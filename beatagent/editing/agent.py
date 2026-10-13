"""One chat turn: interpret (LLM) -> validate -> execute (code) -> echo."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import edl
from .edl import EditError
from .llm import Plan, PSync, make_interpreter
from .session import Session


@dataclass
class Reply:
    text: str
    action: str
    ok: bool = True
    echo: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    interpretation: str = ""
    backend: str = ""
    changed: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# An edit leaving less than this needs a "yes". Absolute, not a fraction: a 5 % rule also
# caught "trim it down to just bars 1-8" (15.5 s of a 6-minute track) -- a deliberate trim.
MIN_KEEP_S = 2.0
_CLAIMS_FRACTION = re.compile(r"fraction|whole (?:beat|bar|number)s?|individual beats", re.IGNORECASE)
_PROPOSES = re.compile(r"\b(bars?|beats?)\s+\d+", re.IGNORECASE)   # a clarify message naming a range
_NUM_REF = re.compile(r"\b(bars?|beats?)\s+(?:number\s+)?(-\s?\d+(?:\.\d+)?|\d+\.\d+)", re.IGNORECASE)


def _bad_numbers(text: str) -> str | None:
    """A bar/beat reference that is negative or fractional, as a user-facing message."""
    m = _NUM_REF.search(text.replace("−", "-"))
    if not m:
        return None
    unit, num = m.group(1).lower(), m.group(2).replace(" ", "")
    if num.startswith("-"):
        return (f"'{unit} {num}' is negative. Bars are numbered from 1 and beats from 0 "
                "(beat N is a grid line). Which one did you mean?")
    return (f"'{unit} {num}' is a fraction. I can only cut on whole beats and bars. "
            "Which whole beat or bar did you mean?")


_DB = re.compile(r"([-+−]?\s?\d+(?:\.\d+)?)\s*(?:db|decibels?)\b", re.IGNORECASE)


def _keep_user_db(text: str, plan: Plan) -> str | None:
    """If the user stated exactly one dB amount, every gain op uses that magnitude;
    the interpreter only chooses the direction (up/down). An LLM once turned
    "60 dB louder" into +24 dB (the engine's limit) and it was applied silently."""
    vals = {abs(float(m.replace("−", "-").replace(" ", ""))) for m in _DB.findall(text)}
    if len(vals) != 1:
        return None
    user = vals.pop()
    changed = False
    for o in plan.ops or []:
        if o.op == "gain" and o.db is not None and abs(o.db) != user:
            o.db = user if o.db >= 0 else -user
            changed = True
    return f"used your {user:g} dB (the interpreter had changed the number)" if changed else None


def measured_sections(session: Session) -> list:
    """Sections of the CURRENT version (recomputed after edits, cached per state)."""
    from .sections import sections
    cache = session.__dict__.setdefault("_sections_cache", {})
    if session.state_id not in cache:
        cache[session.state_id] = sections(session.render(), session.sr, session.grid)
    return cache[session.state_id]


class BeatAgent:
    def __init__(self, session: Session, interpreter=None, sections_fn=measured_sections,
                 partner: Session | None = None):
        self.s = session
        self.llm = interpreter or make_interpreter()
        self.sections_fn = sections_fn          # callable(session) -> measured section list
        self.partner = partner                  # second track for "sync" commands (optional)
        self.pending: list | None = None        # typed ops awaiting "yes"
        self.pending_raw: list | None = None
        self.log: list[str] = []
        self.last_plan = None                   # the interpreter's raw plan, kept for evals/debugging
        self.last_sync: dict | None = None

    def _track_line(self) -> str:
        g = self.s.grid
        return (f"Track now {g.duration:.3f} s, {g.n_beats} beats, {g.n_bars} bars "
                f"({'constant' if g.constant_tempo else 'variable'} grid, {g.tempo_bpm:.2f} BPM median).")

    def handle(self, text: str) -> Reply:
        self.log.append(f"user: {text}")
        bad = _bad_numbers(text)
        if bad:
            # Deterministic, before any LLM: an LLM once "corrected" "remove bar -2" to bar 1
            # and applied it. Numbers that can't address the grid are refused by code.
            r = Reply(f"I didn't change anything: {bad}", "error", ok=False, backend="validator")
            self.log.append(f"agent: {r.text}")
            return r
        sections, partner = None, None         # defined even if the try fails early (used by the fallback)
        try:
            sections = self.sections_fn(self.s) if self.sections_fn else None
            if self.partner is not None:
                pg = self.partner.grid
                partner = {"file": self.partner.path.name, "tempo_bpm": round(pg.tempo_bpm, 2),
                           "n_bars": pg.n_bars, "time_signature": pg.time_signature}
            plan: Plan = self.llm.plan(text, self.s.grid.summary(), sections, self.pending_raw, self.log,
                                       partner=partner)
            self.retried = None
            if plan.action in ("edit", "clarify") and plan.ops:
                try:
                    Session.parse_ops([o.model_dump() for o in plan.ops])
                except EditError as e:
                    # One repair round: give the interpreter its own validation error back
                    # (seen: Gemini emitted "fade" ops with no range). Still validated after.
                    self.retried = str(e)
                    plan = self.llm.plan(f"{text}\n\n[SYSTEM: your previous plan was invalid -- {e} Every op "
                                         "needs a `range` (whole track: start and end null). Return a corrected plan.]",
                                         self.s.grid.summary(), sections, self.pending_raw, self.log,
                                         partner=partner)
            elif plan.action in ("answer", "clarify") and not plan.ops and _CLAIMS_FRACTION.search(plan.message or ""):
                # Seen (flash-lite, 2 of 3 tries): "cut beat 16-32 repeat it twice" refused as
                # "fractions or individual beats cannot be addressed". _bad_numbers already
                # passed, so the message has only whole numbers: the refusal is wrong.
                self.retried = "false fraction claim"
                fixed = self.llm.plan(f"{text}\n\n[SYSTEM: code checked this message: every bar/beat number in it "
                                      "is a whole number, so it is addressable (\"beat 16-32\" = beats 16 to 32). "
                                      "Do not refuse for fractions; return the edit it asks for.]",
                                      self.s.grid.summary(), sections, self.pending_raw, self.log,
                                      partner=partner)
                if not _CLAIMS_FRACTION.search(fixed.message or "") or fixed.ops:
                    plan = fixed
            elif plan.action == "clarify" and not plan.ops and _PROPOSES.search(plan.message or ""):
                # Seen: "Did you mean bars 81 to 96?" with no ops. Only ops can be confirmed,
                # so the user's "yes" would do nothing. One round to attach the edit.
                self.retried = "clarify proposal without ops"
                fixed = self.llm.plan(f"{text}\n\n[SYSTEM: your clarify message proposes a range ({plan.message!r}) "
                                      "but `ops` is empty, so the user cannot confirm it. Return the same clarify "
                                      "with that edit in `ops`, or a question that proposes no range.]",
                                      self.s.grid.summary(), sections, self.pending_raw, self.log,
                                      partner=partner)
                if fixed.action == "clarify":
                    plan = fixed
            db_note = _keep_user_db(text, plan)
            self.last_plan = plan
        except Exception as e:  # noqa: BLE001  (network, quota, malformed JSON)
            # Degrade instead of failing: the offline rule parser covers simple phrasings.
            # Its replies are labelled, so nobody mistakes them for the LLM's.
            from .llm import RuleInterpreter
            if isinstance(self.llm, RuleInterpreter):
                r = Reply(f"I couldn't interpret that ({type(e).__name__}: {str(e)[:200]}). Nothing was changed.",
                          "error", ok=False, backend=self.llm.name)
                self.log.append(f"agent: {r.text}")
                return r
            reason = "quota exhausted" if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) else type(e).__name__
            plan = RuleInterpreter().plan(text, self.s.grid.summary(), sections, self.pending_raw, self.log,
                                          partner=partner)
            self.last_plan = plan
            r = self._execute(plan)
            r.backend = "offline-rules (LLM unavailable)"
            r.text = f"[LLM unavailable: {reason}; used the offline rule parser for this turn]\n" + r.text
            r.interpretation = plan.interpretation
            self.log.append(f"agent: {r.text}")
            return r
        r = self._execute(plan)
        if db_note:
            r.warnings = [*r.warnings, db_note]
        r.backend = f"{self.llm.name}:{getattr(self.llm, 'model', '')}".rstrip(":")
        notes = getattr(self.llm, "notes", None)
        if notes:                              # e.g. configured model retired -> fallback used
            r.warnings = [*r.warnings, *notes]
            r.text += "\nNote: " + " ".join(notes)
            notes.clear()
        r.interpretation = plan.interpretation
        self.log.append(f"agent: {r.text}")
        return r

    def _execute(self, plan: Plan) -> Reply:
        if plan.action == "undo":
            t = self.s.undo()
            return Reply(f"{t.echo[0]}. {self._track_line()}" if t.ok else t.error, "undo", t.ok, changed=t.ok)
        if plan.action == "redo":
            t = self.s.redo()
            return Reply(f"{t.echo[0]}. {self._track_line()}" if t.ok else t.error, "redo", t.ok, changed=t.ok)
        if plan.action == "confirm":
            if not self.pending:
                return Reply("There's no concrete proposal pending, so nothing was changed. Say the edit with "
                             "numbers (e.g. 'remove bars 81-96').", "confirm", False)
            ops, self.pending, self.pending_raw = self.pending, None, None
            return self._apply(ops, "confirm")
        if plan.action == "cancel":
            self.pending = self.pending_raw = None
            return Reply("OK, dropped the proposal. Nothing was changed.", "cancel")
        if plan.action == "answer":
            return Reply(plan.message or "(no answer)", "answer")
        if plan.action == "sync":
            return self._sync(plan)
        raw = [o.model_dump() for o in plan.ops]
        try:
            ops = Session.parse_ops(raw)
        except EditError as e:
            return Reply(f"I didn't change anything: {e}", plan.action, False)
        if plan.action == "clarify":
            if not ops:
                return Reply(plan.message or "Can you be more specific?", "clarify")
            # make the proposal concrete with exact, code-computed numbers
            try:
                echo = self._dry_run(ops)
            except EditError as e:
                return Reply(f"{plan.message}\n(My proposal didn't validate: {e})", "clarify", False)
            self.pending, self.pending_raw = ops, raw
            return Reply(f"{plan.message}\nProposed (not applied yet): " + " | ".join(echo)
                         + "\nSay 'yes' to apply it or tell me what to change.", "clarify", echo=echo)
        if not ops:
            return Reply(plan.message or "I didn't find an edit in that.", "clarify")
        self.pending = self.pending_raw = None
        # Near-total deletion is never applied on one sentence: seen, "remove everything"
        # -> remove bars 1-186, leaving 0.5 s. Make it a proposal that needs a "yes".
        try:
            before = self.s.grid.duration
            after = self._dry_duration(ops)
        except EditError:
            after = None                       # _apply reports the real error
        if after is not None and after < MIN_KEEP_S:
            echo = self._dry_run(ops)
            self.pending, self.pending_raw = ops, raw
            return Reply(f"That would leave only {after:.1f} s of the {before:.1f} s track. I haven't applied it.\n"
                         "Proposed (not applied yet): " + " | ".join(echo)
                         + "\nSay 'yes' to apply it anyway, or tell me what to keep.", "clarify", echo=echo)
        return self._apply(ops, "edit")

    def _dry_duration(self, ops) -> float:
        grid, state = self.s.grid, self.s.state
        for op in ops:
            r = edl.resolve(op, grid)
            state = edl.apply(state, r, self.s.sr)
            grid = edl.map_grid(self.s.src_grid, state, self.s.sr)
        return state.length / self.s.sr

    def _sync(self, plan: Plan) -> Reply:
        """Two-track sync: the LLM only supplied bar numbers; plan, render and
        measurement are the same deterministic code the SYNC panel uses."""
        from ..beats import sync as S
        from ..core.audio_io import save
        if self.partner is None:
            return Reply("Load a second track first: sync needs a partner track to stretch onto this one.",
                         "sync", False)
        p = plan.sync or PSync()                 # LLM said "sync" but gave no details: defaults
        a_bar, b_bar, dur = p.a_bar, p.b_bar, float(p.duration_s)
        a, b = self.s, self.partner
        try:
            sp = S.plan_sync(a.grid, b.grid, a_bar, b_bar, dur)
        except S.SyncError as e:
            return Reply(f"I didn't sync: {e}", "sync", False)
        bg = S.matched_b_grid(sp, b.grid)
        r = S.render(a.render(), a.sr, b.render(), b.sr, sp, a.grid, bg)
        m = S.measure_alignment(r["a"], r["b"], r["sr"], a.grid, sp)
        d = a.workdir / f"sync_A-bar{a_bar}_{b.path.stem}"
        d.mkdir(parents=True, exist_ok=True)
        for k in ("a", "b", "mix"):
            save(d / f"{k}.wav", r[k], r["sr"], "PCM_16", "WAV")
        self.last_sync = {"plan": sp.to_dict(), "alignment": {k: v for k, v in m.items() if k != "offsets_ms"},
                          "dir": str(d)}
        lines = [f"Synced {b.path.name} (B) onto {a.path.name} (A, unchanged): B stretched x{sp.time_ratio:.6f} "
                 f"({(sp.time_ratio - 1) * 100:+.2f}%, pitch preserved), {sp.b_bpm:.2f} -> {sp.a_bpm:.2f} BPM.",
                 f"B bar {sp.b_bar} ({sp.b_start:.3f} s) lands on A bar {sp.a_bar} ({sp.a_start:.3f} s); "
                 f"offset {sp.offset:.3f} s; {sp.duration:.0f} s rendered to {d.name}/mix.wav.",
                 f"Measured onset offset B-A: median {m.get('median_abs_ms')} ms, max {m.get('max_abs_ms')} ms "
                 f"over {m.get('n_measured')}/{m.get('n_beats')} beats."]
        lines += ["Note: " + n for n in sp.notes]
        return Reply("\n".join(lines), "sync", True, echo=lines[:2])

    def _dry_run(self, ops) -> list[str]:
        grid, state, echo = self.s.grid, self.s.state, []
        for op in ops:
            r = edl.resolve(op, grid)
            state = edl.apply(state, r, self.s.sr)
            grid = edl.map_grid(self.s.src_grid, state, self.s.sr)
            echo.append(r.echo)
        return echo

    def _apply(self, ops, action) -> Reply:
        t = self.s.apply(ops)
        if not t.ok:
            return Reply(f"I didn't change anything: {t.error}", action, False, warnings=t.warnings)
        lines = ["Applied: " + e for e in t.echo]
        if t.warnings:
            lines += ["Note: " + w for w in t.warnings]
        lines.append(f"{t.duration_before:.3f} s -> {t.duration_after:.3f} s. " + self._track_line())
        return Reply("\n".join(lines), action, True, echo=t.echo, warnings=t.warnings, changed=True)
