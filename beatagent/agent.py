"""One chat turn: interpret (LLM) -> validate -> execute (code) -> echo."""
from __future__ import annotations

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
                return Reply("There's no pending proposal to confirm.", "confirm", False)
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
        return self._apply(ops, "edit")

    def _sync(self, plan: Plan) -> Reply:
        """Two-track sync: the LLM only supplied bar numbers; plan, render and
        measurement are the same deterministic code the SYNC panel uses."""
        from . import sync as S
        from .audio_io import save
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
