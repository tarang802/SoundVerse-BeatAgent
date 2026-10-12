"""One chat turn: interpret (LLM) -> validate -> execute (code) -> echo."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import edl
from .edl import EditError
from .llm import Plan, make_interpreter
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


class BeatAgent:
    def __init__(self, session: Session, interpreter=None, sections_fn=None):
        self.s = session
        self.llm = interpreter or make_interpreter()
        self.sections_fn = sections_fn          # callable(session) -> measured section list
        self.pending: list | None = None        # typed ops awaiting "yes"
        self.pending_raw: list | None = None
        self.log: list[str] = []

    def _track_line(self) -> str:
        g = self.s.grid
        return (f"Track now {g.duration:.3f} s, {g.n_beats} beats, {g.n_bars} bars "
                f"({'constant' if g.constant_tempo else 'variable'} grid, {g.tempo_bpm:.2f} BPM median).")

    def handle(self, text: str) -> Reply:
        self.log.append(f"user: {text}")
        try:
            sections = self.sections_fn(self.s) if self.sections_fn else None
            plan: Plan = self.llm.plan(text, self.s.grid.summary(), sections, self.pending_raw, self.log)
        except Exception as e:  # noqa: BLE001  (network, quota, malformed JSON)
            r = Reply(f"I couldn't interpret that (interpreter error: {type(e).__name__}: {str(e)[:200]}). "
                      "Nothing was changed.", "error", ok=False, backend=self.llm.name)
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
