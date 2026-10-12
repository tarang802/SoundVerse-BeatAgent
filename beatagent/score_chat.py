"""Part 3 chat changes to a recreation: "mute the drums", "make the bass brighter",
"play it at 140". Same rule as Part 2: the LLM (or the offline rule parser) only turns
the message into typed ScoreOps; code checks them against the actual tracks, applies
them to the score, echoes exactly what changed, and regenerates the page.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel

from .score import Score

WAVES = ("sine", "triangle", "sawtooth", "square")


class ScoreOp(BaseModel):
    op: Literal["tempo", "mute", "unmute", "solo", "gain", "brightness", "transpose", "wave"]
    track: Optional[str] = None      # track name; None = whole score (tempo) or ask
    value: Optional[float] = None    # tempo BPM | gain dB (relative) | brightness factor | semitones
    wave: Optional[str] = None


class ScorePlan(BaseModel):
    action: Literal["edit", "answer", "clarify"]
    ops: list[ScoreOp] = []
    message: str = ""


class ScoreEditError(ValueError):
    pass


def apply_ops(score: Score, ops: list[ScoreOp]) -> tuple[Score, list[str]]:
    """Apply typed ops to a COPY of the score; -> (new score, echo lines). All-or-nothing."""
    s = score.model_copy(deep=True)
    names = [t.name for t in s.tracks]
    echo = []

    def track(op):
        if op.track is None:
            raise ScoreEditError(f"which track? This recreation has: {', '.join(names)}.")
        hit = [t for t in s.tracks if t.name.lower() == op.track.lower()]
        if not hit:
            raise ScoreEditError(f"there is no '{op.track}' track; this recreation has: {', '.join(names)}.")
        return hit[0]

    for op in ops:
        if op.op == "tempo":
            if op.value is None or not 30 <= op.value <= 300:
                raise ScoreEditError("tempo must be between 30 and 300 BPM.")
            echo.append(f"tempo {s.tempo_bpm:.1f} -> {op.value:.1f} BPM (loop {s.loop_seconds:.2f} s -> "
                        f"{s.loop_beats * 60 / op.value:.2f} s; notes stay on their beats)")
            s.tempo_bpm = float(op.value)
        elif op.op in ("mute", "unmute"):
            t = track(op)
            t.mute = op.op == "mute"
            echo.append(f"{t.name}: {'muted' if t.mute else 'unmuted'}")
        elif op.op == "solo":
            t = track(op)
            for u in s.tracks:
                u.mute = u is not t
            echo.append(f"solo {t.name} (muted: {', '.join(u.name for u in s.tracks if u.mute) or 'none'})")
        elif op.op == "gain":
            t = track(op)
            if op.value is None:
                raise ScoreEditError("how many dB?")
            new = t.gain_db + op.value
            if not -60 <= new <= 12:
                raise ScoreEditError(f"{t.name} would be at {new:+.1f} dB; the range is -60 to +12 dB.")
            echo.append(f"{t.name}: level {t.gain_db:+.1f} -> {new:+.1f} dB")
            t.gain_db = new
        elif op.op == "brightness":
            t = track(op)
            if t.kind == "drums":
                raise ScoreEditError("drum brightness is fixed in this version; try the bass or keys.")
            f = op.value if op.value and op.value > 0 else 2.0
            new = min(16000.0, max(80.0, t.patch.cutoff_hz * f))
            echo.append(f"{t.name}: filter cutoff {t.patch.cutoff_hz:.0f} -> {new:.0f} Hz "
                        f"({'brighter' if new > t.patch.cutoff_hz else 'darker'})")
            t.patch.cutoff_hz = new
        elif op.op == "transpose":
            t = track(op)
            if t.kind == "drums":
                raise ScoreEditError("drums can't be transposed.")
            k = int(round(op.value or 0))
            if any(not 0 <= n.pitch + k <= 127 for n in t.notes):
                raise ScoreEditError(f"transposing {t.name} by {k:+d} would leave the MIDI range.")
            for n in t.notes:
                n.pitch += k
            echo.append(f"{t.name}: transposed {k:+d} semitones")
        elif op.op == "wave":
            t = track(op)
            if op.wave not in WAVES or t.kind == "drums":
                raise ScoreEditError(f"oscillator must be one of {', '.join(WAVES)} (pitched tracks only).")
            echo.append(f"{t.name}: oscillator {t.patch.wave} -> {op.wave}")
            t.patch.wave = op.wave
    return Score.model_validate(s.model_dump()), echo


# ---------------------------------------------------------------- interpretation
SYSTEM = """You turn ONE message about a music recreation into a JSON plan of score edits.
You never invent notes or measurements. Ops:
- tempo: value = new BPM ("play it at 140", "speed it up to 128").
- mute / unmute / solo: track = a track name from TRACKS.
- gain: track, value = RELATIVE dB ("turn the bass up 3 dB" -> +3; "quieter" without a number -> -3).
- brightness: track, value = cutoff multiplier ("brighter" -> 2, "darker"/"duller" -> 0.5).
- transpose: track, value = semitones ("bass up an octave" -> 12).
- wave: track, wave in sine|triangle|sawtooth|square.
Use only track names from TRACKS (match "the drums" to "drums" etc.). If the message names a
track that does not exist or is unclear, use action "clarify" and say which tracks exist.
Questions ("what tempo is it?") -> action "answer" from the context. Copy numbers literally."""


def _context(score: Score) -> str:
    tracks = ", ".join("%s (%s, %d notes, mute=%s, level %+.1f dB)" % (t.name, t.kind, len(t.notes), t.mute, t.gain_db)
                       for t in score.tracks)
    return f"TRACKS: {tracks}\nTEMPO: {score.tempo_bpm:.2f} BPM, {score.time_signature}, {score.bars} bars"


class RuleScoreInterpreter:
    """Offline fallback for the common phrasings (no LLM)."""
    name = "offline-rules"

    def plan(self, text: str, score: Score) -> ScorePlan:
        m = text.lower().strip().rstrip(".!")
        names = [t.name for t in score.tracks]

        def find_track():
            for n in names:
                if re.search(rf"\b{re.escape(n.lower())}\b", m) or (n == "drums" and re.search(r"\b(drum|beat|kick|hats?)s?\b", m)):
                    return n
            return None
        x = re.search(r"(?:play it|tempo|bpm)\D{0,12}(\d{2,3}(?:\.\d+)?)|(\d{2,3}(?:\.\d+)?)\s*bpm", m)
        if x:
            return ScorePlan(action="edit", ops=[ScoreOp(op="tempo", value=float(x.group(1) or x.group(2)))])
        t = find_track()
        for word, op in (("unmute", "unmute"), ("mute", "mute"), ("solo", "solo")):
            if re.search(rf"\b{word}\b", m):
                return ScorePlan(action="edit", ops=[ScoreOp(op=op, track=t)]) if t else \
                    ScorePlan(action="clarify", message=f"Which track? This recreation has: {', '.join(names)}.")
        if re.search(r"\b(brighter|brighten)\b", m) and t:
            return ScorePlan(action="edit", ops=[ScoreOp(op="brightness", track=t, value=2.0)])
        if re.search(r"\b(darker|duller|warmer)\b", m) and t:
            return ScorePlan(action="edit", ops=[ScoreOp(op="brightness", track=t, value=0.5)])
        g = re.search(r"\b(louder|quieter|up|down)\b(?:\D*(\d+(?:\.\d+)?)\s*db)?", m)
        if g and t:
            v = float(g.group(2) or 3)
            return ScorePlan(action="edit", ops=[ScoreOp(op="gain", track=t,
                                                         value=v if g.group(1) in ("louder", "up") else -v)])
        return ScorePlan(action="clarify", message="Offline parser: try 'mute the drums', 'make the bass brighter', "
                                                   f"'play it at 140'. Tracks: {', '.join(names)}.")


class GeminiScoreInterpreter:
    """Same Gemini client and model fallback chain as Part 2, different schema."""
    name = "gemini"

    def __init__(self):
        from .llm import GeminiInterpreter
        self._g = GeminiInterpreter()

    @property
    def notes(self):
        return self._g.notes

    def plan(self, text: str, score: Score) -> ScorePlan:
        from google.genai import types
        resp = self._g._generate(f"CONTEXT:\n{_context(score)}\n\nUSER MESSAGE:\n{text}",
                                 types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.0,
                                                             response_mime_type="application/json",
                                                             response_schema=ScorePlan))
        p = resp.parsed if getattr(resp, "parsed", None) is not None else ScorePlan.model_validate_json(resp.text)
        return p


class ScoreChat:
    """Chat session over one recreation: interpret -> validate/apply -> echo; undo."""

    def __init__(self, score: Score, interpreter=None):
        self.history = [score]
        if interpreter is None:
            import os
            try:
                if os.environ.get("BEAT_AGENT_LLM", "").lower() == "offline":
                    raise RuntimeError("offline mode")
                interpreter = GeminiScoreInterpreter()
            except Exception:  # noqa: BLE001  (no key)
                interpreter = RuleScoreInterpreter()
        self.llm = interpreter

    @property
    def score(self) -> Score:
        return self.history[-1]

    def handle(self, text: str) -> dict:
        if re.fullmatch(r"\s*undo\s*\.?\s*", text.lower()):
            if len(self.history) == 1:
                return {"ok": False, "action": "undo", "text": "Nothing to undo."}
            self.history.pop()
            return {"ok": True, "action": "undo", "text": "Undid the last change.", "changed": True}
        backend = self.llm.name
        try:
            plan = self.llm.plan(text, self.score)
        except Exception as e:  # noqa: BLE001  (quota, network)
            plan = RuleScoreInterpreter().plan(text, self.score)
            backend = f"offline-rules (LLM unavailable: {type(e).__name__})"
        if plan.action != "edit" or not plan.ops:
            return {"ok": True, "action": plan.action, "text": plan.message or "Tell me what to change.",
                    "backend": backend}
        try:
            new, echo = apply_ops(self.score, plan.ops)
        except ScoreEditError as e:
            return {"ok": False, "action": "error", "text": f"I didn't change anything: {e}", "backend": backend}
        self.history.append(new)
        return {"ok": True, "action": "edit", "text": "Applied: " + "; ".join(echo), "echo": echo,
                "changed": True, "backend": backend}
