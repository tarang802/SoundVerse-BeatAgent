"""Chat message -> typed plan. The LLM INTERPRETS; it never measures or edits.

Two backends produce the same `Plan`:
  * GeminiInterpreter -- structured JSON output against the Plan schema.
  * RuleInterpreter   -- a small regex parser used only when no API key is set,
                         so the app and evals still run. It is labelled as such
                         in every reply and only covers simple phrasings.

Either way the plan is untrusted input: ops are re-validated into the strict
edl types and range-checked against the measured grid by deterministic code.
"""
from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- plan schema
# Flat (no unions) so it maps cleanly onto Gemini's response_schema.
class PRange(BaseModel):
    unit: Literal["beats", "bars"]
    start: Optional[int] = None
    end: Optional[int] = None
    last: Optional[int] = None


class PPosition(BaseModel):
    unit: Literal["beats", "bars"]
    index: Optional[int] = None


class POp(BaseModel):
    op: Literal["remove", "keep_only", "repeat", "move", "copy", "fade", "gain"]
    range: Optional[PRange] = None
    total_plays: Optional[int] = None
    to: Optional[PPosition] = None
    direction: Optional[Literal["in", "out"]] = None
    db: Optional[float] = None


class Plan(BaseModel):
    action: Literal["edit", "undo", "redo", "clarify", "answer", "confirm", "cancel"]
    ops: list[POp] = Field(default_factory=list)
    interpretation: str = ""      # which reading of the request was used, in one or two sentences
    message: str = ""             # question / answer / refusal text for the user


SYSTEM = """You are the interpreter for Beat Agent, a chat-driven audio editor.
You turn ONE user message into a JSON plan. You never measure audio and never guess
times: all numbers about the track come from the MEASURED GRID provided below.
Deterministic code will validate your ops, check ranges, compute exact seconds,
and echo the result to the user, so prefer emitting the literal op and let code reject it.

CONVENTIONS (fixed; always follow them and mention any you relied on in `interpretation`):
- Beats are 0-based GRID LINES. A beat range is half-open [start, end): "beat 16-32"
  -> {"unit":"beats","start":16,"end":32} = 16 beats. "beat 0 to beat 8" = start 0, end 8.
- Bars are 1-based REGIONS, ranges inclusive: "bars 200-210" -> start 200, end 210.
  "bar 5" -> start 5, end 5. "the first 4 bars" -> bars 1-4.
- "the last N beats/bars" -> use {"unit":..., "last": N}.
- start/end null = start/end of the audio.
- repeat: `total_plays` counts the original. "repeat it twice" / "loop it two times" = 2 plays
  in total (original + 1 copy). "loop four times" = 4. The repeated block is inserted in
  place and the rest of the track follows (state this reading).
- "cut X and repeat it" means take X as the slice to repeat, not delete it.
- "cut"/"remove"/"delete"/"chop out" X on its own means remove X.
- move/copy `to` is measured on the track BEFORE the edit; {"unit":"bars","index":k} is the
  start of bar k; index null = end of track.
- Multiple ops in one plan are applied in order, each against the grid produced by the previous one.
- fade: direction "in" or "out" over the range. gain: db, range null = whole track.

ACTIONS:
- "edit": ops non-empty.
- "undo" / "redo": the user wants to step back / forward (one turn).
- "clarify": the request is subjective or ambiguous in a way the conventions do not settle
  (e.g. "the boring part", "the good bit", "make it shorter"). NEVER guess. If the MEASURED
  SECTIONS make a concrete proposal possible, put it in `ops` and ask the user to confirm,
  quoting the measurements (bars, energy) that support it. Otherwise ask a specific question.
- "confirm": the user accepts the pending proposal (e.g. "yes", "do it").
- "cancel": the user rejects the pending proposal.
- "answer": a question about the track (tempo, bars, length) answered from the measured grid,
  or a request this editor cannot do (explain briefly in `message`, e.g. pitch shifting).
Keep `message` short and concrete. Never invent measurements that are not in the context."""


def _context(grid_summary: dict, sections: list | None, pending: list | None, history: list[str]) -> str:
    return json.dumps({
        "MEASURED_GRID": grid_summary,
        "MEASURED_SECTIONS": sections or "not computed",
        "PENDING_PROPOSAL": pending or None,
        "RECENT_TURNS": history[-6:],
    }, indent=1)


# Tried in order after the configured model. Checked 2026-10-12 against this
# project's key: gemini-2.5-flash -> 404 "no longer available to new users";
# gemini-flash-latest -> 503 (high demand); gemini-3.5-flash -> OK.
DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest")


class GeminiInterpreter:
    name = "gemini"

    def __init__(self, model: str | None = None):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.notes: list[str] = []          # e.g. "configured model unavailable, used X"

    def _generate(self, contents: str, config):
        """Call the configured model; on 404 (retired/unknown model) move to the
        next fallback, on 429/503 (busy) retry once then move on. Raises the last
        error if every model fails."""
        import time as _t
        tried, last = [], None
        for m in dict.fromkeys([self.model, *FALLBACK_MODELS]):
            for attempt in range(2):
                try:
                    resp = self.client.models.generate_content(model=m, contents=contents, config=config)
                    if m != self.model:
                        self.notes.append(f"model {self.model} unavailable ({str(last)[:80]}); using {m}")
                        self.model = m             # remember what works for later turns
                    return resp
                except Exception as e:  # noqa: BLE001
                    last = e
                    code = getattr(e, "code", None) or getattr(e, "status_code", None)
                    if code in (429, 503) and attempt == 0:
                        _t.sleep(2.0)
                        continue
                    break
            tried.append(m)
        raise RuntimeError(f"all Gemini models failed ({', '.join(tried)}): {last}")

    def plan(self, message: str, grid_summary: dict, sections=None, pending=None, history=()) -> Plan:
        from google.genai import types
        ctx = _context(grid_summary, sections, pending, list(history))
        resp = self._generate(
            f"CONTEXT:\n{ctx}\n\nUSER MESSAGE:\n{message}",
            types.GenerateContentConfig(
                system_instruction=SYSTEM, temperature=0.0,
                response_mime_type="application/json", response_schema=Plan),
        )
        self.usage["calls"] += 1
        um = getattr(resp, "usage_metadata", None)
        if um:
            self.usage["input_tokens"] += um.prompt_token_count or 0
            self.usage["output_tokens"] += um.candidates_token_count or 0
        if getattr(resp, "parsed", None) is not None:
            return resp.parsed
        return Plan.model_validate_json(resp.text)


# ---------------------------------------------------------------- offline fallback
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "twelve": 12, "sixteen": 16, "once": 1, "twice": 2, "thrice": 3}
_N = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|sixteen)"
_TIMES = r"(\d+\s*(?:x|times)|once|twice|thrice|(?:one|two|three|four|five|six|seven|eight)\s+times)"


def _n(s: str) -> int:
    s = s.strip().lower()
    m = re.match(r"\d+", s)
    if m:
        return int(m.group())
    return _NUM[s.split()[0]]


class RuleInterpreter:
    """Offline fallback (no LLM). Deliberately narrow; anything else -> clarify."""
    name = "offline-rules"

    def plan(self, message: str, grid_summary: dict, sections=None, pending=None, history=()) -> Plan:
        m = message.lower().strip().rstrip(".!")
        m = m.replace("–", "-").replace("—", "-")
        if re.fullmatch(r"(undo|undo that|undo it|go back)", m):
            return Plan(action="undo")
        if re.fullmatch(r"(redo|redo that|redo it)", m):
            return Plan(action="redo")
        if pending and re.fullmatch(r"(yes|yep|ok|okay|do it|go ahead|confirm|sure)( please)?", m):
            return Plan(action="confirm")
        if pending and re.fullmatch(r"(no|nope|cancel|don't|do not)", m):
            return Plan(action="cancel")

        rng = r"(beats?|bars?)\s+" + _N + r"\s*(?:-|to|through|until)\s*(?:beat |bar )?" + _N
        # cut/take X and repeat/loop it N times
        x = re.search(r"(?:cut|take|grab)\s+" + rng + r".*?(?:repeat|loop)\w*\s+it\s+" + _TIMES, m)
        if x:
            unit = "beats" if x.group(1).startswith("beat") else "bars"
            return Plan(action="edit", ops=[POp(op="repeat", range=PRange(unit=unit, start=_n(x.group(2)), end=_n(x.group(3))),
                                              total_plays=_n(x.group(4)))],
                        interpretation="'cut X and repeat it N times' = repeat X in place, N plays in total.")
        # loop the first N bars M times
        x = re.search(r"(?:loop|repeat)\s+the\s+first\s+" + _N + r"\s+(beats?|bars?)\s+" + _TIMES, m)
        if x:
            unit = "beats" if x.group(2).startswith("beat") else "bars"
            n = _n(x.group(1))
            r = PRange(unit=unit, start=1, end=n) if unit == "bars" else PRange(unit=unit, start=0, end=n)
            return Plan(action="edit", ops=[POp(op="repeat", range=r, total_plays=_n(x.group(3)))],
                        interpretation=f"the first {n} {unit} play {_n(x.group(3))}x in total, then the track continues.")
        # repeat the last bar before the tempo change N times
        x = re.search(r"(?:repeat|loop)\s+the\s+last\s+bar\s+before\s+the\s+tempo\s+change\s+" + _TIMES, m)
        if x:
            segs = grid_summary.get("tempo_segments") or []
            if len(segs) < 2:
                return Plan(action="clarify", message="I didn't measure a tempo change in this track, so there's no "
                                                      "'last bar before the tempo change'. Which bar do you mean?")
            first_new = segs[1]["start_bar"]
            bar = first_new - 1 if segs[1]["start_beat"] is not None else first_new
            return Plan(action="edit", ops=[POp(op="repeat", range=PRange(unit="bars", start=bar, end=bar),
                                              total_plays=_n(x.group(1)))],
                        interpretation=f"the measured tempo change starts in bar {first_new}, so the last bar before it is bar {bar}.")
        # fade in/out the first/last N beats/bars
        x = re.search(r"fade\s+(in|out)\s+the\s+(first|last)\s+" + _N + r"\s+(beats?|bars?)", m)
        if x:
            unit = "beats" if x.group(4).startswith("beat") else "bars"
            n = _n(x.group(3))
            if x.group(2) == "last":
                r = PRange(unit=unit, last=n)
            else:
                r = PRange(unit=unit, start=0, end=n) if unit == "beats" else PRange(unit=unit, start=1, end=n)
            return Plan(action="edit", ops=[POp(op="fade", range=r, direction=x.group(1))])
        # remove/cut/delete X
        x = re.search(r"(?:cut|remove|delete|drop|chop out)\s+" + rng, m)
        if x:
            unit = "beats" if x.group(1).startswith("beat") else "bars"
            return Plan(action="edit", ops=[POp(op="remove", range=PRange(unit=unit, start=_n(x.group(2)), end=_n(x.group(3))))])
        x = re.search(r"(?:cut|remove|delete|drop|chop out)\s+(beat|bar)\s+" + _N + r"\b", m)
        if x:
            k = _n(x.group(2))
            if x.group(1) == "bar":
                return Plan(action="edit", ops=[POp(op="remove", range=PRange(unit="bars", start=k, end=k))])
            return Plan(action="edit", ops=[POp(op="remove", range=PRange(unit="beats", start=k, end=k + 1))],
                        interpretation=f"'beat {k}' = the one-beat span [{k}, {k + 1}).")
        x = re.search(r"(?:turn|make)\s+(?:it|the track|everything)\s+(up|down)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*db", m)
        if x:
            db = float(x.group(2)) * (1 if x.group(1) == "up" else -1)
            return Plan(action="edit", ops=[POp(op="gain", db=db)])
        if re.search(r"boring|good part|best part|interesting|the drop|make it better|shorter", m):
            return Plan(action="clarify",
                        message="That's a judgement call I can't measure directly. Tell me which bars, "
                                "or ask me to list the sections I measured.")
        return Plan(action="clarify", message="I couldn't map that to an edit (offline rule parser: set "
                                              "GEMINI_API_KEY for full language understanding). Try e.g. "
                                              "'remove bars 5-8', 'repeat beats 16-32 twice', 'undo'.")


def make_interpreter(prefer: str = "auto"):
    if prefer in ("auto", "gemini"):
        try:
            return GeminiInterpreter()
        except Exception:
            if prefer == "gemini":
                raise
    return RuleInterpreter()
