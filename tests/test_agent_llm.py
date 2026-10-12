"""The code AROUND the LLM, tested without spending any API quota.

    python -m tests.test_agent_llm

A scripted interpreter stands in for Gemini (it returns the plans we give it and
records what the agent asked), and a fake google-genai client stands in for the
network (it raises 404 / 429 / 503 the way the real API does). Everything else is
the real code: BeatAgent, Session, the edit engine, the offline rule parser, the
model fallback chain and the record/replay cache. What this cannot test is how
well Gemini itself reads English -- that is the job of evals/eval_chat_suite.py.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent import llm  # noqa: E402
from beatagent.agent import BeatAgent  # noqa: E402
from beatagent.audio_io import save  # noqa: E402
from beatagent.grid import BeatGrid  # noqa: E402
from beatagent.llm import Plan, POp, PRange  # noqa: E402
from beatagent.session import Session  # noqa: E402
from evals.synth import SR, fixture_120  # noqa: E402

WORK = ROOT / "work" / "test_agent_llm"
FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


class Scripted:
    """Stands in for GeminiInterpreter: returns queued plans (or raises queued errors)."""
    name = "gemini"

    def __init__(self, *answers):
        self.queue = list(answers)
        self.model = "scripted"
        self.notes: list[str] = []
        self.asked: list[str] = []

    def plan(self, message, grid_summary, sections=None, pending=None, history=(), partner=None):
        self.asked.append(message)
        a = self.queue.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def bars(a, b=None):
    return PRange(unit="bars", start=a, end=a if b is None else b)


def edit(*ops):
    return Plan(action="edit", ops=list(ops))


def session():
    audio, beats, _ = fixture_120()
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / "fixture_120.wav"
    if not p.exists():
        save(p, audio[:, None], SR, "PCM_16", "WAV")
    g = BeatGrid(beats=beats, positions=(np.arange(len(beats)) % 4) + 1, beats_per_bar=4,
                 time_signature="4/4", duration=len(audio) / SR, constant_tempo=True)
    return Session(p, WORK / "s", grid=g)


def agent(*answers, partner=None):
    s = session()
    return BeatAgent(s, Scripted(*answers), sections_fn=None, partner=partner), s


# ---------------------------------------------------------------- agent paths
def test_agent():
    print("agent: plan -> validate -> execute")
    a, s = agent(edit(POp(op="remove", range=bars(2, 3))))
    n0 = s.grid.n_bars
    r = a.handle("drop bars 2 to 3")
    check("plain edit is applied and echoed", r.ok and r.changed and s.grid.n_bars == n0 - 2
          and r.text.startswith("Applied: remove bars 2-3"), r.text.splitlines()[0])
    check("reply names the LLM backend", r.backend == "gemini:scripted", r.backend)

    # invalid op -> one repair round with the validation error, then applied
    a, s = agent(edit(POp(op="gain", db=60, range=bars(1, 2))), edit(POp(op="gain", db=6, range=bars(1, 2))))
    r = a.handle("make the intro much louder")
    check("invalid op triggers one repair round", a.retried is not None and len(a.llm.asked) == 2
          and "previous plan was invalid" in a.llm.asked[1], str(a.retried)[:80])
    check("repaired plan is applied", r.ok and r.changed and "+6.0 dB" in r.text, r.text.splitlines()[0])

    # concrete proposal -> pending -> "yes" applies exactly it
    prop = Plan(action="clarify", ops=[POp(op="remove", range=bars(4, 5))], message="Did you mean bars 4-5?")
    a, s = agent(prop, Plan(action="confirm"))
    n0 = s.grid.n_bars
    r1 = a.handle("cut the boring bit")
    check("proposal is shown, not applied", not r1.changed and a.pending and "Proposed (not applied yet)" in r1.text)
    r2 = a.handle("yes")
    check("'yes' applies the pending proposal", r2.ok and r2.changed and s.grid.n_bars == n0 - 2)

    # proposal only in prose -> repair round attaches the ops
    a, s = agent(Plan(action="clarify", message="Did you mean bars 4 to 5?"), prop)
    r = a.handle("chop out the boring part")
    check("prose-only proposal is repaired into ops", a.pending is not None and len(a.llm.asked) == 2
          and "proposes a range" in a.llm.asked[1])

    # false "fractions" refusal of whole numbers -> repair round -> edit
    a, s = agent(Plan(action="answer", message="Fractions or individual beats cannot be addressed."),
                 edit(POp(op="repeat", range=PRange(unit="beats", start=4, end=8), total_plays=2)))
    r = a.handle("cut beat 4-8 repeat it twice")
    check("false fraction refusal is repaired", r.ok and r.changed and "repeat beats 4-8" in r.text
          and "whole number" in a.llm.asked[1], r.text.splitlines()[0])

    # deterministic number validator runs BEFORE the LLM
    for msg in ("remove bar -2", "delete bars - 3", "remove beat 3.5"):
        a, s = agent(edit(POp(op="remove", range=bars(1))))      # would be wrong if it were used
        r = a.handle(msg)
        check(f"'{msg}' refused by code, LLM not asked", not r.ok and r.backend == "validator"
              and not a.llm.asked and not r.changed, r.text[:70])
    a, s = agent(edit(POp(op="remove", range=bars(2, 3))))
    check("whole-number ranges pass the validator", a.handle("remove bars 2-3").changed)

    # user's dB magnitude is kept: LLM clamped 60 -> 24 must not be applied
    a, s = agent(edit(POp(op="gain", db=24, range=bars(1, 2))))
    r = a.handle("make bars 1-2 60 dB louder")
    check("LLM-changed dB is replaced by the user's and refused", not r.changed and "+60 dB" in r.text, r.text[:90])
    a, s = agent(edit(POp(op="gain", db=-6, range=bars(1, 2))))
    r = a.handle("turn bars 1-2 down 3 dB")
    check("direction from LLM, magnitude from user", r.changed and "-3.0 dB" in r.text, r.text.splitlines()[0])

    # near-total deletion needs a "yes" (seen live: "remove everything" -> bars 1-186 applied)
    s0 = session()
    n_all = s0.grid.n_bars
    a, s = agent(edit(POp(op="remove", range=bars(1, n_all))), Plan(action="confirm"))
    d0 = s.grid.duration
    r = a.handle("remove everything")
    check("whole-track removal is proposed, not applied", not r.changed and a.pending is not None
          and abs(s.grid.duration - d0) < 1e-9 and "haven't applied" in r.text, r.text.splitlines()[0])
    r = a.handle("yes")
    check("...and applied after 'yes'", r.changed and s.grid.duration < 2.0, f"{s.grid.duration:.2f} s left")

    # undo / redo / confirm with nothing pending / sync without partner
    a, s = agent(edit(POp(op="remove", range=bars(2))), Plan(action="undo"), Plan(action="redo"))
    d0 = s.grid.duration
    a.handle("drop bar 2")
    r = a.handle("undo")
    check("undo restores the track", r.ok and abs(s.grid.duration - d0) < 1e-9)
    r = a.handle("redo")
    check("redo re-applies", r.ok and s.grid.duration < d0)
    a, s = agent(Plan(action="confirm"))
    r = a.handle("yes")
    check("'yes' with nothing pending explains itself", not r.ok and "nothing was changed" in r.text)
    a, s = agent(Plan(action="sync"))
    r = a.handle("bring the other track in at bar 3")
    check("sync without a second track explains itself", not r.ok and "second track" in r.text)

    # interpreter notes (model fallback) reach the user
    a, s = agent(edit(POp(op="remove", range=bars(2))))
    a.llm.notes.append("gemini-3.5-flash is unavailable (out of free-tier quota); answered by x instead.")
    r = a.handle("drop bar 2")
    check("fallback-model note is shown", "Note: gemini-3.5-flash is unavailable" in r.text)

    # LLM down -> offline parser, clearly labelled, still correct for simple phrasing
    a, s = agent(RuntimeError("all Gemini models failed: 429 RESOURCE_EXHAUSTED"))
    n0 = s.grid.n_bars
    r = a.handle("remove bars 2-3")
    check("LLM failure falls back to the offline parser", r.changed and s.grid.n_bars == n0 - 2
          and r.text.startswith("[LLM unavailable: quota exhausted") and "offline" in r.backend, r.text.splitlines()[0])
    a, s = agent(RuntimeError("network down"))
    r = a.handle("make it more groovy")
    check("offline parser asks instead of guessing", not r.changed and r.action == "clarify")


# ---------------------------------------------------------------- Gemini wrapper
class FakeErr(Exception):
    def __init__(self, code, text):
        super().__init__(f"{code} {text}")
        self.code = code


class FakeResp:
    def __init__(self, plan):
        self.parsed = plan
        self.text = plan.model_dump_json()
        self.usage_metadata = None


class FakeClient:
    """models.generate_content: per model, a list of outcomes consumed in order."""

    def __init__(self, script):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls: list[str] = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        out = self.script.get(model, [FakeErr(404, "NOT_FOUND")]).pop(0)
        if isinstance(out, Exception):
            raise out
        return FakeResp(out)


def gemini(script, model="gemini-3.5-flash"):
    g = object.__new__(llm.GeminiInterpreter)
    g.client, g.model = FakeClient(script), model
    g.usage, g.notes = {"calls": 0, "input_tokens": 0, "output_tokens": 0}, []
    return g


def test_gemini_wrapper():
    print("gemini wrapper: fallback chain, notes, cache")
    sleep, time.sleep = time.sleep, lambda s: None          # the 429 back-off, not under test
    try:
        ok = edit(POp(op="remove", range=bars(2)))
        q = FakeErr(429, "RESOURCE_EXHAUSTED quota")
        g = gemini({"gemini-3.5-flash": [q, q], "gemini-3.8-flash": [q, q],
                    "gemini-flash-latest": [q, q], "gemini-3.5-flash-lite": [ok]})
        p = g.plan("drop bar 2", {"n_bars": 10})
        check("429 on every big model -> answered by a lite model", p == ok and g.model == "gemini-3.5-flash-lite"
              and g.client.calls.count("gemini-3.5-flash") == 2, str(g.client.calls))
        check("note says quota, in plain words", g.notes and "out of free-tier quota" in g.notes[0]
              and "{" not in g.notes[0], g.notes[0] if g.notes else "")
        g.notes.clear()
        g.client.script["gemini-3.5-flash-lite"] = [ok]
        g.plan("drop bar 2", {"n_bars": 10})
        check("working model is remembered (no re-probing next turn)", g.client.calls[-1] == "gemini-3.5-flash-lite"
              and not g.notes)

        g = gemini({"gemini-2.5-flash": [FakeErr(404, "NOT_FOUND")], "gemini-3.5-flash": [ok]}, model="gemini-2.5-flash")
        g.plan("drop bar 2", {"n_bars": 10})
        check("404 (retired model) -> next model, no retry", g.client.calls == ["gemini-2.5-flash", "gemini-3.5-flash"]
              and "not found" in g.notes[0], str(g.client.calls))

        g = gemini({"gemini-3.5-flash": [FakeErr(503, "UNAVAILABLE"), ok]})
        g.plan("drop bar 2", {"n_bars": 10})
        check("503 (busy) -> one retry on the same model", g.client.calls == ["gemini-3.5-flash"] * 2 and not g.notes)

        g = gemini({m: [q, q] for m in llm.FALLBACK_MODELS})
        try:
            g.plan("drop bar 2", {"n_bars": 10})
            check("all models down raises (so the agent can fall back)", False)
        except RuntimeError as e:
            check("all models down raises (so the agent can fall back)", "all Gemini models failed" in str(e))

        # end to end: real BeatAgent + real wrapper with every model down -> offline, labelled
        s = session()
        n0 = s.grid.n_bars
        r = BeatAgent(s, gemini({m: [q, q] for m in llm.FALLBACK_MODELS}), sections_fn=None).handle("remove bars 2-3")
        check("agent + wrapper, all models down -> offline edit, labelled", r.changed and s.grid.n_bars == n0 - 2
              and "quota exhausted" in r.text)

        # record / replay cache: second identical request spends no call
        d = Path(tempfile.mkdtemp())
        old = os.environ.get("BEAT_AGENT_LLM_CACHE")
        os.environ["BEAT_AGENT_LLM_CACHE"] = str(d)
        try:
            g = gemini({"gemini-3.5-flash": [ok]})
            p1 = g.plan("drop bar 2", {"n_bars": 10})
            p2 = g.plan("drop bar 2", {"n_bars": 10})
            check("cache replays an identical request without a call", p1 == p2 and len(g.client.calls) == 1
                  and g.usage.get("cache_hits") == 1)
        finally:
            if old is None:
                os.environ.pop("BEAT_AGENT_LLM_CACHE", None)
            else:
                os.environ["BEAT_AGENT_LLM_CACHE"] = old
            shutil.rmtree(d, ignore_errors=True)
    finally:
        time.sleep = sleep


def main():
    test_agent()
    test_gemini_wrapper()
    print(f"\n{'all agent/LLM-path tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
