"""Part 3 chat changes to a recreation (beatagent/recreate/score_chat.py), without an API key.

    python -m tests.test_score_chat
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.recreate.score import Note, Score, Track, default_patch  # noqa: E402
from beatagent.recreate.score_chat import (RuleScoreInterpreter, ScoreChat, ScoreEditError, ScoreOp,  # noqa: E402
                                  ScorePlan, apply_ops)

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def score():
    return Score(tempo_bpm=122.0, bars=4, beats_per_bar=4, loop_beats=16, tracks=[
        Track(name="drums", kind="drums", patch=default_patch("drums"),
              notes=[Note(start=0, duration=0.25, pitch=36, velocity=110)]),
        Track(name="bass", kind="bass", patch=default_patch("bass"),
              notes=[Note(start=0, duration=1, pitch=33, velocity=90)]),
        Track(name="keys", kind="keys", patch=default_patch("keys"),
              notes=[Note(start=2, duration=2, pitch=60, velocity=80)])])


class Scripted:
    name = "scripted"

    def __init__(self, plan):
        self.p = plan

    def plan(self, text, score):
        return self.p


def refused(name, ops):
    try:
        apply_ops(score(), ops)
        check(name, False, "accepted")
    except ScoreEditError as e:
        check(name, True, str(e)[:80])


def main():
    s = score()
    new, echo = apply_ops(s, [ScoreOp(op="tempo", value=140)])
    check("tempo changes, notes stay on their beats", new.tempo_bpm == 140 and new.tracks[1].notes == s.tracks[1].notes
          and s.tempo_bpm == 122.0, echo[0])
    new, _ = apply_ops(s, [ScoreOp(op="mute", track="Drums")])
    check("mute (case-insensitive track name)", new.tracks[0].mute and not s.tracks[0].mute)
    new, _ = apply_ops(s, [ScoreOp(op="solo", track="bass")])
    check("solo mutes the others", [t.mute for t in new.tracks] == [True, False, True])
    new, echo = apply_ops(s, [ScoreOp(op="brightness", track="bass", value=2)])
    check("brighter doubles the bass cutoff", new.tracks[1].patch.cutoff_hz == 2 * s.tracks[1].patch.cutoff_hz, echo[0])
    new, _ = apply_ops(s, [ScoreOp(op="gain", track="keys", value=-6)])
    check("gain is relative", new.tracks[2].gain_db == -6)
    new, _ = apply_ops(s, [ScoreOp(op="transpose", track="bass", value=12)])
    check("transpose bass up an octave", new.tracks[1].notes[0].pitch == 45)
    new, _ = apply_ops(s, [ScoreOp(op="wave", track="keys", wave="square")])
    check("change oscillator", new.tracks[2].patch.wave == "square")
    refused("unknown track refused, lists the tracks", [ScoreOp(op="mute", track="vocals")])
    refused("tempo out of range refused", [ScoreOp(op="tempo", value=900)])
    refused("transposing drums refused", [ScoreOp(op="transpose", track="drums", value=2)])
    refused("all-or-nothing: a bad op cancels the batch", [ScoreOp(op="mute", track="bass"), ScoreOp(op="mute", track="x")])

    r = RuleScoreInterpreter()
    for text, exp in [("mute the drums", ("mute", "drums", None)), ("make the bass brighter", ("brightness", "bass", 2.0)),
                      ("play it at 140", ("tempo", None, 140.0)), ("turn the keys down 4 dB", ("gain", "keys", -4.0)),
                      ("unmute the beat", ("unmute", "drums", None))]:
        p = r.plan(text, s)
        got = (p.ops[0].op, p.ops[0].track, p.ops[0].value) if p.ops else None
        check(f"offline parser: '{text}'", got == exp, str(got))
    check("offline parser: unknown request asks", r.plan("make it sound like a choir", s).action == "clarify")

    chat = ScoreChat(s, RuleScoreInterpreter())
    a = chat.handle("mute the drums")
    b = chat.handle("play it at 140")
    check("chat applies and echoes", a["changed"] and b["changed"] and chat.score.tempo_bpm == 140
          and chat.score.tracks[0].mute, b["text"][:80])
    u = chat.handle("undo")
    check("undo restores the previous score", u["ok"] and chat.score.tempo_bpm == 122.0 and chat.score.tracks[0].mute)
    chat = ScoreChat(s, Scripted(ScorePlan(action="edit", ops=[ScoreOp(op="mute", track="guitar")])))
    r2 = chat.handle("mute the guitar")
    check("LLM naming a missing track: refused, nothing changed", not r2["ok"] and len(chat.history) == 1
          and "drums, bass, keys" in r2["text"], r2["text"][:90])
    print(f"\n{'all score chat tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
