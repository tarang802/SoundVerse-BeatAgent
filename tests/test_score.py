"""Part 3 score schema: valid scores load, round-trip, convert to seconds; invalid ones are refused.

    python -m tests.test_score
"""
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.score import Note, Score, Track, default_patch  # noqa: E402

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def base(**kw):
    d = dict(tempo_bpm=120.0, bars=2, beats_per_bar=4, loop_beats=8,
             tracks=[Track(name="drums", kind="drums", patch=default_patch("drums"),
                           notes=[Note(start=0, duration=0.25, pitch=36, velocity=110),
                                  Note(start=1, duration=0.25, pitch=38, velocity=100)]),
                     Track(name="bass", kind="bass", patch=default_patch("bass"),
                           notes=[Note(start=0.5, duration=1.0, pitch=33, velocity=90)])])
    d.update(kw)
    return d


def refused(name, **kw):
    try:
        Score(**base(**kw))
        check(name, False, "accepted")
    except ValidationError as e:
        check(name, True, str(e.errors()[0]["msg"])[:80])


def main():
    s = Score(**base())
    check("valid score loads", s.loop_seconds == 4.0)
    p = ROOT / "work" / "test_score.json"
    p.parent.mkdir(exist_ok=True)
    s.save(p)
    check("save/load round-trip", Score.load(p) == s)
    ev = s.note_events_seconds("bass")
    check("beats -> seconds", ev == [(0.25, 0.75, 33, 90, "bass")], str(ev))
    refused("note outside the loop", tracks=[Track(name="d", kind="drums", notes=[Note(start=8, duration=0.25, pitch=36, velocity=100)])])
    refused("non-GM drum pitch", tracks=[Track(name="d", kind="drums", notes=[Note(start=0, duration=0.25, pitch=40, velocity=100)])])
    refused("loop length != bars x beats", loop_beats=7)
    refused("duplicate track names", tracks=[Track(name="x", kind="bass"), Track(name="x", kind="keys")])
    print(f"\n{'all score tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
