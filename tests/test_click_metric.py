"""Does the splice-click metric (beatagent.core.qc.splice_click_db) actually catch clicks?

    python -m tests.test_click_metric

Every "click-free" claim (fuzz test, brief sessions, sync joins) rests on this one
metric, and some results were exactly 0.0 dB -- which could mean "identical to the
source" or "both measurements caught the same drum hit and a click was masked".
So: build splices with KNOWN quality on synthetic audio (drums on the beat + a
sustained bass/pad, so a hard cut makes a real waveform jump) and check the metric
separates them:
  - hard cut between beats (no drum to hide behind)      -> must be flagged (> 3 dB)
  - hard cut ON a beat (kick starts at the same instant) -> reported, see note
  - the engine's own splice (crossfade before the beat)  -> must be clean (<= 3 dB)
SYNTHETIC material.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.editing import edl  # noqa: E402
from beatagent.core import qc  # noqa: E402
from beatagent.editing.edl import EditState, Piece, Range, Remove  # noqa: E402
from beatagent.core.grid import BeatGrid  # noqa: E402
from evals.common.synth import SR, fixture_120  # noqa: E402

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def source():
    drums, beats, _ = fixture_120(duration=20.0)
    t = np.arange(len(drums)) / SR
    pad = 0.25 * np.sin(2 * np.pi * 55.0 * t) + 0.12 * np.sin(2 * np.pi * 110.7 * t)   # never phase-aligned
    return (drums + pad).astype(np.float32)[:, None], beats


def hard_cut(x, a, b):
    """Remove [a, b) seconds with NO crossfade -> (output, splice time)."""
    ia, ib = int(a * SR), int(b * SR)
    return np.concatenate([x[:ia], x[ib:]]), a


def main():
    x, beats = source()
    period = float(np.median(np.diff(beats)))
    print("metric vs splices of known quality (dB above the source's own HF peak; > 3 = click)")
    # 1. hard cuts BETWEEN beats where nothing high-frequency is ringing on either side
    #    (the old +-50 ms metric scored these +7.9 / +2.6 / +7.0 / +5.7: one missed)
    vals = []
    for k in (8, 12, 16, 20):
        a, b = beats[k] + 0.37 * period, beats[k + 3] + 0.33 * period
        y, t = hard_cut(x, a, b)
        vals.append(qc.splice_click_db(y, x, SR, t, [a, b]))
    check("hard cut between beats is flagged (4/4)", min(vals) > 3.0, f"{[round(v, 1) for v in vals]}")
    # 1b. hard cut INTO a ringing open hi-hat: the hat's own >6 kHz level is above the click's
    masked = []
    for k in (8, 12, 16):
        a, b = beats[k] + 0.37 * period, beats[k + 3] + 0.61 * period
        y, t = hard_cut(x, a, b)
        masked.append(qc.splice_click_db(y, x, SR, t, [a, b]))
    print(f"  [INFO] hard cut into a ringing open hat: {[round(v, 1) for v in masked]} dB "
          "(not flagged: masked by louder HF content -- a documented limit of this metric)")
    # 2. hard cuts ON beats
    on = []
    for k in (8, 12, 16):
        a, b = beats[k], beats[k + 3]
        y, t = hard_cut(x, a, b)
        on.append(qc.splice_click_db(y, x, SR, t, [a, b]))
    print(f"  [INFO] hard cut exactly on a beat: {[round(v, 1) for v in on]} dB "
          "(a click at the same instant as the kick attack)")
    # 3. the engine's splice on the same material, same beats
    g = BeatGrid(beats=beats, positions=(np.arange(len(beats)) % 4) + 1, beats_per_bar=4,
                 time_signature="4/4", duration=len(x) / SR, constant_tempo=True)
    eng = []
    for k in (8, 12, 16):
        st = edl.apply(EditState((Piece(0, len(x)),)), edl.resolve(Remove(range=Range(unit="beats", start=k, end=k + 3)), g), SR)
        y = edl.render(x, st, SR)
        for t, src in qc.splice_points(st, SR):
            eng.append(qc.splice_click_db(y, x, SR, t, src))
    check("engine splice measured", len(eng) == 3, f"{len(eng)} splices")
    check("engine splice is clean", eng and max(eng) <= 3.0, f"{[round(v, 1) for v in eng]}")
    # 4. on drums, the kick attack hides even a hard cut (above), so the crossfade is
    #    proven on a DRUMLESS passage (a breakdown): same beat-aligned edits, pad only
    pad = x - np.asarray(fixture_120(duration=20.0)[0], dtype=np.float32)[:, None]
    gp = BeatGrid(beats=beats, positions=g.positions, beats_per_bar=4, time_signature="4/4",
                  duration=len(pad) / SR, constant_tempo=True)
    eng_p, hard_p = [], []
    for k in (8, 12, 16):
        st = edl.apply(EditState((Piece(0, len(pad)),)), edl.resolve(Remove(range=Range(unit="beats", start=k, end=k + 3)), gp), SR)
        y = edl.render(pad, st, SR)
        (t, src), = qc.splice_points(st, SR)
        eng_p.append(qc.splice_click_db(y, pad, SR, t, src))
        h = np.concatenate([pad[:int(src[0] * SR)], pad[int(src[1] * SR):]])
        hard_p.append(qc.splice_click_db(h, pad, SR, t, src))
    check("drumless: hard cut at the engine's splice point is flagged", min(hard_p) > 3.0,
          f"{[round(v, 1) for v in hard_p]}")
    check("drumless: engine's crossfaded splice is clean", max(eng_p) <= 3.0, f"{[round(v, 1) for v in eng_p]}")
    print(f"\n{'click metric checks passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
