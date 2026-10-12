"""Tune Part 3 drum-detection settings on the DEVELOPMENT set only (data/p3dev).

    python -m evals.tune_p3_drums

Searches: kick relative threshold, kick absolute floor (x clip peak), and a "hat on a
kick" rule (a weak hat detected within 12 ms of a kick is the kick's click, dropped
when its strength is below q x the strongest hat). Objective: mean drum note F on the
dev clips that have drums, subject to ZERO hits on the piano-only dev clips. P1-P4 are
not touched here; they are scored once afterwards as held-out clips.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent import transcribe as T  # noqa: E402
from beatagent.audio_io import load, to_mono  # noqa: E402
from evals.eval_part3 import note_f  # noqa: E402

DEV = ROOT / "data" / "p3dev"


def prepare():
    out = []
    for p in sorted(DEV.glob("D*.wav")):
        a, f = load(p)
        m = to_mono(a).astype(np.float32)
        _, perc = librosa.effects.hpss(m, margin=(1.0, 2.0))
        pm = pretty_midi.PrettyMIDI(str(p.with_suffix(".mid")))
        ref = [(n.start, n.end, n.pitch) for i in pm.instruments if i.is_drum for n in i.notes]
        out.append((p.stem, perc, f.sr, float(np.abs(m).max()), ref, T.drum_candidates(perc, f.sr)))
    return out


def evaluate(prep, kick_rel, kick_floor, hat_kick_q):
    saved = T.DRUM_BANDS["kick"], T.HAT_ON_KICK_Q
    T.DRUM_BANDS["kick"] = (30.0, 120.0, kick_rel, 36, kick_floor)
    T.HAT_ON_KICK_Q = hat_kick_q
    try:
        fs, piano_hits = [], 0
        for _name, perc, sr, peak, ref, cands in prep:
            est = [(t, t + 0.1, p) for t, p, _v in T.transcribe_drums(perc, sr, peak, cands)]
            if ref:
                fs.append(note_f(ref, est)["f"])
            else:
                piano_hits += len(est)
        return float(np.mean(fs)), piano_hits
    finally:
        T.DRUM_BANDS["kick"], T.HAT_ON_KICK_Q = saved


def main():
    prep = prepare()
    current = (T.DRUM_BANDS["kick"][2], T.DRUM_BANDS["kick"][4], T.HAT_ON_KICK_Q)
    rows = []
    for combo in itertools.product((0.15, 0.2, 0.3), (0.03, 0.045, 0.06), (0.0, 0.3, 0.5)):
        f, piano = evaluate(prep, *combo)
        rows.append({"kick_rel": combo[0], "kick_floor": combo[1], "hat_on_kick_q": combo[2],
                     "dev_drums_F": round(f, 4), "piano_only_false_hits": piano})
    ok = [r for r in rows if r["piano_only_false_hits"] == 0]
    best = max(ok, key=lambda r: r["dev_drums_F"]) if ok else None
    cur = next(r for r in rows if (r["kick_rel"], r["kick_floor"], r["hat_on_kick_q"]) == current)
    print("current:", cur)
    print("best (no piano false hits):", best)
    out = ROOT / "evals" / "results" / "part3" / "drum_tuning_dev.json"
    out.write_text(json.dumps({"dev_set": "data/p3dev (synthetic, separate from P1-P4)", "current": cur,
                               "best": best, "grid": rows}, indent=1), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
