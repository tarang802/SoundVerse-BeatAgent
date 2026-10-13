"""Part 3 on REAL music (no reference MIDI): 8-bar excerpts -> score -> page -> render.

    python -m evals.part3.eval_part3_real FILE [FILE ...] [--bar 17] [--bars 8]

Like the brief's P4 ("dense full mix, no reference"), there is no answer key, so the
measurable things are: does the pipeline run end to end, are the pages clean (no
network, no errors), and how well does the RENDER's rhythm line up with the original
(onsets matched within +-50 ms over two loop passes). Excerpts are cut exactly on bar
lines from Part 1's grid. Copyrighted audio, its scores and renders stay in work/;
only the numbers are written to evals/results/part3_real/results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from beatagent.beats.analyze import analyze  # noqa: E402
from beatagent.core.audio_io import load  # noqa: E402
from beatagent.recreate.transcribe import transcribe  # noqa: E402
from evals.part3.part3_render import measure_render  # noqa: E402

OUT = ROOT / "evals" / "results" / "part3_real"
WORK = ROOT / "work" / "part3_real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bar", type=int, default=17)
    ap.add_argument("--bars", type=int, default=8)
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    rows = []
    for f in args.files:
        p = Path(f)
        audio, fmt = load(p)
        g = analyze(audio, fmt.sr)
        downs = g.beats[g.positions == 1]
        if len(downs) < args.bar + args.bars:
            rows.append({"file": p.name, "error": f"only {len(downs)} bars"})
            continue
        a, b = downs[args.bar - 1], downs[args.bar - 1 + args.bars]
        clip = audio[int(a * fmt.sr):int(b * fmt.sr)]
        score = transcribe(clip, fmt.sr, p.stem[:40])
        d = WORK / p.stem[:40].replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        score.save(d / "score.json")
        r = measure_render(score, d, clip, fmt.sr)
        row = {"file": p.name, "excerpt_bars": f"{args.bar}-{args.bar + args.bars - 1}",
               "excerpt_s": round(len(clip) / fmt.sr, 3), "track_bpm_part1": round(g.tempo_bpm, 2),
               "score_bpm": score.tempo_bpm, "score_bars": score.bars,
               "tracks": {t.name: len(t.notes) for t in score.tracks},
               "render": {k: r[k] for k in ("static_ok", "network_requests", "page_errors", "onset_F", "onset_P",
                                            "onset_R", "median_abs_ms", "max_abs_ms", "onsets_ref", "onsets_render")},
               "play_stop_ok": r["play_stop"]["ok"]}
        rows.append(row)
        print(json.dumps(row))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"label": "REAL MUSIC, NO REFERENCE MIDI: pipeline + render "
                                                  "rhythm only; audio, scores and renders not published",
                                                  "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
