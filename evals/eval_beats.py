"""Part 1 evaluation: beat grid accuracy, drift, speed; writes grids + click auditions.

    python -m evals.eval_beats [--pack substitute|testpack] [--tracker beat_this|librosa]

Metrics (mir_eval, +-70 ms, computed on the FULL track: we do not apply
mir_eval's default 5-second trim, so early beats count too):
  beat_F, downbeat_F            fitted grid vs reference
  raw_beat_F                    the tracker's own beats before grid fitting
  drift_max_ms / within_10ms    constant-tempo clips: |grid - reference| for every reference beat
  analysis_s                    wall-clock analysis time on this machine's CPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mir_eval  # noqa: E402

from beatagent.analyze import TRACKERS, analyze  # noqa: E402
from beatagent.audio_io import load, save, to_mono  # noqa: E402
from beatagent.click import audition  # noqa: E402


def load_ref(path: Path):
    rows = np.loadtxt(path, ndmin=2)
    beats = rows[:, 0]
    pos = rows[:, 1].astype(int)
    downs = beats[pos == 1] if np.any(pos > 0) else None
    return beats, downs


def drift(grid_beats: np.ndarray, ref: np.ndarray) -> dict:
    idx = np.clip(np.searchsorted(grid_beats, ref), 1, len(grid_beats) - 1)
    near = np.where(np.abs(grid_beats[idx - 1] - ref) < np.abs(grid_beats[idx] - ref), grid_beats[idx - 1], grid_beats[idx])
    err = (near - ref) * 1000
    q = len(err) // 10 or 1
    return {"drift_max_ms": round(float(np.max(np.abs(err))), 2),
            "drift_median_ms": round(float(np.median(err)), 2),
            "within_10ms": round(float(np.mean(np.abs(err) <= 10)), 4),
            "err_first10pct_ms": round(float(np.median(err[:q])), 2),
            "err_last10pct_ms": round(float(np.median(err[-q:])), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="substitute")
    ap.add_argument("--tracker", default="beat_this")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    pack = ROOT / "data" / args.pack
    out = ROOT / "evals" / "results" / f"part1_{args.pack}_{args.tracker}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ref_path in sorted(pack.glob("*.beats")):
        name = ref_path.stem
        if args.only and name not in args.only:
            continue
        audio_path = next(p for p in pack.glob(f"{name}.*") if p.suffix in (".wav", ".flac", ".mp3"))
        audio, fmt = load(audio_path)
        ref_b, ref_d = load_ref(ref_path)
        t0 = time.time()
        grid = analyze(audio, fmt.sr, tracker=args.tracker)
        el = time.time() - t0
        raw_b, raw_d, _ = TRACKERS[args.tracker](to_mono(audio), fmt.sr)
        grid.save(out / f"{name}.grid.json")
        save(out / f"{name}.click.wav", audition(audio, fmt.sr, grid), fmt.sr, "PCM_16", "WAV")
        r = {"clip": name, "duration_s": round(len(audio) / fmt.sr, 1), "analysis_s": round(el, 2),
             "tracker": grid.meta["tracker"], "grid_type": grid.meta["grid_kind"],
             "tempo_bpm": round(grid.tempo_bpm, 3), "time_signature": grid.time_signature,
             "confidence": grid.confidence.get("overall"),
             "beat_F": round(mir_eval.beat.f_measure(ref_b, grid.beats, 0.07), 4),
             "raw_beat_F": round(mir_eval.beat.f_measure(ref_b, raw_b, 0.07), 4)}
        if ref_d is not None:
            r["downbeat_F"] = round(mir_eval.beat.f_measure(ref_d, grid.downbeats, 0.07), 4)
            r["raw_downbeat_F"] = round(mir_eval.beat.f_measure(ref_d, raw_d, 0.07), 4)
        r.update(drift(grid.beats, ref_b))
        r["raw_drift_max_ms"] = drift(np.sort(raw_b), ref_b)["drift_max_ms"]
        r["segments"] = [(s["start_bar"], s["bpm"]) for s in grid.segments]
        rows.append(r)
        print(json.dumps(r))
    with open(out / "results.json", "w") as f:
        json.dump({"pack": args.pack, "label": ("SYNTHETIC SUBSTITUTE - not the assignment benchmark"
                                                if args.pack == "substitute" else args.pack), "rows": rows}, f, indent=1)
    cols = ["clip", "duration_s", "analysis_s", "grid_type", "tempo_bpm", "time_signature", "beat_F", "raw_beat_F",
            "downbeat_F", "drift_max_ms", "within_10ms", "raw_drift_max_ms", "confidence"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "n/a")) for c in cols) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
