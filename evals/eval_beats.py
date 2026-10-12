"""Part 1 evaluation: beat grid accuracy, drift, speed; writes grids + click auditions.

    python -m evals.eval_beats [--pack substitute|testpack] [--tracker beat_this|librosa]

Metrics (mir_eval, +-70 ms, computed on the FULL track: we do not apply
mir_eval's default 5-second trim, so early beats count too):
  beat_F, downbeat_F            fitted grid vs reference (strict)
  beat_AMLt                     mir_eval continuity: accepts double/half/off-beat counting
  raw_beat_F                    the tracker's own beats before grid fitting
  drift_max_ms / within_10ms    constant-tempo clips: |grid - reference| for every reference beat
  analysis_s                    wall-clock analysis time on this machine's CPU
"""
from __future__ import annotations

import argparse
import json
import re
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


REF_EXTS = (".beats", ".txt", ".csv", ".tsv", ".lab")
AUDIO_EXTS = (".wav", ".flac", ".mp3")


def _rows(path: Path) -> list[list[float]]:
    """Numeric rows of a text annotation: any of tab/space/comma/semicolon separators;
    header, comment ('#', '//') and non-numeric lines are skipped."""
    out = []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        parts = [x for x in re.split(r"[\s,;]+", line) if x]
        try:
            out.append([float(x) for x in parts])
        except ValueError:
            continue                       # header such as "time,position"
    return out


def load_ref(path: Path, downbeat_path: Path | None = None):
    """Reference beats (+ downbeats when the annotation has them).

    Layouts accepted (one beat per row):
      time                      beats only (no downbeat reference)
      time  position            position 1 = downbeat (Beat This!, madmom, GTZAN/Ballroom .beats)
      time  bar  beat           3rd column 1 = downbeat
    A separate downbeat file (one time per row) overrides the columns."""
    rows = _rows(path)
    if not rows:
        raise ValueError(f"No numeric beat rows in {Path(path).name}")
    width = min(len(r) for r in rows)
    beats = np.array([r[0] for r in rows], dtype=float)
    downs = None
    if width >= 2:
        col = 2 if width >= 3 else 1
        pos = np.array([r[col] for r in rows])
        whole = np.all(np.abs(pos - np.round(pos)) < 1e-6)
        if whole and np.any(np.round(pos) == 1):
            downs = beats[np.round(pos) == 1]
        elif not whole:                    # "1.1, 1.2, 2.1" bar.beat style
            frac = np.round((pos - np.floor(pos)) * 10)
            downs = beats[frac == 1] if np.any(frac == 1) else None
    if downbeat_path is not None:
        downs = np.array([r[0] for r in _rows(downbeat_path)], dtype=float)
    order = np.argsort(beats)
    return beats[order], (np.sort(downs) if downs is not None else None)


def find_reference(audio_path: Path) -> tuple[Path | None, Path | None]:
    """(beats file, separate downbeats file) next to an audio file, by stem:
    A1.beats, A1.txt, A1_beats.txt, A1.beats.txt ... and A1.downbeats / A1_downbeats.txt."""
    d, stem = audio_path.parent, audio_path.stem
    beats = downs = None
    for f in sorted(d.iterdir()):
        n = f.name.lower()
        if not f.is_file() or not n.startswith(stem.lower()) or f.suffix.lower() not in REF_EXTS + (".downbeats",):
            continue
        rest = n[len(stem):].lstrip("._- ")
        if "downbeat" in rest:
            downs = downs or f
        elif rest.split(".")[0] in ("", "beats", "beat", "txt", "csv", "tsv", "lab"):
            beats = beats or f
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
    for audio_path in sorted(p for p in pack.iterdir() if p.suffix.lower() in AUDIO_EXTS):
        name = audio_path.stem
        if args.only and name not in args.only:
            continue
        ref_path, down_path = find_reference(audio_path)
        if ref_path is None:
            print(f"  {name}: no reference annotation next to it -- skipped (not scored)")
            continue
        audio, fmt = load(audio_path)
        ref_b, ref_d = load_ref(ref_path, down_path)
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
             # AMLt: also accepts double/half tempo and off-beat counting. Reported next to
             # the strict F so "right beats, different counting convention" is visible.
             "beat_AMLt": round(mir_eval.beat.continuity(ref_b, grid.beats)[3], 4),
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
    cols = ["clip", "duration_s", "analysis_s", "grid_type", "tempo_bpm", "time_signature", "beat_F", "beat_AMLt", "raw_beat_F",
            "downbeat_F", "drift_max_ms", "within_10ms", "raw_drift_max_ms", "confidence"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "n/a")) for c in cols) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
