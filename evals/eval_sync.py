"""Part 1 sync evaluation: beat-match B to A and measure kick alignment.

    python -m evals.eval_sync [--a A1] [--b A7] [--a-bar 33] [--method all]

Methods: constant (one-piece R3 stretch), beats (per-beat R3 slices),
beats_attack (per-beat slices + verbatim 30 ms attack), constant_attack
(one-piece R3 + verbatim patch 40 ms before to 30 ms after every beat).
Writes evals/results/sync_<A>_<B>/<method>_{a,b,mix}.wav and <method>_report.json,
plus comparison.json/.md.
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

from beatagent import qc, sync  # noqa: E402
from beatagent.analyze import analyze  # noqa: E402
from beatagent.audio_io import load, save  # noqa: E402
from beatagent.grid import BeatGrid  # noqa: E402

METHODS = {
    "constant": lambda a, asr, b, bsr, p, ag, bg: sync.render_constant(a, asr, b, bsr, p),
    "beats": lambda a, asr, b, bsr, p, ag, bg: sync.render_beat_anchored(a, asr, b, bsr, p, ag, bg, attack_ms=0),
    "beats_attack": lambda a, asr, b, bsr, p, ag, bg: sync.render_beat_anchored(a, asr, b, bsr, p, ag, bg, attack_ms=30),
    "constant_attack": lambda a, asr, b, bsr, p, ag, bg: sync.render_constant_attack(a, asr, b, bsr, p, ag, bg),
}


def grid_for(name: str, pack: Path, cache: Path):
    audio_path = next(p for p in pack.glob(f"{name}.*") if p.suffix in (".wav", ".flac", ".mp3"))
    audio, fmt = load(audio_path)
    cached = cache / f"{name}.grid.json"
    if cached.exists():
        print(f"  {name}: using cached grid {cached.relative_to(ROOT)}")
        return audio, fmt, BeatGrid.load(cached)
    t0 = time.time()
    g = analyze(audio, fmt.sr)
    print(f"  {name}: analysed in {time.time() - t0:.1f}s")
    return audio, fmt, g


def join_clicks(r: dict, b_src: np.ndarray, plan, a_grid: BeatGrid, b_grid: BeatGrid,
                edges_s=(0.0,)) -> dict:
    """HF energy at each join in B's stem vs the same spot in original B (dB; >~3 = click).

    edges_s: where the joins sit relative to each beat. Per-beat slices join AT
    the beat; constant_attack patches join 45 ms before and 35 ms after it."""
    sr = r["sr"]
    ia = int(np.argmin(np.abs(a_grid.beats - plan.a_start)))
    ib = int(np.argmin(np.abs(b_grid.beats - plan.b_start)))
    vals = []
    for j in range(1, 200):
        if ia + j >= len(a_grid.beats) or ib + j >= len(b_grid.beats):
            break
        t_out = a_grid.beats[ia + j] - plan.a_start
        if t_out > plan.duration - 0.1:
            break
        for e in edges_s:
            vals.append(qc.splice_click_db(r["b"], b_src, sr, t_out + e, [b_grid.beats[ib + j] + e]))
    v = np.array(vals)
    return {"n_joins": int(len(v)), "median_db": round(float(np.median(v)), 1),
            "max_db": round(float(np.max(v)), 1), "joins_over_3db": int(np.sum(v > 3.0))}


def pitch_check(method: str) -> dict:
    """110 Hz sine as B (grid 128 BPM) synced onto a 124 BPM grid: output must stay 110 Hz."""
    sr = 44100
    t = np.arange(sr * 40) / sr
    tone = (0.5 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)[:, None]

    def g(bpm, dur):
        beats = np.arange(0.5, dur - 0.5, 60 / bpm)
        return BeatGrid(beats=beats, positions=(np.arange(len(beats)) % 4) + 1, beats_per_bar=4,
                        time_signature="4/4", duration=dur, constant_tempo=True)
    ga, gb = g(124, 40.0), g(128, 40.0)
    plan = sync.plan_sync(ga, gb, a_bar=2, duration=20.0)
    r = METHODS[method](np.zeros_like(tone), sr, tone, sr, plan, ga, gb)
    seg = r["b"][sr * 2: sr * 18, 0]
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=len(seg) * 4))
    k = int(np.argmax(sp))
    y0, y1, y2 = np.log(sp[k - 1:k + 2] + 1e-12)
    f = (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * sr / (len(seg) * 4)
    return {"in_hz": 110.0, "out_hz": round(float(f), 3), "cents": round(float(1200 * np.log2(f / 110)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="A1")
    ap.add_argument("--b", default="A7")
    ap.add_argument("--a-bar", type=int, default=33)
    ap.add_argument("--b-bar", type=int, default=1)
    ap.add_argument("--method", default="all", choices=["all", *METHODS])
    ap.add_argument("--pack", default="substitute")
    args = ap.parse_args()
    pack = ROOT / "data" / args.pack
    cache = ROOT / "evals" / "results" / f"part1_{args.pack}_beat_this"
    out = ROOT / "evals" / "results" / f"sync_{args.a}_{args.b}"
    out.mkdir(parents=True, exist_ok=True)

    a_audio, a_fmt, a_grid = grid_for(args.a, pack, cache)
    b_audio, b_fmt, b_grid = grid_for(args.b, pack, cache)
    plan = sync.plan_sync(a_grid, b_grid, a_bar=args.a_bar, b_bar=args.b_bar)
    print(f"  plan: B x{plan.time_ratio:.6f} ({plan.b_bpm:.3f} -> {plan.a_bpm:.3f} BPM), "
          f"B bar {plan.b_bar} @ {plan.b_start:.3f}s -> A bar {plan.a_bar} @ {plan.a_start:.3f}s")
    b_src = sync._match_format(b_audio, b_fmt.sr, a_fmt.sr, a_audio.shape[1])

    rows = []
    for name in (METHODS if args.method == "all" else [args.method]):
        t0 = time.time()
        r = METHODS[name](a_audio, a_fmt.sr, b_audio, b_fmt.sr, plan, a_grid, b_grid)
        el = time.time() - t0
        m = sync.measure_alignment(r["a"], r["b"], r["sr"], a_grid, plan)
        edges = {"beats": (0.0,), "beats_attack": (0.0, 0.0325),
                 "constant_attack": (-(sync.JOIN_XFADE / 2 + 0.040), 0.030 + sync.JOIN_XFADE / 2)}
        clicks = join_clicks(r, b_src, plan, a_grid, b_grid, edges[name]) if name in edges else None
        pitch = pitch_check(name)
        for k in ("a", "b", "mix"):
            save(out / f"{name}_{k}.wav", r[k], r["sr"], "PCM_16", "WAV")
        rep = {"label": "SYNTHETIC SUBSTITUTE pair" if args.pack == "substitute" else args.pack,
               "a": args.a, "b": args.b, "method": r["method"], "render_s": round(el, 2),
               "plan": plan.to_dict(), "alignment": m, "join_clicks": clicks, "pitch_check": pitch}
        with open(out / f"{name}_report.json", "w") as f:
            json.dump(rep, f, indent=1)
        rows.append({"method": name, "measured": f"{m['n_measured']}/{m['n_beats']}",
                     "median_abs_ms": m.get("median_abs_ms"), "p95_abs_ms": m.get("p95_abs_ms"),
                     "max_abs_ms": m.get("max_abs_ms"), "pass": m.get("pass"),
                     "join_click_max_db": clicks["max_db"] if clicks else "n/a",
                     "pitch_cents": pitch["cents"], "render_s": round(el, 1)})
        print("  " + json.dumps(rows[-1]))
    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    with open(out / "comparison.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
