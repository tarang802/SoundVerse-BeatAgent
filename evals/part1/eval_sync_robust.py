"""Part 1 sync robustness: many A/B pairs with KNOWN beats, through analysis + sync.

    python -m evals.part1.eval_sync_robust [pair ...]

Each pair is analysed with the real pipeline, planned, rendered and measured.

PRIMARY measure (ground truth, detector-free): every pasted kick is a verbatim copy
of B placed at a known output position, and the true kick times of A and B are
known, so each kick's true landing offset B - A is exact arithmetic. Pass = brief
target (median <= 5 ms, max <= 15 ms) on that, or -- where a refusal is the right
answer -- a SyncError with a message. The two detector measures (onset nearest each
beat; low-frequency kick on downbeats) are reported as secondary information:
since the renderer snaps kicks to onsets, a detector measure alone would be circular.
SYNTHETIC material.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soxr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from beatagent.beats import sync  # noqa: E402
from beatagent.beats.analyze import analyze  # noqa: E402
from evals.part1.eval_robustness import groove, tempo_curve  # noqa: E402
from evals.common.synth import SR, render_groove  # noqa: E402

OUT = ROOT / "evals" / "results" / "part1_sync_robustness"


def mono(x):
    return x[:, None] if x.ndim == 1 else x


def g(bpm, bars=60, seed=1, **kw):
    a, b, _, _ = groove(bpm, bars=bars, seed=seed, **kw)
    return a, b


def drift(bars=60):
    b = tempo_curve(112 * np.exp(np.cumsum(np.random.default_rng(9).normal(0, 0.002, bars * 4))))
    return render_groove(b, duration=b[-1] + 1, style="rock", seed=9), b


def a5_like():
    b = tempo_curve(np.concatenate([[120] * 128, [128] * 160]))
    return render_groove(b, duration=b[-1] + 1, style="house", seed=5), b


def at48(pair):
    a, b = pair
    y = soxr.resample(mono(a), SR, 48000).astype(np.float32)
    return np.repeat(y, 2, axis=1), 48000, b


def B(pair, sr=SR):
    return pair[0], sr, pair[1]


A124 = lambda: g(124, seed=1)  # noqa: E731

PAIRS = [
    # name, A builder -> (audio, beats), B builder -> (audio, sr, beats), a_bar, b_bar, duration, expect
    ("same_tempo", A124, lambda: B(g(124, seed=2, bass_root=40)), 9, 1, 60, "ok"),
    ("b_+2pct", A124, lambda: B(g(126.48, seed=3)), 9, 1, 60, "ok"),
    ("b_+10pct", A124, lambda: B(g(136.4, bars=70, seed=3)), 9, 1, 60, "ok"),
    ("b_+20pct", A124, lambda: B(g(148.8, bars=80, seed=3)), 9, 1, 60, "ok"),
    ("b_-10pct", A124, lambda: B(g(111.6, seed=3)), 9, 1, 60, "ok"),
    ("b_-20pct", A124, lambda: B(g(99.2, seed=3)), 9, 1, 60, "ok"),
    ("b_+30pct_refuse", A124, lambda: B(g(161.2, bars=80, seed=3)), 9, 1, 60, "refuse"),
    ("b_octave_62", A124, lambda: B(g(62, bars=40, style="rock", seed=3)), 9, 1, 60, "ok"),
    ("b_octave_248", A124, lambda: B(g(248, bars=140, seed=3)), 9, 1, 60, "ok"),
    ("b_48k_stereo", A124, lambda: at48(g(128, seed=4)), 9, 1, 60, "ok"),
    ("b_drifting", A124, lambda: B(drift()), 9, 1, 60, "ok"),
    ("a_tempo_change_in_overlap", a5_like, lambda: B(g(124, seed=6)), 25, 1, 60, "ok"),
    ("b_bar_5", A124, lambda: B(g(128, seed=7)), 9, 5, 60, "ok"),
    ("a_bar_near_end_refuse", A124, lambda: B(g(128, seed=7)), 55, 1, 60, "refuse"),
    ("b_too_short_refuse", A124, lambda: B(g(128, bars=20, seed=7)), 9, 1, 60, "refuse"),
    ("meter_mismatch_refuse", A124, lambda: B(g(120, bpb=3, style="waltz", seed=7)), 9, 1, 60, "refuse"),
    ("overlap_30s", A124, lambda: B(g(128, seed=8)), 9, 1, 30, "ok"),
    ("overlap_120s", lambda: g(124, bars=80, seed=1), lambda: B(g(128, bars=80, seed=8)), 9, 1, 120, "ok"),
    ("self_sync", A124, lambda: B(A124()), 9, 1, 60, "ok"),
]


def nearest(x, arr):
    i = int(np.argmin(np.abs(arr - x)))
    return float(arr[i]), float(abs(arr[i] - x))


def ground_truth(patches, a_true, b_true, a_start):
    """True B-A kick offset (ms) for every pasted kick."""
    offs = []
    for d0, s0 in patches:
        ta, da = nearest(d0 + a_start, a_true)       # A's true kick near this output beat
        tb, db = nearest(s0, b_true)                 # B's true kick near the pasted source point
        if da > 0.06 or db > 0.06:                   # not a beat of either track (shouldn't happen)
            continue
        offs.append(((d0 + (tb - s0)) - (ta - a_start)) * 1000)
    return np.array(offs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    only = set(sys.argv[1:])
    for name, ab, bb, a_bar, b_bar, dur, expect in PAIRS:
        if only and name not in only:
            continue
        a, a_true = ab()
        a = mono(a)
        b, bsr, b_true = bb()
        b = mono(b)
        t0 = time.time()
        ga, gb = analyze(a, SR), analyze(b, bsr)
        row = {"pair": name, "expect": expect, "a_bpm": round(ga.tempo_bpm, 2), "b_bpm": round(gb.tempo_bpm, 2),
               "a_meter": ga.time_signature, "b_meter": gb.time_signature,
               "a_grid": ga.meta["grid_kind"], "b_grid": gb.meta["grid_kind"]}
        try:
            plan = sync.plan_sync(ga, gb, a_bar, b_bar, dur)
        except sync.SyncError as e:
            row.update({"result": "refused", "message": str(e)[:200], "pass": expect == "refuse"})
            rows.append(row)
            print(f"  {name:28s} refused ({'pass' if row['pass'] else 'FAIL'}): {str(e)[:110]}")
            continue
        bg = sync.matched_b_grid(plan, gb)
        r = sync.render(a, SR, b, bsr, plan, ga, bg)
        # the analysis may count a track at double tempo (62 BPM -> 124): the true beats
        # at that counting level include the midpoints, so compare at the same level
        b_oct = (gb.meta.get("fit", {}).get("tempo_octave") or {}).get("factor", 1.0) * plan.b_octave
        a_oct = (ga.meta.get("fit", {}).get("tempo_octave") or {}).get("factor", 1.0)
        dbl = lambda x: np.sort(np.concatenate([x, (x[:-1] + x[1:]) / 2]))  # noqa: E731
        b_ref = dbl(b_true) if b_oct >= 2 else b_true
        a_ref = dbl(a_true) if a_oct >= 2 else a_true
        gt = ground_truth(r["patches"], a_ref, b_ref, plan.a_start)
        m = sync.measure_alignment(r["a"], r["b"], r["sr"], ga, plan)
        k = sync.measure_downbeat_kicks(r["a"], r["b"], r["sr"], ga, plan)
        gt_ok = len(gt) > 0 and np.median(np.abs(gt)) <= 5 and np.max(np.abs(gt)) <= 15 and len(gt) >= 0.9 * len(r["patches"])
        row.update({"result": "rendered", "ratio": round(plan.time_ratio, 5), "b_octave": plan.b_octave,
                    "render_len_s": round(len(r["mix"]) / r["sr"], 3),
                    "ground_truth": {"n": int(len(gt)), "median_abs_ms": round(float(np.median(np.abs(gt))), 2) if len(gt) else None,
                                     "max_abs_ms": round(float(np.max(np.abs(gt))), 2) if len(gt) else None},
                    "any_onset": {x: m.get(x) for x in ("n_measured", "n_beats", "median_abs_ms", "max_abs_ms")},
                    "downbeat_kicks": {x: k.get(x) for x in ("n_measured", "n_downbeats", "median_abs_ms", "max_abs_ms")},
                    "elapsed_s": round(time.time() - t0, 1),
                    "pass": expect == "ok" and bool(gt_ok) and abs(len(r["mix"]) / r["sr"] - dur) < 0.01})
        rows.append(row)
        gtx = row["ground_truth"]
        print(f"  {name:28s} {'pass' if row['pass'] else 'FAIL'}  ratio {plan.time_ratio:.4f} oct x{plan.b_octave} "
              f"{row['a_meter']}/{row['b_meter']} | TRUTH n={gtx['n']} med {gtx['median_abs_ms']} max {gtx['max_abs_ms']} | "
              f"(detectors: all {m['n_measured']}/{m['n_beats']} max {m.get('max_abs_ms')}; dbk max {k.get('max_abs_ms')})")
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"\npassed {sum(r['pass'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
