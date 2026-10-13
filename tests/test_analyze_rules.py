"""Unit tests for two analyzer rules: compound bar length (6/8 vs 12/8) and the
live-drift correction (beatagent/analyze.py).

    python -m tests.test_analyze_rules

SYNTHETIC inputs with known answers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent import analyze as A  # noqa: E402
from evals.synth import SR, beat_times_constant, render_groove  # noqa: E402

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def accent_signal(grid, pattern, lfsr=1000.0):
    """A low-band flux signal with one pulse per beat, strength cycling through `pattern`."""
    lflux = np.zeros(int((grid[-1] + 1) * lfsr))
    for i, t in enumerate(grid):
        lflux[int(t * lfsr)] = pattern[i % len(pattern)]
    return lflux, lfsr


def main():
    print("compound bar length (beats = dotted quarters)")
    grid = 0.5 + np.arange(96) * 0.88
    check("accent every 2 beats -> 6/8 (2)", A._compound_bar_length(grid, *accent_signal(grid, [1.0, 0.6])) == 2)
    check("accent every 4 beats -> 12/8 (4)", A._compound_bar_length(grid, *accent_signal(grid, [1.0, 0.6, 0.5, 0.6])) == 4)
    check("no accent pattern -> keep the tracker (None)", A._compound_bar_length(grid, *accent_signal(grid, [1.0])) is None)
    check("ambiguous half-bar accent -> None", A._compound_bar_length(grid, *accent_signal(grid, [1.0, 0.6, 0.78, 0.6])) is None)
    check("too short -> None", A._compound_bar_length(grid[:20], *accent_signal(grid[:20], [1.0, 0.6])) is None)

    print("live-drift correction (end to end, a NEW drifting take -- not A3's seed)")
    from evals.eval_beats import drift
    applied, worse = [], []
    for seed in range(70, 78):                                # 8 new takes, none of them A3's
        rng = np.random.default_rng(seed)
        ibi = np.clip(60 / 112 * np.exp(np.cumsum(rng.normal(0, 0.004, 320))), 60 / 118, 60 / 106)
        true = 0.6 + np.concatenate([[0], np.cumsum(ibi[:-1])])
        audio = render_groove(true, duration=true[-1] + 1.5, style="rock", seed=seed, jitter_ms=12)[:, None]
        g_on = A.analyze(audio, SR)
        saved = A.DRIFT_SHARE
        A.DRIFT_SHARE = 2.0                                   # disables the correction
        try:
            g_off = A.analyze(audio, SR)
        finally:
            A.DRIFT_SHARE = saved
        w_on, w_off = drift(g_on.beats, true)["within_10ms"], drift(g_off.beats, true)["within_10ms"]
        dc = g_on.meta["fit"].get("drift_correction") or {}
        print(f"  [INFO] seed {seed}: {g_off.meta['grid_kind']:18s} applied={dc.get('applied')} "
              f"within 10 ms {w_off:.3f} -> {w_on:.3f}")
        if dc.get("applied"):
            applied.append(w_on - w_off)
        if w_on < w_off - 0.01:
            worse.append(seed)
    check("the correction never makes a take worse (8 new takes)", not worse, f"worse on seeds {worse}")
    check("where it applies, more beats land within 10 ms", bool(applied) and min(applied) > 0,
          f"applied on {len(applied)} takes, gains {[round(x, 3) for x in applied]}")

    print("variable-grid smoothing (the same 8 takes, smoothing on vs off)")
    gains = []
    for seed in range(70, 78):
        rng = np.random.default_rng(seed)
        ibi = np.clip(60 / 112 * np.exp(np.cumsum(rng.normal(0, 0.004, 320))), 60 / 118, 60 / 106)
        true = 0.6 + np.concatenate([[0], np.cumsum(ibi[:-1])])
        audio = render_groove(true, duration=true[-1] + 1.5, style="rock", seed=seed, jitter_ms=12)[:, None]
        g_on = A.analyze(audio, SR)
        if not (g_on.meta["fit"].get("variable_smoothing") or {}).get("applied"):
            continue
        orig = A._smooth_variable
        A._smooth_variable = lambda g: (g, {"applied": False})
        try:
            g_off = A.analyze(audio, SR)
        finally:
            A._smooth_variable = orig
        gains.append(drift(g_on.beats, true)["within_10ms"] - drift(g_off.beats, true)["within_10ms"])
    check("smoothing raises within-10-ms on every variable-path take", bool(gains) and min(gains) > 0,
          f"{len(gains)} takes, gains {[round(x, 3) for x in gains]}")
    steady = beat_times_constant(124.0, 0.5, 240)
    audio2 = render_groove(steady, duration=steady[-1] + 1.5, style="house", seed=6)
    c2, info2 = A._drift_correction(steady, audio2, SR)
    check("steady track: grid left alone", c2 is None and not info2["applied"], str(info2))

    print("abrupt tempo steps (onset-based, the tracker smears them)")
    from evals.eval_robustness import ramp, steps
    for bpms in ([120, 128, 100], [96, 112]):
        audio3, true3, downs3, _ = steps(bpms)
        g = A.analyze(audio3[:, None], SR)
        d = drift(g.beats, true3)
        pw = g.meta["fit"].get("piecewise_by_onsets") or {}
        check(f"steps {bpms}: piecewise grid, every beat within 10 ms", g.meta["grid_kind"] == "piecewise_constant"
              and pw.get("chosen") and len(g.beats) == len(true3) and d["within_10ms"] > 0.97,
              f"{g.meta['grid_kind']}, {len(g.beats)}/{len(true3)} beats, within 10 ms {d['within_10ms']:.3f}")
        gd = g.beats[g.positions == 1]
        check(f"steps {bpms}: bar lines survive the tempo changes",
              len(gd) == len(downs3) and np.max(np.abs(gd - downs3)) < 0.02, f"{len(gd)}/{len(downs3)} downbeats")
    audio4, true4, *_ = ramp(110, 130)
    g4 = A.analyze(audio4[:, None], SR)
    check("a smooth ramp is not mistaken for steps", not (g4.meta["fit"].get("piecewise_by_onsets") or {}).get("chosen"),
          g4.meta["grid_kind"])
    print(f"\n{'all analyzer rule tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
