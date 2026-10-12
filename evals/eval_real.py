"""Real-music check WITHOUT reference annotations.

    python -m evals.eval_real FILE [FILE ...] [--sync A_FILE B_FILE --a-bar 9]

There is no answer key for these files, so this does NOT measure accuracy.
It reports what can be measured honestly:
  * speed, tempo, meter, grid type, confidence, trimmed beats
  * onset support: for each grid line, is there an onset within +-40 ms and how
    far is it? (A grid line with no hit is not necessarily wrong -- hip-hop
    doesn't hit every beat -- so this is broken down by position in bar.)
  * tempo cross-check vs librosa, to flag a half/double-tempo disagreement
  * a click audition per file, for a human to listen to (the real test)
Outputs go to evals/results/real/ (audio is gitignored: the songs are copyrighted).
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

from beatagent import sync  # noqa: E402
from beatagent.analyze import TRACKERS, analyze  # noqa: E402
from beatagent.audio_io import load, save, to_mono  # noqa: E402
from beatagent.click import audition  # noqa: E402
from beatagent.onsets import flux_envelope, kick_flux_near, onsets_near  # noqa: E402

OUT = ROOT / "evals" / "results" / "real"


def slug(p: Path) -> str:
    return "".join(c if c.isalnum() else "_" for c in p.stem)[:40].strip("_")


def independent_tempo(mono: np.ndarray, sr: int, grid, win: float = 20.0) -> dict:
    """Tempo per 20 s window from onset autocorrelation -- does NOT use the
    neural tracker -- compared with the grid's local tempo in the same window.
    Octave (x2, /2) disagreements are folded and reported separately."""
    fx, fsr = flux_envelope(mono, sr)
    # beat evidence relative to the WHOLE track's typical beat (same measure and
    # 0.3 threshold as analyze._trim_unsupported, validated there: drumless
    # breakdown 0.01-0.14, drummed beats >= 0.9). A local/relative onset test was
    # tried first and counted quiet pad notes as hits.
    w = int(0.025 * fsr)
    ev = np.array([fx[max(0, int(b * fsr) - w): int(b * fsr) + w + 1].max(initial=0.0) for b in grid.beats])
    has_onset = ev >= 0.3 * (np.median(ev) + 1e-12)
    k = int(0.2 * fsr)
    fx = fx - np.convolve(fx, np.ones(k) / k, "same")
    rows = []
    t = 0.0
    while t + win <= len(mono) / sr:
        inwin = (grid.beats >= t) & (grid.beats < t + win)
        gb = grid.beats[inwin]
        if len(gb) >= 8:
            support = float(has_onset[inwin].mean())
            g_bpm = 60.0 / np.median(np.diff(gb))
            seg = fx[int(t * fsr): int((t + win) * fsr)]
            ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
            lags = np.arange(int(fsr * 60 / (g_bpm * 1.15)), int(fsr * 60 / (g_bpm * 0.87)) + 1)
            lag = lags[np.argmax(ac[lags])]
            # parabolic refinement of the autocorrelation peak
            y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
            den = y0 - 2 * y1 + y2
            lag_f = lag + (0.5 * (y0 - y2) / den if den else 0.0)
            i_bpm = 60.0 * fsr / lag_f
            strength = float(ac[lag] / (ac[0] + 1e-12))       # 1 = perfectly periodic, ~0 = no pulse
            rows.append({"t": t, "grid_bpm": round(float(g_bpm), 2), "indep_bpm": round(float(i_bpm), 2),
                         "diff_pct": round(float(100 * (i_bpm - g_bpm) / g_bpm), 2),
                         "periodicity": round(strength, 3), "onset_support": round(support, 2)})
        t += win
    # Only windows with a real pulse are graded: >= 50 % of grid beats carry an onset
    # AND the autocorrelation has a periodic peak (periodicity >= 0.1). The second
    # rule comes from observed inconclusive windows (BUN MASKA 0.016, RINGTONE 0.05,
    # ALL CAPS 0.05-0.09) vs agreeing windows (0.24-0.64 real, >= 0.39 substitute).
    graded = [r for r in rows if r["onset_support"] >= 0.5 and r["periodicity"] >= 0.1]
    d = [abs(r["diff_pct"]) for r in graded]
    return {"windows": rows, "n_graded_windows": len(graded), "n_windows": len(rows),
            "max_abs_diff_pct": round(max(d), 2) if d else None,
            "all_within_1_5pct": bool(d) and max(d) <= 1.5}


def check(path: Path) -> dict:
    audio, fmt = load(path)
    t0 = time.time()
    g = analyze(audio, fmt.sr)
    el = time.time() - t0
    name = slug(path)
    g.save(OUT / f"{name}.grid.json")
    save(OUT / f"{name}.click.wav", audition(audio, fmt.sr, g), fmt.sr, "PCM_16", "WAV")

    mono = to_mono(audio)
    on = onsets_near(mono, fmt.sr, g.beats)
    off = (on - g.beats) * 1000
    found = ~np.isnan(off)
    by_pos = {}
    for q in range(1, g.beats_per_bar + 1):
        m = g.positions == q
        by_pos[str(q)] = round(float(found[m].mean()), 3) if m.any() else None
    # how tightly this song's own downbeat hits sit on its own grid (bounds what any
    # grid-based sync can achieve): broadband attack vs low-frequency kick moment
    d = g.downbeats
    att = (onsets_near(mono, fmt.sr, d) - d) * 1000
    kik = (kick_flux_near(mono, fmt.sr, d) - d) * 1000
    def spread(x):
        x = x[~np.isnan(x)]
        return None if len(x) < 5 else {"n": int(len(x)), "median_ms": round(float(np.median(x)), 2),
                                        "p10_p90_spread_ms": round(float(np.percentile(x, 90) - np.percentile(x, 10)), 2)}
    raw, _, _ = TRACKERS["beat_this"](mono, fmt.sr)
    idx = np.clip(np.searchsorted(raw, g.beats), 1, len(raw) - 1)
    d_raw = np.minimum(np.abs(raw[idx - 1] - g.beats), np.abs(raw[idx] - g.beats)) * 1000
    try:
        import librosa
        lt = float(np.atleast_1d(librosa.beat.beat_track(y=mono, sr=fmt.sr, hop_length=512)[0])[0])
    except Exception:  # noqa: BLE001
        lt = float("nan")
    ratio = lt / g.tempo_bpm if g.tempo_bpm else float("nan")
    tempo_flag = ("agree" if abs(ratio - 1) < 0.04 else
                  "librosa says DOUBLE" if abs(ratio - 2) < 0.08 else
                  "librosa says HALF" if abs(ratio - 0.5) < 0.04 else "DISAGREE")
    return {
        "file": path.name, "slug": name, "duration_s": round(len(mono) / fmt.sr, 1),
        "format": f"{fmt.sr} Hz {fmt.subtype} {fmt.channels}ch", "analysis_s": round(el, 1),
        "tempo_bpm": round(g.tempo_bpm, 2), "time_signature": g.time_signature,
        "grid_type": g.meta["grid_kind"], "n_bars": g.n_bars, "confidence": g.confidence.get("overall"),
        "trimmed": g.meta["fit"].get("trimmed_unsupported_beats"),
        "first_beat_s": round(float(g.beats[0]), 3),
        "onset_support": {"share_of_beats_with_onset": round(float(found.mean()), 3), "by_position": by_pos,
                          "median_offset_ms": round(float(np.nanmedian(off)), 2) if found.any() else None,
                          "p95_abs_offset_ms": round(float(np.nanpercentile(np.abs(off), 95)), 2) if found.any() else None},
        "grid_vs_raw_tracker_ms": {"median": round(float(np.median(d_raw)), 2),
                                   "share_over_35ms": round(float(np.mean(d_raw > 35)), 3)},
        "librosa_tempo": round(lt, 2), "tempo_check": tempo_flag,
        "independent_tempo": independent_tempo(mono, fmt.sr, g),
        "downbeat_attack_vs_own_grid": spread(att), "downbeat_lowkick_vs_own_grid": spread(kik),
        "n_tracked_beats": g.meta.get("n_tracked_beats"), "n_consistent_beats": g.meta.get("n_consistent_beats"),
        "tempo_octave": g.meta["fit"].get("tempo_octave"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sync", nargs=2, metavar=("A", "B"))
    ap.add_argument("--a-bar", type=int, default=9)
    ap.add_argument("--b-bar", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in args.files:
        try:
            r = check(Path(f))
        except ValueError as e:              # undecodable file: report it, keep going
            r = {"file": Path(f).name, "error": str(e)}
        rows.append(r)
        print(json.dumps(r))
    if args.sync:
        pa, pb = Path(args.sync[0]), Path(args.sync[1])
        aa, fa = load(pa)
        ab, fb = load(pb)
        ga, gb = analyze(aa, fa.sr), analyze(ab, fb.sr)
        try:
            plan = sync.plan_sync(ga, gb, args.a_bar, args.b_bar)
            r = sync.render(aa, fa.sr, ab, fb.sr, plan, ga, sync.matched_b_grid(plan, gb))
            m = sync.measure_alignment(r["a"], r["b"], r["sr"], ga, plan)
            dk = sync.measure_downbeat_kicks(r["a"], r["b"], r["sr"], ga, plan)
            da = sync.measure_downbeat_attacks(r["a"], r["b"], r["sr"], ga, plan)
            for k in ("a", "b", "mix"):
                save(OUT / f"sync_{slug(pa)}__{slug(pb)}_{k}.wav", r[k], r["sr"], "PCM_16", "WAV")
            res = {"a": pa.name, "b": pb.name, "plan": plan.to_dict(),
                   "alignment_any_onset": {k: v for k, v in m.items() if k != "offsets_ms"},
                   "alignment_downbeat_kicks": dk, "alignment_downbeat_attacks": da}
        except sync.SyncError as e:
            res = {"a": pa.name, "b": pb.name, "refused": str(e)}
        print(json.dumps(res))
        rows.append({"sync": res})
    (OUT / "report.json").write_text(json.dumps({"label": "REAL MUSIC, NO REFERENCE ANNOTATIONS: consistency "
                                                 "checks only, not accuracy", "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
