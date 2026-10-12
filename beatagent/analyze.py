"""Beat/downbeat tracking and grid fitting.

Pipeline
  1. A neural tracker (Beat This!) proposes beats + downbeats. Its output is
     quantised to 20 ms frames, so it cannot meet a 10 ms drift target alone.
  2. Each beat gets an integer grid index (gaps from missed beats are bridged
     by rounding IBI / median period).
  3. Constant-tempo test: robust line fit time = a + b * index. If residuals
     are small and show no slow trend, the grid IS that line (sub-ms period
     precision from ~hundreds of beats).
  4. Otherwise try piecewise-constant (tempo changes): binary segmentation of
     the (index, time) line. Otherwise variable: keep tracked beats, snapped
     to nearby spectral-flux peaks.
  5. Downbeat phase and meter from the tracker's downbeats by voting.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .audio_io import to_mono
from .grid import BeatGrid, tempo_segments_from_beats
from .onsets import flux_envelope

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
_MODEL_CACHE: dict = {}

# thresholds (seconds) -- see DECISIONS.md
OUTLIER = 0.035          # tracked beat further than this from the fitted line is ignored
CONST_P95 = 0.020        # p95 residual allowed for a constant grid
CONST_TREND = 0.012      # max |rolling median residual| allowed (detects slow drift)


# --------------------------------------------------------------------- trackers
def _checkpoint() -> str:
    for name in ("final0", "small0"):
        p = MODEL_DIR / f"{name}.ckpt"
        if p.exists() and p.stat().st_size > 5_000_000:
            return str(p)
    raise FileNotFoundError(f"No Beat This! checkpoint in {MODEL_DIR}. Run `make models`.")


def track_beat_this(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, str]:
    from beat_this.inference import Audio2Beats
    ckpt = _checkpoint()
    if ckpt not in _MODEL_CACHE:
        _MODEL_CACHE[ckpt] = Audio2Beats(checkpoint_path=ckpt, device="cpu", dbn=False)
    beats, downs = _MODEL_CACHE[ckpt](mono, sr)
    return np.asarray(beats, float), np.asarray(downs, float), f"beat_this:{Path(ckpt).stem}"


def track_librosa(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Baseline only. librosa has no downbeat tracker, so downbeats are every
    4th beat starting at the first beat (an explicit, documented guess)."""
    import librosa
    _, frames = librosa.beat.beat_track(y=mono, sr=sr, hop_length=256, units="frames")
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=256)
    return beats, beats[::4], "librosa"


TRACKERS = {"beat_this": track_beat_this, "librosa": track_librosa}


# --------------------------------------------------------------------- fitting
def _grid_indices(times: np.ndarray) -> np.ndarray:
    """Integer beat index per tracked beat; bridges missed beats."""
    ibi = np.diff(times)
    p = np.median(ibi)
    steps = np.maximum(1, np.round(ibi / p)).astype(int)
    # a step of 0 would mean a duplicate; those were removed by the caller
    return np.concatenate([[0], np.cumsum(steps)])


def _line_fit(k: np.ndarray, t: np.ndarray, iters: int = 4):
    keep = np.ones(len(t), bool)
    a = b = 0.0
    for _ in range(iters):
        if keep.sum() < 2:
            break
        b, a = np.polyfit(k[keep], t[keep], 1)
        res = t - (a + b * k)
        keep = np.abs(res) < OUTLIER
    res = t - (a + b * k)
    return a, b, res, keep


def _is_constant(res: np.ndarray, keep: np.ndarray) -> tuple[bool, dict]:
    r = res[keep]
    if len(r) < 8:
        return False, {"reason": "too few beats"}
    p95 = float(np.percentile(np.abs(r), 95))
    # slow drift = rolling median over FULL windows only. (Edge-padding the
    # series let 2-3 frame-quantised beats at the end masquerade as drift.)
    w = min(33, len(r))
    roll = np.array([np.median(r[i:i + w]) for i in range(len(r) - w + 1)])
    trend = float(np.max(np.abs(roll)))
    ok = p95 < CONST_P95 and trend < CONST_TREND and keep.mean() > 0.8
    return ok, {"p95_residual_ms": round(p95 * 1000, 2), "max_trend_ms": round(trend * 1000, 2),
                "inlier_frac": round(float(keep.mean()), 3)}


def _segment(k, t, depth=0, min_len=16):
    """Binary segmentation into constant-tempo pieces. Returns list of (i0, i1) index spans or None."""
    a, b, res, keep = _line_fit(k, t)
    ok, _ = _is_constant(res, keep)
    if ok:
        return [(0, len(t))]
    if depth >= 3 or len(t) < 2 * min_len:
        return None
    best, best_i = np.inf, None
    for i in range(min_len, len(t) - min_len, max(1, len(t) // 200)):
        sse = 0.0
        for sl in (slice(0, i), slice(i, len(t))):
            bb, aa = np.polyfit(k[sl], t[sl], 1)
            sse += float(np.sum((t[sl] - (aa + bb * k[sl])) ** 2))
        if sse < best:
            best, best_i = sse, i
    if best_i is None:
        return None
    # refine split locally
    lo, hi = max(min_len, best_i - len(t) // 200 - 1), min(len(t) - min_len, best_i + len(t) // 200 + 2)
    for i in range(lo, hi):
        sse = 0.0
        for sl in (slice(0, i), slice(i, len(t))):
            bb, aa = np.polyfit(k[sl], t[sl], 1)
            sse += float(np.sum((t[sl] - (aa + bb * k[sl])) ** 2))
        if sse < best:
            best, best_i = sse, i
    left = _segment(k[:best_i], t[:best_i], depth + 1, min_len)
    right = _segment(k[best_i:], t[best_i:], depth + 1, min_len)
    if left is None or right is None:
        return None
    return left + [(s + best_i, e + best_i) for s, e in right]


def _refine_phase(grid: np.ndarray, flux: np.ndarray, fsr: float, period: float) -> float:
    """Shift (s) that maximises summed onset strength at the grid lines.

    The tracker's own phase can't be trusted to 10 ms: Beat This! output is
    quantised to 20 ms frames and, on our synthetic material, sat 20-45 ms
    late depending on the kick sound. So we search up to +-60 ms (capped at
    20% of the beat period so we can't lock onto a 16th-note neighbour)."""
    search = min(0.06, 0.2 * period)
    shifts = np.arange(-search, search + 1e-9, 0.0005)
    xs = np.arange(len(flux)) / fsr
    scores = [np.interp(grid + s, xs, flux).sum() for s in shifts]
    return float(shifts[int(np.argmax(scores))])


def _snap(beats: np.ndarray, flux: np.ndarray, fsr: float, win=0.025) -> np.ndarray:
    out = beats.copy()
    for i, t in enumerate(beats):
        a, b = int((t - win) * fsr), int((t + win) * fsr) + 1
        a = max(a, 0)
        seg = flux[a:b]
        if len(seg) and seg.max() > 2 * np.median(flux[max(0, a - int(fsr)):b + int(fsr)]) + 1e-9:
            out[i] = (a + int(np.argmax(seg))) / fsr
    return out


def _fit_line(beats: np.ndarray, flux: np.ndarray, fsr: float) -> tuple[float, float, dict]:
    """Constant-tempo line t = a + b*n through tracked beats, refined on onsets."""
    k = _grid_indices(beats)
    a, b, res, keep = _line_fit(k, beats)
    info = _is_constant(res, keep)[1]
    g = a + b * k
    shift = _refine_phase(g, flux, fsr, b)
    a += shift
    # The period from 20 ms-quantised tracker frames is biased by ~0.1 ms/beat
    # (tens of ms over 6 min). Snap each line to its onset peak (1.45 ms
    # resolution), refit, re-phase.
    g = a + b * k
    snapped = _snap(g, flux, fsr, win=0.015)
    moved = np.abs(snapped - g) > 1e-9
    if moved.sum() >= max(8, 0.5 * len(g)):
        a2, b2, _, keep2 = _line_fit(k[moved], snapped[moved])
        if keep2.mean() > 0.8:
            a, b = a2, b2
            r2 = _refine_phase(a + b * k, flux, fsr, b * 0.05)   # +-1 % of a beat
            a += r2
            shift += r2
    # express the line with n = 0 at the first tracked beat
    info.update({"period_s": round(float(b), 6), "bpm": round(60 / float(b), 4),
                 "phase_shift_ms": round(shift * 1000, 2), "n_beats_fitted": int(len(beats))})
    return float(a + b * k[0]), float(b), info


def _refine_change(left, right, t_split: float, flux, fsr, span: int = 12) -> float:
    """Time of the last beat on the LEFT line before the tempo switches to the
    RIGHT line, chosen to maximise mean onset strength of the combined grid."""
    (aL, bL), (aR, bR) = left, right
    xs = np.arange(len(flux)) / fsr
    w0, w1 = t_split - (span + 4) * bL, t_split + (span + 4) * bR
    nL = np.arange(np.ceil((w0 - aL) / bL), np.floor((w1 - aL) / bL) + 1)
    lbeats = aL + bL * nL
    nR = np.arange(np.ceil((w0 - aR) / bR), np.floor((w1 - aR) / bR) + 1)
    rbeats = aR + bR * nR
    best, best_T = -np.inf, t_split
    for T in lbeats[(lbeats > t_split - span * bL) & (lbeats < t_split + span * bL)]:
        g = np.concatenate([lbeats[lbeats <= T + 1e-6], rbeats[rbeats > T + 0.5 * bR]])
        score = np.interp(g, xs, flux).mean()
        if score > best:
            best, best_T = score, T
    return float(best_T)


def _drop_silent(grid: np.ndarray, mono: np.ndarray, sr: int, db: float = -50.0) -> np.ndarray:
    """Remove grid beats at the start/end that fall in digital silence (the
    tracker happily emits beats before the music starts)."""
    win = int(0.1 * sr)
    peak = float(np.sqrt(np.mean(mono ** 2)) * 4 + 1e-12)
    def loud(t):
        i = int(t * sr)
        seg = mono[max(0, i - win): i + win]
        return seg.size and np.sqrt(np.mean(seg ** 2)) > peak * 10 ** (db / 20)
    lo, hi = 0, len(grid)
    while lo < hi and not loud(grid[lo]):
        lo += 1
    while hi > lo and not loud(grid[hi - 1]):
        hi -= 1
    return grid[lo:hi]


def _meter(grid: np.ndarray, downs: np.ndarray, flux, fsr) -> tuple[int, int, float, str]:
    """beats_per_bar, phase (index of first downbeat), agreement, time signature."""
    if len(downs) < 2 or len(grid) < 4:
        return 4, 0, 0.0, "4/4"
    idx = np.clip(np.searchsorted(grid, downs), 1, len(grid) - 1)
    near = np.where(np.abs(grid[idx - 1] - downs) < np.abs(grid[idx] - downs), idx - 1, idx)
    close = np.abs(grid[near] - downs) < 0.07
    near = near[close]
    if len(near) < 2:
        return 4, 0, 0.0, "4/4"
    gaps = np.diff(near)
    gaps = gaps[(gaps >= 2) & (gaps <= 7)]
    bpb = int(np.bincount(gaps).argmax()) if len(gaps) else 4
    votes = np.bincount(near % bpb, minlength=bpb)
    phase = int(votes.argmax())
    agree = float(votes.max() / max(1, votes.sum()))
    # compound vs simple: is the beat divided in 3 or in 2?
    xs = np.arange(len(flux)) / fsr
    span = np.diff(grid)
    g = grid[:-1]
    thirds = np.interp(np.concatenate([g + span / 3, g + 2 * span / 3]), xs, flux).mean()
    halves = np.interp(g + span / 2, xs, flux).mean()
    compound = thirds > 1.3 * halves
    if compound:
        ts = {2: "6/8", 3: "9/8", 4: "12/8"}.get(bpb, f"{bpb * 3}/8")
    else:
        ts = f"{bpb}/4"
    return bpb, phase % bpb, agree, ts


def analyze(audio: np.ndarray, sr: int, tracker: str = "beat_this") -> BeatGrid:
    t_start = time.time()
    mono = to_mono(audio).astype(np.float32)
    duration = len(mono) / sr
    beats, downs, tracker_name = TRACKERS[tracker](mono, sr)
    t_track = time.time() - t_start
    beats = np.unique(np.round(beats, 4))
    if len(beats) < 4:
        raise ValueError(f"Only {len(beats)} beats detected; no usable grid (is there a pulse?)")
    flux, fsr = flux_envelope(mono, sr)

    k = _grid_indices(beats)
    spans = _segment(k, beats)
    fit_info: dict = {}
    if spans is not None and len(spans) <= 4:
        # 1. a refined line per segment, fitted on the segment's core so beats
        #    near a tempo change (where the tracker smears) don't bias it
        lines = []
        for si, (s, e) in enumerate(spans):
            trim = 8 if len(spans) > 1 else 0
            s2 = s + (trim if si > 0 else 0)
            e2 = e - (trim if si + 1 < len(spans) else 0)
            if e2 - s2 < 8:
                s2, e2 = s, e
            a, b, info = _fit_line(beats[s2:e2], flux, fsr)
            lines.append((a, b))
            fit_info[f"segment_{si}"] = info
        # 2. change points: where switching from one line to the next best
        #    explains the onsets (the tracker itself drifts early, see DECISIONS.md)
        # outer edges get half a beat of slack: the tracker's first/last beat
        # can sit up to ~45 ms off the onset-refined line
        bounds = [beats[0] - 0.5 * lines[0][1]]
        for si in range(len(spans) - 1):
            T = _refine_change(lines[si], lines[si + 1], beats[spans[si + 1][0]], flux, fsr)
            bounds.append(T)
            fit_info[f"change_{si}"] = {"tracker_split_s": round(float(beats[spans[si + 1][0]]), 3),
                                        "refined_s": round(float(T), 4)}
        bounds.append(beats[-1] + 0.5 * lines[-1][1])
        grid_parts, change_idx = [], []
        for si, (a, b) in enumerate(lines):
            lo, hi = bounds[si], bounds[si + 1]
            if si > 0:
                lo = lo + 0.5 * b             # the change beat itself belongs to the previous line
            n0, n1 = int(np.ceil((lo - a) / b - 1e-9)), int(np.floor((hi - a) / b + 1e-9))
            g = a + b * np.arange(n0, n1 + 1)
            if si > 0:
                change_idx.append(sum(len(p) for p in grid_parts))
            grid_parts.append(g)
        grid = np.concatenate(grid_parts)
        constant = len(spans) == 1
        grid_kind = "constant" if constant else "piecewise_constant"
    else:
        # variable tempo: tracked beats, gaps filled, snapped to onsets
        filled = [beats[0]]
        for i in range(1, len(beats)):
            n = int(k[i] - k[i - 1])
            for j in range(1, n):
                filled.append(beats[i - 1] + (beats[i] - beats[i - 1]) * j / n)
            filled.append(beats[i])
        grid = _snap(np.array(filled), flux, fsr)
        constant, change_idx, grid_kind = False, [], "variable"
    grid = grid[(grid >= 0) & (grid < duration)]
    grid = _drop_silent(grid, mono, sr)

    bpb, phase, down_agree, ts = _meter(grid, downs, flux, fsr)
    positions = ((np.arange(len(grid)) - phase) % bpb) + 1

    # confidence: agreement between fitted grid and raw tracker output (heuristic, not a probability)
    idx = np.clip(np.searchsorted(beats, grid), 1, len(beats) - 1)
    dist = np.minimum(np.abs(beats[idx - 1] - grid), np.abs(beats[idx] - grid))
    beat_agree = float(np.mean(dist < 0.05))
    conf = {"beat_agreement": round(beat_agree, 3), "downbeat_agreement": round(down_agree, 3),
            "overall": round(beat_agree * down_agree, 3),
            "note": "heuristic: share of grid beats confirmed by the tracker x share of tracker downbeats "
                    "that agree with the chosen bar phase; not a calibrated probability"}
    segs = tempo_segments_from_beats(grid, positions, change_idx) if len(grid) > 2 else []
    return BeatGrid(
        beats=grid, positions=positions, beats_per_bar=bpb, time_signature=ts, duration=duration,
        constant_tempo=constant, segments=segs, confidence=conf,
        meta={"tracker": tracker_name, "grid_kind": grid_kind, "fit": fit_info,
              "n_tracked_beats": int(len(beats)), "n_tracked_downbeats": int(len(downs)),
              "tracking_s": round(t_track, 2), "total_s": round(time.time() - t_start, 2)},
    )
