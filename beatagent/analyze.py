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

import os
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
CONST_P95 = 0.030        # p95 residual allowed for a constant grid. 20 ms (tuned on synthetic audio)
                         # rejected a constant 140 BPM real track at 21 ms: Beat This! frames are
                         # 20 ms (+-10 ms) plus real-mix jitter. Drift is caught by CONST_TREND.
CONST_TREND = 0.012      # max |rolling median residual| allowed (detects slow drift)
MAX_LOCAL_SPREAD_PCT = 2.0   # max tempo variation across 32-beat windows for a constant grid
# Preferred tempo octave [BPM_MIN, 2*BPM_MIN), like the BPM-range setting in DJ software.
# Beat This! sometimes reports half-time on hip-hop (real track Naam Kaam Sheher: 70 BPM,
# published 140; the brief's A2 is literally "hip-hop with a half-time feel"). 80 keeps
# boom-bap (85-95) and house as they are; drum & bass (~174) would become 87, so it is
# configurable via BEAT_AGENT_BPM_MIN. Compound meters (6/8) are never doubled/halved.
BPM_MIN = float(os.environ.get("BEAT_AGENT_BPM_MIN", "80"))
MIN_SEGMENT_BEATS = 32       # a tempo section shorter than 8 bars of 4/4 is not trusted
MIN_TEMPO_CHANGE = 0.01      # adjacent sections closer than 1 % are one tempo (phase jumps / noise)
MAX_OUTLIER_RUN = 20         # beats; a long unbroken run that misses the line is drift, not noise.
                             # Measured: drifting A3 = 25, constant tracks 0-14 (thin margin).


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
def _base_period(ibi: np.ndarray) -> float:
    """Beat period that explains the most inter-beat intervals as whole numbers
    of beats. On real hip-hop (Seedhe Maut - Nanchaku) the tracker alternates
    between full-time (430 ms) and half-time (860 ms) sections, so the plain
    median IBI can land on either; we prefer the LARGEST candidate that still
    explains nearly as many intervals as the best one (keep the tracker's tempo
    octave unless the data contradicts it)."""
    med = float(np.median(ibi))
    scores = {}
    for p in (med, med / 2, med / 3, med * 2):
        x = ibi / p
        n = np.round(x)
        scores[p] = float(np.mean((n >= 1) & (np.abs(x - n) < 0.15)))
    best = max(scores.values())
    return max(p for p, s in scores.items() if s >= 0.9 * best)


def _clean_beats(times: np.ndarray, tol: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Integer beat index for each tracked beat, measured from the last TRUSTED
    beat. A beat whose gap is not close to a whole number of beats (a stray hit
    79 ms after a real one, an off-beat) is dropped instead of being counted --
    counting it shifted the index of every later beat by one and broke the fit
    on real music. Gaps of 2, 3... beats (missed beats, half-time sections) are
    bridged.

    Re-anchoring: if two consecutive rejected beats are a whole number of beats
    apart from EACH OTHER, the tracker has changed phase (seen on a real track,
    RINGTONE: a 0.3-beat jump at 7 s); we continue from the new phase instead of
    rejecting every later beat. Returns (kept_times, indices)."""
    ibi = np.diff(times)
    p = _base_period(ibi)
    for _ in range(2):                         # second pass with the refined period
        keep_t, keep_k = [times[0]], [0]
        steps_s, steps_n = [], []
        pending = None                         # last rejected beat, candidate for re-anchoring
        for t in times[1:]:
            x = (t - keep_t[-1]) / p
            n = int(round(x))
            if n >= 1 and abs(x - n) < tol:
                keep_t.append(t)
                keep_k.append(keep_k[-1] + n)
                if n <= 4:
                    steps_s.append(t - keep_t[-2])
                    steps_n.append(n)
                pending = None
                continue
            if pending is not None:
                y = (t - pending) / p
                m = int(round(y))
                if m >= 1 and abs(y - m) < tol:
                    # two rejected beats agree with each other: new phase
                    k_pending = keep_k[-1] + max(1, int(round((pending - keep_t[-1]) / p)))
                    keep_t += [pending, t]
                    keep_k += [k_pending, k_pending + m]
                    pending = None
                    continue
            pending = t
        if steps_s:
            # mean (not median) of the per-beat steps near the median: the tracker's
            # 20 ms frames make single steps alternate (860/880 ms); their mean
            # recovers the true period (~863.5 ms) that the median cannot.
            per = np.array(steps_s) / np.array(steps_n)
            med = np.median(per)
            p = float(per[np.abs(per - med) < 0.1 * med].mean())
    return np.array(keep_t), np.array(keep_k)


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


def _local_tempo_spread(k: np.ndarray, t: np.ndarray, win: int = 32) -> float | None:
    """Spread of tempo fitted over 32-beat windows: (p90 - p10) / median, in %."""
    if len(t) < 3 * win:
        return None
    bpm = [60.0 / np.polyfit(k[i:i + win], t[i:i + win], 1)[0] for i in range(0, len(t) - win, win // 2)]
    return float(100 * (np.percentile(bpm, 90) - np.percentile(bpm, 10)) / np.median(bpm))


def _longest_outlier_run(keep: np.ndarray, k: np.ndarray) -> int:
    """Longest span (in grid beats) covered by consecutive non-inlier beats."""
    best, i = 0, 0
    while i < len(keep):
        if not keep[i]:
            j = i
            while j + 1 < len(keep) and not keep[j + 1]:
                j += 1
            best = max(best, int(k[j] - k[i] + 1))
            i = j + 1
        else:
            i += 1
    return best


def _is_constant(res: np.ndarray, keep: np.ndarray, k: np.ndarray | None = None,
                 t: np.ndarray | None = None) -> tuple[bool, dict]:
    r = res[keep]
    if len(r) < 8:
        return False, {"reason": "too few beats"}
    p95 = float(np.percentile(np.abs(r), 95))
    # slow drift = rolling median over FULL windows only. (Edge-padding the
    # series let 2-3 frame-quantised beats at the end masquerade as drift.)
    w = min(33, len(r))
    roll = np.array([np.median(r[i:i + w]) for i in range(len(r) - w + 1)])
    trend = float(np.max(np.abs(roll)))
    # Local tempo spread is the test that actually separates a drifting take
    # (synthetic live band A3: 3.8 %) from constant tracks (synthetic 0.1-1.1 %,
    # real hip-hop 0.1-0.8 %); the residual tests alone could not.
    spread = _local_tempo_spread(k[keep], t[keep]) if k is not None and t is not None else None
    # Outlier rejection can hide drift: on A3 the beats of a 112->118 BPM ramp were
    # all discarded as outliers and the rest looked constant. Scattered outliers
    # are noise; one long unbroken run of them is a region the line doesn't fit.
    run = _longest_outlier_run(keep, k) if k is not None else 0
    ok = (p95 < CONST_P95 and trend < CONST_TREND and keep.mean() > 0.8
          and (spread is None or spread < MAX_LOCAL_SPREAD_PCT) and run < MAX_OUTLIER_RUN)
    return ok, {"p95_residual_ms": round(p95 * 1000, 2), "max_trend_ms": round(trend * 1000, 2),
                "inlier_frac": round(float(keep.mean()), 3),
                "local_tempo_spread_pct": None if spread is None else round(spread, 2),
                "longest_outlier_run_beats": int(run)}


def _segment(k, t, depth=0, min_len=16):
    """Binary segmentation into constant-tempo pieces. Returns list of (i0, i1) index spans or None."""
    a, b, res, keep = _line_fit(k, t)
    ok, _ = _is_constant(res, keep, k, t)
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


def _fit_line(beats: np.ndarray, k: np.ndarray, flux: np.ndarray, fsr: float) -> tuple[float, float, dict]:
    """Constant-tempo line t = a + b*n through tracked beats (with their integer
    indices k from _clean_beats), refined on onsets."""
    k = k - k[0]
    a, b, res, keep = _line_fit(k, beats)
    info = _is_constant(res, keep, k, beats)[1]
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


def _support(grid: np.ndarray, flux, fsr) -> float:
    """Share of grid beats with onset evidence >= WEAK_EVIDENCE x the median
    (same measure as _trim_unsupported)."""
    w = int(0.025 * fsr)
    ev = np.array([flux[max(0, int(t * fsr) - w): int(t * fsr) + w + 1].max(initial=0.0) for t in grid])
    return float(np.mean(ev >= WEAK_EVIDENCE * (np.median(ev) + 1e-12)))


def _phase_contrast(grid: np.ndarray, flux, fsr) -> float:
    """Mean onset strength ON the grid's beats divided by that at the midpoints
    BETWEEN them. On-beats normally carry the stronger hits, so a grid that has
    slipped onto the off-beats (hi-hats, the "and") scores lower even though it
    still lands on hits -- which is why plain hit "support" could not judge phase."""
    w = int(0.025 * fsr)
    def ev(ts):
        return np.array([flux[max(0, int(t * fsr) - w): int(t * fsr) + w + 1].max(initial=0.0) for t in ts])
    mids = (grid[:-1] + grid[1:]) / 2
    return float(ev(grid).mean() / (ev(mids).mean() + 1e-12))


def _constant_by_onsets(beats: np.ndarray, k: np.ndarray, flux, fsr) -> tuple[np.ndarray, dict]:
    """A constant grid when the tracker's PHASE is unreliable but its TEMPO is
    not (real track Naam Kaam Sheher: local tempo 139.8-140.3 BPM throughout,
    but a 38-beat stretch half a beat off, so line fitting and segmentation
    failed). Period = median of 32-beat local tempos; phase = the offset whose
    grid lands on the most onset energy over the whole track; then refined on
    snapped onsets like _fit_line."""
    w = 32
    per = [np.polyfit(k[i:i + w], beats[i:i + w], 1)[0] for i in range(0, max(1, len(beats) - w), w // 2)]
    p = float(np.median(per))
    xs = np.arange(len(flux)) / fsr
    n = np.arange(int(np.floor((beats[-1] - beats[0]) / p)) + 1)
    offs = np.arange(0, p, 0.001)
    scores = [np.interp(beats[0] + o + p * n, xs, flux).mean() for o in offs]
    a = float(beats[0] + offs[int(np.argmax(scores))])
    g = a + p * n
    snapped = _snap(g, flux, fsr, win=0.015)
    moved = np.abs(snapped - g) > 1e-9
    if moved.sum() >= max(8, 0.5 * len(g)):
        a2, b2, _, keep2 = _line_fit(n[moved], snapped[moved])
        if keep2.mean() > 0.8:
            a, p = float(a2), float(b2)
    g = a + p * n
    if g[0] - p >= beats[0] - 0.5 * p:          # extend back to the first tracked beat
        g = np.concatenate([[g[0] - p], g])
    return g, {"period_s": round(p, 6), "bpm": round(60 / p, 4)}


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


WEAK_EVIDENCE = 0.3      # onset strength at a beat, relative to the track's median beat
MIN_LEADING_RUN = 8      # only trim a weak intro if it lasts at least 2 bars of 4/4


def _trim_unsupported(grid: np.ndarray, flux: np.ndarray, fsr: float) -> tuple[np.ndarray, dict]:
    """Drop grid beats at the START and END that no onset supports.

    The tracker keeps 'hearing' a pulse through drumless intros (A4: a 60 BPM
    pulse invented over a pad) and adds a beat in the reverb tail at the end.
    Measured on the substitute pack: real beats score 0.9-4.1 x the median,
    invented ones 0.00-0.14, so 0.3 separates them with margin. Beats in the
    MIDDLE are never trimmed: a breakdown must keep counting so bar numbers stay.
    Assumption (DECISIONS.md): an unverifiable beat is worse than no beat."""
    if len(grid) < 2 * MIN_LEADING_RUN:
        return grid, {"leading": 0, "trailing": 0}
    w = int(0.025 * fsr)
    ev = np.array([flux[max(0, int(t * fsr) - w): int(t * fsr) + w + 1].max(initial=0.0) for t in grid])
    strong = ev >= WEAK_EVIDENCE * (np.median(ev) + 1e-12)
    lo = int(np.argmax(strong)) if strong.any() else 0
    if lo < MIN_LEADING_RUN:
        lo = 0
    hi = len(grid) - int(np.argmax(strong[::-1])) if strong.any() else len(grid)
    return grid[lo:hi], {"leading": lo, "trailing": len(grid) - hi}


def _is_compound(grid: np.ndarray, flux, fsr) -> bool:
    """Is the beat divided in 3 (6/8, 12/8) rather than 2? Onset energy at the
    beat's thirds vs its half."""
    xs = np.arange(len(flux)) / fsr
    span = np.diff(grid)
    g = grid[:-1]
    thirds = np.interp(np.concatenate([g + span / 3, g + 2 * span / 3]), xs, flux).mean()
    halves = np.interp(g + span / 2, xs, flux).mean()
    return bool(thirds > 1.3 * halves)


def _double(grid: np.ndarray) -> np.ndarray:
    """Twice the tempo: insert the midpoint between consecutive beats."""
    mids = (grid[:-1] + grid[1:]) / 2
    return np.sort(np.concatenate([grid, mids]))


def _halve(grid: np.ndarray, flux, fsr) -> np.ndarray:
    """Half the tempo: keep every other beat, choosing the half that lands on
    more onset energy."""
    xs = np.arange(len(flux)) / fsr
    a, b = grid[0::2], grid[1::2]
    return a if np.interp(a, xs, flux).mean() >= np.interp(b, xs, flux).mean() else b


def _meter(grid: np.ndarray, downs: np.ndarray, flux, fsr, force_bpb: int | None = None) -> tuple[int, int, float, str]:
    """beats_per_bar, phase (index of first downbeat), agreement, time signature.
    force_bpb: after an octave change the tracker's downbeat spacing is in the old
    octave, so the bar length (4) is fixed and only the phase is voted."""
    if len(downs) < 2 or len(grid) < 4:
        return force_bpb or 4, 0, 0.0, f"{force_bpb or 4}/4"
    idx = np.clip(np.searchsorted(grid, downs), 1, len(grid) - 1)
    near = np.where(np.abs(grid[idx - 1] - downs) < np.abs(grid[idx] - downs), idx - 1, idx)
    close = np.abs(grid[near] - downs) < 0.07
    near = near[close]
    if len(near) < 2:
        return 4, 0, 0.0, "4/4"
    gaps = np.diff(near)
    gaps = gaps[(gaps >= 2) & (gaps <= 7)]
    bpb = force_bpb or (int(np.bincount(gaps).argmax()) if len(gaps) else 4)
    votes = np.bincount(near % bpb, minlength=bpb)
    phase = int(votes.argmax())
    agree = float(votes.max() / max(1, votes.sum()))
    if _is_compound(grid, flux, fsr):
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

    n_tracked = len(beats)
    raw_beats = beats                       # kept for the confidence score (must not grade itself)
    beats, k = _clean_beats(beats)
    if len(beats) < 4:
        raise ValueError("Too few consistent beats to build a grid (is there a steady pulse?)")
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
            a, b, info = _fit_line(beats[s2:e2], k[s2:e2], flux, fsr)
            lines.append((a, b))
            fit_info[f"segment_{si}"] = info
        # 1a. a tempo section needs >= MIN_SEGMENT_BEATS AND must not fit its
        #     neighbour's line. Real track Akatsuki: intro "segments" of 11 and 18
        #     beats at 77.6 / 78.5 BPM vs 78.0 -- their beats sit on the 78.0 line,
        #     so they are noise and get merged. Synthetic A3: a 112->118 BPM ramp is
        #     also short, but its beats miss the 118 line by up to ~250 ms, so it
        #     stays a separate section. (A plain length rule merged both.)
        def fits(span, line):
            a_, b_ = line
            x = (beats[span[0]:span[1]] - a_) / b_
            return float(np.percentile(np.abs(x - np.round(x)) * b_, 95)) < OUTLIER

        seg_info = [fit_info.pop(f"segment_{i}") for i in range(len(lines))]   # moves with `lines`
        while len(spans) > 1:
            cand = []
            for i, sp in enumerate(spans):
                if sp[1] - sp[0] >= MIN_SEGMENT_BEATS:
                    continue
                for j in (i - 1, i + 1):
                    if 0 <= j < len(spans) and fits(sp, lines[j]):
                        cand.append((i, j))
                        break
            if not cand:
                break
            i, j = cand[0]
            lo, hi = min(spans[i][0], spans[j][0]), max(spans[i][1], spans[j][1])
            spans[min(i, j)] = (lo, hi)
            lines[min(i, j)] = lines[j]
            seg_info[min(i, j)] = {**seg_info[j], "absorbed_short_section": True}
            del spans[max(i, j)], lines[max(i, j)], seg_info[max(i, j)]
        # 1b. "segments" with the same tempo (within MIN_TEMPO_CHANGE) are tracker
        #     phase jumps, not tempo changes (real track RINGTONE: 69.0 / 69.0 / 68.99 BPM). Use one line
        #     for the whole track, choosing the candidate phase that lands on the
        #     most onset energy across the WHOLE track.
        bpms = [60.0 / bb for _, bb in lines]
        if len(lines) > 1 and max(bpms) / min(bpms) < 1 + MIN_TEMPO_CHANGE:
            xs = np.arange(len(flux)) / fsr

            def score(line):
                a_, b_ = line
                n0 = int(np.ceil((beats[0] - a_) / b_))
                n1 = int(np.floor((beats[-1] - a_) / b_))
                return float(np.interp(a_ + b_ * np.arange(n0, n1 + 1), xs, flux).mean())

            scores = [score(ln) for ln in lines]
            best = int(np.argmax(scores))
            fit_info["merged_phase_jumps"] = {"segment_bpms": [round(x, 3) for x in bpms],
                                              "onset_score_per_candidate": [round(x, 4) for x in scores],
                                              "chosen_segment": best}
            lines, spans, seg_info = [lines[best]], [(0, len(beats))], [seg_info[best]]
        for i, inf in enumerate(seg_info):
            fit_info[f"segment_{i}"] = inf
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
                # the new tempo starts AT the boundary beat (the last beat of the
                # previous line): the interval after it is already the new tempo
                change_idx.append(sum(len(p) for p in grid_parts) - 1)
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
        # Before settling for "variable", test a constant grid with onset-chosen
        # phase: use it if its beats are at least as clearly ON the beat (phase
        # contrast) and it is not much worse at landing on hits (support).
        if len(beats) >= 64:
            cgrid, cinfo = _constant_by_onsets(beats, k, flux, fsr)
            c_var, c_const = _phase_contrast(grid, flux, fsr), _phase_contrast(cgrid, flux, fsr)
            s_var, s_const = _support(grid, flux, fsr), _support(cgrid, flux, fsr)
            choose = c_const >= c_var and s_const >= s_var - 0.10
            fit_info["constant_by_onsets"] = {**cinfo, "phase_contrast_constant": round(c_const, 3),
                                              "phase_contrast_variable": round(c_var, 3),
                                              "support_constant": round(s_const, 3),
                                              "support_variable": round(s_var, 3), "chosen": bool(choose)}
            if choose:
                grid, constant, grid_kind = cgrid, True, "constant"
    # Tempo-change points are kept as TIMES while beats are trimmed: as indices
    # they would point at the wrong beats once leading beats are removed.
    change_t = [float(grid[i]) for i in change_idx]
    grid = grid[(grid >= 0) & (grid < duration)]
    grid = _drop_silent(grid, mono, sr)
    grid, trimmed = _trim_unsupported(grid, flux, fsr)
    octave = {"tracker_bpm": round(float(60 / np.median(np.diff(grid))), 2), "factor": 1.0,
              "preferred_range": [BPM_MIN, 2 * BPM_MIN]}
    force_bpb = None
    grid_tracker_octave = grid          # confidence is judged at the tracker's octave (see below)
    if len(grid) > 8 and not _is_compound(grid, flux, fsr):
        bpm = 60 / np.median(np.diff(grid))
        while bpm < BPM_MIN:
            grid, bpm, octave["factor"] = _double(grid), bpm * 2, octave["factor"] * 2
        while bpm >= 2 * BPM_MIN:
            grid, bpm, octave["factor"] = _halve(grid, flux, fsr), bpm / 2, octave["factor"] / 2
        if octave["factor"] != 1.0:
            force_bpb = 4
    octave["reported_bpm"] = round(float(60 / np.median(np.diff(grid))), 2)
    change_idx = [int(np.searchsorted(grid, t - 1e-6)) for t in change_t if grid[0] < t <= grid[-1]]
    if grid_kind == "piecewise_constant" and not change_idx:
        # the only "tempo change" was into/out of a stretch we just trimmed
        constant, grid_kind = True, "constant"
    fit_info["trimmed_unsupported_beats"] = trimmed
    fit_info["tempo_octave"] = octave

    bpb, phase, down_agree, ts = _meter(grid, downs, flux, fsr, force_bpb)
    positions = ((np.arange(len(grid)) - phase) % bpb) + 1

    # confidence: agreement between fitted grid and raw tracker output (heuristic, not a probability)
    # Measured on the grid BEFORE any octave change: after doubling, half the beats
    # (the inserted midpoints) have no tracker beat and the score halved although
    # the grid was exactly as reliable (Toh Kya 1.00 -> 0.50).
    g_conf = grid_tracker_octave
    idx = np.clip(np.searchsorted(raw_beats, g_conf), 1, len(raw_beats) - 1)
    dist = np.minimum(np.abs(raw_beats[idx - 1] - g_conf), np.abs(raw_beats[idx] - g_conf))
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
              "n_tracked_beats": int(n_tracked), "n_consistent_beats": int(len(beats)),
              "n_tracked_downbeats": int(len(downs)),
              "tracking_s": round(t_track, 2), "total_s": round(time.time() - t_start, 2)},
    )
