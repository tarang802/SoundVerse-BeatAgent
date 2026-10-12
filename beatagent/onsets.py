"""Small, dependency-light onset measurement (numpy + scipy only).

Used to *measure* things (kick alignment in sync renders, stretch checks,
grid phase refinement), so it favours timing precision over recall.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def band_envelope(x: np.ndarray, sr: int, lo: float | None, hi: float | None,
                  hop: int = 32, smooth_ms: float = 1.5) -> tuple[np.ndarray, float]:
    """Amplitude envelope of a band, sampled every `hop` samples. Returns (env, env_sr)."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    if lo and hi:
        sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    elif hi:
        sos = signal.butter(4, hi, btype="low", fs=sr, output="sos")
    elif lo:
        sos = signal.butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = None
    y = signal.sosfiltfilt(sos, x) if sos is not None else x   # zero-phase: no group delay
    e = np.abs(signal.hilbert(y)) if len(y) < 2_000_000 else np.abs(y)
    k = max(1, int(smooth_ms * 1e-3 * sr))
    e = np.convolve(e, np.ones(k) / k, mode="same")
    return e[::hop].astype(np.float64), sr / hop


def onset_times(x: np.ndarray, sr: int, lo=None, hi=None, min_gap_s=0.1,
                rel_thresh=0.15, hop=32) -> np.ndarray:
    """Onset times (s) at the steepest rise of the band envelope's log energy."""
    env, esr = band_envelope(x, sr, lo, hi, hop=hop)
    loge = np.log(env + 1e-4 * env.max() + 1e-12)
    d = np.diff(loge, prepend=loge[0])
    d[d < 0] = 0
    # require a real level, not just a log-ratio bump in silence
    level_ok = env > rel_thresh * np.percentile(env, 99)
    cand, _ = signal.find_peaks(d, distance=max(1, int(min_gap_s * esr)),
                                height=np.percentile(d, 99) * 0.3)
    cand = [c for c in cand if level_ok[min(len(env) - 1, c + int(0.01 * esr))]]
    return np.array(cand, dtype=float) / esr


def flux_envelope(x: np.ndarray, sr: int, hop: int = 64, n_fft: int = 512) -> tuple[np.ndarray, float]:
    """Log-magnitude spectral flux (half-wave rectified), centred frames."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    f, t, Z = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="even",
                          padded=True)
    S = np.log1p(100 * np.abs(Z))
    flux = np.maximum(0, np.diff(S, axis=1)).sum(axis=0)
    flux = np.concatenate([[0], flux])
    return flux, sr / hop


def match_offsets(a: np.ndarray, b: np.ndarray, max_dist: float = 0.05) -> np.ndarray:
    """For each onset in a, signed offset (b - a) to the nearest onset in b within max_dist."""
    if len(a) == 0 or len(b) == 0:
        return np.array([])
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    left, right = b[idx - 1], b[idx]
    nearest = np.where(np.abs(left - a) <= np.abs(right - a), left, right)
    d = nearest - a
    return d[np.abs(d) <= max_dist]


def onsets_near(x: np.ndarray, sr: int, times: np.ndarray, win: float = 0.04,
                min_ratio: float = 3.0) -> np.ndarray:
    """Onset time nearest each expected time: the spectral-flux peak within
    +-win (1.45 ms resolution). NaN where there is no clear peak (flux peak
    below min_ratio x the local median). Used to MEASURE alignment, e.g. kick
    vs kick in a sync render; measuring both sides the same way cancels the
    detector's own bias."""
    flux, fsr = flux_envelope(x, sr)
    out = np.full(len(times), np.nan)
    w = int(win * fsr)
    ctx = int(0.5 * fsr)
    for i, t in enumerate(times):
        c = int(round(t * fsr))
        a, b = max(0, c - w), min(len(flux), c + w + 1)
        if b <= a:
            continue
        seg = flux[a:b]
        k = int(np.argmax(seg))
        local = np.median(flux[max(0, c - ctx): c + ctx]) + 1e-9
        if seg[k] > min_ratio * local:
            # parabolic interpolation for sub-hop precision
            if 0 < k < len(seg) - 1:
                y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
                den = y0 - 2 * y1 + y2
                k = k + (0.5 * (y0 - y2) / den if den else 0.0)
            out[i] = (a + k) / fsr
    return out



def kick_flux_near(x: np.ndarray, sr: int, times: np.ndarray, win: float = 0.04,
                   fmax: float = 150.0, min_ratio: float = 4.0) -> np.ndarray:
    """KICK onset nearest each expected time: peak of LOW-FREQUENCY spectral flux
    (bins <= fmax Hz at ~21 Hz resolution, ~1.5 ms hop)
    within +-win. NaN where no low-band onset stands out (min_ratio x the track's
    median low-band flux) -- i.e. no kick on that beat, so it is not compared.
    This exists because the broadband onsets_near compares "any hit": on real
    hip-hop that paired A's kick with B's hi-hat."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    # Only < fmax Hz matters, so work at 4 kHz: same ~21 Hz bins and ~1.5 ms hop
    # as a 2048-point STFT at 44.1 kHz, ~20x less memory (the full-rate version
    # needed 3.8 GB on a 6-minute track and crashed).
    import soxr
    lsr = 4000
    y = soxr.resample(x.astype(np.float32), sr, lsr)
    n_fft, hop = 192, 6
    f, t, Z = signal.stft(y, fs=lsr, nperseg=n_fft, noverlap=n_fft - hop, boundary="even", padded=True)
    lo = (f >= 30) & (f <= fmax)
    S = np.log1p(100 * np.abs(Z[lo]))
    flux = np.concatenate([[0], np.maximum(0, np.diff(S, axis=1)).sum(axis=0)])
    fsr = lsr / hop
    ref = np.median(flux[flux > 0]) + 1e-12
    out = np.full(len(times), np.nan)
    w = int(win * fsr)
    for i, tt in enumerate(times):
        c = int(round(tt * fsr))
        a, b = max(1, c - w), min(len(flux) - 1, c + w + 1)
        if b - a < 3:
            continue
        seg = flux[a:b]
        k = int(np.argmax(seg))
        if seg[k] < min_ratio * ref:
            continue
        kf = float(k)
        if 0 < k < len(seg) - 1:
            y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
            den = y0 - 2 * y1 + y2
            if den:
                kf = k + 0.5 * (y0 - y2) / den
        out[i] = (a + kf) / fsr
    return out
