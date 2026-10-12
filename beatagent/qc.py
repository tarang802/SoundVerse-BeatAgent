"""Quality checks on rendered edits (used by tests and the eval report)."""
from __future__ import annotations

import numpy as np
from scipy import signal


def _hf(x: np.ndarray, sr: int, cutoff: float = 6000.0) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    sos = signal.butter(4, cutoff, btype="high", fs=sr, output="sos")
    return np.abs(signal.sosfiltfilt(sos, x))


def splice_click_db(out: np.ndarray, src: np.ndarray, sr: int, out_t: float,
                    src_ts: list[float], win: float = 0.015, ctx: float = 0.05) -> float:
    """How much louder the >6 kHz content is at a splice than in the source
    material it came from (dB). A click is broadband, so a hard cut shows up
    as a large positive number; a clean splice is <= ~0 dB."""
    ho = _hf(out[max(0, int((out_t - ctx) * sr)): int((out_t + ctx) * sr)], sr)
    c = int(min(ctx, out_t) * sr)
    peak_out = ho[max(0, c - int(win * sr)): c + int(win * sr)].max(initial=0.0)
    ref, level = 0.0, 0.0
    for t in src_ts:
        seg = src[max(0, int((t - ctx) * sr)): int((t + ctx) * sr)]
        ref = max(ref, _hf(seg, sr).max(initial=0.0))
        level = max(level, float(np.abs(seg).max(initial=0.0)))
    floor = 1e-3 * level + 1e-6     # anything below -60 dB re. the local peak counts as silent
    return float(20 * np.log10((peak_out + floor) / (ref + floor)))


def splice_points(state, sr: int) -> list[tuple[float, list[float]]]:
    """(output time, [source times on either side]) for every non-contiguous join."""
    pts, pos = [], 0
    for i, p in enumerate(state.pieces):
        if i and state.pieces[i - 1].src_end != p.src_start:
            pts.append((pos / sr, [state.pieces[i - 1].src_end / sr, p.src_start / sr]))
        pos += p.length
    return pts
