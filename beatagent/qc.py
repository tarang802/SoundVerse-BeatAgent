"""Quality checks on rendered edits (used by tests and the eval report)."""
from __future__ import annotations

import numpy as np
from scipy import signal


def _hf(x: np.ndarray, sr: int, cutoff: float = 6000.0) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    sos = signal.butter(4, cutoff, btype="high", fs=sr, output="sos")
    return np.abs(signal.sosfiltfilt(sos, x))


def _hf_window(x: np.ndarray, sr: int, t: float, pre: float, post: float, pad: float = 0.03) -> np.ndarray:
    """|>6 kHz| of x in [t - pre, t + post], filtered on a padded segment (no edge ringing)."""
    i0, i1 = max(0, int((t - pre - pad) * sr)), int((t + post + pad) * sr)
    h = _hf(x[i0:i1], sr)
    a = int((t - pre) * sr) - i0
    return h[max(0, a): max(0, a) + int((pre + post) * sr)]


def splice_click_db(out: np.ndarray, src: np.ndarray, sr: int, out_t: float,
                    src_ts: list[float], pre: float = 0.012, post: float = 0.001) -> float:
    """Worse of two position-matched windows (dB, > ~3 = click): the TRANSITION
    [t - pre, t - 1 ms], where the engine's crossfade happens, and the JUNCTION
    [t - 1 ms, t + post], where a hard cut would click. Splitting matters: engine
    splices sit on beats, and a single window reaching past t was dominated by the
    next kick's attack (every engine splice read exactly 0.0 dB)."""
    edge = 0.001
    return max(_window_db(out, src, sr, out_t, src_ts, pre, -edge),
               _window_db(out, src, sr, out_t, src_ts, edge, post))


def _window_db(out: np.ndarray, src: np.ndarray, sr: int, out_t: float,
               src_ts: list[float], pre: float, post: float) -> float:
    """How much louder the >6 kHz content is at a splice than in the source it came
    from (dB): the output window [t - pre, t + post] against the SAME window at each
    source point. > ~3 dB = an added click; a clean splice is <= ~0 dB.

    The window covers the engine's crossfade (8 ms ending 2 ms before the beat) and
    stops 1 ms after the splice. An earlier version compared a +-15 ms output peak with
    the loudest moment within +-50 ms of the source: a nearby hi-hat or kick attack
    was louder than a real click, so a hard cut through a bass note scored -4.7 dB
    (tests/test_click_metric.py). Position-matched windows fix that. A click at the
    very instant a drum hit starts is still masked -- as it is to the ear."""
    peak_out = float(_hf_window(out, sr, out_t, pre, post).max(initial=0.0))
    ref, level = 0.0, 0.0
    for t in src_ts:
        ref = max(ref, float(_hf_window(src, sr, t, pre, post).max(initial=0.0)))
        seg = src[max(0, int((t - 0.05) * sr)): int((t + 0.05) * sr)]
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
