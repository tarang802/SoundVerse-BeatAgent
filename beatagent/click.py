"""Click audition: the original audio with a click on every beat and an
accented (higher, louder) click on downbeats."""
from __future__ import annotations

import numpy as np

from .grid import BeatGrid


def _click(sr: int, freq: float, dur: float = 0.03) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.006)).astype(np.float32)


def click_track(grid: BeatGrid, n: int, sr: int) -> np.ndarray:
    out = np.zeros(n, np.float32)
    weak, strong = _click(sr, 1000) * 0.45, _click(sr, 2000) * 0.8
    for t, pos in zip(grid.beats, grid.positions):
        c = strong if pos == 1 else weak
        i = int(round(t * sr))          # click starts exactly on the grid line
        j = min(n, i + len(c))
        if 0 <= i < n:
            out[i:j] += c[: j - i]
    return out


def audition(audio: np.ndarray, sr: int, grid: BeatGrid, music_gain: float = 0.6) -> np.ndarray:
    clicks = click_track(grid, audio.shape[0], sr)[:, None]
    mix = audio * music_gain + clicks
    peak = np.max(np.abs(mix))
    return (mix / peak * 0.98 if peak > 0.98 else mix).astype(np.float32)
