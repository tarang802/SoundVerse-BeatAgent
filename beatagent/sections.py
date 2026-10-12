"""Measured song sections, so subjective requests ("chop out the boring part")
are answered with evidence instead of an LLM guess.

Per bar we MEASURE: loudness (RMS dB, relative to the track median), drum
activity (onset strength at the bar's beats, relative to the track's typical
beat -- the same evidence measure the grid uses), and low end (< 150 Hz RMS
dB, relative). Consecutive bars with the same coarse label are grouped into
sections. The LLM never sees audio; it only sees this table and must ask the
user to confirm before anything subjective is cut.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .audio_io import to_mono
from .grid import BeatGrid
from .onsets import flux_envelope

DRUMS_ON = 0.3        # drum activity >= 0.3 x the typical beat = drums present (same as grid trimming)
QUIET_DB = -6.0       # bar >= 6 dB below the track median = quiet
MIN_SECTION_BARS = 2  # shorter runs are merged into the previous section


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def bar_features(audio: np.ndarray, sr: int, grid: BeatGrid) -> list[dict]:
    mono = to_mono(audio)
    flux, fsr = flux_envelope(mono, sr)
    sos = signal.butter(4, 150, btype="low", fs=sr, output="sos")
    low = signal.sosfiltfilt(sos, mono)
    w = int(0.025 * fsr)
    ev = np.array([flux[max(0, int(t * fsr) - w): int(t * fsr) + w + 1].max(initial=0.0) for t in grid.beats])
    typical = float(np.median(ev)) + 1e-12
    bars = grid.bars
    rows = []
    for b in range(1, grid.n_bars + 1):
        t0, t1 = grid.bar_start(b), grid.bar_end(b)
        i0, i1 = int(t0 * sr), int(t1 * sr)
        if i1 - i0 < sr * 0.2:
            continue
        rows.append({"bar": b, "start_s": round(t0, 3), "end_s": round(t1, 3),
                     "loud_db": _db(mono[i0:i1]), "low_db": _db(low[i0:i1]),
                     "drums": float(np.mean(ev[bars == b]) / typical) if np.any(bars == b) else 0.0})
    if rows:
        med_loud = float(np.median([r["loud_db"] for r in rows]))
        med_low = float(np.median([r["low_db"] for r in rows]))
        for r in rows:
            r["loud_db"] -= med_loud
            r["low_db"] -= med_low
    return rows


def _label(r: dict) -> str:
    drums = "drums" if r["drums"] >= DRUMS_ON else "no drums"
    level = "quiet" if r["loud_db"] <= QUIET_DB else "full"
    return f"{drums}, {level}"


def sections(audio: np.ndarray, sr: int, grid: BeatGrid) -> list[dict]:
    rows = bar_features(audio, sr, grid)
    if not rows:
        return []
    groups: list[list[dict]] = []
    for r in rows:
        if groups and _label(groups[-1][-1]) == _label(r):
            groups[-1].append(r)
        else:
            groups.append([r])
    # merge very short runs (a fill, a one-bar drop) into the previous section
    merged: list[list[dict]] = []
    for g in groups:
        if merged and len(g) < MIN_SECTION_BARS:
            merged[-1].extend(g)
        else:
            merged.append(g)
    out = []
    for g in merged:
        out.append({
            "bars": [g[0]["bar"], g[-1]["bar"]], "n_bars": len(g),
            "start_s": g[0]["start_s"], "end_s": g[-1]["end_s"],
            "label": max(set(_label(r) for r in g), key=lambda lab: sum(_label(r) == lab for r in g)),
            "loudness_db_vs_track": round(float(np.mean([r["loud_db"] for r in g])), 1),
            "low_end_db_vs_track": round(float(np.mean([r["low_db"] for r in g])), 1),
            "drum_activity": round(float(np.mean([r["drums"] for r in g])), 2),
        })
    return out
