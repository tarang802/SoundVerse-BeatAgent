"""Beat grid data model and the indexing conventions used everywhere.

Indexing conventions (also in README):
  * Beats are 0-based. "beat N" is a grid LINE (a point in time).
    A beat range "beat A to beat B" is half-open [A, B): it starts on line A
    and stops on line B, so "beat 0 to beat 8" is exactly 8 beats.
  * Bars are 1-based and name a REGION. Bar k runs from its downbeat to the
    next downbeat. Bar ranges are inclusive: "bars 3-4" is two bars.
  * Beats before the first downbeat (a pickup) belong to bar 0.
  * position_in_bar is 1-based (1 = downbeat).
  * The last bar ends at the end of the audio.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeatGrid:
    beats: np.ndarray                 # beat times in seconds, strictly increasing
    positions: np.ndarray             # 1-based position in bar for every beat
    beats_per_bar: int
    time_signature: str
    duration: float                   # audio duration in seconds
    constant_tempo: bool
    segments: list[dict] = field(default_factory=list)   # tempo segments
    confidence: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ---------- derived ----------
    @property
    def n_beats(self) -> int:
        return int(len(self.beats))

    @property
    def bars(self) -> np.ndarray:
        """1-based bar number of every beat (0 for pickup beats)."""
        return np.cumsum(self.positions == 1).astype(int)

    @property
    def downbeat_idx(self) -> np.ndarray:
        return np.flatnonzero(self.positions == 1)

    @property
    def downbeats(self) -> np.ndarray:
        return self.beats[self.downbeat_idx]

    @property
    def n_bars(self) -> int:
        return int(len(self.downbeat_idx))

    @property
    def tempo_bpm(self) -> float:
        if self.n_beats < 2:
            return 0.0
        return float(60.0 / np.median(np.diff(self.beats)))

    def bar_start(self, bar: int) -> float:
        """Start time of 1-based bar (its downbeat)."""
        return float(self.downbeats[bar - 1])

    def bar_end(self, bar: int) -> float:
        """End of bar = next downbeat, or end of audio for the last bar."""
        return float(self.downbeats[bar]) if bar < self.n_bars else float(self.duration)

    def line_time(self, beat: int) -> float:
        """Time of grid line `beat`. Line n_beats is the end of the audio."""
        if beat == self.n_beats:
            return float(self.duration)
        return float(self.beats[beat])

    def bar_of_time(self, t: float) -> int:
        i = int(np.searchsorted(self.downbeats, t + 1e-6, side="right"))
        return i  # 0 = before first downbeat

    def beat_of_time(self, t: float) -> int:
        return int(np.searchsorted(self.beats, t + 1e-6, side="right")) - 1

    def local_bpm(self, beat: int) -> float:
        if self.n_beats < 2:
            return 0.0
        i = min(max(beat, 0), self.n_beats - 2)
        return float(60.0 / (self.beats[i + 1] - self.beats[i]))

    # ---------- serialisation ----------
    def to_dict(self) -> dict:
        bars = self.bars
        return {
            "conventions": {
                "beat_index": "0-based; a beat is a grid line (point in time)",
                "bar": "1-based; bar 0 = pickup beats before the first downbeat",
                "position_in_bar": "1-based; 1 = downbeat",
                "beat_range": "half-open [start, end) over grid lines",
                "bar_range": "inclusive; bar k spans its downbeat to the next downbeat",
                "times": "seconds from the start of this audio file",
            },
            "tempo_bpm": round(self.tempo_bpm, 3),
            "time_signature": self.time_signature,
            "beats_per_bar": self.beats_per_bar,
            "grid_type": "constant" if self.constant_tempo else "variable",
            "tempo_segments": self.segments,
            "confidence": self.confidence,
            "duration": round(self.duration, 6),
            "n_beats": self.n_beats,
            "n_bars": self.n_bars,
            "downbeats": [round(float(t), 6) for t in self.downbeats],
            "beats": [
                {"index": i, "time": round(float(t), 6), "bar": int(bars[i]),
                 "position": int(self.positions[i])}
                for i, t in enumerate(self.beats)
            ],
            "meta": self.meta,
        }

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "BeatGrid":
        return cls(
            beats=np.array([b["time"] for b in d["beats"]], dtype=float),
            positions=np.array([b["position"] for b in d["beats"]], dtype=int),
            beats_per_bar=d["beats_per_bar"], time_signature=d["time_signature"],
            duration=d["duration"], constant_tempo=d["grid_type"] == "constant",
            segments=d.get("tempo_segments", []), confidence=d.get("confidence", {}),
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path) -> "BeatGrid":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---------- summary for the LLM (facts only, all measured) ----------
    def summary(self) -> dict:
        return {
            "duration_s": round(self.duration, 3),
            "n_beats": self.n_beats,
            "last_beat_index": self.n_beats - 1,
            "n_bars": self.n_bars,
            "pickup_beats_before_bar_1": int(self.downbeat_idx[0]) if self.n_bars else self.n_beats,
            "time_signature": self.time_signature,
            "beats_per_bar": self.beats_per_bar,
            "tempo_bpm": round(self.tempo_bpm, 2),
            "grid_type": "constant" if self.constant_tempo else "variable",
            "tempo_segments": self.segments,
        }


def tempo_segments_from_beats(beats: np.ndarray, positions: np.ndarray,
                              change_beats: list[int]) -> list[dict]:
    """Describe tempo segments given the beat indices where a new tempo starts."""
    bars = np.cumsum(positions == 1)
    bounds = [0] + sorted(change_beats) + [len(beats)]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        ibi = np.diff(beats[a:b])
        segs.append({
            "start_beat": int(a), "end_beat": int(b - 1),
            "start_bar": int(bars[a]), "end_bar": int(bars[b - 1]),
            "start_time": round(float(beats[a]), 4),
            "bpm": round(float(60.0 / np.median(ibi)), 3),
        })
    return segs
