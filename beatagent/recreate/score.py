"""Part 3 score: the intermediate representation between analysis and playback.

A score is JSON. Note times are in BEATS (quarter notes at `tempo_bpm`), not seconds,
so a tempo change ("play it at 140") is one number and the score stays independent of
playback. seconds = beats * 60 / tempo_bpm. The loop is exactly `loop_beats` long;
every note starts inside [0, loop_beats).

{
  "version": 1,
  "tempo_bpm": 96.0, "time_signature": "4/4", "beats_per_bar": 4, "bars": 2, "loop_beats": 8,
  "source": {...where it came from...}, "analysis": {...how each part was measured...},
  "tracks": [
    {"name": "drums", "kind": "drums", "gain_db": 0, "mute": false, "patch": {...},
     "notes": [{"start": 0.0, "duration": 0.25, "pitch": 36, "velocity": 110}, ...]},
    {"name": "bass", "kind": "bass", ...}
  ]
}
Drum pitches are General MIDI: 36 kick, 38 snare, 42 closed hat, 46 open hat.
Notes come from signal analysis / a transcription model (beatagent/recreate/transcribe.py),
never from the LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

KINDS = ("drums", "bass", "keys", "lead")
DRUM_PITCHES = {36: "kick", 38: "snare", 42: "closed hat", 46: "open hat"}


class Note(BaseModel):
    start: float = Field(ge=0)              # beats from the loop start
    duration: float = Field(gt=0)           # beats
    pitch: int = Field(ge=0, le=127)        # MIDI note (drums: General MIDI)
    velocity: int = Field(ge=1, le=127)


class Patch(BaseModel):
    """Synth settings the page uses. Defaults per kind are filled by default_patch()."""
    wave: Literal["sine", "triangle", "sawtooth", "square"] = "sawtooth"
    cutoff_hz: float = Field(1200.0, gt=20, lt=20000)
    resonance: float = Field(1.0, ge=0.1, le=20)
    attack_s: float = Field(0.005, ge=0, le=2)
    decay_s: float = Field(0.25, gt=0, le=10)
    sustain: float = Field(0.5, ge=0, le=1)
    release_s: float = Field(0.08, ge=0, le=5)


class Track(BaseModel):
    name: str
    kind: Literal["drums", "bass", "keys", "lead"]
    gain_db: float = Field(0.0, ge=-60, le=12)
    mute: bool = False
    patch: Patch = Field(default_factory=Patch)
    notes: list[Note] = Field(default_factory=list)


class Score(BaseModel):
    version: int = 1
    tempo_bpm: float = Field(gt=20, lt=400)
    time_signature: str = "4/4"
    beats_per_bar: int = Field(4, ge=1, le=16)
    bars: int = Field(ge=1, le=64)
    loop_beats: float = Field(gt=0)
    source: dict = Field(default_factory=dict)
    analysis: dict = Field(default_factory=dict)
    tracks: list[Track] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        if abs(self.loop_beats - self.bars * self.beats_per_bar) > 1e-6:
            raise ValueError(f"loop_beats {self.loop_beats} != bars x beats_per_bar "
                             f"({self.bars} x {self.beats_per_bar})")
        for t in self.tracks:
            for n in t.notes:
                if n.start >= self.loop_beats:
                    raise ValueError(f"track '{t.name}': note at beat {n.start} is outside the "
                                     f"{self.loop_beats}-beat loop")
            if t.kind == "drums" and any(n.pitch not in DRUM_PITCHES for n in t.notes):
                raise ValueError(f"track '{t.name}': drum pitches must be one of {sorted(DRUM_PITCHES)}")
        names = [t.name for t in self.tracks]
        if len(set(names)) != len(names):
            raise ValueError("track names must be unique")
        return self

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def loop_seconds(self) -> float:
        return self.loop_beats * self.seconds_per_beat

    def note_events_seconds(self, track: str | None = None) -> list[tuple[float, float, int, int, str]]:
        """(start_s, end_s, pitch, velocity, track) for evaluation against MIDI."""
        spb = self.seconds_per_beat
        return [(n.start * spb, (n.start + n.duration) * spb, n.pitch, n.velocity, t.name)
                for t in self.tracks if track in (None, t.name) for n in t.notes]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Score":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def default_patch(kind: str) -> Patch:
    return {"drums": Patch(wave="sine", cutoff_hz=8000, decay_s=0.2, sustain=0.0),
            "bass": Patch(wave="sawtooth", cutoff_hz=700, resonance=4, decay_s=0.3, sustain=0.6),
            "keys": Patch(wave="triangle", cutoff_hz=3000, attack_s=0.003, decay_s=0.8, sustain=0.2, release_s=0.3),
            "lead": Patch(wave="square", cutoff_hz=2500, decay_s=0.3, sustain=0.5)}[kind]
