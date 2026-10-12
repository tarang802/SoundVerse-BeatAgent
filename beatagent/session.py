"""An editing session on one track: source audio + source grid + edit history."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from . import edl
from .analyze import analyze
from .audio_io import AudioFormat, load, save
from .click import audition
from .edl import EditError, EditState, Piece
from .grid import BeatGrid

_OP_ADAPTER = TypeAdapter(edl.EditOp)


@dataclass
class TurnResult:
    ok: bool
    echo: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    duration_before: float = 0.0
    duration_after: float = 0.0


class Session:
    def __init__(self, audio_path: str | Path, workdir: str | Path, grid: BeatGrid | None = None,
                 tracker: str = "beat_this"):
        self.path = Path(audio_path)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.audio, self.fmt = load(self.path)
        self.sr = self.fmt.sr
        self.src_grid = grid or analyze(self.audio, self.sr, tracker=tracker)
        self.history: list[EditState] = [EditState((Piece(0, self.audio.shape[0]),))]
        self.labels: list[str] = ["original"]
        self.ids: list[int] = [0]          # unique per committed state (version numbers get reused after undo)
        self._next_id = 1
        self.pos = 0
        self._grid_cache: dict[int, BeatGrid] = {}

    # ------------------------------------------------------------ state
    @property
    def state(self) -> EditState:
        return self.history[self.pos]

    @property
    def state_id(self) -> int:
        return self.ids[self.pos]

    @property
    def grid(self) -> BeatGrid:
        if self.pos not in self._grid_cache:
            self._grid_cache[self.pos] = (self.src_grid if self.pos == 0
                                          else edl.map_grid(self.src_grid, self.state, self.sr))
        return self._grid_cache[self.pos]

    def set_octave(self, factor: float) -> None:
        """User override of the tempo octave (DJ-software style x2 / /2): half-time
        music is genuinely ambiguous (Naam Kaam Sheher reads as 70 or 140 to
        different listeners), so the user decides. Rebuilds the grid at that
        octave. Only on an unedited track: edits are addressed in beats, and
        changing what "a beat" means under existing edits would move them."""
        from .sync import octave_grid
        if factor not in (2.0, 0.5):
            raise EditError("Octave factor must be 2 or 0.5.")
        if self.pos != 0 or len(self.history) > 1:
            raise EditError("Change the tempo octave before editing (undo all edits first): "
                            "edits are addressed in beats and would move.")
        if self.src_grid.time_signature.endswith("/8"):
            raise EditError(f"{self.src_grid.time_signature} is a compound meter; doubling or halving "
                            "its beat would break the bar.")
        new = octave_grid(self.src_grid, factor)
        net = (self.src_grid.meta.get("user_octave_override") or 1.0) * factor   # x2 then /2 = back to auto
        new.meta = {k: v for k, v in new.meta.items() if k != "user_octave_override"}
        if net != 1.0:
            new.meta["user_octave_override"] = net
        self.src_grid = new
        self._grid_cache = {}

    # ------------------------------------------------------------ editing
    @staticmethod
    def parse_ops(raw_ops: list[dict]) -> list:
        """Validate untrusted (LLM) op dicts into typed ops. Raises EditError."""
        ops = []
        for o in raw_ops:
            clean = {k: v for k, v in o.items() if v is not None}
            try:
                ops.append(_OP_ADAPTER.validate_python(clean))
            except Exception as e:  # noqa: BLE001
                raise EditError(f"Couldn't turn that into a valid edit ({o.get('op', '?')}): "
                                f"{str(e).splitlines()[0]}") from e
        return ops

    def apply(self, ops: list, label: str = "") -> TurnResult:
        """Apply all ops of one chat turn atomically (all or nothing). Each op
        is resolved against the grid produced by the previous op."""
        before = self.grid.duration
        state, grid = self.state, self.grid
        echo, warns = [], []
        try:
            for i, op in enumerate(ops):
                r = edl.resolve(op, grid)
                state = edl.apply(state, r, self.sr)
                grid = edl.map_grid(self.src_grid, state, self.sr)
                prefix = f"{i + 1}. " if len(ops) > 1 else ""
                echo.append(prefix + r.echo)
                warns += r.warnings
        except EditError as e:
            return TurnResult(False, echo, warns, str(e), before, before)
        # commit: drop any redo branch
        self.history = self.history[: self.pos + 1] + [state]
        self.labels = self.labels[: self.pos + 1] + [label or "; ".join(echo)]
        self.ids = self.ids[: self.pos + 1] + [self._next_id]
        self._next_id += 1
        self._grid_cache = {k: v for k, v in self._grid_cache.items() if k <= self.pos}
        self.pos += 1
        self._grid_cache[self.pos] = grid
        return TurnResult(True, echo, warns, None, before, grid.duration)

    def undo(self) -> TurnResult:
        if self.pos == 0:
            return TurnResult(False, error="Nothing to undo: the track is in its original state.")
        undone = self.labels[self.pos]
        self.pos -= 1
        return TurnResult(True, [f"undid: {undone}"], duration_after=self.grid.duration)

    def redo(self) -> TurnResult:
        if self.pos >= len(self.history) - 1:
            return TurnResult(False, error="Nothing to redo.")
        self.pos += 1
        return TurnResult(True, [f"redid: {self.labels[self.pos]}"], duration_after=self.grid.duration)

    # ------------------------------------------------------------ output
    def render(self) -> np.ndarray:
        return edl.render(self.audio, self.state, self.sr)

    def export(self, stem: str | None = None, with_click: bool = True) -> dict:
        stem = stem or f"{self.path.stem}_v{self.pos}"
        out_audio = self.workdir / f"{stem}{self.fmt.out_ext}"
        y = self.render()
        stats = save(out_audio, y, self.sr, self.fmt.out_subtype, self.fmt.out_container)
        grid_path = self.workdir / f"{stem}.grid.json"
        self.grid.save(grid_path)
        paths = {"audio": str(out_audio), "grid": str(grid_path), **stats}
        if with_click:
            ck = self.workdir / f"{stem}.click.wav"
            save(ck, audition(y, self.sr, self.grid), self.sr, "PCM_16", "WAV")
            paths["click"] = str(ck)
        edl_path = self.workdir / f"{stem}.edl.json"
        with open(edl_path, "w") as f:
            json.dump({"source": str(self.path), "sr": self.sr, "subtype": self.fmt.out_subtype,
                       "history": self.labels[: self.pos + 1],
                       "pieces": [{"src_start": p.src_start, "src_end": p.src_end,
                                   "envs": [e.__dict__ for e in p.envs]} for p in self.state.pieces]},
                      f, indent=1)
        paths["edl"] = str(edl_path)
        return paths
