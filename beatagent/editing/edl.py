"""Typed edit operations on the beat grid, and the deterministic engine that applies them.

The LLM only ever produces `EditOp` objects (validated by pydantic). Everything
after that -- resolving beats/bars to seconds, range checks, splicing,
crossfades, re-deriving the grid -- is plain code in this file.

Model: an edited track is a list of `Piece`s, each a span of the ORIGINAL
source audio plus gain envelopes. Rendering concatenates pieces with short
pre-splice crossfades. The grid of the edited track is the source grid mapped
through the pieces, so it is exact by construction (no re-detection drift),
and undo/redo is just keeping old piece lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..core.grid import BeatGrid, tempo_segments_from_beats

XFADE_S = 0.008      # crossfade length at a splice
LEAD_S = 0.002       # crossfade ends this long BEFORE the splice point so the
                     # next piece's transient (on the grid line) is untouched
EDGE_FADE_S = 0.003  # tiny fade where the file starts/ends mid-signal


class EditError(ValueError):
    """A request that cannot be executed as stated. Message is user-facing."""


# ============================================================ typed operations
class Range(BaseModel):
    """A span of the CURRENT track.

    unit=beats: start/end are grid lines, half-open [start, end).
    unit=bars : start/end are 1-based bars, inclusive.
    start=None means the very start of the audio; end=None the very end.
    last=N means "the last N beats/bars" and overrides start/end.
    """
    unit: Literal["beats", "bars"]
    start: Optional[int] = None
    end: Optional[int] = None
    last: Optional[int] = Field(None, ge=1)


class Position(BaseModel):
    """An insertion point: a beat grid line, the start of a bar, or the end (index=None)."""
    unit: Literal["beats", "bars"]
    index: Optional[int] = None


class Remove(BaseModel):
    op: Literal["remove"] = "remove"
    range: Range


class KeepOnly(BaseModel):
    op: Literal["keep_only"] = "keep_only"
    range: Range


class Repeat(BaseModel):
    """Repeat a range in place. total_plays counts the original: 2 = original + 1 copy."""
    op: Literal["repeat"] = "repeat"
    range: Range
    total_plays: int = Field(ge=2, le=64)


class Move(BaseModel):
    """Cut a range out and re-insert it at `to` (position measured BEFORE the cut)."""
    op: Literal["move"] = "move"
    range: Range
    to: Position


class Copy(BaseModel):
    """Insert a copy of a range at `to` (original stays)."""
    op: Literal["copy"] = "copy"
    range: Range
    to: Position


class Fade(BaseModel):
    op: Literal["fade"] = "fade"
    range: Range
    direction: Literal["in", "out"]


class Gain(BaseModel):
    op: Literal["gain"] = "gain"
    range: Optional[Range] = None         # None = whole track
    db: float = Field(ge=-60, le=24)


EditOp = Annotated[Union[Remove, KeepOnly, Repeat, Move, Copy, Fade, Gain], Field(discriminator="op")]


# ============================================================ edit state
@dataclass(frozen=True)
class Env:
    """Gain envelope over a piece. kind=const: gain `a`. kind=fade_in/out: the
    piece covers fraction [f0, f1] of the fade."""
    kind: str
    f0: float = 0.0
    f1: float = 1.0
    a: float = 1.0

    def split(self, frac: float) -> tuple["Env", "Env"]:
        if self.kind == "const":
            return self, self
        m = self.f0 + (self.f1 - self.f0) * frac
        return replace(self, f1=m), replace(self, f0=m)

    def curve(self, n: int) -> np.ndarray:
        if self.kind == "const":
            return np.full(n, self.a, np.float32)
        f = np.linspace(self.f0, self.f1, n, endpoint=False, dtype=np.float64)
        g = np.sin(f * np.pi / 2) if self.kind == "fade_in" else np.cos(f * np.pi / 2)
        return g.astype(np.float32)

    def at_start(self) -> float:
        return float(self.curve(1)[0]) if self.kind != "const" else self.a


@dataclass(frozen=True)
class Piece:
    src_start: int          # samples into the source
    src_end: int
    envs: tuple = ()

    @property
    def length(self) -> int:
        return self.src_end - self.src_start


@dataclass
class EditState:
    pieces: tuple            # tuple[Piece, ...]

    @property
    def length(self) -> int:
        return sum(p.length for p in self.pieces)


def _split_at(pieces: list[Piece], s: int) -> list[Piece]:
    out, pos = [], 0
    for p in pieces:
        if pos < s < pos + p.length:
            k = s - pos
            frac = k / p.length
            e1, e2 = zip(*[e.split(frac) for e in p.envs]) if p.envs else ((), ())
            out.append(Piece(p.src_start, p.src_start + k, tuple(e1)))
            out.append(Piece(p.src_start + k, p.src_end, tuple(e2)))
        else:
            out.append(p)
        pos += p.length
    return out


def _span(pieces: list[Piece], a: int, b: int) -> tuple[list[Piece], int, int]:
    """Split so [a, b) is covered by whole pieces; return (pieces, i0, i1)."""
    pieces = _split_at(_split_at(pieces, a), b)
    pos, i0, i1 = 0, None, None
    for i, p in enumerate(pieces):
        if i0 is None and pos >= a:
            i0 = i
        pos += p.length
        if pos <= b:
            i1 = i + 1
    return pieces, i0 if i0 is not None else len(pieces), i1 if i1 is not None else 0


# ============================================================ resolution
@dataclass
class Resolved:
    op: BaseModel
    a: float                 # seconds in current track
    b: float
    to: Optional[float] = None
    echo: str = ""
    warnings: list = field(default_factory=list)


def _fmt_line(grid: BeatGrid, beat: int) -> str:
    if beat >= grid.n_beats:
        return "end"
    return f"bar {grid.bars[beat]}.{grid.positions[beat]}"


def resolve_range(r: Range, grid: BeatGrid) -> tuple[float, float, str, list[str]]:
    warn: list[str] = []
    if r.unit == "beats":
        n = grid.n_beats
        if r.last is not None:
            if r.last > n:
                raise EditError(f"The track only has {n} beats, so 'the last {r.last} beats' doesn't exist.")
            s, e = n - r.last, n
        else:
            s = 0 if r.start is None else r.start
            e = n if r.end is None else r.end
        if s < 0 or e > n:
            raise EditError(f"Beat range {s}-{e} is out of range: this track has beats 0-{n - 1} "
                            f"(grid line {n} = end of audio).")
        if e <= s:
            raise EditError(f"Beat range {s}-{e} is empty or reversed (ranges are [start, end), end exclusive).")
        a = 0.0 if (r.start is None and r.last is None) else grid.line_time(s)
        b = grid.line_time(e)
        label = (f"beats {s}-{e} [{s}, {e}) ({e - s} beats, {_fmt_line(grid, s)} to {_fmt_line(grid, e)})")
        if r.start is None and r.last is None and grid.n_beats and grid.beats[0] > 0.05:
            label += f", including the {grid.beats[0]:.3f} s before beat 0"
        elif s == 0 and grid.n_beats and grid.beats[0] > 0.05:
            # "cut beat 0 to beat 8" starts AT beat 0; say what happens to the audio before it
            label += f"; the {grid.beats[0]:.3f} s before beat 0 is kept (say 'from the start' to include it)"
        return a, b, label, warn
    nb = grid.n_bars
    if nb == 0:
        raise EditError("No downbeats were detected, so bar addressing isn't available; use beats instead.")
    if r.last is not None:
        if r.last > nb:
            raise EditError(f"The track only has {nb} bars, so 'the last {r.last} bars' doesn't exist.")
        s, e = nb - r.last + 1, nb
    else:
        s = 1 if r.start is None else r.start
        e = nb if r.end is None else r.end
    if s < 1 or e > nb:
        raise EditError(f"Bar range {s}-{e} is out of range: this track has bars 1-{nb}.")
    if e < s:
        raise EditError(f"Bar range {s}-{e} is reversed.")
    a = grid.bar_start(s) if not (r.start is None and r.last is None) else 0.0
    b = grid.bar_end(e) if r.end is not None or r.last is not None else grid.duration
    if r.start is None and r.last is None and grid.n_bars and grid.downbeats[0] > 0.05:
        pre = " (from 0.000 s, so the pickup before bar 1 is included)"
    else:
        pre = ""
    n_beats_in = int(np.sum((grid.beats >= a - 1e-6) & (grid.beats < b - 1e-6)))
    nb_ = e - s + 1
    label = (f"bar {s}" if nb_ == 1 else f"bars {s}-{e} inclusive") + f" ({nb_} bar{'s' if nb_ != 1 else ''}, {n_beats_in} beats){pre}"
    return a, b, label, warn


def resolve_position(p: Position, grid: BeatGrid) -> tuple[float, str]:
    if p.index is None:
        return grid.duration, "the end of the track"
    if p.unit == "beats":
        if not 0 <= p.index <= grid.n_beats:
            raise EditError(f"Beat {p.index} doesn't exist (beats 0-{grid.n_beats - 1}).")
        return grid.line_time(p.index), f"beat {p.index} ({_fmt_line(grid, p.index)})"
    if not 1 <= p.index <= grid.n_bars + 1:
        raise EditError(f"Bar {p.index} doesn't exist (bars 1-{grid.n_bars}).")
    if p.index == grid.n_bars + 1:
        return grid.duration, f"after the last bar (end of track)"
    return grid.bar_start(p.index), f"the start of bar {p.index}"


def _bar_aligned(grid: BeatGrid, t: float) -> bool:
    if t <= 1e-6 or t >= grid.duration - 1e-6:
        return True
    return bool(np.any(np.abs(grid.downbeats - t) < 1e-4))


def resolve(op: BaseModel, grid: BeatGrid) -> Resolved:
    if isinstance(op, Gain) and op.range is None:
        return Resolved(op, 0.0, grid.duration,
                        echo=f"gain {op.db:+.1f} dB on the whole track (0.000-{grid.duration:.3f} s)")
    a, b, label, warn = resolve_range(op.range, grid)
    span = f"{a:.3f}-{b:.3f} s ({b - a:.3f} s)"
    res = Resolved(op, a, b, warnings=warn)
    structural = isinstance(op, (Remove, KeepOnly, Repeat, Move, Copy))
    if structural and grid.n_bars and not (_bar_aligned(grid, a) and _bar_aligned(grid, b)):
        res.warnings.append("This range doesn't start and end on downbeats, so the bars after it "
                            "will no longer line up with the original bar structure.")
    if isinstance(op, Remove):
        if a <= 1e-6 and b >= grid.duration - 1e-6:
            raise EditError("That would remove the entire track.")
        res.echo = f"remove {label} = {span}"
    elif isinstance(op, KeepOnly):
        res.echo = f"keep only {label} = {span}; everything else is removed"
    elif isinstance(op, Repeat):
        extra = op.total_plays - 1
        res.echo = (f"repeat {label} = {span} so it plays {op.total_plays}x in total "
                    f"({extra} extra cop{'y' if extra == 1 else 'ies'} inserted right after it, "
                    f"+{extra * (b - a):.3f} s); the rest of the track follows")
    elif isinstance(op, (Move, Copy)):
        to, to_label = resolve_position(op.to, grid)
        if isinstance(op, Move) and a < to < b:
            raise EditError(f"Can't move {label} to a point inside itself ({to_label}).")
        res.to = to
        verb = "move" if isinstance(op, Move) else "copy"
        res.echo = f"{verb} {label} = {span} to {to_label} ({to:.3f} s in the track before the edit)"
        if not _bar_aligned(grid, to) and grid.n_bars:
            res.warnings.append("The destination isn't on a downbeat.")
    elif isinstance(op, Fade):
        res.echo = f"fade {op.direction} over {label} = {span} (equal-power curve)"
    elif isinstance(op, Gain):
        res.echo = f"gain {op.db:+.1f} dB on {label} = {span}"
    return res


# ============================================================ application
def apply(state: EditState, r: Resolved, sr: int) -> EditState:
    pieces = list(state.pieces)
    a, b = int(round(r.a * sr)), int(round(r.b * sr))
    op = r.op
    pieces, i0, i1 = _span(pieces, a, b)
    sel = pieces[i0:i1]
    if isinstance(op, Remove):
        new = pieces[:i0] + pieces[i1:]
    elif isinstance(op, KeepOnly):
        new = sel
    elif isinstance(op, Repeat):
        new = pieces[:i1] + sel * (op.total_plays - 1) + pieces[i1:]
    elif isinstance(op, (Move, Copy)):
        to = int(round(r.to * sr))
        pieces = _split_at(pieces, to)
        starts = np.cumsum([0] + [p.length for p in pieces[:-1]])
        sel_idx = {i for i, st in enumerate(starts) if a <= st < b}
        sel = [pieces[i] for i in sorted(sel_idx)]
        j = next((i for i, st in enumerate(starts) if st >= to), len(pieces))
        if isinstance(op, Copy):
            new = pieces[:j] + sel + pieces[j:]
        else:
            before = [p for i, p in enumerate(pieces[:j]) if i not in sel_idx]
            after = [p for i, p in enumerate(pieces) if i >= j and i not in sel_idx]
            new = before + sel + after
    elif isinstance(op, Fade):
        total = sum(p.length for p in sel)
        kind = "fade_in" if op.direction == "in" else "fade_out"
        pos, faded = 0, []
        for p in sel:
            f0, f1 = pos / total, (pos + p.length) / total
            faded.append(Piece(p.src_start, p.src_end, p.envs + (Env(kind, f0, f1),)))
            pos += p.length
        new = pieces[:i0] + faded + pieces[i1:]
    elif isinstance(op, Gain):
        g = float(10 ** (op.db / 20))
        new = pieces[:i0] + [Piece(p.src_start, p.src_end, p.envs + (Env("const", a=g),)) for p in sel] + pieces[i1:]
    else:
        raise EditError(f"Unknown op {op}")
    new = [p for p in new if p.length > 0]
    if not new:
        raise EditError("The edit would leave no audio.")
    return EditState(tuple(new))


# ============================================================ rendering
def _env_gain(p: Piece, n: int) -> np.ndarray | None:
    if not p.envs:
        return None
    g = np.ones(n, np.float32)
    for e in p.envs:
        g *= e.curve(n)
    return g


def render(source: np.ndarray, state: EditState, sr: int) -> np.ndarray:
    """Concatenate pieces. Where consecutive pieces are not contiguous in the
    source, an equal-power crossfade of XFADE_S ends LEAD_S before the splice:
    the outgoing piece fades out while the incoming piece's own pre-roll fades
    in, so the incoming beat's transient is played untouched and the output
    length (and grid) is exactly the sum of piece lengths."""
    n_src, ch = source.shape
    L, lead = int(XFADE_S * sr), int(LEAD_S * sr)
    out = np.zeros((state.length, ch), np.float32)
    pos = 0
    pieces = state.pieces
    for i, p in enumerate(pieces):
        seg = source[p.src_start:p.src_end].copy()
        g = _env_gain(p, len(seg))
        if g is not None:
            seg *= g[:, None]
        out[pos:pos + len(seg)] = seg
        if i > 0:
            prev = pieces[i - 1]
            if prev.src_end != p.src_start:
                w = min(L + lead, pos, p.length)       # region before the splice we rewrite
                s0 = p.src_start - w
                pre = np.zeros((w, ch), np.float32)
                lo = max(0, s0)
                pre[lo - s0:] = source[lo:p.src_start]
                pre *= _env_gain(p, 1)[0] if p.envs else 1.0
                xl = max(1, w - lead)
                # equal-power, with zero slope at both ends (no kink -> no HF splatter)
                t = np.linspace(0, 1, xl, dtype=np.float32)
                u = 0.5 - 0.5 * np.cos(np.pi * t)
                fin = np.concatenate([np.sin(u * np.pi / 2), np.ones(w - xl, np.float32)])[:, None]
                fout = np.concatenate([np.cos(u * np.pi / 2), np.zeros(w - xl, np.float32)])[:, None]
                out[pos - w:pos] = out[pos - w:pos] * fout + pre * fin
        pos += len(seg)
    e = int(EDGE_FADE_S * sr)
    if pieces[0].src_start > 0 and len(out) > e:
        out[:e] *= np.linspace(0, 1, e, dtype=np.float32)[:, None]
    if pieces[-1].src_end < n_src and len(out) > e:
        out[-e:] *= np.linspace(1, 0, e, dtype=np.float32)[:, None]
    return out


# ============================================================ grid mapping
def map_grid(src_grid: BeatGrid, state: EditState, sr: int) -> BeatGrid:
    """The edited track's grid: source beats carried through the pieces."""
    times, pos_labels, seg_start = [], [], []
    out = 0
    for p in state.pieces:
        a, b = p.src_start / sr, p.src_end / sr
        tol = 0.6 / sr      # piece edges are rounded to samples; beats on an edge belong to the later piece
        m = (src_grid.beats >= a - tol) & (src_grid.beats < b - tol)
        first = True
        for t, q in zip(src_grid.beats[m], src_grid.positions[m]):
            times.append(out / sr + (t - a))
            pos_labels.append(q)
            seg_start.append(first)
            first = False
        out += p.length
    beats = np.array(times, float)
    positions = np.array(pos_labels, int)
    # tempo change points: where the inter-beat interval jumps > 1 %
    change = []
    if len(beats) > 3:
        ibi = np.diff(beats)
        for i in range(1, len(ibi)):
            if abs(ibi[i] - ibi[i - 1]) / ibi[i - 1] > 0.01:
                change.append(i + 1)
        # collapse splice artefacts (single odd interval) into real changes only
        change = [c for c in change if c + 1 < len(beats) and
                  abs(np.median(ibi[c:c + 4]) - np.median(ibi[max(0, c - 4):c])) / np.median(ibi[max(0, c - 4):c]) > 0.01]
    constant = src_grid.constant_tempo and not change
    return BeatGrid(beats=beats, positions=positions, beats_per_bar=src_grid.beats_per_bar,
                    time_signature=src_grid.time_signature, duration=out / sr,
                    constant_tempo=bool(constant), segments=tempo_segments_from_beats(beats, positions, change),
                    confidence=src_grid.confidence,
                    meta={**src_grid.meta, "derived_from_edits": True})
