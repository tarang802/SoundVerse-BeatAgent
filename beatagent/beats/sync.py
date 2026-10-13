"""Two-track sync: beat-match track B to master track A.

Step 1 (this file so far): the SYNC PLAN -- pure arithmetic on two beat grids.
No audio is touched here, so every number can be checked by hand.

Conventions
  * A is the master; its audio is never altered.
  * B is time-stretched by `time_ratio` (> 1 = longer/slower, pitch preserved).
  * time_ratio = A's beat period / B's beat period, both measured from the
    fitted grids over the overlap window (not from rounded BPM labels).
  * B's bar `b_bar` downbeat lands exactly on A's bar `a_bar` downbeat.
  * The render timeline starts at that shared downbeat (t = 0).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..core.grid import BeatGrid, tempo_segments_from_beats
from ..core.onsets import kick_flux_near, onsets_near
from .stretch import stretch

MAX_STRETCH = 0.25        # refuse ratios beyond +-25 %: Rubber Band can do it, but it sounds bad


class SyncError(ValueError):
    """A sync request that can't be done as asked. Message is user-facing."""


@dataclass
class SyncPlan:
    a_bar: int                # A's bar where the overlap starts (1-based)
    b_bar: int                # B's bar that lands on it (1-based)
    a_start: float            # seconds in A: A's bar a_bar downbeat
    b_start: float            # seconds in B (original, unstretched): B's bar b_bar downbeat
    duration: float           # length of the rendered overlap, seconds
    a_period: float           # A's median beat period over the overlap, seconds
    b_period: float           # B's median beat period over the matching span, seconds
    time_ratio: float         # stretch applied to B: out_time = in_time * time_ratio
    a_bpm: float
    b_bpm: float
    b_needed: float           # seconds of ORIGINAL B audio the overlap consumes
    notes: list
    b_octave: float = 1.0     # 2 = B matched at double time (its grid's beats doubled), 0.5 = half time

    @property
    def offset(self) -> float:
        """Where B's original t=0 would sit on A's timeline after stretching.
        (A time) = offset + (B original time) * time_ratio. Negative when B
        has audio before its aligned downbeat that falls before A's t=0."""
        return self.a_start - self.b_start * self.time_ratio

    def to_dict(self) -> dict:
        d = asdict(self)
        d["offset"] = self.offset
        d["stretch_percent"] = (self.time_ratio - 1.0) * 100.0
        return d


def octave_grid(g: BeatGrid, factor: float) -> BeatGrid:
    """B's grid at double (factor 2) or half (0.5) time, keeping its downbeats as
    downbeats and 4 beats per bar -- how a DJ matches a 75 BPM track to a 150 one."""
    if factor == 1.0:
        return g
    i0 = int(g.downbeat_idx[0]) if g.n_bars else 0
    if factor == 2.0:
        beats = np.sort(np.concatenate([g.beats, (g.beats[:-1] + g.beats[1:]) / 2]))
        j0 = 2 * i0
    elif factor == 0.5:
        beats = g.beats[i0 % 2::2]
        j0 = (i0 - i0 % 2) // 2
    else:
        raise ValueError(factor)
    pos = ((np.arange(len(beats)) - j0) % g.beats_per_bar) + 1
    return BeatGrid(beats=beats, positions=pos, beats_per_bar=g.beats_per_bar, time_signature=g.time_signature,
                    duration=g.duration, constant_tempo=g.constant_tempo,
                    segments=tempo_segments_from_beats(beats, pos, []), confidence=g.confidence,
                    meta={**g.meta, "octave_factor": factor})


def matched_b_grid(plan: "SyncPlan", b: BeatGrid) -> BeatGrid:
    """The B grid the plan was computed on (use it for rendering and measuring)."""
    return octave_grid(b, plan.b_octave)


def _median_period(beats: np.ndarray, t0: float, t1: float) -> float:
    """Median inter-beat interval of the beats inside [t0, t1]."""
    inside = beats[(beats >= t0 - 1e-6) & (beats <= t1 + 1e-6)]
    if len(inside) < 3:
        raise SyncError(f"Fewer than 3 beats between {t0:.2f}s and {t1:.2f}s; can't measure tempo there.")
    return float(np.median(np.diff(inside)))


def plan_sync(a: BeatGrid, b: BeatGrid, a_bar: int, b_bar: int = 1,
              duration: float = 60.0) -> SyncPlan:
    """Plan at B's own tempo; if that needs more than MAX_STRETCH but B at double
    or half time would not, match at that octave (and say so in notes)."""
    first_err, tried = None, []
    for factor in (1.0, 2.0, 0.5):
        try:
            p = _plan_once(a, octave_grid(b, factor), a_bar, b_bar, duration)
        except SyncError as e:
            if first_err is None:
                first_err = e
            if factor == 1.0 and "stretching" not in str(e):
                raise                       # not a tempo problem: report it as is
            if factor != 1.0:
                tried.append(f"{'double' if factor == 2 else 'half'} time {e.args[0].split('stretching by ')[-1].split('%')[0]}%"
                             if "stretching by" in str(e) else f"{'double' if factor == 2 else 'half'} time: {e}")
            continue
        if factor != 1.0:
            p.b_octave = factor
            p.notes.insert(0, f"B matched at {'double' if factor == 2 else 'half'} time "
                              f"({60 / p.b_period / factor:.2f} BPM counted as {60 / p.b_period:.2f}).")
        return p
    raise SyncError(f"{first_err} Also tried matching B at {' and '.join(tried)} -- still beyond "
                    f"+-{MAX_STRETCH * 100:.0f}%.") if tried else first_err


def _plan_once(a: BeatGrid, b: BeatGrid, a_bar: int, b_bar: int, duration: float) -> SyncPlan:
    notes: list[str] = []

    # ---- the bars must exist
    if not 1 <= a_bar <= a.n_bars:
        raise SyncError(f"Track A has bars 1-{a.n_bars}; there is no bar {a_bar}.")
    if not 1 <= b_bar <= b.n_bars:
        raise SyncError(f"Track B has bars 1-{b.n_bars}; there is no bar {b_bar}.")

    # ---- downbeats can only line up bar-for-bar if the bars match. Compare the
    # full signature, not just beats_per_bar: 12/8 and 4/4 both have 4 beats
    # but one feels in threes, and the two will clash even with aligned downbeats.
    if a.beats_per_bar != b.beats_per_bar or a.time_signature != b.time_signature:
        raise SyncError(f"A is in {a.time_signature} ({a.beats_per_bar} beats/bar) but B is in "
                        f"{b.time_signature} ({b.beats_per_bar} beats/bar); their downbeats can't stay "
                        f"aligned bar by bar.")

    a_start = a.bar_start(a_bar)
    b_start = b.bar_start(b_bar)

    # ---- A must have `duration` seconds left after the start point
    if a_start + duration > a.duration + 1e-6:
        raise SyncError(f"A only has {a.duration - a_start:.1f}s after bar {a_bar}; "
                        f"a {duration:.0f}s overlap doesn't fit.")

    # ---- tempo of each side over the span that will actually be heard
    a_period = _median_period(a.beats, a_start, a_start + duration)
    # first guess of B's span using its tempo near the start, then refine once
    b_period = _median_period(b.beats, b_start, b_start + duration)
    ratio = a_period / b_period
    b_needed = duration / ratio
    b_period = _median_period(b.beats, b_start, b_start + b_needed)
    ratio = a_period / b_period
    b_needed = duration / ratio

    if abs(ratio - 1.0) > MAX_STRETCH:
        raise SyncError(f"B would need stretching by {(ratio - 1) * 100:+.1f}% "
                        f"({60 / b_period:.2f} -> {60 / a_period:.2f} BPM); beyond +-{MAX_STRETCH * 100:.0f}% "
                        f"it sounds bad.")
    if b_start + b_needed > b.duration + 1e-6:
        raise SyncError(f"After stretching, B from bar {b_bar} only covers "
                        f"{(b.duration - b_start) * ratio:.1f}s, less than the {duration:.0f}s requested.")

    if not a.constant_tempo:
        notes.append("A's grid is not constant-tempo; a single ratio uses A's median tempo over the overlap.")
    if not b.constant_tempo:
        notes.append("B's grid is not constant-tempo; a single ratio uses B's median tempo over the overlap.")
    if b_start > 0.05:
        notes.append(f"B's audio before its bar {b_bar} downbeat ({b_start:.3f}s) is not used.")

    return SyncPlan(a_bar=a_bar, b_bar=b_bar, a_start=a_start, b_start=b_start, duration=duration,
                    a_period=a_period, b_period=b_period, time_ratio=ratio,
                    a_bpm=60.0 / a_period, b_bpm=60.0 / b_period, b_needed=b_needed, notes=notes)


# ============================================================ step 2: render
PREROLL = 0.5     # seconds of B before its downbeat fed to the stretcher (warm-up), then cut away
B_FADE_IN = 0.003 # tiny fade where B's audio starts at t=0, so the cut itself doesn't click
SNAP_S = 0.025    # pasted kicks snap to the actual hit within +-25 ms in both tracks
EDGE_SKIP = 0.05  # measurement ignores beats in the first 50 ms of the render (no audio before
                  # them -> detector edge artefact: the t=0 downbeat read 19.7 ms, all others <1 ms)


def _match_format(x: np.ndarray, sr_from: int, sr_to: int, channels: int) -> np.ndarray:
    """Resample and up/down-mix x (n, ch) to sr_to and `channels`."""
    if sr_from != sr_to:
        import soxr
        x = soxr.resample(x, sr_from, sr_to).astype(np.float32)
    if x.shape[1] != channels:
        mono = x.mean(axis=1, keepdims=True)
        x = np.repeat(mono, channels, axis=1)
    return x


def render_constant(a_audio: np.ndarray, a_sr: int, b_audio: np.ndarray, b_sr: int,
                    plan: SyncPlan) -> dict:
    """NOT THE DEFAULT on its own: R3 smears kick transients (downbeats measured
    ~30 ms early, max 18 ms on the measurable beats). Used as the body of
    render_constant_attack.

    Simplest render: stretch B in ONE piece by plan.time_ratio (Rubber Band R3).

    Returns stems aligned on the render timeline (t=0 = A's bar a_bar downbeat):
      a: A's audio, unchanged;  b: B stretched;  mix: (a + b) at -6 dB each.
    """
    sr = a_sr
    n = int(round(plan.duration * sr))
    b_audio = _match_format(b_audio, b_sr, sr, a_audio.shape[1])

    # A: exactly `duration` seconds from its bar downbeat
    a0 = int(round(plan.a_start * sr))
    a_seg = a_audio[a0:a0 + n]

    # B: from (downbeat - preroll) to a little past what the overlap needs
    b_from = max(0.0, plan.b_start - PREROLL)
    b_to = min(b_audio.shape[0] / sr, plan.b_start + plan.b_needed + 1.0)
    chunk = b_audio[int(round(b_from * sr)):int(round(b_to * sr))]
    stretched = stretch(chunk, sr, plan.time_ratio)
    # B's downbeat sat (b_start - b_from) into the chunk; after stretching it is ratio x later
    cut = int(round((plan.b_start - b_from) * plan.time_ratio * sr))
    b_seg = stretched[cut:cut + n]
    if len(b_seg) < n:                        # pad if the stretcher returned a hair short
        b_seg = np.pad(b_seg, ((0, n - len(b_seg)), (0, 0)))
    f = int(B_FADE_IN * sr)
    b_seg[:f] *= np.linspace(0, 1, f, dtype=np.float32)[:, None]

    mix = 0.5 * a_seg + 0.5 * b_seg
    return {"a": a_seg, "b": b_seg, "mix": mix, "sr": sr, "method": "constant_r3"}


# ============================================================ measurement
def measure_alignment(a_seg: np.ndarray, b_seg: np.ndarray, sr: int, a_grid: BeatGrid,
                      plan: SyncPlan) -> dict:
    """Kick-vs-kick alignment in a render, measured on the SEPARATE stems.

    For every A grid beat inside the overlap we find the onset nearest to it in
    A's stem and in B's stem (spectral-flux peak within +-40 ms; on A7 vs its
    known beat times this detector's spread was 0.5 ms p95) and report B - A. Measuring both stems the
    same way cancels the detector's own bias."""
    # skip beats in the first EDGE_SKIP s: the render starts ON A's downbeat, so
    # there is no audio before it and onset detectors give edge artefacts there
    in_win = (a_grid.beats >= plan.a_start + EDGE_SKIP) & (a_grid.beats < plan.a_start + plan.duration - 0.05)
    expected = a_grid.beats[in_win] - plan.a_start          # render-timeline times
    on_a = onsets_near(a_seg, sr, expected)
    on_b = onsets_near(b_seg, sr, expected)
    both = ~np.isnan(on_a) & ~np.isnan(on_b)
    off = (on_b - on_a)[both] * 1000.0
    if len(off) == 0:
        return {"n_beats": int(len(expected)), "n_measured": 0, "error": "no onsets found in both stems"}
    worst = int(np.argmax(np.abs(off)))
    return {
        "n_beats": int(len(expected)),
        "n_measured": int(both.sum()),
        "median_ms": round(float(np.median(off)), 2),
        "median_abs_ms": round(float(np.median(np.abs(off))), 2),
        "p95_abs_ms": round(float(np.percentile(np.abs(off), 95)), 2),
        "max_abs_ms": round(float(np.max(np.abs(off))), 2),
        "worst_at_s": round(float(expected[both][worst]), 3),
        "offsets_ms": [round(float(x), 2) for x in off],
        "target": {"median_abs_ms": 5.0, "max_abs_ms": 15.0},
        "pass": bool(np.median(np.abs(off)) <= 5.0 and np.max(np.abs(off)) <= 15.0),
    }



def measure_downbeat_kicks(a_seg: np.ndarray, b_seg: np.ndarray, sr: int, a_grid: BeatGrid,
                           plan: SyncPlan) -> dict:
    """Kick-vs-kick on DOWNBEATS only, from low-frequency (< 150 Hz) spectral flux.

    measure_alignment compares the nearest onset of any kind; on real hip-hop,
    where many beats have no kick, that paired A's kick with B's hi-hat. Validated
    on known beat times: downbeat spread p95 0.2 ms (A7, A2), 1.1 ms (A1); one false
    kick in 16 drumless breakdown bars. Only downbeats where BOTH stems have a kick
    are compared."""
    bars = a_grid.positions == 1
    in_win = bars & (a_grid.beats >= plan.a_start + EDGE_SKIP) & (a_grid.beats < plan.a_start + plan.duration - 0.05)
    expected = a_grid.beats[in_win] - plan.a_start
    ka, kb = kick_flux_near(a_seg, sr, expected), kick_flux_near(b_seg, sr, expected)
    both = ~np.isnan(ka) & ~np.isnan(kb)
    off = (kb - ka)[both] * 1000.0
    out = {"n_downbeats": int(len(expected)), "n_measured": int(both.sum()),
           "a_has_kick": int((~np.isnan(ka)).sum()), "b_has_kick": int((~np.isnan(kb)).sum())}
    if len(off):
        out.update({"median_ms": round(float(np.median(off)), 2),
                    "median_abs_ms": round(float(np.median(np.abs(off))), 2),
                    "max_abs_ms": round(float(np.max(np.abs(off))), 2),
                    "offsets_ms": [round(float(x), 2) for x in off],
                    "pass": bool(np.median(np.abs(off)) <= 5.0 and np.max(np.abs(off)) <= 15.0)})
    return out



def measure_downbeat_attacks(a_seg: np.ndarray, b_seg: np.ndarray, sr: int, a_grid: BeatGrid,
                             plan: SyncPlan) -> dict:
    """Attack-vs-attack on DOWNBEATS: broadband onset (spectral flux) nearest each
    of A's downbeats, in each stem. On real hip-hop the downbeat attack is what
    sits tightly on the grid (Toh Kya: 1.3 ms spread vs its own grid), while the
    low-frequency 'kick' moment drifts with 808 sub-bass (27 ms spread)."""
    bars = a_grid.positions == 1
    in_win = bars & (a_grid.beats >= plan.a_start + EDGE_SKIP) & (a_grid.beats < plan.a_start + plan.duration - 0.05)
    expected = a_grid.beats[in_win] - plan.a_start
    oa, ob = onsets_near(a_seg, sr, expected), onsets_near(b_seg, sr, expected)
    both = ~np.isnan(oa) & ~np.isnan(ob)
    off = (ob - oa)[both] * 1000.0
    out = {"n_downbeats": int(len(expected)), "n_measured": int(both.sum())}
    if len(off):
        out.update({"median_abs_ms": round(float(np.median(np.abs(off))), 2),
                    "max_abs_ms": round(float(np.max(np.abs(off))), 2),
                    "share_within_15ms": round(float(np.mean(np.abs(off) <= 15)), 3),
                    "offsets_ms": [round(float(x), 2) for x in off],
                    "pass": bool(np.median(np.abs(off)) <= 5.0 and np.max(np.abs(off)) <= 15.0)})
    return out


# ============================================================ step 3: beat-anchored render
SLICE_CONTEXT = 0.05   # s of audio fed to the stretcher on each side of a slice (warm-up), then cut away
JOIN_XFADE = 0.005     # s crossfade at every join, placed BEFORE the beat so the kick starts untouched


def _fade(n: int, rising: bool) -> np.ndarray:
    """Equal-power curve with zero slope at both ends (same shape as edl crossfades)."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    u = 0.5 - 0.5 * np.cos(np.pi * t)
    return np.sin(u * np.pi / 2) if rising else np.cos(u * np.pi / 2)


def _beat_index(grid: BeatGrid, t: float) -> int:
    return int(np.argmin(np.abs(grid.beats - t)))


def render_beat_anchored(a_audio: np.ndarray, a_sr: int, b_audio: np.ndarray, b_sr: int,
                         plan: SyncPlan, a_grid: BeatGrid, b_grid: BeatGrid,
                         attack_ms: float = 30.0) -> dict:
    """NOT THE DEFAULT (kept as measured evidence, see evals/results/sync_A1_A7/comparison.md):
    kick timing is excellent, but R3 on half-second slices detuned a sustained
    110 Hz tone by +11-12 cents, which fails "pitch preserved".

    Stretch B one beat at a time so B's beat j lands exactly on A's beat j.

    For each pair of consecutive beats, B's slice [b_j, b_j+1) is stretched with
    Rubber Band R3 to exactly the length of A's [a_j, a_j+1) and written at a_j.
    If attack_ms > 0, the first attack_ms after each B beat is copied VERBATIM
    (original audio, original pitch) so the kick's transient is never smeared;
    R3 only fills the rest of the beat.
    Each slice gets its own ratio, so this also follows variable-tempo grids.
    """
    sr = a_sr
    n = int(round(plan.duration * sr))
    ch = a_audio.shape[1]
    b = _match_format(b_audio, b_sr, sr, ch)
    a0 = int(round(plan.a_start * sr))
    a_seg = a_audio[a0:a0 + n]

    # beat correspondences: A beat ia+j  <->  B beat ib+j
    ia, ib = _beat_index(a_grid, plan.a_start), _beat_index(b_grid, plan.b_start)
    a_t = a_grid.beats[ia:] - plan.a_start           # render timeline
    b_t = b_grid.beats[ib:]                          # B's original timeline
    m = min(len(a_t), len(b_t)) - 1
    while m > 0 and a_t[m - 1] >= plan.duration:     # only slices that start inside the overlap
        m -= 1

    xf = int(JOIN_XFADE * sr)
    att = int(attack_ms * 1e-3 * sr)
    ctx = int(SLICE_CONTEXT * sr)
    out = np.zeros((n + sr, ch), np.float32)         # 1 s spare so the last slice can overhang
    ratios = []
    for j in range(m):
        d0, d1 = int(round(a_t[j] * sr)), int(round(a_t[j + 1] * sr))
        s0, s1 = int(round(b_t[j] * sr)), int(round(b_t[j + 1] * sr))
        L = d1 - d0
        r = L / (s1 - s0)
        ratios.append(r)
        # stretch the slice WITH context on both sides, then cut out exactly
        # [xf before the beat, end of beat) on the output timeline
        lo, hi = max(0, s0 - ctx), min(len(b), s1 + ctx)
        st = stretch(b[lo:hi], sr, r)
        c0 = int(round((s0 - lo) * r))               # where B's beat j landed inside `st`
        body = np.zeros((L + xf, ch), np.float32)
        piece = st[max(0, c0 - xf): c0 + L]
        body[xf - min(xf, c0):xf - min(xf, c0) + len(piece)] = piece
        w = np.ones(L + xf, np.float32)
        w[:xf] = _fade(xf, True)                     # fade in before the beat ...
        w[-xf:] = _fade(xf, False)                   # ... and out before the next beat
        body *= w[:, None]
        if att > 0:
            # Head region = [xf before the beat, att + xf after it] (k samples).
            #   original audio: fade in over the xf BEFORE the beat, full for att,
            #                   fade out over the last xf
            #   stretched body: silent until the attack ends, then fades in
            # Before the beat we therefore hear only original audio, which cannot
            # contain R3's pre-echo of the kick.
            k = xf + att + xf
            v = np.zeros((k, ch), np.float32)
            v0 = max(0, s0 - xf)
            src = b[v0: s0 + att + xf]
            off = xf - (s0 - v0)                     # < xf only if B's beat is at the file start
            v[off:off + len(src)] = src
            vw = np.ones(k, np.float32)
            vw[:xf] = _fade(xf, True)
            vw[-xf:] = _fade(xf, False)
            bw = np.zeros(k, np.float32)
            bw[-xf:] = _fade(xf, True)
            body[:k] = body[:k] * bw[:, None] + v * vw[:, None]
        start = d0 - xf
        if start < 0:                                # first slice: nothing before t=0
            body, start = body[-start:], 0
        out[start:start + len(body)] += body
    b_seg = out[:n]
    mix = 0.5 * a_seg + 0.5 * b_seg
    return {"a": a_seg, "b": b_seg, "mix": mix, "sr": sr,
            "method": f"beat_anchored_attack{int(attack_ms)}ms" if att else "beat_anchored",
            "n_slices": m, "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios))}


def render_constant_attack(a_audio: np.ndarray, a_sr: int, b_audio: np.ndarray, b_sr: int,
                           plan: SyncPlan, a_grid: BeatGrid, b_grid: BeatGrid,
                           pre_ms: float = 40.0, attack_ms: float = 30.0) -> dict:
    """One-piece R3 stretch (exact pitch, continuous phase) + verbatim patches at every beat.

    At each A beat j, B's ORIGINAL audio from pre_ms before its beat j to
    attack_ms after it is pasted so that B's beat lands exactly on A's beat,
    crossfaded (JOIN_XFADE) into the stretched audio on both sides.
      * the pre_ms part replaces R3's pre-echo (kick energy smeared earlier)
      * the attack_ms part is the kick's transient, untouched and on time
    """
    r = render_constant(a_audio, a_sr, b_audio, b_sr, plan)
    sr, n = r["sr"], len(r["a"])
    b = _match_format(b_audio, b_sr, sr, a_audio.shape[1])
    body = r["b"].copy()
    ia, ib = _beat_index(a_grid, plan.a_start), _beat_index(b_grid, plan.b_start)
    xf, pre, att = int(JOIN_XFADE * sr), int(pre_ms * 1e-3 * sr), int(attack_ms * 1e-3 * sr)
    k = xf + pre + att + xf                          # patch length incl. both crossfades
    w = np.ones(k, np.float32)
    w[:xf], w[-xf:] = _fade(xf, True), _fade(xf, False)
    # Snap each pasted kick to the ACTUAL hit (within +-SNAP_S) in both tracks, so a
    # grid that is a few ms off (drifting live takes, tempo changes) no longer
    # carries the kick off time. No clear hit -> the grid time is kept.
    a_beats = a_grid.beats[ia:] - plan.a_start
    a_beats = a_beats[a_beats < n / sr]
    a_snap = onsets_near(r["a"], sr, a_beats, win=SNAP_S)
    b_beats = b_grid.beats[ib:ib + len(a_beats)]
    b_snap = onsets_near(b, sr, b_beats, win=SNAP_S)
    patches = []
    n_patched = 0
    for j in range(0, len(a_beats)):
        if j >= len(b_beats):
            break
        ta = a_snap[j] if not np.isnan(a_snap[j]) else a_beats[j]
        tb = b_snap[j] if not np.isnan(b_snap[j]) else b_beats[j]
        d0 = int(round(ta * sr))                                      # beat on the render timeline
        if d0 >= n:
            break
        s0 = int(round(tb * sr))                                      # same beat in original B
        patches.append((d0 / sr, s0 / sr))
        o0, v0 = d0 - pre - xf, s0 - pre - xf                         # patch start (output / source)
        lo = max(0, -o0, -v0)                                         # clip at t=0 / file start
        hi = min(k, n - o0, len(b) - v0)
        if hi <= lo:
            continue
        # (a patch clipped at t=0 loses exactly its fade-in, so it starts at full weight)
        seg_w = w[lo:hi, None]
        # w and 1-w sum to 1 (equal-gain): right for these two strongly correlated
        # versions of the same audio (stretched B vs original B)
        body[o0 + lo:o0 + hi] = body[o0 + lo:o0 + hi] * (1 - seg_w) + b[v0 + lo:v0 + hi] * seg_w
        n_patched += 1
    f = int(B_FADE_IN * sr)
    body[:f] *= np.linspace(0, 1, f, dtype=np.float32)[:, None]
    return {"a": r["a"], "b": body, "mix": 0.5 * r["a"] + 0.5 * body, "sr": sr,
            "method": f"constant_r3+verbatim_{int(pre_ms)}ms_pre_{int(attack_ms)}ms_attack+snap",
            "n_patched": n_patched, "patches": patches}


def render(a_audio, a_sr, b_audio, b_sr, plan: SyncPlan, a_grid: BeatGrid, b_grid: BeatGrid) -> dict:
    """The sync renderer the app and evals use (method chosen by measurement:
    one-piece R3 for exact pitch + verbatim patches at each beat for kick timing)."""
    return render_constant_attack(a_audio, a_sr, b_audio, b_sr, plan, a_grid, b_grid)
