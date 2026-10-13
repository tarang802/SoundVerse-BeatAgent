"""Edit-engine tests on a synthetic track with a KNOWN grid (no tracker involved).

    python -m tests.test_edl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatagent.editing import edl  # noqa: E402
from beatagent.core import qc  # noqa: E402
from beatagent.editing.edl import (Copy, EditError, EditState, Fade, Gain, KeepOnly, Move, Piece,  # noqa: E402
                           Position, Range, Remove, Repeat)
from beatagent.core.grid import BeatGrid  # noqa: E402
from evals.common.synth import SR, fixture_120  # noqa: E402

audio, ref_beats, _ = fixture_120()
src = audio[:, None]
G = BeatGrid(beats=ref_beats, positions=(np.arange(len(ref_beats)) % 4) + 1, beats_per_bar=4,
             time_signature="4/4", duration=len(audio) / SR, constant_tempo=True)
S0 = EditState((Piece(0, len(audio)),))


def run(state, grid, op):
    r = edl.resolve(op, grid)
    st = edl.apply(state, r, SR)
    return st, edl.map_grid(G, st, SR), r


def max_jump_near(out, times, win=0.02):
    """Largest sample-to-sample step within +-win of the given times (click detector)."""
    d = np.abs(np.diff(out[:, 0]))
    js = []
    for t in times:
        i = int(t * SR)
        js.append(d[max(0, i - int(win * SR)): i + int(win * SR)].max())
    return max(js) if js else 0.0


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    assert cond, name


def main():
    print("edit engine tests (synthetic 120 BPM fixture, ground-truth grid)")
    # 1. repeat beats 16-32 twice in total
    st, g, r = run(S0, G, Repeat(range=Range(unit="beats", start=16, end=32), total_plays=2))
    check("repeat adds 16 beats", g.n_beats == G.n_beats + 16, f"({G.n_beats} -> {g.n_beats})")
    check("repeat adds exactly 8.000 s", abs(g.duration - G.duration - 8.0) < 1 / SR)
    ibi = np.diff(g.beats)
    check("grid stays constant 0.5 s", np.max(np.abs(ibi - 0.5)) < 1e-4, f"(max dev {np.max(np.abs(ibi - 0.5)) * 1000:.4f} ms)")
    check("grid stays constant-tempo", g.constant_tempo)
    check("bar structure kept (positions cycle 1-4)", np.all(np.diff(g.positions) % 4 == 1))
    out = edl.render(src, st, SR)
    check("rendered length = grid duration", len(out) == round(g.duration * SR))
    # beat-aligned repeat of the same material: output at the repeat should equal source at beat 16
    i_rep = int(round(G.beats[32] * SR))
    i_src = int(round(G.beats[16] * SR))
    seg = 2000
    err = np.max(np.abs(out[i_rep:i_rep + seg, 0] - src[i_src:i_src + seg, 0]))
    check("repeat copy is sample-exact after the crossfade", err < 1e-6, f"(max err {err:.2e})")

    # 2. remove bar 5 (bars 1-based, beats 16-19)
    st2, g2, r2 = run(S0, G, Remove(range=Range(unit="bars", start=5, end=5)))
    check("remove bar 5 drops 4 beats", g2.n_beats == G.n_beats - 4)
    check("remove bar 5 echo", "bar 5 (1 bar, 4 beats)" in r2.echo and "8.500" in r2.echo, f"-> {r2.echo}")
    out2 = edl.render(src, st2, SR)
    splice_t = G.beats[16]
    base = max_jump_near(src, [G.beats[16]])  # natural jump at a kick onset in the source
    jump = max_jump_near(out2, [splice_t])
    check("no click at splice (jump <= source's own onset jump)", jump <= base * 1.05 + 1e-4,
          f"(splice {jump:.4f} vs source {base:.4f})")

    # 3. non-musical splice into a sustained sine must not click
    t = np.arange(SR * 4) / SR
    sine = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None]
    sg = BeatGrid(beats=np.arange(0, 4, 0.5), positions=(np.arange(8) % 4) + 1, beats_per_bar=4,
                  time_signature="4/4", duration=4.0, constant_tempo=True)
    s_st = edl.apply(EditState((Piece(0, len(sine)),)),
                     edl.resolve(Remove(range=Range(unit="beats", start=1, end=2)), sg), SR)
    # 0.5 s of 220 Hz = 110 cycles -> phase-continuous; use 1.1 s offset to force discontinuity
    s_st = EditState((Piece(0, int(1.0 * SR)), Piece(int(2.1137 * SR), len(sine))))
    s_out = edl.render(sine, s_st, SR)
    raw = np.concatenate([sine[:int(1.0 * SR)], sine[int(2.1137 * SR):]])
    c_raw = qc.splice_click_db(raw, sine, SR, 1.0, [1.0, 2.1137])
    c_xf = qc.splice_click_db(s_out, sine, SR, 1.0, [1.0, 2.1137])
    check("hard cut registers as a click", c_raw > 20, f"({c_raw:+.1f} dB HF vs source)")
    check("crossfaded splice is click-free", c_xf < 3, f"({c_xf:+.1f} dB HF vs source)")
    c_bar = qc.splice_click_db(out2, src, SR, G.beats[16], [G.beats[16], G.beats[20]])
    check("remove-bar splice on drums is click-free", c_bar < 3, f"({c_bar:+.1f} dB)")

    # 4. fade out the last 8 beats ends in silence
    st4, g4, r4 = run(S0, G, Fade(range=Range(unit="beats", last=8), direction="out"))
    out4 = edl.render(src, st4, SR)
    check("fade-out reaches silence", np.max(np.abs(out4[-200:])) < 1e-3)
    check("fade keeps length", len(out4) == len(src))

    # 5. move bar 2 to the start of bar 8
    st5, g5, r5 = run(S0, G, Move(range=Range(unit="bars", start=2, end=2), to=Position(unit="bars", index=8)))
    check("move keeps beat count", g5.n_beats == G.n_beats)
    check("move keeps length", st5.length == len(audio))
    # bar-2 material (source 2.5 s) should now begin at original bar-8 start minus one bar (7*2s+0.5-2)
    check("moved bar lands before old bar 8", abs(g5.downbeats[6] - (G.bar_start(8) - 2.0)) < 1e-6)

    # 6. gain + copy + keep_only
    st6, g6, _ = run(S0, G, Gain(db=-6.0))
    out6 = edl.render(src, st6, SR)
    check("-6 dB gain", abs(20 * np.log10(np.max(np.abs(out6)) / np.max(np.abs(src))) + 6.0) < 0.01)
    st7, g7, _ = run(S0, G, Copy(range=Range(unit="bars", start=1, end=1), to=Position(unit="bars", index=None)))
    check("copy to end adds a bar", g7.n_beats == G.n_beats + 4)
    st8, g8, r8 = run(S0, G, KeepOnly(range=Range(unit="bars", start=3, end=4)))
    check("keep only bars 3-4 = 8 beats, 4.0 s", g8.n_beats == 8 and abs(g8.duration - 4.0) < 1e-4)

    # 7. errors
    for bad, frag in [(Remove(range=Range(unit="bars", start=200, end=210)), "bars 1-"),
                      (Remove(range=Range(unit="beats", start=8, end=4)), "reversed"),
                      (Remove(range=Range(unit="beats", start=None, end=None)), "entire track"),
                      (Move(range=Range(unit="bars", start=2, end=4), to=Position(unit="bars", index=3)), "inside itself")]:
        try:
            edl.resolve(bad, G)
            check(f"error raised for {bad}", False)
        except EditError as e:
            check(f"clear error: {str(e)[:70]}", frag in str(e))

    # 8. non-bar-aligned warning
    r9 = edl.resolve(Remove(range=Range(unit="beats", start=3, end=6)), G)
    check("warns on non-bar-aligned structural edit", bool(r9.warnings))
    print("all edit-engine tests passed")


if __name__ == "__main__":
    main()
