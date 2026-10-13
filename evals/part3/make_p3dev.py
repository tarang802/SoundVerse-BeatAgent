"""Part 3 DEVELOPMENT set (for tuning), separate from the P1-P4 test clips.

    python -m evals.part3.make_p3dev

Transcription thresholds must not be tuned on the clips they are scored on, so this
builds 12 different synthetic loops (random seeds, tempos 85-140 BPM, random kick
patterns, hats on 8ths or 16ths, open hats, bass lines at random levels -- the
"kick under a loud bass" case -- and some with piano chords) with reference MIDI.
Tune on data/p3dev, then score P1-P4 once as held-out clips. SYNTHETIC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from beatagent.core.audio_io import save  # noqa: E402
from evals.part3.make_p3pack import HAT, KICK, OHAT, SNARE, _drums, drum_pattern, render  # noqa: E402
from evals.common.synth import SR  # noqa: E402

OUT = ROOT / "data" / "p3dev"


def build(i: int):
    rng = np.random.default_rng(1000 + i)
    bpm = float(rng.integers(85, 141))
    bars = 4
    kicks = sorted({0, *rng.choice([3, 6, 8, 10, 11, 14], size=int(rng.integers(1, 4)), replace=False)})
    hat_step = 2 if rng.random() < 0.6 else 1
    steps = {KICK: [(int(s), int(rng.integers(95, 120))) for s in kicks],
             SNARE: [(4, 105), (12, 105)],
             HAT: [(s, int(rng.integers(55, 85))) for s in range(0, 16, hat_step)]}
    if rng.random() < 0.4:
        steps[OHAT] = [(14, 65)]
        steps[HAT] = [(s, v) for s, v in steps[HAT] if s != 14]
    tracks = {"drums": _drums(drum_pattern(bars, steps), bpm)}
    is_drum = {"drums": True}
    if i >= 10:                                           # piano only: no drum may be "found"
        bt = 60 / bpm
        notes = []
        for b in range(bars):
            base = int(rng.integers(53, 62))
            notes += [(b * 4 * bt + k * bt, b * 4 * bt + k * bt + 1.8 * bt, base + iv, 75)
                      for k in (0, 2) for iv in (0, 4, 7, 12)]
        return bpm, bars, {"piano": notes}, {"piano": False}
    if i % 10 < 6:                                        # bass line, sometimes loud
        root = int(rng.integers(28, 41))
        level = float(rng.choice([60, 90, 120]))
        bass = []
        for b in range(bars):
            for s in sorted(rng.choice(range(16), size=5, replace=False)):
                t0 = (b * 16 + s) * 15 / bpm
                bass.append((t0, t0 + 2 * 15 / bpm * 0.9, root + int(rng.choice([0, 0, 7, 12])), int(min(127, level))))
        tracks["bass"], is_drum["bass"] = bass, False
    if i % 10 >= 8:                                       # piano chords
        chords, bt = [], 60 / bpm
        for b in range(bars):
            base = int(rng.integers(55, 65))
            chords += [(b * 4 * bt, b * 4 * bt + 3.8 * bt, base + iv, 70) for iv in (0, 4, 7)]
        tracks["piano"], is_drum["piano"] = chords, False
    return bpm, bars, tracks, is_drum


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        bpm, bars, tracks, is_drum = build(i)
        name = f"D{i + 1:02d}"
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        for t, notes in tracks.items():
            inst = pretty_midi.Instrument(program=0, is_drum=is_drum[t], name=t)
            inst.notes = [pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(a), end=float(b)) for a, b, p, v in notes]
            pm.instruments.append(inst)
        pm.write(str(OUT / f"{name}.mid"))
        save(OUT / f"{name}.wav", render(tracks, is_drum, bars * 4 * 60 / bpm, name)[:, None], SR, "PCM_16", "WAV")
        print(name, bpm, "BPM", {k: len(v) for k, v in tracks.items()})


if __name__ == "__main__":
    main()
