"""Part 3 substitute test pack: P1-P4 as MIDI (the answer key) + audio rendered from it.

    python -m evals.make_p3pack

The assignment's P1-P4 were not provided. These follow the brief's descriptions:
  P1  2-bar drum loop                      reference MIDI
  P2  4-bar drums and bass                 reference MIDI
  P3  8-bar piano melody with chords       reference MIDI
  P4  8-bar dense full mix                 (brief: no reference; ours is synthetic, so the
                                            MIDI exists and is reported separately)
SYNTHETIC: rendered with evals/synth.py (Python), NOT with the Web Audio page's synths,
so a recreation is not graded against its own sound. Each clip is an exact number of
bars and loops seamlessly (note tails wrap around to the start, like a real loop).
Drums use General MIDI numbers: 36 kick, 38 snare, 42 closed hat, 46 open hat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.audio_io import save  # noqa: E402
from evals.synth import SR, hat, kick, midi_hz, snare, tone  # noqa: E402

OUT = ROOT / "data" / "p3"
KICK, SNARE, HAT, OHAT = 36, 38, 42, 46


def drum_pattern(bars, steps):
    """steps: {gm_note: [(16th-step index within a bar, velocity), ...]} -> (bar, step, note, vel)"""
    return [(b, s, n, v) for b in range(bars) for n, hits in steps.items() for s, v in hits]


def build(name):
    """-> (bpm, bars, description, {track: [(start_s, end_s, pitch, velocity)]}, is_drum per track)"""
    rng = np.random.default_rng({"P1": 1, "P2": 2, "P3": 3, "P4": 4}[name])
    if name == "P1":
        bpm, bars = 96.0, 2
        steps = {KICK: [(0, 110), (7, 80), (10, 100)], SNARE: [(4, 105), (12, 110)],
                 HAT: [(s, 70 + (10 if s % 4 == 0 else 0)) for s in range(0, 16, 2)]}
        desc = "2-bar boom-bap drum loop, 96 BPM, syncopated kick"
        tracks = {"drums": _drums(drum_pattern(bars, steps), bpm)}
        return bpm, bars, desc, tracks, {"drums": True}
    if name == "P2":
        bpm, bars = 122.0, 4
        steps = {KICK: [(s, 112) for s in (0, 4, 8, 12)], SNARE: [(4, 100), (12, 100)],
                 HAT: [(s, 75) for s in (2, 6, 10, 14)], OHAT: [(14, 60)]}
        desc = "4-bar house drums + syncopated bass in A minor, 122 BPM"
        line = [(0, 3, 33), (3, 2, 33), (6, 2, 45), (10, 2, 36), (12, 3, 38)]       # (step, len, midi)
        prog = [0, 0, -2, 3]                                                          # A, A, G, C
        bass = []
        for b in range(bars):
            for s, ln, m in line:
                t0 = (b * 16 + s) * 15 / bpm
                bass.append((t0, t0 + ln * 15 / bpm * 0.9, m + prog[b], int(rng.integers(85, 110))))
        return bpm, bars, desc, {"drums": _drums(drum_pattern(bars, steps), bpm), "bass": bass}, \
            {"drums": True, "bass": False}
    if name == "P3":
        bpm, bars = 84.0, 8
        desc = "8-bar piano: block chords (C-Am-F-G x2) + melody, 84 BPM"
        chords = [(60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62)] * 2
        mel = [[(0, 2, 76), (2, 1, 74), (3, 1, 72)], [(0, 1.5, 72), (1.5, .5, 71), (2, 2, 69)],
               [(0, 1, 69), (1, 1, 72), (2, 2, 77)], [(0, 3, 74), (3, 1, 71)]]
        mel = mel + [[(0, 1, 76), (1, 1, 79), (2, 2, 77)], mel[1][:], [(0, 2, 72), (2, 2, 74)], [(0, 4, 72)]]
        bt = 60 / bpm
        notes = []
        for b, ch in enumerate(chords):
            for beat in (0, 2):
                t0 = (b * 4 + beat) * bt
                notes += [(t0, t0 + 1.9 * bt, p, int(rng.integers(55, 70))) for p in ch]
            for s, ln, p in mel[b]:
                t0 = (b * 4 + s) * bt
                notes.append((t0, t0 + ln * bt * 0.95, p, int(rng.integers(85, 105))))
        return bpm, bars, desc, {"piano": notes}, {"piano": False}
    if name == "P4":
        bpm, bars = 128.0, 8
        desc = "8-bar dense mix: drums, bass, saw chords, lead, delay smear, 128 BPM"
        steps = {KICK: [(s, 115) for s in (0, 4, 8, 12)], SNARE: [(4, 100), (12, 100)],
                 HAT: [(s, 70) for s in range(1, 16, 2)], OHAT: [(s, 65) for s in (2, 6, 10, 14)]}
        roots = [45, 41, 36, 43] * 2                                                  # A F C G
        bass, chords, lead = [], [], []
        for b, r in enumerate(roots):
            for s in range(0, 16, 2):
                t0 = (b * 16 + s) * 15 / bpm
                bass.append((t0, t0 + 1.6 * 15 / bpm, r - 12 + (12 if s in (6, 14) else 0), 100))
            t0 = b * 16 * 15 / bpm
            third = 3 if r in (45,) else 4
            chords += [(t0, t0 + 15 * 15 / bpm, r + 12 + iv, 70) for iv in (0, third, 7)]
            for s, p in ((0, 12), (3, 15), (6, 19), (10, 17), (12, 15)):
                t1 = (b * 16 + s) * 15 / bpm
                lead.append((t1, t1 + 2 * 15 / bpm, r + 24 + p - 12 + 12, 90))
        return bpm, bars, desc, {"drums": _drums(drum_pattern(bars, steps), bpm), "bass": bass,
                                 "chords": chords, "lead": lead}, \
            {"drums": True, "bass": False, "chords": False, "lead": False}
    raise KeyError(name)


def _drums(hits, bpm):
    return [((b * 16 + s) * 15 / bpm, (b * 16 + s) * 15 / bpm + 0.1, n, v) for b, s, n, v in hits]


def render(tracks, is_drum, length_s, name):
    n = int(round(length_s * SR))
    buf = np.zeros(n + int(4 * SR))                   # room for tails, wrapped below
    rng = np.random.default_rng(7)
    for tname, notes in tracks.items():
        for (t0, t1, p, v) in notes:
            g = (v / 127) ** 1.5
            if is_drum[tname]:
                sig = {KICK: kick, SNARE: snare, HAT: hat}.get(p)
                sig = sig(rng=rng) if sig else hat(rng=rng, open_=True)
            elif tname == "bass":
                sig = tone(midi_hz(p), t1 - t0 + 0.05, kind="saw", cutoff=700, decay=0.6) * 0.5
            elif tname in ("piano",):
                sig = tone(midi_hz(p), t1 - t0 + 0.6, kind="pluck") * 0.5
            elif tname == "chords":
                sig = tone(midi_hz(p), t1 - t0, kind="saw", cutoff=2500) * 0.12
            else:
                sig = tone(midi_hz(p), t1 - t0, kind="square", cutoff=3500, decay=0.4) * 0.15
            i = int(round(t0 * SR))
            buf[i:i + len(sig)] += sig * g
    if name == "P4":                                  # dense: feedback delay smears everything
        d = int(0.75 * 60 / 128 * SR)
        for k in range(1, 4):
            buf[d * k:] += 0.35 ** k * buf[:len(buf) - d * k]
    out = buf[:n].copy()
    tail = buf[n:]
    for k in range(0, len(tail), n):                  # wrap tails: the clip loops seamlessly
        seg = tail[k:k + n]
        out[:len(seg)] += seg
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name in ("P1", "P2", "P3", "P4"):
        bpm, bars, desc, tracks, is_drum = build(name)
        length = bars * 4 * 60 / bpm
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        for tname, notes in tracks.items():
            inst = pretty_midi.Instrument(program=0 if tname == "piano" else 33 if tname == "bass" else 81,
                                          is_drum=is_drum[tname], name=tname)
            inst.notes = [pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(a), end=float(b))
                          for a, b, p, v in notes]
            pm.instruments.append(inst)
        pm.write(str(OUT / f"{name}.mid"))
        save(OUT / f"{name}.wav", render(tracks, is_drum, length, name)[:, None], SR, "PCM_16", "WAV")
        meta[name] = {"bpm": bpm, "bars": bars, "time_signature": "4/4", "length_s": round(length, 6),
                      "description": desc, "tracks": {k: {"drum": is_drum[k], "n_notes": len(v)} for k, v in tracks.items()},
                      "reference": "none in the brief (synthetic MIDI kept)" if name == "P4" else "MIDI",
                      "synthetic": True}
        print(f"{name}: {desc}  ({length:.2f} s, {sum(len(v) for v in tracks.values())} notes)")
    (OUT / "pack.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
