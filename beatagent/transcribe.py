"""Part 3 analysis: a short clip (2-16 bars) -> Score (beatagent/score.py).

Every note comes from signal analysis or a transcription model, never from the LLM:
  tempo, bars   Part 1's analyzer (Beat This! + grid fit) on the clip repeated to
                >= 20 s (a 2-bar loop is ~5 s, too short for the tracker; a loop
                repeats by definition). If the clip is a whole number of bars
                (within 3 %), the tempo is taken from its exact length.
  drums         HPSS percussive part -> per-band onset detection:
                kick 30-120 Hz, snare 1-5 kHz noise, hats 7-16 kHz; relative
                thresholds per band; open vs closed hat from the decay.
  pitched       Basic Pitch (ONNX model, Spotify) on the HPSS harmonic part; quiet
                overtone "ghost" notes (an octave / 12th / two octaves above a louder
                note sounding at the same time) are dropped; notes below MIDI 47 go
                to a "bass" track, the rest to "keys" (only if both groups exist).
Times are stored in beats from the first downbeat, folded into the loop.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import signal

from .analyze import analyze
from .audio_io import save, to_mono
from .onsets import band_envelope
from .score import Note, Score, Track, default_patch

BASS_SPLIT = 47                   # MIDI: below -> bass track (B2)
MIN_ANALYSIS_S = 20.0
DRUM_BANDS = {                    # name: (lo, hi, relative threshold, GM pitch, absolute floor)
    "kick": (30.0, 120.0, 0.30, 36, 0.06),
    "snare": (1000.0, 5000.0, 0.40, 38, 0.0),
    "hat": (7000.0, 16000.0, 0.25, 42, 0.01),
}
# A hit must also be loud in ABSOLUTE terms (x the clip's peak), not only relative to
# the band's strongest onset: in a clip without drums the strongest "onset" is a piano
# attack. Measured on the synthetic P1-P4: real kicks 0.15-0.35 of the peak vs piano
# 0.03; hats 0.06-0.14 vs 0.00. Snares cannot be told from piano partials by level
# (piano 0.10 > snares 0.01-0.03), but a snare is NOISE: spectral flatness 1-8 kHz of
# real snares 0.32-0.40, piano attacks 0.00 -> SNARE_MIN_FLATNESS.
SNARE_MIN_FLATNESS = 0.2


def band_onsets(x: np.ndarray, sr: int, lo: float, hi: float, rel: float,
                min_gap_s: float = 0.06) -> list[tuple[float, float]]:
    """(onset time, strength) where a band's envelope rises steeply; strength = the
    envelope peak in the next 30 ms; kept if >= rel x the strongest onset in the band."""
    env, esr = band_envelope(x, sr, lo, hi, hop=32, smooth_ms=2.0)
    rise = np.diff(env, prepend=env[0])
    rise[rise < 0] = 0
    peaks, _ = signal.find_peaks(rise, distance=max(1, int(min_gap_s * esr)),
                                 height=0.05 * rise.max() if rise.max() > 0 else None)
    w = int(0.03 * esr)
    out = [(p / esr, float(env[p:p + w].max(initial=0.0))) for p in peaks]
    if not out:
        return []
    top = max(s for _, s in out)
    return [(t, s) for t, s in out if s >= rel * top]


def _hat_is_open(x: np.ndarray, sr: int, t: float) -> bool:
    env, esr = band_envelope(x, sr, 7000.0, 16000.0, hop=32)
    i = int(t * esr)
    head = env[i:i + int(0.03 * esr)].mean() if i < len(env) else 0.0
    tail = env[i + int(0.06 * esr): i + int(0.12 * esr)].mean() if i < len(env) else 0.0
    return head > 0 and tail / head > 0.25


def flatness(x: np.ndarray, sr: int, t: float, lo=1000.0, hi=8000.0, win=0.03) -> float:
    """Spectral flatness (geometric / arithmetic mean power) in [lo, hi] over win after t:
    ~0.3-0.5 for noise (snare, hats), ~0 for tones (piano partials)."""
    seg = x[int(t * sr): int((t + win) * sr)]
    if len(seg) < 256:
        return 0.0
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    S = S[(f >= lo) & (f <= hi)] + 1e-12
    return float(np.exp(np.mean(np.log(S))) / np.mean(S))


def transcribe_drums(perc: np.ndarray, sr: int, peak: float) -> list[tuple[float, int, int]]:
    """-> [(time_s, GM pitch, velocity)]. peak: the clip's peak |amplitude|."""
    hits = []
    for name, (lo, hi, rel, pitch, floor) in DRUM_BANDS.items():
        ons = [(t, s) for t, s in band_onsets(perc, sr, lo, hi, rel) if s >= floor * peak]
        if name == "snare":
            ons = [(t, s) for t, s in ons if flatness(perc, sr, t) >= SNARE_MIN_FLATNESS]
        if not ons:
            continue
        top = max(s for _, s in ons)
        for t, s in ons:
            p = 46 if name == "hat" and _hat_is_open(perc, sr, t) else pitch
            hits.append((t, p, int(np.clip(40 + 87 * (s / top), 1, 127))))
    # a hat ON a snare hit rings on in the hat band because of the SNARE's noise tail,
    # which the decay test reads as an open hat (P1: all 4 "open hats" sat on snares)
    snares = [t for t, p, _ in hits if p == 38]
    hits = [(t, 42 if p == 46 and any(abs(t - u) < 0.01 for u in snares) else p, v) for t, p, v in hits]
    return sorted(hits)


def _drop_ghosts(notes: list[tuple[float, float, int, float]]) -> list[tuple[float, float, int, float]]:
    """Drop quiet notes 12 / 19 / 24 semitones above a louder note that overlaps them."""
    keep = []
    for i, (s, e, p, a) in enumerate(notes):
        ghost = False
        for j, (s2, e2, p2, a2) in enumerate(notes):
            if i == j or p - p2 not in (12, 19, 24) or a >= 0.75 * a2:
                continue
            overlap = min(e, e2) - max(s, s2)
            if overlap > 0.5 * (e - s):
                ghost = True
                break
        if not ghost:
            keep.append((s, e, p, a))
    return keep


def transcribe_pitched(harm: np.ndarray, sr: int) -> list[tuple[float, float, int, float]]:
    """Basic Pitch -> [(start_s, end_s, MIDI pitch, amplitude 0..1)]"""
    import logging
    logging.getLogger().setLevel(logging.ERROR)      # basic-pitch warns about absent TF/CoreML
    from basic_pitch.inference import predict
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "harmonic.wav"
        save(p, harm[:, None], sr, "PCM_16", "WAV")
        _, _, events = predict(str(p))
    notes = [(float(s), float(e), int(pt), float(a)) for s, e, pt, a, *_ in events]
    notes = [n for n in notes if n[3] >= 0.2]
    return _drop_ghosts(sorted(notes))


def _tempo_and_origin(mono: np.ndarray, sr: int) -> tuple[float, int, int, float, dict]:
    dur = len(mono) / sr
    reps = max(1, int(np.ceil(MIN_ANALYSIS_S / dur)))
    g = analyze(np.tile(mono, reps)[:, None], sr)
    bpb = g.beats_per_bar
    bar_s = bpb * 60.0 / g.tempo_bpm
    bars = max(1, int(round(dur / bar_s)))
    whole = abs(dur - bars * bar_s) / dur < 0.03
    tempo = bars * bpb * 60.0 / dur if whole else g.tempo_bpm
    downs = g.beats[g.positions == 1]
    origin = float(downs[0] % (bars * bpb * 60.0 / tempo)) if len(downs) else 0.0
    bar_len = bpb * 60.0 / tempo
    origin = origin % bar_len
    if origin > bar_len - 0.05:
        origin = 0.0
    info = {"tracker_tempo_bpm": round(g.tempo_bpm, 3), "analysed_copies": reps,
            "clip_is_whole_bars": bool(whole), "tempo_from": "clip length" if whole else "beat tracker",
            "time_signature": g.time_signature, "first_downbeat_s": round(origin, 4)}
    return tempo, bpb, bars, origin, info


def transcribe(audio: np.ndarray, sr: int, name: str = "clip") -> Score:
    import librosa
    t0 = time.time()
    mono = to_mono(audio).astype(np.float32)
    tempo, bpb, bars, origin, info = _tempo_and_origin(mono, sr)
    spb = 60.0 / tempo
    loop_beats = bars * bpb
    harm, perc = librosa.effects.hpss(mono, margin=(1.0, 2.0))

    def beat_of(t: float) -> float:
        b = (t - origin) / spb
        return round(b % loop_beats, 4)

    tracks = []
    drums = transcribe_drums(perc, sr, float(np.abs(mono).max()) or 1.0)
    if drums:
        tracks.append(Track(name="drums", kind="drums", patch=default_patch("drums"),
                            notes=[Note(start=beat_of(t), duration=0.25, pitch=p, velocity=v) for t, p, v in drums]))
    pitched = transcribe_pitched(harm, sr)
    low = [n for n in pitched if n[2] < BASS_SPLIT]
    high = [n for n in pitched if n[2] >= BASS_SPLIT]
    groups = [("bass", "bass", low), ("keys", "keys", high)] if low and high else \
             [("bass" if low else "keys", "bass" if low else "keys", low or high)] if pitched else []
    for tname, kind, ns in groups:
        tracks.append(Track(name=tname, kind=kind, patch=default_patch(kind), notes=[
            Note(start=beat_of(s), duration=max(0.05, round((e - s) / spb, 4)), pitch=p,
                 velocity=int(np.clip(round(a * 127), 1, 127))) for s, e, p, a in ns]))
    for t in tracks:
        t.notes.sort(key=lambda n: (n.start, n.pitch))
    meter = f"{bpb}/4" if info["time_signature"].endswith("/4") else info["time_signature"]
    return Score(tempo_bpm=round(tempo, 4), time_signature=meter, beats_per_bar=bpb, bars=bars,
                 loop_beats=loop_beats,
                 source={"name": name, "duration_s": round(len(mono) / sr, 4), "sr": sr,
                         "loop_start_s": round(origin, 4)},
                 analysis={**info, "drums": "HPSS percussive part, per-band onsets (kick 30-120 Hz, "
                                    "snare 1-5 kHz, hats 7-16 kHz), relative thresholds",
                           "pitched": "Basic Pitch 0.4 (ONNX) on the HPSS harmonic part, overtone ghosts dropped",
                           "llm_used": False, "seconds": round(time.time() - t0, 2)},
                 tracks=tracks)
