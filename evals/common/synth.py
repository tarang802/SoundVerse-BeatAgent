"""Synthetic audio with exactly known beat times.

SYNTHETIC SUBSTITUTE ONLY. The assignment's test pack (A1-A7, P1-P4) was not
provided, so these renders stand in for it. Numbers measured on them say the
pipeline is wired correctly; they are NOT the assignment benchmark and are
easier than real music (clean transients, no mastering, no reverb tails).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SR = 44100


def _env(n, decay_s, sr=SR):
    t = np.arange(n) / sr
    return np.exp(-t / decay_s)


def kick(sr=SR, rng=None):
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    f = 45 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / sr
    click = np.zeros(n)
    click[: int(0.002 * sr)] = np.hanning(int(0.002 * sr)) * 0.5
    return (np.sin(ph) * _env(n, 0.18) + click) * 0.9


def snare(sr=SR, rng=None):
    rng = rng or np.random.default_rng(1)
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0) * 0.5          # crude high-pass
    tone = np.sin(2 * np.pi * 185 * t) * _env(n, 0.05)
    return (noise * _env(n, 0.07) * 0.35 + tone * 0.5) * 0.8


def hat(sr=SR, rng=None, open_=False):
    rng = rng or np.random.default_rng(2)
    n = int((0.2 if open_ else 0.06) * sr)
    noise = rng.standard_normal(n)
    hp = np.diff(np.diff(noise, prepend=0), prepend=0) * 0.25
    return hp * _env(n, 0.08 if open_ else 0.015) * 0.35


def clap(sr=SR, rng=None):
    rng = rng or np.random.default_rng(3)
    n = int(0.2 * sr)
    noise = np.diff(rng.standard_normal(n), prepend=0)
    e = np.zeros(n)
    for d in (0.0, 0.01, 0.02):
        s = int(d * sr)
        e[s:] += _env(n - s, 0.012 if d < 0.02 else 0.06)
    return noise * e * 0.25


def tone(freq, dur, sr=SR, kind="saw", decay=None, cutoff=None):
    n = int(dur * sr)
    t = np.arange(n) / sr
    if kind == "saw":
        x = 2 * ((freq * t) % 1.0) - 1
    elif kind == "square":
        x = np.sign(np.sin(2 * np.pi * freq * t))
    elif kind == "pluck":   # additive, piano-ish: harmonics with faster decay up top
        x = sum(np.sin(2 * np.pi * freq * k * t) * np.exp(-t * (1.5 + 1.2 * k)) / k for k in range(1, 7))
    else:
        x = np.sin(2 * np.pi * freq * t)
    if cutoff:                     # one-pole low-pass
        a = np.exp(-2 * np.pi * cutoff / sr)
        x = lfilter([1 - a], [1, -a], x)
    env = np.ones(n)
    a_n = min(n, int(0.004 * sr))
    env[:a_n] = np.linspace(0, 1, a_n)
    r_n = min(n, int(0.01 * sr))
    env[-r_n:] *= np.linspace(1, 0, r_n)
    if decay:
        env *= _env(n, decay)
    return x * env


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def place(buf, sig, t, gain=1.0, sr=SR):
    i = int(round(t * sr))
    if i >= len(buf) or i < 0:
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def beat_times_constant(bpm, start, n_beats):
    return start + np.arange(n_beats) * 60.0 / bpm


def subdiv_time(beats, x):
    """Time of fractional beat position x (linear between neighbouring beats)."""
    i = int(np.floor(x))
    if i >= len(beats) - 1:
        p = beats[-1] - beats[-2]
        return beats[-1] + (x - (len(beats) - 1)) * p
    return beats[i] + (x - i) * (beats[i + 1] - beats[i])


def render_groove(beats, bpb=4, duration=None, style="house", seed=0, jitter_ms=0.0,
                  drums_from_beat=0, bass_root=36, sections=None):
    """Render a drum+bass groove on the given beat times.

    sections: optional list of (start_beat, end_beat, flags) with flags subset of
    {"kick","snare","hat","bass","pad"} to build arrangement (breakdowns etc.).
    """
    rng = np.random.default_rng(seed)
    duration = duration or (beats[-1] + 2.0)
    buf = np.zeros(int(duration * SR) + SR)
    K, S, H, HO, C = kick(), snare(), hat(rng=rng), hat(rng=rng, open_=True), clap(rng=rng)
    j = lambda: rng.normal(0, jitter_ms / 1000.0) if jitter_ms else 0.0

    def on(flag, b):
        if b < drums_from_beat and flag != "pad":
            return False
        if not sections:
            return True
        for a, e, fl in sections:
            if a <= b < e:
                return flag in fl
        return True

    bass_notes = [0, 0, 3, 5, 7, 5, 3, -2]
    for b in range(len(beats)):
        pos = b % bpb
        t = beats[b]
        if style == "house":
            if on("kick", b): place(buf, K, t + j())
            if on("snare", b) and pos in (1, 3): place(buf, C, t + j(), 0.8)
            if on("hat", b): place(buf, HO, subdiv_time(beats, b + 0.5) + j(), 0.6)
            if on("bass", b):
                bt = subdiv_time(beats, b + 0.5)
                p = beats[min(b + 1, len(beats) - 1)] - t
                place(buf, tone(midi_hz(bass_root + bass_notes[(b // bpb) % 8]), max(p * 0.45, 0.05), cutoff=500), bt, 0.35)
        elif style == "halftime":     # hip-hop: kick 1 (+and of 2), snare on 3, 32nd hats
            if on("kick", b) and pos == 0: place(buf, K, t + j())
            if on("kick", b) and pos == 1: place(buf, K, subdiv_time(beats, b + 0.5) + j(), 0.7)
            if on("snare", b) and pos == 2: place(buf, S, t + j())
            if on("hat", b):
                for q in range(4):
                    place(buf, hat(rng=rng), subdiv_time(beats, b + q / 4) + j(), 0.5 if q else 0.8)
            if on("bass", b) and pos == 0:
                p = beats[min(b + 1, len(beats) - 1)] - t
                place(buf, tone(midi_hz(bass_root + bass_notes[(b // bpb) % 8]), p * 1.8, cutoff=300), t, 0.4)
        elif style == "rock":         # live band: kick 1/3, snare 2/4, 8th hats
            if on("kick", b) and pos in (0, 2): place(buf, K, t + j())
            if on("snare", b) and pos in (1, 3): place(buf, S, t + j())
            if on("hat", b):
                for q in (0, 0.5):
                    place(buf, hat(rng=rng), subdiv_time(beats, b + q) + j(), 0.6)
            if on("bass", b):
                p = beats[min(b + 1, len(beats) - 1)] - t
                place(buf, tone(midi_hz(bass_root + 4 + bass_notes[(b // bpb) % 8]), p * 0.9, cutoff=700), t + j(), 0.35)
        elif style == "six8":         # beats are dotted quarters; 3 eighths per beat; dholak-ish
            if on("kick", b) and pos == 0: place(buf, K, t + j(), 0.8)
            for q in range(3):
                tt = subdiv_time(beats, b + q / 3) + j()
                if on("hat", b): place(buf, S if (q == 0 and pos == 1) else hat(rng=rng), tt, 0.5 if q else 0.8)
            if on("bass", b):
                p = beats[min(b + 1, len(beats) - 1)] - t
                place(buf, tone(midi_hz(bass_root + 7 + bass_notes[(b // bpb) % 8]), p * 0.6, kind="pluck"), t, 0.5)
        elif style == "waltz":        # 3/4: strong kick on 1, lighter snare on 2 and 3, 8th hats
            if on("kick", b) and pos == 0: place(buf, K, t + j())
            if on("snare", b) and pos in (1, 2): place(buf, S, t + j(), 0.45)
            if on("hat", b):
                for q in (0, 0.5):
                    place(buf, hat(rng=rng), subdiv_time(beats, b + q) + j(), 0.5)
            if on("bass", b) and pos == 0:
                p = beats[min(b + bpb, len(beats) - 1)] - t if b + bpb < len(beats) else 1.0
                place(buf, tone(midi_hz(bass_root + bass_notes[(b // bpb) % 8]), p * 0.8, cutoff=400), t, 0.4)
        elif style == "swing":        # house kick/clap with SWUNG off-beat hats (2/3 of the beat)
            if on("kick", b): place(buf, K, t + j())
            if on("snare", b) and pos in (1, 3): place(buf, C, t + j(), 0.8)
            if on("hat", b):
                place(buf, hat(rng=rng), subdiv_time(beats, b + 2 / 3) + j(), 0.6)
                place(buf, hat(rng=rng), t + j(), 0.3)
            if on("bass", b):
                bt = subdiv_time(beats, b + 2 / 3)
                p = beats[min(b + 1, len(beats) - 1)] - t
                place(buf, tone(midi_hz(bass_root + bass_notes[(b // bpb) % 8]), max(p * 0.3, 0.05), cutoff=500), bt, 0.35)
        if on("pad", b) and pos == 0 and sections:
            p = beats[min(b + bpb, len(beats) - 1)] - t if b + bpb < len(beats) else 2.0
            chord = [60, 63, 67] if (b // bpb) % 2 == 0 else [58, 62, 65]
            for m in chord:
                sig = tone(midi_hz(m), max(p, 0.2), kind="sine")
                # slow attack so the pad carries no sharp onset
                a = min(len(sig), int(0.4 * SR))
                sig[:a] *= np.linspace(0, 1, a)
                place(buf, sig, t, 0.08)
    buf = buf[: int(duration * SR)]
    return (buf / max(1e-9, np.max(np.abs(buf))) * 0.89).astype(np.float32)


def fixture_120(duration=30.0, offset=0.5):
    """Tiny smoke-test fixture: 120 BPM house, 4/4, first beat at `offset` s."""
    beats = beat_times_constant(120.0, offset, int((duration - offset - 0.5) * 2))
    audio = render_groove(beats, duration=duration)
    return audio, beats, beats[::4]
