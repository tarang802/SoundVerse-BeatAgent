"""Generate the SYNTHETIC SUBSTITUTE test pack (A1-A7) with exact references.

    python -m evals.make_testpack

The real Soundverse test pack was not provided. These clips mimic each
item's *described property* (tempo change, ambient intro, 6/8, live drift,
half-time feel...) so every code path is exercised, but they are much easier
than real recordings. Results on them must never be presented as the
assignment benchmark. Output: data/substitute/{A1..A7}.{wav|flac|mp3} plus
.beats files ("time<TAB>position", position 1 = downbeat).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.synth import SR, beat_times_constant, render_groove  # noqa: E402

OUT = ROOT / "data" / "substitute"


def _write(name, audio, beats, bpb, fmt, desc, beats_only=False, ref_from=None, sr=SR, subtype="PCM_16"):
    OUT.mkdir(parents=True, exist_ok=True)
    ext = {"WAV": "wav", "FLAC": "flac", "MP3": "mp3"}[fmt]
    if sr != SR:
        import soxr
        audio = soxr.resample(audio, SR, sr)
    path = OUT / f"{name}.{ext}"
    sf.write(path, audio, sr, format=fmt, subtype=subtype)
    positions = (np.arange(len(beats)) % bpb) + 1
    keep = np.ones(len(beats), bool) if ref_from is None else beats >= ref_from - 1e-9
    with open(OUT / f"{name}.beats", "w") as f:
        for t, p in zip(beats[keep], positions[keep]):
            f.write(f"{t:.6f}\t{0 if beats_only else p}\n")
    meta = {"file": path.name, "description": desc, "sr": sr, "subtype": subtype, "beats_per_bar": bpb,
            "n_ref_beats": int(keep.sum()), "has_downbeats": not beats_only,
            "reference_starts_at": ref_from, "duration_s": round(len(audio) / sr, 3)}
    print(f"  {name}: {meta}")
    return meta


def main():
    metas = {}
    rng = np.random.default_rng(7)
    # A1: 124 BPM house, ~6.0 min, with a 16-bar drumless breakdown (pad only) at bars 81-96
    b = beat_times_constant(124.0, 0.48, 186 * 4)
    secs = [(0, 64, {"kick", "hat"}), (64, 320, {"kick", "snare", "hat", "bass"}),
            (320, 384, {"pad"}), (384, 640, {"kick", "snare", "hat", "bass"}), (640, 744, {"kick", "hat"})]
    a = render_groove(b, duration=b[-1] + 1.5, style="house", seed=1, sections=secs)
    metas["A1"] = _write("A1", a, b, 4, "WAV", "house 124 BPM constant, 186 bars, drumless pad breakdown bars 81-96")
    # A2: hip-hop half-time, 86 BPM quarter-note beats, 16th hats, snare on 3
    b = beat_times_constant(86.0, 0.3, 56 * 4)
    a = render_groove(b, duration=b[-1] + 1.5, style="halftime", seed=2, bass_root=33)
    metas["A2"] = _write("A2", a, b, 4, "WAV", "half-time hip-hop 86 BPM, 16th hats", sr=48000, subtype="PCM_24")
    # A3: 'live band' -- tempo random walk around 112 BPM, 12 ms per-hit jitter, beats only
    n = 90 * 4
    ibi = 60 / 112.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    ibi = np.clip(ibi, 60 / 118, 60 / 106)
    b = 0.6 + np.concatenate([[0], np.cumsum(ibi[:-1])])
    a = render_groove(b, duration=b[-1] + 1.5, style="rock", seed=3, jitter_ms=12)
    metas["A3"] = _write("A3", a, b, 4, "FLAC", "rock 'live' 106-118 BPM drift, 12 ms jitter, no click",
                         beats_only=True, subtype="PCM_24")
    # A4: 45 s ambient pad intro (no drums), then house 118 BPM; reference starts at the drum entry
    bpm = 118.0
    bar = 4 * 60 / bpm
    n_intro_bars = int(np.ceil(45 / bar))
    b = beat_times_constant(bpm, 0.5, (n_intro_bars + 48) * 4)
    secs = [(0, n_intro_bars * 4, {"pad"}), (n_intro_bars * 4, len(b), {"kick", "snare", "hat", "bass"})]
    a = render_groove(b, duration=b[-1] + 1.5, style="house", seed=4, sections=secs)
    drums_at = b[n_intro_bars * 4]
    metas["A4"] = _write("A4", a, b, 4, "WAV", f"ambient pad intro {drums_at:.1f} s, then house 118 BPM",
                         ref_from=drums_at)
    # A5: 120 -> 128 BPM step change at bar 33
    b1 = beat_times_constant(120.0, 0.5, 32 * 4)
    b2 = beat_times_constant(128.0, b1[-1] + 0.5, 40 * 4)
    b = np.concatenate([b1, b2])
    a = render_groove(b, duration=b[-1] + 1.5, style="house", seed=5)
    metas["A5"] = _write("A5", a, b, 4, "MP3", "electronic, 120 BPM bars 1-32 then 128 BPM from bar 33",
                         subtype="MPEG_LAYER_III")
    # A6: 6/8 folk-pop with dholak-ish percussion, beats = dotted quarters (2 per bar) at 68 BPM
    b = beat_times_constant(68.0, 0.4, 64 * 2)
    a = render_groove(b, bpb=2, duration=b[-1] + 1.5, style="six8", seed=6)
    metas["A6"] = _write("A6", a, b, 2, "WAV", "6/8 at 68 dotted-quarter BPM; reference beats = dotted quarters")
    # A7: house at 128 BPM (sync partner for A1)
    b = beat_times_constant(128.0, 0.35, 97 * 4)
    a = render_groove(b, duration=b[-1] + 1.5, style="house", seed=8, bass_root=38)
    metas["A7"] = _write("A7", a, b, 4, "WAV", "house 128 BPM (sync partner for A1)")
    with open(OUT / "index.json", "w") as f:
        json.dump({"label": "SYNTHETIC SUBSTITUTE for the unavailable assignment test pack", "items": metas}, f, indent=1)


if __name__ == "__main__":
    main()
