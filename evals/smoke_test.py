"""Environment + pipeline smoke test on a SYNTHETIC fixture.

    python -m evals.smoke_test

This is NOT the assignment benchmark: it renders a 30 s, 120 BPM synthetic
groove with exactly known beat times and checks each stage of the toolchain
against it. Results go to evals/results/smoke_test.json.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.onsets import onset_times, match_offsets  # noqa: E402
from evals.synth import SR, fixture_120  # noqa: E402

OUT = ROOT / "evals" / "results" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)
report: dict = {"label": "SYNTHETIC SMOKE TEST - not an assignment benchmark"}


def section(name):
    print(f"\n== {name} ==")


def xcorr_lag(a, b, max_lag):
    """Lag (samples) of b relative to a maximising correlation."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best, best_l = -1e18, 0
    for l in range(-max_lag, max_lag + 1):
        if l >= 0:
            c = np.dot(a[: n - l], b[l:])
        else:
            c = np.dot(a[-l:], b[: n + l])
        if c > best:
            best, best_l = c, l
    return best_l


# ---------------------------------------------------------------- deps
section("dependencies")
deps = {}
for m in ["numpy", "scipy", "soundfile", "soxr", "mir_eval", "torch", "beat_this", "pylibrb",
          "librosa", "basic_pitch", "playwright", "google.genai"]:
    try:
        mod = importlib.import_module(m)
        deps[m] = f"ok {getattr(mod, '__version__', '')}".strip()
    except Exception as e:  # noqa: BLE001
        deps[m] = f"MISSING ({type(e).__name__})"
    print(f"  {m:12s} {deps[m]}")
deps["ffmpeg_binary"] = shutil.which("ffmpeg") or "MISSING"
print(f"  ffmpeg       {deps['ffmpeg_binary']}")
report["dependencies"] = deps

audio, ref_beats, ref_down = fixture_120()
sf.write(OUT / "fixture_120.wav", audio, SR, subtype="PCM_16")
report["fixture"] = {"bpm": 120, "duration_s": 30.0, "first_beat_s": 0.5,
                     "n_beats": len(ref_beats), "note": "synthetic house groove, exact beat times"}

# ---------------------------------------------------------------- decode
section("decode WAV / FLAC / MP3 (libsndfile)")
dec = {}
for ext, sub in [("wav", "PCM_16"), ("flac", "PCM_24"), ("mp3", "MPEG_LAYER_III")]:
    p = OUT / f"fixture_120.{ext}"
    try:
        sf.write(p, audio, SR, subtype=sub)
        y, sr = sf.read(p, dtype="float32")
        info = sf.info(p)
        lag = xcorr_lag(audio[: SR * 5], y[: SR * 5], 3000)
        n = min(len(audio), len(y)) - abs(lag)
        a = audio[max(0, -lag): max(0, -lag) + n]
        b = y[max(0, lag): max(0, lag) + n]
        err_db = 20 * np.log10(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2)) + 1e-12)
        dec[ext] = {"ok": True, "sr": sr, "subtype": info.subtype, "len_diff_samples": len(y) - len(audio),
                    "decoder_offset_samples": int(lag), "decoder_offset_ms": round(lag / SR * 1000, 2),
                    "residual_db": round(float(err_db), 1)}
    except Exception as e:  # noqa: BLE001
        dec[ext] = {"ok": False, "error": str(e)}
    print(f"  {ext}: {dec[ext]}")
# MP3 decoders can add encoder delay; check kick timing survives the round trip
y3, _ = sf.read(OUT / "fixture_120.mp3", dtype="float32")
k_wav = onset_times(audio, SR, hi=100, min_gap_s=0.4)
k_mp3 = onset_times(y3, SR, hi=100, min_gap_s=0.4)
o = match_offsets(k_wav, k_mp3, 0.05) * 1000
dec["mp3"]["kick_timing_vs_wav_ms"] = {"median": round(float(np.median(o)), 3), "max_abs": round(float(np.max(np.abs(o))), 3)}
print(f"  mp3 kick timing vs wav: {dec['mp3']['kick_timing_vs_wav_ms']}")
report["decode"] = dec

# ---------------------------------------------------------------- mir_eval known example
section("mir_eval known example")
import mir_eval  # noqa: E402

ref = np.array([6.0, 7.0, 8.0, 9.0])
est = np.array([6.01, 7.1, 8.0, 10.0])   # 2 of 4 within 70 ms -> P=R=F=0.5
f = mir_eval.beat.f_measure(ref, est, f_measure_threshold=0.07)
early = mir_eval.beat.evaluate(np.array([1.0, 2.0, 3.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0, 6.0, 7.0]))
report["mir_eval_check"] = {"expected_f": 0.5, "got_f": f, "pass": abs(f - 0.5) < 1e-9,
                            "note": "mir_eval.beat.evaluate() trims beats before 5 s by default; "
                                    "our harness calls trim_beats explicitly and reports it"}
print(f"  f_measure expected 0.5 got {f}")

# ---------------------------------------------------------------- rubber band
section("Rubber Band (pylibrb) time-stretch")
rb = {}
try:
    from beatagent.stretch import stretch
    # pitch preservation: 440 Hz sine stretched by 1.25 must stay 440 Hz
    t = np.arange(SR * 2) / SR
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None]
    t0 = time.time()
    s = stretch(sine, SR, 1.25)[:, 0]
    seg = s[SR // 2: SR // 2 + SR]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f_peak = np.argmax(spec) * SR / len(seg)
    rb["pitch_check"] = {"in_hz": 440, "out_hz": round(float(f_peak), 2),
                         "len_ratio": round(len(s) / len(sine), 5), "expected_len_ratio": 1.25}
    # timing: stretch the groove 120 -> 124 BPM (time_ratio 120/124); kicks must land at t*ratio
    ratio = 120 / 124
    g = stretch(audio[:, None], SR, ratio)[:, 0]
    sf.write(OUT / "fixture_120_to_124.wav", g, SR, subtype="PCM_16")
    k_in = onset_times(audio, SR, hi=100, min_gap_s=0.4)
    k_out = onset_times(g, SR, hi=100, min_gap_s=0.4)
    off = match_offsets(k_in * ratio, k_out, 0.05) * 1000
    rb["timing_check"] = {"time_ratio": round(ratio, 6), "kicks_matched": int(len(off)), "kicks_in": int(len(k_in)),
                          "median_ms": round(float(np.median(off)), 2), "max_abs_ms": round(float(np.max(np.abs(off))), 2)}
    rb["elapsed_s"] = round(time.time() - t0, 2)
    rb["ok"] = bool(abs(f_peak - 440) < 3 and abs(np.median(off)) < 5)
except Exception as e:  # noqa: BLE001
    rb = {"ok": False, "error": repr(e)}
print(" ", json.dumps(rb, indent=1))
report["rubberband"] = rb

# ---------------------------------------------------------------- beat this
section("Beat This! on CPU")
bt = {}
ckpt = ROOT / "models" / "small0.ckpt"
try:
    import torch
    from beat_this.inference import Audio2Beats
    if not ckpt.exists() or ckpt.stat().st_size < 8_000_000:
        raise FileNotFoundError(f"checkpoint not downloaded yet: {ckpt}")
    t0 = time.time()
    a2b = Audio2Beats(checkpoint_path=str(ckpt), device="cpu", dbn=False)
    t1 = time.time()
    beats, downs = a2b(audio, SR)
    t2 = time.time()
    fm = mir_eval.beat.f_measure(ref_beats, beats, 0.07)
    fd = mir_eval.beat.f_measure(ref_down, downs, 0.07)
    off = match_offsets(ref_beats, np.asarray(beats), 0.07) * 1000
    bt = {"ok": bool(fm > 0.9), "checkpoint": ckpt.name, "load_s": round(t1 - t0, 2), "infer_s": round(t2 - t1, 2),
          "n_beats": len(beats), "n_downbeats": len(downs), "beat_F": round(fm, 4), "downbeat_F": round(fd, 4),
          "beat_offset_ms_median": round(float(np.median(off)), 2), "beat_offset_ms_max_abs": round(float(np.max(np.abs(off))), 2),
          "first_beats": [round(float(x), 3) for x in beats[:6]], "torch_threads": torch.get_num_threads()}
except Exception as e:  # noqa: BLE001
    bt = {"ok": False, "error": repr(e)}
print(" ", json.dumps(bt, indent=1))
report["beat_this"] = bt

# ---------------------------------------------------------------- librosa
section("librosa baseline")
lb = {}
try:
    import librosa
    t0 = time.time()
    tempo, frames = librosa.beat.beat_track(y=audio, sr=SR, hop_length=256, units="frames")
    beats = librosa.frames_to_time(frames, sr=SR, hop_length=256)
    fm = mir_eval.beat.f_measure(ref_beats, beats, 0.07)
    off = match_offsets(ref_beats, beats, 0.07) * 1000
    lb = {"ok": True, "elapsed_s": round(time.time() - t0, 2), "tempo": float(np.atleast_1d(tempo)[0]),
          "beat_F": round(fm, 4), "beat_offset_ms_median": round(float(np.median(off)), 2) if len(off) else None}
except Exception as e:  # noqa: BLE001
    lb = {"ok": False, "error": repr(e)}
print(" ", json.dumps(lb, indent=1))
report["librosa"] = lb

# ---------------------------------------------------------------- test pack
section("assignment test pack availability")
tp = ROOT / "data" / "testpack"
found = sorted(p.name for p in tp.glob("*")) if tp.exists() else []
report["assignment_test_pack"] = {"path": str(tp), "found": found,
                                  "status": "AVAILABLE" if found else "UNAVAILABLE - not provided with the brief; "
                                  "all assignment benchmarks marked unavailable"}
print(" ", report["assignment_test_pack"]["status"])

(ROOT / "evals" / "results").mkdir(exist_ok=True)
with open(ROOT / "evals" / "results" / "smoke_test.json", "w") as fh:
    json.dump(report, fh, indent=1, default=str)
print("\nwrote evals/results/smoke_test.json")
