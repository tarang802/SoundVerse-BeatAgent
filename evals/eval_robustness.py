"""Part 1 robustness: ~30 generated tracks with KNOWN beats, through the real file path.

    python -m evals.eval_robustness

Covers tempo range (60-200 BPM), grooves (house, half-time, swing, live drift,
waltz 3/4, 6/8), tempo ramps and multi-step tempo changes, arrangement (breakdown,
fade in/out, leading silence), levels (very quiet, clipped, DC offset), formats
(22.05/44.1/48/96 kHz, 8/16/24-bit/float, mono/stereo, WAV/FLAC/MP3) and things that
are not music (silence, noise, pure tone, very short clips).

Per case: success, tempo (correct up to the 80-160 BPM octave rule), strict beat F,
AMLt (accepts octave/off-beat counting), downbeat F (when the octave is unchanged),
drift on steady clips, grid type, speed. Format cases also edit + export the file
and check sample rate / bit depth survive. Non-music must be refused or flagged
(confidence < 0.5). SYNTHETIC: easier than real music (clean transients, no mix).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.analyze import analyze  # noqa: E402
from beatagent.audio_io import load  # noqa: E402
from beatagent.edl import Range, Remove  # noqa: E402
from beatagent.session import Session  # noqa: E402
from evals.synth import SR, beat_times_constant, render_groove  # noqa: E402

INP = ROOT / "work" / "robust_inputs"
OUT = ROOT / "evals" / "results" / "part1_robustness"


# ------------------------------------------------------------------ track builders
def groove(bpm, bars=32, bpb=4, style="house", start=0.5, seed=0, **kw):
    b = beat_times_constant(bpm, start, bars * bpb)
    a = render_groove(b, bpb=bpb, duration=b[-1] + 1.0, style=style, seed=seed, **kw)
    return a, b, b[::bpb], bpm


def tempo_curve(bpms_per_beat, start=0.5):
    ibi = 60.0 / np.asarray(bpms_per_beat, float)
    return start + np.concatenate([[0], np.cumsum(ibi[:-1])])


def ramp(b0, b1, bars=40, style="house", seed=0):
    n = bars * 4
    b = tempo_curve(np.linspace(b0, b1, n))
    return render_groove(b, duration=b[-1] + 1.0, style=style, seed=seed), b, b[::4], None


def steps(bpms, bars_each=16):
    per = np.concatenate([[x] * bars_each * 4 for x in bpms])
    b = tempo_curve(per)
    return render_groove(b, duration=b[-1] + 1.0, style="house", seed=3), b, b[::4], None


def with_env(case, env_fn):
    a, b, d, bpm = case
    t = np.arange(len(a)) / SR
    return (a * env_fn(t)).astype(np.float32), b, d, bpm


def lead_silence(case, secs):
    a, b, d, bpm = case
    return np.concatenate([np.zeros(int(secs * SR), np.float32), a]), b + secs, d + secs, bpm


def breakdown(bpm=124, seed=5):
    bars = 48
    b = beat_times_constant(bpm, 0.5, bars * 4)
    secs = [(0, 64, {"kick", "snare", "hat", "bass"}), (64, 128, {"pad"}), (128, bars * 4, {"kick", "snare", "hat", "bass"})]
    return render_groove(b, duration=b[-1] + 1.0, style="house", seed=seed, sections=secs), b, b[::4], bpm


def short(secs):
    a, b, d, bpm = groove(124, bars=8)
    n = int(secs * SR)
    keep = b < secs - 0.1
    return a[:n], b[keep], d[d < secs - 0.1], bpm


CASES = [
    # name, category, builder, (container, subtype, sr, channels), kind, expect_grid
    ("house_124", "tempo", lambda: groove(124), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("rock_60", "tempo", lambda: groove(60, bars=20, style="rock"), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("rock_90", "tempo", lambda: groove(90, style="rock"), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("halftime_140", "tempo", lambda: groove(140, style="halftime", bass_root=33), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("dnb_174", "tempo", lambda: groove(174, bars=48, style="rock"), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("fast_200", "tempo", lambda: groove(200, bars=48, style="house"), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("swing_118", "groove", lambda: groove(118, style="swing"), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("waltz_3-4_120", "meter", lambda: groove(120, bars=40, bpb=3, style="waltz"), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("six8_68", "meter", lambda: groove(68, bars=32, bpb=2, style="six8"), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("live_drift_112", "tempo-var", lambda: (lambda b: (render_groove(b, duration=b[-1] + 1, style="rock", seed=4, jitter_ms=10), b, b[::4], None))(
        tempo_curve(112 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.003, 160))))), ("FLAC", "PCM_24", 44100, 2), "music", None),
    ("ramp_110-130", "tempo-var", lambda: ramp(110, 130), ("WAV", "PCM_16", 44100, 2), "music", None),
    ("steps_120-128-100", "tempo-var", lambda: steps([120, 128, 100]), ("WAV", "PCM_16", 44100, 2), "music", "piecewise_constant"),
    ("breakdown_124", "arrangement", breakdown, ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("fade_in_out", "arrangement", lambda: with_env(groove(124), lambda t: np.clip(np.minimum(t / 8, (t[-1] - t) / 8), 0, 1)),
     ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("lead_silence_10s", "arrangement", lambda: lead_silence(groove(124), 10.0), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("very_quiet_-40dB", "level", lambda: with_env(groove(124), lambda t: np.full_like(t, 0.01)), ("WAV", "PCM_24", 44100, 2), "music", "constant"),
    ("clipped_+12dB", "level", lambda: (lambda c: (np.clip(c[0] * 4, -1, 1).astype(np.float32),) + c[1:])(groove(124)),
     ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("dc_offset", "level", lambda: (lambda c: ((c[0] * 0.7 + 0.25).astype(np.float32),) + c[1:])(groove(124)),
     ("WAV", "FLOAT", 44100, 2), "music", "constant"),
    ("fmt_22k_mono_16", "format", lambda: groove(124), ("WAV", "PCM_16", 22050, 1), "music", "constant"),
    ("fmt_48k_stereo_24", "format", lambda: groove(124), ("WAV", "PCM_24", 48000, 2), "music", "constant"),
    ("fmt_96k_stereo_float", "format", lambda: groove(124), ("WAV", "FLOAT", 96000, 2), "music", "constant"),
    ("fmt_44k_mono_8bit", "format", lambda: groove(124), ("WAV", "PCM_U8", 44100, 1), "music", "constant"),
    ("fmt_flac_16", "format", lambda: groove(124), ("FLAC", "PCM_16", 44100, 2), "music", "constant"),
    ("fmt_mp3_48k", "format", lambda: groove(124), ("MP3", "MPEG_LAYER_III", 48000, 2), "music", "constant"),
    ("long_10min", "speed", lambda: groove(124, bars=310), ("WAV", "PCM_16", 44100, 2), "music", "constant"),
    ("silence_30s", "not-music", lambda: (np.zeros(SR * 30, np.float32), None, None, None), ("WAV", "PCM_16", 44100, 2), "no_music", None),
    ("white_noise_30s", "not-music", lambda: (np.random.default_rng(0).normal(0, 0.2, SR * 30).astype(np.float32), None, None, None),
     ("WAV", "PCM_16", 44100, 2), "no_music", None),
    ("sine_440_30s", "not-music", lambda: ((0.4 * np.sin(2 * np.pi * 440 * np.arange(SR * 30) / SR)).astype(np.float32), None, None, None),
     ("WAV", "PCM_16", 44100, 2), "no_music", None),
    ("clip_2s", "short", lambda: short(2.0), ("WAV", "PCM_16", 44100, 2), "short", None),
    ("clip_0.5s", "short", lambda: short(0.5), ("WAV", "PCM_16", 44100, 2), "short", None),
]


def write(name, audio, fmt):
    container, subtype, sr, ch = fmt
    a = audio if audio.ndim == 2 else audio[:, None]
    if sr != SR:
        a = soxr.resample(a, SR, sr).astype(np.float32)
    a = np.repeat(a[:, :1], ch, axis=1) if ch != a.shape[1] else a
    if ch == 2:
        a = a * np.array([1.0, 0.92], np.float32)
    ext = {"WAV": "wav", "FLAC": "flac", "MP3": "mp3"}[container]
    INP.mkdir(parents=True, exist_ok=True)
    p = INP / f"{name}.{ext}"
    sf.write(p, np.clip(a, -1, 1), sr, format=container, subtype=subtype)
    return p


def drift_ms(grid_b: np.ndarray, ref: np.ndarray, factor: float) -> float:
    """Max distance at the COARSER of the two counting levels (so a doubled grid is
    checked at every reference beat and a halved grid at every grid beat)."""
    a, b = (ref, grid_b) if factor >= 1 else (grid_b, ref)
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    d = np.minimum(np.abs(b[idx - 1] - a), np.abs(b[idx] - a))
    return float(d.max() * 1000)


def run_case(name, cat, build, fmt, kind, expect_grid) -> dict:
    audio, ref_b, ref_d, true_bpm = build()
    p = write(name, audio, fmt)
    row = {"case": name, "category": cat, "kind": kind, "format": f"{fmt[0]} {fmt[1]} {fmt[2]} Hz {fmt[3]}ch",
           "duration_s": round(len(audio) / SR, 1)}
    t0 = time.time()
    try:
        a, f = load(p)
        g = analyze(a, f.sr)
        row["analysis_s"] = round(time.time() - t0, 2)
        row["error"] = None
    except Exception as e:  # noqa: BLE001
        row.update({"analysis_s": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}"})
        row["pass"] = kind in ("no_music", "short") and isinstance(e, ValueError)
        row["verdict"] = "refused with message" if row["pass"] else "CRASH/unexpected error"
        return row
    conf = g.confidence.get("overall", 0)
    octave = (g.meta.get("fit", {}).get("tempo_octave") or {}).get("factor", 1.0)
    row.update({"tempo_bpm": round(g.tempo_bpm, 2), "grid_type": g.meta["grid_kind"], "time_signature": g.time_signature,
                "confidence": conf, "octave_factor": octave, "n_beats": g.n_beats,
                "speed_s_per_min": round(row["analysis_s"] / max(0.01, row["duration_s"] / 60), 2)})
    if kind == "no_music":
        row["pass"] = conf < 0.5
        row["verdict"] = "flagged low confidence" if row["pass"] else f"CONFIDENT GRID ON NON-MUSIC (conf {conf})"
        return row
    if kind == "short":
        row["pass"] = True
        row["verdict"] = "no crash"
        return row
    checks = {}
    if true_bpm:
        r = g.tempo_bpm / true_bpm
        checks["tempo"] = min(abs(r - k) / k for k in (0.5, 1.0, 2.0)) < 0.005
    if ref_b is not None and len(ref_b) > 4:
        row["beat_F"] = round(mir_eval.beat.f_measure(ref_b, g.beats, 0.07), 4)
        row["beat_AMLt"] = round(mir_eval.beat.continuity(ref_b, g.beats)[3], 4)
        checks["beats_AMLt>=0.9"] = row["beat_AMLt"] >= 0.9
        if expect_grid == "constant":
            row["drift_max_ms"] = round(drift_ms(g.beats, ref_b, octave), 2)
            checks["drift<=10ms"] = row["drift_max_ms"] <= 10.0
        if ref_d is not None and len(ref_d) > 2 and g.n_bars > 1:
            if octave == 1.0:
                row["downbeat_F"] = round(mir_eval.beat.f_measure(ref_d, g.downbeats, 0.07), 4)
                checks["downbeat_F>=0.8"] = row["downbeat_F"] >= 0.8
            else:
                # bars change length with the octave, so test the containment that must hold:
                # halved -> every grid "1" is a reference "1"; doubled -> every reference "1" is a grid "1"

                hit = lambda xs, ys: float(np.mean([np.min(np.abs(ys - x)) < 0.07 for x in xs]))  # noqa: E731
                score = hit(g.downbeats, ref_d) if octave < 1 else hit(ref_d, g.downbeats)
                row["downbeat_containment"] = round(score, 4)
                checks["downbeats_on_the_1"] = score >= 0.8
    expect_ts = {"waltz_3-4_120": "3/4", "six8_68": "6/8"}.get(name, "4/4")
    checks["meter"] = g.time_signature == expect_ts
    if not g.time_signature.endswith("/8"):
        checks["bpm_in_80-160"] = 80 <= g.tempo_bpm < 160 or g.meta["grid_kind"] != "constant"
    if expect_grid:
        checks["grid_type"] = g.meta["grid_kind"] == expect_grid
    if cat == "format":
        s = Session(p, ROOT / "work" / "robust_edit" / name, grid=g)
        s.apply([Remove(range=Range(unit="bars", start=2, end=2))])
        out = s.export(stem="edit", with_click=False)
        info = sf.info(out["audio"])
        checks["edit_keeps_format"] = info.samplerate == f.sr and info.subtype == f.out_subtype
        row["exported"] = f"{info.samplerate} Hz {info.subtype}"
    row["checks"] = checks
    row["pass"] = all(checks.values())
    row["verdict"] = "pass" if row["pass"] else "FAIL: " + ", ".join(k for k, v in checks.items() if not v)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    rows = []
    for c in CASES:
        if only and c[0] not in only:
            continue
        r = run_case(*c)
        rows.append(r)
        print(f"  {r['case']:22s} {r.get('verdict')}  | bpm {r.get('tempo_bpm')} x{r.get('octave_factor')} "
              f"{r.get('grid_type')} {r.get('time_signature')} F {r.get('beat_F')} AMLt {r.get('beat_AMLt')} "
              f"dF {r.get('downbeat_F')} drift {r.get('drift_max_ms')} conf {r.get('confidence')} {r.get('analysis_s')}s"
              + (f" err={r['error'][:80]}" if r.get('error') else ""))
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    n = len(rows)
    print(f"\npassed {sum(r['pass'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
