"""Part 3 evaluation: clip -> score -> page -> offline render, measured.

    python -m evals.eval_part3 [P1 P2 ...] [--no-render]

For each clip in data/p3 (SYNTHETIC substitutes for the brief's P1-P4, see
evals/make_p3pack.py):
  1. transcribe -> score.json (beatagent/transcribe.py)
  2. note-level F-measure vs the reference MIDI with mir_eval.transcription
     (onset-only: offset_ratio=None, onset tolerance +-50 ms, pitch within 50 cents),
     separately for DRUMS (General MIDI note numbers used as the "pitch", so a snare
     transcribed as a kick is a miss) and PITCHED parts (all non-drum tracks pooled),
     plus a "rhythm-only" drum score that ignores which drum it was
  3. page.html (beatagent/webaudio.py) rendered offline in headless Edge/Chrome to
     render.wav, then onset timing of the render vs the original clip:
     onsets in both, matched within +-50 ms -> F, median and max |offset|
Results: evals/results/part3/<clip>/ and evals/results/part3/results.{json,md}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.audio_io import load  # noqa: E402
from beatagent.score import Score  # noqa: E402
from beatagent.transcribe import transcribe  # noqa: E402

PACK = ROOT / "data" / "p3"
OUT = ROOT / "evals" / "results" / "part3"


def hz(p):
    return 440.0 * 2 ** ((np.asarray(p, dtype=float) - 69) / 12)


def note_f(ref: list[tuple[float, float, int]], est: list[tuple[float, float, int]], ignore_pitch=False) -> dict:
    """mir_eval onset-only note F (+-50 ms). ref/est: (start_s, end_s, midi)."""
    if not ref and not est:
        return {"n_ref": 0, "n_est": 0, "precision": None, "recall": None, "f": None}
    if not ref or not est:
        return {"n_ref": len(ref), "n_est": len(est), "precision": 0.0 if est else None,
                "recall": 0.0 if ref else None, "f": 0.0}
    ri = np.array([[s, max(e, s + 0.01)] for s, e, _ in ref])
    ei = np.array([[s, max(e, s + 0.01)] for s, e, _ in est])
    rp = hz([60 if ignore_pitch else p for *_, p in ref])
    ep = hz([60 if ignore_pitch else p for *_, p in est])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return {"n_ref": len(ref), "n_est": len(est), "precision": round(float(p), 4),
            "recall": round(float(r), 4), "f": round(float(f), 4)}


def reference(name: str, pack: Path = PACK):
    """(drum notes, pitched notes) from <pack>/<name>.mid or .midi; None if there is none
    (the brief's P4 has no reference)."""
    mid = next((pack / f"{name}{ext}" for ext in (".mid", ".midi") if (pack / f"{name}{ext}").exists()), None)
    if mid is None:
        return None, None
    pm = pretty_midi.PrettyMIDI(str(mid))
    drums = [(n.start, n.end, n.pitch) for i in pm.instruments if i.is_drum for n in i.notes]
    pitched = [(n.start, n.end, n.pitch) for i in pm.instruments if not i.is_drum for n in i.notes]
    return drums, pitched


def midi_tempo(name: str, pack: Path) -> float | None:
    mid = next((pack / f"{name}{ext}" for ext in (".mid", ".midi") if (pack / f"{name}{ext}").exists()), None)
    if mid is None:
        return None
    _, tempi = pretty_midi.PrettyMIDI(str(mid)).get_tempo_changes()
    return round(float(tempi[0]), 3) if len(tempi) else None


def estimated(score: Score):
    """Score notes back on the clip's own timeline (loop start offset, wrapped)."""
    off, L = score.source.get("loop_start_s", 0.0), score.loop_seconds
    drums, pitched = [], []
    for t in score.tracks:
        for s, e, p, _v, _n in score.note_events_seconds(t.name):
            s2 = (s + off) % L
            (drums if t.kind == "drums" else pitched).append((s2, s2 + (e - s), p))
    return sorted(drums), sorted(pitched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--pack", default=str(PACK),
                    help="folder of clips (WAV/FLAC/MP3) with optional <clip>.mid references; "
                         "e.g. the official P1-P4. pack.json is optional")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    pack = Path(args.pack).resolve()
    own = pack == PACK.resolve()
    meta = json.loads((pack / "pack.json").read_text(encoding="utf-8")) if (pack / "pack.json").exists() else {}
    audio_files = {p.stem: p for p in sorted(pack.iterdir()) if p.suffix.lower() in (".wav", ".flac", ".mp3")}
    out = OUT if own else ROOT / "evals" / "results" / f"part3_{pack.name}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.clips or sorted(audio_files):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        audio, fmt = load(audio_files[name])
        score = transcribe(audio, fmt.sr, name)
        score.save(d / "score.json")
        rd, rp = reference(name, pack)
        ed, ep = estimated(score)
        m = meta.get(name, {})
        has_ref = rd is not None
        row = {"clip": name, "synthetic": bool(m.get("synthetic", False)), "description": m.get("description", ""),
               "reference": m.get("reference", "MIDI" if has_ref else "none"),
               "tempo_bpm": {"ref": m.get("bpm", midi_tempo(name, pack)), "est": score.tempo_bpm},
               "bars": {"ref": m.get("bars"), "est": score.bars},
               "drums_F": note_f(rd, ed) if has_ref else None,
               "drums_rhythm_only_F": note_f(rd, ed, ignore_pitch=True) if has_ref else None,
               "pitched_F": note_f(rp, ep) if has_ref else None,
               "tracks": {t.name: len(t.notes) for t in score.tracks},
               "transcribe_s": score.analysis.get("seconds")}
        if not args.no_render:
            from evals.part3_render import measure_render
            row["render"] = measure_render(score, d, audio, fmt.sr)
        rows.append(row)
        print(json.dumps(row))
    (out / ("results.json" if not args.clips else f"results_{'_'.join(args.clips)}.json")).write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    if not args.clips:
        write_md(rows, out, synthetic=own)


def _f(x):
    return "—" if x is None else f"{x:.3f}"


def write_md(rows, out: Path = OUT, synthetic: bool = True):
    title = ("# Part 3 results (SYNTHETIC substitute clips; the brief's P1-P4 were not provided)\n" if synthetic
             else f"# Part 3 results ({out.name.replace('part3_', '')} pack)\n")
    none = {"f": None, "precision": None, "recall": None}
    L = [title,
         "Note F: mir_eval.transcription, onset-only (offset_ratio=None), +-50 ms, pitch +-50 cents. "
         "Drums use General MIDI numbers as pitch (a snare heard as a kick is a miss); "
         "'rhythm-only' ignores which drum. Render onsets: onsets of the offline render vs the "
         "original clip, matched within +-50 ms.\n",
         "| clip | tempo ref/est | drums F (P/R) | drums rhythm-only F | pitched F (P/R) | render onset F | render median / max ms |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        d, dr, p = r["drums_F"] or none, r["drums_rhythm_only_F"] or none, r["pitched_F"] or none
        rn = r.get("render") or {}
        L.append(f"| {r['clip']} | {r['tempo_bpm']['ref']} / {r['tempo_bpm']['est']} | "
                 f"{_f(d['f'])} ({_f(d['precision'])}/{_f(d['recall'])}) | {_f(dr['f'])} | "
                 f"{_f(p['f'])} ({_f(p['precision'])}/{_f(p['recall'])}) | {_f(rn.get('onset_F'))} | "
                 f"{rn.get('median_abs_ms', '—')} / {rn.get('max_abs_ms', '—')} |")
    (out / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
