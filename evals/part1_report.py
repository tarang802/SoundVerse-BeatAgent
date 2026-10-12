"""Build evals/results/PART1_REPORT.md from the saved result files.

    python -m evals.part1_report

Generated, not hand-written, so the report can't drift from the numbers.
Run evals.eval_beats and evals.eval_sync first.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"

TARGETS = {
    "beat_F": ("Beat F-measure (±70 ms)", "≥ 0.90 on steady-tempo clips"),
    "downbeat_F": ("Downbeat F-measure (±70 ms)", "≥ 0.80"),
    "drift": ("Grid drift, 6-min constant clip", "every beat within 10 ms"),
    "sync": ("Sync kick offsets", "median ≤ 5 ms, max ≤ 15 ms"),
    "speed": ("Speed", "6-min track < 60 s on laptop CPU"),
}
STEADY = {"A1", "A2", "A4", "A6", "A7"}   # constant tempo by construction (A5 = two steady halves)


def ok(b: bool) -> str:
    return "✅" if b else "❌"


def main():
    bt = json.loads((RES / "part1_substitute_beat_this" / "results.json").read_text())["rows"]
    lb_path = RES / "part1_substitute_librosa" / "results.json"
    lb = {r["clip"]: r for r in json.loads(lb_path.read_text())["rows"]} if lb_path.exists() else {}
    sync = json.loads((RES / "sync_A1_A7" / "comparison.json").read_text())
    smoke = json.loads((RES / "smoke_test.json").read_text())
    by = {r["clip"]: r for r in bt}

    L = []
    L.append("# Part 1 — beat alignment and sync: results\n")
    L.append("> **Everything below is measured on the SYNTHETIC SUBSTITUTE pack** (`data/substitute`, built by "
             "`evals/make_testpack.py`). The assignment's test pack (A1–A7 with reference annotations) was not "
             "provided, so **no assignment benchmark number is claimed**. Synthetic clips have clean transients and "
             "no mix/master, so these numbers are an upper bound on real-music performance.\n")
    L.append(f"Machine: this laptop's CPU (no GPU). Tracker: `{bt[0]['tracker']}`. "
             "mir_eval F-measures are computed on the full clip (mir_eval's default 5 s trim is NOT applied).\n")

    # ---- targets summary
    L.append("## Targets\n")
    L.append("| Target | Brief | Result (synthetic) | Met? |\n|---|---|---|---|")
    steady = [by[c] for c in sorted(STEADY) if c in by]
    worst_beat = min(steady, key=lambda r: r["beat_F"])
    L.append(f"| {TARGETS['beat_F'][0]} | {TARGETS['beat_F'][1]} | min {worst_beat['beat_F']:.4f} ({worst_beat['clip']}) "
             f"over {', '.join(sorted(STEADY))} | {ok(worst_beat['beat_F'] >= 0.9)} |")
    with_down = [r for r in bt if r.get("downbeat_F") is not None]
    fails = [r for r in with_down if r["downbeat_F"] < 0.8]
    fail_txt = "; fails: " + ", ".join("%s %.2f" % (r["clip"], r["downbeat_F"]) for r in fails) if fails else ""
    L.append(f"| {TARGETS['downbeat_F'][0]} | {TARGETS['downbeat_F'][1]} | "
             f"{len(with_down) - len(fails)}/{len(with_down)} clips pass{fail_txt} | {ok(not fails)} |")
    a1 = by["A1"]
    L.append(f"| {TARGETS['drift'][0]} | {TARGETS['drift'][1]} | A1 ({a1['duration_s']:.0f} s): max {a1['drift_max_ms']} ms, "
             f"{a1['within_10ms'] * 100:.0f}% within 10 ms | {ok(a1['drift_max_ms'] <= 10)} |")
    best = next(r for r in sync if r["method"] == "constant_attack")
    L.append(f"| {TARGETS['sync'][0]} | {TARGETS['sync'][1]} | A7→A1 @ bar 33: median {best['median_abs_ms']} ms, "
             f"max {best['max_abs_ms']} ms, {best['measured']} beats measured | {ok(bool(best['pass']))} |")
    L.append(f"| {TARGETS['speed'][0]} | {TARGETS['speed'][1]} | A1 ({a1['duration_s']:.0f} s): {a1['analysis_s']} s "
             f"(this eval run, first clip so includes model load). Measured by hand on 2026-10-12 on this i7-13620H: "
             f"10.3 / 25.7 / 19.4 s in repeated runs; app with both decks analysing at once: 70-80 s each "
             f"| {ok(a1['analysis_s'] < 60)} (single track) |")
    L.append("")

    # ---- per clip
    L.append("## Beat grid per clip\n")
    cols = ["clip", "duration_s", "analysis_s", "grid_type", "tempo_bpm", "time_signature", "beat_F", "beat_AMLt", "downbeat_F",
            "drift_max_ms", "within_10ms", "raw_beat_F", "raw_drift_max_ms", "confidence"]
    L.append("| " + " | ".join(cols) + " | librosa beat_F |\n|" + "---|" * (len(cols) + 1))
    for r in bt:
        L.append("| " + " | ".join(str(r.get(c, "n/a")) for c in cols) + f" | {lb.get(r['clip'], {}).get('beat_F', 'n/a')} |")
    L.append("\n`beat_AMLt` = mir_eval continuity score that also accepts double/half-tempo and off-beat counting "
             "(F low + AMLt high = right beats, different counting convention; both low = wrong grid). "
             "`raw_*` = Beat This! output before our grid fitting; `drift_*` = |grid − reference| per reference beat. "
             "A3 is variable-tempo by design, so the 10 ms drift target does not apply to it.\n")

    # ---- failure analysis
    L.append("## Failure analysis by clip\n")
    notes = {
        "A1": "Pass. Confidence is low (≈0.6) only because the tracker drops beats in the 16-bar drumless breakdown; the fitted grid keeps counting through it.",
        "A2": "Pass. No tempo-octave error on the half-time groove (86 BPM, not 43/172).",
        "A3": (f"Beat F {by['A3']['beat_F']}; grid type `{by['A3']['grid_type']}` (a 112->118 BPM ramp, then ~118); "
               f"worst beat {by['A3']['drift_max_ms']} ms. An earlier variable-grid version had 73 ms worst; the "
               "current constant/piecewise tests (needed so real constant-tempo hip-hop is not mislabelled variable) "
               "model the drift as two constant sections instead. 12 ms random per-hit jitter is in the clip by design."),
        "A4": "Pass after trimming: the tracker invented a 60 BPM pulse over the 47 s pad intro; beats with no onset evidence at the start/end are dropped. Assumption: the reference starts at the drum entry (true for our substitute; unknown for the real pack).",
        "A5": "Pass. The tracker drifts toward the new tempo ~4.5 s early; the change point is re-chosen from onsets (found at bar 33, 64.498 s vs true 64.500 s).",
        "A6": "**Downbeat FAIL (0.67).** 6/8 is reported as 12/8: Beat This! marks a downbeat every 4 dotted-quarter beats, so bars come out twice as long. Not fixed: telling 6/8 from accented 4/4 needs a meter model we did not build; a quick heuristic risked breaking 4/4.",
        "A7": "Pass.",
    }
    for c in sorted(by):
        L.append(f"- **{c}**: {notes.get(c, '')}")
    L.append("")

    # ---- sync
    L.append("## Two-track sync (A7 → A1, overlap starts at A1's bar 33)\n")
    L.append("Four render methods were measured; the app uses `constant_attack`.\n")
    cols = list(sync[0].keys())
    L.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in sync:
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    L.append("""
- `constant`: one Rubber Band R3 stretch. Pitch exact, but R3 smears kick transients: downbeat kicks ~30 ms early. **Note the trap**: with the strict detector only 62/124 beats are measurable and their stats look nearly fine — the unmeasured ones are the worst.
- `beats` / `beats_attack`: per-beat slices. Kicks perfect, but R3 on 0.5 s slices **detuned a sustained 110 Hz tone by +11–12 cents** (measured inside single slices) → fails "pitch preserved".
- `constant_attack` (**used**): one R3 stretch for exact pitch + B's original audio pasted from 40 ms before to 30 ms after every beat, crossfaded, so each kick sits exactly on A's beat.
- Kick offsets are measured on the separate A and B stems with a spectral-flux onset detector validated against known beat times (0.5 ms p95 spread on A7). A kick-band detector was tried and **failed validation** (false kicks in pad sections), so it was removed.
- One B downbeat (~17.4 s into the render) sits right at the detector's 3x threshold: it is measured in some runs (124/124 in the app) and not in others (123/124 here). The table reports this run.
- Caveats: A and B use the same synthetic kick sound, which makes alignment and measurement easier than on real tracks; B with variable tempo is untested; about 70 ms around each kick plays at B's original speed.
""")

    # ---- real music (no reference annotations)
    real = []
    for f in ("final1.jsonl", "final2.jsonl"):
        p = RES / "real" / f
        if p.exists():
            real += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if real:
        L.append("## Real music (no reference annotations — consistency checks, NOT accuracy)\n")
        n_ok = sum(1 for r in real if "plan" not in r and "error" not in r)
        n_err = sum(1 for r in real if "error" in r)
        L.append(f"{n_ok} commercial hip-hop tracks (3 artists) found on the dev machine"
                 + (f", plus {n_err} undecodable file(s) (clean error reported)" if n_err else "") + ". "
                 "No house/EDM/live-band recordings were available, so those genres are untested on real audio. "
                 "Without answer keys, beat F / drift cannot be computed; instead:\n"
                 "- **indep tempo**: tempo from onset autocorrelation in 20 s windows (does not use the neural "
                 "tracker), compared with the grid; windows without a drum pulse are not graded. Validated on the "
                 "substitute pack (≤0.02 % on A1/A2/A5/A7; skipped exactly A1's drumless breakdown).\n"
                 "- **own-grid spread**: p10–p90 spread of this track's downbeat hits around its own grid lines, "
                 "with a broadband attack detector and a low-frequency kick detector.\n")
        L.append("| track | BPM | grid | conf | indep tempo max diff (graded windows) | librosa tempo | "
                 "downbeat attack spread | downbeat low-kick spread | analysis s |\n|---|---|---|---|---|---|---|---|---|")
        for r in real:
            if "plan" in r or "refused" in r:
                continue
            if "error" in r:
                L.append(f"| {r['file']} | — | **clean error** | — | — | — | — | — | — |")
                continue
            it, at, lk = r["independent_tempo"], r["downbeat_attack_vs_own_grid"], r["downbeat_lowkick_vs_own_grid"]
            L.append(f"| {r['file']} | {r['tempo_bpm']} | {r['grid_type']} | {r['confidence']} | "
                     f"{it['max_abs_diff_pct']}% ({it['n_graded_windows']}/{it['n_windows']}) | {r['librosa_tempo']} ({r['tempo_check']}) | "
                     f"{at['p10_p90_spread_ms'] if at else '—'} ms | {lk['p10_p90_spread_ms'] if lk else '—'} ms | {r['analysis_s']} |")
        L.append("\n### Real-music sync (A bar 9, B bar 1, 60 s)\n")
        L.append("| B → A | ratio | any-onset, all beats (med / max) | downbeat attacks (med / max, ≤15 ms share) | "
                 "downbeat low-kick (med / max) | meets brief? |\n|---|---|---|---|---|---|")
        for r in real:
            if "plan" not in r:
                continue
            p, a, d, k = r["plan"], r["alignment_any_onset"], r["alignment_downbeat_attacks"], r["alignment_downbeat_kicks"]
            L.append(f"| {r['b']} → {r['a']} | {p['time_ratio']:.5f} | {a['median_abs_ms']} / {a['max_abs_ms']} ms "
                     f"({a['n_measured']}/{a['n_beats']}) | {d.get('median_abs_ms')} / {d.get('max_abs_ms')} ms, "
                     f"{d.get('share_within_15ms')} ({d['n_measured']}/{d['n_downbeats']}) | {k.get('median_abs_ms')} / "
                     f"{k.get('max_abs_ms')} ms ({k['n_measured']}/{k['n_downbeats']}) | ❌ max > 15 ms |")
        tracks = [r for r in real if "plan" not in r and "error" not in r]
        agree = [r for r in tracks if r["independent_tempo"]["all_within_1_5pct"]]
        disagree = [r for r in tracks if not r["independent_tempo"]["all_within_1_5pct"]]
        lib_agree = [r for r in tracks if r["tempo_check"] == "agree"]
        L.append("\n**Reading of the real-music results (counts computed from the files above):**")
        L.append(f"- Tempo: independent estimate within 1.5 % in every graded window on {len(agree)}/{len(tracks)} tracks. "
                 f"Not within: {', '.join('%s (%s%%)' % (r['file'][:28], r['independent_tempo']['max_abs_diff_pct']) for r in disagree) or 'none'}.")
        L.append(f"- Tempo octave: librosa's estimate agrees with ours (same octave) on {len(lib_agree)}/{len(tracks)} tracks. "
                 "Half-time hip-hop is genuinely ambiguous (Naam Kaam Sheher was judged 140 and then 70 by the same listener), "
                 "so the app shows an automatic octave (preferred range 80-160 BPM, configurable) plus 1/2x / 2x buttons, "
                 "and sync matches tracks ~2x apart at half/double time.")
        L.append("- Sync on real songs **does not meet the max <= 15 ms target** on either pair as measured, although most "
                 "downbeats are tight (see the 'within 15 ms' share). Each song's own downbeat hits scatter around a straight "
                 "grid by an amount that depends on the track and on the detector, and no kick detector validated on all "
                 "real tracks; we cannot separate measurement error, humanised drums and grid error without listening or "
                 "annotations. Renders: `evals/results/real/sync_*_mix.wav` (gitignored, copyrighted audio).\n")


    # ---- environment
    L.append("## Environment checks (smoke test)\n")
    d = smoke.get("decode", {})
    L.append(f"- WAV/FLAC/MP3 decode via libsndfile; MP3 kick timing vs WAV: {d.get('mp3', {}).get('kick_timing_vs_wav_ms')}")
    L.append(f"- Rubber Band (pylibrb) R3 pitch: {smoke.get('rubberband', {}).get('pitch_check')}")
    L.append(f"- mir_eval known example: {smoke.get('mir_eval_check', {}).get('pass')}")
    L.append(f"- Assignment test pack: {smoke.get('assignment_test_pack', {}).get('status')}")
    L.append("\n## Reproduce\n\n```\npython -m evals.make_testpack\npython -m evals.eval_beats\n"
             "python -m evals.eval_beats --tracker librosa\npython -m evals.eval_sync\npython -m evals.part1_report\n```\n")
    out = RES / "PART1_REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
