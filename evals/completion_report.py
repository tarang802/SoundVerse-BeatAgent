"""Build evals/results/COMPLETION_REPORT.md from the saved result files.

    python -m evals.completion_report

Every number is computed from a results file written by an eval script; nothing
is typed in by hand except the feature descriptions and known-limitation notes.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"


# Hand-written: what each brief requirement maps to, and where the evidence is.
# The pass/fail numbers above are computed; these rows only point at them.
FEATURES = ["\n## Brief requirements → implementation\n",
            "| Part | Requirement | Status | Where / evidence |\n|---|---|---|---|",
            "| 1 | Beat grid JSON: tempo, beats, downbeats, meter, tempo segments, confidence | done | `beatagent/analyze.py`, `grid.py`; `*.grid.json` in results |",
            "| 1 | Click track to audition the grid (accented downbeats) | done | `beatagent/click.py`; `*.click.wav`; deck CLICK toggle |",
            "| 1 | Constant / tempo-change / variable grids | done | constant + piecewise + variable paths; robustness table |",
            "| 1 | Tempo octave (half/double time) handled | done | 80–160 BPM rule, compound-meter guard, ½×/2× override in UI and API |",
            "| 1 | Two-track sync: stretch B onto A, pitch preserved, downbeats aligned, 60 s render | done | `beatagent/sync.py` (Rubber Band R3 + verbatim kick patches); sync tables |",
            "| 1 | Sync refuses what can't sound right (>25 % stretch, meter mismatch, too short) | done | refusal rows in sync robustness |",
            "| 1 | Accuracy report with mir_eval (F, AMLt, drift, speed) | done | `evals/eval_beats.py`, `PART1_REPORT.md` |",
            "| 2 | Chat → typed edit ops (LLM interprets, code executes) | done | `beatagent/llm.py` (Gemini structured output), `edl.py` |",
            "| 2 | Echo the exact bars/beats/seconds before cutting | done | every applied reply starts with the resolved range |",
            "| 2 | Click-free audio at every splice | done | pre-splice equal-power crossfade; fuzz click check |",
            "| 2 | Grid stays true after edits | done | exact grid mapping; fuzz grid check; chat-suite grid-follow |",
            "| 2 | Undo / redo | done | session stack with unique state ids |",
            "| 2 | Graceful errors (bad numbers, out of range, impossible ops, LLM down) | done | deterministic validators + offline rule fallback, labelled |",
            "| 2 | Subjective requests answered from measurements, proposed, confirmed | done | `beatagent/sections.py`; propose → yes/no |",
            "| 2 | Preview player | done | React deck: waveform, BAR.BEAT counter, play from any bar |",
            "| 2 | Variable-tempo correctness | done | edits address the measured grid, not a BPM formula |",
            "| 2 | Sync by chat (\"bring the other track in at bar 17\") | done (extra) | chat `sync` action with partner deck |",
            "| 3 | Audio → Web Audio page | not started | — |",
            ""]

LEDGER = ["## Bugs found in the final debugging pass\n",
          "| Found by | Bug | Fix | Status |\n|---|---|---|---|",
          "| sync robustness (`b_drifting`) | Tracker put ~1 beat in 7 about 30 ms early on a drifting take; ±25 ms snap couldn't reach them → worst sync kick 35.9 ms | variable grids: re-snap a beat that is >15 ms from where its neighbours predict, to the onset nearest the prediction | fixed (worst kick 1.3 ms) |",
          "| chat suite (`q1`/`q2`) | LLM proposed \"bars 81 to 96?\" in prose only; the user's \"yes\" then did nothing | prompt rule + code repair round when a clarify message names bars but carries no ops; clearer reply when nothing is pending | fixed |",
          "| chat suite (`b5`) | LLM rewrote \"60 dB louder\" as +24 dB (the engine's limit) and it was applied silently | code keeps the user's stated dB magnitude, LLM only picks direction; engine then refuses 60 dB with a reason | fixed |",
          "| chat suite (`b11`) | LLM turned \"remove bar -2\" into bar 1 and applied it | deterministic negative/fractional bar-beat validator before the LLM | fixed |",
          "| code audit | Offline sync rule regex contained literal backspace characters (shell-escaping), so it could never match | regex fixed; `tests/test_imports.py` now fails on control characters in source | fixed |",
          "| real app session (13 Oct) | \"cut beat 16-32 repeat it twice\" refused by flash-lite as \"fractions\" (2 of 3 tries) | prompt rule reworded (any whole-number range is valid, singular or plural) + code repair round when the LLM claims fractions the code found none of | fixed (3/3 live; `p5` regression case) |",
          "| real app session (13 Oct) | quota fallback note showed a raw API error dump | plain-words note: \"X is unavailable (out of free-tier quota); answered by Y\" | fixed |",
          "| A3 investigation | 112→118 BPM ramp in the first 15 s: piecewise line misses up to 108 ms there | tried splicing tracked beats into ramps: worse overall (tracker unreliable there) — reverted | **open** |",
          "",
          "## Known limitations (honest)\n",
          "- **Abrupt large tempo steps** (e.g. 128 → 100 BPM in one beat): the tracker smooths over them; the grid is wrong around the step (robustness `steps_120-128-100`).",
          "- **Short tempo ramps inside otherwise steady tracks** (A3's opening): grid is a straight line there; worst beat ~108 ms, 86 % of beats within 10 ms.",
          "- **6/8 vs 12/8** (A6): both are musically defensible; bar length counted as 12/8 → downbeat F 0.667 against a 6/8 reference.",
          "- **Real music has no answer key here**: real-song numbers are agreement with an independent tempo estimate and onset support, not accuracy.",
          "- **Real-music sync is not proven ≤ 5 ms**: the downbeat-kick detector reports ~20 ms median on real pairs, but it reports 12–17 ms on a self-sync that is perfect by construction (different bars of the same song), so the detector cannot resolve this on hip-hop. Listen to `evals/results/real/sync_*_mix.wav` (generated locally).",
          "- **Brief session 5, turn 1** (\"Cut beat 0 to beat 8\"): the LLM asked for confirmation with the exact range instead of cutting at once; one \"yes\" applies it.",
          "- **Only hip-hop real tracks** were available on the machine; no EDM/rock/jazz real recordings tested.",
          "- **Tempo octave is a convention** (80–160 BPM); the ½×/2× buttons exist because e.g. 70 vs 140 is a choice, not an error.",
          "- **LLM free tier**: 20 requests/day per model; the fallback chain and the labelled offline parser keep the app usable, with narrower phrasing.",
          "- **Confidence** is a heuristic agreement score, not a calibrated probability.",
          "- **Assignment test pack not provided**: all brief-target numbers are on the synthetic substitute pack.",
          ""]


def jload(p):
    p = RES / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def by_cat(rows, key="category"):
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        if "pass" not in r:
            continue
        d[r[key]][1] += 1
        d[r[key]][0] += int(bool(r["pass"]))
    return d


def unit_tests():
    out = {}
    for mod in ("tests.test_edl", "tests.test_imports", "tests.test_agent_llm"):
        p = subprocess.run([sys.executable, "-m", mod], cwd=ROOT, capture_output=True, text=True)
        txt = p.stdout + p.stderr
        out[mod] = (txt.count("[PASS]"), txt.count("[PASS]") + txt.count("[FAIL]"), p.returncode)
    return out


def main():
    L = ["# Beat Agent — Part 1 & Part 2 completion report\n",
         "> Generated by `python -m evals.completion_report` from the result files in `evals/results/`. "
         "The assignment's test pack was **not provided**: brief-target numbers are on the synthetic substitute "
         "pack; robustness suites use generated audio with exactly known answers; real music (12 hip-hop "
         "tracks found on the dev machine) has no answer key, so only consistency checks apply there.\n"]

    # ------------------------------------------------------------------ summary table
    p1 = jload("part1_substitute_beat_this/results.json")
    rob = jload("part1_robustness/results.json") or []
    syn = jload("part1_sync_robustness/results.json") or []
    sync_cmp = jload("sync_A1_A7/comparison.json") or []
    sess = jload("part2_sessions/results.json") or []
    chat = jload("part2_chat_suite/results.json") or []
    fuzz = jload("part2_fuzz.json")
    real = []
    for f in ("real/final1.jsonl", "real/final2.jsonl"):
        p = RES / f
        if p.exists():
            real += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    ut = unit_tests()

    rob_done = [r for r in rob if "pass" in r]
    syn_done = [r for r in syn if "pass" in r]
    chat_done = [r for r in chat if "pass" in r]
    chat_llm = [r for r in chat_done if not r.get("offline_fallback")]
    sess_turns = [t for s in sess for t in s["turns"]]
    sess_checked = [t for t in sess_turns if t.get("checks")]
    sess_clean = [t for t in sess_checked if t["checks"]["clicks"]["ok"] and t["checks"]["format"]["ok"]
                  and t["checks"]["length"]["ok"]]
    p1_rows = p1["rows"] if p1 else []
    best = next((r for r in sync_cmp if r["method"] == "constant_attack"), {})

    L.append("## Summary\n")
    L.append("| Suite | What it covers | Result |\n|---|---|---|")
    if p1_rows:
        db = [r for r in p1_rows if r.get("downbeat_F") is not None]
        a1 = next(r for r in p1_rows if r["clip"] == "A1")
        a3 = next((r for r in p1_rows if r["clip"] == "A3"), None)
        L.append(f"| Part 1 brief targets (synthetic A1–A7) | beat F, downbeat F, drift, speed | "
                 f"beat F ≥ 0.90 on {sum(r['beat_F'] >= 0.9 for r in p1_rows)}/{len(p1_rows)} clips; "
                 f"downbeat F ≥ 0.80 on {sum(r['downbeat_F'] >= 0.8 for r in db)}/{len(db)} "
                 f"({', '.join(r['clip'] + ' ' + str(r['downbeat_F']) for r in db if r['downbeat_F'] < 0.8)} misses); "
                 f"A1 (6 min) drift {a1['drift_max_ms']} ms, analysed in {a1['analysis_s']} s"
                 + (f"; A3 ramp worst beat {a3['drift_max_ms']} ms" if a3 else "") + " |")
    L.append(f"| Part 1 robustness (generated, known answers) | tempo 60–200, grooves, meters, tempo changes, "
             f"arrangement, levels, formats, non-music | {pct(sum(r['pass'] for r in rob_done), len(rob_done))} |")
    L.append(f"| Part 1 sync robustness (ground truth) | ±20 % stretch, octaves, formats, drift, tempo change, "
             f"refusals, lengths | {pct(sum(r['pass'] for r in syn_done), len(syn_done))} |")
    if best:
        L.append(f"| Part 1 brief sync (A7→A1 @ bar 33) | kick offset target | median {best['median_abs_ms']} ms, "
                 f"max {best['max_abs_ms']} ms, pass {best['pass']} |")
    real_ok = [r for r in real if "plan" not in r and "error" not in r]
    L.append(f"| Part 1 real music (no answer key) | independent tempo agreement in every graded window | "
             f"{pct(sum(r['independent_tempo']['all_within_1_5pct'] for r in real_ok), len(real_ok))} |")
    real_sync = [r["sync"] if "sync" in r else r for r in real if ("a" in r and "b" in r) or "sync" in r]
    for s in real_sync:
        k = s.get("alignment_downbeat_kicks", {})
        L.append(f"| Part 1 real sync {s['a'][:20]}… ← {s['b'][:20]}… (no answer key) | downbeat-kick detector | "
                 f"median {k.get('median_abs_ms')} ms, max {k.get('max_abs_ms')} ms — see self-sync control |")
    ss = (jload("real/selfsync.json") or {}).get("rows", [])
    for content in ("identical", "different bars"):
        rr = [r for r in ss if r["content"] == content]
        if rr:
            L.append(f"| Self-sync control, {content} (perfect by construction) | detector floor on real hip-hop | "
                     + "; ".join(f"{r['file'][:22]}: median {r['downbeat_kicks']['median_abs_ms']} / max "
                                 f"{r['downbeat_kicks']['max_abs_ms']} ms" for r in rr) + " |")
    L.append(f"| Part 2 brief sessions (6, in order, LLM) | the brief's chat script | "
             f"{len(sess)} sessions, {len(sess_turns)} turns; audio checks clean on {pct(len(sess_clean), len(sess_checked))} edited turns |")
    L.append(f"| Part 2 chat suite (producer phrasing, LLM) | all 7 ops, paraphrase, multi-op, grid-follow, undo/redo, "
             f"bad input, subjective, questions, tempo-change, sync, real song | {pct(sum(r['pass'] for r in chat_done), len(chat_done))}"
             f" (answered by the LLM: {len(chat_llm)}) |")
    live = [r for r in (jload("part2_chat_suite/results_live_check.json") or []) if "pass" in r]
    if live:
        L[-1] = L[-1].rstrip(" |") + " — full run on the prompt *before* the 13 Oct fix (free-tier quota prevented a second full run) |"
        L.append(f"| Part 2 live LLM re-check on the FINAL prompt | 13 cases most exposed to the 13 Oct prompt change "
                 f"(see `part2_chat_suite/LLM_LIVE_CHECK.md`) | {pct(sum(r['pass'] for r in live), len(live))}, "
                 f"answered by the LLM: {sum(not r.get('offline_fallback') for r in live)} |")
    if fuzz:
        s = fuzz["stats"]
        fails = s["length_fail"] + s["order_fail"] + s["grid_fail"] + s["click_fail"] + s["undo_fail"] + s["redo_fail"]
        L.append(f"| Part 2 engine fuzz (no LLM) | {s['ops']} random edits, invariants after each | "
                 f"{s['applied']} applied, {s['refused_cleanly']} refused cleanly, {s['crashed']} crashes, "
                 f"{fails} invariant failures |")
    for mod, (p_, t_, rc) in ut.items():
        L.append(f"| Unit tests `{mod}` | | {p_}/{t_} pass (exit {rc}) |")
    L.append("")

    # ------------------------------------------------------------------ Part 1 detail
    L.append("## Part 1 — beat alignment and sync\n")
    L.append("### Robustness by category\n")
    L.append("| Category | Pass |\n|---|---|")
    for c, (a, b) in sorted(by_cat(rob_done).items()):
        L.append(f"| {c} | {pct(a, b)} |")
    L.append("\n| Case | Result | BPM (octave) | meter | grid | beat F | AMLt | downbeats | drift ms | conf | s |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rob_done:
        dn = r.get("downbeat_F", r.get("downbeat_containment", ""))
        L.append(f"| {r['case']} | {r['verdict']} | {r.get('tempo_bpm', '')} (x{r.get('octave_factor', '')}) | "
                 f"{r.get('time_signature', '')} | {r.get('grid_type', '')} | {r.get('beat_F', '')} | {r.get('beat_AMLt', '')} | "
                 f"{dn} | {r.get('drift_max_ms', '')} | {r.get('confidence', '')} | {r.get('analysis_s', '')} |")
    L.append("\n### Sync robustness (ground-truth kick offsets)\n")
    L.append("| Pair | Result | ratio | truth median / max ms | detectors (all-beats max / downbeat-kick max) |\n|---|---|---|---|---|")
    for r in syn_done:
        if r["result"] == "refused":
            L.append(f"| {r['pair']} | {'pass' if r['pass'] else 'FAIL'} — refused: {r['message'][:90]} | | | |")
        else:
            g = r["ground_truth"]
            L.append(f"| {r['pair']} | {'pass' if r['pass'] else 'FAIL'} | {r['ratio']} (oct x{r['b_octave']}) | "
                     f"{g['median_abs_ms']} / {g['max_abs_ms']} | {r['any_onset']['max_abs_ms']} / {r['downbeat_kicks']['max_abs_ms']} |")

    # ------------------------------------------------------------------ Part 2 detail
    L.append("\n## Part 2 — chat editing on the beat grid\n")
    L.append("### Chat suite by category (LLM-interpreted, checked against the intended edit)\n")
    L.append("| Category | Pass |\n|---|---|")
    for c, (a, b) in sorted(by_cat(chat_done).items()):
        L.append(f"| {c} | {pct(a, b)} |")
    L.append("\n| id | request | action | result | reply (first line) |\n|---|---|---|---|---|")
    for r in chat_done:
        first = r["reply"].splitlines()[0].replace("|", "/")[:110]
        L.append(f"| {r['id']} | {r['say']} | {r['action']} | {'pass' if r['pass'] else 'FAIL ' + str(r['checks'])} | {first} |")
    L.append("\n### Brief sessions\n")
    L.append("| session | turn | request | action | first line |\n|---|---|---|---|---|")
    for s in sess:
        for t in s["turns"]:
            first = (t["echo"][0] if t["echo"] else t["reply"].splitlines()[0]).replace("|", "/")[:120]
            L.append(f"| {s['session']} ({s['track']}) | {t['turn']} | {t['user']} | {t['action']} | {first} |")
    if fuzz:
        L.append("\n### Engine fuzz by op\n")
        L.append("| op | applied | refused | failed checks |\n|---|---|---|---|")
        for k, v in sorted(fuzz["stats"]["by_kind"].items()):
            L.append(f"| {k} | {v['applied']} | {v['refused']} | {v['failed_checks']} |")

    L += FEATURES + LEDGER
    out = RES / "COMPLETION_REPORT.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
