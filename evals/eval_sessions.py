"""Part 2: run the brief's chat sessions headlessly, in order, and check every turn.

    python -m evals.eval_sessions [--llm auto|gemini|rules] [--pack substitute]

Per turn we save the user message, the interpreter's typed plan, the agent's
reply/echo, and (when the audio changed) the edited audio + grid + click
audition + EDL. Every changed turn is checked:
  * format    exported sample rate and bit depth == the input's
  * length    rendered samples == grid duration
  * clicks    >6 kHz energy at every splice vs the source material it came from (dB;
              > 3 dB = audible click)
  * grid      onsets in the EDITED audio vs the EDITED grid (median |offset|, share of
              beats with an onset), next to the same numbers for the unedited track
Session 4 ("chop out the boring part") is one message in the brief: the agent
proposes from measured sections and waits for "yes"; the harness does not confirm
on the user's behalf, so no audio changes there by design.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from beatagent import qc  # noqa: E402
from beatagent.agent import BeatAgent  # noqa: E402
from beatagent.llm import GeminiInterpreter, RuleInterpreter, make_interpreter  # noqa: E402
from beatagent.onsets import onsets_near  # noqa: E402
from beatagent.session import Session  # noqa: E402

SESSIONS = [
    {"id": 1, "track": "A1", "turns": ["Cut beat 16–32 and repeat it twice.", "Now fade out the last 8 beats.",
                                       "Undo.", "Remove bar 5."]},
    {"id": 2, "track": "A2", "turns": ["Loop the first 4 bars four times, then let the track play on."]},
    {"id": 3, "track": "A5", "turns": ["Repeat the last bar before the tempo change twice."]},
    {"id": 4, "track": "A1", "turns": ["Chop out the boring part."]},
    {"id": 5, "track": "A3", "turns": ["Cut beat 0 to beat 8.", "Delete bars 200–210."]},
    {"id": 6, "track": "A1", "partner": "A7", "turns": ["Sync A7 to A1, starting the overlap at A1's bar 33."]},
]


def track_path(pack: Path, name: str) -> Path:
    return next(p for p in pack.glob(f"{name}.*") if p.suffix in (".wav", ".flac", ".mp3"))


def grid_truth(y: np.ndarray, sr: int, beats: np.ndarray) -> dict:
    """How well the grid sits on the audio's onsets (broadband, +-40 ms)."""
    on = onsets_near(y, sr, beats)
    off = (on - beats) * 1000
    found = ~np.isnan(off)
    return {"share_with_onset": round(float(found.mean()), 3),
            "median_abs_ms": round(float(np.median(np.abs(off[found]))), 2) if found.any() else None,
            "p95_abs_ms": round(float(np.percentile(np.abs(off[found]), 95)), 2) if found.any() else None}


def check_turn(s: Session, paths: dict, baseline: dict) -> dict:
    y = s.render()
    info = sf.info(paths["audio"])
    clicks = [qc.splice_click_db(y, s.audio, s.sr, t, src) for t, src in qc.splice_points(s.state, s.sr)]
    g = s.grid
    return {
        "format": {"sr": info.samplerate, "subtype": info.subtype, "expected_sr": s.sr,
                   "expected_subtype": s.fmt.out_subtype,
                   "ok": info.samplerate == s.sr and info.subtype == s.fmt.out_subtype},
        "length": {"samples": int(len(y)), "grid_duration_samples": int(round(g.duration * s.sr)),
                   "ok": abs(len(y) - round(g.duration * s.sr)) <= 1},
        "clicks": {"n_splices": len(clicks), "max_db_vs_source": round(max(clicks), 1) if clicks else None,
                   "ok": all(c <= 3.0 for c in clicks)},
        "grid_on_edited_audio": grid_truth(y, s.sr, g.beats),
        "grid_on_original_audio": baseline,
        "peak": paths.get("peak"), "clipped_samples": paths.get("clipped_samples"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="auto", choices=["auto", "gemini", "rules"])
    ap.add_argument("--pack", default="substitute")
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args()
    pack = ROOT / "data" / args.pack
    out = ROOT / "evals" / "results" / "part2_sessions"
    out.mkdir(parents=True, exist_ok=True)
    interp = (RuleInterpreter() if args.llm == "rules" else GeminiInterpreter() if args.llm == "gemini"
              else make_interpreter())
    print(f"interpreter: {interp.name} {getattr(interp, 'model', '')}")
    sessions_cache: dict[str, Session] = {}
    results = []
    for sess in SESSIONS:
        if args.only and sess["id"] not in args.only:
            continue
        sd = out / f"session_{sess['id']}"
        sd.mkdir(exist_ok=True)
        t0 = time.time()
        # a fresh Session per session (edits don't leak between sessions); analysis is reused
        src = track_path(pack, sess["track"])
        cached = sessions_cache.get(sess["track"])
        s = Session(src, sd, grid=cached.src_grid if cached else None)
        sessions_cache.setdefault(sess["track"], s)
        partner = None
        if sess.get("partner"):
            pc = sessions_cache.get(sess["partner"])
            partner = Session(track_path(pack, sess["partner"]), sd, grid=pc.src_grid if pc else None)
            sessions_cache.setdefault(sess["partner"], partner)
        agent = BeatAgent(s, interp, partner=partner)
        baseline = grid_truth(s.audio, s.sr, s.src_grid.beats)
        tokens0 = dict(getattr(interp, "usage", {}))
        turns = []
        for k, text in enumerate(sess["turns"], 1):
            r = agent.handle(text)
            plan = agent.last_plan.model_dump(exclude_none=True) if agent.last_plan else None
            turn = {"turn": k, "user": text, "action": r.action, "ok": r.ok, "changed": r.changed,
                    "reply": r.text, "echo": r.echo, "warnings": r.warnings,
                    "interpretation": r.interpretation, "backend": r.backend, "plan": plan,
                    "llm_repair_round": getattr(agent, "retried", None),
                    "version_after": s.pos, "duration_after_s": round(s.grid.duration, 3)}
            if r.changed or (r.action in ("undo", "redo") and r.ok):
                paths = s.export(stem=f"turn{k}")
                turn["files"] = {k2: str(Path(v).relative_to(ROOT)) for k2, v in paths.items() if isinstance(v, str)}
                turn["checks"] = check_turn(s, paths, baseline)
            if r.action == "sync" and agent.last_sync:
                turn["sync"] = agent.last_sync
            turns.append(turn)
            print(f"  s{sess['id']} t{k} [{r.action} ok={r.ok}] {text}\n      -> {r.text.splitlines()[0][:150]}")
        usage = getattr(interp, "usage", {})
        res = {"session": sess["id"], "track": sess["track"], "partner": sess.get("partner"),
               "elapsed_s": round(time.time() - t0, 1), "turns": turns,
               "llm_tokens": {k: usage.get(k, 0) - tokens0.get(k, 0) for k in ("calls", "input_tokens", "output_tokens")}}
        (sd / "session.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        results.append(res)

    # summary table
    md = ["| session | turn | user | action | ok | echo / reply (first line) | clicks max dB | format | grid median ms (edited / original) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        for t in res["turns"]:
            c = t.get("checks", {})
            first = (t["echo"][0] if t["echo"] else t["reply"].splitlines()[0]).replace("|", "/")
            md.append(f"| {res['session']} ({res['track']}) | {t['turn']} | {t['user']} | {t['action']} | {t['ok']} | "
                      f"{first[:160]} | {c.get('clicks', {}).get('max_db_vs_source', '—')} | "
                      f"{'ok' if c.get('format', {}).get('ok') else ('—' if not c else 'FAIL')} | "
                      f"{c.get('grid_on_edited_audio', {}).get('median_abs_ms', '—')} / "
                      f"{c.get('grid_on_original_audio', {}).get('median_abs_ms', '—')} |")
    tok = {k: sum(r["llm_tokens"][k] for r in results) for k in ("calls", "input_tokens", "output_tokens")}
    md.append(f"\nInterpreter: {interp.name} {getattr(interp, 'model', '')}; LLM calls {tok['calls']}, "
              f"tokens in {tok['input_tokens']} / out {tok['output_tokens']}.")
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
