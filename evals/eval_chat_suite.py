"""Part 2 chat suite: ~55 producer-style requests the prompt has never seen.

    python -m evals.eval_chat_suite [--only id ...] [--llm gemini|rules]

Each case says something to the agent and checks the RESULT, not the wording:
the edit actually applied is compared sample-for-sample with the edit a producer
meant (the expected typed ops applied to a fresh copy). Requests that can fairly be
read two ways list both readings. Refusals must leave the audio unchanged and say
why; answers must contain the measured fact. Every changed result also gets the
audio checks (format, length, clicks, grid on the edited audio).

LLM answers are recorded (BEAT_AGENT_LLM_CACHE) so re-runs are free and repeatable.
Cases answered by the offline fallback (LLM quota exhausted) are reported apart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BEAT_AGENT_LLM_CACHE", str(ROOT / "evals" / "results" / "llm_cache"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from beatagent.agent import BeatAgent  # noqa: E402
from beatagent.analyze import analyze  # noqa: E402
from beatagent.audio_io import load  # noqa: E402
from beatagent.edl import Copy, Fade, Gain, KeepOnly, Move, Position, Range, Remove, Repeat  # noqa: E402
from beatagent.grid import BeatGrid  # noqa: E402
from beatagent.llm import GeminiInterpreter, RuleInterpreter  # noqa: E402
from beatagent.session import Session  # noqa: E402
from evals.eval_sessions import check_turn, grid_truth  # noqa: E402

GRIDS = ROOT / "evals" / "results" / "part1_substitute_beat_this"
REAL = Path("C:/Users/Gupt Ah/Downloads/Telegram Desktop/12. Seedhe Maut - Naam Kaam Sheher [E].flac")
OUT = ROOT / "evals" / "results" / "part2_chat_suite"


def bars(a, b=None):
    return Range(unit="bars", start=a, end=a if b is None else b)


def beats(a, b):
    return Range(unit="beats", start=a, end=b)


WHOLE = Range(unit="beats")

# id, category, track, setup (typed ops applied before, or ["chat", msg]), say, expectation
CASES = [
    ("r1", "remove", "A1", [], "remove bar 5", {"ops": [[Remove(range=bars(5))]]}),
    ("r2", "remove", "A1", [], "delete bars 9 to 12", {"ops": [[Remove(range=bars(9, 12))]]}),
    ("r3", "remove", "A1", [], "get rid of beats 32 through 48", {"ops": [[Remove(range=beats(32, 48))], [Remove(range=beats(32, 49))]]}),
    ("r4", "remove", "A1", [], "cut the first 2 bars", {"ops": [[Remove(range=bars(1, 2))], [Remove(range=Range(unit="bars", end=2))]]}),
    ("r5", "remove", "A1", [], "drop the last bar", {"ops": [[Remove(range=Range(unit="bars", last=1))]]}),
    ("r6", "remove", "A1", [], "kill bars 20-24", {"ops": [[Remove(range=bars(20, 24))]]}),
    ("k1", "keep_only", "A1", [], "keep only bars 17 to 32", {"ops": [[KeepOnly(range=bars(17, 32))]]}),
    ("k2", "keep_only", "A1", [], "trim it down to just bars 1-8", {"ops": [[KeepOnly(range=bars(1, 8))], [KeepOnly(range=Range(unit="bars", end=8))]]}),
    ("k3", "keep_only", "A1", [], "I only want the first 16 bars", {"ops": [[KeepOnly(range=bars(1, 16))], [KeepOnly(range=Range(unit="bars", end=16))]]}),
    ("p1", "repeat", "A1", [], "loop bars 9-12 three times", {"ops": [[Repeat(range=bars(9, 12), total_plays=3)]]}),
    ("p2", "repeat", "A1", [], "repeat bar 7", {"ops": [[Repeat(range=bars(7), total_plays=2)]]}),
    ("p3", "repeat", "A1", [], "double bars 33 to 40", {"ops": [[Repeat(range=bars(33, 40), total_plays=2)]]}),
    ("p4", "repeat", "A1", [], "play the first 8 beats 4 times", {"ops": [[Repeat(range=beats(0, 8), total_plays=4)], [Repeat(range=Range(unit="beats", end=8), total_plays=4)]]}),
    ("m1", "move", "A1", [], "move bar 10 to the start", {"ops": [[Move(range=bars(10), to=Position(unit="bars", index=1))], [Move(range=bars(10), to=Position(unit="beats", index=0))]]}),
    ("m2", "move", "A1", [], "move bars 3-4 after bar 8", {"ops": [[Move(range=bars(3, 4), to=Position(unit="bars", index=9))]]}),
    ("c1", "copy", "A1", [], "copy bar 2 to the end", {"ops": [[Copy(range=bars(2), to=Position(unit="bars", index=None))]]}),
    ("c2", "copy", "A1", [], "put a copy of bars 5-6 right before bar 20", {"ops": [[Copy(range=bars(5, 6), to=Position(unit="bars", index=20))]]}),
    ("f1", "fade", "A1", [], "fade in the first 4 bars", {"ops": [[Fade(range=bars(1, 4), direction="in")], [Fade(range=Range(unit="bars", end=4), direction="in")]]}),
    ("f2", "fade", "A1", [], "fade out the last 2 bars", {"ops": [[Fade(range=Range(unit="bars", last=2), direction="out")]]}),
    ("f3", "fade", "A1", [], "fade out bars 180 to 186", {"ops": [[Fade(range=bars(180, 186), direction="out")]]}),
    ("g1", "gain", "A1", [], "turn bars 9-16 down by 6 dB", {"ops": [[Gain(range=bars(9, 16), db=-6)]]}),
    ("g2", "gain", "A1", [], "make the whole track 3 dB louder", {"ops": [[Gain(range=WHOLE, db=3)]]}),
    ("g3", "gain", "A1", [], "boost the first bar by 4.5 dB", {"ops": [[Gain(range=bars(1), db=4.5)], [Gain(range=Range(unit="bars", end=1), db=4.5)]]}),
    ("x1", "multi-op", "A1", [], "remove bar 5 and fade in the first bar", {"ops": [[Remove(range=bars(5)), Fade(range=bars(1), direction="in")]]}),
    ("x2", "multi-op", "A1", [], "loop bars 1-2 twice, then fade out the last 4 beats",
     {"ops": [[Repeat(range=bars(1, 2), total_plays=2), Fade(range=Range(unit="beats", last=4), direction="out")]]}),
    ("s1", "grid-follow", "A1", [[Remove(range=bars(1, 4))]], "now remove bar 1", {"ops": [[Remove(range=bars(1))]]}),
    ("s2", "grid-follow", "A1", [[Repeat(range=bars(1, 2), total_plays=2)]], "remove bar 3", {"ops": [[Remove(range=bars(3))]]}),
    ("u1", "undo-redo", "A1", [[Remove(range=bars(5))]], "undo", {"state": "undo1"}),
    ("u2", "undo-redo", "A1", [[Remove(range=bars(5))], [Remove(range=bars(6))]], "undo that", {"state": "undo1"}),
    ("u3", "undo-redo", "A1", [], "undo", {"no_change": True}),
    ("u4", "undo-redo", "A1", [[Remove(range=bars(5))], "UNDO"], "redo", {"ops": [[]], "redo": True}),
    ("b1", "bad-input", "A1", [], "delete bar 500", {"no_change": True}),
    ("b2", "bad-input", "A1", [], "remove bars 10 to 5", {"no_change_or_ops": [[Remove(range=bars(5, 10))]]}),
    ("b3", "bad-input", "A1", [], "repeat bar 3 zero times", {"no_change": True}),
    ("b4", "bad-input", "A1", [], "repeat bar 3 a hundred times", {"no_change": True}),
    ("b5", "bad-input", "A1", [], "make bars 1-4 60 dB louder", {"no_change": True}),
    ("b6", "bad-input", "A1", [], "remove everything", {"no_change": True}),
    ("b7", "bad-input", "A1", [], "make it sound better", {"no_change": True}),
    ("b8", "bad-input", "A1", [], "pitch it up a semitone", {"no_change": True}),
    ("b9", "bad-input", "A1", [], "reverse bar 3", {"no_change": True}),
    ("b10", "bad-input", "A1", [], "asdfgh qwerty", {"no_change": True}),
    ("b11", "bad-input", "A1", [], "remove bar -2", {"no_change": True}),
    ("b12", "bad-input", "A1", [], "remove beat 3.5", {"no_change": True}),
    ("q1", "subjective", "A1", [], "chop out the boring part", {"no_change": True, "proposal": [Remove(range=bars(81, 96))]}),
    ("q2", "subjective", "A1", [["chat", "chop out the boring part"]], "yes", {"ops": [[Remove(range=bars(81, 96))]]}),
    ("q3", "subjective", "A1", [], "cut the breakdown", {"ops_or_proposal": [Remove(range=bars(81, 96))]}),
    ("q4", "subjective", "A1", [["chat", "chop out the boring part"]], "no, leave it", {"no_change": True}),
    ("i1", "question", "A1", [], "what's the tempo?", {"no_change": True, "contains": ["124"]}),
    ("i2", "question", "A1", [], "how many bars are there?", {"no_change": True, "contains": ["186"]}),
    ("i3", "question", "A1", [], "how long is the track?", {"no_change": True, "contains_any": ["361", "6:01", "6 min"]}),
    ("v1", "tempo-change", "A5", [], "remove the first bar after the tempo change", {"ops": [[Remove(range=bars(33))]]}),
    ("v2", "tempo-change", "A5", [], "loop the last bar before the tempo change twice", {"ops": [[Repeat(range=bars(32), total_plays=2)]]}),
    ("v3", "variable-grid", "A3", [], "remove bars 10-12", {"ops": [[Remove(range=bars(10, 12))]]}),
    ("y1", "sync", "A1+A7", [], "bring the other track in at bar 17", {"sync": {"a_bar": 17}}),
    ("y2", "sync", "A1+A7", [], "beatmatch the other deck to this from bar 9 for 30 seconds", {"sync": {"a_bar": 9, "duration": 30}}),
    ("y3", "sync", "A1", [], "sync the other track at bar 9", {"no_change": True}),
    ("n1", "real-song", "REAL", [], "remove bar 8", {"ops": [[Remove(range=bars(8))]]}),
    ("n2", "real-song", "REAL", [], "loop the first 2 bars 4 times", {"ops": [[Repeat(range=bars(1, 2), total_plays=4)]]}),
]

_grids: dict = {}


def session_for(track: str, wd: Path) -> tuple[Session, Session | None]:
    if track == "REAL":
        if "REAL" not in _grids:
            a, f = load(REAL)
            _grids["REAL"] = analyze(a, f.sr)
        return Session(REAL, wd, grid=_grids["REAL"]), None
    name, _, partner = track.partition("+")
    path = next(p for p in (ROOT / "data" / "substitute").glob(f"{name}.*") if p.suffix in (".wav", ".flac", ".mp3"))
    s = Session(path, wd, grid=BeatGrid.load(GRIDS / f"{name}.grid.json"))
    p = None
    if partner:
        pp = next(q for q in (ROOT / "data" / "substitute").glob(f"{partner}.*") if q.suffix in (".wav", ".flac", ".mp3"))
        p = Session(pp, wd, grid=BeatGrid.load(GRIDS / f"{partner}.grid.json"))
    return s, p


def apply_setup(s: Session, agent: BeatAgent, setup) -> None:
    for step in setup:
        if step == "UNDO":
            s.undo()
        elif isinstance(step, list) and step and step[0] == "chat":
            agent.handle(step[1])
        else:
            t = s.apply(step)
            assert t.ok, (step, t.error)


def expected_pieces(track: str, wd: Path, setup, ops) -> tuple:
    s, _ = session_for(track, wd)
    for step in setup:
        if step == "UNDO":
            s.undo()
        elif isinstance(step, list) and step and step[0] == "chat":
            continue
        else:
            s.apply(step)
    if ops:
        t = s.apply(ops)
        if not t.ok:
            return ("ERROR", t.error)
    return s.state.pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--llm", default="gemini", choices=["gemini", "rules"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    interp = GeminiInterpreter() if args.llm == "gemini" else RuleInterpreter()
    rows = []
    for cid, cat, track, setup, say, exp in CASES:
        if args.only and cid not in args.only:
            continue
        if track == "REAL" and not REAL.exists():
            rows.append({"id": cid, "category": cat, "skipped": "real song not on this machine"})
            continue
        wd = OUT / "work" / cid
        s, partner = session_for(track, wd)
        agent = BeatAgent(s, interp, partner=partner)
        t0 = time.time()
        apply_setup(s, agent, setup)
        before = s.state.pieces
        base = grid_truth(s.audio, s.sr, s.src_grid.beats)
        r = agent.handle(say)
        after = s.state.pieces
        changed = after != before
        row = {"id": cid, "category": cat, "track": track, "say": say, "action": r.action, "ok": r.ok,
               "changed": changed, "reply": r.text, "interpretation": r.interpretation, "backend": r.backend,
               "llm_model": getattr(interp, "last_model", None),
               "repair_round": getattr(agent, "retried", None),
               "plan": agent.last_plan.model_dump(exclude_none=True) if agent.last_plan else None}
        offline = "offline-rules" in (r.backend or "") and args.llm == "gemini"
        checks: dict = {}
        if "ops" in exp:
            setup_ops = [st for st in setup if not (isinstance(st, list) and st and st[0] == "chat")]
            if exp.get("redo"):
                exp_states = [expected_pieces(track, wd / "exp", [setup[0]], [])]
            else:
                exp_states = [expected_pieces(track, wd / "exp", setup_ops, ops) for ops in exp["ops"]]
            checks["edit_matches_intent"] = any(after == e for e in exp_states)
        if exp.get("state") == "undo1":
            exp_state = expected_pieces(track, wd / "exp", setup[:-1], [])
            checks["undo_restores"] = after == exp_state
        if exp.get("no_change"):
            checks["audio_unchanged"] = not changed
            checks["clear_reply"] = len(r.text) > 15
        if "no_change_or_ops" in exp:
            alt = expected_pieces(track, wd / "exp", [], exp["no_change_or_ops"][0])
            checks["unchanged_or_right_reading"] = (not changed) or after == alt
        if "proposal" in exp:
            exp_state = expected_pieces(track, wd / "exp", [], exp["proposal"])
            pend = agent.pending
            ok_prop = False
            if pend:
                tmp, _ = session_for(track, wd / "prop")
                tt = tmp.apply(pend)
                ok_prop = tt.ok and tmp.state.pieces == exp_state
            checks["measured_proposal"] = ok_prop
        if "ops_or_proposal" in exp:
            exp_state = expected_pieces(track, wd / "exp", [], exp["ops_or_proposal"])
            ok_direct = after == exp_state
            ok_prop = False
            if agent.pending and not changed:
                tmp, _ = session_for(track, wd / "prop")
                tt = tmp.apply(agent.pending)
                ok_prop = tt.ok and tmp.state.pieces == exp_state
            checks["right_section"] = ok_direct or ok_prop
        if "contains" in exp:
            checks["states_measured_fact"] = all(x in r.text for x in exp["contains"])
        if "contains_any" in exp:
            checks["states_measured_fact"] = any(x in r.text for x in exp["contains_any"])
        if "sync" in exp:
            ls = agent.last_sync
            checks["synced"] = bool(r.ok and r.action == "sync" and ls)
            if ls:
                checks["right_bar"] = ls["plan"]["a_bar"] == exp["sync"]["a_bar"]
                if "duration" in exp["sync"]:
                    checks["right_length"] = abs(ls["plan"]["duration"] - exp["sync"]["duration"]) < 0.01
        if changed:
            paths = s.export(stem="out", with_click=False)
            q = check_turn(s, paths, base)
            checks["audio_clean"] = bool(q["clicks"]["ok"] and q["format"]["ok"] and q["length"]["ok"])
            row["audio_checks"] = {"max_click_db": q["clicks"]["max_db_vs_source"], "format": q["format"],
                                   "grid_edited_median_ms": q["grid_on_edited_audio"]["median_abs_ms"]}
        row["checks"] = checks
        row["pass"] = bool(checks) and all(checks.values())
        row["offline_fallback"] = offline
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        flag = "PASS" if row["pass"] else "FAIL"
        print(f"  {cid:4s} {cat:13s} {flag}{' (offline)' if offline else ''}  '{say}' -> [{r.action}] "
              f"{r.text.splitlines()[0][:95]}" + ("" if row["pass"] else f"   checks={checks}"))
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    done = [r for r in rows if "pass" in r]
    llm = [r for r in done if not r["offline_fallback"]]
    print(f"\npassed {sum(r['pass'] for r in done)}/{len(done)}  (answered by LLM: {len(llm)}, "
          f"passed {sum(r['pass'] for r in llm)}; offline fallback: {len(done) - len(llm)})")
    print(f"LLM usage: {getattr(interp, 'usage', {})}")


if __name__ == "__main__":
    main()
