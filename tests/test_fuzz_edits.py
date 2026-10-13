"""Randomised edit sequences on a groove with KNOWN beats (no LLM, no tracker).

    python -m tests.test_fuzz_edits [--seqs 40] [--ops 6] [--seed 0]

After every random typed edit (all 7 kinds, beats and bars, random ranges and
parameters) the invariants of the edit engine are checked:
  length   rendered samples == grid duration
  grid     every grid beat still has its kick within 2 ms (median), >= 95 % found
  clicks   >6 kHz energy at every splice <= +3 dB vs the source material
  order    beats strictly increasing (>= 0.2 s apart here), positions 1..4
  undo     undo restores the previous piece list exactly; redo re-applies it
  errors   impossible edits raise EditError with a message (never another exception)
Results: evals/results/part2_fuzz.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.editing import edl  # noqa: E402
from beatagent.core import qc  # noqa: E402
from beatagent.editing.edl import (Copy, EditError, Fade, Gain, KeepOnly, Move, Position, Range,  # noqa: E402
                           Remove, Repeat)
from beatagent.core.grid import BeatGrid  # noqa: E402
from beatagent.core.onsets import onsets_near  # noqa: E402
from beatagent.editing.session import Session  # noqa: E402
from evals.common.synth import SR, beat_times_constant, render_groove  # noqa: E402


def make_fixture(tmp: Path):
    beats = beat_times_constant(120.0, 0.5, 4 * 40)                  # 40 bars, 80 s
    audio = render_groove(beats, duration=beats[-1] + 1.0, style="house", seed=11)
    stereo = np.stack([audio, audio * 0.9], axis=1)
    p = tmp / "fuzz.wav"
    sf.write(p, stereo, SR, subtype="PCM_24")
    grid = BeatGrid(beats=beats, positions=(np.arange(len(beats)) % 4) + 1, beats_per_bar=4,
                    time_signature="4/4", duration=len(stereo) / SR, constant_tempo=True)
    return p, grid


def rand_range(rng: random.Random, g: BeatGrid) -> Range:
    if rng.random() < 0.5 and g.n_bars >= 2:
        a = rng.randint(1, g.n_bars)
        b = rng.randint(a, min(g.n_bars, a + rng.choice([0, 1, 3, 7])))
        if rng.random() < 0.15:
            return Range(unit="bars", last=rng.randint(1, min(4, g.n_bars)))
        return Range(unit="bars", start=a, end=b)
    n = g.n_beats
    a = rng.randint(0, max(0, n - 2))
    b = rng.randint(a + 1, min(n, a + rng.choice([1, 2, 4, 8, 16])))
    if rng.random() < 0.15:
        return Range(unit="beats", last=rng.randint(1, min(8, n)))
    return Range(unit="beats", start=a, end=b)


def rand_op(rng: random.Random, g: BeatGrid):
    kind = rng.choice(["remove", "keep_only", "repeat", "move", "copy", "fade", "gain"])
    r = rand_range(rng, g)
    if kind == "remove":
        return Remove(range=r)
    if kind == "keep_only":
        return KeepOnly(range=r)
    if kind == "repeat":
        return Repeat(range=r, total_plays=rng.randint(2, 4))
    if kind in ("move", "copy"):
        to = (Position(unit="bars", index=rng.randint(1, g.n_bars)) if rng.random() < 0.6 and g.n_bars >= 1
              else Position(unit="beats", index=rng.randint(0, g.n_beats)) if rng.random() < 0.8
              else Position(unit="bars", index=None))
        return Move(range=r, to=to) if kind == "move" else Copy(range=r, to=to)
    if kind == "fade":
        return Fade(range=r, direction=rng.choice(["in", "out"]))
    return Gain(range=r, db=rng.uniform(-12, 6))


def check(s: Session, base_flux_ok: float) -> dict:
    y = s.render()
    g = s.grid
    out = {}
    out["length_ok"] = abs(len(y) - round(g.duration * s.sr)) <= 1
    b = g.beats
    out["order_ok"] = bool(len(b) == 0 or (np.all(np.diff(b) > 0.2) and set(np.unique(g.positions)) <= {1, 2, 3, 4}))
    if len(b):
        # only beats at full level are a fair grid test (a fade-out tail has no measurable kick)
        # ...and a beat in the first 50 ms has no audio before it, so onset detectors can't
        # measure it (same edge rule as sync.EDGE_SKIP; every failure of the first run was
        # exactly one beat at t~0 after keep_only/copy-to-start).
        # (0.15 not 0.05: a beat at the start of a fade-in from silence has its kick at ~0
        # gain -- inaudible by design, seen in seq 23 -- and must not be graded)
        env_ok = np.array([t >= 0.05 and np.abs(y[max(0, int(t * s.sr) - 200): int(t * s.sr) + 2000]).max() > 0.15
                           for t in b])
        on = onsets_near(y, s.sr, b[env_ok]) if env_ok.any() else np.array([])
        off = (on - b[env_ok]) * 1000 if env_ok.any() else np.array([])
        found = ~np.isnan(off)
        if len(off) == 0:                     # e.g. a 1-beat track: nothing measurable, not a failure
            out["grid_found_share"], out["grid_median_ms"], out["grid_ok"] = None, None, True
            out["grid_measurable"] = False
        else:
            out["grid_found_share"] = float(found.mean())
            out["grid_median_ms"] = float(np.median(np.abs(off[found]))) if found.any() else None
            out["grid_ok"] = bool(out["grid_found_share"] >= 0.95 and out["grid_median_ms"] is not None
                                  and abs(out["grid_median_ms"] - base_flux_ok) <= 2.0)
            out["grid_measurable"] = True
    else:
        out["grid_ok"] = True
    pts = qc.splice_points_gains(s.state, s.sr)
    clicks = [qc.splice_click_db(y, s.audio, s.sr, t, src, src_gains=g) for t, src, g in pts]
    out["max_click_db"] = float(max(clicks)) if clicks else None
    out["clicks_ok"] = all(c <= 3.0 for c in clicks)
    if not out["clicks_ok"]:          # where, and which window: transition (crossfade) or junction
        out["click_detail"] = [{"out_t": round(t, 4), "src": [round(x, 4) for x in src], "db": round(c, 2),
                                "transition_db": round(qc._window_db(y, s.audio, s.sr, t, src, 0.012, -0.001, _g), 2),
                                "junction_db": round(qc._window_db(y, s.audio, s.sr, t, src, 0.001, 0.001, _g), 2)}
                               for (t, src, _g), c in zip(pts, clicks) if c > 3.0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=40)
    ap.add_argument("--ops", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tmp = Path(tempfile.mkdtemp())
    path, grid = make_fixture(tmp)
    s0 = Session(path, tmp / "w", grid=grid)
    base = np.median(np.abs((onsets_near(s0.audio, s0.sr, grid.beats) - grid.beats) * 1000))
    stats = {"ops": 0, "applied": 0, "refused_cleanly": 0, "crashed": 0, "length_fail": 0, "order_fail": 0,
             "grid_fail": 0, "click_fail": 0, "undo_fail": 0, "redo_fail": 0, "by_kind": {}}
    failures = []
    for q in range(args.seqs):
        s = Session(path, tmp / f"s{q}", grid=grid)
        for k in range(args.ops):
            if s.grid.n_beats < 2:              # the sequence trimmed the track to almost nothing
                break
            op = rand_op(rng, s.grid)
            kind = op.op
            bk = stats["by_kind"].setdefault(kind, {"applied": 0, "refused": 0, "failed_checks": 0})
            stats["ops"] += 1
            before = s.state.pieces
            try:
                t = s.apply([op])
            except Exception as e:  # noqa: BLE001  -- anything but a TurnResult/EditError is a bug
                stats["crashed"] += 1
                failures.append({"seq": q, "op": op.model_dump(), "crash": repr(e)})
                continue
            if not t.ok:
                stats["refused_cleanly"] += 1
                bk["refused"] += 1
                if not t.error:
                    failures.append({"seq": q, "op": op.model_dump(), "error": "refused without a message"})
                continue
            stats["applied"] += 1
            bk["applied"] += 1
            c = check(s, base)
            bad = [n for n in ("length_ok", "order_ok", "grid_ok", "clicks_ok") if not c[n]]
            for n in bad:
                stats[{"clicks_ok": "click_fail"}.get(n, n.replace("_ok", "_fail"))] += 1
            if bad:
                bk["failed_checks"] += 1
                failures.append({"seq": q, "step": k, "op": op.model_dump(), "failed": bad, "check": c})
            after = s.state.pieces
            u = s.undo()
            if not (u.ok and s.state.pieces == before):
                stats["undo_fail"] += 1
                failures.append({"seq": q, "step": k, "op": op.model_dump(), "failed": ["undo"]})
            r = s.redo()
            if not (r.ok and s.state.pieces == after):
                stats["redo_fail"] += 1
                failures.append({"seq": q, "step": k, "op": op.model_dump(), "failed": ["redo"]})
        # format check once per sequence
        paths = s.export(stem="final", with_click=False)
        info = sf.info(paths["audio"])
        if info.samplerate != SR or info.subtype != "PCM_24":
            failures.append({"seq": q, "failed": ["format"], "got": [info.samplerate, info.subtype]})
            stats.setdefault("format_fail", 0)
            stats["format_fail"] = stats.get("format_fail", 0) + 1
    total_checks = stats["applied"]
    stats["pass_rate_applied_ops"] = round(
        1 - (stats["length_fail"] + stats["order_fail"] + stats["grid_fail"] + stats["click_fail"]) / max(1, total_checks), 4)
    out = {"label": "engine fuzz on a synthetic 120 BPM groove with known beats (no LLM, no tracker)",
           "seed": args.seed, "stats": stats, "failures": failures[:50], "n_failures": len(failures),
           "baseline_grid_median_ms": round(float(base), 3)}
    (ROOT / "evals" / "results").mkdir(exist_ok=True)
    (ROOT / "evals" / "results" / "part2_fuzz.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(stats, indent=1))
    print(f"failures: {len(failures)}")
    for f in failures[:8]:
        print("  ", json.dumps(f, default=str)[:300])


if __name__ == "__main__":
    main()
