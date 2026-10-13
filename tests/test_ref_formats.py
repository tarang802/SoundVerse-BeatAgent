"""Reference annotation layouts the Part 1 scorer accepts (evals/part1/eval_beats.py).

    python -m tests.test_ref_formats

The official test pack was not provided, so its annotation format is unknown. These
are the layouts in common use; each must give the same beats and downbeats.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.part1.eval_beats import find_reference, load_ref  # noqa: E402

WORK = ROOT / "work" / "test_ref_formats"
FAILS: list[str] = []
BEATS = np.round(0.48 + 0.5 * np.arange(16), 6)
POS = (np.arange(16) % 4) + 1
DOWNS = BEATS[POS == 1]


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def write(name, lines):
    p = WORK / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    cases = {
        "tab time+position (.beats)": (write("a.beats", [f"{t}\t{p}" for t, p in zip(BEATS, POS)]), True),
        "space time+position (.txt)": (write("b.txt", [f"{t} {p}" for t, p in zip(BEATS, POS)]), True),
        "csv with header": (write("c.csv", ["time,position"] + [f"{t},{p}" for t, p in zip(BEATS, POS)]), True),
        "comments + blank lines": (write("d.txt", ["# beats", ""] + [f"{t}\t{p}" for t, p in zip(BEATS, POS)]), True),
        "time bar beat (3 cols)": (write("e.txt", [f"{t} {i // 4 + 1} {p}" for i, (t, p) in enumerate(zip(BEATS, POS))]), True),
        "bar.beat decimal": (write("f.txt", [f"{t} {i // 4 + 1}.{p}" for i, (t, p) in enumerate(zip(BEATS, POS))]), True),
        "times only": (write("g.txt", [f"{t}" for t in BEATS]), False),
        "unsorted rows": (write("h.txt", [f"{t}\t{p}" for t, p in reversed(list(zip(BEATS, POS)))]), True),
    }
    for name, (path, has_downs) in cases.items():
        b, d = load_ref(path)
        ok_b = len(b) == len(BEATS) and np.allclose(b, BEATS)
        ok_d = (d is not None and np.allclose(d, DOWNS)) if has_downs else d is None
        check(name, ok_b and ok_d, f"{len(b)} beats, downbeats {None if d is None else len(d)}")
    # separate downbeat file overrides
    b, d = load_ref(cases["times only"][0], write("g_down.txt", [f"{t}" for t in DOWNS]))
    check("separate downbeat file", d is not None and np.allclose(d, DOWNS))
    try:
        load_ref(write("empty.txt", ["# nothing"]))
        check("empty file raises a clear error", False)
    except ValueError as e:
        check("empty file raises a clear error", "No numeric beat rows" in str(e), str(e))
    # discovery next to the audio
    lib = WORK / "pack"
    lib.mkdir()
    for f in ("A1.wav", "A1.beats", "A2.flac", "A2_beats.txt", "A2_downbeats.txt", "A3.mp3", "A3.txt", "A4.wav",
              "A10.wav", "A10.beats"):
        (lib / f).write_text("0.5\t1\n", encoding="utf-8")
    found = {a: tuple(x.name if x else None for x in find_reference(lib / a))
             for a in ("A1.wav", "A2.flac", "A3.mp3", "A4.wav", "A10.wav")}
    check("finds A1.beats", found["A1.wav"] == ("A1.beats", None), str(found["A1.wav"]))
    check("finds A2_beats.txt + A2_downbeats.txt", found["A2.flac"] == ("A2_beats.txt", "A2_downbeats.txt"), str(found["A2.flac"]))
    check("finds A3.txt", found["A3.mp3"] == ("A3.txt", None), str(found["A3.mp3"]))
    check("no reference -> None (not scored)", found["A4.wav"] == (None, None), str(found["A4.wav"]))
    check("A1 does not grab A10's file", found["A1.wav"][0] == "A1.beats" and found["A10.wav"][0] == "A10.beats",
          str(found["A10.wav"]))
    print(f"\n{'all reference-format tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
