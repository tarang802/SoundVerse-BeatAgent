"""One entry point (no `make` on stock Windows).

    python run.py models      # download Beat This! checkpoints (resumable)
    python run.py smoke       # environment + pipeline smoke test (synthetic)
    python run.py test        # every deterministic test suite (no API key needed), summarised
    python run.py e2e         # the real app in headless Edge/Chrome: the demo path end to end (~3 min)
    python run.py fuzz        # Part 2: 421 random edits, invariants after each (~5 min)
    python run.py testpack    # build the synthetic substitute test pack (A1-A7)
    python run.py p3pack      # build the synthetic Part 3 clips (P1-P4 + reference MIDI)
    python run.py eval        # Part 1 evals on the substitute pack
    python run.py syncval     # Part 1 sync comparison (A7 -> A1 @ bar 33) + PART1_REPORT.md
    python run.py p3eval      # Part 3: transcribe, render pages offline in Edge/Chrome, score vs MIDI
    python run.py chat        # Part 2 LLM evals: 59-case chat suite + the brief's 6 sessions (needs GEMINI_API_KEY)
    python run.py report      # regenerate evals/results/COMPLETION_REPORT.md from the result files
    python run.py app         # web app on http://localhost:8000 (serves the committed frontend/dist)
    python run.py ui          # rebuild the React frontend (needs Node 18+): npm install + npm run build
    python run.py demo        # models + both test packs + tests + Part 1 and Part 3 evals, then the app
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CKPT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/{}.ckpt"
CKPTS = {"small0": 8451101}   # final0 (81 MB) is used automatically if present


def models():
    d = ROOT / "models"
    d.mkdir(exist_ok=True)
    for name, size in CKPTS.items():
        p = d / f"{name}.ckpt"
        have = p.stat().st_size if p.exists() else 0
        if have == size:
            print(f"{name}: present")
            continue
        print(f"{name}: downloading ({have}/{size} bytes so far)")
        req = urllib.request.Request(CKPT_URL.format(name), headers={"Range": f"bytes={have}-"})
        with urllib.request.urlopen(req) as r, open(p, "ab") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
        print(f"{name}: done ({p.stat().st_size} bytes)")


# children print non-ASCII (arrows, en dashes); a Windows pipe defaults to cp1252
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def mod(m, *a):
    subprocess.run([PY, "-m", m, *a], cwd=ROOT, check=True, env=ENV)


TESTS = ["tests.test_imports", "tests.test_edl", "tests.test_score", "tests.test_ref_formats",
         "tests.test_click_metric", "tests.test_agent_llm", "tests.test_audio_paths", "tests.test_recreate_api",
         "tests.test_score_chat", "tests.test_webaudio_page", "tests.test_analyze_rules"]


def run_tests() -> bool:
    """Every deterministic suite, all run even if one fails; summary at the end."""
    if not (ROOT / "data" / "substitute" / "A1.wav").exists():
        mod("evals.part1.make_testpack")                   # test_audio_paths / test_recreate_api open A1
    if not (ROOT / "data" / "p3" / "P1.wav").exists():
        mod("evals.part3.make_p3pack")                     # test_recreate_api needs the clips
    results = []
    for m in TESTS:
        r = subprocess.run([PY, "-m", m], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=ENV)
        out = r.stdout + r.stderr
        results.append((m, r.returncode, out.count("[PASS]"), out.count("[FAIL]")))
        print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {m}: {out.count('[PASS]')} passed, {out.count('[FAIL]')} failed")
        if r.returncode:
            print("\n".join("    " + x for x in out.splitlines() if "[FAIL]" in x or "Error" in x)[:2000])
    bad = [m for m, rc, *_ in results if rc]
    print(f"\n{len(TESTS) - len(bad)}/{len(TESTS)} suites passed; "
          f"{sum(r[2] for r in results)} checks passed, {sum(r[3] for r in results)} failed")
    return not bad


def main():
    # line-buffered: when output goes to a log, the test summary must appear before the
    # app starts serving (it was lost in a buffer when `demo` ran into a file)
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "models":
        models()
    elif cmd == "smoke":
        mod("evals.smoke_test")
    elif cmd == "test":
        sys.exit(0 if run_tests() else 1)
    elif cmd == "e2e":
        mod("tests.test_ui_e2e")
    elif cmd == "fuzz":
        mod("tests.test_fuzz_edits", "--seqs", "60", "--ops", "8", "--seed", "1")
    elif cmd == "testpack":
        mod("evals.part1.make_testpack")
    elif cmd == "p3pack":
        mod("evals.part3.make_p3pack")
    elif cmd == "p3eval":
        mod("evals.part3.eval_part3")
    elif cmd == "chat":
        mod("evals.part2.eval_chat_suite")
        mod("evals.part2.eval_sessions", "--llm", "gemini")
    elif cmd == "report":
        mod("evals.completion_report")
    elif cmd == "eval":
        mod("evals.part1.eval_beats", *sys.argv[2:])
    elif cmd == "app":
        mod("app.server")
    elif cmd == "ui":
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=ROOT / "frontend", check=True)
        subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True)
    elif cmd == "syncval":
        mod("evals.part1.eval_sync")
        mod("evals.part1.part1_report")
    elif cmd == "demo":
        models()
        mod("evals.part1.make_testpack")
        mod("evals.part3.make_p3pack")
        run_tests()
        mod("evals.part1.eval_beats")
        mod("evals.part1.eval_sync")
        mod("evals.part1.part1_report")
        mod("evals.part3.eval_part3")
        mod("app.server")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
