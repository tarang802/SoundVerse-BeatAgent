"""Every module must at least import. A broken string literal in agent.py once
stopped the server from starting while every eval still passed (none imported it).

    python -m tests.test_imports
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = ["beatagent.audio_io", "beatagent.grid", "beatagent.onsets", "beatagent.analyze", "beatagent.click",
           "beatagent.stretch", "beatagent.sync", "beatagent.edl", "beatagent.session", "beatagent.llm",
           "beatagent.agent", "beatagent.qc", "app.server", "evals.eval_beats", "evals.eval_sync",
           "evals.eval_real", "evals.part1_report", "evals.make_testpack", "evals.smoke_test_guard"]


def main():
    bad = []
    for m in MODULES:
        if m == "evals.smoke_test_guard":       # smoke_test runs on import; only compile it
            import py_compile
            py_compile.compile(str(Path(__file__).resolve().parents[1] / "evals" / "smoke_test.py"), doraise=True)
            print("  [PASS] evals/smoke_test.py compiles")
            continue
        try:
            importlib.import_module(m)
            print(f"  [PASS] import {m}")
        except Exception as e:  # noqa: BLE001
            bad.append(m)
            print(f"  [FAIL] import {m}: {type(e).__name__}: {e}")
    assert not bad, bad
    print("all modules import")


if __name__ == "__main__":
    main()
