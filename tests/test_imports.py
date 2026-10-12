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
           "evals.eval_real", "evals.part1_report", "evals.make_testpack", "evals.smoke_test_guard",
           "beatagent.sections", "evals.eval_sessions", "evals.eval_robustness", "evals.eval_sync_robust",
           "evals.eval_chat_suite", "evals.completion_report", "beatagent.score", "beatagent.transcribe",
           "beatagent.webaudio", "evals.make_p3pack", "evals.eval_part3", "evals.part3_render",
           "evals.eval_real_selfsync", "evals.llm_check_report", "beatagent.score_chat", "evals.export_evidence",
           "evals.eval_part3_real"]


def control_chars():
    """Shell-escaped patches once wrote a literal backspace (\\x08) where a regex
    needed \\b, so the offline sync rule could never match. Source has no business
    containing control characters other than newline and tab."""
    root = Path(__file__).resolve().parents[1]
    hits = []
    for d in ("beatagent", "app", "evals", "tests"):
        for f in (root / d).rglob("*.py"):
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if any(ord(c) < 32 and c != "\t" for c in line):
                    hits.append(f"{f.relative_to(root)}:{n}")
    return hits


def main():
    bad = []
    cc = control_chars()
    print(f"  [{'FAIL' if cc else 'PASS'}] no control characters in source {cc}")
    bad += cc
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
