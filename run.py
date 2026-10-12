"""One entry point (no `make` on stock Windows).

    python run.py models      # download Beat This! checkpoints (resumable)
    python run.py smoke       # environment + pipeline smoke test (synthetic)
    python run.py test        # edit-engine unit tests
    python run.py testpack    # build the synthetic substitute test pack
    python run.py eval        # Part 1 evals on the substitute pack
    python run.py app         # web app on http://localhost:8000
    python run.py demo        # models + testpack + test + eval, then the app
"""
from __future__ import annotations

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


def mod(m, *a):
    subprocess.run([PY, "-m", m, *a], cwd=ROOT, check=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "models":
        models()
    elif cmd == "smoke":
        mod("evals.smoke_test")
    elif cmd == "test":
        mod("tests.test_edl")
    elif cmd == "testpack":
        mod("evals.make_testpack")
    elif cmd == "eval":
        mod("evals.eval_beats", *sys.argv[2:])
    elif cmd == "app":
        mod("app.server")
    elif cmd == "demo":
        models()
        mod("evals.make_testpack")
        mod("tests.test_edl")
        mod("evals.eval_beats")
        mod("app.server")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
