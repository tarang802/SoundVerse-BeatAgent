"""Control for real-music sync numbers: sync a real track with ITSELF.

    python -m evals.eval_real_selfsync FILE [FILE ...]

Real songs have no answer key, so the sync detectors cannot be checked against
truth. A self-sync is perfectly aligned by construction (same grid, same audio):
  - B bar 9 -> A bar 9: identical content; any offset is pipeline error.
  - B bar 1 -> A bar 9: different bars of the same, grid-aligned song; any
    offset is what the detector reports when the bars' hits differ -- its floor
    on this music. Real A/B pair numbers near that floor are not evidence of
    misalignment (nor proof of <= 5 ms: listen to the mix).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent import sync  # noqa: E402
from beatagent.analyze import analyze  # noqa: E402
from beatagent.audio_io import load  # noqa: E402

OUT = ROOT / "evals" / "results" / "real" / "selfsync.json"


def main():
    rows = []
    for f in sys.argv[1:]:
        a, fmt = load(Path(f))
        g = analyze(a, fmt.sr)
        for b_bar in (9, 1):
            p = sync.plan_sync(g, g, 9, b_bar, 60)
            r = sync.render(a, fmt.sr, a, fmt.sr, p, g, sync.matched_b_grid(p, g))
            m = sync.measure_alignment(r["a"], r["b"], r["sr"], g, p)
            k = sync.measure_downbeat_kicks(r["a"], r["b"], r["sr"], g, p)
            row = {"file": Path(f).name, "b_bar": b_bar, "a_bar": 9,
                   "content": "identical" if b_bar == 9 else "different bars",
                   "any_onset": {x: m.get(x) for x in ("n_measured", "n_beats", "median_abs_ms", "max_abs_ms")},
                   "downbeat_kicks": {x: k.get(x) for x in ("n_measured", "n_downbeats", "median_abs_ms", "max_abs_ms")}}
            rows.append(row)
            print(json.dumps(row))
    OUT.write_text(json.dumps({"label": "self-sync control (perfect alignment by construction)", "rows": rows},
                              indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
