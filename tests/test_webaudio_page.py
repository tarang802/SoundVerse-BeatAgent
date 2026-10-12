"""The generated Part 3 page is safe and self-contained (beatagent/webaudio.py).

    python -m tests.test_webaudio_page

User-controlled text (a clip's file name, track names) ends up inside the page. It must
stay inert data: it cannot close the <script> element that holds the score, cannot add
markup through the title, and the page must keep its no-network policy.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.score import Note, Score, Track, default_patch  # noqa: E402
from beatagent.webaudio import render_html  # noqa: E402
from evals.part3_render import static_checks  # noqa: E402

FAILS: list[str] = []
EVIL = '</script><script>alert(1)</script><img src=x onerror=alert(2)>'


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def main():
    s = Score(tempo_bpm=120, bars=1, loop_beats=4, source={"name": EVIL},
              tracks=[Track(name="drums" + EVIL, kind="drums", patch=default_patch("drums"),
                            notes=[Note(start=0, duration=0.25, pitch=36, velocity=100)])])
    html = render_html(s)
    scripts = re.findall(r"<script\b", html)
    check("exactly the template's two <script> elements", len(scripts) == 2, str(len(scripts)))
    check("no injected </script> from the data", html.count("</script>") == 2)
    check("no injected markup (img onerror)", "<img" not in html)
    check("mixer never builds markup from track names", not re.search(r"\.innerHTML\s*=", html))
    title = re.search(r"<title>(.*?)</title>", html).group(1)
    check("title sanitised", "<" not in title and ">" not in title, title[:60])
    data = re.search(r'<script id="score" type="application/json">(.*?)</script>', html, re.S).group(1)
    back = json.loads(data)
    check("score data intact after escaping", back["tracks"][0]["name"] == "drums" + EVIL)
    st = static_checks(html)
    check("static check: no network or embedded audio constructs", st["static_ok"], str(st["forbidden_found"]))
    check("no-network content security policy present", "default-src 'none'" in html)
    print(f"\n{'all page safety tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
