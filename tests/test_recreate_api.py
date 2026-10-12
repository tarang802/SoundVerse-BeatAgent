"""Part 3 through the app's HTTP API (the RECREATE panel uses exactly these calls).

    python -m tests.test_recreate_api

SYNTHETIC clips (data/p3, regenerate with `python -m evals.make_p3pack`).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def main():
    logging.disable(logging.WARNING)
    from fastapi.testclient import TestClient
    import app.server as server
    c = TestClient(server.app)
    lib = c.get("/api/p3/library").json()
    check("p3 library lists the clips", [x["name"] for x in lib][:4] == ["P1", "P2", "P3", "P4"], str([x["name"] for x in lib]))
    r = c.post("/api/recreate", json={"path": "data/p3/P2.wav"})
    ok = r.status_code == 200
    check("recreate a clip", ok, r.text[:120] if not ok else f"{r.json()['tempo_bpm']} BPM, {r.json()['tracks']}")
    if ok:
        rid = r.json()["id"]
        page = c.get(f"/api/recreate/{rid}/page.html")
        score = c.get(f"/api/recreate/{rid}/score.json")
        orig = c.get(f"/api/recreate/{rid}/original.wav")
        check("page served as html", page.status_code == 200 and "text/html" in page.headers["content-type"])
        check("page has the no-network CSP", "default-src 'none'" in page.text)
        check("score served and valid json", score.status_code == 200 and json.loads(score.text)["bars"] == 4)
        check("original served for side-by-side listening", orig.status_code == 200 and len(orig.content) > 1000)
        check("unknown file refused", c.get(f"/api/recreate/{rid}/secret.txt").status_code == 404)
        # chat changes, with the offline parser pinned (deterministic, no API quota)
        from beatagent.score import Score
        from beatagent.score_chat import RuleScoreInterpreter, ScoreChat
        server.RECREATE_CHATS[rid] = ScoreChat(Score.load(server.RECREATES[rid] / "score.json"), RuleScoreInterpreter())
        m = c.post(f"/api/recreate/{rid}/chat", json={"text": "mute the drums"}).json()
        page2 = c.get(f"/api/recreate/{rid}/page.html").text
        check("chat 'mute the drums' applied and the page regenerated", m.get("changed") and m["version"] == 2
              and '"name":"drums","kind":"drums","gain_db":0.0,"mute":true' in page2, m.get("text", "")[:80])
        t = c.post(f"/api/recreate/{rid}/chat", json={"text": "play it at 140"}).json()
        check("chat 'play it at 140'", t.get("changed") and abs(t["tempo_bpm"] - 140) < 1e-9, t.get("text", "")[:80])
        bad = c.post(f"/api/recreate/{rid}/chat", json={"text": "mute the vocals"}).json()
        check("chat about a missing track asks instead of guessing", not bad.get("changed"), bad.get("text", "")[:80])
        check("chat on an unknown recreation is a 404", c.post("/api/recreate/nope/chat", json={"text": "x"}).status_code == 404)
    long = c.post("/api/recreate", json={"path": "data/substitute/A1.wav"})
    check("6-minute track refused with a trim hint", long.status_code == 400 and "keep only bars" in long.text,
          long.json().get("detail", "")[:90])
    check("path outside the project refused", c.post("/api/recreate", json={"path": "../../Windows/win.ini"}).status_code == 404)
    check("no input refused", c.post("/api/recreate", json={}).status_code == 400)
    print(f"\n{'all recreate API tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
