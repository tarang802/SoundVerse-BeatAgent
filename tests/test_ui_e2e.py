"""End-to-end UI test: the real app in headless Edge/Chrome, the Loom demo path.

    python -m tests.test_ui_e2e

Starts the server on a spare port with BEAT_AGENT_LLM=offline (deterministic: the
offline parser, no API key, no quota), then drives the UI: load A1 and A7, check the
grid and click downloads, sync at A's bar 33, chat edits with pushback and undo,
recreate P2 and change it by chat. Needs the synthetic packs (`python run.py testpack`
and `p3pack`) and Edge or Chrome. Takes a few minutes (two full analyses).
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        FAILS.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    env = {**os.environ, "PORT": str(port), "BEAT_AGENT_LLM": "offline", "PYTHONIOENCODING": "utf-8"}
    srv = subprocess.Popen([sys.executable, "-m", "app.server"], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/api/library", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        run(base)
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    print(f"\n{'all UI end-to-end checks passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


def run(base: str):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = None
        for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(headless=True, **kw)
                break
            except Exception:  # noqa: BLE001
                continue
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + "/")
        deck_a, deck_b = page.locator(".deck").nth(0), page.locator(".deck").nth(1)
        page.wait_for_selector(".deck-load select option[value='data/substitute/A1.wav']", state="attached")

        # Part 1: grid + click
        deck_a.locator(".deck-load select").select_option("data/substitute/A1.wav")
        deck_a.locator(".bpm-val").wait_for(timeout=180000)
        check("deck A analyses A1", deck_a.locator(".bpm-val").text_content() == "124.00",
              deck_a.locator(".bpm-val").text_content())
        grid_href = deck_a.locator("a:has-text('GRID JSON')").get_attribute("href")
        g = page.request.get(base + grid_href).json()
        check("grid JSON download has beats, bars and positions",
              len(g.get("beats", [])) > 700 and all(k in g["beats"][0] for k in ("time", "bar", "position")),
              f"{len(g.get('beats', []))} beats")
        click_href = deck_a.locator("a:has-text('CLICK WAV')").get_attribute("href")
        ck = page.request.get(base + click_href)
        check("click audition download is audio", ck.ok and ck.headers.get("content-type", "").startswith("audio")
              and len(ck.body()) > 1_000_000, f"{len(ck.body())} bytes")

        # Part 1: sync A7 onto A1 at bar 33
        deck_b.locator(".deck-load select").select_option("data/substitute/A7.wav")
        deck_b.locator(".bpm-val").wait_for(timeout=180000)
        check("deck B analyses A7", deck_b.locator(".bpm-val").text_content() == "128.00")
        page.locator(".sync-controls label:has-text('A bar') input").fill("33")
        page.locator("section.sync .sync-btn").click()
        page.locator(".sync-plan").wait_for(timeout=180000)
        plan = page.locator(".sync-plan").text_content()
        check("sync plan: B bar 1 -> A bar 33, x1.032258", "B 1 → A 33" in plan and "×1.032258" in plan, plan[:120])

        # Part 2: chat edits on deck A (offline parser), pushback, undo
        deck_a.locator("details.edit summary").click()
        log = deck_a.locator(".log")

        def say(text, expect, timeout=120000):
            n = log.locator(".msg").count()
            deck_a.locator("form.chat input").fill(text)
            deck_a.locator("form.chat button:has-text('SEND')").click()
            page.wait_for_function("([el, n]) => el.querySelectorAll('.msg').length >= n + 2",
                                   arg=[log.element_handle(), n], timeout=timeout)
            reply = log.locator(".msg").last.text_content()
            check(f"chat '{text}'", expect in reply, reply[:110])

        say("remove bar -2", "negative")
        say("cut beat 16-32 and repeat it twice", "repeat beats 16-32")
        say("undo", "undid")

        # Part 3: recreate P2 and change it by chat
        page.locator(".recreate select").select_option("data/p3/P2.wav")
        page.locator(".recreate button:has-text('RECREATE')").click()
        ready = "() => /page loaded/.test([...document.querySelectorAll('.recreate .deck-role')].map(d => d.textContent).join(' '))"
        page.wait_for_function(ready, timeout=180000)
        check("recreation page loaded in the panel", True)
        frame = page.frame_locator(".recreate iframe")
        frame.locator("#play").click()
        page.wait_for_timeout(800)
        check("recreation plays (position advances)", (frame.locator("#pos").text_content() or "").startswith("bar"))
        frame.locator("#stop").click()
        page.locator(".recreate-chat input").fill("mute the drums")
        page.locator(".recreate-chat input").press("Enter")
        page.wait_for_selector(".recreate-chat p", timeout=60000)
        page.wait_for_function(ready, timeout=60000)
        check("recreation chat 'mute the drums'", "drums: muted" in page.locator(".recreate-chat p").last.text_content())
        check("no JavaScript errors in the app", not errors, str(errors)[:120])
        browser.close()


if __name__ == "__main__":
    main()
