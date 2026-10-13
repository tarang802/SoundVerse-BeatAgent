"""Render a Part 3 page offline in a headless browser and measure it.

measure_render(score, out_dir, original_audio, sr) writes page.html and render.wav to
out_dir and returns:
  static_ok        the HTML contains no network / embedded-audio constructs
  network_requests requests the page made while loading and rendering (must be 0)
  play_stop        clicking Play schedules notes; Stop stops
  onset_F, median_abs_ms, max_abs_ms
                   onsets of the offline render (TWO loop passes, so the loop seam is
                   included) vs the original clip's onsets on the same timeline,
                   matched within +-50 ms (mir_eval.onset)
Browser: the installed Microsoft Edge or Google Chrome via Playwright (no download);
Playwright's own Chromium if neither exists.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import mir_eval
import numpy as np

from beatagent.core.audio_io import save, to_mono
from beatagent.recreate.score import Score
from beatagent.recreate.webaudio import write_page

FORBIDDEN = [r"https?://", r"<audio", r"data:audio", r"\bfetch\(", r"XMLHttpRequest", r"<link\b",
             r"<[^>]*\ssrc\s*=", r"import\(", r"WebSocket", r"navigator\.sendBeacon"]


def static_checks(html: str) -> dict:
    hits = [p for p in FORBIDDEN if re.search(p, html)]
    return {"static_ok": not hits, "forbidden_found": hits, "bytes": len(html.encode("utf-8"))}


def _launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw), kw.get("channel", "playwright-chromium")
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("No browser for Playwright: install Edge/Chrome or run `playwright install chromium`")


def render_page(page_path: Path, seconds: float) -> tuple[np.ndarray, int, dict]:
    from playwright.sync_api import sync_playwright
    info: dict = {}
    with sync_playwright() as p:
        browser, info["browser"] = _launch(p)
        page = browser.new_page()
        reqs = []
        page.on("request", lambda r: reqs.append(r.url))
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(page_path.resolve().as_uri())
        res = page.evaluate("s => window.renderOffline(s)", seconds)
        # play / stop through the real buttons (a click is a user gesture: audio may start)
        page.click("#play")
        page.wait_for_timeout(700)
        during = page.evaluate("() => ({...window.__beatagent})")
        page.click("#stop")
        page.wait_for_timeout(200)
        after = page.evaluate("() => ({...window.__beatagent})")
        browser.close()
    own = page_path.resolve().as_uri()
    info["network_requests"] = [u for u in reqs if u != own]
    info["page_errors"] = errors
    info["play_stop"] = {"playing_after_play": during.get("playing"), "loops_scheduled": during.get("loopsScheduled"),
                         "notes_scheduled": during.get("notesScheduled"), "playing_after_stop": after.get("playing"),
                         "ok": bool(during.get("playing") and during.get("notesScheduled", 0) > 0 and not after.get("playing"))}
    pcm = np.frombuffer(base64.b64decode(res["pcm16"]), dtype=np.int16).reshape(-1, res["channels"])
    return pcm.astype(np.float32) / 32767.0, int(res["sampleRate"]), info


def onsets(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    # hop 128 (2.9 ms at 44.1 kHz): the default 512 quantised both sides to the same
    # 11.6 ms frame and reported a meaningless median of 0.0 ms
    return librosa.onset.onset_detect(y=to_mono(y).astype(np.float32), sr=sr, units="time",
                                      hop_length=128, backtrack=False)


def measure_render(score: Score, out_dir: Path, original: np.ndarray, sr: int) -> dict:
    out_dir = Path(out_dir)
    page = out_dir / "page.html"
    write_page(score, page)
    res = static_checks(page.read_text(encoding="utf-8"))
    L = score.loop_seconds
    y, rsr, info = render_page(page, 2 * L)
    save(out_dir / "render.wav", y, rsr, "PCM_16", "WAV")
    res.update(info)
    res["network_ok"] = not info["network_requests"]
    # original on the render's timeline: start at the first downbeat, two passes
    o = to_mono(original)
    if rsr != sr:
        import soxr
        o = soxr.resample(o, sr, rsr)
    start = int(round(score.source.get("loop_start_s", 0.0) * rsr))
    n = int(round(L * rsr))
    ref_audio = np.tile(np.roll(o[:n] if len(o) >= n else np.pad(o, (0, n - len(o))), -start), 2)
    ref_on, est_on = onsets(ref_audio, rsr), onsets(y, rsr)
    f, p, r = mir_eval.onset.f_measure(ref_on, est_on, window=0.05)
    pairs = mir_eval.util.match_events(ref_on, est_on, 0.05)
    offs = np.array([(est_on[j] - ref_on[i]) * 1000 for i, j in pairs])
    res.update({"render_seconds": round(len(y) / rsr, 3), "loop_seconds": round(L, 4),
                "onsets_ref": int(len(ref_on)), "onsets_render": int(len(est_on)),
                "onset_F": round(float(f), 4), "onset_P": round(float(p), 4), "onset_R": round(float(r), 4),
                "median_abs_ms": round(float(np.median(np.abs(offs))), 2) if len(offs) else None,
                "max_abs_ms": round(float(np.max(np.abs(offs))), 2) if len(offs) else None})
    return res
