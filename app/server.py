"""Beat Agent API + the built React frontend (frontend/dist).

    python run.py app            # http://localhost:8000

API (all JSON unless noted):
  GET  /api/library                         test tracks available on disk
  POST /api/open {path}                     load + analyse a track -> deck state
  POST /api/upload (multipart file)         same, for an uploaded file
  GET  /api/{sid}/state                     deck state (grid, waveform, history)
  GET  /api/{sid}/audio/{state_id}.wav?click=0|1   preview audio (range requests OK)
  GET  /api/{sid}/grid.json | click.wav     Part 1 deliverables for the current version
  POST /api/{sid}/chat {text}               Part 2 chat edit
  POST /api/{sid}/undo | redo
  POST /api/{sid}/export                    write audio + grid + click + EDL to work/<sid>
  POST /api/sync {a, b, a_bar, b_bar, duration}   Part 1 two-track sync
  GET  /api/sync/{id}/{mix|a|b}.wav         sync render + stems
  GET  /api/sync/{id}/report.json
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from beatagent import sync as syncmod  # noqa: E402
from beatagent.agent import BeatAgent  # noqa: E402
from beatagent.audio_io import load, save  # noqa: E402
from beatagent.click import audition  # noqa: E402
from beatagent.llm import make_interpreter  # noqa: E402
from beatagent.edl import EditError  # noqa: E402
from beatagent.session import Session  # noqa: E402

WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
DIST = ROOT / "frontend" / "dist"
app = FastAPI(title="Beat Agent")
SESSIONS: dict[str, BeatAgent] = {}
SYNCS: dict[str, Path] = {}
LOCK = threading.Lock()
PEAK_RATE = 100          # waveform min/max pairs per second sent to the UI


# ------------------------------------------------------------------ helpers
def _peaks_b64(y: np.ndarray, sr: int, rate: int = PEAK_RATE) -> str:
    """Interleaved int8 (min, max) per 1/rate s, base64. ~70 KB for 6 minutes."""
    m = y.mean(axis=1) if y.ndim == 2 else y
    hop = max(1, sr // rate)
    n = len(m) // hop
    blocks = m[: n * hop].reshape(n, hop)
    mm = np.stack([blocks.min(axis=1), blocks.max(axis=1)], axis=1)
    q = np.clip(np.round(mm * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _state(sid: str) -> dict:
    a = SESSIONS[sid]
    s, g = a.s, a.s.grid
    y = s.render()
    return {
        "sid": sid, "version": s.pos, "state_id": s.state_id,
        "can_undo": s.pos > 0, "can_redo": s.pos < len(s.history) - 1,
        "history": s.labels[: s.pos + 1], "file": s.path.name, "sr": s.sr,
        "subtype": s.fmt.subtype, "channels": s.fmt.channels, "interpreter": a.llm.name,
        "summary": g.summary(), "confidence": g.confidence,
        "beats": [round(float(t), 5) for t in g.beats], "positions": [int(p) for p in g.positions],
        "duration": g.duration, "peaks": _peaks_b64(y, s.sr), "peak_rate": PEAK_RATE,
        "meta": {k: v for k, v in g.meta.items() if k != "fit"},
        "tempo_octave": (s.src_grid.meta.get("fit") or {}).get("tempo_octave"),
        "user_octave": s.src_grid.meta.get("user_octave_override"),
    }


def _agent(sid: str) -> BeatAgent:
    a = SESSIONS.get(sid)
    if a is None:
        raise HTTPException(404, "Unknown track session (the server may have restarted). Load the track again.")
    return a


def _open(p: Path) -> dict:
    sid = uuid.uuid4().hex[:8]
    t0 = time.time()
    try:
        s = Session(p, WORK / sid)
    except ValueError as e:                # our own user-facing messages (decode errors, no pulse...)
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Couldn't analyse {p.name}: {e}") from e
    with LOCK:
        SESSIONS[sid] = BeatAgent(s, make_interpreter())
    st = _state(sid)
    st["analysis_s"] = round(time.time() - t0, 2)
    return st


# ------------------------------------------------------------------ library / loading
@app.get("/api/library")
def library():
    items = []
    for pack in ("testpack", "substitute"):
        d = ROOT / "data" / pack
        if not d.exists():
            continue
        desc = {}
        idx = d / "index.json"
        if idx.exists():
            desc = {k: v.get("description", "") for k, v in json.loads(idx.read_text()).get("items", {}).items()}
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".wav", ".mp3", ".flac"):
                items.append({"name": p.stem, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                              "pack": pack, "synthetic": pack == "substitute",
                              "description": desc.get(p.stem, "")})
    return items


class OpenReq(BaseModel):
    path: str


@app.post("/api/open")
def open_path(req: OpenReq):
    p = (ROOT / req.path).resolve() if not Path(req.path).is_absolute() else Path(req.path)
    if not p.exists() or p.suffix.lower() not in (".wav", ".mp3", ".flac"):
        raise HTTPException(404, f"No WAV/MP3/FLAC file at {req.path}")
    return _open(p)


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_upload_name(raw: str | None) -> str:
    """Last path component, with characters Windows forbids replaced. Seen: an upload
    named 'beat: final?.wav' became a 0-byte file 'xxxx_beat' with the audio hidden in
    an NTFS alternate data stream (':' selects a stream). Also drops '../' parts."""
    name = re.split(r"[\\/]", raw or "")[-1]
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = _UNSAFE.sub("_", stem).strip(" .") or "upload"
    ext = _UNSAFE.sub("", ext).strip(" .")
    return f"{stem}.{ext}" if ext else stem


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = _safe_upload_name(file.filename)
    if Path(name).suffix.lower() not in (".wav", ".mp3", ".flac"):
        raise HTTPException(400, "Only WAV, MP3 or FLAC files are supported.")
    d = WORK / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{uuid.uuid4().hex[:6]}_{name}"
    p.write_bytes(await file.read())
    return _open(p)


@app.get("/api/{sid}/state")
def state(sid: str):
    _agent(sid)
    return _state(sid)


# ------------------------------------------------------------------ audio + deliverables
@app.get("/api/{sid}/audio/{state_id}.wav")
def audio(sid: str, state_id: int, click: int = 0):
    """Preview audio for the CURRENT version, cached as a file so the browser
    gets HTTP range support (seeking). Keyed by the state's unique id, not the
    version number, which is reused after undo + a new edit."""
    s = _agent(sid).s
    p = WORK / sid / f"_preview_s{s.state_id}_{'click' if click else 'plain'}.wav"
    if not p.exists():
        y = s.render()
        if click:
            y = audition(y, s.sr, s.grid)
        sf.write(p, np.clip(y, -1, 1), s.sr, subtype="PCM_16")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/{sid}/grid.json")
def grid_json(sid: str):
    s = _agent(sid).s
    p = WORK / sid / f"{s.path.stem}_v{s.pos}.grid.json"
    s.grid.save(p)
    return FileResponse(p, media_type="application/json", filename=p.name)


@app.get("/api/{sid}/click.wav")
def click_wav(sid: str):
    s = _agent(sid).s
    p = WORK / sid / f"{s.path.stem}_v{s.pos}.click.wav"
    save(p, audition(s.render(), s.sr, s.grid), s.sr, "PCM_16", "WAV")
    return FileResponse(p, media_type="audio/wav", filename=p.name)


# ------------------------------------------------------------------ Part 2: chat edits
class ChatReq(BaseModel):
    text: str
    partner: str | None = None     # the other deck's session id, for "sync ..." commands


@app.post("/api/{sid}/chat")
def chat(sid: str, req: ChatReq):
    a = _agent(sid)
    a.partner = SESSIONS[req.partner].s if req.partner in SESSIONS and req.partner != sid else None
    r = a.handle(req.text)
    out = {"reply": r.to_dict(), "state": _state(sid)}
    if r.action == "sync" and r.ok and a.last_sync:
        d = Path(a.last_sync["dir"])
        sync_id = uuid.uuid4().hex[:8]
        SYNCS[sync_id] = d
        (d / "report.json").write_text(json.dumps(a.last_sync, indent=1))
        out["sync_id"] = sync_id
    return out


@app.post("/api/{sid}/undo")
def undo(sid: str):
    return chat(sid, ChatReq(text="undo"))


@app.post("/api/{sid}/redo")
def redo(sid: str):
    return chat(sid, ChatReq(text="redo"))


class OctaveReq(BaseModel):
    factor: float             # 2 = double the BPM, 0.5 = halve it


@app.post("/api/{sid}/octave")
def octave(sid: str, req: OctaveReq):
    a = _agent(sid)
    try:
        a.s.set_octave(req.factor)
    except EditError as e:
        raise HTTPException(400, str(e)) from e
    return _state(sid)


@app.post("/api/{sid}/export")
def export(sid: str):
    paths = _agent(sid).s.export()
    return {k: (str(Path(v).relative_to(ROOT)) if isinstance(v, str) else v) for k, v in paths.items()}


# ------------------------------------------------------------------ Part 1: sync
class SyncReq(BaseModel):
    a: str                    # deck A session id (master)
    b: str                    # deck B session id
    a_bar: int = 33
    b_bar: int = 1
    duration: float = 60.0


@app.post("/api/sync")
def do_sync(req: SyncReq):
    sa, sb = _agent(req.a).s, _agent(req.b).s
    try:
        plan = syncmod.plan_sync(sa.grid, sb.grid, req.a_bar, req.b_bar, req.duration)
    except syncmod.SyncError as e:
        raise HTTPException(400, str(e)) from e
    t0 = time.time()
    ya, yb = sa.render(), sb.render()        # current (possibly edited) versions of each deck
    r = syncmod.render(ya, sa.sr, yb, sb.sr, plan, sa.grid, syncmod.matched_b_grid(plan, sb.grid))
    m = syncmod.measure_alignment(r["a"], r["b"], r["sr"], sa.grid, plan)
    sync_id = uuid.uuid4().hex[:8]
    d = WORK / "sync" / sync_id
    d.mkdir(parents=True)
    for k in ("a", "b", "mix"):
        save(d / f"{k}.wav", r[k], r["sr"], "PCM_16", "WAV")
    a_beats = sa.grid.beats[(sa.grid.beats >= plan.a_start - 1e-6) & (sa.grid.beats < plan.a_start + plan.duration)]
    a_pos = sa.grid.positions[(sa.grid.beats >= plan.a_start - 1e-6) & (sa.grid.beats < plan.a_start + plan.duration)]
    report = {"a": sa.path.name, "b": sb.path.name, "method": r["method"],
              "render_s": round(time.time() - t0, 2), "plan": plan.to_dict(), "alignment": m}
    (d / "report.json").write_text(json.dumps(report, indent=1))
    SYNCS[sync_id] = d
    return {"id": sync_id, **report,
            "beats": [round(float(t - plan.a_start), 5) for t in a_beats],
            "positions": [int(p) for p in a_pos],
            "peaks": {k: _peaks_b64(r[k], r["sr"]) for k in ("a", "b", "mix")}, "peak_rate": PEAK_RATE}


@app.get("/api/sync/{sync_id}/{which}")
def sync_file(sync_id: str, which: str):
    d = SYNCS.get(sync_id)
    if d is None or which not in ("a.wav", "b.wav", "mix.wav", "report.json"):
        raise HTTPException(404, "Unknown sync render.")
    return FileResponse(d / which, filename=f"sync_{sync_id}_{which}")


# ------------------------------------------------------------------ Part 3: audio -> Web Audio page
RECREATES: dict[str, Path] = {}
MAX_RECREATE_S = 40.0          # the brief's clips are 2-16 bars


@app.get("/api/p3/library")
def p3_library():
    d = ROOT / "data" / "p3"
    meta = json.loads((d / "pack.json").read_text(encoding="utf-8")) if (d / "pack.json").exists() else {}
    return [{"name": p.stem, "path": str(p.relative_to(ROOT)).replace("\\", "/"), "synthetic": True,
             "description": meta.get(p.stem, {}).get("description", "")}
            for p in sorted(d.glob("*.wav"))] if d.exists() else []


class RecreateReq(BaseModel):
    path: str | None = None     # a clip file (library / data/p3)
    sid: str | None = None      # or a deck's CURRENT (possibly edited) version


@app.post("/api/recreate")
def recreate(req: RecreateReq):
    from beatagent.transcribe import transcribe
    from beatagent.webaudio import write_page
    if req.sid:
        s = _agent(req.sid).s
        audio, sr, name = s.render(), s.sr, f"{s.path.stem} (deck v{s.pos})"
    elif req.path:
        p = (ROOT / req.path).resolve()
        if ROOT not in p.parents or not p.exists() or p.suffix.lower() not in (".wav", ".mp3", ".flac"):
            raise HTTPException(404, f"No WAV/MP3/FLAC clip at {req.path}")
        try:
            audio, fmt = load(p)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        sr, name = fmt.sr, p.stem
    else:
        raise HTTPException(400, "Give a clip path or a deck id.")
    dur = len(audio) / sr
    if dur > MAX_RECREATE_S:
        raise HTTPException(400, f"That clip is {dur:.0f} s; recreation takes a short loop (2-16 bars, up to "
                                 f"{MAX_RECREATE_S:.0f} s). Trim the deck first, e.g. chat 'keep only bars 1-8'.")
    try:
        score = transcribe(audio, sr, name)
    except ValueError as e:                       # no pulse / too short
        raise HTTPException(400, f"Couldn't analyse that clip: {e}") from e
    rid = uuid.uuid4().hex[:8]
    d = WORK / "recreate" / rid
    d.mkdir(parents=True)
    score.save(d / "score.json")
    write_page(score, d / "page.html", name)
    save(d / "original.wav", audio, sr, "PCM_16", "WAV")
    RECREATES[rid] = d
    return {"id": rid, "name": name, "tempo_bpm": score.tempo_bpm, "time_signature": score.time_signature,
            "bars": score.bars, "loop_s": round(score.loop_seconds, 3), "analysis": score.analysis,
            "tracks": [{"name": t.name, "kind": t.kind, "notes": len(t.notes)} for t in score.tracks]}


RECREATE_CHATS: dict = {}


class ScoreChatReq(BaseModel):
    text: str


@app.post("/api/recreate/{rid}/chat")
def recreate_chat(rid: str, req: ScoreChatReq):
    """Chat changes to a recreation ("mute the drums", "play it at 140"): typed score ops,
    validated and applied by code; the page is regenerated from the new score."""
    from beatagent.score import Score
    from beatagent.score_chat import ScoreChat
    from beatagent.webaudio import write_page
    d = RECREATES.get(rid)
    if d is None:
        raise HTTPException(404, "Unknown recreation.")
    with LOCK:
        chat = RECREATE_CHATS.get(rid) or RECREATE_CHATS.setdefault(rid, ScoreChat(Score.load(d / "score.json")))
    r = chat.handle(req.text)
    if r.get("changed"):
        chat.score.save(d / "score.json")
        write_page(chat.score, d / "page.html")
    s = chat.score
    return {**r, "version": len(chat.history), "tempo_bpm": s.tempo_bpm,
            "tracks": [{"name": t.name, "kind": t.kind, "notes": len(t.notes), "mute": t.mute,
                        "gain_db": t.gain_db} for t in s.tracks]}


@app.get("/api/recreate/{rid}/{which}")
def recreate_file(rid: str, which: str):
    d = RECREATES.get(rid)
    if d is None or which not in ("page.html", "score.json", "original.wav"):
        raise HTTPException(404, "Unknown recreation.")
    return FileResponse(d / which, filename=f"recreate_{rid}_{which}" if which != "page.html" else None)


# ------------------------------------------------------------------ frontend
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:
    @app.get("/")
    def no_ui():
        return {"error": "frontend not built", "fix": "cd frontend && npm install && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
