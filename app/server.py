"""Minimal web app: load a track, see the waveform + grid, chat edits, play/export.

    python -m app.server            # http://localhost:8000
"""
from __future__ import annotations

import io
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from beatagent.agent import BeatAgent  # noqa: E402
from beatagent.click import audition  # noqa: E402
from beatagent.llm import make_interpreter  # noqa: E402
from beatagent.session import Session  # noqa: E402

WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
app = FastAPI(title="Beat Agent")
SESSIONS: dict[str, BeatAgent] = {}
LOCK = threading.Lock()


def _peaks(y: np.ndarray, n: int = 1600) -> list:
    m = y.mean(axis=1) if y.ndim == 2 else y
    if len(m) < n:
        n = max(1, len(m))
    b = np.array_split(m, n)
    return [[round(float(x.min()), 4), round(float(x.max()), 4)] for x in b]


def _state(sid: str) -> dict:
    a = SESSIONS[sid]
    s = a.s
    g = s.grid
    y = s.render()
    return {
        "sid": sid, "version": s.pos, "state_id": s.state_id, "can_undo": s.pos > 0, "can_redo": s.pos < len(s.history) - 1,
        "history": s.labels[: s.pos + 1], "file": s.path.name, "sr": s.sr, "subtype": s.fmt.subtype,
        "interpreter": a.llm.name, "summary": g.summary(), "confidence": g.confidence,
        "beats": [round(float(t), 4) for t in g.beats], "positions": [int(p) for p in g.positions],
        "duration": g.duration, "peaks": _peaks(y), "meta": {k: v for k, v in g.meta.items() if k != "fit"},
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    sid = uuid.uuid4().hex[:8]
    d = WORK / sid
    d.mkdir(parents=True)
    p = d / Path(file.filename).name
    p.write_bytes(await file.read())
    return _open(sid, p)


class OpenReq(BaseModel):
    path: str


@app.post("/api/open")
def open_path(req: OpenReq):
    p = Path(req.path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise HTTPException(404, f"No such file: {req.path}")
    sid = uuid.uuid4().hex[:8]
    return _open(sid, p)


def _open(sid: str, p: Path) -> dict:
    t0 = time.time()
    try:
        s = Session(p, WORK / sid)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Couldn't analyse {p.name}: {e}") from e
    with LOCK:
        SESSIONS[sid] = BeatAgent(s, make_interpreter())
    st = _state(sid)
    st["analysis_s"] = round(time.time() - t0, 2)
    return st


class ChatReq(BaseModel):
    text: str


@app.post("/api/{sid}/chat")
def chat(sid: str, req: ChatReq):
    a = SESSIONS.get(sid) or _404()
    r = a.handle(req.text)
    return {"reply": r.to_dict(), "state": _state(sid)}


@app.post("/api/{sid}/undo")
def undo(sid: str):
    return chat(sid, ChatReq(text="undo"))


@app.post("/api/{sid}/redo")
def redo(sid: str):
    return chat(sid, ChatReq(text="redo"))


@app.get("/api/{sid}/state")
def state(sid: str):
    SESSIONS.get(sid) or _404()
    return _state(sid)


def _wav(y: np.ndarray, sr: int) -> Response:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.get("/api/{sid}/audio/{version}.wav")
def audio(sid: str, version: int, click: int = 0):
    """Preview audio for the CURRENT version, cached as a file so the browser
    gets HTTP range support (seeking). Keyed by the state's unique id, not
    the version number, which is reused after undo + a new edit."""
    a = SESSIONS.get(sid) or _404()
    s = a.s
    p = WORK / sid / f"_preview_s{s.state_id}_{'click' if click else 'plain'}.wav"
    if not p.exists():
        y = s.render()
        if click:
            y = audition(y, s.sr, s.grid)
        sf.write(p, np.clip(y, -1, 1), s.sr, subtype="PCM_16")
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/{sid}/export")
def export(sid: str):
    a = SESSIONS.get(sid) or _404()
    paths = a.s.export()
    return {k: (str(Path(v).relative_to(ROOT)) if isinstance(v, str) else v) for k, v in paths.items()}


@app.get("/api/{sid}/file")
def file(sid: str, path: str):
    p = (ROOT / path).resolve()
    if not str(p).startswith(str((WORK / sid).resolve())):
        raise HTTPException(403, "outside session dir")
    return FileResponse(p, filename=p.name)


def _404():
    raise HTTPException(404, "Unknown session (the server may have restarted). Load the track again.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
