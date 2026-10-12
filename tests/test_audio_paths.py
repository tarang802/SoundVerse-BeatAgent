"""Audio files with awkward names and formats, through the app's own loader and upload.

    python -m tests.test_audio_paths

Real music libraries are full of "01. Artist - Track [E] (1).flac", ampersands,
quotes and non-Latin titles. Checks: the loader decodes WAV (16/24-bit, float),
FLAC and MP3 under such names; a whole session (analyse, edit, export) works on a
Unicode path; uploads with such names -- including characters Windows forbids in
file names and a path-traversal attempt -- are accepted safely; a corrupt file
gives a readable error. SYNTHETIC audio (evals/synth.fixture_120).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beatagent.audio_io import load  # noqa: E402
from evals.synth import SR, fixture_120  # noqa: E402

WORK = ROOT / "work" / "test_audio_paths"
FAILS: list[str] = []
NAMES = ["plain", "with space", "01. Artist - Track [E] (1)", "R&B 'quotes' #1 %20",
         "ünïcødé 日本語 музыка"]
FORMATS = [("WAV", "PCM_16", ".wav"), ("WAV", "PCM_24", ".wav"), ("WAV", "FLOAT", ".wav"),
           ("FLAC", "PCM_16", ".flac"), ("MP3", "MPEG_LAYER_III", ".mp3")]


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def fixture(seconds=12.0):
    audio, _, _ = fixture_120(duration=seconds)
    return np.stack([audio, audio * 0.9], axis=1).astype(np.float32)      # stereo


def test_loader(x):
    print("loader: names x formats")
    for name in NAMES:
        for container, subtype, ext in FORMATS:
            p = WORK / "lib" / name / f"{name}{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), x, SR, subtype=subtype, format=container)
            try:
                y, fmt = load(p)
                # MP3 adds encoder delay/padding, so only approximately the same length
                tol = 0.15 if container == "MP3" else 1e-6
                ok = fmt.sr == SR and y.shape[1] == 2 and abs(len(y) - len(x)) / SR <= tol
                check(f"{container}/{subtype} '{name}'", ok, f"{len(y) / SR:.3f}s {fmt.subtype}")
            except Exception as e:  # noqa: BLE001
                check(f"{container}/{subtype} '{name}'", False, f"{type(e).__name__}: {e}")


def test_session(x):
    print("session on a unicode path: analyse -> edit -> export")
    from beatagent.edl import Range, Remove
    from beatagent.session import Session
    p = WORK / "lib" / "ünïcødé 日本語 музыка" / "session [E] & co.flac"
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), x, SR, subtype="PCM_24", format="FLAC")
    s = Session(p, WORK / "sess ünï")
    t = s.apply([Remove(range=Range(unit="bars", start=2, end=2))])
    out = s.export()
    files = [Path(v) for v in (out.values() if isinstance(out, dict) else [out]) if isinstance(v, (str, Path))]
    exists = [f for f in files if f.exists()]
    check("edit applied", t.ok, t.error or "")
    check("export wrote files under a unicode work dir", len(exists) > 0, ", ".join(f.name for f in exists)[:120])
    audio_out = [f for f in exists if f.suffix.lower() in (".flac", ".wav")]
    if audio_out:
        info = sf.info(str(audio_out[0]))
        check("export keeps sample rate and bit depth", info.samplerate == SR and info.subtype == "PCM_24",
              f"{info.samplerate} Hz {info.subtype}")


def test_upload(x):
    print("upload endpoint: awkward and hostile file names")
    from fastapi.testclient import TestClient
    import app.server as server
    c = TestClient(server.app)
    src = WORK / "upload_src.wav"
    sf.write(str(src), x, SR, subtype="PCM_16")
    data = src.read_bytes()
    up = server.WORK / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    for fname in ["my track [E] (1).wav", "ünï 日本 & co.wav", "beat: final?*<>|\".wav",
                  "../../evil.wav", "..\\..\\evil2.wav", "CON.wav", "   .wav"]:
        before = set(up.iterdir())
        r = c.post("/api/upload", files={"file": (fname, data, "audio/wav")})
        new = [f for f in set(up.iterdir()) - before]
        # exactly one new, ORDINARY file holding all the bytes, with a portable name, inside uploads/
        # (seen: "beat: final?.wav" made a 0-byte "xxxx_beat" with the audio hidden in an NTFS stream)
        ok_file = (len(new) == 1 and new[0].is_file() and new[0].stat().st_size == len(data)
                   and not any(ch in new[0].name for ch in '<>:"/\\|?*') and up in new[0].resolve().parents
                   and new[0].suffix.lower() == ".wav")
        check(f"upload '{fname}'", r.status_code == 200 and ok_file,
              f"HTTP {r.status_code} saved as {[f.name for f in new]} "
              f"{[f.stat().st_size for f in new]}B {r.text[:80] if r.status_code != 200 else ''}")
    r = c.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    check("unsupported extension refused with 400", r.status_code == 400, r.text[:80])
    bad = WORK / "corrupt.flac"
    bad.write_bytes(b"fLaC" + b"\x00" * 2000)
    r = c.post("/api/upload", files={"file": ("corrupt.flac", bad.read_bytes(), "audio/flac")})
    check("corrupt file refused with a readable 400", r.status_code == 400 and "decode" in r.text.lower(),
          r.text[:100])
    leak = str(ROOT).replace("\\", "\\\\")
    check("error does not reveal the server's folders", str(ROOT) not in r.text and leak not in r.text
          and "Users" not in r.text, r.text[:100])


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    x = fixture()
    test_loader(x)
    test_session(x)
    test_upload(x)
    print(f"\n{'all audio path tests passed' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
