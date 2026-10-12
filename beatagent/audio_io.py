"""Audio file I/O that preserves sample rate and bit depth.

Audio is always held as float32 arrays shaped (n_samples, n_channels).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Lossy inputs have no bit depth; we write their edits as 16-bit WAV
# (documented in DECISIONS.md) instead of re-encoding to MP3.
_LOSSY = {"MPEG_LAYER_I", "MPEG_LAYER_II", "MPEG_LAYER_III", "VORBIS", "OPUS"}


@dataclass
class AudioFormat:
    sr: int
    subtype: str          # e.g. PCM_16, PCM_24, FLOAT
    container: str        # WAV, FLAC, ...
    channels: int

    @property
    def out_subtype(self) -> str:
        return "PCM_16" if self.subtype in _LOSSY else self.subtype

    @property
    def out_container(self) -> str:
        return "WAV" if self.container in ("MP3", "OGG") else self.container

    @property
    def out_ext(self) -> str:
        return {"WAV": ".wav", "FLAC": ".flac", "AIFF": ".aiff"}.get(self.out_container, ".wav")


def load(path: str | Path) -> tuple[np.ndarray, AudioFormat]:
    try:
        info = sf.info(str(path))
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        # Seen on a real FLAC: header says 174 s but the file is 786 KB (an
        # interrupted download). libsndfile's own message ("decoder lost sync")
        # doesn't tell the user that.
        size = Path(path).stat().st_size if Path(path).exists() else 0
        hint = ""
        try:
            dur = sf.info(str(path)).duration
            if dur > 0 and size / dur < 8000:     # < 64 kbit/s: far too small even for MP3
                hint = (f" The file looks truncated or corrupt: {size / 1e6:.1f} MB for a "
                        f"{dur:.0f} s track (an interrupted download?).")
        except Exception:  # noqa: BLE001
            pass
        # libsndfile quotes the full path ("Error opening 'C:\\Users\\...'"): keep only the
        # file name, so an upload's error does not reveal the server's folders.
        why = re.sub(r"Error opening '[^']*'", "Error opening the file", str(e)).rstrip(".")
        raise ValueError(f"Can't decode {Path(path).name}: {why}.{hint}") from e
    fmt = AudioFormat(sr=sr, subtype=info.subtype, container=info.format, channels=audio.shape[1])
    return audio, fmt


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def save(path: str | Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16",
         container: str | None = None, dither: bool = True) -> dict:
    """Write audio. Returns stats (peak, clipped samples) so callers can warn."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    out = audio
    if subtype in ("PCM_16", "PCM_U8", "PCM_S8") and dither:
        # TPDF dither at 1 LSB so long fades into silence don't turn into
        # correlated quantisation distortion.
        lsb = 1.0 / 32768.0
        rng = np.random.default_rng(0)
        out = out + (rng.random(out.shape, dtype=np.float32) - rng.random(out.shape, dtype=np.float32)) * lsb
    clipped = int(np.sum(np.abs(out) > 1.0))
    out = np.clip(out, -1.0, 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out, sr, subtype=subtype, format=container)
    return {"peak": peak, "clipped_samples": clipped}
