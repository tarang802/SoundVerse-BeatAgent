"""Pitch-preserving time-stretch via Rubber Band (pylibrb bundles librubberband).

We use pylibrb rather than pyrubberband because pyrubberband shells out to the
`rubberband` CLI, which isn't on PyPI and whose GitHub release downloads were
unreachable from the dev machine. pylibrb is a 2 MB wheel with the library
compiled in, and it exposes keyframe maps for variable-tempo warping.
"""
from __future__ import annotations

import numpy as np
import pylibrb

CHUNK = 1 << 16
_O = pylibrb.Option
# Engine choice, measured (evals/smoke_test + notes in DECISIONS.md):
#   R3 "finer": pitch exact (+-0.0 cents on 110/440/1000 Hz sines), but kick
#               onsets smeared/shifted by 7-25 ms.
#   R2 "crisp": kicks within ~1.5 ms, but a sustained 110 Hz sine came out
#               ~60 cents off -- unacceptable for basslines.
# Pitch preservation is a hard requirement, so plain stretching uses R3; the
# sync renderer adds its own transient handling. (pylibrb 0.1.2's
# set_keyframe_map rejects every argument on Windows, so warp markers are
# not available.)
DEFAULT_OPTS = _O.PROCESS_OFFLINE | _O.ENGINE_FINER | _O.CHANNELS_TOGETHER
R2_CRISP = _O.PROCESS_OFFLINE | _O.ENGINE_FASTER | _O.TRANSIENTS_CRISP | _O.CHANNELS_TOGETHER


def _run(audio: np.ndarray, sr: int, time_ratio: float,
         keyframes: dict[int, int] | None = None, opts=None) -> np.ndarray:
    """audio: (n, ch) float32. time_ratio > 1 makes it longer (slower)."""
    n, ch = audio.shape
    opts = DEFAULT_OPTS if opts is None else opts
    st = pylibrb.RubberBandStretcher(sample_rate=sr, channels=ch, options=opts,
                                     initial_time_ratio=time_ratio)
    rb_in = np.ascontiguousarray(pylibrb.reorder_to_rb(audio.astype(np.float32), samples_axis=0))
    # the library caps samples per call, so study/process in blocks
    block = min(CHUNK, st.get_process_size_limit())
    sax = pylibrb.SAMPLES_AXIS

    def blocks():
        for s in range(0, n, block):
            yield np.take(rb_in, np.arange(s, min(n, s + block)), axis=sax), s + block >= n

    for chunk, last in blocks():
        st.study(np.ascontiguousarray(chunk), final=last)
    if keyframes:
        st.set_keyframe_map({int(k): int(v) for k, v in keyframes.items()})
    st.set_max_process_size(block)
    outs = []
    for chunk, last in blocks():
        st.process(np.ascontiguousarray(chunk), final=last)
        if st.available() > 0:
            outs.append(st.retrieve_available())
    while not st.is_done() or st.available() > 0:
        if st.available() > 0:
            outs.append(st.retrieve_available())
        elif st.is_done():
            break
    out = np.concatenate(outs, axis=sax)
    return np.ascontiguousarray(pylibrb.reorder_from_rb(out, [None, -1])).astype(np.float32)


def stretch(audio: np.ndarray, sr: int, time_ratio: float) -> np.ndarray:
    """Constant-ratio stretch. Output time t_out = t_in * time_ratio."""
    if abs(time_ratio - 1.0) < 1e-9:
        return audio.copy()
    return _run(audio, sr, time_ratio)


def warp(audio: np.ndarray, sr: int, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Variable stretch so that src_times[i] lands on dst_times[i] (both seconds,
    increasing, src_times[0] and dst_times[0] taken as the anchors at 0)."""
    src = np.asarray(src_times, float)
    dst = np.asarray(dst_times, float)
    total_in = audio.shape[0] / sr
    # overall ratio from the anchor span, extended to the file end
    ratio = (dst[-1] - dst[0]) / (src[-1] - src[0])
    kf = {int(round(s * sr)): int(round(d * sr)) for s, d in zip(src, dst) if 0 < s < total_in}
    return _run(audio, sr, ratio, kf)
