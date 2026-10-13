import { useEffect, useRef, type MouseEvent } from "react";
import type { Peaks } from "../peaks";

interface Props {
  peaks: Peaks;
  duration: number;
  beats: number[];
  positions: number[];
  time: number;                  // playhead, seconds
  color: string;                 // deck accent
  mode: "zoom" | "overview";
  windowSec?: number;            // zoom mode: visible span
  height?: number;
  markers?: { t: number; label: string; color: string }[];
  onSeek?: (t: number) => void;
}

/** Canvas waveform with the beat grid drawn on top.
 * zoom: playhead fixed in the centre, waveform scrolls (DJ-deck style).
 * overview: whole track, played part highlighted, click to seek. */
export function Waveform({ peaks, duration, beats, positions, time, color, mode, windowSec = 6,
                           height = 120, markers = [], onSeek }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.floor(c.clientWidth * dpr));
    const H = Math.floor(height * dpr);
    if (c.width !== W || c.height !== H) {
      c.width = W;
      c.height = H;
    }
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    // visible time range
    const t0 = mode === "zoom" ? time - windowSec / 2 : 0;
    const t1 = mode === "zoom" ? time + windowSec / 2 : duration;
    const xOf = (t: number) => ((t - t0) / (t1 - t0)) * W;
    const mid = H / 2;

    // centre line
    ctx.fillStyle = "#17151f";
    ctx.fillRect(0, mid, W, Math.max(1, dpr));

    // waveform: one min/max bar per device pixel
    const { data, rate, count } = peaks;
    for (let x = 0; x < W; x++) {
      const ta = t0 + (x / W) * (t1 - t0);
      const tb = t0 + ((x + 1) / W) * (t1 - t0);
      let i0 = Math.floor(ta * rate);
      let i1 = Math.max(i0 + 1, Math.floor(tb * rate));
      if (i1 <= 0 || i0 >= count) continue;
      i0 = Math.max(0, i0);
      i1 = Math.min(count, i1);
      let lo = 127;
      let hi = -127;
      for (let i = i0; i < i1; i++) {
        if (data[2 * i] < lo) lo = data[2 * i];
        if (data[2 * i + 1] > hi) hi = data[2 * i + 1];
      }
      const yHi = mid - (hi / 127) * (mid - 2 * dpr);
      const yLo = mid - (lo / 127) * (mid - 2 * dpr);
      const played = mode === "overview" && ta < time;
      ctx.fillStyle = played ? color : mode === "overview" ? "#34303f" : color;
      ctx.globalAlpha = mode === "zoom" ? 0.85 : 1;
      ctx.fillRect(x, yHi, 1, Math.max(dpr, yLo - yHi));
    }
    ctx.globalAlpha = 1;

    // beat grid
    let bar = 0;
    const barEvery = mode === "overview" ? Math.max(1, Math.ceil(beats.length / 4 / (W / (40 * dpr)))) : 1;
    ctx.font = `${10 * dpr}px ui-monospace, Consolas, monospace`;
    for (let i = 0; i < beats.length; i++) {
      const down = positions[i] === 1;
      if (down) bar++;
      const t = beats[i];
      if (t < t0 || t > t1) continue;
      const x = Math.round(xOf(t)) + 0.5;
      if (mode === "zoom") {
        ctx.strokeStyle = down ? "#ff5d6c" : "rgba(255,255,255,0.28)";
        ctx.lineWidth = (down ? 2 : 1) * dpr;
        ctx.beginPath();
        ctx.moveTo(x, down ? 0 : H * 0.15);
        ctx.lineTo(x, down ? H : H * 0.85);
        ctx.stroke();
        if (down) {
          ctx.fillStyle = "#ff5d6c";
          ctx.fillText(String(bar), x + 3 * dpr, 11 * dpr);
        }
      } else if (down && (bar - 1) % barEvery === 0) {
        ctx.strokeStyle = "rgba(255,93,108,0.55)";
        ctx.lineWidth = dpr;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H * 0.22);
        ctx.stroke();
      }
    }

    // extra markers (e.g. sync start point)
    for (const m of markers) {
      if (m.t < t0 || m.t > t1) continue;
      const x = Math.round(xOf(m.t)) + 0.5;
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.fillStyle = m.color;
      ctx.fillText(m.label, x + 3 * dpr, H - 4 * dpr);
    }

    // playhead
    const px = mode === "zoom" ? W / 2 : xOf(time);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(Math.round(px) - dpr, 0, 2 * dpr, H);
  }, [peaks, duration, beats, positions, time, color, mode, windowSec, height, markers]);

  const seek = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const r = e.currentTarget.getBoundingClientRect();
    const f = (e.clientX - r.left) / r.width;
    if (mode === "overview") onSeek(Math.max(0, Math.min(duration, f * duration)));
    else onSeek(Math.max(0, Math.min(duration, time + (f - 0.5) * windowSec)));
  };

  return (
    <canvas
      ref={ref}
      className={`wave wave-${mode}`}
      style={{ height }}
      onClick={seek}
      title={mode === "overview" ? "click to seek" : "click to jump"}
    />
  );
}
