// The server sends the waveform as base64 bytes: interleaved int8 (min, max)
// pairs, `rate` pairs per second (see _peaks_b64 in app/server.py).

export interface Peaks {
  data: Int8Array; // [min0, max0, min1, max1, ...]
  rate: number;    // pairs per second
  count: number;   // number of pairs
}

export function decodePeaks(b64: string, rate: number): Peaks {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const data = new Int8Array(bytes.buffer);
  return { data, rate, count: data.length / 2 };
}

/** Index of the last beat at or before time t (-1 if t is before the first beat). */
export function beatIndexAt(beats: number[], t: number): number {
  let lo = 0;
  let hi = beats.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (beats[mid] <= t + 1e-6) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

/** 1-based bar number of every beat (0 = pickup before the first downbeat). Mirrors BeatGrid.bars. */
export function barNumbers(positions: number[]): number[] {
  let bar = 0;
  return positions.map((p) => (p === 1 ? ++bar : bar));
}

export const fmtTime = (t: number) => {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
};
