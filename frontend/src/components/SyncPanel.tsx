import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { decodePeaks, fmtTime } from "../peaks";
import type { DeckState, SyncResult } from "../types";
import { useAudioClock } from "../useAudioClock";
import { Waveform } from "./Waveform";

interface Props {
  a: DeckState | null;
  b: DeckState | null;
  aBar: number;
  onABar: (n: number) => void;
}

type Src = "mix" | "a" | "b";

export function SyncPanel({ a, b, aBar, onABar }: Props) {
  const [bBar, setBBar] = useState(1);
  const [dur, setDur] = useState(60);
  const [res, setRes] = useState<SyncResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [src, setSrc] = useState<Src>("mix");
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null);
  const time = useAudioClock(audio);

  // a new result or a different stem: swap the audio source, keep the position
  useEffect(() => {
    if (!audio || !res) return;
    const t = audio.currentTime;
    const playing = !audio.paused;
    audio.src = api.syncUrl(res.id, src);
    audio.addEventListener("loadedmetadata", () => {
      audio.currentTime = Math.min(t, audio.duration || 0);
      if (playing) void audio.play();
    }, { once: true });
  }, [audio, res?.id, src]); // eslint-disable-line react-hooks/exhaustive-deps

  const peaks = useMemo(() => (res ? decodePeaks(res.peaks[src], res.peak_rate) : null), [res, src]);

  const run = async () => {
    if (!a || !b) return;
    setBusy(true);
    setError(null);
    audio?.pause();
    try {
      setRes(await api.sync(a.sid, b.sid, aBar, bBar, dur));
      setSrc("mix");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const al = res?.alignment;
  const offs = al?.offsets_ms ?? [];
  const maxAbs = Math.max(1, ...offs.map((x) => Math.abs(x)));

  return (
    <section className="sync">
      <header className="sync-head">
        <h2>SYNC <small>beat-match B to master A</small></h2>
        <div className="sync-controls">
          <label>A bar <input type="number" min={1} value={aBar} onChange={(e) => onABar(+e.target.value)} /></label>
          <label>B bar <input type="number" min={1} value={bBar} onChange={(e) => setBBar(+e.target.value)} /></label>
          <label>seconds <input type="number" min={5} max={300} value={dur} onChange={(e) => setDur(+e.target.value)} /></label>
          <button className="btn sync-btn" disabled={!a || !b || busy} onClick={() => void run()}>
            {busy ? "RENDERING…" : "SYNC ⇄"}
          </button>
        </div>
      </header>
      {!a || !b ? <p className="hint">Load a track on both decks. A is the master and is never changed; B is time-stretched (pitch preserved) so its bar {bBar} downbeat lands on A's bar {aBar} downbeat.</p> : null}
      {error && <div className="deck-error">⚠ {error}</div>}

      {res && al && peaks && (
        <div className="sync-body">
          <div className="sync-plan">
            <div className="big"><span className="label">B STRETCH</span>
              <span className="val">×{res.plan.time_ratio.toFixed(6)}</span>
              <span className="sub">{res.plan.stretch_percent >= 0 ? "+" : ""}{res.plan.stretch_percent.toFixed(2)}% · pitch preserved</span></div>
            <div className="big"><span className="label">TEMPO</span>
              <span className="val">{res.plan.b_bpm.toFixed(2)} → {res.plan.a_bpm.toFixed(2)}</span>
              <span className="sub">B's BPM → A's BPM</span></div>
            <div className="big"><span className="label">ALIGN</span>
              <span className="val">B {res.plan.b_bar} → A {res.plan.a_bar}</span>
              <span className="sub">B@{res.plan.b_start.toFixed(3)}s on A@{res.plan.a_start.toFixed(3)}s · offset {res.plan.offset.toFixed(3)}s</span></div>
          </div>

          <div className="tiles">
            <Tile label="KICK OFFSET median |B−A|" val={`${al.median_abs_ms ?? "—"} ms`} target="≤ 5 ms"
                  ok={(al.median_abs_ms ?? 99) <= 5} />
            <Tile label="KICK OFFSET max |B−A|" val={`${al.max_abs_ms ?? "—"} ms`} target="≤ 15 ms"
                  ok={(al.max_abs_ms ?? 99) <= 15} />
            <Tile label="BEATS MEASURED" val={`${al.n_measured}/${al.n_beats}`} target="coverage of the score"
                  ok={al.n_measured >= 0.9 * al.n_beats} />
            <Tile label="RESULT" val={al.pass ? "PASS" : "FAIL"} target="brief's sync target" ok={!!al.pass} />
          </div>

          <div className="offsets" title="B − A kick offset per beat (ms); red line = ±15 ms limit">
            <span className="label">PER-BEAT OFFSET (B − A), scale ±{maxAbs.toFixed(1)} ms</span>
            <div className="bars">
              {offs.map((o, i) => (
                <div key={i} className={`ob ${Math.abs(o) > 15 ? "bad" : ""}`}
                     style={{ left: `${(100 * i) / offs.length}%`, width: `${Math.max(0.2, 100 / offs.length - 0.15)}%`,
                              height: `${50 * Math.abs(o) / maxAbs}%`, [o >= 0 ? "bottom" : "top"]: "50%" }}
                     title={`beat ${i}: ${o.toFixed(2)} ms`} />
              ))}
            </div>
          </div>

          <Waveform peaks={peaks} duration={res.plan.duration} beats={res.beats} positions={res.positions}
                    time={time} color={src === "b" ? "#ff2bd6" : src === "a" ? "#00e5ff" : "#f5f5f7"}
                    mode="zoom" height={110} onSeek={(t) => { if (audio) audio.currentTime = t; }} />
          <Waveform peaks={peaks} duration={res.plan.duration} beats={res.beats} positions={res.positions}
                    time={time} color={src === "b" ? "#ff2bd6" : src === "a" ? "#00e5ff" : "#f5f5f7"}
                    mode="overview" height={36} onSeek={(t) => { if (audio) audio.currentTime = t; }} />

          <div className="transport">
            <button className="btn play" onClick={() => (audio?.paused ? void audio.play() : audio?.pause())}>
              {audio && !audio.paused ? "❚❚ PAUSE" : "► PLAY"}
            </button>
            <div className="seg">
              {(["mix", "a", "b"] as Src[]).map((k) => (
                <button key={k} className={`btn ${src === k ? "on" : ""}`} onClick={() => setSrc(k)}>
                  {k === "mix" ? "MIX" : k === "a" ? "A ONLY" : "B ONLY"}
                </button>
              ))}
            </div>
            <span className="time">{fmtTime(time)} / {fmtTime(res.plan.duration)}</span>
            <span className="spacer" />
            <a className="btn ghost" href={api.syncUrl(res.id, "mix")} download>MIX WAV</a>
            <a className="btn ghost" href={api.syncUrl(res.id, "report")} download>REPORT</a>
          </div>
          <audio ref={setAudio} preload="auto" />
          <p className="method">method: {res.method} · rendered in {res.render_s}s
            {res.plan.notes.length ? ` · ${res.plan.notes.join(" ")}` : ""}</p>
        </div>
      )}
    </section>
  );
}

function Tile({ label, val, target, ok }: { label: string; val: string; target: string; ok: boolean }) {
  return (
    <div className={`tile ${ok ? "ok" : "bad"}`}>
      <span className="label">{label}</span>
      <span className="val">{val}</span>
      <span className="target">{target}</span>
    </div>
  );
}
