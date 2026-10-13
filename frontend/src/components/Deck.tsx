import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { barNumbers, beatIndexAt, decodePeaks, fmtTime } from "../peaks";
import type { ChatReply, DeckState, LibraryItem } from "../types";
import { useAudioClock } from "../useAudioClock";
import { Waveform } from "./Waveform";

interface Props {
  name: "A" | "B";
  role: string;
  color: string;
  library: LibraryItem[];
  state: DeckState | null;
  onState: (s: DeckState | null) => void;
  markerAtBar?: number;          // sync start bar to highlight on this deck
  onPlayheadBar?: (bar: number) => void;   // bar under the playhead (for "sync from playhead")
  partnerSid?: string;           // the other deck, so "sync ..." works from this deck's chat
}

interface Msg {
  who: "you" | "agent";
  text: string;
  kind?: "ok" | "err" | "ask";
  tag?: string;
}

export function Deck({ name, role, color, library, state, onState, markerAtBar, onPlayheadBar, partnerSid }: Props) {
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null);
  const time = useAudioClock(audio);
  const [click, setClick] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const peaks = useMemo(() => (state ? decodePeaks(state.peaks, state.peak_rate) : null), [state]);
  const bars = useMemo(() => (state ? barNumbers(state.positions) : []), [state]);

  // (re)load audio when the version or click mode changes, keeping position + play state
  useEffect(() => {
    if (!audio || !state) return;
    const t = audio.currentTime;
    const playing = !audio.paused;
    audio.src = api.audioUrl(state, click);
    const restore = () => {
      audio.currentTime = Math.min(t, audio.duration || 0);
      if (playing) void audio.play();
    };
    audio.addEventListener("loadedmetadata", restore, { once: true });
  }, [audio, state?.sid, state?.state_id, click]); // eslint-disable-line react-hooks/exhaustive-deps

  // live seconds counter while the CPU beat tracker runs
  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    setElapsed(0);
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 200);
    return () => clearInterval(id);
  }, [busy]);

  const load = async (label: string, fn: () => Promise<DeckState>) => {
    audio?.pause();
    setError(null);
    setBusy(`Analysing ${label}`);
    try {
      const st = await fn();
      onState(st);
      setMsgs([{ who: "agent", tag: "loaded",
        text: `${st.file}: ${st.summary.tempo_bpm} BPM, ${st.summary.time_signature}, ${st.summary.n_bars} bars, ` +
              `${st.summary.grid_type} grid (analysed in ${st.analysis_s}s). Beats are 0-based grid lines, ` +
              `bars are 1-based.` }]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const send = async (t: string, fn: () => Promise<{ reply: ChatReply; state: DeckState; sync_id?: string }>) => {
    if (!state) return;
    setMsgs((m) => [...m, { who: "you", text: t }]);
    try {
      const { reply, state: st, sync_id } = await fn();
      let body = reply.interpretation ? `${reply.text}\nReading: ${reply.interpretation}` : reply.text;
      if (sync_id) body += `\nListen: ${location.origin}${api.syncUrl(sync_id, "mix")}`;
      setMsgs((m) => [...m, { who: "agent", text: body, tag: `${reply.action} · ${reply.backend}`,
        kind: !reply.ok ? "err" : reply.action === "clarify" ? "ask" : reply.changed ? "ok" : undefined }]);
      onState(st);
    } catch (e) {
      setMsgs((m) => [...m, { who: "agent", text: (e as Error).message, kind: "err", tag: "error" }]);
    }
  };

  const seek = (t: number) => {
    if (audio) audio.currentTime = t;
  };
  const jumpBars = (n: number) => {
    if (!state || !audio) return;
    const downs = state.beats.filter((_, i) => state.positions[i] === 1);
    const i = beatIndexAt(downs, audio.currentTime);
    const target = downs[Math.max(0, Math.min(downs.length - 1, i + n))];
    if (target !== undefined) seek(target);
  };

  const idx = state ? beatIndexAt(state.beats, time) : -1;
  const playBar = state && idx >= 0 ? bars[idx] : 1;
  useEffect(() => { onPlayheadBar?.(Math.max(1, playBar)); }, [playBar]); // eslint-disable-line react-hooks/exhaustive-deps

  const setOctave = async (f: 2 | 0.5) => {
    if (!state) return;
    setError(null);
    try {
      onState(await api.octave(state.sid, f));
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const counter = state && idx >= 0 ? `${bars[idx]}.${state.positions[idx]}` : "--.-";
  const s = state?.summary;
  const conf = state?.confidence.overall ?? 0;
  const markerT = state && markerAtBar && markerAtBar >= 1 && markerAtBar <= bars[bars.length - 1]
    ? state.beats[bars.indexOf(markerAtBar)] : undefined;
  const markers = markerT !== undefined ? [{ t: markerT, label: `SYNC ${markerAtBar}`, color: "#f5c26b" }] : [];

  return (
    <section className="deck" style={{ ["--accent" as string]: color }}>
      <header className="deck-head">
        <div className="deck-id">
          <span className="deck-letter">{name}</span>
          <span className="deck-role">{role}</span>
        </div>
        <div className="deck-load">
          <select value="" onChange={(e) => e.target.value && load(e.target.value.split("/").pop()!, () => api.open(e.target.value))}
                  disabled={!!busy}>
            <option value="">LOAD TRACK…</option>
            {library.map((it) => (
              <option key={it.path} value={it.path}>
                {it.name}{it.synthetic ? " (synthetic)" : ""} — {it.description}
              </option>
            ))}
          </select>
          <button className="btn ghost" disabled={!!busy} onClick={() => fileRef.current?.click()}>UPLOAD</button>
          <input ref={fileRef} type="file" hidden accept=".wav,.mp3,.flac,audio/*"
                 onChange={(e) => { const f = e.target.files?.[0]; if (f) void load(f.name, () => api.upload(f)); e.target.value = ""; }} />
        </div>
      </header>

      {busy && (
        <div className="deck-busy">
          <div className="spinner" /> {busy}… {elapsed.toFixed(0)}s
          <small>neural beat tracker on CPU — about 40 s for a 6-minute track; longer if both decks analyse at once</small>
        </div>
      )}
      {error && <div className="deck-error">⚠ {error}</div>}

      {state && peaks && s && (
        <>
          <div className="deck-display">
            <div className="bpm">
              <span className="bpm-val">{s.tempo_bpm.toFixed(2)}</span><span className="bpm-unit">BPM</span>
              <span className="octave" title="Half-time music can be counted at either tempo (e.g. 70 or 140). Set the one you hear; sync works either way.">
                <button className="btn mini" disabled={state.version > 0 || s.time_signature.endsWith("/8")} onClick={() => void setOctave(0.5)}>½×</button>
                <button className="btn mini" disabled={state.version > 0 || s.time_signature.endsWith("/8")} onClick={() => void setOctave(2)}>2×</button>
              </span>
            </div>
            <div className="counter"><span className="label">BAR.BEAT</span><span className="val">{counter}</span></div>
            <div className="facts">
              <span><b>{s.time_signature}</b> meter</span>
              <span><b>{s.grid_type}</b> grid</span>
              <span><b>{s.n_bars}</b> bars</span>
              <span><b>{fmtTime(time)}</b> / {fmtTime(state.duration)}</span>
            </div>
            <div className="conf" title={state.confidence.note}>
              <span className="label">CONFIDENCE</span>
              <div className="meter"><div style={{ width: `${Math.round(conf * 100)}%` }} /></div>
              <span className="val">{conf.toFixed(2)}</span>
            </div>
          </div>
          {(state.user_octave || (state.tempo_octave && state.tempo_octave.factor !== 1)) && (
            <div className="segments">
              {state.user_octave
                ? `TEMPO OCTAVE set by you (x${state.user_octave === 2 ? "2" : "1/2"})`
                : `TEMPO OCTAVE auto: tracker heard ${state.tempo_octave!.tracker_bpm} BPM, shown x${state.tempo_octave!.factor === 2 ? "2" : "1/2"} to fit ${state.tempo_octave!.preferred_range[0]}-${state.tempo_octave!.preferred_range[1]} BPM. Use ½× / 2× if you count it differently.`}
            </div>
          )}
          {s.tempo_segments.length > 1 && (
            <div className="segments">
              TEMPO CHANGE · {s.tempo_segments.map((g) => `bar ${g.start_bar}: ${g.bpm.toFixed(2)}`).join("  →  ")}
            </div>
          )}

          <Waveform peaks={peaks} duration={state.duration} beats={state.beats} positions={state.positions}
                    time={time} color={color} mode="zoom" height={130} markers={markers} onSeek={seek} />
          <Waveform peaks={peaks} duration={state.duration} beats={state.beats} positions={state.positions}
                    time={time} color={color} mode="overview" height={44} markers={markers} onSeek={seek} />

          <div className="transport">
            <button className="btn play" onClick={() => (audio?.paused ? void audio.play() : audio?.pause())}>
              {audio && !audio.paused ? "❚❚ PAUSE" : "► PLAY"}
            </button>
            <button className="btn" onClick={() => seek(state.beats[state.positions.indexOf(1)] ?? 0)}>CUE</button>
            <button className="btn" onClick={() => jumpBars(-1)}>◄ BAR</button>
            <button className="btn" onClick={() => jumpBars(1)}>BAR ►</button>
            <button className={`btn toggle ${click ? "on" : ""}`} onClick={() => setClick((c) => !c)}
                    title="original audio + click on every beat, accented click on downbeats">
              CLICK {click ? "ON" : "OFF"}
            </button>
            <span className="spacer" />
            <a className="btn ghost" href={api.gridUrl(state)} download>GRID JSON</a>
            <a className="btn ghost" href={api.clickUrl(state)} download>CLICK WAV</a>
          </div>
          <audio ref={setAudio} preload="auto" />

          <details className="edit">
            <summary>EDIT · chat on the grid <small>({state.interpreter}, v{state.version})</small></summary>
            <div className="log">
              {msgs.map((m, i) => (
                <div key={i} className={`msg ${m.who} ${m.kind ?? ""}`}>
                  {m.tag && <span className="tag">{m.tag}</span>}
                  {m.text}
                </div>
              ))}
            </div>
            <form className="chat" onSubmit={(e) => { e.preventDefault(); const t = text.trim(); if (t) { setText(""); void send(t, () => api.chat(state.sid, t, partnerSid)); } }}>
              <input value={text} onChange={(e) => setText(e.target.value)} placeholder='e.g. "cut beat 16-32 and repeat it twice"' />
              <button className="btn">SEND</button>
              <button type="button" className="btn ghost" disabled={!state.can_undo} onClick={() => void send("undo", () => api.undo(state.sid))}>UNDO</button>
              <button type="button" className="btn ghost" disabled={!state.can_redo} onClick={() => void send("redo", () => api.redo(state.sid))}>REDO</button>
            </form>
          </details>
        </>
      )}
      {!state && !busy && <div className="deck-empty">Load a track to analyse its beat grid.</div>}
    </section>
  );
}
