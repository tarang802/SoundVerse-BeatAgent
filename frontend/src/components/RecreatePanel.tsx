import { useEffect, useState } from "react";
import { api } from "../api";
import type { DeckState, LibraryItem, RecreateResult } from "../types";

interface Props {
  deckA: DeckState | null;
}

// Part 3: a short clip -> score (measured, not LLM) -> self-contained Web Audio page.
// The original plays in a normal <audio> element next to the generated page, which runs
// in a sandboxed iframe (scripts only; its own CSP blocks every network request).
export function RecreatePanel({ deckA }: Props) {
  const [clips, setClips] = useState<LibraryItem[]>([]);
  const [choice, setChoice] = useState("");
  const [res, setRes] = useState<RecreateResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState<string | null>(null);

  // the generated page posts {beatagent: "ready"} once its script has run
  useEffect(() => {
    const on = (e: MessageEvent) => {
      const d = e.data as { beatagent?: string; tracks?: number; notes?: number } | null;
      if (d && d.beatagent === "ready") setReady(`page loaded: ${d.tracks} tracks, ${d.notes} notes`);
    };
    window.addEventListener("message", on);
    return () => window.removeEventListener("message", on);
  }, []);

  useEffect(() => {
    api.p3Library().then((c) => {
      setClips(c);
      if (c.length) setChoice(c[0].path);
    }).catch((e: Error) => setError(e.message));
  }, []);

  const run = async () => {
    setBusy(true);
    setError(null);
    setReady(null);
    try {
      setRes(choice === "__deckA" ? await api.recreate({ sid: deckA?.sid }) : await api.recreate({ path: choice }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="recreate">
      <div className="sync-head">
        <h2>RECREATE<small>Part 3 · audio → score → Web Audio page (synthesis only, no network)</small></h2>
        <div className="sync-controls">
          <select value={choice} onChange={(e) => setChoice(e.target.value)}>
            {clips.map((c) => (
              <option key={c.path} value={c.path}>{c.name} — {c.description} (synthetic)</option>
            ))}
            {deckA && <option value="__deckA">Deck A, current version ({deckA.file})</option>}
          </select>
          <button className="btn sync-btn" onClick={run} disabled={busy || !choice}>
            {busy ? "ANALYSING…" : "RECREATE"}
          </button>
        </div>
      </div>
      {error && <p className="warn">{error}</p>}
      {res && (
        <>
          <p className="recreate-meta">
            <b>{res.name}</b>: {res.tempo_bpm.toFixed(1)} BPM ({res.analysis.tempo_from as string}), {res.time_signature},{" "}
            {res.bars} bars, loop {res.loop_s.toFixed(2)} s ·{" "}
            {res.tracks.map((t) => `${t.name} ${t.notes} notes`).join(" · ")} · LLM used: no
          </p>
          <div className="recreate-grid">
            <div>
              <div className="deck-role">ORIGINAL</div>
              <audio controls loop src={api.recreateUrl(res.id, "original.wav")} />
              <div className="recreate-links">
                <a className="btn" href={api.recreateUrl(res.id, "score.json")} target="_blank" rel="noreferrer">SCORE.JSON</a>
                <a className="btn" href={api.recreateUrl(res.id, "page.html")} target="_blank" rel="noreferrer">OPEN PAGE</a>
              </div>
            </div>
            <div>
              <div className="deck-role">RECREATION (generated page){ready ? ` · ${ready}` : " · loading…"}</div>
              <iframe title="recreation" sandbox="allow-scripts" src={api.recreateUrl(res.id, "page.html")} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}
