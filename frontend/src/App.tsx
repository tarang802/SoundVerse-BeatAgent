import { useEffect, useState } from "react";
import { api } from "./api";
import { Deck } from "./components/Deck";
import { RecreatePanel } from "./components/RecreatePanel";
import { SyncPanel } from "./components/SyncPanel";
import type { DeckState, LibraryItem } from "./types";

export function App() {
  const [library, setLibrary] = useState<LibraryItem[]>([]);
  const [libError, setLibError] = useState<string | null>(null);
  const [a, setA] = useState<DeckState | null>(null);
  const [b, setB] = useState<DeckState | null>(null);
  const [aBar, setABar] = useState(1);
  const [aPlayBar, setAPlayBar] = useState(1);
  const [bPlayBar, setBPlayBar] = useState(1);
  // decks remount only on swap (so each deck's audio/chat follows its track); a
  // key on the track id would also remount on every load and wipe the chat log
  const [swapGen, setSwapGen] = useState(0);
  const swap = () => { setA(b); setB(a); setSwapGen((g) => g + 1); };

  useEffect(() => {
    api.library().then(setLibrary).catch((e: Error) => setLibError(e.message));
  }, []);

  const synthetic = library.length > 0 && library.every((i) => i.synthetic);

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
            <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#8b7cff" /><stop offset="1" stopColor="#ff7ab6" /></linearGradient></defs>
            <rect width="32" height="32" rx="9" fill="url(#g)" />
            {[7, 11, 15, 19, 23].map((x, i) => (
              <rect key={x} x={x - 1} y={16 - [4, 8, 11, 7, 3][i]} width="2" height={[8, 16, 22, 14, 6][i]} rx="1" fill="#0b0a10" />
            ))}
          </svg>
          <div className="brand-text">
            <span className="logo">Soundverse</span>
            <span className="product">Beat Agent</span>
          </div>
        </div>
        <nav className="caps" aria-label="What it does">
          <span className="cap"><i>1</i>Beat grid &amp; sync</span>
          <span className="cap"><i>2</i>Chat editing</span>
          <span className="cap"><i>3</i>Recreate as Web Audio</span>
        </nav>
        <div className="top-info">
          {libError ? <span className="pill warn">API unreachable: {libError}</span>
            : synthetic ? <span className="pill" title="The assignment test pack was not provided. Upload real music to try it on real audio.">
                Library: synthetic substitute tracks · upload real music anytime</span>
            : <span className="pill">{library.length} tracks</span>}
        </div>
      </header>
      <main className="decks">
        <Deck key={`A${swapGen}`} name="A" role="TEMPO MASTER · unchanged" color="#8b7cff" library={library} state={a}
              onState={setA} markerAtBar={aBar} onPlayheadBar={setAPlayBar} partnerSid={b?.sid} />
        <Deck key={`B${swapGen}`} name="B" role="FOLLOWS A · stretched, pitch kept" color="#ff7ab6" library={library}
              state={b} onState={setB} onPlayheadBar={setBPlayBar} partnerSid={a?.sid} />
      </main>
      <SyncPanel a={a} b={b} aBar={aBar} onABar={setABar} aPlayBar={aPlayBar} bPlayBar={bPlayBar} onSwap={swap} />
      <RecreatePanel deckA={a} />
      <footer className="foot">
        Grid conventions: beats are 0-based grid lines, beat ranges [start, end); bars are 1-based and inclusive;
        bar 0 = pickup. Tempo, beats and offsets are measured from the audio — the LLM only interprets chat.
      </footer>
    </div>
  );
}
