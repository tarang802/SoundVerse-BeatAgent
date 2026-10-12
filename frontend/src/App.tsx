import { useEffect, useState } from "react";
import { api } from "./api";
import { Deck } from "./components/Deck";
import { SyncPanel } from "./components/SyncPanel";
import type { DeckState, LibraryItem } from "./types";

export function App() {
  const [library, setLibrary] = useState<LibraryItem[]>([]);
  const [libError, setLibError] = useState<string | null>(null);
  const [a, setA] = useState<DeckState | null>(null);
  const [b, setB] = useState<DeckState | null>(null);
  const [aBar, setABar] = useState(33);

  useEffect(() => {
    api.library().then(setLibrary).catch((e: Error) => setLibError(e.message));
  }, []);

  const synthetic = library.length > 0 && library.every((i) => i.synthetic);

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo">SOUNDVERSE</span>
          <span className="product">BEAT AGENT</span>
        </div>
        <div className="top-info">
          {libError ? <span className="warn">API unreachable: {libError}</span>
            : synthetic ? <span className="warn">Library = SYNTHETIC substitute tracks (the assignment test pack was not provided). Upload real music to try it on real audio.</span>
            : <span>{library.length} tracks</span>}
        </div>
      </header>
      <main className="decks">
        <Deck name="A" role="MASTER" color="#00e5ff" library={library} state={a} onState={setA} markerAtBar={aBar} />
        <Deck name="B" role="SYNC TO A" color="#ff2bd6" library={library} state={b} onState={setB} />
      </main>
      <SyncPanel a={a} b={b} aBar={aBar} onABar={setABar} />
      <footer className="foot">
        Grid conventions: beats are 0-based grid lines, beat ranges [start, end); bars are 1-based and inclusive;
        bar 0 = pickup. Tempo, beats and offsets are measured from the audio — the LLM only interprets chat.
      </footer>
    </div>
  );
}
