# Loom guide (~8 min)

Talking points, not lines to read. Say them your own way.

**Setup:** in the VS Code terminal run `python run.py app`. Open http://localhost:8000 in
Chrome and press Ctrl+Shift+R. Close other heavy tabs.

---

### 1. Intro (20 s)
- "Beat Agent: three parts. Beat grid and sync, chat editing, and recreating audio as a Web Audio page."
- "One rule throughout: the LLM only interprets what you type. Code measures and executes."
- "The test pack wasn't provided, so I built synthetic versions with exact answers."

### 2. Beat grid and click (1 min)
- Deck A → **LOAD TRACK → A1**
- Point at the BPM, the 4/4 meter, the confidence bar.
- "Six minutes, worst beat within 2 ms. The raw tracker alone drifts 377 ms."
- **CLICK OFF → ON**, **► PLAY**, about 5 s, then **PAUSE**.
- "Accented click on the downbeats, so you can check the grid by ear."
- **GRID JSON**: "every beat with its bar, position and confidence."

### 3. Sync (1 min)
- Deck B → **A7**
- SYNC: A bar **33**, B bar **1**, seconds **60** → **SYNC ⇄**
- "B is stretched 3.2% with pitch kept. A is never touched."
- "Kicks land within 0.05 ms median. The target was 5."
- **► PLAY**, then **B ONLY**, then **MIX**.

### 4. Chat edits (2 min)
Open **EDIT** on deck A:
- `cut beat 16-32 and repeat it twice`
  - "It tells me exactly what it understood: bars, beats, seconds. Then it edits. The grid updates."
- `undo`
- `remove bar -2`
  - "Pushback. Code catches it before the LLM. An LLM once quietly turned this into bar 1."
- `remove everything`
  - "Won't wipe the track on one sentence. It asks first."
- `no`
- `chop out the boring part`
  - "Subjective request. It measures loudness and drums and proposes bars 81 to 96." → `yes`

### 5. Recreate (1.5 min)
- RECREATE → **P2** → **RECREATE**
- Play **ORIGINAL** for 5 s, then **Play** in the generated page.
- "Tempo from part 1, drums from onsets, bass from Basic Pitch. It's all synthesis: no samples, no network."
- Drag the bass slider.
- Chat: `mute the drums`, then `play it at 140`, then Play.
- "Bass notes score 0.90 F against the MIDI."

### 6. Code (1.5 min)
In VS Code, open briefly:
- `beatagent/`: "one package per part: core, beats, editing, recreate."
- `beats/analyze.py`: "tracker, then my grid fit, so it can't drift."
- `editing/agent.py`: "the LLM boundary. Every guard here is from a real failure in testing."
- `recreate/webaudio.py`: "fixed template. The LLM never writes JS."
- `evals/` + `tests/`: "11 test suites, 59 chat cases, plus Docker CI on GitHub."

### 7. Top 3 problems (40 s)
1. "Sync on real music isn't proven to 5 ms. My detector can't resolve it, so it needs ears."
2. "Part 3 on dense mixes: no source separation, so drums drop to about 0.5 F."
3. "All the numbers are on synthetic clips. The scorers will take the real pack."

### 8. Close (10 s)
- "Run it with `docker compose up`. Details are in the README and DECISIONS. Thanks."

---

**If something breaks:**
- **Memory error on A1:** use A7 instead.
- **Reply says "offline rule parser":** the Gemini quota ran out. Say "that's the fallback" and move on.
