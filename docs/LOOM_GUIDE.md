# Loom guide — line by line (target 9–10 minutes)

The brief asks the video to show: **a click audition, the sync render, at least two chat
edits including one the agent pushed back on, a recreation page playing next to the
original, a walk through the code structure, and the top three problems you know about.**
Camera and mic on, voiceover throughout. Everything below was verified working.

## Before you hit record (5 minutes)

1. Check your Gemini quota at <https://aistudio.google.com> → the Flash-Lite models should
   have requests left (the demo uses ~10). If they are out, chat still works through the
   offline parser for most lines below (noted where it differs).
2. Terminal in the project folder:
   ```bash
   .venv\Scripts\activate
   python run.py app
   ```
   Wait for `Uvicorn running on http://127.0.0.1:8000`.
3. Open **Chrome or Edge** (not the Claude app's preview pane) at http://localhost:8000.
   Browser zoom 90 % so both decks and the panels fit. Close other tabs/notifications.
4. Open VS Code on the project (for the code tour), and File Explorer at
   `C:\Users\Gupt Ah\Downloads\Telegram Desktop` (only if you do the optional real song).
5. Do one dry run of minutes 1–6 without recording (analysis is cached per session,
   so reload the page before the real take).

## The script

### 0:00 – 0:40 · Intro (camera on you)
> "Hi, I'm Tarang. This is Beat Agent: one chat-driven agent that does three things with
> audio — it finds the beat grid and beat-matches two tracks, it edits a track by chat in
> beats and bars, and it recreates a short clip as a Web Audio page. One rule runs through
> all of it: the LLM only interprets what you say; code measures and executes. The test
> pack wasn't provided, so I built synthetic stand-ins with exact answers — the library
> banner at the top says so."

*(Point at the yellow banner: "Library = SYNTHETIC substitute tracks…")*

### 0:40 – 2:10 · Part 1: beat grid + click audition
1. **Deck A → "LOAD TRACK…" → choose `A1 (synthetic) — house 124 BPM constant…`**
   > "A1 is a six-minute house track. Analysis runs on the CPU — Beat This! finds beats,
   > then my code fits a grid that can't drift."
2. When it loads, point at: **124.00 BPM**, the meter **4/4**, the **CONFIDENCE** meter,
   the grid lines on the waveform, the **BAR.BEAT** counter.
   > "Over six minutes every grid beat is within 2 milliseconds of the truth — the target
   > was 10; the raw tracker alone was off by 377."
3. Click **CLICK OFF** so it reads **CLICK ON**, then **► PLAY**. Let ~8 seconds play.
   > "This is the click audition: a click on every beat, an accented one on each downbeat.
   > It's how you check a grid in ten seconds."
4. Click **❚❚ PAUSE**. Click **GRID JSON** (it downloads); open it briefly.
   > "The grid file: tempo, tempo model, every beat with its index, bar and position,
   > downbeats, confidence, and the indexing conventions written into the file."
5. *(Optional, 30 s — real music)* Deck B → **UPLOAD** → pick
   `12. Seedhe Maut - Naam Kaam Sheher [E].flac`. It reads **140 BPM**; click **½×**:
   > "Real hip-hop: 140, or 70 if you count the half-time feel — that's a convention, not
   > an error, so the deck lets you choose." *(Then reload the page or load A7 over it.)*

### 2:10 – 3:30 · Part 1: two-track sync
1. **Deck B → "LOAD TRACK…" → `A7 (synthetic) — house 128 BPM…`**
2. In **SYNC**: set **A bar = 33**, **B bar = 1**, **seconds = 60** → click **SYNC ⇄**.
3. Point at the three tiles: **B STRETCH ×1.032258 (+3.23 %, pitch preserved)**,
   **TEMPO 128.00 → 124.00**, **ALIGN B 1 → A 33**, and the per-beat offset bars.
   > "B is stretched in one piece with Rubber Band so the pitch doesn't change, and its
   > kicks land on A's: median 0.05 ms, worst 0.23 — the target was 5 and 15."
4. **► PLAY** the render ~8 s, then click **B ONLY**, then **MIX** again.
   > "A is the master and is never changed — only B moves."

### 3:30 – 5:30 · Part 2: chat edits, including pushback
1. On **Deck A**, open **EDIT · chat on the grid**.
2. Type **`cut beat 16-32 and repeat it twice`** → **SEND**.
   > "Before it cuts, it says exactly what it understood: beats 16 to 32, which is bar 5.1
   > to bar 9.1, 8.22 to 15.96 seconds, played twice in total. The waveform and grid update,
   > and the grid stays true for the next command."
3. Click **CUE**, then **BAR ►** ×4, **► PLAY** a few seconds over the repeat.
   > "Splices are crossfaded just before the beat, so there's no click."
4. Type **`undo`** → **SEND** (or press **UNDO**).
5. **Pushback 1** — type **`remove bar -2`** → **SEND**.
   > "Code checks the numbers before the LLM is even asked — an LLM once quietly turned
   > this into bar 1."
6. **Pushback 2** — type **`remove everything`** → **SEND**.
   > "It won't delete the whole track on one sentence — it proposes and asks for a yes."
7. Type **`no`** → **SEND**. Then type **`chop out the boring part`** → **SEND**.
   > "Subjective requests get measured answers: bars 81 to 96 have no drums and are quiet,
   > so it proposes those — nothing changes until I say yes." Type **`yes`** → **SEND**.
   *(Offline parser fallback: lines 2, 4, 5 and the proposal work; "remove everything"
   may come back as a clarification instead — say so if it does.)*

### 5:30 – 7:00 · Part 3: recreation next to the original
1. Scroll to **RECREATE** → choose **`P2 — 4-bar house drums + syncopated bass…`** →
   click **RECREATE**. Wait for **"page loaded: 3 tracks, … notes"**.
   > "Part 3 turns a short loop into a score — tempo from Part 1, drums from onset
   > analysis, bass notes from Basic Pitch, no LLM — and the score into a single HTML page
   > that synthesises it. No audio from the original is inside, and the page can't make a
   > network request."
2. Press play on **ORIGINAL** for ~5 s, pause. Then **Play** inside the generated page.
   > "Rhythm first, then pitch, then timbre: the groove lines up; the sounds are synths."
3. Show the mixer: drag **bass** up a few dB.
4. In the chat under the page type **`mute the drums`** → **SEND**; then
   **`play it at 140`** → **SEND**; press **Play** again.
   > "Chat changes are typed score edits too: the page is regenerated from the new score."
5. Mention: "Offline, the same page is rendered to WAV in a headless browser and scored
   against reference MIDI: P2 bass notes 0.90 F, piano 0.96."

### 7:00 – 8:40 · Code tour (VS Code)
Open each briefly:
1. `beatagent/` — "one package per part": `core/`, `beats/`, `editing/`, `recreate/`.
2. `beatagent/beats/analyze.py` — "tracker → cleaned beats → line fits; drift correction for
   live playing; tempo steps found from onsets when the tracker smears them."
3. `beatagent/editing/edl.py` — "typed ops resolved on the grid, pieces with crossfades;
   the edited grid is mapped, never re-tracked."
4. `beatagent/editing/agent.py` + `llm.py` — "the line between interpretation and execution:
   the LLM returns a schema; these guards exist because it misbehaved in testing."
5. `beatagent/recreate/transcribe.py` → `score.py` → `webaudio.py` — "analysis, score,
   fixed page template; the LLM never writes JavaScript."
6. `evals/` and `tests/` → open `evals/results/COMPLETION_REPORT.md` for two seconds:
   > "11 test suites, 222 checks, an end-to-end browser test, 59 chat cases, 30 robustness
   > cases, all reproducible with `python run.py demo`."

### 8:40 – 9:40 · Top three problems (camera)
> "Three things I know are weak:
> 1. **Real-music sync isn't proven to five milliseconds.** On synthetic pairs I have exact
>    ground truth; on real hip-hop my kick detector can't resolve it — it reads 12 to 17
>    ms even when a song is synced with itself. That needs ears, not my detector.
> 2. **Part 3 on dense mixes.** There's no source separation, so on a full mix the drum
>    transcription drops to about 0.5 F and vocals end up in the keys track.
> 3. **The real test pack.** Every graded number is on synthetic stand-ins. The scorers
>    take the real pack as soon as it's dropped in, but I haven't seen it."

### 9:40 – 10:00 · Close
> "Setup, the one command, decisions and how I used AI are in the README, DECISIONS and
> AI_USAGE files. Thanks for watching."

## If something goes wrong while recording
- **Analysis slow:** the first load of a session takes longest (model load); say "it's
  analysing on the CPU" and keep talking.
- **Chat reply says "[LLM unavailable … offline rule parser]":** quota ran out. Say so — it
  is a designed fallback — and continue; skip "remove everything".
- **Recreation page blank:** you are in the Claude app's preview pane; use Chrome/Edge.
