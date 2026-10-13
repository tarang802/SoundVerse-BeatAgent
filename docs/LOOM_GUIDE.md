# Loom recording guide (about 8 minutes)

Show more, talk less. Each step has three parts: **DO** (the click), **WAIT** (how long)
and **SAY** (one or two short lines).

## Before recording
1. Chrome → http://localhost:8000 → **Ctrl+Shift+R**. Zoom 90% (Ctrl + minus). Close other tabs.
2. VS Code tabs open: `beats/analyze.py`, `editing/agent.py`, `recreate/webaudio.py`.
3. Loom: Screen + Camera, bubble at the bottom-right, mic on.

---

## 1. Intro (30 s) · camera

> "Hey, I'm Tarang. This is Beat Agent, my take on the Soundverse assignment. You give it a
> track and talk to it like a producer friend: it finds the beat, syncs two tracks, edits by
> chat, and turns a loop into code that plays in the browser. The AI only interprets what
> you say. Every number comes from measuring the audio. The official test tracks weren't
> included, so I built synthetic ones with exact answers. Let's dive in."

---

## 2. Beat grid + click (1 min)

**DO:** Deck A → **LOAD TRACK…** → **A1 (synthetic) — house 124 BPM…**
**WAIT** 30–40 s.
**SAY:** "A neural tracker finds the beats, then my code fits a grid that can't drift."

**DO:** point at **124.00 BPM**, **4/4**, **CONFIDENCE**.
**SAY:** "124 BPM, 4/4. Worst beat over six minutes: 2 ms. The raw tracker was 377."

**DO:** **CLICK OFF → ON**, then **► PLAY**. **WAIT** 6 s.
**SAY:** "Click audition. The downbeats are accented, and it sits right on the kick."

**DO:** **❚❚ PAUSE** → **GRID JSON**, open it for 3 s → back to the app.
**SAY:** "The grid as JSON: every beat, bar, position and confidence."

---

## 3. Sync (1 min)

**DO:** Deck B → **LOAD TRACK…** → **A7 (synthetic) — house 128 BPM…** **WAIT** 20 s.
**DO:** SYNC panel → A bar `33`, B bar `1`, seconds `60` → **SYNC ⇄**. **WAIT** 10–15 s.
**SAY:** "B is stretched to A's tempo, with the pitch kept. Its bar 1 lands on A's bar 33. A is never changed."

**DO:** point at **B STRETCH ×1.032258** and the green offset tiles.
**SAY:** "Kick offset 0.05 ms median. The target was 5."

**DO:** **► PLAY** → **B ONLY** → **MIX** → **❚❚ PAUSE**.

---

## 4. Chat edits (2 min)

**DO:** deck A → open **▸ EDIT · chat on the grid**. Type each command, then press **SEND**.

| Type | Say |
|---|---|
| `cut beat 16-32 and repeat it twice` | "It states exactly what it understood: beats, bars, seconds. Then it edits, and the grid updates." |
| `undo` | "Undo." |
| `remove bar -2` | "Pushback. Code rejects it before the AI sees it." |
| `remove everything` | "It won't wipe the track on one sentence. It asks first." |
| `no` | — |
| `chop out the boring part` | "It measures loudness and drums, then proposes the quiet bars 81 to 96." |
| `yes` | "Only now does it cut." |

---

## 5. Recreate (1.5 min)

**DO:** RECREATE → **P2 — 4-bar house drums + syncopated bass…** → **RECREATE**. **WAIT** 10–20 s.
**SAY:** "A loop becomes a score, and the score becomes a web page. Tempo from part one, drums from onsets, bass from Basic Pitch."

**DO:** **ORIGINAL** player → play 5 s → pause. Then **Play** in the page for 6 s.
**SAY:** "Original… and the recreation. Pure synthesis, no audio files, no network."

**DO:** drag the **bass** slider. **Stop**. In the chat, type `mute the drums` → **SEND**,
then `play it at 140` → **SEND** → **Play**.
**SAY:** "Chat changes the score, and the page is rebuilt. Bass notes score 0.90 against the MIDI."

**DO:** **Stop**.

---

## 6. Code (1 min) · VS Code

| Show | Say |
|---|---|
| `beatagent/` folder | "One package per part: core, beats, editing, recreate." |
| `analyze.py` | "Beat tracking and the grid fit." |
| `agent.py` | "The AI-to-code boundary. Every guard here came from a real failure in testing." |
| `webaudio.py` | "Fixed template. The AI never writes JavaScript." |
| `evals/`, `tests/` | "11 test suites, 59 chat cases, and Docker CI on GitHub." |

---

## 7. Top 3 problems (30 s) · camera

> "Three known gaps. One: sync on real songs isn't proven to 5 ms; my detector can't resolve
> it. Two: dense mixes in part three. With no source separation, drums drop to about 0.5 F.
> Three: every number is on synthetic tracks; the scorers will take the real pack."

## 8. Close (10 s)

> "Run it with `docker compose up`. Details are in the README. Thanks!"

---

## If something breaks

| Problem | Fix |
|---|---|
| A1 shows "Unable to allocate" | Use **A7** on deck A, then **A4** on deck B |
| Reply says "offline rule parser" | Say "quota's out, that's the fallback" and continue |
| Page won't load | Pause and message Claude "restart the app" |
