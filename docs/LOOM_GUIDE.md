# Loom recording guide (about 9 minutes)

Every step has three parts: **DO** (the click), **SAY** (talk naturally; the quotes are a
guide, not words to read) and **WAIT** (how long, and what to say while you wait).

---

## Before you press record (2 minutes)

1. Chrome → http://localhost:8000 → press **Ctrl+Shift+R**. You should see the
   Soundverse header and two empty decks, A and B.
2. Close other Chrome tabs and heavy apps (the laptop is short on RAM).
3. VS Code open on the project, with these files already in tabs (for the code part):
   `beatagent/beats/analyze.py`, `beatagent/editing/agent.py`,
   `beatagent/recreate/webaudio.py`.
4. Loom: **Screen + Camera**, full screen, mic on. Put the camera bubble at the
   **bottom-right**.
5. Browser zoom 90% (Ctrl + minus once), so the decks and the SYNC panel fit.
6. Volume at a comfortable level. The click and the music will play through your speakers.

---

## 1. Intro (0:00–0:30) · camera bubble big, or just look at the camera

**DO:** nothing yet. Look at the camera, with the app visible behind you.

**SAY:**
> "Hey! I'm Tarang, and this is Beat Agent, my take on the Soundverse assignment.
> The idea: you give it a track and talk to it like you'd talk to a producer friend. It
> finds the beat, lines two tracks up, edits by chat, and can even turn a short loop back
> into code that plays in the browser.
> One thing I was strict about: the AI only figures out what you *mean*. Every number,
> like tempo, beats and timing, comes from measuring the audio, so it doesn't make things up.
> Quick heads-up: the official test tracks weren't included, so I built synthetic ones where
> I know the exact answers. That way I could actually check my work. Let's dive in."

---

## 2. Beat grid + click audition (0:30–2:00)

**DO:** Deck **A** (left) → click the **LOAD TRACK…** dropdown → choose
**A1 (synthetic) — house 124 BPM constant…**

**WAIT ~30–40 s.** A spinner says *"Analysing A1.wav…"* with a timer.
**SAY while it loads:**
> "This is a six-minute house track. The analysis runs on the CPU. A neural beat tracker
> called Beat This! finds the beats first, and then my own code fits a clean grid on top. The
> tracker on its own wobbles a bit, and on a six-minute track that wobble adds up. For a
> DJ that's the difference between kicks landing together and flamming."

**DO:** when the waveform appears, move your mouse over **124.00 BPM**, then over
**4/4 meter**, then over the **CONFIDENCE** bar.

**SAY:**
> "124 BPM, 4/4, high confidence. Over the full six minutes, the worst beat is 2
> milliseconds off. The raw tracker was 377 off, so the grid fit is really earning its keep."

**DO:** click **CLICK OFF** (it turns to **CLICK ON**) → click **► PLAY**.

**WAIT ~6 s** and let it play.
**SAY:**
> "The easiest way to trust a grid is to hear it. A click on every beat and a louder one
> on each downbeat, sitting right on the kick."

**DO:** click **❚❚ PAUSE** → click **GRID JSON** (it downloads) → open it from the
download bar for 3 seconds → go back to the app tab.

**SAY:**
> "And here's the grid as data: tempo, every beat with its bar and position, the downbeats,
> confidence, and whether the tempo is constant."

*(Leave CLICK ON as it is.)*

---

## 3. Sync two tracks (2:00–3:15)

**DO:** Deck **B** (right) → **LOAD TRACK…** → **A7 (synthetic) — house 128 BPM…**

**WAIT ~20 s.**
**SAY while it loads:**
> "Now the DJ part. A7 is another house track, but at 128 BPM. I want it to come in on
> bar 33 of A1."

**DO:** in the **SYNC** panel (below the decks):
- **A bar** box → select the number → type `33`
- **B bar** box → `1`
- **seconds** box → `60`
- click the gold **SYNC ⇄** button

**WAIT ~10–15 s.**
**SAY:**
> "B gets time-stretched as one piece, so its tempo matches A's, with the pitch kept the
> same. Its bar 1 lands exactly on A's bar 33. Deck A is the master; I never touch it."

**DO:** point at the three boxes that appear: **B STRETCH ×1.032258**,
**TEMPO 128.00 → 124.00**, **ALIGN B 1 → A 33**. Then point at the green offset tiles.

**SAY:**
> "About a three percent stretch. The kicks line up within 0.05 milliseconds median, and
> the target was five. On these synthetic tracks I know exactly where every kick should be,
> so that's a real measurement, not a guess."

**DO:** in the sync player, click **► PLAY** → after 5 s click **B ONLY** → after 3 s click
**MIX** → click **❚❚ PAUSE**.

**SAY:**
> "Here's the mix… just B, stretched… and both together again."

---

## 4. Chat editing (3:15–5:30)

**DO:** scroll back up to deck **A** → click **▸ EDIT · chat on the grid** to open it.
Each command below goes in the box followed by **SEND** (or the Enter key).

**4a. DO:** type `cut beat 16-32 and repeat it twice` → **SEND**
**WAIT ~3–5 s** for the reply.
**SAY:**
> "Notice it doesn't just do it. It tells me exactly what it understood: beats 16 to 32,
> which is bar 5 to bar 9, from 8.2 to 15.9 seconds, played twice. Then it edits, and the
> waveform and the grid both update, so the next command still refers to the right bars."

**4b. DO:** type `undo` → **SEND**
**SAY:** "Changed my mind. Undo works the way you'd expect."

**4c. DO:** type `remove bar -2` → **SEND**
**SAY:**
> "Now let me be difficult. Bar minus two doesn't exist. Funny story: during testing, the
> LLM quietly turned this into bar 1 and deleted it. So now code checks the numbers before
> the AI even sees the message."

**4d. DO:** type `remove everything` → **SEND**
**SAY:**
> "This one would wipe the whole track. It won't do that on one sentence. It shows what
> would be left and asks me to confirm."

**4e. DO:** type `no` → **SEND**
**SAY:** "Nope, keep it."

**4f. DO:** type `chop out the boring part` → **SEND**
**WAIT ~5 s.**
**SAY:**
> "My favourite. 'Boring' is subjective, so instead of guessing it measures loudness and
> drum activity per section. It finds the quiet part with no drums, bars 81 to 96, and
> proposes exactly that."

**4g. DO:** type `yes` → **SEND**
**SAY:** "And only now does it actually cut."

> *If a reply mentions the "offline rule parser": say "the Gemini quota ran out, so that's
> the fallback kicking in", and carry on.*

---

## 5. Recreate audio as code (5:30–7:00)

**DO:** scroll down to **RECREATE** → in its dropdown choose
**P2 — 4-bar house drums + syncopated bass…** → click **RECREATE**.

**WAIT ~10–20 s.**
**SAY while it works:**
> "Part three is my favourite. I take a short loop, figure out what's being played, and
> rebuild it as a web page that synthesizes it live. The tempo comes from part one, the
> drums from onset detection, and the bass notes from Spotify's Basic Pitch model."

**DO:** in the **ORIGINAL** player on the left, press play → let it run ~5 s → pause it.
**SAY:** "That's the original loop…"

**DO:** in the generated page on the right, click its **Play** button → ~6 s.
**SAY:**
> "…and that's the page playing it. There's no audio file inside: every drum hit and bass
> note is generated from a score. It also can't make any network request."

**DO:** in the page, drag the **bass** slider up a little.
**SAY:** "Each part gets its own fader."

**DO:** click **Stop** in the page. In the chat box under the page, type
`mute the drums` → **SEND** → then type `play it at 140` → **SEND** → click **Play** in
the page.
**SAY:**
> "Same rule as the editor. The AI turns what I say into small structured changes, and
> code applies them and rebuilds the page. Now it's drumless and faster."

**DO:** click **Stop**.
**SAY:** "Offline, I render these pages in a headless browser and score them against the
real MIDI. The bass here gets 0.90."

---

## 6. Code walkthrough (7:00–8:15) · switch to VS Code

**DO:** collapse the folders in the Explorer, then expand **beatagent**.
**SAY:**
> "Quick tour. One package per part: beats, editing and recreate, plus a shared core."

**DO:** click the `analyze.py` tab, scroll slowly for 3 s.
**SAY:** "Tracker in, cleanup and line fitting out, so the grid can't wander."

**DO:** click the `agent.py` tab.
**SAY:**
> "This is the boundary between the AI and the code. Pretty much every guard in here
> exists because the model did something weird during testing."

**DO:** click the `webaudio.py` tab.
**SAY:**
> "A fixed page template. The AI never writes JavaScript, which matters when the output
> runs in someone's browser."

**DO:** expand **evals** and **tests** in the Explorer.
**SAY:**
> "And this is how I know it works: 11 test suites, 59 chat cases, robustness tests, and a
> GitHub check that builds the Docker image from scratch and runs everything inside it."

---

## 7. Top three problems (8:15–9:00) · look at the camera

**SAY:**
> "Three things I know aren't perfect.
> One: syncing real songs. On synthetic tracks I can prove it's within a fraction of a
> millisecond, but on real hip-hop my own detector isn't precise enough to prove five, so
> that one needs ears.
> Two: busy mixes in part three. There's no source separation yet, so on a dense track the
> drum transcription drops to about half.
> Three: all my numbers are on synthetic tracks. The scorers will take the real test pack
> as soon as I have it."

---

## 8. Goodbye (9:00–9:15) · camera

**SAY:**
> "You can run the whole thing with `docker compose up`, and the README and DECISIONS have
> all the details. Thanks for watching. This was a really fun one to build!"

**DO:** stop the Loom.

---

## If something goes wrong mid-recording

| What happens | What you do | What you say |
|---|---|---|
| A1 shows a red "Unable to allocate" error | Deck A → LOAD TRACK → **A7** instead (then **A4** on deck B for sync; the numbers will differ a bit, which is fine) | "Let me grab another one." |
| Analysis takes a long time | Keep talking about the grid fit | "It's crunching on the CPU, give it a second." |
| Chat reply says "offline rule parser" | Carry on | "Quota's out, so that's the fallback kicking in." |
| Page doesn't load at all | Message Claude "restart the app", then refresh | (pause the recording) |
| You stumble on a line | Keep going | Nothing. It sounds human. |
