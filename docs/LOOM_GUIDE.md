# Loom guide (~8–9 min)

Talk it, don't read it. The lines in quotes are a feel to aim for, not a script.
**Bold** marks what to click; `code` marks what to type.

**Before you hit record:** the app is at http://localhost:8000 in Chrome. Press Ctrl+Shift+R,
close heavy tabs, and put the camera bubble in a corner that doesn't cover the decks.

---

## 1. Hello (30 s) — camera, then screen

> "Hey! I'm Tarang, and this is Beat Agent, my take on the Soundverse assignment.
>
> The idea: you give it a track and talk to it like you'd talk to a producer friend. It
> finds the beat, lines two tracks up, edits by chat, and can even turn a short loop back
> into code that plays in the browser.
>
> One thing I was strict about: the AI only figures out what you *mean*. Every number you'll
> see, like tempo, beats and timing, comes from actually measuring the audio. So it doesn't
> make things up.
>
> Heads-up: the official test tracks weren't included, so I built my own synthetic ones
> where I know the exact answers. That way I could actually check my work. Alright, let's go."

## 2. Finding the beat (1 min)

- Deck A → **LOAD TRACK → A1**
- > "This is a six-minute house track. While it thinks: a neural tracker finds the beats,
  > then my code fits a grid on top so it can't drift."
- Point at BPM, 4/4, confidence:
  > "124 BPM, 4/4, confident. Over six minutes the worst beat is 2 ms off. The tracker on
  > its own was off by 377, so the grid fit is doing real work."
- **CLICK OFF → ON**, **► PLAY**, let it run ~5 s:
  > "The easiest way to trust a grid is to hear it. Every beat clicks, the downbeats are
  > accented, and you can hear it sitting right on the kick."
- **PAUSE**, **GRID JSON**:
  > "And the grid as data: every beat with its bar and position."

## 3. Syncing two tracks (1 min)

- Deck B → **A7**
- SYNC: A bar **33**, B bar **1**, seconds **60** → **SYNC ⇄**
- > "Say I want A7 to come in at bar 33 of A1. A7 is at 128 and A1 is at 124, so B gets
  > stretched about 3% with the pitch kept. A is the master, and I never touch it."
- Point at the offset numbers:
  > "Kicks line up within 0.05 milliseconds on average. The target was 5, so I'm happy."
- **► PLAY**, then **B ONLY**, then back to **MIX**:
  > "Here's the mix, here's B on its own, and back to both."

## 4. Editing by chat (2 min)

Open **EDIT** on deck A.

- `cut beat 16-32 and repeat it twice`
  > "Notice it doesn't just do it. It tells me exactly what it understood: which beats,
  > which bars, how many seconds. Then it edits, and the grid follows along."
- `undo`
  > "Changed my mind."
- `remove bar -2`
  > "Now let's be difficult. Bar minus two doesn't exist. Fun fact: during testing, an LLM
  > quietly turned this into bar 1 and deleted it. So now code checks the numbers before the
  > AI even sees them."
- `remove everything`
  > "This one would wipe the whole track. It won't do that on one sentence; it asks first."
- `no`
- `chop out the boring part`
  > "My favourite one. 'Boring' is subjective, so instead of guessing, it measures loudness
  > and drums per section, finds the quiet drumless bit at bars 81 to 96, and proposes it."
- `yes`
  > "And only now does it actually cut."

## 5. Turning audio into code (1.5 min)

- Scroll to RECREATE → **P2** → **RECREATE**
- > "Part three: take a short loop, figure out what's being played, and rebuild it as a web
  > page that synthesizes it live."
- Play **ORIGINAL** for ~5 s, then **Play** in the generated page:
  > "That's the original… and that's the page. No samples inside: the drums and the bass are
  > generated from a score. Tempo comes from part one, drums from onsets, bass notes from
  > Basic Pitch."
- Drag the **bass** slider up:
  > "Each part has its own fader."
- Chat: `mute the drums`, then `play it at 140`, then **Play**:
  > "Same rule as the editor: the AI turns my words into small structured changes, and code
  > applies them and rebuilds the page."

## 6. Under the hood (1.5 min) — VS Code

- `beatagent/`
  > "One package per part: core, beats, editing, recreate. Plus a shared core."
- `beats/analyze.py`
  > "Beat tracker in, then cleanup and line fitting, so the grid can't wander."
- `editing/agent.py`
  > "This is the line between the AI and the code. Pretty much every guard in here exists
  > because the model did something weird in testing."
- `recreate/webaudio.py`
  > "Fixed template. The AI never writes JavaScript, which matters when the output runs in
  > someone's browser."
- `evals/` and `tests/`
  > "And this is how I know any of it works: 11 test suites, 59 chat cases, robustness
  > tests, plus a GitHub check that builds the Docker image from scratch."

## 7. What I'd fix next (45 s) — camera

> "Three things I know aren't perfect.
>
> One: syncing real songs. On synthetic tracks I can prove it's within a fraction of a
> millisecond. On real hip-hop my own detector isn't precise enough to prove 5 ms, so
> that one needs ears.
>
> Two: busy mixes in part three. There's no source separation yet, so on a dense track the
> drum transcription drops to about half.
>
> Three: everything is measured on my synthetic tracks. The scorers will take the real
> test pack as soon as I have it."

## 8. Bye (15 s)

> "You can run the whole thing with `docker compose up`. The README and DECISIONS have the
> details. Thanks for watching, this was a really fun one to build!"

---

**If something goes sideways while recording:**
- **A1 throws a memory error:** load A7 on deck A instead and say "let's use this one".
- **A reply mentions the "offline rule parser":** the Gemini quota ran out. Say "that's
  the fallback kicking in" and carry on.
- **Analysis is slow:** "it's running on the CPU, give it a second" fills the gap.
