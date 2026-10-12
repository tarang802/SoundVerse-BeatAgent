# Beat Agent

A chat-driven audio agent for the Soundverse Applied AI Engineer assignment. It does
three things with a piece of audio:

1. **Beat alignment and sync** — tempo, beats, downbeats, meter and a DJ-grade grid;
   a click track to audition it; beat-matching track B onto track A (pitch preserved).
2. **Chat editing on the grid** — "cut beat 16–32 and repeat it twice" becomes typed
   edit operations that deterministic code checks, echoes and applies without clicks.
3. **Audio to playable code** — a short loop becomes a score (JSON) and a
   self-contained HTML page that re-plays it with Web Audio synthesis.

**The core rule holds throughout: the LLM interprets; code measures and executes.**
Tempo, beats, notes and offsets come from signal processing and models (Beat This!,
Basic Pitch), never from the LLM.

> **Test data.** The assignment's test pack (A1–A7, P1–P4) was **not provided**. Every
> brief-target number below is measured on **synthetic substitutes** built to the
> brief's descriptions, with exact answer keys. They are labelled as such everywhere and
> are never presented as the official pack. Drop the real pack into `data/testpack/`
> and `python run.py eval --pack testpack` scores it (common annotation layouts accepted).

## Status

| Part | Requirement | Status |
|---|---|---|
| 1 | Beat grid JSON (tempo, every beat with index/bar/position, downbeats, meter, confidence, constant/variable) | ✅ works |
| 1 | Click audition export | ✅ works |
| 1 | Two-track sync, 60 s render, pitch preserved, downbeats aligned | ✅ works |
| 1 | mir_eval accuracy report | ✅ works (synthetic pack) |
| 1 | Targets | 🟡 4 of 5 met on the synthetic pack; downbeat F misses on A6 (6/8 vs 12/8) — see below |
| 2 | Typed edit ops (remove, keep only, repeat/loop, move, copy/insert, fade, gain) | ✅ works |
| 2 | Echo before cut, clean splices, format kept, grid stays true, undo/redo, graceful errors | ✅ works |
| 2 | Preview player with waveform + grid; variable-tempo edits | ✅ works |
| 3 | Score JSON from analysis; one-file page with play/stop/loop, synthesis only, no network | ✅ works |
| 3 | Drums + a pitched part; offline render in a headless browser; note-F report vs MIDI | ✅ works (synthetic P1–P4) |
| 3 | Per-track mixer | ✅ works |
| 3 | Chat changes to the recreation ("mute the drums", "make the bass brighter", "play it at 140") | ✅ works (typed score ops, validated by code; page regenerated) |
| — | Public deployment | ❌ not built (optional) |

Full, generated numbers: [`evals/results/COMPLETION_REPORT.md`](evals/results/COMPLETION_REPORT.md).

## Quick start

**Prerequisites** (what it was developed and tested on): Windows 11, Python **3.11**,
Microsoft Edge or Google Chrome (Part 3's offline render drives the installed browser;
no browser download needed), a Gemini API key for chat (optional — everything else, all
deterministic tests and a narrow offline chat parser work without one). Node 18+ only if
you want to rebuild the UI; the built UI is committed in `frontend/dist`.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps basic-pitch==0.4.0
copy .env.example .env            # then put your GEMINI_API_KEY in .env (never commit it)
```

`basic-pitch` is installed with `--no-deps` on purpose: its default dependencies pull
TensorFlow 2.15 (~400 MB), but the bundled ONNX model only needs `onnxruntime`.

*Verified:* a fresh virtual environment built from exactly these two install lines (with
`uv`, which resolved packages from its local cache) passes all 10 test suites, 201
checks. A different machine with an empty package cache was not tested.

### The one command

```bash
python run.py demo
```

Downloads the Beat This! checkpoint (8.4 MB, resumable), builds both synthetic test
packs, runs every deterministic test suite, the Part 1 and Part 3 evals, then starts the
app on **http://localhost:8000**. Open it in **Chrome or Edge** (the RECREATE panel runs
the generated page in a sandboxed iframe; Claude's built-in preview pane cannot run
scripts there — verified, see Known issues).

Individual steps:

| Command | What it does |
|---|---|
| `python run.py app` | start the web app |
| `python run.py test` | all 10 deterministic test suites (no API key needed), summarised |
| `python run.py e2e` | the real app in headless Edge/Chrome, the demo path end to end: load, grid/click downloads, sync, chat edits with pushback + undo, recreate + chat (offline parser, no key; ~3 min) |
| `python run.py fuzz` | 421 random edits with invariant checks after each (~5 min) |
| `python run.py eval` / `syncval` | Part 1: mir_eval report on A1–A7 / sync comparison + `PART1_REPORT.md` |
| `python run.py p3eval` | Part 3: transcribe P1–P4, render pages offline, score vs MIDI |
| `python run.py chat` | Part 2 with the LLM: 59-case chat suite + the brief's 6 sessions (needs a key) |
| `python run.py report` | regenerate `evals/results/COMPLETION_REPORT.md` from the result files |
| `python run.py testpack` / `p3pack` | rebuild the synthetic A1–A7 / P1–P4 packs |

## Demo workflow (what the Loom shows)

Start with `python run.py app`, open http://localhost:8000 in Chrome or Edge.

1. **Grid + click audition** — deck A: load `A1`; the waveform shows the grid and BPM;
   turn on CLICK and play (accented downbeats); download `grid.json`.
2. **Sync** — deck B: load `A7`; SYNC panel: A bar 33, B bar 1, 60 s → SYNC; play the
   mix, then A and B alone; the measured onset offsets are shown.
3. **Chat edits** on deck A: "cut beat 16-32 and repeat it twice" (exact echo), "fade
   out the last 8 beats", "undo". **Pushback:** "remove bar -2" (refused by code),
   "remove everything" (asks for a yes), "chop out the boring part" (measured proposal,
   then "yes").
4. **Recreation** — RECREATE panel: `P2` → RECREATE; play the original, then Play in the
   generated page next to it; chat "mute the drums", "play it at 140".
5. **Code tour** — `beatagent/analyze.py` (grid), `edl.py` (edits), `agent.py` + `llm.py`
   (the LLM boundary and its guards), `transcribe.py` → `score.py` → `webaudio.py`;
   `tests/` and `evals/`.
6. **Top three known problems** — see Known issues below.

## Architecture

```
audio ─► beatagent/analyze.py ─► BeatGrid (grid.py) ──► click.py      (audition)
          Beat This! + grid fit                     ├─► sync.py       (Rubber Band R3 + kick patches)
                                                    └─► session.py ◄─ edl.py (typed ops, pieces, crossfades)
chat ─► agent.py ─► llm.py (Gemini → Plan) ─► validators ─► session.apply ─► echo / undo / redo
                    └ offline rule parser (labelled) when the LLM is unavailable
clip ─► transcribe.py (HPSS, band onsets, Basic Pitch) ─► score.py (Score JSON)
        ─► webaudio.py (fixed template) ─► page.html ─► evals/part3_render.py (Edge, OfflineAudioContext)
app/server.py (FastAPI) serves the API and frontend/dist (React + TypeScript + Vite)
```

* `beatagent/` — library code: analysis, grid, sync, edit engine, agent, LLM, Part 3.
* `app/server.py` — HTTP API used by the UI (decks, chat, sync, recreate).
* `frontend/` — two DJ decks with waveform + grid, chat per deck, SYNC panel, RECREATE panel.
* `evals/` — every harness and its results (`evals/results/`).
* `tests/` — deterministic tests (no key, no network).

**What the LLM does and does not do.** It turns one chat message into a structured plan
(`action` + typed ops). Code validates every op, resolves bars/beats on the measured
grid, echoes the exact range in bars, beats and seconds, and refuses with a reason. The
LLM never sees audio, never computes times, never writes code that runs. Guards exist
for each way it misbehaved in testing (see `DECISIONS.md` §6).

## Conventions (how commands are read)

* **Beats are 0-based grid lines**; a beat range is half-open: "beats 16–32" = 16 beats.
* **Bars are 1-based and inclusive**: "bars 9–12" = 4 bars. Bar 0 is a pickup.
* "Repeat it twice" / "loop it two times" = **2 plays in total**; "cut X and repeat it"
  repeats X in place. Every reply states the reading it used.
* Subjective requests ("chop out the boring part") get a **measured proposal** and wait
  for "yes"; nothing changes until then. So do edits that would leave < 2 s of audio.
* Tempo is reported in the 80–160 BPM range; ½× / 2× buttons override it per deck.

## How things are measured

* **Beats / downbeats:** `mir_eval` F-measure at ±70 ms on the whole track (no 5 s trim),
  plus AMLt; drift = |grid − reference| for every reference beat. Only where a reference
  annotation exists (synthetic pack: exact by construction).
* **Sync:** ground truth from the render itself — every pasted kick is a verbatim copy of
  B placed at a known position and both tracks' true kick times are known, so the B−A
  offset is arithmetic, not a detector reading. Detector measures are reported as
  secondary (they would be circular after onset snapping).
* **Clicks:** >6 kHz energy at each splice vs the same windows in the source, scaled by
  the edit's own gain; the metric itself is validated against hard cuts of known
  quality (`tests/test_click_metric.py`).
* **Chat:** each case checks the resulting edit state against the intended edit (or, for
  refusals and questions, that nothing changed and the reply states the measured fact) —
  request handling, meaning and audio are checked separately, not just an `ok` flag.
* **Part 3:** `mir_eval.transcription` onset-only note F (±50 ms, pitch ±50 cents) vs the
  reference MIDI, drums and pitched parts separately; render timing = onsets of the
  offline render vs the original over two loop passes (2.9 ms frames).

## Results (synthetic substitutes; regenerate with the commands above)

**Part 1** — A1–A7 (`evals/results/part1_substitute_beat_this/results.md`)

| Target | Result |
|---|---|
| Beat F ≥ 0.90 (steady clips) | 1.00 on A1, A2, A4, A6, A7 (A3 0.99, A5 1.00) |
| Downbeat F ≥ 0.80 | 1.00 on A1, A2, A4, A5, A7; **A6 0.67** (counted 12/8 vs a 6/8 reference) |
| Drift ≤ 10 ms over 6 min (A1) | **1.92 ms** max (raw tracker: 377 ms) |
| Sync kicks median ≤ 5 / max ≤ 15 ms (A7 → A1 @ bar 33) | **0.05 / 0.23 ms** |
| 6-minute track < 60 s on CPU | **11–34 s** across runs on the dev laptop (load-dependent; the `run.py demo` run was the slow one) |

Plus: 29/30 robustness cases (tempo 60–200, grooves, meters, tempo changes, levels,
formats, silence/noise refused with a message); 19/19 sync cases against exact ground
truth; 12 real hip-hop tracks: 11/12 agree with an independent tempo estimate (no answer
key, so consistency only).

**Part 2** — chat suite on the final prompt **59/59**, all answered by Gemini (checked
against the intended edit, not just "ok"); the brief's six sessions 10/10 turns as
intended; 421 random edits: 414 applied, 7 refused cleanly, **0** failures of length,
order, grid, clicks, undo or redo.

**Part 3** — P1–P4 (`evals/results/part3/results.md`)

| Clip | Tempo | Drums note F | Pitched note F | Render onset F |
|---|---|---|---|---|
| P1 2-bar drum loop | exact | 0.93 | — (none, correct) | 0.89 |
| P2 drums + bass | exact | 0.71 | 0.90 | 0.86 |
| P3 piano chords + melody | exact | — (none, correct) | 0.96 | 0.89 |
| P4 dense mix | exact | 0.51 | 0.68 | 0.88 |

All four pages: 0 network requests, no page errors, play/stop working in headless Edge.

**Part 3 on real music** (8-bar excerpts of three real hip-hop tracks, no reference MIDI,
like the brief's P4; `evals/results/part3_real/results.json`, numbers only): pipeline
clean on all three, tempo matches Part 1, render rhythm vs the original onset F
**0.60–0.78**. Without source separation the vocals are transcribed into the "keys" track.

Where the drum transcription goes wrong (P2): kicks under a loud bass line are missed
or their click is heard as a hi-hat; some hats next to snares are labelled snares.

## Known issues (the top ones)

1. **A3-style tempo ramps and abrupt tempo jumps.** A short ramp inside a steady track
   is approximated by a straight line (worst beat 108 ms, 86 % of beats within 10 ms);
   a one-beat jump like 128 → 100 BPM is smoothed over by the tracker.
2. **Real-music sync is not proven to ≤ 5 ms.** The kick detector cannot resolve it on
   hip-hop (it reads 12–17 ms even on a perfect self-sync). Listen to the renders.
3. **Part 3 on dense mixes.** No source separation, so P4 is weak (drums 0.51, pitched
   0.68); drum classes come from band rules, not a trained classifier.

Also: A6 6/8-vs-12/8 counting; the click metric cannot see a click hidden under a kick
attack (proven clean on drumless material instead); only hip-hop real tracks were
available; the app's built-in preview pane (in Claude's desktop app) does not run
scripts in sandboxed iframes, so use Chrome/Edge for the RECREATE panel.

## Security

* Secrets only in `.env` (git-ignored); `.env.example` has no values. The key is used
  server-side only; the frontend never sees it.
* Uploads: names sanitised (a `:` once hid the audio in an NTFS stream), path
  components dropped, only WAV/MP3/FLAC accepted, decode errors never reveal server paths.
  Opening by path is limited to files inside the project; the server listens on
  localhost only.
* Generated pages: built by code from a fixed template; the score is inert JSON with
  `<`, `>`, `&` escaped, and track names are inserted as text, never as markup
  (`tests/test_webaudio_page.py`); `Content-Security-Policy: default-src 'none'`; shown in
  a sandboxed iframe.
* The LLM's output is data validated against a schema; it is never executed.

## More

* [`DECISIONS.md`](DECISIONS.md) — choices, trade-offs, assumptions, what was not built.
* [`AI_USAGE.md`](AI_USAGE.md) — AI at runtime and in development, and where it was wrong.
* [`evals/results/COMPLETION_REPORT.md`](evals/results/COMPLETION_REPORT.md) — every
  suite, every case, the bug ledger.
