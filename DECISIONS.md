# DECISIONS

The decisions that shaped Beat Agent, in the order they matter. Each lists the options
considered, the choice, the trade-off, and what would change my mind. Numbers quoted
here come from the result files in `evals/results/` (see `COMPLETION_REPORT.md`).

## 1. Beat tracking: Beat This! as the tracker, our own grid on top

**Options.** librosa's tracker (baseline the brief names), madmom (strong, but painful to
install on Windows / Python 3.11), Beat This! (CPJKU, 2024; `small0` checkpoint, CPU).

**Choice.** Beat This! for raw beats and downbeats, then a deterministic **grid fit**:
clean the tracker's beats (local period, phase-jump and tempo-change re-anchoring),
fit a line (constant tempo) or piecewise lines (tempo changes), refine the phase on
spectral flux, snap and refit; only if neither fits, keep a *variable* grid of tracked
beats snapped to onsets (and re-snapped where a beat is far from where its neighbours
predict).

**Why.** The tracker's beats sit on ~20 ms frames and wobble; a DJ grid must not drift
(the brief: 20 ms over five minutes makes kicks flam). Fitting turns a 377 ms raw worst
case on the 6-minute substitute A1 into 1.92 ms.

**Trade-off.** The tracker smooths over *abrupt* large tempo jumps (e.g. 128 → 100 BPM in
one beat): the one robustness case that fails. A short tempo ramp inside a steady track
(substitute A3's opening) is approximated by a line: worst beat 108 ms there. A splice
of tracked beats into ramps was tried and **reverted** — it made A3 worse overall.

**Would change my mind.** Real annotated live recordings showing the variable path beats
the piecewise one; or a tracker that follows abrupt changes.

## 2. Tempo octave: a convention plus an override, not a guess

Half/double tempo is a convention, not an error (a track can be counted 70 or 140). The
grid reports tempo in **80–160 BPM** (compound meters excluded), keeps the tracker's
downbeat parity when halving, and the UI has ½× / 2× buttons per deck. Sync tries B at
×1, ×2 and ×0.5 and picks the smallest stretch. *Would change:* a house convention from
Soundverse's DJ product.

## 3. Indexing conventions (the brief asks to document them)

* **Beats are 0-based grid lines**; a beat range is half-open `[start, end)`: "beats 16–32"
  = 16 beats.
* **Bars are 1-based regions**, ranges inclusive: "bars 9–12" = 4 bars. Bar 0 = pickup.
* Every grid JSON stores, per beat: time, beat index, bar, position in bar.
* "Repeat it twice" / "loop it two times" = **2 plays in total** (original + 1 copy); the
  agent states this reading in every reply.

## 4. Two-track sync: Rubber Band R3 plus verbatim kick patches

**Options.** Per-beat stretching (R2) — tight kicks but audibly detuned bass; one-piece
R3 — pitch exact but smeared transients; R3 + patches.

**Choice.** B is stretched in **one piece with Rubber Band R3** (pitch preserved), then
B's own drum hits are pasted **verbatim** at A's beats (40 ms before / 30 ms after each
beat), each snapped to the actual onset in both tracks. A, the master, is never changed.

**Measured.** Brief sync (A7 → A1 at bar 33): median 0.05 ms, max 0.23 ms. Sync suite with
exact ground truth: 19/19 (±20 % stretch, octaves, 48 kHz stereo, drifting take, tempo
change inside the overlap, 30/60/120 s, refusals). Refusals: > ±25 % stretch, meter
mismatch, too little audio.

**Not proven.** Real hip-hop pairs: the kick detector reads ~20 ms median, but it reads
12–17 ms on a self-sync that is perfect by construction, so it cannot resolve this
music. Listening is the check there.

## 5. Edit engine: an edit decision list over the source, rendered on demand

Edits are **typed ops** (`remove`, `keep_only`, `repeat`, `move`, `copy`, `fade`, `gain`)
resolved against the current grid into **pieces** of the source with gain envelopes.
The grid of the edited track is the source grid *mapped exactly* through the pieces —
never re-tracked — so the next command addresses what the user hears. Undo/redo is a
stack of states. Output keeps the input's sample rate and bit depth.

**Splices.** An 8 ms crossfade ends **2 ms before** the beat, so the incoming transient
plays untouched. Validated: on a drumless passage a hard cut at the same point scores
+25 to +32 dB above the source, the engine's splice +0.0 to +0.1 dB.

## 6. The LLM interprets; code measures, validates and executes

Gemini returns a **structured `Plan`** (pydantic schema: action + typed ops). Code then:
validates every op, resolves it on the measured grid, echoes exact bars/beats/seconds
before applying, and refuses with a reason. Guards added because the LLM did these in
testing:

| LLM behaviour seen | Guard (deterministic) |
|---|---|
| "remove bar -2" rewritten to bar 1 and applied | negative / fractional bar or beat numbers refused **before** the LLM |
| "60 dB louder" clamped to +24 dB and applied | the user's stated dB magnitude is kept; the LLM only picks up/down |
| "remove everything" → all 186 bars removed | an edit leaving < 2 s or < 5 % of the track becomes a proposal needing "yes" |
| proposal only in prose ("did you mean bars 81–96?") so "yes" did nothing | one repair round asking for the ops |
| "beat 16-32" refused as "fractions" | one repair round when the LLM claims fractions the code found none of |

Subjective requests ("chop out the boring part") are answered from **measured sections**
(loudness and drum activity per bar range): the agent proposes a concrete range with the
numbers and waits for "yes"; a clarification never changes audio.

**Availability.** Free-tier quotas are per model per day, so the client falls back through
several Gemini models, then to a deliberately narrow **offline rule parser**; every reply
says which answered.

## 7. Measuring clicks: a metric that was itself tested

The first click metric compared a ±15 ms output peak against the loudest moment within
±50 ms of the source; a nearby hi-hat out-shouted real clicks (a hard cut through a bass
note scored −4.7 dB). Replaced by **position-matched windows** (transition 12→1 ms before
the splice, junction ±1 ms) scaled by the edit's own gain, and validated against splices
of known quality (`tests/test_click_metric.py`). Known limit: a click at the very instant
a drum hit starts is masked — to the ear as well.

## 8. Part 3: score in beats, transcription from models, page from a fixed template

* **Score** (`beatagent/score.py`): tempo, meter, bars, tracks, notes with start/duration
  in **beats**, pitch, velocity. Beats make "play it at 140" one number and keep analysis
  separate from playback.
* **Analysis** (`beatagent/transcribe.py`): tempo from Part 1 on the loop repeated to
  ≥ 20 s (a 2-bar loop is too short to track) and from the clip's exact length when it is
  whole bars; drums from per-band onsets on the HPSS percussive part (absolute level and
  spectral-flatness gates); pitched notes from **Basic Pitch** (ONNX, no TensorFlow) on
  the harmonic part, with overtone ghosts and drum body tones removed.
* **Page** (`beatagent/webaudio.py`): **code fills a fixed, reviewed template**; the LLM
  never writes JavaScript (generated code running in a user's browser could do
  anything). Score embedded as inert JSON; `Content-Security-Policy: default-src 'none'`;
  sample-exact loop scheduling; per-track mixer and tempo.
* **Offline render** with Playwright driving the installed Edge/Chrome (no browser
  download), running the *same* scheduling code in an `OfflineAudioContext`.
* **Chat changes** ("mute the drums", "play it at 140") follow the Part 2 rule: the LLM
  returns typed score ops, code checks track names and ranges, applies them to a copy
  (all-or-nothing), echoes what changed, and regenerates the page; undo is a score stack.

**Trade-off.** No source separation (Demucs was not installed: large and slow on CPU);
HPSS is weak on dense mixes (P4).

**Tuning without touching the test clips.** Drum thresholds were checked on a separate
synthetic development set (`evals/make_p3dev.py`: 12 loops with other tempos, patterns
and bass levels, two piano-only), never on P1–P4. The best alternative raised dev drum F
by only 0.006 (noise) and a "hat on a kick" rule lowered it, so nothing changed
(`evals/results/part3/drum_tuning_dev.json`). The remaining drum errors — kicks under a
loud bass line — need a learned drum classifier, not thresholds. *Would change my mind:*
a small labelled drum dataset to train one.

## Assumptions where the brief or a command was unclear

* **Test pack not provided.** All brief-target numbers are on **synthetic substitutes**
  built to the brief's descriptions (`evals/make_testpack.py`, `evals/make_p3pack.py`),
  never presented as the official pack. The scorer accepts common annotation layouts so
  the real pack can be dropped in (`tests/test_ref_formats.py`).
* "Cut beat 16–32 and repeat it twice" = take beats 16–32 and play them twice in place.
* "Cut beat 0 to beat 8" keeps any audio before beat 0 (stated in the reply; "from the
  start" includes it).
* "Delete bars 200–210" on a 90-bar track: refuse with the real bar range.
* "Chop out the boring part": propose the measured drumless/quiet section, wait for yes.
* A6 is counted in 12/8 (dotted-quarter beats); a 6/8 reference scores downbeat F 0.667.
* "Sync A7 to A1": A1 is the master and is never stretched; overlap 60 s by default.

## What I chose not to build

* A public deployment (optional in the brief).
* Source separation (Demucs) and MT3-style multi-instrument transcription.
* Chat edits of the recreation beyond mix / tone / tempo (no note-level editing such
  as "change the bassline"); the chat covers tempo, mute, solo, level, brightness,
  transpose and oscillator.
* JAMS annotation input (plain-text layouts only).
