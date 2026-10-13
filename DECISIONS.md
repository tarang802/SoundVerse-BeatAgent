# Decisions

The main design choices, roughly in order of how much they matter. For each one: what I
considered, what I picked and why, what it costs, and what would make me revisit it.
Numbers come from the result files in `evals/results/`.

## 1. Beat tracking: Beat This! plus my own grid fit

I looked at librosa's tracker (the baseline the brief names), madmom (strong, but hard to
install on Windows with Python 3.11) and Beat This! (CPJKU, 2024, the `small0`
checkpoint on CPU). I went with Beat This! for raw beats and downbeats, and put a
deterministic grid fit on top of it.

The tracker's beats sit on roughly 20 ms frames and wobble a little. That's fine for a
beat F-score but not for a DJ grid, where the brief points out that 20 ms of drift makes
kicks flam. So the fit cleans the tracked beats, fits a straight line (constant tempo) or
a few line segments (tempo changes), refines the phase against spectral flux, and only
falls back to a variable grid when neither fits. On the 6-minute A1 this takes the worst
beat from 377 ms (raw tracker) to 1.92 ms.

**Tempo steps.** The tracker smooths over sudden jumps such as 128 → 100 BPM within one
beat. For those I estimate the local tempo from the onsets, find flat sections, fit each
one separately and place the change at the onset where it actually happens. A piecewise
grid only replaces the variable one if it fits the onsets at least as well. All 30
robustness cases now pass, including the two step cases.

**Live drift.** A band playing without a click (A3: tempo wandering between 106 and 118
BPM, with about 12 ms of jitter per hit) isn't a straight line, and snapping each beat to
its own hit just copies the jitter. Two approaches failed first: stitching tracked beats
into ramps made things worse, and a smoothed-residual test also fired on A1's drumless
breakdown. What worked was a running median of each beat's offset from its onset over
25 beats, followed by a quadratic smoother. The grid follows that curve only when at
least half the beats are more than 5 ms off. On A3, that took beats within 10 ms from 86%
to 98%. Steady clips and all 12 real tracks never trigger it. I also tested on 8 new
drifting takes that weren't used for tuning. Most of them took the variable path, so
variable grids are now smoothed the same way, with outliers removed.

*Would revisit if:* annotated live recordings showed a better approach, or a tracker
followed abrupt changes on its own.

## 2. Tempo octave: a convention, with an override

Whether a track is 70 or 140 BPM is a counting convention, not a measurement error. The
grid reports tempo between 80 and 160 BPM (compound meters excepted) and keeps the
tracker's downbeat phase when halving. Each deck has ½× / 2× buttons. When syncing, B is
tried at ×1, ×2 and ×0.5, and the smallest stretch wins.

*Would revisit if:* Soundverse's DJ product already has a house convention.

## 3. Indexing conventions

- Beats are grid lines counted from 0. A beat range excludes its end: "beats 16-32" is 16 beats.
- Bars are regions counted from 1, and bar ranges include both ends: "bars 9-12" is 4
  bars. Bar 0 is a pickup.
- Every beat in the grid JSON carries its time, beat index, bar and position in the bar.
  The conventions are also written into the file itself.
- "Repeat it twice" means two plays in total (the original plus one copy). The agent
  states this reading in its reply.

## 4. Two-track sync: Rubber Band R3 plus pasted kicks

Stretching B beat by beat (Rubber Band R2) gave tight kicks but a noticeably detuned bass.
Stretching it in one piece with R3 keeps the pitch exact but softens transients. So B is
stretched in one piece with R3, and then B's own drum hits are pasted back unchanged at
A's beats: 40 ms before to 30 ms after each beat, snapped to the real onset in both
tracks. A is the master and is never changed.

On the brief's case (A7 onto A1 at bar 33), the median kick offset is 0.05 ms and the
worst is 0.23 ms. A separate suite of 19 cases with exact ground truth all pass. It covers
±20% stretch, tempo octaves, 48 kHz stereo, a drifting take, a tempo change inside the
overlap, and 30, 60 and 120-second windows. Sync is refused when the stretch would exceed
±25%, the meters differ, or there isn't enough audio.

What this doesn't prove is sync on real hip-hop. There, my kick detector reads about
20 ms, but it reads 12–17 ms even when a track is synced with itself, which is perfect by
construction. So the detector can't settle it, and those renders have to be judged by ear.

## 5. Edit engine: an edit list over the source

Edits are typed operations (`remove`, `keep_only`, `repeat`, `move`, `copy`, `fade`,
`gain`). Each one resolves against the current grid into pieces of the source audio with
gain envelopes. The edited track's grid is the source grid mapped through those pieces,
never re-tracked, so the next command refers to what the user actually hears.
Undo and redo are a stack of states, and the output keeps the input's sample rate and
bit depth.

Splices use an 8 ms crossfade that ends 2 ms before the beat, so the next transient plays
untouched. On a drumless passage, a hard cut at the same point measures 25–32 dB above
the source, while the engine's splice measures 0.0–0.1 dB.

## 6. The LLM interprets; code does everything else

Gemini returns a structured plan (a pydantic schema: an action plus typed operations).
Code validates each operation, resolves it on the measured grid, states the exact bars,
beats and seconds before applying it, and refuses with a reason when something doesn't
add up.

Several guards came from things the LLM actually did during testing:

| What the LLM did | Guard |
|---|---|
| Rewrote "remove bar -2" as bar 1 and applied it | negative or fractional bar/beat numbers are refused before the LLM is called |
| Clamped "60 dB louder" to +24 dB and applied it | the user's number is kept; the LLM only picks louder or quieter |
| Removed all 186 bars for "remove everything" | any edit that would leave less than 2 s becomes a proposal that needs a "yes" |
| Proposed a range only in prose, so "yes" did nothing | one repair round asking for the operations |
| Refused "beat 16-32" as fractions | one repair round when the LLM claims fractions that the code didn't find |

Subjective requests like "chop out the boring part" are answered from measurements of
loudness and drum activity per section. The agent proposes a specific range with the
numbers behind it and waits for a yes. A clarifying question never changes the audio.

Free-tier quotas are per model per day, so the client falls back through several Gemini
models and then to a deliberately narrow offline rule parser. Every reply says which one
answered.

## 7. Measuring clicks

My first click metric compared the output peak around a splice with the loudest moment
near it in the source. A nearby hi-hat could outweigh a real click: a hard cut through a
bass note scored −4.7 dB, which reads as clean. I replaced it with position-matched
windows, scaled by the edit's own gain, and tested it against splices of known quality
(`tests/test_click_metric.py`). One limit remains: a click at the exact moment a drum hit
starts is masked, and it's masked to the ear too.

## 8. Part 3: score in beats, models for transcription, a fixed page template

- **Score** (`beatagent/recreate/score.py`): tempo, meter, bars, tracks, and notes with
  start and duration in beats. Working in beats makes "play it at 140" a single number
  and keeps analysis separate from playback.
- **Analysis** (`beatagent/recreate/transcribe.py`):
  - Tempo comes from Part 1, run on the loop repeated to at least 20 s, because a 2-bar
    loop is too short to track. When the clip is a whole number of bars, its exact length
    is used instead.
  - Drums come from per-band onsets on the percussive part.
  - Pitched notes come from Basic Pitch (ONNX, no TensorFlow), with overtone ghosts and
    drum body tones filtered out.
- **Page** (`beatagent/recreate/webaudio.py`): code fills in a fixed template, and the LLM
  never writes JavaScript. Generated code running in someone's browser could do anything,
  so I didn't want that. The score is embedded as inert JSON, the page has a strict CSP,
  loops are scheduled sample-exactly, and there's a mixer per track.
- **Offline render:** Playwright drives Edge or Chrome and runs the same scheduling code
  in an `OfflineAudioContext`, so the evaluated audio is what the page really plays.
- **Chat changes** ("mute the drums", "play it at 140") follow the Part 2 rule. The LLM
  returns typed score operations, and code checks track names and ranges, applies them all
  or none, says what changed and regenerates the page.

The cost is that there's no source separation. Demucs is large and slow on CPU, so dense
mixes like P4 come out weak.

I tuned the drum thresholds on a separate development set of 12 synthetic loops, with
different tempos, patterns and bass levels (`evals/part3/make_p3dev.py`), never on P1–P4.
The best alternative only raised dev drum F by 0.006, and a "hat on a kick" rule made it
worse, so I left the thresholds alone (`evals/results/part3/drum_tuning_dev.json`). The
remaining drum errors are mostly kicks under a loud bass line. Fixing those needs a
trained drum classifier, not better thresholds.

## Assumptions

- **The test pack wasn't provided.** Every target number is measured on synthetic clips
  built from the brief's descriptions, and labelled as such. The scorers accept common
  annotation layouts, so the real pack can be dropped in (`tests/test_ref_formats.py`).
- "Cut beat 16-32 and repeat it twice" means take beats 16–32 and play them twice in place.
- "Cut beat 0 to beat 8" keeps any audio before beat 0, and the reply says so; "from the
  start" includes it.
- "Delete bars 200-210" on a 90-bar track is refused, and the reply gives the real bar range.
- "Chop out the boring part" gets a measured proposal (quiet, drumless bars) and waits for a yes.
- Compound meters count dotted quarters as beats. 6/8 versus 12/8 is decided from the kick
  accent: if the beat two after the strongest is about as strong, the bar is 6/8. The
  tracker alone called A6 12/8 (downbeat F 0.67); with the accent check it's 6/8 (1.00),
  and no other clip changed.
- In "sync A7 to A1", A1 is the master and is never stretched. The overlap defaults to 60 s.

## Not built

- A public deployment (optional in the brief).
- Source separation and multi-instrument transcription (Demucs, MT3).
- Note-level edits to a recreation, such as "change the bassline". Chat covers tempo,
  mute, solo, level, brightness, transpose and oscillator.
- JAMS annotation input (plain-text layouts only).
