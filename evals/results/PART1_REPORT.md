# Part 1 — beat alignment and sync: results

> **Everything below is measured on the SYNTHETIC SUBSTITUTE pack** (`data/substitute`, built by `evals/make_testpack.py`). The assignment's test pack (A1–A7 with reference annotations) was not provided, so **no assignment benchmark number is claimed**. Synthetic clips have clean transients and no mix/master, so these numbers are an upper bound on real-music performance.

Machine: this laptop's CPU (no GPU). Tracker: `beat_this:small0`. mir_eval F-measures are computed on the full clip (mir_eval's default 5 s trim is NOT applied).

## Targets

| Target | Brief | Result (synthetic) | Met? |
|---|---|---|---|
| Beat F-measure (±70 ms) | ≥ 0.90 on steady-tempo clips | min 1.0000 (A1) over A1, A2, A4, A6, A7 | ✅ |
| Downbeat F-measure (±70 ms) | ≥ 0.80 | 6/6 clips pass | ✅ |
| Grid drift, 6-min constant clip | every beat within 10 ms | A1 (362 s): max 1.92 ms, 100% within 10 ms | ✅ |
| Sync kick offsets | median ≤ 5 ms, max ≤ 15 ms | A7→A1 @ bar 33: median 0.05 ms, max 0.23 ms, 123/123 beats measured | ✅ |
| Speed | 6-min track < 60 s on laptop CPU | A1 (362 s): 14.65 s (this eval run, first clip so includes model load). Measured by hand on 2026-10-12 on this i7-13620H: 10.3 / 25.7 / 19.4 s in repeated runs; app with both decks analysing at once: 70-80 s each | ✅ (single track) |

## Beat grid per clip

| clip | duration_s | analysis_s | grid_type | tempo_bpm | time_signature | beat_F | beat_AMLt | downbeat_F | drift_max_ms | within_10ms | raw_beat_F | raw_drift_max_ms | confidence | librosa beat_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 361.5 | 14.65 | constant | 124.0 | 4/4 | 1.0 | 1.0 | 1.0 | 1.92 | 1.0 | 0.9738 | 377.42 | 0.615 | 0.0 |
| A2 | 157.4 | 6.61 | constant | 86.0 | 4/4 | 1.0 | 1.0 | 1.0 | 1.12 | 1.0 | 0.9956 | 33.02 | 0.966 | 0.6677 |
| A3 | 185.2 | 7.25 | variable | 117.982 | 4/4 | 0.9972 | 1.0 | n/a | 88.57 | 0.9778 | 0.9959 | 52.37 | 0.981 | 0.9903 |
| A4 | 145.9 | 5.62 | constant | 118.0 | 4/4 | 1.0 | 1.0 | 1.0 | 1.63 | 1.0 | 0.8571 | 56.95 | 0.938 | 0.0 |
| A5 | 140.5 | 5.47 | piecewise_constant | 128.001 | 4/4 | 1.0 | 1.0 | 1.0 | 2.08 | 1.0 | 0.9725 | 100.0 | 0.969 | 0.0 |
| A6 | 114.0 | 4.37 | constant | 68.0 | 6/8 | 1.0 | 1.0 | 1.0 | 1.83 | 1.0 | 0.9961 | 22.35 | 1.0 | 0.9961 |
| A7 | 183.3 | 7.49 | constant | 128.0 | 4/4 | 1.0 | 1.0 | 1.0 | 1.88 | 1.0 | 0.9961 | 67.5 | 0.995 | 0.0 |

`beat_AMLt` = mir_eval continuity score that also accepts double/half-tempo and off-beat counting (F low + AMLt high = right beats, different counting convention; both low = wrong grid). `raw_*` = Beat This! output before our grid fitting; `drift_*` = |grid − reference| per reference beat. A3 is variable-tempo by design, so the 10 ms drift target does not apply to it.

## Failure analysis by clip

- **A1**: Pass. Confidence is low (≈0.6) only because the tracker drops beats in the 16-bar drumless breakdown; the fitted grid keeps counting through it.
- **A2**: Pass. No tempo-octave error on the half-time groove (86 BPM, not 43/172).
- **A3**: Beat F 0.9972; grid type `variable` (a 112->118 BPM ramp, then ~118); worst beat 88.57 ms. An earlier variable-grid version had 73 ms worst; the current constant/piecewise tests (needed so real constant-tempo hip-hop is not mislabelled variable) model the drift as two constant sections instead. 12 ms random per-hit jitter is in the clip by design.
- **A4**: Pass after trimming: the tracker invented a 60 BPM pulse over the 47 s pad intro; beats with no onset evidence at the start/end are dropped. Assumption: the reference starts at the drum entry (true for our substitute; unknown for the real pack).
- **A5**: Pass. The tracker drifts toward the new tempo ~4.5 s early; the change point is re-chosen from onsets (found at bar 33, 64.498 s vs true 64.500 s).
- **A6**: **Downbeat FAIL (0.67).** 6/8 is reported as 12/8: Beat This! marks a downbeat every 4 dotted-quarter beats, so bars come out twice as long. Not fixed: telling 6/8 from accented 4/4 needs a meter model we did not build; a quick heuristic risked breaking 4/4.
- **A7**: Pass.

## Two-track sync (A7 → A1, overlap starts at A1's bar 33)

Four render methods were measured; the app uses `constant_attack`.

| method | measured | median_abs_ms | p95_abs_ms | max_abs_ms | pass | downbeat_kicks | join_click_max_db | pitch_cents | render_s |
|---|---|---|---|---|---|---|---|---|---|
| constant | 62/123 | 4.4 | 12.18 | 18.01 | False | 30/30 med 7.1 max 13.84 | n/a | -0.0 | 4.7 |
| beats | 62/123 | 0.11 | 0.17 | 0.25 | True | 30/30 med 2.38 max 6.42 | 22.8 | 10.92 | 10.5 |
| beats_attack | 87/123 | 0.11 | 0.61 | 1.05 | True | 30/30 med 2.34 max 5.26 | 13.9 | 10.92 | 4.7 |
| constant_attack | 123/123 | 0.05 | 0.17 | 0.23 | True | 30/30 med 0.23 max 0.79 | 1.2 | -0.01 | 4.5 |

- `constant`: one Rubber Band R3 stretch. Pitch exact, but R3 smears kick transients: downbeat kicks ~30 ms early. **Note the trap**: with the strict detector only 62/124 beats are measurable and their stats look nearly fine — the unmeasured ones are the worst.
- `beats` / `beats_attack`: per-beat slices. Kicks perfect, but R3 on 0.5 s slices **detuned a sustained 110 Hz tone by +11–12 cents** (measured inside single slices) → fails "pitch preserved".
- `constant_attack` (**used**): one R3 stretch for exact pitch + B's original audio pasted from 40 ms before to 30 ms after every beat, crossfaded, so each kick sits exactly on A's beat.
- Kick offsets are measured on the separate A and B stems with a spectral-flux onset detector validated against known beat times (0.5 ms p95 spread on A7). A kick-band detector was tried and **failed validation** (false kicks in pad sections), so it was removed.
- One B downbeat (~17.4 s into the render) sits right at the detector's 3x threshold: it is measured in some runs (124/124 in the app) and not in others (123/124 here). The table reports this run.
- Caveats: A and B use the same synthetic kick sound, which makes alignment and measurement easier than on real tracks; B with variable tempo is untested; about 70 ms around each kick plays at B's original speed.

## Real music (no reference annotations — consistency checks, NOT accuracy)

12 commercial hip-hop tracks (3 artists) found on the dev machine. No house/EDM/live-band recordings were available, so those genres are untested on real audio. Without answer keys, beat F / drift cannot be computed; instead:
- **indep tempo**: tempo from onset autocorrelation in 20 s windows (does not use the neural tracker), compared with the grid; windows without a drum pulse are not graded. Validated on the substitute pack (≤0.02 % on A1/A2/A5/A7; skipped exactly A1's drumless breakdown).
- **own-grid spread**: p10–p90 spread of this track's downbeat hits around its own grid lines, with a broadband attack detector and a low-frequency kick detector.

| track | BPM | grid | conf | indep tempo max diff (graded windows) | librosa tempo | downbeat attack spread | downbeat low-kick spread | analysis s |
|---|---|---|---|---|---|---|---|---|
| 12. Seedhe Maut - Naam Kaam Sheher [E].flac | 140.19 | variable | 0.499 | 0.36% (8/8) | 140.62 (agree) | 24.23 ms | 52.17 ms | 9.7 |
| 01. Seedhe Maut - Nanchaku [E].flac | 140.0 | constant | 0.51 | 0.38% (9/9) | 139.67 (agree) | 25.86 ms | 49.91 ms | 12.6 |
| 01. Seedhe Maut - Namastute.flac | 80.0 | constant | 0.955 | 0.36% (6/6) | 161.5 (librosa says DOUBLE) | 4.88 ms | 33.76 ms | 8.0 |
| 02. Seedhe Maut - Toh Kya [E].flac | 150.0 | constant | 1.0 | 0.15% (9/9) | 152.0 (agree) | 10.54 ms | 32.08 ms | 8.2 |
| 02 - BUN MASKA - Calm, DL91 ERA, Seedhe Maut.flac | 140.0 | constant | 0.355 | 0.07% (7/9) | 139.67 (agree) | 16.15 ms | 52.45 ms | 9.8 |
| 03 - RINGTONE - Calm, DL91 ERA, Bhaskar.flac | 138.0 | constant | 0.899 | 0.11% (12/13) | 139.67 (agree) | 1.04 ms | 54.56 ms | 16.7 |
| 01. Seedhe Maut - 11K [E].flac | 135.0 | constant | 0.695 | 0.1% (7/8) | 136.0 (agree) | 16.7 ms | 45.69 ms | 7.3 |
| 03 - Elevator Music - Encore ABJ, DL91 ERA.flac | 105.01 | constant | 0.533 | 2.26% (9/10) | 105.47 (agree) | 33.64 ms | 68.0 ms | 15.8 |
| 04 - Kya Mai Yaha - Encore ABJ, DL91 ERA.flac | 130.0 | constant | 0.867 | 0.06% (11/11) | 129.2 (agree) | 44.09 ms | 6.45 ms | 10.8 |
| 01 - ALL CAPS - Encore ABJ, DL91 ERA.flac | 98.44 | variable | 0.213 | 0.49% (1/5) | 97.51 (agree) | 61.68 ms | 42.63 ms | 3.8 |
| 29. Seedhe Maut - Kavi [E].flac | 110.0 | constant | 0.842 | 0.77% (8/10) | 110.29 (agree) | 12.65 ms | 17.59 ms | 8.3 |
| 24. Seedhe Maut - Akatsuki [E].flac | 156.0 | piecewise_constant | 0.811 | 0.11% (6/7) | 156.25 (agree) | 55.06 ms | 43.91 ms | 6.8 |

### Real-music sync (A bar 9, B bar 1, 60 s)

| B → A | ratio | any-onset, all beats (med / max) | downbeat attacks (med / max, ≤15 ms share) | downbeat low-kick (med / max) | meets brief? |
|---|---|---|---|---|---|
| 01. Seedhe Maut - Namastute.flac → 02. Seedhe Maut - Toh Kya [E].flac | 1.06671 | 0.06 / 37.69 ms (83/149) | 0.04 / 35.95 ms, 0.958 (24/37) | 19.39 / 55.41 ms (22/37) | ❌ max > 15 ms |
| 02 - BUN MASKA - Calm, DL91 ERA, Seedhe Maut.flac → 01. Seedhe Maut - Nanchaku [E].flac | 1.00004 | 0.06 / 37.72 ms (83/139) | 0.04 / 36.48 ms, 0.85 (20/34) | 20.63 / 72.55 ms (22/34) | ❌ max > 15 ms |

**Reading of the real-music results (counts computed from the files above):**
- Tempo: independent estimate within 1.5 % in every graded window on 11/12 tracks. Not within: 03 - Elevator Music - Encore (2.26%).
- Tempo octave: librosa's estimate agrees with ours (same octave) on 11/12 tracks. Half-time hip-hop is genuinely ambiguous (Naam Kaam Sheher was judged 140 and then 70 by the same listener), so the app shows an automatic octave (preferred range 80-160 BPM, configurable) plus 1/2x / 2x buttons, and sync matches tracks ~2x apart at half/double time.
- Sync on real songs **does not meet the max <= 15 ms target** on either pair as measured, although most downbeats are tight (see the 'within 15 ms' share). Each song's own downbeat hits scatter around a straight grid by an amount that depends on the track and on the detector, and no kick detector validated on all real tracks; we cannot separate measurement error, humanised drums and grid error without listening or annotations. Renders: `evals/results/real/sync_*_mix.wav` (gitignored, copyrighted audio).

## Environment checks (smoke test)

- WAV/FLAC/MP3 decode via libsndfile; MP3 kick timing vs WAV: {'median': 0.0, 'max_abs': 0.726}
- Rubber Band (pylibrb) R3 pitch: {'in_hz': 440, 'out_hz': 440.0, 'len_ratio': 1.25, 'expected_len_ratio': 1.25}
- mir_eval known example: True
- Assignment test pack: UNAVAILABLE - not provided with the brief; all assignment benchmarks marked unavailable

## Reproduce

```
python -m evals.make_testpack
python -m evals.eval_beats
python -m evals.eval_beats --tracker librosa
python -m evals.eval_sync
python -m evals.part1_report
```
