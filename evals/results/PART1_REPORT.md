# Part 1 — beat alignment and sync: results

> **Everything below is measured on the SYNTHETIC SUBSTITUTE pack** (`data/substitute`, built by `evals/make_testpack.py`). The assignment's test pack (A1–A7 with reference annotations) was not provided, so **no assignment benchmark number is claimed**. Synthetic clips have clean transients and no mix/master, so these numbers are an upper bound on real-music performance.

Machine: this laptop's CPU (no GPU). Tracker: `beat_this:small0`. mir_eval F-measures are computed on the full clip (mir_eval's default 5 s trim is NOT applied).

## Targets

| Target | Brief | Result (synthetic) | Met? |
|---|---|---|---|
| Beat F-measure (±70 ms) | ≥ 0.90 on steady-tempo clips | min 1.0000 (A1) over A1, A2, A4, A6, A7 | ✅ |
| Downbeat F-measure (±70 ms) | ≥ 0.80 | 5/6 clips pass; fails: A6 0.67 | ❌ |
| Grid drift, 6-min constant clip | every beat within 10 ms | A1 (362 s): max 1.92 ms, 100% within 10 ms | ✅ |
| Sync kick offsets | median ≤ 5 ms, max ≤ 15 ms | A7→A1 @ bar 33: median 0.14 ms, max 0.28 ms, 123/124 beats measured | ✅ |
| Speed | 6-min track < 60 s on laptop CPU | A1 (362 s): 40.64 s (eval run, first clip so includes model load) | ✅ |

## Beat grid per clip

| clip | duration_s | analysis_s | grid_type | tempo_bpm | time_signature | beat_F | downbeat_F | drift_max_ms | within_10ms | raw_beat_F | raw_drift_max_ms | confidence | librosa beat_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 361.5 | 40.64 | constant | 124.0 | 4/4 | 1.0 | 1.0 | 1.92 | 1.0 | 0.9738 | 377.42 | 0.615 | 0.0121 |
| A2 | 157.4 | 4.15 | constant | 86.0 | 4/4 | 1.0 | 1.0 | 1.12 | 1.0 | 0.9956 | 33.02 | 0.966 | 0.6677 |
| A3 | 185.2 | 5.11 | variable | 117.788 | 4/4 | 0.9945 | n/a | 72.96 | 0.5806 | 0.9959 | 52.37 | 0.637 | 0.9528 |
| A4 | 145.9 | 3.62 | constant | 118.0 | 4/4 | 1.0 | 1.0 | 1.63 | 1.0 | 0.8571 | 56.95 | 0.938 | 0.0 |
| A5 | 140.5 | 3.67 | piecewise_constant | 128.0 | 4/4 | 1.0 | 1.0 | 1.96 | 1.0 | 0.9725 | 100.0 | 0.969 | 0.0 |
| A6 | 114.0 | 2.78 | constant | 68.0 | 12/8 | 1.0 | 0.6667 | 1.83 | 1.0 | 0.9961 | 22.35 | 0.857 | 0.9961 |
| A7 | 183.3 | 4.76 | constant | 128.0 | 4/4 | 1.0 | 1.0 | 1.88 | 1.0 | 0.9961 | 67.5 | 0.995 | 0.0 |

`raw_*` = Beat This! output before our grid fitting; `drift_*` = |grid − reference| per reference beat. A3 is variable-tempo by design, so the 10 ms drift target does not apply to it.

## Failure analysis by clip

- **A1**: Pass. Confidence is low (≈0.6) only because the tracker drops beats in the 16-bar drumless breakdown; the fitted grid keeps counting through it.
- **A2**: Pass. No tempo-octave error on the half-time groove (86 BPM, not 43/172).
- **A3**: Beat F passes; worst beat is ~73 ms off. Variable-tempo grids keep the tracker's beats snapped to onsets — with 12 ms random per-hit jitter, the snap sometimes lands on a neighbouring hi-hat.
- **A4**: Pass after trimming: the tracker invented a 60 BPM pulse over the 47 s pad intro; beats with no onset evidence at the start/end are dropped. Assumption: the reference starts at the drum entry (true for our substitute; unknown for the real pack).
- **A5**: Pass. The tracker drifts toward the new tempo ~4.5 s early; the change point is re-chosen from onsets (found at bar 33, 64.498 s vs true 64.500 s).
- **A6**: **Downbeat FAIL (0.67).** 6/8 is reported as 12/8: Beat This! marks a downbeat every 4 dotted-quarter beats, so bars come out twice as long. Not fixed: telling 6/8 from accented 4/4 needs a meter model we did not build; a quick heuristic risked breaking 4/4.
- **A7**: Pass.

## Two-track sync (A7 → A1, overlap starts at A1's bar 33)

Four render methods were measured; the app uses `constant_attack`.

| method | measured | median_abs_ms | p95_abs_ms | max_abs_ms | pass | join_click_max_db | pitch_cents | render_s |
|---|---|---|---|---|---|---|---|---|
| constant | 62/124 | 4.4 | 12.18 | 18.01 | False | n/a | -0.0 | 7.9 |
| beats | 62/124 | 0.11 | 0.17 | 0.25 | True | -2.4 | 10.92 | 14.7 |
| beats_attack | 88/124 | 0.11 | 0.61 | 1.05 | True | 0.0 | 10.92 | 13.7 |
| constant_attack | 123/124 | 0.14 | 0.25 | 0.28 | True | 0.0 | -0.0 | 8.6 |

- `constant`: one Rubber Band R3 stretch. Pitch exact, but R3 smears kick transients: downbeat kicks ~30 ms early. **Note the trap**: with the strict detector only 62/124 beats are measurable and their stats look nearly fine — the unmeasured ones are the worst.
- `beats` / `beats_attack`: per-beat slices. Kicks perfect, but R3 on 0.5 s slices **detuned a sustained 110 Hz tone by +11–12 cents** (measured inside single slices) → fails "pitch preserved".
- `constant_attack` (**used**): one R3 stretch for exact pitch + B's original audio pasted from 40 ms before to 30 ms after every beat, crossfaded, so each kick sits exactly on A's beat.
- Kick offsets are measured on the separate A and B stems with a spectral-flux onset detector validated against known beat times (0.5 ms p95 spread on A7). A kick-band detector was tried and **failed validation** (false kicks in pad sections), so it was removed.
- One B downbeat (~17.4 s into the render) sits right at the detector's 3x threshold: it is measured in some runs (124/124 in the app) and not in others (123/124 here). The table reports this run.
- Caveats: A and B use the same synthetic kick sound, which makes alignment and measurement easier than on real tracks; B with variable tempo is untested; about 70 ms around each kick plays at B's original speed.

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
