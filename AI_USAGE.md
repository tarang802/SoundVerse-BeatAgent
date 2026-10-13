# AI_USAGE

Two different things: AI **inside the product** (what runs when you use Beat Agent) and
AI **used to build it**.

## 1. AI at runtime

| Component | What it is | What it is trusted with |
|---|---|---|
| **Beat This!** (CPJKU, `small0` checkpoint, CPU) | neural beat + downbeat tracker | raw beat/downbeat times — then fitted, cleaned and checked by our code (`beatagent/beats/analyze.py`) |
| **Basic Pitch** (Spotify, 0.4, ONNX model) | neural note transcription | pitched notes for Part 3 — then filtered (overtone ghosts, drum body tones) by our code |
| **Gemini** (`gemini-3.5-flash`, falling back to other Flash / Flash-Lite models on quota) | LLM, structured output | **interpreting one chat message into a plan**: typed edit ops for the track (Part 2), or typed score ops for a recreation (Part 3: tempo, mute, solo, level, brightness, transpose, oscillator). Nothing else. |

**The LLM never:** measures audio, computes times or tempo, transcribes notes, writes
code that runs, or applies an edit itself. Its output is data, validated against a
pydantic schema; deterministic code resolves every op on the measured grid, echoes the
exact bars/beats/seconds, applies it, or refuses with a reason.

**Runtime failure modes we saw, and the guards** (all deterministic, all tested in
`tests/test_agent_llm.py` with a scripted stand-in for the LLM, so they are checked
without spending API quota):

| Seen in testing | Guard |
|---|---|
| "remove bar -2" rewritten to bar 1 and applied | negative / fractional bar or beat numbers are refused before the LLM is called |
| "make bars 1-4 60 dB louder" clamped to +24 dB and applied | the user's stated dB magnitude is kept; the LLM only chooses up or down |
| "remove everything" → all 186 bars removed | an edit that would leave < 2 s becomes a proposal needing "yes" |
| "chop out the boring part" → proposal only in prose, so "yes" did nothing | one repair round asking for the ops |
| "cut beat 16-32 repeat it twice" refused as "fractions" (found by Tarang in the live app) | prompt rule reworded + a repair round when the LLM claims fractions the code found none of |
| quota exhausted / network down | model fallback chain, then a narrow offline rule parser; every reply says which answered |

## 2. AI used to build it

**Tool:** Claude Code (Anthropic's Claude, in the Claude desktop app), used throughout.

**Delegated to the AI:** most of the implementation (Python library, FastAPI server,
React UI), the evaluation harnesses and test suites, the synthetic test packs, the
investigation of failures, and first drafts of these documents.

**Kept by Tarang:** scope and priorities (e.g. finish Parts 1–2 before Part 3), product
calls (React + TypeScript + Vite UI; a DJ-style interface; building for real DJs rather
than to the test cases), testing the app on real music and against outside references,
and deciding what ships.

### What was reviewed line by line

> **To be completed by Tarang before submission** — list the files or functions you
> read line by line (and anything you deliberately did not), so this section is true.

### Cases where the AI was wrong or subtly off, and how it was caught

1. **Tempo octave on a real track (caught by Tarang).** The analyser reported a real
   hip-hop track at 70 BPM; Tarang compared it with the published tempo (140). The
   investigation showed this is a counting convention, not a measurement error, and
   Tarang later noted the track can be heard at half time as well. Outcome: a documented
   80–160 BPM convention plus ½× / 2× buttons per deck, and octave-aware sync.
2. **Building to the test cases (caught by Tarang).** Early versions defaulted the sync
   to bar 33 (from the brief's session 6) and used a "master timeline" that confused a
   real user. Tarang pushed back ("build for real DJs"); defaults and labels changed.
3. **A click metric that could not see clicks (caught in review).** Every
   "click-free" claim rested on a metric the AI wrote. When the review asked to
   investigate suspicious 0 dB readings, validating the metric against splices of known
   quality showed a hard cut through a bass note scored −4.7 dB (a nearby hi-hat
   out-shouted the click), and engine splices read 0.0 dB only because the next kick
   masked them. The metric was rebuilt and is now tested both ways
   (`tests/test_click_metric.py`).
4. **A wrong hypothesis, disproved by measurement.** With the new metric the fuzz test
   flagged 8 edits; the AI guessed the equal-power crossfade was adding +3 dB on
   correlated audio and changed the crossfade law. Re-running gave identical numbers, so
   the change was reverted; the real cause was the metric ignoring the edit's own gain.
5. **Subtle code corruption from the AI's own editing method.** Shell-escaped patches
   wrote a literal backspace character into a regex (the offline "sync at bar N" rule
   could never match) and, another time, a raw newline into a string. Caught by an audit
   and a compile error; `tests/test_imports.py` now fails on any control character in
   source.
6. **A guard that over-reached.** The AI's first "near-total deletion" guard used a
   5 % rule and blocked a deliberate "trim it down to just bars 1-8" on a long track; the
   chat suite caught it and the rule became an absolute 2 s floor.
7. **An injectable page, from the AI that wrote the template.** The generated Part 3 page
   built its mixer rows with `innerHTML` from track names and only escaped `</` in its
   data, so a crafted name could add markup and run script inside the page. Caught by a
   safety test written afterwards (`tests/test_webaudio_page.py`); names are now inserted
   as text and `< > &` are escaped.
8. **Ignoring a standing instruction.** The AI added a `Co-Authored-By` trailer to
   commits although Tarang had asked for commits under Tarang's own name only; Tarang caught it
   on GitHub and the affected commits were rewritten.

## 3. Testing: automated vs manual

**Automated** (re-runnable, see README): 11 deterministic test suites (`python run.py
test`), a 421-edit fuzz test, Part 1 evals on the synthetic pack plus 30 robustness and
19 sync cases with exact answers, a 59-case chat suite and the brief's six sessions
against the live LLM (answers recorded so they replay without quota), and the Part 3
pipeline evaluated on P1–P4 including headless-browser renders.

**Manual:** Tarang used the app on real music (and supplied the 12 real hip-hop tracks
used in the consistency evals) and reported issues that automated tests had not caught (tempo octave, the "beat 16-32" refusal, confusing
sync defaults). Listening notes on the click auditions and sync renders:

> **To be completed by Tarang** — what you listened to and what you heard.
