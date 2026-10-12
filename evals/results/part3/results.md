# Part 3 results (SYNTHETIC substitute clips; the brief's P1-P4 were not provided)

Note F: mir_eval.transcription, onset-only (offset_ratio=None), +-50 ms, pitch +-50 cents. Drums use General MIDI numbers as pitch (a snare heard as a kick is a miss); 'rhythm-only' ignores which drum. Render onsets: onsets of the offline render vs the original clip, matched within +-50 ms.

| clip | tempo ref/est | drums F (P/R) | drums rhythm-only F | pitched F (P/R) | render onset F | render median / max ms |
|---|---|---|---|---|---|---|
| P1 | 96.0 / 96.0 | 0.929 (0.867/1.000) | 0.929 | — (—/—) | 0.886 | 5.8 / 8.71 |
| P2 | 122.0 / 122.0001 | 0.707 (0.636/0.795) | 0.707 | 0.905 (0.864/0.950) | 0.859 | 5.8 / 46.44 |
| P3 | 84.0 / 84.0 | — (—/—) | — | 0.958 (0.919/1.000) | 0.887 | 5.8 / 17.41 |
| P4 | 128.0 / 128.0 | 0.509 (0.461/0.569) | 0.764 | 0.684 (0.755/0.625) | 0.878 | 2.9 / 29.02 |
