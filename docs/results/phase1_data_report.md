# Phase 1 data report: Bengali-Loop to Nemotron-3-Diarization

Date: 2026-09-25. Everything was run on CPU (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `nice -n 10`). Every number below comes from a command run in this integration pass. The rules are in `docs/SPEC.md`.

## Pipeline as run

| step | command | output |
|---|---|---|
| 0 | waited for the Silero job | all 10 `runs/probe_alignment/silero/train_*.npz` were already present (last one written 13:32:30; checked at 13:48), so no recording was skipped |
| 1 | `python -m bnd.labels --out data/prepared` | `label_audit.json`, `ref_rttm/`, `train_rttm/`; byte-identical to the previous outputs |
| 2 | `python -m bnd.trust --out data/prepared/trust --gaps 2 3 5 10` | `trust/{2,3,5,10}/spans.json`, `trust/loss_report.json`; 10/10 recordings, none skipped (2.0 s wall) |
| 3 | `python -m bnd.nemo_data --spans data/prepared/trust/5/spans.json --out data/prepared/manifests` | canonical gap-5 manifests plus the real-loader probe (200 samples) |
| 3b | the same for gap 2, 3 and 10, with `--out data/prepared/manifests_by_gap/<g>` | 200-sample real-loader probe for each |
| 4 | `pytest -q tests/` | **146 passed, 0 failed** (labels 69, trust 16, nemo_data 14, score 47) |
| 5 | `python -m bnd.score der`, ref_rttm scored against ref_rttm and against train_rttm | DER 0.00 % at collars 0 / 0.25 / 0.5 on all 10 recordings. This checks that the scorer's same-speaker union is exact on the official references. |

The old 8-recording `data/prepared/manifests/` snapshot is stale. I moved it to the session scratchpad before regenerating, so the new rttm directory holds no leftover per-span RTTMs. `data/raw` was not modified: `find data/raw -newer docs/SPEC.md` returns 0 files.

## 1. Label defects per recording (`bnd.labels`)

Row counts come from the physical CSV lines. Overlap classes count rows, and the pairs columns count unique row pairs.

| rec | rows | dropped | zero_length | reversed | missing_speaker | beyond_eof | cross-spk overlap rows | same-spk overlap rows | cross pairs | same pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| train_001 | 260 | 13 | 13 | 0 | 0 | 0 | 12 | 2 | 7 | 1 |
| train_002 | 309 | 24 | 23 | 1 | 0 | 0 | 9 | 4 | 7 | 3 |
| train_003 | 259 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train_004 | 114 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 |
| train_005 | 274 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train_006 | 348 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train_008 | 122 | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 1 | 0 |
| train_009 | 332 | 47 | 44 | 3 | 0 | 0 | 4 | 0 | 2 | 0 |
| train_011 | 320 | 8 | 7 | 0 | 1 | 0 | 12 | 3 | 9 | 2 |
| train_013 | 274 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | **2612** | **115** | **109** | **5** | **1** | **0** | **41** | **9** | **27** | **6** |

- The zero-length (109), reversed (5) and cross-speaker pair (27) counts match the SPEC audit exactly.
- The `missing_speaker` class (train_011.csv:253, 2927–2929 s, empty speaker field) is not in the SPEC. That row is dropped from both RTTMs and padded as a defect by `bnd.trust`.
- `train_rttm` holds 2488 lines in total (clipped, with touching same-speaker rows merged). `ref_rttm` keeps the original rows.

## 2. Trusted spans per gap_s (`bnd.trust`, pad 1.0 s, min_span 45 s)

Totals over all 10 recordings (9.609 h audio; 7.708 h labelled speech = the union over speakers on the 10 ms grid):

- **raw** columns: seconds each rule flags. One frame can count under several rules.
- **exclusive** columns: seconds flagged by that rule alone.
- The last column shows where the dropped time went: removed by the union of rules, trimmed at span edges so that no kept labelled turn is split, or in leftovers shorter than 45 s. These three add up exactly to the dropped total.

| gap_s | kept h (of total) | kept labelled-speech h (of total) | labelled speech lost | spans | dropped total s | raw: defect / label_silent / unlabelled_speech / gt8 (s) | exclusive: defect / label_silent / unlabelled_speech / gt8 (s) | union excluded / edge trim / short span (s) |
|---|---|---|---|---|---|---|---|---|
| 2 | 6.994 / 9.609 | 5.399 / 7.708 | 30.0 % | 128 | 9411 | 677 / 1047 / 6.05 / 127 | 637 / 1008 / 6.05 / 127 | 1818 / 5500 / 2093 |
| 3 | 8.010 / 9.609 | 6.333 / 7.708 | 17.8 % | 107 | 5753 | 677 / 611 / 0.00 / 127 | 648 / 582 / 0.00 / 127 | 1386 / 2940 / 1427 |
| 5 | 8.712 / 9.609 | 6.954 / 7.708 | 9.8 % | 95 | 3226 | 677 / 272 / 0.00 / 127 | 665 / 260 / 0.00 / 127 | 1064 / 1427 / 735 |
| 10 | 9.104 / 9.609 | 7.329 / 7.708 | 4.9 % | 81 | 1817 | 677 / 103 / 0.00 / 127 | 677 / 103 / 0.00 / 127 | 907 / 308 / 602 |

Reading the table:
- Rule (c), unlabelled speech, is almost empty: 6.05 s at gap 2 and 0 at gap ≥3. Both detectors rarely hear speech for 2 s or more outside the labels, so there is no evidence of missing labels.
- Rule (d), more than 8 speakers in a 105 s window, fires only on train_006 (127 s), as the SPEC predicted.
- At small gap_s, most of the loss is **edge trim**, not the rules themselves: 5500 s at gap 2 and 1427 s at gap 5. When an excluded region falls inside a long turn, the whole turn goes, because a kept labelled turn is never split.

The per-recording "lost %" is labelled speech lost. train_009 and train_011 lose the most at gap 5 (20.9 % and 25.7 %). train_011's loss is mostly edge trim (718 s), where excluded regions fall inside long turns.

### Per recording


Per recording, kept hours / kept labelled-speech hours / spans, and labelled speech lost (%):

| recording | duration h | labelled h | gap 2: kept h / lab h / spans / lost % | gap 3: kept h / lab h / spans / lost % | gap 5: kept h / lab h / spans / lost % | gap 10: kept h / lab h / spans / lost % |
|---|---|---|---|---|---|---|
| train_001 | 0.936 | 0.544 | 0.554 / 0.233 / 20 / 57.1 | 0.674 / 0.323 / 17 / 40.6 | 0.820 / 0.441 / 16 / 18.8 | 0.838 / 0.459 / 14 / 15.6 |
| train_002 | 0.677 | 0.482 | 0.481 / 0.325 / 14 / 32.5 | 0.527 / 0.365 / 13 / 24.3 | 0.592 / 0.427 / 11 / 11.4 | 0.592 / 0.427 / 11 / 11.4 |
| train_003 | 1.084 | 0.961 | 0.669 / 0.574 / 20 / 40.3 | 0.946 / 0.842 / 16 / 12.4 | 1.019 / 0.912 / 17 / 5.2 | 1.056 / 0.949 / 16 / 1.3 |
| train_004 | 1.129 | 1.084 | 1.083 / 1.039 / 6 / 4.2 | 1.094 / 1.050 / 4 / 3.2 | 1.126 / 1.081 / 2 / 0.3 | 1.126 / 1.081 / 2 / 0.3 |
| train_005 | 0.947 | 0.895 | 0.924 / 0.873 / 5 / 2.5 | 0.929 / 0.878 / 4 / 1.9 | 0.929 / 0.878 / 4 / 1.9 | 0.939 / 0.888 / 3 / 0.8 |
| train_006 | 0.917 | 0.694 | 0.839 / 0.626 / 7 / 9.8 | 0.866 / 0.647 / 6 / 6.8 | 0.871 / 0.652 / 4 / 6.0 | 0.878 / 0.658 / 2 / 5.1 |
| train_008 | 1.133 | 1.096 | 0.716 / 0.689 / 9 / 37.2 | 0.944 / 0.915 / 5 / 16.5 | 1.038 / 1.008 / 3 / 8.0 | 1.089 / 1.054 / 3 / 3.9 |
| train_009 | 0.716 | 0.429 | 0.425 / 0.214 / 18 / 50.1 | 0.526 / 0.303 / 20 / 29.4 | 0.571 / 0.339 / 20 / 20.9 | 0.580 / 0.348 / 18 / 19.0 |
| train_011 | 1.113 | 0.982 | 0.581 / 0.479 / 14 / 51.2 | 0.728 / 0.619 / 13 / 36.9 | 0.848 / 0.729 / 13 / 25.7 | 1.055 / 0.930 / 10 / 5.3 |
| train_013 | 0.956 | 0.540 | 0.722 / 0.347 / 15 / 35.8 | 0.776 / 0.391 / 9 / 27.7 | 0.897 / 0.485 / 5 / 10.2 | 0.951 / 0.536 / 2 / 0.8 |

### Dropped by reason per recording at gap_s = 5


| recording | defect | label_silent | unlabelled_speech | gt8_speakers | union excluded | edge trim | short span |
|---|---|---|---|---|---|---|---|
| train_001 | 106.0 | 97.7 | 0.0 | 0.0 | 203.7 | 136.3 | 78.0 |
| train_002 | 160.0 | 12.3 | 0.0 | 0.0 | 160.0 | 30.0 | 114.0 |
| train_003 | 37.0 | 9.5 | 0.0 | 0.0 | 46.5 | 126.5 | 59.0 |
| train_004 | 13.0 | 0.0 | 0.0 | 0.0 | 13.0 | 0.0 | 0.0 |
| train_005 | 4.0 | 5.3 | 0.0 | 0.0 | 9.3 | 53.7 | 0.0 |
| train_006 | 0.0 | 16.4 | 0.0 | 127.0 | 143.3 | 17.7 | 4.0 |
| train_008 | 118.0 | 11.1 | 0.0 | 0.0 | 129.1 | 160.0 | 56.0 |
| train_009 | 139.0 | 18.4 | 0.0 | 0.0 | 157.4 | 54.6 | 310.0 |
| train_011 | 100.0 | 54.3 | 0.0 | 0.0 | 154.3 | 717.7 | 81.0 |
| train_013 | 0.0 | 47.6 | 0.0 | 0.0 | 47.6 | 130.4 | 33.0 |

### Per fold (SPEC folds F1 001+008, F2 011+009, F3 013+005, F4 004+002, F5 003+006)


| fold | recordings | total h | labelled h | gap 2 kept h / lab h | gap 3 kept h / lab h | gap 5 kept h / lab h | gap 10 kept h / lab h |
|---|---|---|---|---|---|---|---|
| F1 | 001+008 | 2.070 | 1.640 | 1.270 / 0.922 | 1.618 / 1.238 | 1.858 / 1.450 | 1.928 / 1.512 |
| F2 | 011+009 | 1.829 | 1.411 | 1.007 / 0.693 | 1.254 / 0.922 | 1.420 / 1.069 | 1.634 / 1.278 |
| F3 | 013+005 | 1.902 | 1.435 | 1.646 / 1.220 | 1.705 / 1.269 | 1.826 / 1.363 | 1.890 / 1.423 |
| F4 | 004+002 | 1.806 | 1.566 | 1.564 / 1.364 | 1.622 / 1.415 | 1.718 / 1.508 | 1.718 / 1.508 |
| F5 | 003+006 | 2.001 | 1.655 | 1.508 / 1.200 | 1.812 / 1.489 | 1.891 / 1.564 | 1.934 / 1.607 |

## 3. NeMo manifests and real-loader sampling (`bnd.nemo_data`)

Setup:
- Per-span RTTMs use **absolute recording time**, clipped to the span. The loader's offset handling was verified against `audio_to_diar_label.py` at cf724ac; the citations are in the module docstring.
- The dataset is the real `SortformerEncLabelModel` stage-2 `train_ds` from `sortformer_streaming_8spk.yaml`, confirmed at runtime: session_len_sec 105, num_spks 8, subsegment_mode true, single_chunk_min 45 s, two_chunks_rate 0.5, two_chunk_min 22 s, nspk_bias 1.5, start_guard 0.25 s, min_first_spk 0.5 s, splice_silence 0.1 s, soft_targets false, 100 frames/s.
- Each probe draws 200 samples with seed 20260925.
- Session lengths were measured in a second pass. That pass wrapped `__getitem__` with the same seed and reproduced the CLI probe's statistics exactly.

| gap_s | manifest lines (spans) | turns | hours | labelled h | max spk / span | turns clipped at edge | balanced lines | exposure ratio one-line / balanced |
|---|---|---|---|---|---|---|---|---|
| 2 | 128 | 1954 | 6.994 | 5.399 | 12 | 0 | 257 | 15.3 / 1.9 |
| 3 | 107 | 2118 | 8.010 | 6.333 | 18 | 0 | 292 | 25.9 / 1.8 |
| **5** | **95** | **2246** | **8.712** | **6.954** | **18** | **0** | **306** | **25.9 / 1.9** |
| 10 | 81 | 2294 | 9.104 | 7.329 | 18 | 0 | 317 | 28.1 / 1.9 |

Loader probe, 200 samples per gap:

| gap_s | spans hit | fallback (noise) | empty targets | two-chunk | active speakers per sample (0/1/2/3/4/5/6/7/8) | mean active | session length s: min / p10 / median / mean / max | samples at full 105 s |
|---|---|---|---|---|---|---|---|---|
| 2 | 102 | 0 (0.0 %) | 18 (9.0 %) | 76 | 18/13/75/52/30/7/2/2/1 | 2.54 | 45.00 / 46.67 / 79.27 / 78.04 / 105.0 | 65 |
| 3 | 95 | 0 (0.0 %) | 10 (5.0 %) | 80 | 10/23/52/58/32/11/8/4/2 | 2.88 | 45.02 / 47.22 / 93.87 / 82.77 / 105.0 | 87 |
| **5** | **86** | **0 (0.0 %)** | **1 (0.5 %)** | **86** | **1/13/84/57/27/10/5/2/1** | **2.81** | **45.04 / 49.30 / 105.0 / 86.81 / 105.0** | **106** |
| 10 | 74 | 0 (0.0 %) | 0 (0.0 %) | 91 | 0/13/59/73/23/16/10/5/1 | 3.13 | 45.01 / 51.77 / 105.0 / 91.46 / 105.0 | 124 |

Assertions that held on every gap:
- 0 unexplained target/RTTM mismatch frames and 0 off-by-one edge frames (all real times are whole seconds).
- Every arrival frame matches the RTTM, and 0 arrival ties.
- The first arrival is a single speaker in every non-empty sample: 199/199 at gap 5, 182/182 at gap 2, 190/190 at gap 3, 200/200 at gap 10.
- At most 8 active channels.
- 0 audio reads outside the span.
- All audio chunks are bit-exact against the WAV: 286/286 at gap 5, 276 at gap 2, 280 at gap 3, 291 at gap 10.
- The largest seek shift is 1 sample (10 chunks at gap 5).
- A few chunks were zero-padded at the loader's EOF: 4 at gap 5.
- One real collate batch has shape [4, 10500, 8].
- The balanced manifest loads with exactly as many items as it has lines.

**Where the empty targets come from:** I checked this directly. The empty samples are drawn from trusted spans that contain **no labelled turn at all**:
- gap 2 has 7 such spans (681 s): train_001 s000/s002/s009/s019, train_009_s012, train_013 s009/s010.
- gap 3 has 3 such spans (365 s).
- gap 5 and gap 10 have 0.

These spans are genuinely silent by both detectors: Nemotron and Silero both flag speech in ≤0.5 % of frames. The exceptions are train_013 s009/s010, where Nemotron alone flags 14–16 % and Silero flags 0–0.3 %. The labels and audio therefore agree (silence labelled as silence), so this is not a bug. It does put all-silence training chunks into the data at small gap_s. The single empty sample at gap 5 comes from train_001_s015, a 554 s span with a 110 s unlabelled silent stretch.

## 4. Integration edits

- `tests/test_nemo_data.py::test_probe_real_spans`: the assertion `n_first_arrival_unique == n_samples - n_fallback` failed (199 vs 200) once train_011/013 entered the gap-5 spans. The reason is one legitimately empty real sample, which has no first arrival. I changed the assertion to `n_samples - n_empty_targets` (fallbacks are a subset of empties, and a new assertion checks that). No code in `bnd/` changed.

## 5. Open issues (owner decisions and known limitations)

1. **Which gap_s to train on.**
   - gap 5 (the SPEC default) keeps 8.712 h / 6.954 h labelled, losing 9.8 % of labelled speech.
   - gap 10 keeps 9.104 h and loses 4.9 %.
   - gap 2 loses 30.0 %, and 9 % of its loader samples have empty targets.
2. **Edge trim dominates the loss at small gaps.** Never splitting a kept labelled turn means an excluded region inside a long turn drops the whole turn: 1427 s at gap 5, 5500 s at gap 2. The SPEC wording also allows cuts at the edges of excluded regions. Allowing them would recover time but would split turns.
3. **Rule (a) excludes both whole rows of every overlap pair (±1 s),** including long parent turns such as train_002 spk13 822–927 s. That totals 677 s raw. Padding only the overlap region would recover part of it.
4. **Kept data contains 0 overlapped speech,** a direct consequence of SPEC rule 2(a). The fine-tuned model will see no overlap in training, which is a distribution shift from the test audio.
5. **Manifest exposure.** One line per span gives the long spans few draws (exposure ratio 25.9× at gap 5). `train_manifest_balanced.json` (306 lines at gap 5) brings this to 1.9×. Which manifest to train on is a phase-2 decision.
6. **Spans with no labels (gap 2/3 only),** described in §3. Keep them (training on silence) or drop spans with a labelled fraction below some threshold. This is an owner decision.
7. **Spans can hold more than 8 speakers across their whole length** (max 18 per span at gap 3/5/10). This is fine for subsegment training, because the 105 s chunks are filtered to ≤8 speakers and there were 0 fallbacks. It also means spans must **not** be used as whole-span val/test manifest entries. Validation stays at recording level through `bnd.score`.
8. **`missing_speaker`** is a defect class the SPEC does not list (1 row, train_011:253). It is handled consistently in labels and trust.
9. **Beyond-EOF tolerance differs:** trust uses 0 and labels uses EOF_TOL_S. There were 0 beyond_eof rows, so it had no effect.
10. **Defect classification is duplicated:** `trust.classify_defects` repeats `labels.classify`, so the two can drift apart. Today they agree on every class count.
11. **Scorer policy to confirm:** same-speaker overlapping rows are unioned for the counted speech, but collars are taken from the original rows. On the official refs this gives 0 % DER against themselves. The unions are 1/3/2 on 001/002/011.
12. **Kaggle export:** "non-overlapping" is read as global. `--overlap flatten` is an unvalidated heuristic, and the validator is stricter than Kaggle may be.
13. **NeMo precision quirk (upstream):** `int((t-offset)*100)` truncation causes ±1-frame edge jitter at non-integer times. Today there is none, because all real times and offsets are whole seconds.
14. **The `slow` pytest mark is not registered,** which produces warnings only. There is no conftest/pytest.ini.
15. **The CLIs resolve `--out` relative to the working directory,** so run them from the repo root.
