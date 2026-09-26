# Phase 3b results: fair re-test of fine-tuning, plus linking (second-look)

Protocol: `docs/PREREG_phase3b.md` (binding; supersedes the withdrawn `PREREG_phase3.md`).
Sources for every number: `runs/p3b/results.json` (frozen identity `657353de1498…`, results sha
`c7b5b325…`, one scoring in `runs/p3b/scored.jsonl`), `runs/p3b/selection.json`,
`runs/p3b/gpu_ledger.jsonl`, the independent read-only audit (§7) and `runs/final/`.

> **SECOND-LOOK results.** Every phase-3b factor was motivated by phase-2 OUTER-fold diagnostics:
> 80 s sessions, θ_act below 0.40, the refine linker and gap-fill. Selection used inner folds only,
> so the outer numbers are protected from the *choice* among candidates. They are not protected from
> the *design* of the search space. The phase-2 disclosures D1/D2 still apply. There was no Kaggle
> submission.

## 1. Q1 verdict first: does fine-tuning help under equal decoding?

**No. Fine-tuning shows no demonstrated benefit over calibrated zero-shot decoding.** Harm is not
shown either. Each system searched the same 108 decoding configurations and was selected with the
same 1-SE rule on inner folds.

The comparison is FT\* − ZS\*: corpus DER at collar 0, paired recording bootstrap with 50,000
resamples and seed 20260925.

| Claim condition (all four required) | Value | Result |
|---|---|---|
| Corpus Δ ≤ −1.0 pp | ZS\* 26.97 % → FT\* 27.69 %, **Δ +0.72 pp** | FAIL |
| Bootstrap 97.5th percentile < 0 | 95 % CI **[−2.48, +4.17]** | FAIL |
| ≥ 6/10 recordings improved | FT\* better on **4**, worse on 6 | FAIL |
| Macro-mean Δ < 0 | **+1.58 pp** (macro DER 31.07 vs 32.65) | FAIL |
| Harm clause (2.5th percentile > 0) | −2.48 | not met |

At collar 0.25, Δ is +0.62 [−2.59, +4.10] and FT\* wins 5/10. At collar 0.5, Δ is +0.48
[−2.70, +3.97], again 5/10.

This result reverses the phase-2 reading. The phase-2 gain (42.75 → 28.69) compared fine-tuning
against a zero-shot system with a weak decoder. When both systems get the same model-free decoder
search, which in practice means same-speaker gap-fill at g = 5 s, the gain disappears.

**Default system, by the rule fixed before scoring: ZS.** The pinned base Nemotron-3-Diarization is
used with configuration **`s20_avg_g5_a0.40_c0.60`**. That is 20 s sessions, the phase-2 average-linkage
linker, gap-fill 5 s, θ_act 0.40 and θ_clu 0.60. It is the joint mode of the five ZS inner
selections: all 5 of 5 folds chose it, so no tie-break was needed. No training was run, because the
all-10 fine-tune is registered only for an FT default.

## 2. Corpus DER (outer, 10 recordings, 27,856 s of reference speech at collar 0)

Values are DER % [MISS / FA / CONF].

| System | Collar 0 | Collar 0.25 | Collar 0.5 |
|---|---|---|---|
| **ZS\* (default)** (= ZS Q2) | **26.97** [6.90/6.85/13.22] | 25.13 [6.35/5.85/12.93] | 23.34 [5.83/4.86/12.65] |
| FT\* | 27.69 [5.32/8.56/13.82] | 25.75 [4.98/7.34/13.42] | 23.82 [4.67/6.10/13.05] |
| FT Q2 (S=20, avg) | 28.23 [4.30/9.14/14.79] | 26.28 [4.03/7.89/14.36] | 24.33 [3.78/6.60/13.95] |
| ZS\* + oracle link (own tracks) | 16.08 [6.04/6.97/3.06] | 14.18 [5.51/5.96/2.71] | 12.33 [5.00/4.95/2.38] |
| FT\* + oracle link (own tracks) | 16.94 [5.29/8.56/3.09] | 14.98 [4.96/7.35/2.68] | 13.06 [4.65/6.10/2.31] |
| p2 zs_sel (reference) | 42.75 [26.01/5.63/11.10] | 41.29 [25.75/4.72/10.81] | 39.88 [25.54/3.82/10.52] |
| p2 ft_sel (reference) | 28.69 [7.49/7.35/13.84] | 26.81 [7.00/6.26/13.55] | 24.94 [6.54/5.14/13.26] |
| p2 zs_strict (reference) | 43.67 [27.60/5.43/10.64] | 42.26 [27.33/4.55/10.38] | 40.89 [27.10/3.67/10.12] |
| p2 ft_strict (reference) | 30.96 [11.89/6.95/12.12] | 29.22 [11.52/5.88/11.83] | 27.52 [11.16/4.81/11.55] |

## 3. Per recording (collar 0) and speaker counts

The speaker columns give predicted counts. The reference counts come from `results.json`
`speaker_count_errors`.

| Recording | ZS\* DER | FT\* DER | Δ pp | Ref spk | ZS\* spk | FT\* spk |
|---|---|---|---|---|---|---|
| train_001 | 45.35 | 47.22 | +1.88 | 11 | 40 | 44 |
| train_002 | 38.07 | 50.96 | +12.89 | 20 | 30 | 35 |
| train_003 | 13.51 | 19.97 | +6.46 | 21 | 20 | 15 |
| train_004 | 11.39 | 9.59 | −1.80 | 10 | 35 | 28 |
| train_005 | 30.78 | 21.87 | −8.90 | 19 | 58 | 32 |
| train_006 | 42.31 | 43.18 | +0.87 | 17 | 68 | 53 |
| train_008 | 7.24 | 6.63 | −0.61 | 11 | 10 | 14 |
| train_009 | 36.58 | 44.40 | +7.82 | 21 | 35 | 44 |
| train_011 | 28.59 | 28.75 | +0.16 | 18 | 19 | 25 |
| train_013 | 56.90 | 53.95 | −2.96 | 21 | 42 | 33 |

Sum of |predicted − reference| speakers for each system:

- ZS\* 192, FT\* 166, FT Q2 180.
- Oracle link: ZS 35, FT 33.
- Phase 2: zs_sel 261, ft_sel 155, zs_strict 131, ft_strict 98.

**Both selected systems overcount speakers heavily.** On speaker count, ZS\* is *worse* than phase-2
zs_strict even though its DER is 16.7 pp lower. Gap-fill improves DER but does nothing about
fragmentation.

## 4. Selection (inner folds only; 1-SE rule; LORO = leave-one-recording-out recoveries)

| Fold | ZS selected (inner DER) | ZS argmin | LORO | FT selected (inner DER) | FT argmin | LORO |
|---|---|---|---|---|---|---|
| F1 | s20_avg_g5_a0.40_c0.60 (28.87) | s80_refine_g5_a0.40_c0.55 | 7/8 | s20_avg_g0_a0.40_c0.55 (28.28) | s20_avg_g2_a0.40_c0.60 | 7/8 |
| F2 | s20_avg_g5_a0.40_c0.60 (26.06) | s80_refine_g5_a0.40_c0.55 | 5/8 | s20_avg_g0_a0.25_c0.55 (27.31) | s20_avg_g2_a0.40_c0.60 | 3/8 |
| F3 | s20_avg_g5_a0.40_c0.60 (23.86) | s80_refine_g5_a0.40_c0.55 | 5/8 | s20_avg_g0_a0.40_c0.60 (25.55) | s80_refine_g5_a0.40_c0.55 | 6/8 |
| F4 | s20_avg_g5_a0.40_c0.60 (28.82) | s80_avg_g5_a0.25_c0.55 | 6/8 | s20_avg_g0_a0.40_c0.55 (28.34) | s20_avg_g5_a0.40_c0.60 | 2/8 |
| F5 | s20_avg_g5_a0.40_c0.60 (27.35) | s20_avg_g5_a0.40_c0.60 | 7/8 | s20_avg_g2_a0.40_c0.60 (25.88) | s20_avg_g5_a0.40_c0.60 | 5/8 |

The FT recipe was F1 lr5e-5_raw and lr5e-5_shift in F2–F5.

- The refine linker and S=80 win some argmins but are never 1-SE-selected.
- θ_act 0.10 is never selected and is never the argmin. The council's activity-only diagnostic,
  which found a 3–5 pp gain for lower θ_act, did not survive a full retune.
- The FT selections are unstable: LORO recovers them 2/8 in F4 and 3/8 in F2.

## 5. Q2 (linking) and other comparisons (collar 0, paired bootstrap, Δ = B − A)

| Comparison | A | B | Δ pp | 95 % CI | B better on |
|---|---|---|---|---|---|
| **Q2, ZS** (the better Q1 system): restricted → selected | 26.97 | 26.97 | 0.00 | [0.00, 0.00] | 0/10 (10 ties) |
| Q2, FT: restricted → selected | 28.23 | 27.69 | −0.54 | [−1.50, 0.00] | 2/10 (8 ties) |
| ZS\* → ZS\* + oracle link | 26.97 | 16.08 | −10.89 | [−15.69, −6.88] | 10/10 |
| FT\* → FT\* + oracle link | 27.69 | 16.94 | −10.75 | [−15.94, −6.57] | 10/10 |
| p2 zs_sel → p3b ZS\* | 42.75 | 26.97 | −15.78 | [−18.97, −12.34] | 10/10 |
| p2 ft_sel → p3b FT\* | 28.69 | 27.69 | −1.00 | [−3.77, +1.89] | 4/10 |

For ZS, Q2 is **degenerate**. The restricted pick (S=20, L=avg) is the same configuration as ZS\* in
every fold, so the comparison contains nothing and supports no linking change.

**Linking is still the dominant remaining error.** Oracle linking from each system's own tracks
removes about 10.8 pp, and CONF falls from roughly 13 % to 3 %. None of the linking alternatives
passed the 1-SE rule.

## 6. Final artefact (`runs/final/`)

- **`model_reference.json`**: ZS, the pinned base checkpoint with no training.
  - HF `nvidia/Nemotron-3-Diarization`, revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`.
  - Local file `data/weights/nemotron3/Nemotron-3-Diarization.nemo`, sha256
    `867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d` (re-verified).
  - NeMo commit `cf724ac3…`.
- **`final_config.json`**: the full decoding configuration, the embedder id and revision, and
  provenance:
  - the five inner selections and the joint-mode computation;
  - the Q1 numbers and pass flags;
  - hashes of `results.json`, `selection.json` and `frozen.json`, and the frozen identity;
  - code hashes;
  - disclosures.
- **`ft_refits_provenance.json`**: all 10 phase-2 fine-tuned refits (5 selected-recipe, 5 strict).
  - Each has its path, a sha256 re-verified against its fit marker, the 8 training recordings, the
    held-out fold, lr, timing, δ, manifest and code hashes, and steps.
  - It states plainly that FT did **not** beat ZS under equal decoding.
  - None of these models was trained on all 10 recordings.
- **Builder:** `python -m bnd.final` (CPU). It re-derives the default from `results.json` and
  `selection.json` and refuses to continue on any mismatch.
- **CLI:** `python -m bnd.diarize <audio> --out <dir>`. It runs ffmpeg to 16 kHz mono and checks the
  model sha256 (failing closed on a mismatch). It then runs native streaming (264/40/340/40/300,
  bf16) over 20 s sessions with 5 s context, extracts θ_act tracks, computes WeSpeaker embeddings,
  links with avg-link at θ_clu 0.60 and applies gap-fill of 5 s. It writes `<stem>.rttm`,
  `<stem>.csv` (`start_time,end_time,speaker_id`) and `<stem>.diarize_report.json`.

Smoke run through `scripts/gpu_run.sh` on card 1. Nothing was scored.

| Audio | Duration | Speakers out | Turns | RTF total | RTF NeMo | Wall (gpu_run) |
|---|---|---|---|---|---|---|
| test_029 (unlabelled test split) | 2408.8 s | 35 | 290 | 0.0163 | 0.0055 | 42.3 s |
| train_001 | 3371.4 s | 40 | 329 | 0.0131 | 0.0052 | 47.3 s |

- About 8.5 s of each run is loading the model and about 14 s is embedding, which runs as a
  subprocess in its own venv. Peak NeMo allocation is 0.57 GiB.
- **Consistency check (no labels read):** the train_001 RTTM that `bnd.diarize` produced is
  byte-identical to the frozen outer ZS\* RTTM `runs/p3b/outer/zs_star/F1/train_001.rttm` (sha256
  `48aff85e…`). The CLI therefore reproduces the evaluated system end to end.
- Its 40 speakers against 11 in the reference is the known over-counting problem (§3).

## 7. Audit (independent, read-only)

- All 100 frozen outer RTTMs match their sha256 in `runs/p3b/frozen.json`. The hashes of
  `selection.json` and of the phase-2 `frozen.json` also match. `results.json` matches
  `scored.jsonl`, and there was a single scoring. The file order was selection, then outer RTTMs,
  then freeze.
- Phase 2 was not modified:
  - All 80 phase-2 frozen hashes match, and `runs/cv/results.json` matches its scored hash.
  - No file in runs/cv, runs/models, the reference RTTMs or the raw data changed after phase 3b
    started. This check relies on modification times.
- **Regen reproduction passed.** The check covered 14 fits and 56 recording × fit pairs.
  - The window lists are identical and max |Δprob| = 0.0.
  - The regenerated `.nemo` files have a different sha256 from phase 2, but their probabilities are
    identical.
- The audit recomputed inner DER for 5,400 RTTMs with pyannote and an independent RTTM parser. The
  maximum difference was 0.0.
- An independent implementation of the 1-SE rule reproduced every selection, argmin, Q2 pick and
  LORO count.
- ZS and FT used the same 108 configurations and code path. Every pool contained inner recordings
  only.
- Q1, Q2, oracle link, the speaker counts and every other comparison reproduced exactly.
- Phase 3b used 1.606 GPU-h over 48 jobs, all on card 1. No orphan processes remained.

## 8. Caveats (state all of these in any report)

1. **Second-look.** The design of the search space came from phase-2 outer diagnostics. Any DER of
   `s20_avg_g5_a0.40_c0.60` over all 10 labelled recordings is a **development** number, never a
   held-out one.
2. **One training seed.** The FT results reflect one seed per fit, and training variance is not in
   any confidence interval.
3. **Coarse references.** All 4,994 onset and duration fields are whole seconds, so collar-0 DER
   includes quantisation error from the reference itself. Gap-fill partly works by matching the
   annotators' convention of continuing a speaker through pauses. Part of the gap-fill "gain" is
   therefore agreement with the convention, not better acoustics.
4. **Ten recordings.** Each fold has 8 inner recordings. The CIs are wide and dominated by
   individual recordings; train_002 alone contributes +12.9 pp to Q1.
5. **Unstable selections.**
   - FT LORO recovery is 2/8 in F4 and 3/8 in F2.
   - F3 FT has an SE of 2.68 pp with 35 eligible configurations.
   - F5 ZS has an SE of 0.
6. **Speaker over-counting.** ZS\* has a sum |error| of 192, including 68 predicted speakers against
   17 in the reference on train_006. This is worse than the phase-2 strict systems.
7. **Regenerated files differ.** The regenerated inner `.nemo` files have a different sha256 from
   phase 2, although their probabilities are identical.

## 9. What a model card can and cannot claim

**Can claim:**

- On this 10-recording Bengali drama set, fine-tuning Nemotron-3-Diarization (phase-2 recipe, one
  seed) showed **no demonstrated benefit** over the base model once both used the same
  inner-selected decoder: +0.72 pp, CI [−2.48, +4.17].
- The recommended system is the **unmodified base model**. Its decoding is 20 s sessions with 5 s
  context, θ_act 0.40, WeSpeaker average-linkage cannot-link linking at θ_clu 0.60, and
  same-speaker gap-fill of 5 s. All five inner selections chose this configuration.
- Outer-fold, second-look DER is 26.97 % at collar 0, 25.13 % at collar 0.25 and 23.34 % at
  collar 0.5, and it must carry the second-look label.
- This decoder improves zero-shot DER by 15.8 pp over the phase-2 zero-shot decoder (10/10
  recordings; second-look).
- Linking is the main remaining error, with about a 10.9 pp oracle-link gap.
- The speed figures are measured: RTF about 0.013–0.016 end to end on one A6000.

**Cannot claim:**

- That "zero-shot beats fine-tuning". The CI spans 0.
- That fine-tuning hurts. The harm clause was not met.
- That fine-tuning "fixes speech detection".
- Any held-out number for the final configuration over all 10 recordings.
- Better speaker counting. The system overcounts.
- Any benefit from the refine linker, 80 s sessions or θ_act 0.10. None was selected.
- Q2 support for any linking change.
- That a fine-tuned model on HF is "better" than the base model. The phase-2 refits are documented
  in `runs/final/ft_refits_provenance.json`, but each was trained on 8 recordings. If one is
  published, its card must state that it did **not** beat the base model under equal decoding.
- Anything about the Kaggle test split, which was not scored.

## 10. Compute

| Item | GPU-h |
|---|---|
| Before phase 3b (phase 1–2 cumulative) | 4.11 |
| Phase 3b: regen of 14 fits, inference and prep (48 jobs, card 1) | 1.606 |
| Final: two `bnd.diarize` smoke runs (gpu_run wall, an upper bound) | 0.025 |
| **Cumulative** | **5.741 of the 12 GPU-h budget** |

No all-10 training was run. That is correct under the rule, because the default is ZS.
