# Phase 2 results: registered nested CV, zero-shot vs fine-tuned Nemotron-3-Diarization on Bengali-Loop

Frozen 2026-09-25T19:09:45, scored once at 2026-09-25T19:11:09 (`runs/cv/scored.jsonl`: one entry,
deviation null). Protocol: `docs/PREREG_phase2.md` plus Deviations D1–D7. Source of every number
below: `runs/cv/results.json` (frozen identity `12035a5f0eba…`, history copy
`runs/cv/results_history/results_001_12035a5f0eba.json`), `runs/cv/selection.json`,
`runs/cv/gpu_ledger.jsonl`, and two independent read-only audits (see §8). Smoke artefacts
(`runs/smoke2a`, `runs/link_smoke`, `runs/models/smoke_*`, `runs/cache/ft_smoke_*`) are not used.

## 1. Executive verdict

**Adoption gate: PASS on all six conditions. The fine-tuned system is the default.**

The default system is: Nemotron-3-Diarization fine-tuned on gap-5 trusted spans with the
inner-selected recipe (lr 5e-5; timing `shift` in 4 of 5 folds, `raw` in F1), native streaming
inference (264/40/340/40/300, bf16, 20 s cores + 5 s context), WeSpeaker cannot-link average-linkage
linking, with (θ_act, θ_clu) chosen on inner folds (0.40/0.55 in 3 folds, 0.50/0.65 and 0.50/0.55 in
the others).

| Gate condition (ft_sel vs zs_sel, outer folds, collar 0) | Value | Result |
|---|---|---|
| Corpus DER improvement ≥ 2.0 pp | 42.75 → 28.69, **14.06 pp** | PASS |
| Macro mean delta < 0 | −12.47 pp | PASS |
| Improves on ≥ 7/10 recordings | 8 better, 2 worse | PASS |
| No recording worse by > 5.0 pp | worst +1.71 pp (train_006) | PASS |
| Bootstrap 97.5th percentile of delta < 0 | 95% CI [−18.58, −8.51] pp (50k, seed 20260925) | PASS |
| No corpus regression at collar 0.25 | −14.48 pp | PASS |

Both independent audits reproduced the verdict from the frozen RTTMs. The whole gain comes from
speech activity (MISS 26.01% → 7.49%). Speaker linking did not get better, and after fine-tuning
it is the largest remaining error (§5).

## 2. Corpus DER (outer folds, 10 recordings, 27,856 s reference speech at collar 0)

DER % [MISS / FA / CONF]; macro = mean of per-recording DER.

| System | Collar 0 | Collar 0.25 | Collar 0.5 | Macro (c0) |
|---|---|---|---|---|
| zs_sel (calibrated zero-shot) | 42.75 [26.01/5.63/11.10] | 41.29 [25.75/4.72/10.81] | 39.88 [25.54/3.82/10.52] | 45.75 |
| **ft_sel (default)** | **28.69** [7.49/7.35/13.84] | **26.81** [7.00/6.26/13.55] | **24.94** [6.54/5.14/13.26] | 33.28 |
| zs_strict (0.50, 0.65, no selection) | 43.67 [27.60/5.43/10.64] | 42.26 [27.33/4.55/10.38] | 40.89 [27.10/3.67/10.12] | 46.13 |
| ft_strict (lr1e-5_raw, 0.50, 0.65) | 30.96 [11.89/6.95/12.12] | 29.22 [11.52/5.88/11.83] | 27.52 [11.16/4.81/11.55] | 34.38 |
| zs_sel + oracle link | 34.11 [26.01/5.63/2.47] | 32.65 [25.75/4.72/2.17] | 31.25 [25.54/3.82/1.89] | 36.20 |
| zs_sel + oracle speech | 19.26 [0.38/0.00/18.88] | 18.66 [0.36/0.00/18.29] | 18.10 [0.35/0.00/17.75] | 21.57 |
| ft_sel + oracle link | 17.35 [7.49/7.35/2.51] | 15.43 [7.00/6.26/2.17] | 13.55 [6.54/5.14/1.87] | 20.33 |
| ft_sel + oracle speech | 18.36 [0.38/0.00/17.98] | 17.75 [0.36/0.00/17.38] | 17.16 [0.35/0.00/16.81] | 20.91 |

### Paired bootstrap comparisons (delta = B − A, pp; 50k resamples, seed 20260925)

| Comparison | Collar | A | B | Delta | 95% CI | B better on |
|---|---|---|---|---|---|---|
| Primary: zs_sel → ft_sel | 0 | 42.75 | 28.69 | −14.06 | [−18.58, −8.51] | 8/10 |
| Primary | 0.25 | 41.29 | 26.81 | −14.48 | [−18.97, −8.89] | 8/10 |
| Primary | 0.5 | 39.88 | 24.94 | −14.94 | [−19.39, −9.32] | 9/10 |
| Strict: zs vs ft lr1e-5_raw @ (0.50, 0.65) | 0 | 43.67 | 30.96 | −12.71 | [−15.53, −9.44] | 10/10 |
| Strict | 0.25 | 42.26 | 29.22 | −13.03 | [−15.83, −9.73] | 10/10 |
| Strict | 0.5 | 40.89 | 27.52 | −13.37 | [−16.15, −10.03] | 10/10 |

## 3. Per-recording DER (collar 0, %)

Delta = ft_sel − zs_sel (pp). Fold and D2 exposure shown per recording.

| Rec | Fold | D2-exposed | zs_sel | ft_sel | Delta | zs_strict | ft_strict | Strict delta |
|---|---|---|---|---|---|---|---|---|
| train_001 | F1 | yes | 59.97 | 47.22 | −12.74 | 60.68 | 49.87 | −10.81 |
| train_008 | F1 | yes | 32.07 | 6.63 | −25.45 | 32.38 | 12.15 | −20.23 |
| train_011 | F2 | yes | 45.00 | 34.31 | −10.69 | 50.89 | 37.06 | −13.83 |
| train_009 | F2 | yes | 45.72 | 33.92 | −11.79 | 45.88 | 36.05 | −9.83 |
| train_013 | F3 | no | 69.50 | 57.02 | −12.48 | 66.59 | 58.19 | −8.40 |
| train_005 | F3 | no | 41.81 | 25.78 | −16.03 | 42.41 | 33.47 | −8.94 |
| train_004 | F4 | no | 29.18 | 9.59 | −19.59 | 30.37 | 13.86 | −16.51 |
| train_002 | F4 | no | 50.78 | 50.96 | **+0.18** | 49.33 | 39.03 | −10.30 |
| train_003 | F5 | no | 32.75 | 14.90 | −17.85 | 38.26 | 23.86 | −14.40 |
| train_006 | F5 | no | 50.71 | 52.42 | **+1.71** | 44.47 | 40.27 | −4.20 |

(Strict deltas are differences of the table values.) The two losses (train_002, train_006) are both
in folds F4/F5, where the selected recipe was lr5e-5_shift at cell (0.40, 0.55). On those two
recordings ft_sel is 11.92 and 12.16 pp worse than ft_strict, and ft_strict beats zs_strict on both.
The losses come from the recipe and cell choice, not from fine-tuning as such. Robustness (auditor):
dropping any one recording leaves the corpus delta between −12.18 and −15.62 pp. Exact Wilcoxon
p = 0.0098, one-sample t p = 0.0011, sign test for 8/10 p = 0.109.

Error components per recording (auditor): MISS falls on 10/10 recordings (by 8.1–27.6 pp), FA rises
on 10/10 (by 0.03–3.7 pp), and CONF rises on 7/10.

## 4. Selected recipes and cells (inner folds only)

Selection metric: pooled inner out-of-fold DER at collar 0 over each fold's 8 inner recordings, D1
grid (θ_act {0.40, 0.50, 0.60} × θ_clu {0.45, 0.55, 0.65, 0.75, 0.85}), D3 tie-breaks. No ties occurred.

| Outer fold | Zero-shot cell | Inner ZS DER | Selected recipe | FT cell | Inner FT DER | Refit δ (s) |
|---|---|---|---|---|---|---|
| F1 | (0.40, 0.55) | 43.12 | lr5e-5_raw | (0.40, 0.55) | 28.28 | n/a (raw) |
| F2 | (0.40, 0.55) | 42.19 | lr5e-5_shift | (0.50, 0.65) | 25.84 | 0.46 |
| F3 | (0.40, 0.55) | 40.59 | lr5e-5_shift | (0.50, 0.55) | 23.86 | 0.45 |
| F4 | (0.40, 0.55) | 44.49 | lr5e-5_shift | (0.40, 0.55) | 28.34 | 0.46 |
| F5 | (0.40, 0.55) | 43.42 | lr5e-5_shift | (0.40, 0.55) | 26.66 | 0.44 |

Best inner DER per recipe (at that recipe's own best cell):

| Fold | lr1e-5_raw | lr1e-5_shift | lr5e-5_raw | lr5e-5_shift |
|---|---|---|---|---|
| F1 | 31.98 | 32.21 | **28.28** | 30.03 |
| F2 | 27.21 | 28.47 | 26.97 | **25.84** |
| F3 | 26.47 | 26.33 | 25.01 | **23.86** |
| F4 | 31.09 | 32.20 | 28.56 | **28.34** |
| F5 | 30.00 | 30.35 | 26.95 | **26.66** |

lr 5e-5 won every fold, and shift beat raw in 4 of 5, but the margins between recipes are 0.2–2 pp on
inner folds and were never tested on outer folds. Inner-fit δ values ranged +0.37 to +0.56 s. δ was
estimated only from each fit's training recordings (verified per fit by the audit) and applied
signed (D4). Full 15-cell tables for every fold and recipe are in `runs/cv/selection.json`.

Grid edge: zero-shot chose θ_act = 0.40, the lowest registered value, in all 5 folds, and so did
3 of 5 fine-tuned picks. The optimum may lie below the grid (§7).

## 5. Oracle diagnostics: what limits DER

| Base | Real | Oracle linking (a) | Oracle speech (b) |
|---|---|---|---|
| zs_sel | 42.75 | 34.11 | 19.26 |
| ft_sel | 28.69 | 17.35 | 18.36 |

- **Zero-shot is limited by speech detection.** MISS is 26.01%. Oracle speech removes 23.5 pp,
  oracle linking only 8.6 pp.
- **Fine-tuning fixes speech detection, not linking.** MISS 26.01 → 7.49%, FA 5.63 → 7.35%. CONF
  rises 11.10 → 13.84% because more speech is now exposed to linking. Measured against detected
  speech (CONF / (ref − MISS)), CONF is flat: 15.01% before, 14.96% after (auditor). Local tracks
  are equally clean in both systems: oracle-link CONF is 2.47% vs 2.51%.
- **After fine-tuning, linking is the dominant remaining error.** Oracle linking cuts ft_sel by
  11.3 pp, to 17.35%. Oracle speech on ft_sel gives 18.36%, so the two remaining bottlenecks are now of
  similar size. Linking has the larger headroom from where the system is today.
- **Heavy over-splitting.** Predicted minus reference speakers (counted by the stage runner and
  confirmed by an auditor from the frozen RTTMs; not stored in results.json). Reference counts
  are 11, 20, 21, 10, 19, 17, 11, 21, 18, 21 for 001, 002, 003, 004, 005, 006, 008, 009, 011, 013.

| System | 001 | 002 | 003 | 004 | 005 | 006 | 008 | 009 | 011 | 013 | Σ\|err\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zs_sel | +36 | +21 | +5 | +29 | +48 | +68 | +1 | +21 | +2 | +30 | 261 |
| ft_sel | +33 | +15 | −4 | +18 | +15 | +47 | +3 | +6 | −1 | +13 | 155 |
| zs_strict | +19 | +4 | −4 | +18 | +26 | +36 | −1 | +8 | −3 | +12 | 131 |
| ft_strict | +14 | +5 | −5 | +11 | +17 | +23 | −2 | +8 | −4 | +9 | 98 |

  Oracle-link Σ|err| is 35 (zs) and 31 (ft). Cells at θ_clu = 0.55 split more than those at 0.65.

## 6. Compute (GPU-hours)

| Item | GPU-h |
|---|---|
| Phase 2a (smoke, before registration) | 0.27 |
| Plan + inner (zero-shot cache, 40 inner fits + inference) | 3.380 |
| Select, freeze, score (CPU only) | 0 |
| Outer (10 refits + outer inference) | 0.460 |
| **Registered protocol total** (`runs/cv/gpu_ledger.jsonl`, 101 jobs, 13,825.6 s) | **3.840** |
| **Cumulative** | **4.11 of the 12 GPU-h budget** |

All 101 jobs returned 0, all ran through `scripts/gpu_run.sh` on card 1 (A6000), none overlapped in
time, and peak reserved memory was 2.95 GiB (limit 20 GB). A co-tenant load spike slowed three x13
inner fits by 2–2.5×, adding about 0.1 GPU-h.

## 7. Deviations and exposure (all disclosed)

| ID | Deviation | Effect on interpretation |
|---|---|---|
| D1 | θ_clu grid changed from {0.25 … 0.65} to {0.45, 0.55, 0.65, 0.75, 0.85}, owner-approved before any registered DER | Motivated by the phase-2a smoke on **outer fold F1** (train_001: 61 predicted vs 11 reference speakers). Selection stayed inner-only, but outer-fold information shaped the grid. |
| D2 | Smoke runs had scored outer F1 (train_001, 008) and F2 (train_011, 009) | Those labels were seen before the protocol ran. The registered run rebuilt everything from scratch (`reused_phase2a` false, no symlinks in the cv cache). See the subset analysis below. |
| D3 | Tie-breaks: nearest to (0.50, 0.65), then lower θ_act, then lower θ_clu; recipe ties → lr 1e-5, then raw. Strict cell moved to (0.50, 0.65) | No ties occurred. The strict comparison uses (0.50, 0.65). |
| D4 | δ applied signed | train_004's own shift is negative (−0.21 s); all fit medians are positive. |
| D5 | TitaNet secondary embedder descoped | Not reported. |
| D6 | Fixed 165 steps gives about 7 passes (6.9–7.9) for inner fits vs 5.3–5.7 for refits | Inner selection saw a different exposure than the refits used on outer folds. |
| D7 | Tracks with 1.0 ≤ usable < 1.5 s are embedded from one piece shorter than 1.5 s; "confident" means p ≥ θ_act | Recorded in every embedding meta. |

**D2 subset analysis (auditor).** On the six recordings no smoke run ever scored (002, 003, 004,
005, 006, 013), the delta is −12.47 pp, 95% CI [−17.66, −4.48], and fine-tuned wins 4 of 6. Both
losses are in this clean subset, and all four exposed recordings are wins. With n = 6 this does not
show leakage, and the clean-subset CI still excludes 0, but it must be reported with the headline.

**Registered item not run.** PREREG registers overlap correspondence between adjacent windows as a
SECONDARY linking variant that must be reported. It was **not run**: all 655 link markers are
variant "primary", and no deviation covers the omission (D5 descopes only TitaNet). It exists only
in the phase-2a smoke on F1, which is not a registered result. This is an open protocol gap. It does
not affect the gate, which is defined on the primary variant.

## 8. Audits: discrepancies and how they were resolved

Two independent read-only audits ran CPU-only. One rescored with pyannote.metrics and its own RTTM
parser; the other used pyannote directly with its own bootstrap. Neither edited repository files.
The audits confirmed:
- No self-scoring in any of the 50 fits: plan, fit report, NeMo manifest, ledger arguments and caches
  all agree.
- Selection used inner folds only.
- δ came from training recordings only.
- All 80 frozen sha256 values match the files on disk.
- One scoring pass.
- PREREG hyperparameters were followed.
- Every collar-0 per-recording DER matches to within 3e-15.
- D3 selections re-derive to the same picks.

| # | Discrepancy | Resolution |
|---|---|---|
| 1 | Zero-shot outer RTTMs are byte-identical to inner zero-shot hypotheses. Each recording is an inner recording for the other 4 folds, so its zero-shot DER existed in `selection.json` before freeze. | By design, not leakage: no fold's own labels entered its own selection. The claim "no outer DER seen before freeze" holds only for the fine-tuned and oracle systems, and is stated that way here. |
| 2 | Secondary overlap-correspondence variant not run. | Not resolved. Disclosed in §7 as an unregistered omission; see next steps. |
| 3 | `bnd/score.py` unions overlapping same-speaker reference rows before scoring (train_001: 1, train_002: 3, train_011: 2 merges), with collars placed at the original row boundaries. PREREG's Scoring section does not state this. | Scoring with pyannote directly reproduces the stored 42.75 / 28.69 exactly. One auditor's own parser without the union gave 42.80 / 28.75 (delta 14.05 vs 14.06 pp); train_011 oracle-speech was 0.248585 vs 0.247325. The gate verdict is identical either way. The rule is documented here as part of the scoring definition. At collars 0.25 and 0.5, pyannote's native collar handling differs per recording by at most 0.07 pp (corpus by 0.01 pp), because score.py places collars at the original boundaries. |
| 4 | D1 grid change was informed by outer-F1 labels; D2 exposure of F1 and F2. | Disclosed (§7). The results stand as registered but must not be presented as fully blind. |
| 5 | Zero-shot pick at the θ_act grid edge (0.40) in all folds, with MISS at 26%. | Disclosure only; the grid is registered and was not changed. The 14 pp gain may overstate the gain over a best-tuned zero-shot. |
| 6 | Losses on train_002 and train_006 trace to the lr5e-5_shift @ (0.40, 0.55) choice in F4 and F5. | Noted (§3). No change; the selection rule is registered. |

## 9. Claims

**Supported**
1. Under the registered nested CV, the selected fine-tuned system beats calibrated zero-shot by
   14.06 pp corpus DER at collar 0 (95% CI [−18.58, −8.51]). The gain holds at collars 0.25
   (−14.48) and 0.5 (−14.94).
2. The gain holds without any selection: the strict comparison gives −12.71 pp and improves 10/10
   recordings.
3. The gain comes from speech activity (MISS 26.0 → 7.5%). Linking quality per detected second is
   unchanged.
4. After fine-tuning, cross-window speaker linking (CONF 13.84%, heavy over-splitting) is the
   dominant remaining error.
5. The protocol ran within budget: 4.11 of 12 GPU-h, one card, peak reserved memory under 3 GB.

**Not supported**
- A population-level CI. The data are 10 recordings from one corpus, with one training seed (42)
  and one refit per fold, and the bootstrap ignores seed variance. With n = 10 the percentile
  bootstrap is somewhat anti-conservative.
- "Improves on most recordings" as a significance claim: the sign test for 8/10 gives p = 0.11.
- "Shift timing helps" or "lr 5e-5 is better". These rest on 0.2–2 pp inner differences and were
  never tested on outer folds.
- "Fine-tuning improves speaker discrimination".
- A fully blind or unseen-data framing (D1, D2).
- The size of the gain over a best-tuned zero-shot (grid edge, discrepancy 5).
- Transfer of the 165-step recipe to other data amounts (D6).
- Any claim about the Kaggle test set. It is unlabelled, and no test-set predictions have been made.

## 10. Recommended next steps

1. **Final model, all 10 recordings (per PREREG).** Train once on the gap-5 spans of all 10
   recordings with the default recipe. The recipe is lr 5e-5 (4/5 folds chose shift, and the
   majority rule should be written down before training). Estimate δ from all 10 recordings, and
   keep the fixed 165 steps and seed 42. Because 165 steps on 10 recordings gives about 4.4 passes, a
   further exposure change beyond D6, log whether to keep 165 steps or scale by epochs as a
   deviation **before** training. Pick (θ_act, θ_clu) by pooled out-of-fold DER over all 10 CV
   predictions of the selected recipe, not by a new search on outer results. By analogy with the refit
   train jobs (92–97 s of GPU wall each), training should cost a few minutes of GPU time.
2. **Kaggle late submission.** Run the final model on the 14 test WAVs through `scripts/gpu_run.sh`.
   Export with the `bnd/score.py` exporter and validator (sample_submission order; sorted,
   non-overlapping, 0 ≤ start < end ≤ duration). Report the leaderboard score as external and
   separate from the CV claims.
3. **Close the protocol gap.** Run the registered secondary overlap-correspondence variant on the
   frozen outer caches and report it, clearly labelled as run after scoring. Selection and the gate
   stay unchanged.
4. **Next research target: linking.** Oracle linking is worth about 11 pp, and systems over-split by
   up to +47 speakers. Candidates are a higher or extended θ_clu range, extending θ_act below 0.40,
   overlap correspondence, and a second embedder. Any new grid needs a fresh preregistration and
   fresh folds or held-out data, because these outer folds are now spent.
5. **Release (later phase, per PREREG).** A private GitHub repo with the core reproducible files, and
   a public HF model (MIT if the licences permit; author Emrul Zawad). The model card must carry the
   D1/D2 exposure disclosures, the n = 10 single-corpus caveat, the MISS-driven nature of the gain,
   and the unsupported claims above.
