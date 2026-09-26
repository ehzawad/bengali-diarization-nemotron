# Phase 5 results: linking challenger + independent synthetic hold-out

Protocol: `docs/PREREG_phase5.md` (binding; frozen 2026-09-26 before any phase-5 number and before
the hold-out was downloaded). Subset: `docs/phase5_holdout_subset.txt` (60 stems).
Every number below comes from these sources, which were produced by commands run in this phase and
then checked by two independent read-only audits:

| Artefact | sha256 (prefix) |
|---|---|
| `runs/p5/dev_results.json` (Part A, scored once, `runs/p5/scored.jsonl`) | `45b1ec9a8728…` |
| `runs/p5/selection.json` | `e0d53cfbe417…` |
| `runs/p5/frozen.json` (80 outer RTTMs; frozen identity `e7ff878d78bd…`) | `44ff79af2ff2…` |
| `runs/p5/final_configs.json` | `19b6554dcabd…` |
| `runs/p5/holdout_validation.json` (written before any system output) | `536457eeb750…` |
| `runs/p5/holdout_frozen.json` (420 RTTMs; frozen identity `277e139e…`) | `f136f705d882…` |
| `runs/p5/holdout_results.json` (scored once, `runs/p5/holdout_scored.jsonl`) | `6bbb8f610313…` |
| `runs/p5/gpu_ledger.jsonl` | — |

> **Status of evidence.** All 10 Bengali-Loop recordings are **DEVELOPMENT** data (five or more
> looks). Every Part-A number is a development number, never a held-out one. Part B
> (smam/bengali-diarization-synthetic-v4, revision `c3c7cb03…`, 60 mixtures, 30 h) is the only
> independent evaluation. It is **synthetic**: IndicVoices/Kathbath read and extempore speech placed
> at random. It is not natural drama. The base model Nemotron-3-Diarization `f667ed73` has DISPLACE
> 2024 (which includes Bengali) in its training data.

## 1. Executive verdict

**Part A, dev adoption criterion: the challenger is NOT adopted.** The challenger is 120 s native
sessions plus a contrastively fine-tuned WeSpeaker embedder (`s120_ft`). The incumbent is 20 s
sessions with the stock embedder (`s20_stock`).

| Criterion (both required) | Value | Result |
|---|---|---|
| Corpus DER, collar 0, no adapter, Δ ≤ −1.0 pp | 26.97 → 28.59 %, **Δ +1.62 pp** (95 % CI [−0.54, +4.01]; challenger wins 2/10) | **FAIL** |
| Σ\|count err\| not up by more than 10 % | 192 → 45 | PASS |

**Part B, primary hold-out claim: NO demonstrated benefit of the challenger.** The comparison is in
acoustic mode (g = 0), collar 0, paired over mixtures, 50k resamples, seed 20260926.

| Claim condition (both required) | Value | Result |
|---|---|---|
| Δ ≤ −1.0 pp | incumbent 50.11 → challenger 49.37 %, **Δ −0.74 pp** | **FAIL** |
| 97.5th percentile < 0 | 95 % CI **[−1.97, +0.51]**; challenger wins 36/60 | **FAIL** |

**The incumbent stays**: base Nemotron-3-Diarization, 20 s sessions, stock WeSpeaker, avg-link. The
fine-tuned embedder is the one consistent dev effect, and it is **harmful**: it raises CONF by about
1.7 pp in both session-length cells. The 120 s sessions do not change dev DER, but they sharply
reduce speaker over-counting. On the hold-out, that count gain does not turn into a DER gain.

Every Nemotron system is limited by overlap. About 33 pp of the hold-out DER is overlap MISS, close
to the floor for a system that outputs one speaker per frame. The untuned pyannote community-1 has
lower hold-out DER than every Nemotron system (exploratory; §4.4).

## 2. Part A: development factorial (nested CV, folds F1–F5)

### 2.1 Selection (inner folds only; simplest config within 0.5 pp of the minimum)

Each cell searched the same 20 configurations: θ_act ∈ {0.25, 0.40} × θ_clu ∈ {0.45, 0.55, 0.60,
0.65, 0.75} × g ∈ {0, 5}, with the avg linker. Every selection used avg linking, g = 5 and
θ_act = 0.40. The argmin sometimes had θ_act = 0.25, but 0.40 was always inside the 0.5 pp band. The
table shows θ_clu and pooled inner DER at collar 0. LORO is leave-one-recording-out stability of the
selection.

| Fold | s20_stock | s120_stock | s20_ft | s120_ft |
|---|---|---|---|---|
| F1 | c0.60, 28.87 (8/8) | c0.55, 28.88 (8/8) | c0.75, 32.37 (5/8) | c0.65, 29.41 (8/8) |
| F2 | c0.60, 26.06 (6/8) | c0.55, 26.11 (8/8) | c0.65, 25.72 (8/8) | c0.65, 26.71 (5/8) |
| F3 | c0.60, 23.86 (8/8) | c0.55, 22.69 (8/8) | c0.60, 26.12 (6/8) | c0.60, 25.15 (**2/8**) |
| F4 | c0.60, 28.82 (8/8) | c0.55, 29.29 (7/8) | c0.75, 28.98 (8/8) | c0.60, 28.79 (8/8) |
| F5 | c0.60, 27.35 (7/8) | c0.55, 28.03 (7/8) | c0.65, 27.20 (7/8) | c0.60, 29.27 (8/8) |

Final configurations are the joint mode of the five fold selections (`runs/p5/final_configs.json`):

| Cell | Final config | Mode support |
|---|---|---|
| s20_stock (incumbent) | `s20_avg_g5_a0.40_c0.60` | 5/5 (= phase-3b ZS\* default) |
| s120_stock | `s120_avg_g5_a0.40_c0.55` | 5/5 |
| s20_ft | `s20_avg_g5_a0.40_c0.65` | 2/5, tied with c0.75; the tie order picked c0.65 |
| s120_ft (challenger) | `s120_avg_g5_a0.40_c0.60` | 3/5 |

### 2.2 Outer corpus DER, 4-cell factorial, with and without the floor adapter

Values are DER % [MISS/FA/CONF]. There are 80 outer RTTMs, sha256-frozen before scoring. The floor
adapter (`bnd.p5.floor_adapter`) floors both boundaries to whole seconds after selection. It was not
part of the search.

| Cell | Collar 0 | Collar 0.25 | Collar 0.5 | Σ\|count err\| |
|---|---|---|---|---|
| **s20_stock (incumbent)** | **26.97** [6.90/6.85/13.22] | 25.13 [6.35/5.85/12.93] | 23.34 [5.83/4.86/12.65] | 192 |
| s120_stock | 26.97 [7.01/6.62/13.34] | 25.13 [6.45/5.63/13.05] | 23.33 [5.91/4.64/12.78] | 64 |
| s20_ft | 28.70 [6.90/6.84/14.96] | 26.84 [6.34/5.84/14.66] | 25.03 [5.81/4.85/14.37] | 141 |
| **s120_ft (challenger)** | **28.59** [7.01/6.62/14.95] | 26.75 [6.45/5.63/14.67] | 24.95 [5.92/4.64/14.39] | 45 |
| s20_stock + floor | 24.91 [6.15/6.03/12.73] | 23.69 [5.87/5.27/12.55] | 22.42 [5.59/4.47/12.36] | 167 |
| s120_stock + floor | 24.92 [6.25/5.78/12.89] | 23.70 [5.96/5.02/12.72] | 22.43 [5.66/4.22/12.54] | 62 |
| s20_ft + floor | 26.77 [6.14/6.02/14.62] | 25.52 [5.86/5.26/14.41] | 24.21 [5.57/4.46/14.18] | 120 |
| s120_ft + floor | 26.57 [6.25/5.78/14.54] | 25.34 [5.97/5.02/14.36] | 24.06 [5.66/4.22/14.17] | 45 |

The factorial read as main effects:

- **Session length (S).** 20 → 120 s leaves DER unchanged but cuts speaker-count error about
  threefold: 192 → 64 with stock, 141 → 45 with FT.
- **Embedder (E).** Stock → FT adds about 1.6–1.7 pp of CONF in both S cells. MISS and FA are
  identical within an S level, because the embedder only affects linking.
- **Floor adapter.** It lowers DER by about 2 pp in every cell and does not change the ranking.

### 2.3 Paired comparisons (Δ = B − A, pp; recording bootstrap, 50k resamples, seed 20260925)

| Comparison | Collar | Δ | 95 % CI | B wins/10 |
|---|---|---|---|---|
| s20_stock → s120_ft (challenger) | 0 | +1.62 | [−0.54, +4.01] | 2 |
| s20_stock → s120_ft | 0.25 | +1.62 | [−0.56, +4.03] | 2 |
| s20_stock → s120_ft | 0.5 | +1.61 | [−0.59, +4.04] | 2 |
| s20_stock → s120_stock | 0 | −0.00 | [−2.28, +2.72] | 8 |
| s20_stock → s20_ft | 0 | +1.73 | [−2.07, +4.77] | 1 |
| s20_stock+floor → s120_ft+floor | 0 | +1.66 | [−0.54, +4.08] | 3 |
| s120_stock → s120_ft (audit, not in the stage report) | 0 | +1.62 | [+0.45, +3.05] | 1 |

The worst recording for the challenger is train_005, at +7.9 pp. For s120_stock, train_005 is
+9.2 pp worse, while train_006 is 6.8 pp better.

Per-recording DER at collar 0 (%):

| Rec | s20_stock | s120_stock | s20_ft | s120_ft |
|---|---|---|---|---|
| train_001 | 45.35 | 45.30 | 29.60 | 51.03 |
| train_002 | 38.07 | 35.23 | 42.91 | 38.42 |
| train_003 | 13.51 | 14.06 | 23.14 | 18.29 |
| train_004 | 11.39 | 10.03 | 13.04 | 10.06 |
| train_005 | 30.78 | 39.98 | 31.81 | 38.69 |
| train_006 | 42.31 | 35.54 | 46.72 | 37.73 |
| train_008 | 7.24 | 7.19 | 7.30 | 7.25 |
| train_009 | 36.58 | 36.08 | 40.23 | 39.44 |
| train_011 | 28.59 | 28.51 | 29.51 | 29.44 |
| train_013 | 56.90 | 55.34 | 60.54 | 57.07 |

## 3. Part B: reference validation (before any system output existed)

- **Data.** 60 mixtures × 1800 s = 30.0 h. `sha256sum -c data/holdout_synth_SHA256.txt` passed on all
  180 entries (wav, rttm and json). The directory is read-only.
- **Labelled speech.** 71,604 s, which is 66.3 % of the audio. Overlap is 37.6 % of labelled speech.
- **Silent to both detectors.** 5.33 % of labelled speech is called silent by both Silero and
  zero-shot Nemotron activity. None of it lies in runs of 5 s or longer. The worst single mixture is
  8.8 %. Overall speech fraction is 0.607 for Nemotron and 0.619 for Silero. The audit did not
  independently re-verify these detector statistics.
- **Turns.** There are 5,848 turns. The mean of per-mixture mean turn length is 19.6 s, and the median
  of per-mixture medians is 14.0 s. Turns run from 3.0 s to 180 s.
- **Speaker counts.** Every mixture's speaker count equals its folder k. The two 1-speaker mixtures
  have no overlap.

| Speaker bin | Mixtures | Labelled fraction | Overlap | Silent to both |
|---|---|---|---|---|
| 1–4 | 8 | 0.684 | 0.227 | 7.5 % |
| 5–8 | 8 | 0.658 | 0.366 | 6.5 % |
| 9–16 | 16 | 0.661 | 0.386 | 5.0 % |
| 17–30 | 28 | 0.660 | 0.417 | 4.6 % |

**Scoring convention.** Whole-file UEM, overlap included, collar 0 primary. The RTTM files are used
unedited, but **same-speaker self-overlapping reference rows are unioned** (phase-1 SPEC convention,
`bnd.score`). The synthetic RTTMs contain many of these. Raw pyannote (`unique=False`) would count them
twice, raising the reference total by 5.74 % (106,602 → 112,721 s) and every DER by about 3 pp. The
audit re-ran every comparison under the raw convention and **no conclusion changes**: the primary Δ
becomes −0.69 [−1.85, +0.49].

## 4. Part B: results (frozen batch, scored once)

The frozen batch is 7 system-modes × 60 mixtures = 420 RTTMs, sha256-frozen at 04:10:39 and scored
at 04:12:25.

- **Incumbent.** Base model, S=20, stock embedder. Acoustic mode `s20_avg_g0_a0.40_c0.60`; benchmark
  mode is the Part-A final `…g5…`.
- **Challenger.** Base model, S=120, all-10 FT embedder (sha `9bed16ff…`). Acoustic mode
  `s120_avg_g0_a0.40_c0.60`; benchmark mode `…g5…`.
- **community-1.** pyannote/speaker-diarization-community-1 @`3533c8cf`, untuned, exclusive output.
- **FT-Nemotron.** All-10 model (sha `1d72a071…`): 165 steps, lr 5e-5, shift timing, δ 0.46 s,
  seed 42. It uses the phase-3b FT joint-mode config `s20_avg_g0_a0.40_c0.55`. That config already
  has g = 0, so its benchmark mode is byte-identical to its acoustic mode, as the PREREG specifies.

### 4.1 Corpus DER (%) [MISS/FA/CONF] and Σ|count err|

| System | Collar 0 | Collar 0.25 | Collar 0.5 | Σ\|count err\| |
|---|---|---|---|---|
| incumbent, acoustic | 50.11 [38.15/0.02/11.94] | 49.75 [37.83/0.00/11.93] | 49.43 [37.53/0.00/11.90] | 2216 |
| incumbent, benchmark | 46.26 [33.50/0.18/12.58] | 45.76 [33.03/0.15/12.58] | 45.31 [32.60/0.14/12.57] | 2216 |
| challenger, acoustic | 49.37 [38.28/0.02/11.07] | 49.02 [37.95/0.00/11.07] | 48.72 [37.65/0.00/11.07] | 520 |
| challenger, benchmark | 45.46 [33.55/0.18/11.73] | 44.98 [33.07/0.15/11.75] | 44.54 [32.64/0.14/11.76] | 520 |
| community-1 | **44.70** [35.00/0.15/9.56] | 44.03 [34.64/0.00/9.39] | 43.53 [34.29/0.00/9.24] | **144** |
| FT-Nemotron (both modes) | 45.72 [33.64/2.02/10.06] | 45.12 [33.25/1.84/10.03] | 44.52 [32.87/1.67/9.99] | 2236 |

Signed speaker-count error (mean, and how many of the 60 mixtures are over- or under-counted):

| System | Mean signed error | Over / under |
|---|---|---|
| Incumbent | +36.9 | 60 / 0 |
| FT-Nemotron | +37.3 | 60 / 0 |
| Challenger | +8.7 | 59 / 0 |
| community-1 | −0.2 | 34 / 20 |

### 4.2 By speaker-count bin (collar 0 DER % / speaker-count MAE)

| System | 1–4 | 5–8 | 9–16 | 17–30 |
|---|---|---|---|---|
| incumbent, acoustic | 38.61 / 26.62 | 47.09 / 33.62 | 50.05 / 38.50 | 53.66 / 39.93 |
| incumbent, benchmark | 34.05 / 26.62 | 42.24 / 33.62 | 46.21 / 38.50 | 50.23 / 39.93 |
| challenger, acoustic | 33.41 / 6.62 | 43.86 / 10.75 | 47.55 / 11.31 | 55.60 / 7.14 |
| challenger, benchmark | 28.58 / 6.62 | 38.77 / 10.75 | 43.51 / 11.31 | 52.30 / 7.14 |
| community-1 | 38.10 / 1.88 | 42.06 / 2.25 | 42.93 / 1.50 | 47.94 / 3.11 |
| FT-Nemotron | 32.43 / 21.25 | 43.41 / 35.75 | 46.18 / 38.25 | 49.21 / 41.71 |

The challenger beats the incumbent in bins 1–4, 5–8 and 9–16. It is worse in bin 17–30, by +1.94 pp
in acoustic mode.

### 4.3 Primary comparison (registered)

Acoustic mode, Δ = challenger − incumbent, paired mixture bootstrap, 50k resamples, seed 20260926.

| Collar | Δ pp | 95 % CI | Challenger wins/60 |
|---|---|---|---|
| **0 (primary)** | **−0.74** | **[−1.97, +0.51]** | 36 |
| 0.25 | −0.73 | [−1.97, +0.54] | 36 |
| 0.5 | −0.71 | [−1.97, +0.56] | 36 |

P(Δ ≥ 0) = 0.12, and seed 1 gives [−1.99, +0.51]. **Verdict: no demonstrated hold-out benefit.**

### 4.4 Secondary comparisons (exploratory; Δ = B − A, pp, collar 0)

The first five rows are the registered secondary comparisons. The last four were **computed after
scoring** for this report, from the scored per-mixture values in `holdout_results.json`, with
`bnd.score.paired_bootstrap` (50k resamples, seed 20260926). They are post-hoc.

| Comparison (A → B) | A | B | Δ | 95 % CI | B wins/60 |
|---|---|---|---|---|---|
| incumbent → challenger, benchmark mode | 46.26 | 45.46 | −0.80 | [−2.13, +0.55] | 37 |
| incumbent (ac) → community-1 | 50.11 | 44.70 | −5.41 | [−6.70, −4.12] | 51 |
| base (ac) → FT-Nemotron | 50.11 | 45.72 | −4.39 | [−5.08, −3.72] | 57 |
| base (bm) → FT-Nemotron | 46.26 | 45.72 | −0.54 | [−1.26, +0.15] | 35 |
| base (bm) → FT-Nemotron, collar 0.5 | 45.31 | 44.52 | −0.79 | [−1.53, −0.09] | 38 |
| *post-hoc:* incumbent acoustic → incumbent benchmark (g0 → g5) | 50.11 | 46.26 | −3.85 | [−4.10, −3.61] | **60** |
| *post-hoc:* challenger acoustic → challenger benchmark | 49.37 | 45.46 | −3.91 | [−4.19, −3.63] | **60** |
| *post-hoc:* incumbent benchmark → community-1 | 46.26 | 44.70 | −1.56 | [−2.94, −0.19] | 35 |
| *post-hoc:* FT-Nemotron → community-1 | 45.72 | 44.70 | −1.02 | [−2.25, +0.26] | 30 |

Voice reuse makes every mixture-bootstrap CI here optimistic (too narrow); see §5.

### 4.5 Where the hold-out error lives (audit, 10 ms frame grid, matches pyannote MISS to 0.01 pp)

58.1 % of the scored reference lies in overlap frames. A system that outputs one speaker per frame
therefore has a MISS floor of **32.83 %**.

| System | Overlap-frame MISS | Single-speaker-frame MISS |
|---|---|---|
| Incumbent | 33.14 | 5.02 |
| Challenger | 33.16 | 5.12 |
| community-1 | 32.91 | 2.08 |
| FT-Nemotron | 33.01 | 0.63 |

- **Overlap.** Nemotron can emit up to 4 speakers, yet recovers almost no overlap here. About 33 pp
  of every system's DER is shared overlap MISS. All differences between systems come from the
  remaining 12–17 pp.
- **FT-Nemotron's −4.39 pp.** It comes essentially entirely from single-speaker MISS, while FA rises
  from 0.02 to 2.02. Gap-fill g = 5 gives the base model most of the same gain (−3.85 pp, 60/60).
- **Many-speaker overlap.** Regions with five or more simultaneous speakers are only 0.19 % of the
  reference.

## 5. Audits and discrepancies

Two independent read-only audits ran, one per part. Neither edited files or used the GPU.

**Part A: no defects found.**

- **Hashes.** All 80 frozen hashes match. Re-linking the 40 no-adapter outer RTTMs from caches, then
  applying the adapter, gives 80/80 byte-identical files.
- **DER.** An independent pyannote rescoring matches every per-recording DER exactly at collar 0.
  At collars 0.25 and 0.5 it differs by at most 0.032 pp per recording (0.004 pp corpus). The
  difference comes from the registered collar convention, which builds collars from the original
  rows, including nested same-speaker rows.
- **Inner tables.** All 2000 inner predictions were rescored; the maximum pooled difference is
  1.4e-15.
- **Selection.** An independent reimplementation reproduced all 20 selections and all 4 joint
  modes.
- **Leakage.** No embedder leakage: 16 fit manifests and 200 embedding metas were checked, and no
  embedder was ever used on its own training recordings.
- **Determinism.** Fits are deterministic: `p5emb_x12` reproduced the smoke fit's sha256
  `f594dc4a…`.
- **Default path unchanged.** The incumbent's outer RTTMs are byte-identical to phase-3b `zs_star`
  (DER 0.269698808), and its F1 inner DER equals phase-3b's value (0.28873912). The default
  link/infer path is therefore unchanged.
- **Earlier artefacts.** `runs/cv` (80/80) and `runs/p3b` (100/100) hashes are intact. No
  file under runs/cv, runs/p3b, runs/final, runs/diag, data/prepared/ref_rttm or data/raw is newer
  than the PREREG.

**Part B: the stage report is correct.**

- **Subset.** Independently re-derived; it equals the subset file.
- **Freeze before score.** Order of events: final configs 02:45:49, first hold-out output 02:47:57,
  validation 03:04:03, freeze 04:10:39, score 04:12:25. There is one line in `holdout_scored.jsonl`.
- **Hashes.** All 420 RTTM hashes and all 10 config hashes match.
- **DER.** Per-mixture DER recomputed without `bnd` imports matches exactly (maximum difference 0 s)
  for 7 systems × 60 mixtures × 3 collars.
- **Bootstrap.** Every bootstrap reproduces, and the primary result is seed-stable.

Disclosures and discrepancies:

1. **Pre-freeze read of the hold-out references (unattributed).** ext4 relatime atimes show every
   hold-out RTTM was read between 01:01:16 and 01:04:51, and the JSONs between 01:04:40 and 01:04:51.
   The download finished at 00:52:52, well before the Part-A final configs (02:45:49). The reading
   process was not identified; a download or verification step is most likely.
   - No Part-A code path takes hold-out input.
   - Every Part-B config traces to hash-verified Part-A or phase-3b artefacts.
   - **No evidence of tuning was found, but the atime record cannot exclude inspection** before
     Part A was finalised.
2. **Voice reuse is heavier than "324 voices" suggests.**
   - The 60-mixture subset fills 930 speaker slots with only **203 unique voices** (192 IndicVoices,
     11 Kathbath).
   - 172 voices appear in more than one mixture, and one voice appears in up to 14.
   - 1204 of the 1770 mixture pairs share at least one voice.
   - The mixture-bootstrap CIs are therefore optimistic. This makes the primary non-claim *more*
     robust, but weakens every "CI excludes 0" among the secondary rows.
3. **Same-speaker union convention** (§3): absolute DER is about 3 pp lower than raw pyannote.
   Conclusions are unchanged.
4. **Fragile challenger selections.**
   - F3 s120_ft has LORO 2/8.
   - s20_ft's final config is a 2/5 joint mode tied with c0.75.
   - F1 s20_ft picked c0.75 because c0.60 sat about 0.014 pp outside the 0.5 pp band.
5. **FT-Nemotron benchmark mode equals acoustic mode** (g = 0 already). This is not a bug.
6. **GPU projections vs actuals.**
   - The A-fits log prints `projected_gpu_h 2.194`, a pre-run estimate. The Part-A actual was
     1.442 h.
   - community-1 took 2653 s against a 0.946 h projection.
7. **Council artefact label.** In `docs/council_round3_artifacts/compare_systems.json`, the rows
   labelled "zs_default g0" and "g5" carry identical numbers, because `zs_star` already includes
   g = 5. The label is misleading; the numbers are not affected.

## 6. What can and cannot be claimed

**Can claim (development-grade, Bengali-Loop, 10 recordings, no significance):**

- The challenger (120 s + FT embedder) failed the preregistered adoption criterion: +1.62 pp
  [−0.54, +4.01], 2/10 wins.
- With this one fixed recipe (no search, 6–8 training recordings, about 88 classes), the FT WeSpeaker
  embedder raised CONF by about 1.6–1.7 pp at both session lengths. In the s120 pairing the CI
  excludes 0.
- 120 s sessions cut Σ|count err| from 192 to 64 with DER unchanged at the corpus level. The CI is
  wide, and train_005 is 9.2 pp worse.
- The floor adapter lowers Bengali-Loop DER by about 2 pp in every cell. This is agreement with the
  whole-second reference convention, not better acoustics.
- Phase 5 reproduced phase-3b `zs_star` exactly.

**Can claim (independent, synthetic hold-out, 60 mixtures):**

- No demonstrated hold-out benefit of the challenger over the incumbent: Δ −0.74 pp [−1.97, +0.51].
  The incumbent stays.
- The challenger greatly reduces over-counting (Σ 2216 → 520), but still over-counts 59 of 60
  mixtures, and is worse in the 17–30 bin.
- All Nemotron systems are bounded by overlap MISS (about 33 pp, near the exclusive-output floor).
- *Exploratory:* untuned community-1 has lower DER than the Nemotron incumbent (−5.41 pp acoustic;
  −1.56 pp against benchmark mode, post-hoc), and it is the only system with unbiased speaker counts.
- *Exploratory:* Bengali-Loop convention fine-tuning did not hurt on exactly-timed references.
- *Exploratory, post-hoc:* same-speaker gap-fill g = 5 lowers hold-out DER by 3.85 pp and improves
  60/60 mixtures, so it is not only a Bengali-Loop convention artefact on this data. It fills
  Nemotron's single-speaker MISS.

**Cannot claim:**

- Any held-out or generalisation number from Part A.
- "FT embedding hurts in general" (one recipe), or "120 s improves DER".
- Any benefit on natural Bengali drama or conversation from Part B. Part B is random placement of
  clean read and extempore speech, with 37.6 % overlap and turns of at least 3 s.
- That community-1 beats Nemotron on real Bengali. On Bengali-Loop dev, the council-round-3 artefact
  (`compare_systems.json`) has community-1 at 46.04 % (g0) and 29.91 % (g5) against 26.97 % for
  Nemotron ZS\*. IndicVoices overlap with the training data of pyannote, WeSpeaker or Nemotron is
  unknown.
- Exact or significant secondary CIs (voice reuse).
- Absolute hold-out DER as a quality measure. It is convention-dependent and dominated by overlap.
- That the hold-out references were never read before Part A was frozen (§5.1).
- Anything about the Kaggle test split, which was not scored.

## 7. Compute

| Item | GPU-h |
|---|---|
| Before phase 5 (phases 1–3b, final, council round 3) | ≈ 6.03 |
| Phase 5 smoke fit and diagnostic (another agent) | 0.071 |
| Part A: prep, S=120 probs and embeddings, 16 embedder fits, 15 FT embedding jobs | 1.371 |
| Part B: probs, embeddings, community-1 (2653 s), FT-Nemotron train (97 s) and inference | 1.225 |
| **Phase 5 total** | **2.667 of the 5.5 h allowance** |
| **Cumulative** | **8.697 of the 12 h hard stop** |

Every GPU job ran through `scripts/gpu_run.sh` on card 1 (A6000). Silero ran on CPU. No orphan
processes remain, and both GPUs were idle at the end of each stage.

## 8. Recommended final release configuration

**The challenger is not released.** The release is the incumbent. Base model:
Nemotron-3-Diarization @`f667ed73`, `.nemo` sha256 `867c53f5…`. Embedder: stock WeSpeaker
resnet34-LM @`837717dd`. The decoder is shared by both modes: 20 s native sessions with 5 s context
each side, streaming 264/40/340/40/300, bf16, θ_act 0.40, avg-link cannot-link at θ_clu 0.60.

| Mode | Decoding after linking | Provenance | Evidence |
|---|---|---|---|
| **Default** (`bnd.diarize` as shipped, `runs/final/final_config.json`, sha `82bf7245…`) | same-speaker gap-fill g = 5 s; full-resolution timestamps | 5/5 inner selections in phase 3b and again in phase 5 (s20_stock) | dev 26.97 % (second-look); hold-out 46.26 % (benchmark mode) |
| **Bengali-Loop benchmark mode** | default + `floor_adapter` (whole-second floor) | council round 3 adapter, reported separately in Part A, outside the search | dev 24.91 % (development only; not evaluated on the hold-out, whose references are not whole-second) |
| Strict acoustic (g = 0), optional | no gap-fill; native boundaries | PREREG Part-B acoustic mode | hold-out 50.11 %; no outer dev number |

**Recommendation: keep g = 5 as the default.** Offer the floor adapter only as an explicit Bengali-Loop
benchmark flag.

This departs from the council-round-3 suggestion of a g = 0 acoustic default, and the reason is the
only independent evidence available. On the exactly-timed synthetic hold-out, g = 5 beat g = 0 on
**60/60 mixtures** (−3.85 pp; post-hoc, exploratory). Gap-fill is therefore compensating for real
single-speaker MISS in Nemotron's activity, not only matching an annotation convention.

The registered hold-out acoustic mode (g = 0) stays the documented primary-comparison mode and
remains available as an option. **Choosing the default from the hold-out is a second look at Part B,
so this is the owner's decision.** The options are:

- (a) **g = 5 default** (recommended). It is the shipped CLI unchanged, with no new code.
- (b) **g = 0 default**, matching the PREREG acoustic label, at a measured hold-out cost of 3.85 pp.

**Implementation note.** `bnd.diarize` has no floor or mode switch today. The adapter exists as
`bnd.p5.floor_adapter` and is applied offline. Exposing benchmark mode in the CLI is a small
opt-in change, and the default output must stay byte-identical.

**Model card must state:**

- no demonstrated benefit of the phase-5 challenger (dev or hold-out), nor of fine-tuning (phase 3b);
- Bengali-Loop numbers are development numbers;
- the hold-out is synthetic, with the voice-reuse and same-speaker-union caveats;
- severe speaker over-counting: mean +36.9 speakers per 30-min synthetic mixture, and 192 summed on
  Bengali-Loop;
- overlapped speech is effectively not recovered;
- the unattributed pre-freeze atime read of the hold-out references.

Any per-mixture or per-recording speaker count should be treated as unreliable. If counts matter,
120 s sessions (s120_stock: 64 against 192 on dev) are the lead for a future preregistered test.
They are not part of this release.
