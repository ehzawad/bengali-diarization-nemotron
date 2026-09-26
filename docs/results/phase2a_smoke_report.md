# Phase 2a end-to-end smoke report (2026-09-25)

This report checks that the pipeline runs end to end. **None of the DER numbers below can be reported as results.**
The F1 zero-shot DER is a pipeline sanity check on one outer fold, which the workflow allowed. The F2 DER is
the inner-fold output of a smoke fit. No threshold was chosen using any of these numbers, and no other outer fold was scored.

## Code edits in this phase
None. All sanity bars either passed, or failed for a reason that is not a code bug (see "Speaker-count bar").

## 1. Zero-shot cache
`runs/cache/zeroshot/` has all 10 train recordings at θ_act 0.40, 0.50 and 0.60. For all 30 (recording, θ) pairs,
`bnd.infer.emb_is_current` is True, and the `tracks_<θ>.json` bytes match a fresh `tracks_for_recording` recompute.
Nothing needed regenerating.

## 2. Zero-shot linking at the PREREG tie-break cell (θ_act 0.50, θ_clu 0.45), CPU
Outputs are in `runs/smoke2a/zeroshot/primary_0.50_0.45/{F1,rest}/`, kept away from `runs/hyp/`.

| rec | tracks | strong | output speakers | turns | speech s |
|---|---|---|---|---|---|
| 001 | 264 | 247 | 61 | 762 | 1372.9 |
| 008 | 295 | 293 | 13 | 1080 | 2812.3 |
| 002 | 246 | 226 | 52 | 688 | 1442.9 |
| 003 | 347 | 339 | 33 | 1026 | 2740.4 |
| 004 | 338 | 317 | 43 | 1225 | 2981.4 |
| 005 | 349 | 318 | 76 | 970 | 2353.2 |
| 006 | 410 | 384 | 106 | 723 | 2070.0 |
| 009 | 260 | 239 | 59 | 626 | 1384.3 |
| 011 | 394 | 380 | 28 | 1023 | 2790.0 |
| 013 | 326 | 295 | 65 | 778 | 1733.4 |

The oracle diagnostics and the secondary overlap variant were run only on F1 (001, 008):
`runs/smoke2a/zeroshot/{oracle_link_0.50,oracle_speech_0.50_0.45,overlap_0.50_0.45}/F1/`.

## 3. F1 sanity scoring (outer fold, NON-REPORTABLE)
`bnd.score der`, ids train_001,train_008. The JSON is in `.../F1_score_NONREPORTABLE.json`.

| system | collar | DER | MISS | FA | CONF |
|---|---|---|---|---|---|
| zero-shot primary | 0.0 | 43.13 | 32.53 | 3.24 | 7.36 |
| zero-shot primary | 0.25 | 42.07 | 32.31 | 2.57 | 7.19 |
| zero-shot primary | 0.5 | 41.08 | 32.13 | 1.92 | 7.03 |
| (a) oracle linking | 0.0 | 36.48 | 32.53 | 3.24 | 0.71 |
| (b) oracle speech mask | 0.0 | 16.27 | 0.25 | 0.00 | 16.01 |
| overlap variant (secondary) | 0.0 | 55.03 | 32.53 | 3.24 | 19.27 |

Per recording at collar 0 (DER / MISS / FA / CONF, % of reference):
001 64.02 / 38.1 / 7.8 / 18.1; 008 32.71 / 29.7 / 0.9 / 2.0.
Predicted vs reference speakers: **001: 61 vs 11**; **008: 13 vs 11**.

Sanity bars:
- DER is finite and below 100 %: **pass**.
- Oracle-linking DER ≤ system DER (36.48 ≤ 43.13): **pass**.
- Predicted speakers between 3 and 40: **fails for train_001 (61)**. train_008 passes.

### Speaker-count bar: diagnosed, not a code bug
- On train_001 and train_008 at θ 0.45, the constrained AHC gives exactly the same partition as scipy
  `linkage(average)` + `fcluster(distance)` (ARI 1.0, K 58 and 12). At θ 0.45 no cannot-link constraint
  changes the result: every same-window strong-track pair has cosine distance ≥ 0.59 (1st percentile 0.63 / 0.59).
  At 0.55 and 0.65 the two results differ only where cannot-link applies (ARI 0.97 / 0.87 on 001). These were
  label-free K counts only, with no DER at those cells.
- train_001 has 39 singleton clusters. The singleton strong tracks have median usable speech of 1.84 s, against
  7.77 s for all strong tracks. Their nearest-neighbour distance is below 0.45 about half the time, but the two
  large clusters have mean intra-cluster distances of 0.37 and 0.38. Under average linkage, the distance from
  one of these tracks to a cluster therefore ends up above 0.45. The top 11 output speakers hold 82.9 % of the output speech.
- Oracle linking brings CONF down to 0.71 %, so the local tracks are nearly pure. The over-splitting comes from
  where θ_clu sits for short, noisy embeddings, not from a defect.
- No fix and no threshold change was made. Choosing θ_clu on F1 would break PREREG. The registered grid already
  includes 0.55 and 0.65, and inner-fold selection will choose among them.

Other observations, not defects:
- MISS is about 32 % even with oracle linking. The reference labels speech that the model does not detect
  (whole-second turns, and single segments of 352–443 s in 008). This matches the phase-1 finding that
  25–48 % of labelled time is model-silent.
- Oracle-speech CONF (16 %) is higher than the primary CONF. At reference-speech frames the system missed, the
  arg-max channel often has no track, so it becomes an unlinked speaker of its own.
- Overlap variant: I checked the hints against oracle track labels. Hint precision is 0.952 on 001 and 0.983 on 008.
  Pair precision within a cluster falls from 0.995 to 0.985 on 001 and from 0.965 to 0.935 on 008, because union-find
  chains the few wrong hints together. This is how the method behaves, not a bug. It is a secondary variant and is reported only.

## 4. Fine-tuned smoke model on inner fold F2 (train_011, train_009), held out from the fit
Inference ran as
`scripts/gpu_run.sh python -m bnd.infer --system ft_smoke_F1inner_F2 --model runs/models/smoke_F1inner_F2.nemo --recordings train_011 train_009`
on card 1 (A6000). It exited 0 with a wall time of 1:37.17 and a host max RSS of 2.71 GB. Log: `runs/smoke2a/infer_ft_smoke_F2.log`.
- The streaming config in the cache meta matches zero-shot (264/40/340/40/300, bf16 autocast, dt 0.01, batch 1).
  Model sha256 30784e2f…
- NeMo wall time was 31.5 s (RTF 0.00478 overall; 011: 0.00514, 009: 0.00422). Embedding took 49.2 s. Peak allocation was
  0.76 GiB for NeMo and 0.05 GiB for WeSpeaker. The embeddings match the tracks at all θ values.
- The windowed speech fraction, a label-free check against the zero-shot full-recording probe, was 0.845 on 011 and 0.588 on 009.
  The fine-tuned model marks more frames as speech.

The link at (0.50, 0.45) is in `runs/smoke2a/ft_smoke_F1inner_F2/primary_0.50_0.45/F2/`.

| system | collar | DER | MISS | FA | CONF |
|---|---|---|---|---|---|
| FT smoke primary | 0.0 | 37.73 | 13.25 | 8.56 | 15.92 |
| FT smoke primary | 0.25 | 35.34 | 12.77 | 7.04 | 15.52 |
| FT smoke primary | 0.5 | 32.93 | 12.30 | 5.48 | 15.15 |
| (a) oracle linking | 0.0 | 24.60 | 13.25 | 8.56 | 2.79 |
| (b) oracle speech mask | 0.0 | 21.13 | 0.84 | 0.00 | 20.29 |

Per recording at collar 0: 009 48.01 (MISS 15.6 / FA 13.2 / CONF 19.2); 011 33.28 (12.2 / 6.5 / 14.5).
Speakers, predicted vs reference: 011 25 vs 18; **009 55 vs 21**. This is the same over-splitting pattern at θ_clu 0.45.
No zero-shot DER was computed on F2 because that would be an outer-fold DER. These F2 numbers therefore cannot
be compared with the zero-shot system or with the F1 numbers, which come from different recordings.

## 5. GPU time and projected cost
Wall time of GPU-holding jobs in this workflow, taken from the logs and reports:

| job | wall s |
|---|---|
| zero-shot infer smoke (train_001), /usr/bin/time | 54.8 |
| zero-shot infer all 10, /usr/bin/time | 462.7 |
| embedding regeneration (fix phase), embed_report wall_s | 171.7 (the process start is not included) |
| smoke fit, total_wall_s | 101.1 |
| recheck fit, total_wall_s | 92.9 |
| FT inference F2 (this phase), /usr/bin/time | 97.2 |
| **known total** | **980.4 s = 0.27 GPU-h** |

Timings not logged: the GPU pytest run (tests/test_infer.py -k gpu) and the failed first fit launch, which
crashed at step 1. Both were short, but I have no number for them. The earlier throughput pilots
(`runs/pilot_*.log`) and `runs/zeroshot_test.log` are pre-workflow and are not counted.

Per-recording zero-shot NeMo RTF on the A6000 was 0.00414–0.00609 (overall 0.00489). Total zero-shot RTF including tracks and
embedding was 0.0106.

Projection for the full PREREG protocol, A6000, current code:
- **Fits:** per outer fold, 4 inner folds × 4 recipes = 16 inner fits, plus 1 selected refit on 8 recordings, so
  17 × 5 = 85. The STRICT comparison needs a (1e-5, raw) refit on 8 recordings per outer fold wherever the selected recipe
  differs: up to 5 more, **≤ 90 fits**. The step count is fixed at 165, so a fit takes about the same time as the smoke fit
  (101 s total), giving **≈ 2.5 GPU-h**.
- **Inference passes:** 80 inner fits × 1 pass over their inner fold (2 recordings), plus ≤ 10 outer passes (selected
  and strict refits), so **≤ 90 passes**. Measured cost: (31.5 + 49.2) s / 6586 s of audio = 0.0123 s per audio second, plus
  about 16 s of fixed overhead per job. An average fold is 6918 s of audio, so a pass takes about 100 s and all passes about **2.5 GPU-h**.
  Zero-shot needs nothing new because its cache is reused.
- **TitaNet secondary embedder (reported only):** not built. Guess is about 0.2 GPU-h.
- **Total including spend so far: ≈ 5.5 GPU-h, against the 12 GPU-h budget.** If gpu_run picks the A5000,
  runs will be slower by an amount I have not measured.

## 6. Tests
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 third_party/nemo-speech/.venv/bin/python -m pytest -q tests/`:
**225 passed, 1 skipped** (the GPU test), 34.3 s.

## Open items for the orchestrator
1. At θ_clu 0.45 the linker over-splits on recordings that have many short tracks (001: 61, 006: 106, 009-FT: 55).
   Inner-fold selection over the registered grid is the legitimate way to handle this. The 3–40 speaker bar should
   not be read as a test of code correctness.
2. Both systems have large MISS: 32 % on F1 for zero-shot, 13 % on F2 for the fine-tuned model. The 1-s whole-second reference
   labels are part of the cause.
3. The fine-tuned model marks much more speech than zero-shot in the windowed setup (011: 0.845 windowed vs 0.50 on the
   full-recording probe). This is worth watching as a source of false alarms.
