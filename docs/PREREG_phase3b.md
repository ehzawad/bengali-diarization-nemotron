# Phase-3b preregistration — fair re-test + linking (frozen 2026-09-25, before any phase-3b DER)

Supersedes docs/PREREG_phase3.md, which is WITHDRAWN before any phase-3 DER existed (its workflow
was stopped in the prep stage; partial runs/p3 artefacts contain no scores and are not used).

## Why (council round 2, docs/council_round2_2026-09-25.md)
- Results skeptic: a model-free decoder rule — fill gaps ≤ g s between consecutive turns of the SAME
  speaker — chosen on inner folds only, brings zero-shot to 27.56 % outer DER vs fine-tuned 28.69 %
  (post-hoc outer diagnostic); with the rule on both, 27.62 vs 27.53. Lower θ_act (< 0.40 grid floor)
  improves zero-shot by 3–5 pp (activity-only diagnostic). Fine-tuned CONF on matched support rose
  13.7 → 15.2 %. No leakage found; merge policy immaterial (0.01 pp).
- Linking critic: weak tracks ≈ 0.8 % of confusion; large strong-track clusters dominate (≈43 %
  mixing / 57 % fragmentation); WeSpeaker separates speakers (distant-pair AUC 0.945); the hard
  same-window cannot-link blocks correct reunions (355 same-speaker same-window pairs, 261 with
  < 0.1 s coactivity); 20 s windows reset AOSC every call; 15-cell selections are unstable
  (winner recovered 37–73 % under resampling).

## Honest framing (disclose in every report)
Every factor below was motivated by phase-2 OUTER-fold diagnostics. Selection is inner-fold only, so
outer numbers stay protected from the *choice* among candidates, but not from the *design* of the
space: phase-3b outer numbers are "second-look" results and must be labelled so. No Kaggle.

## Systems (no change to model weights except regenerating deleted inner fits)
- ZS = pinned base Nemotron-3-Diarization.
- FT = phase-2 selected recipe per outer fold (F1 lr5e-5_raw; F2–F5 lr5e-5_shift). Outer: the kept
  refits runs/models/cv_refit_<Fi>_<recipe>.nemo. Inner: that recipe's inner fits excluding Fi.
  Inner .nemo files were deleted after phase 2; regenerate the needed ones with the identical
  command/seed (14 unique fits) and REQUIRE that their 20 s-window probs reproduce the cached
  phase-2 probs (max |Δ| ≤ 1e-3 on every cached recording) before any use; any mismatch = stop.

## Search space (identical for ZS and FT; 108 configurations each)
- S session: {20 s cores + 5 s context (phase 2), 80 s cores + 5 s context} — native streaming config
  264/40/340/40/300 inside each session; one forward call per session.
- θ_act ∈ {0.10, 0.25, 0.40}.
- L linker ∈ {avg: phase-2 linker exactly (hard same-window cannot-link, weak rule, θ_clu stop),
  refine: see below}.
- θ_clu ∈ {0.55, 0.60, 0.65}.
- g gap-fill ∈ {0, 2, 5} s: after the exclusive timeline, a gap ≤ g between two consecutive turns of
  the same speaker is filled iff no other speaker has any output in the gap.
3 × 3 × 2 × 3 × 2 = 108.

### refine (frozen spec)
1. Coactivity cannot-link: two tracks of the same session are cannot-linked iff both have
   p ≥ θ_act on ≥ 0.20 s of common frames (instead of any shared window).
2. Seeds: average-linkage AHC over tracks with ≥ 4.0 s usable embedding speech, cannot-link checked
   at every merge, stop at θ_clu.
3. Reassignment: two passes over all tracks with an embedding (seed tracks included); centroid =
   L2-normalized mean of member embeddings weighted by usable seconds; a track moves to its nearest
   centroid c* iff d(c*) < θ_clu, runner-up margin ≥ 0.05, and no cannot-link with c*'s members;
   tracks that match no centroid form singleton clusters; centroids recomputed after each pass.
4. Tracks without an embedding: own speaker (as phase 2). Then the phase-2 exclusive decoder.

## Selection (per system, per outer fold Fi; inner recordings only)
Pooled inner DER at collar 0 over Fi's 8 inner recordings. **1-SE rule:** let c_min be the argmin;
SE = sample std of per-recording DER differences (config − c_min) / √8; choose the SIMPLEST config
with pooled DER ≤ DER(c_min) + SE. Simplicity order (lexicographic, simpler first): S 20 < 80;
L avg < refine; g 0 < 2 < 5; θ_act 0.40 < 0.25 < 0.10; θ_clu 0.60 < 0.55 < 0.65. Report the argmin
too, and leave-one-recording-out selection stability.

## Evaluation (outer; freeze all RTTMs by sha256 before any scoring)
- **Q1 (primary) — does fine-tuning help under equal decoding?** FT* vs ZS* (each its own inner-
  selected config), paired per recording, corpus DER collar 0, bootstrap 50k seed 20260925.
  "Fine-tuning helps" is claimed ONLY if: corpus Δ ≤ −1.0 pp, bootstrap 97.5th pct < 0, ≥ 6/10
  recordings improved, macro-mean Δ < 0. Otherwise report "no demonstrated benefit over calibrated
  zero-shot decoding" (or harm, if the 2.5th pct > 0).
- **Q2 (secondary) — linking.** For the better Q1 system: its selected config vs the best config
  restricted to S=20, L=avg (inner-selected with the same 1-SE rule), paired outer Δ with CI.
- Also: collars 0.25/0.5, MISS/FA/CONF, speaker-count errors, oracle-link DER from each selected
  system's OWN tracks, and phase-2 registered systems for reference.

## Default system and final artefact (rule fixed now)
Default = FT if Q1 "fine-tuning helps", else ZS. Final configuration = the joint mode of the five
inner selections of the default system (ties → simpler by the order above). If FT is default: train
one all-10 model (lr 5e-5, timing shift, δ from all 10, 165 steps, seed 42). Any OOF number over all
10 recordings for the final configuration is a DEVELOPMENT number, never presented as held-out.

## Budget
≤ 2.0 GPU-h additional (cumulative ≤ 12; 4.11 used before phase 3b). All GPU work via
scripts/gpu_run.sh.
