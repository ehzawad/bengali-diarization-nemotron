> **WITHDRAWN 2026-09-25 before any phase-3 DER — superseded by docs/PREREG_phase3b.md.**

# Phase-3 preregistration — linking (frozen 2026-09-25, before any phase-3 DER)

Owner decision: after phase 2 (fine-tuned 28.69 % vs zero-shot 42.75 % DER, gate PASS), attack the
linking gap (ft_sel 28.69 → 17.35 with oracle linking) before the final model and release.

## Honest framing (disclose in every report)
Phase-3 ideas were motivated by phase-2 OUTER results (over-splitting, θ_act at grid floor,
speaker-count errors). Selection below uses inner folds only, so the nested outer estimate remains
protected from the choice of cell/variant, but the outer folds are no longer untouched with respect
to the DESIGN of the search space. Phase-3 outer numbers are "second-look" numbers and must be
labelled so. No Kaggle submission (owner).

## What is fixed (from phase 2)
Folds, fits and refits of runs/cv (no new training in phase 3): the fine-tuned system per outer fold =
its phase-2 selected recipe refit (runs/models/cv_refit_<Fi>_<recipe>.nemo); inner candidates = the
phase-2 inner fits of that recipe (their cached probs in runs/cv/cache). The zero-shot system is
re-evaluated with the SAME phase-3 search space (equal budget) for reference only.
Native inference, windows (20 s + 5 s), WeSpeaker embedder and crop policy D7, exclusive decoder,
scorer (collar 0 primary) — unchanged.

## Search space (linker only)
- θ_act ∈ {0.25, 0.30, 0.35, 0.40, 0.50}
- θ_clu ∈ {0.55, 0.60, 0.65, 0.70, 0.75}
- variant V ∈ {avg (phase-2 linker), avg+ov (adjacent-window overlap correspondence must-link hints,
  Hungarian on shared-context activity, abstention when shared speech < 1.0 s or margin < 0.2; a hint
  that would violate cannot-link is dropped)}
- min-cluster absorption m ∈ {0, 5, 15} seconds: after clustering, any cluster whose total OWNED-core
  speech < m s is merged into its nearest cluster (centroid cosine) with no cannot-link conflict,
  if that distance < θ_clu + 0.15; otherwise kept.
= 5 × 5 × 2 × 3 = 150 configurations. Nothing else may be added after results exist.

## Selection
Per outer fold Fi: pooled inner DER (collar 0) over the 8 inner recordings, using the inner fits of
Fi's selected recipe that exclude Fi (exactly as phase-2 select); argmin; ties → smallest m, then
V=avg, then smallest L1 grid-index distance from (0.40, 0.65), then lower θ_act, lower θ_clu.
Phase-2 cell (for the ft system) is included in the space so "no change" is always selectable.

## Evaluation
Apply the selected configuration per fold to the phase-2 refit's outer predictions (re-link from
cached outer probs; new θ_act values need new tracks+embeddings from the same cached probs — no new
model forward pass for probs). Freeze all 10 outer RTTMs (sha256) before scoring. Compare
**P3 = ft + selected linker** vs **P2 = ft_sel (phase-2 frozen RTTMs)**, paired per recording.
Report also: oracle-linking gap, speaker-count errors, zero-shot under the same search (reference).

## Gate (P3 replaces P2 as default only if ALL hold, collar 0)
corpus improvement ≥ 1.0 pp; ≥ 6/10 recordings improved; no recording worse by > 3.0 pp; paired
bootstrap (50k, seed 20260925) 97.5th percentile < 0; no corpus regression at collar 0.25.
Otherwise P2's linker stays.

## Final model (after the gate; logged now, before results)
Train ONE all-10 model: recipe = phase-2 majority (lr 5e-5, timing shift; 4/5 folds), δ estimated on
all 10 recordings, fixed 165 steps (≈4.4 passes; disclosed), seed 42. Linker configuration = argmin
of pooled OUT-OF-FOLD DER over all 10 recordings using the 5 refits' outer predictions within the
winning system's search space (P3 space if P3 passed, else the phase-2 D1 grid) — a development
choice; its score is NOT a held-out estimate and must not be reported as one.

## Budget
Phase 3 + final model ≤ 1.5 GPU-h additional (cumulative ≤ 12). All GPU via scripts/gpu_run.sh.
