# Phase-4 preregistration — speaker counting (frozen 2026-09-25, before any phase-4 number)

Owner decision: after phase 3b (default = base Nemotron + s20_avg_g5_a0.40_c0.60, 26.97 % DER, but
Σ|pred−ref speakers| = 192 over 10 recordings, e.g. 40 vs 11 on train_001), run one bounded phase on
over-splitting, then release.

## Framing (disclose)
Third look at the same 10 recordings; the search space is motivated by phase-2/3b outer diagnostics
and council round 2 (large strong-track fragments, hard same-window cannot-link blocking reunions,
many tiny clusters). Inner-only selection protects the choice among candidates, not the design.

## Fixed
Zero-shot base model; S = 20 s cores + 5 s context; θ_act = 0.40; average-linkage AHC; weak-track
rule; exclusive decoder; gap-fill g = 5 s (applied last). Cached tracks/embeddings from
runs/p3b/cache/zeroshot/s20 (read-only). No GPU, no training.

## Search space (36 configurations)
- θ_clu ∈ {0.60, 0.70}
- CL ∈ {hard (same window, phase 2), coact (both tracks p ≥ θ_act on ≥ 0.20 s of common frames)}
- θ2 ∈ {none, 0.70, 0.80}: second-stage merge — after first-stage AHC, clusters with ≥ 30 s owned
  speech are merged by average linkage over their usable-seconds-weighted L2-normalized centroids,
  stopping at θ2, cannot-link (union of members) checked at every merge.
- m ∈ {0, 5, 15} s: after the second stage, every cluster with < m s owned speech is merged into its
  nearest (centroid cosine) cluster with ≥ m s and no cannot-link conflict; if none, kept.
Phase-3b default = (0.60, hard, none, 0) is in the space.

## Selection (per outer fold Fi; 8 inner recordings; collar 0)
1. DER-eligible set = configs with pooled inner DER ≤ min + SE (SE as in PREREG_phase3b, 1-SE rule).
2. Among them choose the lowest inner Σ|pred−ref speakers|; ties → simpler (θ2 none < 0.80 < 0.70;
   m 0 < 5 < 15; CL hard < coact; θ_clu 0.60 < 0.70).
Report argmin-DER config, the selected config, and leave-one-recording-out stability.

## Evaluation (outer; freeze RTTMs by sha256 before scoring)
P4 = selected config per fold vs D = phase-3b frozen zs_star RTTMs (runs/p4 must not modify them).
Gate — P4 becomes the default only if ALL hold:
1. outer Σ|pred−ref speakers| reduced by ≥ 30 % (≤ 134);
2. per-recording |count error| better on ≥ 6/10 recordings;
3. corpus DER Δ (P4 − D, collar 0) ≤ +0.5 pp;
4. paired bootstrap (50k, seed 20260925) 97.5th percentile of DER Δ ≤ +1.5 pp.
Report DER at collars 0/0.25/0.5 with MISS/FA/CONF, per-recording counts, oracle-link.
Final configuration if the gate passes = joint mode of the five selections (ties → simpler).

## Budget
0 GPU-h expected (CPU only). Cumulative stays ≤ 12 (5.741 used).
