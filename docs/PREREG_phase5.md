# Phase-5 preregistration — linking challenger + independent synthetic hold-out
(frozen 2026-09-26, before any phase-5 number and before the hold-out data are downloaded)

Owner decision (after council round 3): run ONE challenger package targeting the dominant error
(cross-window speaker linking) — contrastive fine-tuning of the speaker EMBEDDER plus 120 s native
sessions — and evaluate the frozen systems once on an independent synthetic Bengali hold-out
(smam/bengali-diarization-synthetic-v4). Then finalize and release.

## Status of evidence (disclose everywhere)
All 10 Bengali-Loop recordings are DEVELOPMENT data (≥5 looks). Part A numbers are development
numbers. Part B is the only independent evaluation; it is synthetic (random chunk placement from
IndicVoices/Kathbath read/extempore speech; 324 voices reused across mixtures), not natural drama.
Base model: Nemotron-3-Diarization f667ed73 (DISPLACE 2024 incl. Bengali is in its training data).

## Part A — development (nested CV on the 10 recordings; folds F1–F5 as before)
Systems: 2 × 2 factorial on BASE Nemotron weights:
- S (native session core) ∈ {20 s (incumbent), 120 s}, both with 5 s context each side, native config
  264/40/340/40/300, one forward call per session (existing bnd.infer --core-s).
- E (embedder) ∈ {stock WeSpeaker resnet34-LM @837717dd, FT WeSpeaker}.
Incumbent = (20 s, stock). Challenger = (120 s, FT).

FT embedder recipe (fixed; no search): initialise from the pinned WeSpeaker resnet34-LM; train with
additive-angular-margin softmax (margin 0.2, scale 30) over (recording, speaker) classes of the fit's
training recordings only; training crops 2.0–4.0 s taken strictly inside a single-speaker reference
turn (ref_rttm, trusted gap-5 spans), ≥0.5 s from any turn boundary, no other speaker active;
per-speaker crop count capped at 200 per epoch (balance); AdamW lr 1e-4 backbone / 1e-3 head, wd 1e-4,
batch 64 crops, 1500 optimizer steps, cosine to lr/10, fp32/bf16-autocast, seed 42; the classifier
head is discarded; the embedding interface (dim 256, pooling, frontend) is unchanged. Fits: 10 unique
6-recording inner sets, 5 eight-recording refits, 1 all-10 final (16 fits). An embedder is NEVER used
on a recording it was trained on (assert).

Calibration family (identical for all four systems): θ_act ∈ {0.25, 0.40}; θ_clu ∈ {0.45, 0.55, 0.60,
0.65, 0.75}; gap-fill g ∈ {0, 5}; linker = phase-2 avg (hard same-window cannot-link). = 20 configs.
Selection per system per outer fold on pooled inner DER (collar 0), inner predictions only
(embedders/models that exclude the outer fold and the scored recording):
**simplest config within 0.5 pp of the minimum**; tie order: g 0 < 5; θ_act 0.40 < 0.25;
θ_clu 0.60 < 0.55 < 0.65 < 0.45 < 0.75. Floor snapping is NOT in the search; it is a separately
reported benchmark adapter applied after selection.
Outer: freeze all outer RTTMs (sha256) before scoring. Report all 4 cells (collars 0/0.25/0.5,
MISS/FA/CONF, Σ|count err|, per-recording), each with and without the floor adapter.
Dev adoption criterion (challenger replaces incumbent): corpus DER improvement ≥ 1.0 pp at collar 0
(no adapter) AND Σ|count err| not increased by > 10 %. Development-grade; no significance claim.
Final configuration per system = joint mode of its five fold selections (ties → simpler order).

## Part B — independent synthetic hold-out (score ONCE)
Subset (fixed now): from each of the 30 folders k_speaker (k = 1…30), take the 2 mixtures whose
indices are chosen by numpy.random.default_rng(20260926).choice(20, size=2, replace=False) drawn
sequentially for k = 1, 2, …, 30 → 60 mixtures (~30 h). Download only these (plus their RTTM/JSON).
Reference validation BEFORE any system output exists: report labelled-speech fraction that both
Silero and zero-shot Nemotron activity call silent (as in phase 1), overlap fraction, and turn stats;
scoring uses the RTTMs as provided (no edits); whole-file UEM; overlap INCLUDED; collar 0 primary,
0.25/0.5 total-width secondary. No system parameter may be tuned on Part B.
Frozen batch (all predictions produced and sha256-frozen before any Part-B score):
1. incumbent: base, 20 s, stock embedder, its Part-A final config;
2. challenger: base, 120 s, all-10 FT embedder, its Part-A final config;
3. pyannote/speaker-diarization-community-1 @3533c8cf (untuned; exclusive output);
4. FT-Nemotron: one all-10 model (lr 5e-5, shift, δ from all 10, 165 steps, seed 42) with the phase-3b
   FT decoding family's joint-mode config (secondary; tests whether convention learning hurts on
   exactly-timed synthetic references).
Output modes: ACOUSTIC (g = 0, native boundaries) is primary on Part B for systems 1, 2, 4 (their
selected θ_act/θ_clu kept); the Part-A-selected config incl. its g is reported as "benchmark mode"
(secondary). Community-1 as provided.
Primary Part-B comparison: challenger vs incumbent (acoustic mode), paired per mixture, corpus DER
collar 0, bootstrap over mixtures (50k, seed 20260926; mixtures share voices → approximate). Claim
"challenger better on the hold-out" only if Δ ≤ −1.0 pp and 97.5th pct < 0. Secondary (exploratory):
incumbent vs community-1; FT-Nemotron vs base; DER by speaker-count bin (1–4, 5–8, 9–16, 17–30).
Also report Σ|count err| and per-bin speaker-count MAE.

## Budget
≤ 5.5 GPU-h for phase 5 (cumulative ≈ 6.03 before it; hard stop at 12). All GPU via
scripts/gpu_run.sh (one card, ≤ 20 GB). Ledger runs/p5/gpu_ledger.jsonl.

Subset file list (derived by the rule above, dataset revision c3c7cb03053144e98effbcaf764e3fb1f61f3004): docs/phase5_holdout_subset.txt (60 stems).
