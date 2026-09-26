# Phase-2 preregistration (frozen 2026-09-25, before any DER is computed)

Binding for workflow 2. Changing anything here after a DER number exists must be logged as a
deviation with the reason, and the pre-deviation result reported too.

## Owner decisions
- Training data: trusted spans at **gap_s = 5** (`data/prepared/trust/5`, manifests
  `data/prepared/manifests`), 6.95 h labelled speech kept (−9.8 %).
- Label timing: **measured inside CV**. Variant `raw` = targets as-is; variant `shift` = every
  training-RTTM turn START moved later by δ_f, where δ_f is estimated ONLY from fold f's training
  recordings (median over those recordings of the shift maximizing label-mask vs both-detector-speech
  frame agreement, search −1.0…+1.0 s step 10 ms), ends unchanged, turns that would become
  ≤ 0.1 s dropped. The scoring reference is NEVER shifted.
- Release: private GitHub (core reproducible files) + public HF model (MIT if licences permit,
  author Emrul Zawad) — later phase.

## Data / folds
Outer folds: F1 001+008, F2 011+009, F3 013+005, F4 004+002, F5 003+006. Inner folds for outer Fi =
the other four folds (leave-one-fold-out over the remaining 8 recordings). No test audio in any
selection. Training manifests for a fit = gap-5 spans of its training recordings only (assert by
recording_id).

## Inference system (identical for zero-shot and fine-tuned)
- Native Nemotron streaming, model-card offline config (264/40/340/40/300), bf16.
- Windows: 20 s owned cores + 5 s context each side (≤30 s input); every channel of every window is
  a local track with provenance `(recording, window, channel)`; raw per-window probs cached.
- Local decoding: per-channel activity ≥ θ_act, then 0.1 s minimum turn.
- Embeddings: `pyannote/wespeaker-voxceleb-resnet34-LM` @ 837717ddb9ff5507820346191109dc79c958d614
  (pyannote.audio in a SEPARATE venv if it conflicts with the NeMo lock). Crops: confident,
  non-overlapped activity of the track, 1.5–4 s pieces, valid lengths (no padding counted as speech);
  tracks with < 1.0 s usable speech are "weak".
- Linking: average-linkage agglomerative clustering on cosine distance of strong tracks with a
  HARD cannot-link between tracks of the same window (checked at every merge), stopping at
  θ_clu; weak tracks then attached to the nearest cluster if within θ_clu and no cannot-link
  conflict, else become their own speaker. Overlap correspondence between adjacent windows is a
  registered SECONDARY variant (reported, not used for selection).
- Final output: exclusive single-label timeline — at each frame the active linked speaker with the
  highest probability, ties broken by earliest onset then lowest id; same-speaker touching turns merged.
- TitaNet-large is a registered secondary embedder (reported only).

## Tuning grid (equal for every system)
θ_act ∈ {0.40, 0.50, 0.60} × θ_clu ∈ {0.25, 0.35, 0.45, 0.55, 0.65} = 15 cells. (0.55/0.65 added
because WeSpeaker cosine distances are not TitaNet's; declared before seeing data.) Selection
metric: pooled inner-fold DER at collar 0.0; ties → closest to (0.50, 0.45).

## Fine-tuning recipe (fixed except the registered inner-selected factors)
Pinned base; streaming training settings 264/0/264/264/0/0; subsegment_mode stage-2 settings
(session 105 s); train last 4 encoder blocks + final_norm + active sortformer heads; freeze
learnable_sil_emb and hidden_to_spks; AdamW betas (0.9,0.98) wd 1e-3; clip 1.0; bf16; batch 8
sessions, no accumulation; 20 warmup steps then cosine to lr/10; **fixed 165 optimizer steps**
(≈5 audio passes); no early stopping; export final weights; seed 42.
Inner-selected factors: lr ∈ {1e-5, 5e-5} × timing ∈ {raw, shift} = 4 recipes. Selection: pooled
inner out-of-fold DER (collar 0.0) at each recipe's own best grid cell.

## Protocol
For each outer fold Fi: (1) zero-shot: choose (θ_act, θ_clu) on the 8 inner recordings, apply to
Fi; (2) fine-tuned: for each inner fold Fj (j≠i) and recipe r, fit on the other 6 recordings, infer
Fj; choose r and (θ_act, θ_clu) on pooled inner predictions; refit r on all 8; apply to Fi.
Complete all 5 outer prediction sets before computing any outer DER.
Also report STRICT comparison: zero-shot vs fine-tuned (lr 1e-5, raw) at fixed (0.50, 0.45), no
selection. And oracle diagnostics on outer folds: (a) oracle linking with predicted local tracks,
(b) oracle speech mask.

## Scoring
`bnd/score.py`: whole-file UEM, skip_overlap=False, primary collar 0.0, secondary 0.25 and 0.5
(pyannote total width); one mapping per recording; corpus DER = Σerrors/Σref; paired recording
bootstrap 50k, seed 20260925.

## Adoption gate (fine-tuned becomes the default only if ALL hold, on outer folds, collar 0.0)
corpus DER improvement ≥ 2.0 pp; macro mean delta < 0; improvement on ≥ 7/10 recordings; no
recording worse by > 5.0 pp; bootstrap 97.5th percentile of the delta < 0; no corpus regression at
collar 0.25. Otherwise the calibrated zero-shot system is the default and the result is reported
as-is.

## Budget / hardware
≤ 12 GPU-h total; every GPU command via `scripts/gpu_run.sh` (flock-serialized, one card, ≤ 20 GB).

## Deviations (logged 2026-09-25, before any REGISTERED DER; owner-approved D1)
- **D1 grid:** θ_clu grid replaced by {0.45, 0.55, 0.65, 0.75, 0.85} (still 15 cells with θ_act
  {0.40,0.50,0.60}). Reason: phase-2a smoke on outer fold F1 over-split (train_001: 61 predicted vs
  11 reference speakers at 0.45; clustering verified identical to scipy average linkage). Selection
  remains inner-fold only. Disclosed in the final report and model card.
- **D2 exposure:** smoke runs scored outer F1 (train_001, 008; zero-shot at (0.50,0.45) + oracles)
  and F2 (train_011, 009; smoke fine-tuned model). These labels have been seen by humans/agents;
  disclosed in the final report. The registered protocol re-runs everything from scratch.
- **D3 tie-breaks:** grid ties → smallest L1 distance in grid-index units from (0.50, 0.65), then
  lower θ_act, then lower θ_clu. Recipe ties (within 1e-8 pp) → prefer lr 1e-5, then timing raw.
  Strict comparison fixed cell changes from (0.50,0.45) to (0.50,0.65) (grid centre).
- **D4 timing sign:** δ is applied signed (starts move by +δ; a negative δ moves starts earlier).
- **D5 TitaNet:** secondary embedder descoped (report-only in PREREG; not implemented).
- **D6 exposure mismatch:** fixed 165 steps gives ≈7 passes for 6-recording inner fits vs ≈5.5 for
  8-recording refits; kept as registered, disclosed.
- **D7 crops:** strong tracks with 1.0 ≤ usable < 1.5 s are embedded from one <1.5 s piece
  (≈100/3000 strong tracks per θ_act); "confident" = p ≥ θ_act.
