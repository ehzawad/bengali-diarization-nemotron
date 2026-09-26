# SPEC — Bengali-Loop × Nemotron-3-Diarization (reconciled 2026-09-25)

Owner goal: fine-tune `nvidia/Nemotron-3-Diarization` on Bengali-Loop into a practical Bengali
diarizer: proper data prep, recording-level validation, zero-shot vs fine-tuned DER, cross-window
speaker linking for >8 speakers per recording.

**Owner ruling (binding): make the DATASET adapt to Nemotron's native training format FIRST, even
if that means losing data.** Losing data = excluding untrusted time ranges entirely (audio AND
labels). NEVER delete a label inside a kept range: Nemotron reads unlabelled audio as silence.

## Environment (fixed)
- Repo root: this directory. Python package: `bnd/`. Tests: `tests/` (pytest).
- Python: `third_party/nemo-speech/.venv/bin/python` (py3.12, torch 2.11+cu129, NeMo 3.1.0 @ cf724ac,
  pyannote.metrics 4.1, pytest). NeMo source: `third_party/nemo-speech/`.
- Model: HF `nvidia/Nemotron-3-Diarization` rev `f667ed73aee57d40cc39428eb768b4fd87a0a29e`,
  cached at `data/weights/nemotron3/Nemotron-3-Diarization.nemo`. Class `SortformerEncLabelModel`,
  strict restore works, output frames 10 ms (high_resolution, output_subsampling_factor=1),
  encoder/cache frames 80 ms.
- Raw data (read-only, never modify): `data/raw/diarization/diarization/train/{audio,annotation}/`
  10 recs `train_{001,002,003,004,005,006,008,009,011,013}` (16 kHz mono WAV, 40–68 min, 10–22
  speakers each); `.../test/audio/` 14 unlabelled WAVs. `data/raw/sample_submission .csv`.
- CSV quirks: headers vary in case (`Start_time,End_time,Speaker_ID` in train_013), CRLF in 9/10,
  times `HH:MM:SS` whole seconds. Defects (training-recipe audit): 109 zero-length rows, 5 reversed
  rows (e.g. train_002.csv:133), 27 positive-duration cross-speaker overlaps (nested interjections,
  e.g. train_002 spk13 822–927 s containing spk1 828–838 s), 352–443 s single segments (004, 008).
- Alignment evidence already computed:
  - `runs/probe_alignment/probs/<rid>.npz` — zero-shot Nemotron probs [T,8], `dt`≈0.01 s (offline
    streaming config 264/40/340/40/300).
  - `runs/probe_alignment/silero/<rid>.npz` — Silero VAD probs, `dt`=0.032 s (being produced; may be
    incomplete for some recordings — code must handle a missing file by failing loudly, tests use
    synthetic arrays).
  - `runs/probe_alignment/report.json` — 25–48 % of labelled time is model-silent; of that, 62 % is
    in runs <2 s (intra-turn pauses / 1-s rounding, KEEP), 19 % in runs ≥5 s (suspect).
- Hardware policy: shared box; ONE GPU at a time, ≤20 GB, never both cards. Phase 1 (data) is CPU
  only — do NOT use the GPU. Pin BLAS/OMP threads ≤4, nice. Never ffmpeg-per-file swarms.

## Data design (phase 1)
1. `bnd/labels.py` — robust CSV → `Turn(start,end,speaker)`; defect classification per row
   (zero_length, reversed, beyond_eof, cross_speaker_overlap, same_speaker_overlap); writes
   - scoring reference RTTM `data/prepared/ref_rttm/<rid>.rttm`: original rows kept as-is (NO
     same-speaker merging — merging deletes collar boundaries), drop only zero-length/reversed rows,
     every drop logged in `data/prepared/label_audit.json` with file:line.
   - training RTTM `data/prepared/train_rttm/<rid>.rttm`: valid rows, same-speaker touching rows may
     be merged (training only).
2. `bnd/trust.py` — untrusted-region mask per recording (10 ms grid), union of:
   (a) every defect row interval ± `pad` (default 1.0 s), including both sides of every
       cross-speaker overlap;
   (b) labelled-but-silent runs: frames labelled speech where BOTH Nemotron max-prob < 0.5 AND
       Silero < 0.5, for runs ≥ `gap_s` (grid {2,3,5,10}; default 5);
   (c) unlabelled-but-speech runs: frames unlabelled where BOTH detectors say speech, runs ≥ `gap_s`
       (missing labels);
   (d) any 105 s window with >8 reference speakers (only train_006, ~1 %).
   Trusted spans = complement, cut only at points that are unlabelled (or at excluded-region
   edges), each span ≥ `min_span_s` (default 45 s, so NeMo can sample single ≥45 s chunks) —
   shorter leftovers are dropped. Report per rec and total: kept hours, kept labelled-speech hours,
   dropped by reason, for every gap_s in the grid. Output `data/prepared/trust/<gap_s>/spans.json`.
3. `bnd/nemo_data.py` — NeMo manifests from spans: one JSONL line per span
   `{"audio_filepath","offset","duration","num_speakers","rttm_filepath","recording_id","span_id"}`.
   Must VERIFY against the real NeMo loader (`nemo/collections/asr/data/audio_to_diar_label.py`)
   whether RTTM times are absolute (recording time) with manifest offset applied, or span-relative;
   write per-span RTTMs in whichever convention the loader actually uses. Must instantiate the real
   NeMo diarization dataset on CPU with the official stage-2 train_ds settings
   (subsegment_mode=true, session_len_sec=105, num_spks=8, the subsegment_* params from
   `examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_streaming_8spk.yaml`) and
   draw ≥200 samples, asserting: targets align with RTTM, arrival order correct, ≤8 active channels,
   no sample crosses a span edge, fallback/empty-target rate reported.
4. `bnd/score.py` — DER with pyannote.metrics: UEM = whole file [0, duration); `skip_overlap=False`;
   collars 0.0 (primary), 0.25, 0.5 **total width** (pyannote semantics: ±collar/2); one Hungarian
   mapping per recording; per-rec MISS/FA/CONF/total seconds; corpus DER = sum errors / sum ref;
   paired recording bootstrap (50k, seed 20260925) for system deltas. Plus Kaggle exporter/validator:
   `filename,diarization` rows in sample_submission order, JSON list of
   `{"start_time":float,"end_time":float,"speaker_id":"SPEAKER_k"}`, sorted, non-overlapping,
   0≤start<end≤duration.

Folds (frozen, eval-protocol council): F1 001+008, F2 011+009, F3 013+005, F4 004+002, F5 003+006.

## Later phases (not phase 1)
Training: NeMo native streaming fine-tune from pinned base, subsegment_mode on trusted spans,
training streaming settings 264/0/264/264/0/0, train last 4 encoder blocks + final_norm + active
sortformer heads, FREEZE learnable_sil_emb + legacy hidden_to_spks, AdamW 1e-5 (0.9,0.98) wd 1e-3,
clip 1.0, bf16, fixed horizon, no val-loss early stopping. Inference: short overlapping windows +
cannot-link-constrained clustering + overlap correspondence; embedder
`pyannote/wespeaker-voxceleb-resnet34-LM` (TitaNet baseline). Equal 9-cell threshold grid per
system; paired bootstrap; adoption gate ≥2 pp corpus DER, ≥7/10 wins, no rec worse >5 pp.

## Rigor rules
No overclaiming; every number from a command you ran; tests must exercise real code paths; failing
loudly beats silent fallbacks; raw data immutable; no GPU in phase 1.
