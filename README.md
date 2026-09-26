# Bengali long-form speaker diarization with Nemotron-3-Diarization

A reproducible pipeline and a controlled study of adapting NVIDIA
[Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (8-channel streaming
Sortformer) to Bengali long-form drama from the [Bengali-Loop](https://arxiv.org/abs/2602.14291)
benchmark: 40–68-minute recordings with 10–22 speakers each.

**Author:** Emrul Zawad · **Code licence:** MIT (weights and data are not included; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)) · **Cite:** [CITATION.cff](CITATION.cff)

## Verdict

1. **Fine-tuning did not demonstrably help.** Nemotron-3 fine-tuned on the 10 labelled Bengali-Loop
   recordings first looked 14 DER points better than zero-shot. That gain disappears once zero-shot gets
   the same calibrated decoding: 27.69 % vs 26.97 %, Δ +0.72 pp, 95 % CI [−2.48, +4.17], FT better on
   only 4/10 recordings. Fine-tuning a speaker embedder fared no better; it *increased* speaker confusion
   by 1.6–1.7 pp.
2. **The gains come from matching the reference convention at inference time.** Bengali-Loop labels are
   whole-second, pause-inclusive turns whose timestamps sit early (rounded down). Two rules recover most
   of what "fine-tuning" learned:
   - fill same-speaker pauses of ≤ 5 s (42.75 % → 26.97 %);
   - floor boundaries to whole seconds (→ 24.91 %).
3. **Nemotron-3 + windowed linking beats pyannote community-1 on Bengali drama but not on the synthetic
   hold-out.**
   - Development recordings, identical decoding: 24.91 % vs 28.51 %.
   - Synthetic Bengali hold-out: 46.26 % vs 44.70 %; pyannote also counts speakers far better there.
4. **The dominant remaining error is cross-window speaker linking.** Oracle linking would cut DER by
   about 11 pp. Longer native sessions (120 s) keep DER unchanged and cut speaker-count error by about 3×
   on the development recordings.

The root cause is data. The only verified, human-labelled natural Bengali diarization data are the
10 Bengali-Loop train recordings (9.6 h, ~170 speakers), and their labels are coarse. That is too
little, and too coarse, to teach a model pre-trained on ~92k hours anything that transfers.

## Results

**Development:** nested 5-fold cross-validation on the 10 Bengali-Loop train recordings. These
recordings were examined repeatedly while the system was designed, so treat the numbers as
development results. DER is at collar 0, whole-file UEM, overlap scored, as
[MISS / FA / CONF] and Σ|speaker-count error|.

| System | DER | Components | Σ\|count err\| |
|---|---|---|---|
| Zero-shot Nemotron, phase-2 decoding | 42.75 % | [26.01 / 5.63 / 11.10] | 261 |
| Fine-tuned Nemotron, phase-2 decoding | 28.69 % | [7.49 / 7.35 / 13.84] | 155 |
| **Zero-shot Nemotron, calibrated decoding (default)** | **26.97 %** | [6.90 / 6.85 / 13.22] | 192 |
| Fine-tuned Nemotron, same calibrated search | 27.69 % | [5.32 / 8.56 / 13.82] | 166 |
| Default + 120 s native sessions | 26.97 % | [7.01 / 6.62 / 13.34] | 64 |
| Default, benchmark mode (whole-second boundaries) | 24.91 % | [6.15 / 6.03 / 12.73] | 167 |
| pyannote community-1, same decoding rules | 28.51 % | [5.70 / 6.79 / 16.02] | 52 |
| Default + oracle linking (diagnostic) | 16.08 % | [6.04 / 6.97 / 3.06] | — |

**Independent hold-out:** 60 fixed mixtures (30 h, 1–30 speakers) of
[smam/bengali-diarization-synthetic-v4](https://huggingface.co/datasets/smam/bengali-diarization-synthetic-v4).
The subset was fixed by seed before download and scored once. It is synthetic read and extempore
speech with 37.6 % overlapped speech, not natural drama.

| System | DER (collar 0) | Σ\|count err\| |
|---|---|---|
| Default, acoustic mode (no gap fill) | 50.11 % | 2216 |
| Default, turns mode (gap fill 5 s) | 46.26 % | 2216 |
| 120 s sessions + fine-tuned embedder | 49.37 % (Δ −0.74, CI [−1.97, +0.51]; no claim) | 520 |
| Fine-tuned Nemotron | 45.72 % | 2236 |
| pyannote community-1 (untuned) | 44.70 % | 144 |

Missed speech dominates every system on the hold-out (33–38 %), because each system emits one speaker
per frame while the references overlap heavily.

Full protocols, deviations, audits and disclosures:
- [docs/results/](docs/results/): phase reports;
- [docs/PREREG_*.md](docs/): preregistrations, including one withdrawn and one paused before scoring;
- [docs/council/](docs/council/): independent reviews.

## Pipeline

1. **Native inference.** Nemotron-3 streaming at the model-card offline configuration
   (264/40/340/40/300, bf16). One forward call per session: 20 s owned core plus 5 s context each side.
2. **Local tracks.** One track per active channel with probability ≥ 0.40.
3. **Speaker embeddings.** WeSpeaker ResNet34-LM on confident, non-overlapped 1.5–4 s crops.
4. **Linking.** Average-linkage clustering with a hard same-window cannot-link, stopping at cosine
   distance 0.60, so the pipeline handles any number of speakers.
5. **Decoding.** Exclusive single-speaker timeline, then an output mode:

| `--mode` | What you get |
|---|---|
| `turns` (default) | same-speaker pauses ≤ 5 s filled — clean speaker turns |
| `acoustic` | no filling — native 10 ms speech boundaries |
| `benchmark` | `turns` + boundaries floored to whole seconds (Bengali-Loop reference convention) |

## Quick start

Requirements: Linux, NVIDIA GPU (≥ 4 GB for inference), CUDA 12.x driver, [uv](https://docs.astral.sh/uv/), ffmpeg.

```bash
scripts/setup_env.sh        # NeMo Speech @ cf724ac (py3.12, torch 2.11 cu129) + pyannote.audio 4.0.7 env
scripts/fetch_weights.sh    # nvidia/Nemotron-3-Diarization @ f667ed73 (WeSpeaker is fetched on first use)
third_party/nemo-speech/.venv/bin/python -m bnd.diarize path/to/audio.mp4 --out results/
# options: --mode acoustic|benchmark   --session-core-s 120   (fewer spurious speakers, same DER)
```

Outputs: `results/<name>.rttm`, `results/<name>.csv` (`start_time,end_time,speaker_id` in seconds) and
a JSON run report. On an RTX A6000, speed is about 1/75 of real time (RTF ≈ 0.013); a 56-minute
recording takes about 45 s.

## Reproducing the study

```bash
scripts/fetch_data.sh                      # Bengali-Loop from Kaggle (accept the competition rules first)
PY=third_party/nemo-speech/.venv/bin/python
$PY -m bnd.labels                          # label audit + scoring references (data/prepared/)
$PY -m bnd.trust --out data/prepared/trust # trusted spans (training data selection; gap 5 s)
$PY -m bnd.nemo_data --spans data/prepared/trust/5/spans.json --out data/prepared/manifests
$PY -m bnd.cv plan && $PY -m bnd.cv inner && $PY -m bnd.cv select && $PY -m bnd.cv outer \
  && $PY -m bnd.cv freeze && $PY -m bnd.cv score                      # phase 2: nested CV
$PY -m bnd.p3b regen && $PY -m bnd.p3b prep && $PY -m bnd.p3b search && $PY -m bnd.p3b outer \
  && $PY -m bnd.p3b freeze && $PY -m bnd.p3b score                    # phase 3b: fair re-test
scripts/fetch_holdout.sh                   # the fixed 60-mixture hold-out (accept its HF terms first)
for s in A-prep A-fits A-search A-outer A-freeze A-score B-validate B-infer B-freeze B-score; do
  $PY -m bnd.p5 $s; done                                              # phase 5: challenger + hold-out
$PY -m pytest -q tests                     # ~400 CPU tests; real-data ones skip until the pipeline has run
```

- All GPU work goes through `scripts/gpu_run.sh`, which is flock-serialised to one card at a time,
  under 20 GB.
- The complete study used about 8.7 GPU-hours on an RTX A6000.
- Drivers are resumable, and every stage refuses to score outer folds or the hold-out before its
  predictions are frozen by sha256.

## Limitations

- **Development numbers.** The 10 labelled recordings were used for every design decision, so their
  numbers are development results. The only untouched evaluation is synthetic.
- **Single-speaker output.** The pipeline emits one speaker per frame (the benchmark's convention), so
  overlapped speech is always partly missed.
- **Speaker overcounting.** Speakers are overcounted; use `--session-core-s 120` when counts matter.
- **Coarse references.** The references are coarse, so collar-0 DER includes their quantisation error.
- **Upstream exposure.** DISPLACE 2024, which includes Bengali, is part of Nemotron-3's training data.

## Layout

```
bnd/        pipeline + experiment drivers (labels, trust, nemo_data, infer, embed, link, score,
            train, timing, embed_ft, cv, p3b, p5, final, diarize)
configs/    final_config.json — the shipped default system (with provenance)
scripts/    setup/fetch scripts, GPU lock, alignment probes, diagnostics (scripts/diag)
tests/      pytest suite
docs/       specification, preregistrations, results, council reviews, hold-out subset list
```
