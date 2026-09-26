H4 found a material benchmark convention effect. Read [REPORT.md](REPORT.md) for the result, evidence, release recommendation, limits, and open questions. All ten-recording numbers are exploratory looks. No repository edits, GPU use, model training, or submission was performed.

The practical recommendation is full-resolution speech activity with no multi-second pause fill by default, plus an explicit Bengali-Loop benchmark mode using frozen ZS upstream settings, ≤5-second same-speaker silent-gap filling, then floor both boundaries. This changes the exploratory DER from 26.97% to 24.91%. Longer gaps add at most a small conditional benefit. FT already learned turn continuity; its best main inner-selected simple correction is offset-only floor, without demonstrating an advantage over decoded ZS. Historical universal flooring remains unproved.

Artifacts are grouped by purpose:

- `PLAN.md`: main analysis plan plus disclosed follow-up.
- `DECODER_SPEC.md`, `common.py`, `run_decoders.py`: literal rules, scorer, and inner/outer experiment.
- `decoder_summary.json`, `outer_scores.json`, `inner_scores.json`, `selection.json`, `scores.csv`: complete main numerical results. Main selected outputs are under `predictions/`; reconstructed ungapped outputs are under `raw/`.
- `reference_audit.json`, `raw_rows.json`, `reference_gaps.json`, `labelled_pauses.json`: annotation audit with CSV line provenance.
- `boundary_summary.json`, `boundary_matches.csv`: detector-based boundary audit with matching sensitivity.
- `cache_diagnostics.json`: cached probabilities, pause learning, first-onset ownership, gap occupancy, and inner selections.
- `target_alignment.json`, `target_pair_provenance.json`: eight-recording F1-inner paired raw/shift target comparison, without new training.
- `gap_sensitivity.json`: disclosed later single-gap-parameter sweep and inner-selected results.
- `extra_summary.json`: conditional bootstrap intervals, output structure, and detector-agreement proxies.
- `overlap_and_rounding.json`: stored overlap exceptions, scorer sensitivity, and rounding tie semantics.
- `audio_evidence/`, `audio_evidence.json`: six reproducible WAV excerpts and waveform/detector/reference figures. These were not human auditory adjudications.
- `convention_summary.png`, `convention_summary.pdf`: graphical overview.
- `sources.json`, `provenance.json`, `scorer_validation.json`, `verification.log`: input identity and checks.

Run `./reproduce.sh` from any directory on this machine to rerun the analyses sequentially on CPU with the prescribed environment. The NeMo venv supplies scientific/audio/scoring dependencies; the system Python supplies matplotlib and uses the standard-library WAV reader for figures. The script writes only here, but overwrites this directory's generated results. It requires the existing repository caches and models' already-computed predictions, not model inference. The written report is a fixed narrative of this run, not regenerated text.
