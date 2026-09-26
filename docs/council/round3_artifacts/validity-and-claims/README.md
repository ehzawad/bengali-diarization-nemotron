**Validity and claims audit — council round 3**

Completed 2026-09-26. Scope: H5 and the validity implications of H1–H4. No repository files were changed, no git commands were used, no subagents were spawned, no messages or submissions were sent, and this role used **0 GPU-hours**.

Read [00-reconciliation.md](00-reconciliation.md) first. The detailed deliverables are:

- [01-evidence-and-hypotheses.md](01-evidence-and-hypotheses.md): exposure history, verified numerical evidence, claim limits, and H1–H5 verdicts.
- [02-final-protocol.md](02-final-protocol.md): a prospective protocol capped at 5.5 additional GPU-hours, including independent evaluation, label conventions, test-audio use, and stopping rules.
- [03-external-data-and-licences.md](03-external-data-and-licences.md): primary-source availability/licence audit, Kaggle interpretation, and hold-out candidates.
- [04-model-card-and-paper.md](04-model-card-and-paper.md): usable wording, claims to remove, and publication assessment.

The evidence directory separates measurements from interpretation. [artifact_audit.json](evidence/artifact_audit.json) records hash checks, reproduction of the existing four main results, and GPU-ledger summaries. [shift_snap_reproduction.txt](evidence/shift_snap_reproduction.txt) and [boundary_offsets_reproduction.txt](evidence/boundary_offsets_reproduction.txt) reproduce the orchestrator's post-hoc looks. These are **exploratory looks on all ten development recordings**, not new tests. [local/](evidence/local/) contains line-numbered snapshots of the inspected documents and relevant source; [web/manifest.json](evidence/web/manifest.json) and [supplemental_manifest.json](evidence/web/supplemental_manifest.json) record public HTTP availability, URLs, snapshot hashes, and failures. The supplemental set includes the Babel agreement and NSU report, both successfully retrieved, with extracted text saved alongside their PDFs.

Repository file:line references in the reports resolve against `/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron`. Saved upstream model-card references resolve inside `evidence/web/`. Claims labelled “verified” come from commands or retrieved primary sources; claims labelled “inference” are interpretations, and the proposed protocol has **not** been executed.

Reproduction commands, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 10 third_party/nemo-speech/.venv/bin/python /tmp/bnd_council3/validity-and-claims/scripts/audit_artifacts.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 10 third_party/nemo-speech/.venv/bin/python runs/diag/shift_snap.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 10 third_party/nemo-speech/.venv/bin/python runs/diag/boundary_offsets.py
```

The first script writes only here. The two repository diagnostic scripts were inspected before execution and their stdout/stderr redirected here. The native-full-recording diagnostic was **not** rerun, because it writes into `runs/`; its existing result and source were inspected. The audit's rescore uses the existing scorer and is a reproduction check, not a second independent implementation. Earlier independent audits are attributed to their saved reports rather than presented as work repeated in this role.
