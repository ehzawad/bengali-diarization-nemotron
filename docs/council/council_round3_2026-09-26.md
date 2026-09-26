# Codex Council — 4/4 roles responded (1849.3s)

## Summary

- **Where linking should live: native session length and fine-tuning** [session-architecture]: ok (model: gpt-6-astra, effort: xhigh) — 1086.5s
- **Clustering, speaker counting and embedding scale** [clustering-and-count]: ok (model: gpt-6-astra, effort: xhigh) — 1849.3s
- **Reference convention and decoding** [benchmark-convention]: ok (model: gpt-6-astra, effort: xhigh) — 1820.3s
- **Validity, remaining looks and what can be claimed** [validity-and-claims]: ok (model: gpt-6-astra, effort: xhigh) — 1823.7s

## Where linking should live: native session length and fine-tuning (session-architecture)

H2 is partly supported: medium sessions improve real linking, but their tracks are **less pure**, and within-session tracking worsens. H1 identifies a substantial linking bottleneck; the measurements do not support saying fine-tuning failure was predetermined. Full evidence, per-recording results, scripts, and outputs are in [REPORT.md](/tmp/bnd_council3/session-architecture/REPORT.md). **Every measurement below is exploratory.**

I ran the requested zero-shot sessions on `train_001`, `train_006`, and `train_008`, holding thresholds, linking, and decoding fixed. With one-to-one ORACLE cross-session mapping, pooled DER for **20/60/120/180/300-second cores** was **15.71/17.00/17.38/18.16/19.95%**; full-recording inference scored **31.97%** under the same decoder. Within-session confusion rose from **3.53%** at 20 seconds to **5.02%** at 120 seconds and **7.01%** at 300 seconds. This degradation persists without gap filling.

With actual WeSpeaker embeddings and existing AHC, exploratory DER for those same lengths was **26.56/25.29/24.06/24.15/27.32%**. The prespecified 120-second system therefore gains **2.49 percentage points**, despite worse local tracking. The gain is heterogeneous: `train_006` improves substantially while `train_001` slightly worsens. At 120 seconds, track count falls from **994 to 271**, mean owned speech per track rises from **7.40 to 24.18 seconds**, and reference-weighted purity falls from **97.47% to 95.45%**. Tracks are fewer and longer, but not cleaner.

A material oracle caveat changes H1’s interpretation: [the existing oracle](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/bnd/link.py:810) can merge multiple channels *inside* a session, repairing local fragmentation that the practical linker forbids. Its exploratory 20-second DER is **13.52%**, versus **15.71%** for the constraint-preserving oracle. The practical-to-strict gap remains substantial at **10.85 points**, but residual speech errors and local confusion leave meaningful room for upstream improvements.

Fine-tuning could plausibly improve returning-speaker consistency, fragmentation, and embedding purity; no new fine-tuning benefit was measured here. The minimal targeted experiment is **masked conditional speaker-identity loss on confidently voiced reference interiors**, excluding pauses and uncertain boundaries, with one identity mapping across the whole session. Keep the speech gate and convention decoder fixed during evaluation. Simply extending sessions while retaining all-frame activity BCE would not isolate tracking from label-convention learning.

Capacity-aware resets are implementable through NeMo’s [explicit streaming-step API](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/third_party/nemo-speech/nemo/collections/asr/models/sortformer_diar_models.py:1276) and [state initializer](/mnt/sdb/arafat/ehz/llm/bengali-diarization-nemotron/third_party/nemo-speech/nemo/collections/asr/modules/sortformer_modules.py:530), using a caller-managed loop. Reset the complete state and session output accumulator. However, exploratory sustained activity on the last channels flagged only **2 of 92** 120-second sessions, while every input fit within reference-speaker capacity. Most contamination occurred without that signal. A maximum session age should accompany an occupancy trigger; adaptive-reset performance remains untested.

Both GPU jobs used `scripts/gpu_run.sh`; exploratory measured consumption was **0.04068 GPU-hours**. Reconstructed baseline RTTMs matched frozen outputs exactly. No repository source was modified, and no other council work was required. These deliberately selected, previously examined recordings cannot establish generalization. The open decision is whether to preregister an inner-fold medium-session comparison and a separate identity-loss ablation before changing the default.

## Clustering, speaker counting and embedding scale (clustering-and-count)

H3’s distance-scale mismatch is real, but **no reliable overall improvement supports replacing frozen ZS\***. Combined inner-fold 1-SE selection retains the baseline in all five folds: **26.97% / 25.13% DER** at collars 0 / 0.25, with summed speaker-count error **192**.

Exploratory results: median normalization reaches **26.87% / 25.04%**, NME-SC **36.83% / 35.08%**, and the cosine-HMM family **27.39% / 25.55%**. Quantile normalization selected by inner argmin reaches **26.01% / 24.15%**, but improves only two recordings; its collar-0 difference has CI **[−4.07, +0.71] pp**. An exact-output-count oracle achieves zero count error yet **28.61% DER**—count accuracy alone does not fix identity assignments.

Everything is under the requested directory. The [full report](/tmp/bnd_council3/clustering-and-count/REPORT.md) and [metrics CSV](/tmp/bnd_council3/clustering-and-count/metrics.csv) include MISS/FA/CONF, counts, fold selections, diagnostics, evidence references, and reproduction instructions. All ten baseline RTTMs reproduce byte for byte. CPU only; no repository modifications.

Longer clean tracks improve embedding discrimination but retain substantial recording-scale variation. Track-level averaging is already implemented. Different crops, another embedder, and genuine matched-PLDA VBx remain untested; the implemented HMM does not establish their effectiveness. No other council work was required. The remaining decision is whether to preregister those alternatives against fresh held-out labels, with speaker counting treated as a separate objective if needed.

## Reference convention and decoding (benchmark-convention)

H4 is materially supported. **Exploratory all-ten DER: ZS* improves from 26.97% to 24.91% by flooring both boundaries after its existing five-second gap filling.** Longer gaps offer only small gains; minimum-duration, interruption-merging, and earliest-onset overlap rules add no robust improvement. Conditional inner 1-SE selection yields **24.97% ZS versus 26.74% FT**.

FT already learned pause inclusion: at a common threshold, it activates 90.75% of detector-silent turn interiors versus 8.89% for ZS. Existing raw-target FT nearly ties decoded ZS in the matched eight-recording inner comparison, but does not demonstrate superiority. No new fine-tune was trained.

Recommend full-resolution speech intervals without multi-second pause filling by default, with a separate Bengali-Loop benchmark mode using same-speaker gap filling ≤5 seconds and floor snapping. The evidence supports early integer timestamps, but **does not prove universal historical flooring**.

Everything is in [REPORT.md](/tmp/bnd_council3/benchmark-convention/REPORT.md), with a concise [council handoff](/tmp/bnd_council3/benchmark-convention/COUNCIL_HANDOFF.txt), scores, selected RTTMs, audio excerpts, figures, and reproduction scripts. Verification passed 272 assertions, including 216 scorer comparisons. CPU only; no repository edits or GPU use. Results depend on the existing caches and frozen upstream choices.

All ten-recording results remain exploratory. Detector agreement is not acoustic ground truth. Exact annotation-tool semantics, official scorer details, independent acoustic labels, and a prospective convention-aligned FT comparison remain open.

## Validity, remaining looks and what can be claimed (validity-and-claims)

Completed the audit under [validity-and-claims](/tmp/bnd_council3/validity-and-claims/README.md), with a [reconciliation report](/tmp/bnd_council3/validity-and-claims/00-reconciliation.md), evidence audit, final protocol, external-data/licence review, and model-card/paper guidance.

H5 is supported: all ten recordings now serve as development data. The defensible conclusion is **no demonstrated incremental benefit from the tested fine-tuning recipe over the calibrated base pipeline**. Earlier comparisons remain reproducible observations; floor snapping and new linking/session experiments remain exploratory.

I verified 180 frozen prediction hashes and reproduced all four primary scores without discrepancies. Two material findings: DISPLACE appears in the [pinned checkpoint’s training inventory](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md), disqualifying it as a clean hold-out, and gap merging already has [Bengali prior art](https://arxiv.org/html/2602.23070v1).

The proposed protocol caps additional work at 5.5 GPU-hours and prioritizes one frozen independent comparison. Babel is a conditional labelled-data route; fresh blinded annotations offer stronger independence. Kaggle late scoring remains conditional on authorization and verified scoring access.

Open dependencies are independent labels and rights, frozen candidates from the other lenses, and the intended acoustic versus pause-inclusive target. No repository edits, submissions, or GPU work were performed.
