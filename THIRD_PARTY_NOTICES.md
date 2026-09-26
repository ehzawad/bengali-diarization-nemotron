# Third-party components

This repository contains only original source code (MIT, see `LICENSE`). It does **not**
redistribute any model weights, audio or annotations. The components below are downloaded
by the user at pinned revisions and remain under their own terms.

| Component | Used for | Pinned revision | Licence / terms |
|---|---|---|---|
| [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | local speaker activity (the diarizer) | `f667ed73aee57d40cc39428eb768b4fd87a0a29e` | [OpenMDW License 1.1](https://openmdw.ai/license/1-1/) — any fine-tuned derivative stays under OpenMDW-1.1 obligations |
| [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech) | model code, streaming inference, training | commit `cf724ac337d1ebc7d0dda1e23fb80916f52927a5` | Apache-2.0 |
| [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | speaker embeddings for cross-window linking | `837717ddb9ff5507820346191109dc79c958d614` | CC-BY-4.0 (attribution required) |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) 4.0.7, pyannote.metrics 4.1 | embedding frontend, DER scoring | — | MIT |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | external baseline only (not part of the system) | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | gated; accept its terms on Hugging Face |
| [Silero VAD](https://github.com/snakers4/silero-vad) 6.2.3 | label-alignment diagnostics only | — | MIT |
| Bengali-Loop diarization data ([paper](https://arxiv.org/abs/2602.14291); [Kaggle DL Sprint 4.0](https://www.kaggle.com/competitions/dl-sprint-4-0-bengali-speaker-diarization-challenge)) | development data (10 labelled recordings) | Kaggle files of 2026-01-29 | competition rules; accept them on Kaggle; not redistributed |
| [smam/bengali-diarization-synthetic-v4](https://huggingface.co/datasets/smam/bengali-diarization-synthetic-v4) | independent synthetic hold-out (60 fixed mixtures) | `c3c7cb03053144e98effbcaf764e3fb1f61f3004` | CC-BY-4.0; gated; not redistributed (only the subset list is included) |

If you publish results produced with this code, cite the upstream model and datasets as well as
this repository (`CITATION.cff`).
