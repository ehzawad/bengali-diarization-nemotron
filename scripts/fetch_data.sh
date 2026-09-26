#!/usr/bin/env bash
# Bengali-Loop diarization data from Kaggle (DL Sprint 4.0). You must first accept the competition rules at
# https://www.kaggle.com/competitions/dl-sprint-4-0-bengali-speaker-diarization-challenge/rules
# and have a Kaggle API token in ~/.kaggle/. The data are not redistributed by this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/raw" && cd "$ROOT/data/raw"
uvx kaggle competitions download dl-sprint-4-0-bengali-speaker-diarization-challenge -p .
unzip -q -o dl-sprint-4-0-bengali-speaker-diarization-challenge.zip
ls diarization/diarization/train/audio | wc -l   # expect 10
