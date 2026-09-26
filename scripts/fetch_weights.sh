#!/usr/bin/env bash
# Download the pinned base diarizer. (The WeSpeaker embedder is fetched automatically by bnd/embed.py
# at revision 837717dd on first use.) Weights are NOT redistributed by this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/third_party/nemo-speech/.venv/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("nvidia/Nemotron-3-Diarization", "Nemotron-3-Diarization.nemo",
                    revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e", local_dir="data/weights/nemotron3")
print(p)
PY
sha256sum "$ROOT/data/weights/nemotron3/Nemotron-3-Diarization.nemo"
