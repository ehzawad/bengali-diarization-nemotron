#!/usr/bin/env bash
# Build the two pinned environments used by this repository (Linux, NVIDIA GPU, CUDA 12.x driver, uv).
#   third_party/nemo-speech/.venv : NeMo Speech @ cf724ac (Python 3.12, torch 2.11 cu129) — diarizer, training, scoring
#   third_party/embed-venv        : pyannote.audio 4.0.7 (torch 2.11 cu129)                 — speaker embeddings, baseline
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEMO_COMMIT=cf724ac337d1ebc7d0dda1e23fb80916f52927a5
mkdir -p "$ROOT/third_party"
if [ ! -d "$ROOT/third_party/nemo-speech/.git" ]; then
  git clone https://github.com/NVIDIA-NeMo/Speech.git "$ROOT/third_party/nemo-speech"
fi
git -C "$ROOT/third_party/nemo-speech" fetch --depth 1 origin "$NEMO_COMMIT" 2>/dev/null || true
git -C "$ROOT/third_party/nemo-speech" checkout --detach "$NEMO_COMMIT"
(cd "$ROOT/third_party/nemo-speech" && uv sync --python 3.12 --extra asr --extra cu12 --no-dev --frozen)
uv pip install --python "$ROOT/third_party/nemo-speech/.venv/bin/python" -r "$ROOT/requirements/nemo-extras.txt"
uv venv --python 3.12 "$ROOT/third_party/embed-venv"
uv pip install --python "$ROOT/third_party/embed-venv/bin/python" \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
  -r "$ROOT/requirements/embed-venv.lock.txt"
"$ROOT/third_party/nemo-speech/.venv/bin/python" -c "import nemo, torch; print('nemo', nemo.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$ROOT/third_party/embed-venv/bin/python" -c "import pyannote.audio, torch; print('pyannote.audio', pyannote.audio.__version__, 'torch', torch.__version__)"
