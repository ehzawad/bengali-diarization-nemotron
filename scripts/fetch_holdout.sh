#!/usr/bin/env bash
# The fixed 60-mixture synthetic hold-out (docs/phase5_holdout_subset.txt) from
# smam/bengali-diarization-synthetic-v4 @ c3c7cb03 (CC-BY-4.0, gated: accept its terms on Hugging Face first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV=c3c7cb03053144e98effbcaf764e3fb1f61f3004
TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}"
B="https://huggingface.co/datasets/smam/bengali-diarization-synthetic-v4/resolve/$REV"
mkdir -p "$ROOT/data/holdout_synth"
while read -r s; do
  for ext in wav rttm json; do
    f="$ROOT/data/holdout_synth/$s.$ext"; mkdir -p "$(dirname "$f")"
    [ -s "$f" ] || curl -sfL -H "Authorization: Bearer $TOKEN" -o "$f" "$B/$s.$ext"
  done
done < "$ROOT/docs/phase5_holdout_subset.txt"
(cd "$ROOT/data/holdout_synth" && find . -type f | sort | xargs sha256sum) > "$ROOT/data/holdout_synth_SHA256.txt"
find "$ROOT/data/holdout_synth" -name '*.wav' | wc -l   # expect 60
