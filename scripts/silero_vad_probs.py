"""Independent speech-activity evidence: Silero VAD frame probabilities (CPU).

Writes runs/probe_alignment/silero/<rid>.npz with probs sampled every 32 ms
(512 samples at 16 kHz). Used only to triangulate label trimming with the
zero-shot Nemotron activity; never used at inference.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad

torch.set_num_threads(4)
HOP = 512
model = load_silero_vad()
out = Path("runs/probe_alignment/silero")
out.mkdir(parents=True, exist_ok=True)
for wav in sorted(Path(sys.argv[1]).glob("*.wav")):
    target = out / f"{wav.stem}.npz"
    if target.exists():
        continue
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    model.reset_states()
    x = torch.from_numpy(audio)
    probs = np.empty(len(audio) // HOP, np.float32)
    with torch.inference_mode():
        for i in range(len(probs)):
            probs[i] = model(x[i * HOP:(i + 1) * HOP], sr).item()
    np.savez_compressed(target, probs=probs, dt=HOP / sr)
    print(wav.stem, f"{len(probs)} frames, speech frac {(probs >= 0.5).mean():.3f}", flush=True)
