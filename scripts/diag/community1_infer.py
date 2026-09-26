"""EXTERNAL BASELINE (diagnostic, no tuning on these recordings): pyannote speaker-diarization-community-1
(pinned revision) on the 10 Bengali-Loop train recordings; writes EXCLUSIVE (single-label) and regular RTTMs."""
import sys, time
from pathlib import Path
import torch
from pyannote.audio import Pipeline
REV = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
tok = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=tok, revision=REV)
pipe.to(torch.device("cuda"))
out = Path("runs/diag/community1"); (out / "exclusive").mkdir(parents=True, exist_ok=True); (out / "regular").mkdir(exist_ok=True)
torch.cuda.reset_peak_memory_stats(); t0 = time.time(); tot = 0.0
for wav in sorted(Path("data/raw/diarization/diarization/train/audio").glob("*.wav")):
    rid = wav.stem; t = time.time()
    res = pipe(str(wav))
    for kind, ann in (("exclusive", res.exclusive_speaker_diarization), ("regular", res.speaker_diarization)):
        with open(out / kind / f"{rid}.rttm", "w") as f:
            for seg, _, spk in ann.itertracks(yield_label=True):
                f.write(f"SPEAKER {rid} 1 {seg.start:.3f} {seg.duration:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    import soundfile as sf; d = sf.info(str(wav)).duration; tot += d
    print(rid, f"{time.time()-t:.1f}s for {d:.0f}s audio, speakers {len(res.exclusive_speaker_diarization.labels())}", flush=True)
print(f"total {time.time()-t0:.1f}s, RTF {(time.time()-t0)/tot:.4f}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
