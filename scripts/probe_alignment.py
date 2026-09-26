"""Zero-shot label-alignment probe.

Runs the pinned Nemotron-3-Diarization checkpoint over each full training
recording (model-card offline streaming config) and compares its frame-level
speech activity with the human CSV labels. Answers: how much labelled time
does the model hear as non-speech, how much unlabelled time does it hear as
speech, and how far are label boundaries from model boundaries.
Speaker identity is NOT compared here; this is speech-activity alignment only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

MODEL_ID = "nvidia/Nemotron-3-Diarization"
MODEL_REV = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
OFFLINE = dict(spkcache_len=264, fifo_len=40, chunk_len=340,
               chunk_right_context=40, spkcache_update_period=300)


def hms(s):
    parts = [float(x) for x in s.strip().split(":")]
    return sum(x * 60 ** i for i, x in enumerate(reversed(parts)))


def read_labels(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): v for k, v in r.items()}
            rows.append((hms(r["start_time"]), hms(r["end_time"]), r["speaker_id"].strip()))
    return sorted(rows)


def load_model(weights_dir):
    from nemo.collections.asr.models import SortformerEncLabelModel
    path = hf_hub_download(MODEL_ID, "Nemotron-3-Diarization.nemo", revision=MODEL_REV,
                           local_dir=weights_dir)
    model = SortformerEncLabelModel.restore_from(path, map_location="cuda", strict=True)
    model.eval()
    for k, v in OFFLINE.items():
        setattr(model.sortformer_modules, k, v)
    model._check_streaming_parameters()
    return model, path


def probs_for(model, wav, cache):
    if cache.exists():
        z = np.load(cache)
        return z["probs"], float(z["dt"])
    import soundfile as sf
    dur = sf.info(wav).duration
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, probs = model.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=True)
    p = probs[0]
    p = (p[0] if p.ndim == 3 else p).float().cpu().numpy()
    dt = dur / p.shape[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probs=p, dt=dt)
    return p, dt


def mask(rows, n, dt):
    m = np.zeros(n, bool)
    for s, e, _ in rows:
        m[int(round(s / dt)):int(round(e / dt))] = True
    return m


def onsets(m):
    return np.flatnonzero(m[1:] & ~m[:-1]) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/diarization/diarization/train")
    ap.add_argument("--out", default="runs/probe_alignment")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    model, path = load_model("data/weights/nemotron3")
    torch.cuda.reset_peak_memory_stats()
    report = {"model": MODEL_ID, "revision": MODEL_REV, "config": OFFLINE,
              "threshold": a.threshold, "recordings": {}}
    for wav in sorted(Path(a.data, "audio").glob("*.wav")):
        rid = wav.stem
        rows = read_labels(Path(a.data, "annotation", rid + ".csv"))
        p, dt = probs_for(model, wav, out / "probs" / f"{rid}.npz")
        n = p.shape[0]
        speech = p.max(1) >= a.threshold
        lab = mask(rows, n, dt)
        seg_cov = []
        for s, e, spk in rows:
            i, j = int(round(s / dt)), int(round(e / dt))
            if j > i:
                seg_cov.append((s, e, spk, float(speech[i:j].mean())))
        mo, lo = onsets(speech), onsets(lab)
        deltas = [float((mo[np.argmin(np.abs(mo - x))] - x) * dt) for x in lo
                  if len(mo) and abs(mo[np.argmin(np.abs(mo - x))] - x) * dt <= 2.0]
        long_bad = [(s, e, spk, round(c, 3)) for s, e, spk, c in seg_cov if e - s >= 20 and c < 0.6]
        r = {
            "frames": n, "dt": dt, "minutes": n * dt / 60,
            "label_speech_frac": float(lab.mean()), "model_speech_frac": float(speech.mean()),
            "labelled_but_model_silent_frac_of_label": float((lab & ~speech).sum() / max(lab.sum(), 1)),
            "model_speech_but_unlabelled_frac_of_model": float((speech & ~lab).sum() / max(speech.sum(), 1)),
            "frame_agreement": float((lab == speech).mean()),
            "segments": len(seg_cov),
            "segments_cov_lt_30pct": sum(c < 0.3 for *_, c in seg_cov),
            "long_segments_ge20s_cov_lt_60pct": long_bad[:15],
            "onset_delta_s_median": float(np.median(deltas)) if deltas else None,
            "onset_delta_s_abs_p50": float(np.median(np.abs(deltas))) if deltas else None,
            "onset_delta_s_abs_p90": float(np.percentile(np.abs(deltas), 90)) if deltas else None,
            "onsets_matched_within_2s": f"{len(deltas)}/{len(lo)}",
            "max_channels_active": int((p >= a.threshold).any(0).sum()),
        }
        report["recordings"][rid] = r
        print(rid, json.dumps({k: v for k, v in r.items() if k != "long_segments_ge20s_cov_lt_60pct"}), flush=True)
    report["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("peak_vram_gib", report["peak_vram_gib"])


if __name__ == "__main__":
    main()
