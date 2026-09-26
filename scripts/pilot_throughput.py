"""Training throughput / VRAM pilot (throwaway weights).

Real NeMo streaming fine-tuning steps with the official stage-2 subsegment planner
(105 s sessions) on whole training recordings, labels = raw CSV minus invalid rows.
Measures seconds per microbatch and peak VRAM per batch size. Nothing is saved.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).parent))
from probe_alignment import hms  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NEMO = REPO / "third_party/nemo-speech"
RAW = REPO / "data/raw/diarization/diarization/train"
CKPT = REPO / "data/weights/nemotron3/Nemotron-3-Diarization.nemo"
TRAIN_STREAM = dict(spkcache_len=264, fifo_len=0, chunk_len=264, spkcache_update_period=264,
                    chunk_left_context=0, chunk_right_context=0)


def build_manifest(out):
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for wav in sorted((RAW / "audio").glob("*.wav")):
        rid = wav.stem
        rows = []
        with open(RAW / "annotation" / f"{rid}.csv", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                r = {k.strip().lower(): v for k, v in r.items()}
                s, e = hms(r["start_time"]), hms(r["end_time"])
                if e > s:
                    rows.append((s, e, r["speaker_id"].strip()))
        rttm = out / f"{rid}.rttm"
        rttm.write_text("".join(f"SPEAKER {rid} 1 {s:.3f} {e - s:.3f} <NA> <NA> spk{k} <NA> <NA>\n"
                                for s, e, k in rows))
        lines.append(json.dumps({"audio_filepath": str(wav), "offset": 0.0,
                                 "duration": sf.info(wav).duration,
                                 "num_speakers": len({k for *_, k in rows}),
                                 "rttm_filepath": str(rttm)}))
    (out / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return out / "manifest.jsonl"


def set_trainable(model, top_k=4):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in list(model.encoder.layers[-top_k:]) + [model.encoder.final_norm, model.sortformer_modules]:
        for p in m.parameters():
            p.requires_grad_(True)
    sm = model.sortformer_modules
    sm.hidden_to_spks.requires_grad_(False)
    if getattr(sm, "learnable_sil_emb", None) is not None:
        sm.learnable_sil_emb.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--session", type=float, default=105)
    ap.add_argument("--top-k", type=int, default=4)
    a = ap.parse_args()
    import lightning.pytorch as pl
    from nemo.collections.asr.models import SortformerEncLabelModel

    manifest = build_manifest(REPO / "runs/pilot/data")
    yaml = OmegaConf.load(NEMO / "examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_streaming_8spk.yaml")
    tds = yaml.model.train_ds
    with open_dict(tds):
        tds.manifest_filepath = str(manifest)
        tds.batch_size = a.batch
        tds.num_workers = 4
        tds.session_len_sec = a.session
        tds.sample_rate = 16000
        tds.num_spks = 8
        tds.validate_manifest_paths = False

    times = []

    class Timer(pl.Callback):
        def on_train_batch_start(self, *args):
            torch.cuda.synchronize()
            self.t = time.perf_counter()

        def on_train_batch_end(self, trainer, module, outputs, *args):
            torch.cuda.synchronize()
            times.append(time.perf_counter() - self.t)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_steps=a.steps,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         num_sanity_val_steps=0, limit_val_batches=0, callbacks=[Timer()],
                         gradient_clip_val=1.0)
    model = SortformerEncLabelModel.restore_from(str(CKPT), map_location="cpu", strict=True, trainer=trainer)
    for k, v in TRAIN_STREAM.items():
        setattr(model.sortformer_modules, k, v)
    model.streaming_mode = True
    model._check_streaming_parameters()
    ntrain = set_trainable(model, a.top_k)
    with open_dict(model.cfg):
        model.cfg.train_ds = tds
        model.cfg.validation_ds = None
        model.cfg.optim = OmegaConf.create({"name": "adamw", "lr": 1e-5, "betas": [0.9, 0.98],
                                            "weight_decay": 1e-3})
    model.setup_training_data(model.cfg.train_ds)
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model)
    warm = times[5:] or times
    res = {"batch": a.batch, "top_k": a.top_k, "session_s": a.session, "steps": len(times), "trainable_params": ntrain,
           "s_per_microbatch_median": sorted(warm)[len(warm) // 2],
           "audio_s_per_microbatch": a.batch * a.session,
           "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    print("PILOT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
