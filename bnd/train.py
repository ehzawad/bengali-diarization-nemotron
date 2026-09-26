"""PREREG_phase2 "Fine-tuning recipe": one native NeMo streaming fine-tune of the pinned base.

Fixed recipe (only lr and timing are CLI factors):
  * data  : gap-5 trusted-span manifest (``data/prepared/manifests/train_manifest_balanced.json``,
            span lines repeated ~duration/105 s so exposure is audio-proportional) filtered to an
            explicit recording list; every line is asserted to belong to that list. ``--timing
            shift`` re-points each line at per-span RTTMs whose starts are moved later by delta
            (``bnd.timing``; delta estimated from the SAME training recordings only).
  * model : ``Nemotron-3-Diarization.nemo`` restored with ``strict=True``; training streaming
            settings spkcache/fifo/chunk/update/left/right = 264/0/264/264/0/0.
  * loader: official stage-2 ``train_ds`` (subsegment_mode, session 105 s) with batch 8,
            num_workers 4, drop_last (every step is exactly 8 sessions).
  * params: last 4 encoder layers + encoder.final_norm + sortformer_modules, minus
            sortformer_modules.learnable_sil_emb and sortformer_modules.hidden_to_spks.
  * optim : AdamW(lr, betas (0.9, 0.98), wd 1e-3) on the trainable params only, grad clip 1.0
            (norm), bf16-mixed, linear warmup over 20 steps then cosine to lr/10 at step 165,
            exactly 165 optimizer steps, no validation / early stopping, seed 42.
  * export: inference streaming config 264/40/340/40/300 set on the modules AND the saved config,
            ``runs/models/<tag>.nemo``; strict re-restore; the restored model's predictions on a
            30 s probe must equal the in-memory model's; state dicts must be identical.

GPU use ONLY through ``scripts/gpu_run.sh``::

    scripts/gpu_run.sh python -m bnd.train --tag T --train-recs 013,005,... --lr 1e-5 --timing shift
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

from bnd import nemo_data as nd

REPO = Path(__file__).resolve().parents[1]
BASE_NEMO = REPO / "data/weights/nemotron3/Nemotron-3-Diarization.nemo"
MANIFEST = REPO / "data/prepared/manifests/train_manifest_balanced.json"
FITS_DIR = REPO / "runs/fits"
MODELS_DIR = REPO / "runs/models"
RECORDINGS = ("train_001", "train_002", "train_003", "train_004", "train_005",
              "train_006", "train_008", "train_009", "train_011", "train_013")

TRAIN_STREAM = dict(spkcache_len=264, fifo_len=0, chunk_len=264, spkcache_update_period=264,
                    chunk_left_context=0, chunk_right_context=0)
INFER_STREAM = dict(spkcache_len=264, fifo_len=40, chunk_len=340, chunk_right_context=40,
                    spkcache_update_period=300)
TOP_K_LAYERS = 4
FROZEN_IN_SCOPE = ("sortformer_modules.learnable_sil_emb", "sortformer_modules.hidden_to_spks")
BATCH = 8
NUM_WORKERS = 4
SESSION_S = 105.0
MAX_STEPS = 165
WARMUP = 20
MIN_LR_FRAC = 0.1
BETAS = (0.9, 0.98)
WEIGHT_DECAY = 1e-3
CLIP = 1.0
SEED = 42
MEM_LIMIT_GIB = 20.0
PROBE_REC, PROBE_OFFSET_S, PROBE_S = None, 120.0, 30.0  # probe audio = first training recording


class TrainError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------

def norm_recs(recs) -> list[str]:
    if isinstance(recs, str):
        recs = [r for r in recs.split(",") if r.strip()]
    out = [r.strip() if r.strip().startswith("train_") else f"train_{r.strip()}" for r in recs]
    if not out:
        raise TrainError("empty recording list")
    if len(set(out)) != len(out):
        raise TrainError(f"duplicate recordings in {out}")
    bad = [r for r in out if r not in RECORDINGS]
    if bad:
        raise TrainError(f"unknown recordings {bad}")
    return out


def filter_manifest(entries: list[dict], recs: list[str]) -> list[dict]:
    """Lines of ``entries`` whose recording is in ``recs``; asserts nothing else leaks in (by
    recording_id, audio file stem and span_id prefix) and every requested recording is present."""
    keep = set(recs)
    out = []
    for e in entries:
        rid = e.get("recording_id")
        if rid is None:
            raise TrainError(f"manifest line without recording_id: {e}")
        stem = Path(e["audio_filepath"]).stem
        if stem != rid or not str(e.get("span_id", "")).startswith(rid + "_"):
            raise TrainError(f"inconsistent line (recording_id {rid}, audio {stem}, span {e.get('span_id')})")
        if rid in keep:
            out.append(e)
    got = {e["recording_id"] for e in out}
    if not got <= keep:  # defensive: cannot happen by construction
        raise TrainError(f"leaked recordings {sorted(got - keep)}")
    missing = keep - got
    if missing:
        raise TrainError(f"requested recordings with no spans in the manifest: {sorted(missing)}")
    return out


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in entries))
    tmp.replace(path)


def build_fit_manifest(recs: list[str], timing: str, fit_dir: Path, base_manifest: Path = MANIFEST,
                       **delta_kw) -> tuple[Path, dict]:
    entries = filter_manifest(read_manifest(base_manifest), recs)
    info: dict = {"base_manifest": str(base_manifest), "timing": timing, "n_lines": len(entries),
                  "n_spans": len({e["span_id"] for e in entries}),
                  "recordings": sorted({e["recording_id"] for e in entries}),
                  "span_hours": sum(e["duration"] for e in {e["span_id"]: e for e in entries}.values()) / 3600}
    if timing == "shift":
        from bnd import timing as tm

        est = tm.estimate_delta(recs, **delta_kw)
        entries, stats = tm.write_shifted_manifest(entries, est["delta_s"], fit_dir / "rttm_shift")
        info["delta"] = est
        info["shift_stats"] = stats
    elif timing != "raw":
        raise TrainError(f"timing must be raw|shift, got {timing}")
    path = fit_dir / "train_manifest.json"
    write_jsonl(path, entries)
    # re-read and re-check what NeMo will actually read
    filter_manifest(read_manifest(path), recs)
    return path, info


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------

def set_trainable(model, top_k: int = TOP_K_LAYERS) -> dict:
    """Freeze all, unfreeze the recipe's scope, return the census."""
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = len(model.encoder.layers)
    scope = [f"encoder.layers.{i}." for i in range(n_layers - top_k, n_layers)]
    scope += ["encoder.final_norm.", "sortformer_modules."]
    frozen_in_scope = tuple(f + "." for f in FROZEN_IN_SCOPE) + FROZEN_IN_SCOPE
    for name, p in model.named_parameters():
        if any(name.startswith(s) for s in scope) and not (name in FROZEN_IN_SCOPE or name.startswith(frozen_in_scope)):
            p.requires_grad_(True)
    return census(model)


def census(model) -> dict:
    groups: dict[str, int] = {}
    frozen_named: dict[str, int] = {}
    total = trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        parts = name.split(".")
        key = ".".join(parts[:3]) if parts[:2] == ["encoder", "layers"] else ".".join(parts[:2])
        if p.requires_grad:
            trainable += p.numel()
            groups[key] = groups.get(key, 0) + p.numel()
        elif name.startswith("sortformer_modules.") or name.startswith("encoder.final_norm"):
            frozen_named[key] = frozen_named.get(key, 0) + p.numel()
    return {"total_params": total, "trainable_params": trainable, "trainable_by_group": groups,
            "frozen_in_scope": frozen_named, "n_encoder_layers": len(model.encoder.layers)}


def lr_factor(i: int, max_steps: int = MAX_STEPS, warmup: int = WARMUP, min_frac: float = MIN_LR_FRAC) -> float:
    """LambdaLR factor for the (i+1)-th optimizer step: linear warmup k/warmup for k <= warmup,
    then cosine from 1 to min_frac, reaching min_frac exactly at k = max_steps."""
    k = i + 1
    if k <= warmup:
        return k / warmup
    t = min((k - warmup) / (max_steps - warmup), 1.0)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))


def set_stream(model, cfg_too: bool, **kw) -> None:
    from omegaconf import open_dict

    for k, v in kw.items():
        setattr(model.sortformer_modules, k, v)
    if cfg_too:
        with open_dict(model.cfg):
            for k, v in kw.items():
                model.cfg.sortformer_modules[k] = v
    model._check_streaming_parameters()


def restore(path: Path, map_location: str, trainer=None):
    from nemo.collections.asr.models import SortformerEncLabelModel

    return SortformerEncLabelModel.restore_from(str(path), map_location=map_location, strict=True,
                                                trainer=trainer)


def probe_preds(model, audio, device) -> "torch.Tensor":
    import torch

    model.eval()
    x = torch.as_tensor(audio, dtype=torch.float32, device=device)[None]
    n = torch.tensor([x.shape[1]], device=device)
    with torch.inference_mode():
        return model.forward(audio_signal=x, audio_signal_length=n).float().cpu()


# ----------------------------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------------------------

def fit(tag: str, recs: list[str], lr: float, timing: str) -> dict:
    import lightning.pytorch as pl
    import numpy as np
    import soundfile as sf
    import torch
    from omegaconf import OmegaConf, open_dict

    t_all = time.perf_counter()
    if not torch.cuda.is_available():
        raise TrainError("no CUDA device (launch through scripts/gpu_run.sh)")
    fit_dir = FITS_DIR / tag
    model_path = MODELS_DIR / f"{tag}.nemo"
    if model_path.exists():
        raise TrainError(f"{model_path} exists; refusing to overwrite")
    fit_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"tag": tag, "train_recs": recs, "lr": lr, "timing": timing, "status": "started",
                    "device": torch.cuda.get_device_name(0)}
    manifest, report["data"] = build_fit_manifest(recs, timing, fit_dir)
    if timing == "shift":
        print(f"[train] delta = {report['data']['delta']['delta_s']:+.2f} s "
              f"(median {report['data']['delta']['median_raw_s']:+.3f}; per rec "
              f"{ {p['recording_id']: p['shift_s'] for p in report['data']['delta']['per_recording']} })", flush=True)
    print(f"[train] manifest {manifest}: {report['data']['n_lines']} lines / {report['data']['n_spans']} spans "
          f"/ {report['data']['span_hours']:.2f} h, recordings {report['data']['recordings']}", flush=True)

    pl.seed_everything(SEED, workers=True)
    s2 = nd.stage2_config()
    s2.model.train_ds.manifest_filepath = str(manifest)
    tds = OmegaConf.create(OmegaConf.to_container(s2.model.train_ds, resolve=True))
    with open_dict(tds):
        tds.batch_size = BATCH
        tds.num_workers = NUM_WORKERS
        tds.sample_rate = 16000
        tds.num_spks = 8
        tds.validate_manifest_paths = False
        tds.drop_last = True
    if not (tds.subsegment_mode and float(tds.session_len_sec) == SESSION_S):
        raise TrainError(f"stage-2 train_ds not subsegment/105 s: {OmegaConf.to_container(tds)}")

    rows: list[dict] = []
    state = {"grad_norm": None, "lr": None, "t0": None}

    class Recorder(pl.Callback):
        def on_train_start(self, trainer, module):
            state["t0"] = time.perf_counter()

        def on_before_optimizer_step(self, trainer, module, optimizer):
            g = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
            state["grad_norm"] = float(torch.linalg.vector_norm(torch.stack(g))) if g else float("nan")
            state["lr"] = optimizer.param_groups[0]["lr"]  # lr this optimizer step uses

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss = float(loss.detach())
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {trainer.global_step}")
            rows.append({"step": trainer.global_step, "loss": loss,
                         "lr": state["lr"],
                         "grad_norm_preclip": state["grad_norm"], "batch_size": int(batch[0].shape[0]),
                         "wall_s": time.perf_counter() - state["t0"]})
            if trainer.global_step % 10 == 0 or trainer.global_step == 1:
                r = rows[-1]
                print(f"[train] step {r['step']:3d} loss {loss:.4f} lr {r['lr']:.3e} "
                      f"gnorm {r['grad_norm_preclip']:.3f} {r['wall_s']:.0f}s", flush=True)

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_steps=MAX_STEPS,
                         max_epochs=-1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         num_sanity_val_steps=0, limit_val_batches=0, gradient_clip_val=CLIP,
                         gradient_clip_algorithm="norm", accumulate_grad_batches=1, callbacks=[Recorder()])
    model = restore(BASE_NEMO, "cpu", trainer=trainer)
    orig_cfg = copy.deepcopy(model.cfg)
    set_stream(model, cfg_too=False, **TRAIN_STREAM)
    if not model.streaming_mode:
        raise TrainError("base model is not in streaming mode")
    report["census"] = set_trainable(model)
    print("[train] trainable census " + json.dumps(report["census"]), flush=True)
    with open_dict(model.cfg):
        model.cfg.train_ds = tds
        model.cfg.validation_ds = None
        model.cfg.optim = OmegaConf.create({"name": "adamw", "lr": lr, "betas": list(BETAS),
                                            "weight_decay": WEIGHT_DECAY,
                                            "sched": {"name": "warmup_cosine(bnd.train.lr_factor)",
                                                      "warmup_steps": WARMUP, "max_steps": MAX_STEPS,
                                                      "min_lr": lr * MIN_LR_FRAC}})
    params = [p for p in model.parameters() if p.requires_grad]

    def configure_optimizers():
        opt = torch.optim.AdamW(params, lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        model._optimizer, model._scheduler = opt, sched  # NeMo's training_step logs self._optimizer's lr
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

    model.configure_optimizers = configure_optimizers
    model.setup_training_data(model.cfg.train_ds)
    report["loader_batches_per_epoch"] = len(model._train_dl)
    torch.cuda.reset_peak_memory_stats()
    t_fit = time.perf_counter()
    trainer.fit(model)
    torch.cuda.synchronize()
    report["fit_wall_s"] = time.perf_counter() - t_fit
    report["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    report["optimizer_steps"] = int(trainer.global_step)
    report["epochs_equiv"] = MAX_STEPS / report["loader_batches_per_epoch"]

    with open(fit_dir / "loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    losses = np.array([r["loss"] for r in rows])
    report["loss"] = {"first": float(losses[0]), "last": float(losses[-1]),
                      "mean_first10": float(losses[:10].mean()), "mean_last10": float(losses[-10:].mean()),
                      "min": float(losses.min()), "max": float(losses.max())}
    report["lr_trace"] = {"step1": rows[0]["lr"], "step20": rows[min(19, len(rows) - 1)]["lr"], "last": rows[-1]["lr"]}
    report["frac_steps_clipped"] = float(np.mean([r["grad_norm_preclip"] > CLIP for r in rows]))
    report["batch_sizes"] = sorted({r["batch_size"] for r in rows})

    def fail(msg):
        report["status"] = f"FAILED: {msg}"
        (fit_dir / "fit_report.json").write_text(json.dumps(report, indent=1))
        raise TrainError(msg)

    if trainer.global_step != MAX_STEPS or len(rows) != MAX_STEPS:
        fail(f"expected {MAX_STEPS} optimizer steps, got global_step {trainer.global_step}, {len(rows)} batches")
    if report["batch_sizes"] != [BATCH]:
        fail(f"batch sizes {report['batch_sizes']}")
    if report["peak_reserved_gib"] >= MEM_LIMIT_GIB:
        fail(f"peak reserved {report['peak_reserved_gib']:.2f} GiB >= {MEM_LIMIT_GIB}")

    # ---- export ------------------------------------------------------------------------------
    device = torch.device("cuda")
    model = model.to(device).float()
    model.eval()
    set_stream(model, cfg_too=True, **INFER_STREAM)
    with open_dict(model.cfg):
        model.cfg.train_ds = orig_cfg.train_ds  # no local paths in the exported config
        model.cfg.validation_ds = orig_cfg.validation_ds
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_to(str(model_path))
    probe_rec = recs[0]
    audio, sr = sf.read(str(nd.RAW_TRAIN / "audio" / f"{probe_rec}.wav"), dtype="float32",
                        start=int(PROBE_OFFSET_S * 16000), frames=int(PROBE_S * 16000))
    if sr != 16000 or audio.ndim != 1 or audio.shape[0] != int(PROBE_S * 16000):
        fail(f"probe audio {audio.shape} @ {sr}")
    p_mem = probe_preds(model, audio, device)
    sd_mem = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    re = restore(model_path, "cuda").float()
    for k, v in INFER_STREAM.items():
        if getattr(re.sortformer_modules, k) != v:
            fail(f"re-restored {k}={getattr(re.sortformer_modules, k)} != {v}")
    sd_re = re.state_dict()
    if sd_re.keys() != sd_mem.keys() or any(not torch.equal(sd_re[k].cpu(), sd_mem[k]) for k in sd_mem):
        fail("re-restored state dict differs from the in-memory model")
    p_re = probe_preds(re, audio, device)
    diff = float((p_re - p_mem).abs().max())
    report["export"] = {"path": str(model_path), "bytes": model_path.stat().st_size,
                        "probe": {"recording": probe_rec, "offset_s": PROBE_OFFSET_S, "seconds": PROBE_S,
                                  "shape": list(p_mem.shape), "max_abs_diff": diff,
                                  "exactly_equal": bool(torch.equal(p_re, p_mem)),
                                  "mean_prob": float(p_mem.mean())},
                        "state_dict_identical": True, "infer_stream": INFER_STREAM}
    if not torch.equal(p_re, p_mem):
        fail(f"probe predictions differ after re-restore (max |diff| {diff:.3e})")
    report["total_wall_s"] = time.perf_counter() - t_all
    report["status"] = "ok"
    (fit_dir / "fit_report.json").write_text(json.dumps(report, indent=1))
    print("[train] FIT " + json.dumps({k: report[k] for k in ("tag", "optimizer_steps", "loss", "peak_reserved_gib",
                                                             "fit_wall_s", "total_wall_s")}), flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PREREG phase-2 fine-tune (run via scripts/gpu_run.sh)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-recs", required=True, help="comma-separated ids, e.g. 013,005 or train_013")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--timing", choices=("raw", "shift"), required=True)
    a = ap.parse_args(argv)
    fit(a.tag, norm_recs(a.train_recs), a.lr, a.timing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
