"""Diarize one audio file with the default system (configs/final_config.json).

    python -m bnd.diarize <audio> --out <dir> [--mode turns|acoustic|benchmark] [--session-core-s S]
                          [--config configs/final_config.json] [--keep-work]

Output modes (the model, linker and thresholds are identical in all three):
* ``turns`` (default): speaker turns; same-speaker pauses <= ``gap_fill_s`` (5 s) are filled.
* ``acoustic``: no gap filling; native 10 ms speech boundaries.
* ``benchmark``: ``turns`` + boundaries floored to whole seconds, matching the Bengali-Loop reference
  convention (bnd.link.floor_adapter). Use it only to score against Bengali-Loop-style references.
``--session-core-s 120`` uses longer native sessions: same development DER, ~3x lower speaker-count
error on the 10 development recordings; not validated on the hold-out.

Pipeline (every decoding parameter comes from final_config.json; nothing is tuned here):
1. ffmpeg -> 16 kHz mono PCM s16 WAV (``<out>/work/<stem>.wav``).
2. Model: the checkpoint named in the config; its sha256 must equal the config's (fail closed).
   Native streaming (264/40/340/40/300, bf16), one forward call per session of ``session_core_s``
   owned seconds + ``context_s`` each side (bnd.infer.make_windows / native_probs, batch 1).
3. Local tracks at θ_act (bnd.infer.write_tracks) -> WeSpeaker embeddings in the separate pyannote venv
   (bnd.embed, subprocess) -> linker (bnd.link.run_recording: ``linker``, ``theta_clu``) -> phase-2
   exclusive decoder -> same-speaker gap-fill ``gap_fill_s``.
4. Writes ``<out>/<stem>.rttm``, ``<out>/<stem>.csv`` (header ``start_time,end_time,speaker_id``,
   seconds with 3 decimals, speaker ids as in the RTTM) and ``<out>/<stem>.diarize_report.json``
   (timings, RTF, speaker count, model/config hashes).

No reference labels are read and nothing is scored. GPU: run through ``scripts/gpu_run.sh``.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bnd import infer
from bnd import link as L
from bnd import score as S

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/final_config.json"
MODES = ("turns", "acoustic", "benchmark")
CSV_HEADER = ("start_time", "end_time", "speaker_id")


class DiarizeError(RuntimeError):
    pass


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    d = cfg.get("decoding", {})
    need = ("session_core_s", "context_s", "theta_act", "linker", "theta_clu", "gap_fill_s")
    miss = [k for k in need if k not in d]
    if miss or "model" not in cfg or "sha256" not in cfg["model"]:
        raise DiarizeError(f"{path}: incomplete final config (missing {miss or 'model.sha256'})")
    if d["linker"] not in L.LINKERS:
        raise DiarizeError(f"{path}: unknown linker {d['linker']!r}")
    if abs(float(d["context_s"]) - infer.CONTEXT_S) > 1e-9:
        raise DiarizeError(f"{path}: context_s {d['context_s']} != implemented {infer.CONTEXT_S}")
    if float(d["gap_fill_s"]) < 0 or not 0 < float(d["theta_act"]) < 1 or not 0 < float(d["theta_clu"]) < 2:
        raise DiarizeError(f"{path}: out-of-range decoding parameter")
    return cfg


def model_path(cfg: dict) -> Path:
    p = Path(cfg["model"]["path_rel"])
    return p if p.is_absolute() else REPO / p


def to_wav16k(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(infer.SR), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def write_csv(path: Path, segs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for s in segs:
            w.writerow([f"{s.start:.3f}", f"{s.end:.3f}", s.speaker])
    tmp.replace(path)


def run_model(model_file: Path, wav_path: Path, rid: str, cache: Path, core_s: float) -> dict:
    """Probs for every session window -> cache/{probs_<w>.npy, windows.json, meta.json} (bnd.infer contract)."""
    import torch

    wav = infer.read_audio(wav_path)
    n = len(wav)
    windows = infer.make_windows(n, core_s=core_s)
    t0 = time.perf_counter()
    model = infer.load_model(model_file)
    t_load = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache.mkdir(parents=True, exist_ok=True)
    for w in windows:  # batch 1: each window's probs independent of batch composition (bnd.infer --batch)
        p = infer.native_probs(model, [wav[w["start_sample"]:w["end_sample"]]])[0]
        np.save(cache / f"probs_{w['window']}.npy", p.astype(np.float16))
    torch.cuda.synchronize()
    t_nemo = time.perf_counter() - t0
    infer.write_json_atomic(cache / "windows.json", infer.public_windows(windows))
    meta = dict(infer.session_meta_base("diarize", infer.sha256_file(model_file), infer.nemo_commit(),
                                        infer.model_dt(model), infer.streaming_config(model), 1),
                recording_duration_s=n / infer.SR, recording_id=rid, n_windows=len(windows),
                audio_filepath=str(wav_path), core_s=core_s, context_s=infer.CONTEXT_S)
    infer.write_json_atomic(cache / "meta.json", meta)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return {"duration_s": n / infer.SR, "windows": len(windows), "model_load_s": t_load, "nemo_s": t_nemo,
            "nemo_peak_alloc_gib": peak}


def resolve_settings(d: dict, mode: str = "turns", session_core_s: float | None = None) -> dict:
    """Effective (session_core_s, gap_fill_s, floor) for an output mode; ``turns`` + no override == config."""
    if mode not in MODES:
        raise DiarizeError(f"unknown mode {mode!r}; choose one of {MODES}")
    core = float(d["session_core_s"]) if session_core_s is None else float(session_core_s)
    if core < 1.0:
        raise DiarizeError(f"session_core_s must be >= 1 s, got {core}")
    gap = 0.0 if mode == "acoustic" else float(d["gap_fill_s"])
    return {"mode": mode, "session_core_s": core, "gap_fill_s": gap, "floor": mode == "benchmark"}


def diarize(audio: Path, out: Path, cfg_path: Path = DEFAULT_CONFIG, keep_work: bool = False,
            device: str = "cuda", mode: str = "turns", session_core_s: float | None = None) -> dict:
    t_start = time.perf_counter()
    cfg = load_config(cfg_path)
    d = cfg["decoding"]
    eff = resolve_settings(d, mode, session_core_s)
    audio = Path(audio).resolve()
    if not audio.exists():
        raise DiarizeError(f"no such audio: {audio}")
    rid = audio.stem
    out = Path(out)
    work = out / "work"
    wav16 = work / f"{rid}.wav"
    cache_root = work / "cache"
    cache = cache_root / rid
    if cache.exists():
        shutil.rmtree(cache)  # never reuse a previous run's probs/embeddings

    mfile = model_path(cfg)
    t0 = time.perf_counter()
    msha = infer.sha256_file(mfile)
    if msha != cfg["model"]["sha256"]:
        raise DiarizeError(f"{mfile}: sha256 {msha} != final_config {cfg['model']['sha256']}")
    t_sha = time.perf_counter() - t0

    t0 = time.perf_counter()
    to_wav16k(audio, wav16)
    t_ffmpeg = time.perf_counter() - t0

    theta = float(d["theta_act"])
    rep_model = run_model(mfile, wav16, rid, cache, eff["session_core_s"])

    t0 = time.perf_counter()
    tracks = infer.write_tracks(cache, theta)
    t_tracks = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb_rep = infer.run_embed(cache_root, [rid], [theta], device)
    t_embed = time.perf_counter() - t0
    if not infer.emb_is_current(cache, theta):
        raise DiarizeError("embeddings not aligned with tracks")

    t0 = time.perf_counter()
    c = L.load_cache(cache, theta, rid)
    segs, info = L.run_recording(c, theta, float(d["theta_clu"]), linker=d["linker"],
                                 gap_fill_s=eff["gap_fill_s"])
    if eff["floor"]:
        segs = L.floor_adapter(segs)
    t_link = time.perf_counter() - t0

    out.mkdir(parents=True, exist_ok=True)
    rttm, csvp = out / f"{rid}.rttm", out / f"{rid}.csv"
    S.write_rttm(rttm, rid, segs)
    write_csv(csvp, segs)
    total = time.perf_counter() - t_start
    dur = rep_model["duration_s"]
    report = {
        "audio": str(audio), "recording_id": rid, "duration_s": dur,
        "config": str(cfg_path), "config_key": cfg.get("config_key"), "default_system": cfg.get("default_system"),
        "model": str(mfile), "model_sha256": msha,
        "decoding": d, "effective": eff,
        "n_speakers": len({s.speaker for s in segs}), "n_turns": len(segs),
        "output_speech_s": round(sum(s.end - s.start for s in segs), 3),
        "n_tracks": len(tracks), "n_windows": rep_model["windows"],
        "gap_filled_s": info.get("gap_filled_s"),
        "timing_s": {"model_sha": t_sha, "ffmpeg": t_ffmpeg, "model_load": rep_model["model_load_s"],
                     "nemo": rep_model["nemo_s"], "tracks": t_tracks, "embed_total": t_embed,
                     "embed_inner": emb_rep.get("wall_s"), "link_decode": t_link, "total": total},
        "rtf_total": total / dur, "rtf_nemo": rep_model["nemo_s"] / dur,
        "nemo_peak_alloc_gib": rep_model["nemo_peak_alloc_gib"],
        "embed_peak_alloc_gib": emb_rep.get("peak_alloc_gib"),
        "outputs": {"rttm": str(rttm), "csv": str(csvp)},
    }
    infer.write_json_atomic(out / f"{rid}.diarize_report.json", report)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--keep-work", action="store_true", help="keep <out>/work (16 kHz wav, probs, tracks, embeddings)")
    ap.add_argument("--mode", choices=MODES, default="turns",
                    help="turns (default): fill same-speaker pauses <= 5 s; acoustic: no filling; "
                         "benchmark: turns + whole-second boundaries (Bengali-Loop convention)")
    ap.add_argument("--session-core-s", type=float, default=None,
                    help="native session core seconds (default from config: 20; 120 = fewer spurious speakers)")
    a = ap.parse_args(argv)
    r = diarize(a.audio, a.out, a.config, a.keep_work, mode=a.mode, session_core_s=a.session_core_s)
    print(json.dumps({k: r[k] for k in ("recording_id", "duration_s", "n_speakers", "n_turns", "rtf_total",
                                        "rtf_nemo", "outputs")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
