"""Windowed native Nemotron inference -> local tracks (PREREG_phase2 "Inference system").

Writes the SHARED CONTRACT cache consumed by the linking stage, per recording
``runs/cache/<system>/<rid>/``:

* ``windows.json``  list of ``{"window","start_s","end_s","core_start_s","core_end_s"}``,
  absolute seconds; cores are ``core_s`` pieces (20 s = phase 2, the default) that partition
  ``[0, duration)`` exactly once, inputs add 5 s context each side, clipped at the file edges
  (input <= core_s + 10 s).  Session length (PREREG_phase3b): ``--core-s 80`` runs ONE native
  forward call (264/40/340/40/300 inside) per 80 s core + context; with ``--session-subdir`` the
  cache lives in ``<cache-root>/<system>/s<core>/<rid>/`` (distinct identity per session length;
  ``meta.json`` records ``core_s`` and a cache with another ``core_s`` is never treated as complete).
  ``--core-s 20`` without ``--session-subdir`` is byte-identical to phase 2.
* ``probs_<window>.npy``  float16 ``[frames, 8]`` native probabilities for the whole window
  input, ``frames = floor(input_samples / 160)`` (NeMo feature count, asserted per window);
  frame ``i`` covers ``[start_s + i*dt, start_s + (i+1)*dt)`` (< 10 ms tail of an input that is
  not a multiple of 160 samples is not covered).
* ``meta.json``  ``{"dt","model_sha256","nemo_commit","streaming","recording_duration_s","system",...}``.
* ``tracks_<θ>.json`` (θ formatted ``%.2f``, e.g. ``tracks_0.40.json``; a ``tracks_0.4.json``
  symlink is also written) list of tracks, one per (window, channel) whose activity survives
  ``p >= θ_act`` + 0.1 s minimum turn.  Fields: ``track, window, channel, segments`` (absolute,
  core+context), ``core_segments`` (absolute, owned core only), ``usable_speech_s, weak`` and
  provenance/crop extras (``crops``, ``active_s``, ``core_active_s``, ``max_prob``).
* ``emb_<θ>.npy`` + ``emb_<θ>_meta.json``/``emb_meta.json``: written by :mod:`bnd.embed` in the
  separate pyannote venv (``third_party/embed-venv``), launched from here as a subprocess.

Definitions (fixed here, before any DER):
* active frame of channel c: ``p[:, c] >= θ_act``; runs shorter than 0.1 s are dropped.
* usable ("confident, non-overlapped") frame of track c: an active frame of the (min-turn
  filtered) track at which every OTHER channel is ``< θ_act``; usable runs shorter than 0.1 s are
  dropped.  ``usable_speech_s`` = total usable duration over the whole window input.
  ``weak`` = ``usable_speech_s < 1.0``.
* crops: usable runs concatenated in time order (true audio only, no padding) and cut into
  ``k = ceil(L/4)`` equal pieces, so every piece is 1.5-4 s whenever ``L >= 1.5`` s.  If
  ``0.5 <= L < 1.5`` a single short piece of all usable speech is used; ``L < 0.5`` -> no crop
  -> NaN embedding row.
* DECISION (logged 2026-09-25, before any DER; PREREG leaves this band undefined): a STRONG track
  (usable >= 1.0 s) with ``1.0 <= L < 1.5`` s has no conforming 1.5-4 s piece.  It stays strong
  (the PREREG weak rule is ``< 1.0 s``) and is embedded from its single short piece of all usable
  speech (no padding, no non-usable audio added).  Measured on the zero-shot cache: 100/3061
  (θ 0.40), 103/3038 (0.50), 95/2964 (0.60) strong tracks; counted per recording as
  ``strong_short_crop`` in ``infer_report.json`` and recorded in the embedding meta crop policy.
* ``tracks_<θ>.json`` rewrites invalidate embeddings: ``emb_<θ>_meta.json`` records the sha256 of
  the tracks file it embedded (``tracks_sha256``); :func:`write_tracks` deletes ``emb_<θ>.npy`` and
  its meta whenever that hash is missing or differs from the new tracks file, so stale rows can
  never be paired with new tracks (a missing emb file fails closed in the linker).

GPU: run only through ``scripts/gpu_run.sh``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bnd.labels import RECORDINGS

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/raw/diarization/diarization"
NEMO_ROOT = REPO / "third_party/nemo-speech"
EMBED_PY = REPO / "third_party/embed-venv/bin/python"
DEFAULT_MODEL = REPO / "data/weights/nemotron3/Nemotron-3-Diarization.nemo"
CACHE_ROOT = REPO / "runs/cache"

SR = 16000
CORE_S = 20.0
CONTEXT_S = 5.0
N_CH = 8
MIN_TURN_S = 0.1
WEAK_S = 1.0
CROP_MIN_S = 1.5
CROP_MAX_S = 4.0
SHORT_CROP_MIN_S = 0.5
# Model-card offline streaming config (SPEC / PREREG): 264/40/340/40/300.
OFFLINE = dict(spkcache_len=264, fifo_len=40, chunk_len=340, chunk_right_context=40,
               spkcache_update_period=300)
EPS = 1e-9


class InferError(RuntimeError):
    pass


# ------------------------------------------------------------------ bookkeeping
def theta_tag(theta: float) -> str:
    return f"{theta:.2f}"


def session_tag(core_s: float) -> str:
    """Cache sub-directory of a session length (PREREG_phase3b): 20 -> 's20', 80 -> 's80'."""
    if abs(core_s - round(core_s)) > 1e-9:
        raise InferError(f"session core must be whole seconds, got {core_s}")
    return f"s{int(round(core_s))}"


def make_windows(n_samples: int, sr: int = SR, core_s: float = CORE_S,
                 context_s: float = CONTEXT_S) -> list[dict]:
    """Cores of ``core_s`` partition [0, n_samples) (last core shorter); inputs add context,
    clipped at the file edges.  All boundaries are integer samples; seconds = samples / sr."""
    if n_samples <= 0:
        raise InferError("empty recording")
    core, ctx = int(round(core_s * sr)), int(round(context_s * sr))
    out = []
    for w, cs in enumerate(range(0, n_samples, core)):
        ce = min(cs + core, n_samples)
        s, e = max(0, cs - ctx), min(n_samples, ce + ctx)
        out.append({"window": w, "start_s": s / sr, "end_s": e / sr,
                    "core_start_s": cs / sr, "core_end_s": ce / sr,
                    "start_sample": s, "end_sample": e})
    check_windows(out, n_samples / sr, max_input_s=core_s + 2 * context_s)
    return out


def check_windows(windows: list[dict], duration: float, max_input_s: float = CORE_S + 2 * CONTEXT_S):
    """Cores tile [0, duration) exactly once; each input contains its core; input <= 30 s."""
    t = 0.0
    for i, w in enumerate(windows):
        if w["window"] != i:
            raise InferError(f"window index {w['window']} at position {i}")
        if abs(w["core_start_s"] - t) > EPS or w["core_end_s"] <= w["core_start_s"]:
            raise InferError(f"core gap/overlap at window {i}: {w}")
        if not (w["start_s"] <= w["core_start_s"] and w["core_end_s"] <= w["end_s"]):
            raise InferError(f"window {i} input does not contain its core")
        if w["start_s"] < -EPS or w["end_s"] > duration + EPS:
            raise InferError(f"window {i} outside the recording")
        if w["end_s"] - w["start_s"] > max_input_s + EPS:
            raise InferError(f"window {i} input longer than {max_input_s} s")
        t = w["core_end_s"]
    if abs(t - duration) > EPS:
        raise InferError(f"cores end at {t}, recording at {duration}")


def public_windows(windows: list[dict]) -> list[dict]:
    keys = ("window", "start_s", "end_s", "core_start_s", "core_end_s")
    return [{k: w[k] for k in keys} for w in windows]


# ------------------------------------------------------------------ track extraction
def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [a, b) frame runs of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b)) for a, b in zip(d[::2], d[1::2])]


def min_turn_filter(mask: np.ndarray, dt: float, min_s: float = MIN_TURN_S) -> np.ndarray:
    out = np.zeros(len(mask), bool)
    for a, b in runs_of(mask):
        if (b - a) * dt >= min_s - EPS:
            out[a:b] = True
    return out


def frames_to_segments(mask: np.ndarray, dt: float, t0: float, t1: float) -> list[list[float]]:
    """Frame runs -> absolute [start, end] seconds, frame i = [t0 + i*dt, t0 + (i+1)*dt), ends
    clipped to the window end t1."""
    segs = []
    for a, b in runs_of(mask):
        s, e = t0 + a * dt, min(t0 + b * dt, t1)
        if e > s + EPS:
            segs.append([round(s, 6), round(e, 6)])
    return segs


def clip_segments(segs: list[list[float]], lo: float, hi: float) -> list[list[float]]:
    out = []
    for s, e in segs:
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2 + EPS:
            out.append([round(s2, 6), round(e2, 6)])
    return out


def seg_total(segs) -> float:
    return float(sum(e - s for s, e in segs))


def plan_crops(usable: list[list[float]]) -> list[list[list[float]]]:
    """Concatenate usable runs in time order and cut into equal pieces of CROP_MIN_S..CROP_MAX_S
    (k = ceil(L / CROP_MAX_S)); single short piece for SHORT_CROP_MIN_S <= L < CROP_MIN_S;
    nothing below that.  Each piece is a list of absolute [start, end] audio intervals."""
    L = seg_total(usable)
    if L < SHORT_CROP_MIN_S - EPS:
        return []
    if L < CROP_MIN_S - EPS:
        return [[list(x) for x in usable]]
    k = max(1, math.ceil(L / CROP_MAX_S - EPS))
    piece = L / k
    pieces, cur, acc = [], [], 0.0
    queue = [list(x) for x in usable]
    i = 0
    while i < len(queue):
        s, e = queue[i]
        need = piece - acc
        if len(pieces) == k - 1 or e - s <= need + EPS:  # last piece takes the remainder
            cur.append([s, e])
            acc += e - s
            i += 1
        else:
            cut = round(s + need, 6)
            cur.append([s, cut])
            acc += cut - s
            queue[i] = [cut, e]
        if len(pieces) < k - 1 and acc >= piece - 1e-6:
            pieces.append(cur)
            cur, acc = [], 0.0
    if cur:
        pieces.append(cur)
    return [p for p in pieces if seg_total(p) > EPS]


def extract_tracks(probs: np.ndarray, dt: float, win: dict, theta: float,
                   duration: float, first_track_id: int = 0) -> list[dict]:
    """Local tracks of one window: every channel with activity >= theta after 0.1 s min turn."""
    p = np.asarray(probs, np.float32)
    if p.ndim != 2 or p.shape[1] != N_CH:
        raise InferError(f"probs shape {p.shape}, expected [frames, {N_CH}]")
    t0, t1 = win["start_s"], min(win["end_s"], duration)
    raw_active = p >= theta
    tracks = []
    for c in range(N_CH):
        act = min_turn_filter(raw_active[:, c], dt)
        segs = frames_to_segments(act, dt, t0, t1)
        if not segs:
            continue
        others = np.delete(raw_active, c, axis=1).any(axis=1)
        usable_mask = min_turn_filter(act & ~others, dt)
        usable = frames_to_segments(usable_mask, dt, t0, t1)
        core = clip_segments(segs, win["core_start_s"], win["core_end_s"])
        u = seg_total(usable)
        tracks.append({
            "track": first_track_id + len(tracks), "window": win["window"], "channel": c,
            "segments": segs, "core_segments": core,
            "usable_speech_s": round(u, 6), "weak": bool(u < WEAK_S - EPS),
            "active_s": round(seg_total(segs), 6), "core_active_s": round(seg_total(core), 6),
            "max_prob": float(p[:, c].max()),
            "crops": plan_crops(usable),
        })
    return tracks


def tracks_for_recording(cache: Path, theta: float) -> list[dict]:
    meta = json.loads((cache / "meta.json").read_text())
    windows = json.loads((cache / "windows.json").read_text())
    tracks = []
    for w in windows:
        p = np.load(cache / f"probs_{w['window']}.npy")
        tracks += extract_tracks(p, meta["dt"], w, theta, meta["recording_duration_s"], len(tracks))
    return tracks


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def tracks_bytes(tracks: list[dict]) -> bytes:
    return json.dumps(tracks, indent=1).encode()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def emb_paths(cache: Path, theta: float) -> tuple[Path, Path]:
    tag = theta_tag(theta)
    return cache / f"emb_{tag}.npy", cache / f"emb_{tag}_meta.json"


def emb_tracks_sha(cache: Path, theta: float) -> str | None:
    try:
        return json.loads(emb_paths(cache, theta)[1].read_text()).get("tracks_sha256")
    except (OSError, ValueError):
        return None


def invalidate_stale_embeddings(cache: Path, theta: float, tracks_sha: str) -> bool:
    """Delete emb_<θ>.npy + meta unless the meta says they were computed from exactly the tracks
    file with sha256 ``tracks_sha``.  Returns True if something was removed."""
    npy, meta = emb_paths(cache, theta)
    if emb_tracks_sha(cache, theta) == tracks_sha and npy.exists():
        return False
    removed = False
    for p in (npy, meta):  # npy first: a crash in between leaves no emb rows to mis-pair
        if p.exists() or p.is_symlink():
            p.unlink()
            removed = True
    return removed


def emb_is_current(cache: Path, theta: float) -> bool:
    """True iff emb_<θ>.npy exists and its meta hash matches the tracks file on disk."""
    npy, _ = emb_paths(cache, theta)
    tpath = cache / f"tracks_{theta_tag(theta)}.json"
    return (npy.exists() and tpath.exists()
            and emb_tracks_sha(cache, theta) == sha256_bytes(tpath.read_bytes()))


def write_tracks(cache: Path, theta: float) -> list[dict]:
    tracks = tracks_for_recording(cache, theta)
    path = cache / f"tracks_{theta_tag(theta)}.json"
    data = tracks_bytes(tracks)
    invalidate_stale_embeddings(cache, theta, sha256_bytes(data))  # before the tracks change
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    alias = cache / f"tracks_{theta!r}.json"
    if alias.name != path.name:
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        alias.symlink_to(path.name)
    return tracks


# ------------------------------------------------------------------ model
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def nemo_commit() -> str:
    return subprocess.check_output(["git", "-C", str(NEMO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def load_model(path: Path, device: str = "cuda"):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(str(path), map_location=device, strict=True)
    model.eval()
    if not model.streaming_mode:
        raise InferError("pinned model is expected to run in native streaming mode")
    for k, v in OFFLINE.items():
        setattr(model.sortformer_modules, k, v)
    model._check_streaming_parameters()
    if not (model.high_resolution and model.output_subsampling_factor == 1):
        raise InferError("expected high-resolution 10 ms output frames")
    torch.set_grad_enabled(False)
    return model


def streaming_config(model) -> dict:
    sm = model.sortformer_modules
    keys = ("spkcache_len", "fifo_len", "chunk_len", "chunk_left_context", "chunk_right_context",
            "spkcache_update_period", "subsampling_factor")
    d = {k: getattr(sm, k) for k in keys}
    d.update(async_streaming=bool(model.async_streaming), streaming_mode=bool(model.streaming_mode),
             precision="bf16-autocast")
    return d


def model_dt(model) -> float:
    hop = model.preprocessor.featurizer.hop_length
    return hop / model.preprocessor._cfg.sample_rate * model.output_subsampling_factor


def expected_frames(model, n_samples: int) -> int:
    import torch
    n = model.preprocessor.featurizer.get_seq_len(torch.tensor([n_samples], dtype=torch.float)).item()
    return math.ceil(int(n) / model.output_subsampling_factor)


def native_probs(model, waves: list[np.ndarray]) -> list[np.ndarray]:
    """Native streaming forward of a batch of equal-length waveforms (float32, 16 kHz)."""
    import torch
    lens = {len(w) for w in waves}
    if len(lens) != 1:
        raise InferError("native_probs batches must be equal length (no padding)")
    n = lens.pop()
    x = torch.from_numpy(np.stack(waves).astype(np.float32)).to(model.device)
    xl = torch.full((len(waves),), n, dtype=torch.long, device=model.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        p = model.forward(audio_signal=x, audio_signal_length=xl)
    p = p.float().cpu().numpy()
    exp = expected_frames(model, n)
    dt = model_dt(model)
    if p.shape[1] != exp or p.shape[2] != N_CH:
        raise InferError(f"model returned {p.shape} for {n} samples, expected [*, {exp}, {N_CH}]")
    # NeMo: frames = floor(n / hop); the frames cover [0, exp*dt] and leave < 1 frame of tail.
    if not exp * dt <= n / SR + EPS < (exp + 1) * dt:
        raise InferError(f"{exp} frames x {dt} s inconsistent with a {n / SR} s input")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise InferError("non-finite or out-of-range probabilities")
    return [p[i] for i in range(len(waves))]


def audio_path(rid: str) -> Path:
    for split in ("train", "test"):
        p = RAW / split / "audio" / f"{rid}.wav"
        if p.exists():
            return p
    raise InferError(f"no audio for {rid}")


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR or x.ndim != 1:
        raise InferError(f"{path}: expected {SR} Hz mono, got sr={sr} shape={x.shape}")
    return x


def cache_complete(cache: Path, meta_expect: dict) -> bool:
    try:
        meta = json.loads((cache / "meta.json").read_text())
        windows = json.loads((cache / "windows.json").read_text())
    except (OSError, ValueError):
        return False
    if any(meta.get(k) != v for k, v in meta_expect.items()):
        return False
    return all((cache / f"probs_{w['window']}.npy").exists() for w in windows)


def session_meta_base(system: str, model_sha: str, commit, dt: float, streaming, batch: int) -> dict:
    """meta.json prefix. core_s is deliberately NOT here: infer_recording appends it after
    audio_filepath, so the key order (and bytes) of meta.json for --core-s 20 match phase 2."""
    return dict(system=system, model_sha256=model_sha, nemo_commit=commit, dt=dt, streaming=streaming,
                n_channels=N_CH, batch=batch, frame_convention="frame i = [start_s+i*dt, start_s+(i+1)*dt)")


def infer_recording(model, rid: str, cache: Path, meta_base: dict, batch: int, core_s: float = CORE_S) -> dict:
    """One native forward call per window ("session") of ``core_s`` owned seconds + CONTEXT_S each side
    (batch 1 default); probs cached per window."""
    wav = read_audio(audio_path(rid))
    n = len(wav)
    windows = make_windows(n, core_s=core_s)
    cache.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[dict]] = {}
    for w in windows:
        groups.setdefault(w["end_sample"] - w["start_sample"], []).append(w)
    frames = 0
    for length, ws in groups.items():
        for i in range(0, len(ws), batch):
            chunk = ws[i:i + batch]
            outs = native_probs(model, [wav[w["start_sample"]:w["end_sample"]] for w in chunk])
            for w, p in zip(chunk, outs):
                np.save(cache / f"probs_{w['window']}.npy", p.astype(np.float16))
                frames += p.shape[0]
    write_json_atomic(cache / "windows.json", public_windows(windows))
    meta = dict(meta_base, recording_duration_s=n / SR, recording_id=rid, n_windows=len(windows),
                audio_filepath=str(audio_path(rid)), core_s=core_s, context_s=CONTEXT_S)
    write_json_atomic(cache / "meta.json", meta)  # written last: marks the probs complete
    return {"windows": len(windows), "frames": frames, "duration_s": n / SR,
            "window_lengths_s": sorted({round(k / SR, 4) for k in groups})}


def stitched_speech_agreement(cache: Path, full_probs: Path, thr: float = 0.5) -> dict | None:
    """Diagnostic (no labels, no DER): agreement of max-prob speech/non-speech between the
    core-stitched windowed probs and the full-recording probe probs."""
    if not full_probs.exists():
        return None
    z = np.load(full_probs)
    full = z["probs"].max(1) >= thr
    meta = json.loads((cache / "meta.json").read_text())
    dt = meta["dt"]
    win = np.zeros_like(full)
    for w in json.loads((cache / "windows.json").read_text()):
        p = np.load(cache / f"probs_{w['window']}.npy").astype(np.float32).max(1) >= thr
        a, b = int(round(w["core_start_s"] / dt)), min(int(round(w["core_end_s"] / dt)), len(full))
        off = int(round((w["core_start_s"] - w["start_s"]) / dt))
        seg = p[off:off + (b - a)]
        win[a:a + len(seg)] = seg
    return {"frame_agreement": float((full == win).mean()), "full_speech_frac": float(full.mean()),
            "windowed_speech_frac": float(win.mean())}


def run_embed(system_dir: Path, rids: list[str], thetas: list[float], device: str) -> dict:
    if not EMBED_PY.exists():
        raise InferError(f"embedding venv missing: {EMBED_PY}")
    out = system_dir / "embed_report.json"
    cmd = [str(EMBED_PY), "-m", "bnd.embed", "--cache-dir", str(system_dir), "--device", device,
           "--report", str(out), "--recordings", *rids, "--theta-act", *[theta_tag(t) for t in thetas]]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)
    return json.loads(out.read_text())


def is_short_crop(t: dict) -> bool:
    """Track embedded from a single sub-1.5 s piece (see the DECISION in the module doc)."""
    return len(t["crops"]) == 1 and seg_total(t["crops"][0]) < CROP_MIN_S - EPS


def track_stats(tracks: list[dict], duration: float, core_s: float = CORE_S) -> dict:
    n = len(tracks)
    return {"tracks": n, "weak": sum(t["weak"] for t in tracks),
            "weak_share": (sum(t["weak"] for t in tracks) / n) if n else None,
            "no_crop": sum(not t["crops"] for t in tracks),
            "short_crop_only": sum(is_short_crop(t) for t in tracks),
            "strong_short_crop": sum(is_short_crop(t) and not t["weak"] for t in tracks),
            "tracks_with_core_speech": sum(bool(t["core_segments"]) for t in tracks),
            "tracks_per_window_mean": n / max(1, math.ceil(duration / core_s))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--recordings", nargs="+", default=["all"])
    ap.add_argument("--theta-act", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--batch", type=int, default=1,
                    help="windows per forward; 1 (default) makes each window's probs independent of batch "
                         "composition (bf16 batched vs single differed by up to 0.012 in the GPU test)")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute probs even if cached")
    ap.add_argument("--core-s", type=float, default=CORE_S,
                    help="session length: owned core seconds per native forward call (+5 s context each side); "
                         "20 = phase 2 (PREREG_phase3b S in {20, 80})")
    ap.add_argument("--session-subdir", action="store_true",
                    help="cache under <cache-root>/<system>/s<core>/<rid>/ (distinct identity per session length)")
    a = ap.parse_args(argv)
    if not a.core_s > 0:
        ap.error("--core-s must be > 0")

    import torch
    rids = list(RECORDINGS) if a.recordings == ["all"] else a.recordings
    system_dir = a.cache_root / a.system
    if a.session_subdir:
        system_dir = system_dir / session_tag(a.core_s)
    t_start = time.perf_counter()
    model_sha = sha256_file(a.model)
    commit = nemo_commit()
    report = {"system": a.system, "model": str(a.model), "model_sha256": model_sha,
              "nemo_commit": commit, "recordings": {}}
    model = None
    t_model = 0.0
    for rid in rids:
        cache = system_dir / rid
        expect = {"system": a.system, "model_sha256": model_sha, "nemo_commit": commit, "core_s": a.core_s}
        if not a.force and cache_complete(cache, expect):
            print(f"[infer] {rid}: probs cached", flush=True)
            report["recordings"][rid] = {"cached": True}
            continue
        if model is None:
            model = load_model(a.model)
            torch.cuda.reset_peak_memory_stats()
            meta_base = session_meta_base(a.system, model_sha, commit, model_dt(model), streaming_config(model), a.batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = infer_recording(model, rid, cache, meta_base, a.batch, a.core_s)
        torch.cuda.synchronize()
        r["wall_s"] = time.perf_counter() - t0
        r["rtf"] = r["wall_s"] / r["duration_s"]
        t_model += r["wall_s"]
        r["speech_agreement_vs_fullrec"] = stitched_speech_agreement(
            cache, REPO / "runs/probe_alignment/probs" / f"{rid}.npz")
        report["recordings"][rid] = r
        print(f"[infer] {rid}: {json.dumps(r)}", flush=True)
    if model is not None:
        report["nemo_peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["nemo_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        del model
        torch.cuda.empty_cache()
    report["nemo_wall_s"] = t_model

    t0 = time.perf_counter()
    for rid in rids:
        cache = system_dir / rid
        dur = json.loads((cache / "meta.json").read_text())["recording_duration_s"]
        rep = report["recordings"].setdefault(rid, {})
        rep["duration_s"] = dur
        rep["tracks"] = {theta_tag(t): track_stats(write_tracks(cache, t), dur, a.core_s) for t in a.theta_act}
    report["tracks_wall_s"] = time.perf_counter() - t0
    if not a.no_embed:
        t0 = time.perf_counter()
        report["embed"] = run_embed(system_dir, rids, a.theta_act, "cuda")
        report["embed_wall_s"] = time.perf_counter() - t0
    stale = [(rid, theta_tag(t)) for rid in rids for t in a.theta_act
             if not emb_is_current(system_dir / rid, t)]
    report["emb_missing_or_stale"] = [f"{r}@{t}" for r, t in stale]
    if stale and not a.no_embed:
        raise InferError(f"embeddings not aligned with tracks after embedding: {stale[:5]}")
    report["total_wall_s"] = time.perf_counter() - t_start
    total_audio = sum(report["recordings"][r]["duration_s"] for r in rids)
    report["total_audio_s"] = total_audio
    report["rtf_total"] = report["total_wall_s"] / total_audio
    report["rtf_nemo"] = t_model / total_audio if t_model else None
    write_json_atomic(system_dir / "infer_report.json", report)
    print("[infer] REPORT", json.dumps({k: v for k, v in report.items() if k != "recordings"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
