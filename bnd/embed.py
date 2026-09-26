"""WeSpeaker embeddings of local tracks (PREREG_phase2 "Inference system" -> Embeddings).

Runs in the SEPARATE pyannote venv (``third_party/embed-venv``); imports nothing from NeMo.
For each recording cache ``runs/cache/<system>/<rid>/`` and each θ_act it reads
``tracks_<θ>.json`` (crop plans made by :mod:`bnd.infer`) and writes

* ``emb_<θ>.npy``  float32 ``[n_tracks, D]``, rows aligned with the tracks list, L2-normalised;
  a NaN row for tracks with no crop.
* ``emb_<θ>_meta.json`` (per θ) and ``emb_meta.json`` (shared): model, revision, crop policy;
  the per-θ meta also records ``tracks_sha256`` (sha256 of the exact tracks file bytes embedded),
  written AFTER the npy so an interrupted run leaves a mismatch that :mod:`bnd.infer` treats as
  stale.

Every crop is embedded ALONE (batch 1) at its true length: WeSpeaker's fbank mean-normalisation
runs over all frames, so zero padding would leak into the features.  A crop made of several
usable runs is the concatenation of those audio intervals (same local track, no padding).
Track embedding = duration-weighted mean of its L2-normalised crop embeddings, re-normalised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
MODEL_REV = "837717ddb9ff5507820346191109dc79c958d614"
MODEL_DIR = REPO / "data/weights/wespeaker-resnet34-LM"
SR = 16000

CROP_POLICY = {
    "usable_frames": "track active (p>=theta_act, 0.1 s min turn) and all other 7 channels < theta_act; "
                     "usable runs < 0.1 s dropped",
    "pieces": "usable runs concatenated in time order, cut into k=ceil(L/4) equal pieces (1.5-4 s each "
              "when L>=1.5 s); 0.5<=L<1.5 s -> one short piece; L<0.5 s -> no crop (NaN row)",
    "strong_short_band": "DECISION 2026-09-25 (pre-DER; PREREG undefined): strong tracks with "
                         "1.0<=usable<1.5 s stay strong and are embedded from their single short "
                         "piece of all usable speech (no padding, no non-usable audio)",
    "valid_length": "each crop embedded alone at its true sample length, no padding",
    "track_embedding": "duration-weighted mean of L2-normalised crop embeddings, L2-normalised",
    "weak": "usable_speech_s < 1.0",
    "window_scope": "core + context (whole window input)",
}


def load_embedder(device: str):
    import torch
    from huggingface_hub import snapshot_download
    from pyannote.audio import Model
    path = snapshot_download(MODEL_ID, revision=MODEL_REV, local_dir=str(MODEL_DIR))
    m = Model.from_pretrained(path)
    m.eval().to(torch.device(device))
    return m


def crop_audio(wav: np.ndarray, piece: list[list[float]], sr: int = SR) -> np.ndarray:
    parts = []
    for s, e in piece:
        a, b = int(round(s * sr)), int(round(e * sr))
        a, b = max(0, a), min(len(wav), b)
        if b > a:
            parts.append(wav[a:b])
    if not parts:
        raise ValueError(f"empty crop {piece}")
    return np.concatenate(parts)


def l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n <= 0) or not np.all(np.isfinite(n)):
        raise ValueError("zero or non-finite embedding norm")
    return v / n


def combine(embs: list[np.ndarray], durs: list[float]) -> np.ndarray:
    """Duration-weighted mean of L2-normalised crop embeddings, re-normalised."""
    e = l2n(np.stack(embs).astype(np.float64))
    w = np.asarray(durs, np.float64)[:, None]
    return l2n((e * w).sum(0) / w.sum()).astype(np.float32)


def embed_tracks(tracks: list[dict], wav: np.ndarray, embed_fn, dim: int) -> tuple[np.ndarray, dict]:
    out = np.full((len(tracks), dim), np.nan, np.float32)
    n_crops = 0
    for i, t in enumerate(tracks):
        if t["track"] != i:
            raise ValueError(f"track ids not aligned with list order at {i}")
        if not t["crops"]:
            continue
        embs, durs = [], []
        for piece in t["crops"]:
            x = crop_audio(wav, piece)
            embs.append(np.asarray(embed_fn(x), np.float64))
            durs.append(len(x) / SR)
            n_crops += 1
        out[i] = combine(embs, durs)
    return out, {"n_tracks": len(tracks), "n_crops": n_crops,
                 "nan_rows": int(np.isnan(out).any(1).sum())}


def write_emb(cache: Path, tag: str, emb: np.ndarray, meta: dict) -> None:
    """npy first, meta (with tracks_sha256) last, both atomic: the meta hash is the commit mark."""
    mpath = cache / f"emb_{tag}_meta.json"
    if mpath.exists():
        mpath.unlink()
    tmp = cache / f"emb_{tag}.tmp.npy"
    np.save(tmp, emb)
    tmp.replace(cache / f"emb_{tag}.npy")
    mtmp = cache / f"emb_{tag}_meta.json.tmp"
    mtmp.write_text(json.dumps(meta, indent=1))
    mtmp.replace(mpath)


def make_embed_fn(model, device: str):
    import torch

    def fn(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None].to(device)
        with torch.inference_mode():
            return model(t)[0].float().cpu().numpy()
    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", type=Path, required=True, help="runs/cache/<system>")
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--theta-act", nargs="+", required=True, help="tags as in tracks_<tag>.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--embedder-ckpt", type=Path, default=None,
                    help="runs/p5/embedders/<tag>: FT weights (bnd.embed_ft); refuses its training recordings")
    a = ap.parse_args(argv)
    import soundfile as sf
    import torch

    if a.embedder_ckpt is None:
        model = load_embedder(a.device)
    else:
        from bnd import embed_ft
        embed_ft.assert_not_trained_on(a.embedder_ckpt, a.recordings)
        embed_ft.check_ft_cache_target(a.cache_dir, a.recordings, a.theta_act,
                                       embed_ft.read_manifest(a.embedder_ckpt)["sha256"])
        model = embed_ft.load_ft_embedder(a.embedder_ckpt, a.device, a.recordings)
    fn = make_embed_fn(model, a.device)
    dim = int(fn(np.zeros(SR * 2, np.float32) + 1e-3 * np.random.default_rng(0).standard_normal(SR * 2)
                 .astype(np.float32)).shape[0])
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    meta = {"model": MODEL_ID, "revision": MODEL_REV, "dim": dim, "sample_rate": SR,
            "crop_policy": CROP_POLICY, "torch": torch.__version__}
    if a.embedder_ckpt is not None:
        from bnd import embed_ft
        man = embed_ft.read_manifest(a.embedder_ckpt)
        meta["embedder_ckpt"] = {"dir": str(a.embedder_ckpt), "tag": man["tag"], "sha256": man["sha256"],
                                 "train_recordings": man["train_recordings"]}
    report = {"recordings": {}}
    t0 = time.perf_counter()
    for rid in a.recordings:
        cache = a.cache_dir / rid
        rmeta = json.loads((cache / "meta.json").read_text())
        wav, sr = sf.read(rmeta["audio_filepath"], dtype="float32")
        if sr != SR or wav.ndim != 1:
            raise ValueError(f"{rid}: expected 16 kHz mono")
        (cache / "emb_meta.json").write_text(json.dumps(meta, indent=1))
        report["recordings"][rid] = {}
        for tag in a.theta_act:
            raw = (cache / f"tracks_{tag}.json").read_bytes()
            tracks = json.loads(raw)
            emb, st = embed_tracks(tracks, wav, fn, dim)
            write_emb(cache, tag, emb, dict(meta, **st, theta_act=tag,
                                            tracks_sha256=hashlib.sha256(raw).hexdigest()))
            report["recordings"][rid][tag] = st
        print(f"[embed] {rid}: {json.dumps(report['recordings'][rid])}", flush=True)
    report["wall_s"] = time.perf_counter() - t0
    if a.device.startswith("cuda"):
        report["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["meta"] = meta
    if a.report:
        a.report.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
