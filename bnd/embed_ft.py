"""Fine-tuned WeSpeaker embedder (PREREG_phase5 "FT embedder recipe"; fixed, no search).

Runs in the SEPARATE pyannote venv (``third_party/embed-venv``), GPU through scripts/gpu_run.sh.

Recipe (verbatim from docs/PREREG_phase5.md):
* init: pinned WeSpeaker resnet34-LM (``bnd.embed.MODEL_ID`` @ ``bnd.embed.MODEL_REV``);
* loss: additive-angular-margin softmax (m 0.2, s 30) over (recording, speaker) classes of the
  fit's training recordings only; the head is discarded after training;
* crops: 2.0-4.0 s, strictly inside ONE single-speaker reference turn (ref_rttm), >= 0.5 s from any
  turn boundary (own and every other speaker's), no other reference speaker active, and wholly
  inside a trusted gap-5 span (data/prepared/trust/5/spans.json);
* balance: per (recording, speaker) class at most 200 crops per epoch;
* AdamW lr 1e-4 backbone / 1e-3 head, weight decay 1e-4, batch 64 crops, 1500 optimizer steps,
  cosine from lr to lr/10, bf16 autocast on CUDA (fbank in fp32), seed 42;
* saved: the embedding network only (same pyannote WeSpeakerResNet34 module, same fbank frontend,
  256-d, stats pooling) as a state_dict at runs/p5/embedders/<tag>/embedder.pt + manifest.json.

Implementation decisions (the PREREG does not pin them; fixed before any number):
* "turn" = union run of one speaker's ref_rttm rows (overlapping/touching rows of the same speaker
  merged); eligible region = turn interior shrunk by 0.5 s on both ends, minus every other
  speaker's turns dilated by 0.5 s, intersected with the trusted spans, kept if >= 2.0 s.
* class capacity = sum over its regions of floor(len / 2.0 s) (distinct minimum-length crops);
  crops per epoch = min(200, capacity). Classes with capacity 0 are not classes.
* the epoch plan (class ids, each repeated its per-epoch count) is shuffled and the epochs are
  concatenated into one stream cut into batches of 64 (an epoch boundary may fall inside a batch).
* one crop length per batch, L ~ U(2.0, 4.0) s clipped to the longest region of every class in the
  batch (>= 2.0 by construction), so a batch is a dense [64, 1, L*16000] tensor: no padding (the
  WeSpeaker fbank mean-normalisation and stats pooling would see padding) and BatchNorm sees 64
  real crops. Within a class, a region is chosen with probability proportional to its number of
  valid start positions (len - L) and the start is uniform.

An embedder is NEVER applied to a recording it was trained on: every consumer must call
:func:`assert_not_trained_on`; :func:`load_ft_embedder` requires the recordings it will be applied to and
calls it itself (``bnd.embed --embedder-ckpt`` does both). ``bnd.embed --embedder-ckpt`` also refuses to
write into a cache that holds stock or another embedder's embeddings (:func:`check_ft_cache_target`).

CLI
  fit : python -m bnd.embed_ft --tag <tag> --train-recs 013 005 ...
  diag: python -m bnd.embed_ft --mode diag --tag <tag> --diag-recs 011 009 [--theta-act 0.40]
        (stock vs FT same-/different-speaker cosine AUC over the cached zero-shot tracks of held-out
        recordings; inner diagnostic only.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from bnd import embed as E
from bnd.score import load_rttm_file

REPO = Path(__file__).resolve().parents[1]
REF_DIR = REPO / "data/prepared/ref_rttm"
SPANS_PATH = REPO / "data/prepared/trust/5/spans.json"
OUT_ROOT = REPO / "runs/p5/embedders"
LEDGER = REPO / "runs/p5/gpu_ledger.jsonl"
SR = E.SR

RECIPE = {
    "init": f"{E.MODEL_ID}@{E.MODEL_REV}",
    "loss": "AAM-softmax", "margin": 0.2, "scale": 30.0,
    "crop_min_s": 2.0, "crop_max_s": 4.0, "boundary_margin_s": 0.5,
    "cap_per_class_epoch": 200, "lr_backbone": 1e-4, "lr_head": 1e-3, "weight_decay": 1e-4,
    "batch": 64, "steps": 1500, "schedule": "cosine lr -> lr/10 over all steps, no warmup",
    "precision": "bf16-autocast (CUDA) for the ResNet, fbank fp32", "seed": 42,
    "trust_spans": "data/prepared/trust/5/spans.json",
}


class EmbedFTError(RuntimeError):
    """Contract violation (own-training-recording use, bad checkpoint, malformed inputs)."""


# ------------------------------------------------------------------------------------------
# interval algebra (sorted, disjoint lists of (start, end))
# ------------------------------------------------------------------------------------------
Iv = tuple[float, float]


def merge(iv: Sequence[Iv]) -> list[Iv]:
    """Union; touching intervals are merged."""
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract(a: Sequence[Iv], b: Sequence[Iv]) -> list[Iv]:
    b = merge(b)
    out = []
    for s, e in merge(a):
        cur = s
        for x, y in b:
            if y <= cur or x >= e:
                continue
            if x > cur:
                out.append((cur, x))
            cur = max(cur, y)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return out


def intersect(a: Sequence[Iv], b: Sequence[Iv]) -> list[Iv]:
    a, b = merge(a), merge(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


# ------------------------------------------------------------------------------------------
# eligible single-speaker regions and classes
# ------------------------------------------------------------------------------------------
def rec_id(x: str) -> str:
    """Bengali-Loop shorthand ("13", "013") -> "train_013"; every other id (``train_*``, Part-B hold-out
    stems such as ``1_speaker/simple_1spk_015``) is returned unchanged."""
    x = str(x)
    return f"train_{int(x):03d}" if x.isdigit() else x


def speaker_turns(ref) -> dict[str, list[Iv]]:
    by: dict[str, list[Iv]] = {}
    for s in ref:
        by.setdefault(s.speaker, []).append((s.start, s.end))
    return {k: merge(v) for k, v in by.items()}


def eligible_regions(ref, spans: Sequence[Iv], margin: float = RECIPE["boundary_margin_s"],
                     min_len: float = RECIPE["crop_min_s"]) -> dict[str, list[Iv]]:
    """speaker -> regions where ANY crop lies inside one of its turns, >= margin from every turn
    boundary of every speaker, with no other speaker active, inside a trusted span."""
    turns = speaker_turns(ref)
    out = {}
    for spk, tv in turns.items():
        interior = [(a + margin, b - margin) for a, b in tv if b - a > 2 * margin]
        others = [(a - margin, b + margin) for o, ov in turns.items() if o != spk for a, b in ov]
        reg = intersect(subtract(interior, others), spans)
        reg = [(a, b) for a, b in reg if b - a >= min_len - 1e-9]
        if reg:
            out[spk] = reg
    return out


def load_spans(rid: str, spans_path: Path = SPANS_PATH) -> list[Iv]:
    d = json.loads(Path(spans_path).read_text())
    if d.get("gap_s") != 5:
        raise EmbedFTError(f"{spans_path}: expected gap-5 spans, got gap_s={d.get('gap_s')}")
    if rid not in d["recordings"]:
        raise EmbedFTError(f"{rid}: no trusted spans")
    return [(float(s["start"]), float(s["end"])) for s in d["recordings"][rid]["spans"]]


@dataclass
class Cls:
    cls: int
    rid: str
    speaker: str
    regions: list[Iv]

    @property
    def capacity(self) -> int:
        return int(sum(math.floor((b - a) / RECIPE["crop_min_s"] + 1e-9) for a, b in self.regions))

    @property
    def max_len(self) -> float:
        return max(b - a for a, b in self.regions)


def build_classes(rids: Sequence[str], ref_dir: Path = REF_DIR, spans_path: Path = SPANS_PATH) -> list[Cls]:
    """(recording, speaker) classes of the given recordings only, ids in sorted (rid, speaker) order."""
    out = []
    for rid in sorted({rec_id(r) for r in rids}):
        ref = load_rttm_file(Path(ref_dir) / f"{rid}.rttm", rid)
        reg = eligible_regions(ref, load_spans(rid, spans_path))
        for spk in sorted(reg):
            c = Cls(len(out), rid, spk, reg[spk])
            if c.capacity > 0:
                out.append(c)
    if not out:
        raise EmbedFTError("no eligible classes")
    return out


# ------------------------------------------------------------------------------------------
# crop sampler
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Crop:
    cls: int
    rid: str
    start: float
    end: float


class CropSampler:
    def __init__(self, classes: Sequence[Cls], batch: int = RECIPE["batch"], cap: int = RECIPE["cap_per_class_epoch"],
                 seed: int = RECIPE["seed"], lmin: float = RECIPE["crop_min_s"], lmax: float = RECIPE["crop_max_s"]):
        self.classes, self.batch, self.cap, self.lmin, self.lmax = list(classes), batch, cap, lmin, lmax
        self.rng = np.random.default_rng(seed)
        self.per_epoch = {c.cls: min(cap, c.capacity) for c in self.classes}
        self._stream: list[int] = []
        self.epochs = 0
        self.drawn = {c.cls: 0 for c in self.classes}

    def epoch_plan(self) -> np.ndarray:
        plan = np.concatenate([np.full(self.per_epoch[c.cls], c.cls, np.int64) for c in self.classes])
        self.rng.shuffle(plan)
        return plan

    def _place(self, c: Cls, L: float) -> Crop:
        reg = [(a, b) for a, b in c.regions if b - a >= L - 1e-9]
        w = np.array([b - a - L for a, b in reg]) + 1e-6
        a, b = reg[int(self.rng.choice(len(reg), p=w / w.sum()))]
        s = a + self.rng.uniform(0.0, max(0.0, b - a - L))
        s = math.ceil(s * SR) / SR          # sample-aligned, still >= a
        e = s + L
        if e > b:                            # float guard: shift back inside the region
            s, e = b - L, b
        return Crop(c.cls, c.rid, s, e)

    def next_batch(self) -> list[Crop]:
        while len(self._stream) < self.batch:
            self._stream.extend(self.epoch_plan().tolist())
            self.epochs += 1
        ids, self._stream = self._stream[:self.batch], self._stream[self.batch:]
        L = min(self.rng.uniform(self.lmin, self.lmax), *(self.classes[i].max_len for i in ids))
        L = math.floor(L * SR) / SR
        if L < self.lmin - 1e-9:
            raise EmbedFTError(f"batch crop length {L} < {self.lmin}")
        for i in ids:
            self.drawn[i] += 1
        return [self._place(self.classes[i], L) for i in ids]


def check_crop(crop: Crop, ref, spans: Sequence[Iv], speaker: str, margin: float = RECIPE["boundary_margin_s"],
               lmin: float = RECIPE["crop_min_s"], lmax: float = RECIPE["crop_max_s"], tol: float = 1e-6) -> None:
    """Raise unless ``crop`` satisfies every recipe constraint (independent re-check, used by tests
    and by the fit on every drawn crop)."""
    L = crop.end - crop.start
    if not (lmin - tol <= L <= lmax + tol):
        raise EmbedFTError(f"crop length {L}")
    if not any(a - tol <= crop.start and crop.end <= b + tol for a, b in spans):
        raise EmbedFTError(f"crop {crop} not inside a trusted span")
    turns = speaker_turns(ref)
    own = [t for t in turns.get(speaker, []) if t[0] - tol <= crop.start and crop.end <= t[1] + tol]
    if len(own) != 1:
        raise EmbedFTError(f"crop {crop} not inside one turn of {speaker}")
    a, b = own[0]
    if crop.start - a < margin - tol or b - crop.end < margin - tol:
        raise EmbedFTError(f"crop {crop} within {margin}s of its turn boundary")
    for o, ov in turns.items():
        if o == speaker:
            continue
        for x, y in ov:
            if x < crop.end + margin - tol and y > crop.start - margin + tol:
                raise EmbedFTError(f"crop {crop} overlaps or is within {margin}s of speaker {o} turn {(x, y)}")


# ------------------------------------------------------------------------------------------
# never-embed-own-training guard + checkpoint I/O
# ------------------------------------------------------------------------------------------
def train_binding(manifest: dict) -> str:
    """sha256 binding the training-recording list (and tag) to the weights' sha256. Stored in the manifest
    by :func:`save_embedder`; a manifest whose ``train_recordings`` (or ``sha256``) was edited no longer
    matches it. (It is a consistency seal against accidental/naive edits, not a signature: whoever can
    rewrite the manifest can recompute it. The out-of-band record is the driver's fit spec/done-marker.)"""
    key = {"tag": manifest.get("tag"), "sha256": manifest.get("sha256"),
           "train_recordings": sorted(manifest.get("train_recordings", []))}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def read_manifest(ckpt_dir: Path) -> dict:
    p = Path(ckpt_dir) / "manifest.json"
    if not p.exists():
        raise EmbedFTError(f"{ckpt_dir}: no manifest.json")
    m = json.loads(p.read_text())
    if not isinstance(m.get("train_recordings"), list):
        raise EmbedFTError(f"{p}: train_recordings missing or not a list")
    if "train_binding" in m and m["train_binding"] != train_binding(m):
        raise EmbedFTError(f"{p}: train_recordings/sha256 do not match train_binding (manifest edited?)")
    return m


def assert_not_trained_on(ckpt_dir: Path, rids: Sequence[str]) -> None:
    """Raise if any recording in ``rids`` was a training recording of the embedder at ``ckpt_dir``.
    Every driver that applies an FT embedder MUST call this before embedding (:func:`load_ft_embedder`
    also calls it for the recordings it is told it will embed)."""
    train = set(read_manifest(ckpt_dir)["train_recordings"])
    bad = sorted(train & {rec_id(r) for r in rids})
    if bad:
        raise EmbedFTError(f"embedder {ckpt_dir} was trained on {bad}; refusing to embed them")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_embedder(model, out_dir: Path, manifest: dict) -> dict:
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = out_dir / "embedder.pt.tmp"
    torch.save(sd, tmp)
    tmp.replace(out_dir / "embedder.pt")
    manifest = dict(manifest, checkpoint="embedder.pt", sha256=sha256_file(out_dir / "embedder.pt"),
                    interface={"class": type(model).__module__ + "." + type(model).__name__,
                               "dim": int(model.dimension), "frontend": "pyannote WeSpeaker fbank (unchanged)",
                               "base": RECIPE["init"]})
    manifest["train_binding"] = train_binding(manifest)
    mtmp = out_dir / "manifest.json.tmp"
    mtmp.write_text(json.dumps(manifest, indent=1))
    mtmp.replace(out_dir / "manifest.json")
    return manifest


def load_ft_embedder(ckpt_dir: Path, device: str, apply_to: Sequence[str]):
    """Stock architecture + frontend (bnd.embed.load_embedder), FT weights (strict), sha-checked.
    ``apply_to`` = the recordings the caller will embed: refused if any was a training recording, and the
    manifest must carry a valid ``train_binding`` (written by :func:`save_embedder`)."""
    import torch
    ckpt_dir = Path(ckpt_dir)
    man = read_manifest(ckpt_dir)
    if "train_binding" not in man:
        raise EmbedFTError(f"{ckpt_dir}: manifest has no train_binding (pre-binding checkpoint); refusing to load")
    assert_not_trained_on(ckpt_dir, apply_to)
    p = ckpt_dir / man["checkpoint"]
    if sha256_file(p) != man["sha256"]:
        raise EmbedFTError(f"{p}: sha256 mismatch with manifest")
    m = E.load_embedder(device)
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
    m.eval().to(torch.device(device))
    return m


EMB_WRITE_ROOT = REPO / "runs/p5"


def check_ft_cache_target(cache_dir: Path, rids: Sequence[str], thetas: Sequence[str], ckpt_sha: str) -> None:
    """Refuse to write FT embeddings (``bnd.embed --embedder-ckpt``) where they could be mistaken for, or
    overwrite, embeddings of another embedder. The FT meta keeps model/revision = the stock pin (the
    architecture/frontend), so readers that check only model/revision/tracks_sha256 cannot tell; hence:

    * inside the repository only under ``runs/p5/`` (never a registered stock cache such as runs/cache,
      runs/cv, runs/p3b, runs/final, runs/diag);
    * no ``emb_*`` file may be a symlink (bnd.embed would write through it into the link target);
    * every existing embedding meta (``emb_meta.json``, ``emb_<θ>_meta.json``) must name THIS checkpoint's
      sha256 under ``embedder_ckpt`` (stock embeddings or another FT embedder's -> refuse);
    * existing ``emb_*.npy`` with no meta at all (unknown provenance) -> refuse."""
    repo, root = REPO.resolve(), Path(EMB_WRITE_ROOT).resolve()
    for rid in rids:
        d = Path(cache_dir) / rid
        rd = d.resolve()
        if rd.is_relative_to(repo) and not rd.is_relative_to(root):
            raise EmbedFTError(f"{d}: FT embeddings may only be written under {root} inside the repository")
        if not d.is_dir():
            continue
        names = {"emb_meta.json"} | {f"emb_{t}.npy" for t in thetas} | {f"emb_{t}_meta.json" for t in thetas}
        files = sorted({p for p in d.glob("emb_*") if not p.name.endswith((".tmp", ".tmp.npy"))}
                       | {d / n for n in names if (d / n).is_symlink()})
        links = [p.name for p in files if p.is_symlink()]
        if links:
            raise EmbedFTError(f"{d}: embedding files are symlinks {links}; refusing to write through them")
        metas = [p for p in files if p.name.endswith("meta.json")]
        for p in metas:
            try:
                ck = json.loads(p.read_text()).get("embedder_ckpt")
            except (OSError, ValueError) as e:
                raise EmbedFTError(f"{p}: unreadable embedding meta ({e}); refusing to overwrite") from e
            if not isinstance(ck, dict) or ck.get("sha256") != ckpt_sha:
                who = "stock embedder" if ck is None else f"FT embedder {ck.get('tag') if isinstance(ck, dict) else ck}"
                raise EmbedFTError(f"{p}: embeddings of the {who} (not checkpoint {ckpt_sha[:12]}); refusing to "
                                   f"overwrite them -- use a separate cache dir per embedder")
        if not metas and any(p.suffix == ".npy" for p in files):
            raise EmbedFTError(f"{d}: emb_*.npy without any embedding meta (unknown embedder); refusing to overwrite")


# ------------------------------------------------------------------------------------------
# AAM softmax + fit
# ------------------------------------------------------------------------------------------
def aam_logits(emb, weight, labels, margin: float, scale: float):
    """ArcFace logits: s*cos(theta + m) at the target class (with the standard monotonic fallback
    cos - m*sin(m) when theta + m > pi), s*cos(theta) elsewhere."""
    import torch
    import torch.nn.functional as F
    cos = F.linear(F.normalize(emb.float()), F.normalize(weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
    sin = torch.sqrt(1.0 - cos * cos)
    phi = cos * math.cos(margin) - sin * math.sin(margin)
    phi = torch.where(cos > math.cos(math.pi - margin), phi, cos - math.sin(math.pi - margin) * margin)
    onehot = F.one_hot(labels, cos.shape[1]).bool()
    return scale * torch.where(onehot, phi, cos)


def lr_factor(step: int, total: int) -> float:
    """Cosine from 1 to 0.1 over ``total`` steps (step 0 -> 1.0, step total -> 0.1)."""
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def fit(tag: str, train_rids: Sequence[str], device: str = "cuda", out_root: Path = OUT_ROOT,
        steps: int = RECIPE["steps"], log_every: int = 50) -> dict:
    import random
    import torch
    from bnd.infer import audio_path, read_audio

    rids = sorted({rec_id(r) for r in train_rids})
    out_dir = Path(out_root) / tag
    if (out_dir / "manifest.json").exists():
        raise EmbedFTError(f"{out_dir} already holds an embedder; refusing to overwrite")
    seed = RECIPE["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    classes = build_classes(rids)
    refs = {r: load_rttm_file(REF_DIR / f"{r}.rttm", r) for r in rids}
    spans = {r: load_spans(r) for r in rids}
    wavs = {r: read_audio(audio_path(r)) for r in rids}
    sampler = CropSampler(classes)

    model = E.load_embedder(device)
    model.train()
    dim = int(model.dimension)
    head = torch.nn.Parameter(torch.empty(len(classes), dim, device=device))
    torch.nn.init.xavier_uniform_(head)
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": RECIPE["lr_backbone"]},
                             {"params": [head], "lr": RECIPE["lr_head"]}], weight_decay=RECIPE["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    losses, accs, lens = [], [], []
    t0 = time.perf_counter()
    for step in range(steps):
        f = lr_factor(step, steps)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        crops = sampler.next_batch()
        for c in crops:
            check_crop(c, refs[c.rid], spans[c.rid], classes[c.cls].speaker)
        n = int(round((crops[0].end - crops[0].start) * SR))
        x = np.stack([wavs[c.rid][int(round(c.start * SR)):int(round(c.start * SR)) + n] for c in crops])
        if x.shape[1] != n:
            raise EmbedFTError("crop ran past the end of the audio")
        xt = torch.from_numpy(x)[:, None].to(device)
        y = torch.tensor([c.cls for c in crops], device=device)
        with torch.no_grad():
            fb = model.compute_fbank(xt)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            emb = model.resnet(fb)[1]
        logits = aam_logits(emb, head, y, RECIPE["margin"], RECIPE["scale"])
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        if not math.isfinite(lv):
            raise EmbedFTError(f"non-finite loss at step {step}")
        losses.append(lv)
        accs.append(float((logits.detach().argmax(1) == y).float().mean()))
        lens.append(n / SR)
        if step % log_every == 0 or step == steps - 1:
            print(f"[embed_ft] {tag} step {step} loss {lv:.4f} acc {accs[-1]:.3f} L {lens[-1]:.2f}s "
                  f"lr {opt.param_groups[0]['lr']:.2e} t {time.perf_counter() - t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    model.eval()
    manifest = {
        "tag": tag, "train_recordings": rids, "recipe": RECIPE, "steps": steps,
        "classes": [{"cls": c.cls, "rid": c.rid, "speaker": c.speaker, "n_regions": len(c.regions),
                     "region_s": round(sum(b - a for a, b in c.regions), 3), "capacity": c.capacity,
                     "per_epoch": sampler.per_epoch[c.cls], "crops_drawn": sampler.drawn[c.cls]}
                    for c in classes],
        "crop_counts": {"n_classes": len(classes), "per_epoch_total": int(sum(sampler.per_epoch.values())),
                        "crops_drawn_total": int(sum(sampler.drawn.values())), "epochs_started": sampler.epochs,
                        "per_recording_drawn": {r: int(sum(sampler.drawn[c.cls] for c in classes if c.rid == r))
                                                for r in rids}},
        "loss_curve": [round(v, 5) for v in losses], "acc_curve": [round(v, 4) for v in accs],
        "batch_len_s_mean": float(np.mean(lens)),
        "loss_first10_mean": float(np.mean(losses[:10])), "loss_last10_mean": float(np.mean(losses[-10:])),
        "wall_s": wall, "device": device, "torch": torch.__version__,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        "head": "discarded (not saved)",
    }
    return save_embedder(model, out_dir, manifest)


# ------------------------------------------------------------------------------------------
# inner diagnostic: same- vs different-speaker cosine AUC over cached tracks
# ------------------------------------------------------------------------------------------
def crop_intervals(track: dict) -> list[Iv]:
    return merge([(a, b) for piece in track["crops"] for a, b in piece])


def track_ref_labels(tracks: Sequence[dict], ref, purity: float = 0.8) -> list[str | None]:
    """Reference speaker owning >= ``purity`` of the track's embedded (crop) audio, else None."""
    turns = speaker_turns(ref)
    out = []
    for t in tracks:
        iv = crop_intervals(t)
        tot = sum(b - a for a, b in iv)
        if tot <= 0:
            out.append(None)
            continue
        ov = {s: sum(b - a for a, b in intersect(iv, tv)) for s, tv in turns.items()}
        best = max(sorted(ov), key=lambda s: ov[s]) if ov else None
        out.append(best if best is not None and ov[best] >= purity * tot else None)
    return out


def pair_auc(emb: np.ndarray, labels: Sequence[str | None]) -> dict:
    """AUC (Mann-Whitney, ties = 1/2) that a same-speaker pair has higher cosine than a
    different-speaker pair, over all pairs of labelled tracks with a finite embedding."""
    from scipy.stats import rankdata
    keep = [i for i, l in enumerate(labels) if l is not None and np.isfinite(emb[i]).all()]
    X = E.l2n(emb[keep].astype(np.float64))
    lab = np.array([labels[i] for i in keep])
    iu = np.triu_indices(len(keep), 1)
    cos = (X @ X.T)[iu]
    same = (lab[:, None] == lab[None, :])[iu]
    n1, n0 = int(same.sum()), int((~same).sum())
    if not n1 or not n0:
        raise EmbedFTError("AUC needs both same- and different-speaker pairs")
    r = rankdata(cos)
    auc = (r[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"auc": float(auc), "n_tracks": len(keep), "n_same": n1, "n_diff": n0,
            "mean_cos_same": float(cos[same].mean()), "mean_cos_diff": float(cos[~same].mean())}


def diag(tag: str, diag_rids: Sequence[str], theta_act: str = "0.40", cache_dir: Path = REPO / "runs/cache/zeroshot",
         device: str = "cuda", out_root: Path = OUT_ROOT) -> dict:
    import torch
    from bnd.infer import audio_path, read_audio
    ckpt = Path(out_root) / tag
    rids = [rec_id(r) for r in diag_rids]
    assert_not_trained_on(ckpt, rids)
    models = {"stock": E.load_embedder(device), "ft": load_ft_embedder(ckpt, device, rids)}
    fns = {k: E.make_embed_fn(m, device) for k, m in models.items()}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    rep = {"tag": tag, "theta_act": theta_act, "cache_dir": str(cache_dir), "purity": 0.8, "recordings": {}}
    t0 = time.perf_counter()
    for rid in rids:
        raw = (Path(cache_dir) / rid / f"tracks_{theta_act}.json").read_bytes()
        tracks = json.loads(raw)
        wav = read_audio(audio_path(rid))
        labels = track_ref_labels(tracks, load_rttm_file(REF_DIR / f"{rid}.rttm", rid))
        r = {"tracks_sha256": hashlib.sha256(raw).hexdigest(), "n_tracks": len(tracks),
             "n_labelled": sum(l is not None for l in labels)}
        embs = {}
        for k, fn in fns.items():
            embs[k], st = E.embed_tracks(tracks, wav, fn, 256)
            r[k] = dict(pair_auc(embs[k], labels), embed_stats=st)
        cached = Path(cache_dir) / rid / f"emb_{theta_act}.npy"
        if cached.exists():
            c = np.load(cached)
            ok = np.isfinite(c).all(1) & np.isfinite(embs["stock"]).all(1)
            r["stock_vs_cached_min_cos"] = float((c[ok] * embs["stock"][ok]).sum(1).min())
        rep["recordings"][rid] = r
        print(f"[embed_ft diag] {rid}: stock AUC {r['stock']['auc']:.4f} ft AUC {r['ft']['auc']:.4f}", flush=True)
    rep["wall_s"] = time.perf_counter() - t0
    if device.startswith("cuda"):
        rep["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    out = ckpt / f"diag_{'_'.join(rids)}_{theta_act}.json"
    out.write_text(json.dumps(rep, indent=1))
    return rep


def ledger(kind: str, wall: float, **kw) -> None:
    import os
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(dict(kind=kind, gpu_wall_s=wall, card=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                note="in-process wall (excl. interpreter start)", **kw)) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("fit", "diag"), default="fit")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-recs", nargs="+")
    ap.add_argument("--diag-recs", nargs="+")
    ap.add_argument("--theta-act", default="0.40")
    ap.add_argument("--cache-dir", type=Path, default=REPO / "runs/cache/zeroshot")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=RECIPE["steps"], help="recipe: 1500 (do not change for registered fits)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    if a.mode == "fit":
        if not a.train_recs:
            ap.error("--train-recs required")
        m = fit(a.tag, a.train_recs, a.device, steps=a.steps)
        print(json.dumps({k: m[k] for k in ("tag", "train_recordings", "loss_first10_mean", "loss_last10_mean",
                                            "wall_s", "peak_alloc_gib", "sha256", "crop_counts")}, indent=1))
        if a.device.startswith("cuda"):
            ledger("embed_ft-fit", time.perf_counter() - t0, tag=a.tag)
    else:
        if not a.diag_recs:
            ap.error("--diag-recs required")
        r = diag(a.tag, a.diag_recs, a.theta_act, a.cache_dir, a.device)
        print(json.dumps(r, indent=1))
        if a.device.startswith("cuda"):
            ledger("embed_ft-diag", time.perf_counter() - t0, tag=a.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
