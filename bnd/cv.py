"""Resumable, idempotent driver for the PREREG phase-2 nested cross-validation (docs/PREREG_phase2.md,
including Deviations D1-D7, which override the body).

Subcommands (each safe to re-run; finished work is skipped when its done-marker's input hashes still
match; a marker whose inputs changed is STALE and GPU steps refuse to redo it unless --allow-stale):

  plan    enumerate outer folds F1..F5, the 40 DEDUPLICATED inner fits (a 6-recording training set
          {all} - Fi - Fj serves outer Fi / inner Fj AND outer Fj / inner Fi: C(5,2)=10 sets x 4
          recipes) and the per-outer refits; write runs/cv/plan.json.
  inner   zero-shot inference from scratch under runs/cv/cache/zeroshot (D2: the phase-2a cache
          runs/cache/zeroshot is never reused);
          every inner fit: bnd.train -> bnd.infer on its 4 held-out recordings at theta_act
          {0.40,0.50,0.60} (both via scripts/gpu_run.sh) -> delete the fit's .nemo (inner fits only,
          training log / loss CSV / fit_report kept) -> bnd.link for all 15 D1 cells
          (runs/cv/hyp/<system>/<cell>/).
  select  per outer Fi: pooled inner DER at collar 0 over symlinked hyp dirs
          (runs/cv/select/<Fi>/<system>/<cell>/): zero-shot over the 8 inner recordings; recipe r over
          the predictions of the fits {Fi,Fj} (j != i) on Fj, i.e. fits that never saw Fi. D3
          tie-breaks. Writes runs/cv/selection.json with all 15-cell tables.
  outer   refits cv_refit_<Fi>_<recipe> (selected recipe, plus the strict recipe lr 1e-5 raw when
          different) on the 8 non-Fi recordings, outer inference on Fi, links at the selected cells
          and at the strict cell (0.50, 0.65), oracle diagnostics (a) link / (b) speech for the selected
          systems. Outputs runs/cv/outer/<system>/<Fi>/<rid>.rttm + runs/cv/outer/index.json.
  freeze  refuses unless every outer prediction of every system exists (and matches its link
          marker); writes runs/cv/frozen.json with the sha256 of every outer RTTM.
  score   OUTER LOCK: refuses unless frozen.json exists and every hash still matches (and, if outer
          DER was computed before, unless the new freeze records a --deviation). Per-recording and
          corpus DER at collars 0/0.25/0.5, paired bootstrap, the six-condition adoption gate, strict
          comparison, oracle diagnostics -> runs/cv/results.json + runs/cv/results.md.
  ledger  print the GPU ledger (runs/cv/gpu_ledger.jsonl) and budget balance.
  status  print progress.

The outer lock is enforced at the single scoring entry point (:func:`score_paths`): a recording of
outer fold F may be scored in the context of F only after freeze, and only from the exact frozen file;
independently of the context, a file under runs/cv/outer/ or from a cv_refit_* system may be scored
only after freeze and only if it is the exact frozen file. Selection for outer Fi runs in context Fi,
so it can never score an Fi recording. Once any outer DER exists (runs/cv/scored.jsonl), select /
outer / freeze / inner refuse to run again without --deviation "<reason>", and every scoring keeps the
previous results under runs/cv/results_history/ (PREREG: log the deviation, report the pre-deviation
result too).

Every GPU command goes through scripts/gpu_run.sh and is timed into the ledger (card parsed from
gpu_run's ``[gpu_run] card=N`` stderr line; GPU time is counted from that line, i.e. lock/card waits
are excluded). A job is refused before launch if ledger + estimate would exceed 12 GPU-h (phase-2a
usage 0.27 h is the starting balance).

Run on CPU:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 <venv python> -m bnd.cv <cmd>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from bnd import score as S
from bnd.labels import RECORDINGS

REPO = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------------------------
# registered constants (PREREG + D1/D3)
# ----------------------------------------------------------------------------------------------
FOLDS: dict[str, tuple[str, str]] = {
    "F1": ("train_001", "train_008"),
    "F2": ("train_011", "train_009"),
    "F3": ("train_013", "train_005"),
    "F4": ("train_004", "train_002"),
    "F5": ("train_003", "train_006"),
}
THETA_ACT = (0.40, 0.50, 0.60)
THETA_CLU = (0.45, 0.55, 0.65, 0.75, 0.85)  # D1
GRID_CENTRE = (0.50, 0.65)                  # D3 tie-break anchor
STRICT_CELL = (0.50, 0.65)                  # D3 strict fixed cell
LRS = {"1e-5": 1e-5, "5e-5": 5e-5}
TIMINGS = ("raw", "shift")
RECIPES = tuple(f"lr{l}_{t}" for l in LRS for t in TIMINGS)
STRICT_RECIPE = "lr1e-5_raw"
TIE_TOL = 1e-10          # DER as a fraction; 1e-10 == 1e-8 pp (D3)
BUDGET_GPU_H = 12.0
PHASE2A_GPU_H = 0.27
GATE_MIN_IMPROVEMENT_PP = 2.0
GATE_MIN_WINS = 7
GATE_MAX_WORSE_PP = 5.0
PRIMARY_COLLAR = S.PRIMARY_COLLAR
COLLARS = S.COLLARS
# default GPU cost estimates (phase-2a measurements, docs/phase2a_smoke_report.md) with headroom
EST_FIT_S = 150.0
EST_INFER_FIXED_S = 30.0
EST_INFER_PER_AUDIO_S = 0.0123 * 1.3

OUTER_SYSTEMS = ("zs_sel", "ft_sel", "zs_strict", "ft_strict",
                 "zs_sel_oracle_link", "zs_sel_oracle_speech",
                 "ft_sel_oracle_link", "ft_sel_oracle_speech")

assert sorted(r for f in FOLDS.values() for r in f) == sorted(RECORDINGS), "folds must partition the corpus"


class CVError(RuntimeError):
    pass


class OuterLockError(CVError):
    pass


class BudgetError(CVError):
    pass


class StaleError(CVError):
    pass


# ----------------------------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    repo: Path = REPO
    cv_dir: Path = REPO / "runs/cv"
    python: Path = REPO / "third_party/nemo-speech/.venv/bin/python"
    gpu_run: Path = REPO / "scripts/gpu_run.sh"
    ref_dir: Path = REPO / "data/prepared/ref_rttm"
    audio_dir: Path = REPO / "data/raw/diarization/diarization/train/audio"
    base_model: Path = REPO / "data/weights/nemotron3/Nemotron-3-Diarization.nemo"
    fits_dir: Path = REPO / "runs/fits"          # fixed by bnd.train
    models_dir: Path = REPO / "runs/models"      # fixed by bnd.train
    budget_h: float = BUDGET_GPU_H
    start_h: float = PHASE2A_GPU_H
    bootstrap_n: int = S.BOOTSTRAP_N
    dry_run: bool = False
    allow_stale: bool = False
    deviation: str | None = None
    jobs: int = 4
    out: Callable[[str], None] = print
    _sha_memo: dict = field(default_factory=dict)

    @property
    def cache_root(self) -> Path:
        return self.cv_dir / "cache"

    @property
    def hyp_root(self) -> Path:
        return self.cv_dir / "hyp"

    @property
    def select_root(self) -> Path:
        return self.cv_dir / "select"

    @property
    def outer_root(self) -> Path:
        return self.cv_dir / "outer"

    @property
    def logs(self) -> Path:
        return self.cv_dir / "logs"

    @property
    def ledger_path(self) -> Path:
        return self.cv_dir / "gpu_ledger.jsonl"

    @property
    def frozen_path(self) -> Path:
        return self.cv_dir / "frozen.json"

    @property
    def selection_path(self) -> Path:
        return self.cv_dir / "selection.json"

    @property
    def scored_path(self) -> Path:
        return self.cv_dir / "scored.jsonl"

    @property
    def history_root(self) -> Path:
        return self.cv_dir / "results_history"


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def cell_key(a: float, c: float) -> str:
    return f"{a:.2f}_{c:.2f}"


def parse_cell(k: str) -> tuple[float, float]:
    a, c = k.split("_")
    return float(a), float(c)


GRID = [(a, c) for a in THETA_ACT for c in THETA_CLU]


def fold_of(rid: str) -> str:
    for f, rs in FOLDS.items():
        if rid in rs:
            return f
    raise CVError(f"{rid} is in no fold")


def recipe_parts(recipe: str) -> tuple[str, str]:
    m = re.fullmatch(r"lr(1e-5|5e-5)_(raw|shift)", recipe)
    if not m:
        raise CVError(f"unknown recipe {recipe!r}")
    return m.group(1), m.group(2)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(ctx: Ctx, path: Path, missing_ok: bool = False) -> str:
    path = Path(path)
    try:
        st = path.stat()
    except FileNotFoundError:
        if missing_ok:
            return "MISSING"
        raise CVError(f"missing input {path}")
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in ctx._sha_memo:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        ctx._sha_memo[key] = h.hexdigest()
    return ctx._sha_memo[key]


def canon_sha(obj) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str))
    tmp.replace(path)


def code_shas(ctx: Ctx, *mods: str) -> dict:
    return {m: file_sha(ctx, ctx.repo / "bnd" / f"{m}.py") for m in mods}


def move_aside(p: Path) -> Path | None:
    """Rename (never delete) ``p`` to ``p.stale_<timestamp>``."""
    if not (p.exists() or p.is_symlink()):
        return None
    dst = p.with_name(p.name + f".stale_{time.strftime('%Y%m%d%H%M%S')}")
    p.rename(dst)
    return dst


# ----------------------------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------------------------
def fold_recs(folds: Iterable[str]) -> list[str]:
    return sorted(r for f in folds for r in FOLDS[f])


def set_id(excluded: Sequence[str]) -> str:
    return "x" + "".join(sorted(f[1:] for f in excluded))


def inner_fits() -> list[dict]:
    """The deduplicated inner fits: one per (unordered excluded fold pair, recipe)."""
    out = []
    for pair in itertools.combinations(sorted(FOLDS), 2):
        train = sorted(r for f in FOLDS if f not in pair for r in FOLDS[f])
        sid = set_id(pair)
        for recipe in RECIPES:
            lr, timing = recipe_parts(recipe)
            tag = f"cv_inner_{sid}_{recipe}"
            out.append({"set": sid, "excluded_folds": list(pair), "train_recs": train,
                        "heldout": {f: list(FOLDS[f]) for f in pair},
                        "heldout_recs": fold_recs(pair), "recipe": recipe, "lr": lr, "timing": timing,
                        "tag": tag, "system": tag, "kind": "inner"})
    return out


def fit_for(excluded: Iterable[str], recipe: str) -> dict:
    ex = sorted(excluded)
    hits = [f for f in inner_fits() if f["excluded_folds"] == ex and f["recipe"] == recipe]
    if len(hits) != 1:
        raise CVError(f"no unique inner fit for {ex} {recipe}")
    return hits[0]


def inner_sources(outer: str, recipe: str) -> list[tuple[dict, str, list[str]]]:
    """(fit, inner fold Fj, recordings of Fj) whose predictions form recipe r's pooled inner set for
    outer ``outer``. Every such fit excludes ``outer`` (and Fj) from training."""
    out = []
    for fj in sorted(FOLDS):
        if fj == outer:
            continue
        fit = fit_for((outer, fj), recipe)
        recs = list(FOLDS[fj])
        if set(fit["train_recs"]) & (set(recs) | set(FOLDS[outer])):
            raise CVError(f"leak: {fit['tag']} trains on {set(fit['train_recs']) & (set(recs) | set(FOLDS[outer]))}")
        out.append((fit, fj, recs))
    return out


def refit_spec(outer: str, recipe: str) -> dict:
    lr, timing = recipe_parts(recipe)
    train = sorted(r for f in FOLDS if f != outer for r in FOLDS[f])
    tag = f"cv_refit_{outer}_{recipe}"
    return {"outer": outer, "train_recs": train, "heldout_recs": list(FOLDS[outer]), "recipe": recipe,
            "lr": lr, "timing": timing, "tag": tag, "system": tag, "kind": "refit"}


def durations(ctx: Ctx, rids: Iterable[str]) -> dict[str, float]:
    return {r: S.audio_duration(ctx.audio_dir / f"{r}.wav") for r in rids}


def build_plan(ctx: Ctx) -> dict:
    fits = inner_fits()
    durs = durations(ctx, RECORDINGS)
    for f in fits:
        f["heldout_audio_s"] = sum(durs[r] for r in f["heldout_recs"])
    outer = {}
    for fi in sorted(FOLDS):
        outer[fi] = {"test_recs": list(FOLDS[fi]),
                     "inner_recs": sorted(r for f in FOLDS if f != fi for r in FOLDS[f]),
                     "inner_fits": {r: [src[0]["tag"] for src in inner_sources(fi, r)] for r in RECIPES},
                     "refit_train_recs": refit_spec(fi, STRICT_RECIPE)["train_recs"],
                     "refit": "selected recipe (from select), plus the strict recipe lr1e-5_raw if different"}
    n_refits_max = 2 * len(FOLDS)
    audio_heldout = sum(f["heldout_audio_s"] for f in fits)
    proj_s = (len(fits) + n_refits_max) * EST_FIT_S + len(fits) * EST_INFER_FIXED_S \
        + audio_heldout * EST_INFER_PER_AUDIO_S \
        + n_refits_max * EST_INFER_FIXED_S + 2 * sum(durs.values()) * EST_INFER_PER_AUDIO_S \
        + EST_INFER_FIXED_S + sum(durs.values()) * EST_INFER_PER_AUDIO_S  # D2: zero-shot from scratch
    return {
        "created": now(), "prereg": "docs/PREREG_phase2.md (+ deviations D1-D7)",
        "folds": {k: list(v) for k, v in FOLDS.items()},
        "grid": {"theta_act": list(THETA_ACT), "theta_clu": list(THETA_CLU), "n_cells": len(GRID)},
        "recipes": list(RECIPES), "strict_recipe": STRICT_RECIPE, "strict_cell": list(STRICT_CELL),
        "n_inner_fits": len(fits), "inner_fits": fits, "outer": outer,
        "n_refits_max": n_refits_max, "recording_duration_s": durs,
        "budget": {"budget_gpu_h": ctx.budget_h, "start_gpu_h": ctx.start_h,
                   "projected_gpu_h_upper": round(proj_s / 3600, 3),
                   "estimate_basis": {"fit_s": EST_FIT_S, "infer_fixed_s": EST_INFER_FIXED_S,
                                      "infer_per_audio_s": EST_INFER_PER_AUDIO_S}},
    }


# ----------------------------------------------------------------------------------------------
# done markers
# ----------------------------------------------------------------------------------------------
def marker_path(ctx: Ctx, kind: str, name: str) -> Path:
    return ctx.cv_dir / "done" / kind / f"{name}.json"


def read_marker(ctx: Ctx, kind: str, name: str) -> dict | None:
    p = marker_path(ctx, kind, name)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def marker_status(ctx: Ctx, kind: str, name: str, inputs: dict) -> tuple[str, dict | None]:
    m = read_marker(ctx, kind, name)
    if m is None:
        return "missing", None
    if m.get("inputs_sha") != canon_sha(inputs):
        return "stale", m
    return "done", m


def write_marker(ctx: Ctx, kind: str, name: str, inputs: dict, outputs: dict) -> dict:
    m = {"kind": kind, "name": name, "completed": now(), "inputs": inputs, "inputs_sha": canon_sha(inputs),
         "outputs": outputs}
    write_json(marker_path(ctx, kind, name), m)
    return m


def dir_marker_name(ctx: Ctx, out_dir: Path) -> str:
    return str(Path(out_dir).resolve().relative_to(ctx.cv_dir.resolve())).replace("/", "__")


# ----------------------------------------------------------------------------------------------
# ledger
# ----------------------------------------------------------------------------------------------
CARD_RE = re.compile(r"\[gpu_run\] card=(\S+)")


class Ledger:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.path = ctx.ledger_path

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def used_h(self) -> float:
        return self.ctx.start_h + sum(e.get("gpu_wall_s", 0.0) for e in self.entries()) / 3600

    def estimate_s(self, kind: str, audio_s: float = 0.0) -> float:
        prev = [e for e in self.entries() if e.get("kind") == kind and e.get("returncode") == 0
                and e.get("gpu_wall_s")]
        if kind == "train":
            return max([EST_FIT_S] + [e["gpu_wall_s"] for e in prev])
        rate = max([EST_INFER_PER_AUDIO_S] + [(e["gpu_wall_s"] - EST_INFER_FIXED_S) / e["audio_s"]
                                             for e in prev if e.get("audio_s")])
        return EST_INFER_FIXED_S + rate * audio_s

    def check(self, estimate_s: float, what: str = "") -> None:
        used = self.used_h()
        if used + estimate_s / 3600 > self.ctx.budget_h:
            raise BudgetError(f"GPU budget: used {used:.3f} h + estimate {estimate_s / 3600:.3f} h for {what} "
                              f"> {self.ctx.budget_h} h; aborting before launch")

    def run(self, cmd: list[str], kind: str, tag: str, log: Path, audio_s: float = 0.0) -> dict:
        est = self.estimate_s(kind, audio_s)
        self.check(est, f"{kind} {tag}")
        full = [str(self.ctx.gpu_run)] + [str(c) for c in cmd]
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        card, t_card = None, None
        with open(log, "a") as lf:
            lf.write(f"\n===== {now()} {shlex.join(full)}\n")
            lf.flush()
            p = subprocess.Popen(full, cwd=str(self.ctx.repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            assert p.stdout is not None
            for line in p.stdout:
                lf.write(line)
                if card is None:
                    m = CARD_RE.search(line)
                    if m:
                        card, t_card = m.group(1), time.time()
            rc = p.wait()
        t1 = time.time()
        e = {"ts_start": _dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
             "ts_end": _dt.datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
             "kind": kind, "tag": tag, "cmd": shlex.join(full), "card": card, "returncode": rc,
             "wall_s": round(t1 - t0, 3), "gpu_wall_s": round(t1 - t_card, 3) if t_card else 0.0,
             "audio_s": audio_s, "estimate_s": round(est, 1), "log": str(log)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(e) + "\n")
        self.ctx.out(f"[cv] gpu {kind} {tag}: rc={rc} card={card} gpu_wall={e['gpu_wall_s']:.0f}s "
                     f"ledger={self.used_h():.3f}/{self.ctx.budget_h} h")
        if rc != 0:
            raise CVError(f"GPU command failed (rc {rc}): {shlex.join(full)}; log {log}")
        if self.used_h() > self.ctx.budget_h:
            raise BudgetError(f"GPU budget exceeded after {tag}: {self.used_h():.3f} h")
        return e


def run_cpu(ctx: Ctx, cmd: list[str], log: Path) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    full = ["nice", "-n", "10"] + [str(c) for c in cmd]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as lf:
        lf.write(f"\n===== {now()} {shlex.join(full)}\n")
        lf.flush()
        rc = subprocess.run(full, cwd=str(ctx.repo), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise CVError(f"CPU command failed (rc {rc}): {shlex.join(full)}; log {log}")


def say_dry(ctx: Ctx, where: str, cmd: list[str]) -> None:
    pre = [str(ctx.gpu_run)] if where == "gpu" else ["nice", "-n", "10"]
    ctx.out(f"DRY-RUN [{where}] {shlex.join(pre + [str(c) for c in cmd])}")


# ----------------------------------------------------------------------------------------------
# zero-shot provenance
# ----------------------------------------------------------------------------------------------
def cache_current(ctx: Ctx, root: Path, system: str, rids: Iterable[str], model_sha: str,
                  thetas: Iterable[float], commit: str) -> tuple[bool, list[str]]:
    from bnd import embed, infer

    bad = []
    for rid in rids:
        d = root / system / rid
        if not infer.cache_complete(d, {"system": system, "model_sha256": model_sha, "nemo_commit": commit}):
            bad.append(f"{rid}: probs/meta missing or provenance mismatch")
            continue
        for t in thetas:
            if not infer.emb_is_current(d, t):
                bad.append(f"{rid}@{t:.2f}: embeddings missing/stale")
                continue
            try:
                em = json.loads(infer.emb_paths(d, t)[1].read_text())
            except (OSError, ValueError):
                bad.append(f"{rid}@{t:.2f}: no emb meta")
                continue
            if em.get("revision") != embed.MODEL_REV or em.get("model") != embed.MODEL_ID:
                bad.append(f"{rid}@{t:.2f}: embedder {em.get('model')}@{em.get('revision')}")
    return not bad, bad


def zeroshot_root(ctx: Ctx, ledger: Ledger | None) -> Path:
    """Zero-shot probs/tracks/embeddings for all recordings under runs/cv/cache/zeroshot.

    D2: "the registered protocol re-runs everything from scratch", so the phase-2a cache
    (runs/cache/zeroshot) is never reused, however well its provenance matches. The infer done-marker
    hashes bnd/infer.py + bnd/embed.py, so a later code change makes it stale (redo needs --allow-stale)."""
    from bnd import infer

    model_sha = file_sha(ctx, ctx.base_model)
    spec = {"system": "zeroshot", "tag": "zeroshot", "kind": "zeroshot"}
    done = ensure_infer(ctx, ledger, spec, ctx.base_model, model_sha, list(RECORDINGS), THETA_ACT)
    if done and not ctx.dry_run:
        write_json(ctx.cv_dir / "zeroshot_provenance.json",
                   {"checked": now(), "pinned_model": str(ctx.base_model), "model_sha256": model_sha,
                    "nemo_commit": infer.nemo_commit(), "root": str(ctx.cache_root), "reused_phase2a": False,
                    "code": code_shas(ctx, "infer", "embed")})
    return ctx.cache_root


# ----------------------------------------------------------------------------------------------
# GPU steps
# ----------------------------------------------------------------------------------------------
def fit_inputs(ctx: Ctx, spec: dict) -> dict:
    from bnd import nemo_data as nd
    from bnd import timing as tm
    from bnd import train as tr

    mk = ctx.dry_run
    ins = {"tag": spec["tag"], "train_recs": spec["train_recs"], "lr": spec["lr"], "timing": spec["timing"],
           "base_model_sha256": file_sha(ctx, ctx.base_model, mk),
           "manifest_sha256": file_sha(ctx, tr.MANIFEST, mk),
           "train_rttm": {r: file_sha(ctx, nd.TRAIN_RTTM_DIR / f"{r}.rttm", mk) for r in spec["train_recs"]},
           "code": code_shas(ctx, "train", "timing", "nemo_data", "trust")}
    if spec["timing"] == "shift":
        ins["detectors"] = {r: [file_sha(ctx, tm.PROBS_DIR / f"{r}.npz", mk),
                                file_sha(ctx, tm.SILERO_DIR / f"{r}.npz", mk)] for r in spec["train_recs"]}
    return ins


def ensure_fit(ctx: Ctx, ledger: Ledger, spec: dict) -> dict | None:
    tag = spec["tag"]
    if set(spec["train_recs"]) & set(spec["heldout_recs"]):
        raise CVError(f"{tag}: training set overlaps held-out recordings")
    inputs = fit_inputs(ctx, spec)
    st, m = marker_status(ctx, "fit", tag, inputs)
    if st == "done":
        return m["outputs"]
    model = ctx.models_dir / f"{tag}.nemo"
    fit_dir = ctx.fits_dir / tag
    report = fit_dir / "fit_report.json"
    if st == "stale":
        if not ctx.allow_stale:
            raise StaleError(f"fit {tag}: done-marker inputs changed (code/data/model); rerun with --allow-stale "
                             f"to move the old fit aside and refit")
        if not ctx.dry_run:
            for p in (model, fit_dir, marker_path(ctx, "fit", tag)):
                move_aside(p)
    elif model.exists():
        r = json.loads(report.read_text()) if report.exists() else {}
        if (r.get("status") == "ok" and r.get("train_recs") == spec["train_recs"]
                and r.get("timing") == spec["timing"] and math.isclose(r.get("lr", -1), LRS[spec["lr"]])):
            ctx.out(f"[cv] adopting finished fit {tag} (model + ok fit_report, marker was missing)")
            if not ctx.dry_run:
                return write_marker(ctx, "fit", tag, inputs, fit_outputs(ctx, spec))["outputs"]
            return None
        raise CVError(f"{model} exists without a valid done-marker (fit_report status {r.get('status')!r}); "
                      f"inspect it and move it aside by hand")
    cmd = [ctx.python, "-m", "bnd.train", "--tag", tag, "--train-recs", ",".join(spec["train_recs"]),
           "--lr", spec["lr"], "--timing", spec["timing"]]
    if ctx.dry_run:
        say_dry(ctx, "gpu", cmd)
        return None
    ledger.run(cmd, "train", tag, ctx.logs / f"train_{tag}.log")
    out = fit_outputs(ctx, spec)
    return write_marker(ctx, "fit", tag, inputs, out)["outputs"]


def fit_outputs(ctx: Ctx, spec: dict) -> dict:
    tag = spec["tag"]
    model = ctx.models_dir / f"{tag}.nemo"
    rep = json.loads((ctx.fits_dir / tag / "fit_report.json").read_text())
    if rep.get("status") != "ok" or not model.exists():
        raise CVError(f"fit {tag} did not finish ok: status {rep.get('status')!r}, model exists {model.exists()}")
    if rep.get("train_recs") != spec["train_recs"]:
        raise CVError(f"fit {tag}: report train_recs {rep.get('train_recs')} != {spec['train_recs']}")
    return {"model": str(model), "model_sha256": file_sha(ctx, model), "fit_report": str(ctx.fits_dir / tag / "fit_report.json"),
            "loss_csv": str(ctx.fits_dir / tag / "loss.csv"), "total_wall_s": rep.get("total_wall_s"),
            "delta_s": (rep.get("data", {}).get("delta") or {}).get("delta_s")}


def infer_inputs(ctx: Ctx, spec: dict, model_sha: str, rids: Sequence[str], thetas: Iterable[float]) -> dict:
    return {"system": spec["system"], "model_sha256": model_sha, "recordings": list(rids),
            "theta_act": [f"{t:.2f}" for t in thetas], "code": code_shas(ctx, "infer", "embed")}


def infer_name(system: str, rids: Sequence[str], thetas: Iterable[float]) -> str:
    return f"{system}__{'-'.join(rids)}__{'-'.join(f'{t:.2f}' for t in thetas)}"


def refit_if_infer_stale(ctx: Ctx, spec: dict, rids: Sequence[str], thetas: Iterable[float]) -> bool:
    """--allow-stale for a finished fit whose .nemo is gone (inner fits delete it after inference) but
    whose inference marker is STALE (e.g. bnd/infer.py or bnd/embed.py changed): the model is needed to
    redo inference, so move the fit's done-marker and fit dir aside; ensure_fit then refits from
    scratch. Returns True when a refit was arranged (or would be, in dry-run)."""
    if not ctx.allow_stale:
        return False
    fo = read_marker(ctx, "fit", spec["tag"])
    model = ctx.models_dir / f"{spec['tag']}.nemo"
    if fo is None or model.exists():
        return False
    thetas = tuple(thetas)
    st, _ = marker_status(ctx, "infer", infer_name(spec["system"], rids, thetas),
                          infer_inputs(ctx, spec, fo["outputs"]["model_sha256"], rids, thetas))
    if st != "stale":
        return False
    if ctx.dry_run:
        ctx.out(f"DRY-RUN --allow-stale: infer marker of {spec['tag']} is stale and its model was deleted; "
                f"would move its fit marker + fit dir aside and refit")
        return True
    for p in (marker_path(ctx, "fit", spec["tag"]), ctx.fits_dir / spec["tag"]):
        dst = move_aside(p)
        if dst:
            ctx.out(f"[cv] --allow-stale: moved {p} -> {dst} (stale inference, model deleted: refit)")
    return True


def ensure_infer(ctx: Ctx, ledger: Ledger | None, spec: dict, model: Path, model_sha: str | None,
                 rids: Sequence[str], thetas: Iterable[float]) -> bool:
    """Probs + tracks + embeddings for ``rids`` under ctx.cache_root/<system>. Returns True when done."""
    from bnd import infer

    system = spec["system"]
    thetas = tuple(thetas)
    bad_rec = set(rids) & set(spec.get("train_recs", ()))
    if bad_rec:
        raise CVError(f"{system}: inferring on its own training recordings {sorted(bad_rec)}")
    name = infer_name(system, rids, thetas)
    force = False
    if model_sha is not None:
        inputs = infer_inputs(ctx, spec, model_sha, rids, thetas)
        st, m = marker_status(ctx, "infer", name, inputs)
        if st == "done":
            ok, bad = cache_current(ctx, ctx.cache_root, system, rids, model_sha, thetas, infer.nemo_commit())
            if ok:
                return True
            ctx.out(f"[cv] infer {name}: marker present but cache not current ({bad[:2]}); redoing")
        elif st == "stale":
            if not ctx.allow_stale:
                raise StaleError(f"infer {name}: done-marker inputs changed; rerun with --allow-stale")
            force = True  # recompute probs too: the code that produced them changed
        elif st == "missing" and not ctx.dry_run:
            ok, _ = cache_current(ctx, ctx.cache_root, system, rids, model_sha, thetas, infer.nemo_commit())
            if ok:  # finished earlier, marker lost: adopt (provenance verified by model sha + emb hashes)
                write_marker(ctx, "infer", name, inputs, {"cache_root": str(ctx.cache_root),
                                                         "recordings": list(rids), "adopted": True})
                return True
    cmd = [ctx.python, "-m", "bnd.infer", "--system", system, "--model", model, "--recordings", *rids,
           "--theta-act", *[f"{t:.2f}" for t in thetas], "--cache-root", ctx.cache_root]
    if force:
        cmd.append("--force")
    if ctx.dry_run or model_sha is None:
        say_dry(ctx, "gpu", cmd)
        return False
    if not model.exists():
        raise CVError(f"infer {system}: model {model} missing but inference not done (fit marker says done); "
                      f"rerun with --allow-stale (stale inference) or move runs/cv/done/fit/{spec['tag']}.json "
                      f"aside to refit")
    if file_sha(ctx, model) != model_sha:
        raise CVError(f"{model}: sha256 differs from the fit marker")
    audio = sum(durations(ctx, rids).values())
    ledger.run(cmd, "infer", f"{system}:{','.join(rids)}", ctx.logs / f"infer_{system}.log", audio_s=audio)
    ok, bad = cache_current(ctx, ctx.cache_root, system, rids, model_sha, thetas, infer.nemo_commit())
    if not ok:
        raise CVError(f"infer {system}: cache not current after inference: {bad[:5]}")
    write_marker(ctx, "infer", name, inputs, {"cache_root": str(ctx.cache_root), "recordings": list(rids)})
    return True


def cleanup_inner_model(ctx: Ctx, spec: dict, fit_out: dict) -> None:
    """Delete an INNER fit's .nemo once its inference is done (disk hygiene); verify before destroy."""
    if spec["kind"] != "inner":
        return
    model = ctx.models_dir / f"{spec['tag']}.nemo"
    if not model.exists():
        return
    if file_sha(ctx, model) != fit_out["model_sha256"]:
        raise CVError(f"{model}: sha differs from its fit marker; not deleting")
    if ctx.dry_run:
        ctx.out(f"DRY-RUN rm {model}")
        return
    model.unlink()
    ctx.out(f"[cv] removed {model} (inner fit, inference done; log/loss CSV/fit_report kept)")


# ----------------------------------------------------------------------------------------------
# link (CPU)
# ----------------------------------------------------------------------------------------------
def link_inputs(ctx: Ctx, cache_root: Path, system: str, rids: Sequence[str], cell: tuple[float, float],
                variant: str, oracle: str) -> dict:
    from bnd import infer

    a, c = cell
    per = {}
    for rid in rids:
        d = cache_root / system / rid
        npy, emeta = infer.emb_paths(d, a)
        per[rid] = [file_sha(ctx, p, ctx.dry_run) for p in
                    (d / "meta.json", d / f"tracks_{a:.2f}.json", npy, emeta)]
        if oracle != "none":
            per[rid].append(file_sha(ctx, ctx.ref_dir / f"{rid}.rttm", ctx.dry_run))
    return {"system": system, "cache_root": str(cache_root), "recordings": list(rids), "cell": cell_key(a, c),
            "variant": variant, "oracle": oracle, "cache": per, "code": code_shas(ctx, "link", "score")}


def link_done(ctx: Ctx, out_dir: Path, inputs: dict) -> bool:
    st, m = marker_status(ctx, "link", dir_marker_name(ctx, out_dir), inputs)
    if st != "done":
        return False
    for rid, sha in m["outputs"]["rttm_sha256"].items():
        if file_sha(ctx, out_dir / f"{rid}.rttm", True) != sha:
            return False
    return True


def ensure_link(ctx: Ctx, cache_root: Path, system: str, rids: Sequence[str], cell: tuple[float, float],
                out_dir: Path, variant: str = "primary", oracle: str = "none") -> bool:
    inputs = link_inputs(ctx, cache_root, system, rids, cell, variant, oracle)
    if link_done(ctx, out_dir, inputs):
        return True
    a, c = cell
    cmd = [ctx.python, "-m", "bnd.link", "--system", system, "--recordings", *rids, "--theta-act", f"{a:.2f}",
           "--theta-clu", f"{c:.2f}", "--out", out_dir, "--cache-root", cache_root, "--variant", variant,
           "--oracle", oracle, "--ref-dir", ctx.ref_dir]
    if ctx.dry_run:
        say_dry(ctx, "cpu", cmd)
        return False
    stray = sorted(p.stem for p in out_dir.glob("*.rttm")) if out_dir.exists() else []
    stray = [s for s in stray if s not in rids]
    if stray:
        raise CVError(f"{out_dir}: unexpected RTTMs {stray}")
    run_cpu(ctx, cmd, ctx.logs / "link" / f"{dir_marker_name(ctx, out_dir)}.log")
    shas = {rid: file_sha(ctx, out_dir / f"{rid}.rttm") for rid in rids}
    write_marker(ctx, "link", dir_marker_name(ctx, out_dir), inputs, {"out_dir": str(out_dir), "rttm_sha256": shas})
    return True


def link_grid(ctx: Ctx, cache_root: Path, system: str, rids: Sequence[str]) -> None:
    def one(cell):
        return ensure_link(ctx, cache_root, system, rids, cell, ctx.hyp_root / system / cell_key(*cell))

    if ctx.dry_run or ctx.jobs <= 1:
        for cell in GRID:
            one(cell)
        return
    with ThreadPoolExecutor(max_workers=ctx.jobs) as ex:
        list(ex.map(one, GRID))


# ----------------------------------------------------------------------------------------------
# inner
# ----------------------------------------------------------------------------------------------
def cmd_inner(ctx: Ctx, only: Sequence[str] = (), max_fits: int | None = None) -> None:
    guard_after_scoring(ctx, "inner")
    ledger = Ledger(ctx)
    zroot = zeroshot_root(ctx, ledger)
    link_grid(ctx, zroot, "zeroshot", list(RECORDINGS))
    n_new = 0
    for spec in inner_fits():
        if only and spec["tag"] not in only and spec["set"] not in only and spec["recipe"] not in only:
            continue
        if max_fits is None or n_new < max_fits:  # a stale-inference refit counts as a new fit
            refit_if_infer_stale(ctx, spec, spec["heldout_recs"], THETA_ACT)
        fo = read_marker(ctx, "fit", spec["tag"])
        if max_fits is not None and fo is None and n_new >= max_fits:
            continue
        if fo is None:
            n_new += 1
        out = ensure_fit(ctx, ledger, spec)
        model = ctx.models_dir / f"{spec['tag']}.nemo"
        done = ensure_infer(ctx, ledger, spec, model, out["model_sha256"] if out else None,
                            spec["heldout_recs"], THETA_ACT)
        if done:
            cleanup_inner_model(ctx, spec, out)
        link_grid(ctx, ctx.cache_root, spec["system"], spec["heldout_recs"])


# ----------------------------------------------------------------------------------------------
# scoring with the outer lock
# ----------------------------------------------------------------------------------------------
def scored_entries(ctx: Ctx) -> list[dict]:
    """Append-only record of every outer scoring (runs/cv/scored.jsonl); never rewritten."""
    if not ctx.scored_path.exists():
        return []
    return [json.loads(l) for l in ctx.scored_path.read_text().splitlines() if l.strip()]


def frozen_identity(fz: dict) -> str:
    """What was frozen (outer RTTM hashes + selection hash), independent of the freeze timestamp."""
    return canon_sha({"systems": fz["systems"], "selection_sha256": fz["selection_sha256"]})


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def guard_after_scoring(ctx: Ctx, what: str) -> None:
    """PREREG: any change after an outer DER exists is a logged deviation. Once runs/cv/scored.jsonl has
    an entry and frozen.json is absent (unfrozen after scoring), ``what`` refuses unless --deviation
    gives the reason, which is appended to runs/cv/deviations.jsonl. Dry runs only warn."""
    ents = scored_entries(ctx)
    if not ents or ctx.frozen_path.exists():
        return
    last = ents[-1]
    if ctx.deviation:
        if ctx.dry_run:
            ctx.out(f"DRY-RUN {what}: would log deviation {ctx.deviation!r}")
        else:
            append_jsonl(ctx.cv_dir / "deviations.jsonl",
                         {"ts": now(), "command": what, "deviation": ctx.deviation,
                          "after_scoring": last["scored"], "prior_frozen_identity": last["frozen_identity"],
                          "prior_results": last["history"]})
            ctx.out(f"[cv] {what}: deviation logged after scoring: {ctx.deviation!r}")
        return
    msg = (f"OUTER LOCK: outer DER was already computed ({last['scored']}, results kept at {last['history']}) "
           f"and frozen.json is gone; `{what}` after scoring is a deviation: rerun with "
           f"--deviation \"<reason>\" (PREREG: logged, pre-deviation result reported too)")
    if ctx.dry_run:
        ctx.out("DRY-RUN WARNING " + msg)
        return
    raise OuterLockError(msg)


def verify_frozen(ctx: Ctx) -> dict:
    """frozen.json exists and every listed outer RTTM still has its frozen sha256."""
    if not ctx.frozen_path.exists():
        raise OuterLockError(f"OUTER LOCK: {ctx.frozen_path} does not exist; run `freeze` after all outer "
                             f"predictions are complete")
    fz = json.loads(ctx.frozen_path.read_text())
    bad = []
    for system, recs in fz["systems"].items():
        for rid, e in recs.items():
            p = ctx.cv_dir / e["path"]
            if file_sha(ctx, p, True) != e["sha256"]:
                bad.append(f"{system}/{rid}")
    if fz.get("selection_sha256") != file_sha(ctx, ctx.selection_path, True):
        bad.append("selection.json")
    if bad:
        raise OuterLockError(f"OUTER LOCK: frozen hashes no longer match for {bad[:6]}")
    return fz


def is_outer_artifact(ctx: Ctx, path: Path) -> bool:
    """A prediction that only exists for outer evaluation: anything under runs/cv/outer/ or produced
    from an outer refit (a ``cv_refit_*`` path component), symlinks resolved."""
    rp = Path(path).resolve()
    return rp.is_relative_to(ctx.outer_root.resolve()) or any(pt.startswith("cv_refit_") for pt in rp.parts)


def check_lock(ctx: Ctx, context: str, rid: str, path: Path) -> None:
    """Scoring ``rid`` from ``path`` in the context of outer fold ``context`` is forbidden before freeze,
    and afterwards allowed only from the exact frozen file, when EITHER ``rid`` belongs to that fold
    (context lock) OR ``path`` is an outer artifact (file lock: it cannot be scored in any context, so a
    wrong context cannot leak outer DER into selection)."""
    if context not in FOLDS:
        raise CVError(f"unknown context {context}")
    if rid not in FOLDS[context] and not is_outer_artifact(ctx, path):
        return
    fz = verify_frozen(ctx)
    rel = str(Path(path).resolve().relative_to(ctx.cv_dir.resolve())) if Path(path).resolve().is_relative_to(
        ctx.cv_dir.resolve()) else None
    shas = {e["path"]: e["sha256"] for recs in fz["systems"].values() for r, e in recs.items() if r == rid}
    if rel not in shas or shas[rel] != file_sha(ctx, path):
        raise OuterLockError(f"OUTER LOCK: {rid} (context {context}) may only be scored from a frozen outer "
                             f"prediction here; {path} is not one")


def _score_one(args) -> dict:
    rid, hyp, ref, dur, collar = args
    sc = S.score_recording(S.load_rttm_file(ref, rid), S.load_rttm_file(hyp, rid), dur, collar, recording_id=rid)
    return sc.as_dict()


def _rec_from_dict(d: dict) -> S.RecScore:
    return S.RecScore(**{k: d[k] for k in ("recording_id", "collar", "duration", "total", "miss", "fa", "conf",
                                           "ref_unions", "hyp_unions")})


_DUR_MEMO: dict[str, float] = {}


def score_paths(ctx: Ctx, context: str, paths: dict[str, Path], collars: Sequence[float]) -> dict[float, list[S.RecScore]]:
    """THE scoring entry point: per-recording DER of hyp RTTM ``paths[rid]`` (sorted by rid), with the
    outer lock applied to every recording, and a content-addressed per-recording score cache."""
    rids = sorted(paths)
    for rid in rids:
        check_lock(ctx, context, rid, paths[rid])
    dur = {}
    for rid in rids:
        wav = str((ctx.audio_dir / f"{rid}.wav").resolve())
        if wav not in _DUR_MEMO:
            _DUR_MEMO[wav] = S.audio_duration(wav)
        dur[rid] = _DUR_MEMO[wav]
    cache_dir = ctx.cv_dir / "score_cache"
    todo, keys, out = [], {}, {c: {} for c in collars}
    for rid in rids:
        ref = ctx.ref_dir / f"{rid}.rttm"
        hsha, rsha = file_sha(ctx, paths[rid]), file_sha(ctx, ref)
        for c in collars:
            key = canon_sha([rid, hsha, rsha, dur[rid], c, file_sha(ctx, ctx.repo / "bnd/score.py")])
            cp = cache_dir / key[:2] / f"{key}.json"
            if cp.exists():
                out[c][rid] = _rec_from_dict(json.loads(cp.read_text()))
            else:
                keys[(rid, c)] = cp
                todo.append((rid, str(paths[rid]), str(ref), dur[rid], c))
    if todo:
        if ctx.jobs > 1 and len(todo) > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(ctx.jobs, len(todo))) as pool:
                res = pool.map(_score_one, todo)
        else:
            res = [_score_one(t) for t in todo]
        for t, d in zip(todo, res):
            write_json(keys[(t[0], t[4])], d)
            out[t[4]][t[0]] = _rec_from_dict(d)
    return {c: [out[c][r] for r in rids] for c in collars}


def pooled(scores: Sequence[S.RecScore]) -> dict:
    tot = sum(s.total for s in scores)
    err = sum(s.errors for s in scores)
    return {"der": err / tot, "errors_s": err, "total_s": tot, "n_recordings": len(scores)}


# ----------------------------------------------------------------------------------------------
# selection (D3)
# ----------------------------------------------------------------------------------------------
def _grid_l1(cell: tuple[float, float]) -> int:
    ia = min(range(len(THETA_ACT)), key=lambda i: abs(THETA_ACT[i] - cell[0]))
    ic = min(range(len(THETA_CLU)), key=lambda i: abs(THETA_CLU[i] - cell[1]))
    ca = THETA_ACT.index(GRID_CENTRE[0])
    cc = THETA_CLU.index(GRID_CENTRE[1])
    return abs(ia - ca) + abs(ic - cc)


def choose_cell(table: dict[tuple[float, float], float], tol: float = TIE_TOL) -> tuple[tuple[float, float], list]:
    """Min pooled DER; ties (within tol) -> smallest grid-index L1 distance from (0.50, 0.65), then lower
    theta_act, then lower theta_clu (D3). Returns (cell, tied cells)."""
    if not table:
        raise CVError("empty table")
    m = min(table.values())
    tied = sorted((c for c, v in table.items() if v <= m + tol), key=lambda c: (_grid_l1(c), c[0], c[1]))
    return tied[0], tied


def choose_recipe(best: dict[str, float], tol: float = TIE_TOL) -> tuple[str, list]:
    """Min of each recipe's best-cell DER; ties (within tol) -> lr 1e-5, then timing raw (D3)."""
    if not best:
        raise CVError("no recipes")
    m = min(best.values())
    lr_order = list(LRS)
    tied = sorted((r for r, v in best.items() if v <= m + tol),
                  key=lambda r: (lr_order.index(recipe_parts(r)[0]), TIMINGS.index(recipe_parts(r)[1])))
    return tied[0], tied


def build_select_dirs(ctx: Ctx, outer: str) -> dict[str, dict[str, dict[str, Path]]]:
    """{system_key: {cell: {rid: hyp path}}} for outer ``outer``; also materialised as symlink dirs
    runs/cv/select/<outer>/<system_key>/<cell>/. Never contains a recording of ``outer``."""
    res: dict[str, dict[str, dict[str, Path]]] = {}
    inner = [r for f in sorted(FOLDS) if f != outer for r in FOLDS[f]]
    sources: dict[str, list[tuple[str, str]]] = {"zeroshot": [("zeroshot", r) for r in inner]}
    for recipe in RECIPES:
        sources[recipe] = [(fit["system"], r) for fit, _fj, recs in inner_sources(outer, recipe) for r in recs]
    for key, src in sources.items():
        res[key] = {}
        for cell in GRID:
            ck = cell_key(*cell)
            d = ctx.select_root / outer / key / ck
            paths = {}
            for system, rid in src:
                if rid in FOLDS[outer]:
                    raise CVError(f"selection for {outer} would include outer recording {rid}")
                tgt = ctx.hyp_root / system / ck / f"{rid}.rttm"
                paths[rid] = tgt
            if not ctx.dry_run:
                d.mkdir(parents=True, exist_ok=True)
                for p in d.iterdir():
                    if p.is_symlink():
                        p.unlink()
                    else:
                        raise CVError(f"{d}: non-symlink {p.name}")
                for rid, tgt in paths.items():
                    (d / f"{rid}.rttm").symlink_to(tgt.resolve())
            res[key][ck] = {rid: d / f"{rid}.rttm" for rid in paths}
            if sorted(paths) != sorted(inner):
                raise CVError(f"{outer}/{key}: pooled set {sorted(paths)} != inner recordings {sorted(inner)}")
    return res


def select_outer(ctx: Ctx, outer: str) -> dict:
    dirs = build_select_dirs(ctx, outer)
    missing = [str(p) for sysd in dirs.values() for cd in sysd.values() for p in cd.values()
               if not p.resolve().is_file()]
    if missing:
        raise CVError(f"select {outer}: {len(missing)} inner predictions missing (run `inner` first), e.g. {missing[:3]}")
    out: dict = {"inner_recs": sorted(r for f in FOLDS if f != outer for r in FOLDS[f]), "collar": PRIMARY_COLLAR}
    tables = {}
    for key, cells in dirs.items():
        tab = {}
        for ck, paths in cells.items():
            sc = score_paths(ctx, outer, paths, [PRIMARY_COLLAR])[PRIMARY_COLLAR]
            a, c = parse_cell(ck)
            tab[ck] = dict(theta_act=a, theta_clu=c, **pooled(sc),
                           per_recording={s.recording_id: s.der for s in sc})
        cell, tied = choose_cell({parse_cell(k): v["der"] for k, v in tab.items()})
        tables[key] = {"cells": tab, "selected_cell": cell_key(*cell), "selected_der": tab[cell_key(*cell)]["der"],
                       "tied_cells": [cell_key(*t) for t in tied]}
    out["zeroshot"] = tables.pop("zeroshot")
    out["recipes"] = tables
    rec, tied = choose_recipe({r: t["selected_der"] for r, t in tables.items()})
    out["selected_recipe"] = rec
    out["tied_recipes"] = tied
    out["strict_recipe"] = STRICT_RECIPE
    out["sources"] = {r: {rid: fit["tag"] for fit, _fj, recs in inner_sources(outer, r) for rid in recs}
                      for r in RECIPES}
    return out


def cmd_select(ctx: Ctx) -> dict | None:
    if ctx.dry_run:
        for fi in sorted(FOLDS):
            build_select_dirs(ctx, fi)
            ctx.out(f"DRY-RUN select {fi}: zero-shot pooled over {len(FOLDS) * 2 - 2} inner recordings, "
                    f"{len(RECIPES)} recipes x {len(GRID)} cells from fits "
                    f"{sorted({f['tag'] for r in RECIPES for f, _, _ in inner_sources(fi, r)})[:4]}...")
        return None
    if ctx.frozen_path.exists():
        verify_frozen(ctx)
        ctx.out("[cv] frozen; selection is locked")
        return json.loads(ctx.selection_path.read_text())
    guard_after_scoring(ctx, "select")
    sel = {"created": now(), "grid": {"theta_act": list(THETA_ACT), "theta_clu": list(THETA_CLU)},
           "tie_breaks": "D3", "folds": {}}
    for fi in sorted(FOLDS):
        sel["folds"][fi] = select_outer(ctx, fi)
        f = sel["folds"][fi]
        ctx.out(f"[cv] {fi}: zero-shot {f['zeroshot']['selected_cell']} "
                f"({100 * f['zeroshot']['selected_der']:.2f}%), recipe {f['selected_recipe']} "
                f"{f['recipes'][f['selected_recipe']]['selected_cell']} "
                f"({100 * f['recipes'][f['selected_recipe']]['selected_der']:.2f}%)")
    write_json(ctx.selection_path, sel)
    return sel


# ----------------------------------------------------------------------------------------------
# outer
# ----------------------------------------------------------------------------------------------
def outer_jobs(ctx: Ctx, sel: dict, zroot: Path) -> list[dict]:
    """Every outer link job: system, fold, cache root, cache system, cell, oracle."""
    jobs = []
    for fi in sorted(FOLDS):
        f = sel["folds"][fi]
        rsel = f["selected_recipe"]
        zs_cell = parse_cell(f["zeroshot"]["selected_cell"])
        ft_cell = parse_cell(f["recipes"][rsel]["selected_cell"])
        ft_sel_sys = refit_spec(fi, rsel)["system"]
        ft_strict_sys = refit_spec(fi, STRICT_RECIPE)["system"]
        for name, root, system, cell, oracle in (
                ("zs_sel", zroot, "zeroshot", zs_cell, "none"),
                ("zs_strict", zroot, "zeroshot", STRICT_CELL, "none"),
                ("ft_sel", ctx.cache_root, ft_sel_sys, ft_cell, "none"),
                ("ft_strict", ctx.cache_root, ft_strict_sys, STRICT_CELL, "none"),
                ("zs_sel_oracle_link", zroot, "zeroshot", zs_cell, "link"),
                ("zs_sel_oracle_speech", zroot, "zeroshot", zs_cell, "speech"),
                ("ft_sel_oracle_link", ctx.cache_root, ft_sel_sys, ft_cell, "link"),
                ("ft_sel_oracle_speech", ctx.cache_root, ft_sel_sys, ft_cell, "speech")):
            jobs.append({"name": name, "fold": fi, "cache_root": root, "system": system, "cell": cell,
                         "oracle": oracle, "recs": list(FOLDS[fi]),
                         "out_dir": ctx.outer_root / name / fi})
    return jobs


def cmd_outer(ctx: Ctx) -> None:
    if ctx.frozen_path.exists():
        verify_frozen(ctx)
        ctx.out("[cv] outer predictions are frozen; nothing to do")
        return
    guard_after_scoring(ctx, "outer")
    if not ctx.selection_path.exists():
        if ctx.dry_run:
            ctx.out("DRY-RUN outer: selection.json missing; per fold: refit selected recipe (+ strict lr1e-5_raw "
                    "if different) on 8 recordings, infer the 2 outer recordings, link 8 systems")
            for fi in sorted(FOLDS):
                sp = refit_spec(fi, STRICT_RECIPE)
                say_dry(ctx, "gpu", [ctx.python, "-m", "bnd.train", "--tag", f"cv_refit_{fi}_<selected>",
                                     "--train-recs", ",".join(sp["train_recs"]), "--lr", "<sel>", "--timing", "<sel>"])
            return
        raise CVError("selection.json missing; run `select` first")
    sel = json.loads(ctx.selection_path.read_text())
    ledger = Ledger(ctx)
    zroot = zeroshot_root(ctx, ledger)
    for fi in sorted(FOLDS):
        f = sel["folds"][fi]
        rsel = f["selected_recipe"]
        need: dict[str, set] = {rsel: {parse_cell(f["recipes"][rsel]["selected_cell"])[0]}}
        need.setdefault(STRICT_RECIPE, set()).add(STRICT_CELL[0])
        for recipe, thetas in need.items():
            spec = refit_spec(fi, recipe)
            refit_if_infer_stale(ctx, spec, spec["heldout_recs"], sorted(thetas))
            out = ensure_fit(ctx, ledger, spec)
            ensure_infer(ctx, ledger, spec, ctx.models_dir / f"{spec['tag']}.nemo",
                         out["model_sha256"] if out else None, spec["heldout_recs"], sorted(thetas))
    jobs = outer_jobs(ctx, sel, zroot)
    all_done = True
    for j in jobs:
        all_done &= ensure_link(ctx, j["cache_root"], j["system"], j["recs"], j["cell"], j["out_dir"],
                                oracle=j["oracle"])
    if all_done and not ctx.dry_run:
        write_outer_index(ctx, jobs)


def write_outer_index(ctx: Ctx, jobs: list[dict]) -> None:
    write_json(ctx.outer_root / "index.json", {
        "created": now(), "selection_sha256": file_sha(ctx, ctx.selection_path),
        "jobs": [{**{k: v for k, v in j.items() if k not in ("cache_root", "out_dir", "cell")},
                  "cell": cell_key(*j["cell"]), "cache_root": str(j["cache_root"]),
                  "out_dir": str(Path(j["out_dir"]).resolve().relative_to(ctx.cv_dir.resolve()))}
                 for j in jobs]})


# ----------------------------------------------------------------------------------------------
# freeze
# ----------------------------------------------------------------------------------------------
def cmd_freeze(ctx: Ctx) -> dict | None:
    idx_p = ctx.outer_root / "index.json"
    if not idx_p.exists():
        raise CVError("freeze refused: runs/cv/outer/index.json missing (outer predictions incomplete)")
    if not ctx.selection_path.exists():
        raise CVError("freeze refused: selection.json missing")
    idx = json.loads(idx_p.read_text())
    if idx.get("selection_sha256") != file_sha(ctx, ctx.selection_path):
        raise CVError("freeze refused: selection.json changed after the outer predictions were made")
    systems: dict[str, dict] = {s: {} for s in OUTER_SYSTEMS}
    problems = []
    for j in idx["jobs"]:
        d = ctx.cv_dir / j["out_dir"]
        m = read_marker(ctx, "link", dir_marker_name(ctx, d))
        if m is None:
            problems.append(f"{j['out_dir']}: no link marker")
            continue
        for rid in j["recs"]:
            p = d / f"{rid}.rttm"
            sha = file_sha(ctx, p, True)
            if sha == "MISSING" or m["outputs"]["rttm_sha256"].get(rid) != sha:
                problems.append(f"{j['out_dir']}/{rid}.rttm missing or differs from its link marker")
                continue
            systems[j["name"]][rid] = {"path": str(p.relative_to(ctx.cv_dir)), "sha256": sha,
                                       "fold": j["fold"], "source_system": j["system"], "cell": j["cell"],
                                       "oracle": j["oracle"]}
    for s in OUTER_SYSTEMS:
        miss = sorted(set(RECORDINGS) - set(systems[s]))
        if miss:
            problems.append(f"{s}: missing outer predictions for {miss}")
    if problems:
        raise CVError(f"freeze refused: {problems[:8]}")
    fz = {"created": now(), "selection_sha256": file_sha(ctx, ctx.selection_path), "systems": systems}
    if ctx.frozen_path.exists():
        old = verify_frozen(ctx)
        if old["systems"] != systems:
            raise OuterLockError("already frozen with different outer predictions; refusing to re-freeze")
        ctx.out("[cv] already frozen; hashes verified")
        return old
    ents = scored_entries(ctx)
    if ents:
        last = ents[-1]
        if frozen_identity(fz) == last["frozen_identity"]:
            ctx.out(f"[cv] re-freezing exactly what was scored at {last['scored']} (restoring frozen.json)")
            fz["restored_from_scoring"] = last["scored"]
        else:
            guard_after_scoring(ctx, "freeze")
            fz["deviation"] = ctx.deviation
            fz["supersedes"] = {"frozen_identity": last["frozen_identity"], "scored": last["scored"],
                                "results": last["history"]}
    if ctx.dry_run:
        ctx.out(f"DRY-RUN freeze {sum(len(v) for v in systems.values())} outer RTTMs")
        return None
    write_json(ctx.frozen_path, fz)
    ctx.out(f"[cv] frozen {sum(len(v) for v in systems.values())} outer RTTMs -> {ctx.frozen_path}")
    return fz


# ----------------------------------------------------------------------------------------------
# gate + score
# ----------------------------------------------------------------------------------------------
def gate_verdict(a0: Sequence[S.RecScore], b0: Sequence[S.RecScore], a25: Sequence[S.RecScore],
                 b25: Sequence[S.RecScore], boot0: dict) -> dict:
    """PREREG adoption gate for B (fine-tuned) over A (zero-shot), outer folds; all six must hold."""
    for x, y in ((a0, b0), (a25, b25), (a0, a25)):
        if [s.recording_id for s in x] != [s.recording_id for s in y]:
            raise CVError("gate inputs on different recordings")
    ca, cb = pooled(a0)["der"], pooled(b0)["der"]
    delta = np.array([b.der - a.der for a, b in zip(a0, b0)])
    improvement_pp = 100 * (ca - cb)
    wins = int((delta < 0).sum())
    worst_pp = float(100 * delta.max())
    worst_rec = a0[int(delta.argmax())].recording_id
    d25_pp = 100 * (pooled(b25)["der"] - pooled(a25)["der"])
    conds = {
        "corpus_improvement_ge_2pp": {"value_pp": improvement_pp,
                                      "pass": bool(improvement_pp >= GATE_MIN_IMPROVEMENT_PP - 1e-9)},
        "macro_mean_delta_lt_0": {"value_pp": float(100 * delta.mean()), "pass": bool(delta.mean() < 0)},
        "wins_ge_7": {"wins": wins, "losses": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                      "n": len(delta), "pass": bool(wins >= GATE_MIN_WINS)},
        "no_recording_worse_gt_5pp": {"worst_delta_pp": worst_pp, "worst_recording": worst_rec,
                                      "pass": bool(worst_pp <= GATE_MAX_WORSE_PP + 1e-9)},
        "bootstrap_p97_5_lt_0": {"ci_high_pp": 100 * boot0["ci_high"], "ci_low_pp": 100 * boot0["ci_low"],
                                 "pass": bool(boot0["ci_high"] < 0)},
        "no_corpus_regression_collar_0.25": {"delta_pp": d25_pp, "pass": bool(d25_pp <= 0)},
    }
    adopt = all(c["pass"] for c in conds.values())
    return {"conditions": conds, "adopt_finetuned": adopt,
            "default_system": "fine-tuned" if adopt else "zero-shot (calibrated)"}


def compare(a: dict, b: dict, n: int) -> dict:
    return {str(c): S.paired_bootstrap(a[c], b[c], n=n, seed=S.BOOTSTRAP_SEED) for c in COLLARS}


def cmd_score(ctx: Ctx) -> dict:
    fz = verify_frozen(ctx)
    ident = frozen_identity(fz)
    ents = scored_entries(ctx)
    if ents and ents[-1]["frozen_identity"] != ident and not fz.get("deviation"):
        raise OuterLockError(f"OUTER LOCK: frozen.json differs from what was scored at {ents[-1]['scored']} but "
                             f"records no deviation; refusing to replace those results")
    sel = json.loads(ctx.selection_path.read_text())
    scores: dict[str, dict[float, list[S.RecScore]]] = {}
    for system, recs in fz["systems"].items():
        per_c: dict[float, list] = {c: [] for c in COLLARS}
        for fi in sorted(FOLDS):
            paths = {rid: ctx.cv_dir / recs[rid]["path"] for rid in FOLDS[fi]}
            r = score_paths(ctx, fi, paths, COLLARS)
            for c in COLLARS:
                per_c[c] += r[c]
        for c in COLLARS:
            per_c[c].sort(key=lambda s: s.recording_id)
        scores[system] = per_c
    res: dict = {"created": now(), "frozen_created": fz["created"], "collars": list(COLLARS),
                 "bootstrap": {"n": ctx.bootstrap_n, "seed": S.BOOTSTRAP_SEED}, "systems": {}}
    for system, per_c in scores.items():
        res["systems"][system] = {str(c): {"corpus": S.corpus_summary(sc), "per_recording": [s.as_dict() for s in sc]}
                                  for c, sc in per_c.items()}
    prim = compare(scores["zs_sel"], scores["ft_sel"], ctx.bootstrap_n)
    res["primary_comparison"] = {"a": "zs_sel", "b": "ft_sel", "by_collar": prim}
    res["gate"] = gate_verdict(scores["zs_sel"][0.0], scores["ft_sel"][0.0], scores["zs_sel"][0.25],
                               scores["ft_sel"][0.25], prim["0.0"])
    strict = compare(scores["zs_strict"], scores["ft_strict"], ctx.bootstrap_n)
    res["strict_comparison"] = {"a": "zs_strict", "b": "ft_strict", "cell": cell_key(*STRICT_CELL),
                                "recipe": STRICT_RECIPE, "by_collar": strict}
    res["oracle_diagnostics"] = {
        base: {kind: {str(c): S.corpus_summary(scores[f"{base}_oracle_{kind}"][c]) for c in COLLARS}
               for kind in ("link", "speech")} for base in ("zs_sel", "ft_sel")}
    res["selection"] = {fi: {"zeroshot_cell": f["zeroshot"]["selected_cell"], "recipe": f["selected_recipe"],
                             "ft_cell": f["recipes"][f["selected_recipe"]]["selected_cell"]}
                        for fi, f in sel["folds"].items()}
    res["frozen_identity"] = ident
    if fz.get("deviation"):
        res["deviation"] = fz["deviation"]
        res["supersedes"] = fz.get("supersedes")
    # every scoring is kept (never overwritten): results_history/<n>_<identity>.json + scored.jsonl
    n = len(ents) + 1
    hist = ctx.history_root / f"results_{n:03d}_{ident[:12]}.json"
    if hist.exists():
        raise CVError(f"{hist} already exists; refusing to overwrite a scored result")
    write_json(hist, res)
    write_json(ctx.cv_dir / "results.json", res)
    (ctx.cv_dir / "results.md").write_text(results_markdown(res))
    append_jsonl(ctx.scored_path, {"scored": res["created"], "frozen_identity": ident,
                                   "results_sha256": file_sha(ctx, hist),
                                   "history": str(hist.relative_to(ctx.cv_dir)),
                                   "deviation": fz.get("deviation")})
    ctx.out(results_markdown(res))
    return res


def results_markdown(res: dict) -> str:
    L = ["# Phase-2 registered CV results (outer folds)", "",
         f"Frozen {res['frozen_created']}; scored {res['created']}. Bootstrap n={res['bootstrap']['n']}, "
         f"seed {res['bootstrap']['seed']}.", "", "## Selection (inner folds only)", "",
         "| fold | zero-shot cell | FT recipe | FT cell |", "|---|---|---|---|"]
    for fi, s in sorted(res["selection"].items()):
        L.append(f"| {fi} | {s['zeroshot_cell']} | {s['recipe']} | {s['ft_cell']} |")
    L += ["", "## Corpus DER (%)", "", "| system | " + " | ".join(f"collar {c}" for c in res["collars"]) + " |",
          "|---|" + "---|" * len(res["collars"])]
    for system, byc in res["systems"].items():
        L.append(f"| {system} | " + " | ".join(f"{100 * byc[str(c)]['corpus']['der']:.2f}" for c in res["collars"]) + " |")
    L += ["", "## Per-recording DER, collar 0 (%)", "", "| rec | zs_sel | ft_sel | delta pp | zs_strict | ft_strict |",
          "|---|---|---|---|---|---|"]
    get = {s: {d["recording_id"]: d["der"] for d in res["systems"][s]["0.0"]["per_recording"]}
           for s in ("zs_sel", "ft_sel", "zs_strict", "ft_strict")}
    for rid in sorted(get["zs_sel"]):
        L.append(f"| {rid} | {100 * get['zs_sel'][rid]:.2f} | {100 * get['ft_sel'][rid]:.2f} | "
                 f"{100 * (get['ft_sel'][rid] - get['zs_sel'][rid]):+.2f} | {100 * get['zs_strict'][rid]:.2f} | "
                 f"{100 * get['ft_strict'][rid]:.2f} |")
    L += ["", "## Adoption gate (ft_sel vs zs_sel, collar 0)", "", "| condition | value | pass |", "|---|---|---|"]
    for k, v in res["gate"]["conditions"].items():
        val = {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "pass"}
        L.append(f"| {k} | {val} | {'PASS' if v['pass'] else 'FAIL'} |")
    L += ["", f"**Verdict: default system = {res['gate']['default_system']}**", "",
          "## Comparisons (paired bootstrap, delta = B - A, pp)", "", "| comparison | collar | A | B | delta | 95% CI | wins/n |",
          "|---|---|---|---|---|---|---|"]
    for name in ("primary_comparison", "strict_comparison"):
        for c, r in res[name]["by_collar"].items():
            L.append(f"| {name} | {c} | {100 * r['der_a']:.2f} | {100 * r['der_b']:.2f} | "
                     f"{100 * r['delta_b_minus_a']:+.2f} | [{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] | "
                     f"{r['b_wins']}/{r['n_recordings']} |")
    L += ["", "## Oracle diagnostics (corpus DER %, collar 0)", "", "| base | (a) oracle link | (b) oracle speech |",
          "|---|---|---|"]
    for base, d in res["oracle_diagnostics"].items():
        L.append(f"| {base} | {100 * d['link']['0.0']['der']:.2f} | {100 * d['speech']['0.0']['der']:.2f} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------
# status / ledger / CLI
# ----------------------------------------------------------------------------------------------
def cmd_status(ctx: Ctx) -> dict:
    fits = inner_fits()
    st = {"inner_fits_done": sum(read_marker(ctx, "fit", f["tag"]) is not None for f in fits),
          "inner_fits_total": len(fits),
          "selection": ctx.selection_path.exists(), "outer_index": (ctx.outer_root / "index.json").exists(),
          "frozen": ctx.frozen_path.exists(), "results": (ctx.cv_dir / "results.json").exists(),
          "times_scored": len(scored_entries(ctx)),
          "gpu_h_used": round(Ledger(ctx).used_h(), 3), "gpu_h_budget": ctx.budget_h}
    ctx.out(json.dumps(st, indent=1))
    return st


def cmd_ledger(ctx: Ctx) -> None:
    led = Ledger(ctx)
    es = led.entries()
    for e in es:
        ctx.out(f"{e['ts_start']} {e['kind']:6s} {e['tag']:40s} card={e['card']} rc={e['returncode']} "
                f"gpu_wall={e['gpu_wall_s']:.0f}s")
    ctx.out(f"{len(es)} jobs; used {led.used_h():.3f} h (incl. phase-2a {ctx.start_h} h) of {ctx.budget_h} h")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bnd.cv", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("plan", "inner", "select", "outer", "freeze", "score", "ledger", "status"))
    ap.add_argument("--cv-dir", type=Path, default=Ctx.cv_dir)
    ap.add_argument("--dry-run", action="store_true", help="print planned commands; write nothing")
    ap.add_argument("--allow-stale", action="store_true", help="move stale GPU results aside and redo them")
    ap.add_argument("--jobs", type=int, default=4, help="parallel CPU link/score workers (<= 4 on the shared box)")
    ap.add_argument("--only", nargs="*", default=(), help="inner: restrict to these tags / set ids / recipes")
    ap.add_argument("--max-new-fits", type=int, default=None, help="inner: start at most N new fits this run")
    ap.add_argument("--bootstrap-n", type=int, default=S.BOOTSTRAP_N)
    ap.add_argument("--deviation", default=None,
                    help="reason for re-running inner/select/outer/freeze after outer DER was computed (logged)")
    a = ap.parse_args(argv)
    ctx = Ctx(cv_dir=a.cv_dir, dry_run=a.dry_run, allow_stale=a.allow_stale, jobs=max(1, min(a.jobs, 4)),
              bootstrap_n=a.bootstrap_n, deviation=a.deviation)
    if a.cmd == "plan":
        plan = build_plan(ctx)
        if ctx.dry_run:
            ctx.out(json.dumps({k: v for k, v in plan.items() if k != "inner_fits"}, indent=1))
            ctx.out(f"DRY-RUN plan: {plan['n_inner_fits']} unique inner fits; not written")
        else:
            write_json(ctx.cv_dir / "plan.json", plan)
            ctx.out(f"[cv] wrote {ctx.cv_dir / 'plan.json'}: {plan['n_inner_fits']} inner fits, "
                    f"projected <= {plan['budget']['projected_gpu_h_upper']} GPU-h")
    elif a.cmd == "inner":
        cmd_inner(ctx, a.only, a.max_new_fits)
    elif a.cmd == "select":
        cmd_select(ctx)
    elif a.cmd == "outer":
        cmd_outer(ctx)
    elif a.cmd == "freeze":
        cmd_freeze(ctx)
    elif a.cmd == "score":
        if ctx.dry_run:
            verify_frozen(ctx)
            ctx.out("DRY-RUN score: frozen hashes verified; would score 8 systems x 10 recordings x 3 collars")
        else:
            cmd_score(ctx)
    elif a.cmd == "ledger":
        cmd_ledger(ctx)
    elif a.cmd == "status":
        cmd_status(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
