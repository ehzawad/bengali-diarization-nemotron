"""Resumable driver for PHASE 5 (docs/PREREG_phase5.md, binding): linking challenger (fine-tuned WeSpeaker
embedder x 120 s native sessions) on the 10 Bengali-Loop recordings (Part A, DEVELOPMENT numbers) and one
frozen evaluation on the independent synthetic hold-out smam/bengali-diarization-synthetic-v4 (Part B).

Every output lives under runs/p5/. Earlier registered artefacts (runs/cv, runs/p3b, runs/final, runs/diag)
and the data (data/raw, data/prepared/ref_rttm, data/holdout_synth) are read only; the phase-3b zero-shot
S=20 cache runs/p3b/cache/zeroshot/s20 is reused through read-only symlinks.

EVIDENCE STATUS (carried into every report): all 10 Bengali-Loop recordings are DEVELOPMENT data (>= 5
looks); Part-A numbers are development numbers. Part B is the only independent evaluation; it is synthetic
(random chunk placement of IndicVoices/Kathbath read/extempore speech, 324 voices reused across mixtures),
not natural drama. Base model Nemotron-3-Diarization f667ed73 (DISPLACE 2024 incl. Bengali in training).

Part A: 2 x 2 factorial on the BASE model, cells (``CELLS``)
  s20_stock (INCUMBENT)  s120_stock  s20_ft  s120_ft (CHALLENGER)
S = native session core (+5 s context each side, one forward call per session, bnd.infer --core-s),
E = embedder (stock WeSpeaker resnet34-LM @837717dd, or the FT embedder of bnd.embed_ft). Calibration family
(20 configs per cell): theta_act {0.25, 0.40} x theta_clu {0.45, 0.55, 0.60, 0.65, 0.75} x gap-fill g {0, 5},
linker avg (phase 2, hard same-window cannot-link). Selection per cell per outer fold on pooled inner DER
(collar 0): SIMPLEST config within 0.5 pp of the minimum; tie order g 0<5, theta_act 0.40<0.25, theta_clu
0.60<0.55<0.65<0.45<0.75 (lexicographic in that order). Inner pools: stock cells = the zero-shot caches of
the 8 inner recordings; FT cells = recording r of inner fold Fj embedded by the fit trained on the 6
recordings outside {Fi, Fj} (never an embedder trained on r or on Fi). Outer: FT cells use the 8-recording
refit that excludes Fi. The floor adapter (both boundaries floored to whole seconds, after g) is NOT in the
search; it is a separately reported benchmark adapter (``<cell>_floor``).

Subcommands (resumable via done-markers with input hashes under runs/p5/done/; ``--dry-run`` prints the
planned work and projected GPU-h and writes nothing; ``plan`` = dry run of everything):
  A-prep      zero-shot S=120 caches (bnd.infer --core-s 120 --session-subdir --no-embed, tracks theta_act
              {0.25, 0.40}) of all 10 recordings into runs/p5/cache/zeroshot/s120; S=20 = read-only symlinks
              to runs/p3b/cache/zeroshot/s20 (probs, tracks and stock embeddings, verified).
  A-fits      16 embedder fits (bnd.embed_ft; 10 inner 6-recording sets, 5 eight-recording refits, 1 all-10),
              FT embeddings of each fit's held-out recordings for S {20,120} x theta_act {0.25,0.40}
              (bnd.embed --embedder-ckpt, never on a training recording), stock S=120 embeddings.
  A-search    link 4 cells x 20 configs for every inner unit (CPU, <= 4 workers), selection per fold + LORO
              -> runs/p5/selection.json.
  A-outer     outer RTTMs of all 4 cells at the fold's selected config (+ floor-adapter variants).
  A-freeze    sha256 of every outer RTTM -> runs/p5/frozen.json.
  A-score     OUTER LOCK; 8 systems (4 cells x {no adapter, floor}), collars 0/0.25/0.5, MISS/FA/CONF,
              sum|count err|, per recording, paired bootstrap (50k, seed 20260925), dev adoption criterion,
              final config per cell = joint mode -> runs/p5/dev_results.{json,md} + final_configs.json.
  B-validate  (hold-out downloaded: 60 stems + data/holdout_synth_SHA256.txt) zero-shot Nemotron S=20 probs
              + Silero on the hold-out audio; reference statistics (labelled fraction, overlap fraction,
              labelled speech silent to both detectors, turn stats) -> runs/p5/holdout_validation.json.
  B-infer     frozen batch: incumbent, challenger (all-10 FT embedder), pyannote community-1 @3533c8cf,
              FT-Nemotron (bnd.train all 10, lr 5e-5, shift, 165 steps, seed 42; phase-3b FT joint-mode
              decoding); ACOUSTIC (g=0) and BENCHMARK (selected g) modes -> runs/p5/holdout/rttm/.
  B-freeze    sha256 of every hold-out RTTM + configs/models -> runs/p5/holdout_frozen.json.
  B-score     HOLD-OUT LOCK (refuses before freeze, after tampering, and a second scoring); corpus DER
              collars 0/0.25/0.5 (overlap included, whole-file UEM), per mixture, per speaker-count bin,
              sum|count err| + per-bin MAE, paired bootstrap over mixtures (50k, seed 20260926), claim rule
              -> runs/p5/holdout_results.{json,md}.
  ledger / status.   (internal, launched by the driver under scripts/gpu_run.sh: embed-job,
  holdout-nemo-job, community1-job; FT-Nemotron training reuses ``bnd.p3b train-fit``.)

GPU: every GPU command through scripts/gpu_run.sh (one card, <= 20 GB). Ledger runs/p5/gpu_ledger.jsonl,
starting balance 6.03 GPU-h (cumulative before phase 5); a job is refused before launch if it would exceed
12 GPU-h cumulative; a WARNING when phase-5 use exceeds 5.5 h. bnd.embed_ft appends its own in-process
entries to the same ledger; the driver's gpu_run entry for that job lists them in ``covers_lines`` and they
are not counted twice. Running GPU children and the driver are recorded in runs/p5/pids/.

Run on CPU:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 <nemo venv python> -m bnd.p5 <cmd>
"""
from __future__ import annotations

import argparse
import atexit
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from bnd import cv, p3b
from bnd import score as S
from bnd.labels import RECORDINGS

REPO = cv.REPO
FOLDS = cv.FOLDS
Config = p3b.Config

# ----------------------------------------------------------------------------------------------
# registered constants (PREREG_phase5)
# ----------------------------------------------------------------------------------------------
SESSIONS = (20, 120)
THETA_ACT = (0.25, 0.40)
THETA_CLU = (0.45, 0.55, 0.60, 0.65, 0.75)
GAPS = (0, 5)
RANK_G = {0: 0, 5: 1}
RANK_A = {0.40: 0, 0.25: 1}
RANK_C = {0.60: 0, 0.55: 1, 0.65: 2, 0.45: 3, 0.75: 4}
SELECT_TOL_PP = 0.5
CELLS = {"s20_stock": (20, "stock"), "s120_stock": (120, "stock"), "s20_ft": (20, "ft"), "s120_ft": (120, "ft")}
INCUMBENT, CHALLENGER = "s20_stock", "s120_ft"
FLOOR = "_floor"
DEV_SYSTEMS = tuple(CELLS) + tuple(c + FLOOR for c in CELLS)
DEV_MIN_IMPROVEMENT_PP = 1.0
DEV_MAX_COUNT_INCREASE = 0.10
PHASE_START_H = 6.03
ALLOWANCE_H = 5.5
BUDGET_GPU_H = 12.0
DEV_BOOT_SEED = S.BOOTSTRAP_SEED            # 20260925
HOLDOUT_BOOT_SEED = 20260926
HOLDOUT_BOOT_N = 50_000                     # PREREG Part B: 50k resamples (not a CLI choice)
HOLDOUT_MIN_IMPROVEMENT_PP = 1.0
SUBSET_SEED = 20260926
SUBSET_K = range(1, 31)
SUBSET_PER_K = 2
SUBSET_POOL = 20
DATASET_REVISION = "c3c7cb03053144e98effbcaf764e3fb1f61f3004"
SPK_BINS = (("1-4", 1, 4), ("5-8", 5, 8), ("9-16", 9, 16), ("17-30", 17, 30))
COMMUNITY1_ID = "pyannote/speaker-diarization-community-1"
COMMUNITY1_REV = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
FTN_TAG = "p5_final_ftnemotron"
FTN_LR = "5e-5"
FTN_TIMING = "shift"
FINAL_EMBEDDER = "p5emb_all10"
HOLDOUT_SYSTEMS = ("incumbent_acoustic", "incumbent_benchmark", "challenger_acoustic", "challenger_benchmark",
                   "community1", "ftnemotron_acoustic", "ftnemotron_benchmark")
SILERO_CMD = ["uv", "run", "--no-project", "--python", "3.12", "--with", "silero-vad==6.2.3", "--with", "soundfile",
              "--with", "numpy", "python"]
SILERO_WORKERS = 4
THRESH = 0.5
# cost model (projection / pre-launch check); fits and community-1 are unmeasured -> conservative
DEF_EMBFIT_S = 600.0                         # replaced by measured bnd.embed_ft fits (ledger) when present
COMMUNITY1_PER_AUDIO_S = 0.025 * 1.25        # runs/diag/community1_infer.log: RTF 0.0250 on the A6000
HOLDOUT_ASSUMED_S = 1800.0                   # dataset card: ~30-minute mixtures (dry run before download)
EVIDENCE = ("All 10 Bengali-Loop recordings are DEVELOPMENT data (>= 5 looks): Part-A numbers are development "
            "numbers, never held-out. Part B (smam/bengali-diarization-synthetic-v4, 60 mixtures) is the only "
            "independent evaluation; it is synthetic (random chunk placement of IndicVoices/Kathbath read/extempore "
            "speech; 324 voices reused across mixtures, so the mixture bootstrap is approximate), not natural drama. "
            "Base model Nemotron-3-Diarization f667ed73 has DISPLACE 2024 (incl. Bengali) in its training data.")


class P5Error(p3b.P3bError):
    pass


class HoldoutLockError(cv.OuterLockError):
    pass


def cfg_simplicity(c: Config) -> tuple:
    return (RANK_G[int(c.gap)], RANK_A[round(c.theta_act, 2)], RANK_C[round(c.theta_clu, 2)])


CONFIGS = [Config(s, "avg", g, a, c) for s in SESSIONS for a in THETA_ACT for c in THETA_CLU for g in GAPS]
assert len(CONFIGS) == 40 and len({c.key for c in CONFIGS}) == 40


def cell_configs(cell: str) -> list[Config]:
    return [c for c in CONFIGS if c.session == CELLS[cell][0]]


def configs_for(session: int, theta: float) -> list[Config]:
    return [c for c in CONFIGS if c.session == session and math.isclose(c.theta_act, theta)]


def acoustic(cfg: Config) -> Config:
    return Config(cfg.session, cfg.linker, 0, cfg.theta_act, cfg.theta_clu)


# ----------------------------------------------------------------------------------------------
# embedder fits and pools
# ----------------------------------------------------------------------------------------------
def embedder_fits() -> list[dict]:
    """The 16 registered FT-embedder fits."""
    out = []
    for pair in itertools.combinations(sorted(FOLDS), 2):
        out.append({"tag": f"p5emb_{cv.set_id(pair)}", "kind": "inner", "excluded_folds": list(pair),
                    "train_recs": sorted(r for f in FOLDS if f not in pair for r in FOLDS[f]),
                    "heldout_recs": cv.fold_recs(pair)})
    for fi in sorted(FOLDS):
        out.append({"tag": f"p5emb_refit_{fi}", "kind": "refit", "excluded_folds": [fi],
                    "train_recs": sorted(r for f in FOLDS if f != fi for r in FOLDS[f]),
                    "heldout_recs": list(FOLDS[fi])})
    out.append({"tag": FINAL_EMBEDDER, "kind": "final", "excluded_folds": [], "train_recs": sorted(RECORDINGS),
                "heldout_recs": []})
    return out


def fit_by_tag(tag: str) -> dict:
    hits = [f for f in embedder_fits() if f["tag"] == tag]
    if len(hits) != 1:
        raise P5Error(f"no registered embedder fit {tag}")
    return hits[0]


def ft_source(tag: str) -> str:
    return f"ftemb_{tag}"


def check_not_trained_on(fit: dict, rids: Iterable[str]) -> None:
    """PREREG: an embedder is NEVER used on a recording it was trained on."""
    bad = sorted(set(fit["train_recs"]) & set(rids))
    if bad:
        raise P5Error(f"embedder {fit['tag']} was trained on {bad}; refusing to use it there")


def inner_units(cell: str, fi: str) -> list[tuple[str, str]]:
    """(cache source, rid) of cell's pooled inner set for outer fold ``fi`` (8 recordings)."""
    if fi not in FOLDS:
        raise P5Error(f"unknown fold {fi}")
    _, emb = CELLS[cell]
    out = []
    for fj in sorted(FOLDS):
        if fj == fi:
            continue
        for rid in FOLDS[fj]:
            if emb == "stock":
                out.append(("zeroshot", rid))
                continue
            fit = fit_by_tag(f"p5emb_{cv.set_id((fi, fj))}")
            check_not_trained_on(fit, [rid, *FOLDS[fi]])
            out.append((ft_source(fit["tag"]), rid))
    return out


def outer_source(cell: str, fi: str) -> str:
    if CELLS[cell][1] == "stock":
        return "zeroshot"
    fit = fit_by_tag(f"p5emb_refit_{fi}")
    check_not_trained_on(fit, FOLDS[fi])
    return ft_source(fit["tag"])


# ----------------------------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------------------------
def select_within(table: dict[Config, dict[str, dict]], rids: Sequence[str] | None = None,
                  tol_pp: float = SELECT_TOL_PP) -> dict:
    """PREREG_phase5: the SIMPLEST config whose pooled DER is within ``tol_pp`` of the minimum."""
    if not table:
        raise P5Error("empty selection table")
    rids = sorted(rids if rids is not None else next(iter(table.values())))
    if not rids:
        raise P5Error("no recordings to select on")
    der = {c: p3b.pooled_der(v, rids) for c, v in table.items()}
    m = min(der.values())
    argmin = min((c for c in table if der[c] <= m + cv.TIE_TOL), key=cfg_simplicity)
    eligible = [c for c in table if der[c] <= m + tol_pp / 100 + 1e-12]
    return {"selected": min(eligible, key=cfg_simplicity), "argmin": argmin, "der": der, "eligible": eligible,
            "n": len(rids)}


def loro_stability(table: dict[Config, dict[str, dict]]) -> dict:
    rids = sorted(next(iter(table.values())))
    full = select_within(table, rids)
    per = {}
    for r in rids:
        x = select_within(table, [q for q in rids if q != r])
        per[r] = {"selected": x["selected"].key, "argmin": x["argmin"].key}
    return {"dropped": per, "n": len(rids),
            "selected_recovered": sum(v["selected"] == full["selected"].key for v in per.values()),
            "argmin_recovered": sum(v["argmin"] == full["argmin"].key for v in per.values())}


def joint_mode(keys: Sequence[str]) -> dict:
    """Final configuration = joint mode of the five fold selections; ties -> simpler order."""
    cnt: dict[str, int] = {}
    for k in keys:
        cnt[k] = cnt.get(k, 0) + 1
    top = max(cnt.values())
    tied = [k for k, v in cnt.items() if v == top]
    best = min(tied, key=lambda k: cfg_simplicity(p3b.parse_key(k)))
    return {"config": best, "count": top, "of": len(keys), "tied": sorted(tied), "counts": cnt}


def dev_adoption(der_inc: float, der_chal: float, count_inc: int, count_chal: int) -> dict:
    """Challenger replaces incumbent iff corpus DER (collar 0, no adapter) improves by >= 1.0 pp AND
    sum|count err| does not increase by more than 10 %."""
    delta_pp = 100 * (der_chal - der_inc)
    c1 = bool(delta_pp <= -DEV_MIN_IMPROVEMENT_PP + 1e-9)
    c2 = bool(count_chal <= (1 + DEV_MAX_COUNT_INCREASE) * count_inc + 1e-9)
    return {"delta_pp": delta_pp, "der_improves_ge_1pp": c1, "count_inc": count_inc, "count_chal": count_chal,
            "count_not_up_gt_10pct": c2, "adopt_challenger": c1 and c2,
            "grade": "development-grade; no significance claim"}


def holdout_claim(boot0: dict) -> dict:
    """Claim 'challenger better on the hold-out' only if delta <= -1.0 pp and the 97.5th pct < 0."""
    d = 100 * boot0["delta_b_minus_a"]
    c1 = bool(d <= -HOLDOUT_MIN_IMPROVEMENT_PP + 1e-9)
    c2 = bool(boot0["ci_high"] < 0)
    return {"delta_pp": d, "ci_low_pp": 100 * boot0["ci_low"], "ci_high_pp": 100 * boot0["ci_high"],
            "delta_le_-1pp": c1, "p97_5_lt_0": c2, "claim_challenger_better": c1 and c2,
            "verdict": ("challenger better on the hold-out" if c1 and c2 else
                        "no demonstrated hold-out benefit of the challenger over the incumbent")}


def speaker_bin(k: int) -> str:
    for name, lo, hi in SPK_BINS:
        if lo <= k <= hi:
            return name
    raise P5Error(f"speaker count {k} outside the registered bins")


from bnd.link import floor_adapter  # noqa: E402,F401  moved to bnd.link for the release; identical behaviour


# ----------------------------------------------------------------------------------------------
# hold-out subset
# ----------------------------------------------------------------------------------------------
def derive_subset(seed: int = SUBSET_SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for k in SUBSET_K:
        idx = sorted(int(i) for i in rng.choice(SUBSET_POOL, size=SUBSET_PER_K, replace=False))
        out += [f"{k}_speaker/simple_{k}spk_{i:03d}" for i in idx]
    return out


def load_subset(path: Path) -> list[str]:
    """docs/phase5_holdout_subset.txt, which must equal the PREREG rule exactly (order included)."""
    stems = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    rule = derive_subset()
    if stems != rule:
        raise P5Error(f"{path} does not match the PREREG subset rule (seed {SUBSET_SEED})")
    if len(set(stems)) != 60 or len({rid_of(s) for s in stems}) != 60:
        raise P5Error("hold-out subset must be 60 unique mixtures")
    return stems


def rid_of(stem: str) -> str:
    return stem.split("/")[-1]


def k_of(stem: str) -> int:
    m = re.fullmatch(r"(\d+)_speaker/simple_(\d+)spk_(\d{3})", stem)
    if not m or m.group(1) != m.group(2):
        raise P5Error(f"bad hold-out stem {stem!r}")
    return int(m.group(1))


# ----------------------------------------------------------------------------------------------
# context + ledger
# ----------------------------------------------------------------------------------------------
@dataclass
class P5Ctx(p3b.P3bCtx):
    cv_dir: Path = REPO / "runs/p5"
    start_h: float = PHASE_START_H
    allowance_h: float = ALLOWANCE_H
    budget_h: float = BUDGET_GPU_H
    p3b_dir: Path = REPO / "runs/p3b"
    holdout_dir: Path = REPO / "data/holdout_synth"
    holdout_sha_file: Path = REPO / "data/holdout_synth_SHA256.txt"
    subset_file: Path = REPO / "docs/phase5_holdout_subset.txt"

    @property
    def embedders_root(self) -> Path:
        return self.cv_dir / "embedders"

    @property
    def ho(self) -> Path:
        return self.cv_dir / "holdout"

    @property
    def ho_cache(self) -> Path:
        return self.ho / "cache"

    @property
    def ho_rttm(self) -> Path:
        return self.ho / "rttm"

    @property
    def dev_results_path(self) -> Path:
        return self.cv_dir / "dev_results.json"

    @property
    def final_configs_path(self) -> Path:
        return self.cv_dir / "final_configs.json"

    @property
    def validation_path(self) -> Path:
        return self.cv_dir / "holdout_validation.json"

    @property
    def ho_index_path(self) -> Path:
        return self.ho / "index.json"

    @property
    def ho_frozen_path(self) -> Path:
        return self.cv_dir / "holdout_frozen.json"

    @property
    def ho_scored_path(self) -> Path:
        return self.cv_dir / "holdout_scored.jsonl"

    @property
    def ho_results_path(self) -> Path:
        return self.cv_dir / "holdout_results.json"


class P5Ledger(p3b.P3bLedger):
    """P3bLedger (abort before launch above 12 h cumulative, WARNING above start + 5.5 h) with phase-5
    tagging and no double counting of bnd.embed_ft's own in-process entries."""

    _pending_n0: int | None = None

    def used_h(self) -> float:
        ents = self.entries()
        cov = {i for e in ents for i in e.get("covers_lines", [])}
        if self._pending_n0 is not None:     # job in flight: its child's self entries are covered by the job's
            cov |= {i for i in range(self._pending_n0, len(ents)) if "ts_start" not in ents[i]}   # own entry
        return self.ctx.start_h + sum(e.get("gpu_wall_s", 0.0) for i, e in enumerate(ents) if i not in cov) / 3600

    def run_est(self, cmd: list, kind: str, tag: str, log: Path, est_s: float, audio_s: float = 0.0) -> dict:
        n0 = len(self.entries())
        self._pending_n0 = n0
        try:
            return super().run_est(cmd, kind, tag, log, est_s, audio_s)
        finally:
            self._pending_n0 = None
            self._retag(n0, tag)

    def _retag(self, n0: int, tag: str) -> None:
        if not self.path.exists():
            return
        lines = self.path.read_text().splitlines()
        ents = [json.loads(l) if l.strip() else None for l in lines]
        j = next((i for i in range(len(ents) - 1, n0 - 1, -1)
                  if ents[i] and ents[i].get("tag") == tag and "ts_start" in ents[i]), None)
        if j is None:
            return
        e = ents[j]
        e["phase"] = "5"
        e["covers_lines"] = [i for i in range(n0, j) if ents[i] and "ts_start" not in ents[i]]
        lines[j] = json.dumps(e)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(self.path)


@dataclass
class Rates:
    base: p3b.Rates
    embfit_s: float = DEF_EMBFIT_S
    ftn_fit_s: float = p3b.DEF_FIT_S

    def nemo(self, audio_s: float, core_s: int) -> float:
        return self.base.infer(audio_s, 0, embed=False, s80=core_s > 20)

    def embed(self, audio_s: float, n_theta: int) -> float:
        return self.base.embed(audio_s, n_theta)


def rates(ctx: P5Ctx) -> Rates:
    r = Rates(p3b.phase2_rates(ctx))
    led = P5Ledger(ctx).entries()
    fits = [e["gpu_wall_s"] for e in led if e.get("kind") == "embfit" and e.get("returncode") == 0] + \
        [e["gpu_wall_s"] + p3b.JOB_FIXED_S for e in led if e.get("kind") == "embed_ft-fit" and e.get("gpu_wall_s")]
    if fits:
        r.embfit_s = max(fits) * p3b.HEADROOM
    trains = [e["gpu_wall_s"] for p in (ctx.p2_dir / "gpu_ledger.jsonl", ctx.p3b_dir / "gpu_ledger.jsonl")
              if p.exists() for e in map(json.loads, filter(str.strip, p.read_text().splitlines()))
              if e.get("kind") == "train" and e.get("returncode") == 0 and e.get("gpu_wall_s")]
    if trains:
        r.ftn_fit_s = max(trains) * p3b.HEADROOM
    return r


def gpu_job(ctx: P5Ctx, ledger: P5Ledger | None, cmd: list, kind: str, tag: str, est_s: float,
            audio_s: float = 0.0) -> dict | None:
    if ctx.dry_run:
        cv.say_dry(ctx, "gpu", cmd)
        ctx.out(f"DRY-RUN   ({kind} {tag}: est {est_s / 3600:.3f} GPU-h)")
        return None
    assert ledger is not None
    return ledger.run_est(cmd, kind, tag, ctx.logs / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', tag)}.log", est_s, audio_s)


# ----------------------------------------------------------------------------------------------
# caches
# ----------------------------------------------------------------------------------------------
def cdir(ctx: P5Ctx, source: str, session: int, rid: str) -> Path:
    return p3b.cache_dir(ctx, source, session, rid)


def base_sha(ctx: P5Ctx) -> str:
    return cv.file_sha(ctx, ctx.base_model)


def emb_state(d: Path, theta: float) -> str | None:
    """'stock' / FT checkpoint sha256 of the CURRENT embeddings of cache ``d`` at ``theta`` (None if missing
    or not aligned with the tracks file)."""
    from bnd import embed, infer

    if not infer.emb_is_current(d, theta):
        return None
    try:
        m = json.loads(infer.emb_paths(d, theta)[1].read_text())
    except (OSError, ValueError):
        return None
    if m.get("model") != embed.MODEL_ID or m.get("revision") != embed.MODEL_REV:
        return None
    ck = m.get("embedder_ckpt")
    return "stock" if ck is None else ck.get("sha256")


def tracks_ok(d: Path, thetas: Iterable[float]) -> bool:
    return all((d / f"tracks_{t:.2f}.json").is_file() for t in thetas)


def _mirror(dst: Path, src: Path, names: Iterable[str], forbid: Path | None = None) -> None:
    if forbid is not None and dst.resolve().is_relative_to(forbid.resolve()):
        raise P5Error(f"mirror {dst} would lie inside {forbid}")
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        if not (src / n).exists():
            raise P5Error(f"mirror source {src / n} missing")
        p3b._symlink(dst / n, src / n)


def base_names(src: Path) -> list[str]:
    ws = json.loads((src / "windows.json").read_text())
    return ["meta.json", "windows.json"] + [f"probs_{w['window']}.npy" for w in ws]


def mirror_s20(ctx: P5Ctx, rid: str) -> Path:
    """runs/p5/cache/zeroshot/s20/<rid>/ = read-only symlinks to the verified phase-3b zero-shot S=20 cache
    (probs, tracks and STOCK embeddings for theta_act 0.25/0.40)."""
    src = ctx.p3b_dir / "cache/zeroshot/s20" / rid
    names = base_names(src) + [n for t in THETA_ACT for n in (f"tracks_{t:.2f}.json", f"emb_{t:.2f}.npy",
                                                               f"emb_{t:.2f}_meta.json")] + ["emb_meta.json"]
    dst = cdir(ctx, "zeroshot", 20, rid)
    _mirror(dst, src, names, forbid=ctx.p3b_dir)
    return dst


def mirror_for_ft(ctx: P5Ctx, source: str, session: int, rid: str) -> Path:
    """FT embedding cache: symlinks to the zero-shot probs/windows/meta/tracks, real emb files."""
    src = cdir(ctx, "zeroshot", session, rid)
    dst = cdir(ctx, source, session, rid)
    _mirror(dst, src, base_names(src) + [f"tracks_{t:.2f}.json" for t in THETA_ACT])
    for t in THETA_ACT:
        for n in (f"emb_{t:.2f}.npy", f"emb_{t:.2f}_meta.json"):
            if (dst / n).is_symlink():
                raise P5Error(f"{dst / n} is a symlink; FT embeddings must be real files")
    if (dst / "emb_meta.json").is_symlink():
        raise P5Error(f"{dst / 'emb_meta.json'} is a symlink (bnd.embed would write through it)")
    return dst


def s20_source_ok(ctx: P5Ctx, rid: str) -> bool:
    from bnd import infer

    src = ctx.p3b_dir / "cache/zeroshot/s20" / rid
    return (infer.cache_complete(src, {"system": "zeroshot", "core_s": 20.0, "model_sha256": base_sha(ctx)})
            and all(emb_state(src, t) == "stock" for t in THETA_ACT))


def s120_ok(ctx: P5Ctx, rid: str) -> bool:
    from bnd import infer

    d = cdir(ctx, "zeroshot", 120, rid)
    return infer.cache_complete(d, {"system": "zeroshot", "core_s": 120.0, "model_sha256": base_sha(ctx)}) \
        and tracks_ok(d, THETA_ACT)


def track_shas(ctx: P5Ctx, source: str, session: int, rids: Iterable[str]) -> dict:
    return {f"s{session}/{r}": {f"{t:.2f}": cv.file_sha(ctx, cdir(ctx, source, session, r) / f"tracks_{t:.2f}.json",
                                                         True) for t in THETA_ACT} for r in rids}


def embed_job_cmd(ctx: P5Ctx, dirs: Sequence[Path], rids: Sequence[str], thetas: Sequence[float],
                  ckpt: Path | None) -> list:
    cmd = [ctx.embed_python, "-m", "bnd.p5", "embed-job", "--recordings", *rids, "--theta-act",
           *[f"{t:.2f}" for t in thetas]]
    for d in dirs:
        cmd += ["--dir", d]
    return cmd + (["--ckpt", ckpt] if ckpt else [])


# ----------------------------------------------------------------------------------------------
# internal GPU entry points
# ----------------------------------------------------------------------------------------------
def embed_job_main(argv: Sequence[str]) -> int:
    """(embed venv) bnd.embed on every ``--dir`` (same recordings), stock or ``--ckpt`` FT weights."""
    ap = argparse.ArgumentParser(prog="bnd.p5 embed-job")
    ap.add_argument("--dir", type=Path, action="append", required=True)
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--theta-act", nargs="+", required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    a = ap.parse_args(argv)
    from bnd import embed

    for d in a.dir:
        args = ["--cache-dir", str(d), "--recordings", *a.recordings, "--theta-act", *a.theta_act, "--device", "cuda",
                "--report", str(d / "embed_report.json")]
        if a.ckpt:
            args += ["--embedder-ckpt", str(a.ckpt)]
        embed.main(args)
    return 0


def holdout_nemo_job_main(argv: Sequence[str]) -> int:
    """(NeMo venv) native Nemotron probs of hold-out mixtures via bnd.diarize.run_model (the model is loaded
    once per job: infer.load_model is memoised in this process only)."""
    import functools

    ap = argparse.ArgumentParser(prog="bnd.p5 holdout-nemo-job")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--core-s", type=float, required=True)
    ap.add_argument("--system-dir", type=Path, required=True)
    ap.add_argument("--item", nargs="+", required=True, help="rid=wav")
    a = ap.parse_args(argv)
    from bnd import diarize, infer

    infer.load_model = functools.lru_cache(maxsize=1)(infer.load_model)
    sha = infer.sha256_file(a.model)
    rep = {"model": str(a.model), "model_sha256": sha, "core_s": a.core_s, "recordings": {}}
    for it in a.item:
        rid, wav = it.split("=", 1)
        d = a.system_dir / rid
        if infer.cache_complete(d, {"model_sha256": sha, "core_s": a.core_s}):
            rep["recordings"][rid] = {"cached": True}
            continue
        rep["recordings"][rid] = diarize.run_model(a.model, Path(wav), rid, d, a.core_s)
        print(f"[p5 holdout-nemo] {rid}: {json.dumps(rep['recordings'][rid])}", flush=True)
    infer.write_json_atomic(a.system_dir / f"nemo_report_{os.getpid()}.json", rep)
    return 0


def community1_job_main(argv: Sequence[str]) -> int:
    """(embed venv) pyannote speaker-diarization-community-1 @COMMUNITY1_REV, untuned, EXCLUSIVE output
    (as scripts/diag/community1_infer.py) for each ``rid=wav``."""
    import time

    import torch
    from pyannote.audio import Pipeline

    ap = argparse.ArgumentParser(prog="bnd.p5 community1-job")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--item", nargs="+", required=True)
    a = ap.parse_args(argv)
    tok = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    pipe = Pipeline.from_pretrained(COMMUNITY1_ID, token=tok, revision=COMMUNITY1_REV)
    pipe.to(torch.device("cuda"))
    torch.cuda.reset_peak_memory_stats()
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {"pipeline": COMMUNITY1_ID, "revision": COMMUNITY1_REV, "output": "exclusive_speaker_diarization",
           "recordings": {}}
    for it in a.item:
        rid, wav = it.split("=", 1)
        dst = a.out / f"{rid}.rttm"
        if dst.exists():
            continue
        t0 = time.time()
        res = pipe(wav)
        segs = [S.Seg(round(seg.start, 3), round(seg.end, 3), spk)
                for seg, _, spk in res.exclusive_speaker_diarization.itertracks(yield_label=True)]
        tmp = dst.with_name(dst.name + ".tmp")
        S.write_rttm(tmp, rid, segs)
        tmp.replace(dst)
        rep["recordings"][rid] = {"wall_s": time.time() - t0, "speakers": len({s.speaker for s in segs})}
        print(f"[p5 community1] {rid}: {json.dumps(rep['recordings'][rid])}", flush=True)
    rep["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (a.out / f"_community1_report_{os.getpid()}.json").write_text(json.dumps(rep, indent=1))
    return 0


# ----------------------------------------------------------------------------------------------
# A-prep
# ----------------------------------------------------------------------------------------------
def a_prep_cmd(ctx: P5Ctx, rids: Sequence[str]) -> list:
    return [ctx.python, "-m", "bnd.infer", "--system", "zeroshot", "--model", ctx.base_model, "--recordings", *rids,
            "--theta-act", *[f"{t:.2f}" for t in THETA_ACT], "--cache-root", ctx.cache_root, "--core-s", "120",
            "--session-subdir", "--no-embed"]


def cmd_a_prep(ctx: P5Ctx) -> dict:
    ledger = None if ctx.dry_run else P5Ledger(ctx)
    r = rates(ctx)
    rids = list(RECORDINGS)
    bad20 = [x for x in rids if not s20_source_ok(ctx, x)]
    if bad20:
        raise P5Error(f"phase-3b zero-shot S=20 cache incomplete/stale for {bad20} (runs/p3b/cache/zeroshot/s20)")
    todo = [x for x in rids if not s120_ok(ctx, x)]
    audio = sum(cv.durations(ctx, todo).values()) if todo else 0.0
    est = r.nemo(audio, 120) if todo else 0.0
    if ctx.dry_run:
        ctx.out(f"DRY-RUN A-prep: S=20 = symlinks to runs/p3b/cache/zeroshot/s20 (10 recordings, verified); S=120 "
                f"inference + tracks theta {list(THETA_ACT)} for {todo}")
        if todo:
            gpu_job(ctx, None, a_prep_cmd(ctx, todo), "infer", "aprep:zeroshot_s120", est, audio)
        return {"projected_gpu_h": round(est / 3600, 3)}
    for x in rids:
        mirror_s20(ctx, x)
    if todo:
        gpu_job(ctx, ledger, a_prep_cmd(ctx, todo), "infer", "aprep:zeroshot_s120", est, audio)
    bad = [x for x in rids if not s120_ok(ctx, x)]
    if bad:
        raise P5Error(f"S=120 cache incomplete after inference: {bad}")
    inputs = {"base_model_sha256": base_sha(ctx), "recordings": rids, "code": cv.code_shas(ctx, "infer")}
    cv.write_marker(ctx, "aprep", "zeroshot", inputs, {**track_shas(ctx, "zeroshot", 20, rids),
                                                        **track_shas(ctx, "zeroshot", 120, rids)})
    ctx.out("[p5] A-prep done")
    return {"projected_gpu_h": round(est / 3600, 3)}


# ----------------------------------------------------------------------------------------------
# A-fits
# ----------------------------------------------------------------------------------------------
def fit_dir(ctx: P5Ctx, tag: str) -> Path:
    return ctx.embedders_root / tag


def fit_inputs(ctx: P5Ctx, spec: dict) -> dict:
    from bnd import embed_ft

    return {"tag": spec["tag"], "train_recs": spec["train_recs"], "recipe": embed_ft.RECIPE,
            "refs": {r: cv.file_sha(ctx, ctx.ref_dir / f"{r}.rttm", True) for r in spec["train_recs"]},
            "spans": cv.file_sha(ctx, embed_ft.SPANS_PATH, True), "code": cv.code_shas(ctx, "embed_ft", "embed")}


def fit_manifest(ctx: P5Ctx, tag: str) -> dict | None:
    p = fit_dir(ctx, tag) / "manifest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def fit_done(ctx: P5Ctx, spec: dict) -> dict | None:
    """The fit's manifest if its checkpoint is complete (sha matches) AND it is the registered fit: its tag,
    training recordings, sealed train_binding, recipe and step count equal the PREREG ones. A complete
    checkpoint under a registered tag that was trained any other way (e.g. ``bnd.embed_ft --steps 50``)
    raises instead of being adopted or silently replaced."""
    from bnd import embed_ft

    m = fit_manifest(ctx, spec["tag"])
    if m is None or m.get("train_recordings") != spec["train_recs"]:
        return None
    ck = fit_dir(ctx, spec["tag"]) / m.get("checkpoint", "embedder.pt")
    if cv.file_sha(ctx, ck, True) != m.get("sha256"):
        return None
    bad = [k for k, ok in (("tag", m.get("tag") == spec["tag"]),
                           ("train_binding", m.get("train_binding") == embed_ft.train_binding(m)),
                           ("recipe", m.get("recipe") == embed_ft.RECIPE),
                           ("steps", m.get("steps") == embed_ft.RECIPE["steps"])) if not ok]
    if bad:
        raise P5Error(f"embedder fit {spec['tag']} at {fit_dir(ctx, spec['tag'])} is not the registered fit "
                      f"(mismatch: {bad}; steps {m.get('steps')} vs {embed_ft.RECIPE['steps']}); refusing to adopt it. "
                      f"Move it aside by hand, then rerun")
    return m


def fit_cmd(ctx: P5Ctx, spec: dict) -> list:
    return [ctx.embed_python, "-m", "bnd.embed_ft", "--tag", spec["tag"], "--train-recs", *spec["train_recs"]]


def ensure_fit(ctx: P5Ctx, ledger: P5Ledger | None, r: Rates, spec: dict) -> float:
    from bnd import embed_ft

    if Path(embed_ft.OUT_ROOT).resolve() != ctx.embedders_root.resolve():
        raise P5Error(f"bnd.embed_ft writes to {embed_ft.OUT_ROOT}, the driver expects {ctx.embedders_root}")
    inputs = fit_inputs(ctx, spec)
    st, mk = cv.marker_status(ctx, "embfit", spec["tag"], inputs)
    man = fit_done(ctx, spec)
    if st == "done" and man and man["sha256"] == mk["outputs"]["sha256"]:
        return 0.0
    if st == "stale" and not ctx.allow_stale:
        raise cv.StaleError(f"embedder fit {spec['tag']}: done-marker inputs changed; rerun with --allow-stale")
    est = 0.0 if man else r.embfit_s
    if ctx.dry_run:
        if not man:
            gpu_job(ctx, None, fit_cmd(ctx, spec), "embfit", f"embfit:{spec['tag']}", est)
        return est
    if not man:
        d = cv.move_aside(fit_dir(ctx, spec["tag"]))
        if d:
            ctx.out(f"[p5] moved incomplete embedder dir -> {d}")
        gpu_job(ctx, ledger, fit_cmd(ctx, spec), "embfit", f"embfit:{spec['tag']}", est)
        man = fit_done(ctx, spec)
        if man is None:
            raise P5Error(f"embedder fit {spec['tag']} did not produce a valid manifest/checkpoint")
    cv.write_marker(ctx, "embfit", spec["tag"], inputs, {"sha256": man["sha256"], "dir": str(fit_dir(ctx, spec["tag"])),
                                                        "train_recordings": man["train_recordings"]})
    return est


def ftemb_ok(ctx: P5Ctx, spec: dict, sha: str) -> bool:
    return all(emb_state(cdir(ctx, ft_source(spec["tag"]), s, r), t) == sha
               for s in SESSIONS for r in spec["heldout_recs"] for t in THETA_ACT)


def ensure_ftemb(ctx: P5Ctx, ledger: P5Ledger | None, r: Rates, spec: dict) -> float:
    """FT embeddings of the fit's held-out recordings, S {20, 120} x theta_act {0.25, 0.40}."""
    from bnd import embed_ft

    rids = spec["heldout_recs"]
    if not rids:
        return 0.0
    check_not_trained_on(spec, rids)
    src = ft_source(spec["tag"])
    audio = sum(cv.durations(ctx, rids).values())
    est = r.embed(audio * len(SESSIONS), len(THETA_ACT))
    man = fit_done(ctx, spec)
    if ctx.dry_run:
        if man is None or not ftemb_ok(ctx, spec, man["sha256"]):
            dirs = [ctx.cache_root / src / f"s{s}" for s in SESSIONS]
            gpu_job(ctx, None, embed_job_cmd(ctx, dirs, rids, THETA_ACT, fit_dir(ctx, spec["tag"])), "embed",
                    f"ftemb:{spec['tag']}", est, audio)
            return est
        return 0.0
    if man is None:
        raise P5Error(f"{spec['tag']}: embedder not fitted")
    embed_ft.assert_not_trained_on(fit_dir(ctx, spec["tag"]), rids)
    for s in SESSIONS:
        for x in rids:
            mirror_for_ft(ctx, src, s, x)
    inputs = {"ckpt_sha256": man["sha256"], "recordings": rids, "sessions": list(SESSIONS),
              "tracks": {s: track_shas(ctx, src, s, rids) for s in SESSIONS}, "code": cv.code_shas(ctx, "embed")}
    st, _ = cv.marker_status(ctx, "ftemb", spec["tag"], inputs)
    if st == "done" and ftemb_ok(ctx, spec, man["sha256"]):
        return 0.0
    dirs = [ctx.cache_root / src / f"s{s}" for s in SESSIONS]
    gpu_job(ctx, ledger, embed_job_cmd(ctx, dirs, rids, THETA_ACT, fit_dir(ctx, spec["tag"])), "embed",
            f"ftemb:{spec['tag']}", est, audio)
    if not ftemb_ok(ctx, spec, man["sha256"]):
        raise P5Error(f"{spec['tag']}: FT embeddings missing/stale after the embed job")
    cv.write_marker(ctx, "ftemb", spec["tag"], inputs, {
        f"s{s}/{x}/{t:.2f}": cv.file_sha(ctx, cdir(ctx, src, s, x) / f"emb_{t:.2f}.npy")
        for s in SESSIONS for x in rids for t in THETA_ACT})
    return est


def ensure_stock_s120(ctx: P5Ctx, ledger: P5Ledger | None, r: Rates) -> float:
    rids = [x for x in RECORDINGS if not all(emb_state(cdir(ctx, "zeroshot", 120, x), t) == "stock" for t in THETA_ACT)]
    if not rids:
        return 0.0
    audio = sum(cv.durations(ctx, rids).values())
    est = r.embed(audio, len(THETA_ACT))
    gpu_job(ctx, ledger, embed_job_cmd(ctx, [ctx.cache_root / "zeroshot/s120"], rids, THETA_ACT, None), "embed",
            "stockemb:zeroshot_s120", est, audio)
    if not ctx.dry_run and any(emb_state(cdir(ctx, "zeroshot", 120, x), t) != "stock" for x in rids for t in THETA_ACT):
        raise P5Error("stock S=120 embeddings missing/stale after the embed job")
    return est


def cmd_a_fits(ctx: P5Ctx) -> dict:
    if not ctx.dry_run and cv.read_marker(ctx, "aprep", "zeroshot") is None:
        raise P5Error("A-prep not done")
    ledger = None if ctx.dry_run else P5Ledger(ctx)
    r = rates(ctx)
    tot = ensure_stock_s120(ctx, ledger, r)
    specs = embedder_fits()
    assert len(specs) == 16
    for spec in specs:
        tot += ensure_fit(ctx, ledger, r, spec)
        tot += ensure_ftemb(ctx, ledger, r, spec)
    s = {"fits": len(specs), "projected_gpu_h": round(tot / 3600, 3), "embfit_s_assumed": r.embfit_s,
         "rates": r.base.basis}
    ctx.out(("DRY-RUN " if ctx.dry_run else "[p5] ") + "A-fits " + json.dumps(s))
    return s


# ----------------------------------------------------------------------------------------------
# A-search
# ----------------------------------------------------------------------------------------------
def search_units() -> list[tuple[str, str]]:
    u = {("zeroshot", r) for r in RECORDINGS}
    for cell in CELLS:
        for fi in FOLDS:
            u |= set(inner_units(cell, fi))
    return sorted(u)


def search_tasks(ctx: P5Ctx) -> list[dict]:
    tasks = []
    for source, rid in search_units():
        for s in SESSIONS:
            for t in THETA_ACT:
                cfgs = configs_for(s, t)
                tasks.append({"name": f"{source}__s{s}__{rid}__{t:.2f}", "system": source, "rid": rid, "session": s,
                              "theta": t, "cfgs": cfgs, "outs": [p3b.hyp_path(ctx, source, c, rid) for c in cfgs],
                              "report": ctx.hyp_root / source / "_reports" / f"s{s}__{rid}__{t:.2f}.json"})
    return tasks


def select_cell(ctx: P5Ctx, fi: str, cell: str) -> dict:
    units = inner_units(cell, fi)
    cfgs = cell_configs(cell)
    items, spans = [], []
    for c in cfgs:
        its = p3b.pool_items(ctx, units, fi, c)
        spans.append((len(items), len(items) + len(its)))
        items += its
    missing = [str(p) for _, p in items if not p.is_file()]
    if missing:
        raise P5Error(f"select {fi}/{cell}: {len(missing)} inner predictions missing, e.g. {missing[:3]}")
    sc = p3b.score_items(ctx, fi, items, [S.PRIMARY_COLLAR])
    table = {c: {x.recording_id: {"errors": x.errors, "total": x.total, "der": x.der}
                 for x in (sc[i][S.PRIMARY_COLLAR] for i in range(a, b))} for c, (a, b) in zip(cfgs, spans)}
    sel = select_within(table)
    return {"sources": {rid: src for src, rid in units}, "selected": sel["selected"].key,
            "selected_config": sel["selected"].as_dict(), "selected_der": sel["der"][sel["selected"]],
            "argmin": sel["argmin"].key, "argmin_der": sel["der"][sel["argmin"]],
            "eligible": sorted(c.key for c in sel["eligible"]), "loro_stability": loro_stability(table),
            "configs": {c.key: dict(**c.as_dict(), der=sel["der"][c], per_recording={x: v["der"] for x, v in table[c].items()},
                                    errors_s=sum(v["errors"] for v in table[c].values()),
                                    total_s=sum(v["total"] for v in table[c].values())) for c in cfgs}}


def check_fits_done(ctx: P5Ctx, kinds: Iterable[str]) -> None:
    bad = [s["tag"] for s in embedder_fits() if s["kind"] in kinds
           and (cv.read_marker(ctx, "embfit", s["tag"]) is None or cv.read_marker(ctx, "ftemb", s["tag"]) is None)]
    if bad:
        raise P5Error(f"A-fits not done for {bad[:5]}")


def cmd_a_search(ctx: P5Ctx) -> dict | None:
    tasks = search_tasks(ctx)
    if ctx.dry_run:
        ctx.out(f"DRY-RUN A-search: link {len(tasks)} (source, S, theta_act) tasks x 10 configs = {len(tasks) * 10} "
                f"RTTMs (CPU, <= {ctx.jobs} workers); per fold x cell: 20 pooled inner DERs over 8 recordings "
                f"(collar 0), simplest within {SELECT_TOL_PP} pp, LORO. GPU-h: 0")
        for fi in sorted(FOLDS):
            ctx.out(f"DRY-RUN A-search {fi}: FT pool {sorted({s for s, _ in inner_units('s20_ft', fi)})}")
        return None
    if ctx.frozen_path.exists():
        p3b.verify_frozen_p3b(ctx)
        ctx.out("[p5] Part A frozen; selection is locked")
        return json.loads(ctx.selection_path.read_text())
    cv.guard_after_scoring(ctx, "A-search")
    if cv.read_marker(ctx, "aprep", "zeroshot") is None:
        raise P5Error("A-prep not done")
    check_fits_done(ctx, ("inner",))
    n = p3b.run_link_tasks(ctx, tasks)
    ctx.out(f"[p5] linked {n} new tasks of {len(tasks)}")
    sel = {"created": cv.now(), "prereg": "docs/PREREG_phase5.md", "evidence": EVIDENCE,
           "space": {"theta_act": list(THETA_ACT), "theta_clu": list(THETA_CLU), "gap_fill_s": list(GAPS),
                     "linker": "avg", "sessions": {c: CELLS[c][0] for c in CELLS}, "configs_per_cell": 20},
           "rule": f"simplest config within {SELECT_TOL_PP} pp of the minimum pooled inner DER (collar 0); tie order "
                   "g 0<5, theta_act 0.40<0.25, theta_clu 0.60<0.55<0.65<0.45<0.75 (lexicographic)",
           "collar": S.PRIMARY_COLLAR, "folds": {}}
    for fi in sorted(FOLDS):
        f = {"inner_recs": sorted(r for x in FOLDS if x != fi for r in FOLDS[x])}
        for cell in CELLS:
            f[cell] = select_cell(ctx, fi, cell)
            ctx.out(f"[p5] {fi} {cell}: {f[cell]['selected']} ({100 * f[cell]['selected_der']:.2f}%), argmin "
                    f"{f[cell]['argmin']} ({100 * f[cell]['argmin_der']:.2f}%), LORO "
                    f"{f[cell]['loro_stability']['selected_recovered']}/8")
        sel["folds"][fi] = f
    cv.write_json(ctx.selection_path, sel)
    return sel


# ----------------------------------------------------------------------------------------------
# A-outer / A-freeze
# ----------------------------------------------------------------------------------------------
def outer_jobs(sel: dict) -> list[dict]:
    return [{"cell": cell, "fold": fi, "source": outer_source(cell, fi),
             "cfg": p3b.parse_key(sel["folds"][fi][cell]["selected"]), "recs": list(FOLDS[fi])}
            for cell in CELLS for fi in sorted(FOLDS)]


def outer_path(ctx: P5Ctx, system: str, fi: str, rid: str) -> Path:
    return ctx.outer_root / system / fi / f"{rid}.rttm"


def ensure_outer(ctx: P5Ctx, j: dict) -> dict | None:
    per = {rid: p3b.cache_shas(ctx, j["source"], j["cfg"].session, rid, j["cfg"].theta_act) for rid in j["recs"]}
    inputs = {"source": j["source"], "config": j["cfg"].key, "recs": j["recs"], "cache": per,
              "code": cv.code_shas(ctx, "link", "score", "p3b", "p5")}
    name = f"{j['cell']}__{j['fold']}"
    st, m = cv.marker_status(ctx, "outer", name, inputs)
    outs = {s: {r: outer_path(ctx, s, j["fold"], r) for r in j["recs"]} for s in (j["cell"], j["cell"] + FLOOR)}
    if st == "done" and all(cv.file_sha(ctx, p, True) == m["outputs"][s][r] for s, d in outs.items() for r, p in d.items()):
        return m["outputs"]
    if ctx.dry_run:
        ctx.out(f"DRY-RUN [cpu] outer {j['cell']} {j['fold']} {j['source']} {j['cfg'].key} (+ floor adapter)")
        return None
    if j["source"].startswith("ftemb_"):
        check_not_trained_on(fit_by_tag(j["source"][len("ftemb_"):]), j["recs"])
    shas: dict[str, dict] = {j["cell"]: {}, j["cell"] + FLOOR: {}}
    for rid in j["recs"]:
        p = outs[j["cell"]][rid]
        r = p3b._link_task((str(cdir(ctx, j["source"], j["cfg"].session, rid)), rid, j["cfg"].theta_act, [j["cfg"].key],
                            [str(p)], str(p.parent / f"_report_{rid}.json"), "none", None))
        shas[j["cell"]][rid] = r[j["cfg"].key]
        fp = outs[j["cell"] + FLOOR][rid]
        p3b._write_rttm_atomic(fp, rid, floor_adapter(S.load_rttm_file(p, rid)))
        shas[j["cell"] + FLOOR][rid] = cv.file_sha(ctx, fp)
    cv.write_marker(ctx, "outer", name, inputs, shas)
    return shas


def cmd_a_outer(ctx: P5Ctx) -> None:
    if ctx.frozen_path.exists():
        p3b.verify_frozen_p3b(ctx)
        ctx.out("[p5] Part-A outer predictions are frozen; nothing to do")
        return
    cv.guard_after_scoring(ctx, "A-outer")
    if not ctx.selection_path.exists():
        if ctx.dry_run:
            ctx.out("DRY-RUN A-outer: selection.json missing; per fold x cell link the 2 outer recordings at the "
                    "selected config (stock: zero-shot cache; FT: the 8-recording refit embedder p5emb_refit_Fi), plus "
                    "floor-adapter copies (8 systems x 10 RTTMs, CPU, GPU-h 0)")
            return
        raise P5Error("selection.json missing; run A-search first")
    sel = json.loads(ctx.selection_path.read_text())
    if not ctx.dry_run:
        check_fits_done(ctx, ("refit",))
    jobs = outer_jobs(sel)
    res = [ensure_outer(ctx, j) for j in jobs]
    if ctx.dry_run or any(r is None for r in res):
        return
    idx = {"created": cv.now(), "selection_sha256": cv.file_sha(ctx, ctx.selection_path),
           "jobs": [{"cell": j["cell"], "fold": j["fold"], "source": j["source"], "config": j["cfg"].key,
                     "recs": j["recs"], "rttm_sha256": r} for j, r in zip(jobs, res)]}
    cv.write_json(ctx.outer_root / "index.json", idx)
    ctx.out(f"[p5] A-outer: {len(jobs)} jobs -> {ctx.outer_root / 'index.json'}")


def cmd_a_freeze(ctx: P5Ctx) -> dict | None:
    idx_p = ctx.outer_root / "index.json"
    if not idx_p.exists() or not ctx.selection_path.exists():
        raise P5Error("A-freeze refused: outer index or selection.json missing")
    idx = json.loads(idx_p.read_text())
    if idx.get("selection_sha256") != cv.file_sha(ctx, ctx.selection_path):
        raise P5Error("A-freeze refused: selection.json changed after the outer predictions were made")
    systems: dict[str, dict] = {s: {} for s in DEV_SYSTEMS}
    problems = []
    for j in idx["jobs"]:
        for system, shas in j["rttm_sha256"].items():
            for rid in j["recs"]:
                p = outer_path(ctx, system, j["fold"], rid)
                sha = cv.file_sha(ctx, p, True)
                if sha != shas[rid]:
                    problems.append(f"{p} missing or differs from the outer index")
                    continue
                systems[system][rid] = {"path": p3b.rel(ctx, p), "sha256": sha, "fold": j["fold"],
                                        "source_system": j["source"], "config": j["config"],
                                        "adapter": "floor" if system.endswith(FLOOR) else "none"}
    for s in systems:
        miss = sorted(set(RECORDINGS) - set(systems[s]))
        if miss:
            problems.append(f"{s}: missing outer predictions for {miss}")
    if problems:
        raise P5Error(f"A-freeze refused: {problems[:8]}")
    fz = {"created": cv.now(), "prereg": "docs/PREREG_phase5.md", "evidence": EVIDENCE,
          "selection_sha256": cv.file_sha(ctx, ctx.selection_path),
          "phase2_frozen_sha256": cv.file_sha(ctx, ctx.p2.frozen_path, True), "path_anchor": "runs/",
          "refit_embedders": {s["tag"]: (fit_manifest(ctx, s["tag"]) or {}).get("sha256")
                              for s in embedder_fits() if s["kind"] == "refit"},
          "systems": systems}
    if ctx.frozen_path.exists():
        old = p3b.verify_frozen_p3b(ctx)
        if old["systems"] != systems:
            raise cv.OuterLockError("already frozen with different outer predictions; refusing to re-freeze")
        ctx.out("[p5] already frozen; hashes verified")
        return old
    if cv.scored_entries(ctx):
        cv.guard_after_scoring(ctx, "A-freeze")
        fz["deviation"] = ctx.deviation
    if ctx.dry_run:
        ctx.out(f"DRY-RUN A-freeze {sum(len(v) for v in systems.values())} outer RTTMs")
        return None
    cv.write_json(ctx.frozen_path, fz)
    ctx.out(f"[p5] frozen {sum(len(v) for v in systems.values())} outer RTTMs -> {ctx.frozen_path}")
    return fz


# ----------------------------------------------------------------------------------------------
# A-score
# ----------------------------------------------------------------------------------------------
def cmd_a_score(ctx: P5Ctx) -> dict:
    fz = p3b.verify_frozen_p3b(ctx)
    ident = cv.frozen_identity(fz)
    ents = cv.scored_entries(ctx)
    if ents and ents[-1]["frozen_identity"] != ident and not fz.get("deviation"):
        raise cv.OuterLockError("OUTER LOCK: frozen.json differs from what was scored and records no deviation")
    sel = json.loads(ctx.selection_path.read_text())
    scores: dict[str, dict[float, list[S.RecScore]]] = {}
    for system, recs in fz["systems"].items():
        per_c: dict[float, list] = {c: [] for c in S.COLLARS}
        for fi in sorted(FOLDS):
            for x in p3b.score_items(ctx, fi, [(rid, p3b.unrel(ctx, recs[rid]["path"])) for rid in FOLDS[fi]], S.COLLARS):
                for c in S.COLLARS:
                    per_c[c].append(x[c])
        scores[system] = {c: sorted(v, key=lambda s: s.recording_id) for c, v in per_c.items()}
    cnt = p3b.speaker_counts(ctx, fz)
    res: dict = {"created": cv.now(), "frozen_created": fz["created"], "prereg": "docs/PREREG_phase5.md",
                 "evidence": EVIDENCE, "label": "DEVELOPMENT numbers (Part A)", "collars": list(S.COLLARS),
                 "bootstrap": {"n": ctx.bootstrap_n, "seed": DEV_BOOT_SEED}, "systems": {}}
    for system, per_c in scores.items():
        res["systems"][system] = {str(c): {"corpus": S.corpus_summary(sc), "per_recording": [s.as_dict() for s in sc]}
                                  for c, sc in per_c.items()}
        res["systems"][system]["sum_abs_count_error"] = cnt["systems"][system]["sum_abs_error"]
    res["speaker_count_errors"] = cnt
    res["comparisons"] = {f"{INCUMBENT}{a} -> {cell}{a}": cv.compare(scores[INCUMBENT + a], scores[cell + a], ctx.bootstrap_n)
                          for a in ("", FLOOR) for cell in CELLS if cell != INCUMBENT}
    res["primary"] = {"a": INCUMBENT, "b": CHALLENGER, "adapter": "none",
                      "by_collar": res["comparisons"][f"{INCUMBENT} -> {CHALLENGER}"]}
    res["dev_adoption"] = dev_adoption(res["systems"][INCUMBENT]["0.0"]["corpus"]["der"],
                                       res["systems"][CHALLENGER]["0.0"]["corpus"]["der"],
                                       cnt["systems"][INCUMBENT]["sum_abs_error"], cnt["systems"][CHALLENGER]["sum_abs_error"])
    res["selection"] = {fi: {cell: {k: f[cell][k] for k in ("selected", "selected_der", "argmin", "argmin_der")}
                             | {"loro_selected_recovered": f[cell]["loro_stability"]["selected_recovered"]}
                             for cell in CELLS} for fi, f in sel["folds"].items()}
    res["final_configs"] = {cell: joint_mode([f[cell]["selected"] for f in sel["folds"].values()]) for cell in CELLS}
    res["frozen_identity"] = ident
    n = len(ents) + 1
    hist = ctx.history_root / f"dev_results_{n:03d}_{ident[:12]}.json"
    if hist.exists():
        raise P5Error(f"{hist} exists; refusing to overwrite a scored result")
    cv.write_json(hist, res)
    cv.write_json(ctx.dev_results_path, res)
    (ctx.cv_dir / "dev_results.md").write_text(dev_markdown(res))
    cv.append_jsonl(ctx.scored_path, {"scored": res["created"], "frozen_identity": ident,
                                      "results_sha256": cv.file_sha(ctx, hist),
                                      "history": str(hist.relative_to(ctx.cv_dir)), "deviation": fz.get("deviation")})
    cv.write_json(ctx.final_configs_path, {
        "created": cv.now(), "rule": "final configuration per cell = joint mode of its five fold selections (ties -> "
                                     "simpler order)", "dev_results_sha256": cv.file_sha(ctx, ctx.dev_results_path),
        "selection_sha256": cv.file_sha(ctx, ctx.selection_path), "frozen_identity": ident,
        "cells": {cell: dict(res["final_configs"][cell], decoding=p3b.parse_key(res["final_configs"][cell]["config"]).as_dict())
                  for cell in CELLS}, "dev_adoption": res["dev_adoption"]})
    ctx.out(dev_markdown(res))
    return res


def dev_markdown(res: dict) -> str:
    pc = lambda x: f"{100 * x:.2f}"  # noqa: E731
    L = ["# Phase-5 Part A (DEVELOPMENT numbers, outer folds)", "", f"> {res['evidence']}", "",
         f"Frozen {res['frozen_created']}; scored {res['created']}; bootstrap n={res['bootstrap']['n']} seed "
         f"{res['bootstrap']['seed']}.", "", "## Corpus DER (%) [MISS/FA/CONF], sum|count err|", "",
         "| system | " + " | ".join(f"collar {c}" for c in res["collars"]) + " | sum|cnt err| |",
         "|---|" + "---|" * (len(res["collars"]) + 1)]
    for system, byc in res["systems"].items():
        cells = [f"{pc(byc[str(c)]['corpus']['der'])} [{pc(byc[str(c)]['corpus']['miss_rate'])}/"
                 f"{pc(byc[str(c)]['corpus']['fa_rate'])}/{pc(byc[str(c)]['corpus']['conf_rate'])}]" for c in res["collars"]]
        L.append(f"| {system} | " + " | ".join(cells) + f" | {byc['sum_abs_count_error']} |")
    L += ["", "## Per recording, collar 0 (%)", "", "| rec | " + " | ".join(res["systems"]) + " |",
          "|---|" + "---|" * len(res["systems"])]
    get = {s: {d["recording_id"]: d["der"] for d in v["0.0"]["per_recording"]} for s, v in res["systems"].items()}
    for rid in sorted(get[INCUMBENT]):
        L.append(f"| {rid} | " + " | ".join(pc(get[s][rid]) for s in res["systems"]) + " |")
    L += ["", "## Paired comparisons (delta = B - A, pp; bootstrap over recordings)", "",
          "| comparison | collar | A | B | delta | 95% CI | B wins/n |", "|---|---|---|---|---|---|---|"]
    for name, byc in res["comparisons"].items():
        for c, r in byc.items():
            L.append(f"| {name} | {c} | {pc(r['der_a'])} | {pc(r['der_b'])} | {100 * r['delta_b_minus_a']:+.2f} | "
                     f"[{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] | {r['b_wins']}/{r['n_recordings']} |")
    d = res["dev_adoption"]
    L += ["", "## Dev adoption criterion (challenger s120_ft vs incumbent s20_stock, collar 0, no adapter)", "",
          f"- corpus delta {d['delta_pp']:+.2f} pp (needs <= -1.0): {'PASS' if d['der_improves_ge_1pp'] else 'FAIL'}",
          f"- sum|count err| {d['count_inc']} -> {d['count_chal']} (<= +10 %): "
          f"{'PASS' if d['count_not_up_gt_10pct'] else 'FAIL'}",
          f"- **adopt challenger: {d['adopt_challenger']}** ({d['grade']})", "",
          "## Selection per fold (inner only) and final configurations (joint mode)", "",
          "| fold | cell | selected | inner DER | argmin | LORO |", "|---|---|---|---|---|---|"]
    for fi, f in sorted(res["selection"].items()):
        for cell, x in f.items():
            L.append(f"| {fi} | {cell} | {x['selected']} | {pc(x['selected_der'])} | {x['argmin']} | "
                     f"{x['loro_selected_recovered']}/8 |")
    L.append("")
    for cell, fc in res["final_configs"].items():
        L.append(f"- {cell}: **{fc['config']}** ({fc['count']}/{fc['of']}{', tied ' + str(fc['tied']) if len(fc['tied']) > 1 else ''})")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------
# Part B: data, validation
# ----------------------------------------------------------------------------------------------
def ho_stems(ctx: P5Ctx) -> list[str]:
    return load_subset(ctx.subset_file)


def ho_file(ctx: P5Ctx, stem: str, ext: str) -> Path:
    return ctx.holdout_dir / f"{stem}.{ext}"


def holdout_present(ctx: P5Ctx) -> tuple[bool, list[str]]:
    miss = [f"{s}.{e}" for s in ho_stems(ctx) for e in ("wav", "rttm", "json") if not ho_file(ctx, s, e).is_file()]
    if not ctx.holdout_sha_file.is_file():
        miss.append(str(ctx.holdout_sha_file))
    return not miss, miss


def parse_sha_file(ctx: P5Ctx) -> dict[str, str]:
    """{resolved path: sha256} from a sha256sum-style file (paths relative to the repo, data/ or the hold-out
    directory, or absolute)."""
    out = {}
    for line in ctx.holdout_sha_file.read_text().splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if not m:
            continue
        rp = m.group(2)
        for base in (ctx.repo, ctx.holdout_dir.parent, ctx.holdout_dir):
            p = (base / rp) if not Path(rp).is_absolute() else Path(rp)
            if p.is_file():
                out[str(p.resolve())] = m.group(1).lower()
                break
    return out


def verify_holdout_data(ctx: P5Ctx) -> dict:
    ok, miss = holdout_present(ctx)
    if not ok:
        raise P5Error(f"hold-out data incomplete ({len(miss)} missing, e.g. {miss[:4]})")
    listed = parse_sha_file(ctx)
    bad, unlisted = [], []
    for s in ho_stems(ctx):
        for e in ("wav", "rttm", "json"):
            p = str(ho_file(ctx, s, e).resolve())
            if p not in listed:
                if e != "json":
                    unlisted.append(f"{s}.{e}")
                continue
            if cv.file_sha(ctx, Path(p)) != listed[p]:
                bad.append(f"{s}.{e}")
    if bad or unlisted:
        raise P5Error(f"hold-out files do not match {ctx.holdout_sha_file}: mismatched {bad[:4]}, unlisted {unlisted[:4]}")
    return {"sha_file_sha256": cv.file_sha(ctx, ctx.holdout_sha_file), "n_files_verified": len(ho_stems(ctx)) * 2,
            "dataset_revision_registered": DATASET_REVISION}


def ho_ref(ctx: P5Ctx, stem: str) -> list[S.Seg]:
    """Reference exactly as provided (single file id, any name)."""
    d = S.read_rttm(ho_file(ctx, stem, "rttm"))
    if len(d) > 1:
        raise P5Error(f"{stem}.rttm holds several file ids {sorted(d)}")
    return next(iter(d.values()), [])


def ho_wav(ctx: P5Ctx, stem: str) -> Path:
    """16 kHz mono audio of a mixture: the provided file when it already is, else an ffmpeg copy under
    runs/p5/holdout/wav16 (written only outside dry runs)."""
    import soundfile as sf

    src = ho_file(ctx, stem, "wav")
    info = sf.info(str(src))
    if info.samplerate == 16000 and info.channels == 1:
        return src
    dst = ctx.ho / "wav16" / f"{rid_of(stem)}.wav"
    if not dst.exists() and not ctx.dry_run:
        from bnd import diarize

        diarize.to_wav16k(src, dst)
    return dst


def ho_durations(ctx: P5Ctx) -> dict[str, float]:
    ok, _ = holdout_present(ctx)
    if not ok:
        return {rid_of(s): HOLDOUT_ASSUMED_S for s in ho_stems(ctx)}
    return {rid_of(s): S.audio_duration(ho_file(ctx, s, "wav")) for s in ho_stems(ctx)}


def ho_nemo_ok(ctx: P5Ctx, system: str, session: int, rid: str, model_sha: str) -> bool:
    from bnd import infer

    return infer.cache_complete(ctx.ho_cache / system / f"s{session}" / rid, {"model_sha256": model_sha,
                                                                            "core_s": float(session)})


def ho_nemo(ctx: P5Ctx, ledger: P5Ledger | None, r: Rates, system: str, model: Path, model_sha: str | None,
            session: int, tag: str) -> float:
    stems = ho_stems(ctx)
    todo = [s for s in stems if model_sha is None or not ho_nemo_ok(ctx, system, session, rid_of(s), model_sha)]
    if not todo:
        return 0.0
    durs = ho_durations(ctx)
    audio = sum(durs[rid_of(s)] for s in todo)
    est = r.nemo(audio, session)
    items = [f"{rid_of(s)}={ho_wav(ctx, s) if not ctx.dry_run else ho_file(ctx, s, 'wav')}" for s in todo]
    cmd = [ctx.python, "-m", "bnd.p5", "holdout-nemo-job", "--model", model, "--core-s", str(session),
           "--system-dir", ctx.ho_cache / system / f"s{session}", "--item", *items]
    if ctx.dry_run:
        ctx.out(f"DRY-RUN [gpu] holdout-nemo-job {system} S={session} on {len(todo)} mixtures ({audio:.0f} audio-s; "
                f"est {est / 3600:.3f} GPU-h)")
        return est
    gpu_job(ctx, ledger, cmd, "infer", tag, est, audio)
    bad = [s for s in todo if not ho_nemo_ok(ctx, system, session, rid_of(s), model_sha)]
    if bad:
        raise P5Error(f"hold-out {system} S={session}: probs incomplete for {bad[:4]}")
    return est


def stitched_max(cache: Path) -> tuple[np.ndarray, float]:
    """Max-over-channels native prob on the owned-core frames, stitched over windows (frame i of the
    recording = [i*dt, (i+1)*dt))."""
    meta = json.loads((cache / "meta.json").read_text())
    dt = float(meta["dt"])
    from bnd import link as L

    n = L.n_frames(float(meta["recording_duration_s"]), dt)
    out = np.zeros(n, np.float32)
    for w in json.loads((cache / "windows.json").read_text()):
        p = np.load(cache / f"probs_{w['window']}.npy").astype(np.float32).max(1)
        a, b = int(round(w["core_start_s"] / dt)), min(int(round(w["core_end_s"] / dt)), n)
        off = int(round((w["core_start_s"] - w["start_s"]) / dt))
        seg = p[off:off + (b - a)]
        out[a:a + len(seg)] = seg
    return out, dt


def ref_stats(ref: Sequence[S.Seg], duration: float, nemo: np.ndarray | None, nemo_dt: float | None,
              silero: np.ndarray | None, silero_dt: float | None) -> dict:
    """Reference statistics on the 10 ms grid (bnd.trust conventions)."""
    from bnd import trust as T

    n = T.n_grid(duration)
    cnt = np.zeros(n, np.int32)
    by_spk: dict[str, list] = {}
    for s in ref:
        by_spk.setdefault(s.speaker, []).append(s)
    for segs in by_spk.values():
        cnt += T.label_mask(segs, n).astype(np.int32)
    lab = cnt > 0
    durs = np.array([s.end - s.start for s in ref if s.end > s.start]) if ref else np.zeros(0)
    st = {"duration_s": duration, "n_speakers": len(by_spk), "n_turns": int(len(durs)),
          "labelled_s": float(lab.sum() * T.GRID_DT), "labelled_fraction": float(lab.mean()) if n else 0.0,
          "overlap_s": float((cnt >= 2).sum() * T.GRID_DT),
          "overlap_fraction_of_labelled": float((cnt >= 2).sum() / max(1, lab.sum())),
          "turn_s": {"mean": float(durs.mean()) if len(durs) else None, "median": float(np.median(durs)) if len(durs) else None,
                     "min": float(durs.min()) if len(durs) else None, "max": float(durs.max()) if len(durs) else None}}
    if nemo is not None and silero is not None:
        nm = T.resample_to_grid(nemo, nemo_dt, n, "nemotron")
        sl = T.resample_to_grid(silero, silero_dt, n, "silero", tail_tol_frames=2)
        silent = lab & (nm < THRESH) & (sl < THRESH)
        st["silent_both_fraction_of_labelled"] = float(silent.sum() / max(1, lab.sum()))
        st["silent_both_runs_ge_5s_fraction_of_labelled"] = float(T.long_runs(silent, 500).sum() / max(1, lab.sum()))
        st["nemotron_speech_fraction"] = float((nm >= THRESH).mean())
        st["silero_speech_fraction"] = float((sl >= THRESH).mean())
    return st


def silero_dir(ctx: P5Ctx) -> Path:
    return ctx.ho / "silero_work" / "runs/probe_alignment/silero"


def run_silero(ctx: P5Ctx, stems: Sequence[str]) -> None:
    """scripts/silero_vad_probs.py unchanged (CPU): run with cwd runs/p5/holdout/silero_work so its fixed output
    path lands there; <= 4 shards of symlinked 16 kHz wavs, each a nice'd process recorded in runs/p5/pids/."""
    todo = [s for s in stems if not (silero_dir(ctx) / f"{rid_of(s)}.npz").exists()]
    if not todo:
        return
    work = ctx.ho / "silero_work"
    procs = []
    for k in range(SILERO_WORKERS):
        shard = todo[k::SILERO_WORKERS]
        if not shard:
            continue
        d = work / "in" / f"shard{k}"
        d.mkdir(parents=True, exist_ok=True)
        for s in shard:
            p3b._symlink(d / f"{rid_of(s)}.wav", ho_wav(ctx, s))
        cmd = ["nice", "-n", "10", *SILERO_CMD, str(ctx.repo / "scripts/silero_vad_probs.py"), str(d)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
        log = open(ctx.logs / f"silero_shard{k}.log", "a")
        p = subprocess.Popen(cmd, cwd=str(work), env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, p3b.write_pid(ctx, f"silero_shard{k}", p.pid, shlex.join(cmd)), log))
    rcs = []
    try:
        for p, _, _ in procs:
            rcs.append(p.wait())
    except BaseException:
        for p, _, _ in procs:
            if p.poll() is None:
                p.terminate()
        for p, _, _ in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        raise
    finally:
        for p, pf, lf in procs:
            lf.close()
            if p.poll() is not None and pf.exists():
                pf.unlink()
    if any(rcs):
        raise P5Error(f"Silero shards failed (rc {rcs}); logs {ctx.logs}/silero_shard*.log")


def any_holdout_output(ctx: P5Ctx) -> bool:
    return ctx.ho_rttm.is_dir() and any(ctx.ho_rttm.rglob("*.rttm"))


def cmd_b_validate(ctx: P5Ctx) -> dict | None:
    stems = ho_stems(ctx)
    ok, miss = holdout_present(ctx)
    r = rates(ctx)
    if ctx.dry_run:
        ctx.out(f"DRY-RUN B-validate: requires data/holdout_synth (60 stems x wav/rttm/json) + {ctx.holdout_sha_file.name}"
                f" ({'present' if ok else f'{len(miss)} missing'}); zero-shot Nemotron S=20 probs + Silero (CPU, "
                f"{SILERO_WORKERS} nice'd shards) -> labelled fraction, overlap fraction, labelled speech silent to both, "
                "turn stats; references used as provided")
        return {"projected_gpu_h": round(ho_nemo(ctx, None, r, "zeroshot", ctx.base_model, None, 20, "") / 3600, 3)}
    if not ok:
        raise P5Error(f"B-validate refused: hold-out data incomplete ({len(miss)} missing, e.g. {miss[:4]})")
    if any_holdout_output(ctx) and not ctx.validation_path.exists():
        raise P5Error("B-validate refused: hold-out system outputs exist before the reference validation")
    data = verify_holdout_data(ctx)
    ledger = P5Ledger(ctx)
    bsha = base_sha(ctx)
    ho_nemo(ctx, ledger, r, "zeroshot", ctx.base_model, bsha, 20, "bvalidate:zeroshot_s20")
    run_silero(ctx, stems)
    from bnd import trust as T

    per = {}
    for s in stems:
        rid = rid_of(s)
        nemo, ndt = stitched_max(ctx.ho_cache / "zeroshot/s20" / rid)
        sil, sdt = T.load_detector(silero_dir(ctx) / f"{rid}.npz")
        dur = S.audio_duration(ho_file(ctx, s, "wav"))
        per[rid] = dict(ref_stats(ho_ref(ctx, s), dur, nemo, ndt, sil, sdt), stem=s, k=k_of(s), bin=speaker_bin(k_of(s)),
                        json_sha256=cv.file_sha(ctx, ho_file(ctx, s, "json")))
        per[rid]["n_speakers_equals_k"] = per[rid]["n_speakers"] == per[rid]["k"]
    lab = sum(v["labelled_s"] for v in per.values())
    agg = {"n_mixtures": len(per), "total_audio_s": sum(v["duration_s"] for v in per.values()), "labelled_s": lab,
           "labelled_fraction": lab / sum(v["duration_s"] for v in per.values()),
           "overlap_fraction_of_labelled": sum(v["overlap_s"] for v in per.values()) / max(lab, 1e-9),
           "silent_both_fraction_of_labelled": sum(v["silent_both_fraction_of_labelled"] * v["labelled_s"]
                                                   for v in per.values()) / max(lab, 1e-9),
           "silent_both_runs_ge_5s_fraction_of_labelled": sum(v["silent_both_runs_ge_5s_fraction_of_labelled"] * v["labelled_s"]
                                                              for v in per.values()) / max(lab, 1e-9),
           "n_turns": sum(v["n_turns"] for v in per.values()),
           "mixtures_with_n_speakers_ne_k": sorted(x for x, v in per.items() if not v["n_speakers_equals_k"])}
    out = {"created": cv.now(), "prereg": "docs/PREREG_phase5.md Part B", "evidence": EVIDENCE,
           "note": "references recorded as provided, never edited; computed before any hold-out system output",
           "data": data, "aggregate": agg, "per_mixture": per,
           "detectors": {"nemotron": "zero-shot base, S=20 sessions, max over channels on owned cores >= 0.5",
                         "silero": "scripts/silero_vad_probs.py (silero-vad 6.2.3), >= 0.5"}}
    cv.write_json(ctx.validation_path, out)
    cv.write_marker(ctx, "bvalidate", "holdout", {"data": data, "subset": cv.file_sha(ctx, ctx.subset_file)},
                    {"validation_sha256": cv.file_sha(ctx, ctx.validation_path)})
    ctx.out("[p5] B-validate " + json.dumps(agg))
    return out


# ----------------------------------------------------------------------------------------------
# Part B: frozen batch
# ----------------------------------------------------------------------------------------------
def load_final_configs(ctx: P5Ctx) -> dict:
    """Part-A final configs, only if they come from the last scored Part-A result."""
    if not ctx.final_configs_path.exists() or not ctx.dev_results_path.exists():
        raise P5Error("Part-A final configurations missing (run A-score first)")
    fc = json.loads(ctx.final_configs_path.read_text())
    ents = cv.scored_entries(ctx)
    if not ents or fc.get("dev_results_sha256") != cv.file_sha(ctx, ctx.dev_results_path) \
            or fc.get("frozen_identity") != ents[-1]["frozen_identity"]:
        raise P5Error("final_configs.json does not match the last scored Part-A results")
    return fc


def ftn_config(ctx: P5Ctx) -> dict:
    """FT-Nemotron decoding = joint mode of the phase-3b FT inner selections (runs/p3b, verified frozen)."""
    fz = json.loads((ctx.p3b_dir / "frozen.json").read_text())
    selp = ctx.p3b_dir / "selection.json"
    if fz.get("selection_sha256") != cv.file_sha(ctx, selp):
        raise P5Error("runs/p3b/selection.json no longer matches runs/p3b/frozen.json")
    sel = json.loads(selp.read_text())
    return dict(p3b.joint_mode([f["ft"]["selected"] for f in sel["folds"].values()]),
                selection_sha256=cv.file_sha(ctx, selp))


def ho_batch(ctx: P5Ctx, fc: dict | None) -> dict:
    """The frozen batch: system -> {cache source, session, config, mode}."""
    ftn = p3b.parse_key(ftn_config(ctx)["config"])
    out = {}
    for name, cell, src in (("incumbent", INCUMBENT, "zeroshot"), ("challenger", CHALLENGER, ft_source(FINAL_EMBEDDER))):
        cfg = p3b.parse_key(fc["cells"][cell]["config"]) if fc else Config(CELLS[cell][0], "avg", 0, 0.40, 0.60)
        out[f"{name}_acoustic"] = {"source": src, "cfg": acoustic(cfg), "mode": "acoustic (g=0)", "cell": cell}
        out[f"{name}_benchmark"] = {"source": src, "cfg": cfg, "mode": "benchmark (selected g)", "cell": cell}
    out["ftnemotron_acoustic"] = {"source": "ftnemotron", "cfg": acoustic(ftn), "mode": "acoustic (g=0)"}
    out["ftnemotron_benchmark"] = {"source": "ftnemotron", "cfg": ftn, "mode": "benchmark (selected g)"}
    out["community1"] = {"source": "community1", "cfg": None, "mode": "as provided (exclusive)"}
    return out


def ftn_paths(ctx: P5Ctx) -> tuple[Path, Path]:
    return ctx.cv_dir / "models" / f"{FTN_TAG}.nemo", ctx.cv_dir / "fits" / FTN_TAG / "fit_report.json"


def ftn_train_cmd(ctx: P5Ctx) -> list:
    return [ctx.python, "-m", "bnd.p3b", "train-fit", "--fits-dir", ctx.cv_dir / "fits", "--models-dir",
            ctx.cv_dir / "models", "--tag", FTN_TAG, "--train-recs", ",".join(sorted(RECORDINGS)), "--lr", FTN_LR,
            "--timing", FTN_TIMING]


def ensure_ftn_model(ctx: P5Ctx, ledger: P5Ledger | None, r: Rates) -> tuple[float, str | None]:
    model, rep = ftn_paths(ctx)

    def ok() -> bool:
        """fit_report says ok, all 10, the registered lr / timing / step count, and it describes THIS .nemo."""
        from bnd import train as tr

        try:
            d = json.loads(rep.read_text())
            lr_ok = float(d.get("lr")) == float(FTN_LR)
        except (OSError, ValueError, TypeError):
            return False
        ex = d.get("export") or {}
        return (model.exists() and d.get("status") == "ok" and d.get("train_recs") == sorted(RECORDINGS) and lr_ok
                and d.get("timing") == FTN_TIMING and d.get("optimizer_steps") == tr.MAX_STEPS
                and bool(ex.get("path")) and Path(ex["path"]).resolve() == model.resolve()
                and ex.get("bytes") == model.stat().st_size)

    if ok():
        return 0.0, cv.file_sha(ctx, model)
    if ctx.dry_run:
        gpu_job(ctx, None, ftn_train_cmd(ctx), "train", f"train:{FTN_TAG}", r.ftn_fit_s)
        return r.ftn_fit_s, None
    for p in (model, rep.parent):
        d = cv.move_aside(p)
        if d:
            ctx.out(f"[p5] moved incomplete {p} -> {d}")
    gpu_job(ctx, ledger, ftn_train_cmd(ctx), "train", f"train:{FTN_TAG}", r.ftn_fit_s)
    if not ok():
        raise P5Error(f"{FTN_TAG}: training did not finish ok")
    return r.ftn_fit_s, cv.file_sha(ctx, model)


def ho_tracks(ctx: P5Ctx, system: str, session: int, theta: float) -> None:
    from bnd import infer

    for s in ho_stems(ctx):
        d = ctx.ho_cache / system / f"s{session}" / rid_of(s)
        if not (d / f"tracks_{theta:.2f}.json").exists():
            infer.write_tracks(d, theta)


def ho_embed(ctx: P5Ctx, ledger: P5Ledger | None, r: Rates, source: str, session: int, theta: float,
             ckpt: Path | None, want: str, tag: str) -> float:
    rids = [rid_of(s) for s in ho_stems(ctx)]
    sysdir = ctx.ho_cache / source / f"s{session}"
    todo = [x for x in rids if ctx.dry_run or emb_state(sysdir / x, theta) != want]
    if not todo:
        return 0.0
    durs = ho_durations(ctx)
    audio = sum(durs[x] for x in todo)
    est = r.embed(audio, 1)
    if ctx.dry_run:
        ctx.out(f"DRY-RUN [gpu] embed-job {source} S={session} theta {theta:.2f} "
                f"{'FT ' + str(ckpt) if ckpt else 'stock'} on {len(todo)} mixtures (est {est / 3600:.3f} GPU-h)")
        return est
    gpu_job(ctx, ledger, embed_job_cmd(ctx, [sysdir], todo, [theta], ckpt), "embed", tag, est, audio)
    bad = [x for x in todo if emb_state(sysdir / x, theta) != want]
    if bad:
        raise P5Error(f"hold-out embeddings {source} S={session}: missing/stale for {bad[:4]}")
    return est


def preflight_ft_embed(rids: Sequence[str]) -> None:
    """bnd.embed --embedder-ckpt calls bnd.embed_ft.assert_not_trained_on(rids); fail before any GPU use if
    it cannot handle hold-out ids."""
    from bnd import embed_ft

    try:
        [embed_ft.rec_id(x) for x in rids[:1]]
    except Exception as e:  # noqa: BLE001
        raise P5Error(f"bnd.embed_ft.rec_id cannot handle hold-out id {rids[0]!r} ({type(e).__name__}: {e}); "
                      f"bnd.embed --embedder-ckpt would fail on Part B. Fix bnd.embed_ft first.") from e


def ho_link(ctx: P5Ctx, name: str, spec: dict) -> dict:
    out = {}
    for s in ho_stems(ctx):
        rid = rid_of(s)
        p = ctx.ho_rttm / name / f"{rid}.rttm"
        cfg = spec["cfg"]
        cache = ctx.ho_cache / spec["source"] / f"s{cfg.session}" / rid
        r = p3b._link_task((str(cache), rid, cfg.theta_act, [cfg.key], [str(p)],
                            str(ctx.ho_rttm / name / "_reports" / f"{rid}.json"), "none", None))
        out[rid] = r[cfg.key]
    return out


def cmd_b_infer(ctx: P5Ctx) -> dict | None:
    stems = ho_stems(ctx)
    rids = [rid_of(s) for s in stems]
    r = rates(ctx)
    if ctx.dry_run:
        fc = None
        try:
            fc = load_final_configs(ctx)
        except P5Error:
            ctx.out("DRY-RUN B-infer: Part-A final configs not yet available; placeholder configs for the plan")
        batch = ho_batch(ctx, fc)
        ctx.out("DRY-RUN B-infer: base S=20 hold-out probs come from B-validate (not projected again)")
        tot = ho_embed(ctx, None, r, "zeroshot", 20, batch["incumbent_acoustic"]["cfg"].theta_act, None, "stock", "")
        tot += ho_nemo(ctx, None, r, "zeroshot", ctx.base_model, None, 120, "")
        tot += ho_embed(ctx, None, r, ft_source(FINAL_EMBEDDER), 120, batch["challenger_acoustic"]["cfg"].theta_act,
                        ctx.embedders_root / FINAL_EMBEDDER, "ft", "")
        audio = sum(ho_durations(ctx).values())
        c1 = p3b.JOB_FIXED_S + audio * COMMUNITY1_PER_AUDIO_S
        ctx.out(f"DRY-RUN [gpu] community1-job ({COMMUNITY1_ID}@{COMMUNITY1_REV[:8]}, embed venv) on 60 mixtures "
                f"(est {c1 / 3600:.3f} GPU-h)")
        t, _ = ensure_ftn_model(ctx, None, r)
        tot += c1 + t
        tot += ho_nemo(ctx, None, r, "ftnemotron", ftn_paths(ctx)[0], None, 20, "")
        tot += ho_embed(ctx, None, r, "ftnemotron", 20, batch["ftnemotron_acoustic"]["cfg"].theta_act, None, "stock", "")
        for name, spec in batch.items():
            ctx.out(f"DRY-RUN [cpu] link {name}: {spec['source']} {spec['cfg'].key if spec['cfg'] else '-'} "
                    f"({spec['mode']}) -> {ctx.ho_rttm / name}/<60 mixtures>.rttm")
        return {"projected_gpu_h": round(tot / 3600, 3)}
    if ctx.ho_frozen_path.exists():
        verify_holdout_frozen(ctx)
        ctx.out("[p5] hold-out batch frozen; nothing to do")
        return None
    if cv.read_marker(ctx, "bvalidate", "holdout") is None or not ctx.validation_path.exists():
        raise P5Error("B-infer refused: run B-validate first (reference validation before any system output)")
    verify_holdout_data(ctx)
    fc = load_final_configs(ctx)
    batch = ho_batch(ctx, fc)
    ledger = P5Ledger(ctx)
    bsha = base_sha(ctx)
    # 1. incumbent: base S=20 (probs from B-validate), stock embedder
    ho_nemo(ctx, ledger, r, "zeroshot", ctx.base_model, bsha, 20, "binfer:zeroshot_s20")
    ti = batch["incumbent_acoustic"]["cfg"].theta_act
    ho_tracks(ctx, "zeroshot", 20, ti)
    ho_embed(ctx, ledger, r, "zeroshot", 20, ti, None, "stock", "binfer:stockemb_s20")
    # 2. challenger: base S=120, all-10 FT embedder
    spec10 = fit_by_tag(FINAL_EMBEDDER)
    man = fit_done(ctx, spec10)
    if man is None:
        raise P5Error(f"{FINAL_EMBEDDER} not fitted (A-fits)")
    preflight_ft_embed(rids)
    from bnd import embed_ft

    embed_ft.assert_not_trained_on(fit_dir(ctx, FINAL_EMBEDDER), rids)
    ho_nemo(ctx, ledger, r, "zeroshot", ctx.base_model, bsha, 120, "binfer:zeroshot_s120")
    tc = batch["challenger_acoustic"]["cfg"].theta_act
    ho_tracks(ctx, "zeroshot", 120, tc)
    src = ft_source(FINAL_EMBEDDER)
    for x in rids:
        d = ctx.ho_cache / src / "s120" / x
        base = ctx.ho_cache / "zeroshot/s120" / x
        _mirror(d, base, base_names(base) + [f"tracks_{tc:.2f}.json"])
    ho_embed(ctx, ledger, r, src, 120, tc, fit_dir(ctx, FINAL_EMBEDDER), man["sha256"], "binfer:ftemb_all10_s120")
    # 3. community-1
    c1dir = ctx.ho_rttm / "community1"
    todo = [s for s in stems if not (c1dir / f"{rid_of(s)}.rttm").exists()]
    if todo:
        audio = sum(ho_durations(ctx)[rid_of(s)] for s in todo)
        gpu_job(ctx, ledger, [ctx.embed_python, "-m", "bnd.p5", "community1-job", "--out", c1dir, "--item",
                              *[f"{rid_of(s)}={ho_wav(ctx, s)}" for s in todo]], "infer", "binfer:community1",
                p3b.JOB_FIXED_S + audio * COMMUNITY1_PER_AUDIO_S, audio)
    # 4. FT-Nemotron (all 10) with the phase-3b FT joint-mode decoding, stock embedder
    _, ftn_sha = ensure_ftn_model(ctx, ledger, r)
    ho_nemo(ctx, ledger, r, "ftnemotron", ftn_paths(ctx)[0], ftn_sha, 20, "binfer:ftnemotron_s20")
    tf = batch["ftnemotron_acoustic"]["cfg"].theta_act
    ho_tracks(ctx, "ftnemotron", 20, tf)
    ho_embed(ctx, ledger, r, "ftnemotron", 20, tf, None, "stock", "binfer:stockemb_ftnemotron")
    shas = {}
    for name, spec in batch.items():
        if spec["cfg"] is None:
            shas[name] = {rid_of(s): cv.file_sha(ctx, c1dir / f"{rid_of(s)}.rttm") for s in stems}
        else:
            shas[name] = ho_link(ctx, name, spec)
    idx = {"created": cv.now(), "final_configs_sha256": cv.file_sha(ctx, ctx.final_configs_path),
           "systems": {n: {"source": s["source"], "config": s["cfg"].key if s["cfg"] else None, "mode": s["mode"],
                           "rttm_sha256": shas[n]} for n, s in batch.items()},
           "models": {"base_sha256": bsha, "ftnemotron_sha256": ftn_sha, "all10_embedder_sha256": man["sha256"],
                      "ftnemotron_decoding": ftn_config(ctx), "community1": f"{COMMUNITY1_ID}@{COMMUNITY1_REV}"}}
    cv.write_json(ctx.ho_index_path, idx)
    ctx.out(f"[p5] B-infer: {len(batch)} systems x {len(stems)} mixtures -> {ctx.ho_index_path}")
    return idx


def ho_config_hashes(ctx: P5Ctx) -> dict:
    return {"final_configs": cv.file_sha(ctx, ctx.final_configs_path, True),
            "dev_results": cv.file_sha(ctx, ctx.dev_results_path, True),
            "validation": cv.file_sha(ctx, ctx.validation_path, True),
            "holdout_index": cv.file_sha(ctx, ctx.ho_index_path, True),
            "p3b_selection": cv.file_sha(ctx, ctx.p3b_dir / "selection.json", True),
            "subset": cv.file_sha(ctx, ctx.subset_file, True),
            "holdout_sha_file": cv.file_sha(ctx, ctx.holdout_sha_file, True),
            "ftnemotron_model": cv.file_sha(ctx, ftn_paths(ctx)[0], True),
            "all10_embedder": cv.file_sha(ctx, fit_dir(ctx, FINAL_EMBEDDER) / "embedder.pt", True),
            "base_model": cv.file_sha(ctx, ctx.base_model, True)}


def cmd_b_freeze(ctx: P5Ctx) -> dict | None:
    if not ctx.ho_index_path.exists():
        raise P5Error("B-freeze refused: runs/p5/holdout/index.json missing (B-infer incomplete)")
    idx = json.loads(ctx.ho_index_path.read_text())
    stems = ho_stems(ctx)
    systems, problems = {}, []
    for name in HOLDOUT_SYSTEMS:
        e = idx["systems"].get(name)
        if e is None:
            problems.append(f"{name}: not in the batch")
            continue
        systems[name] = {}
        for s in stems:
            rid = rid_of(s)
            p = ctx.ho_rttm / name / f"{rid}.rttm"
            sha = cv.file_sha(ctx, p, True)
            if sha == "MISSING" or sha != e["rttm_sha256"].get(rid):
                problems.append(f"{p} missing or differs from the hold-out index")
                continue
            systems[name][rid] = {"path": p3b.rel(ctx, p), "sha256": sha, "stem": s}
    if problems:
        raise P5Error(f"B-freeze refused: {problems[:8]}")
    fz = {"created": cv.now(), "prereg": "docs/PREREG_phase5.md Part B", "evidence": EVIDENCE, "path_anchor": "runs/",
          "configs": ho_config_hashes(ctx), "batch": {n: {k: v for k, v in e.items() if k != "rttm_sha256"}
                                                     for n, e in idx["systems"].items()},
          "models": idx["models"], "systems": systems}
    if ctx.ho_frozen_path.exists():
        old = verify_holdout_frozen(ctx)
        if old["systems"] != systems:
            raise HoldoutLockError("hold-out already frozen with different predictions; refusing to re-freeze")
        ctx.out("[p5] hold-out already frozen; hashes verified")
        return old
    if ctx.dry_run:
        ctx.out(f"DRY-RUN B-freeze {sum(len(v) for v in systems.values())} hold-out RTTMs")
        return None
    cv.write_json(ctx.ho_frozen_path, fz)
    ctx.out(f"[p5] hold-out frozen: {sum(len(v) for v in systems.values())} RTTMs -> {ctx.ho_frozen_path}")
    return fz


def verify_holdout_frozen(ctx: P5Ctx) -> dict:
    if not ctx.ho_frozen_path.exists():
        raise HoldoutLockError(f"HOLD-OUT LOCK: {ctx.ho_frozen_path} does not exist; run B-freeze after B-infer")
    fz = json.loads(ctx.ho_frozen_path.read_text())
    bad = [f"{s}/{r}" for s, recs in fz["systems"].items() for r, e in recs.items()
           if cv.file_sha(ctx, p3b.unrel(ctx, e["path"]), True) != e["sha256"]]
    cur = ho_config_hashes(ctx)
    bad += [f"config {k}" for k, v in fz["configs"].items() if cur.get(k) != v]
    if set(fz["systems"]) != set(HOLDOUT_SYSTEMS) or any(len(v) != 60 for v in fz["systems"].values()):
        bad.append("system/mixture set")
    if bad:
        raise HoldoutLockError(f"HOLD-OUT LOCK: frozen hashes no longer match for {bad[:6]}")
    return fz


def _score_ho_one(args) -> dict:
    rid, hyp, ref, dur, collar = args
    d = S.read_rttm(ref)
    if len(d) > 1:
        raise P5Error(f"{ref}: several file ids")
    return S.score_recording(next(iter(d.values()), []), S.load_rttm_file(hyp, rid), dur, collar, recording_id=rid).as_dict()


def check_holdout_boot_n(ctx: P5Ctx) -> None:
    """Part B is scored ONCE, so its bootstrap must be the registered one (50k resamples); refuse before the
    score-once lock can record a scoring made with any other count."""
    if ctx.bootstrap_n != HOLDOUT_BOOT_N:
        raise P5Error(f"Part B bootstrap must use the registered {HOLDOUT_BOOT_N} resamples (PREREG Part B), got "
                      f"--bootstrap-n {ctx.bootstrap_n}; refusing to score the hold-out")


def cmd_b_score(ctx: P5Ctx) -> dict:
    check_holdout_boot_n(ctx)
    fz = verify_holdout_frozen(ctx)
    ident = cv.canon_sha({"systems": fz["systems"], "configs": fz["configs"]})
    prev = [json.loads(l) for l in ctx.ho_scored_path.read_text().splitlines() if l.strip()] \
        if ctx.ho_scored_path.exists() else []
    if prev:
        raise HoldoutLockError(f"HOLD-OUT LOCK: Part B was already scored ({prev[-1]['scored']}, identity "
                               f"{prev[-1]['frozen_identity'][:12]}); PREREG: score ONCE")
    stems = ho_stems(ctx)
    ks = {rid_of(s): k_of(s) for s in stems}
    durs = {rid_of(s): S.audio_duration(ho_file(ctx, s, "wav")) for s in stems}
    refn = {rid_of(s): len({x.speaker for x in ho_ref(ctx, s)}) for s in stems}
    todo = [(rid_of(s), str(p3b.unrel(ctx, fz["systems"][name][rid_of(s)]["path"])), str(ho_file(ctx, s, "rttm")),
             durs[rid_of(s)], c) for name in HOLDOUT_SYSTEMS for s in stems for c in S.COLLARS]
    if ctx.jobs > 1:
        with mp.get_context("fork").Pool(min(ctx.jobs, 4)) as pool:
            res = pool.map(_score_ho_one, todo, chunksize=4)
    else:
        res = [_score_ho_one(t) for t in todo]
    it = iter(res)
    scores: dict[str, dict[float, list[S.RecScore]]] = {n: {c: [] for c in S.COLLARS} for n in HOLDOUT_SYSTEMS}
    for name in HOLDOUT_SYSTEMS:
        for _s in stems:
            for c in S.COLLARS:
                scores[name][c].append(cv._rec_from_dict(next(it)))
    for name in scores:
        for c in S.COLLARS:
            scores[name][c].sort(key=lambda x: x.recording_id)
    out: dict = {"created": cv.now(), "frozen_created": fz["created"], "prereg": "docs/PREREG_phase5.md Part B",
                 "evidence": EVIDENCE, "label": "INDEPENDENT synthetic hold-out (scored once)",
                 "scoring": "corpus DER, overlap INCLUDED, whole-file UEM, references as provided; collar 0 primary, "
                            "0.25/0.5 total-width secondary", "collars": list(S.COLLARS),
                 "bootstrap": {"n": ctx.bootstrap_n, "seed": HOLDOUT_BOOT_SEED, "unit": "mixture",
                               "note": "mixtures share voices -> approximate"},
                 "reference_speakers": refn, "systems": {}}
    for name, per_c in scores.items():
        counts = {}
        for s in stems:
            rid = rid_of(s)
            n = len({x.speaker for x in S.load_rttm_file(p3b.unrel(ctx, fz["systems"][name][rid]["path"]), rid)})
            counts[rid] = {"predicted": n, "reference": refn[rid], "error": n - refn[rid], "k": ks[rid],
                           "bin": speaker_bin(ks[rid])}
        by_bin = {}
        for b, _, _ in SPK_BINS:
            rb = sorted(x for x in ks if speaker_bin(ks[x]) == b)
            by_bin[b] = {"n_mixtures": len(rb),
                         "der_by_collar": {str(c): S.corpus_summary([x for x in per_c[c] if x.recording_id in rb])["der"]
                                           for c in S.COLLARS},
                         "count_mae": float(np.mean([abs(counts[x]["error"]) for x in rb]))}
        out["systems"][name] = {"by_collar": {str(c): {"corpus": S.corpus_summary(sc), "per_mixture": [x.as_dict() for x in sc]}
                                              for c, sc in per_c.items()},
                                "speaker_counts": counts, "sum_abs_count_error": sum(abs(v["error"]) for v in counts.values()),
                                "by_bin": by_bin}

    def cmp(a: str, b: str) -> dict:
        return {str(c): S.paired_bootstrap(scores[a][c], scores[b][c], n=ctx.bootstrap_n, seed=HOLDOUT_BOOT_SEED)
                for c in S.COLLARS}

    prim = cmp("incumbent_acoustic", "challenger_acoustic")
    out["primary"] = {"a": "incumbent_acoustic", "b": "challenger_acoustic", "by_collar": prim, **holdout_claim(prim["0.0"])}
    out["secondary"] = {"exploratory": True,
                        "challenger_vs_incumbent_benchmark": cmp("incumbent_benchmark", "challenger_benchmark"),
                        "incumbent_vs_community1": cmp("incumbent_acoustic", "community1"),
                        "ftnemotron_vs_base": cmp("incumbent_acoustic", "ftnemotron_acoustic"),
                        "ftnemotron_vs_base_benchmark": cmp("incumbent_benchmark", "ftnemotron_benchmark")}
    out["frozen_identity"] = ident
    if ctx.dry_run:
        return out
    cv.write_json(ctx.ho_results_path, out)
    (ctx.cv_dir / "holdout_results.md").write_text(holdout_markdown(out))
    cv.append_jsonl(ctx.ho_scored_path, {"scored": out["created"], "frozen_identity": ident,
                                         "results_sha256": cv.file_sha(ctx, ctx.ho_results_path)})
    ctx.out(holdout_markdown(out))
    return out


def holdout_markdown(res: dict) -> str:
    pc = lambda x: f"{100 * x:.2f}"  # noqa: E731
    L = ["# Phase-5 Part B: independent synthetic hold-out (scored once)", "", f"> {res['evidence']}", "",
         f"Frozen {res['frozen_created']}; scored {res['created']}. {res['scoring']}. Bootstrap over mixtures n="
         f"{res['bootstrap']['n']} seed {res['bootstrap']['seed']} ({res['bootstrap']['note']}).", "",
         "## Corpus DER (%) [MISS/FA/CONF], sum|count err|", "",
         "| system | " + " | ".join(f"collar {c}" for c in res["collars"]) + " | sum|cnt err| |",
         "|---|" + "---|" * (len(res["collars"]) + 1)]
    for name, v in res["systems"].items():
        cells = []
        for c in res["collars"]:
            k = v["by_collar"][str(c)]["corpus"]
            cells.append(f"{pc(k['der'])} [{pc(k['miss_rate'])}/{pc(k['fa_rate'])}/{pc(k['conf_rate'])}]")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {v['sum_abs_count_error']} |")
    L += ["", "## By speaker-count bin (collar 0 DER %, count MAE)", "",
          "| system | " + " | ".join(b for b, _, _ in SPK_BINS) + " |", "|---|" + "---|" * len(SPK_BINS)]
    for name, v in res["systems"].items():
        L.append(f"| {name} | " + " | ".join(f"{pc(v['by_bin'][b]['der_by_collar']['0.0'])} / {v['by_bin'][b]['count_mae']:.2f}"
                                              for b, _, _ in SPK_BINS) + " |")
    p = res["primary"]
    L += ["", "## Primary: challenger vs incumbent (acoustic mode, collar 0)", "",
          f"delta {p['delta_pp']:+.2f} pp, 95% CI [{p['ci_low_pp']:+.2f}, {p['ci_high_pp']:+.2f}] -> "
          f"delta <= -1.0 pp: {'PASS' if p['delta_le_-1pp'] else 'FAIL'}; 97.5th pct < 0: "
          f"{'PASS' if p['p97_5_lt_0'] else 'FAIL'}.", "", f"**{p['verdict']}.**", "",
          "## Secondary (exploratory; delta = B - A, pp)", "", "| comparison | collar | A | B | delta | 95% CI | B wins/n |",
          "|---|---|---|---|---|---|---|"]
    for name, byc in [("primary", p["by_collar"])] + [(k, v) for k, v in res["secondary"].items() if k != "exploratory"]:
        for c, r in byc.items():
            L.append(f"| {name} | {c} | {pc(r['der_a'])} | {pc(r['der_b'])} | {100 * r['delta_b_minus_a']:+.2f} | "
                     f"[{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] | {r['b_wins']}/{r['n_recordings']} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------
# plan / status / CLI
# ----------------------------------------------------------------------------------------------
def cmd_plan(ctx: P5Ctx) -> dict:
    ctx.dry_run = True
    led = P5Ledger(ctx)
    ctx.out(f"DRY-RUN plan: PREREG_phase5; ledger {ctx.ledger_path} starts at {ctx.start_h:.2f} GPU-h, used now "
            f"{led.used_h():.3f} h; WARNING above {ctx.start_h + ctx.allowance_h:.2f} h (phase-5 allowance "
            f"{ctx.allowance_h} h); abort before launch at {ctx.budget_h} h cumulative")
    parts = {"A-prep": cmd_a_prep(ctx)["projected_gpu_h"], "A-fits": cmd_a_fits(ctx)["projected_gpu_h"]}
    cmd_a_search(ctx)
    cmd_a_outer(ctx)
    ctx.out(f"DRY-RUN A-freeze: sha256 of {len(DEV_SYSTEMS)} systems x 10 outer RTTMs")
    ctx.out(f"DRY-RUN A-score: 4 cells x {{none, floor}}, collars 0/0.25/0.5, MISS/FA/CONF, sum|count err|, paired "
            f"bootstrap {ctx.bootstrap_n} seed {DEV_BOOT_SEED}, dev adoption, joint-mode final configs")
    parts["B-validate"] = cmd_b_validate(ctx)["projected_gpu_h"]
    parts["B-infer"] = cmd_b_infer(ctx)["projected_gpu_h"]
    ctx.out(f"DRY-RUN B-freeze: sha256 of {len(HOLDOUT_SYSTEMS)} systems x 60 hold-out RTTMs + configs/models")
    ctx.out(f"DRY-RUN B-score: corpus DER collars 0/0.25/0.5 (overlap included, whole-file UEM), per mixture, per bin "
            f"{[b for b, _, _ in SPK_BINS]}, sum|count err| + per-bin MAE, bootstrap over mixtures {ctx.bootstrap_n} "
            f"seed {HOLDOUT_BOOT_SEED}; claim iff delta <= -1.0 pp and 97.5th pct < 0")
    tot = sum(parts.values())
    s = {"projected_gpu_h_by_step": parts, "projected_gpu_h_phase5": round(tot, 3),
         "projected_cumulative_h": round(led.used_h() + tot, 3),
         "allowance_line_h": round(ctx.start_h + ctx.allowance_h, 2), "budget_h": ctx.budget_h,
         "exceeds_allowance": led.used_h() + tot > ctx.start_h + ctx.allowance_h,
         "exceeds_budget": led.used_h() + tot > ctx.budget_h}
    ctx.out("DRY-RUN plan total " + json.dumps(s))
    return s


def cmd_status(ctx: P5Ctx) -> dict:
    led = P5Ledger(ctx)
    d = ctx.cv_dir / "done"
    n = lambda k: len(list((d / k).glob("*.json"))) if (d / k).is_dir() else 0  # noqa: E731
    st = {"aprep": n("aprep"), "embedder_fits": n("embfit"), "ft_embeddings": n("ftemb"), "link_tasks": n("plink"),
          "selection": ctx.selection_path.exists(), "outer_markers": n("outer"), "frozen": ctx.frozen_path.exists(),
          "dev_results": ctx.dev_results_path.exists(), "validation": ctx.validation_path.exists(),
          "holdout_index": ctx.ho_index_path.exists(), "holdout_frozen": ctx.ho_frozen_path.exists(),
          "holdout_results": ctx.ho_results_path.exists(), "gpu_h_used": round(led.used_h(), 3),
          "phase5_h": round(led.extra_h(), 3), "gpu_h_budget": ctx.budget_h,
          "running_pids": sorted(p.name for p in ctx.pids.glob("*.pid")) if ctx.pids.is_dir() else []}
    ctx.out(json.dumps(st, indent=1))
    return st


def _p5_out(s: str) -> None:
    print(s.replace("[p3b]", "[p5]").replace("phase-3b", "phase-5"), flush=True)


COMMANDS = ("plan", "A-prep", "A-fits", "A-search", "A-outer", "A-freeze", "A-score", "B-validate", "B-infer",
            "B-freeze", "B-score", "ledger", "status")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    internal = {"embed-job": embed_job_main, "holdout-nemo-job": holdout_nemo_job_main,
                "community1-job": community1_job_main}
    if argv[:1] and argv[0] in internal:
        return internal[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(prog="bnd.p5", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=COMMANDS)
    ap.add_argument("--dry-run", action="store_true", help="print planned work + projected GPU-h; write nothing")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--jobs", type=int, default=4, help="parallel CPU workers (<= 4 on the shared box)")
    ap.add_argument("--bootstrap-n", type=int, default=S.BOOTSTRAP_N)
    ap.add_argument("--deviation", default=None)
    a = ap.parse_args(argv)
    ctx = P5Ctx(dry_run=a.dry_run, allow_stale=a.allow_stale, jobs=max(1, min(a.jobs, 4)), bootstrap_n=a.bootstrap_n,
                deviation=a.deviation, out=_p5_out)
    if not (a.dry_run or a.cmd in ("plan", "ledger", "status")):
        pidf = p3b.write_pid(ctx, f"driver_{a.cmd}_{os.getpid()}", os.getpid(), shlex.join(["bnd.p5", *argv]))
        atexit.register(lambda: pidf.exists() and pidf.unlink())
    fn = {"plan": cmd_plan, "A-prep": cmd_a_prep, "A-fits": cmd_a_fits, "A-search": cmd_a_search, "A-outer": cmd_a_outer,
          "A-freeze": cmd_a_freeze, "B-validate": cmd_b_validate, "B-infer": cmd_b_infer, "B-freeze": cmd_b_freeze,
          "status": cmd_status}
    if a.cmd in fn:
        fn[a.cmd](ctx)
    elif a.cmd == "A-score":
        if ctx.dry_run:
            p3b.verify_frozen_p3b(ctx)
            ctx.out("DRY-RUN A-score: frozen hashes verified")
        else:
            cmd_a_score(ctx)
    elif a.cmd == "B-score":
        check_holdout_boot_n(ctx)
        verify_holdout_frozen(ctx)
        if ctx.dry_run:
            ctx.out("DRY-RUN B-score: hold-out frozen hashes verified")
        else:
            cmd_b_score(ctx)
    elif a.cmd == "ledger":
        led = P5Ledger(ctx)
        for e in led.entries():
            ctx.out(f"{e.get('ts_start', '-')} {e.get('kind', '?'):14s} {e.get('tag', ''):40s} card={e.get('card')} "
                    f"rc={e.get('returncode')} gpu_wall={e.get('gpu_wall_s', 0):.0f}s")
        ctx.out(f"used {led.used_h():.3f} h (start {ctx.start_h:.2f} h; phase-5 {led.extra_h():.3f} of "
                f"{ctx.allowance_h} h allowance) of {ctx.budget_h} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
