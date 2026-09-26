"""Resumable driver for PHASE 3b (docs/PREREG_phase3b.md, binding; it supersedes the WITHDRAWN
docs/PREREG_phase3.md). Every output lives under runs/p3b/. Phase-2 artefacts (runs/cv, runs/models,
runs/fits) are read only; runs/p3 (withdrawn, unscored) is ignored.

HONEST FRAMING (carried into every report): every factor of the phase-3b space was motivated by
phase-2 OUTER-fold diagnostics. Selection is inner-fold only, so outer numbers are protected from the
CHOICE among candidates, not from the DESIGN of the space: phase-3b outer numbers are "second-look"
results. No Kaggle.

Systems: ZS = pinned base model; FT = the phase-2 selected recipe per outer fold (F1 lr5e-5_raw,
F2-F5 lr5e-5_shift): outer = the kept refits runs/models/cv_refit_<Fi>_<recipe>.nemo, inner = that
recipe's inner fits excluding Fi (regenerated, see `regen`).

Search space (108 per system): S session {20, 80} s cores (+5 s context, one native forward call per
session) x theta_act {0.10, 0.25, 0.40} x L linker {avg, refine} x theta_clu {0.55, 0.60, 0.65} x
g gap-fill {0, 2, 5} s (bnd.link: ``linker``, ``gap_fill_s``; bnd.infer: ``--core-s``).

Subcommands (each resumable through done-markers with input hashes under runs/p3b/done/; with
``--dry-run`` a subcommand prints its planned work and projected GPU-h and writes nothing):

  regen   the 14 unique inner fits of the selected recipes: bnd.train with the IDENTICAL arguments and
          seed (train/timing/nemo_data/trust code, manifest, RTTMs and base model hashes must equal the
          phase-2 fit marker) into runs/p3b/{fits,models}/, then 20 s inference of ALL their held-out
          recordings into runs/p3b/cache/<fit>/s20/ and max|p_new - p_phase2| <= 1e-3 on every window
          of every recording, else ABORT (runs/p3b/regen_failure.json; PREREG: any mismatch = stop).
  prep    caches for S in {20, 80} x theta_act {0.10, 0.25, 0.40} (tracks + WeSpeaker embeddings):
          ZS on all 10 recordings, FT inner fits on the held-out recordings they serve, the 5 selected
          refits on their outer recordings. S=20 probs of ZS/refits are the phase-2 cached probs
          (read-only symlinks, same model and code); S=20 probs of inner fits are the verified regen
          probs; S=80 probs are new (bnd.infer --core-s 80). Cache: runs/p3b/cache/<system>/s<S>/<rid>/.
  search  link all 108 configurations for every ZS and FT inner unit (CPU, <= 4 workers); per outer
          fold Fi and system, pooled inner DER (collar 0) over Fi's 8 inner recordings; 1-SE rule with
          the PREREG simplicity order; argmin; leave-one-recording-out stability; the Q2 restricted
          (S=20, L=avg) selection -> runs/p3b/selection.json (full tables).
  outer   apply each system's selected config per fold (zs_star, ft_star), the Q2 restricted
          selections (zs_q2, ft_q2) and oracle linking from each selected system's own tracks
          (zs_star_oracle_link, ft_star_oracle_link) -> runs/p3b/outer/<name>/<Fi>/<rid>.rttm.
  freeze  sha256 of every outer RTTM (+ the phase-2 registered systems as references) -> frozen.json.
  score   OUTER LOCK (only frozen files; outer DER never before freeze); Q1, Q2, collars 0/0.25/0.5,
          MISS/FA/CONF, speaker-count errors, oracle link, phase-2 references, default-system rule ->
          runs/p3b/results.json + results.md (history kept; scored.jsonl append-only).
  plan    dry run of everything.  ledger / status.
  (internal, run by the driver under scripts/gpu_run.sh: train-fit, prep-job)

GPU: every GPU command through scripts/gpu_run.sh (one card). Ledger runs/p3b/gpu_ledger.jsonl, starting
balance 4.11 GPU-h (phase-2 cumulative); a job is refused before launch if it would exceed 12 GPU-h
cumulative; a WARNING is printed when the phase-3b allowance (2.0 h, i.e. 6.11 h cumulative) is or
would be exceeded. Running GPU children and the driver itself are recorded in runs/p3b/pids/.

Run on CPU:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 10 <venv python> -m bnd.p3b <cmd>
"""
from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from bnd import cv
from bnd import score as S
from bnd.labels import RECORDINGS

REPO = cv.REPO
FOLDS = cv.FOLDS

# ----------------------------------------------------------------------------------------------
# registered constants (PREREG_phase3b)
# ----------------------------------------------------------------------------------------------
SESSIONS = (20, 80)
THETA_ACT = (0.10, 0.25, 0.40)
LINKERS = ("avg", "refine")
THETA_CLU = (0.55, 0.60, 0.65)
GAPS = (0, 2, 5)
# simplicity order, lexicographic S, L, g, theta_act, theta_clu (simpler first)
RANK_S = {20: 0, 80: 1}
RANK_L = {"avg": 0, "refine": 1}
RANK_G = {0: 0, 2: 1, 5: 2}
RANK_A = {0.40: 0, 0.25: 1, 0.10: 2}
RANK_C = {0.60: 0, 0.55: 1, 0.65: 2}
REGISTERED_FT_RECIPE = {"F1": "lr5e-5_raw", "F2": "lr5e-5_shift", "F3": "lr5e-5_shift", "F4": "lr5e-5_shift",
                        "F5": "lr5e-5_shift"}
REGEN_TOL = 1e-3
PHASE2_CUM_H = 4.11
ALLOWANCE_H = 2.0
BUDGET_GPU_H = 12.0
Q1_MIN_IMPROVEMENT_PP = 1.0
Q1_MIN_WINS = 6
# GPU cost model for projections / pre-launch checks. Rates are re-derived from the phase-2
# infer reports when present (runs/cv/cache/*/infer_report.json); S=80 throughput is unmeasured.
DEF_NEMO_PER_AUDIO_S = 0.0049
DEF_EMBED_PER_AUDIO_THETA_S = 0.0022
DEF_FIT_S = 150.0
JOB_FIXED_S = 30.0
HEADROOM = 1.25
STAR = ("zs_star", "ft_star")
OUTER_SYSTEMS = ("zs_star", "ft_star", "zs_q2", "ft_q2", "zs_star_oracle_link", "ft_star_oracle_link")
P2_REFERENCE = {"p2_zs_sel": "zs_sel", "p2_ft_sel": "ft_sel", "p2_zs_strict": "zs_strict", "p2_ft_strict": "ft_strict"}
FRAMING = ("SECOND-LOOK results: every phase-3b factor (session length, theta_act below 0.40, the refine linker, "
           "gap-fill) was motivated by phase-2 OUTER-fold diagnostics. Selection used inner folds only, so these "
           "outer numbers are protected from the choice among candidates but not from the design of the space. "
           "Phase-2 exposure disclosures (D1, D2) still apply. No Kaggle submission.")


class P3bError(cv.CVError):
    pass


class RegenError(P3bError):
    pass


@dataclass(frozen=True)
class Config:
    session: int
    linker: str
    gap: int
    theta_act: float
    theta_clu: float

    @property
    def key(self) -> str:
        return f"s{self.session}_{self.linker}_g{self.gap}_a{self.theta_act:.2f}_c{self.theta_clu:.2f}"

    @property
    def simplicity(self) -> tuple:
        return (RANK_S[self.session], RANK_L[self.linker], RANK_G[self.gap], RANK_A[round(self.theta_act, 2)],
                RANK_C[round(self.theta_clu, 2)])

    def as_dict(self) -> dict:
        return {"session_core_s": self.session, "linker": self.linker, "gap_fill_s": self.gap,
                "theta_act": self.theta_act, "theta_clu": self.theta_clu}


def parse_key(k: str) -> Config:
    m = re.fullmatch(r"s(\d+)_(avg|refine)_g(\d+)_a([0-9.]+)_c([0-9.]+)", k)
    if not m:
        raise P3bError(f"bad config key {k!r}")
    return Config(int(m.group(1)), m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5)))


CONFIGS = [Config(s, l, g, a, c) for s in SESSIONS for a in THETA_ACT for l in LINKERS for c in THETA_CLU for g in GAPS]
assert len(CONFIGS) == 108 and len({c.key for c in CONFIGS}) == 108
Q2_CONFIGS = [c for c in CONFIGS if c.session == 20 and c.linker == "avg"]
assert len(Q2_CONFIGS) == 27


def configs_for(session: int, theta: float) -> list[Config]:
    return [c for c in CONFIGS if c.session == session and math.isclose(c.theta_act, theta)]


# ----------------------------------------------------------------------------------------------
# context, paths, ledger
# ----------------------------------------------------------------------------------------------
@dataclass
class P3bCtx(cv.Ctx):
    cv_dir: Path = REPO / "runs/p3b"
    p2_dir: Path = REPO / "runs/cv"
    p2_models: Path = REPO / "runs/models"
    start_h: float = PHASE2_CUM_H
    budget_h: float = BUDGET_GPU_H
    allowance_h: float = ALLOWANCE_H
    embed_python: Path = REPO / "third_party/embed-venv/bin/python"
    _p2_ctx: cv.Ctx | None = field(default=None, repr=False)

    @property
    def p3b_models(self) -> Path:
        return self.cv_dir / "models"

    @property
    def p3b_fits(self) -> Path:
        return self.cv_dir / "fits"

    @property
    def p2_cache(self) -> Path:
        return self.p2_dir / "cache"

    @property
    def pids(self) -> Path:
        return self.cv_dir / "pids"

    @property
    def anchor(self) -> Path:
        """Frozen paths are stored relative to runs/ so runs/p3b and runs/cv both resolve."""
        return self.cv_dir.parent

    @property
    def p2(self) -> cv.Ctx:
        if self._p2_ctx is None:
            self._p2_ctx = cv.Ctx(cv_dir=self.p2_dir, ref_dir=self.ref_dir, audio_dir=self.audio_dir, jobs=1,
                                  out=self.out)
        return self._p2_ctx


def rel(ctx: P3bCtx, p: Path) -> str:
    return os.path.relpath(Path(p).absolute(), ctx.anchor.absolute())


def unrel(ctx: P3bCtx, r: str) -> Path:
    return ctx.anchor / r


def cache_dir(ctx: P3bCtx, system: str, session: int, rid: str) -> Path:
    return ctx.cache_root / system / f"s{session}" / rid


def write_pid(ctx: P3bCtx, name: str, pid: int, cmd: str) -> Path:
    ctx.pids.mkdir(parents=True, exist_ok=True)
    p = ctx.pids / f"{name}.pid"
    p.write_text(json.dumps({"pid": pid, "cmd": cmd, "started": cv.now(), "host_pid_parent": os.getpid()}) + "\n")
    return p


class P3bLedger(cv.Ledger):
    """bnd.cv ledger semantics (abort before launch above 12 GPU-h cumulative) + a WARNING at the
    phase-3b allowance (start + 2.0 h = 6.11 h) + pid files for running GPU children."""

    def extra_h(self) -> float:
        return self.used_h() - self.ctx.start_h

    def warn_line(self) -> float:
        return self.ctx.start_h + self.ctx.allowance_h

    def check(self, estimate_s: float, what: str = "") -> None:
        super().check(estimate_s, what)
        if self.used_h() + estimate_s / 3600 > self.warn_line():
            self.ctx.out(f"[p3b] WARNING: GPU ledger {self.used_h():.3f} h + estimate {estimate_s / 3600:.3f} h for "
                         f"{what} exceeds the phase-3b allowance line {self.warn_line():.2f} h "
                         f"({self.ctx.allowance_h} h over {self.ctx.start_h} h); continuing (abort only at "
                         f"{self.ctx.budget_h} h)")

    def run_est(self, cmd: list, kind: str, tag: str, log: Path, est_s: float, audio_s: float = 0.0) -> dict:
        self.check(est_s, f"{kind} {tag}")
        full = [str(self.ctx.gpu_run)] + [str(c) for c in cmd]
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        card, t_card = None, None
        pidf = None
        with open(log, "a") as lf:
            lf.write(f"\n===== {cv.now()} {shlex.join(full)}\n")
            lf.flush()
            p = subprocess.Popen(full, cwd=str(self.ctx.repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            pidf = write_pid(self.ctx, "gpu_" + re.sub(r"[^A-Za-z0-9_.-]", "_", tag), p.pid, shlex.join(full))
            try:
                assert p.stdout is not None
                for line in p.stdout:
                    lf.write(line)
                    if card is None:
                        m = cv.CARD_RE.search(line)
                        if m:
                            card, t_card = m.group(1), time.time()
                rc = p.wait()
            except BaseException:
                p.terminate()
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                raise
            finally:
                if p.poll() is not None and pidf.exists():
                    pidf.unlink()
        t1 = time.time()
        e = {"ts_start": _dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
             "ts_end": _dt.datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
             "kind": kind, "tag": tag, "cmd": shlex.join(full), "card": card, "returncode": rc,
             "wall_s": round(t1 - t0, 3), "gpu_wall_s": round(t1 - t_card, 3) if t_card else 0.0,
             "audio_s": audio_s, "estimate_s": round(est_s, 1), "log": str(log), "phase": "3b"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(e) + "\n")
        self.ctx.out(f"[p3b] gpu {kind} {tag}: rc={rc} card={card} gpu_wall={e['gpu_wall_s']:.0f}s "
                     f"ledger={self.used_h():.3f}/{self.ctx.budget_h} h (phase-3b {self.extra_h():.3f}/"
                     f"{self.ctx.allowance_h} h)")
        if rc != 0:
            raise P3bError(f"GPU command failed (rc {rc}): {shlex.join(full)}; log {log}")
        if self.used_h() > self.ctx.budget_h:
            raise cv.BudgetError(f"GPU budget exceeded after {tag}: {self.used_h():.3f} h")
        if self.used_h() > self.warn_line():
            self.ctx.out(f"[p3b] WARNING: phase-3b GPU allowance exceeded: {self.extra_h():.3f} h > "
                         f"{self.ctx.allowance_h} h")
        return e


# ----------------------------------------------------------------------------------------------
# cost model
# ----------------------------------------------------------------------------------------------
@dataclass
class Rates:
    nemo_per_audio_s: float = DEF_NEMO_PER_AUDIO_S
    embed_per_audio_theta_s: float = DEF_EMBED_PER_AUDIO_THETA_S
    fit_s: dict = field(default_factory=dict)
    basis: str = "defaults"

    def fit(self, tag: str) -> float:
        return self.fit_s.get(tag, DEF_FIT_S)

    def infer(self, audio_s: float, n_theta: int, embed: bool, s80: bool = False) -> float:
        n = audio_s * self.nemo_per_audio_s * (HEADROOM if s80 else 1.0)
        e = audio_s * n_theta * self.embed_per_audio_theta_s * HEADROOM if embed else 0.0
        return JOB_FIXED_S + n + e

    def embed(self, audio_s: float, n_theta: int) -> float:
        return JOB_FIXED_S + audio_s * n_theta * self.embed_per_audio_theta_s * HEADROOM


def phase2_rates(ctx: P3bCtx) -> Rates:
    r = Rates()
    tn = te = ta = 0.0
    for p in sorted(ctx.p2_cache.glob("*/infer_report.json")) if ctx.p2_cache.is_dir() else []:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        n_t = max(1, len(next(iter(d.get("recordings", {}).values()), {}).get("tracks", {}) or [1, 1, 1]))
        if d.get("nemo_wall_s") and d.get("embed_wall_s") and d.get("total_audio_s"):
            tn += d["nemo_wall_s"]
            te += d["embed_wall_s"] / n_t
            ta += d["total_audio_s"]
    if ta > 0:
        r.nemo_per_audio_s, r.embed_per_audio_theta_s = tn / ta, te / ta
        r.basis = f"phase-2 infer reports ({ta:.0f} audio-s)"
    led = ctx.p2_dir / "gpu_ledger.jsonl"
    if led.exists():
        for l in led.read_text().splitlines():
            if l.strip():
                e = json.loads(l)
                if e.get("kind") == "train" and e.get("returncode") == 0:
                    r.fit_s[e["tag"]] = max(r.fit_s.get(e["tag"], 0.0), e["gpu_wall_s"])
    return r


# ----------------------------------------------------------------------------------------------
# phase-2 inputs (read only)
# ----------------------------------------------------------------------------------------------
def phase2_selection(ctx: P3bCtx) -> dict:
    """Phase-2 selection, after re-verifying that phase 2 is still frozen, and checking that its
    selected recipes are the ones PREREG_phase3b registers."""
    cv.verify_frozen(ctx.p2)
    sel2 = json.loads((ctx.p2_dir / "selection.json").read_text())
    got = {fi: sel2["folds"][fi]["selected_recipe"] for fi in sorted(FOLDS)}
    if got != REGISTERED_FT_RECIPE:
        raise P3bError(f"phase-2 selected recipes {got} != PREREG_phase3b {REGISTERED_FT_RECIPE}")
    return sel2


def ft_recipe(fi: str) -> str:
    return REGISTERED_FT_RECIPE[fi]


def ft_inner_units(fi: str) -> list[tuple[str, str]]:
    """(inner-fit system, rid): fold ``fi``'s pooled FT inner set = the selected recipe's inner fits
    that exclude ``fi``, each on its own held-out inner fold (bnd.cv.inner_sources)."""
    out = []
    for fit, _fj, recs in cv.inner_sources(fi, ft_recipe(fi)):
        for rid in recs:
            if rid in FOLDS[fi] or rid in fit["train_recs"] or set(FOLDS[fi]) & set(fit["train_recs"]):
                raise P3bError(f"leak: {fit['system']} / {rid} for outer {fi}")
            out.append((fit["system"], rid))
    return out


def zs_inner_units(fi: str) -> list[tuple[str, str]]:
    return [("zeroshot", r) for f in sorted(FOLDS) if f != fi for r in FOLDS[f]]


def units_for(system_kind: str, fi: str) -> list[tuple[str, str]]:
    return zs_inner_units(fi) if system_kind == "zs" else ft_inner_units(fi)


def regen_specs() -> list[dict]:
    """The 14 unique inner fits of the selected recipes (every fit serving some fold's FT pool)."""
    tags = sorted({s for fi in sorted(FOLDS) for s, _ in ft_inner_units(fi)})
    fits = {f["tag"]: f for f in cv.inner_fits()}
    return [fits[t] for t in tags]


def refit_system(fi: str) -> str:
    return cv.refit_spec(fi, ft_recipe(fi))["system"]


# ----------------------------------------------------------------------------------------------
# internal GPU entry points (run under scripts/gpu_run.sh by the driver)
# ----------------------------------------------------------------------------------------------
def train_fit_main(argv: Sequence[str]) -> int:
    """bnd.train with its fixed output dirs redirected to runs/p3b (the recipe, data, code and seed are
    bnd.train's own; only FITS_DIR / MODELS_DIR change)."""
    ap = argparse.ArgumentParser(prog="bnd.p3b train-fit")
    ap.add_argument("--fits-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, required=True)
    a, rest = ap.parse_known_args(argv)
    from bnd import train as tr

    tr.FITS_DIR, tr.MODELS_DIR = a.fits_dir, a.models_dir
    return tr.main(rest)


def prep_job_main(argv: Sequence[str]) -> int:
    """One GPU job per system: S=80 inference (+ tracks + embeddings) of ``--s80`` recordings, then
    S=20 embeddings of ``--s20-embed`` recordings (tracks already written by the driver)."""
    ap = argparse.ArgumentParser(prog="bnd.p3b prep-job")
    ap.add_argument("--system", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--theta-act", nargs="+", required=True)
    ap.add_argument("--s80", nargs="*", default=[])
    ap.add_argument("--s20-embed", nargs="*", default=[])
    a = ap.parse_args(argv)
    from bnd import infer

    if a.s80:
        infer.main(["--system", a.system, "--model", str(a.model), "--recordings", *a.s80, "--theta-act",
                    *a.theta_act, "--cache-root", str(a.cache_root), "--core-s", "80", "--session-subdir"])
    if a.s20_embed:
        infer.run_embed(a.cache_root / a.system / "s20", list(a.s20_embed), [float(t) for t in a.theta_act], "cuda")
    return 0


# ----------------------------------------------------------------------------------------------
# regen
# ----------------------------------------------------------------------------------------------
def p2_fit_marker(ctx: P3bCtx, tag: str) -> dict:
    m = cv.read_marker(ctx.p2, "fit", tag)
    if m is None:
        raise P3bError(f"phase-2 fit marker for {tag} missing")
    return m


def regen_fit_cmd(ctx: P3bCtx, spec: dict) -> list:
    return [ctx.python, "-m", "bnd.p3b", "train-fit", "--fits-dir", ctx.p3b_fits, "--models-dir", ctx.p3b_models,
            "--tag", spec["tag"], "--train-recs", ",".join(spec["train_recs"]), "--lr", spec["lr"],
            "--timing", spec["timing"]]


def regen_infer_cmd(ctx: P3bCtx, spec: dict) -> list:
    return [ctx.python, "-m", "bnd.infer", "--system", spec["system"], "--model", ctx.p3b_models / f"{spec['tag']}.nemo",
            "--recordings", *spec["heldout_recs"], "--theta-act", *[f"{t:.2f}" for t in THETA_ACT],
            "--cache-root", ctx.cache_root, "--core-s", "20", "--session-subdir", "--no-embed"]


def compare_probs(new: Path, old: Path) -> dict:
    """max |p_new - p_phase2| over every window of one recording (identical windows required)."""
    wn = json.loads((new / "windows.json").read_text())
    wo = json.loads((old / "windows.json").read_text())
    if wn != wo:
        return {"ok": False, "reason": "windows differ", "max_abs_diff": None}
    mx = 0.0
    for w in wo:
        a = np.load(old / f"probs_{w['window']}.npy").astype(np.float32)
        b = np.load(new / f"probs_{w['window']}.npy").astype(np.float32)
        if a.shape != b.shape:
            return {"ok": False, "reason": f"window {w['window']} shape {b.shape} != {a.shape}", "max_abs_diff": None}
        mx = max(mx, float(np.abs(a - b).max()) if a.size else 0.0)
    return {"ok": bool(mx <= REGEN_TOL), "max_abs_diff": mx, "windows": len(wo)}


def verify_regen(ctx: P3bCtx, spec: dict) -> dict:
    per = {rid: compare_probs(cache_dir(ctx, spec["system"], 20, rid), ctx.p2_cache / spec["system"] / rid)
           for rid in spec["heldout_recs"]}
    bad = {r: v for r, v in per.items() if not v["ok"]}
    if bad:
        cv.write_json(ctx.cv_dir / "regen_failure.json", {"ts": cv.now(), "fit": spec["tag"], "tolerance": REGEN_TOL,
                                                          "per_recording": per})
        raise RegenError(f"STOP (PREREG_phase3b): regenerated {spec['tag']} does not reproduce the phase-2 probs "
                         f"within {REGEN_TOL}: {bad}; see {ctx.cv_dir / 'regen_failure.json'}")
    return per


def regen_one(ctx: P3bCtx, ledger: P3bLedger | None, rates: Rates, spec: dict) -> dict:
    """Returns {"done": bool, "est_s": float}."""
    from bnd import infer

    tag = spec["tag"]
    m2 = p2_fit_marker(ctx, tag)
    ins = cv.fit_inputs(ctx, spec)
    if ins != m2["inputs"]:
        diff = sorted(k for k in set(ins) | set(m2["inputs"]) if ins.get(k) != m2["inputs"].get(k))
        raise P3bError(f"regen {tag}: fit inputs differ from the phase-2 fit marker in {diff}; not an identical refit")
    for rid in spec["heldout_recs"]:
        if not infer.cache_complete(ctx.p2_cache / tag / rid, {"system": tag}):
            raise P3bError(f"regen {tag}: phase-2 cached probs of {rid} missing")
    audio = sum(cv.durations(ctx, spec["heldout_recs"]).values())
    inputs = {"fit_inputs_sha": cv.canon_sha(ins), "phase2_model_sha256": m2["outputs"]["model_sha256"],
              "heldout": spec["heldout_recs"], "tolerance": REGEN_TOL, "code": cv.code_shas(ctx, "infer")}
    st, m = cv.marker_status(ctx, "regen", tag, inputs)
    model = ctx.p3b_models / f"{tag}.nemo"
    if st == "done" and model.exists() and cv.file_sha(ctx, model) == m["outputs"]["model_sha256"]:
        return {"done": True, "est_s": 0.0}
    if st == "stale" and not ctx.allow_stale:
        raise cv.StaleError(f"regen {tag}: done-marker inputs changed; rerun with --allow-stale")
    fit_rep = ctx.p3b_fits / tag / "fit_report.json"
    fit_ok = model.exists() and fit_rep.exists() and json.loads(fit_rep.read_text()).get("status") == "ok"
    est_fit = 0.0 if fit_ok else rates.fit(tag)
    est_inf = rates.infer(audio, len(THETA_ACT), embed=False)
    if ctx.dry_run:
        if not fit_ok:
            cv.say_dry(ctx, "gpu", regen_fit_cmd(ctx, spec))
        cv.say_dry(ctx, "gpu", regen_infer_cmd(ctx, spec))
        ctx.out(f"DRY-RUN regen {tag}: verify max|p - p_phase2| <= {REGEN_TOL} on {spec['heldout_recs']} "
                f"(est {(est_fit + est_inf) / 3600:.3f} GPU-h)")
        return {"done": False, "est_s": est_fit + est_inf}
    if not fit_ok:
        if model.exists() or (ctx.p3b_fits / tag).exists():
            for p in (model, ctx.p3b_fits / tag):
                d = cv.move_aside(p)
                if d:
                    ctx.out(f"[p3b] moved incomplete {p} -> {d}")
        ledger.run_est(regen_fit_cmd(ctx, spec), "train", f"regen:{tag}", ctx.logs / f"train_{tag}.log", est_fit)
        rep = json.loads(fit_rep.read_text())
        if rep.get("status") != "ok" or rep.get("train_recs") != spec["train_recs"] or not model.exists():
            raise P3bError(f"regen {tag}: fit did not finish ok ({rep.get('status')!r})")
    model_sha = cv.file_sha(ctx, model)
    s20_expect = {"system": tag, "model_sha256": model_sha, "core_s": 20.0}
    if not all(infer.cache_complete(cache_dir(ctx, tag, 20, r), s20_expect) for r in spec["heldout_recs"]):
        ledger.run_est(regen_infer_cmd(ctx, spec), "infer", f"regen_s20:{tag}", ctx.logs / f"infer_{tag}_s20.log",
                       est_inf, audio_s=audio)
    for rid in spec["heldout_recs"]:
        if not infer.cache_complete(cache_dir(ctx, tag, 20, rid), s20_expect):
            raise P3bError(f"regen {tag}: S=20 cache of {rid} incomplete after inference")
    per = verify_regen(ctx, spec)
    delta = (json.loads(fit_rep.read_text()).get("data", {}).get("delta") or {}).get("delta_s")
    cv.write_marker(ctx, "regen", tag, inputs, {"model": str(model), "model_sha256": model_sha,
                                                "phase2_model_sha256": m2["outputs"]["model_sha256"],
                                                "model_bytes_identical_to_phase2": model_sha == m2["outputs"]["model_sha256"],
                                                "delta_s": delta, "phase2_delta_s": m2["outputs"].get("delta_s"),
                                                "probs_check": per})
    ctx.out(f"[p3b] regen {tag}: reproduced (max |dp| {max(v['max_abs_diff'] for v in per.values()):.2e})")
    return {"done": True, "est_s": est_fit + est_inf}


def cmd_regen(ctx: P3bCtx) -> dict:
    phase2_selection(ctx)
    rates = phase2_rates(ctx)
    ledger = P3bLedger(ctx)
    specs = regen_specs()
    if len(specs) != 14:
        raise P3bError(f"expected 14 unique inner fits, got {len(specs)}")
    tot, done = 0.0, 0
    for spec in specs:
        r = regen_one(ctx, ledger, rates, spec)
        tot += r["est_s"]
        done += r["done"]
    s = {"fits": len(specs), "done": done, "projected_gpu_h": round(tot / 3600, 3), "rates": rates.basis}
    ctx.out(("DRY-RUN " if ctx.dry_run else "[p3b] ") + "regen " + json.dumps(s))
    return s


def regen_done(ctx: P3bCtx, tag: str) -> dict:
    m = cv.read_marker(ctx, "regen", tag)
    if m is None:
        raise P3bError(f"{tag}: regenerated fit not verified (run `regen` first; PREREG: no use before the check)")
    return m


# ----------------------------------------------------------------------------------------------
# prep
# ----------------------------------------------------------------------------------------------
def prep_groups() -> list[dict]:
    """One group per system: {system, role, rids}."""
    groups: dict[str, dict] = {}

    def add(system, role, rid):
        groups.setdefault(system, {"system": system, "role": role, "rids": set()})["rids"].add(rid)

    for rid in RECORDINGS:
        add("zeroshot", "zeroshot", rid)
    for fi in sorted(FOLDS):
        for system, rid in ft_inner_units(fi):
            add(system, "inner", rid)
        for rid in FOLDS[fi]:
            add(refit_system(fi), "refit", rid)
    order = {"zeroshot": 0, "inner": 1, "refit": 2}
    return [dict(g, rids=sorted(g["rids"])) for g in sorted(groups.values(), key=lambda g: (order[g["role"]], g["system"]))]


def group_model(ctx: P3bCtx, g: dict) -> tuple[Path, str | None]:
    """(model path, expected sha256). Base/refit: the model that produced the phase-2 S=20 probs;
    inner: the verified regenerated fit."""
    if g["role"] == "inner":
        if ctx.dry_run and cv.read_marker(ctx, "regen", g["system"]) is None:
            return ctx.p3b_models / f"{g['system']}.nemo", None
        return ctx.p3b_models / f"{g['system']}.nemo", regen_done(ctx, g["system"])["outputs"]["model_sha256"]
    model = ctx.base_model if g["role"] == "zeroshot" else ctx.p2_models / f"{g['system']}.nemo"
    shas = {json.loads((ctx.p2_cache / g["system"] / r / "meta.json").read_text())["model_sha256"] for r in g["rids"]}
    if len(shas) != 1:
        raise P3bError(f"{g['system']}: phase-2 caches from different models {shas}")
    return model, shas.pop()


def _symlink(dst: Path, target: Path) -> None:
    target = target.resolve()
    if dst.is_symlink():
        if dst.resolve() == target:
            return
        raise P3bError(f"{dst} is a symlink to {dst.resolve()}, expected {target}")
    if dst.exists():
        raise P3bError(f"{dst} exists and is not the expected symlink to {target}")
    dst.symlink_to(target)


def mirror_phase2_s20(ctx: P3bCtx, system: str, rid: str) -> Path:
    """runs/p3b/cache/<system>/s20/<rid>/: read-only symlinks to the phase-2 probs/windows/meta (tracks
    and embeddings for the phase-3b theta_act are real files written here)."""
    src, dst = ctx.p2_cache / system / rid, cache_dir(ctx, system, 20, rid)
    if dst.resolve().is_relative_to(ctx.p2_dir.resolve()):
        raise P3bError(f"mirror {dst} lies inside the phase-2 tree")
    dst.mkdir(parents=True, exist_ok=True)
    windows = json.loads((src / "windows.json").read_text())
    for name in ["meta.json", "windows.json"] + [f"probs_{w['window']}.npy" for w in windows]:
        _symlink(dst / name, src / name)
    for t in THETA_ACT:
        for name in (f"tracks_{t:.2f}.json", f"emb_{t:.2f}.npy", f"emb_{t:.2f}_meta.json"):
            if (dst / name).is_symlink():
                raise P3bError(f"{dst / name} is a symlink; phase-3b tracks/embeddings must be real files")
    return dst


def _emb_ok(d: Path, theta: float) -> bool:
    from bnd import embed, infer

    if not infer.emb_is_current(d, theta):
        return False
    try:
        em = json.loads(infer.emb_paths(d, theta)[1].read_text())
    except (OSError, ValueError):
        return False
    return em.get("model") == embed.MODEL_ID and em.get("revision") == embed.MODEL_REV


def _cache_ok(ctx: P3bCtx, system: str, session: int, rid: str, model_sha: str | None) -> bool:
    from bnd import infer

    d = cache_dir(ctx, system, session, rid)
    exp = {"system": system, "core_s": float(session)}
    if model_sha:
        exp["model_sha256"] = model_sha
    return infer.cache_complete(d, exp) and all(_emb_ok(d, t) for t in THETA_ACT)


def prep_inputs(ctx: P3bCtx, g: dict, model_sha: str | None) -> dict:
    src = {}
    if g["role"] == "inner":
        m = cv.read_marker(ctx, "regen", g["system"])
        src = {"regen_inputs_sha": m["inputs_sha"] if m else "MISSING"}
    else:
        for rid in g["rids"]:
            d = ctx.p2_cache / g["system"] / rid
            probs = sorted(d.glob("probs_*.npy"))
            src[rid] = {"meta": cv.file_sha(ctx, d / "meta.json", True), "windows": cv.file_sha(ctx, d / "windows.json", True),
                        "probs": cv.canon_sha([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in probs])}
    return {"system": g["system"], "role": g["role"], "rids": g["rids"], "sessions": list(SESSIONS),
            "theta_act": [f"{t:.2f}" for t in THETA_ACT], "model_sha256": model_sha, "s20_source": src,
            "code": cv.code_shas(ctx, "infer", "embed")}


def prep_outputs(ctx: P3bCtx, g: dict, model_sha: str) -> dict:
    out = {}
    for s in SESSIONS:
        for rid in g["rids"]:
            if not _cache_ok(ctx, g["system"], s, rid, model_sha):
                raise P3bError(f"{g['system']}/s{s}/{rid}: probs/tracks/embeddings missing or stale after prep")
            d = cache_dir(ctx, g["system"], s, rid)
            out[f"s{s}/{rid}"] = {f"{t:.2f}": [cv.file_sha(ctx, d / f"tracks_{t:.2f}.json"),
                                               cv.file_sha(ctx, d / f"emb_{t:.2f}.npy")] for t in THETA_ACT}
    return out


def prep_job_cmd(ctx: P3bCtx, g: dict, model: Path, s80: Sequence[str], s20: Sequence[str]) -> list:
    return [ctx.python, "-m", "bnd.p3b", "prep-job", "--system", g["system"], "--model", model, "--cache-root",
            ctx.cache_root, "--theta-act", *[f"{t:.2f}" for t in THETA_ACT], "--s80", *s80, "--s20-embed", *s20]


def prep_group(ctx: P3bCtx, ledger: P3bLedger | None, rates: Rates, g: dict) -> dict:
    from bnd import infer

    model, model_sha = group_model(ctx, g)
    inputs = prep_inputs(ctx, g, model_sha)
    st, m = cv.marker_status(ctx, "prep", g["system"], inputs)
    if st == "done":
        try:
            if prep_outputs(ctx, g, model_sha) == m["outputs"]:
                return {"done": True, "est_s": 0.0}
        except P3bError:
            pass
        ctx.out(f"[p3b] prep {g['system']}: marker present but outputs changed; redoing")
    elif st == "stale" and not ctx.allow_stale:
        raise cv.StaleError(f"prep {g['system']}: done-marker inputs changed; rerun with --allow-stale")
    s80 = [r for r in g["rids"] if st == "stale" or not _cache_ok(ctx, g["system"], 80, r, model_sha)]
    s20 = [r for r in g["rids"] if st == "stale" or not _cache_ok(ctx, g["system"], 20, r, model_sha)]
    durs = cv.durations(ctx, g["rids"])
    est = 0.0
    if s80 or s20:
        est = (rates.infer(sum(durs[r] for r in s80), len(THETA_ACT), embed=True, s80=True) if s80 else 0.0) \
            + (rates.embed(sum(durs[r] for r in s20), len(THETA_ACT)) - (JOB_FIXED_S if s80 else 0.0) if s20 else 0.0)
    if ctx.dry_run:
        src = "phase-2 cached probs (symlinks)" if g["role"] != "inner" else "verified regen probs"
        ctx.out(f"DRY-RUN prep {g['role']:8s} {g['system']}: S=20 from {src}, tracks+embeddings theta "
                f"{[f'{t:.2f}' for t in THETA_ACT]} for {s20}; S=80 inference+tracks+embeddings for {s80} "
                f"(est {est / 3600:.3f} GPU-h)")
        if s80 or s20:
            cv.say_dry(ctx, "gpu", prep_job_cmd(ctx, g, model, s80, s20))
        return {"done": False, "est_s": est}
    if not model.exists():
        raise P3bError(f"{g['system']}: model {model} missing")
    if model_sha is None or cv.file_sha(ctx, model) != model_sha:
        raise P3bError(f"{g['system']}: {model} sha256 differs from the model that produced its S=20 probs")
    for rid in s20:
        if g["role"] == "inner":
            if not infer.cache_complete(cache_dir(ctx, g["system"], 20, rid),
                                        {"system": g["system"], "model_sha256": model_sha, "core_s": 20.0}):
                raise P3bError(f"{g['system']}/s20/{rid}: regen cache missing (is {rid} a held-out recording?)")
        else:
            mirror_phase2_s20(ctx, g["system"], rid)
        for t in THETA_ACT:
            infer.write_tracks(cache_dir(ctx, g["system"], 20, rid), t)  # CPU, from cached probs
    if s80 or s20:
        ledger.run_est(prep_job_cmd(ctx, g, model, s80, s20), "prep", f"prep:{g['system']}",
                       ctx.logs / f"prep_{g['system']}.log", est, audio_s=sum(durs[r] for r in set(s80) | set(s20)))
    cv.write_marker(ctx, "prep", g["system"], inputs, prep_outputs(ctx, g, model_sha))
    return {"done": True, "est_s": est}


def cmd_prep(ctx: P3bCtx) -> dict:
    phase2_selection(ctx)
    rates = phase2_rates(ctx)
    ledger = P3bLedger(ctx)
    groups = prep_groups()
    if not ctx.dry_run:
        for g in groups:
            if g["role"] == "inner":
                regen_done(ctx, g["system"])
    tot, n_done = 0.0, 0
    for g in groups:
        r = prep_group(ctx, ledger, rates, g)
        tot += r["est_s"]
        n_done += r["done"]
    used = ledger.used_h()
    s = {"groups": len(groups), "groups_done": n_done, "caches": 2 * sum(len(g["rids"]) for g in groups),
         "projected_gpu_h": round(tot / 3600, 3), "ledger_used_h": round(used, 3),
         "phase3b_used_h": round(used - ctx.start_h, 3), "rates": rates.basis}
    ctx.out(("DRY-RUN " if ctx.dry_run else "[p3b] ") + "prep " + json.dumps(s))
    return s


def check_prepped(ctx: P3bCtx, systems: Iterable[str]) -> None:
    groups = {g["system"]: g for g in prep_groups()}
    bad = []
    for s in sorted(set(systems)):
        if cv.read_marker(ctx, "prep", s) is None:
            bad.append(s)
            continue
        _, sha = group_model(ctx, groups[s])
        st, _ = cv.marker_status(ctx, "prep", s, prep_inputs(ctx, groups[s], sha))
        if st != "done":
            bad.append(f"{s} ({st})")
    if bad:
        raise P3bError(f"prep not done for {bad[:5]}; run `prep` first")


# ----------------------------------------------------------------------------------------------
# linking (CPU, in process)
# ----------------------------------------------------------------------------------------------
def hyp_path(ctx: P3bCtx, system: str, cfg: Config, rid: str) -> Path:
    return ctx.hyp_root / system / cfg.key / f"{rid}.rttm"


def _write_rttm_atomic(path: Path, rid: str, segs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    S.write_rttm(tmp, rid, segs)
    tmp.replace(path)


def link_configs(cache, theta: float, cfgs: Sequence[Config], oracle: str = "none", ref=None) -> dict:
    """{cfg.key: (segs, info)} for configs sharing one (session, theta_act) cache: each (linker,
    theta_clu) labelling and its frame timeline are computed once, gap-fill variants reuse them.
    Equivalent to bnd.link.run_recording per config (tested)."""
    from bnd import link as L

    out = {}
    groups: dict[tuple, list[Config]] = {}
    for c in cfgs:
        if not math.isclose(c.theta_act, theta):
            raise P3bError(f"{c.key} does not match cache theta {theta}")
        groups.setdefault((c.linker, c.theta_clu) if oracle == "none" else ("oracle", 0.0), []).append(c)
    for (linker, clu), cs in groups.items():
        labels, info = L.track_labels(cache, theta, clu, linker if oracle == "none" else "avg", oracle, ref)
        lab0 = L.frame_labels(cache, labels)
        for c in cs:
            lab = L.gap_fill(lab0, cache.dt, float(c.gap))
            segs = L.frames_to_segs(lab, cache.dt, cache.duration)
            inf = dict(info)
            inf.update(n_output_speakers=len({s.speaker for s in segs}), n_output_turns=len(segs),
                       output_speech_s=round(sum(s.end - s.start for s in segs), 3),
                       gap_filled_s=round(float(((lab0 < 0) & (lab >= 0)).sum() * cache.dt), 3))
            out[c.key] = (segs, inf)
    return out


def _link_task(args) -> dict:
    """Worker: one (system, session, rid, theta_act) cache -> RTTMs of the given configurations."""
    from bnd import link as L

    cache_d, rid, theta, cfg_keys, out_paths, report_path, oracle, ref_path = args
    cache = L.load_cache(Path(cache_d), theta, rid)
    ref = S.load_rttm_file(ref_path, rid) if oracle != "none" else None
    res = link_configs(cache, theta, [parse_key(k) for k in cfg_keys], oracle, ref)
    shas, infos = {}, {}
    for k, outp in zip(cfg_keys, out_paths):
        segs, info = res[k]
        _write_rttm_atomic(Path(outp), rid, segs)
        shas[k] = cv.sha_bytes(Path(outp).read_bytes())
        infos[k] = info
    if report_path:
        cv.write_json(Path(report_path), infos)
    return shas


def cache_shas(ctx: P3bCtx, system: str, session: int, rid: str, theta: float) -> list[str]:
    d = cache_dir(ctx, system, session, rid)
    return [cv.file_sha(ctx, d / n, ctx.dry_run) for n in
            ("meta.json", "windows.json", f"tracks_{theta:.2f}.json", f"emb_{theta:.2f}.npy")]


def search_units() -> list[tuple[str, str]]:
    return sorted({u for fi in sorted(FOLDS) for u in ft_inner_units(fi)} | {("zeroshot", r) for r in RECORDINGS})


def search_tasks(ctx: P3bCtx) -> list[dict]:
    tasks = []
    for system, rid in search_units():
        for s in SESSIONS:
            for t in THETA_ACT:
                cfgs = configs_for(s, t)
                tasks.append({"name": f"{system}__s{s}__{rid}__{t:.2f}", "system": system, "rid": rid, "session": s,
                              "theta": t, "cfgs": cfgs, "outs": [hyp_path(ctx, system, c, rid) for c in cfgs],
                              "report": ctx.hyp_root / system / "_reports" / f"s{s}__{rid}__{t:.2f}.json"})
    return tasks


def _task_inputs(ctx: P3bCtx, t: dict) -> dict:
    return {"system": t["system"], "rid": t["rid"], "session": t["session"], "theta_act": f"{t['theta']:.2f}",
            "configs": [c.key for c in t["cfgs"]], "cache": cache_shas(ctx, t["system"], t["session"], t["rid"], t["theta"]),
            "code": cv.code_shas(ctx, "link", "score", "p3b")}


def run_link_tasks(ctx: P3bCtx, tasks: list[dict]) -> int:
    todo = []
    for t in tasks:
        inputs = _task_inputs(ctx, t)
        st, m = cv.marker_status(ctx, "plink", t["name"], inputs)
        if st == "done" and all(cv.file_sha(ctx, p, True) == m["outputs"]["rttm_sha256"].get(c.key)
                                for c, p in zip(t["cfgs"], t["outs"])):
            continue
        todo.append((t, inputs))
    if ctx.dry_run:
        return len(todo)
    args = [(str(cache_dir(ctx, t["system"], t["session"], t["rid"])), t["rid"], t["theta"], [c.key for c in t["cfgs"]],
             [str(p) for p in t["outs"]], str(t["report"]), "none", None) for t, _ in todo]
    if ctx.jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=min(ctx.jobs, 4), mp_context=mp.get_context("fork")) as ex:
            results = ex.map(_link_task, args)
            for (t, inputs), shas in zip(todo, results):
                cv.write_marker(ctx, "plink", t["name"], inputs, {"rttm_sha256": shas})
    else:
        for (t, inputs), a in zip(todo, args):
            cv.write_marker(ctx, "plink", t["name"], inputs, {"rttm_sha256": _link_task(a)})
    return len(todo)


# ----------------------------------------------------------------------------------------------
# scoring with the outer lock (frozen paths relative to runs/)
# ----------------------------------------------------------------------------------------------
def verify_frozen_p3b(ctx: P3bCtx) -> dict:
    if not ctx.frozen_path.exists():
        raise cv.OuterLockError(f"OUTER LOCK: {ctx.frozen_path} does not exist; run `freeze` after all outer "
                                f"predictions are complete")
    fz = json.loads(ctx.frozen_path.read_text())
    bad = [f"{s}/{r}" for s, recs in fz["systems"].items() for r, e in recs.items()
           if cv.file_sha(ctx, unrel(ctx, e["path"]), True) != e["sha256"]]
    if fz.get("selection_sha256") != cv.file_sha(ctx, ctx.selection_path, True):
        bad.append("selection.json")
    if fz.get("phase2_frozen_sha256") != cv.file_sha(ctx, ctx.p2.frozen_path, True):
        bad.append("phase-2 frozen.json")
    if bad:
        raise cv.OuterLockError(f"OUTER LOCK: frozen hashes no longer match for {bad[:6]}")
    return fz


def is_outer_artifact(ctx: P3bCtx, path: Path) -> bool:
    rp = Path(path).resolve()
    return (rp.is_relative_to(ctx.outer_root.resolve()) or rp.is_relative_to((ctx.p2_dir / "outer").resolve())
            or any(pt.startswith("cv_refit_") for pt in rp.parts))


def check_lock(ctx: P3bCtx, context: str, rid: str, path: Path) -> None:
    """Scoring ``rid`` in the context of outer fold ``context`` is allowed only from an exact frozen file
    (after `freeze`) when ``rid`` belongs to that fold, or when ``path`` is an outer artefact of either
    phase (then in ANY context)."""
    if context not in FOLDS:
        raise P3bError(f"unknown context {context}")
    if rid not in FOLDS[context] and not is_outer_artifact(ctx, path):
        return
    fz = verify_frozen_p3b(ctx)
    r = rel(ctx, path)
    shas = {e["path"]: e["sha256"] for recs in fz["systems"].values() for x, e in recs.items() if x == rid}
    if r not in shas or shas[r] != cv.file_sha(ctx, path):
        raise cv.OuterLockError(f"OUTER LOCK: {rid} (context {context}) may only be scored from a frozen outer "
                                f"prediction here; {path} is not one")


_DUR: dict[str, float] = {}


def score_items(ctx: P3bCtx, context: str, items: Sequence[tuple[str, Path]], collars: Sequence[float]
                ) -> list[dict[float, S.RecScore]]:
    """THE phase-3b scoring entry point: [(rid, hyp path)] -> per item {collar: RecScore}, outer lock on
    every item, content-addressed cache in runs/p3b/score_cache."""
    for rid, p in items:
        check_lock(ctx, context, rid, p)
    score_sha = cv.file_sha(ctx, ctx.repo / "bnd/score.py")
    todo, keys, out = [], [], [dict() for _ in items]
    seen: dict[str, int] = {}
    for n, (rid, p) in enumerate(items):
        wav = str((ctx.audio_dir / f"{rid}.wav").resolve())
        if wav not in _DUR:
            _DUR[wav] = S.audio_duration(wav)
        ref = ctx.ref_dir / f"{rid}.rttm"
        hsha, rsha = cv.file_sha(ctx, p), cv.file_sha(ctx, ref)
        for c in collars:
            key = cv.canon_sha([rid, hsha, rsha, _DUR[wav], c, score_sha])
            cp = ctx.cv_dir / "score_cache" / key[:2] / f"{key}.json"
            if cp.exists():
                out[n][c] = cv._rec_from_dict(json.loads(cp.read_text()))
            elif key in seen:
                keys.append((n, c, cp, seen[key]))
            else:
                seen[key] = len(todo)
                keys.append((n, c, cp, len(todo)))
                todo.append((rid, str(p), str(ref), _DUR[wav], c))
    if todo:
        if ctx.jobs > 1 and len(todo) > 1:
            with mp.get_context("fork").Pool(min(ctx.jobs, 4, len(todo))) as pool:
                res = pool.map(cv._score_one, todo, chunksize=4)
        else:
            res = [cv._score_one(t) for t in todo]
        for n, c, cp, j in keys:
            if not cp.exists():
                cv.write_json(cp, res[j])
            out[n][c] = cv._rec_from_dict(res[j])
    return out


# ----------------------------------------------------------------------------------------------
# selection: 1-SE rule + simplicity order
# ----------------------------------------------------------------------------------------------
def pooled_der(per: dict[str, dict], rids: Sequence[str]) -> float:
    return sum(per[r]["errors"] for r in rids) / sum(per[r]["total"] for r in rids)


def one_se_select(table: dict[Config, dict[str, dict]], rids: Sequence[str] | None = None) -> dict:
    """PREREG_phase3b selection over ``table`` {config: {rid: {"errors", "total"}}} restricted to ``rids``
    (default: all). c_min = argmin pooled DER (exact ties -> simpler); for every config
    SE = sample std (ddof 1) of per-recording DER differences (config - c_min) / sqrt(n); the chosen
    config is the SIMPLEST with pooled DER <= DER(c_min) + SE."""
    if not table:
        raise P3bError("empty selection table")
    rids = sorted(rids if rids is not None else next(iter(table.values())))
    n = len(rids)
    if n < 2:
        raise P3bError("1-SE rule needs >= 2 recordings")
    der = {c: pooled_der(v, rids) for c, v in table.items()}
    rec = {c: np.array([v[r]["errors"] / v[r]["total"] for r in rids]) for c, v in table.items()}
    m = min(der.values())
    cmin = min((c for c in table if der[c] <= m + cv.TIE_TOL), key=lambda c: c.simplicity)
    se = {c: float(np.std(rec[c] - rec[cmin], ddof=1) / math.sqrt(n)) for c in table}
    ok = [c for c in table if der[c] <= der[cmin] + se[c] + 1e-12]
    chosen = min(ok, key=lambda c: c.simplicity)
    return {"selected": chosen, "argmin": cmin, "der": der, "se": se, "eligible": ok, "n": n}


def loro_stability(table: dict[Config, dict[str, dict]]) -> dict:
    rids = sorted(next(iter(table.values())))
    full = one_se_select(table, rids)
    per = {}
    for r in rids:
        x = one_se_select(table, [q for q in rids if q != r])
        per[r] = {"selected": x["selected"].key, "argmin": x["argmin"].key}
    return {"dropped": per,
            "selected_recovered": sum(v["selected"] == full["selected"].key for v in per.values()),
            "argmin_recovered": sum(v["argmin"] == full["argmin"].key for v in per.values()), "n": len(rids)}


def pool_items(ctx: P3bCtx, units: Sequence[tuple[str, str]], fi: str, cfg: Config) -> list[tuple[str, Path]]:
    items = []
    for system, rid in units:
        if rid in FOLDS[fi]:
            raise P3bError(f"selection for {fi} would include outer recording {rid}")
        items.append((rid, hyp_path(ctx, system, cfg, rid)))
    inner = sorted(r for f in FOLDS if f != fi for r in FOLDS[f])
    if sorted(r for r, _ in items) != inner:
        raise P3bError(f"{fi}: pooled set {sorted(r for r, _ in items)} != inner recordings {inner}")
    return items


def select_table(ctx: P3bCtx, fi: str, units: Sequence[tuple[str, str]]) -> dict:
    cfgs = list(CONFIGS)
    items, spans = [], []
    for c in cfgs:
        its = pool_items(ctx, units, fi, c)
        spans.append((len(items), len(items) + len(its)))
        items += its
    missing = [str(p) for _, p in items if not p.is_file()]
    if missing:
        raise P3bError(f"select {fi}: {len(missing)} inner predictions missing (run `search` linking), e.g. {missing[:3]}")
    sc = score_items(ctx, fi, items, [S.PRIMARY_COLLAR])
    table: dict[Config, dict] = {}
    for c, (a, b) in zip(cfgs, spans):
        table[c] = {s.recording_id: {"errors": s.errors, "total": s.total, "der": s.der}
                    for s in (sc[i][S.PRIMARY_COLLAR] for i in range(a, b))}
    full = one_se_select(table)
    q2 = one_se_select({c: v for c, v in table.items() if c in Q2_CONFIGS})
    stab = loro_stability(table)
    return {"sources": {rid: system for system, rid in units},
            "selected": full["selected"].key, "selected_config": full["selected"].as_dict(),
            "selected_der": full["der"][full["selected"]], "argmin": full["argmin"].key,
            "argmin_der": full["der"][full["argmin"]], "se_of_selected": full["se"][full["selected"]],
            "eligible": sorted(c.key for c in full["eligible"]),
            "q2_selected": q2["selected"].key, "q2_selected_der": q2["der"][q2["selected"]],
            "q2_argmin": q2["argmin"].key, "loro_stability": stab,
            "configs": {c.key: dict(**c.as_dict(), der=full["der"][c], se_vs_argmin=full["se"][c],
                                    per_recording={r: v["der"] for r, v in table[c].items()},
                                    errors_s=sum(v["errors"] for v in table[c].values()),
                                    total_s=sum(v["total"] for v in table[c].values())) for c in cfgs}}


def select_all(ctx: P3bCtx) -> dict:
    sel = {"created": cv.now(), "prereg": "docs/PREREG_phase3b.md", "framing": FRAMING,
           "space": {"session_core_s": list(SESSIONS), "theta_act": list(THETA_ACT), "linker": list(LINKERS),
                     "theta_clu": list(THETA_CLU), "gap_fill_s": list(GAPS), "n_configs": len(CONFIGS)},
           "rule": "1-SE: SE = sample std (ddof 1) of per-recording DER differences (config - argmin) / sqrt(8); "
                   "simplest config with pooled DER <= DER(argmin) + SE; simplicity S 20<80, L avg<refine, "
                   "g 0<2<5, theta_act 0.40<0.25<0.10, theta_clu 0.60<0.55<0.65 (lexicographic)",
           "q2_space": "S=20, L=avg (27 configs), same rule", "collar": S.PRIMARY_COLLAR,
           "phase2_selection_sha256": cv.file_sha(ctx, ctx.p2_dir / "selection.json"), "folds": {}}
    for fi in sorted(FOLDS):
        f = {"inner_recs": sorted(r for x in FOLDS if x != fi for r in FOLDS[x]), "ft_recipe": ft_recipe(fi)}
        for kind in ("zs", "ft"):
            f[kind] = select_table(ctx, fi, units_for(kind, fi))
            ctx.out(f"[p3b] {fi} {kind}: 1-SE {f[kind]['selected']} ({100 * f[kind]['selected_der']:.2f}%), argmin "
                    f"{f[kind]['argmin']} ({100 * f[kind]['argmin_der']:.2f}%), Q2 {f[kind]['q2_selected']}, LORO "
                    f"{f[kind]['loro_stability']['selected_recovered']}/8")
        sel["folds"][fi] = f
    return sel


def cmd_search(ctx: P3bCtx) -> dict | None:
    tasks = search_tasks(ctx)
    if ctx.dry_run:
        n_units = len(search_units())
        ctx.out(f"DRY-RUN search: link {len(tasks)} (cache, S, theta_act) tasks = {n_units} units x {len(SESSIONS)} S x "
                f"{len(THETA_ACT)} theta_act, {len(CONFIGS) // (len(SESSIONS) * len(THETA_ACT))} configs each "
                f"({len(tasks) * len(CONFIGS) // (len(SESSIONS) * len(THETA_ACT))} RTTMs, CPU, <= {ctx.jobs} workers); "
                f"then per fold x {{zs, ft}}: {len(CONFIGS)} pooled inner DERs over 8 recordings (<= "
                f"{n_units * len(CONFIGS)} unique scorings, collar 0), 1-SE rule, argmin, LORO, Q2. GPU-h: 0")
        for fi in sorted(FOLDS):
            ctx.out(f"DRY-RUN search {fi}: ft {ft_recipe(fi)} from {sorted({s for s, _ in ft_inner_units(fi)})}; "
                    f"zs on {sorted(r for _, r in zs_inner_units(fi))}")
        return None
    if ctx.frozen_path.exists():
        verify_frozen_p3b(ctx)
        ctx.out("[p3b] frozen; selection is locked")
        return json.loads(ctx.selection_path.read_text())
    cv.guard_after_scoring(ctx, "search")
    phase2_selection(ctx)
    check_prepped(ctx, {t["system"] for t in tasks})
    n = run_link_tasks(ctx, tasks)
    ctx.out(f"[p3b] linked {n} new tasks of {len(tasks)}")
    sel = select_all(ctx)
    cv.write_json(ctx.selection_path, sel)
    return sel


# ----------------------------------------------------------------------------------------------
# outer
# ----------------------------------------------------------------------------------------------
def outer_jobs(ctx: P3bCtx, sel: dict) -> list[dict]:
    jobs = []
    for fi in sorted(FOLDS):
        f = sel["folds"][fi]
        for kind, system in (("zs", "zeroshot"), ("ft", refit_system(fi))):
            star, q2 = parse_key(f[kind]["selected"]), parse_key(f[kind]["q2_selected"])
            for name, cfg, oracle in ((f"{kind}_star", star, "none"), (f"{kind}_q2", q2, "none"),
                                      (f"{kind}_star_oracle_link", star, "link")):
                jobs.append({"name": name, "fold": fi, "system": system, "cfg": cfg, "oracle": oracle,
                             "recs": list(FOLDS[fi]), "out_dir": ctx.outer_root / name / fi})
    return jobs


def ensure_outer_link(ctx: P3bCtx, j: dict) -> bool:
    per = {rid: cache_shas(ctx, j["system"], j["cfg"].session, rid, j["cfg"].theta_act) for rid in j["recs"]}
    if j["oracle"] != "none":
        for rid in j["recs"]:
            per[rid].append(cv.file_sha(ctx, ctx.ref_dir / f"{rid}.rttm", ctx.dry_run))
    inputs = {"system": j["system"], "recordings": j["recs"], "config": j["cfg"].key, "oracle": j["oracle"],
              "cache": per, "code": cv.code_shas(ctx, "link", "score", "p3b")}
    name = cv.dir_marker_name(ctx, j["out_dir"])
    st, m = cv.marker_status(ctx, "link", name, inputs)
    if st == "done" and all(cv.file_sha(ctx, j["out_dir"] / f"{r}.rttm", True) == m["outputs"]["rttm_sha256"][r]
                            for r in j["recs"]):
        return True
    if ctx.dry_run:
        ctx.out(f"DRY-RUN [cpu] link {j['name']} {j['fold']} {j['system']} {j['cfg'].key} oracle={j['oracle']} "
                f"-> {j['out_dir']}")
        return False
    stray = [p.stem for p in j["out_dir"].glob("*.rttm") if p.stem not in j["recs"]] if j["out_dir"].exists() else []
    if stray:
        raise P3bError(f"{j['out_dir']}: unexpected RTTMs {stray}")
    shas = {}
    for rid in j["recs"]:
        r = _link_task((str(cache_dir(ctx, j["system"], j["cfg"].session, rid)), rid, j["cfg"].theta_act,
                        [j["cfg"].key], [str(j["out_dir"] / f"{rid}.rttm")], str(j["out_dir"] / f"_report_{rid}.json"),
                        j["oracle"], str(ctx.ref_dir / f"{rid}.rttm")))
        shas[rid] = r[j["cfg"].key]
    cv.write_marker(ctx, "link", name, inputs, {"out_dir": str(j["out_dir"]), "rttm_sha256": shas})
    return True


def cmd_outer(ctx: P3bCtx) -> None:
    if ctx.frozen_path.exists():
        verify_frozen_p3b(ctx)
        ctx.out("[p3b] outer predictions are frozen; nothing to do")
        return
    cv.guard_after_scoring(ctx, "outer")
    if not ctx.selection_path.exists():
        if ctx.dry_run:
            ctx.out("DRY-RUN outer: selection.json missing; per fold link the 2 outer recordings with zs_star/zs_q2 "
                    "(zero-shot cache) and ft_star/ft_q2 (selected refit cache) at the inner-selected configs, plus "
                    "oracle-link from each star system's own tracks (6 systems x 10 RTTMs, CPU, GPU-h 0)")
            for fi in sorted(FOLDS):
                ctx.out(f"DRY-RUN outer {fi}: {refit_system(fi)} + zeroshot on {list(FOLDS[fi])}")
            return
        raise P3bError("selection.json missing; run `search` first")
    sel = json.loads(ctx.selection_path.read_text())
    jobs = outer_jobs(ctx, sel)
    if not ctx.dry_run:
        check_prepped(ctx, {j["system"] for j in jobs})
    ok = all([ensure_outer_link(ctx, j) for j in jobs])
    if ok and not ctx.dry_run:
        idx = {"created": cv.now(), "selection_sha256": cv.file_sha(ctx, ctx.selection_path),
               "jobs": [dict({k: v for k, v in j.items() if k not in ("cfg", "out_dir")}, config=j["cfg"].key,
                             out_dir=rel(ctx, j["out_dir"])) for j in jobs]}
        cv.write_json(ctx.outer_root / "index.json", idx)
        ctx.out(f"[p3b] outer: {len(jobs)} link jobs done -> {ctx.outer_root / 'index.json'}")


# ----------------------------------------------------------------------------------------------
# freeze
# ----------------------------------------------------------------------------------------------
def cmd_freeze(ctx: P3bCtx) -> dict | None:
    idx_p = ctx.outer_root / "index.json"
    if not idx_p.exists():
        raise P3bError("freeze refused: runs/p3b/outer/index.json missing (outer predictions incomplete)")
    if not ctx.selection_path.exists():
        raise P3bError("freeze refused: selection.json missing")
    idx = json.loads(idx_p.read_text())
    if idx.get("selection_sha256") != cv.file_sha(ctx, ctx.selection_path):
        raise P3bError("freeze refused: selection.json changed after the outer predictions were made")
    fz2 = cv.verify_frozen(ctx.p2)
    systems: dict[str, dict] = {s: {} for s in OUTER_SYSTEMS + tuple(P2_REFERENCE)}
    problems = []
    for j in idx["jobs"]:
        d = unrel(ctx, j["out_dir"])
        m = cv.read_marker(ctx, "link", cv.dir_marker_name(ctx, d))
        if m is None:
            problems.append(f"{j['out_dir']}: no link marker")
            continue
        for rid in j["recs"]:
            p = d / f"{rid}.rttm"
            sha = cv.file_sha(ctx, p, True)
            if sha == "MISSING" or m["outputs"]["rttm_sha256"].get(rid) != sha:
                problems.append(f"{j['out_dir']}/{rid}.rttm missing or differs from its link marker")
                continue
            systems[j["name"]][rid] = {"path": rel(ctx, p), "sha256": sha, "fold": j["fold"],
                                       "source_system": j["system"], "config": j["config"], "oracle": j["oracle"]}
    for name, p2name in P2_REFERENCE.items():
        for rid, e in fz2["systems"][p2name].items():
            systems[name][rid] = {"path": rel(ctx, ctx.p2_dir / e["path"]), "sha256": e["sha256"], "fold": e["fold"],
                                  "source_system": e["source_system"], "phase2_system": p2name, "cell": e.get("cell"),
                                  "oracle": e.get("oracle")}
    for s in systems:
        miss = sorted(set(RECORDINGS) - set(systems[s]))
        if miss:
            problems.append(f"{s}: missing outer predictions for {miss}")
    if problems:
        raise P3bError(f"freeze refused: {problems[:8]}")
    fz = {"created": cv.now(), "prereg": "docs/PREREG_phase3b.md", "framing": FRAMING,
          "selection_sha256": cv.file_sha(ctx, ctx.selection_path),
          "phase2_frozen_sha256": cv.file_sha(ctx, ctx.p2.frozen_path),
          "phase2_frozen_identity": cv.frozen_identity(fz2), "path_anchor": "runs/", "systems": systems}
    if ctx.frozen_path.exists():
        old = verify_frozen_p3b(ctx)
        if old["systems"] != systems:
            raise cv.OuterLockError("already frozen with different outer predictions; refusing to re-freeze")
        ctx.out("[p3b] already frozen; hashes verified")
        return old
    ents = cv.scored_entries(ctx)
    if ents:
        last = ents[-1]
        if cv.frozen_identity(fz) == last["frozen_identity"]:
            fz["restored_from_scoring"] = last["scored"]
        else:
            cv.guard_after_scoring(ctx, "freeze")
            fz["deviation"] = ctx.deviation
            fz["supersedes"] = {"frozen_identity": last["frozen_identity"], "scored": last["scored"],
                                "results": last["history"]}
    if ctx.dry_run:
        ctx.out(f"DRY-RUN freeze {sum(len(v) for v in systems.values())} outer RTTMs")
        return None
    cv.write_json(ctx.frozen_path, fz)
    ctx.out(f"[p3b] frozen {sum(len(v) for v in systems.values())} outer RTTMs -> {ctx.frozen_path}")
    return fz


# ----------------------------------------------------------------------------------------------
# score
# ----------------------------------------------------------------------------------------------
def q1_verdict(zs0: Sequence[S.RecScore], ft0: Sequence[S.RecScore], boot0: dict) -> dict:
    """PREREG_phase3b Q1 claim conditions (delta = FT* - ZS*, collar 0)."""
    if [s.recording_id for s in zs0] != [s.recording_id for s in ft0]:
        raise P3bError("Q1 inputs on different recordings")
    delta = np.array([b.der - a.der for a, b in zip(zs0, ft0)])
    corpus_pp = 100 * (cv.pooled(ft0)["der"] - cv.pooled(zs0)["der"])
    wins = int((delta < 0).sum())
    conds = {
        "corpus_delta_le_-1pp": {"value_pp": corpus_pp, "pass": bool(corpus_pp <= -Q1_MIN_IMPROVEMENT_PP + 1e-9)},
        "bootstrap_p97_5_lt_0": {"ci_low_pp": 100 * boot0["ci_low"], "ci_high_pp": 100 * boot0["ci_high"],
                                 "pass": bool(boot0["ci_high"] < 0)},
        "improved_ge_6_of_10": {"wins": wins, "losses": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                                "n": len(delta), "pass": bool(wins >= Q1_MIN_WINS)},
        "macro_mean_delta_lt_0": {"value_pp": float(100 * delta.mean()), "pass": bool(delta.mean() < 0)},
    }
    helps = all(c["pass"] for c in conds.values())
    harm = bool(boot0["ci_low"] > 0)
    if helps:
        verdict = "fine-tuning helps"
    elif harm:
        verdict = "harm: fine-tuning is worse than calibrated zero-shot decoding (bootstrap 2.5th pct > 0)"
    else:
        verdict = "no demonstrated benefit over calibrated zero-shot decoding"
    return {"conditions": conds, "finetuning_helps": helps, "harm": harm, "verdict": verdict,
            "default_system": "FT" if helps else "ZS"}


def joint_mode(keys: Sequence[str]) -> dict:
    cnt: dict[str, int] = {}
    for k in keys:
        cnt[k] = cnt.get(k, 0) + 1
    top = max(cnt.values())
    tied = [k for k, v in cnt.items() if v == top]
    best = min(tied, key=lambda k: parse_key(k).simplicity)
    return {"config": best, "count": top, "of": len(keys), "tied": sorted(tied), "counts": cnt}


def speaker_counts(ctx: P3bCtx, fz: dict) -> dict:
    refn = {r: len({s.speaker for s in S.load_rttm_file(ctx.ref_dir / f"{r}.rttm", r)}) for r in RECORDINGS}
    out = {"reference": refn, "systems": {}}
    for system, recs in fz["systems"].items():
        err = {}
        for rid, e in recs.items():
            n = len({s.speaker for s in S.load_rttm_file(unrel(ctx, e["path"]), rid)})
            err[rid] = {"predicted": n, "error": n - refn[rid]}
        out["systems"][system] = {"per_recording": dict(sorted(err.items())),
                                  "sum_abs_error": sum(abs(v["error"]) for v in err.values())}
    return out


def cmd_score(ctx: P3bCtx) -> dict:
    fz = verify_frozen_p3b(ctx)
    ident = cv.frozen_identity(fz)
    ents = cv.scored_entries(ctx)
    if ents and ents[-1]["frozen_identity"] != ident and not fz.get("deviation"):
        raise cv.OuterLockError(f"OUTER LOCK: frozen.json differs from what was scored at {ents[-1]['scored']} but "
                                f"records no deviation; refusing to replace those results")
    sel = json.loads(ctx.selection_path.read_text())
    scores: dict[str, dict[float, list[S.RecScore]]] = {}
    for system, recs in fz["systems"].items():
        per_c: dict[float, list] = {c: [] for c in S.COLLARS}
        for fi in sorted(FOLDS):
            r = score_items(ctx, fi, [(rid, unrel(ctx, recs[rid]["path"])) for rid in FOLDS[fi]], S.COLLARS)
            for x in r:
                for c in S.COLLARS:
                    per_c[c].append(x[c])
        for c in S.COLLARS:
            per_c[c].sort(key=lambda s: s.recording_id)
        scores[system] = per_c
    res: dict = {"created": cv.now(), "frozen_created": fz["created"], "prereg": "docs/PREREG_phase3b.md",
                 "framing": FRAMING, "collars": list(S.COLLARS),
                 "bootstrap": {"n": ctx.bootstrap_n, "seed": S.BOOTSTRAP_SEED}, "systems": {}}
    for system, per_c in scores.items():
        res["systems"][system] = {str(c): {"corpus": S.corpus_summary(sc), "per_recording": [s.as_dict() for s in sc]}
                                  for c, sc in per_c.items()}
    q1 = cv.compare(scores["zs_star"], scores["ft_star"], ctx.bootstrap_n)
    res["q1"] = {"a": "zs_star (ZS*, inner-selected)", "b": "ft_star (FT*, inner-selected)", "by_collar": q1,
                 **q1_verdict(scores["zs_star"][0.0], scores["ft_star"][0.0], q1["0.0"])}
    der0 = {k: res["systems"][f"{k}_star"]["0.0"]["corpus"]["der"] for k in ("zs", "ft")}
    better = "ft" if der0["ft"] < der0["zs"] else "zs"
    res["q2"] = {"rule": "the better Q1 system = lower corpus DER at collar 0 among ZS*/FT*; delta = selected - "
                         "restricted (S=20, L=avg), paired bootstrap; both systems reported",
                 "better_q1_system": better,
                 "by_system": {k: {"a": f"{k}_q2 (restricted S=20, L=avg)", "b": f"{k}_star (selected)",
                                   "by_collar": cv.compare(scores[f"{k}_q2"], scores[f"{k}_star"], ctx.bootstrap_n)}
                               for k in ("zs", "ft")}}
    res["oracle_link"] = {k: {str(c): {"der": res["systems"][f"{k}_star"][str(c)]["corpus"]["der"],
                                       "oracle_link_der": res["systems"][f"{k}_star_oracle_link"][str(c)]["corpus"]["der"],
                                       "gap_pp": 100 * (res["systems"][f"{k}_star"][str(c)]["corpus"]["der"]
                                                        - res["systems"][f"{k}_star_oracle_link"][str(c)]["corpus"]["der"])}
                              for c in S.COLLARS} for k in ("zs", "ft")}
    res["phase2_reference"] = {"note": "phase-2 registered frozen RTTMs (runs/cv/outer), reference only",
                               "p3b_vs_p2": {k: cv.compare(scores[f"p2_{k}_sel"], scores[f"{k}_star"], ctx.bootstrap_n)
                                             for k in ("zs", "ft")}}
    res["speaker_count_errors"] = speaker_counts(ctx, fz)
    res["selection"] = {fi: {k: {"selected": f[k]["selected"], "selected_inner_der": f[k]["selected_der"],
                                 "argmin": f[k]["argmin"], "argmin_inner_der": f[k]["argmin_der"],
                                 "q2_selected": f[k]["q2_selected"],
                                 "loro_selected_recovered": f[k]["loro_stability"]["selected_recovered"]}
                             for k in ("zs", "ft")} | {"ft_recipe": f["ft_recipe"]} for fi, f in sel["folds"].items()}
    dflt = res["q1"]["default_system"].lower()
    res["default_system"] = {
        "rule": "Default = FT if Q1 'fine-tuning helps', else ZS. Final configuration = joint mode of the five inner "
                "selections of the default system (ties -> simpler).",
        "system": res["q1"]["default_system"],
        "final_configuration": joint_mode([f[dflt]["selected"] for f in sel["folds"].values()]),
        "next_step": ("train one all-10 model (lr 5e-5, timing shift, delta from all 10, 165 steps, seed 42); any OOF "
                      "number over all 10 recordings for the final configuration is a DEVELOPMENT number")
        if dflt == "ft" else "no training: the pinned base model with the final configuration is the default"}
    res["frozen_identity"] = ident
    if fz.get("deviation"):
        res["deviation"] = fz["deviation"]
        res["supersedes"] = fz.get("supersedes")
    n = len(ents) + 1
    hist = ctx.history_root / f"results_{n:03d}_{ident[:12]}.json"
    if hist.exists():
        raise P3bError(f"{hist} already exists; refusing to overwrite a scored result")
    cv.write_json(hist, res)
    cv.write_json(ctx.cv_dir / "results.json", res)
    (ctx.cv_dir / "results.md").write_text(results_markdown(res))
    cv.append_jsonl(ctx.scored_path, {"scored": res["created"], "frozen_identity": ident,
                                      "results_sha256": cv.file_sha(ctx, hist),
                                      "history": str(hist.relative_to(ctx.cv_dir)), "deviation": fz.get("deviation")})
    ctx.out(results_markdown(res))
    return res


def results_markdown(res: dict) -> str:
    pc = lambda x: f"{100 * x:.2f}"  # noqa: E731
    L = ["# Phase-3b results (outer folds): SECOND-LOOK numbers", "", f"> {res['framing']}", "",
         f"Frozen {res['frozen_created']}; scored {res['created']}. Bootstrap n={res['bootstrap']['n']}, seed "
         f"{res['bootstrap']['seed']}. Protocol docs/PREREG_phase3b.md.", "",
         "## Selection (inner folds only; 1-SE rule; LORO = leave-one-recording-out recoveries of the selection)", "",
         "| fold | system | selected | inner DER | argmin | argmin DER | Q2 (S=20, avg) | LORO |", "|---|---|---|---|---|---|---|---|"]
    for fi, s in sorted(res["selection"].items()):
        for k in ("zs", "ft"):
            x = s[k]
            L.append(f"| {fi} | {k.upper()}{' ' + s['ft_recipe'] if k == 'ft' else ''} | {x['selected']} | "
                     f"{pc(x['selected_inner_der'])} | {x['argmin']} | {pc(x['argmin_inner_der'])} | {x['q2_selected']} | "
                     f"{x['loro_selected_recovered']}/8 |")
    L += ["", "## Corpus DER (%) [MISS/FA/CONF]", "",
          "| system | " + " | ".join(f"collar {c}" for c in res["collars"]) + " |", "|---|" + "---|" * len(res["collars"])]
    for system, byc in res["systems"].items():
        cells = []
        for c in res["collars"]:
            k = byc[str(c)]["corpus"]
            cells.append(f"{pc(k['der'])} [{pc(k['miss_rate'])}/{pc(k['fa_rate'])}/{pc(k['conf_rate'])}]")
        L.append(f"| {system} | " + " | ".join(cells) + " |")
    names = ("zs_star", "ft_star", "zs_q2", "ft_q2", "p2_zs_sel", "p2_ft_sel")
    get = {s: {d["recording_id"]: d["der"] for d in res["systems"][s]["0.0"]["per_recording"]} for s in names}
    sc = res["speaker_count_errors"]["systems"]
    L += ["", "## Per-recording DER, collar 0 (%), and speaker-count error (pred - ref)", "",
          "| rec | ZS* | FT* | FT*-ZS* pp | ZS q2 | FT q2 | p2 zs_sel | p2 ft_sel | ZS* spk | FT* spk |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for rid in sorted(get["zs_star"]):
        L.append(f"| {rid} | {pc(get['zs_star'][rid])} | {pc(get['ft_star'][rid])} | "
                 f"{100 * (get['ft_star'][rid] - get['zs_star'][rid]):+.2f} | {pc(get['zs_q2'][rid])} | {pc(get['ft_q2'][rid])} | "
                 f"{pc(get['p2_zs_sel'][rid])} | {pc(get['p2_ft_sel'][rid])} | "
                 f"{sc['zs_star']['per_recording'][rid]['error']:+d} | {sc['ft_star']['per_recording'][rid]['error']:+d} |")
    L += ["", "Speaker-count sum |error|: " + ", ".join(f"{s} {v['sum_abs_error']}" for s, v in sc.items()), "",
          "## Q1 (primary): does fine-tuning help under equal decoding? (FT* vs ZS*, collar 0)", "",
          "| condition | value | pass |", "|---|---|---|"]
    for k, v in res["q1"]["conditions"].items():
        val = {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "pass"}
        L.append(f"| {k} | {val} | {'PASS' if v['pass'] else 'FAIL'} |")
    L += ["", f"**Q1 verdict: {res['q1']['verdict']}.**", "",
          "## Paired comparisons (bootstrap, delta = B - A, pp)", "",
          "| comparison | collar | A | B | delta | 95% CI | B wins/n |", "|---|---|---|---|---|---|---|"]
    rows = [("Q1 ZS* -> FT*", res["q1"]["by_collar"])]
    rows += [(f"Q2 {k.upper()} q2 -> star", v["by_collar"]) for k, v in res["q2"]["by_system"].items()]
    rows += [(f"p2 {k}_sel -> p3b {k}_star (ref.)", v) for k, v in res["phase2_reference"]["p3b_vs_p2"].items()]
    for name, byc in rows:
        for c, r in byc.items():
            L.append(f"| {name} | {c} | {pc(r['der_a'])} | {pc(r['der_b'])} | {100 * r['delta_b_minus_a']:+.2f} | "
                     f"[{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}] | {r['b_wins']}/{r['n_recordings']} |")
    L += ["", f"Q2 is reported for both systems; the better Q1 system (lower collar-0 corpus DER) is "
              f"**{res['q2']['better_q1_system'].upper()}**.", "",
          "## Oracle linking from each selected system's own tracks (corpus DER %, collar 0)", "",
          "| system | DER | oracle link | gap pp |", "|---|---|---|---|"]
    for k, d in res["oracle_link"].items():
        x = d["0.0"]
        L.append(f"| {k}_star | {pc(x['der'])} | {pc(x['oracle_link_der'])} | {x['gap_pp']:.2f} |")
    ds = res["default_system"]
    fc = ds["final_configuration"]
    L += ["", "## Default system (rule fixed before results)", "", ds["rule"], "",
          f"**Default = {ds['system']}; final configuration = {fc['config']}** (joint mode {fc['count']}/{fc['of']}"
          f"{', tied ' + str(fc['tied']) + ' -> simpler' if len(fc['tied']) > 1 else ''}). Next: {ds['next_step']}.", "",
          "ZS outer RTTMs are byte-identical to inner hypotheses of the other folds' searches (same cache and config), "
          "as in phase 2, by design: no fold's own labels entered its own selection."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------
# plan / status / CLI
# ----------------------------------------------------------------------------------------------
def cmd_plan(ctx: P3bCtx) -> dict:
    ctx.dry_run = True
    led = P3bLedger(ctx)
    ctx.out(f"DRY-RUN plan: PREREG_phase3b, {len(CONFIGS)} configurations per system; ledger starts at "
            f"{ctx.start_h:.2f} GPU-h (phase-2 cumulative), used now {led.used_h():.3f} h; abort before launch at "
            f"{ctx.budget_h} h cumulative; WARNING above {ctx.start_h + ctx.allowance_h:.2f} h (phase-3b allowance "
            f"{ctx.allowance_h} h)")
    r = cmd_regen(ctx)
    p = cmd_prep(ctx)
    cmd_search(ctx)
    cmd_outer(ctx)
    ctx.out("DRY-RUN freeze: sha256 of 6 phase-3b systems x 10 outer RTTMs + 4 phase-2 registered reference systems")
    ctx.out(f"DRY-RUN score: Q1 FT* vs ZS* (collar 0, bootstrap {ctx.bootstrap_n} seed {S.BOOTSTRAP_SEED}), Q2, collars "
            f"0/0.25/0.5, MISS/FA/CONF, speaker counts, oracle link, phase-2 references, default-system rule")
    total = r["projected_gpu_h"] + p["projected_gpu_h"]
    proj = led.used_h() + total
    s = {"projected_gpu_h_phase3b": round(total, 3), "projected_cumulative_h": round(proj, 3),
         "allowance_line_h": round(ctx.start_h + ctx.allowance_h, 2), "budget_h": ctx.budget_h,
         "exceeds_allowance": proj > ctx.start_h + ctx.allowance_h, "exceeds_budget": proj > ctx.budget_h}
    ctx.out("DRY-RUN plan total " + json.dumps(s))
    if s["exceeds_allowance"]:
        ctx.out(f"DRY-RUN WARNING: projected phase-3b GPU use {total:.3f} h exceeds the {ctx.allowance_h} h allowance "
                f"(the driver warns, and aborts only at {ctx.budget_h} h cumulative)")
    return s


def cmd_status(ctx: P3bCtx) -> dict:
    led = P3bLedger(ctx)
    d = ctx.cv_dir / "done"
    st = {"regen_markers": len(list((d / "regen").glob("*.json"))), "prep_markers": len(list((d / "prep").glob("*.json"))),
          "link_task_markers": len(list((d / "plink").glob("*.json"))), "selection": ctx.selection_path.exists(),
          "outer_index": (ctx.outer_root / "index.json").exists(), "frozen": ctx.frozen_path.exists(),
          "results": (ctx.cv_dir / "results.json").exists(), "times_scored": len(cv.scored_entries(ctx)),
          "gpu_h_used": round(led.used_h(), 3), "phase3b_h": round(led.extra_h(), 3), "gpu_h_budget": ctx.budget_h,
          "running_pids": sorted(p.name for p in ctx.pids.glob("*.pid")) if ctx.pids.is_dir() else []}
    ctx.out(json.dumps(st, indent=1))
    return st


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["train-fit"]:
        return train_fit_main(argv[1:])
    if argv[:1] == ["prep-job"]:
        return prep_job_main(argv[1:])
    ap = argparse.ArgumentParser(prog="bnd.p3b", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("plan", "regen", "prep", "search", "outer", "freeze", "score", "ledger", "status"))
    ap.add_argument("--dry-run", action="store_true", help="print planned work + projected GPU-h; write nothing")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--jobs", type=int, default=4, help="parallel CPU workers (<= 4 on the shared box)")
    ap.add_argument("--bootstrap-n", type=int, default=S.BOOTSTRAP_N)
    ap.add_argument("--deviation", default=None)
    a = ap.parse_args(argv)
    ctx = P3bCtx(dry_run=a.dry_run, allow_stale=a.allow_stale, jobs=max(1, min(a.jobs, 4)),
                 bootstrap_n=a.bootstrap_n, deviation=a.deviation)
    dry = a.dry_run or a.cmd in ("plan", "ledger", "status")
    if not dry:
        pidf = write_pid(ctx, f"driver_{a.cmd}_{os.getpid()}", os.getpid(), shlex.join(["bnd.p3b", *argv]))
        atexit.register(lambda: pidf.exists() and pidf.unlink())
    if a.cmd == "plan":
        cmd_plan(ctx)
    elif a.cmd == "regen":
        cmd_regen(ctx)
    elif a.cmd == "prep":
        cmd_prep(ctx)
    elif a.cmd == "search":
        cmd_search(ctx)
    elif a.cmd == "outer":
        cmd_outer(ctx)
    elif a.cmd == "freeze":
        cmd_freeze(ctx)
    elif a.cmd == "score":
        if ctx.dry_run:
            verify_frozen_p3b(ctx)
            ctx.out("DRY-RUN score: frozen hashes verified")
        else:
            cmd_score(ctx)
    elif a.cmd == "ledger":
        led = P3bLedger(ctx)
        for e in led.entries():
            ctx.out(f"{e['ts_start']} {e['kind']:6s} {e['tag']:44s} card={e['card']} rc={e['returncode']} "
                    f"gpu_wall={e['gpu_wall_s']:.0f}s")
        ctx.out(f"used {led.used_h():.3f} h (start {ctx.start_h:.2f} h phase 2; phase-3b {led.extra_h():.3f} of "
                f"{ctx.allowance_h} h allowance) of {ctx.budget_h} h")
    elif a.cmd == "status":
        cmd_status(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
