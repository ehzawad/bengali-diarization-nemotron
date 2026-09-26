"""CPU tests for bnd/p3b.py (PREREG_phase3b driver) on synthetic runs/{cv,p3b} worlds: the 108-config space and
simplicity order, the 1-SE rule, pooling that never touches outer Fi, session bookkeeping (80 s cores tile
exactly; 20 s = phase 2), the regen reproduction check (abort on mismatch), the outer lock / freeze / score,
Q1 claim conditions, the ledger warning/abort, and the dry run (writes nothing). Never writes into the real runs/."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import soundfile as sf

from bnd import cv, infer
from bnd import link as L
from bnd import p3b
from bnd import score as S
from bnd.labels import RECORDINGS

_PREPARED_MANIFEST = Path(__file__).resolve().parents[1] / "data/prepared/manifests/train_manifest_balanced.json"
needs_prepared = pytest.mark.skipif(not _PREPARED_MANIFEST.exists(),
                                    reason="needs data/prepared (run bnd.labels, bnd.trust, bnd.nemo_data first)")


REPO = Path(__file__).resolve().parents[1]
DUR = 40.0
C = p3b.Config


def hyp(miss: float) -> list[S.Seg]:
    return [S.Seg(0.0, 10.0, "x"), S.Seg(10.0, 20.0, "y"), S.Seg(20.0, 30.0 - miss, "x")]


@pytest.fixture
def world(tmp_path):
    runs = tmp_path / "runs"
    audio, ref = tmp_path / "audio", tmp_path / "ref"
    audio.mkdir()
    ref.mkdir()
    for rid in RECORDINGS:
        sf.write(audio / f"{rid}.wav", np.zeros(int(DUR * 1000), np.float32), 1000)
        S.write_rttm(ref / f"{rid}.rttm", rid, [S.Seg(0.0, 10.0, "A"), S.Seg(10.0, 20.0, "B"), S.Seg(20.0, 30.0, "A")])
    p2 = runs / "cv"
    sel2 = {"folds": {fi: {"selected_recipe": r} for fi, r in p3b.REGISTERED_FT_RECIPE.items()}}
    cv.write_json(p2 / "selection.json", sel2)
    systems = {}
    for name, miss in (("ft_sel", 3.0), ("zs_sel", 6.0), ("ft_strict", 4.0), ("zs_strict", 6.0)):
        systems[name] = {}
        for fi, rids in cv.FOLDS.items():
            for rid in rids:
                p = p2 / "outer" / name / fi / f"{rid}.rttm"
                S.write_rttm(p, rid, hyp(miss))
                systems[name][rid] = {"path": str(p.relative_to(p2)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                      "fold": fi, "source_system": "x", "cell": "0.40_0.55", "oracle": "none"}
    cv.write_json(p2 / "frozen.json", {"created": "t", "selection_sha256": hashlib.sha256(
        (p2 / "selection.json").read_bytes()).hexdigest(), "systems": systems})
    lines = []
    ctx = p3b.P3bCtx(cv_dir=runs / "p3b", p2_dir=p2, p2_models=tmp_path / "models", ref_dir=ref, audio_dir=audio,
                     bootstrap_n=500, jobs=1, out=lines.append)
    ctx._lines = lines
    return ctx


def tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}


# ------------------------------------------------------------------ space + simplicity order
def test_space_is_the_registered_108_and_keys_roundtrip():
    assert len(p3b.CONFIGS) == 108
    assert {c.session for c in p3b.CONFIGS} == {20, 80} and {c.linker for c in p3b.CONFIGS} == {"avg", "refine"}
    assert {c.gap for c in p3b.CONFIGS} == {0, 2, 5}
    assert {round(c.theta_act, 2) for c in p3b.CONFIGS} == {0.10, 0.25, 0.40}
    assert {round(c.theta_clu, 2) for c in p3b.CONFIGS} == {0.55, 0.60, 0.65}
    assert all(p3b.parse_key(c.key) == c for c in p3b.CONFIGS)
    assert all(c.session == 20 and c.linker == "avg" for c in p3b.Q2_CONFIGS) and len(p3b.Q2_CONFIGS) == 27


def test_simplicity_order_is_lexicographic_S_L_g_act_clu():
    order = sorted(p3b.CONFIGS, key=lambda c: c.simplicity)
    assert order[0] == C(20, "avg", 0, 0.40, 0.60) and order[-1] == C(80, "refine", 5, 0.10, 0.65)
    assert C(20, "refine", 5, 0.10, 0.65).simplicity < C(80, "avg", 0, 0.40, 0.60).simplicity      # S first
    assert C(20, "avg", 5, 0.10, 0.65).simplicity < C(20, "refine", 0, 0.40, 0.60).simplicity      # then L
    assert C(20, "avg", 2, 0.10, 0.65).simplicity < C(20, "avg", 5, 0.40, 0.60).simplicity         # then g
    acts = sorted([0.10, 0.25, 0.40], key=lambda a: C(20, "avg", 0, a, 0.60).simplicity)
    assert acts == [0.40, 0.25, 0.10]
    clus = sorted([0.55, 0.60, 0.65], key=lambda c: C(20, "avg", 0, 0.40, c).simplicity)
    assert clus == [0.60, 0.55, 0.65]


# ------------------------------------------------------------------ 1-SE rule
RIDS8 = [f"r{i}" for i in range(8)]


def table_from(ders: dict) -> dict:
    return {c: {r: {"errors": d * 100.0, "total": 100.0} for r, d in zip(RIDS8, v)} for c, v in ders.items()}


def test_one_se_rule_picks_simplest_within_one_se():
    best = C(80, "refine", 5, 0.10, 0.65)          # argmin, most complex
    near = C(20, "refine", 2, 0.25, 0.55)          # within 1 SE
    simple_far = C(20, "avg", 0, 0.40, 0.60)       # simplest but worse than argmin + its SE
    base = np.array([0.30, 0.32, 0.28, 0.35, 0.31, 0.29, 0.33, 0.30])
    tab = table_from({best: base, near: base + np.array([0.02, -0.01, 0.03, 0.0, -0.01, 0.02, -0.02, 0.0]),
                      simple_far: base + 0.05})
    r = p3b.one_se_select(tab)
    d = base + np.array([0.02, -0.01, 0.03, 0.0, -0.01, 0.02, -0.02, 0.0]) - base
    se = np.std(d, ddof=1) / math.sqrt(8)
    assert r["argmin"] == best and r["se"][near] == pytest.approx(se)
    assert r["der"][near] - r["der"][best] == pytest.approx(d.mean()) and d.mean() <= se
    assert r["se"][simple_far] == pytest.approx(0.0, abs=1e-12)   # constant shift: SE 0 -> not eligible
    assert r["selected"] == near and simple_far not in r["eligible"]
    # a constant offset of 0 (identical per recording) ties with the argmin -> the simpler one wins
    tab[simple_far] = tab[best]
    r = p3b.one_se_select(tab)
    assert r["argmin"] == simple_far and r["selected"] == simple_far


def test_one_se_rule_uses_pooled_der_and_loro_stability():
    a, b = C(20, "avg", 0, 0.40, 0.60), C(80, "avg", 0, 0.40, 0.60)
    tab = {a: {r: {"errors": 30.0, "total": 100.0} for r in RIDS8}, b: {r: {"errors": 29.0, "total": 100.0} for r in RIDS8}}
    tab[b]["r0"] = {"errors": 60.0, "total": 400.0}  # pooled DER weights recordings by reference speech
    r = p3b.one_se_select(tab)
    assert r["der"][b] == pytest.approx((29 * 7 + 60) / (700 + 400))
    st = p3b.loro_stability(tab)
    assert st["n"] == 8 and set(st["dropped"]) == set(RIDS8)
    assert st["selected_recovered"] == sum(v["selected"] == r["selected"].key for v in st["dropped"].values())


def test_joint_mode_ties_go_to_the_simpler_config():
    k1, k2 = C(80, "avg", 0, 0.40, 0.60).key, C(20, "refine", 5, 0.10, 0.65).key
    jm = p3b.joint_mode([k1, k2, k1, k2, C(20, "avg", 0, 0.40, 0.60).key])
    assert jm["config"] == k2 and jm["count"] == 2 and jm["tied"] == sorted([k1, k2])


# ------------------------------------------------------------------ pooling never touches outer Fi
def test_inner_units_exclude_outer_fold_and_regen_covers_14_fits():
    for fi in cv.FOLDS:
        for kind in ("zs", "ft"):
            units = p3b.units_for(kind, fi)
            assert sorted(r for _, r in units) == sorted(r for f in cv.FOLDS if f != fi for r in cv.FOLDS[f])
            assert not {r for _, r in units} & set(cv.FOLDS[fi])
        for system, rid in p3b.ft_inner_units(fi):
            fit = next(f for f in cv.inner_fits() if f["tag"] == system)
            assert fi in fit["excluded_folds"] and rid in fit["heldout_recs"]
            assert fit["recipe"] == p3b.REGISTERED_FT_RECIPE[fi]
    assert len(p3b.regen_specs()) == 14
    assert sum(len(g["rids"]) for g in p3b.prep_groups()) == 60


def test_pool_items_refuses_outer_recordings(world):
    ctx = world
    c = p3b.CONFIGS[0]
    ok = p3b.pool_items(ctx, p3b.zs_inner_units("F1"), "F1", c)
    assert all(r not in cv.FOLDS["F1"] for r, _ in ok)
    with pytest.raises(p3b.P3bError, match="outer recording"):
        p3b.pool_items(ctx, p3b.zs_inner_units("F1") + [("zeroshot", "train_001")], "F1", c)
    with pytest.raises(p3b.P3bError, match="pooled set"):
        p3b.pool_items(ctx, p3b.zs_inner_units("F1")[:-1], "F1", c)


# ------------------------------------------------------------------ session bookkeeping
@pytest.mark.parametrize("n_s", [1.0, 79.99, 80.0, 80.01, 245.3, 3601.7])
def test_80s_cores_tile_exactly(n_s):
    n = int(round(n_s * infer.SR))
    ws = infer.make_windows(n, core_s=80.0)
    assert ws[0]["core_start_s"] == 0.0 and ws[-1]["core_end_s"] == n / infer.SR
    assert all(a["core_end_s"] == b["core_start_s"] for a, b in zip(ws, ws[1:]))
    assert all(w["end_s"] - w["start_s"] <= 90.0 + 1e-9 for w in ws)
    assert all(w["core_end_s"] - w["core_start_s"] <= 80.0 + 1e-9 for w in ws)
    assert len(ws) == math.ceil(n / (80 * infer.SR))
    for w in ws[1:]:
        if w["core_end_s"] + 5.0 <= n / infer.SR:  # interior: full 5 s context on both sides
            assert w["end_s"] - w["start_s"] == pytest.approx(90.0)
    assert infer.session_tag(80.0) == "s80" and infer.session_tag(20) == "s20"
    with pytest.raises(infer.InferError):
        infer.check_windows(ws, n / infer.SR) if len(ws) > 1 else infer.check_windows(
            [dict(ws[0], end_s=ws[0]["start_s"] + 31)], ws[0]["start_s"] + 31)


def test_default_windows_equal_phase2_and_20s_explicit():
    n = int(3601.7 * infer.SR)
    assert infer.make_windows(n) == infer.make_windows(n, core_s=20.0)
    d = REPO / "runs/cv/cache/zeroshot/train_001"
    if not (d / "windows.json").exists():
        pytest.skip("phase-2 cache not present")
    meta = json.loads((d / "meta.json").read_text())
    n = sf.info(meta["audio_filepath"]).frames
    assert infer.public_windows(infer.make_windows(n)) == json.loads((d / "windows.json").read_text())
    assert meta["core_s"] == 20.0


def test_link_configs_equal_run_recording(tmp_path):
    from test_link import make_cache, unit_dirs

    turns = [(0, 12, "s0"), (12, 13.5, "s1"), (14.0, 30, "s0"), (31, 55, "s2"), (55.5, 70, "s1")]
    d = make_cache(tmp_path, "r", turns, 70.0, unit_dirs(3), noise=0.2, seed=4, theta_act=0.25)
    cache = L.load_cache(d, 0.25)
    cfgs = p3b.configs_for(20, 0.25)
    assert len(cfgs) == 18
    res = p3b.link_configs(cache, 0.25, cfgs)
    for c in cfgs:
        segs, _ = L.run_recording(cache, 0.25, c.theta_clu, linker=c.linker, gap_fill_s=float(c.gap))
        assert res[c.key][0] == segs, c.key
    ref = [S.Seg(s, e, n) for s, e, n in turns]
    o = p3b.link_configs(cache, 0.25, [cfgs[-1]], "link", ref)
    segs, _ = L.run_recording(cache, 0.25, 0.6, oracle="link", ref=ref, gap_fill_s=float(cfgs[-1].gap))
    assert o[cfgs[-1].key][0] == segs


# ------------------------------------------------------------------ regen reproduction check
TAG = "cv_inner_x12_lr5e-5_raw"


def make_probs_cache(d: Path, system: str, model_sha: str, rng, n_windows=3, core=20.0):
    d.mkdir(parents=True, exist_ok=True)
    ws = [{"window": i, "start_s": max(0.0, i * core - 5), "end_s": i * core + core + 5, "core_start_s": i * core,
           "core_end_s": i * core + core} for i in range(n_windows)]
    for w in ws:
        np.save(d / f"probs_{w['window']}.npy", rng.random((int((w["end_s"] - w["start_s"]) / 0.01), 8)).astype(np.float16))
    (d / "windows.json").write_text(json.dumps(ws))
    (d / "meta.json").write_text(json.dumps({"system": system, "model_sha256": model_sha, "nemo_commit": "c",
                                             "dt": 0.01, "core_s": core, "recording_duration_s": n_windows * core}))


def regen_world(ctx, tmp_path, delta: float, bad_rec: str = "train_009"):
    spec = next(s for s in p3b.regen_specs() if s["tag"] == TAG)
    rng = np.random.default_rng(0)
    for rid in spec["heldout_recs"]:
        make_probs_cache(ctx.p2_cache / TAG / rid, TAG, "p2sha", rng)
    ins = cv.fit_inputs(ctx, spec)
    cv.write_marker(ctx.p2, "fit", TAG, ins, {"model_sha256": "p2sha", "delta_s": None})

    def fake_run(cmd, kind, tag, log, est_s, audio_s=0.0):
        if kind == "train":
            (ctx.p3b_models).mkdir(parents=True, exist_ok=True)
            (ctx.p3b_models / f"{TAG}.nemo").write_bytes(b"regenerated model")
            cv.write_json(ctx.p3b_fits / TAG / "fit_report.json", {"status": "ok", "train_recs": spec["train_recs"]})
        else:
            assert "--core-s" in [str(c) for c in cmd] and "20" in [str(c) for c in cmd]
            sha = hashlib.sha256(b"regenerated model").hexdigest()
            for rid in spec["heldout_recs"]:
                src, dst = ctx.p2_cache / TAG / rid, p3b.cache_dir(ctx, TAG, 20, rid)
                dst.mkdir(parents=True, exist_ok=True)
                (dst / "windows.json").write_bytes((src / "windows.json").read_bytes())
                for p in src.glob("probs_*.npy"):
                    a = np.load(p).astype(np.float32)
                    if rid == bad_rec and p.name == "probs_1.npy":
                        a[5, 3] = min(1.0, a[5, 3] + delta) if a[5, 3] < 0.5 else a[5, 3] - delta
                    np.save(dst / p.name, a.astype(np.float16))
                m = json.loads((src / "meta.json").read_text())
                m.update(model_sha256=sha, core_s=20.0)
                (dst / "meta.json").write_text(json.dumps(m))
        ran.append(kind)
        return {}

    ran = []
    led = p3b.P3bLedger(ctx)
    led.run_est = fake_run
    return spec, led, ran


@needs_prepared
def test_regen_reproduction_check_passes_within_tolerance(world, tmp_path):
    ctx = world
    spec, led, ran = regen_world(ctx, tmp_path, delta=5e-4)
    r = p3b.regen_one(ctx, led, p3b.Rates(), spec)
    assert r["done"] and ran == ["train", "infer"]
    m = cv.read_marker(ctx, "regen", TAG)
    assert m["outputs"]["probs_check"]["train_009"]["max_abs_diff"] <= p3b.REGEN_TOL
    assert m["outputs"]["model_bytes_identical_to_phase2"] is False
    assert p3b.regen_done(ctx, TAG)
    # resumable: no GPU work the second time
    ran.clear()
    assert p3b.regen_one(ctx, led, p3b.Rates(), spec)["done"] and ran == []


@needs_prepared
def test_regen_reproduction_check_aborts_on_mismatch(world, tmp_path):
    ctx = world
    spec, led, ran = regen_world(ctx, tmp_path, delta=3e-3)
    before = tree_digest(ctx.p2_dir)
    with pytest.raises(p3b.RegenError, match="STOP"):
        p3b.regen_one(ctx, led, p3b.Rates(), spec)
    assert cv.read_marker(ctx, "regen", TAG) is None
    fail = json.loads((ctx.cv_dir / "regen_failure.json").read_text())
    assert not fail["per_recording"]["train_009"]["ok"] and fail["per_recording"]["train_001"]["ok"]
    with pytest.raises(p3b.P3bError, match="not verified"):
        p3b.regen_done(ctx, TAG)
    assert tree_digest(ctx.p2_dir) == before


@needs_prepared
def test_regen_refuses_when_fit_inputs_differ_from_phase2(world, tmp_path):
    ctx = world
    spec, led, _ = regen_world(ctx, tmp_path, delta=0.0)
    m = cv.read_marker(ctx.p2, "fit", TAG)
    m["inputs"]["lr"] = "1e-5"
    cv.write_marker(ctx.p2, "fit", TAG, m["inputs"], m["outputs"])
    with pytest.raises(p3b.P3bError, match="not an identical refit"):
        p3b.regen_one(ctx, led, p3b.Rates(), spec)


def test_regen_window_mismatch_fails(tmp_path):
    rng = np.random.default_rng(1)
    make_probs_cache(tmp_path / "a", "x", "s", rng)
    make_probs_cache(tmp_path / "b", "x", "s", np.random.default_rng(1), n_windows=2)
    assert not p3b.compare_probs(tmp_path / "b", tmp_path / "a")["ok"]
    assert p3b.compare_probs(tmp_path / "a", tmp_path / "a") == {"ok": True, "max_abs_diff": 0.0, "windows": 3}


# ------------------------------------------------------------------ S=20 phase-2 mirror
def test_mirror_is_readonly_symlinks_and_refuses_symlinked_outputs(world):
    ctx = world
    make_probs_cache(ctx.p2_cache / "zeroshot" / "train_001", "zeroshot", "z", np.random.default_rng(2))
    before = tree_digest(ctx.p2_dir)
    d = p3b.mirror_phase2_s20(ctx, "zeroshot", "train_001")
    assert d == ctx.cv_dir / "cache" / "zeroshot" / "s20" / "train_001"
    assert (d / "probs_0.npy").is_symlink() and (d / "meta.json").is_symlink()
    p3b.mirror_phase2_s20(ctx, "zeroshot", "train_001")  # idempotent
    infer.write_tracks(d, 0.10)
    assert (d / "tracks_0.10.json").is_file() and not (d / "tracks_0.10.json").is_symlink()
    assert tree_digest(ctx.p2_dir) == before
    (d / "tracks_0.25.json").symlink_to(d / "tracks_0.10.json")
    with pytest.raises(p3b.P3bError, match="real files"):
        p3b.mirror_phase2_s20(ctx, "zeroshot", "train_001")


# ------------------------------------------------------------------ outer lock, freeze, score, Q1
def synth_selection(ctx) -> dict:
    zs, ft = C(20, "avg", 0, 0.40, 0.60), C(80, "refine", 2, 0.25, 0.55)
    sel = {"created": "t", "folds": {}}
    for fi in cv.FOLDS:
        f = {"ft_recipe": p3b.REGISTERED_FT_RECIPE[fi]}
        for k, c in (("zs", zs), ("ft", ft)):
            f[k] = {"selected": c.key, "selected_der": 0.3, "argmin": c.key, "argmin_der": 0.3,
                    "q2_selected": C(20, "avg", 2, 0.40, 0.60).key, "loro_stability": {"selected_recovered": 6}}
        sel["folds"][fi] = f
    cv.write_json(ctx.selection_path, sel)
    return sel


def synth_outer(ctx, misses: dict[str, dict] | None = None):
    """Outer RTTMs + link markers + index for all 6 systems (default miss 4 s; ZS 6 s)."""
    sel = json.loads(ctx.selection_path.read_text())
    jobs = p3b.outer_jobs(ctx, sel)
    for j in jobs:
        shas = {}
        for rid in j["recs"]:
            miss = (misses or {}).get(j["name"], {}).get(rid, 6.0 if j["name"].startswith("zs") else 4.0)
            p = j["out_dir"] / f"{rid}.rttm"
            S.write_rttm(p, rid, hyp(miss))
            shas[rid] = hashlib.sha256(p.read_bytes()).hexdigest()
        cv.write_marker(ctx, "link", cv.dir_marker_name(ctx, j["out_dir"]), {"x": 1},
                        {"out_dir": str(j["out_dir"]), "rttm_sha256": shas})
    cv.write_json(ctx.outer_root / "index.json", {
        "created": "t", "selection_sha256": cv.file_sha(ctx, ctx.selection_path),
        "jobs": [dict({k: v for k, v in j.items() if k not in ("cfg", "out_dir")}, config=j["cfg"].key,
                      out_dir=p3b.rel(ctx, j["out_dir"])) for j in jobs]})
    return jobs


def test_outer_lock_before_freeze(world):
    ctx = world
    synth_selection(ctx)
    synth_outer(ctx)
    c = C(20, "avg", 0, 0.40, 0.60)
    for rid in ("train_001", "train_011"):
        S.write_rttm(p3b.hyp_path(ctx, "zeroshot", c, rid), rid, hyp(6.0))
    with pytest.raises(cv.OuterLockError, match="does not exist"):  # context lock (F1 recording in context F1)
        p3b.score_items(ctx, "F1", [("train_001", p3b.hyp_path(ctx, "zeroshot", c, "train_001"))], [0.0])
    with pytest.raises(cv.OuterLockError):  # file lock: outer artefacts of either phase, any context
        p3b.score_items(ctx, "F1", [("train_011", ctx.outer_root / "ft_star" / "F2" / "train_011.rttm")], [0.0])
    with pytest.raises(cv.OuterLockError):
        p3b.score_items(ctx, "F3", [("train_001", ctx.p2_dir / "outer" / "ft_sel" / "F1" / "train_001.rttm")], [0.0])
    with pytest.raises(cv.OuterLockError):
        p3b.cmd_score(ctx)
    r = p3b.score_items(ctx, "F1", [("train_011", p3b.hyp_path(ctx, "zeroshot", c, "train_011"))], [0.0])
    assert r[0][0.0].der == pytest.approx(6.0 / 30.0)


def test_freeze_refuses_incomplete(world):
    ctx = world
    synth_selection(ctx)
    synth_outer(ctx)
    (ctx.outer_root / "ft_q2" / "F3" / "train_005.rttm").unlink()
    with pytest.raises(p3b.P3bError, match="freeze refused"):
        p3b.cmd_freeze(ctx)
    assert not ctx.frozen_path.exists()


def test_freeze_score_q1_and_tamper(world):
    ctx = world
    synth_selection(ctx)
    ft = {rid: 2.0 for rid in RECORDINGS}   # FT* misses 2 s, ZS* 6 s -> -13.3 pp, 10/10 wins
    synth_outer(ctx, {"ft_star": ft})
    before = tree_digest(ctx.p2_dir)
    fz = p3b.cmd_freeze(ctx)
    assert set(fz["systems"]) == set(p3b.OUTER_SYSTEMS) | set(p3b.P2_REFERENCE)
    assert fz["systems"]["p2_ft_sel"]["train_001"]["path"] == "cv/outer/ft_sel/F1/train_001.rttm"
    res = p3b.cmd_score(ctx)
    q = res["q1"]
    assert q["finetuning_helps"] and q["default_system"] == "FT"
    assert q["conditions"]["corpus_delta_le_-1pp"]["value_pp"] == pytest.approx(-100 * 4 / 30)
    assert res["q2"]["better_q1_system"] == "ft"
    assert res["default_system"]["final_configuration"]["config"] == C(80, "refine", 2, 0.25, 0.55).key
    assert res["speaker_count_errors"]["reference"]["train_001"] == 2
    assert set(res["oracle_link"]) == {"zs", "ft"}
    md = (ctx.cv_dir / "results.md").read_text()
    assert md.startswith("# Phase-3b results (outer folds): SECOND-LOOK") and "fine-tuning helps" in md
    assert len(cv.scored_entries(ctx)) == 1
    assert tree_digest(ctx.p2_dir) == before
    p = ctx.outer_root / "zs_star" / "F4" / "train_002.rttm"
    orig = p.read_bytes()
    S.write_rttm(p, "train_002", hyp(0.0))
    with pytest.raises(cv.OuterLockError, match="no longer match"):
        p3b.cmd_score(ctx)
    p.write_bytes(orig)
    q2 = ctx.p2_dir / "outer" / "zs_sel" / "F2" / "train_009.rttm"
    q2.write_text(q2.read_text() + "\n")
    with pytest.raises(cv.OuterLockError):
        p3b.cmd_score(ctx)


def recs(errors, totals=None, collar=0.0):
    totals = totals or [30.0] * len(errors)
    return [S.RecScore(f"r{i:02d}", collar, 40.0, t, e, 0.0, 0.0) for i, (e, t) in enumerate(zip(errors, totals))]


def test_q1_claim_conditions():
    zs = recs([9.0] * 10)
    boot = {"ci_high": -0.01, "ci_low": -0.05}
    v = p3b.q1_verdict(zs, recs([8.7] * 10), boot)  # exactly -1.0 pp
    assert v["finetuning_helps"] and v["default_system"] == "FT"
    v = p3b.q1_verdict(zs, recs([8.71] * 10), boot)
    assert not v["finetuning_helps"] and v["verdict"].startswith("no demonstrated benefit")
    v = p3b.q1_verdict(zs, recs([6.0] * 5 + [9.0] * 5), boot)  # 5/10 improved
    assert not v["conditions"]["improved_ge_6_of_10"]["pass"] and v["default_system"] == "ZS"
    v = p3b.q1_verdict(zs, recs([5.0] * 10), {"ci_high": 0.01, "ci_low": -0.2})
    assert not v["conditions"]["bootstrap_p97_5_lt_0"]["pass"]
    # macro mean can disagree with the corpus delta when totals differ
    v = p3b.q1_verdict(recs([10.0, 1.0] * 5, [100.0, 3.0] * 5), recs([5.0, 2.0] * 5, [100.0, 3.0] * 5), boot)
    assert v["conditions"]["corpus_delta_le_-1pp"]["pass"] and not v["conditions"]["macro_mean_delta_lt_0"]["pass"]
    v = p3b.q1_verdict(zs, recs([12.0] * 10), {"ci_high": 0.2, "ci_low": 0.05})
    assert v["harm"] and v["verdict"].startswith("harm")


# ------------------------------------------------------------------ ledger
FAKE_GPU_RUN = """#!/usr/bin/env bash
echo "[gpu_run] card=1 pid=$$ cmd=$*" >&2
exec "$@"
"""


def test_ledger_starts_at_4p11_warns_at_6p11_aborts_at_12(world, tmp_path):
    ctx = world
    g = tmp_path / "gpu_run.sh"
    g.write_text(FAKE_GPU_RUN)
    g.chmod(0o755)
    ctx.gpu_run = g
    led = p3b.P3bLedger(ctx)
    assert led.used_h() == pytest.approx(4.11)
    led.check(3600 * 1.5, "x")
    assert not any("WARNING" in l for l in ctx._lines)
    led.check(3600 * 2.5, "x")
    assert any("WARNING" in l and "6.11" in l for l in ctx._lines)
    with pytest.raises(cv.BudgetError, match="aborting before launch"):
        led.check(3600 * 8.0, "x")
    e = led.run_est([sys.executable, "-c", "print('hi')"], "infer", "t:1", ctx.logs / "t.log", 10.0)
    assert e["returncode"] == 0 and e["card"] == "1" and e["phase"] == "3b"
    assert len(led.entries()) == 1 and not list(ctx.pids.glob("*.pid"))  # pid file removed after exit
    with pytest.raises(p3b.P3bError, match="failed"):
        led.run_est([sys.executable, "-c", "raise SystemExit(3)"], "infer", "t:2", ctx.logs / "t.log", 10.0)
    assert not list(ctx.pids.glob("*.pid"))


# ------------------------------------------------------------------ dry run (real phase-2 inputs, read only)
@pytest.mark.skipif(not (REPO / "runs/cv/frozen.json").exists(), reason="phase-2 artefacts not present")
def test_dry_run_plan_lists_work_and_gpu_h_and_writes_nothing(tmp_path):
    lines = []
    ctx = p3b.P3bCtx(cv_dir=tmp_path / "p3b", out=lines.append)
    p2_before = {p: p.stat().st_mtime_ns for p in (REPO / "runs/cv").glob("*.json")}
    s = p3b.cmd_plan(ctx)
    out = "\n".join(lines)
    assert not (tmp_path / "p3b").exists()
    assert out.count("DRY-RUN regen cv_inner_") == 14 and "bnd.p3b train-fit" in out and "--core-s 20" in out
    assert "prep-job" in out and "gpu_run.sh" in out and "--s80" in out
    assert "5400 RTTMs" in out and "DRY-RUN plan total" in out
    assert s["projected_gpu_h_phase3b"] > 0 and s["allowance_line_h"] == 6.11
    assert {p: p.stat().st_mtime_ns for p in (REPO / "runs/cv").glob("*.json")} == p2_before


@pytest.mark.skipif(not (REPO / "runs/cv/done/link").is_dir(), reason="phase-2 artefacts not present")
@pytest.mark.parametrize("system,rid", [("cv_inner_x12_lr5e-5_raw", "train_011"), ("zeroshot", "train_004")])
def test_search_worker_s20_avg_g0_reproduces_phase2_inner_rttm(system, rid, tmp_path):
    """Regression (read-only): the p3b search worker at (S=20, avg, g=0, 0.40, 0.55) on a phase-2 20 s cache
    writes exactly the registered phase-2 inner hypothesis (bytes and link-marker hash)."""
    cfg = C(20, "avg", 0, 0.40, 0.55)
    m = json.loads((REPO / f"runs/cv/done/link/hyp__{system}__0.40_0.55.json").read_text())
    out = tmp_path / f"{rid}.rttm"
    shas = p3b._link_task((str(REPO / "runs/cv/cache" / system / rid), rid, 0.40, [cfg.key], [str(out)], "", "none", None))
    assert shas[cfg.key] == m["outputs"]["rttm_sha256"][rid]
    assert out.read_bytes() == (REPO / "runs/cv/hyp" / system / "0.40_0.55" / f"{rid}.rttm").read_bytes()
