"""CPU tests for bnd/p5.py (PREREG_phase5 driver) on synthetic worlds under tmp_path: the registered fits and
pools (never an embedder trained on the scored recording or the outer fold), the selection rule and its tie
order, LORO, joint mode, the dev adoption criterion, the floor adapter, the hold-out subset loader and
speaker-count bins, the hold-out claim rule, the Part-A outer lock (freeze -> score -> tamper), the hold-out
lock (refuse before freeze, after tampering and a second scoring), the phase-5 ledger (start 6.03 h, warn,
abort, no double counting of bnd.embed_ft's own entries) and dry runs that write nothing. Never writes into
the real runs/ or data/."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import soundfile as sf

from bnd import cv, p3b
from bnd import p5
from bnd import score as S
from bnd.labels import RECORDINGS

REPO = Path(__file__).resolve().parents[1]
DUR = 40.0
C = p5.Config


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()} \
        if root.exists() else {}


def hyp(miss: float, spk2: str = "y") -> list[S.Seg]:
    return [S.Seg(0.0, 10.0, "x"), S.Seg(10.0, 20.0, spk2), S.Seg(20.0, 30.0 - miss, "x")]


@pytest.fixture
def world(tmp_path):
    runs = tmp_path / "runs"
    audio, ref = tmp_path / "audio", tmp_path / "ref"
    audio.mkdir()
    ref.mkdir()
    for rid in RECORDINGS:
        sf.write(audio / f"{rid}.wav", np.zeros(int(DUR * 1000), np.float32), 1000)
        S.write_rttm(ref / f"{rid}.rttm", rid, [S.Seg(0.0, 10.0, "A"), S.Seg(10.0, 20.0, "B"), S.Seg(20.0, 30.0, "A")])
    cv.write_json(runs / "cv" / "frozen.json", {"systems": {}})
    lines = []
    ctx = p5.P5Ctx(cv_dir=runs / "p5", p2_dir=runs / "cv", p3b_dir=runs / "p3b", p2_models=tmp_path / "models",
                   ref_dir=ref, audio_dir=audio, holdout_dir=tmp_path / "holdout",
                   holdout_sha_file=tmp_path / "holdout_SHA256.txt", base_model=tmp_path / "base.nemo",
                   bootstrap_n=200, jobs=1, out=lines.append)
    ctx._lines = lines
    return ctx


# ------------------------------------------------------------------ registered space, fits, pools
def test_space_and_simplicity_order():
    for cell in p5.CELLS:
        cs = p5.cell_configs(cell)
        assert len(cs) == 20 and {c.session for c in cs} == {p5.CELLS[cell][0]} and {c.linker for c in cs} == {"avg"}
    order = sorted(p5.cell_configs("s20_stock"), key=p5.cfg_simplicity)
    assert order[0].key == "s20_avg_g0_a0.40_c0.60" and order[-1].key == "s20_avg_g5_a0.25_c0.75"
    assert [round(c.theta_clu, 2) for c in order[:5]] == [0.60, 0.55, 0.65, 0.45, 0.75]
    assert all(p3b.parse_key(c.key) == c for c in p5.CONFIGS)
    assert (p5.INCUMBENT, p5.CHALLENGER) == ("s20_stock", "s120_ft")


def test_sixteen_registered_fits():
    fits = p5.embedder_fits()
    kinds = [f["kind"] for f in fits]
    assert len(fits) == 16 and kinds.count("inner") == 10 and kinds.count("refit") == 5 and kinds.count("final") == 1
    assert len({f["tag"] for f in fits}) == 16
    for f in fits:
        assert not set(f["train_recs"]) & set(f["heldout_recs"])
        n = {"inner": 6, "refit": 8, "final": 10}[f["kind"]]
        assert len(f["train_recs"]) == n and sorted(f["train_recs"] + f["heldout_recs"]) == sorted(RECORDINGS)
    assert len({tuple(f["train_recs"]) for f in fits if f["kind"] == "inner"}) == 10


@pytest.mark.parametrize("fi", sorted(cv.FOLDS))
@pytest.mark.parametrize("cell", sorted(p5.CELLS))
def test_pool_excludes_outer_fold_and_scored_recording(cell, fi):
    units = p5.inner_units(cell, fi)
    inner = sorted(r for f in cv.FOLDS if f != fi for r in cv.FOLDS[f])
    assert sorted(r for _, r in units) == inner
    for src, rid in units:
        assert rid not in cv.FOLDS[fi]
        if p5.CELLS[cell][1] == "stock":
            assert src == "zeroshot"
            continue
        fit = p5.fit_by_tag(src[len("ftemb_"):])
        assert fit["kind"] == "inner"
        assert rid not in fit["train_recs"], "embedder trained on the scored recording"
        assert not set(cv.FOLDS[fi]) & set(fit["train_recs"]), "embedder trained on the outer fold"
    out = p5.outer_source(cell, fi)
    if p5.CELLS[cell][1] == "ft":
        f = p5.fit_by_tag(out[len("ftemb_"):])
        assert f["kind"] == "refit" and not set(cv.FOLDS[fi]) & set(f["train_recs"])
    else:
        assert out == "zeroshot"


def test_never_embed_own_training(tmp_path):
    with pytest.raises(p5.P5Error):
        p5.check_not_trained_on(p5.fit_by_tag("p5emb_all10"), ["train_001"])
    with pytest.raises(p5.P5Error):
        p5.check_not_trained_on(p5.fit_by_tag("p5emb_refit_F1"), ["train_002"])
    p5.check_not_trained_on(p5.fit_by_tag("p5emb_refit_F1"), list(cv.FOLDS["F1"]))
    # every inner fit is only ever asked to embed its own held-out recordings
    for f in p5.embedder_fits():
        p5.check_not_trained_on(f, f["heldout_recs"])
    # the second, manifest-based guard of bnd.embed_ft
    from bnd import embed_ft

    d = tmp_path / "emb"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"train_recordings": ["train_001", "train_002"]}))
    with pytest.raises(embed_ft.EmbedFTError):
        embed_ft.assert_not_trained_on(d, ["train_002"])
    embed_ft.assert_not_trained_on(d, ["train_003"])


def test_mirror_for_ft_refuses_symlinked_embeddings(world, tmp_path):
    src = p5.cdir(world, "zeroshot", 20, "train_001")
    src.mkdir(parents=True)
    (src / "windows.json").write_text(json.dumps([{"window": 0}]))
    for n in ("meta.json", "probs_0.npy", "tracks_0.25.json", "tracks_0.40.json"):
        (src / n).write_text("x")
    d = p5.mirror_for_ft(world, "ftemb_p5emb_x23", 20, "train_001")
    assert (d / "tracks_0.40.json").is_symlink() and not (d / "emb_0.40.npy").exists()
    (d / "emb_0.40.npy").symlink_to(src / "meta.json")
    with pytest.raises(p5.P5Error):
        p5.mirror_for_ft(world, "ftemb_p5emb_x23", 20, "train_001")


# ------------------------------------------------------------------ selection
def table_from(ders: dict[str, float], rids=("r1", "r2")) -> dict:
    return {p3b.parse_key(k): {r: {"errors": d * 100, "total": 100.0, "der": d} for r in rids} for k, d in ders.items()}


def test_selection_simplest_within_half_pp():
    t = table_from({"s20_avg_g5_a0.25_c0.75": 0.2000,     # argmin (most complex)
                    "s20_avg_g5_a0.40_c0.60": 0.2049,     # within 0.5 pp
                    "s20_avg_g0_a0.25_c0.45": 0.2050,     # exactly 0.5 pp -> eligible, g0 wins
                    "s20_avg_g0_a0.40_c0.60": 0.2051})    # just beyond
    x = p5.select_within(t)
    assert x["argmin"].key == "s20_avg_g5_a0.25_c0.75"
    assert x["selected"].key == "s20_avg_g0_a0.25_c0.45"
    assert "s20_avg_g0_a0.40_c0.60" not in {c.key for c in x["eligible"]}


def test_tie_order_g_then_theta_act_then_theta_clu():
    same = 0.3
    x = p5.select_within(table_from({"s20_avg_g5_a0.40_c0.60": same, "s20_avg_g0_a0.25_c0.75": same}))
    assert x["selected"].key == "s20_avg_g0_a0.25_c0.75"                      # g first
    x = p5.select_within(table_from({"s20_avg_g0_a0.25_c0.60": same, "s20_avg_g0_a0.40_c0.75": same}))
    assert x["selected"].key == "s20_avg_g0_a0.40_c0.75"                      # then theta_act 0.40 < 0.25
    keys = [f"s120_avg_g0_a0.40_c{c}" for c in ("0.75", "0.45", "0.65", "0.55", "0.60")]
    for i in range(len(keys)):
        x = p5.select_within(table_from({k: same for k in keys[i:]}))
        assert x["selected"].key == keys[-1]                                    # 0.60 < 0.55 < 0.65 < 0.45 < 0.75
    x = p5.select_within(table_from({k: same for k in keys[:3]}))
    assert x["selected"].key == "s120_avg_g0_a0.40_c0.65"


def test_selection_is_pooled_not_macro():
    t = {p3b.parse_key("s20_avg_g5_a0.40_c0.60"): {"a": {"errors": 10, "total": 100}, "b": {"errors": 0, "total": 10}},
         p3b.parse_key("s20_avg_g0_a0.40_c0.60"): {"a": {"errors": 13, "total": 100}, "b": {"errors": 0, "total": 10}}}
    x = p5.select_within(t)   # pooled 9.09 % vs 11.82 %: 2.7 pp apart -> the complex one
    assert x["selected"].key == "s20_avg_g5_a0.40_c0.60"


def test_loro_stability_counts():
    rids = [f"r{i}" for i in range(8)]
    t = {p3b.parse_key("s20_avg_g0_a0.40_c0.60"): {r: {"errors": 30, "total": 100} for r in rids},
         p3b.parse_key("s20_avg_g5_a0.40_c0.60"): {r: {"errors": 20 if r != "r0" else 100, "total": 100} for r in rids}}
    st = p5.loro_stability(t)
    assert st["n"] == 8 and st["dropped"]["r0"]["selected"] == "s20_avg_g5_a0.40_c0.60"
    assert st["selected_recovered"] == 7


def test_joint_mode_ties_to_simpler():
    jm = p5.joint_mode(["s20_avg_g5_a0.40_c0.60"] * 2 + ["s20_avg_g0_a0.25_c0.55"] * 2 + ["s20_avg_g0_a0.40_c0.75"])
    assert jm["config"] == "s20_avg_g0_a0.25_c0.55" and jm["count"] == 2 and len(jm["tied"]) == 2
    jm = p5.joint_mode(["s120_avg_g0_a0.40_c0.45", "s120_avg_g0_a0.40_c0.65"])
    assert jm["config"] == "s120_avg_g0_a0.40_c0.65"
    assert p5.joint_mode(["s20_avg_g5_a0.25_c0.75"] * 3 + ["s20_avg_g0_a0.40_c0.60"] * 2)["config"] == \
        "s20_avg_g5_a0.25_c0.75"


def test_dev_adoption_criterion():
    assert p5.dev_adoption(0.27, 0.259, 100, 110)["adopt_challenger"]
    assert not p5.dev_adoption(0.27, 0.261, 100, 100)["adopt_challenger"]          # 0.9 pp
    x = p5.dev_adoption(0.27, 0.20, 100, 111)
    assert x["der_improves_ge_1pp"] and not x["count_not_up_gt_10pct"] and not x["adopt_challenger"]


def test_floor_adapter():
    segs = [S.Seg(0.4, 3.7, "a"), S.Seg(3.7, 3.9, "b"), S.Seg(3.9, 6.2, "a"), S.Seg(6.5, 6.9, "c")]
    out = p5.floor_adapter(segs)
    assert out == [S.Seg(0.0, 3.0, "a"), S.Seg(3.0, 6.0, "a")]
    assert all(float(s.start).is_integer() and float(s.end).is_integer() for s in out)


# ------------------------------------------------------------------ hold-out subset, bins, claim
def test_subset_loader_matches_prereg_file_exactly(tmp_path):
    stems = p5.load_subset(REPO / "docs/phase5_holdout_subset.txt")
    assert stems == p5.derive_subset() and len(stems) == 60
    assert stems[:3] == ["1_speaker/simple_1spk_015", "1_speaker/simple_1spk_016", "2_speaker/simple_2spk_015"]
    assert [p5.k_of(s) for s in stems] == [k for k in range(1, 31) for _ in range(2)]
    bad = tmp_path / "sub.txt"
    bad.write_text("\n".join(stems[:-1] + ["30_speaker/simple_30spk_012"]) + "\n")
    with pytest.raises(p5.P5Error):
        p5.load_subset(bad)
    bad.write_text("\n".join(stems[1:2] + stems[:1] + stems[2:]) + "\n")
    with pytest.raises(p5.P5Error):
        p5.load_subset(bad)


def test_speaker_bins():
    assert [p5.speaker_bin(k) for k in (1, 4, 5, 8, 9, 16, 17, 30)] == ["1-4", "1-4", "5-8", "5-8", "9-16", "9-16",
                                                                       "17-30", "17-30"]
    for k in (0, 31):
        with pytest.raises(p5.P5Error):
            p5.speaker_bin(k)
    stems = p5.derive_subset()
    counts = {b: sum(p5.speaker_bin(p5.k_of(s)) == b for s in stems) for b, _, _ in p5.SPK_BINS}
    assert counts == {"1-4": 8, "5-8": 8, "9-16": 16, "17-30": 28}


def test_holdout_claim_rule():
    b = lambda d, lo, hi: {"delta_b_minus_a": d, "ci_low": lo, "ci_high": hi}  # noqa: E731
    assert p5.holdout_claim(b(-0.012, -0.02, -0.001))["claim_challenger_better"]
    assert not p5.holdout_claim(b(-0.009, -0.02, -0.001))["claim_challenger_better"]    # < 1 pp
    assert not p5.holdout_claim(b(-0.03, -0.05, 0.0))["claim_challenger_better"]        # 97.5th pct not < 0
    assert p5.holdout_claim(b(-0.010, -0.02, -0.0001))["claim_challenger_better"]      # exactly -1.0 pp


# ------------------------------------------------------------------ Part-A outer lock
def fake_selection(ctx) -> None:
    sel = {"folds": {}}
    for fi in sorted(cv.FOLDS):
        sel["folds"][fi] = {cell: {"selected": p5.cell_configs(cell)[0].key, "selected_der": 0.2, "argmin":
                                   p5.cell_configs(cell)[1].key, "argmin_der": 0.19,
                                   "loro_stability": {"selected_recovered": 8}} for cell in p5.CELLS}
    cv.write_json(ctx.selection_path, sel)


def fake_outer(ctx, miss=None) -> None:
    miss = miss or {"s20_stock": 6.0, "s120_stock": 5.0, "s20_ft": 5.0, "s120_ft": 2.0}
    jobs = []
    for cell in p5.CELLS:
        for fi in sorted(cv.FOLDS):
            shas = {}
            for system in (cell, cell + p5.FLOOR):
                shas[system] = {}
                for rid in cv.FOLDS[fi]:
                    p = p5.outer_path(ctx, system, fi, rid)
                    segs = hyp(miss[cell])
                    S.write_rttm(p, rid, p5.floor_adapter(segs) if system.endswith(p5.FLOOR) else segs)
                    shas[system][rid] = sha(p)
            jobs.append({"cell": cell, "fold": fi, "source": p5.outer_source(cell, fi),
                         "config": p5.cell_configs(cell)[0].key, "recs": list(cv.FOLDS[fi]), "rttm_sha256": shas})
    cv.write_json(ctx.outer_root / "index.json", {"selection_sha256": sha(ctx.selection_path), "jobs": jobs})


def test_outer_lock_freeze_score_tamper(world):
    ctx = world
    fake_selection(ctx)
    fake_outer(ctx)
    p = p5.outer_path(ctx, "s120_ft", "F1", "train_001")
    with pytest.raises(cv.OuterLockError):
        p3b.score_items(ctx, "F1", [("train_001", p)], [0.0])        # before freeze
    with pytest.raises(cv.OuterLockError):
        p3b.score_items(ctx, "F2", [("train_001", p)], [0.0])        # an outer artefact in any context
    with pytest.raises(cv.OuterLockError):
        p5.cmd_a_score(ctx)
    fz = p5.cmd_a_freeze(ctx)
    assert set(fz["systems"]) == set(p5.DEV_SYSTEMS) and all(len(v) == 10 for v in fz["systems"].values())
    res = p5.cmd_a_score(ctx)
    d = res["dev_adoption"]
    assert d["delta_pp"] < -1.0 and d["adopt_challenger"]
    assert set(res["systems"]) == set(p5.DEV_SYSTEMS)
    assert res["final_configs"]["s120_ft"]["config"] == p5.cell_configs("s120_ft")[0].key
    assert res["bootstrap"]["seed"] == 20260925 and res["primary"]["b"] == "s120_ft"
    fc = json.loads(ctx.final_configs_path.read_text())
    assert p5.load_final_configs(ctx) == fc
    assert (ctx.cv_dir / "dev_results.md").read_text().startswith("# Phase-5 Part A (DEVELOPMENT")
    # tamper with a frozen RTTM -> every scoring path refuses
    S.write_rttm(p, "train_001", hyp(0.0))
    with pytest.raises(cv.OuterLockError):
        p5.cmd_a_score(ctx)
    with pytest.raises(cv.OuterLockError):
        p3b.score_items(ctx, "F1", [("train_001", p)], [0.0])


def test_final_configs_must_match_last_scoring(world):
    ctx = world
    fake_selection(ctx)
    fake_outer(ctx)
    p5.cmd_a_freeze(ctx)
    p5.cmd_a_score(ctx)
    fc = json.loads(ctx.final_configs_path.read_text())
    fc["cells"]["s120_ft"]["config"] = "s120_avg_g0_a0.25_c0.75"
    fc["dev_results_sha256"] = "0" * 64
    cv.write_json(ctx.final_configs_path, fc)
    with pytest.raises(p5.P5Error):
        p5.load_final_configs(ctx)


# ------------------------------------------------------------------ hold-out lock
@pytest.fixture
def ho_world(world, monkeypatch):
    ctx = world
    monkeypatch.setattr(p5, "HOLDOUT_BOOT_N", ctx.bootstrap_n)   # registered 50k -> 200 for test speed
    stems = p5.derive_subset()
    lines = []
    for s in stems:
        rid = p5.rid_of(s)
        w = p5.ho_file(ctx, s, "wav")
        w.parent.mkdir(parents=True, exist_ok=True)
        sf.write(w, np.zeros(int(DUR * 1000), np.float32), 1000)
        S.write_rttm(p5.ho_file(ctx, s, "rttm"), rid, [S.Seg(0.0, 10.0, "A"), S.Seg(8.0, 20.0, "B"),
                                                      S.Seg(20.0, 30.0, "A")])
        p5.ho_file(ctx, s, "json").write_text("{}")
        for e in ("json", "rttm", "wav"):
            lines.append(f"{sha(p5.ho_file(ctx, s, e))}  ./{s}.{e}")
    ctx.holdout_sha_file.write_text("\n".join(lines) + "\n")
    miss = {"incumbent_acoustic": 6.0, "incumbent_benchmark": 5.0, "challenger_acoustic": 2.0,
            "challenger_benchmark": 2.0, "community1": 8.0, "ftnemotron_acoustic": 4.0, "ftnemotron_benchmark": 4.0}
    systems = {}
    for name in p5.HOLDOUT_SYSTEMS:
        systems[name] = {"source": "x", "config": None, "mode": "m", "rttm_sha256": {}}
        for i, s in enumerate(stems):
            rid = p5.rid_of(s)
            p = ctx.ho_rttm / name / f"{rid}.rttm"
            S.write_rttm(p, rid, hyp(miss[name] + (i % 3) * 0.5))
            systems[name]["rttm_sha256"][rid] = sha(p)
    cv.write_json(ctx.ho_index_path, {"systems": systems, "models": {}})
    return ctx


def test_holdout_data_verification(ho_world):
    ctx = ho_world
    assert p5.holdout_present(ctx)[0]
    assert p5.verify_holdout_data(ctx)["n_files_verified"] == 120
    s = p5.derive_subset()[5]
    S.write_rttm(p5.ho_file(ctx, s, "rttm"), p5.rid_of(s), [S.Seg(0.0, 1.0, "Z")])
    with pytest.raises(p5.P5Error):
        p5.verify_holdout_data(ctx)


def test_holdout_lock(ho_world):
    ctx = ho_world
    with pytest.raises(p5.HoldoutLockError):
        p5.cmd_b_score(ctx)                                  # before freeze
    fz = p5.cmd_b_freeze(ctx)
    assert set(fz["systems"]) == set(p5.HOLDOUT_SYSTEMS) and all(len(v) == 60 for v in fz["systems"].values())
    res = p5.cmd_b_score(ctx)
    assert res["bootstrap"]["seed"] == 20260926 and res["primary"]["a"] == "incumbent_acoustic"
    assert res["primary"]["claim_challenger_better"] and res["primary"]["delta_pp"] < -1.0
    inc = res["systems"]["incumbent_acoustic"]
    assert set(inc["by_bin"]) == {"1-4", "5-8", "9-16", "17-30"} and inc["by_bin"]["17-30"]["n_mixtures"] == 28
    assert inc["sum_abs_count_error"] == 0 and res["reference_speakers"]["simple_1spk_015"] == 2
    assert set(inc["by_collar"]) == {"0.0", "0.25", "0.5"}
    assert len(inc["by_collar"]["0.0"]["per_mixture"]) == 60
    assert set(res["secondary"]) >= {"incumbent_vs_community1", "ftnemotron_vs_base"}
    with pytest.raises(p5.HoldoutLockError):
        p5.cmd_b_score(ctx)                                  # score ONCE
    s = p5.derive_subset()[0]
    S.write_rttm(ctx.ho_rttm / "challenger_acoustic" / f"{p5.rid_of(s)}.rttm", p5.rid_of(s), hyp(0.0))
    with pytest.raises(p5.HoldoutLockError):
        p5.verify_holdout_frozen(ctx)                        # tampered


def test_b_score_refuses_non_registered_bootstrap_n(ho_world):
    """Part B is scored once: a --bootstrap-n other than the registered count is refused BEFORE the lock
    records a scoring (it used to decide the claim on 7 resamples and then lock out the 50k rerun)."""
    ctx = ho_world
    p5.cmd_b_freeze(ctx)
    assert p5.HOLDOUT_BOOT_N == ctx.bootstrap_n
    good = ctx.bootstrap_n
    ctx.bootstrap_n = 7
    with pytest.raises(p5.P5Error, match="registered"):
        p5.cmd_b_score(ctx)
    with pytest.raises(p5.P5Error, match="registered"):
        p5.check_holdout_boot_n(ctx)
    assert not ctx.ho_scored_path.exists() and not ctx.ho_results_path.exists()
    ctx.bootstrap_n = good
    assert p5.cmd_b_score(ctx)["bootstrap"]["n"] == good


def test_holdout_boot_n_is_the_registered_50k():
    assert p5.HOLDOUT_BOOT_N == 50_000 == S.BOOTSTRAP_N


def test_holdout_lock_config_tamper(ho_world):
    ctx = ho_world
    p5.cmd_b_freeze(ctx)
    cv.write_json(ctx.final_configs_path, {"cells": {}})     # a config appears/changes after the freeze
    with pytest.raises(p5.HoldoutLockError):
        p5.cmd_b_score(ctx)


def test_b_infer_refuses_without_validation(ho_world):
    with pytest.raises(p5.P5Error):
        p5.cmd_b_infer(ho_world)


def test_preflight_rejects_holdout_ids_if_embed_ft_cannot_parse_them():
    from bnd import embed_ft

    try:
        embed_ft.rec_id("simple_1spk_015")
    except Exception:
        with pytest.raises(p5.P5Error):
            p5.preflight_ft_embed(["simple_1spk_015"])
    else:
        p5.preflight_ft_embed(["simple_1spk_015"])


def test_ref_stats_overlap_and_silence():
    ref = [S.Seg(0.0, 10.0, "A"), S.Seg(5.0, 15.0, "B")]
    nemo = np.zeros(2000, np.float32)
    nemo[:500] = 0.9                                           # nemotron speech 0-5 s
    sil = np.zeros(int(20 / 0.032), np.float32)
    sil[int(10 / 0.032):] = 0.9                                # silero speech 10-20 s
    st = p5.ref_stats(ref, 20.0, nemo, 0.01, sil, 0.032)
    assert st["n_speakers"] == 2 and st["labelled_fraction"] == pytest.approx(0.75)
    assert st["overlap_fraction_of_labelled"] == pytest.approx(5 / 15)
    assert st["silent_both_fraction_of_labelled"] == pytest.approx(5 / 15, abs=0.01)


def write_fake_fit(ctx, spec: dict, **over) -> Path:
    from bnd import embed_ft

    d = p5.fit_dir(ctx, spec["tag"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "embedder.pt").write_bytes(b"weights")
    m = {"tag": spec["tag"], "train_recordings": spec["train_recs"], "recipe": embed_ft.RECIPE,
         "steps": embed_ft.RECIPE["steps"], "checkpoint": "embedder.pt", "sha256": sha(d / "embedder.pt")}
    m.update(over)
    m["train_binding"] = embed_ft.train_binding(m)
    (d / "manifest.json").write_text(json.dumps(m))
    return d


@pytest.mark.parametrize("over", [{"steps": 50}, {"recipe": {"steps": 50}}, {"tag": "smoke"}])
def test_fit_done_refuses_non_registered_fit(world, monkeypatch, over):
    """A complete checkpoint under a registered tag trained with another recipe/step count is neither
    adopted (no done-marker claiming the registered RECIPE) nor silently replaced."""
    from bnd import embed_ft

    ctx = world
    spec = p5.fit_by_tag("p5emb_all10")
    write_fake_fit(ctx, spec, **over)
    monkeypatch.setattr(embed_ft, "OUT_ROOT", ctx.embedders_root)
    with pytest.raises(p5.P5Error, match="not the registered fit"):
        p5.fit_done(ctx, spec)
    r = p5.Rates(p3b.Rates())
    with pytest.raises(p5.P5Error, match="not the registered fit"):
        p5.ensure_fit(ctx, None, r, spec)
    assert not list(ctx.cv_dir.rglob("*embfit*")), "a done-marker was written"
    # an edited manifest (train_binding no longer matches) is refused too
    d = write_fake_fit(ctx, spec)
    m = json.loads((d / "manifest.json").read_text())
    m["steps"] = 50
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(p5.P5Error):
        p5.fit_done(ctx, spec)
    # the registered fit is adopted
    write_fake_fit(ctx, spec)
    assert p5.fit_done(ctx, spec)["steps"] == 1500


def write_fake_ftn(ctx, **over) -> None:
    model, rep = p5.ftn_paths(ctx)
    model.parent.mkdir(parents=True, exist_ok=True)
    rep.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"nemo-model")
    d = {"tag": p5.FTN_TAG, "train_recs": sorted(RECORDINGS), "lr": 5e-05, "timing": "shift", "status": "ok",
         "optimizer_steps": 165, "export": {"path": str(model), "bytes": model.stat().st_size}}
    d.update(over)
    rep.write_text(json.dumps(d))


@pytest.mark.parametrize("over", [{"lr": 1e-4}, {"timing": "none"}, {"timing": "raw"}, {"optimizer_steps": 50},
                                  {"export": {"path": "/elsewhere/x.nemo", "bytes": 10}}, {"export": None},
                                  {"lr": None}])
def test_ftn_model_requires_registered_hparams(world, over):
    ctx = world
    ctx.dry_run = True                   # a refused model is re-planned (dry run: no GPU, nothing moved)
    r = p5.Rates(p3b.Rates())
    write_fake_ftn(ctx, **over)
    est, msha = p5.ensure_ftn_model(ctx, None, r)
    assert msha is None and est == r.ftn_fit_s
    write_fake_ftn(ctx)
    est, msha = p5.ensure_ftn_model(ctx, None, r)
    assert est == 0.0 and msha == sha(p5.ftn_paths(ctx)[0])


# ------------------------------------------------------------------ ledger
def fake_gpu_run(tmp_path: Path) -> Path:
    g = tmp_path / "gpu_run.sh"
    g.write_text("#!/usr/bin/env bash\necho \"[gpu_run] card=0 pid=$$\" >&2\nexec \"$@\"\n")
    g.chmod(g.stat().st_mode | stat.S_IEXEC)
    return g


def test_ledger_start_warn_abort_and_no_double_count(world, tmp_path):
    ctx = world
    ctx.gpu_run = fake_gpu_run(tmp_path)
    led = p5.P5Ledger(ctx)
    assert led.used_h() == pytest.approx(6.03)
    # a job whose child appends a bnd.embed_ft-style self entry (1800 s) to the same ledger
    child = (f"import json; open({str(ctx.ledger_path)!r}, 'a').write(json.dumps({{'kind': 'embed_ft-fit', "
             f"'gpu_wall_s': 1800.0, 'tag': 't'}}) + '\\n')")
    ctx.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    e = led.run_est([sys.executable, "-c", child], "embfit", "embfit:t", tmp_path / "log.txt", 10.0)
    ents = led.entries()
    assert ents[-1]["phase"] == "5" and ents[-1]["covers_lines"] == [0] and ents[-1]["tag"] == "embfit:t"
    assert led.used_h() == pytest.approx(6.03 + e["gpu_wall_s"] / 3600, abs=1e-6)   # the 1800 s not added
    ledger_line = [l for l in ctx._lines if "gpu embfit embfit:t" in l][-1]
    assert f"ledger={led.used_h():.3f}/" in ledger_line   # not counted twice even before the retag
    # an uncovered stand-alone self entry does count
    with open(ctx.ledger_path, "a") as f:
        f.write(json.dumps({"kind": "embed_ft-fit", "gpu_wall_s": 3600.0 * 5.4, "tag": "x"}) + "\n")
    assert led.used_h() > 11.4
    ctx._lines.clear()
    led.check(0.2 * 3600, "warn")                          # 11.4 + 0.2 > 11.53 -> warning, not abort
    assert any("WARNING" in l for l in ctx._lines)
    with pytest.raises(cv.BudgetError):
        led.check(0.7 * 3600, "abort")                     # would exceed 12 h


def test_ledger_no_transient_double_count_near_budget(world, tmp_path):
    """The child's own 1800 s entry must not be counted on top of the job's entry in the post-job budget
    check (it used to raise a spurious BudgetError after a successful fit near the line)."""
    ctx = world
    ctx.gpu_run = fake_gpu_run(tmp_path)
    ctx.budget_h = 6.03 + 0.4                            # job ~0 h fits; job + a double-counted 0.5 h does not
    led = p5.P5Ledger(ctx)
    child = (f"import json; open({str(ctx.ledger_path)!r}, 'a').write(json.dumps({{'kind': 'embed_ft-fit', "
             f"'gpu_wall_s': 1800.0, 'tag': 't'}}) + '\\n')")
    ctx.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    e = led.run_est([sys.executable, "-c", child], "embfit", "embfit:t", tmp_path / "log.txt", 10.0)
    assert led.used_h() == pytest.approx(6.03 + e["gpu_wall_s"] / 3600, abs=1e-6)
    assert led._pending_n0 is None


# ------------------------------------------------------------------ dry runs write nothing
def test_dry_runs_write_nothing(world):
    ctx = world
    ctx.dry_run = True
    before = tree_digest(ctx.cv_dir.parent)
    p5.cmd_a_search(ctx)
    p5.cmd_a_outer(ctx)
    r = p5.cmd_b_validate(ctx)
    assert r["projected_gpu_h"] > 0
    assert tree_digest(ctx.cv_dir.parent) == before and not ctx.cv_dir.exists()
    assert any("DRY-RUN" in l for l in ctx._lines)
