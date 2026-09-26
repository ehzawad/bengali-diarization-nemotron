"""CPU tests for bnd/cv.py on synthetic data: plan dedup, leakage invariants, pooled selection,
D3 tie-breaks, outer lock (before freeze / after tampering), adoption gate, GPU ledger budget,
and the --dry-run path of `inner`. Nothing here touches the GPU or the registered runs/cv tree."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import soundfile as sf

from bnd import cv
from bnd import score as S
from bnd.labels import RECORDINGS

DUR = 40.0
SMALL_GRID = [(0.50, 0.65), (0.60, 0.85)]


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def ctx(tmp_path):
    audio = tmp_path / "audio"
    ref = tmp_path / "ref"
    audio.mkdir()
    ref.mkdir()
    for rid in RECORDINGS:
        sf.write(audio / f"{rid}.wav", np.zeros(int(DUR * 1000), np.float32), 1000)
        S.write_rttm(ref / f"{rid}.rttm", rid, [S.Seg(0.0, 10.0, "A"), S.Seg(10.0, 20.0, "B"),
                                                 S.Seg(20.0, 30.0, "A")])
    fake_model = tmp_path / "base.nemo"
    fake_model.write_bytes(b"not a model")
    lines = []
    c = cv.Ctx(cv_dir=tmp_path / "cv", ref_dir=ref, audio_dir=audio, base_model=fake_model,
               fits_dir=tmp_path / "fits",
               models_dir=tmp_path / "models", bootstrap_n=500, jobs=1, out=lines.append)
    c._lines = lines
    return c


def hyp(rid: str, miss: float) -> list[S.Seg]:
    """Reference with the last ``miss`` seconds of the final A turn missing (DER = miss / 30)."""
    return [S.Seg(0.0, 10.0, "x"), S.Seg(10.0, 20.0, "y"), S.Seg(20.0, 30.0 - miss, "x")]


# ------------------------------------------------------------------ plan / dedup / leakage
def test_dedup_40_unique_inner_fits():
    fits = cv.inner_fits()
    assert len(fits) == 40
    assert len({f["tag"] for f in fits}) == 40
    assert len({f["set"] for f in fits}) == 10  # C(5,2)
    # every (outer i, inner j) ordered pair maps to exactly one fit per recipe, shared with (j, i)
    for recipe in cv.RECIPES:
        for i, j in itertools.permutations(sorted(cv.FOLDS), 2):
            f = cv.fit_for((i, j), recipe)
            assert f is cv.fit_for((j, i), recipe) or f["tag"] == cv.fit_for((j, i), recipe)["tag"]
            assert set(f["heldout"]) == {i, j}


def test_no_fit_trains_on_a_recording_it_is_scored_on():
    for f in cv.inner_fits():
        assert len(f["train_recs"]) == 6 and len(f["heldout_recs"]) == 4
        assert not set(f["train_recs"]) & set(f["heldout_recs"])
        assert sorted(f["train_recs"] + f["heldout_recs"]) == sorted(RECORDINGS)
    for fi in cv.FOLDS:
        for recipe in cv.RECIPES:
            for fit, fj, recs in cv.inner_sources(fi, recipe):
                assert fj != fi
                assert not set(fit["train_recs"]) & set(recs)            # scored recordings
                assert not set(fit["train_recs"]) & set(cv.FOLDS[fi])    # the outer fold
            r = cv.refit_spec(fi, recipe)
            assert len(r["train_recs"]) == 8 and not set(r["train_recs"]) & set(cv.FOLDS[fi])
    # each recording is held out by 4 sets x 4 recipes
    for rid in RECORDINGS:
        assert sum(rid in f["heldout_recs"] for f in cv.inner_fits()) == 16


def test_ensure_fit_and_infer_refuse_leaks(ctx):
    spec = dict(cv.inner_fits()[0])
    spec["heldout_recs"] = spec["heldout_recs"] + [spec["train_recs"][0]]
    with pytest.raises(cv.CVError, match="overlaps"):
        cv.ensure_fit(ctx, cv.Ledger(ctx), spec)
    spec = cv.inner_fits()[0]
    with pytest.raises(cv.CVError, match="own training"):
        cv.ensure_infer(ctx, cv.Ledger(ctx), spec, Path("/x"), "sha", [spec["train_recs"][0]], cv.THETA_ACT)


def test_plan_written_structure(ctx):
    plan = cv.build_plan(ctx)
    assert plan["n_inner_fits"] == 40 and plan["n_refits_max"] == 10
    for fi, o in plan["outer"].items():
        assert not set(o["inner_recs"]) & set(cv.FOLDS[fi]) and len(o["inner_recs"]) == 8
        for recipe, tags in o["inner_fits"].items():
            assert len(tags) == 4


# ------------------------------------------------------------------ D3 tie-breaks
def test_choose_cell_tie_breaks():
    grid = {c: 0.3 for c in cv.GRID}
    assert cv.choose_cell(grid)[0] == (0.50, 0.65)                         # centre
    t = {c: 0.5 for c in cv.GRID}
    t[(0.40, 0.65)] = t[(0.60, 0.65)] = 0.2                                # both L1 = 1
    assert cv.choose_cell(t)[0] == (0.40, 0.65)                            # lower theta_act
    t = {c: 0.5 for c in cv.GRID}
    t[(0.50, 0.55)] = t[(0.50, 0.75)] = 0.2
    assert cv.choose_cell(t)[0] == (0.50, 0.55)                            # lower theta_clu
    t = {c: 0.5 for c in cv.GRID}
    t[(0.50, 0.45)] = t[(0.60, 0.75)] = 0.2                                # both L1 = 2
    assert cv.choose_cell(t)[0] == (0.50, 0.45)
    t = {c: 0.5 for c in cv.GRID}
    t[(0.40, 0.45)] = 0.2
    t[(0.50, 0.65)] = 0.2 + 1e-12                                          # within 1e-8 pp: tie
    assert cv.choose_cell(t)[0] == (0.50, 0.65)
    t[(0.50, 0.65)] = 0.2 + 1e-9                                           # beyond tolerance
    assert cv.choose_cell(t)[0] == (0.40, 0.45)


def test_choose_recipe_tie_breaks():
    assert cv.choose_recipe({r: 0.3 for r in cv.RECIPES})[0] == "lr1e-5_raw"
    assert cv.choose_recipe({"lr5e-5_raw": 0.3, "lr1e-5_shift": 0.3, "lr5e-5_shift": 0.4,
                             "lr1e-5_raw": 0.31})[0] == "lr1e-5_shift"
    assert cv.choose_recipe({"lr5e-5_raw": 0.3, "lr5e-5_shift": 0.3, "lr1e-5_raw": 0.4,
                             "lr1e-5_shift": 0.4})[0] == "lr5e-5_raw"
    assert cv.choose_recipe({"lr5e-5_shift": 0.1, "lr1e-5_raw": 0.1 + 1e-9})[0] == "lr5e-5_shift"


# ------------------------------------------------------------------ pooled selection on synthetic hyps
def write_inner_hyps(ctx, grid):
    """zero-shot: cell (0.60,0.85) perfect, others miss 6 s. fits: lr5e-5_shift perfect, others miss 3 s.
    Every fit also writes predictions for BOTH held-out folds, including the future outer fold."""
    for a, c in grid:
        ck = cv.cell_key(a, c)
        for rid in RECORDINGS:
            S.write_rttm(ctx.hyp_root / "zeroshot" / ck / f"{rid}.rttm", rid,
                         hyp(rid, 0.0 if (a, c) == (0.60, 0.85) else 6.0))
        for f in cv.inner_fits():
            for rid in f["heldout_recs"]:
                S.write_rttm(ctx.hyp_root / f["system"] / ck / f"{rid}.rttm", rid,
                             hyp(rid, 0.0 if f["recipe"] == "lr5e-5_shift" else 3.0))


def test_pooled_selection_excludes_outer_fold(ctx, monkeypatch):
    monkeypatch.setattr(cv, "GRID", SMALL_GRID)
    write_inner_hyps(ctx, SMALL_GRID)
    ctx.jobs = 2  # exercises the forked scoring pool
    sel = cv.cmd_select(ctx)
    assert ctx.selection_path.exists()
    for fi, f in sel["folds"].items():
        outer = set(cv.FOLDS[fi])
        assert not outer & set(f["inner_recs"])
        tabs = [f["zeroshot"]] + list(f["recipes"].values())
        for t in tabs:
            for cell in t["cells"].values():
                assert set(cell["per_recording"]) == set(f["inner_recs"]) and cell["n_recordings"] == 8
        # symlink dirs contain only inner recordings, and point at fits that never saw Fi
        for d in (ctx.select_root / fi).rglob("*.rttm"):
            assert d.stem not in outer
            assert d.is_symlink()
        for recipe, src in f["sources"].items():
            for rid, tag in src.items():
                fit = next(x for x in cv.inner_fits() if x["tag"] == tag)
                assert not set(fit["train_recs"]) & (outer | {rid})
        assert f["zeroshot"]["selected_cell"] == "0.60_0.85"
        assert f["selected_recipe"] == "lr5e-5_shift"
        # the three imperfect recipes tie on both cells -> D3 centre cell
        assert f["recipes"]["lr1e-5_raw"]["selected_cell"] == "0.50_0.65"
        assert f["recipes"]["lr1e-5_raw"]["selected_der"] == pytest.approx(3.0 / 30.0)
    # a stray outer-fold file in a select dir is refused by the scorer (load via dir contents)
    with pytest.raises(cv.OuterLockError):
        cv.score_paths(ctx, "F1", {"train_001": ctx.hyp_root / "zeroshot" / "0.50_0.65" / "train_001.rttm"},
                       [0.0])


def test_select_refuses_when_inner_predictions_missing(ctx, monkeypatch):
    monkeypatch.setattr(cv, "GRID", SMALL_GRID)
    write_inner_hyps(ctx, SMALL_GRID)
    (ctx.hyp_root / "cv_inner_x12_lr1e-5_raw" / "0.50_0.65" / "train_011.rttm").unlink()
    with pytest.raises(cv.CVError, match="missing"):
        cv.cmd_select(ctx)


# ------------------------------------------------------------------ outer lock / freeze / score
def synth_selection(ctx):
    sel = {"created": "x", "folds": {}}
    for fi in cv.FOLDS:
        sel["folds"][fi] = {"zeroshot": {"selected_cell": "0.50_0.65"}, "selected_recipe": "lr5e-5_raw",
                            "recipes": {"lr5e-5_raw": {"selected_cell": "0.60_0.85"}}}
    cv.write_json(ctx.selection_path, sel)
    return sel


def synth_outer(ctx, miss_by_system=None):
    miss_by_system = miss_by_system or {}
    sel = synth_selection(ctx)
    jobs = cv.outer_jobs(ctx, sel, ctx.cache_root)
    for j in jobs:
        shas = {}
        for rid in j["recs"]:
            p = Path(j["out_dir"]) / f"{rid}.rttm"
            S.write_rttm(p, rid, hyp(rid, miss_by_system.get(j["name"], {}).get(rid, 2.0)))
            shas[rid] = cv.file_sha(ctx, p)
        cv.write_marker(ctx, "link", cv.dir_marker_name(ctx, Path(j["out_dir"])), {"synthetic": True},
                        {"out_dir": str(j["out_dir"]), "rttm_sha256": shas})
    cv.write_outer_index(ctx, jobs)
    return jobs


def test_outer_lock_refuses_before_freeze(ctx):
    synth_outer(ctx)
    with pytest.raises(cv.OuterLockError, match="does not exist"):
        cv.cmd_score(ctx)
    p = ctx.outer_root / "ft_sel" / "F2" / "train_011.rttm"
    with pytest.raises(cv.OuterLockError):
        cv.score_paths(ctx, "F2", {"train_011": p}, [0.0])
    # file lock: an outer prediction is refused in ANY context before freeze
    with pytest.raises(cv.OuterLockError):
        cv.score_paths(ctx, "F1", {"train_011": p}, [0.0])


def test_freeze_refuses_incomplete(ctx):
    synth_outer(ctx)
    (ctx.outer_root / "ft_strict" / "F3" / "train_005.rttm").unlink()
    with pytest.raises(cv.CVError, match="freeze refused"):
        cv.cmd_freeze(ctx)
    assert not ctx.frozen_path.exists()


def test_freeze_score_and_tamper(ctx):
    # ft_sel better on every recording by 1 s of 30 s (3.33 pp), except train_006 (worse by 1 s)
    zs = {rid: 5.0 for rid in RECORDINGS}
    ft = {rid: 4.0 for rid in RECORDINGS}
    ft["train_006"] = 6.0
    synth_outer(ctx, {"zs_sel": zs, "ft_sel": ft})
    fz = cv.cmd_freeze(ctx)
    assert sum(len(v) for v in fz["systems"].values()) == len(cv.OUTER_SYSTEMS) * 10
    res = cv.cmd_score(ctx)
    g = res["gate"]["conditions"]
    assert len(g) == 6
    assert g["wins_ge_7"]["wins"] == 9 and g["wins_ge_7"]["pass"]
    assert g["corpus_improvement_ge_2pp"]["value_pp"] == pytest.approx(100 * 8 / 300)
    assert g["no_recording_worse_gt_5pp"]["worst_recording"] == "train_006"
    assert res["systems"]["zs_sel"]["0.0"]["corpus"]["der"] == pytest.approx(5 / 30)
    assert (ctx.cv_dir / "results.md").read_text().startswith("# Phase-2 registered CV results")
    # tamper with one frozen outer RTTM -> scoring and re-freezing refuse
    p = ctx.outer_root / "zs_sel" / "F4" / "train_002.rttm"
    S.write_rttm(p, "train_002", hyp("train_002", 0.0))
    with pytest.raises(cv.OuterLockError, match="no longer match"):
        cv.cmd_score(ctx)
    with pytest.raises(cv.OuterLockError):
        cv.score_paths(ctx, "F4", {"train_002": p}, [0.0])


def test_tampered_selection_breaks_lock(ctx):
    synth_outer(ctx)
    cv.cmd_freeze(ctx)
    sel = json.loads(ctx.selection_path.read_text())
    sel["folds"]["F1"]["selected_recipe"] = "lr1e-5_raw"
    ctx.selection_path.write_text(json.dumps(sel))
    with pytest.raises(cv.OuterLockError, match="selection.json"):
        cv.cmd_score(ctx)


# ------------------------------------------------------------------ gate
def recs(errors, totals=None, collar=0.0):
    totals = totals or [100.0] * len(errors)
    return [S.RecScore(f"r{i:02d}", collar, 500.0, t, e, 0.0, 0.0) for i, (e, t) in enumerate(zip(errors, totals))]


def gate(a, b, a25=None, b25=None, boot=None, totals=None):
    A, B = recs(a, totals), recs(b, totals)
    A25, B25 = recs(a25 or a, totals, 0.25), recs(b25 or b, totals, 0.25)
    boot = boot or S.paired_bootstrap(A, B, n=2000)
    return cv.gate_verdict(A, B, A25, B25, boot)


def fails(v):
    return sorted(k for k, c in v["conditions"].items() if not c["pass"])


def test_gate_all_pass():
    v = gate([30] * 10, [25] * 10)
    assert v["adopt_finetuned"] and fails(v) == [] and v["default_system"] == "fine-tuned"


def test_gate_each_condition_can_fail_alone():
    assert fails(gate([30] * 10, [29] * 10)) == ["corpus_improvement_ge_2pp"]
    assert fails(gate([30] * 10, [20] * 6 + [31] * 4)) == ["wins_ge_7"]
    v = gate([30] * 10, [25] * 9 + [36])
    assert fails(v) == ["no_recording_worse_gt_5pp"] and v["conditions"]["no_recording_worse_gt_5pp"]["worst_recording"] == "r09"
    assert fails(gate([30] * 10, [25] * 10, a25=[30] * 10, b25=[31] * 10)) == ["no_corpus_regression_collar_0.25"]
    A, B = recs([30] * 10), recs([25] * 10)
    fake = dict(S.paired_bootstrap(A, B, n=100), ci_high=0.001)
    assert fails(cv.gate_verdict(A, B, recs([30] * 10, collar=.25), recs([25] * 10, collar=.25), fake)) == \
        ["bootstrap_p97_5_lt_0"]
    # corpus improves (one huge recording) but macro mean delta > 0 and most recordings lose
    v = gate([500] + [3] * 9, [300] + [3.3] * 9, totals=[1000] + [10] * 9)
    assert "macro_mean_delta_lt_0" in fails(v) and not v["adopt_finetuned"]
    assert v["conditions"]["macro_mean_delta_lt_0"]["value_pp"] == pytest.approx((-20 + 9 * 3) / 10)


def test_gate_boundaries():
    # exactly 2.0 pp improvement and exactly 5.0 pp worst regression pass
    v = gate([30] * 10, [25] * 5 + [30] * 4 + [35])  # -25 + 5 = -20 s of 1000 s
    assert v["conditions"]["corpus_improvement_ge_2pp"]["value_pp"] == pytest.approx(2.0)
    assert v["conditions"]["corpus_improvement_ge_2pp"]["pass"]
    assert v["conditions"]["no_recording_worse_gt_5pp"]["pass"]


# ------------------------------------------------------------------ ledger
def fake_gpu_run(tmp_path, card="1"):
    p = tmp_path / "fake_gpu_run.sh"
    p.write_text(f'#!/usr/bin/env bash\necho "[gpu_run] card={card} pid=$$ cmd=$*" >&2\nexec "$@"\n')
    p.chmod(0o755)
    return p


def test_ledger_records_card_and_time(ctx, tmp_path):
    ctx.gpu_run = fake_gpu_run(tmp_path)
    led = cv.Ledger(ctx)
    e = led.run(["python3", "-c", "import time; time.sleep(0.2); print('ok')"], "train", "t1", tmp_path / "l.log")
    assert e["card"] == "1" and e["returncode"] == 0 and e["gpu_wall_s"] >= 0.2
    assert len(led.entries()) == 1
    assert led.used_h() == pytest.approx(ctx.start_h + e["gpu_wall_s"] / 3600)
    assert "ok" in (tmp_path / "l.log").read_text()
    with pytest.raises(cv.CVError, match="failed"):
        led.run(["python3", "-c", "raise SystemExit(3)"], "infer", "t2", tmp_path / "l.log", audio_s=10)
    assert led.entries()[-1]["returncode"] == 3  # failures are charged too


def test_ledger_budget_abort_before_launch(ctx, tmp_path):
    ctx.gpu_run = fake_gpu_run(tmp_path)
    ctx.start_h = 11.99  # 0.27 h phase-2a start replaced by a nearly spent budget
    led = cv.Ledger(ctx)
    marker = tmp_path / "launched"
    with pytest.raises(cv.BudgetError, match="aborting before launch"):
        led.run(["touch", str(marker)], "train", "t", tmp_path / "l.log")
    assert not marker.exists() and led.entries() == []
    # the default start balance is phase-2a's 0.27 h
    assert cv.Ctx().start_h == 0.27 and cv.Ctx().budget_h == 12.0


def test_ledger_budget_uses_cumulative_entries(ctx, tmp_path):
    ctx.gpu_run = fake_gpu_run(tmp_path)
    ctx.ledger_path.parent.mkdir(parents=True)
    ctx.ledger_path.write_text(json.dumps({"kind": "train", "gpu_wall_s": 11.7 * 3600, "returncode": 0}) + "\n")
    with pytest.raises(cv.BudgetError):
        cv.Ledger(ctx).run(["true"], "infer", "t", tmp_path / "l.log", audio_s=7000)


# ------------------------------------------------------------------ dry run
def test_inner_dry_run_prints_commands_and_writes_nothing(ctx):
    ctx.dry_run = True
    cv.cmd_inner(ctx)
    lines = ctx._lines
    gpu = [l for l in lines if l.startswith("DRY-RUN [gpu]")]
    cpu = [l for l in lines if l.startswith("DRY-RUN [cpu]")]
    assert sum("bnd.train" in l for l in gpu) == 40
    # 40 fits + zero-shot regeneration (synthetic cache root has no provenance-matching cache)
    assert sum("bnd.infer" in l for l in gpu) == 41
    assert all(str(ctx.gpu_run) in l for l in gpu)
    assert len(cpu) == 15 + 40 * 15 and all("bnd.link" in l for l in cpu)
    assert not ctx.cv_dir.exists()
    # every inner training command excludes the recordings its inference runs on
    for f in cv.inner_fits():
        tr = next(l for l in gpu if f"--tag {f['tag']} " in l)
        inf = next(l for l in gpu if f"--system {f['tag']} " in l)
        for rid in f["heldout_recs"]:
            assert rid not in tr and rid in inf


# ------------------------------------------------------------------ regression: review findings
def test_outer_artifacts_unscorable_prefreeze_in_any_context(ctx, tmp_path):
    """Finding 1: the lock is on the file, not only on the caller's context."""
    synth_outer(ctx)
    for system in ("ft_sel", "ft_strict", "zs_sel", "zs_strict", "ft_sel_oracle_link"):
        for fi, recs in cv.FOLDS.items():
            p = ctx.outer_root / system / fi / f"{recs[0]}.rttm"
            for other in cv.FOLDS:
                with pytest.raises(cv.OuterLockError):
                    cv.score_paths(ctx, other, {recs[0]: p}, [0.0])
    # a symlink into runs/cv/outer and any cv_refit_* output elsewhere are refused too
    link = tmp_path / "sneaky" / "train_011.rttm"
    link.parent.mkdir()
    link.symlink_to(ctx.outer_root / "ft_sel" / "F2" / "train_011.rttm")
    with pytest.raises(cv.OuterLockError):
        cv.score_paths(ctx, "F1", {"train_011": link}, [0.0])
    refit_hyp = ctx.hyp_root / "cv_refit_F2_lr1e-5_raw" / "0.50_0.65" / "train_011.rttm"
    S.write_rttm(refit_hyp, "train_011", hyp("train_011", 0.0))
    with pytest.raises(cv.OuterLockError):
        cv.score_paths(ctx, "F1", {"train_011": refit_hyp}, [0.0])
    # an ordinary inner hyp is still scorable in a foreign context
    inner = ctx.hyp_root / "cv_inner_x12_lr1e-5_raw" / "0.50_0.65" / "train_011.rttm"
    S.write_rttm(inner, "train_011", hyp("train_011", 3.0))
    assert cv.score_paths(ctx, "F1", {"train_011": inner}, [0.0])[0.0][0].der == pytest.approx(0.1)
    # after freeze the exact frozen file is scorable (any context); the unfrozen refit hyp still is not
    cv.cmd_freeze(ctx)
    p = ctx.outer_root / "ft_sel" / "F2" / "train_011.rttm"
    assert cv.score_paths(ctx, "F1", {"train_011": p}, [0.0])[0.0][0].der == pytest.approx(2.0 / 30)
    with pytest.raises(cv.OuterLockError):
        cv.score_paths(ctx, "F1", {"train_011": refit_hyp}, [0.0])


def _relink(ctx, rel_dir, rid, miss):
    d = ctx.outer_root / rel_dir
    p = d / f"{rid}.rttm"
    S.write_rttm(p, rid, hyp(rid, miss))
    name = cv.dir_marker_name(ctx, d)
    m = cv.read_marker(ctx, "link", name)
    m["outputs"]["rttm_sha256"][rid] = cv.file_sha(ctx, p)
    cv.write_json(cv.marker_path(ctx, "link", name), m)


def test_rescore_after_unfreeze_needs_deviation_and_keeps_old_results(ctx):
    """Finding 2: once outer DER exists, unfreeze + change + re-score is refused without a logged
    deviation, and the pre-deviation results are kept."""
    synth_outer(ctx)
    cv.cmd_freeze(ctx)
    r1 = cv.cmd_score(ctx)
    old = json.loads((ctx.cv_dir / "results.json").read_text())
    assert len(cv.scored_entries(ctx)) == 1
    # restoring an accidentally deleted frozen.json with identical content needs no deviation
    ctx.frozen_path.unlink()
    assert cv.cmd_freeze(ctx)["restored_from_scoring"] == r1["created"]
    # unfreeze and change an outer prediction
    ctx.frozen_path.unlink()
    _relink(ctx, "ft_sel/F1", "train_001", 0.0)
    for fn in (cv.cmd_freeze, cv.cmd_select, cv.cmd_outer, cv.cmd_inner):
        with pytest.raises(cv.OuterLockError, match="deviation"):
            fn(ctx)
    assert not ctx.frozen_path.exists()
    # dry runs only warn
    ctx.dry_run = True
    cv.cmd_freeze(ctx)
    assert any("DRY-RUN WARNING" in l for l in ctx._lines)
    ctx.dry_run = False
    # with a logged deviation: freeze records it, score keeps both results
    ctx.deviation = "test: changed ft_sel/F1/train_001"
    fz = cv.cmd_freeze(ctx)
    assert fz["deviation"] == ctx.deviation and fz["supersedes"]["scored"] == r1["created"]
    dev = [json.loads(l) for l in (ctx.cv_dir / "deviations.jsonl").read_text().splitlines()]
    assert dev[-1]["command"] == "freeze" and dev[-1]["prior_results"] == cv.scored_entries(ctx)[0]["history"]
    r2 = cv.cmd_score(ctx)
    assert r2["deviation"] == ctx.deviation
    ents = cv.scored_entries(ctx)
    assert len(ents) == 2 and ents[0]["history"] != ents[1]["history"]
    kept = json.loads((ctx.cv_dir / ents[0]["history"]).read_text())
    assert kept == old
    assert kept["systems"]["ft_sel"]["0.0"]["corpus"]["der"] != r2["systems"]["ft_sel"]["0.0"]["corpus"]["der"]
    # a frozen.json swapped in without a deviation cannot be scored over an earlier result
    fz2 = json.loads(ctx.frozen_path.read_text())
    ctx.frozen_path.unlink()
    ctx.deviation = None
    _relink(ctx, "ft_sel/F1", "train_001", 1.0)
    fz2["systems"]["ft_sel"]["train_001"]["sha256"] = cv.file_sha(
        ctx, ctx.outer_root / "ft_sel" / "F1" / "train_001.rttm")
    fz2.pop("deviation")
    cv.write_json(ctx.frozen_path, fz2)
    with pytest.raises(cv.OuterLockError, match="records no deviation"):
        cv.cmd_score(ctx)
    assert len(cv.scored_entries(ctx)) == 2


def _stale_inner_fit(ctx):
    spec = cv.inner_fits()[0]
    fit_dir = ctx.fits_dir / spec["tag"]
    fit_dir.mkdir(parents=True)
    (fit_dir / "fit_report.json").write_text("{}")
    cv.write_marker(ctx, "fit", spec["tag"], {"x": 1}, {"model_sha256": "abc"})
    inputs = cv.infer_inputs(ctx, spec, "abc", spec["heldout_recs"], cv.THETA_ACT)
    inputs["code"] = {"infer": "OLD", "embed": "OLD"}
    cv.write_marker(ctx, "infer", cv.infer_name(spec["system"], spec["heldout_recs"], cv.THETA_ACT), inputs, {})
    return spec


def test_allow_stale_refits_inner_fit_whose_model_was_deleted(ctx):
    """Finding 3: a stale inference marker on an inner fit (model already deleted) is recoverable with
    --allow-stale: the fit marker and fit dir are moved aside (never deleted) so the fit reruns."""
    spec = _stale_inner_fit(ctx)
    model = ctx.models_dir / f"{spec['tag']}.nemo"
    # without --allow-stale: refused, nothing moved
    assert not cv.refit_if_infer_stale(ctx, spec, spec["heldout_recs"], cv.THETA_ACT)
    with pytest.raises(cv.StaleError):
        cv.ensure_infer(ctx, None, spec, model, "abc", spec["heldout_recs"], cv.THETA_ACT)
    # dry run with --allow-stale: announces, moves nothing
    ctx.allow_stale, ctx.dry_run = True, True
    assert cv.refit_if_infer_stale(ctx, spec, spec["heldout_recs"], cv.THETA_ACT)
    assert cv.read_marker(ctx, "fit", spec["tag"]) is not None
    ctx.dry_run = False
    assert cv.refit_if_infer_stale(ctx, spec, spec["heldout_recs"], cv.THETA_ACT)
    assert cv.read_marker(ctx, "fit", spec["tag"]) is None
    assert not (ctx.fits_dir / spec["tag"]).exists()
    assert len(list(ctx.fits_dir.glob(f"{spec['tag']}.stale_*"))) == 1
    assert len(list((ctx.cv_dir / "done" / "fit").glob(f"{spec['tag']}.json.stale_*"))) == 1
    # ensure_fit now plans a fresh training run, and the stale inference is redone with --force
    ctx.dry_run = True
    ctx._lines.clear()
    assert cv.ensure_fit(ctx, cv.Ledger(ctx), spec) is None
    assert any("bnd.train" in l and f"--tag {spec['tag']} " in l for l in ctx._lines)
    ctx._lines.clear()
    cv.ensure_infer(ctx, None, spec, model, "abc", spec["heldout_recs"], cv.THETA_ACT)
    assert any("bnd.infer" in l and l.endswith("--force") for l in ctx._lines)


def test_allow_stale_leaves_current_or_present_models_alone(ctx):
    spec = _stale_inner_fit(ctx)
    ctx.allow_stale = True
    model = ctx.models_dir / f"{spec['tag']}.nemo"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"m")  # model still present: inference can be redone without a refit
    assert not cv.refit_if_infer_stale(ctx, spec, spec["heldout_recs"], cv.THETA_ACT)
    assert cv.read_marker(ctx, "fit", spec["tag"]) is not None


def test_zeroshot_regenerated_under_cv_cache_never_phase2a(ctx):
    """Finding 4 / D2: the registered run never reuses runs/cache/zeroshot."""
    assert not hasattr(cv.Ctx(), "zeroshot_cache_root")
    ctx.dry_run = True
    root = cv.zeroshot_root(ctx, None)
    assert root == ctx.cache_root
    zl = [l for l in ctx._lines if "bnd.infer" in l and "--system zeroshot" in l]
    assert len(zl) == 1 and f"--cache-root {ctx.cache_root}" in zl[0]
    assert "runs/cache " not in zl[0] and not zl[0].rstrip().endswith("runs/cache")
