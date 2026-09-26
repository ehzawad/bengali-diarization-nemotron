import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bnd import embed_ft as F
from bnd.score import Seg

REPO = Path(__file__).resolve().parents[1]
EMBED_PY = REPO / "third_party/embed-venv/bin/python"


def _ref():
    # A: 0-20 (two touching rows -> one turn), 30-34; B: 12-14 (inside A's turn), 40-60 ; C: 55-70 (overlaps B)
    return [Seg(0.0, 10.0, "A"), Seg(10.0, 20.0, "A"), Seg(30.0, 34.0, "A"), Seg(12.0, 14.0, "B"),
            Seg(40.0, 60.0, "B"), Seg(55.0, 70.0, "C"), Seg(80.0, 82.2, "D")]


SPANS = [(0.0, 50.0), (52.0, 100.0)]


def test_interval_algebra():
    assert F.merge([(0, 1), (1, 2), (3, 4)]) == [(0, 2), (3, 4)]
    assert F.subtract([(0, 10)], [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]
    assert F.intersect([(0, 5), (8, 10)], [(4, 9)]) == [(4, 5), (8, 9)]


def test_eligible_regions_exact():
    reg = F.eligible_regions(_ref(), SPANS)
    # A turn 0-20 -> interior 0.5-19.5 minus B(12-14) dilated 11.5-14.5 -> 0.5-11.5, 14.5-19.5
    # A turn 30-34 -> 30.5-33.5 (3.0 s)
    assert reg["A"] == [(0.5, 11.5), (14.5, 19.5), (30.5, 33.5)]
    # B 12-14 interior 12.5-13.5 is < 2 s and inside A anyway; B 40-60 -> 40.5-59.5 minus C 54.5-70.5
    # -> 40.5-54.5, then the span gap 50-52 splits it
    assert reg["B"] == [(40.5, 50.0), (52.0, 54.5)]
    # C 55-70 -> 55.5-69.5 minus B dilated 39.5-60.5 -> 60.5-69.5
    assert reg["C"] == [(60.5, 69.5)]
    # D 80-82.2 -> interior 1.2 s < 2 s -> no class
    assert "D" not in reg


def test_sampler_constraints_and_cap(tmp_path):
    ref = _ref()
    reg = F.eligible_regions(ref, SPANS)
    classes = [F.Cls(i, "train_x", s, reg[s]) for i, s in enumerate(sorted(reg))]
    assert [c.capacity for c in classes] == [5 + 2 + 1, 4 + 1, 4]
    s = F.CropSampler(classes, batch=8, cap=5, seed=42)
    assert s.per_epoch == {0: 5, 1: 5, 2: 4}
    n_batches = 40
    for _ in range(n_batches):
        b = s.next_batch()
        assert len(b) == 8
        L = {round(c.end - c.start, 9) for c in b}
        assert len(L) == 1 and 2.0 - 1e-9 <= L.pop() <= 4.0 + 1e-9  # one length per batch, in range
        for c in b:
            F.check_crop(c, ref, SPANS, classes[c.cls].speaker)
            assert abs(c.start * 16000 - round(c.start * 16000)) < 1e-6
    # cap per epoch: every full epoch contributes exactly per_epoch crops per class
    total = n_batches * 8
    full_epochs = total // sum(s.per_epoch.values())
    for k, v in s.per_epoch.items():
        assert full_epochs * v <= s.drawn[k] <= (full_epochs + 1) * v


def test_epoch_plan_respects_cap():
    classes = [F.Cls(0, "r", "A", [(0.0, 1000.0)]), F.Cls(1, "r", "B", [(0.0, 6.0)])]
    s = F.CropSampler(classes, cap=200)
    plan = s.epoch_plan()
    assert (plan == 0).sum() == 200 and (plan == 1).sum() == 3


def test_check_crop_rejects_violations():
    ref = _ref()
    ok = F.Crop(0, "r", 1.0, 3.5)
    F.check_crop(ok, ref, SPANS, "A")
    bad = [
        ("A", F.Crop(0, "r", 0.3, 2.8)),     # too close to own turn start
        ("A", F.Crop(0, "r", 9.8, 11.8)),    # within 0.5 s of B's turn at 12
        ("A", F.Crop(0, "r", 11.0, 14.0)),   # B active
        ("B", F.Crop(1, "r", 48.0, 51.0)),   # leaves the trusted span at 50
        ("A", F.Crop(0, "r", 1.0, 5.5)),     # > 4 s
        ("A", F.Crop(0, "r", 1.0, 2.5)),     # < 2 s
        ("B", F.Crop(0, "r", 1.0, 3.5)),     # not a turn of B
    ]
    for spk, c in bad:
        with pytest.raises(F.EmbedFTError):
            F.check_crop(c, ref, SPANS, spk)


def test_class_mapping_per_recording(tmp_path):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    # same speaker label "1" in two recordings must be two classes
    (ref_dir / "train_001.rttm").write_text(
        "SPEAKER train_001 1 0.0 10.0 <NA> <NA> 1 <NA> <NA>\nSPEAKER train_001 1 20.0 10.0 <NA> <NA> 2 <NA> <NA>\n")
    (ref_dir / "train_002.rttm").write_text("SPEAKER train_002 1 0.0 10.0 <NA> <NA> 1 <NA> <NA>\n")
    (ref_dir / "train_003.rttm").write_text("SPEAKER train_003 1 0.0 10.0 <NA> <NA> 9 <NA> <NA>\n")
    spans = {"gap_s": 5, "recordings": {r: {"spans": [{"start": 0.0, "end": 100.0}]}
                                        for r in ("train_001", "train_002", "train_003")}}
    sp = tmp_path / "spans.json"
    sp.write_text(json.dumps(spans))
    cl = F.build_classes(["002", "train_001"], ref_dir, sp)
    assert [(c.cls, c.rid, c.speaker) for c in cl] == [(0, "train_001", "1"), (1, "train_001", "2"),
                                                       (2, "train_002", "1")]
    assert all(c.rid != "train_003" for c in cl)  # only the given training recordings


def test_rec_id():
    assert F.rec_id("013") == "train_013" and F.rec_id("train_005") == "train_005" and F.rec_id("4") == "train_004"


def test_never_embed_own_training(tmp_path):
    d = tmp_path / "emb"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"train_recordings": ["train_002", "train_013"]}))
    F.assert_not_trained_on(d, ["011", "train_009"])
    with pytest.raises(F.EmbedFTError):
        F.assert_not_trained_on(d, ["011", "013"])
    with pytest.raises(F.EmbedFTError):
        F.assert_not_trained_on(tmp_path, ["011"])  # no manifest -> refuse


def test_lr_factor_cosine_to_tenth():
    assert F.lr_factor(0, 1500) == pytest.approx(1.0)
    assert F.lr_factor(750, 1500) == pytest.approx(0.55)
    assert F.lr_factor(1500, 1500) == pytest.approx(0.1)


def test_pair_auc_and_labels():
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [np.nan, np.nan]], np.float32)
    r = F.pair_auc(emb, ["a", "a", "b", "b", "a"])
    assert r["auc"] == pytest.approx(1.0) and r["n_tracks"] == 4 and r["n_same"] == 2 and r["n_diff"] == 4
    tracks = [{"crops": [[[0.0, 2.0]]]}, {"crops": [[[9.0, 11.0], [11.0, 13.0]]]}, {"crops": []}]
    ref = [Seg(0.0, 10.0, "A"), Seg(10.0, 20.0, "B")]
    assert F.track_ref_labels(tracks, ref) == ["A", None, None]  # 50/50 track is impure


@pytest.mark.skipif(not EMBED_PY.exists(), reason="embed venv missing")
def test_checkpoint_round_trip_and_aam(tmp_path):
    """Real pinned WeSpeaker on CPU: perturb, save, reload through load_ft_embedder -> identical
    outputs; sha tamper is refused; AAM logits have the margin at the target only."""
    code = r"""
import json, math, sys, numpy as np, torch
from pathlib import Path
from bnd import embed as E, embed_ft as F
out = Path(sys.argv[1])
m = E.load_embedder('cpu')
with torch.no_grad():
    for p in m.parameters():
        p.add_(1e-3 * torch.randn_like(p))
m.eval()
F.save_embedder(m, out, {"tag": "t", "train_recordings": ["train_002"]})
m2 = F.load_ft_embedder(out, 'cpu', ['train_011'])
x = np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32) * 0.1
a = E.make_embed_fn(m, 'cpu')(x); b = E.make_embed_fn(m2, 'cpu')(x)
s = E.make_embed_fn(E.load_embedder('cpu'), 'cpu')(x)
man = json.loads((out / 'manifest.json').read_text())
emb = torch.randn(4, 256); W = torch.randn(3, 256); y = torch.tensor([0, 1, 2, 0])
lg = F.aam_logits(emb, W, y, 0.2, 30.0)
cos = torch.nn.functional.linear(torch.nn.functional.normalize(emb), torch.nn.functional.normalize(W))
off = (lg / 30.0 - cos).abs()
mask = torch.nn.functional.one_hot(y, 3).bool()
print(json.dumps({"maxdiff": float(np.abs(a - b).max()), "dim": int(a.shape[0]),
                  "differs_from_stock": float(np.abs(a - s).max()), "manifest_dim": man["interface"]["dim"],
                  "nontarget_equal": float(off[~mask].max()), "target_moved": float(off[mask].min())}))
"""
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    out = subprocess.run([str(EMBED_PY), "-c", code, str(tmp_path / "ck")], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    r = json.loads(out)
    assert r["maxdiff"] == 0.0 and r["dim"] == 256 and r["manifest_dim"] == 256
    assert r["differs_from_stock"] > 1e-4
    assert r["nontarget_equal"] < 1e-5 and r["target_moved"] > 1e-3
    # tamper -> refused
    ck = tmp_path / "ck" / "embedder.pt"
    ck.write_bytes(ck.read_bytes() + b"x")
    code2 = "import sys; from bnd import embed_ft as F; F.load_ft_embedder(sys.argv[1], 'cpu', ['train_011'])"
    p = subprocess.run([str(EMBED_PY), "-c", code2, str(tmp_path / "ck")], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert p.returncode != 0 and "sha256 mismatch" in p.stderr


@pytest.mark.skipif(not EMBED_PY.exists(), reason="embed venv missing")
def test_embed_cli_refuses_own_training_recording(tmp_path):
    d = tmp_path / "emb"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"train_recordings": ["train_011"], "tag": "t", "sha256": "0",
                                                 "checkpoint": "embedder.pt"}))
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    p = subprocess.run([str(EMBED_PY), "-m", "bnd.embed", "--cache-dir", str(tmp_path), "--recordings", "train_011",
                        "--theta-act", "0.40", "--device", "cpu", "--embedder-ckpt", str(d)],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "refusing to embed" in p.stderr


def test_real_regions_inside_trusted_spans():
    """On the real registered inputs: every region of every class satisfies the recipe (checked
    by sampling crops and re-validating them independently)."""
    if not (F.REF_DIR / "train_013.rttm").exists():
        pytest.skip("no reference")
    rids = ["train_013", "train_002"]
    cl = F.build_classes(rids)
    s = F.CropSampler(cl)
    refs = {r: F.load_rttm_file(F.REF_DIR / f"{r}.rttm", r) for r in rids}
    spans = {r: F.load_spans(r) for r in rids}
    for _ in range(20):
        for c in s.next_batch():
            F.check_crop(c, refs[c.rid], spans[c.rid], cl[c.cls].speaker)


# ------------------------------------------------------------------------------------------
# regression tests (review round: hold-out ids, manifest binding, per-embedder cache identity)
# ------------------------------------------------------------------------------------------
HOLDOUT = ["1_speaker/simple_1spk_015", "simple_1spk_015", "30_speaker/simple_30spk_007"]


def test_rec_id_passes_holdout_stems_through(tmp_path):
    for x in HOLDOUT:
        assert F.rec_id(x) == x
    d = tmp_path / "emb"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"train_recordings": [f"train_{i:03d}" for i in range(1, 11)]}))
    F.assert_not_trained_on(d, HOLDOUT)                       # all-10 embedder on Part B: allowed
    with pytest.raises(F.EmbedFTError):
        F.assert_not_trained_on(d, HOLDOUT + ["7"])
    real = REPO / "docs/phase5_holdout_subset.txt"
    if real.exists():
        stems = [l.strip() for l in real.read_text().splitlines() if l.strip() and not l.startswith("#")]
        F.assert_not_trained_on(d, stems)


def _bound_manifest(d, train, sha="ab" * 32, tag="t"):
    m = {"tag": tag, "train_recordings": train, "sha256": sha, "checkpoint": "embedder.pt"}
    m["train_binding"] = F.train_binding(m)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(m))
    return m


def test_manifest_binding_detects_edited_train_list(tmp_path):
    d = tmp_path / "emb"
    m = _bound_manifest(d, ["train_002", "train_013"])
    with pytest.raises(F.EmbedFTError):
        F.assert_not_trained_on(d, ["train_002"])
    for edit in ({"train_recordings": []}, {"train_recordings": ["train_013"]}, {"sha256": "cd" * 32}):
        (d / "manifest.json").write_text(json.dumps(dict(m, **edit)))
        with pytest.raises(F.EmbedFTError, match="train_binding"):
            F.assert_not_trained_on(d, ["train_002"])
        with pytest.raises(F.EmbedFTError, match="train_binding"):
            F.load_ft_embedder(d, "cpu", ["train_011"])


def test_load_ft_embedder_guards_before_touching_weights(tmp_path):
    d = tmp_path / "emb"
    _bound_manifest(d, ["train_002"])                          # no embedder.pt on disk at all
    with pytest.raises(F.EmbedFTError, match="refusing to embed"):
        F.load_ft_embedder(d, "cpu", ["002"])
    with pytest.raises(TypeError):
        F.load_ft_embedder(d, "cpu")                           # apply_to is mandatory
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text(json.dumps({"tag": "t", "train_recordings": ["train_002"],
                                                      "sha256": "0", "checkpoint": "embedder.pt"}))
    with pytest.raises(F.EmbedFTError, match="no train_binding"):
        F.load_ft_embedder(legacy, "cpu", ["train_011"])


def _meta(d, name, ck):
    m = {"model": F.E.MODEL_ID, "revision": F.E.MODEL_REV}
    if ck is not None:
        m["embedder_ckpt"] = ck
    (d / name).write_text(json.dumps(m))


def test_ft_cache_target_refuses_foreign_embeddings(tmp_path, monkeypatch):
    sha, other = "aa" * 32, "bb" * 32
    root = tmp_path / "c"
    ok = root / "r1"
    ok.mkdir(parents=True)
    F.check_ft_cache_target(root, ["r1", "missing"], ["0.40"], sha)          # empty / absent dirs: fine
    # stock embeddings present -> refuse (any meta, shared or per-theta, any theta)
    for name in ("emb_meta.json", "emb_0.40_meta.json", "emb_0.25_meta.json"):
        d = root / f"stock_{name}"
        d.mkdir()
        _meta(d, name, None)
        with pytest.raises(F.EmbedFTError, match="stock embedder"):
            F.check_ft_cache_target(root, [d.name], ["0.40"], sha)
    # another FT embedder (e.g. a different fold's fit) -> refuse
    d = root / "other"
    d.mkdir()
    _meta(d, "emb_meta.json", {"tag": "p5emb_inner_x", "sha256": other})
    with pytest.raises(F.EmbedFTError, match="p5emb_inner_x"):
        F.check_ft_cache_target(root, ["other"], ["0.40"], sha)
    # the same checkpoint (re-embedding after a tracks change / interrupted run) -> allowed
    _meta(ok, "emb_meta.json", {"tag": "t", "sha256": sha})
    _meta(ok, "emb_0.40_meta.json", {"tag": "t", "sha256": sha})
    np.save(ok / "emb_0.40.npy", np.zeros((1, 2), np.float32))
    F.check_ft_cache_target(root, ["r1"], ["0.40"], sha)
    # npy of unknown provenance -> refuse
    d = root / "orphan"
    d.mkdir()
    np.save(d / "emb_0.40.npy", np.zeros((1, 2), np.float32))
    with pytest.raises(F.EmbedFTError, match="unknown embedder"):
        F.check_ft_cache_target(root, ["orphan"], ["0.40"], sha)
    # symlinked embedding files (bnd.embed would write through them) -> refuse, even dangling ones
    d = root / "linked"
    d.mkdir()
    (d / "emb_meta.json").symlink_to(ok / "emb_meta.json")
    with pytest.raises(F.EmbedFTError, match="symlink"):
        F.check_ft_cache_target(root, ["linked"], ["0.40"], sha)
    d = root / "dangling"
    d.mkdir()
    (d / "emb_0.40.npy").symlink_to(root / "nowhere.npy")
    with pytest.raises(F.EmbedFTError, match="symlink"):
        F.check_ft_cache_target(root, ["dangling"], ["0.40"], sha)
    # hold-out stems with a folder component work as cache sub-paths
    F.check_ft_cache_target(root, ["1_speaker/simple_1spk_015"], ["0.40"], sha)


def test_ft_cache_target_only_under_runs_p5_inside_repo():
    for reg in ("runs/cache/zeroshot", "runs/cv/cache", "runs/p3b/cache/zeroshot/s20", "runs/final", "runs/diag"):
        with pytest.raises(F.EmbedFTError, match="only be written under"):
            F.check_ft_cache_target(REPO / reg, ["train_011"], ["0.40"], "aa" * 32)
    F.check_ft_cache_target(REPO / "runs/p5/cache/ft_x/s20", ["__no_such_rec__"], ["0.40"], "aa" * 32)


@pytest.mark.skipif(not EMBED_PY.exists(), reason="embed venv missing")
def test_embed_cli_ft_never_overwrites_stock_or_other_embedder(tmp_path):
    """End to end (real WeSpeaker, CPU): stock embeddings in a cache dir are NOT overwritten by
    ``--embedder-ckpt`` (refused, bytes unchanged); a second FT embedder cannot overwrite the first;
    FT into a fresh dir works (incl. a hold-out-style stem) and records the checkpoint sha256."""
    import soundfile as sf

    cache = tmp_path / "cache"
    wav = tmp_path / "a.wav"
    sf.write(wav, (0.05 * np.random.default_rng(1).standard_normal(16000 * 8)).astype(np.float32), 16000)
    tracks = [{"track": 0, "crops": [[[0.0, 2.5]]]}, {"track": 1, "crops": [[[3.0, 5.0], [5.5, 7.5]]]}]
    for rid in ("rec_a", "1_speaker/simple_1spk_015", "fresh"):
        d = cache / rid
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"audio_filepath": str(wav)}))
        (d / "tracks_0.40.json").write_text(json.dumps(tracks))
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    mk = r"""
import sys, torch
from pathlib import Path
from bnd import embed as E, embed_ft as F
for i, tag in enumerate(sys.argv[2:]):
    m = E.load_embedder('cpu')
    torch.manual_seed(i)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1e-3 * torch.randn_like(p))
    F.save_embedder(m.eval(), Path(sys.argv[1]) / tag, {"tag": tag, "train_recordings": ["train_002"]})
"""
    subprocess.run([str(EMBED_PY), "-c", mk, str(tmp_path / "ck"), "ft1", "ft2"], cwd=REPO, env=env, check=True,
                   capture_output=True, text=True)

    def run(rids, ck=None):
        cmd = [str(EMBED_PY), "-m", "bnd.embed", "--cache-dir", str(cache), "--recordings", *rids,
               "--theta-act", "0.40", "--device", "cpu"] + (["--embedder-ckpt", str(tmp_path / "ck" / ck)] if ck else [])
        return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)

    assert run(["rec_a"]).returncode == 0                                    # stock
    before = {n: (cache / "rec_a" / n).read_bytes() for n in ("emb_0.40.npy", "emb_0.40_meta.json", "emb_meta.json")}
    p = run(["rec_a"], "ft1")
    assert p.returncode != 0 and "stock embedder" in p.stderr
    assert {n: (cache / "rec_a" / n).read_bytes() for n in before} == before
    p = run(["fresh", "1_speaker/simple_1spk_015"], "ft1")                    # hold-out stem accepted
    assert p.returncode == 0, p.stderr
    m1 = json.loads((tmp_path / "ck/ft1/manifest.json").read_text())
    for rid in ("fresh", "1_speaker/simple_1spk_015"):
        em = json.loads((cache / rid / "emb_0.40_meta.json").read_text())
        assert em["embedder_ckpt"]["sha256"] == m1["sha256"]
    assert run(["fresh"], "ft1").returncode == 0                             # same embedder: re-embed ok
    snap = (cache / "fresh" / "emb_0.40.npy").read_bytes()
    p = run(["fresh"], "ft2")                                                # another embedder: refused
    assert p.returncode != 0 and "ft1" in p.stderr
    assert (cache / "fresh" / "emb_0.40.npy").read_bytes() == snap
