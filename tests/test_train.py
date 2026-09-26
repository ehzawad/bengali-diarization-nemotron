"""Tests for bnd.train (CPU only): manifest recording filter, LR schedule, trainable-set census on
the real pinned model."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from bnd import train as tr

_PREPARED_MANIFEST = Path(__file__).resolve().parents[1] / "data/prepared/manifests/train_manifest_balanced.json"
needs_prepared = pytest.mark.skipif(not _PREPARED_MANIFEST.exists(),
                                    reason="needs data/prepared (run bnd.labels, bnd.trust, bnd.nemo_data first)")


REPO = Path(__file__).resolve().parents[1]
F1_OUT = {"train_001", "train_008"}
F2_INNER = {"train_011", "train_009"}
SMOKE_RECS = ["train_013", "train_005", "train_004", "train_002", "train_003", "train_006"]


def _line(rid, k=0, **kw):
    e = {"audio_filepath": f"/x/audio/{rid}.wav", "offset": 0.0, "duration": 100.0, "num_speakers": 2,
         "rttm_filepath": f"/x/rttm/{rid}_s{k:03d}.rttm", "recording_id": rid, "span_id": f"{rid}_s{k:03d}"}
    e.update(kw)
    return e


def test_norm_recs():
    assert tr.norm_recs("013,005") == ["train_013", "train_005"]
    assert tr.norm_recs(["train_001"]) == ["train_001"]
    for bad in ("", "013,013", "007", "train_012"):
        with pytest.raises(tr.TrainError):
            tr.norm_recs(bad)


def test_filter_manifest_synthetic():
    ents = [_line("train_001"), _line("train_002"), _line("train_002", 1), _line("train_003")]
    out = tr.filter_manifest(ents, ["train_002", "train_003"])
    assert [e["span_id"] for e in out] == ["train_002_s000", "train_002_s001", "train_003_s000"]
    with pytest.raises(tr.TrainError, match="no spans"):
        tr.filter_manifest(ents, ["train_002", "train_004"])
    # a line whose recording_id disagrees with its audio file must not slip through
    with pytest.raises(tr.TrainError, match="inconsistent"):
        tr.filter_manifest([_line("train_002", audio_filepath="/x/audio/train_001.wav")], ["train_002"])
    with pytest.raises(tr.TrainError, match="inconsistent"):
        tr.filter_manifest([_line("train_002", span_id="train_001_s000")], ["train_002"])


@needs_prepared
def test_filter_real_manifest_excludes_heldout():
    ents = tr.read_manifest(tr.MANIFEST)
    out = tr.filter_manifest(ents, SMOKE_RECS)
    rids = {e["recording_id"] for e in out}
    assert rids == set(SMOKE_RECS)
    assert not rids & (F1_OUT | F2_INNER)
    assert not any(Path(e["audio_filepath"]).stem in F1_OUT | F2_INNER for e in out)
    # every kept line is byte-identical to a base line, and all base lines of those recs are kept
    assert out == [e for e in ents if e["recording_id"] in set(SMOKE_RECS)]


@needs_prepared
def test_build_fit_manifest_raw_and_shift(tmp_path):
    p, info = tr.build_fit_manifest(SMOKE_RECS, "raw", tmp_path / "raw")
    raw = tr.read_manifest(p)
    assert {e["recording_id"] for e in raw} == set(SMOKE_RECS)
    assert info["recordings"] == sorted(SMOKE_RECS)
    p2, info2 = tr.build_fit_manifest(SMOKE_RECS, "shift", tmp_path / "shift")
    sh = tr.read_manifest(p2)
    assert len(sh) == len(raw)
    assert {e["recording_id"] for e in sh} == set(SMOKE_RECS)
    # delta estimated from the training recordings only
    assert sorted(info2["delta"]["recordings"]) == sorted(SMOKE_RECS)
    assert all(Path(e["rttm_filepath"]).parent == (tmp_path / "shift" / "rttm_shift").resolve() for e in sh)
    with pytest.raises(tr.TrainError):
        tr.build_fit_manifest(SMOKE_RECS, "bogus", tmp_path / "x")


def test_lr_schedule():
    f = [tr.lr_factor(i) for i in range(tr.MAX_STEPS)]
    assert f[0] == pytest.approx(1 / 20)
    assert f[19] == pytest.approx(1.0)
    assert f[-1] == pytest.approx(0.1)
    assert all(a < b for a, b in zip(f[:20], f[1:20]))  # warmup increasing
    assert all(a >= b for a, b in zip(f[19:], f[20:]))  # cosine non-increasing
    mid = 20 + (165 - 20) / 2
    assert tr.lr_factor(int(mid) - 1) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (int(mid) - 20) / 145)))


@pytest.fixture(scope="module")
def base_model():
    from nemo.collections.asr.models import SortformerEncLabelModel

    return SortformerEncLabelModel.restore_from(str(tr.BASE_NEMO), map_location="cpu", strict=True)


def test_trainable_census_on_real_model(base_model):
    m = base_model
    c = tr.set_trainable(m)
    print("CENSUS", json.dumps(c))
    n = len(m.encoder.layers)
    assert n == 31 and c["n_encoder_layers"] == 31
    want_layers = {f"encoder.layers.{i}" for i in range(n - 4, n)}
    names = {k for k, p in m.named_parameters() if p.requires_grad}
    for k, p in m.named_parameters():
        in_scope = (any(k.startswith(l + ".") for l in want_layers) or k.startswith("encoder.final_norm.")
                    or k.startswith("sortformer_modules."))
        excluded = k.startswith("sortformer_modules.learnable_sil_emb") or k.startswith("sortformer_modules.hidden_to_spks")
        assert p.requires_grad == (in_scope and not excluded), k
    # the two frozen heads exist and are frozen
    assert any(k.startswith("sortformer_modules.learnable_sil_emb") for k, _ in m.named_parameters())
    assert any(k.startswith("sortformer_modules.hidden_to_spks") for k, _ in m.named_parameters())
    assert set(c["frozen_in_scope"]) == {"sortformer_modules.learnable_sil_emb", "sortformer_modules.hidden_to_spks"}
    assert {g for g in c["trainable_by_group"] if g.startswith("encoder.layers")} == want_layers
    assert "encoder.final_norm" in c["trainable_by_group"]
    assert c["trainable_params"] == sum(p.numel() for k, p in m.named_parameters() if k in names)
    assert c["trainable_params"] == sum(c["trainable_by_group"].values())
    # same scope as the throughput pilot (runs/pilot_b8_k4.log: 13,628,747 trainable)
    assert c["trainable_params"] == 13_628_747


def test_stream_settings_on_real_model(base_model):
    m = base_model
    tr.set_stream(m, cfg_too=False, **tr.TRAIN_STREAM)
    for k, v in tr.TRAIN_STREAM.items():
        assert getattr(m.sortformer_modules, k) == v
    tr.set_stream(m, cfg_too=True, **tr.INFER_STREAM)
    for k, v in tr.INFER_STREAM.items():
        assert getattr(m.sortformer_modules, k) == v and m.cfg.sortformer_modules[k] == v
