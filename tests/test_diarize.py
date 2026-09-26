"""CPU tests for bnd/diarize.py and bnd/final.py: config validation, CSV format, ffmpeg resampling,
fail-closed model hash, and the default-system rule (FT only if Q1 passes; joint mode, ties -> simpler).
Never writes into the real runs/."""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import soundfile as sf

from bnd import diarize as D
from bnd import final as F
from bnd import infer
from bnd import score as S

REPO = Path(__file__).resolve().parents[1]


def cfg_dict(**over) -> dict:
    d = {"session_core_s": 20, "context_s": 5.0, "theta_act": 0.4, "linker": "avg", "theta_clu": 0.6,
         "gap_fill_s": 5.0}
    d.update(over)
    return {"decoding": d, "model": {"sha256": "0" * 64, "path_rel": "nope.nemo"}}


def write_cfg(tmp_path: Path, obj: dict) -> Path:
    p = tmp_path / "final_config.json"
    p.write_text(json.dumps(obj))
    return p


def test_load_config_accepts_final_shape(tmp_path):
    c = D.load_config(write_cfg(tmp_path, cfg_dict()))
    assert c["decoding"]["linker"] == "avg"


@pytest.mark.parametrize("over", [{"linker": "vbx"}, {"context_s": 3.0}, {"gap_fill_s": -1},
                                  {"theta_act": 1.5}])
def test_load_config_rejects_bad(tmp_path, over):
    with pytest.raises(D.DiarizeError):
        D.load_config(write_cfg(tmp_path, cfg_dict(**over)))


def test_load_config_rejects_missing_key(tmp_path):
    c = cfg_dict()
    del c["decoding"]["theta_clu"]
    with pytest.raises(D.DiarizeError):
        D.load_config(write_cfg(tmp_path, c))


def test_csv_header_and_format(tmp_path):
    segs = [S.Seg(0.0, 1.23456, "SPEAKER_0"), S.Seg(2.0, 3.5, "SPEAKER_1")]
    p = tmp_path / "x.csv"
    D.write_csv(p, segs)
    rows = list(csv.reader(p.open()))
    assert rows[0] == ["start_time", "end_time", "speaker_id"]
    assert rows[1] == ["0.000", "1.235", "SPEAKER_0"] and rows[2] == ["2.000", "3.500", "SPEAKER_1"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_ffmpeg_resamples_to_16k_mono(tmp_path):
    x = (0.1 * np.sin(np.linspace(0, 2000, 44100 * 2))).astype(np.float32)
    src = tmp_path / "in.wav"
    sf.write(src, np.stack([x, x], 1), 44100)
    dst = tmp_path / "out" / "in.wav"
    D.to_wav16k(src, dst)
    y = infer.read_audio(dst)  # asserts 16 kHz mono
    assert abs(len(y) - 32000) <= 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_ffmpeg_is_lossless_on_16k_mono_pcm(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(-3000, 3000, 16000, dtype=np.int16)
    src = tmp_path / "a.wav"
    sf.write(src, x, 16000, subtype="PCM_16")
    dst = tmp_path / "b.wav"
    D.to_wav16k(src, dst)
    assert np.array_equal(sf.read(dst, dtype="int16")[0], x)


def test_diarize_fails_closed_on_model_hash(tmp_path, monkeypatch):
    m = tmp_path / "m.nemo"
    m.write_bytes(b"not the model")
    c = cfg_dict()
    c["model"]["path_rel"] = str(m)
    audio = tmp_path / "a.wav"
    sf.write(audio, np.zeros(16000, np.float32), 16000)
    with pytest.raises(D.DiarizeError, match="sha256"):
        D.diarize(audio, tmp_path / "out", write_cfg(tmp_path, c))


def _results(helps: bool, zs_cfg: str, ft_cfg: str) -> dict:
    return {"q1": {"finetuning_helps": helps},
            "default_system": {"system": "FT" if helps else "ZS",
                               "final_configuration": {"config": ft_cfg if helps else zs_cfg}}}


def _selection(zs: list[str], ft: list[str]) -> dict:
    return {"folds": {f"F{i + 1}": {"zs": {"selected": z}, "ft": {"selected": f}}
                      for i, (z, f) in enumerate(zip(zs, ft))}}


def test_decide_zs_when_q1_fails():
    zs = ["s20_avg_g5_a0.40_c0.60"] * 5
    ft = ["s20_avg_g0_a0.40_c0.55"] * 5
    d = F.decide(_results(False, zs[0], ft[0]), _selection(zs, ft))
    assert d["system"] == "ZS" and d["joint_mode"]["config"] == zs[0] and d["joint_mode"]["count"] == 5


def test_decide_ft_when_q1_passes_and_ties_go_simpler():
    zs = ["s20_avg_g5_a0.40_c0.60"] * 5
    ft = ["s80_avg_g5_a0.40_c0.60", "s80_avg_g5_a0.40_c0.60", "s20_avg_g2_a0.40_c0.60",
          "s20_avg_g2_a0.40_c0.60", "s20_refine_g0_a0.40_c0.60"]
    d = F.decide(_results(True, zs[0], "s20_avg_g2_a0.40_c0.60"), _selection(zs, ft))
    assert d["system"] == "FT" and d["joint_mode"]["config"] == "s20_avg_g2_a0.40_c0.60"


def test_decide_refuses_disagreement_with_results():
    zs = ["s20_avg_g5_a0.40_c0.60"] * 5
    with pytest.raises(F.FinalError):
        F.decide(_results(False, "s20_avg_g2_a0.40_c0.60", zs[0]), _selection(zs, zs))


def test_check_results_requires_single_matching_scoring(tmp_path):
    r = tmp_path / "results.json"
    r.write_text("{}")
    sc = tmp_path / "scored.jsonl"
    sc.write_text(json.dumps({"results_sha256": F.sha_file(r)}) + "\n")
    assert F.check_results(r, sc)["results_sha256"] == F.sha_file(r)
    sc.write_text(json.dumps({"results_sha256": "x"}) + "\n")
    with pytest.raises(F.FinalError):
        F.check_results(r, sc)
    sc.write_text((json.dumps({"results_sha256": F.sha_file(r)}) + "\n") * 2)
    with pytest.raises(F.FinalError):
        F.check_results(r, sc)


# --- output modes (release) -------------------------------------------------------------------
def test_resolve_settings_default_equals_config():
    from bnd import diarize as D
    d = {"session_core_s": 20, "gap_fill_s": 5.0}
    assert D.resolve_settings(d) == {"mode": "turns", "session_core_s": 20.0, "gap_fill_s": 5.0, "floor": False}


def test_resolve_settings_acoustic_and_benchmark_and_override():
    from bnd import diarize as D
    d = {"session_core_s": 20, "gap_fill_s": 5.0}
    assert D.resolve_settings(d, "acoustic")["gap_fill_s"] == 0.0
    b = D.resolve_settings(d, "benchmark")
    assert b["gap_fill_s"] == 5.0 and b["floor"] is True
    assert D.resolve_settings(d, "turns", 120)["session_core_s"] == 120.0


def test_resolve_settings_rejects_bad_input():
    import pytest
    from bnd import diarize as D
    d = {"session_core_s": 20, "gap_fill_s": 5.0}
    with pytest.raises(D.DiarizeError):
        D.resolve_settings(d, "fancy")
    with pytest.raises(D.DiarizeError):
        D.resolve_settings(d, "turns", 0.5)


def test_shipped_config_loads_and_matches_final_record():
    import json
    from bnd import diarize as D
    cfg = D.load_config(D.DEFAULT_CONFIG)
    assert cfg["decoding"]["gap_fill_s"] == 5.0 and cfg["decoding"]["session_core_s"] == 20
    assert cfg["model"]["sha256"] == "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d"


def test_floor_adapter_is_exclusive_and_integer():
    from bnd.link import floor_adapter
    from bnd.score import Seg
    out = floor_adapter([Seg(1.2, 3.7, "A"), Seg(3.5, 6.9, "B"), Seg(7.0, 7.4, "A")])
    assert out == [Seg(1.0, 3.0, "A"), Seg(3.0, 6.0, "B")]
    assert all(s.start == int(s.start) and s.end == int(s.end) for s in out)
