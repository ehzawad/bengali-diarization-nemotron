import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bnd import embed as E

REPO = Path(__file__).resolve().parents[1]
EMBED_PY = REPO / "third_party/embed-venv/bin/python"


def test_crop_audio_concatenates_exact_samples():
    wav = np.arange(16000 * 10, dtype=np.float32)
    x = E.crop_audio(wav, [[1.0, 1.5], [3.0, 3.25]])
    assert len(x) == 8000 + 4000
    assert x[0] == 16000 and x[7999] == 16000 + 7999 and x[8000] == 48000
    with pytest.raises(ValueError):
        E.crop_audio(wav, [[20.0, 21.0]])


def test_combine_weighted_and_normalised():
    a, b = np.array([1.0, 0, 0]), np.array([0, 2.0, 0])
    v = E.combine([a, b], [3.0, 1.0])
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v[0] / v[1] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        E.combine([np.zeros(3)], [1.0])


def test_embed_tracks_alignment_nan_rows_and_true_lengths():
    wav = np.random.default_rng(0).standard_normal(16000 * 30).astype(np.float32)
    tracks = [
        {"track": 0, "crops": [[[0.0, 2.0]], [[5.0, 6.0], [7.0, 8.5]]]},
        {"track": 1, "crops": []},
        {"track": 2, "crops": [[[10.0, 10.7]]]},
    ]
    seen = []

    def fake(x):
        seen.append(len(x))
        return np.array([len(x), 1.0, 0.0])

    emb, st = E.embed_tracks(tracks, wav, fake, 3)
    assert seen == [32000, 40000, 11200]  # every crop at its true length, alone
    assert np.isnan(emb[1]).all() and not np.isnan(emb[[0, 2]]).any()
    assert np.linalg.norm(emb[0]) == pytest.approx(1.0)
    assert st == {"n_tracks": 3, "n_crops": 3, "nan_rows": 1}
    with pytest.raises(ValueError):
        E.embed_tracks([{"track": 5, "crops": []}], wav, fake, 3)


@pytest.mark.skipif(not EMBED_PY.exists(), reason="embed venv missing")
def test_real_wespeaker_cpu_valid_length_matters(tmp_path):
    """Real pinned WeSpeaker on CPU in the separate venv: 256-d output, and zero padding changes
    the embedding (so embedding each crop alone at true length is necessary)."""
    code = r"""
import json, numpy as np, torch
from bnd import embed as E
m = E.load_embedder('cpu'); fn = E.make_embed_fn(m, 'cpu')
import soundfile as sf
wav, sr = sf.read('data/raw/diarization/diarization/train/audio/train_001.wav', dtype='float32', start=16000*300, stop=16000*303)
a = fn(wav); b = fn(np.concatenate([wav, np.zeros(16000*2, np.float32)]))
cos = float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
print(json.dumps({"dim": int(a.shape[0]), "cos_pad": cos, "finite": bool(np.isfinite(a).all())}))
"""
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    out = subprocess.run([str(EMBED_PY), "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()[-1]
    r = json.loads(out)
    assert r["dim"] == 256 and r["finite"]
    assert r["cos_pad"] < 0.999


def test_write_emb_records_tracks_hash_after_npy(tmp_path):
    import hashlib
    emb = np.ones((2, 3), np.float32)
    E.write_emb(tmp_path, "0.50", emb, {"tracks_sha256": hashlib.sha256(b"x").hexdigest()})
    assert np.array_equal(np.load(tmp_path / "emb_0.50.npy"), emb)
    meta = json.loads((tmp_path / "emb_0.50_meta.json").read_text())
    assert meta["tracks_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert not list(tmp_path.glob("*.tmp*"))


def test_crop_policy_logs_strong_short_band_decision():
    assert "1.0<=usable<1.5" in E.CROP_POLICY["strong_short_band"]
