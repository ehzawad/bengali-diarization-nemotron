import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from bnd import infer as I

DT = 0.01


# ---------------------------------------------------------------- windows
@pytest.mark.parametrize("n", [1, 159, 16000 * 3, 16000 * 20, 16000 * 20 + 1, 16000 * 25,
                               16000 * 45 + 777, 16000 * 3600 + 12345])
def test_windows_partition_and_context(n):
    ws = I.make_windows(n)
    dur = n / I.SR
    assert ws[0]["core_start_s"] == 0 and ws[-1]["core_end_s"] == dur
    for a, b in zip(ws, ws[1:]):
        assert a["core_end_s"] == b["core_start_s"]
    for w in ws:
        assert w["end_s"] - w["start_s"] <= 30.0 + 1e-9
        assert w["start_s"] == max(0.0, w["core_start_s"] - 5.0)
        assert w["end_s"] == min(dur, w["core_end_s"] + 5.0)
        assert w["core_end_s"] - w["core_start_s"] <= 20.0 + 1e-9
        assert w["end_sample"] - w["start_sample"] == round((w["end_s"] - w["start_s"]) * I.SR)
    assert len(ws) == math.ceil(n / (20 * I.SR))
    assert set(I.public_windows(ws)[0]) == {"window", "start_s", "end_s", "core_start_s", "core_end_s"}


def test_windows_interior_full_length():
    ws = I.make_windows(16000 * 100)
    assert [round(w["end_s"] - w["start_s"], 6) for w in ws] == [25.0, 30.0, 30.0, 30.0, 25.0]


def test_check_windows_rejects_gap_and_overlap():
    ws = I.make_windows(16000 * 60)
    bad = [dict(w) for w in ws]
    bad[1]["core_start_s"] += 0.01
    with pytest.raises(I.InferError):
        I.check_windows(bad, 60.0)
    bad = [dict(w) for w in ws]
    bad[-1]["core_end_s"] -= 0.5
    with pytest.raises(I.InferError):
        I.check_windows(bad, 60.0)
    bad = [dict(w) for w in ws]
    bad[1]["start_s"] -= 1.0  # 31 s input
    with pytest.raises(I.InferError):
        I.check_windows(bad, 60.0)


# ---------------------------------------------------------------- tracks
def win(start=10.0, core_start=15.0, core_end=35.0, end=40.0, w=3):
    return {"window": w, "start_s": start, "end_s": end, "core_start_s": core_start, "core_end_s": core_end}


def probs_with(n, spans, base=0.05):
    p = np.full((n, 8), base, np.float32)
    for ch, a, b, v in spans:
        p[a:b, ch] = v
    return p


def test_runs_and_min_turn():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], bool)
    assert I.runs_of(m) == [(1, 3), (4, 14), (15, 16)]
    f = I.min_turn_filter(m, DT, 0.1)
    assert I.runs_of(f) == [(4, 14)]  # exactly 10 frames = 0.1 s kept, shorter dropped


def test_threshold_is_inclusive_and_segments_absolute():
    p = probs_with(3000, [(2, 100, 300, 0.5), (2, 300, 309, 0.9)])  # 0.09 s blip merges into run
    tr = I.extract_tracks(p, DT, win(), 0.5, 1e9)
    assert len(tr) == 1 and tr[0]["channel"] == 2 and tr[0]["window"] == 3
    assert tr[0]["segments"] == [[11.0, 13.09]]
    assert I.extract_tracks(p, DT, win(), 0.51, 1e9) == []  # only a 0.09 s run passes 0.51


def test_short_run_dropped_and_empty_channels_not_tracks():
    p = probs_with(3000, [(0, 100, 109, 0.9), (5, 500, 700, 0.8)])
    tr = I.extract_tracks(p, DT, win(), 0.5, 1e9)
    assert [t["channel"] for t in tr] == [5]
    assert tr[0]["track"] == 0


def test_core_segments_clipped_to_core():
    # active 12 s..20 s abs (frames 200..1000) and 30..38 s abs (2000..2800); core 15..35
    p = probs_with(3000, [(1, 200, 1000, 0.9), (1, 2000, 2800, 0.9)])
    t = I.extract_tracks(p, DT, win(), 0.5, 1e9)[0]
    assert t["segments"] == [[12.0, 20.0], [30.0, 38.0]]
    assert t["core_segments"] == [[15.0, 20.0], [30.0, 35.0]]
    assert t["active_s"] == pytest.approx(16.0) and t["core_active_s"] == pytest.approx(10.0)


def test_context_only_track_kept_with_empty_core():
    p = probs_with(3000, [(4, 0, 300, 0.9)])  # 10..13 s, before core
    t = I.extract_tracks(p, DT, win(), 0.5, 1e9)[0]
    assert t["core_segments"] == [] and t["segments"] == [[10.0, 13.0]]


def test_segments_clipped_to_recording_end():
    w = win(start=100.0, core_start=105.0, core_end=112.345, end=112.345)
    p = probs_with(1234, [(0, 1000, 1234, 0.9)])  # frames reach 112.34
    t = I.extract_tracks(p, DT, w, 0.5, 112.345)[0]
    assert t["segments"][-1][1] <= 112.345 + 1e-9


def test_usable_excludes_overlap_and_weak_flag():
    # ch0 active 0..5 s; ch1 overlaps 1..4.5 s -> ch0 usable = 0..1 + 4.5..5 = 1.5 s
    p = probs_with(3000, [(0, 0, 500, 0.9), (1, 100, 450, 0.9)])
    tr = {t["channel"]: t for t in I.extract_tracks(p, DT, win(), 0.5, 1e9)}
    assert tr[0]["usable_speech_s"] == pytest.approx(1.5)
    assert tr[0]["weak"] is False
    assert tr[1]["usable_speech_s"] == pytest.approx(0.0) and tr[1]["weak"] is True
    assert tr[1]["crops"] == []


def test_usable_threshold_depends_on_theta():
    # ch1 at 0.45 overlaps ch0: overlap counts at theta 0.4 but not at 0.5
    p = probs_with(3000, [(0, 0, 300, 0.9), (1, 0, 200, 0.45)])
    t04 = {t["channel"]: t for t in I.extract_tracks(p, DT, win(), 0.4, 1e9)}
    t05 = {t["channel"]: t for t in I.extract_tracks(p, DT, win(), 0.5, 1e9)}
    assert t04[0]["usable_speech_s"] == pytest.approx(1.0) and t04[0]["weak"] is False
    assert t05[0]["usable_speech_s"] == pytest.approx(3.0)
    assert 1 not in t05


def test_weak_boundary():
    # ch3 active 0..1 s and 5..6 s; ch0 active 0..0.99 s overlaps the first run
    p = probs_with(3000, [(0, 0, 99, 0.9), (3, 0, 100, 0.9), (3, 500, 600, 0.9)])
    tr = {t["channel"]: t for t in I.extract_tracks(p, DT, win(), 0.5, 1e9)}
    assert tr[3]["usable_speech_s"] == pytest.approx(1.0)  # 0.99..1.0 is < 0.1 s -> dropped
    assert tr[3]["weak"] is False
    q = probs_with(3000, [(2, 0, 99, 0.9)])
    assert I.extract_tracks(q, DT, win(), 0.5, 1e9)[0]["weak"] is True


# ---------------------------------------------------------------- crops
def total(piece):
    return sum(e - s for s, e in piece)


@pytest.mark.parametrize("L", [1.5, 2.0, 3.99, 4.0, 4.01, 6.0, 8.0, 9.7, 37.3])
def test_crops_1p5_to_4_and_cover_usable(L):
    usable = [[0.0, L]]
    pieces = I.plan_crops(usable)
    assert len(pieces) == math.ceil(L / 4 - 1e-9)
    for p in pieces:
        assert 1.5 - 1e-6 <= total(p) <= 4.0 + 1e-6
    assert sum(total(p) for p in pieces) == pytest.approx(L)


def test_crops_concatenate_runs_in_order_without_overlap():
    usable = [[10.0, 10.8], [12.0, 13.1], [20.0, 23.5], [30.0, 30.3]]  # 5.7 s -> 2 x 2.85 s
    pieces = I.plan_crops(usable)
    assert len(pieces) == 2
    assert [total(p) for p in pieces] == pytest.approx([2.85, 2.85])
    flat = [iv for p in pieces for iv in p]
    for (s, e), (s2, e2) in zip(flat, flat[1:]):
        assert e <= s2 + 1e-9
    covered = sum(e - s for s, e in flat)
    assert covered == pytest.approx(5.7)
    for s, e in flat:  # every crop interval lies inside a usable run
        assert any(a - 1e-9 <= s and e <= b + 1e-9 for a, b in usable)


def test_short_and_no_crop():
    assert I.plan_crops([[0.0, 0.4]]) == []
    p = I.plan_crops([[0.0, 0.3], [1.0, 1.5]])
    assert len(p) == 1 and total(p[0]) == pytest.approx(0.8)


def test_extracted_crops_are_usable_only():
    p = probs_with(3000, [(0, 0, 1000, 0.9), (1, 200, 400, 0.9)])
    t = [t for t in I.extract_tracks(p, DT, win(start=0.0, core_start=0.0), 0.5, 1e9) if t["channel"] == 0][0]
    flat = [iv for piece in t["crops"] for iv in piece]
    for s, e in flat:
        assert not (s < 4.0 and e > 2.0)  # never inside ch1's 2..4 s
    assert sum(e - s for s, e in flat) == pytest.approx(t["usable_speech_s"])


# ---------------------------------------------------------------- cache round trip
def test_tracks_files_roundtrip(tmp_path):
    n = 16000 * 50
    ws = I.make_windows(n)
    (tmp_path / "windows.json").write_text(json.dumps(I.public_windows(ws)))
    (tmp_path / "meta.json").write_text(json.dumps({"dt": DT, "recording_duration_s": n / I.SR}))
    rng = np.random.default_rng(0)
    for w in ws:
        frames = (w["end_sample"] - w["start_sample"]) // 160
        p = np.full((frames, 8), 0.02, np.float16)
        p[100:400, w["window"] % 8] = 0.8
        p[:, 7] = rng.uniform(0, 0.3, frames)
        np.save(tmp_path / f"probs_{w['window']}.npy", p)
    tr = I.write_tracks(tmp_path, 0.4)
    assert (tmp_path / "tracks_0.40.json").exists()
    assert (tmp_path / "tracks_0.4.json").resolve() == (tmp_path / "tracks_0.40.json").resolve()
    loaded = json.loads((tmp_path / "tracks_0.40.json").read_text())
    assert loaded == tr and [t["track"] for t in tr] == list(range(len(tr)))
    assert [t["window"] for t in tr] == [0, 1, 2]
    assert tr[1]["segments"] == [[16.0, 19.0]]  # window 1 starts at 15 s


def _cache_with_probs(tmp_path, n=16000 * 50):
    ws = I.make_windows(n)
    (tmp_path / "windows.json").write_text(json.dumps(I.public_windows(ws)))
    (tmp_path / "meta.json").write_text(json.dumps({"dt": DT, "recording_duration_s": n / I.SR}))
    for w in ws:
        frames = (w["end_sample"] - w["start_sample"]) // 160
        p = np.full((frames, 8), 0.02, np.float16)
        p[100:1200, 0] = 0.8   # 11 s speaker on ch0
        p[1500:1590, 1] = 0.8  # 0.9 s weak speaker on ch1
        np.save(tmp_path / f"probs_{w['window']}.npy", p)
    return ws


def _fake_emb(cache, theta, n_rows, sha):
    npy, meta = I.emb_paths(cache, theta)
    np.save(npy, np.eye(n_rows, 4, dtype=np.float32))
    meta.write_text(json.dumps({"tracks_sha256": sha}) if sha is not None else json.dumps({"dim": 4}))


def test_rewritten_tracks_invalidate_stale_embeddings(tmp_path):
    """Reviewer repro: probs change (channels reversed), tracks rewritten with the same count, old
    emb rows would be silently mis-paired.  Now the stale emb is deleted (linker fails closed)."""
    ws = _cache_with_probs(tmp_path)
    tr = I.write_tracks(tmp_path, 0.5)
    tpath = tmp_path / "tracks_0.50.json"
    npy, meta = I.emb_paths(tmp_path, 0.5)
    _fake_emb(tmp_path, 0.5, len(tr), I.sha256_bytes(tpath.read_bytes()))
    assert I.emb_is_current(tmp_path, 0.5)
    # identical probs -> identical tracks bytes -> embeddings kept
    assert I.write_tracks(tmp_path, 0.5) == tr and npy.exists() and I.emb_is_current(tmp_path, 0.5)
    # reverse channel order: same track count, different (window, channel) pairing
    for w in ws:
        f = tmp_path / f"probs_{w['window']}.npy"
        np.save(f, np.load(f)[:, ::-1].copy())
    tr2 = I.write_tracks(tmp_path, 0.5)
    assert len(tr2) == len(tr) and [t["channel"] for t in tr2] != [t["channel"] for t in tr]
    assert not npy.exists() and not meta.exists() and not I.emb_is_current(tmp_path, 0.5)


def test_legacy_or_mismatched_emb_meta_is_invalidated(tmp_path):
    _cache_with_probs(tmp_path)
    tr = I.write_tracks(tmp_path, 0.4)
    npy, meta = I.emb_paths(tmp_path, 0.4)
    _fake_emb(tmp_path, 0.4, len(tr), None)  # meta without tracks_sha256 (pre-fix cache)
    assert not I.emb_is_current(tmp_path, 0.4)
    I.write_tracks(tmp_path, 0.4)
    assert not npy.exists()
    _fake_emb(tmp_path, 0.4, len(tr), "0" * 64)  # wrong hash
    I.write_tracks(tmp_path, 0.4)
    assert not npy.exists() and not meta.exists()
    # tracks file edited behind the meta's back -> not current
    tpath = tmp_path / "tracks_0.40.json"
    _fake_emb(tmp_path, 0.4, len(tr), I.sha256_bytes(tpath.read_bytes()))
    assert I.emb_is_current(tmp_path, 0.4)
    tpath.write_text(tpath.read_text() + " ")
    assert not I.emb_is_current(tmp_path, 0.4)


def test_strong_track_in_1p0_to_1p5_band_gets_one_short_crop_and_is_counted():
    """Logged decision: strong (>= 1.0 s usable) but < 1.5 s -> stays strong, one short piece."""
    p = probs_with(3000, [(0, 0, 120, 0.9), (1, 500, 1500, 0.9), (2, 2000, 2080, 0.9)])
    tr = {t["channel"]: t for t in I.extract_tracks(p, DT, win(), 0.5, 1e9)}
    t0 = tr[0]
    assert t0["usable_speech_s"] == pytest.approx(1.2) and t0["weak"] is False
    assert len(t0["crops"]) == 1 and total(t0["crops"][0]) == pytest.approx(1.2)
    assert I.is_short_crop(t0) and not I.is_short_crop(tr[1])
    st = I.track_stats(list(tr.values()), 60.0)
    assert st["short_crop_only"] == 2 and st["strong_short_crop"] == 1  # ch2 (0.8 s) is weak


def test_fp16_probs_threshold_consistency():
    # float16 storage must not flip a 0.5 decision for values well away from it
    p = np.array([[0.4995, 0.5, 0.5005] + [0] * 5], np.float32)
    assert list(p.astype(np.float16).astype(np.float32)[0, :3] >= 0.5) == [False, True, True]


# ---------------------------------------------------------------- GPU integration (slow)
GPU = os.environ.get("BND_GPU_TESTS") == "1"


@pytest.mark.skipif(not GPU, reason="GPU test: run via scripts/gpu_run.sh with BND_GPU_TESTS=1")
def test_native_frame_counts_on_gpu():
    model = I.load_model(I.DEFAULT_MODEL)
    assert I.model_dt(model) == pytest.approx(0.01)
    wav = I.read_audio(I.audio_path("train_001"))
    hop = 160
    chunk_feats = model.sortformer_modules.chunk_len * model.sortformer_modules.subsampling_factor
    lengths = {
        "short_3.7s": int(3.7 * 16000) + 33,
        "exact_chunk": chunk_feats * hop,
        "exact_chunk_plus_1": chunk_feats * hop + 1,
        "full_30s": 30 * 16000,
        "tail_like_7.3s": int(7.3 * 16000) + 91,
    }
    for name, n in lengths.items():
        (p,) = I.native_probs(model, [wav[600 * 16000: 600 * 16000 + n]])
        assert p.shape == (n // hop, 8), name
    # batch equivalence: equal-length windows batched vs one at a time
    segs = [wav[s * 16000: s * 16000 + 30 * 16000] for s in (100, 300, 900)]
    batched = I.native_probs(model, segs)
    single = [I.native_probs(model, [s])[0] for s in segs]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(batched, single))
    print("batch_vs_single_max_abs_diff", diff)
    assert diff < 0.05


# ---------------------------------------------------------------- meta.json byte identity (--core-s 20)
PHASE2_META = I.REPO / "runs/cv/cache/zeroshot/train_001/meta.json"
PHASE2_META_KEYS = ["system", "model_sha256", "nemo_commit", "dt", "streaming", "n_channels", "batch",
                    "frame_convention", "recording_duration_s", "recording_id", "n_windows",
                    "audio_filepath", "core_s", "context_s"]


def _fake_infer_meta(tmp_path, monkeypatch, meta_base, n, audio_fp, rid="train_001"):
    monkeypatch.setattr(I, "read_audio", lambda p: np.zeros(n, np.float32))
    monkeypatch.setattr(I, "audio_path", lambda r: audio_fp)
    monkeypatch.setattr(I, "native_probs", lambda m, waves: [np.zeros((len(w) // 160, 8), np.float32) for w in waves])
    I.infer_recording(None, rid, tmp_path, meta_base, 1, 20.0)
    return (tmp_path / "meta.json").read_bytes()


def test_meta_key_order_matches_phase2_at_core_20(tmp_path, monkeypatch):
    mb = I.session_meta_base("zeroshot", "0" * 64, "abc", 0.01, {"x": 1}, 1)
    assert "core_s" not in mb
    out = json.loads(_fake_infer_meta(tmp_path, monkeypatch, mb, I.SR * 50, "/a/train_001.wav"))
    assert list(out) == PHASE2_META_KEYS and out["core_s"] == 20.0


@pytest.mark.skipif(not PHASE2_META.exists(), reason="phase-2 cache not present")
def test_meta_bytes_identical_to_phase2_cache(tmp_path, monkeypatch):
    p2 = json.loads(PHASE2_META.read_text())
    assert list(p2) == PHASE2_META_KEYS
    mb = I.session_meta_base(p2["system"], p2["model_sha256"], p2["nemo_commit"], p2["dt"], p2["streaming"], p2["batch"])
    n = int(round(p2["recording_duration_s"] * I.SR))
    got = _fake_infer_meta(tmp_path, monkeypatch, mb, n, p2["audio_filepath"], p2["recording_id"])
    assert got == PHASE2_META.read_bytes()
