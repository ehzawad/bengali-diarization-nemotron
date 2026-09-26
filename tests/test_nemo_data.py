"""Tests for bnd.nemo_data against the REAL NeMo diarization loader (CPU only).

Fast tests: span parsing, per-span RTTM writing. Tests marked ``slow`` import NeMo (~10 s) and
instantiate ``AudioToSpeechE2ESpkDiarDataset`` through the model's own dataloader factory with the
official stage-2 ``train_ds`` (subsegment_mode, 105 s, 8 speakers).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import soundfile as sf

from bnd import nemo_data as nd

REPO = Path(__file__).resolve().parents[1]
SR = 16000
N_PROBE = 200


# ------------------------------------------------------------------------------------------------
# synthetic corpus with known RTTMs
# ------------------------------------------------------------------------------------------------

def _grid(x: float) -> float:
    return round(x * 100) / 100


def _synth_turns(rng: random.Random, t0: float, t1: float, speakers: list[str]) -> list[nd.Turn]:
    """Sequential turns with gaps and ~25 % overlapping interjections, all on the 10 ms grid."""
    turns, t, prev = [], t0, None
    while True:
        spk = rng.choice([s for s in speakers if s != prev])
        dur = _grid(rng.uniform(1.5, 8.0))
        if t + dur > t1:
            break
        turns.append(nd.Turn(_grid(t), _grid(t + dur), spk))
        if rng.random() < 0.25:  # interjection by another speaker inside this turn
            other = rng.choice([s for s in speakers if s != spk])
            s = _grid(t + dur * 0.4)
            turns.append(nd.Turn(s, _grid(min(s + 1.3, t + dur + 0.6)), other))
        prev = spk
        t = _grid(t + dur + rng.uniform(0.2, 1.5))
    return turns


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    rng = random.Random(7)
    nrng = np.random.default_rng(7)
    recs = {"recA": 300.0, "recB": 200.0}
    wavs = {}
    for rid, dur in recs.items():
        wavs[rid] = root / f"{rid}.wav"
        sf.write(wavs[rid], nrng.integers(-20000, 20000, int(dur * SR), dtype=np.int16), SR, subtype="PCM_16")
    turns = {
        # 10 speakers so the loader's <=8-speaker capacity filter is exercised; OUT only outside the span;
        # S1 10-15 s crosses the span start 12.34 and must be clipped.
        "recA": [nd.Turn(2.0, 9.0, "OUT"), nd.Turn(10.0, 15.0, "S1")]
        + _synth_turns(rng, 15.5, 299.0, [f"S{i}" for i in range(10)]),
        "recB": _synth_turns(rng, 0.5, 199.5, ["X", "Y", "Z"]),
    }
    spans = [
        nd.Span("recA", "recA_s000", 12.34, 287.65, str(wavs["recA"])),
        nd.Span("recB", "recB_s000", 0.0, 95.5, str(wavs["recB"])),
        nd.Span("recB", "recB_s001", 100.25, 199.99, str(wavs["recB"])),
    ]
    stats = nd.write_manifest(spans, root / "man", turns)
    # variant: same spans but pointing at whole-recording absolute RTTMs
    whole = root / "whole"
    whole.mkdir()
    lines = []
    for sp in spans:
        r = whole / f"{sp.recording_id}.rttm"
        r.write_text("".join(nd.rttm_line(sp.recording_id, t)
                             for t in sorted(turns[sp.recording_id], key=lambda x: (x.start, x.end, x.speaker))))
        lines.append(json.dumps({"audio_filepath": sp.audio_filepath, "offset": sp.start,
                                 "duration": round(sp.duration, 2), "rttm_filepath": str(r)}))
    (whole / "manifest.json").write_text("\n".join(lines) + "\n")
    return {"root": root, "spans": spans, "turns": turns, "stats": stats, "whole_manifest": whole / "manifest.json"}


# ------------------------------------------------------------------------------------------------
# fast tests
# ------------------------------------------------------------------------------------------------

def test_load_spans_shapes(tmp_path):
    shapes = [
        [{"recording_id": "r1", "start": 0.0, "end": 50.0}, {"rid": "r1", "start_s": 60.5, "end_s": 120.0}],
        {"spans": [{"recording_id": "r1", "start": 0.0, "end": 50.0, "span_id": "a"}]},
        {"recordings": {"r1": {"spans": [{"span_id": "r1_s000", "start": 0.0, "end": 50.0, "duration": 50.0}]}}},
        {"r1": [[0.0, 50.0], [60.0, 110.0]]},
    ]
    for i, obj in enumerate(shapes):
        p = tmp_path / f"s{i}.json"
        p.write_text(json.dumps(obj))
        spans = nd.load_spans(p, raw_train=tmp_path)
        assert spans and all(s.end > s.start for s in spans)
        assert spans[0].audio_filepath.endswith("r1.wav")
    bad = tmp_path / "bad.json"
    for obj in ({"r1": [[0.0, 50.004]]}, {"r1": [[0.0, 50.0], [40.0, 90.0]]}, {"r1": [[5.0, 5.0]]}, 42):
        bad.write_text(json.dumps(obj))
        with pytest.raises(nd.NemoDataError):
            nd.load_spans(bad, raw_train=tmp_path)


def test_span_rttm_absolute_clipped_sorted(synth):
    man = [json.loads(l) for l in (synth["root"] / "man/train_manifest.json").read_text().splitlines()]
    assert [m["span_id"] for m in man] == ["recA_s000", "recB_s000", "recB_s001"]
    for m, sp in zip(man, synth["spans"]):
        assert set(m) == {"audio_filepath", "offset", "duration", "num_speakers", "rttm_filepath",
                          "recording_id", "span_id"}
        assert m["offset"] == sp.start and m["duration"] == round(sp.end - sp.start, 2)
        tt = nd.read_rttm(Path(m["rttm_filepath"]))
        assert all(sp.start - 1e-9 <= t.start < t.end <= sp.end + 1e-9 for t in tt)  # absolute, inside span
        assert tt == sorted(tt, key=lambda x: (x.start, x.end, x.speaker))
        assert m["num_speakers"] == len({t.speaker for t in tt})
    a = nd.read_rttm(Path(man[0]["rttm_filepath"]))
    assert "OUT" not in {t.speaker for t in a}
    assert a[0] == nd.Turn(12.34, 15.0, "S1")  # clipped at span start
    assert synth["stats"]["n_turns_clipped_at_span_edge"] >= 1


def test_eof_overshoot_clamped_or_rejected(tmp_path):
    wav = tmp_path / "r.wav"
    sf.write(wav, np.zeros(int(60.005 * SR), dtype=np.int16), SR, subtype="PCM_16")  # 60.005 s
    turns = {"r": [nd.Turn(1.0, 59.0, "A")]}
    st = nd.write_manifest([nd.Span("r", "r_s0", 0.0, 60.01, str(wav))], tmp_path / "o", turns)
    m = json.loads((tmp_path / "o/train_manifest.json").read_text())
    assert m["duration"] == 60.0 and st["spans_clamped_to_eof"][0]["new_end"] == 60.0
    with pytest.raises(nd.NemoDataError):
        nd.write_manifest([nd.Span("r", "r_s0", 0.0, 60.03, str(wav))], tmp_path / "o2", turns)


# ------------------------------------------------------------------------------------------------
# slow tests: the real NeMo loader
# ------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_relative_rttm_is_misread(synth):
    """The loader's own RTTM parsers treat RTTM times as absolute (offset subtracted inside)."""
    from nemo.collections.asr.data.audio_to_diar_label import (extract_frame_info_from_rttm,
                                                               get_frame_targets_from_rttm)

    sp = synth["spans"][0]
    tt, _ = nd.span_turns(sp, synth["turns"]["recA"])
    dur = round(sp.end - sp.start, 2)
    n = int(dur * 100)
    ref = nd.expected_activity(tt, sp.start, n)

    def loader_activity(lines):
        ts, names = extract_frame_info_from_rttm(sp.start, dur, lines)
        tgt = get_frame_targets_from_rttm(ts, sp.start, dur, 2, 100, max(1, len(names))).numpy() > 0.5
        return {names[c]: tgt[:, c] for c in names}

    absolute = loader_activity([nd.rttm_line("recA", t) for t in tt])
    assert set(absolute) == set(ref)
    # With a non-integer offset (12.34 s) the loader's int((t - offset) * 100) truncation can move a
    # turn edge by one 10 ms frame; every absolute-convention mismatch must be such an edge frame.
    edges = np.array(sorted(nd._edges(tt, sp.start)))
    abs_wrong = 0
    for k in ref:
        bad = np.flatnonzero(absolute[k] != ref[k])
        abs_wrong += bad.size
        assert all(np.min(np.abs(edges - f)) <= 1 for f in bad), k
    assert abs_wrong <= 2 * len(tt)
    relative = loader_activity([nd.rttm_line("recA", nd.Turn(round(t.start - sp.start, 2),
                                                               round(t.end - sp.start, 2), t.speaker)) for t in tt])
    wrong = sum(int((relative.get(k, np.zeros(n, bool)) != ref[k]).sum()) for k in ref)
    assert wrong > 1000, "a span-relative RTTM should be misaligned by the offset"


@pytest.fixture(scope="module")
def synth_ds(synth):
    ds, dl, train_ds = nd.build_real_dataset(synth["root"] / "man/train_manifest.json")
    return ds, dl, train_ds


@pytest.mark.slow
def test_stage2_params_and_pinned_model(synth_ds):
    ds, dl, train_ds = synth_ds
    got = nd.check_stage2_params(ds, train_ds)
    assert got["session_len_sec"] == 105 and got["max_spks"] == 8
    assert type(ds).__name__ == "AudioToSpeechE2ESpkDiarDataset"
    nemo = REPO / "data/weights/nemotron3/Nemotron-3-Diarization.nemo"
    if not nemo.exists():
        pytest.skip("pinned .nemo not present")
    import tarfile

    import yaml

    with tarfile.open(nemo) as tf:
        mc = yaml.safe_load(tf.extractfile("model_config.yaml"))
    keys = [k for k in mc["train_ds"] if k.startswith("subsegment_")] + ["session_len_sec", "num_spks", "soft_targets"]
    y = nd.stage2_config().model.train_ds
    for k in keys:
        assert float(mc["train_ds"][k]) == float(y[k]), k
    assert mc["output_subsampling_factor"] == 1 and mc["preprocessor"]["window_stride"] == 0.01


@pytest.mark.slow
def test_probe_synthetic(synth_ds, synth):
    ds, dl, _ = synth_ds
    st = nd.probe(ds, N_PROBE, seed=1)
    print("synthetic probe:", {k: v for k, v in st.items() if k != "per_manifest_item"})
    assert st["n_samples"] == N_PROBE
    assert st["n_manifest_items_sampled"] == 3
    ok = st["n_samples"] - st["n_fallback"]
    assert st["n_first_arrival_unique"] == ok
    assert st["n_audio_chunks_exact"] == st["n_audio_chunks"] > 0
    assert st["n_unexplained_mismatch_frames"] == 0
    # offsets 12.34 / 100.25 make int((t - offset) * 100) truncate some edges: the explained path runs
    assert st["n_boundary_offbyone_frames"] > 0
    assert st["max_read_before_span_s"] <= 1 / SR and st["max_read_after_span_s"] <= 1 / SR
    assert max(int(k) for k in st["active_channels_hist"]) <= 8
    assert st["n_two_chunk"] > 0 and st["n_one_chunk"] > 0
    assert 0.0 <= st["fallback_rate"] <= 1.0
    shapes = nd.collate_check(dl, 1)
    assert shapes[0][2] == 8


@pytest.mark.slow
def test_per_span_rttm_equals_whole_recording_rttm(synth, synth_ds):
    """Absolute convention: a whole-recording RTTM + manifest offset yields identical samples."""
    import torch

    ds_span = synth_ds[0]
    ds_whole = nd.build_real_dataset(synth["whole_manifest"])[0]
    for i in range(30):
        idx = i % 3
        out = []
        for ds in (ds_span, ds_whole):
            random.seed(100 + i)
            np.random.seed(100 + i)
            torch.manual_seed(100 + i)
            out.append(ds[idx])
        (a1, l1, t1, n1, s1), (a2, l2, t2, n2, s2) = out
        assert torch.equal(a1, a2) and torch.equal(t1, t2) and s1 == s2 and int(n1) == int(n2)
        assert "OUT" not in s1


@pytest.mark.slow
def test_probe_catches_misaligned_targets(synth_ds, monkeypatch):
    """Negative control: shift the loader's frame targets by 0.3 s and the probe must fail."""
    import torch
    import nemo.collections.asr.data.audio_to_diar_label as m

    orig = m.get_frame_targets_from_rttm

    def shifted(*a, **kw):
        return torch.roll(orig(*a, **kw), shifts=30, dims=0)

    monkeypatch.setattr(m, "get_frame_targets_from_rttm", shifted)
    with pytest.raises(nd.NemoDataError):
        nd.probe(synth_ds[0], 20, seed=3)


@pytest.mark.slow
def test_probe_real_spans(tmp_path):
    """Real Bengali-Loop audio: trusted spans if bnd.trust has produced them, else whole recordings."""
    spans_json = REPO / "data/prepared/trust/5/spans.json"
    raw = nd.RAW_TRAIN
    if not (raw / "audio").exists():
        pytest.skip("raw data not present")
    spans = nd.load_spans(spans_json) if spans_json.exists() else nd.whole_recording_spans(raw)
    turns = {}
    for rid in sorted({s.recording_id for s in spans}):
        n, sr = nd.audio_info(next(s.audio_filepath for s in spans if s.recording_id == rid))
        turns[rid] = nd.load_turns(rid, duration=n / sr)[0]
    stats = nd.write_manifest(spans, tmp_path / "man", turns)
    ds, dl, train_ds = nd.build_real_dataset(Path(stats["manifest"]))
    nd.check_stage2_params(ds, train_ds)
    st = nd.probe(ds, N_PROBE, seed=20260925)
    print("real probe (%s):" % ("trust/5" if spans_json.exists() else "whole recordings"),
          {k: v for k, v in st.items() if k != "per_manifest_item"})
    # A sample with no active speaker (fallback noise, or a real chunk drawn from a long unlabelled
    # silent stretch inside a trusted span) has no first arrival. Fallbacks are counted inside
    # n_empty_targets too, so subtracting n_empty_targets covers both.
    ok = st["n_samples"] - st["n_empty_targets"]
    assert st["n_fallback"] <= st["n_empty_targets"]
    assert st["n_samples"] == N_PROBE
    assert st["n_first_arrival_unique"] == ok
    assert st["n_audio_chunks_exact"] == st["n_audio_chunks"]
    assert st["max_read_before_span_s"] <= 1 / SR and st["max_read_after_span_s"] <= 1 / SR
    assert max(int(k) for k in st["active_channels_hist"]) <= 8


def _frame_shift(t, k: int):
    """Shift frame targets by k frames along time, zero-padding (no wrap-around)."""
    import torch

    out = torch.zeros_like(t)
    if k > 0:
        out[k:] = t[:-k]
    elif k < 0:
        out[:k] = t[-k:]
    else:
        out[:] = t
    return out


@pytest.mark.slow
@pytest.mark.parametrize("k", [-1, 1])
def test_probe_catches_one_frame_shift(synth_ds, monkeypatch, k):
    """Negative control at the finest resolution: a systematic 10 ms shift in EITHER direction must
    fail the probe. (Regression: float-truncation tolerance used to hide a -1 frame shift.)"""
    import nemo.collections.asr.data.audio_to_diar_label as m

    orig = m.get_frame_targets_from_rttm
    monkeypatch.setattr(m, "get_frame_targets_from_rttm", lambda *a, **kw: _frame_shift(orig(*a, **kw), k))
    with pytest.raises(nd.NemoDataError):
        nd.probe(synth_ds[0], 20, seed=3)


def test_truncation_reference_matches_loader_math():
    """loader_arith_activity == exact grid except the documented one-frame truncation at edges."""
    off, dur = 12.34, 60.0
    turns = [nd.Turn(13.02, 20.0, "A"), nd.Turn(15.0, 16.5, "B")]
    lines = [nd.rttm_line("r", t) for t in turns]
    n = int(dur * 100)
    got = nd.loader_arith_activity(lines, off, dur, n)
    ref = nd.expected_activity(turns, off, n)
    assert int((13.02 - off) * 100) == 67  # the float truncation itself (exact: 68)
    assert np.flatnonzero(got["A"]).min() == 67 and np.flatnonzero(ref["A"]).min() == 68
    edges = nd._edges(turns, off)
    for k in ref:
        d = np.flatnonzero(got[k] != ref[k])
        assert all(f + 1 in edges for f in d)
    assert np.array_equal(got["B"], ref["B"])  # 15.0 - 12.34 does not truncate
    # whole-second turns and offset (the real Bengali-Loop case): no truncation at all
    whole = [nd.Turn(13.0, 20.0, "A"), nd.Turn(15.0, 16.0, "B")]
    got0 = nd.loader_arith_activity([nd.rttm_line("r", t) for t in whole], 12.0, dur, n)
    ref0 = nd.expected_activity(whole, 12.0, n)
    assert all(np.array_equal(got0[k], ref0[k]) for k in ref0)


def test_balanced_manifest_evens_exposure(tmp_path):
    """Regression: one line per span gives a 2720 s span the same draws as a 48 s one under NeMo's
    uniform per-line sampling; the balanced manifest repeats lines ~duration/105 s."""
    durs = {"a_s0": 48.0, "a_s1": 143.03, "b_s0": 2720.13, "b_s1": 300.0}
    man = tmp_path / "m.json"
    man.write_text("\n".join(json.dumps({"audio_filepath": "/x.wav", "offset": 0.0, "duration": d,
                                          "num_speakers": 2, "rttm_filepath": "/x.rttm",
                                          "recording_id": k[0], "span_id": k}) for k, d in durs.items()) + "\n")
    st = nd.write_balanced_manifest(man, tmp_path / "b.json", 105.0)
    one, bal = st["one_line_per_span"], st["balanced"]
    assert one["n_lines"] == 4 and one["exposure_ratio_max_over_min"] > 25  # 1.0 vs 105/2720.13
    assert [nd.balanced_repeats(d) for d in durs.values()] == [1, 1, 26, 3]
    assert bal["n_lines"] == 31 and bal["exposure_ratio_max_over_min"] < 2.0
    lines = (tmp_path / "b.json").read_text().splitlines()
    orig = man.read_text().splitlines()
    assert set(lines) == set(orig) and sorted(lines.count(l) for l in orig) == [1, 1, 3, 26]
    assert bal["per_recording"]["b"]["draw_share"] == round(29 / 31, 4)
    dup = tmp_path / "dup.json"
    dup.write_text(orig[0] + "\n" + orig[0] + "\n")
    with pytest.raises(nd.NemoDataError):
        nd.write_balanced_manifest(dup, tmp_path / "x.json")


@pytest.mark.slow
def test_balanced_manifest_real_loader_frequency(synth, tmp_path):
    """The real loader sees one item per balanced line: draw counts per span follow the repeats."""
    import torch

    st = nd.write_balanced_manifest(synth["root"] / "man/train_manifest.json", tmp_path / "b.json", 105.0)
    ds, dl, _ = nd.build_real_dataset(tmp_path / "b.json")
    reps = [nd.balanced_repeats(round(sp.end - sp.start, 2)) for sp in synth["spans"]]
    assert reps == [3, 1, 1] and len(ds) == st["balanced"]["n_lines"] == 5
    torch.manual_seed(0)
    counts: dict[str, int] = {}
    for idx in torch.utils.data.RandomSampler(ds, num_samples=5000):  # what shuffle=True draws from
        counts[ds.collection[int(idx)].rttm_file] = counts.get(ds.collection[int(idx)].rttm_file, 0) + 1
    by_span = {Path(k).stem: v / 5000 for k, v in counts.items()}
    assert abs(by_span["recA_s000"] - 0.6) < 0.03 and abs(by_span["recB_s000"] - 0.2) < 0.03
