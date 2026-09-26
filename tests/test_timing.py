"""Tests for bnd.timing: delta estimator on synthetic shifted masks and the shifted-RTTM writer."""

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
from bnd import timing as tm


def _turns(rng: random.Random, dur: float, spk=("a", "b", "c")) -> list[nd.Turn]:
    out, t = [], 2.0
    while True:
        t += rng.randint(30, 300) / 100  # gap 0.3..3 s
        d = rng.randint(50, 800) / 100
        if t + d > dur - 2:
            return out
        out.append(nd.Turn(round(t, 2), round(t + d, 2), rng.choice(spk)))
        t += d


def _mask(turns, n):
    return tm._grid_turns(turns, n)


@pytest.mark.parametrize("true_shift", [0.37, -0.25, 0.0, 0.8])
def test_best_shift_recovers_known_shift(true_shift):
    rng = random.Random(7)
    dur = 600.0
    n = int(dur * 100)
    turns = _turns(rng, dur)
    lab = _mask(turns, n)
    # detector speech = the labels moved LATER by true_shift (labels are early by true_shift),
    # plus 3 % random frame flips
    shifted = [nd.Turn(t.start + true_shift, t.end + true_shift, t.speaker) for t in turns]
    speech = _mask(shifted, n)
    flip = np.random.default_rng(1).random(n) < 0.03
    speech = speech ^ flip
    r = tm.best_shift(lab, speech)
    assert r["shift_s"] == pytest.approx(true_shift, abs=1e-9)
    assert r["agreement"] > r["agreement_at_0"] or true_shift == 0.0


def test_agreement_curve_uses_fixed_frame_range():
    lab = np.zeros(1000, bool)
    lab[400:600] = True
    shifts, agree = tm.agreement_curve(lab, lab, 100)
    assert shifts[0] == -100 and shifts[-1] == 100 and shifts.size == 201
    assert agree[100] == 1.0  # shift 0
    # a shift of s frames mismatches exactly 2|s| frames of the 800 compared
    assert agree[110] == pytest.approx(1 - 20 / 800)
    assert agree[90] == pytest.approx(1 - 20 / 800)


def test_tie_prefers_smallest_abs_then_negative():
    lab = np.zeros(500, bool)
    speech = np.zeros(500, bool)  # everything agrees at every shift
    assert tm.best_shift(lab, speech)["shift_frames"] == 0
    assert tm.best_shift(lab, speech)["n_tied"] == 201


def test_median_delta_rounds_to_grid():
    assert tm.median_delta([0.68, 0.45, -0.21, 0.48, 0.47, 0.65]) == (pytest.approx(0.475), 0.48)
    assert tm.median_delta([0.1, 0.2, 0.3]) == (pytest.approx(0.2), 0.2)
    with pytest.raises(tm.TimingError):
        tm.median_delta([])


def test_estimate_delta_end_to_end_on_synthetic_files(tmp_path):
    """Real file path: WAV + training RTTM + Nemotron probs [T, 8] (dt~0.01) + Silero (dt 0.032)."""
    rng = random.Random(3)
    raw = tmp_path / "raw"
    (raw / "audio").mkdir(parents=True)
    rttm_dir, probs, sil = tmp_path / "rttm", tmp_path / "probs", tmp_path / "silero"
    for d in (rttm_dir, probs, sil):
        d.mkdir()
    shifts = {"train_101": 0.30, "train_102": 0.50, "train_103": 0.42}
    for rid, sh in shifts.items():
        dur = 400.0
        sf.write(raw / "audio" / f"{rid}.wav", np.zeros(int(dur * 16000), np.float32), 16000)
        turns = _turns(rng, dur)
        (rttm_dir / f"{rid}.rttm").write_text("".join(nd.rttm_line(rid, t) for t in turns))
        true = [nd.Turn(t.start + sh, t.end + sh, t.speaker) for t in turns]
        ndt = 0.0100000189
        tn = int(dur / ndt)
        pn = np.zeros((tn, 8), np.float32)
        for t in true:
            pn[int(t.start / ndt):int(t.end / ndt), 2] = 0.9
        np.savez(probs / f"{rid}.npz", probs=pn, dt=ndt)
        # silero says speech everywhere nemotron does, and also in some extra places
        ts = int(dur / 0.032)
        ps = np.full(ts, 0.1, np.float32)
        for t in true:
            ps[int(t.start / 0.032):int(t.end / 0.032) + 1] = 0.8
        ps[1000:1100] = 0.8
        np.savez(sil / f"{rid}.npz", probs=ps, dt=0.032)
    est = tm.estimate_delta(list(shifts), probs, sil, rttm_dir, raw)
    got = {p["recording_id"]: p["shift_s"] for p in est["per_recording"]}
    for rid, sh in shifts.items():
        assert abs(got[rid] - sh) <= 0.01 + 1e-9, (rid, got[rid], sh)  # one 10 ms frame (Silero is 32 ms)
    assert est["delta_s"] == pytest.approx(np.median(list(got.values())))
    # missing detector output fails loudly
    (sil / "train_103.npz").unlink()
    with pytest.raises(FileNotFoundError):
        tm.estimate_delta(list(shifts), probs, sil, rttm_dir, raw)


def test_shift_turns_exact():
    turns = [nd.Turn(10.0, 20.0, "a"),   # -> 10.48..20.0 kept
             nd.Turn(20.0, 20.5, "b"),   # -> 0.02 s dropped
             nd.Turn(21.0, 21.58, "a"),  # -> 0.10 s dropped (<= 0.1)
             nd.Turn(30.0, 30.59, "c")]  # -> 0.11 s kept
    out, dropped = tm.shift_turns(turns, 0.48, span_start=0.0)
    assert dropped == 2
    assert [(t.start, t.end, t.speaker) for t in out] == [(10.48, 20.0, "a"), (30.48, 30.59, "c")]
    # negative delta: start clipped at the span start, ends unchanged
    out, dropped = tm.shift_turns([nd.Turn(100.2, 105.0, "a")], -0.5, span_start=100.0)
    assert dropped == 0 and (out[0].start, out[0].end) == (100.0, 105.0)


def test_write_shifted_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    r1 = src / "train_101_s000.rttm"
    r1.write_text("".join(nd.rttm_line("train_101", t) for t in
                          [nd.Turn(50.0, 60.0, "1"), nd.Turn(61.0, 61.5, "2"), nd.Turn(62.0, 70.0, "3")]))
    e = {"audio_filepath": "/x/train_101.wav", "offset": 50.0, "duration": 100.0, "num_speakers": 3,
         "rttm_filepath": str(r1), "recording_id": "train_101", "span_id": "train_101_s000"}
    new, stats = tm.write_shifted_manifest([e, dict(e)], 0.48, tmp_path / "out")
    assert len(new) == 2 and new[0] == new[1]
    assert stats == {"delta_s": 0.48, "n_spans": 1, "n_turns_in": 3, "n_turns_out": 2, "n_dropped": 1}
    p = Path(new[0]["rttm_filepath"])
    assert p.parent == (tmp_path / "out").resolve() and p.name == "train_101_s000.rttm"
    got = nd.read_rttm(p)
    assert [(t.start, t.end, t.speaker) for t in got] == [(50.48, 60.0, "1"), (62.48, 70.0, "3")]
    assert new[0]["num_speakers"] == 2
    for k in ("audio_filepath", "offset", "duration", "recording_id", "span_id"):
        assert new[0][k] == e[k]
    assert r1.read_text().startswith("SPEAKER train_101 1 50.000 10.000")  # source untouched
