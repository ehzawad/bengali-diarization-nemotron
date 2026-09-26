import json

import numpy as np
import pytest

from bnd import trust as T
from bnd.trust import Row

G = T.GRID_DT


def F(t):  # seconds -> grid frame
    return int(round(t / G))


def speech_arrays(dur, speech_intervals, nemo_dt=0.01, sil_dt=0.032, nemo_on=True, sil_on=True):
    """Synthetic detector outputs: speech (p=0.9) inside intervals, 0.1 outside."""
    nn = int(round(dur / nemo_dt))
    ns = int(dur / sil_dt)  # floor, like Silero's 512-sample hops
    nemo = np.full((nn, 8), 0.1, np.float32)
    sil = np.full(ns, 0.1, np.float32)
    for s, e in speech_intervals:
        if nemo_on:
            nemo[int(round(s / nemo_dt)):int(round(e / nemo_dt)), 3] = 0.9
        if sil_on:
            sil[int(np.floor(s / sil_dt)):int(np.ceil(e / sil_dt))] = 0.9
    return nemo, sil


# ---------------------------------------------------------------- parsing
def test_local_parser_header_case_crlf(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"Start_time,End_time,Speaker_ID\r\n00:00:15,00:00:25,1\r\n\r\n01:00:02,01:00:01,2\r\n")
    expect = [Row(2, 15.0, 25.0, "1"), Row(4, 3602.0, 3601.0, "2")]
    assert T._local_parse_csv(p) == expect
    rows, name = T.load_label_rows(p)  # bnd.labels when importable; same rows either way
    assert name in ("bnd.labels.read_rows", "bnd.trust._local_parse_csv")
    assert rows == expect


def test_classify_defects():
    rows = [Row(2, 10, 10, "a"), Row(3, 30, 20, "a"), Row(4, 100, 200, "a"),
            Row(5, 120, 130, "b"), Row(6, 190, 210, "a"), Row(7, 200, 205, "c"),
            Row(8, 400, 410, "a"), Row(9, 410, 420, "a"),  # touching: not an overlap
            Row(10, 590, 605, "d")]
    kinds = T._count_kinds(T.classify_defects(rows, 600.0))
    assert kinds == {"zero_length": 1, "reversed": 1, "beyond_eof": 1,
                     "cross_speaker_overlap": 4,  # (4,5) and (6,7): both rows each
                     "same_speaker_overlap": 2}   # (4,6)
    d = T.classify_defects(rows, 600.0)
    rev = [x for x in d if x["kind"] == "reversed"][0]
    assert rev["interval"] == [20, 30]
    cross = sorted(tuple(x["interval"]) for x in d if x["kind"] == "cross_speaker_overlap")
    assert (100, 200) in cross and (120, 130) in cross


# ---------------------------------------------------------------- grid resampling
def test_resample_piecewise_constant_silero():
    dt = 0.032
    p = np.arange(10, dtype=np.float32)  # frame i covers [0.032 i, 0.032 (i+1))
    n = 32  # 0.32 s grid, source covers 0.32 s exactly
    g = T.resample_to_grid(p, dt, n)
    centres = (np.arange(n) + 0.5) * G
    assert np.array_equal(g, np.floor(centres / dt))
    # grid frame 3 (centre 0.035) is in silero frame 1; frame 2 (0.025) in frame 0
    assert g[2] == 0 and g[3] == 1


def test_resample_nemotron_nonexact_dt_is_identity():
    n = 337136
    dt = 3371.36325 / n  # dt ~= 0.01000001 as in the real probe
    p = np.random.default_rng(0).random(n).astype(np.float32)
    g = T.resample_to_grid(p, dt, T.n_grid(3371.36325))
    assert g.shape == (n,) and np.array_equal(g, p)


def test_resample_tail_tolerance_and_loud_failure():
    dur = 10.0
    n = T.n_grid(dur)
    short_by_partial_hop = np.zeros(int(dur / 0.032), np.float32)  # 312 frames = 9.984 s
    short_by_partial_hop[-1] = 7
    g = T.resample_to_grid(short_by_partial_hop, 0.032, n)
    assert g.shape == (n,) and g[-1] == 7
    with pytest.raises(ValueError):
        T.resample_to_grid(np.zeros(250, np.float32), 0.032, n)  # 8 s of 10 s
    with pytest.raises(ValueError):
        T.resample_to_grid(np.zeros(400, np.float32), 0.032, n)  # runs past the end


def test_missing_detector_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.load_detector(tmp_path / "nope.npz")


# ---------------------------------------------------------------- rules
def test_rule_a_defect_padding():
    n = T.n_grid(100)
    m = T.rule_defect([{"kind": "zero_length", "interval": [50, 50]},
                       {"kind": "cross_speaker_overlap", "interval": [0.5, 3]}], n, 1.0)
    assert np.flatnonzero(m[F(40):F(60)]).tolist() == list(range(F(49) - F(40), F(51) - F(40)))
    assert m[:F(4)].all() and not m[F(4)]  # clipped at 0, padded to 4 s
    assert m.sum() == F(2) + F(4)


def test_rule_b_labelled_silent_runs_ge_gap():
    n = T.n_grid(100)
    lab = np.zeros(n, bool)
    lab[F(10):F(40)] = True
    nemo = np.ones(n)
    sil = np.ones(n)
    nemo[F(12):F(18)] = 0.1; sil[F(12):F(18)] = 0.1   # 6 s both silent -> excluded at gap 5
    nemo[F(20):F(24)] = 0.1; sil[F(20):F(24)] = 0.1   # 4 s -> kept at gap 5
    nemo[F(30):F(38)] = 0.1                            # only one detector silent -> kept
    m = T.rule_label_silent(lab, nemo, sil, 5)
    assert m.sum() == F(6) and m[F(12):F(18)].all()
    m3 = T.rule_label_silent(lab, nemo, sil, 3)
    assert m3.sum() == F(10)
    # silence outside labels never counts
    lab2 = np.zeros(n, bool)
    assert not T.rule_label_silent(lab2, nemo, sil, 2).any()


def test_rule_c_unlabelled_speech_runs_ge_gap():
    n = T.n_grid(100)
    lab = np.zeros(n, bool)
    lab[F(50):F(60)] = True
    nemo = np.zeros(n)
    sil = np.zeros(n)
    nemo[F(10):F(17)] = 0.9; sil[F(10):F(17)] = 0.9   # 7 s unlabelled speech
    nemo[F(48):F(62)] = 0.9; sil[F(48):F(62)] = 0.9   # only 2 s either side of the label
    sil[F(70):F(90)] = 0.9                             # silero only -> not both
    m = T.rule_unlabelled_speech(lab, nemo, sil, 5)
    assert m.sum() == F(7) and m[F(10):F(17)].all()
    assert T.rule_unlabelled_speech(lab, nemo, sil, 2).sum() == F(7) + F(4)


def test_rule_d_more_than_8_speakers_in_105s():
    dur = 400
    n = T.n_grid(dur)
    turns = [Row(i, 200 + 5 * i, 202 + 5 * i, str(i)) for i in range(9)]  # 9 spk in 200..242
    m = T.rule_gt8_speakers(turns, n)
    lo, hi = np.flatnonzero(m)[[0, -1]]
    W = F(105)
    # window [w, w+W) holds speaker 0 (200-202 s) and speaker 8 (240-242 s) iff
    # F(240) - W < w <= F(202) - 1
    assert lo == F(240) - W + 1 and hi == F(202) - 1 + W - 1
    assert m.sum() == hi - lo + 1
    assert not T.rule_gt8_speakers(turns[:8], n).any()
    # 8 speakers at 0,15,...,105 s plus a 9th at 220 s: never 9 within one 105 s window
    spread = [Row(i, 15.0 * i, 15.0 * i + 1, str(i)) for i in range(8)] + [Row(9, 220, 221, "8")]
    assert not T.rule_gt8_speakers(spread, n).any()


# ---------------------------------------------------------------- spans
def test_span_edges_never_split_a_labelled_turn():
    n = T.n_grid(300)
    turns = [Row(1, 50, 70, "a"), Row(2, 70, 80, "b"), Row(3, 200, 260, "a")]
    cross = T.crossing_count(turns, n)
    assert cross[F(60)] == 1 and cross[F(70)] == 0  # touching turns: boundary 70 is cuttable
    excl = np.zeros(n, bool)
    excl[F(0):F(60)] = True     # ends inside turn 1 -> start pulled to 70
    excl[F(230):F(240)] = True  # inside turn 3 -> spans pulled back to 200 / forward to 260
    spans, edge, short = T.trusted_spans(excl, cross, 45)
    for s, e in spans:
        for r in turns:
            rs, re = F(r.start), F(r.end)
            assert not (rs < s < re) and not (rs < e < re)
    assert edge[F(60):F(70)].all() and edge[F(200):F(230)].all() and edge[F(240):F(260)].all()
    # [260,300) is 40 s < 45 s -> dropped as short
    assert spans == [(F(70), F(200))] and short[F(260):F(300)].all()


def test_min_span_filter():
    n = T.n_grid(200)
    excl = np.zeros(n, bool)
    excl[F(44.99):F(46)] = True
    excl[F(150):F(160)] = True  # leaves [160,200) = 40 s
    spans, edge, short = T.trusted_spans(excl, T.crossing_count([], n), 45)
    assert spans == [(F(46), F(150))]
    assert short[:F(44.99)].all() and short[F(160):].all() and not edge.any()
    spans2, _, _ = T.trusted_spans(excl, T.crossing_count([], n), 40)
    assert (0, F(44.99)) in spans2 and (F(160), F(200)) in spans2


def test_analyse_end_to_end_accounting_and_no_split():
    dur = 600.0
    rows = [Row(2, 20, 60, "1"), Row(3, 60, 100, "2"), Row(4, 110, 250, "1"),  # 110-250 long turn
            Row(5, 300, 300, "3"),                       # zero-length defect at 300
            Row(6, 350, 420, "1"), Row(7, 360, 365, "2"),  # cross overlap -> both rows
            Row(8, 450, 590, "2")]
    speech = [(20, 100), (110, 150), (200, 250), (350, 420), (450, 590), (10, 18)]  # 150-200 silent
    nemo, sil = speech_arrays(dur, speech)
    res = T.analyse(rows, dur, nemo, 0.01, sil, 0.032, gaps=(2, 5, 10), pad_s=1.0, min_span_s=45)
    for g, r in res.items():
        s = r["seconds"]
        assert abs(s["kept"] + s["dropped_union_excluded"] + s["dropped_edge_trim"]
                   + s["dropped_short_span"] - dur) < 1e-6
        for a, b in r["spans"]:
            assert b - a >= F(45)
            for row in T.valid_turns(rows):
                rs, re = F(row.start), F(row.end)
                assert not (rs < a < re) and not (rs < b < re)
    s5 = res[5]["seconds"]
    assert s5["dropped_by_reason_raw"]["label_silent"] == pytest.approx(50.0, abs=0.05)  # 150-200 (silero hop edge)
    assert s5["dropped_by_reason_raw"]["unlabelled_speech"] == pytest.approx(8.0)  # 10-18
    assert s5["dropped_by_reason_raw"]["defect"] == pytest.approx(2 + 72)  # 299-301, 349-421
    assert s5["dropped_by_reason_raw"]["gt8_speakers"] == 0
    assert res[10]["seconds"]["dropped_by_reason_raw"]["unlabelled_speech"] == 0
    # gap 5: excluded 10-18 (unlabelled speech), ~150-200 (inside turn 110-250), 299-301,
    # 349-421. Edges inside turn 110-250 are pulled out to 110 and 250; [0,10) is short.
    assert res[5]["spans"] == [(F(18), F(110)), (F(250), F(299)), (F(301), F(349)), (F(421), F(600))]
    # gap 10: 10-18 is no longer excluded, so the first span starts at 0
    assert res[10]["spans"][0] == (0, F(110))
    assert res[5]["defect_counts"] == {"zero_length": 1, "cross_speaker_overlap": 2}


def test_cli_writes_spans_json(tmp_path):
    import soundfile as sf
    train = tmp_path / "train"
    (train / "audio").mkdir(parents=True)
    (train / "annotation").mkdir()
    dur = 120.0
    for rid in ("r1", "r2"):
        sf.write(train / "audio" / f"{rid}.wav", np.zeros(int(dur * 16000), np.float32), 16000)
        (train / "annotation" / f"{rid}.csv").write_text(
            "start_time,end_time,speaker_id\r\n00:00:10,00:01:00,1\r\n00:01:00,00:01:50,2\r\n")
    probs, sil_dir = tmp_path / "probs", tmp_path / "silero"
    probs.mkdir(); sil_dir.mkdir()
    nemo, sil = speech_arrays(dur, [(10, 110)])
    for rid in ("r1", "r2"):
        np.savez(probs / f"{rid}.npz", probs=nemo, dt=0.01)
    np.savez(sil_dir / "r1.npz", probs=sil, dt=0.032)  # r2 has no silero
    args = ["--out", str(tmp_path / "out"), "--train-dir", str(train), "--probs-dir", str(probs),
            "--silero-dir", str(sil_dir)]
    with pytest.raises(SystemExit):
        T.main(args)
    assert T.main(args + ["--skip-missing"]) == 0
    for g in (2, 3, 5, 10):
        doc = json.loads((tmp_path / "out" / str(g) / "spans.json").read_text())
        assert list(doc["skipped"]) == ["r2"]
        assert doc["recordings"]["r1"]["spans"] == [
            {"span_id": "r1_s000", "start": 0.0, "end": 120.0, "duration": 120.0}]
        assert doc["total"]["kept_h"] == pytest.approx(120 / 3600)
    assert (tmp_path / "out" / "loss_report.json").exists()


# ---------------------------------------------------------------- review regressions
def test_missing_speaker_row_is_padded_defect_not_a_turn():
    # Real case: train_011.csv:253 `00:48:47,00:48:49,` (empty speaker). bnd.labels drops it
    # from every RTTM, so its time must be excluded, not kept as unlabelled speech.
    dur = 600.0
    rows = [Row(2, 10, 290, "1"), Row(3, 300, 302, ""), Row(4, 302, 590, "2"),
            Row(5, 350, 360, " ")]  # whitespace-only speaker also counts as missing
    d = T.classify_defects(rows, dur)
    assert T._count_kinds(d) == {"missing_speaker": 2}  # not an overlap with row 4
    assert [x["interval"] for x in d if x["lines"] == [3]] == [[300, 302]]
    assert T.valid_turns(rows) == [rows[0], rows[2]]
    nemo, sil = speech_arrays(dur, [(10, 290), (300, 590)])
    res = T.analyse(rows, dur, nemo, 0.01, sil, 0.032, gaps=(5,), pad_s=1.0, min_span_s=45)[5]
    for a, b in res["spans"]:
        assert not (a < F(302) and b > F(300)), "missing-speaker time kept inside a span"
    assert res["defect_counts"] == {"missing_speaker": 2}
    assert res["seconds"]["dropped_by_reason_raw"]["defect"] == pytest.approx(4 + 12)
    # the empty speaker never counts as a speaker for rule (d)
    many = [Row(i, 10 + 5 * i, 12 + 5 * i, str(i)) for i in range(8)] + [Row(99, 60, 61, "")]
    assert not T.rule_gt8_speakers(T.valid_turns(many), T.n_grid(dur)).any()


def test_grid_never_runs_past_eof():
    # Real durations whose round() grid overshot the WAV by < 5 ms (train_002/006/009).
    for dur in (2436.45825, 3301.169375, 2579.249375, 3371.36325, 120.0, 0.015):
        n = T.n_grid(dur)
        assert n * G <= dur + 1e-9 and (n + 1) * G > dur
    assert T.n_grid(120.0) == 12000 and T.n_grid(3371.36325) == 337136
    dur = 2436.45825
    nemo, sil = speech_arrays(dur, [])
    res = T.analyse([Row(2, 100, 200, "1")], dur, nemo, 0.01, sil, 0.032, gaps=(5,))[5]
    assert res["spans"][-1][1] * G <= dur
