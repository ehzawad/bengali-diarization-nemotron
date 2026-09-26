"""Tests for bnd/score.py -- hand-computed DER cases, bootstrap, Kaggle exporter/validator."""
import csv
import io
import json
import math
from pathlib import Path

import pytest

from bnd import score as S
from bnd.score import Seg

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "raw" / "sample_submission .csv"
TEST_AUDIO = ROOT / "data" / "raw" / "diarization" / "diarization" / "test" / "audio"


def sc(ref, hyp, dur=20.0, collar=0.0):
    return S.score_recording([Seg(*r) for r in ref], [Seg(*h) for h in hyp], dur, collar, "r")


def approx(x, y):
    return math.isclose(x, y, abs_tol=1e-9)


# ------------------------------------------------------------------ DER hand-computed cases
def test_perfect_with_different_labels():
    r = sc([(0, 10, "A"), (12, 15, "B")], [(0, 10, "x"), (12, 15, "y")])
    assert approx(r.total, 13) and approx(r.errors, 0) and r.der == 0


def test_miss():
    r = sc([(0, 10, "A")], [(0, 6, "x")])
    assert approx(r.miss, 4) and approx(r.fa, 0) and approx(r.conf, 0) and approx(r.der, 0.4)


def test_false_alarm_inside_and_outside_reference_speech_counts():
    # whole-file UEM: hypothesis speech in a region with no reference at all is still false alarm
    r = sc([(0, 10, "A")], [(0, 12, "x"), (15, 18, "x")])
    assert approx(r.fa, 5) and approx(r.miss, 0) and approx(r.total, 10) and approx(r.der, 0.5)


def test_confusion_one_mapping_per_recording():
    # x covers A (10 s) and B (4 s): Hungarian maps x->A, the 4 s of B are confusion
    r = sc([(0, 10, "A"), (10, 14, "B")], [(0, 14, "x")])
    assert approx(r.conf, 4) and approx(r.miss, 0) and approx(r.fa, 0) and approx(r.der, 4 / 14)


def test_overlapping_reference_rows_count_both_speakers():
    ref = [(0, 10, "A"), (4, 6, "B")]
    r1 = sc(ref, [(0, 10, "x")])
    assert approx(r1.total, 12) and approx(r1.miss, 2) and approx(r1.conf, 0)
    r2 = sc(ref, [(0, 10, "x"), (4, 6, "y")])
    assert approx(r2.total, 12) and approx(r2.errors, 0)
    # overlap scored even when hypothesis puts the wrong speaker in: x->A, y->B, z unmapped
    r3 = sc(ref, [(0, 10, "x"), (4, 6, "y"), (4, 6, "z")])
    assert approx(r3.fa, 2)


def test_reference_beyond_eof_is_cropped_by_uem():
    r = sc([(0, 12, "A")], [(0, 10, "x")], dur=10.0)
    assert approx(r.total, 10) and approx(r.errors, 0)


def test_hypothesis_beyond_eof_fails_loudly():
    with pytest.raises(S.ScoringError, match="beyond audio duration"):
        sc([(0, 5, "A")], [(0, 11, "x")], dur=10.0)


def test_collar_is_total_width_half_each_side():
    ref = [(0, 10, "A")]
    # collar 0: 0.2 s missed
    assert approx(sc(ref, [(0.2, 10, "x")], collar=0.0).miss, 0.2)
    # collar 0.25 removes [-0.125, 0.125] and [9.875, 10.125]: miss 0.2 - 0.125 = 0.075
    r = sc(ref, [(0.2, 10, "x")], collar=0.25)
    assert approx(r.miss, 0.075) and approx(r.total, 9.75)
    # if pyannote used +-collar (not +-collar/2) this would be 0 -- it is not
    assert r.miss > 0
    # collar 0.5 removes +-0.25: the whole 0.2 s miss disappears
    r = sc(ref, [(0.2, 10, "x")], collar=0.5)
    assert approx(r.miss, 0) and approx(r.total, 9.5)
    # hypothesis ending 0.3 s late: FA 0.3 - 0.25 = 0.05 at collar 0.5
    assert approx(sc(ref, [(0, 10.3, "x")], collar=0.5).fa, 0.05)


def test_collar_on_interior_boundary_and_touching_rows_not_merged():
    # two touching same-speaker rows keep the boundary at 5 s -> it gets a collar
    r = sc([(0, 5, "A"), (5, 10, "A")], [(0, 10, "x")], collar=0.5)
    assert approx(r.total, 10 - 0.25 - 0.5 - 0.25)
    assert r.ref_unions == 0


def test_same_label_overlap_is_unioned_not_double_counted():
    r = sc([(0, 10, "A"), (5, 15, "A")], [(0, 15, "x")])
    assert approx(r.total, 15) and approx(r.errors, 0) and r.ref_unions == 1
    # raw pyannote on the un-unioned annotation double counts (why we union)
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    raw = DiarizationErrorRate()(
        S.to_annotation([Seg(0, 10, "A"), Seg(5, 15, "A")], "r"),
        S.to_annotation([Seg(0, 15, "x")], "r"),
        uem=Timeline([Segment(0, 20)]), detailed=True)
    assert approx(raw["total"], 20)


def test_zero_reference_speech_der_undefined():
    r = sc([], [(0, 1, "x")])
    with pytest.raises(S.ScoringError):
        _ = r.der


# ------------------------------------------------------------------ corpus / strictness
def _corpus():
    refs = {"a": [Seg(0, 10, "A")], "b": [Seg(0, 30, "A")]}
    hyps = {"a": [Seg(0, 5, "x")], "b": [Seg(0, 30, "x")]}
    return refs, hyps, {"a": 40.0, "b": 40.0}


def test_corpus_der_is_sum_errors_over_sum_ref():
    refs, hyps, durs = _corpus()
    res = S.score_corpus(refs, hyps, durs, collars=(0.0,))
    sm = S.corpus_summary(res[0.0])
    assert approx(sm["der"], 5 / 40)  # not the macro mean
    assert approx(sm["macro_der_mean"], (0.5 + 0.0) / 2)
    assert approx(sm["macro_der_median"], 0.25)
    assert [s.recording_id for s in res[0.0]] == ["a", "b"]
    assert set(S.score_corpus(refs, hyps, durs)) == {0.0, 0.25, 0.5}


def test_missing_prediction_fails():
    refs, hyps, durs = _corpus()
    del hyps["b"]
    with pytest.raises(S.ScoringError, match="missing predictions"):
        S.score_corpus(refs, hyps, durs)
    hyps["b"], hyps["zz"] = [], []
    with pytest.raises(S.ScoringError, match="unknown"):
        S.score_corpus(refs, hyps, durs)


def test_load_system_strict_on_files(tmp_path):
    ref, hyp = tmp_path / "ref", tmp_path / "hyp"
    S.write_rttm(ref / "a.rttm", "a", [Seg(0, 10, "A")])
    S.write_rttm(ref / "b.rttm", "b", [Seg(0, 10, "A")])
    S.write_rttm(hyp / "a.rttm", "a", [Seg(0, 10, "x")])
    with pytest.raises(S.ScoringError, match="missing prediction"):
        S.load_system(["a", "b"], ref, hyp)
    (hyp / "b.rttm").write_text("")  # empty prediction file = explicit "no speech": allowed
    refs, hyps = S.load_system(["a", "b"], ref, hyp)
    assert hyps["b"] == [] and refs["a"] == [Seg(0.0, 10.0, "A")]
    S.write_rttm(hyp / "c.rttm", "c", [Seg(0, 1, "x")])
    with pytest.raises(S.ScoringError, match="unexpected"):
        S.load_system(["a", "b"], ref, hyp)
    (hyp / "c.rttm").unlink()
    S.write_rttm(hyp / "b.rttm", "a", [Seg(0, 1, "x")])  # wrong file id inside
    with pytest.raises(S.ScoringError, match="foreign"):
        S.load_system(["a", "b"], ref, hyp)


def test_cli_der_end_to_end(tmp_path):
    import numpy as np
    import soundfile as sf
    ref, hyp, aud = tmp_path / "ref", tmp_path / "hyp", tmp_path / "aud"
    aud.mkdir()
    sf.write(aud / "a.wav", np.zeros(16000 * 20, dtype=np.int16), 16000)
    S.write_rttm(ref / "a.rttm", "a", [Seg(0, 10, "A"), Seg(10, 14, "B")])
    S.write_rttm(hyp / "a.rttm", "a", [Seg(0, 14, "x")])
    out = tmp_path / "der.json"
    assert S.main(["der", "--ref-dir", str(ref), "--hyp-dir", str(hyp), "--audio-dir", str(aud),
                   "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert approx(rep["0.0"]["corpus"]["der"], 4 / 14)
    assert approx(rep["0.0"]["per_recording"][0]["duration"], 20.0)


# ------------------------------------------------------------------ bootstrap
def _rs(rid, total, err):
    return S.RecScore(rid, 0.0, 100.0, total, err, 0.0, 0.0)


def test_bootstrap_deterministic_and_correct_point():
    a = [_rs(f"r{i}", 100 + 10 * i, 30 + i) for i in range(10)]
    b = [_rs(f"r{i}", 100 + 10 * i, 25 + (i % 3)) for i in range(10)]
    x = S.paired_bootstrap(a, b, n=5000)
    y = S.paired_bootstrap(a, b, n=5000)
    assert x == y
    assert x["seed"] == 20260925 and S.BOOTSTRAP_N == 50_000
    tot = sum(s.total for s in a)
    assert approx(x["delta_b_minus_a"], (sum(s.errors for s in b) - sum(s.errors for s in a)) / tot)
    assert x["ci_low"] <= x["delta_b_minus_a"] <= x["ci_high"] < 0
    assert x["b_wins"] == 10 and x["p_b_not_better"] == 0.0
    z = S.paired_bootstrap(a, b, n=5000, seed=1)
    assert z["ci_low"] != x["ci_low"] or z["ci_high"] != x["ci_high"]


def test_bootstrap_identical_systems_zero_and_mismatch_fails():
    a = [_rs(f"r{i}", 50 + i, 10) for i in range(5)]
    r = S.paired_bootstrap(a, a, n=1000)
    assert r["delta_b_minus_a"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 and r["ties"] == 5
    with pytest.raises(S.ScoringError):
        S.paired_bootstrap(a, a[::-1], n=10)
    with pytest.raises(S.ScoringError, match="reference total"):
        S.paired_bootstrap(a, [_rs(s.recording_id, s.total + 1, 1) for s in a], n=10)


def test_bootstrap_default_50k_runs():
    a = [_rs(f"r{i}", 100, 20 + i) for i in range(10)]
    b = [_rs(f"r{i}", 100, 20 + (-1) ** i) for i in range(10)]
    r = S.paired_bootstrap(a, b)
    assert r["n_resamples"] == 50_000 and r == S.paired_bootstrap(a, b)


# ------------------------------------------------------------------ Kaggle exporter / validator
def _sample_rows():
    with open(SAMPLE, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_renderer_reproduces_sample_file_bytes_exactly():
    rows = _sample_rows()
    order = S.read_sample_order(SAMPLE)
    assert order == [r[0] for r in rows[1:]] and len(order) == 14
    items = {r[0]: json.loads(r[1]) for r in rows[1:]}
    assert S.render_submission(order, items).encode("utf-8") == SAMPLE.read_bytes()


def test_validator_accepts_real_sample_with_real_durations():
    order = S.read_sample_order(SAMPLE)
    durs = S.durations_for(order, TEST_AUDIO)
    info = S.validate_submission_text(SAMPLE.read_bytes().decode("utf-8"), order, durs)
    assert info == {"n_files": 14, "n_segments": 70}


def test_export_round_trip():
    order = ["t1", "t2", "t3"]
    durs = {"t1": 30.0, "t2": 12.3456, "t3": 5.0}
    hyp = {
        "t1": [Seg(5, 8, "spk9"), Seg(0.12345, 4.5, "spk3"), Seg(8, 9, "spk3")],
        "t2": [Seg(0, 12.3456, "a")],  # end at a non-ms duration -> clipped to 12.345
        "t3": [],
    }
    items = {f: S.to_kaggle_items(hyp[f], durs[f]) for f in order}
    text = S.render_submission(order, items)
    assert S.validate_submission_text(text, order, durs) == {"n_files": 3, "n_segments": 4}
    back = {r[0]: json.loads(r[1]) for r in list(csv.reader(io.StringIO(text, newline="")))[1:]}
    assert back["t1"] == [
        {"start_time": 0.123, "end_time": 4.5, "speaker_id": "SPEAKER_1"},
        {"start_time": 5.0, "end_time": 8.0, "speaker_id": "SPEAKER_2"},
        {"start_time": 8.0, "end_time": 9.0, "speaker_id": "SPEAKER_1"},
    ]
    assert back["t2"] == [{"start_time": 0.0, "end_time": 12.345, "speaker_id": "SPEAKER_1"}]
    assert back["t3"] == []
    # the exported timeline scores DER 0 against the original hypothesis (up to rounding)
    rt = [Seg(i["start_time"], i["end_time"], i["speaker_id"]) for i in back["t1"]]
    assert sc([(s.start, s.end, s.speaker) for s in hyp["t1"]], [(s.start, s.end, s.speaker) for s in rt],
              dur=30.0).errors < 0.001


def test_export_overlap_error_and_flatten():
    segs = [Seg(0, 10, "A"), Seg(4, 6, "B"), Seg(9, 12, "B")]
    with pytest.raises(S.SubmissionError, match="overlapping"):
        S.to_kaggle_items(segs, 20.0)
    items = S.to_kaggle_items(segs, 20.0, overlap="flatten")
    assert items == [
        {"start_time": 0.0, "end_time": 10.0, "speaker_id": "SPEAKER_1"},
        {"start_time": 10.0, "end_time": 12.0, "speaker_id": "SPEAKER_2"},
    ]
    S.validate_submission_text(S.render_submission(["f"], {"f": items}), ["f"], {"f": 20.0})


def test_export_missing_or_extra_file_fails():
    with pytest.raises(S.SubmissionError, match="missing"):
        S.render_submission(["a", "b"], {"a": []})
    with pytest.raises(S.SubmissionError, match="not in sample"):
        S.render_submission(["a"], {"a": [], "z": []})


def _one(items, name="f"):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "diarization"])
    w.writerow([name, items if isinstance(items, str) else json.dumps(items)])
    return buf.getvalue()


GOOD = {"start_time": 1.0, "end_time": 2.0, "speaker_id": "SPEAKER_1"}


@pytest.mark.parametrize(
    "items,match",
    [
        ("not json", "not JSON"),
        (json.dumps({"a": 1}), "JSON list"),
        ([{"start_time": 1.0, "end_time": 2.0}], "exactly keys"),
        ([dict(GOOD, extra=1)], "exactly keys"),
        ([dict(GOOD, start_time=1)], "finite float"),
        ([dict(GOOD, end_time="2.0")], "finite float"),
        ([dict(GOOD, speaker_id="spk1")], "SPEAKER_k"),
        ([dict(GOOD, speaker_id=1)], "SPEAKER_k"),
        ([dict(GOOD, start_time=2.0)], "start < end"),
        ([dict(GOOD, start_time=-0.5)], "start < end"),
        ([dict(GOOD, end_time=10.5)], "duration"),
        ([dict(GOOD, start_time=3.0, end_time=4.0), GOOD], "sorted"),
        ([GOOD, dict(GOOD, start_time=1.5, end_time=3.0)], "overlaps"),
    ],
)
def test_validator_rejects_malformed(items, match):
    with pytest.raises(S.SubmissionError, match=match):
        S.validate_submission_text(_one(items), ["f"], {"f": 10.0})


def test_validator_accepts_touching_and_rejects_structure_errors():
    ok = [GOOD, dict(GOOD, start_time=2.0, end_time=3.0, speaker_id="SPEAKER_2")]
    S.validate_submission_text(_one(ok), ["f"], {"f": 10.0})
    with pytest.raises(S.SubmissionError, match="row order"):
        S.validate_submission_text(_one(ok, name="f.wav"), ["f"], {"f": 10.0})
    with pytest.raises(S.SubmissionError, match="row order"):
        S.validate_submission_text(_one(ok), ["f", "g"], {"f": 10.0, "g": 10.0})
    with pytest.raises(S.SubmissionError, match="header"):
        S.validate_submission_text("file,diarization\r\nf,[]\r\n", ["f"], {"f": 10.0})


# ------------------------------------------------------------------ review regressions
def test_rttm_round_trip_touching_rows_stay_touching(tmp_path):
    # onset+dur in float gave 0.1 + 0.2 = 0.30000000000000004 > 0.3: a false epsilon overlap
    S.write_rttm(tmp_path / "a.rttm", "a", [Seg(0.1, 0.3, "A"), Seg(0.3, 0.5, "B")])
    segs = S.load_rttm_file(tmp_path / "a.rttm", "a")
    assert segs == [Seg(0.1, 0.3, "A"), Seg(0.3, 0.5, "B")]
    assert S.to_kaggle_items(segs, 1.0) == [
        {"start_time": 0.1, "end_time": 0.3, "speaker_id": "SPEAKER_1"},
        {"start_time": 0.3, "end_time": 0.5, "speaker_id": "SPEAKER_2"},
    ]
    # many touching segments on a 10 ms grid survive write -> read -> export (default overlap='error')
    import random
    rng = random.Random(0)
    t, segs = 0, []
    for i in range(2000):
        d = rng.randint(1, 300)
        segs.append(Seg(t / 100, (t + d) / 100, f"s{i % 7}"))
        t += d
    S.write_rttm(tmp_path / "b.rttm", "b", segs)
    back = S.load_rttm_file(tmp_path / "b.rttm", "b")
    assert all(x.end <= y.start for x, y in zip(back, back[1:]))
    assert len(S.to_kaggle_items(back, t / 100 + 1)) == 2000


def test_rttm_round_trip_same_speaker_touching_keeps_collar(tmp_path):
    rows = [Seg(0.1, 0.3, "A"), Seg(0.3, 0.9, "A")]
    S.write_rttm(tmp_path / "r.rttm", "r", rows)
    rt = S.score_recording(S.load_rttm_file(tmp_path / "r.rttm", "r"), [Seg(0.1, 0.9, "x")], 2.0, 0.5)
    direct = S.score_recording(rows, [Seg(0.1, 0.9, "x")], 2.0, 0.5)
    assert rt.ref_unions == 0 and approx(rt.total, direct.total) and approx(rt.total, 0.1)


def test_export_float_epsilon_is_not_an_overlap_but_real_overlap_is():
    assert len(S.to_kaggle_items([Seg(0.1, 0.1 + 0.2, "A"), Seg(0.3, 0.5, "B")], 1.0)) == 2
    with pytest.raises(S.SubmissionError, match="overlapping"):
        S.to_kaggle_items([Seg(0.1, 0.302, "A"), Seg(0.3, 0.5, "B")], 1.0)


def test_nested_same_speaker_row_keeps_its_collar_boundaries():
    # ref A 0-10 with nested A 3-5: counted speech is unioned (no double count) but the boundaries
    # at 3 s and 5 s are original reference rows and keep their collars (SPEC item 1)
    r = sc([(0, 10, "A"), (3, 5, "A")], [(0, 3.2, "x")], collar=0.5)
    assert r.ref_unions == 1
    assert approx(r.total, 10 - 0.25 - 0.5 - 0.5 - 0.25)  # collars at 0, 3, 5, 10
    # speech 3.25-4.75 and 5.25-9.75 missed; 3.0-3.2 inside the collar at 3
    assert approx(r.miss, 1.5 + 4.5) and approx(r.fa, 0) and approx(r.conf, 0)
    # without the nested row the 3/5 s boundaries do not exist
    assert approx(sc([(0, 10, "A")], [(0, 3.2, "x")], collar=0.5).total, 9.5)
    # collar 0 is unaffected by the change
    assert approx(sc([(0, 10, "A"), (3, 5, "A")], [(0, 3.2, "x")]).total, 10)


def test_collars_match_pyannote_when_nothing_is_unioned():
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref = [Seg(0, 5, "A"), Seg(4, 9.3, "B"), Seg(9.3, 12, "A"), Seg(15, 15.1, "C")]
    hyp = [Seg(0.4, 8, "x"), Seg(7, 13, "y"), Seg(14.9, 16, "x")]
    for c in S.COLLARS:
        ours = S.score_recording(ref, hyp, 20.0, c)
        raw = DiarizationErrorRate(collar=c, skip_overlap=False)(
            S.to_annotation(ref, "r"), S.to_annotation(hyp, "r"), uem=Timeline([Segment(0, 20.0)]), detailed=True)
        assert approx(ours.total, raw["total"]) and approx(ours.errors, raw["missed detection"]
                                                           + raw["false alarm"] + raw["confusion"])


@pytest.mark.parametrize("body", [
    "start,end,speaker\n62.0,75.0,A\n",
    "SPKR train_001 1 62 13 <NA> <NA> A <NA> <NA>\n",
    "SPEAKER train_001 1 0 1 <NA> <NA> A <NA> <NA>\nSPKR train_001 1 62 13 <NA> <NA> A <NA> <NA>\n",
    "SPKR-INFO train_001 1 <NA> <NA> <NA> unknown A <NA> <NA>\n",
])
def test_non_rttm_hypothesis_fails_loudly(tmp_path, body):
    (tmp_path / "h.rttm").write_text(body)
    with pytest.raises(S.ScoringError):
        S.load_rttm_file(tmp_path / "h.rttm", "train_001")


def test_empty_or_comment_only_rttm_is_no_speech_and_other_types_ignored(tmp_path):
    (tmp_path / "e.rttm").write_text("")
    (tmp_path / "c.rttm").write_text(";; comment\n\n# note\n")
    (tmp_path / "m.rttm").write_text("SPKR-INFO f 1 <NA> <NA> <NA> unknown A <NA> <NA>\n"
                                     "SPEAKER f 1 1.5 2.25 <NA> <NA> A <NA> <NA>\n")
    assert S.load_rttm_file(tmp_path / "e.rttm", "f") == []
    assert S.load_rttm_file(tmp_path / "c.rttm", "f") == []
    assert S.load_rttm_file(tmp_path / "m.rttm", "f") == [Seg(1.5, 3.75, "A")]
