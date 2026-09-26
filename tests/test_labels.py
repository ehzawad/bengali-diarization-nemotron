"""Tests for bnd.labels on synthetic CSVs (every defect class and header variant)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from bnd import labels as L

from pathlib import Path
_PREPARED_MANIFEST = Path(__file__).resolve().parents[1] / "data/prepared/manifests/train_manifest_balanced.json"
needs_prepared = pytest.mark.skipif(not _PREPARED_MANIFEST.exists(),
                                    reason="needs data/prepared (run bnd.labels, bnd.trust, bnd.nemo_data first)")



def write_csv(path, text, *, bom=False, crlf=False):
    data = text.replace("\n", "\r\n") if crlf else text
    raw = data.encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)
    return path


# ---------------------------------------------------------------- time parsing
@pytest.mark.parametrize("text,sec", [
    ("00:00:00", 0.0), ("00:01:02", 62.0), ("01:02:03", 3723.0), ("1:02:03", 3723.0),
    ("02:03", 123.0), ("75", 75.0), ("75.25", 75.25), (" 00:00:05 ", 5.0),
    ("00:00:05.5", 5.5), ("100:00:00", 360000.0),
])
def test_parse_time_ok(text, sec):
    assert L.parse_time(text) == pytest.approx(sec)


@pytest.mark.parametrize("text", ["", "  ", "ab", "00:61:00", "00:00:60", "1:2:3:4",
                                  "-5", "00:-1:00", "5s"])
def test_parse_time_bad(text):
    with pytest.raises(ValueError):
        L.parse_time(text)


# ------------------------------------------------------------- header variants
BODY = "00:00:01,00:00:03,1\n00:00:04,00:00:06,2\n"


@pytest.mark.parametrize("header", [
    "start_time,end_time,speaker_id",
    "Start_time,End_time,Speaker_ID",
    "START_TIME,END_TIME,SPEAKER_ID",
    " start_time , end_time , speaker_id ",
    "Start Time,End Time,Speaker ID",
    "start,end,speaker",
    "start-time,end-time,speaker-id",
])
@pytest.mark.parametrize("bom", [False, True])
@pytest.mark.parametrize("crlf", [False, True])
def test_header_variants(tmp_path, header, bom, crlf):
    p = write_csv(tmp_path / "x.csv", header + "\n" + BODY, bom=bom, crlf=crlf)
    rows = L.read_rows(p)
    assert [(r.line, r.start, r.end, r.speaker) for r in rows] == [
        (2, 1.0, 3.0, "1"), (3, 4.0, 6.0, "2")]


def test_column_order_independent(tmp_path):
    p = write_csv(tmp_path / "x.csv", "speaker_id,end_time,start_time\n7,00:00:09,00:00:02\n")
    (r,) = L.read_rows(p)
    assert (r.start, r.end, r.speaker) == (2.0, 9.0, "7")


def test_plain_seconds_and_no_trailing_newline(tmp_path):
    p = write_csv(tmp_path / "x.csv", "start_time,end_time,speaker_id\n1.5,3,A\n4,6.25,B")
    rows = L.read_rows(p)
    assert [(r.start, r.end, r.speaker) for r in rows] == [(1.5, 3.0, "A"), (4.0, 6.25, "B")]


def test_blank_lines_skipped_and_logged_line_numbers_physical(tmp_path):
    p = write_csv(tmp_path / "x.csv",
                  "start_time,end_time,speaker_id\n00:00:01,00:00:02,1\n\n00:00:03,00:00:04,2 \n\n",
                  crlf=True)
    log = []
    rows = L.read_rows(p, log)
    assert [r.line for r in rows] == [2, 4]
    assert rows[1].speaker == "2"
    assert any("x.csv:3: blank line" in m for m in log)
    assert any("x.csv:4: speaker id" in m for m in log)


@pytest.mark.parametrize("text,frag", [
    ("", ":1: missing header"),
    ("start_time,end_time\n1,2\n", "lacks ['speaker_id']"),
    ("start_time,end_time,speaker_id,start\n1,2,3,4\n", "duplicate header"),
    ("start_time,end_time,speaker_id\n00:00:01,00:00:02\n", "x.csv:2: expected 3 fields"),
    ("start_time,end_time,speaker_id\n00:00:01,00:00:02,1\n00:0x:01,00:00:02,1\n",
     "x.csv:3: unparseable time"),
    ("start_time,end_time,speaker_id\n", "no data rows"),
])
def test_malformed_fails_loudly(tmp_path, text, frag):
    p = write_csv(tmp_path / "x.csv", text)
    with pytest.raises(L.LabelError) as e:
        L.read_rows(p)
    assert frag in str(e.value)


# ------------------------------------------------------------ defect classes
DEFECT_CSV = """Start_time,End_time,Speaker_ID
00:00:00,00:00:10,1
00:00:10,00:00:10,2
00:00:20,00:00:15,3
00:00:30,00:00:40,4
00:00:32,00:00:35,5
00:00:36,00:00:38,4
00:00:50,00:00:52,
00:00:40,00:00:45,4
00:01:35,00:01:45,6
00:02:00,00:02:05,7
"""
# line: 2 ok | 3 zero_length | 4 reversed | 5 spk4 contains 6(spk5, cross) and 7(spk4, same)
# 8 missing_speaker | 9 touches line 5 (no overlap) | 10 beyond_eof (dur=100) | 11 entirely past EOF


@pytest.fixture
def defect_rows(tmp_path):
    p = write_csv(tmp_path / "rec.csv", DEFECT_CSV, bom=True, crlf=True)
    return L.classify(L.read_rows(p), duration=100.0)


def test_each_defect_class(defect_rows):
    got = {r.line: sorted(r.defects) for r in defect_rows}
    assert got == {
        2: [],
        3: ["zero_length"],
        4: ["reversed"],
        5: ["cross_speaker_overlap", "same_speaker_overlap"],
        6: ["cross_speaker_overlap"],
        7: ["same_speaker_overlap"],
        8: ["missing_speaker"],
        9: [],
        10: ["beyond_eof"],
        11: ["beyond_eof"],
    }
    assert set(d for r in defect_rows for d in r.defects) == set(L.DEFECT_CLASSES)
    by = {r.line: r for r in defect_rows}
    assert sorted(by[5].overlaps_with) == [6, 7]
    assert by[3].dropped and by[4].dropped and by[8].dropped
    assert not by[10].dropped and not by[5].dropped


def test_overlap_pairs(defect_rows):
    pairs = L.overlap_pairs(defect_rows)
    assert pairs["cross_speaker_overlap"] == [(5, 6, 3.0)]
    assert pairs["same_speaker_overlap"] == [(5, 7, 2.0)]


def test_dropped_rows_do_not_create_overlaps(tmp_path):
    # reversed row 00:00:09 -> 00:00:01 spans the other row if swapped; must not count
    p = write_csv(tmp_path / "x.csv",
                  "start_time,end_time,speaker_id\n00:00:02,00:00:05,1\n00:00:09,00:00:01,2\n")
    rows = L.classify(L.read_rows(p), None)
    assert [r.defects for r in rows] == [[], ["reversed"]]


def test_touching_is_not_overlap_and_no_eof_check_without_duration(tmp_path):
    p = write_csv(tmp_path / "x.csv",
                  "start_time,end_time,speaker_id\n0,5,1\n5,9,2\n9,99999,1\n")
    rows = L.classify(L.read_rows(p), None)
    assert all(r.defects == [] for r in rows)


def test_classify_idempotent(defect_rows):
    before = [list(r.defects) for r in defect_rows]
    L.classify(defect_rows, 100.0)
    assert [list(r.defects) for r in defect_rows] == before


# -------------------------------------------------------------- RTTM outputs
def parse_rttm(path):
    out = []
    for ln in path.read_text().splitlines():
        f = ln.split()
        assert f[0] == "SPEAKER" and f[2] == "1" and len(f) == 10
        out.append((f[1], float(f[3]), round(float(f[3]) + float(f[4]), 3), f[7]))
    return out


def test_scoring_reference_keeps_original_rows(defect_rows, tmp_path):
    turns = L.scoring_turns(defect_rows)
    L.write_rttm(tmp_path / "ref.rttm", "rec", turns)
    got = parse_rttm(tmp_path / "ref.rttm")
    # file order, no merging (line 5 and 7 same speaker stay separate, 5 and 9 touching stay
    # separate), beyond-EOF rows kept unclipped, only lines 3/4/8 dropped
    assert got == [("rec", 0.0, 10.0, "1"), ("rec", 30.0, 40.0, "4"), ("rec", 32.0, 35.0, "5"),
                   ("rec", 36.0, 38.0, "4"), ("rec", 40.0, 45.0, "4"),
                   ("rec", 95.0, 105.0, "6"), ("rec", 120.0, 125.0, "7")]


def test_training_rttm_merges_and_clips(defect_rows, tmp_path):
    turns = L.training_turns(defect_rows, 100.0)
    L.write_rttm(tmp_path / "tr.rttm", "rec", turns)
    got = parse_rttm(tmp_path / "tr.rttm")
    # spk4 30-40 + 36-38 (nested) + 40-45 (touching) -> 30-45; spk6 clipped to 100; spk7 gone
    assert got == [("rec", 0.0, 10.0, "1"), ("rec", 30.0, 45.0, "4"), ("rec", 32.0, 35.0, "5"),
                   ("rec", 95.0, 100.0, "6")]


def test_rttm_fractional_times_precision(tmp_path):
    L.write_rttm(tmp_path / "f.rttm", "r", [L.Turn(1.25, 3.5, "A")])
    assert tmp_path.joinpath("f.rttm").read_text() == \
        "SPEAKER r 1 1.250 2.250 <NA> <NA> A <NA> <NA>\n"


# ------------------------------------------------------------------ end to end
def test_run_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    (raw / "annotation").mkdir(parents=True)
    (raw / "audio").mkdir()
    sr = 16000
    sf.write(raw / "audio" / "recA.wav", np.zeros(100 * sr, np.int16), sr, subtype="PCM_16")
    sf.write(raw / "audio" / "recB.wav", np.zeros(20 * sr, np.int16), sr, subtype="PCM_16")
    write_csv(raw / "annotation" / "recA.csv", DEFECT_CSV, bom=True, crlf=True)
    write_csv(raw / "annotation" / "recB.csv", "start,end,speaker\n1,4,x\n4,4,y\n")
    out = tmp_path / "prepared"
    audit = L.run(out, raw, ("recA", "recB"))
    disk = json.loads((out / "label_audit.json").read_text())
    assert disk == json.loads(json.dumps(audit))
    a = disk["recordings"]["recA"]
    assert a["audio_duration_s"] == 100.0
    assert a["defect_row_counts"] == {"zero_length": 1, "reversed": 1, "missing_speaker": 1,
                                      "beyond_eof": 2, "cross_speaker_overlap": 2,
                                      "same_speaker_overlap": 2}
    assert a["overlap_pair_counts"] == {"cross_speaker_overlap": 1, "same_speaker_overlap": 1}
    assert a["n_rows"] == 10 and a["n_dropped"] == 3 and a["n_ref_rows"] == 7
    drops = {d["where"] for d in a["defects"] if d["action"] == "dropped"}
    assert drops == {"recA.csv:3", "recA.csv:4", "recA.csv:8"}
    assert disk["totals"]["defect_row_counts"]["zero_length"] == 2
    assert disk["totals"]["n_dropped"] == 4
    assert parse_rttm(out / "ref_rttm" / "recB.rttm") == [("recB", 1.0, 4.0, "x")]
    assert len(parse_rttm(out / "ref_rttm" / "recA.rttm")) == 7
    assert len(parse_rttm(out / "train_rttm" / "recA.rttm")) == 4


def test_run_missing_audio_fails(tmp_path):
    raw = tmp_path / "raw"
    (raw / "annotation").mkdir(parents=True)
    write_csv(raw / "annotation" / "recA.csv", "start,end,speaker\n1,4,x\n")
    with pytest.raises(FileNotFoundError):
        L.run(tmp_path / "out", raw, ("recA",))


# ------------------------------------------------ real data (regression guard)
# Frozen from `python -m bnd.labels` on the read-only data/raw annotations (2026-09-25).
REAL_ROWS = {  # rid: (n_rows, n_ref_rows, n_train_rows)
    "train_001": (260, 247, 246), "train_002": (309, 285, 280), "train_003": (259, 239, 239),
    "train_004": (114, 114, 114), "train_005": (274, 272, 272), "train_006": (348, 348, 347),
    "train_008": (122, 121, 121), "train_009": (332, 285, 285), "train_011": (320, 312, 310),
    "train_013": (274, 274, 274),
}
REAL_TOTALS = {"n_rows": 2612, "n_dropped": 115,
               "defect_row_counts": {"zero_length": 109, "reversed": 5, "missing_speaker": 1,
                                     "beyond_eof": 0, "cross_speaker_overlap": 41,
                                     "same_speaker_overlap": 9},
               "overlap_pair_counts": {"cross_speaker_overlap": 27, "same_speaker_overlap": 6}}
PREPARED = L.REPO / "data/prepared"
needs_raw = pytest.mark.skipif(not (L.RAW_TRAIN / "annotation").is_dir(),
                               reason=f"raw Bengali-Loop annotations absent at {L.RAW_TRAIN}")


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("real_prepared")
    return out, L.run(out)


@needs_raw
def test_real_data_counts(real_run):
    _, audit = real_run
    assert audit["totals"] == REAL_TOTALS
    got = {rid: (r["n_rows"], r["n_ref_rows"], r["n_train_rows"])
           for rid, r in audit["recordings"].items()}
    assert got == REAL_ROWS
    assert sum(n for _, _, n in REAL_ROWS.values()) == 2488
    # known raw rows (SPEC: reversed train_002.csv:133; one empty-speaker row in train_011)
    where = {d["where"]: d for r in audit["recordings"].values() for d in r["defects"]}
    assert "reversed" in where["train_002.csv:133"]["defects"]
    ms = [w for w, d in where.items() if "missing_speaker" in d["defects"]]
    assert ms == ["train_011.csv:253"] and where[ms[0]]["action"] == "dropped"


@needs_prepared
@needs_raw
def test_real_data_matches_committed_outputs(real_run):
    out, _ = real_run
    if not (PREPARED / "label_audit.json").exists():
        pytest.fail(f"committed outputs missing: {PREPARED}/label_audit.json (run python -m bnd.labels)")
    assert json.loads((out / "label_audit.json").read_text()) == \
        json.loads((PREPARED / "label_audit.json").read_text())
    for sub in ("ref_rttm", "train_rttm"):
        for rid in L.RECORDINGS:
            assert (out / sub / f"{rid}.rttm").read_text() == \
                (PREPARED / sub / f"{rid}.rttm").read_text(), f"{sub}/{rid}.rttm drifted"


@needs_raw
def test_real_ref_rttm_scores_zero_against_its_merged_self(real_run):
    """The ref RTTM keeps same-speaker overlapping rows unmerged (SPEC). pyannote counts labels
    with unique=False, so a scorer that feeds those rows straight to pyannote would charge an
    unrecoverable miss (e.g. 18 s on train_002). bnd.score must union them first: a hypothesis
    equal to the merged reference must score exactly 0 on every recording."""
    from bnd import score as S
    out, audit = real_run
    for rid in L.RECORDINGS:
        dur = audit["recordings"][rid]["audio_duration_s"]
        ref = S.load_rttm_file(out / "ref_rttm" / f"{rid}.rttm", rid)
        hyp, _ = S.union_same_label_overlaps(ref)
        r = S.score_recording(ref, hyp, dur, 0.0, rid)
        assert (r.miss, r.fa, r.conf) == (0.0, 0.0, 0.0), rid
        assert r.total == pytest.approx(sum(s.end - s.start for s in hyp)), rid
