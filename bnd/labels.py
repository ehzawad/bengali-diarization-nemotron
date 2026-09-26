"""Bengali-Loop CSV labels -> audited turns and RTTMs (SPEC "Data design" item 1).

Reads the per-recording annotation CSVs robustly (UTF-8 BOM, CRLF/LF, header
case and whitespace variants, ``HH:MM:SS`` / ``MM:SS`` / plain-second times,
optional fractional seconds), classifies every row, and writes:

* scoring reference RTTM (``ref_rttm/<rid>.rttm``): the original rows as-is,
  in file order, NO merging; only rows that cannot be a segment are dropped
  (zero_length, reversed, missing_speaker), each drop logged with file:line.
* training RTTM (``train_rttm/<rid>.rttm``): the same valid rows, clipped to
  [0, audio duration], with same-speaker overlapping/touching rows merged.
* ``label_audit.json``: per-recording defect counts and every defect with
  file:line, so later stages (trust mask) can exclude those time ranges.

Row defect classes (a row can carry several):

``zero_length``            end == start                         -> dropped
``reversed``               end <  start                         -> dropped
``missing_speaker``        empty speaker field                  -> dropped
``beyond_eof``             end > audio duration (+ tolerance)   -> kept (ref), clipped (train)
``cross_speaker_overlap``  positive-duration intersection with a
                           valid row of a DIFFERENT speaker     -> kept
``same_speaker_overlap``   positive-duration intersection with a
                           valid row of the SAME speaker        -> kept (ref), merged (train)

Overlaps are only computed among valid (non-dropped) rows; touching rows
(end == next start) are not overlaps. Structurally malformed rows (wrong field
count, unparseable time) raise ``LabelError`` with file:line: failing loudly
beats silently guessing. Blank lines are skipped and logged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DROP_CLASSES = ("zero_length", "reversed", "missing_speaker")
DEFECT_CLASSES = DROP_CLASSES + ("beyond_eof", "cross_speaker_overlap", "same_speaker_overlap")
EOF_TOL_S = 0.0  # labels are whole seconds; any end past the audio is flagged

REPO = Path(__file__).resolve().parents[1]
RAW_TRAIN = REPO / "data/raw/diarization/diarization/train"
RECORDINGS = ("train_001", "train_002", "train_003", "train_004", "train_005",
              "train_006", "train_008", "train_009", "train_011", "train_013")

_HEADER_ALIASES = {
    "start": "start_time", "start_time": "start_time", "starttime": "start_time",
    "end": "end_time", "end_time": "end_time", "endtime": "end_time",
    "speaker": "speaker_id", "speaker_id": "speaker_id", "speakerid": "speaker_id",
    "spk": "speaker_id",
}
_TIME_RE = re.compile(r"^\d+(\.\d+)?$")


class LabelError(ValueError):
    """Structural problem in a label file (reported with file:line)."""


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Row:
    """One CSV data row with its provenance and defect flags."""
    line: int          # 1-based physical line in the file (header is line 1)
    start: float
    end: float
    speaker: str
    raw: str
    defects: list[str] = field(default_factory=list)
    overlaps_with: list[int] = field(default_factory=list)  # lines of overlapping rows

    @property
    def dropped(self) -> bool:
        return any(d in DROP_CLASSES for d in self.defects)

    @property
    def turn(self) -> Turn:
        return Turn(self.start, self.end, self.speaker)


def parse_time(text: str) -> float:
    """``HH:MM:SS[.f]``, ``MM:SS[.f]`` or plain seconds ``S[.f]`` -> seconds."""
    s = text.strip()
    if not s:
        raise ValueError("empty time")
    parts = s.split(":")
    if len(parts) > 3 or not all(_TIME_RE.match(p) for p in parts):
        raise ValueError(f"unparseable time {text!r}")
    vals = [float(p) for p in parts]
    # every component except the leading one must be < 60
    if any(v >= 60 for v in vals[1:]):
        raise ValueError(f"minute/second field >= 60 in {text!r}")
    total = 0.0
    for v in vals:
        total = total * 60 + v
    return total


def _norm_header(name: str) -> str:
    key = re.sub(r"[\s\-]+", "_", name.strip().lstrip("﻿").strip().lower())
    return _HEADER_ALIASES.get(key, _HEADER_ALIASES.get(key.replace("_", ""), key))


def read_rows(path: Path, log: list[str] | None = None) -> list[Row]:
    """Parse one annotation CSV into ``Row`` objects (no defect classification)."""
    path = Path(path)
    text = path.read_bytes().decode("utf-8-sig")
    lines = text.splitlines()  # handles \r\n, \n and stray \r
    if not lines or not lines[0].strip():
        raise LabelError(f"{path}:1: missing header")
    header = [_norm_header(h) for h in next(csv.reader([lines[0]]))]
    need = ("start_time", "end_time", "speaker_id")
    missing = [n for n in need if n not in header]
    if missing:
        raise LabelError(f"{path}:1: header {lines[0]!r} lacks {missing}")
    if len(set(header)) != len(header):
        raise LabelError(f"{path}:1: duplicate header columns {header}")
    idx = {n: header.index(n) for n in need}
    rows: list[Row] = []
    for lineno, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            if log is not None:
                log.append(f"{path.name}:{lineno}: blank line skipped")
            continue
        fields = next(csv.reader([raw]))
        if len(fields) != len(header):
            raise LabelError(f"{path}:{lineno}: expected {len(header)} fields, got "
                             f"{len(fields)}: {raw!r}")
        try:
            start = parse_time(fields[idx["start_time"]])
            end = parse_time(fields[idx["end_time"]])
        except ValueError as e:
            raise LabelError(f"{path}:{lineno}: {e}: {raw!r}") from None
        spk_raw = fields[idx["speaker_id"]]
        if spk_raw.strip() and spk_raw != spk_raw.strip() and log is not None:
            log.append(f"{path.name}:{lineno}: speaker id {spk_raw!r} whitespace-stripped")
        rows.append(Row(lineno, start, end, spk_raw.strip(), raw))
    if not rows:
        raise LabelError(f"{path}: no data rows")
    return rows


def classify(rows: list[Row], duration: float | None) -> list[Row]:
    """Attach defect flags in place; returns ``rows``. ``duration`` None skips EOF check."""
    for r in rows:
        r.defects.clear()
        r.overlaps_with.clear()
        if not r.speaker:
            r.defects.append("missing_speaker")
        if r.end == r.start:
            r.defects.append("zero_length")
        elif r.end < r.start:
            r.defects.append("reversed")
        if duration is not None and max(r.start, r.end) > duration + EOF_TOL_S:
            r.defects.append("beyond_eof")
    valid = sorted((r for r in rows if not r.dropped), key=lambda r: (r.start, r.end, r.line))
    # sweep: compare each row with later-starting rows until they start at/after its end
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            if b.start >= a.end:
                break
            if min(a.end, b.end) - b.start <= 0:
                continue
            kind = "same_speaker_overlap" if a.speaker == b.speaker else "cross_speaker_overlap"
            for x, y in ((a, b), (b, a)):
                if kind not in x.defects:
                    x.defects.append(kind)
                x.overlaps_with.append(y.line)
    return rows


def overlap_pairs(rows: list[Row]) -> dict[str, list[tuple[int, int, float]]]:
    """Unique overlapping (line_a, line_b, overlap_seconds) pairs by kind."""
    by_line = {r.line: r for r in rows}
    out: dict[str, list[tuple[int, int, float]]] = {"cross_speaker_overlap": [],
                                                     "same_speaker_overlap": []}
    for r in rows:
        for other in r.overlaps_with:
            if other <= r.line:
                continue
            o = by_line[other]
            ov = min(r.end, o.end) - max(r.start, o.start)
            kind = "same_speaker_overlap" if r.speaker == o.speaker else "cross_speaker_overlap"
            out[kind].append((r.line, other, ov))
    return out


def _rttm_line(rid: str, t: Turn) -> str:
    return (f"SPEAKER {rid} 1 {t.start:.3f} {t.end - t.start:.3f} <NA> <NA> "
            f"{t.speaker} <NA> <NA>\n")


def scoring_turns(rows: list[Row]) -> list[Turn]:
    """Original valid rows, file order, no merging, no clipping.

    Same-speaker overlapping rows stay separate here (SPEC: keep collar boundaries). pyannote
    counts labels with ``unique=False``, so these rows must NOT be fed to pyannote raw: the
    scorer (``bnd.score.union_same_label_overlaps``) unions positive-duration same-label overlaps
    first. Guarded by tests/test_labels.py::test_real_ref_rttm_scores_zero_against_its_merged_self.
    """
    return [r.turn for r in rows if not r.dropped]


def training_turns(rows: list[Row], duration: float | None) -> list[Turn]:
    """Valid rows clipped to [0, duration]; same-speaker overlapping/touching rows merged."""
    by_spk: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r.dropped:
            continue
        s, e = max(0.0, r.start), r.end
        if duration is not None:
            e = min(e, duration)
        if e <= s:
            continue
        by_spk.setdefault(r.speaker, []).append((s, e))
    out: list[Turn] = []
    for spk, ivs in by_spk.items():
        ivs.sort()
        cs, ce = ivs[0]
        for s, e in ivs[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                out.append(Turn(cs, ce, spk))
                cs, ce = s, e
        out.append(Turn(cs, ce, spk))
    out.sort(key=lambda t: (t.start, t.end, t.speaker))
    return out


def write_rttm(path: Path, rid: str, turns: list[Turn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(_rttm_line(rid, t) for t in turns))
    tmp.replace(path)


def audio_duration(wav: Path) -> float:
    import soundfile as sf
    info = sf.info(str(wav))
    return info.frames / info.samplerate


def audit_recording(rid: str, csv_path: Path, duration: float | None) -> tuple[list[Row], dict]:
    log: list[str] = []
    rows = classify(read_rows(csv_path, log), duration)
    name = Path(csv_path).name
    counts = Counter(d for r in rows for d in r.defects)
    pairs = overlap_pairs(rows)
    defects = [{"where": f"{name}:{r.line}", "line": r.line, "start": r.start, "end": r.end,
                "speaker": r.speaker, "defects": list(r.defects),
                "overlaps_with_lines": sorted(r.overlaps_with),
                "action": "dropped" if r.dropped else "kept", "raw": r.raw}
               for r in rows if r.defects]
    ref = scoring_turns(rows)
    rec = {
        "recording_id": rid,
        "csv": str(csv_path),
        "audio_duration_s": duration,
        "n_rows": len(rows),
        "n_ref_rows": len(ref),
        "n_dropped": sum(r.dropped for r in rows),
        "speakers": sorted({t.speaker for t in ref}, key=lambda s: (len(s), s)),
        "defect_row_counts": {k: counts.get(k, 0) for k in DEFECT_CLASSES},
        "overlap_pair_counts": {k: len(v) for k, v in pairs.items()},
        "overlap_pairs": {k: [{"lines": [a, b], "overlap_s": ov} for a, b, ov in v]
                          for k, v in pairs.items()},
        "log": log,
        "defects": defects,
    }
    return rows, rec


def run(out: Path, raw_train: Path = RAW_TRAIN, recordings=RECORDINGS) -> dict:
    out = Path(out)
    recs = {}
    for rid in recordings:
        csv_path = raw_train / "annotation" / f"{rid}.csv"
        wav = raw_train / "audio" / f"{rid}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"missing audio for {rid}: {wav}")
        dur = audio_duration(wav)
        rows, rec = audit_recording(rid, csv_path, dur)
        write_rttm(out / "ref_rttm" / f"{rid}.rttm", rid, scoring_turns(rows))
        tt = training_turns(rows, dur)
        write_rttm(out / "train_rttm" / f"{rid}.rttm", rid, tt)
        rec["n_train_rows"] = len(tt)
        recs[rid] = rec
    total_rows = Counter()
    total_pairs = Counter()
    for rec in recs.values():
        total_rows.update(rec["defect_row_counts"])
        total_pairs.update(rec["overlap_pair_counts"])
    audit = {
        "defect_classes": list(DEFECT_CLASSES),
        "dropped_classes_in_ref": list(DROP_CLASSES),
        "eof_tolerance_s": EOF_TOL_S,
        "totals": {"n_rows": sum(r["n_rows"] for r in recs.values()),
                   "n_dropped": sum(r["n_dropped"] for r in recs.values()),
                   "defect_row_counts": {k: total_rows.get(k, 0) for k in DEFECT_CLASSES},
                   "overlap_pair_counts": dict(total_pairs)},
        "recordings": recs,
    }
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "label_audit.json.tmp"
    tmp.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    tmp.replace(out / "label_audit.json")
    return audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "data/prepared")
    ap.add_argument("--raw-train", type=Path, default=RAW_TRAIN)
    a = ap.parse_args(argv)
    audit = run(a.out, a.raw_train)
    w = sys.stdout.write
    w(f"{'rec':<10} {'rows':>5} {'drop':>5} " + " ".join(f"{k[:12]:>12}" for k in DEFECT_CLASSES)
      + f" {'xspk_pairs':>10} {'same_pairs':>10}\n")
    for rid, r in audit["recordings"].items():
        c, p = r["defect_row_counts"], r["overlap_pair_counts"]
        w(f"{rid:<10} {r['n_rows']:>5} {r['n_dropped']:>5} "
          + " ".join(f"{c[k]:>12}" for k in DEFECT_CLASSES)
          + f" {p['cross_speaker_overlap']:>10} {p['same_speaker_overlap']:>10}\n")
    t = audit["totals"]
    w(f"{'TOTAL':<10} {t['n_rows']:>5} {t['n_dropped']:>5} "
      + " ".join(f"{t['defect_row_counts'][k]:>12}" for k in DEFECT_CLASSES)
      + f" {t['overlap_pair_counts'].get('cross_speaker_overlap', 0):>10}"
      + f" {t['overlap_pair_counts'].get('same_speaker_overlap', 0):>10}\n")
    w(f"wrote {a.out}/label_audit.json, ref_rttm/, train_rttm/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
