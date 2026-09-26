"""Untrusted-region mask and trusted spans per recording (SPEC item 2).

Everything is computed on a fixed 10 ms grid: grid frame ``g`` covers
``[g * GRID_DT, (g + 1) * GRID_DT)`` and the recording has
``floor(duration / GRID_DT)`` frames (floor, so no span can end past EOF; the
< 10 ms tail after the last whole frame is outside the grid).

Untrusted = union of
  (a) ``defect``          every defect row interval +- ``pad_s`` (both rows of a
                          cross-speaker overlap pair, both rows of a same-speaker
                          overlap pair, zero-length / reversed / beyond-EOF rows,
                          rows with an empty speaker ID -- ``bnd.labels`` drops
                          those from every RTTM, so their time must not be kept);
  (b) ``label_silent``    labelled frames where Nemotron max-prob < 0.5 AND
                          Silero < 0.5, in runs >= ``gap_s``;
  (c) ``unlabelled_speech`` unlabelled frames where Nemotron max-prob >= 0.5
                          AND Silero >= 0.5, in runs >= ``gap_s``;
  (d) ``gt8_speakers``    every 105 s window with > 8 reference speakers.

Trusted spans are the maximal runs of the complement, then each edge is pulled
inward to the nearest grid boundary that no labelled turn crosses (so a kept
labelled turn is never split; the trimmed seconds are reported as
``edge_trim``), then spans shorter than ``min_span_s`` are dropped
(``short_span``). ``union_excluded + edge_trim + short_span + kept == duration``
holds exactly on the grid.

Consequences of the SPEC rules that callers must know:
  * rule (a) pads BOTH rows of every overlap, so kept spans contain no
    overlapped speech (at most one active speaker per frame);
  * rule (d) bounds speakers per 105 s window, not per span: a whole span can
    hold > 8 speakers (up to 18 on this corpus), so spans are fit for NeMo
    subsegment_mode training chunks (session_len_sec=105) but NOT as whole-span
    validation/test entries (session_len_sec=-1), where NeMo keeps only 8.

Label parsing sits behind :func:`load_label_rows` so it can be swapped for
``bnd.labels`` once that module exists (see the function docstring).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

GRID_DT = 0.01
THRESH = 0.5
WINDOW_S = 105.0
MAX_SPK = 8
GAP_GRID = (2, 3, 5, 10)
REASONS = ("defect", "label_silent", "unlabelled_speech", "gt8_speakers")

TRAIN_DIR = Path("data/raw/diarization/diarization/train")
PROBS_DIR = Path("runs/probe_alignment/probs")
SILERO_DIR = Path("runs/probe_alignment/silero")


# --------------------------------------------------------------------------
# Label rows (swap boundary)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Row:
    """One CSV row as written (times in seconds, not yet validated)."""
    line: int          # 1-based physical line number in the CSV
    start: float
    end: float
    speaker: str


def _hms(s: str) -> float:
    parts = s.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"bad time {s!r}")
    h, m, sec = parts
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _local_parse_csv(path: Path) -> list[Row]:
    """Minimal CSV parser: case-insensitive header, CRLF tolerant, HH:MM:SS."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        try:
            i_s, i_e, i_k = (header.index(k) for k in ("start_time", "end_time", "speaker_id"))
        except ValueError as exc:
            raise ValueError(f"{path}: unexpected header {header}") from exc
        rows = []
        for n, rec in enumerate(reader, start=2):
            if not rec or all(not c.strip() for c in rec):
                continue
            rows.append(Row(n, _hms(rec[i_s]), _hms(rec[i_e]), rec[i_k].strip()))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def load_label_rows(csv_path: Path) -> tuple[list[Row], str]:
    """Return every CSV row (defective ones included) plus the parser used.

    Swap boundary: this module needs only (line, start, end, speaker) per raw
    row; defect classification for the mask is done by :func:`classify_defects`.
    Parsing is delegated to ``bnd.labels.read_rows`` (the single CSV parser of
    the package); the local minimal parser is used only if ``bnd.labels`` is
    not importable, and the choice is recorded in the output (``label_parser``).
    """
    try:
        from bnd import labels as _labels
    except ImportError:
        return _local_parse_csv(csv_path), "bnd.trust._local_parse_csv"
    rows = [Row(r.line, float(r.start), float(r.end), str(r.speaker))
            for r in _labels.read_rows(Path(csv_path))]
    return rows, "bnd.labels.read_rows"


def classify_defects(rows: list[Row], duration: float) -> list[dict]:
    """Defect rows: missing_speaker, zero_length, reversed, beyond_eof,
    cross/same_speaker_overlap.

    Each entry is {kind, line(s), interval=[lo, hi]} where ``interval`` is the
    span to be padded. Overlaps are positive-duration only (touching rows are
    not overlaps); for an overlap pair BOTH rows' full intervals are defects.
    A missing-speaker row is dropped by ``bnd.labels`` (no RTTM row), so it is
    padded like any other dropped row and, as in ``bnd.labels.classify``, left
    out of the overlap sweep.
    """
    out = []
    valid = []
    for r in rows:
        if not r.speaker.strip():
            out.append({"kind": "missing_speaker", "lines": [r.line],
                        "interval": [min(r.start, r.end), max(r.start, r.end)]})
            continue
        if r.end == r.start:
            out.append({"kind": "zero_length", "lines": [r.line], "interval": [r.start, r.end]})
        elif r.end < r.start:
            out.append({"kind": "reversed", "lines": [r.line], "interval": [r.end, r.start]})
        else:
            if r.end > duration:
                out.append({"kind": "beyond_eof", "lines": [r.line],
                            "interval": [min(r.start, duration), duration]})
            valid.append(r)
    valid.sort(key=lambda r: (r.start, r.end))
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            if b.start >= a.end:
                break
            if min(a.end, b.end) - max(a.start, b.start) <= 0:
                continue
            kind = "same_speaker_overlap" if a.speaker == b.speaker else "cross_speaker_overlap"
            for r in (a, b):
                out.append({"kind": kind, "lines": [a.line, b.line], "interval": [r.start, r.end]})
    return out


def valid_turns(rows: list[Row]) -> list[Row]:
    """Rows with start < end and a speaker ID (the labels Nemotron trains on;
    ``bnd.labels`` drops empty-speaker rows from the training RTTM)."""
    return [r for r in rows if r.end > r.start and r.speaker.strip()]


# --------------------------------------------------------------------------
# Grid helpers
# --------------------------------------------------------------------------
def n_grid(duration: float) -> int:
    """Whole 10 ms frames inside the audio (floor; the 1e-6 guards float error
    such as 3371.36325 / 0.01 = 337136.32499999995 or 120 / 0.01)."""
    return int(np.floor(duration / GRID_DT + 1e-6))


def sec_to_frame(t: float, n: int) -> int:
    return int(min(max(round(t / GRID_DT), 0), n))


def resample_to_grid(p: np.ndarray, dt: float, n: int, name: str = "probs",
                     tail_tol_frames: int = 1) -> np.ndarray:
    """Sample a piecewise-constant source (frame i covers [i*dt, (i+1)*dt)) at
    each grid frame centre ``(g + 0.5) * GRID_DT``.

    The source may end before the grid by at most ``tail_tol_frames`` source
    frames (Silero drops the last partial 512-sample hop); those grid frames
    take the last source value. A larger shortfall or a source that runs more
    than ``tail_tol_frames`` past the grid raises.
    """
    p = np.asarray(p)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"{name}: expected non-empty 1-D array, got {p.shape}")
    grid_end = n * GRID_DT
    src_end = p.size * dt
    if abs(src_end - grid_end) > tail_tol_frames * dt + GRID_DT:
        raise ValueError(f"{name}: covers {src_end:.3f}s but grid is {grid_end:.3f}s "
                         f"(|diff| > {tail_tol_frames} x dt={dt})")
    centres = (np.arange(n) + 0.5) * GRID_DT
    idx = np.floor(centres / dt).astype(np.int64)
    np.clip(idx, 0, p.size - 1, out=idx)
    return p[idx]


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) runs of True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2].tolist(), d[1::2].tolist()))


def long_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    out = np.zeros(mask.shape, bool)
    for s, e in runs_of(mask):
        if e - s >= min_frames:
            out[s:e] = True
    return out


def label_mask(turns: list[Row], n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for r in turns:
        m[sec_to_frame(r.start, n):sec_to_frame(r.end, n)] = True
    return m


def crossing_count(turns: list[Row], n: int) -> np.ndarray:
    """c[b] for boundary b in 0..n = number of turns with start_f < b < end_f."""
    d = np.zeros(n + 2, np.int64)
    for r in turns:
        s, e = sec_to_frame(r.start, n), sec_to_frame(r.end, n)
        if e - s >= 2:
            d[s + 1] += 1
            d[e] -= 1
    return np.cumsum(d)[: n + 1]


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def rule_defect(defects: list[dict], n: int, pad_s: float) -> np.ndarray:
    m = np.zeros(n, bool)
    for d in defects:
        lo, hi = d["interval"]
        m[sec_to_frame(lo - pad_s, n):sec_to_frame(hi + pad_s, n)] = True
    return m


def rule_label_silent(lab: np.ndarray, nemo_max: np.ndarray, silero: np.ndarray,
                      gap_s: float) -> np.ndarray:
    cand = lab & (nemo_max < THRESH) & (silero < THRESH)
    return long_runs(cand, int(round(gap_s / GRID_DT)))


def rule_unlabelled_speech(lab: np.ndarray, nemo_max: np.ndarray, silero: np.ndarray,
                           gap_s: float) -> np.ndarray:
    cand = ~lab & (nemo_max >= THRESH) & (silero >= THRESH)
    return long_runs(cand, int(round(gap_s / GRID_DT)))


def rule_gt8_speakers(turns: list[Row], n: int, window_s: float = WINDOW_S,
                      max_spk: int = MAX_SPK) -> np.ndarray:
    """Union of every window [w, w+W) (w on the grid, fully inside the
    recording; the whole recording if shorter than W) that contains activity of
    more than ``max_spk`` distinct reference speakers."""
    W = min(int(round(window_s / GRID_DT)), n)
    n_win = n - W + 1
    count = np.zeros(n_win, np.int64)
    by_spk: dict[str, np.ndarray] = {}
    for r in turns:
        by_spk.setdefault(r.speaker, np.zeros(n, bool))[
            sec_to_frame(r.start, n):sec_to_frame(r.end, n)] = True
    for m in by_spk.values():
        cs = np.concatenate([[0], np.cumsum(m)])
        count += (cs[W:W + n_win] - cs[:n_win]) > 0
    bad = np.flatnonzero(count > max_spk)
    out = np.zeros(n, bool)
    if bad.size:
        d = np.zeros(n + 1, np.int64)
        np.add.at(d, bad, 1)
        np.add.at(d, bad + W, -1)
        out = np.cumsum(d)[:n] > 0
    return out


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------
def trusted_spans(excluded: np.ndarray, crossing: np.ndarray, min_span_s: float):
    """Return (kept spans [(s,e) frames], edge_trim mask, short_span mask)."""
    n = excluded.size
    edge = np.zeros(n, bool)
    short = np.zeros(n, bool)
    kept = []
    min_f = int(round(min_span_s / GRID_DT))
    for s, e in runs_of(~excluded):
        s2, e2 = s, e
        while s2 < e2 and crossing[s2] != 0:
            s2 += 1
        while e2 > s2 and crossing[e2] != 0:
            e2 -= 1
        edge[s:s2] = True
        edge[e2:e] = True
        if e2 - s2 <= 0:
            continue
        if e2 - s2 < min_f:
            short[s2:e2] = True
            continue
        kept.append((s2, e2))
    return kept, edge, short


# --------------------------------------------------------------------------
# Per-recording driver
# --------------------------------------------------------------------------
def load_detector(path) -> tuple[np.ndarray, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing detector output {path}")
    z = np.load(path)
    return z["probs"], float(z["dt"])


def analyse(rows: list[Row], duration: float, nemo: np.ndarray, nemo_dt: float,
            silero: np.ndarray, silero_dt: float, gaps=GAP_GRID, pad_s: float = 1.0,
            min_span_s: float = 45.0) -> dict:
    """Pure function over arrays; returns {gap_s: result dict}."""
    n = n_grid(duration)
    nemo = np.asarray(nemo)
    nemo_max = nemo.max(axis=1) if nemo.ndim == 2 else nemo
    nemo_g = resample_to_grid(nemo_max, nemo_dt, n, "nemotron")
    sil_g = resample_to_grid(silero, silero_dt, n, "silero")
    turns = valid_turns(rows)
    defects = classify_defects(rows, duration)
    lab = label_mask(turns, n)
    cross = crossing_count(turns, n)
    fixed = {"defect": rule_defect(defects, n, pad_s), "gt8_speakers": rule_gt8_speakers(turns, n)}
    out = {}
    for g in gaps:
        masks = dict(fixed)
        masks["label_silent"] = rule_label_silent(lab, nemo_g, sil_g, g)
        masks["unlabelled_speech"] = rule_unlabelled_speech(lab, nemo_g, sil_g, g)
        excl = np.zeros(n, bool)
        for k in REASONS:
            excl |= masks[k]
        spans, edge, short = trusted_spans(excl, cross, min_span_s)
        keep = np.zeros(n, bool)
        for s, e in spans:
            keep[s:e] = True
        assert not (keep & (excl | edge | short)).any()
        assert int(keep.sum() + excl.sum() + edge.sum() + short.sum()) == n
        out[g] = {
            "spans": spans,
            "n_frames": n,
            "defect_counts": _count_kinds(defects),
            "seconds": {
                "duration": n * GRID_DT,
                "kept": float(keep.sum() * GRID_DT),
                "kept_labelled_speech": float((keep & lab).sum() * GRID_DT),
                "labelled_speech": float(lab.sum() * GRID_DT),
                "dropped_total": float((n - keep.sum()) * GRID_DT),
                "dropped_by_reason_raw": {k: float(masks[k].sum() * GRID_DT) for k in REASONS},
                "dropped_by_reason_exclusive": {
                    k: float((masks[k] & ~np.logical_or.reduce(
                        [masks[j] for j in REASONS if j != k])).sum() * GRID_DT) for k in REASONS},
                "dropped_union_excluded": float(excl.sum() * GRID_DT),
                "dropped_edge_trim": float(edge.sum() * GRID_DT),
                "dropped_short_span": float(short.sum() * GRID_DT),
            },
        }
    return out


def _count_kinds(defects):
    c: dict[str, int] = {}
    for d in defects:
        c[d["kind"]] = c.get(d["kind"], 0) + 1
    return c


def recording_ids(train_dir: Path) -> list[str]:
    return sorted(p.stem for p in (train_dir / "annotation").glob("*.csv"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    ap.add_argument("--probs-dir", type=Path, default=PROBS_DIR)
    ap.add_argument("--silero-dir", type=Path, default=SILERO_DIR)
    ap.add_argument("--pad", type=float, default=1.0)
    ap.add_argument("--min-span", type=float, default=45.0)
    ap.add_argument("--gaps", type=float, nargs="+", default=list(GAP_GRID))
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip recordings lacking a detector file (listed in output) "
                         "instead of failing")
    a = ap.parse_args(argv)
    import soundfile as sf

    gaps = [int(g) if float(g).is_integer() else g for g in a.gaps]
    rids = recording_ids(a.train_dir)
    if not rids:
        raise SystemExit(f"no annotations under {a.train_dir}")
    missing = {r: [str(p) for p in (a.probs_dir / f"{r}.npz", a.silero_dir / f"{r}.npz")
                   if not p.exists()] for r in rids}
    missing = {r: v for r, v in missing.items() if v}
    if missing and not a.skip_missing:
        raise SystemExit(f"missing detector files: {missing}")
    per_gap = {g: {} for g in gaps}
    parser_name = None
    for rid in rids:
        if rid in missing:
            print(f"SKIP {rid}: missing {missing[rid]}", file=sys.stderr)
            continue
        wav = a.train_dir / "audio" / f"{rid}.wav"
        info = sf.info(str(wav))
        duration = info.frames / info.samplerate
        rows, parser_name = load_label_rows(a.train_dir / "annotation" / f"{rid}.csv")
        nemo, ndt = load_detector(a.probs_dir / f"{rid}.npz")
        sil, sdt = load_detector(a.silero_dir / f"{rid}.npz")
        res = analyse(rows, duration, nemo, ndt, sil, sdt, gaps, a.pad, a.min_span)
        for g in gaps:
            r = res[g]
            r["audio_filepath"] = str(wav.resolve())
            r["duration_s"] = duration
            r["spans"] = [{"span_id": f"{rid}_s{i:03d}", "start": round(s * GRID_DT, 2),
                           "end": round(e * GRID_DT, 2), "duration": round((e - s) * GRID_DT, 2)}
                          for i, (s, e) in enumerate(r["spans"])]
            if r["spans"] and r["spans"][-1]["end"] > duration + 1e-6:
                raise AssertionError(f"{rid}: span ends {r['spans'][-1]['end']} past EOF {duration}")
            per_gap[g][rid] = r
        print(f"{rid}: " + "  ".join(
            f"gap{g}: kept {res[g]['seconds']['kept'] / 3600:.3f}h/{len(res[g]['spans'])} spans"
            for g in gaps), file=sys.stderr)

    summary = {}
    for g in gaps:
        recs = per_gap[g]
        tot = _total(recs)
        doc = {"gap_s": g, "pad_s": a.pad, "min_span_s": a.min_span, "grid_dt": GRID_DT,
               "threshold": THRESH, "window_s": WINDOW_S, "max_speakers": MAX_SPK,
               "label_parser": parser_name, "skipped": missing, "total": tot,
               "recordings": recs}
        d = a.out / str(g)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "spans.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, d / "spans.json")
        summary[str(g)] = tot
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "loss_report.json").write_text(json.dumps(
        {"skipped": missing, "label_parser": parser_name, "by_gap": summary,
         "per_recording": {str(g): {r: v["seconds"] | {"n_spans": len(v["spans"])}
                                    for r, v in per_gap[g].items()} for g in gaps}}, indent=1))
    for g in gaps:
        t = summary[str(g)]
        print(f"gap {g}s: kept {t['kept_h']:.3f}h of {t['duration_h']:.3f}h, labelled speech kept "
              f"{t['kept_labelled_speech_h']:.3f}h of {t['labelled_speech_h']:.3f}h, "
              f"{t['n_spans']} spans")
    return 0


def _total(recs: dict) -> dict:
    s = lambda k: sum(r["seconds"][k] for r in recs.values())
    return {
        "recordings": sorted(recs),
        "n_spans": sum(len(r["spans"]) for r in recs.values()),
        "duration_h": s("duration") / 3600,
        "kept_h": s("kept") / 3600,
        "labelled_speech_h": s("labelled_speech") / 3600,
        "kept_labelled_speech_h": s("kept_labelled_speech") / 3600,
        "dropped_total_s": s("dropped_total"),
        "dropped_by_reason_raw_s": {k: sum(r["seconds"]["dropped_by_reason_raw"][k]
                                           for r in recs.values()) for k in REASONS},
        "dropped_by_reason_exclusive_s": {k: sum(r["seconds"]["dropped_by_reason_exclusive"][k]
                                                 for r in recs.values()) for k in REASONS},
        "dropped_union_excluded_s": s("dropped_union_excluded"),
        "dropped_edge_trim_s": s("dropped_edge_trim"),
        "dropped_short_span_s": s("dropped_short_span"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
