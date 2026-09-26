"""DER scoring (pyannote.metrics) and Kaggle submission export/validation.

Scoring contract (docs/SPEC.md item 4):
  * UEM = whole file [0, duration) -- nothing outside the audio is scored, nothing inside is skipped.
  * skip_overlap=False: overlapped reference speech is scored (each overlapping speaker counts).
  * collars 0.0 (primary), 0.25, 0.5 seconds of TOTAL width, i.e. pyannote semantics: every reference
    segment boundary t removes [t - collar/2, t + collar/2] from both reference and hypothesis.
  * one optimal (Hungarian) speaker mapping per recording (pyannote computes it per call).
  * per-recording MISS / FA / CONF / total reference seconds; corpus DER = sum errors / sum total.
  * paired recording bootstrap (50k resamples, seed 20260925) for system deltas.
  * missing predictions fail loudly -- no recording is ever silently scored as "empty".

Same-label rows that overlap with POSITIVE duration are unioned before scoring (touching rows are
kept separate).  pyannote counts labels with ``unique=False``, so two overlapping rows of the SAME
speaker would otherwise count that speaker twice in the reference total (or twice as false alarm in
a hypothesis).  The number of such unions is reported per recording.  The union only affects the
counted speech: collars are built from the ORIGINAL reference rows (every row boundary, including
rows nested inside a same-speaker row), because merging must never delete a collar boundary
(SPEC item 1).  We therefore extrude the collars from the UEM ourselves -- exactly pyannote's
``uemify`` construction -- and call pyannote with collar=0.

RTTM times are parsed as decimals: ``end = round(onset + dur, RTTM_DECIMALS)`` so a row that only
touches the next one in the file never comes back overlapping it by a float epsilon.

This module never runs a model.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

COLLARS: tuple[float, ...] = (0.0, 0.25, 0.5)
PRIMARY_COLLAR = 0.0
BOOTSTRAP_N = 50_000
BOOTSTRAP_SEED = 20260925
KAGGLE_HEADER = ["filename", "diarization"]
KAGGLE_KEYS = ("start_time", "end_time", "speaker_id")
SPEAKER_RE = re.compile(r"^SPEAKER_[0-9]+$")
EXPORT_DECIMALS = 3
RTTM_DECIMALS = 6  # end = round(onset + dur, 6): removes onset+dur float error (1e-16 s), keeps us precision
# NIST RTTM row types. SPEAKER rows are scored; the others are legal and ignored; anything else is an error.
RTTM_TYPES = frozenset({"SPEAKER", "SPKR-INFO", "SEGMENT", "NOSCORE", "NO_RT_METADATA", "LEXEME",
                        "NON-LEX", "NON-SPEECH", "FILLER", "EDIT", "IP", "SU", "CB", "A/P"})


class ScoringError(RuntimeError):
    """Raised whenever inputs are incomplete or malformed. Never caught inside this module."""


class SubmissionError(ValueError):
    """Raised by the Kaggle validator / exporter on any schema violation."""


# --------------------------------------------------------------------------------------------
# Turns and RTTM I/O
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Seg:
    start: float
    end: float
    speaker: str


def read_rttm(path: str | Path) -> dict[str, list[Seg]]:
    """Parse an RTTM file into {file_id: [Seg]} (row order preserved). Strict on malformed rows:
    a row whose type is not a NIST RTTM type (CSV, typo such as 'SPKR') is an error, and so is a
    non-empty file with no SPEAKER row at all. Only a file with no data rows means "no speech"."""
    out: dict[str, list[Seg]] = {}
    path = Path(path)
    n_data = 0
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            s = line.strip()
            if not s or s.startswith(";;") or s.startswith("#"):
                continue
            n_data += 1
            parts = s.split()
            if parts[0] not in RTTM_TYPES:
                raise ScoringError(f"{path}:{ln}: not an RTTM row (type {parts[0]!r}): {s!r}")
            if parts[0] != "SPEAKER":
                continue
            if len(parts) < 8:
                raise ScoringError(f"{path}:{ln}: malformed RTTM row: {s!r}")
            try:
                onset, dur = float(parts[3]), float(parts[4])
            except ValueError as exc:
                raise ScoringError(f"{path}:{ln}: non-numeric time: {s!r}") from exc
            if not (math.isfinite(onset) and math.isfinite(dur)) or dur < 0 or onset < 0:
                raise ScoringError(f"{path}:{ln}: invalid onset/duration: {s!r}")
            out.setdefault(parts[1], []).append(Seg(onset, round(onset + dur, RTTM_DECIMALS), parts[7]))
    if n_data and not out:
        raise ScoringError(f"{path}: {n_data} data row(s) but no SPEAKER row; refusing to score as 'no speech'")
    return out


def write_rttm(path: str | Path, file_id: str, segs: Iterable[Seg]) -> None:
    lines = [
        f"SPEAKER {file_id} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {s.speaker} <NA> <NA>\n"
        for s in segs
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(lines), encoding="utf-8")


def load_rttm_file(path: str | Path, recording_id: str) -> list[Seg]:
    """Load one recording's segments from ``path``. The file may contain only that recording id
    (or be empty = no speech). Any other file id inside is an error."""
    data = read_rttm(path)
    extra = set(data) - {recording_id}
    if extra:
        raise ScoringError(f"{path}: contains foreign file ids {sorted(extra)} (expected {recording_id})")
    return data.get(recording_id, [])


def union_same_label_overlaps(segs: Sequence[Seg]) -> tuple[list[Seg], int]:
    """Union same-speaker segments that overlap with positive duration. Touching segments
    (a.end == b.start) are NOT merged. Zero-length segments are dropped. Returns (segs, n_unions)."""
    by_spk: dict[str, list[Seg]] = {}
    for s in segs:
        if s.end > s.start:
            by_spk.setdefault(s.speaker, []).append(s)
    out: list[Seg] = []
    n_unions = 0
    for spk, lst in by_spk.items():
        lst.sort(key=lambda x: (x.start, x.end))
        cur_s, cur_e = lst[0].start, lst[0].end
        for s in lst[1:]:
            if s.start < cur_e:  # strictly overlapping
                cur_e = max(cur_e, s.end)
                n_unions += 1
            else:
                out.append(Seg(cur_s, cur_e, spk))
                cur_s, cur_e = s.start, s.end
        out.append(Seg(cur_s, cur_e, spk))
    out.sort(key=lambda x: (x.start, x.end, x.speaker))
    return out, n_unions


def to_annotation(segs: Sequence[Seg], uri: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for i, s in enumerate(segs):
        ann[Segment(s.start, s.end), i] = s.speaker
    return ann


def audio_duration(path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        raise ScoringError(f"{path}: empty or unreadable audio")
    return info.frames / info.samplerate


# --------------------------------------------------------------------------------------------
# DER
# --------------------------------------------------------------------------------------------
@dataclass
class RecScore:
    recording_id: str
    collar: float
    duration: float
    total: float
    miss: float
    fa: float
    conf: float
    ref_unions: int = 0
    hyp_unions: int = 0

    @property
    def errors(self) -> float:
        return self.miss + self.fa + self.conf

    @property
    def der(self) -> float:
        if self.total <= 0:
            raise ScoringError(f"{self.recording_id}: zero reference speech inside UEM; DER undefined")
        return self.errors / self.total

    def as_dict(self) -> dict:
        d = asdict(self)
        d["errors"] = self.errors
        d["der"] = self.der
        return d


def score_recording(
    ref: Sequence[Seg],
    hyp: Sequence[Seg],
    duration: float,
    collar: float,
    recording_id: str = "rec",
    hyp_tolerance_s: float = 0.05,
) -> RecScore:
    """Score one recording with pyannote DiarizationErrorRate, UEM [0, duration)."""
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    if not (duration > 0 and math.isfinite(duration)):
        raise ScoringError(f"{recording_id}: invalid duration {duration}")
    if collar < 0:
        raise ScoringError(f"collar must be >= 0, got {collar}")
    for s in hyp:
        if s.end > duration + hyp_tolerance_s:
            raise ScoringError(
                f"{recording_id}: hypothesis segment ends at {s.end:.3f}s beyond audio duration {duration:.3f}s"
            )
    ref_u, n_ref = union_same_label_overlaps(ref)
    hyp_u, n_hyp = union_same_label_overlaps(hyp)
    uem = Timeline([Segment(0.0, duration)], uri=recording_id)
    if collar > 0:
        # pyannote's uemify, but with boundaries from the ORIGINAL (non-empty) reference rows, so a
        # same-speaker union never deletes a collar boundary.
        half = 0.5 * collar
        collars = [Segment(t - half, t + half) for s in ref if s.end > s.start for t in (s.start, s.end)]
        if collars:
            uem = Timeline(segments=collars).support().gaps(support=uem)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    comp = metric(to_annotation(ref_u, recording_id), to_annotation(hyp_u, recording_id), uem=uem, detailed=True)
    return RecScore(
        recording_id=recording_id,
        collar=collar,
        duration=duration,
        total=float(comp["total"]),
        miss=float(comp["missed detection"]),
        fa=float(comp["false alarm"]),
        conf=float(comp["confusion"]),
        ref_unions=n_ref,
        hyp_unions=n_hyp,
    )


def corpus_summary(scores: Sequence[RecScore]) -> dict:
    """Corpus DER (sum errors / sum total) plus component rates and macro statistics."""
    if not scores:
        raise ScoringError("no recordings to summarise")
    tot = sum(s.total for s in scores)
    if tot <= 0:
        raise ScoringError("zero total reference speech")
    miss, fa, conf = (sum(getattr(s, k) for s in scores) for k in ("miss", "fa", "conf"))
    ders = np.array([s.der for s in scores])
    return {
        "n_recordings": len(scores),
        "total_ref_s": tot,
        "miss_s": miss,
        "fa_s": fa,
        "conf_s": conf,
        "der": (miss + fa + conf) / tot,
        "miss_rate": miss / tot,
        "fa_rate": fa / tot,
        "conf_rate": conf / tot,
        "macro_der_mean": float(ders.mean()),
        "macro_der_median": float(np.median(ders)),
        "macro_der_min": float(ders.min()),
        "macro_der_max": float(ders.max()),
    }


def load_system(
    recording_ids: Sequence[str],
    ref_dir: str | Path,
    hyp_dir: str | Path,
) -> tuple[dict[str, list[Seg]], dict[str, list[Seg]]]:
    """Load ref + hyp RTTMs for exactly ``recording_ids``. Missing ref or hyp -> ScoringError.
    Hypothesis RTTMs for recordings NOT in the list are also an error (wrong fold/mix-up)."""
    ref_dir, hyp_dir = Path(ref_dir), Path(hyp_dir)
    missing_ref = [r for r in recording_ids if not (ref_dir / f"{r}.rttm").is_file()]
    missing_hyp = [r for r in recording_ids if not (hyp_dir / f"{r}.rttm").is_file()]
    if missing_ref:
        raise ScoringError(f"missing reference RTTM(s) in {ref_dir}: {missing_ref}")
    if missing_hyp:
        raise ScoringError(f"missing prediction RTTM(s) in {hyp_dir}: {missing_hyp}")
    extra = sorted({p.stem for p in hyp_dir.glob("*.rttm")} - set(recording_ids))
    if extra:
        raise ScoringError(f"prediction RTTM(s) for unexpected recordings in {hyp_dir}: {extra}")
    refs = {r: load_rttm_file(ref_dir / f"{r}.rttm", r) for r in recording_ids}
    hyps = {r: load_rttm_file(hyp_dir / f"{r}.rttm", r) for r in recording_ids}
    return refs, hyps


def score_corpus(
    refs: Mapping[str, Sequence[Seg]],
    hyps: Mapping[str, Sequence[Seg]],
    durations: Mapping[str, float],
    collars: Sequence[float] = COLLARS,
) -> dict[float, list[RecScore]]:
    """Score every recording in ``refs`` at every collar. Strict: a recording without a prediction
    or a duration is an error; predictions for unknown recordings are an error."""
    rids = sorted(refs)
    if not rids:
        raise ScoringError("empty reference set")
    missing = [r for r in rids if r not in hyps]
    if missing:
        raise ScoringError(f"missing predictions for {missing}")
    extra = sorted(set(hyps) - set(refs))
    if extra:
        raise ScoringError(f"predictions for unknown recordings {extra}")
    nodur = [r for r in rids if r not in durations]
    if nodur:
        raise ScoringError(f"missing durations for {nodur}")
    return {
        c: [score_recording(refs[r], hyps[r], durations[r], c, recording_id=r) for r in rids]
        for c in collars
    }


# --------------------------------------------------------------------------------------------
# Paired recording bootstrap
# --------------------------------------------------------------------------------------------
def paired_bootstrap(
    a: Sequence[RecScore],
    b: Sequence[RecScore],
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> dict:
    """Paired recording-level bootstrap of corpus DER(b) - DER(a) (negative = b better).

    Both systems must be scored on identical recordings, collar and reference totals. Each resample
    draws recordings with replacement and recomputes corpus DER for both systems on that draw.
    """
    ida = [s.recording_id for s in a]
    idb = [s.recording_id for s in b]
    if ida != idb or not ida:
        raise ScoringError(f"systems scored on different/empty recording lists: {ida} vs {idb}")
    for sa, sb in zip(a, b):
        if sa.collar != sb.collar:
            raise ScoringError(f"{sa.recording_id}: collar mismatch {sa.collar} vs {sb.collar}")
        if not math.isclose(sa.total, sb.total, rel_tol=0, abs_tol=1e-6):
            raise ScoringError(f"{sa.recording_id}: reference total differs ({sa.total} vs {sb.total})")
    tot = np.array([s.total for s in a], dtype=np.float64)
    ea = np.array([s.errors for s in a], dtype=np.float64)
    eb = np.array([s.errors for s in b], dtype=np.float64)
    k = len(tot)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n, dtype=np.float64)
    chunk = 10_000
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        idx = rng.integers(0, k, size=(hi - lo, k))
        t = tot[idx].sum(axis=1)
        deltas[lo:hi] = (eb[idx].sum(axis=1) - ea[idx].sum(axis=1)) / t
    point = (eb.sum() - ea.sum()) / tot.sum()
    alpha = (1.0 - ci) / 2.0
    per_rec = eb / tot - ea / tot
    return {
        "n_resamples": n,
        "seed": seed,
        "n_recordings": k,
        "collar": a[0].collar,
        "der_a": float(ea.sum() / tot.sum()),
        "der_b": float(eb.sum() / tot.sum()),
        "delta_b_minus_a": float(point),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
        "ci": ci,
        "p_b_not_better": float(np.mean(deltas >= 0.0)),
        "b_wins": int(np.sum(per_rec < 0)),
        "b_losses": int(np.sum(per_rec > 0)),
        "ties": int(np.sum(per_rec == 0)),
        "worst_rec_delta": float(per_rec.max()),
        "worst_rec": ida[int(per_rec.argmax())],
    }


# --------------------------------------------------------------------------------------------
# Kaggle exporter / validator
# --------------------------------------------------------------------------------------------
def read_sample_order(sample_csv: str | Path) -> list[str]:
    with open(sample_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] != KAGGLE_HEADER:
        raise SubmissionError(f"{sample_csv}: unexpected header {rows[0] if rows else None}")
    names = [r[0] for r in rows[1:]]
    if len(set(names)) != len(names):
        raise SubmissionError(f"{sample_csv}: duplicate filenames")
    return names


def flatten_overlaps(segs: Sequence[Seg]) -> list[Seg]:
    """Make segments globally non-overlapping. In an overlapped region the speaker whose segment
    started earliest keeps the floor (ties: longer segment, then label). Adjacent pieces of the same
    speaker are merged. Deterministic."""
    segs = [s for s in segs if s.end > s.start]
    if not segs:
        return []
    bounds = sorted({t for s in segs for t in (s.start, s.end)})
    order = sorted(segs, key=lambda s: (s.start, -(s.end - s.start), s.speaker))
    out: list[Seg] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        winner = next((s for s in order if s.start <= lo and s.end >= hi), None)
        if winner is None:
            continue
        if out and out[-1].speaker == winner.speaker and out[-1].end == lo:
            out[-1] = Seg(out[-1].start, hi, winner.speaker)
        else:
            out.append(Seg(lo, hi, winner.speaker))
    return out


def to_kaggle_items(segs: Sequence[Seg], duration: float, overlap: str = "error") -> list[dict]:
    """Convert one recording's hypothesis to the Kaggle JSON list.

    overlap="error": any overlap between segments that survives rounding to 1 ms raises SubmissionError.
    overlap="flatten": resolved with :func:`flatten_overlaps` (earliest-start keeps the floor).
    Speakers are renamed SPEAKER_1..N in order of first appearance; times rounded to 1 ms and
    clipped to [0, duration]; segments that vanish after rounding are dropped.
    """
    if overlap not in ("error", "flatten"):
        raise ValueError(f"overlap must be 'error' or 'flatten', got {overlap!r}")
    for s in segs:
        if not (math.isfinite(s.start) and math.isfinite(s.end)) or s.start < 0 or s.end < s.start:
            raise SubmissionError(f"invalid segment {s}")
    live = sorted((s for s in segs if s.end > s.start), key=lambda s: (s.start, s.end))
    if overlap == "error":
        # compared at export precision (1 ms): a float-epsilon overlap (e.g. 0.1 + 0.2 vs 0.3) is not
        # an overlap in the exported file, a real one survives rounding.
        for x, y in zip(live, live[1:]):
            if round(y.start, EXPORT_DECIMALS) < round(x.end, EXPORT_DECIMALS):
                raise SubmissionError(f"overlapping segments {x} / {y}; pass overlap='flatten' to resolve")
        flat = live
    else:
        flat = flatten_overlaps(live)
    cap = math.floor(duration * 10**EXPORT_DECIMALS) / 10**EXPORT_DECIMALS
    names: dict[str, str] = {}
    items: list[dict] = []
    for s in flat:
        st = round(min(max(s.start, 0.0), duration), EXPORT_DECIMALS)
        en = round(min(s.end, duration), EXPORT_DECIMALS)
        en = min(en, cap)
        if en <= st:
            continue
        if s.speaker not in names:
            names[s.speaker] = f"SPEAKER_{len(names) + 1}"
        items.append({"start_time": float(st), "end_time": float(en), "speaker_id": names[s.speaker]})
    return items


def render_submission(order: Sequence[str], items_by_file: Mapping[str, list[dict]]) -> str:
    missing = [f for f in order if f not in items_by_file]
    if missing:
        raise SubmissionError(f"missing predictions for {missing}")
    extra = sorted(set(items_by_file) - set(order))
    if extra:
        raise SubmissionError(f"predictions for files not in sample submission: {extra}")
    buf = io.StringIO()
    w = csv.writer(buf)  # default dialect == sample file bytes (CRLF, minimal quoting)
    w.writerow(KAGGLE_HEADER)
    for f in order:
        w.writerow([f, json.dumps(items_by_file[f])])
    return buf.getvalue()


def validate_submission_text(
    text: str, order: Sequence[str], durations: Mapping[str, float]
) -> dict:
    """Validate submission CSV text. Raises SubmissionError on the first violation."""
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != KAGGLE_HEADER:
        raise SubmissionError(f"header must be {KAGGLE_HEADER}, got {rows[0] if rows else None}")
    body = rows[1:]
    names = [r[0] if r else "" for r in body]
    if names != list(order):
        raise SubmissionError(f"row order/filenames differ from sample submission: {names} vs {list(order)}")
    n_segs = 0
    for ln, r in enumerate(body, 2):
        if len(r) != 2:
            raise SubmissionError(f"line {ln}: expected 2 columns, got {len(r)}")
        fname = r[0]
        if fname not in durations:
            raise SubmissionError(f"line {ln}: no duration known for {fname}")
        dur = durations[fname]
        try:
            items = json.loads(r[1])
        except json.JSONDecodeError as exc:
            raise SubmissionError(f"line {ln} ({fname}): diarization is not JSON: {exc}") from exc
        if not isinstance(items, list):
            raise SubmissionError(f"line {ln} ({fname}): diarization must be a JSON list")
        prev_end = -math.inf
        prev_start = -math.inf
        for j, it in enumerate(items):
            where = f"line {ln} ({fname}) item {j}"
            if not isinstance(it, dict) or set(it) != set(KAGGLE_KEYS):
                raise SubmissionError(f"{where}: must be an object with exactly keys {KAGGLE_KEYS}: {it!r}")
            st, en, spk = it["start_time"], it["end_time"], it["speaker_id"]
            for nm, v in (("start_time", st), ("end_time", en)):
                if not isinstance(v, float) or not math.isfinite(v):
                    raise SubmissionError(f"{where}: {nm} must be a finite float, got {v!r}")
            if not isinstance(spk, str) or not SPEAKER_RE.match(spk):
                raise SubmissionError(f"{where}: speaker_id must match SPEAKER_k, got {spk!r}")
            if not (0.0 <= st < en <= dur):
                raise SubmissionError(f"{where}: need 0 <= start < end <= duration ({dur:.3f}), got {st}, {en}")
            if st < prev_start:
                raise SubmissionError(f"{where}: not sorted by start_time")
            if st < prev_end:
                raise SubmissionError(f"{where}: overlaps previous segment (start {st} < previous end {prev_end})")
            prev_start, prev_end = st, en
            n_segs += 1
    return {"n_files": len(body), "n_segments": n_segs}


def durations_for(stems: Sequence[str], audio_dir: str | Path) -> dict[str, float]:
    audio_dir = Path(audio_dir)
    missing = [s for s in stems if not (audio_dir / f"{s}.wav").is_file()]
    if missing:
        raise ScoringError(f"missing audio in {audio_dir}: {missing}")
    return {s: audio_duration(audio_dir / f"{s}.wav") for s in stems}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _parse_ids(ids: str | None, ref_dir: Path) -> list[str]:
    if ids:
        return [x.strip() for x in ids.split(",") if x.strip()]
    found = sorted(p.stem for p in ref_dir.glob("*.rttm"))
    if not found:
        raise ScoringError(f"no reference RTTMs in {ref_dir}")
    return found


def _score_dir(ref_dir: Path, hyp_dir: Path, audio_dir: Path, ids: list[str], collars) -> dict:
    refs, hyps = load_system(ids, ref_dir, hyp_dir)
    durs = durations_for(ids, audio_dir)
    return score_corpus(refs, hyps, durs, collars)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bnd.score", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("der", help="score a hypothesis RTTM dir")
    d.add_argument("--ref-dir", required=True, type=Path)
    d.add_argument("--hyp-dir", required=True, type=Path)
    d.add_argument("--audio-dir", required=True, type=Path)
    d.add_argument("--ids", help="comma-separated recording ids (default: all ref RTTMs)")
    d.add_argument("--out", type=Path)

    c = sub.add_parser("compare", help="paired bootstrap of system B vs system A")
    c.add_argument("--ref-dir", required=True, type=Path)
    c.add_argument("--hyp-a", required=True, type=Path)
    c.add_argument("--hyp-b", required=True, type=Path)
    c.add_argument("--audio-dir", required=True, type=Path)
    c.add_argument("--ids")
    c.add_argument("--n", type=int, default=BOOTSTRAP_N)
    c.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    c.add_argument("--out", type=Path)

    e = sub.add_parser("export", help="hypothesis RTTM dir -> Kaggle CSV")
    e.add_argument("--hyp-dir", required=True, type=Path)
    e.add_argument("--audio-dir", required=True, type=Path)
    e.add_argument("--sample", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--overlap", choices=["error", "flatten"], default="error")

    v = sub.add_parser("validate", help="validate a Kaggle CSV")
    v.add_argument("--csv", required=True, type=Path)
    v.add_argument("--audio-dir", required=True, type=Path)
    v.add_argument("--sample", required=True, type=Path)

    a = ap.parse_args(argv)

    if a.cmd == "der":
        ids = _parse_ids(a.ids, a.ref_dir)
        res = _score_dir(a.ref_dir, a.hyp_dir, a.audio_dir, ids, COLLARS)
        report = {
            str(col): {"corpus": corpus_summary(sc), "per_recording": [s.as_dict() for s in sc]}
            for col, sc in res.items()
        }
        for col, sc in res.items():
            sm = corpus_summary(sc)
            print(f"collar={col:<5} DER={100*sm['der']:.2f}%  MISS={100*sm['miss_rate']:.2f}  "
                  f"FA={100*sm['fa_rate']:.2f}  CONF={100*sm['conf_rate']:.2f}  "
                  f"macro_mean={100*sm['macro_der_mean']:.2f}  n={sm['n_recordings']}")
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(report, indent=2))
    elif a.cmd == "compare":
        ids = _parse_ids(a.ids, a.ref_dir)
        ra = _score_dir(a.ref_dir, a.hyp_a, a.audio_dir, ids, COLLARS)
        rb = _score_dir(a.ref_dir, a.hyp_b, a.audio_dir, ids, COLLARS)
        report = {str(col): paired_bootstrap(ra[col], rb[col], n=a.n, seed=a.seed) for col in COLLARS}
        for col, r in report.items():
            print(f"collar={col:<5} A={100*r['der_a']:.2f}% B={100*r['der_b']:.2f}% "
                  f"delta={100*r['delta_b_minus_a']:+.2f} pp  95%CI[{100*r['ci_low']:+.2f},{100*r['ci_high']:+.2f}]  "
                  f"wins={r['b_wins']}/{r['n_recordings']}")
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(report, indent=2))
    elif a.cmd == "export":
        order = read_sample_order(a.sample)
        durs = durations_for(order, a.audio_dir)
        missing = [f for f in order if not (a.hyp_dir / f"{f}.rttm").is_file()]
        if missing:
            raise ScoringError(f"missing prediction RTTM(s) in {a.hyp_dir}: {missing}")
        items = {
            f: to_kaggle_items(load_rttm_file(a.hyp_dir / f"{f}.rttm", f), durs[f], overlap=a.overlap)
            for f in order
        }
        text = render_submission(order, items)
        info = validate_submission_text(text, order, durs)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(text.encode("utf-8"))
        print(f"wrote {a.out}: {info}")
    elif a.cmd == "validate":
        order = read_sample_order(a.sample)
        durs = durations_for(order, a.audio_dir)
        text = a.csv.read_bytes().decode("utf-8")
        info = validate_submission_text(text, order, durs)
        print(f"VALID {a.csv}: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
