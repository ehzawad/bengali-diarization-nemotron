"""NeMo diarization manifests + per-span RTTMs from trusted spans, verified against the real loader.

Loader convention (NeMo 3.1.0 @ cf724ac, verified by reading the source; paths relative to
``third_party/nemo-speech``):

* Manifest parsing — ``nemo/collections/common/parts/preprocessing/collections.py``
  ``EndtoEndDiarizationSpeechLabel.__parse_item_rttm`` (l.1422-1486): accepts ``audio_filepath``
  (l.1435) and ``rttm_filepath`` (l.1461), defaults a missing ``offset`` to 0 (l.1426), requires
  ``duration`` (l.1476) and keeps only audio/uniq_id/duration/rttm/offset (l.1479); extra keys such
  as ``recording_id``/``span_id``/``num_speakers`` are ignored.
* RTTM times are ABSOLUTE recording time. ``nemo/collections/asr/data/audio_to_diar_label.py``
  ``extract_frame_info_from_rttm`` (l.80-124) keeps lines intersecting
  ``[offset, offset + duration]`` (l.95, l.107) and clips them to that window (l.109), without
  subtracting the offset; ``get_frame_targets_from_rttm`` (l.127-174) then converts to frames with
  ``int((t - offset) * feat_per_sec)`` (l.172). A span-relative RTTM would therefore be misread
  (see ``tests/test_nemo_data.py::test_relative_rttm_is_misread``).
* ``subsegment_mode`` (``_create_subsegment`` l.1137-1239): plans on the whole manifest region —
  frames ``[0, int(duration * 100))`` relative to ``offset`` (l.1171-1181) — samples one chunk of
  up to ``session_len_sec`` or, with probability ``subsegment_two_chunks_rate``, two chunks
  (l.1189-1201), each ``(start, end)`` in 10 ms feature frames relative to ``offset``; then loads
  audio at ``offset + start / feat_per_sec`` for ``(end - start) / feat_per_sec`` s (l.1053-1054).
  So audio and labels never leave ``[offset, offset + duration]`` by construction.
* Speaker columns are assigned in order of FIRST APPEARANCE OF A LINE in the RTTM file within the
  region (l.117-121), inactive columns in the chosen chunk(s) are dropped and the rest padded to
  ``num_spks`` (l.1209-1224). Columns are NOT re-sorted by arrival in the chunk; the model sorts
  targets by arrival itself in the ATS loss (``get_ats_targets_hungarian``,
  ``nemo/collections/asr/parts/utils/asr_multispeaker_utils.py`` l.313;
  ``sortformer_diar_models.py`` l.1651). We write RTTM lines sorted by start time so column order
  is at least span-arrival order.
* Chunks with more than ``num_spks`` active speakers are filtered out at planning time
  (single chunk l.878, two-chunk l.943); when no candidate survives, a full-length Gaussian-noise sample with all-zero
  targets is returned (``_fallback_noise_subsegment``, l.1067-1135).
* Target frame rate: ``feat_per_sec = int(1 / window_stride) = 100`` (l.315) and, with the
  Nemotron-3 ``output_subsampling_factor = 1``, ``diar_frame_length = 0.01`` so targets are the
  10 ms feature-level activity (``get_soft_targets_seg`` stride <= 1 branch, l.568-569).
* Sampling exposure: ``__len__`` is the number of manifest lines (l.386-387) and each
  ``__getitem__`` turns one line into ONE plan of at most ``session_len_sec`` (l.1241-1247); the
  model's dataloader shuffles items uniformly (``sortformer_diar_models.py``
  ``__setup_dataloader_from_config``: ``sampler=None`` without an int ``limit_train_batches``, and
  ``_OversamplingDistributedSampler`` is also per-item). With one line per span (SPEC item 3) a
  2720 s span is drawn exactly as often as a 48 s one, so audio seconds seen per hour of span differ
  by ~57x. ``train_manifest.json`` stays one line per span (the SPEC contract);
  ``train_manifest_balanced.json`` repeats each line ``max(1, round(duration / session_len_sec))``
  times (same audio, offset, RTTM - no new cut points, full chunk-position diversity) so expected
  audio seconds drawn per span-second are within ~2x across spans. ``exposure_stats`` quantifies
  both in the report; which manifest trains is a phase-2 decision.

CLI::

    python -m bnd.nemo_data --spans data/prepared/trust/5/spans.json --out data/prepared/manifests
    python -m bnd.nemo_data --whole-recordings --out <scratch dir>   # temporary, untrusted spans
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NEMO_ROOT = REPO / "third_party/nemo-speech"
STAGE2_YAML = NEMO_ROOT / "examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_streaming_8spk.yaml"
RAW_TRAIN = REPO / "data/raw/diarization/diarization/train"
TRAIN_RTTM_DIR = REPO / "data/prepared/train_rttm"
FRAME_S = 0.01  # loader target frame (10 ms)
CS = 100  # centiseconds per second


class NemoDataError(ValueError):
    """Malformed spans / labels / loader behaviour that violates the contract."""


@dataclass(frozen=True)
class Span:
    recording_id: str
    span_id: str
    start: float  # absolute recording seconds
    end: float
    audio_filepath: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


# ----------------------------------------------------------------------------------------------
# spans / labels input
# ----------------------------------------------------------------------------------------------

def _cs(t: float) -> int:
    """Seconds -> integer centiseconds, failing loudly when not on the 10 ms grid."""
    v = round(t * CS)
    if abs(v - t * CS) > 1e-6:
        raise NemoDataError(f"time {t!r} is not on the 10 ms grid")
    return v


def _pick(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    raise NemoDataError(f"span entry {d!r} lacks any of {keys}")


def _audio_for(rid: str, raw_train: Path) -> str:
    return str((raw_train / "audio" / f"{rid}.wav").resolve())


def load_spans(path: Path, raw_train: Path = RAW_TRAIN) -> list[Span]:
    """Read spans.json. Accepted shapes (fails loudly on anything else):

    * ``[{recording_id|rid, start|start_s, end|end_s, [span_id], [audio_filepath]}, ...]``
    * ``{"spans": [<entry>, ...], ...}``
    * ``{"recordings": {rid: {"spans": [[start, end] | <entry>, ...]}}}`` or ``{rid: [[s, e], ...]}``
    """
    obj = json.loads(Path(path).read_text())
    entries: list[dict] = []

    def add_rec(rid: str, items: Iterable):
        for it in items:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                entries.append({"recording_id": rid, "start": it[0], "end": it[1]})
            elif isinstance(it, dict):
                entries.append({"recording_id": rid, **it})
            else:
                raise NemoDataError(f"{path}: bad span item {it!r} for {rid}")

    if isinstance(obj, list):
        entries = list(obj)
    elif isinstance(obj, dict) and isinstance(obj.get("spans"), list):
        entries = list(obj["spans"])
    elif isinstance(obj, dict) and isinstance(obj.get("recordings"), dict):
        for rid, rec in obj["recordings"].items():
            add_rec(rid, rec["spans"] if isinstance(rec, dict) else rec)
    elif isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()):
        for rid, items in obj.items():
            add_rec(rid, items)
    else:
        raise NemoDataError(f"{path}: unrecognised spans.json shape (top-level {type(obj).__name__})")

    spans: list[Span] = []
    counters: dict[str, int] = {}
    for e in entries:
        rid = str(_pick(e, "recording_id", "rid", "recording"))
        s, t = float(_pick(e, "start", "start_s", "offset")), float(_pick(e, "end", "end_s"))
        _cs(s), _cs(t)
        if not t > s:
            raise NemoDataError(f"{path}: span {e!r} has end <= start")
        k = counters.get(rid, 0)
        counters[rid] = k + 1
        sid = str(e.get("span_id") or f"{rid}_s{k:03d}")
        audio = e.get("audio_filepath") or _audio_for(rid, raw_train)
        spans.append(Span(rid, sid, s, t, str(audio)))
    if not spans:
        raise NemoDataError(f"{path}: no spans")
    ids = [s.span_id for s in spans]
    if len(set(ids)) != len(ids):
        raise NemoDataError(f"{path}: duplicate span_id")
    by_rec: dict[str, list[Span]] = {}
    for sp in spans:
        by_rec.setdefault(sp.recording_id, []).append(sp)
    for rid, ss in by_rec.items():
        ss = sorted(ss, key=lambda x: x.start)
        for a, b in zip(ss, ss[1:]):
            if b.start < a.end:
                raise NemoDataError(f"{path}: overlapping spans {a.span_id} / {b.span_id}")
    return spans


def read_rttm(path: Path) -> list[Turn]:
    turns = []
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        f = line.split()
        if not f:
            continue
        if f[0] != "SPEAKER" or len(f) < 8:
            raise NemoDataError(f"{path}:{ln}: not an RTTM SPEAKER line: {line!r}")
        s, d = float(f[3]), float(f[4])
        turns.append(Turn(s, s + d, f[7]))
    return turns


def load_turns(rid: str, train_rttm_dir: Path = TRAIN_RTTM_DIR, raw_train: Path = RAW_TRAIN,
               duration: float | None = None) -> tuple[list[Turn], str]:
    """Training turns for a recording: prepared training RTTM if present, else the raw CSV parsed
    with ``bnd.labels`` (valid rows, same-speaker merge, clipped to the audio)."""
    p = Path(train_rttm_dir) / f"{rid}.rttm"
    if p.exists():
        return read_rttm(p), str(p)
    from bnd import labels  # raw-CSV route; fails loudly if the parser rejects the file

    csv_path = Path(raw_train) / "annotation" / f"{rid}.csv"
    rows = labels.classify(labels.read_rows(csv_path), duration)
    return [Turn(t.start, t.end, t.speaker) for t in labels.training_turns(rows, duration)], str(csv_path)


def audio_info(path: str) -> tuple[int, int]:
    import soundfile as sf

    info = sf.info(path)
    return info.frames, info.samplerate


def whole_recording_spans(raw_train: Path = RAW_TRAIN) -> list[Span]:
    """Temporary untrusted spans: each full recording floored to the 10 ms grid."""
    spans = []
    for wav in sorted((Path(raw_train) / "audio").glob("*.wav")):
        n, sr = audio_info(str(wav))
        end = math.floor(n * CS / sr) / CS
        spans.append(Span(wav.stem, f"{wav.stem}_whole", 0.0, end, str(wav.resolve())))
    if not spans:
        raise NemoDataError(f"no WAVs under {raw_train}/audio")
    return spans


# ----------------------------------------------------------------------------------------------
# manifest + per-span RTTM writing
# ----------------------------------------------------------------------------------------------

def rttm_line(rid: str, t: Turn) -> str:
    return f"SPEAKER {rid} 1 {t.start:.3f} {t.end - t.start:.3f} <NA> <NA> {t.speaker} <NA> <NA>\n"


def span_turns(span: Span, turns: list[Turn]) -> tuple[list[Turn], int]:
    """Turns intersecting the span, clipped to it (absolute times), sorted by start.
    Returns (turns, n_clipped). Clipping only happens when a span edge is an excluded-region edge
    that cuts a labelled turn; the audio is cut at the same point so labels stay consistent."""
    out, clipped = [], 0
    for t in turns:
        if t.end <= span.start or t.start >= span.end:
            continue
        s, e = max(t.start, span.start), min(t.end, span.end)
        if (s, e) != (t.start, t.end):
            clipped += 1
        if e > s:
            out.append(Turn(s, e, t.speaker))
    out.sort(key=lambda x: (x.start, x.end, x.speaker))
    return out, clipped


def write_manifest(spans: list[Span], out_dir: Path, turns_by_rec: dict[str, list[Turn]],
                   manifest_name: str = "train_manifest.json") -> dict:
    """Write ``<out>/rttm/<span_id>.rttm`` (absolute times) and ``<out>/<manifest_name>`` JSONL."""
    out_dir = Path(out_dir).resolve()
    rttm_dir = out_dir / "rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    lines, stats = [], {"n_spans": 0, "n_turns": 0, "n_turns_clipped_at_span_edge": 0,
                        "hours": 0.0, "labelled_speaker_hours": 0.0, "max_speakers_in_span": 0}
    for sp in spans:
        if sp.recording_id not in turns_by_rec:
            raise NemoDataError(f"no turns loaded for {sp.recording_id}")
        n, sr = audio_info(sp.audio_filepath)
        if sp.end > n / sr + 1e-9:
            # The 10 ms grid can round the last span end a few ms past EOF (grid rounding upstream).
            # Up to one grid step we clamp to the last whole grid point inside the audio and log it;
            # anything larger is a real error.
            eof_grid = math.floor(n * CS / sr) / CS
            if sp.end - n / sr >= FRAME_S:
                raise NemoDataError(f"span {sp.span_id} ends at {sp.end} beyond audio EOF {n / sr:.3f}")
            stats.setdefault("spans_clamped_to_eof", []).append(
                {"span_id": sp.span_id, "end": sp.end, "eof": n / sr, "new_end": eof_grid})
            sp = Span(sp.recording_id, sp.span_id, sp.start, eof_grid, sp.audio_filepath)
        tt, clipped = span_turns(sp, turns_by_rec[sp.recording_id])
        spk = sorted({t.speaker for t in tt})
        rpath = (rttm_dir / f"{sp.span_id}.rttm").resolve()
        tmp = rpath.with_suffix(".rttm.tmp")
        tmp.write_text("".join(rttm_line(sp.recording_id, t) for t in tt))
        tmp.replace(rpath)
        lines.append(json.dumps({
            "audio_filepath": sp.audio_filepath,
            "offset": round(sp.start, 2),
            "duration": round(sp.duration, 2),
            "num_speakers": len(spk),
            "rttm_filepath": str(rpath),
            "recording_id": sp.recording_id,
            "span_id": sp.span_id,
        }, ensure_ascii=False))
        stats["n_spans"] += 1
        stats["n_turns"] += len(tt)
        stats["n_turns_clipped_at_span_edge"] += clipped
        stats["hours"] += sp.duration / 3600
        stats["labelled_speaker_hours"] += sum(t.end - t.start for t in tt) / 3600
        stats["max_speakers_in_span"] = max(stats["max_speakers_in_span"], len(spk))
    mpath = out_dir / manifest_name
    tmp = mpath.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(mpath)
    stats["manifest"] = str(mpath)
    return stats


# ----------------------------------------------------------------------------------------------
# real NeMo dataset (official stage-2 train_ds) + probe
# ----------------------------------------------------------------------------------------------

def stage2_config(yaml_path: Path = STAGE2_YAML):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(yaml_path))
    return cfg


def build_real_dataset(manifest: Path, yaml_path: Path = STAGE2_YAML):
    """Build the training dataset through the model's own private dataloader factory
    (``SortformerEncLabelModel.__setup_dataloader_from_config``) with the official stage-2
    ``train_ds`` (only loader-process knobs changed: num_workers=0, pin_memory=False, batch 4).
    Returns (dataset, dataloader, resolved_train_ds)."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

    cfg = stage2_config(yaml_path)
    cfg.model.train_ds.manifest_filepath = str(manifest)
    train_ds = OmegaConf.create(OmegaConf.to_container(cfg.model.train_ds, resolve=True))
    train_ds.num_workers = 0
    train_ds.pin_memory = False
    train_ds.batch_size = 4
    stub = SimpleNamespace(
        output_subsampling_factor=int(cfg.model.output_subsampling_factor),
        augmentor=None,
        _cfg=OmegaConf.create({"preprocessor": OmegaConf.to_container(cfg.model.preprocessor, resolve=True)}),
        _trainer=None,
        device="cpu",
        world_size=1,
    )
    setup = SortformerEncLabelModel.__dict__["_SortformerEncLabelModel__setup_dataloader_from_config"]
    dl = setup(stub, train_ds, is_training=True)
    return dl.dataset, dl, train_ds


class _Spy:
    """Records the loader's audio reads and fallbacks without changing behaviour."""

    def __init__(self, ds):
        self.ds = ds
        self.reads: list[dict] = []
        self.loads: list[dict] = []
        self.fallbacks = 0

    @contextlib.contextmanager
    def attach(self):
        import nemo.collections.asr.data.audio_to_diar_label as m

        orig_cls = m.AudioSegment
        spy = self

        class SpyAudioSegment(orig_cls):
            @classmethod
            def from_file(cls, audio_file, *a, **kw):
                seg = orig_cls.from_file(audio_file, *a, **kw)
                spy.reads.append({"audio_file": audio_file, "offset": kw.get("offset"),
                                  "duration": kw.get("duration"), "n": int(seg.samples.shape[0]),
                                  "samples": seg.samples.copy()})
                return seg

        orig_load = self.ds._load_audio_chunks
        orig_fb = self.ds._fallback_noise_subsegment

        def load(audio_file, offset, bounds):
            spy.loads.append({"offset": offset, "bounds": [tuple(map(int, b)) for b in bounds]})
            return orig_load(audio_file, offset, bounds)

        def fb(*a, **kw):
            spy.fallbacks += 1
            return orig_fb(*a, **kw)

        m.AudioSegment = SpyAudioSegment
        self.ds._load_audio_chunks = load
        self.ds._fallback_noise_subsegment = fb
        try:
            yield self
        finally:
            m.AudioSegment = orig_cls
            del self.ds._load_audio_chunks
            del self.ds._fallback_noise_subsegment

    def reset(self):
        self.reads.clear()
        self.loads.clear()
        self.fallbacks = 0


def expected_activity(turns: list[Turn], offset: float, n_frames: int) -> dict[str, np.ndarray]:
    """Independent reference: speaker -> bool[n_frames] on the 10 ms grid relative to ``offset``,
    frame f active iff [f, f+1) centiseconds lies in a turn (turns rounded to 10 ms)."""
    off = round(offset * CS)
    act: dict[str, np.ndarray] = {}
    for t in turns:
        s = max(0, round(t.start * CS) - off)
        e = min(n_frames, round(t.end * CS) - off)
        a = act.setdefault(t.speaker, np.zeros(n_frames, dtype=bool))
        if e > s:
            a[s:e] = True
    return act


def loader_arith_activity(rttm_lines: list[str], offset: float, duration: float,
                          n_frames: int) -> dict[str, np.ndarray]:
    """What the loader's float arithmetic should produce for these RTTM lines: identical to
    ``expected_activity`` except where ``int((t - offset) * 100)`` truncates a grid time one frame
    low (e.g. ``int((13.02 - 12.34) * 100) == 67``). Mirrors ``convert_rttm_line`` (round 3) ->
    ``extract_frame_info_from_rttm`` (clip, round 3) -> ``get_frame_targets_from_rttm`` (clip, int)."""
    lo, hi = offset, offset + duration
    act: dict[str, np.ndarray] = {}
    for line in rttm_lines:
        f = line.split()
        if not f:
            continue
        s = round(float(f[3]), 3)
        e = round(float(f[4]), 3) + s
        if s > e or not (e > lo and s < hi):
            continue
        s, e = round(max(s, lo), 3), round(min(e, hi), 3)
        if e < offset or s > offset + duration:
            continue
        s, e = max(offset, s), min(offset + duration, e)
        a = act.setdefault(f[7], np.zeros(n_frames, dtype=bool))
        a[int((s - offset) * CS):int((e - offset) * CS)] = True
    return act


def _edges(turns: list[Turn], offset: float) -> set[int]:
    off = round(offset * CS)
    out = set()
    for t in turns:
        out.add(round(t.start * CS) - off)
        out.add(round(t.end * CS) - off)
    return out


def probe(ds, n_samples: int, seed: int = 20260925, compare_audio: bool = True) -> dict:
    """Draw ``n_samples`` real ``__getitem__`` samples and check every contract item.

    Raises ``NemoDataError`` on the first hard violation; returns counts/rates otherwise."""
    import soundfile as sf
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    coll = ds.collection
    if ds.feat_per_sec != CS or abs(ds.diar_frame_length - FRAME_S) > 1e-9:
        raise NemoDataError(f"loader frame rate {ds.feat_per_sec}/s, diar frame {ds.diar_frame_length}s; "
                            f"expected 100/s and 0.01 s")
    if not ds.subsegment_mode:
        raise NemoDataError("dataset not in subsegment_mode")
    max_spks = ds.max_spks
    turn_cache: dict[str, list[Turn]] = {}
    line_cache: dict[str, list[str]] = {}
    audio_cache: dict[str, Any] = {}
    stats = {"n_samples": 0, "n_fallback": 0, "n_empty_targets": 0, "n_two_chunk": 0,
             "n_one_chunk": 0, "active_channels_hist": {}, "n_boundary_offbyone_frames": 0,
             "n_samples_with_boundary_offbyone": 0, "n_unexplained_mismatch_frames": 0,
             "n_columns_in_chunk_arrival_order": 0, "n_arrival_ties": 0,
             "n_first_arrival_unique": 0, "n_audio_chunks": 0, "n_audio_chunks_exact": 0,
             "max_audio_sample_shift": 0, "n_audio_chunks_shifted": 0, "n_audio_eof_padded": 0,
             "max_read_before_span_s": 0.0, "max_read_after_span_s": 0.0, "per_manifest_item": {}}
    spy = _Spy(ds)
    with spy.attach():
        for k in range(n_samples):
            idx = rng.randrange(len(coll))
            item = coll[idx]
            spy.reset()
            audio, alen, targets, tlen, names = ds[idx]
            stats["n_samples"] += 1
            pm = stats["per_manifest_item"].setdefault(str(idx), 0)
            stats["per_manifest_item"][str(idx)] = pm + 1
            T = int(tlen.item()) if hasattr(tlen, "item") else int(tlen)
            if targets.shape != (T, max_spks):
                raise NemoDataError(f"sample {k}: targets {tuple(targets.shape)} != ({T}, {max_spks})")
            if len(names) != max_spks:
                raise NemoDataError(f"sample {k}: {len(names)} speaker names")
            tg = targets.numpy() > 0.5
            if not np.isin(targets.numpy(), (0.0, 1.0)).all():
                raise NemoDataError(f"sample {k}: non-binary targets with soft_targets=false")
            active = tg.any(axis=0)
            n_act = int(active.sum())
            stats["active_channels_hist"][str(n_act)] = stats["active_channels_hist"].get(str(n_act), 0) + 1
            if n_act > max_spks:
                raise NemoDataError(f"sample {k}: {n_act} active channels > {max_spks}")
            if n_act == 0:
                stats["n_empty_targets"] += 1
            if spy.fallbacks:
                stats["n_fallback"] += 1
                if tg.any() or spy.loads or spy.reads:
                    raise NemoDataError(f"sample {k}: fallback with labels or audio reads")
                continue
            if len(spy.loads) != 1:
                raise NemoDataError(f"sample {k}: expected one audio load, saw {len(spy.loads)}")
            ld = spy.loads[0]
            bounds = ld["bounds"]
            offset = float(item.offset)
            if abs(ld["offset"] - offset) > 1e-9:
                raise NemoDataError(f"sample {k}: load offset {ld['offset']} != manifest offset {offset}")
            n_span_frames = int(item.duration * CS)
            stats["n_two_chunk" if len(bounds) == 2 else "n_one_chunk"] += 1
            for s, e in bounds:
                if not (0 <= s < e <= n_span_frames):
                    raise NemoDataError(f"sample {k}: bounds {bounds} outside span frames [0,{n_span_frames}]")
            if len(bounds) == 2 and not (bounds[0][1] <= bounds[1][0] or bounds[1][1] <= bounds[0][0]):
                raise NemoDataError(f"sample {k}: overlapping chunks {bounds}")
            n_frames = sum(e - s for s, e in bounds)
            if T != n_frames:
                raise NemoDataError(f"sample {k}: target length {T} != planned frames {n_frames}")
            if int(alen) != n_frames * (ds.featurizer.sample_rate // CS):
                raise NemoDataError(f"sample {k}: audio length {int(alen)} != {n_frames} frames")
            # ---- labels ----
            rttm = item.rttm_file
            if rttm not in turn_cache:
                turn_cache[rttm] = read_rttm(Path(rttm))
            turns = turn_cache[rttm]
            ref = expected_activity(turns, offset, n_span_frames)
            if rttm not in line_cache:
                line_cache[rttm] = Path(rttm).read_text().splitlines()
            lref = loader_arith_activity(line_cache[rttm], offset, float(item.duration), n_span_frames)
            # The float-truncation reference may deviate from the exact grid only by moving a turn
            # edge E to E-1, i.e. only at frames E-1 of edges E; anything else is a broken reference.
            edges = _edges(turns, offset)
            allowed = np.zeros(n_span_frames, dtype=bool)
            for e_ in edges:
                if 1 <= e_ <= n_span_frames:
                    allowed[e_ - 1] = True
            if set(lref) != set(ref):
                raise NemoDataError(f"sample {k}: truncation reference speakers differ from grid reference")
            for spk in ref:
                if ((lref[spk] != ref[spk]) & ~allowed).any():
                    raise NemoDataError(f"sample {k}: truncation reference for {spk} deviates from the grid "
                                        f"away from a turn edge")
            ref_cat = {spk: np.concatenate([a[s:e] for s, e in bounds]) for spk, a in ref.items()}
            lref_cat = {spk: np.concatenate([a[s:e] for s, e in bounds]) for spk, a in lref.items()}
            ref_active = {spk for spk, a in lref_cat.items() if a.any()}
            named = [n for n in names if n is not None]
            if set(named) != ref_active:
                raise NemoDataError(f"sample {k}: channel speakers {sorted(named)} != RTTM speakers "
                                    f"{sorted(ref_active)} in chunk(s) {bounds}")
            for c, n in enumerate(names):
                if n is None and tg[:, c].any():
                    raise NemoDataError(f"sample {k}: unnamed channel {c} is active")
            # Targets must equal the truncation reference EXACTLY: a systematic shift of even one
            # frame (either direction) moves every non-truncated edge and raises here. Deviations
            # from the exact grid are counted only when explained by float truncation.
            sample_offby = 0
            for c, n in enumerate(names):
                if n is None:
                    continue
                bad = tg[:, c] != lref_cat[n]
                if bad.any():
                    stats["n_unexplained_mismatch_frames"] += int(bad.sum())
                    raise NemoDataError(f"sample {k}: speaker {n} mismatches RTTM at {int(bad.sum())} frames "
                                        f"not explained by float truncation of a turn edge")
                sample_offby += int((tg[:, c] != ref_cat[n]).sum())
            stats["n_boundary_offbyone_frames"] += sample_offby
            stats["n_samples_with_boundary_offbyone"] += int(sample_offby > 0)
            # ---- arrival order (exact against the truncation reference) ----
            arr_tg = {n: int(np.argmax(tg[:, c])) for c, n in enumerate(names) if n is not None}
            arr_ref = {n: int(np.argmax(lref_cat[n])) for n in arr_tg}
            for n in arr_tg:
                if arr_tg[n] != arr_ref[n]:
                    raise NemoDataError(f"sample {k}: speaker {n} arrives at frame {arr_tg[n]}, RTTM says {arr_ref[n]}")
            order_cols = [arr_tg[n] for n in names if n is not None]
            if order_cols == sorted(order_cols):
                stats["n_columns_in_chunk_arrival_order"] += 1
            if len(set(order_cols)) != len(order_cols):
                stats["n_arrival_ties"] += 1
            if order_cols:
                first = min(order_cols)
                if int(tg[first].sum()) == 1:
                    stats["n_first_arrival_unique"] += 1
            # ---- audio: every read inside the span, and equal to the file's samples ----
            span_lo, span_hi = offset, offset + float(item.duration)
            if len(spy.reads) != len(bounds):
                raise NemoDataError(f"sample {k}: {len(spy.reads)} audio reads for {len(bounds)} chunks")
            sr = ds.featurizer.sample_rate
            for (s, e), rd in zip(bounds, spy.reads):
                stats["n_audio_chunks"] += 1
                if rd["audio_file"] != item.audio_file:
                    raise NemoDataError(f"sample {k}: read {rd['audio_file']} not {item.audio_file}")
                seek = int(rd["offset"] * sr)  # exactly what AudioSegment.from_file does
                n_read = rd["n"]
                lo_s = seek / sr - span_lo
                hi_s = (seek + n_read) / sr - span_hi
                stats["max_read_before_span_s"] = max(stats["max_read_before_span_s"], -lo_s)
                stats["max_read_after_span_s"] = max(stats["max_read_after_span_s"], hi_s)
                if lo_s < -1.5 / sr or hi_s > 1.5 / sr:
                    raise NemoDataError(f"sample {k}: read [{seek / sr:.5f}, {(seek + n_read) / sr:.5f}) "
                                        f"outside span [{span_lo}, {span_hi})")
                ideal = round((offset + s / CS) * sr)
                shift = seek - ideal
                stats["max_audio_sample_shift"] = max(stats["max_audio_sample_shift"], abs(shift))
                stats["n_audio_chunks_shifted"] += int(shift != 0)
                if n_read < (e - s) * (sr // CS):
                    stats["n_audio_eof_padded"] += 1
                if compare_audio:
                    f = item.audio_file
                    if f not in audio_cache:
                        audio_cache.clear()
                        audio_cache[f] = sf.SoundFile(f)
                    fh = audio_cache[f]
                    fh.seek(seek)
                    truth = fh.read(n_read, dtype="float32")
                    if not np.array_equal(truth, rd["samples"]):
                        raise NemoDataError(f"sample {k}: loaded audio differs from file at seek {seek}")
                    stats["n_audio_chunks_exact"] += 1
            # the concatenated output waveform equals the concatenated reads (augmentor=None)
            cat = np.concatenate([np.pad(r["samples"], (0, max(0, (e - s) * (sr // CS) - r["n"])))
                                  [: (e - s) * (sr // CS)] for (s, e), r in zip(bounds, spy.reads)])
            if not np.array_equal(cat, audio.numpy()):
                raise NemoDataError(f"sample {k}: returned waveform != concatenated chunk reads")
    for fh in audio_cache.values():
        fh.close()
    n = max(1, stats["n_samples"])
    stats["fallback_rate"] = stats["n_fallback"] / n
    stats["empty_target_rate"] = stats["n_empty_targets"] / n
    stats["two_chunk_rate"] = stats["n_two_chunk"] / n
    stats["n_manifest_items_sampled"] = len(stats["per_manifest_item"])
    return stats


def check_stage2_params(ds, train_ds) -> dict:
    """Assert the instantiated dataset carries the official stage-2 values."""
    want = {"session_len_sec": 105, "max_spks": 8, "soft_targets": False, "subsegment_mode": True,
            "subsegment_single_chunk_min_len_sec": 45.0, "subsegment_two_chunks_rate": 0.5,
            "subsegment_two_chunk_min_len_sec": 22.0, "subsegment_nspk_bias": 1.5,
            "subsegment_start_guard_sec": 0.25, "subsegment_min_first_spk_sec": 0.5,
            "subsegment_splice_silence_sec": 0.1, "feat_per_sec": 100, "diar_frame_length": 0.01}
    got = {k: getattr(ds, k) for k in want}
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        raise NemoDataError(f"dataset params differ from stage-2 yaml: {bad}")
    return got


def balanced_repeats(duration: float, session_len_sec: float = 105.0) -> int:
    """Manifest repeats giving each span ~duration/session_len draws (>= 1)."""
    if not duration > 0 or not session_len_sec > 0:
        raise NemoDataError(f"bad duration {duration} / session_len_sec {session_len_sec}")
    return max(1, int(round(duration / session_len_sec)))


def exposure_stats(entries: list[dict], session_len_sec: float = 105.0) -> dict:
    """Training exposure under NeMo's uniform per-line sampling, one plan (<= session_len_sec of
    audio) per draw. ``entries`` are manifest lines (repeats appear as repeated lines).
    ``sec_per_span_sec`` = expected audio seconds drawn per span second per epoch."""
    if not entries:
        raise NemoDataError("no manifest entries")
    by_span: dict[str, dict] = {}
    for e in entries:
        d = by_span.setdefault(e["span_id"], {"duration": float(e["duration"]), "draws": 0,
                                              "recording_id": e["recording_id"]})
        if abs(d["duration"] - float(e["duration"])) > 1e-9:
            raise NemoDataError(f"span {e['span_id']} listed with different durations")
        d["draws"] += 1
    n_lines = len(entries)
    tot_audio = sum(v["duration"] for v in by_span.values())
    rates = {k: v["draws"] * min(v["duration"], session_len_sec) / v["duration"] for k, v in by_span.items()}
    per_rec: dict[str, dict] = {}
    for k, v in by_span.items():
        r = per_rec.setdefault(v["recording_id"], {"audio_share": 0.0, "draw_share": 0.0})
        r["audio_share"] += v["duration"] / tot_audio
        r["draw_share"] += v["draws"] / n_lines
    long = [k for k, v in by_span.items() if v["duration"] > 1000]
    return {
        "n_lines": n_lines, "n_spans": len(by_span), "session_len_sec": session_len_sec,
        "span_duration_min_s": min(v["duration"] for v in by_span.values()),
        "span_duration_max_s": max(v["duration"] for v in by_span.values()),
        "sec_per_span_sec_min": min(rates.values()), "sec_per_span_sec_max": max(rates.values()),
        "exposure_ratio_max_over_min": max(rates.values()) / min(rates.values()),
        "spans_over_1000s_audio_share": sum(by_span[k]["duration"] for k in long) / tot_audio,
        "spans_over_1000s_draw_share": sum(by_span[k]["draws"] for k in long) / n_lines,
        "per_recording": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in sorted(per_rec.items())},
    }


def write_balanced_manifest(manifest: Path, out_path: Path, session_len_sec: float = 105.0) -> dict:
    """Duration-balanced copy of a one-line-per-span manifest: each line repeated
    ``balanced_repeats(duration)`` times, byte-identical otherwise. Returns exposure stats for
    both manifests."""
    lines = [l for l in Path(manifest).read_text().splitlines() if l.strip()]
    entries = [json.loads(l) for l in lines]
    if len({e["span_id"] for e in entries}) != len(entries):
        raise NemoDataError(f"{manifest}: not one line per span")
    out_lines, out_entries = [], []
    for l, e in zip(lines, entries):
        k = balanced_repeats(float(e["duration"]), session_len_sec)
        out_lines += [l] * k
        out_entries += [e] * k
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text("\n".join(out_lines) + "\n")
    tmp.replace(out_path)
    return {"manifest": str(out_path), "one_line_per_span": exposure_stats(entries, session_len_sec),
            "balanced": exposure_stats(out_entries, session_len_sec)}


def collate_check(dl, n_batches: int = 1) -> list[list[int]]:
    shapes = []
    for i, batch in enumerate(dl):
        audio, flen, targets, tlens, names = batch
        if targets.shape[2] != 8 or targets.shape[0] != audio.shape[0]:
            raise NemoDataError(f"collated batch shapes audio {tuple(audio.shape)} targets {tuple(targets.shape)}")
        shapes.append(list(targets.shape))
        if i + 1 >= n_batches:
            break
    return shapes


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NeMo diarization manifests from spans, verified against the real loader")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spans", type=Path, help="spans.json from bnd.trust")
    g.add_argument("--whole-recordings", action="store_true",
                   help="TEMPORARY untrusted spans = whole recordings (for loader checks only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw-train", type=Path, default=RAW_TRAIN)
    ap.add_argument("--train-rttm", type=Path, default=TRAIN_RTTM_DIR)
    ap.add_argument("--verify-samples", type=int, default=200, help="0 disables the loader check")
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args(argv)

    spans = whole_recording_spans(a.raw_train) if a.whole_recordings else load_spans(a.spans, a.raw_train)
    turns_by_rec, sources = {}, {}
    for rid in sorted({s.recording_id for s in spans}):
        audio = next(s.audio_filepath for s in spans if s.recording_id == rid)
        n, sr = audio_info(audio)
        turns_by_rec[rid], sources[rid] = load_turns(rid, a.train_rttm, a.raw_train, n / sr)
    stats = write_manifest(spans, a.out, turns_by_rec)
    report = {"spans_source": "whole_recordings(TEMPORARY)" if a.whole_recordings else str(a.spans),
              "label_sources": sources, "manifest_stats": stats}
    print(json.dumps({k: v for k, v in stats.items()}, indent=1))
    session_len = float(stage2_config().model.train_ds.session_len_sec)
    report["exposure"] = write_balanced_manifest(Path(stats["manifest"]),
                                                 Path(a.out) / "train_manifest_balanced.json", session_len)
    for name in ("one_line_per_span", "balanced"):
        x = report["exposure"][name]
        print(f"exposure ({name}): {x['n_lines']} lines / {x['n_spans']} spans, audio-s drawn per span-s "
              f"{x['sec_per_span_sec_min']:.3f}..{x['sec_per_span_sec_max']:.3f} "
              f"(ratio {x['exposure_ratio_max_over_min']:.1f}); spans >1000 s hold "
              f"{x['spans_over_1000s_audio_share']:.3f} of audio, {x['spans_over_1000s_draw_share']:.3f} of draws")
    if a.verify_samples > 0:
        ds, dl, train_ds = build_real_dataset(Path(stats["manifest"]))
        report["dataset_params"] = check_stage2_params(ds, train_ds)
        report["probe"] = probe(ds, a.verify_samples, a.seed)
        report["collate_batch_shapes"] = collate_check(dl, 1)
        ds_b = build_real_dataset(Path(report["exposure"]["manifest"]))[0]
        if len(ds_b) != report["exposure"]["balanced"]["n_lines"]:
            raise NemoDataError(f"balanced manifest: loader has {len(ds_b)} items, "
                                f"manifest {report['exposure']['balanced']['n_lines']} lines")
        report["exposure"]["balanced_loader_len"] = len(ds_b)
        p = report["probe"]
        print(f"probe: {p['n_samples']} samples over {p['n_manifest_items_sampled']} spans; "
              f"fallback {p['n_fallback']} ({p['fallback_rate']:.3f}), empty targets {p['n_empty_targets']} "
              f"({p['empty_target_rate']:.3f}), two-chunk {p['n_two_chunk']}, active-channel hist "
              f"{dict(sorted(p['active_channels_hist'].items(), key=lambda x: int(x[0])))}, boundary off-by-one "
              f"frames {p['n_boundary_offbyone_frames']}, audio chunks exact {p['n_audio_chunks_exact']}/"
              f"{p['n_audio_chunks']} (shifted {p['n_audio_chunks_shifted']}, max shift "
              f"{p['max_audio_sample_shift']} samples), columns already in chunk-arrival order "
              f"{p['n_columns_in_chunk_arrival_order']}, first arrival unique {p['n_first_arrival_unique']}")
    rp = Path(a.out) / "nemo_data_report.json"
    rp.write_text(json.dumps(report, indent=1))
    print(f"report: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
