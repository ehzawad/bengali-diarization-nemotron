"""Label-timing variant ``shift`` (PREREG_phase2 "Label timing").

delta estimator (training recordings ONLY)
    For each training recording: label mask = union of its training-RTTM turns on the 10 ms grid;
    detector speech = frames where BOTH the zero-shot Nemotron max-prob >= 0.5 AND Silero >= 0.5
    (``runs/probe_alignment/{probs,silero}/<rid>.npz``, resampled with ``bnd.trust``). For every
    shift s in -1.00 ... +1.00 s (10 ms steps) the label mask is translated by s (s > 0 = labels
    moved LATER) and the frame agreement ``mean(label[t - s] == speech[t])`` is computed over the
    SAME frame range for every s (the first and last ``max_shift`` frames are excluded so no shift
    reads outside the recording). The recording's shift is the arg-max (ties: smallest |s|, then
    the negative one). delta = median over recordings, rounded to the 10 ms grid (half up).

shifted RTTM writer
    Every per-span training RTTM turn: start += delta (clipped to the span start), end unchanged;
    turns whose shifted duration is <= 0.1 s are dropped. A new manifest points at the shifted
    RTTMs; ``num_speakers`` is recomputed. The scoring reference is never touched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from bnd import nemo_data as nd
from bnd import trust

REPO = Path(__file__).resolve().parents[1]
PROBS_DIR = REPO / "runs/probe_alignment/probs"
SILERO_DIR = REPO / "runs/probe_alignment/silero"
MAX_SHIFT_S = 1.0
STEP_S = 0.01
MIN_TURN_S = 0.1
THRESH = trust.THRESH  # 0.5


class TimingError(ValueError):
    pass


def _grid_turns(turns, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for t in turns:
        m[trust.sec_to_frame(t.start, n):trust.sec_to_frame(t.end, n)] = True
    return m


def agreement_curve(lab: np.ndarray, speech: np.ndarray, max_shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Shifts (frames, -max_shift..+max_shift) and frame agreement of label[t - s] with speech[t]
    over t in [max_shift, n - max_shift)."""
    lab = np.asarray(lab, bool)
    speech = np.asarray(speech, bool)
    if lab.shape != speech.shape or lab.ndim != 1:
        raise TimingError(f"mask shapes differ: {lab.shape} vs {speech.shape}")
    n = lab.size
    if n <= 2 * max_shift + 1:
        raise TimingError(f"recording too short ({n} frames) for max shift {max_shift}")
    shifts = np.arange(-max_shift, max_shift + 1)
    tgt = speech[max_shift:n - max_shift]
    agree = np.empty(shifts.size)
    for i, s in enumerate(shifts):
        agree[i] = np.count_nonzero(lab[max_shift - s:n - max_shift - s] == tgt) / tgt.size
    return shifts, agree


def best_shift(lab: np.ndarray, speech: np.ndarray, max_shift: int = round(MAX_SHIFT_S / STEP_S)) -> dict:
    shifts, agree = agreement_curve(lab, speech, max_shift)
    best = agree.max()
    cand = shifts[agree == best]
    s = int(sorted(cand.tolist(), key=lambda x: (abs(x), x))[0])
    return {"shift_frames": s, "shift_s": round(s * STEP_S, 2), "agreement": float(best),
            "agreement_at_0": float(agree[shifts == 0][0]),
            "n_tied": int(cand.size)}


def detector_speech(nemo: np.ndarray, nemo_dt: float, silero: np.ndarray, silero_dt: float, n: int) -> np.ndarray:
    nemo = np.asarray(nemo)
    nemo_max = nemo.max(axis=1) if nemo.ndim == 2 else nemo
    ng = trust.resample_to_grid(nemo_max, nemo_dt, n, "nemotron")
    sg = trust.resample_to_grid(np.asarray(silero), silero_dt, n, "silero")
    return (ng >= THRESH) & (sg >= THRESH)


def recording_shift(rid: str, duration: float, turns, probs_dir: Path = PROBS_DIR,
                    silero_dir: Path = SILERO_DIR) -> dict:
    """Best shift for one recording; missing detector files raise FileNotFoundError."""
    n = trust.n_grid(duration)
    nemo, ndt = trust.load_detector(Path(probs_dir) / f"{rid}.npz")
    sil, sdt = trust.load_detector(Path(silero_dir) / f"{rid}.npz")
    speech = detector_speech(nemo, ndt, sil, sdt, n)
    lab = _grid_turns(turns, n)
    r = best_shift(lab, speech)
    r.update(recording_id=rid, label_s=float(lab.sum() * trust.GRID_DT),
             detector_speech_s=float(speech.sum() * trust.GRID_DT))
    return r


def median_delta(shifts_s: list[float]) -> tuple[float, float]:
    """(raw median, median rounded half-up to the 10 ms grid)."""
    if not shifts_s:
        raise TimingError("no recordings for delta estimation")
    med = float(np.median(np.asarray(shifts_s, float)))
    return med, round(math.floor(round(med / STEP_S, 6) + 0.5) * STEP_S, 2)


def estimate_delta(train_recs: list[str], probs_dir: Path = PROBS_DIR, silero_dir: Path = SILERO_DIR,
                   train_rttm_dir: Path = nd.TRAIN_RTTM_DIR, raw_train: Path = nd.RAW_TRAIN) -> dict:
    """delta from the given (training) recordings only."""
    if len(set(train_recs)) != len(train_recs):
        raise TimingError(f"duplicate recordings {train_recs}")
    per = []
    for rid in train_recs:
        frames, sr = nd.audio_info(str(Path(raw_train) / "audio" / f"{rid}.wav"))
        duration = frames / sr
        turns, _src = nd.load_turns(rid, train_rttm_dir, raw_train, duration)
        per.append(recording_shift(rid, duration, turns, probs_dir, silero_dir))
    med, delta = median_delta([p["shift_s"] for p in per])
    return {"delta_s": round(delta, 2), "median_raw_s": med, "recordings": list(train_recs),
            "per_recording": per, "search_s": [-MAX_SHIFT_S, MAX_SHIFT_S, STEP_S]}


# ----------------------------------------------------------------------------------------------
# shifted per-span RTTMs
# ----------------------------------------------------------------------------------------------

def shift_turns(turns, delta_s: float, span_start: float, min_turn_s: float = MIN_TURN_S):
    """Starts moved later by delta (clipped at the span start), ends unchanged, turns <= min_turn_s
    dropped. Returns (turns, n_dropped)."""
    out, dropped = [], 0
    for t in turns:
        s = max(round(t.start + delta_s, 6), span_start)
        if t.end - s <= min_turn_s + 1e-9:
            dropped += 1
            continue
        out.append(nd.Turn(s, t.end, t.speaker))
    out.sort(key=lambda x: (x.start, x.end, x.speaker))
    return out, dropped


def write_shifted_manifest(entries: list[dict], delta_s: float, out_dir: Path) -> tuple[list[dict], dict]:
    """For manifest ``entries`` (dicts; repeated lines allowed) write ``out_dir/<span_id>.rttm``
    shifted once per span and return new entries pointing at them."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, tuple[str, int]] = {}
    stats = {"delta_s": delta_s, "n_spans": 0, "n_turns_in": 0, "n_turns_out": 0, "n_dropped": 0}
    new = []
    for e in entries:
        sid = e["span_id"]
        if sid not in done:
            turns = nd.read_rttm(Path(e["rttm_filepath"]))
            sh, dropped = shift_turns(turns, delta_s, float(e["offset"]))
            p = out_dir / f"{sid}.rttm"
            p.write_text("".join(nd.rttm_line(e["recording_id"], t) for t in sh))
            done[sid] = (str(p), len({t.speaker for t in sh}))
            stats["n_spans"] += 1
            stats["n_turns_in"] += len(turns)
            stats["n_turns_out"] += len(sh)
            stats["n_dropped"] += dropped
        e2 = dict(e)
        e2["rttm_filepath"], e2["num_speakers"] = done[sid]
        new.append(e2)
    return new, stats


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="estimate the label-timing delta from training recordings")
    ap.add_argument("--recs", required=True, help="comma-separated recording ids (001 or train_001)")
    a = ap.parse_args(argv)
    recs = [r if r.startswith("train_") else f"train_{r}" for r in a.recs.split(",") if r]
    print(json.dumps(estimate_delta(recs), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
