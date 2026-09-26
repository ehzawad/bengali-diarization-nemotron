"""Cross-window speaker linking + final exclusive timeline (docs/PREREG_phase2.md "Linking",
"Final output"). CPU only; never runs a model and never computes DER.

Input is ONLY the shared infer->link cache (read-only) plus theta_clu:

    runs/cache/<system>/<rid>/
        windows.json          [{"window", "start_s", "end_s", "core_start_s", "core_end_s"}]
        meta.json             {"dt", "model_sha256", "nemo_commit", "streaming", "recording_duration_s", "system"}
        probs_<window>.npy    float16 [frames, 8], frame i covers window-relative [i*dt, (i+1)*dt)
        tracks_<theta_act>.json  [{"track", "window", "channel", "segments", "core_segments",
                                   "usable_speech_s", "weak"}]
        emb_<theta_act>.npy   float32 [n_tracks, D], L2-normalised, NaN row = no usable crop
        emb_meta.json

Linking (PRIMARY):
  1. strong tracks = not weak AND a finite embedding. Average-linkage (UPGMA) agglomerative
     clustering on cosine distance, own implementation, with a HARD cannot-link between any two
     tracks of the same window. A cluster carries the set of windows it holds; two clusters may
     merge only if those sets are disjoint. This is checked at EVERY merge (so it holds
     transitively: A~C merged, B cannot-link A => B can never join A∪C). Merging continues while
     the closest allowed pair has distance <= theta_clu, and stops at the first step where it is
     > theta_clu. Ties in distance -> lowest (cluster index) pair.
  2. weak tracks (weak flag, or a non-weak track whose embedding row is NaN -- counted and
     reported as ``strong_nan``). Each weak track's NEAREST strong cluster (average cosine
     distance to that cluster's embedded strong members; ties -> lowest cluster) is found; weak
     tracks are processed in order of that distance (ties -> lowest track id) and attached iff the
     distance <= theta_clu AND the cluster holds no track of the weak track's window at that
     moment (window sets are updated after every attachment). Otherwise -- including when the
     nearest cluster is blocked by cannot-link (no fallback to the 2nd nearest, PREREG literal) --
     it becomes its own speaker. Weak tracks with no embedding are always their own speaker.

Linking (SECONDARY, reported only, ``--variant overlap``): adjacent windows share context audio;
for each adjacent pair the channels are matched with the Hungarian algorithm on activity IoU over
the shared frames (activity = prob >= theta_act); a match is a must-link hint only if IoU >= 0.5
and both channels have >= 0.5 s of activity in the shared region (abstain otherwise). Hints are
applied with union-find BEFORE clustering, each one only if it does not violate a cannot-link.
The resulting groups are then clustered exactly as in the primary variant. (Phase-2 rule,
unchanged; valid only with ``--linker avg``. The withdrawn PREREG_phase3 hint rule and its
min-cluster absorption are NOT part of this module any more.)

Linker ``refine`` (docs/PREREG_phase3b.md "refine (frozen spec)"; ``--linker refine``):
  1. Coactivity cannot-link: two tracks of the SAME window (session) are cannot-linked iff both
     channels have raw p >= theta_act on >= 0.20 s of common frames of that window's input
     (core + context; frames counted on the cached probs, 0.20 s = 20 frames at dt 0.01).
     Tracks of different windows are never cannot-linked (as in phase 2).
  2. Seeds: tracks with usable_speech_s >= 4.0 s (and a finite embedding). Average-linkage (UPGMA)
     AHC on cosine distance, the coactivity cannot-link checked at every merge (transitively: a
     cluster conflicts with everything any member conflicts with), merging while the closest
     allowed pair has distance <= theta_clu (phase-2 stop convention). Ties -> lowest pair.
  3. Reassignment, two passes over ALL tracks with a finite embedding (seeds included). The
     centroid set of pass 1 = the seed clusters; of pass 2 = the non-singleton clusters produced
     by pass 1 (pass-1 singletons are candidates to join, not centroids). Centroid = L2-normalised
     mean of the members' embeddings weighted by usable_speech_s, fixed during a pass and
     recomputed after it. For each track, c* = nearest centroid (cosine distance; ties -> lowest
     centroid index), runner-up c2 (none -> margin +inf). A track is a candidate for c* iff
     d(c*) < theta_clu (strict) AND d(c2) - d(c*) >= 0.05. Candidates are placed in order of
     d(c*) (ties -> lowest track id); a candidate joins c* iff it has no cannot-link with any of
     c*'s members (the tracks whose embeddings define c* this pass, itself excluded) nor with any
     track already placed in c* during this pass, else (no fallback to c2) it is a singleton. A
     current member is therefore never displaced by a coactive newcomer (members are mutually
     cannot-link-free, and a newcomer placed in c* is cannot-link-free with all of them); a
     member that leaves c* this pass still blocks its coactive tracks from c* (literal reading). Tracks
     that are not candidates are singletons. Clusters left empty disappear.
  4. Tracks without a finite embedding: their own speaker (as phase 2). Then the phase-2 exclusive
     decoder below.

Gap-fill post-decoder ``g`` (``--gap-fill g`` seconds, PREREG_phase3b; 0 = off, the identical label
array): on the exclusive frame timeline, a run of silent frames of total length <= g (1e-9 s
tolerance) whose neighbours on BOTH sides are frames of the same speaker is given to that speaker.
Because the timeline is exclusive, "no other speaker has any output in the gap" is exactly "the gap
is silent": a gap between two turns of speaker X that contains any output of another speaker is
never filled, and nothing is filled before the first or after the last turn.

Final output: frames of dt on the recording grid; frame g (centre (g+0.5)*dt) is owned by the
single window whose core contains its centre. Among that window's tracks whose owned
``core_segments`` contain the centre, the linked speaker with the highest probability wins; ties
-> earliest onset (start of the track's core+context segment containing the frame) -> lowest
global speaker id. Consecutive frames of the same speaker form one turn (touching same-speaker
turns across window boundaries merge automatically). Global speaker ids are numbered by the
earliest output onset.

Oracle diagnostics (given a reference RTTM; NOT for outer-fold use by this workflow):
  --oracle link   : every local track takes the reference speaker it overlaps most (over its
                    core+context segments); zero-overlap tracks stay their own speaker.
  --oracle speech : the linked system, but at every frame with reference speech the owning
                    window's arg-max channel over all 8 channels (ignoring theta_act) is output --
                    a channel with no cached track is its own unlinked speaker -- and nothing is
                    output where the reference is silent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from bnd.score import Seg, load_rttm_file, write_rttm

REPO = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO / "runs" / "cache"
REF_DIR = REPO / "data" / "prepared" / "ref_rttm"
OVERLAP_IOU_MIN = 0.5
OVERLAP_MIN_ACTIVE_S = 0.5
LINKERS = ("avg", "refine")
REFINE_COACT_S = 0.20      # PREREG_phase3b refine 1: cannot-link iff >= 0.20 s common confident frames
REFINE_SEED_S = 4.0        # refine 2: seeds = tracks with >= 4.0 s usable embedding speech
REFINE_MARGIN = 0.05       # refine 3: runner-up margin
REFINE_PASSES = 2          # refine 3: two reassignment passes


class LinkError(RuntimeError):
    """Malformed cache / contract violation. Never caught inside this module."""


# ------------------------------------------------------------------------------------------
# cache loading
# ------------------------------------------------------------------------------------------
@dataclass
class Track:
    track: int
    window: int
    channel: int
    segments: list[tuple[float, float]]
    core_segments: list[tuple[float, float]]
    usable_speech_s: float
    weak: bool


@dataclass
class Window:
    window: int
    start_s: float
    end_s: float
    core_start_s: float
    core_end_s: float


@dataclass
class Cache:
    rid: str
    dt: float
    duration: float
    windows: list[Window]
    tracks: list[Track]
    emb: np.ndarray                       # [n_tracks, D]
    probs: dict[int, np.ndarray]          # window -> float32 [frames, 8]
    meta: dict = field(default_factory=dict)
    emb_meta: dict = field(default_factory=dict)


def _find_theta_file(d: Path, prefix: str, suffix: str, theta: float) -> Path:
    """``<prefix><theta><suffix>``: the canonical tag is ``%.2f`` (bnd.infer.theta_tag); other
    spellings of the same value (e.g. a ``0.4`` symlink) are accepted only if they resolve to the
    same file -- two different files for one theta is a contract violation."""
    hits = []
    for p in d.glob(f"{prefix}*{suffix}"):
        tag = p.name[len(prefix):-len(suffix)]
        try:
            v = float(tag)
        except ValueError:
            continue
        if math.isclose(v, theta, abs_tol=1e-9):
            hits.append(p)
    if not hits:
        raise LinkError(f"{d}: no {prefix}<{theta}>{suffix}")
    if len({h.resolve() for h in hits}) != 1:
        raise LinkError(f"{d}: several different files for theta {theta}: {sorted(h.name for h in hits)}")
    canon = d / f"{prefix}{theta:.2f}{suffix}"
    return canon if canon in hits else sorted(hits)[0]


def _pairs(x, what: str) -> list[tuple[float, float]]:
    out = []
    for s in x:
        a, b = float(s[0]), float(s[1])
        if not (math.isfinite(a) and math.isfinite(b)) or b < a:
            raise LinkError(f"bad segment {s!r} in {what}")
        out.append((a, b))
    return out


def load_cache(cache_dir: Path, theta_act: float, rid: str | None = None) -> Cache:
    cache_dir = Path(cache_dir)
    rid = rid or cache_dir.name
    meta = json.loads((cache_dir / "meta.json").read_text())
    for k in ("dt", "recording_duration_s"):
        if k not in meta:
            raise LinkError(f"{cache_dir}/meta.json lacks {k!r}")
    dt, dur = float(meta["dt"]), float(meta["recording_duration_s"])
    if not (dt > 0 and dur > 0):
        raise LinkError(f"{cache_dir}: bad dt/duration {dt}/{dur}")
    windows = [Window(int(w["window"]), float(w["start_s"]), float(w["end_s"]),
                      float(w["core_start_s"]), float(w["core_end_s"]))
               for w in json.loads((cache_dir / "windows.json").read_text())]
    windows.sort(key=lambda w: w.core_start_s)
    check_cores(windows, dur)
    tpath = _find_theta_file(cache_dir, "tracks_", ".json", theta_act)
    epath = _find_theta_file(cache_dir, "emb_", ".npy", theta_act)
    raw = json.loads(tpath.read_text())
    tracks = [Track(int(t["track"]), int(t["window"]), int(t["channel"]),
                    _pairs(t["segments"], f"{tpath} track {t['track']}"),
                    _pairs(t["core_segments"], f"{tpath} track {t['track']}"),
                    float(t["usable_speech_s"]), bool(t["weak"])) for t in raw]
    if [t.track for t in tracks] != list(range(len(tracks))):
        raise LinkError(f"{tpath}: track ids must be 0..n-1 in file order (aligned with emb rows)")
    emb = np.load(epath).astype(np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(tracks):
        raise LinkError(f"{epath}: shape {emb.shape} not aligned with {len(tracks)} tracks")
    check_unit_norm(emb, str(epath))
    wids = {w.window for w in windows}
    probs = {}
    for w in windows:
        p = np.load(cache_dir / f"probs_{w.window}.npy")
        if p.ndim != 2 or p.shape[1] != 8:
            raise LinkError(f"probs_{w.window}.npy shape {p.shape}")
        need = int(round((w.end_s - w.start_s) / dt))
        if abs(p.shape[0] - need) > 2:
            raise LinkError(f"probs_{w.window}.npy has {p.shape[0]} frames, window needs ~{need}")
        probs[w.window] = p.astype(np.float32)
    for t in tracks:
        if t.window not in wids or not 0 <= t.channel < 8:
            raise LinkError(f"track {t.track}: bad window/channel {t.window}/{t.channel}")
        w = next(x for x in windows if x.window == t.window)
        for a, b in t.core_segments:
            if a < w.core_start_s - 1e-6 or b > w.core_end_s + 1e-6:
                raise LinkError(f"track {t.track}: core segment {a,b} outside core of window {w.window}")
    emb_meta = {}
    for cand in (cache_dir / f"emb_{theta_act:.2f}_meta.json", cache_dir / "emb_meta.json"):
        if cand.is_file():
            emb_meta = json.loads(cand.read_text())
            break
    return Cache(rid, dt, dur, windows, tracks, emb, probs, meta, emb_meta)


def check_cores(windows: Sequence[Window], duration: float, tol: float = 1e-6) -> None:
    """Cores must partition [0, duration) exactly once (sorted, contiguous, no gaps/overlap)."""
    if not windows:
        raise LinkError("no windows")
    if abs(windows[0].core_start_s) > tol or abs(windows[-1].core_end_s - duration) > 1e-3:
        raise LinkError(f"cores cover [{windows[0].core_start_s}, {windows[-1].core_end_s}) "
                        f"not [0, {duration})")
    for a, b in zip(windows, windows[1:]):
        if abs(a.core_end_s - b.core_start_s) > tol:
            raise LinkError(f"core gap/overlap between windows {a.window} and {b.window}")
    for w in windows:
        if not (w.start_s - tol <= w.core_start_s < w.core_end_s <= w.end_s + tol):
            raise LinkError(f"window {w.window}: core not inside window input")


# ------------------------------------------------------------------------------------------
# constrained average-linkage AHC
# ------------------------------------------------------------------------------------------
UNIT_NORM_TOL = 1e-3


def check_unit_norm(emb: np.ndarray, what: str = "emb") -> None:
    """Finite embedding rows must be L2-normalised (cache contract). A zero/degenerate row would
    turn cosine distances into NaN and silently stop the clustering, so it fails loudly."""
    if emb.size == 0:
        return
    fin = np.all(np.isfinite(emb), axis=1)
    nrm = np.linalg.norm(emb[fin].astype(np.float64), axis=1)
    bad = np.flatnonzero(fin)[np.abs(nrm - 1.0) > UNIT_NORM_TOL]
    if len(bad):
        raise LinkError(f"{what}: {len(bad)} finite row(s) not unit-norm, e.g. row {int(bad[0])} "
                        f"norm {float(np.linalg.norm(emb[bad[0]])):.6g}")


def cosine_dist(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    if not (np.all(np.isfinite(nrm)) and np.all(nrm > 0)):
        raise LinkError("cosine_dist: non-finite or zero-norm embedding row")
    e = emb / nrm
    return np.clip(1.0 - e @ e.T, 0.0, 2.0).astype(np.float64)


def constrained_ahc(dist: np.ndarray, windows: Sequence[frozenset], sizes: Sequence[int],
                    theta: float, log: list | None = None) -> list[list[int]]:
    """Average-linkage AHC on a precomputed item distance matrix with a hard cannot-link.

    Items may already be groups: ``dist[i, j]`` must be the mean pairwise member distance and
    ``sizes[i]`` the number of embedded members (UPGMA update is then exact). ``windows[i]`` is
    the set of windows held by item i; clusters sharing a window can never merge. Returns
    clusters as sorted lists of item indices, ordered by their smallest item.
    """
    n = len(windows)
    if dist.shape != (n, n) or len(sizes) != n:
        raise LinkError("constrained_ahc: shape mismatch")
    D = dist.astype(np.float64).copy()
    np.fill_diagonal(D, np.inf)
    win = [set(w) for w in windows]
    size = np.asarray(sizes, dtype=np.float64)
    members = [[i] for i in range(n)]
    alive = np.ones(n, dtype=bool)
    conflict = np.zeros((n, n), dtype=bool)
    holders: dict = {}
    for i, w in enumerate(win):
        for x in w:
            holders.setdefault(x, []).append(i)
    for hs in holders.values():
        conflict[np.ix_(hs, hs)] = True
    np.fill_diagonal(conflict, False)
    while True:
        M = np.where(conflict | ~alive[None, :] | ~alive[:, None], np.inf, D)
        k = int(np.argmin(M))  # first minimum in row-major order = lowest (i, j): deterministic
        i, j = divmod(k, n)
        d = M[i, j]
        if not np.isfinite(d) or d > theta:
            break
        i, j = min(i, j), max(i, j)
        if win[i] & win[j]:  # hard check at EVERY merge (independent of the conflict matrix)
            raise LinkError(f"cannot-link violated merging {members[i]} and {members[j]}")
        ni, nj = size[i], size[j]
        newrow = (ni * D[i] + nj * D[j]) / (ni + nj)
        D[i, :] = newrow
        D[:, i] = newrow
        D[i, i] = np.inf
        D[j, :] = np.inf
        D[:, j] = np.inf
        size[i] = ni + nj
        win[i] |= win[j]
        conflict[i, :] |= conflict[j, :]
        conflict[:, i] = conflict[i, :]
        conflict[i, i] = False
        members[i] += members[j]
        members[j] = []
        alive[j] = False
        if log is not None:
            log.append({"merge": [i, j], "dist": float(d)})
    out = [sorted(m) for m, a in zip(members, alive) if a]
    out.sort(key=lambda m: m[0])
    return out


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x


def link_tracks(tracks: Sequence[Track], emb: np.ndarray, theta_clu: float,
                must_link: Sequence[tuple[int, int]] = ()) -> tuple[np.ndarray, dict]:
    """Return (speaker label per track, info). Labels are cluster ids 0..K-1 (arbitrary order;
    the timeline renumbers them). Implements PREREG linking; ``must_link`` = secondary variant."""
    n = len(tracks)
    if emb.shape[0] != n:
        raise LinkError("emb not aligned with tracks")
    check_unit_norm(emb)
    finite = np.all(np.isfinite(emb), axis=1) if n else np.zeros(0, bool)
    weak_flag = np.array([t.weak for t in tracks], dtype=bool)
    strong = finite & ~weak_flag
    info = {"n_tracks": n, "n_strong": int(strong.sum()), "n_weak_flag": int(weak_flag.sum()),
            "strong_nan": int((~weak_flag & ~finite).sum()), "must_link_hints": len(must_link),
            "must_link_applied": 0, "must_link_abstained_cannot_link": 0}
    # ---- groups from must-link hints (union-find, never across a cannot-link)
    uf = _UF(n)
    gwin = {i: {tracks[i].window} for i in range(n)}
    for a, b in sorted(must_link):
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if gwin[ra] & gwin[rb]:
            info["must_link_abstained_cannot_link"] += 1
            continue
        lo, hi = min(ra, rb), max(ra, rb)
        uf.p[hi] = lo
        gwin[lo] |= gwin.pop(hi)
        info["must_link_applied"] += 1
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    glist = [sorted(g) for g in groups.values()]
    glist.sort(key=lambda g: g[0])
    # a group is "strong" if it holds a strong track; its linkage members = its strong tracks
    strong_groups = [g for g in glist if any(strong[i] for i in g)]
    weak_groups = [g for g in glist if not any(strong[i] for i in g)]
    gm = [[i for i in g if strong[i]] for g in strong_groups]
    if strong_groups:
        idx = np.array([i for m in gm for i in m])
        Dt = cosine_dist(emb[idx])
        pos = {t: k for k, t in enumerate(idx)}
        G = len(gm)
        S = np.zeros((G, len(idx)))
        for a, m in enumerate(gm):
            S[a, [pos[t] for t in m]] = 1.0
        cnt = S.sum(1)
        Dg = (S @ Dt @ S.T) / np.outer(cnt, cnt)  # exact mean pairwise member distance
        clusters_g = constrained_ahc(Dg, [frozenset(tracks[i].window for i in g) for g in strong_groups],
                                     [len(m) for m in gm], theta_clu)
    else:
        clusters_g = []
    clusters = [sorted(i for gi in c for i in strong_groups[gi]) for c in clusters_g]
    cl_members = [[i for i in c if strong[i]] for c in clusters]
    cl_win = [set(tracks[i].window for i in c) for c in clusters]
    # ---- weak attachment (PREREG literal): nearest strong cluster, attach iff <= theta and no cannot-link
    pending = [g for g in weak_groups if any(finite[i] for i in g)]
    orphans = [g for g in weak_groups if not any(finite[i] for i in g)]
    n_attached = 0
    n_blocked = 0
    if pending and clusters:
        dmat = np.full((len(pending), len(clusters)), np.inf)
        for a, g in enumerate(pending):
            ge = emb[[i for i in g if finite[i]]]
            ge = ge / np.linalg.norm(ge, axis=1, keepdims=True)
            for k, m in enumerate(cl_members):
                ce = emb[m] / np.linalg.norm(emb[m], axis=1, keepdims=True)
                dmat[a, k] = float(np.clip(1.0 - ge @ ce.T, 0, 2).mean())
        pwin = [{tracks[i].window for i in g} for g in pending]
        nearest = np.argmin(dmat, axis=1)  # first minimum -> lowest cluster index on ties
        order = sorted(range(len(pending)), key=lambda x: (dmat[x, nearest[x]], x))
        done = np.zeros(len(pending), dtype=bool)
        for a in order:
            k = int(nearest[a])
            if dmat[a, k] <= theta_clu and not (pwin[a] & cl_win[k]):
                clusters[k] = sorted(clusters[k] + pending[a])
                cl_win[k] |= pwin[a]
                done[a] = True
                n_attached += 1
            elif dmat[a, k] <= theta_clu:
                n_blocked += 1
        rest = [g for a, g in enumerate(pending) if not done[a]]
    else:
        rest = pending
    all_clusters = clusters + rest + orphans
    labels = np.full(n, -1, dtype=np.int64)
    for k, c in enumerate(all_clusters):
        for i in c:
            labels[i] = k
    assert (labels >= 0).all()
    info.update({"n_strong_clusters": len(clusters), "weak_groups": len(weak_groups),
                 "weak_attached": n_attached, "weak_blocked_cannot_link": n_blocked, "weak_own_speaker": len(rest) + len(orphans),
                 "n_speakers_linked": len(all_clusters)})
    assert_cannot_link(tracks, labels)
    return labels, info


def assert_cannot_link(tracks: Sequence[Track], labels: np.ndarray) -> None:
    seen = {}
    for t, l in zip(tracks, labels):
        key = (int(l), t.window)
        if key in seen:
            raise LinkError(f"cannot-link violated: tracks {seen[key]} and {t.track} share speaker {l} "
                            f"in window {t.window}")
        seen[key] = t.track


# ------------------------------------------------------------------------------------------
# secondary: adjacent-window overlap correspondence
# ------------------------------------------------------------------------------------------
def overlap_hints(cache: Cache, theta_act: float, iou_min: float = OVERLAP_IOU_MIN,
                  min_active_s: float = OVERLAP_MIN_ACTIVE_S) -> tuple[list[tuple[int, int]], dict]:
    from scipy.optimize import linear_sum_assignment

    dt = cache.dt
    by_wc = {(t.window, t.channel): t.track for t in cache.tracks}
    hints, n_pairs, n_abst = [], 0, 0
    for w1, w2 in zip(cache.windows, cache.windows[1:]):
        a, b = max(w1.start_s, w2.start_s), min(w1.end_s, w2.end_s)
        if b - a < dt:
            continue
        g = np.arange(int(math.ceil(a / dt - 1e-9)), int(math.floor(b / dt + 1e-9)))
        c = (g + 0.5) * dt
        c = c[(c > a) & (c < b)]
        p1 = cache.probs[w1.window]
        p2 = cache.probs[w2.window]
        i1 = np.clip(np.floor((c - w1.start_s) / dt).astype(int), 0, len(p1) - 1)
        i2 = np.clip(np.floor((c - w2.start_s) / dt).astype(int), 0, len(p2) - 1)
        A1 = p1[i1] >= theta_act  # [F, 8]
        A2 = p2[i2] >= theta_act
        inter = A1.T.astype(np.float64) @ A2.astype(np.float64)
        union = A1.sum(0)[:, None] + A2.sum(0)[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        r, cc = linear_sum_assignment(-iou)
        for x, y in zip(r, cc):
            n_pairs += 1
            ok = (iou[x, y] >= iou_min and A1[:, x].sum() * dt >= min_active_s
                  and A2[:, y].sum() * dt >= min_active_s)
            if ok and (w1.window, x) in by_wc and (w2.window, y) in by_wc:
                hints.append((by_wc[(w1.window, x)], by_wc[(w2.window, y)]))
            else:
                n_abst += 1
    return hints, {"overlap_pairs": n_pairs, "overlap_hints": len(hints), "overlap_abstained": n_abst}


# ------------------------------------------------------------------------------------------
# linker "refine" (PREREG_phase3b, frozen spec)
# ------------------------------------------------------------------------------------------
def coactivity_cannot_link(cache: Cache, theta_act: float, min_common_s: float = REFINE_COACT_S) -> np.ndarray:
    """bool [n_tracks, n_tracks]: True iff the two tracks belong to the same window and both
    channels have raw p >= theta_act on >= ``min_common_s`` of common frames of that window."""
    n = len(cache.tracks)
    cl = np.zeros((n, n), dtype=bool)
    by_w: dict[int, list[Track]] = {}
    for t in cache.tracks:
        by_w.setdefault(t.window, []).append(t)
    need = min_common_s / cache.dt - 1e-6  # frames
    for w, trs in by_w.items():
        if len(trs) < 2:
            continue
        A = (cache.probs[w] >= theta_act).astype(np.int64)  # [F, 8]
        common = A.T @ A                                       # [8, 8] common active frames
        for i, a in enumerate(trs):
            for b in trs[i + 1:]:
                if common[a.channel, b.channel] >= need:
                    cl[a.track, b.track] = cl[b.track, a.track] = True
    return cl


def constrained_ahc_cl(dist: np.ndarray, cl: np.ndarray, sizes: Sequence[int], theta: float,
                       log: list | None = None) -> list[list[int]]:
    """Average-linkage AHC (exact UPGMA on a precomputed distance matrix) with an arbitrary
    symmetric cannot-link matrix ``cl`` between items, propagated to clusters (a cluster conflicts
    with whatever any member conflicts with) and re-checked against the ORIGINAL item matrix at
    every merge. Merges while the closest allowed pair is <= theta. Clusters are returned as sorted
    item lists ordered by their smallest item."""
    n = dist.shape[0]
    if dist.shape != (n, n) or cl.shape != (n, n) or len(sizes) != n:
        raise LinkError("constrained_ahc_cl: shape mismatch")
    D = dist.astype(np.float64).copy()
    np.fill_diagonal(D, np.inf)
    cl0 = np.asarray(cl, dtype=bool)
    conflict = cl0.copy()
    np.fill_diagonal(conflict, False)
    size = np.asarray(sizes, dtype=np.float64)
    members = [[i] for i in range(n)]
    alive = np.ones(n, dtype=bool)
    while n:
        M = np.where(conflict | ~alive[None, :] | ~alive[:, None], np.inf, D)
        k = int(np.argmin(M))
        i, j = divmod(k, n)
        d = M[i, j]
        if not np.isfinite(d) or d > theta:
            break
        i, j = min(i, j), max(i, j)
        if cl0[np.ix_(members[i], members[j])].any():  # hard check, independent of `conflict`
            raise LinkError(f"cannot-link violated merging {members[i]} and {members[j]}")
        ni, nj = size[i], size[j]
        newrow = (ni * D[i] + nj * D[j]) / (ni + nj)
        D[i, :] = newrow
        D[:, i] = newrow
        D[i, i] = np.inf
        D[j, :] = np.inf
        D[:, j] = np.inf
        size[i] = ni + nj
        conflict[i, :] |= conflict[j, :]
        conflict[:, i] = conflict[i, :]
        conflict[i, i] = False
        members[i] += members[j]
        members[j] = []
        alive[j] = False
        if log is not None:
            log.append({"merge": [i, j], "dist": float(d)})
    out = [sorted(m) for m, a in zip(members, alive) if a]
    out.sort(key=lambda m: m[0])
    return out


def weighted_centroid(emb: np.ndarray, members: Sequence[int], weights: np.ndarray) -> np.ndarray:
    """L2-normalised usable-seconds-weighted mean of the (unit) member embeddings."""
    rows = emb[list(members)].astype(np.float64)
    w = weights[list(members)].astype(np.float64)
    if not np.all(np.isfinite(rows)) or not (w > 0).all():
        raise LinkError("weighted_centroid: non-finite member embedding or non-positive weight")
    m = (rows * w[:, None]).sum(0)
    nrm = np.linalg.norm(m)
    if not (nrm > 0 and np.isfinite(nrm)):
        raise LinkError("weighted_centroid: degenerate centroid")
    return m / nrm


def refine_link(tracks: Sequence[Track], emb: np.ndarray, theta_clu: float, cl: np.ndarray,
                seed_s: float = REFINE_SEED_S, margin: float = REFINE_MARGIN,
                passes: int = REFINE_PASSES) -> tuple[np.ndarray, dict]:
    """PREREG_phase3b linker ``refine`` (module doc). ``cl`` = coactivity cannot-link matrix.
    Returns (speaker label per track, info)."""
    n = len(tracks)
    if emb.shape[0] != n or cl.shape != (n, n):
        raise LinkError("refine_link: emb / cannot-link not aligned with tracks")
    check_unit_norm(emb)
    finite = np.all(np.isfinite(emb), axis=1) if n else np.zeros(0, bool)
    usable = np.array([t.usable_speech_s for t in tracks], dtype=np.float64)
    seeds = np.flatnonzero(finite & (usable >= seed_s - 1e-9))
    info = {"linker": "refine", "n_tracks": n, "n_embedded": int(finite.sum()), "n_no_embedding": int((~finite).sum()),
            "n_seeds": int(len(seeds)), "cannot_link_pairs": int(np.triu(cl, 1).sum()), "passes": []}
    if len(seeds):
        seed_cl = constrained_ahc_cl(cosine_dist(emb[seeds]), cl[np.ix_(seeds, seeds)], [1] * len(seeds), theta_clu)
        clusters = [[int(seeds[i]) for i in c] for c in seed_cl]
    else:
        clusters = []
    info["n_seed_clusters"] = len(clusters)
    emb_idx = np.flatnonzero(finite)
    prev = {t: k for k, c in enumerate(clusters) for t in c}
    singletons: list[int] = list(emb_idx) if not clusters else []
    for p in range(passes):
        if not clusters:  # nothing left to attract tracks: the previous singletons stand
            break
        cents = np.stack([weighted_centroid(emb, c, usable) for c in clusters])  # [K, D]
        E = emb[emb_idx].astype(np.float64)
        Dm = np.clip(1.0 - E @ cents.T, 0.0, 2.0)                                  # [T, K]
        order_k = np.argsort(Dm, axis=1, kind="stable")                            # ties -> lowest centroid
        best = order_k[:, 0]
        d1 = Dm[np.arange(len(emb_idx)), best]
        d2 = Dm[np.arange(len(emb_idx)), order_k[:, 1]] if len(clusters) > 1 else np.full(len(emb_idx), np.inf)
        cand = (d1 < theta_clu) & (d2 - d1 >= margin - 1e-12)
        new = [[] for _ in clusters]
        singletons = []
        st = {"pass": p + 1, "centroids": len(clusters), "no_match_threshold": int((d1 >= theta_clu).sum()),
              "no_match_margin": int(((d1 < theta_clu) & ~cand).sum()), "blocked_cannot_link": 0}
        for x in sorted(np.flatnonzero(cand), key=lambda x: (d1[x], int(emb_idx[x]))):
            t, k = int(emb_idx[x]), int(best[x])
            # "no cannot-link with c*'s members": the members that define c* this pass (other
            # than t itself) AND every track already placed in c* during this pass
            block = [m for m in clusters[k] if m != t] + new[k]
            if block and cl[t, block].any():
                singletons.append(t)
                st["blocked_cannot_link"] += 1
            else:
                new[k].append(t)
        singletons += [int(emb_idx[x]) for x in np.flatnonzero(~cand)]
        newmap = {t: k for k, c in enumerate(new) for t in c}  # indexed like the centroids (= prev)
        st.update(assigned=len(newmap), singletons=len(singletons),
                  moved=int(sum(prev.get(int(t)) != newmap.get(int(t)) for t in emb_idx)))
        clusters = [sorted(c) for c in new if c]
        st["clusters_after"] = len(clusters)
        prev = {t: k for k, c in enumerate(clusters) for t in c}
        info["passes"].append(st)
    labels = np.full(n, -1, dtype=np.int64)
    k = 0
    for c in clusters:
        labels[c] = k
        k += 1
    for t in sorted(set(singletons)) + [int(i) for i in np.flatnonzero(~finite)]:
        if labels[t] >= 0:
            raise LinkError(f"refine: track {t} assigned twice")
        labels[t] = k
        k += 1
    if (labels < 0).any():
        raise LinkError("refine: unassigned tracks")
    info.update(n_clusters=len(clusters), n_singletons=len(set(singletons)), n_speakers_linked=int(k))
    assert_pair_cannot_link(cl, labels)
    return labels, info


def assert_pair_cannot_link(cl: np.ndarray, labels: np.ndarray) -> None:
    i, j = np.nonzero(np.triu(cl, 1))
    bad = labels[i] == labels[j]
    if bad.any():
        x = int(np.flatnonzero(bad)[0])
        raise LinkError(f"cannot-link violated: tracks {int(i[x])} and {int(j[x])} share speaker {int(labels[i[x]])}")


# ------------------------------------------------------------------------------------------
# gap-fill post-decoder (PREREG_phase3b g)
# ------------------------------------------------------------------------------------------
def gap_fill(lab: np.ndarray, dt: float, g: float) -> np.ndarray:
    """Fill silent runs of <= g seconds bounded on both sides by the same speaker (module doc).
    ``g <= 0`` returns ``lab`` itself (no copy)."""
    if g <= 0:
        return lab
    out = np.array(lab, copy=True)
    n = len(out)
    sil = np.concatenate([[False], out < 0, [False]])
    d = np.flatnonzero(sil[1:] != sil[:-1])
    for a, b in zip(d[::2], d[1::2]):  # silent run [a, b)
        if a == 0 or b >= n:
            continue
        if out[a - 1] == out[b] and (b - a) * dt <= g + 1e-9:
            out[a:b] = out[a - 1]
    return out


# ------------------------------------------------------------------------------------------
# exclusive timeline
# ------------------------------------------------------------------------------------------
def n_frames(duration: float, dt: float) -> int:
    x = duration / dt
    return int(round(x)) if abs(x - round(x)) < 1e-6 else int(math.ceil(x))


def _mask_from_segments(c: np.ndarray, segs: Sequence[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(c), dtype=bool)
    for a, b in segs:
        m |= (c >= a) & (c < b)
    return m


def _onset_from_segments(c: np.ndarray, segs, core_segs) -> np.ndarray:
    on = np.full(len(c), np.inf)
    for a, b in sorted(core_segs):  # fallback: core segment start
        sel = (c >= a) & (c < b)
        on[sel] = a
    for a, b in sorted(segs, reverse=True):  # earliest containing segment wins
        sel = (c >= a) & (c < b)
        on[sel] = a
    return on


def frame_labels(cache: Cache, labels: np.ndarray, speech: np.ndarray | None = None) -> np.ndarray:
    """Per-frame global label (-1 = silence) on the recording grid.

    Default: exclusive single-label from owned core activity (PREREG). ``speech`` (bool [n_frames])
    switches to the oracle-speech-mask mode: at speech frames the arg-max over all 8 channels of
    the owning window is output (ignoring theta_act: channels with no cached track -- they never
    crossed theta_act -- are their own unlinked speaker, one per (window, channel)), nothing
    elsewhere. Windows with no track at all still output at reference-speech frames."""
    dt = cache.dt
    N = n_frames(cache.duration, dt)
    out = np.full(N, -1, dtype=np.int64)
    owner = np.full(N, -1, dtype=np.int64)
    by_w: dict[int, list[Track]] = {}
    for t in cache.tracks:
        by_w.setdefault(t.window, []).append(t)
    centres = (np.arange(N) + 0.5) * dt
    if speech is not None and len(speech) != N:
        raise LinkError(f"speech mask has {len(speech)} frames, recording grid has {N}")
    next_free = int(labels.max()) + 1 if len(labels) else 0
    for w in cache.windows:
        g0 = int(np.searchsorted(centres, w.core_start_s, side="left"))
        g1 = int(np.searchsorted(centres, w.core_end_s, side="left"))
        if w is cache.windows[-1]:
            g1 = N
        if g1 <= g0:
            continue
        if (owner[g0:g1] >= 0).any():
            raise LinkError(f"frame owned twice (window {w.window})")
        owner[g0:g1] = w.window
        c = centres[g0:g1]
        p = cache.probs[w.window]
        li = np.floor((c - w.start_s) / dt + 1e-9).astype(int)
        if li.min() < 0 or li.max() > len(p):
            raise LinkError(f"window {w.window}: core frames outside its probs ({li.min()}..{li.max()} vs {len(p)})")
        li = np.minimum(li, len(p) - 1)
        tr = sorted(by_w.get(w.window, []), key=lambda t: t.channel)
        if speech is not None:
            # oracle speech: arg-max over ALL 8 channels (theta_act ignored). A channel with no
            # cached track (never crossed theta_act) is an unlinked local track = its own speaker.
            ch_spk = np.empty(8, dtype=np.int64)
            for ch in range(8):
                ch_spk[ch] = next_free
                next_free += 1
            for t in tr:
                ch_spk[t.channel] = labels[t.track]
            best = np.argmax(p[li, :], axis=1)  # first max = lowest channel on exact ties
            sp = speech[g0:g1]
            out[g0:g1] = np.where(sp, ch_spk[best], -1)
            continue
        if not tr:
            continue
        spk = np.array([labels[t.track] for t in tr])
        P = np.stack([p[li, t.channel] for t in tr])  # [K, F]
        act = np.stack([_mask_from_segments(c, t.core_segments) for t in tr])
        onset = np.stack([_onset_from_segments(c, t.segments, t.core_segments) for t in tr])
        Pm = np.where(act, P, -np.inf)
        maxp = Pm.max(axis=0)
        cand = act & (Pm == maxp[None, :])
        on_m = np.where(cand, onset, np.inf)
        min_on = on_m.min(axis=0)
        cand &= on_m == min_on[None, :]
        big = np.iinfo(np.int64).max
        s_m = np.where(cand, spk[:, None], big)
        best = s_m.min(axis=0)
        out[g0:g1] = np.where(act.any(axis=0), best, -1)
    if (owner < 0).any():
        raise LinkError(f"{int((owner < 0).sum())} frame(s) owned by no window")
    return out


def frames_to_segs(lab: np.ndarray, dt: float, duration: float) -> list[Seg]:
    """Runs of equal labels -> Segs (touching same-speaker turns are one run). Global speaker ids
    are renumbered by earliest onset -> SPEAKER_0, SPEAKER_1, ..."""
    segs = []
    n = len(lab)
    if n == 0:
        return segs
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    order: dict[int, int] = {}
    for s, e in zip(starts, ends):
        l = int(lab[s])
        if l < 0:
            continue
        if l not in order:
            order[l] = len(order)
        segs.append(Seg(round(s * dt, 6), round(min(e * dt, duration), 6), f"SPEAKER_{order[l]}"))
    return segs


def floor_adapter(segs: Sequence[Seg]) -> list[Seg]:
    """Bengali-Loop benchmark adapter: floor both boundaries to whole seconds (the references' timestamp
    convention), drop empty turns, keep the timeline exclusive (an earlier turn is truncated by a later one)."""
    fl = [Seg(float(math.floor(s.start)), float(math.floor(s.end)), s.speaker) for s in segs
          if math.floor(s.end) > math.floor(s.start)]
    out: list[Seg] = []
    for s in sorted(fl, key=lambda s: (s.start, s.end)):
        if out and s.start < out[-1].end:
            p = out[-1]
            out[-1] = Seg(p.start, s.start, p.speaker)
            if out[-1].end <= out[-1].start:
                out.pop()
        if s.end > s.start:
            out.append(s)
    return out


# ------------------------------------------------------------------------------------------
# oracle diagnostics
# ------------------------------------------------------------------------------------------
def _overlap(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    tot = 0.0
    for x0, x1 in a:
        for y0, y1 in b:
            tot += max(0.0, min(x1, y1) - max(x0, y0))
    return tot


def oracle_link_labels(tracks: Sequence[Track], ref: Sequence[Seg]) -> np.ndarray:
    by_spk: dict[str, list[tuple[float, float]]] = {}
    for s in ref:
        by_spk.setdefault(s.speaker, []).append((s.start, s.end))
    names = sorted(by_spk)
    labels = np.empty(len(tracks), dtype=np.int64)
    nxt = len(names)
    for t in tracks:
        ov = [_overlap(t.segments, by_spk[s]) for s in names]
        if names and max(ov) > 0:
            labels[t.track] = int(np.argmax(ov))  # ties -> first name in sorted order
        else:
            labels[t.track] = nxt
            nxt += 1
    return labels


def ref_speech_mask(ref: Sequence[Seg], duration: float, dt: float) -> np.ndarray:
    N = n_frames(duration, dt)
    c = (np.arange(N) + 0.5) * dt
    return _mask_from_segments(c, [(s.start, s.end) for s in ref if s.end > s.start])


# ------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------
def track_labels(cache: Cache, theta_act: float, theta_clu: float, linker: str = "avg", oracle: str = "none",
                 ref: Sequence[Seg] | None = None, must_link: Sequence[tuple[int, int]] = ()) -> tuple[np.ndarray, dict]:
    """Speaker label per local track: oracle link (reference), linker refine, or linker avg (phase 2)."""
    if oracle == "link":
        if ref is None:
            raise LinkError("oracle link needs a reference")
        return oracle_link_labels(cache.tracks, ref), {"oracle": "link"}
    if linker == "refine":
        if must_link:
            raise LinkError("must-link hints are not defined for linker refine")
        return refine_link(cache.tracks, cache.emb, theta_clu, coactivity_cannot_link(cache, theta_act))
    if linker != "avg":
        raise LinkError(f"unknown linker {linker}")
    return link_tracks(cache.tracks, cache.emb, theta_clu, must_link)


def run_recording(cache: Cache, theta_act: float, theta_clu: float, variant: str = "primary",
                  oracle: str = "none", ref: Sequence[Seg] | None = None, linker: str = "avg",
                  gap_fill_s: float = 0.0) -> tuple[list[Seg], dict]:
    """Link one recording and decode it. ``linker="avg"`` + ``gap_fill_s=0`` is the phase-2 system
    (byte-identical output); ``linker="refine"`` and ``gap_fill_s>0`` are the PREREG_phase3b factors.
    With ``oracle="link"`` the linker is replaced by reference labels (tracks, decoder and gap-fill
    unchanged)."""
    if linker not in LINKERS:
        raise LinkError(f"unknown linker {linker}")
    if gap_fill_s < 0:
        raise LinkError(f"gap_fill_s must be >= 0, got {gap_fill_s}")
    hints, hinfo = [], {}
    if variant == "overlap":
        if linker != "avg":
            raise LinkError("the phase-2 overlap variant is defined for linker avg only")
        hints, hinfo = overlap_hints(cache, theta_act)
    elif variant != "primary":
        raise LinkError(f"unknown variant {variant}")
    labels, info = track_labels(cache, theta_act, theta_clu, linker, oracle, ref, hints)
    info.update(hinfo)
    speech = None
    if oracle == "speech":
        if ref is None:
            raise LinkError("oracle speech needs a reference")
        speech = ref_speech_mask(ref, cache.duration, cache.dt)
        info["oracle"] = "speech"
    elif oracle not in ("none", "link"):
        raise LinkError(f"unknown oracle {oracle}")
    lab = frame_labels(cache, labels, speech)
    if gap_fill_s > 0:
        filled = gap_fill(lab, cache.dt, gap_fill_s)
        info["gap_fill_s"] = gap_fill_s
        info["gap_filled_s"] = round(float(((lab < 0) & (filled >= 0)).sum() * cache.dt), 3)
        lab = filled
    segs = frames_to_segs(lab, cache.dt, cache.duration)
    info.update({"n_output_speakers": len({s.speaker for s in segs}), "n_output_turns": len(segs),
                 "output_speech_s": round(sum(s.end - s.start for s in segs), 3)})
    info["track_speaker"] = [int(x) for x in labels]
    return segs, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--theta-act", type=float, required=True)
    ap.add_argument("--theta-clu", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--session-subdir", default=None,
                    help="cache layout <cache-root>/<system>/<session-subdir>/<rid> (e.g. s80, PREREG_phase3b)")
    ap.add_argument("--variant", choices=("primary", "overlap"), default="primary",
                    help="overlap = the phase-2 registered secondary variant (linker avg only)")
    ap.add_argument("--linker", choices=LINKERS, default="avg")
    ap.add_argument("--gap-fill", type=float, default=0.0, help="PREREG_phase3b gap-fill g (seconds); 0 = off")
    ap.add_argument("--oracle", choices=("none", "link", "speech"), default="none")
    ap.add_argument("--ref-dir", type=Path, default=REF_DIR)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"system": a.system, "theta_act": a.theta_act, "theta_clu": a.theta_clu,
              "variant": a.variant, "oracle": a.oracle, "recordings": {}}
    if a.linker != "avg" or a.gap_fill or a.session_subdir:
        report.update(linker=a.linker, gap_fill_s=a.gap_fill, session_subdir=a.session_subdir)
    for rid in a.recordings:
        sysdir = a.cache_root / a.system / a.session_subdir if a.session_subdir else a.cache_root / a.system
        cache = load_cache(sysdir / rid, a.theta_act, rid)
        if cache.meta.get("system") not in (None, a.system):
            raise LinkError(f"{rid}: cache meta system {cache.meta.get('system')!r} != {a.system!r}")
        ref = load_rttm_file(a.ref_dir / f"{rid}.rttm", rid) if a.oracle != "none" else None
        segs, info = run_recording(cache, a.theta_act, a.theta_clu, a.variant, a.oracle, ref,
                                   linker=a.linker, gap_fill_s=a.gap_fill)
        write_rttm(a.out / f"{rid}.rttm", rid, segs)
        info["cache_meta"] = {k: cache.meta.get(k) for k in ("model_sha256", "nemo_commit", "dt")}
        info["emb_meta"] = cache.emb_meta
        report["recordings"][rid] = info
        print(f"{rid}: tracks={info.get('n_tracks')} strong={info.get('n_strong')} "
              f"speakers={info['n_output_speakers']} turns={info['n_output_turns']} "
              f"speech={info['output_speech_s']}s", file=sys.stderr)
    tmp = a.out / "link_report.json.tmp"
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(a.out / "link_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
