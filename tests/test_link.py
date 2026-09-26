"""Tests for bnd/link.py on synthetic caches that follow the shared infer->link contract."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bnd import link as L
from bnd.score import Seg, read_rttm, score_recording

DT = 0.01


# ------------------------------------------------------------------ synthetic cache builder
def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def make_cache(tmp: Path, rid: str, turns, duration, dirs, core=20.0, ctx=5.0, theta_act=0.5,
               hi=0.9, lo=0.05, noise=0.0, weak_s=1.0, seed=0, nan_tracks=(), dim=16):
    """turns: [(start, end, spk)] global truth; dirs: spk -> unit embedding. Channels in each
    window are assigned by arrival order inside the window input (<= 8 asserted)."""
    rng = np.random.default_rng(seed)
    d = tmp / rid
    d.mkdir(parents=True)
    N = L.n_frames(duration, DT)
    windows, tracks, embs = [], [], []
    k, cs = 0, 0.0
    while cs < duration - 1e-9:
        ce = min(cs + core, duration)
        ws, we = max(0.0, cs - ctx), min(duration, ce + ctx)
        windows.append({"window": k, "start_s": ws, "end_s": we, "core_start_s": cs, "core_end_s": ce})
        nf = int(round((we - ws) / DT))
        c = ws + (np.arange(nf) + 0.5) * DT
        spk_on = {}
        for s, e, spk in turns:
            if e > ws and s < we:
                spk_on[spk] = min(spk_on.get(spk, 1e9), max(s, ws))
        order = sorted(spk_on, key=lambda x: (spk_on[x], x))
        assert len(order) <= 8, "synthetic window has > 8 speakers"
        P = np.full((nf, 8), lo, dtype=np.float32)
        for ch, spk in enumerate(order):
            m = np.zeros(nf, bool)
            for s, e, sp in turns:
                if sp == spk:
                    m |= (c >= s) & (c < e)
            P[m, ch] = hi
        np.save(d / f"probs_{k}.npy", P.astype(np.float16))
        for ch in range(8):
            act = P[:, ch] >= theta_act
            segs = [(round(ws + a * DT, 6), round(ws + b * DT, 6)) for a, b in _runs(act) if (b - a) * DT >= 0.1]
            core_segs = [(max(a, cs), min(b, ce)) for a, b in segs if min(b, ce) > max(a, cs)]
            usable = sum(b - a for a, b in segs)
            tid = len(tracks)
            tracks.append({"track": tid, "window": k, "channel": ch, "segments": [list(x) for x in segs],
                           "core_segments": [list(x) for x in core_segs], "usable_speech_s": usable,
                           "weak": usable < weak_s})
            if ch < len(order) and usable > 0 and tid not in nan_tracks:
                v = dirs[order[ch]] + noise * rng.standard_normal(dim)
                embs.append(v / np.linalg.norm(v))
            else:
                embs.append(np.full(dim, np.nan))
        k += 1
        cs = ce
    (d / "windows.json").write_text(json.dumps(windows))
    (d / "meta.json").write_text(json.dumps({"dt": DT, "model_sha256": "x", "nemo_commit": "y",
                                             "streaming": {}, "recording_duration_s": duration,
                                             "system": "synth"}))
    (d / f"tracks_{theta_act:.2f}.json").write_text(json.dumps(tracks))
    np.save(d / f"emb_{theta_act:.2f}.npy", np.asarray(embs, dtype=np.float32))
    (d / "emb_meta.json").write_text(json.dumps({"model": "synthetic"}))
    return d


def unit_dirs(n, dim=16, seed=1):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return {f"s{i}": q[:, i] for i in range(n)}


def truth_frames(turns, duration, names):
    N = L.n_frames(duration, DT)
    c = (np.arange(N) + 0.5) * DT
    lab = np.full(N, -1)
    for s, e, spk in turns:
        lab[(c >= s) & (c < e)] = names.index(spk)
    return lab


def mapped_accuracy(hyp_segs, turns, duration, names):
    """Fraction of frames where hyp == truth under the best 1:1 mapping (hungarian)."""
    from scipy.optimize import linear_sum_assignment

    t = truth_frames(turns, duration, names)
    hn = sorted({s.speaker for s in hyp_segs})
    h = truth_frames([(s.start, s.end, s.speaker) for s in hyp_segs], duration, hn)
    C = np.zeros((len(names), len(hn)))
    for i in range(len(names)):
        for j in range(len(hn)):
            C[i, j] = ((t == i) & (h == j)).sum()
    r, cc = linear_sum_assignment(-C)
    mp = {j: i for i, j in zip(r, cc)}
    hm = np.array([mp.get(x, -2) if x >= 0 else -1 for x in h])
    return float((hm == t).mean())


def mk_tracks(windows, weak=None):
    weak = weak or [False] * len(windows)
    return [L.Track(i, w, i % 8, [], [], 0.0 if wk else 5.0, wk) for i, (w, wk) in enumerate(zip(windows, weak))]


# ------------------------------------------------------------------ cannot-link
def test_cannot_link_blocks_transitive_merge():
    # A, B in window 0; C in window 1; all identical embeddings (distance 0), theta huge
    e = np.tile(np.eye(4)[0], (3, 1)).astype(np.float32)
    lab, info = L.link_tracks(mk_tracks([0, 0, 1]), e, theta_clu=2.0)
    assert lab[0] == lab[2] and lab[1] != lab[0]  # A~C merged (lowest pair), B can never join
    assert info["n_speakers_linked"] == 2


def test_cannot_link_randomised_never_violated():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(3, 40))
        wins = rng.integers(0, max(1, n // 3), size=n).tolist()
        e = rng.standard_normal((n, 8)).astype(np.float32)
        if trial % 3 == 0:
            e[:] = e[0]  # everything identical: only the constraint stops merging
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        tr = mk_tracks(wins, weak=(rng.random(n) < 0.3).tolist())
        lab, info = L.link_tracks(tr, e, theta_clu=2.0)
        for k in set(lab.tolist()):
            ws = [wins[i] for i in range(n) if lab[i] == k]
            assert len(ws) == len(set(ws))
        if trial % 3 == 0 and not any(t.weak for t in tr):
            assert info["n_speakers_linked"] == max(np.bincount(wins))


def test_constrained_ahc_hard_check_is_independent():
    # an inconsistent caller cannot sneak a violation through: windows sets are re-checked
    D = np.array([[0, .1], [.1, 0]])
    assert L.constrained_ahc(D, [frozenset({0}), frozenset({0})], [1, 1], 1.0) == [[0], [1]]


# ------------------------------------------------------------------ threshold stop / average linkage
def test_threshold_stop_and_average_linkage():
    D = np.array([[0, .1, .3], [.1, 0, .5], [.3, .5, 0]])
    W = [frozenset({i}) for i in range(3)]
    assert L.constrained_ahc(D, W, [1, 1, 1], 0.09) == [[0], [1], [2]]
    assert L.constrained_ahc(D, W, [1, 1, 1], 0.1) == [[0, 1], [2]]  # inclusive at theta
    # average linkage: d({0,1},2) = (0.3+0.5)/2 = 0.4 (single linkage would give 0.3)
    assert L.constrained_ahc(D, W, [1, 1, 1], 0.39) == [[0, 1], [2]]
    assert L.constrained_ahc(D, W, [1, 1, 1], 0.40) == [[0, 1, 2]]


def test_upgma_size_weighting_matches_bruteforce():
    rng = np.random.default_rng(3)
    e = rng.standard_normal((9, 5))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    D = L.cosine_dist(e)
    log = []
    cl = L.constrained_ahc(D, [frozenset({i}) for i in range(9)], [1] * 9, 0.9, log)
    # every merge distance equals the brute-force mean pairwise distance between the two clusters
    members = {i: [i] for i in range(9)}
    for m in log:
        i, j = m["merge"]
        bf = D[np.ix_(members[i], members[j])].mean()
        assert math.isclose(bf, m["dist"], abs_tol=1e-9)
        members[i] = members[i] + members[j]
    assert sorted(sum(cl, [])) == list(range(9))


# ------------------------------------------------------------------ weak attachment
def test_weak_attachment_rules():
    x, y = np.eye(4)[0], np.eye(4)[1]
    near = (x + 0.1 * y) / np.linalg.norm(x + 0.1 * y)
    e = np.array([x, x, y, near, near, y * -1, np.full(4, np.nan), np.full(4, np.nan)], dtype=np.float32)
    #            t0 w0  t1 w1 t2 w2 t3 weak w3   t4 weak w0  t5 weak w4  t6 weak NaN  t7 strong NaN
    tr = mk_tracks([0, 1, 2, 3, 0, 4, 5, 6], weak=[False, False, False, True, True, True, True, False])
    lab, info = L.link_tracks(tr, e, theta_clu=0.3)
    assert lab[0] == lab[1] != lab[2]
    assert lab[3] == lab[0]                      # near + no conflict -> attached
    assert lab[4] not in (lab[0], lab[2])        # nearest cluster holds window 0 -> own speaker
    assert lab[5] not in (lab[0], lab[2])        # too far -> own speaker
    assert len({lab[5], lab[6], lab[7], lab[4]}) == 4
    assert info["strong_nan"] == 1 and info["weak_attached"] == 1 and info["weak_blocked_cannot_link"] == 1


def test_weak_attachment_updates_window_sets():
    # two weak tracks of the SAME window both nearest to one cluster: only the closer one joins
    x, y = np.eye(4)[0], np.eye(4)[1]
    w1 = (x + 0.05 * y); w1 /= np.linalg.norm(w1)
    w2 = (x + 0.2 * y); w2 /= np.linalg.norm(w2)
    e = np.array([x, w2, w1], dtype=np.float32)
    lab, _ = L.link_tracks(mk_tracks([0, 1, 1], weak=[False, True, True]), e, theta_clu=0.3)
    assert lab[2] == lab[0] and lab[1] != lab[0]


# ------------------------------------------------------------------ timeline
def _two_track_cache(tmp_path, p_a, p_b, seg_a, seg_b):
    d = tmp_path / "r"
    d.mkdir(parents=True)
    dur = 4.0
    (d / "windows.json").write_text(json.dumps([{"window": 0, "start_s": 0, "end_s": dur,
                                                  "core_start_s": 0, "core_end_s": dur}]))
    (d / "meta.json").write_text(json.dumps({"dt": DT, "recording_duration_s": dur}))
    P = np.zeros((400, 8), np.float16)
    P[:, 0], P[:, 1] = p_a, p_b
    np.save(d / "probs_0.npy", P)
    tr = [{"track": 0, "window": 0, "channel": 0, "segments": [seg_a], "core_segments": [seg_a],
           "usable_speech_s": 3, "weak": False},
          {"track": 1, "window": 0, "channel": 1, "segments": [seg_b], "core_segments": [seg_b],
           "usable_speech_s": 3, "weak": False}]
    (d / "tracks_0.5.json").write_text(json.dumps(tr))
    np.save(d / "emb_0.5.npy", np.eye(2, 4, dtype=np.float32))
    return L.load_cache(d, 0.5)


def test_tie_break_earliest_onset_then_lowest_id(tmp_path):
    # equal probs everywhere; track 1 starts earlier -> it owns the overlap
    c = _two_track_cache(tmp_path, 0.8, 0.8, [1.0, 3.0], [0.5, 2.0])
    lab = L.frame_labels(c, np.array([0, 1]))
    g = int(1.5 / DT)
    assert lab[g] == 1
    # equal onsets -> lowest global speaker id
    c2 = _two_track_cache(tmp_path / "b", 0.8, 0.8, [1.0, 3.0], [1.0, 2.0])
    assert L.frame_labels(c2, np.array([5, 3]))[g] == 3
    assert L.frame_labels(c2, np.array([3, 5]))[g] == 3
    # higher probability beats onset
    c3 = _two_track_cache(tmp_path / "c", 0.9, 0.8, [1.0, 3.0], [0.5, 2.0])
    assert L.frame_labels(c3, np.array([0, 1]))[g] == 0
    # deterministic: identical output on repeat
    a = L.frame_labels(c, np.array([0, 1]))
    assert np.array_equal(a, L.frame_labels(c, np.array([0, 1])))


def test_core_ownership_no_dup_no_gap(tmp_path):
    # one speaker talking through the whole file: all windows (whose contexts overlap) see it;
    # linked into one speaker, the output must be exactly one turn [0, dur) -- no duplicated
    # context time, no hole at core boundaries.
    dur = 67.3
    dirs = unit_dirs(1)
    d = make_cache(tmp_path, "r", [(0.0, dur, "s0")], dur, dirs)
    c = L.load_cache(d, 0.5)
    segs, info = L.run_recording(c, 0.5, 0.45)
    assert len(segs) == 1 and segs[0].start == 0 and math.isclose(segs[0].end, dur, abs_tol=1e-6)
    # output never overlaps itself and sums to the owned time
    lab = L.frame_labels(c, np.array(info["track_speaker"]))
    assert len(lab) == L.n_frames(dur, DT) and (lab >= 0).all()


def test_check_cores_rejects_gap_and_overlap():
    W = L.Window
    with pytest.raises(L.LinkError):
        L.check_cores([W(0, 0, 25, 0, 20), W(1, 15, 40, 20.5, 40)], 40)
    with pytest.raises(L.LinkError):
        L.check_cores([W(0, 0, 25, 0, 20), W(1, 15, 40, 19.0, 40)], 40)
    with pytest.raises(L.LinkError):
        L.check_cores([W(0, 0, 25, 0, 20), W(1, 15, 40, 20, 39)], 40)
    L.check_cores([W(0, 0, 25, 0, 20), W(1, 15, 40, 20, 40)], 40)


def test_frames_to_segs_merges_touching_and_numbers_by_onset():
    lab = np.array([-1, 7, 7, 2, 2, 7, -1, 7])
    s = L.frames_to_segs(lab, 1.0, 8.0)
    assert [(x.start, x.end, x.speaker) for x in s] == [
        (1, 3, "SPEAKER_0"), (3, 5, "SPEAKER_1"), (5, 6, "SPEAKER_0"), (7, 8, "SPEAKER_0")]


# ------------------------------------------------------------------ >8 global speakers
def _many_speaker_turns(n_spk=12, dur=200.0, seed=5):
    """Sequential turns (2-6 s) cycling through speakers in blocks so every 30 s window input
    holds <= 8 speakers, but the recording holds n_spk."""
    rng = np.random.default_rng(seed)
    turns, t, i = [], 0.0, 0
    block = [f"s{j}" for j in range(n_spk)]
    while t < dur - 1:
        grp = block[(i // 12 * 4) % n_spk:][:4] or block[:4]
        spk = grp[i % len(grp)] if len(grp) else block[0]
        L_ = round(float(rng.uniform(2.0, 6.0)), 2)
        e = min(dur, round(t + L_, 2))
        turns.append((t, e, spk))
        t = round(e + float(rng.choice([0.0, 0.3])), 2)
        i += 1
    return turns


def test_more_than_8_speakers_reconstructed(tmp_path):
    dur = 200.0
    turns = _many_speaker_turns(dur=dur)
    names = sorted({s for _, _, s in turns})
    assert len(names) > 8
    dirs = unit_dirs(len(names))
    d = make_cache(tmp_path, "r", turns, dur, {n: dirs[f"s{k}"] for k, n in enumerate(names)} | dirs,
                   noise=0.05)
    c = L.load_cache(d, 0.5)
    assert all(len({t.channel for t in c.tracks if t.window == w.window and t.segments}) <= 8
               for w in c.windows)
    segs, info = L.run_recording(c, 0.5, 0.45)
    assert info["n_output_speakers"] == len(names)
    assert mapped_accuracy(segs, turns, dur, names) > 0.999
    ref = [Seg(s, e, n) for s, e, n in turns]
    r = score_recording(ref, segs, dur, 0.0, "r")
    assert r.der < 1e-3


def test_cli_end_to_end_writes_rttm(tmp_path):
    dur = 90.0
    turns = [(0.0, 12.0, "s0"), (12.0, 30.0, "s1"), (31.0, 55.0, "s0"), (55.0, 90.0, "s2")]
    make_cache(tmp_path / "cache" / "synth", "rec1", turns, dur, unit_dirs(3), noise=0.05)
    out = tmp_path / "hyp"
    rc = L.main(["--system", "synth", "--recordings", "rec1", "--theta-act", "0.5", "--theta-clu",
                 "0.45", "--out", str(out), "--cache-root", str(tmp_path / "cache")])
    assert rc == 0
    segs = read_rttm(out / "rec1.rttm")["rec1"]
    assert [(s.start, s.end) for s in segs] == [(0, 12), (12, 30), (31, 55), (55, 90)]
    assert segs[0].speaker == segs[2].speaker != segs[1].speaker
    rep = json.loads((out / "link_report.json").read_text())
    assert rep["recordings"]["rec1"]["n_output_speakers"] == 3


def test_missing_theta_file_fails_loudly(tmp_path):
    make_cache(tmp_path, "r", [(0, 10, "s0")], 20.0, unit_dirs(1))
    with pytest.raises(L.LinkError):
        L.load_cache(tmp_path / "r", 0.6)


def test_theta_alias_symlink_ok_but_conflicting_copy_fails(tmp_path):
    d = make_cache(tmp_path, "r", [(0, 10, "s0")], 20.0, unit_dirs(1))
    (d / "tracks_0.5.json").symlink_to("tracks_0.50.json")  # what bnd.infer writes
    L.load_cache(d, 0.5)
    (d / "tracks_0.5.json").unlink()
    (d / "tracks_0.5.json").write_text("[]")
    with pytest.raises(L.LinkError):
        L.load_cache(d, 0.5)


# ------------------------------------------------------------------ secondary + oracles
def test_overlap_hints_follow_true_correspondence(tmp_path):
    dur = 80.0
    turns = [(0, 18.0, "s0"), (18.0, 23.0, "s1"), (23.0, 44.0, "s0"), (44.0, 80.0, "s1")]
    d = make_cache(tmp_path, "r", turns, dur, unit_dirs(2), noise=0.0)
    c = L.load_cache(d, 0.5)
    hints, info = L.overlap_hints(c, 0.5)
    assert info["overlap_hints"] > 0
    # every hinted pair must connect tracks of the same true speaker: same embedding direction
    for a, b in hints:
        assert np.allclose(c.emb[a], c.emb[b], atol=1e-6)
    # with hints, the primary clustering result is unchanged on this easy case
    s1, _ = L.run_recording(c, 0.5, 0.45, "primary")
    s2, i2 = L.run_recording(c, 0.5, 0.45, "overlap")
    assert s1 == s2 and i2["must_link_applied"] > 0


def test_must_link_hint_never_breaks_cannot_link():
    e = np.eye(3, 4, dtype=np.float32)
    lab, info = L.link_tracks(mk_tracks([0, 0, 1]), e, 0.1, must_link=[(0, 2), (1, 2)])
    assert lab[0] == lab[2] != lab[1]
    assert info["must_link_abstained_cannot_link"] == 1


def test_oracle_link_and_speech(tmp_path):
    dur = 60.0
    turns = [(0.0, 20.0, "s0"), (20.0, 40.0, "s1"), (41.0, 60.0, "s0")]
    # embeddings deliberately useless (all identical): real linking would merge within cannot-link
    dirs = {k: np.eye(16)[0] for k in ("s0", "s1")}
    d = make_cache(tmp_path, "r", turns, dur, dirs)
    c = L.load_cache(d, 0.5)
    ref = [Seg(s, e, n) for s, e, n in turns]
    segs, _ = L.run_recording(c, 0.5, 0.45, oracle="link", ref=ref)
    assert score_recording(ref, segs, dur, 0.0, "r").der < 1e-3
    segs2, _ = L.run_recording(c, 0.5, 0.45, oracle="speech", ref=ref)
    sp = sum(s.end - s.start for s in segs2)
    assert math.isclose(sp, 59.0, abs_tol=0.02)


# ------------------------------------------------------------------ review regressions
def _drop_tracks(d: Path, keep):
    """Rewrite a synthetic cache keeping only tracks for which keep(track_dict) is True
    (mimics the real infer cache, which writes tracks only for channels that crossed theta_act)."""
    tp = d / "tracks_0.50.json"
    ep = d / "emb_0.50.npy"
    tr = json.loads(tp.read_text())
    em = np.load(ep)
    idx = [i for i, t in enumerate(tr) if keep(t)]
    new = []
    for k, i in enumerate(idx):
        t = dict(tr[i])
        t["track"] = k
        new.append(t)
    tp.write_text(json.dumps(new))
    np.save(ep, em[idx])


def test_oracle_speech_ignores_theta_act_untracked_channels(tmp_path):
    """Regression: oracle speech must take the arg-max over ALL 8 channels, including channels
    that have no track (under theta_act), and must output in windows with no track at all."""
    dur = 40.0
    turns = [(0.0, 20.0, "s0"), (20.0, 40.0, "s1")]
    dirs = unit_dirs(2)
    d = make_cache(tmp_path, "r", turns, dur, dirs)
    # window 1 (core 20-40): s1's channel probability 0.45 (< theta_act 0.5) -> no track there
    p1 = np.load(d / "probs_1.npy").astype(np.float32)
    ch = int(np.argmax(p1.max(axis=0)))
    p1[:, ch] = np.where(p1[:, ch] > 0.5, 0.45, p1[:, ch])
    np.save(d / "probs_1.npy", p1.astype(np.float16))
    _drop_tracks(d, lambda t: bool(t["segments"]) and t["window"] != 1)
    c = L.load_cache(d, 0.5)
    assert not [t for t in c.tracks if t.window == 1]  # window 1 has no track at all
    ref = [Seg(s, e, n) for s, e, n in turns]
    segs, _ = L.run_recording(c, 0.5, 0.45, oracle="speech", ref=ref)
    assert math.isclose(sum(s.end - s.start for s in segs), 40.0, abs_tol=0.02)
    assert score_recording(ref, segs, dur, 0.0, "r").der < 2e-3
    # arg-max over an untracked channel with higher prob than a tracked one
    labels = np.array([7] * len(c.tracks))
    p0 = c.probs[0]
    tracked = {t.channel for t in c.tracks if t.window == 0}
    un = next(k for k in range(8) if k not in tracked)
    p0[:, un] = 0.95
    lab = L.frame_labels(c, labels, np.ones(L.n_frames(dur, DT), bool))
    assert (lab[:2000] != 7).all() and (lab[:2000] >= 0).all()


def test_zero_norm_embedding_fails_loudly():
    """Regression: a finite zero-norm row used to make cosine NaN and silently stop clustering."""
    tr = mk_tracks([0, 1, 2, 3])
    e = np.array([[1, 0], [1, 0], [1, 0.01], [0, 0]], np.float32)
    e[2] /= np.linalg.norm(e[2])
    with pytest.raises(L.LinkError):
        L.link_tracks(tr, e, 0.45)
    lab, _ = L.link_tracks(tr[:3], e[:3], 0.45)
    assert list(lab) == [0, 0, 0]
    with pytest.raises(L.LinkError):
        L.cosine_dist(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(L.LinkError):  # non-unit (e.g. un-normalised) rows violate the contract
        L.link_tracks(tr[:2], np.array([[2.0, 0.0], [1.0, 0.0]], np.float32), 0.45)


# ------------------------------------------------------------------ PREREG_phase3b: phase-2 default unchanged
REPO = Path(__file__).resolve().parents[1]
P2_FROZEN = REPO / "runs/cv/frozen.json"
P2_LINK_MARKERS = REPO / "runs/cv/done/link"


@pytest.mark.skipif(not P2_FROZEN.exists(), reason="phase-2 frozen artefacts not present")
@pytest.mark.parametrize("system", ["ft_sel", "zs_sel", "ft_strict"])
def test_phase2_frozen_rttms_regenerate_byte_identical(system, tmp_path):
    """Regression (read-only): linker avg, g=0 regenerates every phase-2 frozen outer RTTM from its cache."""
    import hashlib

    from bnd.score import write_rttm

    fz = json.loads(P2_FROZEN.read_text())
    for rid, e in fz["systems"][system].items():
        a, c = map(float, e["cell"].split("_"))
        cache = L.load_cache(REPO / "runs/cv/cache" / e["source_system"] / rid, a, rid)
        segs, _ = L.run_recording(cache, a, c, "primary", linker="avg", gap_fill_s=0.0)
        out = tmp_path / f"{rid}.rttm"
        write_rttm(out, rid, segs)
        assert hashlib.sha256(out.read_bytes()).hexdigest() == e["sha256"], f"{system}/{rid}"


@pytest.mark.skipif(not P2_LINK_MARKERS.is_dir(), reason="phase-2 link markers not present")
@pytest.mark.parametrize("hyp", ["cv_inner_x12_lr5e-5_raw/0.40_0.55", "cv_inner_x45_lr5e-5_shift/0.50_0.65",
                                 "zeroshot/0.40_0.55"])
def test_phase2_inner_rttm_regenerates_byte_identical_via_cli(hyp, tmp_path):
    """Regression (read-only): a phase-2 INNER hypothesis dir regenerated through the CLI with explicit
    ``--linker avg --gap-fill 0`` is byte-identical to the registered files and their link-marker hashes."""
    import hashlib

    system, cell = hyp.split("/")
    m = json.loads((P2_LINK_MARKERS / f"hyp__{system}__{cell}.json").read_text())
    rids = sorted(m["outputs"]["rttm_sha256"])
    if system == "zeroshot":
        rids = rids[:3]  # keep the test short; all 10 are covered by the frozen test above
    a, c = cell.split("_")
    out = tmp_path / "o"
    L.main(["--system", system, "--recordings", *rids, "--theta-act", a, "--theta-clu", c, "--out", str(out),
            "--cache-root", str(REPO / "runs/cv/cache"), "--linker", "avg", "--gap-fill", "0"])
    for rid in rids:
        new = (out / f"{rid}.rttm").read_bytes()
        assert hashlib.sha256(new).hexdigest() == m["outputs"]["rttm_sha256"][rid], f"{hyp}/{rid}"
        assert new == (REPO / "runs/cv/hyp" / system / cell / f"{rid}.rttm").read_bytes()
