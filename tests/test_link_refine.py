"""PREREG_phase3b linker ``refine`` (frozen spec) and the gap-fill post-decoder, on synthetic data (CPU)."""
import math

import numpy as np
import pytest

from bnd import link as L
from test_link import make_cache, unit_dirs

DT = 0.01


def unit(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r), 0.0, 0.0], np.float32)


def trk(i, window, usable, channel=None):
    return L.Track(i, window, i % 8 if channel is None else channel, [], [], usable, usable < 1.0)


def no_cl(n):
    return np.zeros((n, n), bool)


# ------------------------------------------------------------------ 1. coactivity cannot-link
def cache_two_channels(common_frames: int, theta=0.25, second_window=False):
    """Window 0 (and optionally window 1), 30 s input; ch0 active 0-10 s, ch1 active from 10 s -
    common_frames*dt; both have a track."""
    F = 3000
    P = np.full((F, 8), 0.05, np.float32)
    P[:1000, 0] = theta          # inclusive threshold
    P[1000 - common_frames:2000, 1] = 0.9
    wins = [L.Window(0, 0.0, 30.0, 0.0, 20.0)]
    probs = {0: P}
    tracks = [L.Track(0, 0, 0, [(0.0, 10.0)], [(0.0, 10.0)], 5.0, False),
              L.Track(1, 0, 1, [(10.0 - common_frames * DT, 20.0)], [(10.0 - common_frames * DT, 20.0)], 5.0, False)]
    if second_window:
        wins = [L.Window(0, 0.0, 25.0, 0.0, 20.0), L.Window(1, 15.0, 30.0, 20.0, 30.0)]
        P2 = P[1500:].copy()
        probs = {0: P[:2500], 1: P2}
        tracks.append(L.Track(2, 1, 0, [(15.0, 25.0)], [(20.0, 25.0)], 5.0, False))
    emb = np.stack([unit(0), unit(0)] + ([unit(0)] if second_window else []))
    return L.Cache("r", DT, 30.0, wins, tracks, emb, probs)


def test_coactivity_cannot_link_needs_0p20s_common_confident_frames():
    assert not L.coactivity_cannot_link(cache_two_channels(19), 0.25)[0, 1]
    cl = L.coactivity_cannot_link(cache_two_channels(20), 0.25)
    assert cl[0, 1] and cl[1, 0] and not cl.diagonal().any()
    # raw p >= theta is inclusive; a higher theta removes channel 0's activity entirely
    assert not L.coactivity_cannot_link(cache_two_channels(40), 0.26)[0, 1]


def test_coactivity_never_links_across_windows():
    c = cache_two_channels(40, second_window=True)
    cl = L.coactivity_cannot_link(c, 0.25)
    assert cl[0, 1] and not cl[0, 2] and not cl[1, 2]


def test_refine_merges_same_window_tracks_without_coactivity():
    """The phase-2 linker forbids any same-window pair; refine only forbids coactive pairs."""
    tr = [trk(0, 0, 5.0, 0), trk(1, 0, 5.0, 1)]
    emb = np.stack([unit(0), unit(5)])
    lab_avg, _ = L.link_tracks(tr, emb, 0.55)
    assert lab_avg[0] != lab_avg[1]
    lab, info = L.refine_link(tr, emb, 0.55, no_cl(2))
    assert lab[0] == lab[1] and info["n_seeds"] == 2 and info["n_seed_clusters"] == 1
    cl = no_cl(2)
    cl[0, 1] = cl[1, 0] = True
    lab, _ = L.refine_link(tr, emb, 0.55, cl)
    assert lab[0] != lab[1]


# ------------------------------------------------------------------ 2. seeds
def test_seeds_are_tracks_with_at_least_4s_usable_and_cannot_link_checked_in_ahc():
    tr = [trk(0, 0, 4.0), trk(1, 1, 3.99), trk(2, 2, 6.0), trk(3, 3, 8.0)]
    emb = np.stack([unit(0), unit(2), unit(1), unit(90)])
    cl = no_cl(4)
    cl[0, 2] = cl[2, 0] = True  # the two closest seeds are coactive -> never one seed cluster
    lab, info = L.refine_link(tr, emb, 0.3, cl)
    assert info["n_seeds"] == 3  # 3.99 s is not a seed
    assert info["n_seed_clusters"] == 3  # 0 and 2 blocked; 3 is far
    assert lab[0] != lab[2]
    # track 1 (non-seed) is nearest to seed 0 (2 deg) but seed 2 is 1 deg away from it: margin < 0.05 -> singleton
    assert lab[1] not in (lab[0], lab[2], lab[3])


def test_transitive_cannot_link_in_seed_ahc():
    tr = [trk(0, 0, 5.0), trk(1, 1, 5.0), trk(2, 2, 5.0)]
    emb = np.stack([unit(0), unit(3), unit(6)])
    cl = no_cl(3)
    cl[0, 2] = cl[2, 0] = True
    log = []
    out = L.constrained_ahc_cl(L.cosine_dist(emb), cl, [1, 1, 1], 0.5, log)
    assert sorted(map(sorted, out)) == [[0, 1], [2]]
    assert len(log) == 1


# ------------------------------------------------------------------ 3. reassignment
def two_seed_world(track_deg, usable=1.5):
    """Seeds at 0 and 60 degrees (8 s each, different windows) + one probe track."""
    tr = [trk(0, 0, 8.0), trk(1, 1, 8.0), trk(2, 2, usable)]
    emb = np.stack([unit(0), unit(60), unit(track_deg)])
    return tr, emb


def test_reassignment_runner_up_margin():
    # nearest at 28 deg vs 32 deg: d = 0.1171 vs 0.1340 -> margin 0.017 < 0.05 -> singleton
    tr, emb = two_seed_world(28)  # seeds are 0.5 apart: separate seed clusters at theta 0.45
    lab, info = L.refine_link(tr, emb, 0.45, no_cl(3))
    assert info["n_seed_clusters"] == 2
    assert lab[2] not in (lab[0], lab[1]) and info["passes"][0]["no_match_margin"] == 1
    # 10 deg: d = 0.0152 vs 0.2340 -> joins seed 0
    tr, emb = two_seed_world(10)
    lab, _ = L.refine_link(tr, emb, 0.45, no_cl(3))
    assert lab[2] == lab[0] != lab[1]


def test_reassignment_threshold_is_strict():
    tr = [trk(0, 0, 8.0), trk(1, 1, 2.0)]
    ang = math.degrees(math.acos(1 - 0.5))  # d == 0.5 exactly
    emb = np.stack([unit(0), unit(ang)])
    lab, _ = L.refine_link(tr, emb, 0.5 + 1e-7, no_cl(2))
    assert lab[0] == lab[1]
    lab, info = L.refine_link(tr, emb, 0.5 - 1e-7, no_cl(2))
    assert lab[0] != lab[1] and info["passes"][0]["no_match_threshold"] == 1


def test_reassignment_respects_cannot_link_most_confident_first():
    tr = [trk(0, 0, 8.0), trk(1, 1, 2.0), trk(2, 1, 2.0)]
    emb = np.stack([unit(0), unit(20), unit(10)])  # track 2 is closer -> placed first
    cl = no_cl(3)
    cl[1, 2] = cl[2, 1] = True
    lab, info = L.refine_link(tr, emb, 0.55, cl)
    assert lab[2] == lab[0] and lab[1] != lab[0]
    assert info["passes"][0]["blocked_cannot_link"] == 1
    L.assert_pair_cannot_link(cl, lab)


def test_cannot_link_checked_against_centroid_members_not_only_placed_tracks():
    # seeds t0 (window 0) and t1 (window 1) merge; short t2 is coactive with seed t0 and nearest
    # the centroid (placed first). Frozen spec: "no cannot-link with c*'s members" -> t2 is blocked,
    # the seed t0 keeps its place (regression: t0 used to be displaced into a singleton).
    tr = [trk(0, 0, 6.0), trk(1, 1, 6.0), trk(2, 0, 1.0)]
    emb = np.stack([unit(0), unit(20), unit(10)])
    cl = no_cl(3)
    cl[0, 2] = cl[2, 0] = True
    lab, info = L.refine_link(tr, emb, 0.6, cl)
    assert info["n_seed_clusters"] == 1
    assert lab[0] == lab[1] != lab[2]
    assert [p["blocked_cannot_link"] for p in info["passes"]] == [1, 1]
    L.assert_pair_cannot_link(cl, lab)


def test_seed_tracks_are_reassigned_too_and_can_leave_their_seed_cluster():
    # three seeds merged by AHC at a loose threshold; in reassignment, seed 2 is > theta from the centroid
    tr = [trk(0, 0, 8.0), trk(1, 1, 8.0), trk(2, 2, 4.0)]
    emb = np.stack([unit(0), unit(4), unit(70)])
    lab, info = L.refine_link(tr, emb, 0.7, no_cl(3))
    assert info["n_seed_clusters"] == 1
    # centroid (weights 8, 8, 4) is at ~15 deg: d(seed 2) = 1 - cos(55) = 0.426 < 0.7 -> stays
    assert lab[0] == lab[1] == lab[2]
    lab, info = L.refine_link(tr, emb, 0.4, no_cl(3))  # AHC: 0-1 merge (0.0024); 2 is 0.66+ away -> separate
    assert info["n_seed_clusters"] == 2 and lab[2] != lab[0]


def test_centroid_is_usable_seconds_weighted():
    e = np.stack([unit(0), unit(90)])
    c = L.weighted_centroid(e, [0, 1], np.array([3.0, 1.0]))
    assert math.degrees(math.atan2(c[1], c[0])) == pytest.approx(math.degrees(math.atan2(1, 3)))
    assert np.linalg.norm(c) == pytest.approx(1.0)


def test_second_pass_uses_recomputed_centroids():
    tr = [trk(0, 0, 4.0), trk(1, 1, 1.0), trk(2, 2, 1.0)]
    emb = np.stack([unit(0), unit(50), unit(66.4)])  # d(t2, seed) = 0.600 > 0.55
    lab1, _ = L.refine_link(tr, emb, 0.55, no_cl(3), passes=1)
    assert lab1[0] == lab1[1] != lab1[2]
    lab2, info = L.refine_link(tr, emb, 0.55, no_cl(3))
    assert lab2[0] == lab2[1] == lab2[2]  # centroid moved to ~9.4 deg: d(t2) = 0.456
    assert [p["pass"] for p in info["passes"]] == [1, 2] and info["passes"][1]["moved"] == 1


def test_tracks_without_embedding_are_their_own_speaker_and_no_seeds_means_singletons():
    tr = [trk(0, 0, 8.0), trk(1, 1, 0.3), trk(2, 2, 2.0)]
    emb = np.stack([unit(0), np.full(4, np.nan, np.float32), unit(1)])
    lab, info = L.refine_link(tr, emb, 0.55, no_cl(3))
    assert lab[0] == lab[2] and lab[1] not in (lab[0],) and info["n_no_embedding"] == 1
    tr2 = [trk(0, 0, 2.0), trk(1, 1, 2.0)]
    lab, info = L.refine_link(tr2, np.stack([unit(0), unit(0)]), 0.55, no_cl(2))
    assert info["n_seeds"] == 0 and lab[0] != lab[1]


def test_refine_end_to_end_on_synthetic_cache_respects_coactivity(tmp_path):
    turns = [(0, 12, "s0"), (12, 30, "s1"), (30, 55, "s0"), (50, 70, "s2"), (70, 100, "s1")]
    d = make_cache(tmp_path, "r", turns, 100.0, unit_dirs(3), noise=0.05, seed=2)
    c = L.load_cache(d, 0.5)
    segs, info = L.run_recording(c, 0.5, 0.55, linker="refine")
    assert info["linker"] == "refine" and len({s.speaker for s in segs}) == 3
    lab = np.array(info["track_speaker"])
    L.assert_pair_cannot_link(L.coactivity_cannot_link(c, 0.5), lab)
    with pytest.raises(L.LinkError):
        L.run_recording(c, 0.5, 0.55, "overlap", linker="refine")


# ------------------------------------------------------------------ gap-fill
def test_gap_fill_rule():
    X, Y, _ = 3, 4, -1
    lab = np.array([X, X, _, _, X, _, Y, _, _, _, X, _])
    assert L.gap_fill(lab, 1.0, 0) is lab
    out = L.gap_fill(lab, 1.0, 2.0)
    assert list(out) == [X, X, X, X, X, _, Y, _, _, _, X, _]   # X_X filled; X_Y and Y___X blocked; tail kept
    assert list(L.gap_fill(lab, 1.0, 1.999)) == list(lab)     # a 2 s gap needs g >= 2
    assert list(L.gap_fill(np.array([_, X, _, X, _]), 1.0, 5.0)) == [_, X, X, X, _]  # never before/after


def test_gap_fill_blocked_when_another_speaker_talks_in_the_gap(tmp_path):
    # s0 [0,10), s1 [10.5,11), s0 [11.5,20): the 1.5 s gap between the s0 turns contains s1 -> not filled;
    # the 0.5 s silences are between different speakers -> not filled
    turns = [(0, 10, "s0"), (10.5, 11.0, "s1"), (11.5, 20.0, "s0"), (21.0, 30.0, "s0")]
    d = make_cache(tmp_path, "r", turns, 30.0, unit_dirs(2))
    c = L.load_cache(d, 0.5)
    a, _ = L.run_recording(c, 0.5, 0.45)
    b, info = L.run_recording(c, 0.5, 0.45, gap_fill_s=5.0)
    assert len(a) == 4 and [(s.start, s.end) for s in b] == [(0.0, 10.0), (10.5, 11.0), (11.5, 30.0)]
    assert info["gap_filled_s"] == pytest.approx(1.0)
    c2, _ = L.run_recording(c, 0.5, 0.45, gap_fill_s=0.99)
    assert c2 == a


def test_default_avg_g0_equals_phase2_call_and_cli(tmp_path):
    turns = [(0, 12, "s0"), (12, 30, "s1"), (31, 55, "s0"), (56, 70, "s2")]
    d = make_cache(tmp_path / "synth", "r", turns, 70.0, unit_dirs(3), noise=0.3, seed=3)
    c = L.load_cache(d, 0.5)
    a, _ = L.run_recording(c, 0.5, 0.45)
    b, _ = L.run_recording(c, 0.5, 0.45, "primary", "none", None, linker="avg", gap_fill_s=0.0)
    assert a == b
    outs = []
    for extra in ([], ["--linker", "avg", "--gap-fill", "0"]):
        o = tmp_path / f"o{len(outs)}"
        L.main(["--system", "synth", "--recordings", "r", "--theta-act", "0.5", "--theta-clu", "0.45",
                "--out", str(o), "--cache-root", str(tmp_path)] + extra)
        outs.append((o / "r.rttm").read_bytes())
    assert outs[0] == outs[1]
