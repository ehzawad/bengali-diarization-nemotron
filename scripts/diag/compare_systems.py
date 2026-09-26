"""DIAGNOSTIC: score systems x fixed decoding variants (no tuning), same scorer, all 10 recordings."""
import glob, json, math, sys
import numpy as np
sys.path.insert(0, '.')
from bnd.score import load_rttm_file, score_recording, audio_duration, Seg
from bnd.link import gap_fill, frames_to_segs
DT = 0.01
def to_frames(segs, dur):
    lab = np.full(int(math.ceil(dur / DT)), -1, dtype=np.int64); names = {}
    for s in sorted(segs, key=lambda s: s.start):
        k = names.setdefault(s.speaker, len(names)); lab[int(round(s.start / DT)):int(round(s.end / DT))] = k
    return lab
def exclusive(segs):
    segs = sorted(segs, key=lambda s: (s.start, s.end)); out = []
    for s in segs:
        if out and s.start < out[-1].end:
            p = out[-1]; out[-1] = Seg(p.start, s.start, p.speaker)
            if out[-1].end <= out[-1].start: out.pop()
        if s.end > s.start: out.append(s)
    return out
def variant(segs, dur, g, floor):
    if g > 0: segs = frames_to_segs(gap_fill(to_frames(segs, dur), DT, g), DT, dur)
    if floor: segs = exclusive([Seg(math.floor(s.start), math.floor(s.end), s.speaker) for s in segs if math.floor(s.end) > math.floor(s.start)])
    return segs
systems = {
    "nemotron_zs_default(p3b zs_star)": {f.split('/')[-1][:-5]: f for f in glob.glob('runs/p3b/outer/zs_star/F*/*.rttm')},
    "pyannote_community1_exclusive": {f.split('/')[-1][:-5]: f for f in glob.glob('runs/diag/community1/exclusive/*.rttm')},
}
res = {}
for name, paths in systems.items():
    if len(paths) != 10: print(name, "missing", len(paths)); continue
    for g, fl in [(0, False), (5, False), (0, True), (5, True)]:
        T = {c: [0, 0, 0, 0] for c in (0.0, 0.25, 0.5)}; cnt = 0; per = {}
        for rid, p in sorted(paths.items()):
            dur = audio_duration(f'data/raw/diarization/diarization/train/audio/{rid}.wav')
            ref = load_rttm_file(f'data/prepared/ref_rttm/{rid}.rttm', rid)
            hyp = variant(load_rttm_file(p, rid), dur, g, fl)
            cnt += abs(len({s.speaker for s in hyp}) - len({s.speaker for s in ref}))
            for c in T:
                s = score_recording(ref, hyp, dur, c, rid); t = T[c]; t[0] += s.total; t[1] += s.miss; t[2] += s.fa; t[3] += s.conf
                if c == 0.0: per[rid] = round(100 * s.der, 2)
        f = lambda t: f"{100*(t[1]+t[2]+t[3])/t[0]:6.2f} [M {100*t[1]/t[0]:5.2f} FA {100*t[2]/t[0]:5.2f} C {100*t[3]/t[0]:5.2f}]"
        key = f"{name} g{g}{' floor' if fl else ''}"
        res[key] = {"c0": f(T[0.0]), "c0.25": f(T[0.25]), "c0.5(=NeMo 0.25)": f(T[0.5]), "sum_abs_count_err": cnt, "per_rec_c0": per}
        print(f"{key:48s} c0 {f(T[0.0])} | c0.25 {100*sum(T[0.25][1:])/T[0.25][0]:6.2f} | c0.5 {100*sum(T[0.5][1:])/T[0.5][0]:6.2f} | Σ|cnt err| {cnt}", flush=True)
json.dump(res, open('runs/diag/compare_systems.json', 'w'), indent=1)
