"""DIAGNOSTIC: post-hoc boundary conventions applied to the frozen phase-3b default (zs_star) outputs."""
import glob, math, sys
sys.path.insert(0, '.')
from bnd.score import load_rttm_file, score_recording, audio_duration, Seg
recs = sorted(f.split('/')[-1][:-5] for f in glob.glob('runs/p3b/outer/zs_star/F*/*.rttm'))
paths = {f.split('/')[-1][:-5]: f for f in glob.glob('runs/p3b/outer/zs_star/F*/*.rttm')}
def exclusive(segs):
    segs = sorted(segs, key=lambda s: (s.start, s.end)); out = []
    for s in segs:
        if out and s.start < out[-1].end:  # later-starting speaker takes over (annotation: new segment starts when a different speaker begins)
            p = out[-1]; out[-1] = Seg(p.start, s.start, p.speaker)
            if out[-1].end <= out[-1].start: out.pop()
        if s.end > s.start: out.append(s)
    return out
def xf(segs, mode, d, dur):
    r = []
    for s in segs:
        if mode == 'shift': a, b = s.start - d, s.end - d
        elif mode == 'floor': a, b = math.floor(s.start), math.floor(s.end)
        elif mode == 'round': a, b = round(s.start), round(s.end)
        else: a, b = s.start, s.end
        a = max(0.0, a); b = min(dur, b)
        if b > a: r.append(Seg(a, b, s.speaker))
    return exclusive(r)
for mode, d in [('none', 0), ('shift', 0.1), ('shift', 0.2), ('shift', 0.3), ('shift', 0.5), ('floor', 0), ('round', 0)]:
    T = {0.0: [0, 0, 0, 0], 0.25: [0, 0, 0, 0]}
    for rid in recs:
        dur = audio_duration(f'data/raw/diarization/diarization/train/audio/{rid}.wav')
        ref = load_rttm_file(f'data/prepared/ref_rttm/{rid}.rttm', rid)
        hyp = xf(load_rttm_file(paths[rid], rid), mode, d, dur)
        for c in T:
            s = score_recording(ref, hyp, dur, c, rid); t = T[c]; t[0] += s.total; t[1] += s.miss; t[2] += s.fa; t[3] += s.conf
    f = lambda t: f"{100*(t[1]+t[2]+t[3])/t[0]:6.2f} [M {100*t[1]/t[0]:5.2f} FA {100*t[2]/t[0]:5.2f} C {100*t[3]/t[0]:5.2f}]"
    print(f"{mode:5s} {d:4.1f}  c0 {f(T[0.0])} | c0.25 {f(T[0.25])}", flush=True)
