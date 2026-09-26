"""DIAGNOSTIC (not registered): native full-recording Nemotron (one streaming session, 8 channels,
model-card offline config) decoded exclusively + gap-fill, scored with bnd.score. Looks at all 10."""
import glob, json, sys
import numpy as np
sys.path.insert(0, '.')
from bnd.score import load_rttm_file, score_recording, audio_duration
from bnd.link import gap_fill, frames_to_segs

def decode(p, dt, dur, th, g):
    lab = np.where(p.max(1) >= th, p.argmax(1), -1)
    lab = gap_fill(lab, dt, g)
    return frames_to_segs(lab, dt, dur)

res = {}
recs = sorted(f.split('/')[-1][:-4] for f in glob.glob('runs/probe_alignment/probs/*.npz'))
for th in (0.3, 0.4, 0.5):
    for g in (0, 2, 5, 10):
        tot = {0.0: [0, 0, 0, 0], 0.25: [0, 0, 0, 0]}; per = {}; spk = {}
        for rid in recs:
            z = np.load(f'runs/probe_alignment/probs/{rid}.npz'); p = z['probs'].astype(np.float32); dt = float(z['dt'])
            dur = audio_duration(f'data/raw/diarization/diarization/train/audio/{rid}.wav')
            ref = load_rttm_file(f'data/prepared/ref_rttm/{rid}.rttm', rid)
            hyp = decode(p, dt, dur, th, g)
            spk[rid] = len({s.speaker for s in hyp})
            for c in (0.0, 0.25):
                s = score_recording(ref, hyp, dur, c, rid)
                t = tot[c]; t[0] += s.total; t[1] += s.miss; t[2] += s.fa; t[3] += s.conf
                if c == 0.0: per[rid] = round(100 * s.der, 2)
        k = f"th{th}_g{g}"
        res[k] = {c: {"der": 100 * (t[1] + t[2] + t[3]) / t[0], "miss": 100 * t[1] / t[0], "fa": 100 * t[2] / t[0], "conf": 100 * t[3] / t[0]} for c, t in tot.items()}
        res[k]["per_rec_c0"] = per; res[k]["n_spk"] = spk
        r0 = res[k][0.0]; r25 = res[k][0.25]
        print(f"{k:10s} c0 DER {r0['der']:6.2f} [M {r0['miss']:5.2f} FA {r0['fa']:5.2f} C {r0['conf']:5.2f}] | c0.25 {r25['der']:6.2f}", flush=True)
json.dump(res, open('runs/diag/native_fullrec.json', 'w'), indent=1, default=str)
b = min(res, key=lambda k: res[k][0.0]['der']); print('best', b, res[b]['per_rec_c0'], res[b]['n_spk'])
