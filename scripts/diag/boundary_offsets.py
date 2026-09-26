"""DIAGNOSTIC: label start/end offsets vs acoustic speech boundaries (both detectors agree),
per recording. Positive onset delta = detector speech starts AFTER the label start."""
import glob, sys
import numpy as np
sys.path.insert(0, '.')
from bnd.score import load_rttm_file
out = {}
allS, allE = [], []
for f in sorted(glob.glob('data/prepared/ref_rttm/*.rttm')):
    rid = f.split('/')[-1][:-5]
    z = np.load(f'runs/probe_alignment/probs/{rid}.npz'); p = z['probs'].max(1); dt = float(z['dt'])
    s = np.load(f'runs/probe_alignment/silero/{rid}.npz'); sp = s['probs']; sdt = float(s['dt'])
    n = len(p); t = np.arange(n) * dt
    sil = sp[np.minimum((t / sdt).astype(int), len(sp) - 1)]
    both = (p >= 0.5) & (sil >= 0.5)
    on = np.flatnonzero(both[1:] & ~both[:-1]) + 1; off = np.flatnonzero(~both[1:] & both[:-1]) + 1
    ref = load_rttm_file(f, rid)
    ds, de = [], []
    for seg in ref:
        if seg.end - seg.start < 1.0: continue
        i = int(round(seg.start / dt)); j = int(round(seg.end / dt))
        if len(on):
            k = on[np.argmin(np.abs(on - i))]
            if abs(k - i) * dt <= 1.5: ds.append((k - i) * dt)
        if len(off):
            k = off[np.argmin(np.abs(off - j))]
            if abs(k - j) * dt <= 1.5: de.append((k - j) * dt)
    ds, de = np.array(ds), np.array(de); allS += list(ds); allE += list(de)
    print(f"{rid}: onset delta median {np.median(ds):+.2f} s (n={len(ds)}, IQR {np.percentile(ds,25):+.2f}..{np.percentile(ds,75):+.2f}) | offset delta median {np.median(de):+.2f} s (n={len(de)}, IQR {np.percentile(de,25):+.2f}..{np.percentile(de,75):+.2f})")
allS, allE = np.array(allS), np.array(allE)
print(f"ALL onset median {np.median(allS):+.3f} mean {allS.mean():+.3f} | offset median {np.median(allE):+.3f} mean {allE.mean():+.3f}")
h1 = np.histogram(allS, bins=np.arange(-1.5, 1.51, 0.25))[0]; h2 = np.histogram(allE, bins=np.arange(-1.5, 1.51, 0.25))[0]
print("onset hist (-1.5..1.5 by .25):", h1.tolist()); print("offset hist:", h2.tolist())
