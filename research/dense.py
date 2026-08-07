"""Плотное измерение смещения на участке — понять характер дрейфа."""
import sys
import numpy as np
import afeat as A

FPS = A.FPS
hd = np.load('f_hd.npy')
k4 = [np.load('f_4k_rus.npy'), np.load('f_4k_eng.npy')]
THD = hd.shape[1]

t0, t1, dlo, dhi = (float(x) for x in sys.argv[1:5])
step = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
win = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0

n = int(win * FPS)
print(f'{"время":>9} {"смещ":>8} {"пик":>6} {"conf":>6}')
for t in np.arange(t0, t1, step):
    i = int(t * FPS)
    lo, hi = max(0, i + int(dlo * FPS)), min(THD, i + int(dhi * FPS) + n)
    if hi - lo < n:
        continue
    tot = None
    for f in k4:
        s = A.ncc(f[:, i:i + n], hd[:, lo:hi])
        tot = s if tot is None else tot + s
    sc = tot / len(k4)
    b = int(np.argmax(sc))
    m = np.ones(len(sc), bool)
    m[max(0, b - int(1.5 * FPS)):b + int(1.5 * FPS)] = False
    conf = sc[b] - (sc[m].max() if m.any() else 0)
    flag = '' if conf > 0.15 else ('  ?' if conf > 0.07 else '  ??')
    print(f'{t:9.1f} {(lo+b-i)/FPS:8.3f} {sc[b]:6.3f} {conf:6.3f}{flag}')
