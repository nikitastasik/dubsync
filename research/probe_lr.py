"""Сравнение качества корреляции: обычный downmix против разностного канала L-R."""
import numpy as np
import afeat as A

FPS = A.FPS
hd_lr = A.normalize(A.features(A.read_wav_mono('hd_lr.wav')))
np.save('f_hd_lr.npy', hd_lr)

for name in ('rus', 'eng'):
    f = A.normalize(A.features(A.read_wav_mono(f'4k_{name}_lr.wav')))
    np.save(f'f_4k_{name}_lr.npy', f)
    print(f'--- 4k_{name} L-R')
    for t, exp in ((900, 40.5), (2000, 42.5), (3000, 43.0), (4200, 75.2), (5400, 76.9), (6200, 76.7)):
        i = int(t * FPS)
        n = int(40 * FPS)
        lo, hi = max(0, i - int(5 * FPS)), min(hd_lr.shape[1], i + int(95 * FPS) + n)
        sc = A.ncc(f[:, i:i + n], hd_lr[:, lo:hi])
        b = int(np.argmax(sc))
        off = (lo + b - i) / FPS
        m = np.ones(len(sc), bool)
        m[max(0, b - int(2 * FPS)):b + int(2 * FPS)] = False
        print(f'  t={t:5d}  off={off:8.3f} (ожид ~{exp:5.1f})  peak={sc[b]:.3f}  conf={sc[b]-sc[m].max():.3f}')
