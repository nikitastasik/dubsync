"""Смены сцен как независимые опорные точки.

Момент склейки — однозначное событие в обоих релизах, не зависящее от
кодека, HDR и микширования звука.
"""
import numpy as np
from vmatch2 import load4k, loadhd, VFPS


def cutcurve(a):
    """Разница между соседними кадрами после нормировки каждого кадра."""
    n = a.shape[0]
    f = a.reshape(n, -1).astype(np.float32)
    f -= f.mean(axis=1, keepdims=True)
    f /= (f.std(axis=1, keepdims=True) + 1e-6)
    return np.r_[0, np.abs(np.diff(f, axis=0)).mean(axis=1)]


def find_cuts(c, thr):
    idx = np.where(c > thr)[0]
    keep = []
    for j in idx:
        if keep and j - keep[-1] < 3:
            if c[j] > c[keep[-1]]:
                keep[-1] = j
            continue
        keep.append(j)
    return np.array(keep)


c4 = cutcurve(load4k())
ch = cutcurve(loadhd())
np.save('cut4.npy', c4)
np.save('cuth.npy', ch)
t4 = find_cuts(c4, np.median(c4) + 6 * np.median(np.abs(c4 - np.median(c4))))
th = find_cuts(ch, np.median(ch) + 6 * np.median(np.abs(ch - np.median(ch))))
print(f'смен сцен: 4K {len(t4)}, HD {len(th)}')
np.save('cutidx4.npy', t4)
np.save('cutidxh.npy', th)

# для опорных моментов 4K — ближайшая смена в HD около заданного смещения
print('\nсоответствие смен сцен (4K -> HD):')
for t, guess in ((300, 28.5), (900, 43), (1420, 43.75), (1650, 39.5),
                 (2000, 43.5), (2500, 42), (3000, 43), (3500, 43),
                 (4000, 76), (4500, 76), (5000, 77), (5500, 76), (6000, 76)):
    i = int(t * VFPS)
    near4 = t4[np.argmin(np.abs(t4 - i))]
    j = near4 + int(guess * VFPS)
    cand = th[np.abs(th - j) < 5 * VFPS]
    if len(cand) == 0:
        print(f'  4K {near4/VFPS:8.2f}s: пары не найдено')
        continue
    k = cand[np.argmin(np.abs(cand - j))]
    print(f'  4K {near4/VFPS:8.2f}s -> HD {k/VFPS:8.2f}s   смещение {(k-near4)/VFPS:+7.2f}s'
          f'   (сила {c4[near4]:.2f}/{ch[k]:.2f})')
