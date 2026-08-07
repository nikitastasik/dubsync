"""Точная карта соответствия по видео: профили + DP с фиксированной ценой скачка.

Скачки не ограничены знаком: если рекламу не только вставляли, но и вырезали
часть фильма, смещение может и уменьшаться — пусть данные скажут сами.
"""
import numpy as np
import afeat as A
from vmatch2 import load4k, loadhd, feats, VFPS

DMIN, DMAX = int(-10 * VFPS), int(100 * VFPS)
D = DMAX - DMIN + 1
WIN = int(10 * VFPS)
STEP = int(4 * VFPS)
PEN = 1.2

f4 = feats(load4k())
fh = feats(loadhd())
T4, THD = f4.shape[1], fh.shape[1]

starts = list(range(0, T4 - WIN, STEP))
S = np.full((len(starts), D), -1.0, dtype=np.float32)
for w, i in enumerate(starts):
    clo, chi = max(0, i + DMIN), min(THD, i + DMAX + WIN)
    if chi - clo < WIN:
        continue
    sc = A.ncc(f4[:, i:i + WIN], fh[:, clo:chi])
    j0 = clo - (i + DMIN)
    S[w, j0:j0 + len(sc)] = sc
np.save('vS2.npy', S)
np.save('vstarts2.npy', np.array(starts))
print('профили готовы', S.shape, flush=True)

best = S[0].astype(np.float64).copy()
back = np.zeros(S.shape, dtype=np.int32)
ar = np.arange(D)
for w in range(1, S.shape[0]):
    g = int(np.argmax(best))
    jump = best[g] - PEN
    take = jump > best
    best = np.where(take, jump, best) + S[w]
    back[w] = np.where(take, g, ar)
path = np.zeros(S.shape[0], dtype=np.int32)
path[-1] = int(np.argmax(best))
for w in range(S.shape[0] - 1, 0, -1):
    path[w - 1] = back[w][path[w]]
offs = (path + DMIN) / VFPS
qual = np.array([S[w, path[w]] for w in range(len(starts))])
np.save('v_offsets2.npy', offs)
np.save('v_qual2.npy', qual)

print(f'качество: медиана {np.median(qual):.3f}, 25-й проц {np.percentile(qual,25):.3f}')
print('\n=== сегменты (по видео):')
prev, seg_start = offs[0], 0.0
tot = 0.0
for w in range(1, len(starts)):
    if abs(offs[w] - prev) > 0.05:
        t = starts[w] / VFPS
        print(f'  сегмент 4K [{seg_start:8.2f} .. {t:8.2f}]  смещение {prev:+7.2f}s')
        print(f'      -> скачок в {t:8.2f}s: {prev:+.2f} -> {offs[w]:+.2f}  ({offs[w]-prev:+.2f}s)')
        tot += offs[w] - prev
        prev, seg_start = offs[w], t
print(f'  сегмент 4K [{seg_start:8.2f} .. {T4/VFPS:8.2f}]  смещение {prev:+7.2f}s')
print(f'\nсуммарно добавлено {tot:.2f}s; разница длительностей {(THD-T4)/VFPS:.2f}s')
