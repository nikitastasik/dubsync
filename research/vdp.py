"""Карта соответствия по видео: профили корреляции + монотонный DP."""
import numpy as np
import afeat as A
from vmatch import load, VFPS

DMIN = int(-5 * VFPS)
DMAX = int(95 * VFPS)
D = DMAX - DMIN + 1
WIN = int(20 * VFPS)
STEP = int(5 * VFPS)
PEN = 3.0
EPS = 0.0001

v4 = np.load('v4.npy')
vh = np.load('vh.npy')
T4, THD = v4.shape[1], vh.shape[1]
print(f'4K {T4/VFPS:.1f}s   HD {THD/VFPS:.1f}s   diff {(THD-T4)/VFPS:.2f}s', flush=True)

starts = list(range(0, T4 - WIN, STEP))
S = np.full((len(starts), D), -1.0, dtype=np.float32)
for w, i in enumerate(starts):
    clo, chi = max(0, i + DMIN), min(THD, i + DMAX + WIN)
    if chi - clo < WIN:
        continue
    sc = A.ncc(v4[:, i:i + WIN], vh[:, clo:chi])
    S[w, clo - (i + DMIN):clo - (i + DMIN) + len(sc)] = sc
np.save('vS.npy', S)
np.save('vstarts.npy', np.array(starts))
print('профили готовы', S.shape, flush=True)

best = S[0].astype(np.float64).copy()
back = np.zeros(S.shape, dtype=np.int32)
ramp = np.arange(D) * EPS
ar = np.arange(D)
for w in range(1, S.shape[0]):
    shifted = best + ramp
    pm = np.maximum.accumulate(shifted)
    argp = np.maximum.accumulate(np.where(shifted >= pm, ar, 0)).astype(np.int32)
    jump = pm - PEN - ramp
    take = jump > best
    best = np.where(take, jump, best) + S[w]
    back[w] = np.where(take, argp, ar)
path = np.zeros(S.shape[0], dtype=np.int32)
path[-1] = int(np.argmax(best))
for w in range(S.shape[0] - 1, 0, -1):
    path[w - 1] = back[w][path[w]]
offs = (path + DMIN) / VFPS
np.save('v_offsets.npy', offs)

qual = np.array([S[w, path[w]] for w in range(len(starts))])
print(f'качество совпадения: медиана {np.median(qual):.3f}, 10-й проц. {np.percentile(qual,10):.3f}')
print('\n=== ступени (видео):')
prev = offs[0]
print(f'старт {prev:.2f}s')
tot = 0.0
for w in range(1, len(starts)):
    if abs(offs[w] - prev) > 0.1:
        t = starts[w] / VFPS
        print(f'  4K ~{t/60:6.2f} мин ({t:8.1f}s)   {prev:7.2f} -> {offs[w]:7.2f}   {offs[w]-prev:+7.2f}s   q={qual[w]:.2f}')
        tot += offs[w] - prev
        prev = offs[w]
print(f'конец {offs[-1]:.2f}s, суммарно вставлено {tot:.2f}s (разница длительностей {(THD-T4)/VFPS:.2f}s)')
