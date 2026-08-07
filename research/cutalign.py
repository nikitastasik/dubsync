"""Выравнивание последовательностей смен сцен без априорных догадок.

DP сопоставляет смены 4K со сменами 1080p: пропуск смены штрафуется, изменение
смещения штрафуется пропорционально величине. Данные сами решают, постоянное
смещение или дрейфующее.
"""
import numpy as np
from vmatch2 import VFPS

t4 = np.load('cutidx4.npy') / VFPS
th = np.load('cutidxh.npy') / VFPS
DLO, DHI = -10.0, 100.0
SKIP = 2.0          # цена пропуска смены
W = 1.0             # цена за секунду изменения смещения
INF = 1e18

cand = [np.where((th - a >= DLO) & (th - a <= DHI))[0] for a in t4]
N = len(t4)
f = [dict() for _ in range(N)]      # (j) -> (cost, prev_i, prev_j)
for j in cand[0]:
    f[0][j] = (0.0, -1, -1)

for i in range(1, N):
    for j in cand[i]:
        d = th[j] - t4[i]
        best = (INF, -1, -1)
        for pi in range(max(0, i - 3), i):
            for pj, (c, _, _) in f[pi].items():
                if pj >= j:
                    continue
                pd = th[pj] - t4[pi]
                cost = c + SKIP * ((i - pi - 1) + (j - pj - 1)) + W * abs(d - pd)
                if cost < best[0]:
                    best = (cost, pi, pj)
        if best[1] >= 0:
            f[i][j] = best
        elif i < 3:
            f[i][j] = (SKIP * i, -1, -1)

# лучший финал
bi = bj = -1
bc = INF
for i in range(N - 1, max(0, N - 6), -1):
    for j, (c, _, _) in f[i].items():
        tot = c + SKIP * (N - 1 - i)
        if tot < bc:
            bc, bi, bj = tot, i, j

pairs = []
i, j = bi, bj
while i >= 0 and j >= 0:
    pairs.append((t4[i], th[j]))
    _, i, j = f[i][j]
pairs.reverse()
np.save('cut_pairs.npy', np.array(pairs))
print(f'сопоставлено {len(pairs)} смен из {N} (4K) / {len(th)} (HD)')

print('\nсмещение по фильму (по сменам сцен):')
prev = None
for a, b in pairs:
    d = b - a
    if prev is None or abs(d - prev) > 0.2:
        mark = '' if prev is None else f'   изменение {d-prev:+.2f}'
        print(f'  4K {a:8.2f}s -> HD {b:8.2f}s   смещение {d:+7.2f}{mark}')
        prev = d
