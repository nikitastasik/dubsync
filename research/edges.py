"""Точные границы рекламных вставок — с точностью до кадра.

В правильной точке разрыва T окно 4K слева от T ложится на 1080p со «старым»
смещением, а окно справа — с «новым». Перебираем T покадрово и берём максимум
суммы двух корреляций.
"""
import numpy as np
import afeat as A
from vfeat import RATE
from vmatch2 import VFPS

F4 = np.load('F4.npy')
FH = np.load('FH.npy')
THD = FH.shape[1]
nodes = np.load('fmap_nodes.npy')
ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45)
tt, dd = nodes[ok, 0], nodes[ok, 1]

vo = np.load('v_offsets2.npy')
vs = np.load('vstarts2.npy') / VFPS
JUMPS = [(vs[w], vo[w - 1], vo[w]) for w in range(1, len(vo)) if vo[w] - vo[w - 1] > 5]

SIDE = 8.0          # длина проверочного окна, с


def corr_at(i4, off, n):
    lo = i4 + int(round(off * RATE))
    if lo < 0 or lo + n > THD or i4 < 0 or i4 + n > F4.shape[1]:
        return -1.0
    a = F4[:, i4:i4 + n]
    b = FH[:, lo:lo + n]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


results = []
for Tguess, _, _ in JUMPS:
    # уточняем смещения по обе стороны по надёжным узлам рядом
    before = dd[(tt > Tguess - 90) & (tt < Tguess - 5)]
    after = dd[(tt > Tguess + 15) & (tt < Tguess + 100)]
    ob, oa = float(np.median(before)), float(np.median(after))
    n = int(SIDE * RATE)
    best = (-9, None)
    for T in np.arange(Tguess - 4, Tguess + 4, 1 / RATE):
        i = int(round(T * RATE))
        s = corr_at(i - n, ob, n) + corr_at(i, oa, n)
        if s > best[0]:
            best = (s, T)
    T = best[1]
    results.append((T, ob, oa))
    print(f'вставка: 4K {T:9.3f}s  смещение {ob:+.3f} -> {oa:+.3f}  '
          f'(длина {oa-ob:.3f}s)  сумма корреляций {best[0]:.3f}')
    print(f'    в 1080p реклама занимает [{T+ob:9.3f} .. {T+oa:9.3f}]')

np.save('edges.npy', np.array(results))
