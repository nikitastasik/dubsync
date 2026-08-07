"""Границы вставок по скользящей корреляции 4-секундных окон.

R_before(t): насколько 1080p перед моментом t ещё совпадает с 4K при левом
смещении. R_after(t): насколько 1080p после t совпадает при правом смещении.
Реклама — провал обеих кривых между ними.
"""
import numpy as np
from vfeat import RATE

F4 = np.load('F4.npy')
FH = np.load('FH.npy')
T4, THD = F4.shape[1], FH.shape[1]
nodes = np.load('fmap_nodes.npy')
ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45) & (nodes[:, 3] > 0.10)
tt, dd = nodes[ok, 0], nodes[ok, 1]
edges = np.load('edges.npy')
W = int(round(4.0 * RATE))
THR = 0.70


def extrapolate(T, side):
    m = ((tt > T - 80) & (tt < T - 6)) if side == 'l' else ((tt > T + 6) & (tt < T + 80))
    p = np.polyfit(tt[m] - T, dd[m], 1)
    return float(p[1])


def corr(j_hd, i_4k, n):
    if i_4k < 0 or i_4k + n > T4 or j_hd < 0 or j_hd + n > THD:
        return -1.0
    a = FH[:, j_hd:j_hd + n]
    b = F4[:, i_4k:i_4k + n]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


out = []
for T0, _, _ in edges:
    ob, oa = extrapolate(T0, 'l'), extrapolate(T0, 'r')
    db, da = int(round(ob * RATE)), int(round(oa * RATE))
    lo = int(round((T0 + ob - 10) * RATE))
    hi = int(round((T0 + oa + 10) * RATE))
    js = np.arange(lo, hi)
    rb = np.array([corr(j - W, j - W - db, W) for j in js])
    ra = np.array([corr(j, j - da, W) for j in js])
    gb, ga = np.where(rb > THR)[0], np.where(ra > THR)[0]
    A = js[gb[-1]] / RATE if len(gb) else np.nan
    B = js[ga[0]] / RATE if len(ga) else np.nan
    cutL, cutR = A - ob, B - oa
    print(f'вставка около 4K {T0:8.2f}s:  смещения {ob:+.3f} / {oa:+.3f}')
    print(f'    реклама в 1080p [{A:9.3f} .. {B:9.3f}]  длина {B-A:6.3f}s')
    print(f'    склейка в 4K: слева {cutL:9.3f}s, справа {cutR:9.3f}s  '
          f'-> расхождение {cutR-cutL:+.3f}s')
    out.append((cutL, cutR, ob, oa, A, B))

np.save('edges4.npy', np.array(out))
