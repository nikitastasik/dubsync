"""Границы вставок по признаку «кадру 1080p нет соответствия в 4K».

Для каждого кадра 1080p в окрестности считаем, насколько он похож на кадр 4K
при старом и при новом смещении. Реклама — участок, где обе похожести падают.
"""
import numpy as np
from vfeat import RATE

F4 = np.load('F4.npy')
FH = np.load('FH.npy')
T4, THD = F4.shape[1], FH.shape[1]
edges = np.load('edges.npy')
SM = int(round(0.5 * RATE))     # сглаживание похожести, ±0.5 с


def sim_curve(j0, j1, off):
    """Похожесть кадров 1080p [j0,j1) на 4K при данном смещении."""
    d = int(round(off * RATE))
    out = np.full(j1 - j0, -1.0)
    for k, j in enumerate(range(j0, j1)):
        i = j - d
        if 0 <= i < T4:
            out[k] = float(F4[:, i] @ FH[:, j] / 32)
    ker = np.ones(2 * SM + 1) / (2 * SM + 1)
    return np.convolve(out, ker, mode='same')


res = []
for T, ob, oa in edges:
    j_lo = int(round((T + ob - 12) * RATE))
    j_hi = int(round((T + oa + 12) * RATE))
    cb = sim_curve(j_lo, j_hi, ob)
    ca = sim_curve(j_lo, j_hi, oa)
    tj = np.arange(j_lo, j_hi) / RATE
    thr = 0.35
    inside = (cb < thr) & (ca < thr)
    idx = np.where(inside)[0]
    if len(idx) == 0:
        print(f'вставка около {T:.2f}: границы не выделились')
        res.append((T, ob, oa))
        continue
    # самый длинный непрерывный участок «нет соответствия»
    splits = np.split(idx, np.where(np.diff(idx) > int(0.5 * RATE))[0] + 1)
    seg = max(splits, key=len)
    a_hd, b_hd = tj[seg[0]], tj[seg[-1]]
    print(f'вставка около 4K {T:8.2f}s:')
    print(f'    реклама в 1080p [{a_hd:9.3f} .. {b_hd:9.3f}]  длина {b_hd-a_hd:6.3f}s')
    print(f'    точка склейки в 4K: {a_hd-ob:9.3f}s (по началу) / {b_hd-oa:9.3f}s (по концу)')
    res.append(((a_hd - ob + b_hd - oa) / 2, ob, oa))

np.save('edges2.npy', np.array(res))
