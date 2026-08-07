"""Границы вставок и проверка, не вырезан ли вместе с ними кусок фильма.

Слева и справа от вставки смещение берём экстраполяцией карты на саму границу
(из-за дрейфа медиана по участку для этого не годится). Затем ищем, где
обрывается совпадение при левом смещении и где начинается при правом.
Если точки склейки слева и справа не сходятся — часть фильма в 1080p потеряна.
"""
import numpy as np
from vfeat import RATE

F4 = np.load('F4.npy')
FH = np.load('FH.npy')
T4 = F4.shape[1]
nodes = np.load('fmap_nodes.npy')
ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45) & (nodes[:, 3] > 0.10)
tt, dd = nodes[ok, 0], nodes[ok, 1]
edges = np.load('edges.npy')


def extrapolate(T, side):
    """Смещение на границе: локальная прямая по узлам с нужной стороны."""
    m = ((tt > T - 80) & (tt < T - 6)) if side == 'l' else ((tt > T + 6) & (tt < T + 80))
    if m.sum() < 3:
        return None
    p = np.polyfit(tt[m] - T, dd[m], 1)
    return float(p[1])


def sim(j, off, half=3):
    d = int(round(off * RATE))
    vals = []
    for k in range(j - half, j + half + 1):
        i = k - d
        if 0 <= i < T4 and 0 <= k < FH.shape[1]:
            vals.append(float(F4[:, i] @ FH[:, k] / 32))
    return np.mean(vals) if vals else -1.0


out = []
for T0, _, _ in edges:
    ob, oa = extrapolate(T0, 'l'), extrapolate(T0, 'r')
    step = 1 / RATE
    # A: последний момент, где ещё держится совпадение при левом смещении
    ts = np.arange(T0 + ob - 8, T0 + ob + 8, step)
    cb = np.array([sim(int(round(t * RATE)), ob) for t in ts])
    good = cb > 0.55
    A = ts[np.where(good)[0][-1]] if good.any() else np.nan
    # B: первый момент устойчивого совпадения при правом смещении
    ts2 = np.arange(T0 + oa - 8, T0 + oa + 8, step)
    ca = np.array([sim(int(round(t * RATE)), oa) for t in ts2])
    good2 = ca > 0.55
    B = ts2[np.where(good2)[0][0]] if good2.any() else np.nan
    cutL, cutR = A - ob, B - oa
    lost = cutR - cutL
    print(f'вставка около 4K {T0:8.2f}s:  смещения {ob:+.3f} / {oa:+.3f}')
    print(f'    реклама в 1080p [{A:9.3f} .. {B:9.3f}]  длина {B-A:6.3f}s')
    print(f'    склейка в 4K: слева {cutL:9.3f}s, справа {cutR:9.3f}s  '
          f'-> потеряно фильма {lost:+.3f}s')
    out.append((cutL, cutR, ob, oa, A, B))

np.save('edges3.npy', np.array(out))
