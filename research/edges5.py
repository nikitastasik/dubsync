"""Точка склейки и смещения по обе стороны — одной оптимизацией.

Для каждой кандидатной точки T подбираем лучшее смещение отдельно для куска
слева и для куска справа. Верная точка склейки — та, где оба куска ложатся
на 1080p одинаково хорошо.
"""
import numpy as np
import afeat as A
from vfeat import RATE

F4 = np.load('F4.npy')
FH = np.load('FH.npy')
T4, THD = F4.shape[1], FH.shape[1]
nodes = np.load('fmap_nodes.npy')
ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45) & (nodes[:, 3] > 0.10)
tt, dd = nodes[ok, 0], nodes[ok, 1]
edges = np.load('edges.npy')

SIDE = int(round(6.0 * RATE))     # длина куска для проверки
SPAN = 2.0                        # коридор поиска смещения, ±с


def pred(T, side):
    m = ((tt > T - 80) & (tt < T - 6)) if side == 'l' else ((tt > T + 6) & (tt < T + 80))
    p = np.polyfit(tt[m] - T, dd[m], 1)
    return float(p[1])


def best_off(i4, n, center):
    """Лучшее смещение для куска 4K [i4,i4+n) в коридоре center±SPAN."""
    lo = max(0, i4 + int((center - SPAN) * RATE))
    hi = min(THD, i4 + int((center + SPAN) * RATE) + n)
    if hi - lo < n or i4 < 0 or i4 + n > T4:
        return -1.0, np.nan
    sc = A.ncc(F4[:, i4:i4 + n], FH[:, lo:hi])
    b = int(np.argmax(sc))
    return float(sc[b]), (lo + b - i4) / RATE


out = []
for T0, _, _ in edges:
    pb, pa = pred(T0, 'l'), pred(T0, 'r')
    rows = []
    for T in np.arange(T0 - 6, T0 + 6, 1 / RATE):
        i = int(round(T * RATE))
        sl, ol = best_off(i - SIDE, SIDE, pb)
        sr, orr = best_off(i, SIDE, pa)
        rows.append((T, sl, ol, sr, orr))
    r = np.array(rows)
    k = int(np.argmax(r[:, 1] + r[:, 3]))
    T, sl, ol, sr, orr = r[k]
    # где кусок слева/справа ещё уверенно совпадает
    goodL = r[r[:, 1] > 0.90, 0]
    goodR = r[r[:, 3] > 0.90, 0]
    T1 = goodL.max() if len(goodL) else np.nan
    T2 = goodR.min() if len(goodR) else np.nan
    print(f'вставка около 4K {T0:8.2f}s:')
    print(f'    склейка в 4K {T:9.3f}s   качество слева {sl:.3f}, справа {sr:.3f}')
    print(f'    смещения {ol:+.3f} -> {orr:+.3f}   реклама в 1080p '
          f'[{T+ol:9.3f} .. {T+orr:9.3f}]  длина {orr-ol:6.3f}s')
    if not (np.isnan(T1) or np.isnan(T2)):
        print(f'    фильм слева уверен до {T1:.3f}s, справа с {T2:.3f}s '
              f'-> потеряно {max(0.0, T2-T1):.3f}s')
    out.append((T, ol, orr))

np.save('edges5.npy', np.array(out))
