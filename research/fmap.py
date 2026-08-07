"""Финальная карта смещения на полном фреймрейте.

Узлы каждые NODE секунд; в каждом — корреляция окна кадров 4K с 1080p в
узком коридоре вокруг предсказания, плюс парабола для субкадровой точности.
"""
import numpy as np
import afeat as A
from vfeat import RATE
from vmatch2 import VFPS

NODE = 5.0
WIN = 12.0
SEARCH = 1.6

F4 = np.load('F4.npy')
FH = np.load('FH.npy')
T4, THD = F4.shape[1], FH.shape[1]

vo = np.load('v_offsets2.npy')
vs = np.load('vstarts2.npy') / VFPS
JUMPS = [vs[w] for w in range(1, len(vo)) if vo[w] - vo[w - 1] > 5]


def predict(t):
    return float(np.interp(t, vs, vo))


rows = []
n = int(WIN * RATE)
for t in np.arange(0, T4 / RATE - WIN, NODE):
    if any(t - 1 < j < t + WIN + 1 for j in JUMPS):
        rows.append((t + WIN / 2, np.nan, 0.0, 0.0))
        continue
    p = predict(t + WIN / 2)
    i = int(t * RATE)
    lo = max(0, i + int((p - SEARCH) * RATE))
    hi = min(THD, i + int((p + SEARCH) * RATE) + n)
    if hi - lo < n:
        rows.append((t + WIN / 2, np.nan, 0.0, 0.0))
        continue
    sc = A.ncc(F4[:, i:i + n], FH[:, lo:hi])
    b = int(np.argmax(sc))
    if 0 < b < len(sc) - 1:
        y0, y1, y2 = sc[b - 1], sc[b], sc[b + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
        delta = float(np.clip(delta, -1, 1))
    else:
        delta = 0.0
    off = (lo + b + delta - i) / RATE
    m = np.ones(len(sc), bool)
    m[max(0, b - int(0.5 * RATE)):b + int(0.5 * RATE)] = False
    conf = float(sc[b] - (sc[m].max() if m.any() else 0))
    rows.append((t + WIN / 2, off, float(sc[b]), conf))

arr = np.array(rows)
np.save('fmap_nodes.npy', arr)
ok = ~np.isnan(arr[:, 1])
good = ok & (arr[:, 2] > 0.45) & (arr[:, 3] > 0.10)
print(f'узлов {len(arr)}, измерено {ok.sum()}, надёжных {good.sum()}')
print(f'качество: медиана {np.median(arr[ok,2]):.3f}')
d = arr[good, 1]
print(f'смещение: мин {d.min():.2f}s, макс {d.max():.2f}s')
# гладкость: разница соседних надёжных узлов
tt = arr[good, 0]
step = np.diff(d) / np.diff(tt)
print(f'скорость дрейфа: медиана {np.median(np.abs(step))*100:.2f} %, '
      f'макс {np.abs(step).max()*100:.2f} %')
