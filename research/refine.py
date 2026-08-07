"""Точная карта смещения: видео задаёт структуру, аудио — миллисекунды.

Видео надёжно ловит вставки и общий ход дрейфа, но с шагом 0.25 с. Аудио
в узком окне вокруг видео-предсказания даёт позицию пика с точностью до
единиц миллисекунд (парабола по трём точкам вокруг максимума).
"""
import numpy as np
import afeat as A
from vmatch2 import VFPS

FPS = A.FPS
NODE = 10.0        # шаг узлов, с
WIN = 24.0         # окно корреляции, с
SEARCH = 1.5       # полуширина поиска вокруг видео-предсказания, с

hd = np.load('f_hd.npy')
k4 = [np.load('f_4k_rus.npy'), np.load('f_4k_eng.npy')]
THD, T4 = hd.shape[1], k4[0].shape[1]

vo = np.load('v_offsets2.npy')
vs = np.load('vstarts2.npy') / VFPS
# границы крупных вставок (там функция разрывна)
JUMPS = []
for w in range(1, len(vo)):
    if vo[w] - vo[w - 1] > 5:
        JUMPS.append(vs[w])
print('крупные вставки около:', [f'{t:.1f}s' for t in JUMPS])


def predict(t):
    return float(np.interp(t, vs, vo))


rows = []
for t in np.arange(0, (T4 / FPS) - WIN, NODE):
    # окна, пересекающие вставку, пропускаем: там нет единого смещения
    if any(t - 2 < j < t + WIN + 2 for j in JUMPS):
        rows.append((t, np.nan, 0.0, 0.0))
        continue
    p = predict(t + WIN / 2)
    i = int(t * FPS)
    n = int(WIN * FPS)
    lo = max(0, i + int((p - SEARCH) * FPS))
    hi = min(THD, i + int((p + SEARCH) * FPS) + n)
    if hi - lo < n:
        rows.append((t, np.nan, 0.0, 0.0))
        continue
    tot = None
    for f in k4:
        s = A.ncc(f[:, i:i + n], hd[:, lo:hi])
        tot = s if tot is None else tot + s
    sc = tot / len(k4)
    b = int(np.argmax(sc))
    # субфреймовое уточнение параболой
    if 0 < b < len(sc) - 1:
        y0, y1, y2 = sc[b - 1], sc[b], sc[b + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
    else:
        delta = 0.0
    off = (lo + b + delta - i) / FPS
    m = np.ones(len(sc), bool)
    m[max(0, b - int(0.4 * FPS)):b + int(0.4 * FPS)] = False
    conf = float(sc[b] - (sc[m].max() if m.any() else 0))
    rows.append((t + WIN / 2, off, float(sc[b]), conf))

arr = np.array(rows)
np.save('refined_nodes.npy', arr)
ok = ~np.isnan(arr[:, 1])
good = ok & (arr[:, 2] > 0.25) & (arr[:, 3] > 0.05)
print(f'узлов {len(arr)}, измерено {ok.sum()}, надёжных {good.sum()}')
resid = arr[good, 1] - np.array([predict(t) for t in arr[good, 0]])
print(f'расхождение аудио и видео: медиана {np.median(resid)*1000:+.0f} мс, '
      f'разброс {np.percentile(np.abs(resid-np.median(resid)),90)*1000:.0f} мс (90-й проц.)')
