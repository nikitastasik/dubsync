"""Итоговая функция смещения off(t): гладкая внутри кусков, разрывная на вставках."""
import numpy as np

DUR_4K = 6631.232
nodes = np.load('fmap_nodes.npy')
edges = np.load('edges5.npy')            # (точка склейки, смещение слева, справа)
bias = float(np.load('audio_bias.npy')[0])

ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45) & (nodes[:, 3] > 0.10)
t, d = nodes[ok, 0], nodes[ok, 1]

# выбросы относительно локальной медианы
med = np.array([np.median(d[np.abs(t - x) < 30]) for x in t])
keep = np.abs(d - med) < 0.30
print(f'узлов {len(t)}, отброшено выбросов {(~keep).sum()}')
t, d = t[keep], d[keep]

cuts = [float(e[0]) for e in edges]
bounds = [-1e9] + cuts + [1e9]
T_OUT, D_OUT, resid = [], [], []
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    m = (t >= a) & (t < b)
    if m.sum() < 3:
        continue
    tk, dk = t[m], d[m]
    sm = np.empty_like(dk)
    for i, x in enumerate(tk):
        w = np.abs(tk - x) < 25
        sm[i] = np.polyfit(tk[w] - x, dk[w], 1)[1] if w.sum() >= 3 else dk[i]
    resid.append(dk - sm)
    # края куска: продлеваем измеренными смещениями с самой границы склейки
    lo_t = max(a, 0.0) if k == 0 else cuts[k - 1] + 1e-6
    hi_t = min(b, DUR_4K) if k == len(bounds) - 2 else cuts[k]
    lo_d = float(edges[k - 1][2]) if k > 0 else sm[0]
    hi_d = float(edges[k][1]) if k < len(cuts) else sm[-1]
    T_OUT.append(np.r_[lo_t, tk, hi_t])
    D_OUT.append(np.r_[lo_d, sm, hi_d])

r = np.concatenate(resid)
print(f'шум узлов после сглаживания: СКО {np.std(r)*1000:.0f} мс, '
      f'90-й проц {np.percentile(np.abs(r),90)*1000:.0f} мс')

tm = np.concatenate(T_OUT)
dm = np.concatenate(D_OUT) + bias
np.save('map_t.npy', tm)
np.save('map_d.npy', dm)
np.save('map_bounds.npy', np.array(cuts))
print(f'карта: {len(tm)} узлов, смещение {dm.min():+.2f}..{dm.max():+.2f}s, '
      f'разрывов {len(cuts)}, поправка звука {bias*1000:+.0f} мс')
for c in cuts:
    print(f'   разрыв в 4K {c:.3f}s')
