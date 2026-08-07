"""Сборка гладкой функции смещения off(t) с разрывами на вставках."""
import numpy as np

nodes = np.load('fmap_nodes.npy')
edges = np.load('edges.npy')          # (T, off_before, off_after)
AUDIO_BIAS = float(np.load('audio_bias.npy')) if __import__('os').path.exists('audio_bias.npy') else 0.0

ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45) & (nodes[:, 3] > 0.10)
t, d = nodes[ok, 0], nodes[ok, 1]

# выбросы: отклонение от локальной медианы больше 0.35 с
med = np.array([np.median(d[np.abs(t - x) < 40]) for x in t])
keep = np.abs(d - med) < 0.35
t, d = t[keep], d[keep]
print(f'узлов после чистки: {len(t)}')

# сглаживание внутри каждого куска между вставками
bounds = [0.0] + [float(e[0]) for e in edges] + [1e9]
T_OUT, D_OUT = [], []
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    m = (t >= a) & (t < b)
    if m.sum() < 2:
        continue
    tk, dk = t[m], d[m]
    # локальная линейная регрессия (LOESS-подобная) окном 60 с
    sm = np.empty_like(dk)
    for i, x in enumerate(tk):
        w = np.abs(tk - x) < 45
        if w.sum() >= 3:
            p = np.polyfit(tk[w] - x, dk[w], 1)
            sm[i] = p[1]
        else:
            sm[i] = dk[i]
    # края куска продлеваем до самих границ вставки
    T_OUT.append(np.r_[max(a, tk[0] - 60), tk, min(b, tk[-1] + 60)])
    D_OUT.append(np.r_[sm[0], sm, sm[-1]])

tm = np.concatenate(T_OUT)
dm = np.concatenate(D_OUT) + AUDIO_BIAS
np.save('map_t.npy', tm)
np.save('map_d.npy', dm)
np.save('map_bounds.npy', np.array(bounds[1:-1]))
print(f'карта: {len(tm)} узлов, смещение {dm.min():.2f}..{dm.max():.2f}s, '
      f'поправка звука {AUDIO_BIAS*1000:+.0f} мс')
