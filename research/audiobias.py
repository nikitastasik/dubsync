"""Систематический сдвиг звука относительно видео.

Карта строится по видео. Если в одном из релизов звук смещён относительно
картинки, это надо учесть — иначе идеально совпадёт видеоряд, а не звук.
"""
import numpy as np
import afeat as A
from vfeat import RATE

FPS = A.FPS
hd = np.load('f_hd.npy')
k4 = [np.load('f_4k_rus.npy'), np.load('f_4k_eng.npy')]
THD, T4 = hd.shape[1], k4[0].shape[1]

nodes = np.load('fmap_nodes.npy')
ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45) & (nodes[:, 3] > 0.10)
tt, dd = nodes[ok, 0], nodes[ok, 1]
edges = np.load('edges5.npy')
cuts = [e[0] for e in edges]

WIN = 30.0
SEARCH = 0.8
res = []
for t in np.arange(20, T4 / FPS - WIN, 20.0):
    if any(t - 2 < c < t + WIN + 2 for c in cuts):
        continue
    p = float(np.interp(t + WIN / 2, tt, dd))
    i = int(t * FPS)
    n = int(WIN * FPS)
    lo = max(0, i + int((p - SEARCH) * FPS))
    hi = min(THD, i + int((p + SEARCH) * FPS) + n)
    if hi - lo < n:
        continue
    tot = None
    for f in k4:
        s = A.ncc(f[:, i:i + n], hd[:, lo:hi])
        tot = s if tot is None else tot + s
    sc = tot / len(k4)
    b = int(np.argmax(sc))
    if not (0 < b < len(sc) - 1):
        continue
    y0, y1, y2 = sc[b - 1], sc[b], sc[b + 1]
    den = y0 - 2 * y1 + y2
    delta = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-9 else 0.0
    off = (lo + b + delta - i) / FPS
    m = np.ones(len(sc), bool)
    m[max(0, b - int(0.4 * FPS)):b + int(0.4 * FPS)] = False
    conf = float(sc[b] - (sc[m].max() if m.any() else 0))
    if sc[b] > 0.30 and conf > 0.08:
        res.append(off - p)

r = np.array(res)
bias = float(np.median(r))
np.save('audio_bias.npy', np.array([bias]))
print(f'замеров {len(r)}')
print(f'сдвиг звука относительно видео: {bias*1000:+.0f} мс '
      f'(разброс ±{np.percentile(np.abs(r-bias),68)*1000:.0f} мс)')
print(f'точность оценки: ±{np.std(r)/np.sqrt(len(r))*1000:.0f} мс')
