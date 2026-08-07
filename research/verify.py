"""Проверка результата: остаточный рассинхрон нового звука относительно 4K."""
import numpy as np
import afeat as A

FPS = A.FPS
chk = np.load('f_check.npy')
k4 = [np.load('f_4k_rus.npy'), np.load('f_4k_eng.npy')]
T = min(chk.shape[1], k4[0].shape[1])
cuts = [float(e[0]) for e in np.load('edges5.npy')]

WIN, STEP, SEARCH = 30.0, 30.0, 1.0
n = int(WIN * FPS)
res = []
for t in np.arange(10, T / FPS - WIN, STEP):
    i = int(t * FPS)
    lo, hi = max(0, i - int(SEARCH * FPS)), min(T, i + int(SEARCH * FPS) + n)
    if hi - lo < n:
        continue
    tot = None
    for f in k4:
        s = A.ncc(f[:, i:i + n], chk[:, lo:hi])
        tot = s if tot is None else tot + s
    sc = tot / len(k4)
    b = int(np.argmax(sc))
    if not (0 < b < len(sc) - 1):
        continue
    y0, y1, y2 = sc[b - 1], sc[b], sc[b + 1]
    den = y0 - 2 * y1 + y2
    dl = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-9 else 0.0
    off = (lo + b + dl - i) / FPS
    m = np.ones(len(sc), bool)
    m[max(0, b - int(0.4 * FPS)):b + int(0.4 * FPS)] = False
    conf = float(sc[b] - (sc[m].max() if m.any() else 0))
    res.append((t, off, float(sc[b]), conf))

a = np.array(res)
good = (a[:, 2] > 0.30) & (a[:, 3] > 0.08)
d = a[good, 1]
print(f'замеров {len(a)}, надёжных {good.sum()}')
print(f'остаточный рассинхрон: медиана {np.median(d)*1000:+.0f} мс, '
      f'СКО {np.std(d)*1000:.0f} мс')
print(f'   |ошибка| < 40 мс: {(np.abs(d) < 0.040).mean()*100:.0f}% замеров')
print(f'   |ошибка| < 80 мс: {(np.abs(d) < 0.080).mean()*100:.0f}% замеров')
print(f'   максимум {np.abs(d).max()*1000:.0f} мс')
worst = a[good][np.argsort(-np.abs(a[good, 1]))][:8]
print('\nхудшие точки:')
for t, off, pk, cf in worst:
    near = min(abs(t - c) for c in cuts)
    print(f'   {t/60:6.2f} мин  {off*1000:+7.0f} мс  (пик {pk:.2f}, до склейки {near:6.0f}s)')
