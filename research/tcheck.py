"""Проверка карты по звуковым событиям между исходными дорожками.

Берём резкое событие в звуке 4K, смотрим, куда его переносит карта, и ищем
соответствующее событие в звуке 1080p. Систематическая разница = ошибка карты.
"""
import numpy as np
import afeat as A

SR = A.SR
HOP = 40                     # 5 мс
ref = A.read_wav_mono('4k_rus_8k.wav')
eng = A.read_wav_mono('4k_eng_8k.wav')
hd = A.read_wav_mono('hd_8k.wav')
tm, dm = np.load('map_t.npy'), np.load('map_d.npy')
cuts = [float(e[0]) for e in np.load('edges5.npy')]


def env(x):
    m = len(x) // HOP
    return np.abs(x[:m * HOP].reshape(m, HOP)).max(axis=1)


def onset(e):
    prev = np.convolve(e, np.ones(20) / 20, mode='same')
    prev = np.r_[np.zeros(20), prev[:-20]] + 1e-4
    return e / prev


er, ee, eh = env(ref), env(eng), env(hd)
orr, oe, oh = onset(er), onset(ee), onset(eh)

# событие должно быть резким в обеих дорожках 4K — значит оно из общей фонограммы
cand = np.where((orr > 3.0) & (er > 0.06) & (oe > 3.0))[0]
keep = [c for i, c in enumerate(cand)
        if (i == 0 or c - cand[i - 1] > 40) and (i == len(cand) - 1 or cand[i + 1] - c > 40)]
print(f'событий общей фонограммы: {len(keep)}')

res = []
for c in keep:
    t = c * HOP / SR
    if any(abs(t - x) < 20 for x in cuts):
        continue
    off = float(np.interp(t, tm, dm))
    j = int(round((t + off) * SR / HOP))
    lo, hi = max(0, j - 60), min(len(oh), j + 60)         # ±300 мс
    seg = oh[lo:hi] * eh[lo:hi]
    if len(seg) == 0 or seg.max() < 0.04:
        continue
    k = lo + int(np.argmax(seg))
    if oh[k] < 2.5:
        continue
    res.append((t, (k - j) * HOP / SR * 1000))

d = np.array([x[1] for x in res])
print(f'сопоставлено: {len(d)}')
print(f'ошибка карты по событиям: медиана {np.median(d):+.0f} мс, '
      f'СКО {np.std(d):.0f} мс, среднее {np.mean(d):+.0f} мс')
for lim in (20, 40, 80):
    print(f'   |ошибка| <= {lim} мс: {(np.abs(d) <= lim).mean()*100:.0f}%')
