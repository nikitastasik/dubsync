"""Точность синхронизации по резким звуковым событиям.

Удары и выстрелы приходят из общей фонограммы и одинаковы в обеих озвучках,
поэтому разница их моментов — прямая мера рассинхрона, без корреляций.
"""
import numpy as np
import afeat as A

SR = A.SR
HOP = 40                      # 5 мс
ref = A.read_wav_mono('4k_rus_8k.wav')
new = A.read_wav_mono('check_8k.wav')
n = min(len(ref), len(new))


def env(x):
    m = len(x) // HOP
    e = np.abs(x[:m * HOP].reshape(m, HOP)).max(axis=1)
    return e


er, en = env(ref[:n]), env(new[:n])
# резкость нарастания: отношение к предыдущим 100 мс
def onset(e):
    prev = np.convolve(e, np.ones(20) / 20, mode='same')
    prev = np.r_[np.zeros(20), prev[:-20]] + 1e-4
    return e / prev


orr, onn = onset(er), onset(en)
cand = np.where((orr > 6) & (er > 0.15))[0]
# оставляем изолированные события
keep = [c for i, c in enumerate(cand)
        if (i == 0 or c - cand[i - 1] > 60) and (i == len(cand) - 1 or cand[i + 1] - c > 60)]
print(f'найдено резких событий в опорной дорожке: {len(keep)}')

diffs = []
for c in keep:
    lo, hi = max(0, c - 40), min(len(onn), c + 40)       # ±200 мс
    seg = onn[lo:hi] * en[lo:hi]
    if seg.max() < 0.1:
        continue
    k = lo + int(np.argmax(seg))
    if onn[k] < 4:
        continue
    diffs.append((c * HOP / SR, (k - c) * HOP / SR * 1000))

d = np.array([x[1] for x in diffs])
print(f'сопоставлено событий: {len(d)}')
print(f'смещение события: медиана {np.median(d):+.0f} мс, СКО {np.std(d):.0f} мс')
for lim in (20, 40, 80):
    print(f'   |ошибка| <= {lim} мс: {(np.abs(d) <= lim).mean()*100:.0f}%')
big = [x for x in diffs if abs(x[1]) > 80]
print(f'\nсобытий с ошибкой > 80 мс: {len(big)}')
for t, v in big[:10]:
    print(f'   {t/60:6.2f} мин: {v:+.0f} мс')
