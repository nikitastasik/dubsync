"""Карта смещений 4K->1080p: профили корреляции + монотонный путь через DP.

Смещение может только расти (реклама добавляет время), скачки редки и дороги,
поэтому DP со штрафом за скачок выделяет настоящие ступени из шума.
"""
import numpy as np
import afeat as A

FPS = A.FPS
DMIN = int(-5 * FPS)
DMAX = int(95 * FPS)
D = DMAX - DMIN + 1
WIN = int(30 * FPS)
STEP = int(5 * FPS)

hd = np.load('f_hd.npy')
k4 = [np.load('f_4k_rus.npy'), np.load('f_4k_eng.npy')]
T4, THD = k4[0].shape[1], hd.shape[1]

starts = list(range(0, T4 - WIN, STEP))
S = np.full((len(starts), D), -1.0, dtype=np.float32)

for w, i in enumerate(starts):
    lo = i + DMIN
    hi = i + DMAX + WIN
    clo, chi = max(0, lo), min(THD, hi)
    if chi - clo < WIN:
        continue
    tot = None
    for f in k4:
        s = A.ncc(f[:, i:i + WIN], hd[:, clo:chi])
        tot = s if tot is None else tot + s
    sc = tot / len(k4)
    j0 = clo - lo                      # индекс первого посчитанного смещения
    S[w, j0:j0 + len(sc)] = sc
    if w % 100 == 0:
        print(f'  {w}/{len(starts)}', flush=True)

np.save('S.npy', S)
np.save('starts.npy', np.array(starts))
print('профили готовы', S.shape, flush=True)

# --- DP: путь неубывающий по смещению, штраф за скачок
PEN = 0.60      # цена самого факта скачка
EPS = 0.00004   # цена за фрейм величины скачка (10 мс)

best = S[0].astype(np.float64).copy()
back = np.zeros((len(starts), D), dtype=np.int32)
ramp = np.arange(D) * EPS
for w in range(1, len(starts)):
    shifted = best + ramp
    pm = np.maximum.accumulate(shifted)
    argmax_pref = np.maximum.accumulate(
        np.where(shifted >= pm, np.arange(D), 0)).astype(np.int32)
    jump = pm - PEN - ramp
    stay = best
    take_jump = jump > stay
    best = np.where(take_jump, jump, stay) + S[w]
    back[w] = np.where(take_jump, argmax_pref, np.arange(D))

path = np.zeros(len(starts), dtype=np.int32)
path[-1] = int(np.argmax(best))
for w in range(len(starts) - 1, 0, -1):
    path[w - 1] = back[w][path[w]]

offs = (path + DMIN) / FPS
np.save('dp_offsets.npy', offs)

print('\n=== найденные ступени:')
prev = offs[0]
print(f'старт: смещение {prev:.3f}s')
for w in range(1, len(starts)):
    if abs(offs[w] - prev) > 0.02:
        t = starts[w] / FPS
        print(f'  ~{t:8.1f}s ({t//60:3.0f}:{t%60:05.2f})  {prev:8.3f} -> {offs[w]:8.3f}   вставка {offs[w]-prev:+.3f}s')
        prev = offs[w]
print(f'конец: смещение {offs[-1]:.3f}s   (разница длительностей {(THD-T4)/FPS:.2f}s)')
