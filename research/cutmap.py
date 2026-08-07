"""Независимая карта смещений по сменам сцен.

Для каждого участка фильма строим гистограмму смещений между всеми парами
смен (4K, HD). Верное смещение собирает сразу много совпадений — без
каких-либо догадок о его величине.
"""
import numpy as np
from vmatch2 import VFPS

t4 = np.load('cutidx4.npy') / VFPS
th = np.load('cutidxh.npy') / VFPS
DLO, DHI, BIN = -10.0, 100.0, 0.25
nb = int((DHI - DLO) / BIN) + 1

print(f'{"участок 4K":>16} {"смещение":>9} {"совп":>5} {"2-е место":>10} {"совп":>5}')
for T in np.arange(60, 6600, 120):
    sel = t4[(t4 >= T - 90) & (t4 < T + 90)]
    if len(sel) < 4:
        continue
    hist = np.zeros(nb)
    for a in sel:
        d = th - a
        d = d[(d >= DLO) & (d <= DHI)]
        for x in d:
            hist[int(round((x - DLO) / BIN))] += 1
    b = int(np.argmax(hist))
    best_d, best_n = DLO + b * BIN, hist[b]
    h2 = hist.copy()
    h2[max(0, b - 3):b + 4] = 0
    b2 = int(np.argmax(h2))
    print(f'{T-90:7.0f}..{T+90:<7.0f} {best_d:+9.2f} {best_n:5.0f} {DLO+b2*BIN:+10.2f} {h2[b2]:5.0f}'
          f'   {"" if best_n >= 2 * max(h2[b2], 1) else "<- неуверенно"}')
