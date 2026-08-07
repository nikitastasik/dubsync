"""Где корреляция надёжна, а где нет: лучший пик и его позиция по минутам."""
import numpy as np
import afeat as A

FPS = A.FPS
DMIN = int(-5 * FPS)
S = np.load('S.npy')
starts = np.load('starts.npy')

print(' мин |  пик  | 2-й  | смещение лучшего пика (топ-3)')
for w in range(0, S.shape[0], 12):     # раз в минуту
    row = S[w]
    b = int(np.argmax(row))
    tops = []
    r = row.copy()
    for _ in range(3):
        j = int(np.argmax(r))
        tops.append((j + DMIN) / FPS)
        r[max(0, j - int(3 * FPS)):j + int(3 * FPS)] = -9
    second = r.max()
    print(f'{starts[w]/FPS/60:5.1f} | {row[b]:.3f} | {second:.3f} | ' +
          '  '.join(f'{t:7.2f}' for t in tops))
