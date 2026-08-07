"""Форма корреляционного пика: острый или пологий?"""
import numpy as np
import afeat as A

FPS = A.FPS
DMIN = int(-5 * FPS)
S = np.load('S.npy')
starts = np.load('starts.npy')

for minute in (22, 32, 48, 55, 78, 81, 102, 106):
    w = int(np.argmin(np.abs(starts / FPS - minute * 60)))
    row = S[w]
    b = int(np.argmax(row))
    print(f'\n=== мин {minute}: максимум на {(b+DMIN)/FPS:.2f}s, значение {row[b]:.3f}')
    # ширина пика на половине высоты
    half = row[b] / 2
    l = b
    while l > 0 and row[l] > half:
        l -= 1
    r = b
    while r < len(row) - 1 and row[r] > half:
        r += 1
    print(f'    ширина на половине высоты: {(r-l)/FPS*1000:.0f} мс')
    lo, hi = b - int(2.5 * FPS), b + int(2.5 * FPS) + 1
    prof = row[max(0, lo):hi]
    show = prof[::10]                                    # шаг 100 мс
    ts = (np.arange(len(show)) * 10 + max(0, lo) + DMIN) / FPS
    for k in range(0, len(show), 10):
        chunk = ' '.join(f'{v:5.2f}' for v in show[k:k + 10])
        print(f'    {ts[k]:7.2f}s: {chunk}')
