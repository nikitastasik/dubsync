"""Карта смещений 4K -> 1080p последовательным трекингом.

Смещение может только расти (реклама добавляет время в 1080p), поэтому идём
слева направо в узком окне поиска и фиксируем ступени.
"""
import numpy as np
import afeat as A

FPS = A.FPS

# спектральный поток пробовали — на разных микшированиях он некоррелирован
# и разбавляет пик (0.5 -> 0.03), поэтому только полосы уровня
hd = np.load('f_hd.npy')
k4 = [np.load(f'f_4k_{n}.npy') for n in ('rus', 'eng')]
T4, THD = k4[0].shape[1], hd.shape[1]


def search(i, n, dmin, dmax):
    """Корреляция окна 4K [i,i+n) при смещениях [dmin,dmax] (в фреймах).

    Возвращает (смещение, пик, уверенность) — уверенность = пик минус лучший
    конкурент за пределами ±2 c от найденного максимума.
    """
    lo = max(0, i + dmin)
    hi = min(THD, i + dmax + n)
    if hi - lo < n:
        return None, 0.0, 0.0
    tot = None
    for f in k4:
        s = A.ncc(f[:, i:i + n], hd[:, lo:hi])
        tot = s if tot is None else tot + s
    sc = tot / len(k4)
    b = int(np.argmax(sc))
    guard = int(2 * FPS)
    mask = np.ones(len(sc), bool)
    mask[max(0, b - guard):b + guard] = False
    second = float(sc[mask].max()) if mask.any() else 0.0
    return lo + b - i, float(sc[b]), float(sc[b]) - second


if __name__ == '__main__':
    print(f'4K {T4/FPS:.1f}s   HD {THD/FPS:.1f}s   diff {(THD-T4)/FPS:.2f}s', flush=True)

    # стартовое смещение: громкий кусок ближе к концу, где вставки уже все позади
    off, peak, conf = search(int(5400 * FPS), int(60 * FPS), int(-10 * FPS), int(120 * FPS))
    print(f'seed @5400s: off={off/FPS:.3f}s peak={peak:.3f} conf={conf:.3f}', flush=True)

    WIN = int(40 * FPS)
    STEP = int(10 * FPS)
    BACK = int(-4 * FPS)   # допуск назад на шум
    FWD = int(50 * FPS)    # максимум добавленного времени между замерами

    # идём назад от seed к началу, затем вперёд — так стартуем от надёжной точки
    rows = []
    cur = off
    for i in range(int(5400 * FPS), -1, -STEP):
        d, p, c = search(i, WIN, cur - FWD, cur - BACK)
        if d is not None and p > 0.20 and c > 0.05:
            cur = d
        rows.append((i, cur, d, p, c))
    back_rows = rows[::-1]

    cur = off
    rows = []
    for i in range(int(5400 * FPS) + STEP, T4 - WIN, STEP):
        d, p, c = search(i, WIN, cur + BACK, cur + FWD)
        if d is not None and p > 0.20 and c > 0.05:
            cur = d
        rows.append((i, cur, d, p, c))
    all_rows = back_rows + rows

    np.save('track_rows.npy', np.array(all_rows, dtype=float))
    prev = None
    for i, cur, d, p, c in all_rows:
        mark = ''
        if prev is not None and abs(cur - prev) > int(0.5 * FPS):
            mark = f'   <<< STEP {(cur - prev)/FPS:+.2f}s'
        raw = 'None' if d is None else f'{d/FPS:8.3f}'
        print(f't={i/FPS:7.1f}  off={cur/FPS:8.3f}  raw={raw}  peak={p:.3f} conf={c:.3f}{mark}', flush=True)
        prev = cur
