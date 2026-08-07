"""Видео-признаки с одинаковой геометрией кадра.

4K хранит картинку 2.4:1 внутри кадра 16:9 (чёрные полосы ~13% сверху и снизу),
1080p уже обрезан до 2.4:1. Без учёта этого сигнатуры описывают разные области.
"""
import numpy as np
from scipy.ndimage import zoom

W, H, VFPS = 48, 27, 4.0
ROWS = 20            # общая сетка активной области


def _raw(path):
    a = np.fromfile(path, dtype=np.uint8)
    n = len(a) // (W * H)
    return a[:n * W * H].reshape(n, H, W).astype(np.float32)


def load4k(path='v4k.raw'):
    a = _raw(path)
    # активная часть кадра 16:9 -> строки с 3.5 по 23.5 из 27
    return a[:, 4:4 + ROWS, :]


def loadhd(path='vhd.raw'):
    a = _raw(path)
    return zoom(a, (1, ROWS / H, 1), order=1)


def feats(a):
    """(32, n): блоки 4x8, нормировано по кадру."""
    n = a.shape[0]
    b = a[:, :ROWS, :].reshape(n, 4, ROWS // 4, 8, W // 8).mean(axis=(2, 4))
    f = b.reshape(n, 32).T.astype(np.float32)
    f -= f.mean(axis=0, keepdims=True)
    f /= (f.std(axis=0, keepdims=True) + 1e-6)
    return f


if __name__ == '__main__':
    f4 = feats(load4k())
    fh = feats(loadhd())
    np.save('g4.npy', f4)
    np.save('gh.npy', fh)
    print('4K', f4.shape, '  HD', fh.shape)

    # где в 1080p находится сцена из 4K
    import sys
    for t in (float(x) for x in sys.argv[1:] or [1420, 300, 2500, 5000]):
        i = int(t * VFPS)
        n = int(6 * VFPS)
        win = f4[:, i:i + n]
        sc = np.zeros(fh.shape[1] - n + 1)
        w = win - win.mean(axis=1, keepdims=True)
        wn = np.linalg.norm(w)
        for j in range(len(sc)):
            r = fh[:, j:j + n]
            sc[j] = (w * (r - r.mean(axis=1, keepdims=True))).sum() / (wn * np.linalg.norm(r - r.mean(axis=1, keepdims=True)) + 1e-9)
        order = np.argsort(sc)[::-1]
        print(f'\n4K t={t}s — лучшие совпадения в 1080p:')
        shown = []
        for j in order:
            if any(abs(j - s) < 4 * VFPS for s in shown):
                continue
            shown.append(j)
            print(f'   HD {j/VFPS:9.2f}s  (смещение {(j-i)/VFPS:+7.2f}s)  сходство {sc[j]:.3f}')
            if len(shown) >= 4:
                break
