"""Самые яркие моменты в обоих потоках — независимые опорные точки."""
import numpy as np

W, H, VFPS = 48, 27, 4.0


def curve(path, crop):
    a = np.fromfile(path, dtype=np.uint8)
    n = len(a) // (W * H)
    f = a[:n * W * H].reshape(n, H, W).astype(np.float32)
    if crop:
        f = f[:, 4:24, :]
    return f.mean(axis=(1, 2))


c4 = curve('v4k.raw', True)
ch = curve('vhd.raw', False)
np.save('bright4.npy', c4)
np.save('brighth.npy', ch)
print(f'4K: среднее {c4.mean():.1f}, макс {c4.max():.1f}')
print(f'HD: среднее {ch.mean():.1f}, макс {ch.max():.1f}')


def peaks(c, name):
    print(f'\n--- {name}: самые яркие моменты')
    order = np.argsort(c)[::-1]
    shown = []
    for j in order:
        if any(abs(j - s) < 20 * VFPS for s in shown):
            continue
        shown.append(j)
        print(f'   {j/VFPS:9.2f}s  ({j/VFPS/60:6.2f} мин)  яркость {c[j]:.0f}')
        if len(shown) >= 12:
            break
    return shown


peaks(c4, '4K')
peaks(ch, 'HD')
