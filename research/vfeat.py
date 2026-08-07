"""Видео-признаки на полном фреймрейте, блочно (потоки по ~200 МБ)."""
import numpy as np
from scipy.ndimage import zoom

W, H = 48, 27
ROWS = 20
RATE = 24000 / 1001


def feats_stream(path, crop, chunk=20000):
    import os
    n_total = os.path.getsize(path) // (W * H)
    out = np.empty((32, n_total), dtype=np.float32)
    for s in range(0, n_total, chunk):
        cnt = min(chunk, n_total - s)
        a = np.fromfile(path, dtype=np.uint8, count=cnt * W * H,
                        offset=s * W * H).reshape(cnt, H, W).astype(np.float32)
        a = a[:, 4:4 + ROWS, :] if crop else zoom(a, (1, ROWS / H, 1), order=1)
        b = a.reshape(cnt, 4, ROWS // 4, 8, W // 8).mean(axis=(2, 4)).reshape(cnt, 32).T
        b -= b.mean(axis=0, keepdims=True)
        b /= (b.std(axis=0, keepdims=True) + 1e-6)
        out[:, s:s + cnt] = b
    return out


if __name__ == '__main__':
    f4 = feats_stream('f4k.raw', True)
    np.save('F4.npy', f4)
    print('4K', f4.shape, f4.shape[1] / RATE, 'с')
    fh = feats_stream('fhd.raw', False)
    np.save('FH.npy', fh)
    print('HD', fh.shape, fh.shape[1] / RATE, 'с')
