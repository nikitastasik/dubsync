"""Сопоставление по видео: надёжная карта соответствия 4K -> 1080p.

Кадры уменьшены до 48x27 серых пикселей при 4 fps. Каждый кадр сводится к
32 признакам (блоки 6x4), нормированным по кадру — это гасит разницу
HDR/SDR и оставляет структуру сцены.
"""
import numpy as np
import afeat as A

W, H, VFPS = 48, 27, 4.0


def load(path):
    a = np.fromfile(path, dtype=np.uint8)
    n = len(a) // (W * H)
    a = a[:n * W * H].reshape(n, H, W).astype(np.float32)
    # блоки: 27x48 -> 4x8 (усреднение)
    b = a[:, :24, :48].reshape(n, 4, 6, 8, 6).mean(axis=(2, 4))   # (n,4,8)
    f = b.reshape(n, 32).T                                        # (32, n)
    f -= f.mean(axis=0, keepdims=True)                            # убрать общую яркость кадра
    f /= (f.std(axis=0, keepdims=True) + 1e-6)                    # и общий контраст
    return f


if __name__ == '__main__':
    v4 = load('v4k.raw')
    vh = load('vhd.raw')
    np.save('v4.npy', v4)
    np.save('vh.npy', vh)
    print(f'4K {v4.shape[1]/VFPS:.1f}s   HD {vh.shape[1]/VFPS:.1f}s')
