"""Перекладка звука 1080p на временную шкалу 4K.

Для каждого выходного отсчёта берём позицию в исходном звуке по карте
off(t) и интерполируем Catmull-Rom: дрейф компенсируется непрерывно, а не
ступеньками, поэтому стыков не слышно.
"""
import sys
import numpy as np
import wave
from scipy.io import wavfile

SR = 48000
OUT = sys.argv[1] if len(sys.argv) > 1 else 'ru_synced.wav'
DUR_4K = 6631.232

tm = np.load('map_t.npy')
dm = np.load('map_d.npy')
bounds = list(np.load('map_bounds.npy'))

sr, src = wavfile.read('hd_full.wav', mmap=True)
assert sr == SR, sr
NS = src.shape[0]
print(f'исходник {NS/SR:.2f}s, цель {DUR_4K:.2f}s, кусков {len(bounds)+1}')

# узлы по кускам между вставками
edges_all = [-1e9] + bounds + [1e9]
pieces = []
for k in range(len(edges_all) - 1):
    m = (tm >= edges_all[k]) & (tm < edges_all[k + 1])
    if m.sum() >= 2:
        pieces.append((edges_all[k], edges_all[k + 1], tm[m], dm[m]))
    else:
        pieces.append((edges_all[k], edges_all[k + 1], None, None))

n_out = int(round(DUR_4K * SR))
w = wave.open(OUT, 'wb')
w.setnchannels(2)
w.setsampwidth(2)
w.setframerate(SR)

BLK = SR * 20
for start in range(0, n_out, BLK):
    cnt = min(BLK, n_out - start)
    t = (start + np.arange(cnt)) / SR
    off = np.empty(cnt)
    for a, b, tk, dk in pieces:
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        off[m] = np.interp(t[m], tk, dk) if tk is not None else 0.0
    pos = (t + off) * SR
    i0 = np.floor(pos).astype(np.int64)
    f = (pos - i0).astype(np.float32)
    idx = np.clip(np.stack([i0 - 1, i0, i0 + 1, i0 + 2]), 0, NS - 1)
    p = src[idx].astype(np.float32)                    # (4, cnt, 2)
    f = f[:, None]
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    y = 0.5 * ((2 * p1) + (-p0 + p2) * f +
               (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 +
               (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    bad = (pos < 0) | (pos >= NS - 2)
    y[bad] = 0
    w.writeframes(np.clip(y, -32768, 32767).astype('<i2').tobytes())
    if start % (BLK * 15) == 0:
        print(f'  {start/SR:7.0f}s / {DUR_4K:.0f}s', flush=True)

w.close()
print('готово:', OUT)
