"""Проверка: какая дорожка 4K лучше коррелирует с 1080p, и каково стартовое смещение."""
import numpy as np
import afeat as A

hd = A.normalize(A.features(A.read_wav_mono('hd_8k.wav')))
np.save('f_hd.npy', hd)
print('hd frames', hd.shape)

for name, path in (('rus', '4k_rus_8k.wav'), ('eng', '4k_eng_8k.wav')):
    f = A.normalize(A.features(A.read_wav_mono(path)))
    np.save(f'f_4k_{name}.npy', f)
    print(f'4k_{name} frames', f.shape)
    for t in (600, 1800, 3000, 4200, 5400):
        i = int(t * A.FPS)
        n = int(20 * A.FPS)
        win = f[:, i:i + n]
        lo = max(0, i - int(30 * A.FPS))
        hi = min(hd.shape[1], i + n + int(150 * A.FPS))
        sc = A.ncc(win, hd[:, lo:hi])
        best = int(np.argmax(sc))
        srt = np.sort(sc)[::-1]
        print(f'  t={t:5d}s  off={(lo + best - i) / A.FPS:+8.3f}s  peak={srt[0]:.3f}  2nd={srt[int(2 * A.FPS)]:.3f}')
