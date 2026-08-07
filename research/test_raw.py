"""Проверка: коррелируют ли сами сэмплы (один ли мастер фонограммы).

Если да — точность выравнивания будет на уровне миллисекунд, а не 10 мс.
"""
import numpy as np
import afeat as A

SR = A.SR


def load(p):
    return A.read_wav_mono(p)


hd = load('hd_8k.wav')
tracks = {'rus': load('4k_rus_8k.wav'), 'eng': load('4k_eng_8k.wav')}


def gcc(x, y, phat):
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    X = np.fft.rfft(x, n)
    Y = np.fft.rfft(y, n)
    R = X * np.conj(Y)
    if phat:
        R /= (np.abs(R) + 1e-12)
    return np.fft.irfft(R, n)


for t, exp in ((2000, 42.5), (3000, 43.0), (5400, 76.9)):
    print(f'--- t={t}s (ожидаем ~{exp}s)')
    L = int(60 * SR)
    for name, tr in tracks.items():
        x = tr[int(t * SR):int(t * SR) + L].astype(np.float64)
        # окно поиска в HD: exp +/- 8 c
        s0 = int((t + exp - 8) * SR)
        y = hd[s0:s0 + L + int(16 * SR)].astype(np.float64)
        x -= x.mean(); y -= y.mean()
        for phat in (False, True):
            r = gcc(y, x, phat)          # пик на позиции сдвига y относительно x
            seg = r[:int(16 * SR)]
            b = int(np.argmax(seg))
            off = (s0 + b - t * SR) / SR
            noise = np.sqrt((seg ** 2).mean())
            print(f'   {name:3s} phat={int(phat)}  off={off:9.4f}s  peak/rms={seg[b]/noise:7.1f}')
