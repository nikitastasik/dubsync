"""Признаки звука: лог-энергия по полосам, шаг 10 мс.

Работает даже когда озвучки разные: общая основа (музыка, шумы, эффекты)
даёт корреляцию, а вычитание медленной составляющей убирает разницу
громкости и мастеринга.
"""
import wave

import numpy as np

SR = 8000
HOP = 80                  # 10 мс
WIN = 512                 # 64 мс
FPS = SR / HOP            # 100 кадров признаков в секунду

BANDS = [(50, 150), (150, 300), (300, 500), (500, 800),
         (800, 1200), (1200, 1800), (1800, 2600), (2600, 3900)]


def read_wav_mono(path):
    with wave.open(path, 'rb') as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
            raise ValueError(f'{path}: ожидается моно 16 бит {SR} Гц')
        data = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
    return data.astype(np.float32) / 32768.0


def spectra(x):
    nfr = 1 + (len(x) - WIN) // HOP
    window = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    idx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in BANDS]
    out = np.empty((len(BANDS), nfr), dtype=np.float32)
    for s in range(0, nfr, 40000):
        e = min(s + 40000, nfr)
        starts = np.arange(s, e) * HOP
        frames = x[np.add.outer(starts, np.arange(WIN))] * window
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        for k, ix in enumerate(idx):
            out[k, s:e] = p[:, ix].sum(axis=1)
    np.log10(out + 1e-10, out=out)
    return out


def normalize(f, smooth_s=3.0):
    """Убрать медленные изменения громкости, оставить рисунок."""
    n = int(smooth_s * FPS)
    ker = np.ones(n, dtype=np.float32) / n
    out = np.empty_like(f)
    for k in range(f.shape[0]):
        pad = np.pad(f[k], (n // 2, n - n // 2 - 1), mode='edge')
        out[k] = f[k] - np.convolve(pad, ker, mode='valid')[:f.shape[1]]
    out /= (out.std(axis=1, keepdims=True) + 1e-6)
    return out


def features(path):
    return normalize(spectra(read_wav_mono(path)))
