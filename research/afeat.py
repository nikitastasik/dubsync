"""Спектральные признаки и нормированная кросс-корреляция для выравнивания дорожек."""
import numpy as np
import wave

SR = 8000
HOP = 80          # 10 ms -> 100 fps
WIN = 512         # 64 ms
FPS = SR / HOP    # 100

BANDS = [(50, 150), (150, 300), (300, 500), (500, 800),
         (800, 1200), (1200, 1800), (1800, 2600), (2600, 3900)]


def read_wav_mono(path):
    with wave.open(path, 'rb') as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, (w.getnchannels(), w.getsampwidth())
        assert w.getframerate() == SR, w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype='<i2')
    return data.astype(np.float32) / 32768.0


def features(x):
    """(K, T) float32 — лог-энергия по полосам, шаг 10 мс."""
    nfr = 1 + (len(x) - WIN) // HOP
    window = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    idx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in BANDS]
    out = np.empty((len(BANDS), nfr), dtype=np.float32)
    block = 40000
    for s in range(0, nfr, block):
        e = min(s + block, nfr)
        starts = np.arange(s, e) * HOP
        frames = x[np.add.outer(starts, np.arange(WIN))] * window
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        for k, ix in enumerate(idx):
            out[k, s:e] = p[:, ix].sum(axis=1)
    np.log10(out + 1e-10, out=out)
    return out


def normalize(f, smooth_s=3.0):
    """Убрать медленную составляющую (разница громкости/мастеринга)."""
    n = int(smooth_s * FPS)
    ker = np.ones(n, dtype=np.float32) / n
    out = np.empty_like(f)
    for k in range(f.shape[0]):
        pad = np.pad(f[k], (n // 2, n - n // 2 - 1), mode='edge')
        out[k] = f[k] - np.convolve(pad, ker, mode='valid')[:f.shape[1]]
    out /= (out.std(axis=1, keepdims=True) + 1e-6)
    return out


def ncc(win, region):
    """Нормированная корреляция окна win (K,n) по region (K,m). -> (m-n+1,) баллов."""
    K, n = win.shape
    m = region.shape[1]
    if m < n:
        return None
    size = 1 << int(np.ceil(np.log2(m + n)))
    scores = np.zeros(m - n + 1, dtype=np.float64)
    for k in range(K):
        w = win[k] - win[k].mean()
        wn = np.linalg.norm(w)
        if wn < 1e-6:
            continue
        w = w / wn
        r = region[k].astype(np.float64)
        # скользящие сумма и сумма квадратов
        c1 = np.concatenate(([0.0], np.cumsum(r)))
        c2 = np.concatenate(([0.0], np.cumsum(r * r)))
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        denom = np.sqrt(np.maximum(s2 - s1 * s1 / n, 1e-12))
        conv = np.fft.irfft(np.fft.rfft(r, size) * np.fft.rfft(w[::-1], size), size)
        corr = conv[n - 1:n - 1 + (m - n + 1)]
        scores += (corr - s1 * w.sum() / n) / denom
    return scores / K
