"""Нормированная кросс-корреляция многоканальных признаков.

Одна и та же функция работает и для аудио (полосы энергии), и для видео
(блоки яркости): на входе матрица (K, T) — K признаков на кадр.
"""
import numpy as np


def ncc(win, region):
    """Корреляция окна win (K, n) со всеми позициями region (K, m).

    Возвращает массив длины m-n+1: среднее по признакам значение корреляции
    Пирсона для каждого сдвига. None, если регион короче окна.
    """
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
        c1 = np.concatenate(([0.0], np.cumsum(r)))
        c2 = np.concatenate(([0.0], np.cumsum(r * r)))
        s1 = c1[n:] - c1[:-n]
        s2 = c2[n:] - c2[:-n]
        denom = np.sqrt(np.maximum(s2 - s1 * s1 / n, 1e-12))
        conv = np.fft.irfft(np.fft.rfft(r, size) * np.fft.rfft(w[::-1], size), size)
        corr = conv[n - 1:n - 1 + (m - n + 1)]
        scores += (corr - s1 * w.sum() / n) / denom
    return scores / K


def direct(a, b):
    """Корреляция двух блоков одинакового размера (K, n)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def parabolic(scores, b):
    """Субпиксельное уточнение позиции максимума параболой по трём точкам."""
    if not (0 < b < len(scores) - 1):
        return 0.0
    y0, y1, y2 = scores[b - 1], scores[b], scores[b + 1]
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))


def peak(scores, guard_frames):
    """Позиция максимума, его значение и отрыв от лучшего конкурента."""
    b = int(np.argmax(scores))
    mask = np.ones(len(scores), bool)
    mask[max(0, b - guard_frames):b + guard_frames] = False
    second = float(scores[mask].max()) if mask.any() else 0.0
    return b, float(scores[b]), float(scores[b]) - second
