"""Признаки видеоряда для сопоставления двух релизов.

Кадр сводится к 32 числам (сетка 4x8 средних яркостей), нормированным внутри
кадра — это гасит разницу HDR/SDR, кодеков и общей яркости, оставляя структуру
сцены. Главная тонкость — геометрия: один релиз может хранить картинку 2.4:1
внутри кадра 16:9 (чёрные полосы), другой — уже обрезанной. Без приведения к
общей активной области сопоставление разваливается.
"""
import os

import numpy as np
from scipy.ndimage import zoom

ROWS = 20          # общая сетка активной области
GRID = (4, 8)      # блоки признаков: строк x столбцов


def open_raw(path, width=48, height=27):
    n = os.path.getsize(path) // (width * height)
    return np.memmap(path, dtype=np.uint8, mode='r',
                     shape=(n, height, width))


def detect_active_rows(frames, sample=200):
    """Строки, где есть картинка (не чёрные полосы letterbox)."""
    idx = np.linspace(0, frames.shape[0] - 1, min(sample, frames.shape[0])).astype(int)
    sub = frames[idx].astype(np.float32)
    std = sub.std(axis=(0, 2))
    thr = max(1.5, 0.12 * float(np.median(std)))
    active = std > thr
    if not active.any():
        return 0, frames.shape[1]
    # самая длинная непрерывная полоса активных строк
    best = cur = None
    for i, a in enumerate(active):
        if a:
            cur = (cur[0], i + 1) if cur else (i, i + 1)
            if best is None or cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    return best


def features(frames, rows=None, chunk=20000):
    """(32, n) float32: блоки яркости активной области, нормированные по кадру."""
    if rows is None:
        rows = detect_active_rows(frames)
    top, bot = rows
    n_total = frames.shape[0]
    gr, gc = GRID
    out = np.empty((gr * gc, n_total), dtype=np.float32)
    for s in range(0, n_total, chunk):
        a = frames[s:s + chunk, top:bot, :].astype(np.float32)
        cnt = a.shape[0]
        if a.shape[1] != ROWS:
            a = zoom(a, (1, ROWS / a.shape[1], 1), order=1)
        h, w = a.shape[1], a.shape[2]
        a = a[:, :ROWS, :(w // gc) * gc]
        b = a.reshape(cnt, gr, ROWS // gr, gc, a.shape[2] // gc).mean(axis=(2, 4))
        b = b.reshape(cnt, gr * gc).T
        b -= b.mean(axis=0, keepdims=True)
        b /= (b.std(axis=0, keepdims=True) + 1e-6)
        out[:, s:s + cnt] = b
    return out


def describe_geometry(frames, name):
    top, bot = detect_active_rows(frames)
    h = frames.shape[1]
    return (f'{name}: активные строки {top}..{bot} из {h} '
            f'({"есть чёрные полосы" if (top > 0 or bot < h) else "полос нет"})')
