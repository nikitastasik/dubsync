"""Перекладка звука на временную шкалу целевого релиза.

Смещение меняется плавно, поэтому звук не режется кусками, а читается по
непрерывной карте позиций с интерполяцией Catmull-Rom. Дрейф компенсируется
незаметным изменением скорости, стыки на месте вырезанной рекламы приходятся
на монтажные склейки и не щёлкают.
"""
import wave

import numpy as np
from scipy.io import wavfile


def resample(src_wav, out_wav, map_t, map_d, cuts, duration, block_s=20.0,
             progress=None):
    sr, src = wavfile.read(src_wav, mmap=True)
    if src.ndim == 1:
        src = src[:, None]
    ns, nch = src.shape

    edges_all = [-1e9] + list(cuts) + [1e9]
    pieces = []
    for k in range(len(edges_all) - 1):
        m = (map_t >= edges_all[k]) & (map_t < edges_all[k + 1])
        pieces.append((edges_all[k], edges_all[k + 1],
                       map_t[m] if m.sum() >= 2 else None,
                       map_d[m] if m.sum() >= 2 else None))

    n_out = int(round(duration * sr))
    w = wave.open(out_wav, 'wb')
    w.setnchannels(nch)
    w.setsampwidth(2)
    w.setframerate(sr)
    blk = int(block_s * sr)
    for start in range(0, n_out, blk):
        cnt = min(blk, n_out - start)
        t = (start + np.arange(cnt)) / sr
        off = np.zeros(cnt)
        for a, b, tk, dk in pieces:
            m = (t >= a) & (t < b)
            if m.any() and tk is not None:
                off[m] = np.interp(t[m], tk, dk)
        pos = (t + off) * sr
        i0 = np.floor(pos).astype(np.int64)
        f = (pos - i0).astype(np.float32)[:, None]
        idx = np.clip(np.stack([i0 - 1, i0, i0 + 1, i0 + 2]), 0, ns - 1)
        p = src[idx].astype(np.float32)
        p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
        y = 0.5 * ((2 * p1) + (-p0 + p2) * f +
                   (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 +
                   (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
        y[(pos < 0) | (pos >= ns - 2)] = 0
        w.writeframes(np.clip(y, -32768, 32767).astype('<i2').tobytes())
        if progress:
            progress(start / sr, duration)
    w.close()


def splice_check(wav_path, cuts, half_s=0.4):
    """Скачок сигнала на стыках в сравнении с обычными перепадами рядом."""
    sr, a = wavfile.read(wav_path, mmap=True)
    out = []
    for c in cuts:
        i = int(c * sr)
        seg = a[max(0, i - int(half_s * sr)):i + int(half_s * sr)].astype(np.float32)
        jump = float(np.abs(a[i].astype(np.float32) - a[i - 1].astype(np.float32)).max())
        nearby = float(np.abs(np.diff(seg, axis=0)).max()) if len(seg) > 1 else 0.0
        out.append({'cut': float(c), 'jump': jump, 'typical': nearby,
                    'ok': jump <= max(1.0, nearby)})
    return out
