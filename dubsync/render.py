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


def _offsets(t, pieces):
    off = np.zeros(len(t))
    for a, b, tk, dk in pieces:
        m = (t >= a) & (t < b)
        if m.any() and tk is not None:
            off[m] = np.interp(t[m], tk, dk)
    return off


def _pieces(map_t, map_d, cuts):
    edges_all = [-1e9] + list(cuts) + [1e9]
    out = []
    for k in range(len(edges_all) - 1):
        m = (map_t >= edges_all[k]) & (map_t < edges_all[k + 1])
        ok = m.sum() >= 2
        out.append((edges_all[k], edges_all[k + 1],
                    map_t[m] if ok else None, map_d[m] if ok else None))
    return out


def stretch(src_wav, out_wav, map_t, map_d, cuts, duration,
            frame=2048, tol=512, progress=None):
    """Перекладка с сохранением тона (WSOLA).

    Нужна, когда источник сам был ускорен или замедлен без изменения тона:
    так часто делают в TS, подгоняя звук из зала под запись камеры. Обычный
    resample тогда вернёт темп, но сдвинет тон на те же проценты.

    Звук режется на кадры по frame отсчётов с перекрытием 50%. Положение
    каждого кадра в источнике задаёт карта, а в пределах ±tol отсчётов
    выбирается то, где форма волны лучше всего продолжает предыдущий кадр.
    """
    sr, src = wavfile.read(src_wav, mmap=True)
    if src.ndim == 1:
        src = src[:, None]
    ns, nch = src.shape
    hs = frame // 2
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)).astype(np.float32)
    pieces = _pieces(map_t, map_d, cuts)

    n_out = int(round(duration * sr))
    n_fr = n_out // hs + 2
    centres = (np.arange(n_fr) * hs + frame / 2) / sr
    ideal = np.round((centres + _offsets(centres, pieces)) * sr - frame / 2).astype(np.int64)

    def grab(p, n):
        out = np.zeros((n, nch), dtype=np.float32)
        a, b = max(p, 0), min(p + n, ns)
        if b > a:
            out[a - p:b - p] = src[a:b]
        return out

    size = 1 << int(np.ceil(np.log2(frame + 2 * tol + frame)))
    w = wave.open(out_wav, 'wb')
    w.setnchannels(nch)
    w.setsampwidth(2)
    w.setframerate(sr)
    blk = sr * 20
    buf = np.zeros((blk + 2 * frame, nch), dtype=np.float32)
    buf0 = 0
    written = 0
    prev = None
    for k in range(n_fr):
        o = k * hs - buf0
        if o >= blk:
            m = min(o, n_out - written)
            if m > 0:
                w.writeframes(np.clip(buf[:m], -32768, 32767).astype('<i2').tobytes())
                written += m
            buf[:-o] = buf[o:]
            buf[-o:] = 0
            buf0 += o
            o = 0
            if progress:
                progress(buf0 / sr, duration)
        p = int(ideal[k])
        if prev is not None and abs(p - (prev + hs)) < 4 * frame:
            nat = grab(prev + hs, frame).mean(axis=1)
            reg = grab(p - tol, frame + 2 * tol).mean(axis=1)
            num = np.fft.irfft(np.fft.rfft(reg, size) *
                               np.conj(np.fft.rfft(nat, size)), size)[:2 * tol + 1]
            e = np.concatenate(([0.0], np.cumsum(reg.astype(np.float64) ** 2)))
            en = e[frame:frame + 2 * tol + 1] - e[:2 * tol + 1]
            score = num / np.sqrt(en + 1e-9)
            p = p - tol + int(np.argmax(score))
        buf[o:o + frame] += grab(p, frame) * win[:, None]
        prev = p
    rest = n_out - written
    if rest > 0:
        w.writeframes(np.clip(buf[:rest], -32768, 32767).astype('<i2').tobytes())
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
