"""Построение соответствия времени между двумя релизами.

Порядок такой:
  1. грубая карта по прореженным кадрам — находит рекламные вставки и общий ход;
  2. точные точки склейки — с точностью до кадра;
  3. точная карта в узлах — на полном фреймрейте, в узком коридоре;
  4. сглаживание — гасит шум измерений, сохраняя реальный дрейф.

Смещение между релизами не обязано быть постоянным: встречается плавный
дрейф на секунды, поэтому карта строится как функция времени, а не число.
"""
import numpy as np

from . import corr


def coarse_map(f4, fh, rate, decim=6, dmin=-10.0, dmax=100.0,
               win=20.0, step=5.0, penalty=1.2):
    """Грубая карта смещений. Возвращает (моменты 4K, смещения) в секундах.

    Путь ищется динамическим программированием: скачок стоит фиксированно,
    поэтому карта получается кусочно-постоянной, а не пляшущей за шумом.
    """
    a = np.ascontiguousarray(f4[:, ::decim])
    b = np.ascontiguousarray(fh[:, ::decim])
    r = rate / decim
    lo_d, hi_d = int(dmin * r), int(dmax * r)
    D = hi_d - lo_d + 1
    n = int(win * r)
    stp = max(1, int(step * r))
    T4, THD = a.shape[1], b.shape[1]

    starts = list(range(0, max(1, T4 - n), stp))
    S = np.full((len(starts), D), -1.0, dtype=np.float32)
    for w, i in enumerate(starts):
        clo, chi = max(0, i + lo_d), min(THD, i + hi_d + n)
        if chi - clo < n:
            continue
        sc = corr.ncc(a[:, i:i + n], b[:, clo:chi])
        if sc is None:
            continue
        j0 = clo - (i + lo_d)
        S[w, j0:j0 + len(sc)] = sc

    best = S[0].astype(np.float64).copy()
    back = np.zeros(S.shape, dtype=np.int32)
    ar = np.arange(D)
    for w in range(1, S.shape[0]):
        g = int(np.argmax(best))
        jump = best[g] - penalty
        take = jump > best
        best = np.where(take, jump, best) + S[w]
        back[w] = np.where(take, g, ar)
    path = np.zeros(S.shape[0], dtype=np.int32)
    path[-1] = int(np.argmax(best))
    for w in range(S.shape[0] - 1, 0, -1):
        path[w - 1] = back[w][path[w]]

    quality = np.array([S[w, path[w]] for w in range(len(starts))])
    return (np.array(starts) / r, (path + lo_d) / r, quality)


def find_inserts(times, offsets, min_jump=5.0):
    """Места, где смещение скачком выросло — кандидаты в рекламные вставки."""
    out = []
    for w in range(1, len(offsets)):
        d = offsets[w] - offsets[w - 1]
        if d > min_jump:
            out.append((float(times[w]), float(offsets[w - 1]), float(offsets[w])))
    return out


def refine_edge(f4, fh, rate, t_guess, pred_left, pred_right,
                side=6.0, span=2.0, search=6.0):
    """Точка склейки и смещения по обе стороны от неё.

    Для каждой кандидатной точки подбираем лучшее смещение отдельно слева и
    справа; верная точка — там, где оба куска ложатся одинаково хорошо.
    """
    T4, THD = f4.shape[1], fh.shape[1]
    n = int(side * rate)

    def best_off(i4, center):
        lo = max(0, i4 + int((center - span) * rate))
        hi = min(THD, i4 + int((center + span) * rate) + n)
        if hi - lo < n or i4 < 0 or i4 + n > T4:
            return -1.0, np.nan
        sc = corr.ncc(f4[:, i4:i4 + n], fh[:, lo:hi])
        if sc is None:
            return -1.0, np.nan
        b = int(np.argmax(sc))
        return float(sc[b]), (lo + b - i4) / rate

    rows = []
    for t in np.arange(t_guess - search, t_guess + search, 1.0 / rate):
        i = int(round(t * rate))
        sl, ol = best_off(i - n, pred_left)
        sr, orr = best_off(i, pred_right)
        rows.append((t, sl, ol, sr, orr))
    r = np.array(rows)
    k = int(np.argmax(r[:, 1] + r[:, 3]))
    t, sl, ol, sr, orr = r[k]
    return {'cut': float(t), 'off_left': float(ol), 'off_right': float(orr),
            'q_left': float(sl), 'q_right': float(sr),
            'ad_start': float(t + ol), 'ad_end': float(t + orr),
            'length': float(orr - ol)}


def fine_map(f4, fh, rate, predict, cuts, node=5.0, win=12.0, search=1.6):
    """Точная карта: для каждого узла — смещение с субкадровой точностью."""
    T4, THD = f4.shape[1], fh.shape[1]
    n = int(win * rate)
    rows = []
    for t in np.arange(0, T4 / rate - win, node):
        centre = t + win / 2
        if any(t - 1 < c < t + win + 1 for c in cuts):
            rows.append((centre, np.nan, 0.0, 0.0))
            continue
        p = predict(centre)
        i = int(t * rate)
        lo = max(0, i + int((p - search) * rate))
        hi = min(THD, i + int((p + search) * rate) + n)
        if hi - lo < n:
            rows.append((centre, np.nan, 0.0, 0.0))
            continue
        sc = corr.ncc(f4[:, i:i + n], fh[:, lo:hi])
        if sc is None:
            rows.append((centre, np.nan, 0.0, 0.0))
            continue
        b, pk, conf = corr.peak(sc, int(0.5 * rate))
        off = (lo + b + corr.parabolic(sc, b) - i) / rate
        rows.append((centre, off, pk, conf))
    return np.array(rows)


def smooth_map(nodes, edges, duration, q_min=0.45, conf_min=0.10,
               outlier=0.30, window=25.0, bias=0.0):
    """Гладкая функция смещения с разрывами на склейках.

    Возвращает (моменты, смещения, точки разрыва) — готово для рендера.
    """
    ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > q_min) & (nodes[:, 3] > conf_min)
    t, d = nodes[ok, 0], nodes[ok, 1]
    if len(t) < 3:
        raise RuntimeError('слишком мало надёжных узлов — проверьте исходники')
    med = np.array([np.median(d[np.abs(t - x) < 30]) for x in t])
    keep = np.abs(d - med) < outlier
    t, d = t[keep], d[keep]

    cuts = [e['cut'] for e in edges]
    bounds = [-1e9] + cuts + [1e9]
    T_OUT, D_OUT, resid = [], [], []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        m = (t >= a) & (t < b)
        if m.sum() < 3:
            continue
        tk, dk = t[m], d[m]
        sm = np.empty_like(dk)
        for i, x in enumerate(tk):
            w = np.abs(tk - x) < window
            sm[i] = np.polyfit(tk[w] - x, dk[w], 1)[1] if w.sum() >= 3 else dk[i]
        resid.append(dk - sm)
        lo_t = max(a, 0.0) if k == 0 else cuts[k - 1] + 1e-6
        hi_t = min(b, duration) if k == len(bounds) - 2 else cuts[k]
        lo_d = edges[k - 1]['off_right'] if k > 0 else sm[0]
        hi_d = edges[k]['off_left'] if k < len(cuts) else sm[-1]
        T_OUT.append(np.r_[lo_t, tk, hi_t])
        D_OUT.append(np.r_[lo_d, sm, hi_d])

    noise = float(np.std(np.concatenate(resid))) if resid else 0.0
    return (np.concatenate(T_OUT), np.concatenate(D_OUT) + bias,
            np.array(cuts), noise)


def audio_bias(a4_list, ahd, nodes, cuts, fps, win=30.0, step=20.0, search=0.8):
    """Сдвиг звука относительно картинки (если в релизе звук смещён)."""
    ok = ~np.isnan(nodes[:, 1]) & (nodes[:, 2] > 0.45) & (nodes[:, 3] > 0.10)
    tt, dd = nodes[ok, 0], nodes[ok, 1]
    THD = ahd.shape[1]
    T4 = a4_list[0].shape[1]
    n = int(win * fps)
    res = []
    for t in np.arange(20, T4 / fps - win, step):
        if any(t - 2 < c < t + win + 2 for c in cuts):
            continue
        p = float(np.interp(t + win / 2, tt, dd))
        i = int(t * fps)
        lo = max(0, i + int((p - search) * fps))
        hi = min(THD, i + int((p + search) * fps) + n)
        if hi - lo < n:
            continue
        tot = None
        for f in a4_list:
            s = corr.ncc(f[:, i:i + n], ahd[:, lo:hi])
            tot = s if tot is None else tot + s
        sc = tot / len(a4_list)
        b, pk, conf = corr.peak(sc, int(0.4 * fps))
        if pk > 0.30 and conf > 0.08:
            res.append((lo + b + corr.parabolic(sc, b) - i) / fps - p)
    if not res:
        return 0.0, 0
    r = np.array(res)
    return float(np.median(r)), len(r)
