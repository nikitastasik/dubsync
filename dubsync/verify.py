"""Проверки готового результата."""
import subprocess
from collections import defaultdict

import numpy as np

from . import corr


def residual_sync(a4_list, achk, fps, cuts, win=30.0, step=30.0, search=1.0):
    """Остаточный рассинхрон нового звука относительно опорного релиза."""
    T = min(achk.shape[1], a4_list[0].shape[1])
    n = int(win * fps)
    res = []
    for t in np.arange(10, T / fps - win, step):
        i = int(t * fps)
        lo, hi = max(0, i - int(search * fps)), min(T, i + int(search * fps) + n)
        if hi - lo < n:
            continue
        tot = None
        for f in a4_list:
            s = corr.ncc(f[:, i:i + n], achk[:, lo:hi])
            tot = s if tot is None else tot + s
        sc = tot / len(a4_list)
        b, pk, conf = corr.peak(sc, int(0.4 * fps))
        off = (lo + b + corr.parabolic(sc, b) - i) / fps
        res.append((t, off, pk, conf))
    a = np.array(res)
    good = (a[:, 2] > 0.30) & (a[:, 3] > 0.08)
    d = a[good, 1] if good.any() else a[:, 1]
    return {
        'n': int(len(a)), 'n_good': int(good.sum()),
        'median_ms': float(np.median(d) * 1000),
        'std_ms': float(np.std(d) * 1000),
        'within_40ms': float((np.abs(d) < 0.040).mean()),
        'within_80ms': float((np.abs(d) < 0.080).mean()),
        'max_ms': float(np.abs(d).max() * 1000),
    }


def interleaving(path, times):
    """Насколько далеко разнесены данные одного момента по разным дорожкам.

    Если добавленная дорожка лежит в другом конце файла, плеер захлёбывается:
    картинка замирает или сыплется, хотя звук идёт.
    """
    worst = 0.0
    rows = []
    for t in times:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'packet=stream_index,pos',
             '-read_intervals', f'{t}%+1', '-of', 'csv=p=0', path],
            capture_output=True, text=True)
        d = defaultdict(list)
        for line in r.stdout.splitlines():
            p = line.strip().split(',')
            if len(p) < 2 or p[1] == 'N/A':
                continue
            d[int(p[0])].append(int(p[1]))
        if not d:
            continue
        lo = min(min(v) for v in d.values())
        hi = max(max(v) for v in d.values())
        spread = (hi - lo) / 1e6
        worst = max(worst, spread)
        rows.append({'t': float(t), 'spread_mb': spread, 'tracks': sorted(d)})
    return {'worst_mb': worst, 'ok': worst < 8.0, 'points': rows}
