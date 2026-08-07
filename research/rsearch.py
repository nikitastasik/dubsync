"""Обратный поиск: где окно из 1080p находится в 4K (и наоборот)."""
import sys
import numpy as np
from vmatch2 import load4k, loadhd, feats, VFPS

f4 = feats(load4k())
fh = feats(loadhd())


def best(win, target, k=5, gap=8.0):
    n = win.shape[1]
    w = win - win.mean(axis=1, keepdims=True)
    wn = np.linalg.norm(w)
    m = target.shape[1] - n + 1
    sc = np.empty(m)
    for j in range(m):
        r = target[:, j:j + n]
        r = r - r.mean(axis=1, keepdims=True)
        sc[j] = (w * r).sum() / (wn * np.linalg.norm(r) + 1e-9)
    out, shown = [], []
    for j in np.argsort(sc)[::-1]:
        if any(abs(j - s) < gap * VFPS for s in shown):
            continue
        shown.append(j)
        out.append((j / VFPS, sc[j]))
        if len(out) >= k:
            break
    return out


mode, t, win_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
i, n = int(t * VFPS), int(win_s * VFPS)
if mode == 'hd2k':
    print(f'окно 1080p [{t}..{t+win_s}] — где это в 4K:')
    for tt, s in best(fh[:, i:i + n], f4):
        print(f'   4K {tt:9.2f}s   сходство {s:.3f}   (смещение 4K->HD {t-tt:+7.2f}s)')
else:
    print(f'окно 4K [{t}..{t+win_s}] — где это в 1080p:')
    for tt, s in best(f4[:, i:i + n], fh):
        print(f'   HD {tt:9.2f}s   сходство {s:.3f}   (смещение {tt-t:+7.2f}s)')
