"""DP по сохранённым профилям корреляции при разных штрафах за скачок."""
import sys
import numpy as np
import afeat as A

FPS = A.FPS
DMIN = int(-5 * FPS)
S = np.load('S.npy')
starts = np.load('starts.npy')
D = S.shape[1]
EPS = 0.00004


def run(pen, eps=EPS):
    best = S[0].astype(np.float64).copy()
    back = np.zeros(S.shape, dtype=np.int32)
    ramp = np.arange(D) * eps
    ar = np.arange(D)
    for w in range(1, S.shape[0]):
        shifted = best + ramp
        pm = np.maximum.accumulate(shifted)
        argp = np.maximum.accumulate(np.where(shifted >= pm, ar, 0)).astype(np.int32)
        jump = pm - pen - ramp
        take = jump > best
        best = np.where(take, jump, best) + S[w]
        back[w] = np.where(take, argp, ar)
    path = np.zeros(S.shape[0], dtype=np.int32)
    path[-1] = int(np.argmax(best))
    for w in range(S.shape[0] - 1, 0, -1):
        path[w - 1] = back[w][path[w]]
    return (path + DMIN) / FPS


for pen in (float(x) for x in sys.argv[1:] or [1.5, 2.5, 4.0, 6.0, 9.0]):
    offs = run(pen)
    steps = []
    prev = offs[0]
    for w in range(1, len(offs)):
        if abs(offs[w] - prev) > 0.02:
            steps.append((starts[w] / FPS, prev, offs[w]))
            prev = offs[w]
    total = offs[-1] - offs[0]
    print(f'\n### PEN={pen}: ступеней {len(steps)}, старт {offs[0]:.2f}s, конец {offs[-1]:.2f}s, сумма {total:.2f}s')
    for t, a, b in steps:
        print(f'   {t/60:6.2f} мин  {a:7.2f} -> {b:7.2f}   {b-a:+7.2f}s')
