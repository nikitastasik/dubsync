"""Показать кадр из raw-потока как ASCII — проверка временной привязки."""
import sys
import numpy as np

W, H, VFPS = 48, 27, 4.0
CH = ' .:-=+*#%@'


def show(path, t):
    a = np.fromfile(path, dtype=np.uint8)
    n = len(a) // (W * H)
    idx = int(round(t * VFPS))
    fr = a[:n * W * H].reshape(n, H, W)[idx]
    print(f'{path} t={t}s (кадр {idx} из {n}, среднее {fr.mean():.0f})')
    for row in fr[::2]:
        print('  ' + ''.join(CH[min(9, int(v) * 10 // 256)] for v in row))


for arg in sys.argv[1:]:
    path, t = arg.split('@')
    show(path, float(t))
    print()
