"""Проверка чередования дорожек: насколько далеко разнесены данные одного момента."""
import subprocess
import sys
from collections import defaultdict

path = sys.argv[1]
times = [float(x) for x in sys.argv[2:]] or [10, 258, 688, 1000, 2500, 3847, 5000, 6500]

print(f'{"момент":>8} {"разброс позиций":>18}  дорожки')
worst = 0.0
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
        print(f'{t:8.0f}  нет данных')
        continue
    lo = min(min(v) for v in d.values())
    hi = max(max(v) for v in d.values())
    spread = (hi - lo) / 1e6
    worst = max(worst, spread)
    flag = '' if spread < 8 else '   <- ПЛОХО'
    print(f'{t:8.0f} {spread:15.2f} МБ  {sorted(d)}{flag}')
print(f'\nхудший разброс: {worst:.2f} МБ  '
      f'({"норма" if worst < 8 else "чередование нарушено"})')
