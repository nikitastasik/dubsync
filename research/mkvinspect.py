"""Минимальный разбор MKV: где лежит индекс Cues, сколько в нём точек и на какие дорожки."""
import sys

IDS = {
    0x1A45DFA3: 'EBML', 0x18538067: 'Segment', 0x114D9B74: 'SeekHead',
    0x1549A966: 'Info', 0x1654AE6B: 'Tracks', 0x1F43B675: 'Cluster',
    0x1C53BB6B: 'Cues', 0x1254C367: 'Tags', 0x1043A770: 'Chapters',
    0xBB: 'CuePoint', 0xB3: 'CueTime', 0xB7: 'CueTrackPositions', 0xF7: 'CueTrack',
    0xF1: 'CueClusterPosition', 0xA3: 'SimpleBlock', 0xA0: 'BlockGroup', 0xA1: 'Block',
}


def read_vint(f, keep_marker=False):
    b = f.read(1)
    if not b:
        return None, 0
    v = b[0]
    if v == 0:
        return None, 1
    n = 1
    mask = 0x80
    while not (v & mask):
        mask >>= 1
        n += 1
    val = v if keep_marker else (v & (mask - 1))
    for _ in range(n - 1):
        val = (val << 8) | f.read(1)[0]
    return val, n


def parse(path, descend_into, depth=0, end=None, out=None):
    f = path
    while True:
        if end is not None and f.tell() >= end:
            return
        pos = f.tell()
        eid, n1 = read_vint(f, keep_marker=True)
        if eid is None:
            return
        size, n2 = read_vint(f)
        if size is None:
            return
        name = IDS.get(eid, hex(eid))
        body = f.tell()
        if depth == 0 or name in ('Cues', 'CuePoint', 'CueTrackPositions'):
            out.append((depth, name, pos, size))
        if name in descend_into:
            parse(f, descend_into, depth + 1, body + size, out)
            f.seek(body + size)
        else:
            f.seek(body + size)
        if name == 'Segment':
            # внутрь сегмента заходим на том же уровне
            return


for path in sys.argv[1:]:
    print(f'=== {path.split("/")[-1]}')
    with open(path, 'rb') as f:
        out = []
        # верхний уровень
        while True:
            pos = f.tell()
            eid, _ = read_vint(f, keep_marker=True)
            if eid is None:
                break
            size, _ = read_vint(f)
            if size is None:
                break
            name = IDS.get(eid, hex(eid))
            body = f.tell()
            if name == 'Segment':
                seg_end = body + size
                cues_info = None
                clusters = 0
                while f.tell() < seg_end:
                    p2 = f.tell()
                    e2, _ = read_vint(f, keep_marker=True)
                    if e2 is None:
                        break
                    s2, _ = read_vint(f)
                    if s2 is None:
                        break
                    n2 = IDS.get(e2, hex(e2))
                    b2 = f.tell()
                    if n2 == 'Cluster':
                        clusters += 1
                        if clusters > 3 and cues_info is None:
                            # пропускаем кластеры быстро
                            pass
                    if n2 == 'Cues':
                        data_end = b2 + s2
                        pts, tracks = 0, {}
                        while f.tell() < data_end:
                            e3, _ = read_vint(f, keep_marker=True)
                            s3, _ = read_vint(f)
                            if e3 is None or s3 is None:
                                break
                            b3 = f.tell()
                            if IDS.get(e3) == 'CuePoint':
                                pts += 1
                                sub_end = b3 + s3
                                while f.tell() < sub_end:
                                    e4, _ = read_vint(f, keep_marker=True)
                                    s4, _ = read_vint(f)
                                    if e4 is None or s4 is None:
                                        break
                                    b4 = f.tell()
                                    if IDS.get(e4) == 'CueTrackPositions':
                                        se = b4 + s4
                                        while f.tell() < se:
                                            e5, _ = read_vint(f, keep_marker=True)
                                            s5, _ = read_vint(f)
                                            if e5 is None or s5 is None:
                                                break
                                            b5 = f.tell()
                                            if IDS.get(e5) == 'CueTrack':
                                                tv = int.from_bytes(f.read(s5), 'big')
                                                tracks[tv] = tracks.get(tv, 0) + 1
                                            f.seek(b5 + s5)
                                    f.seek(b4 + s4)
                            f.seek(b3 + s3)
                        cues_info = (p2, s2, pts, tracks)
                        break
                    f.seek(b2 + s2)
                sz = __import__('os').path.getsize(path)
                if cues_info:
                    p, s, pts, tracks = cues_info
                    where = 'в начале' if p < sz * 0.5 else 'в конце'
                    print(f'  Cues: позиция {p:,} ({where} файла), размер {s:,} байт')
                    print(f'        точек {pts}, по дорожкам {tracks}')
                else:
                    print('  Cues: НЕ НАЙДЕН среди верхних элементов сегмента')
                break
            f.seek(body + size)
