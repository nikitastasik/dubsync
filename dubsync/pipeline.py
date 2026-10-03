"""Полный проход: от двух файлов до готового MKV.

Все промежуточные данные складываются в рабочую папку и переиспользуются при
повторном запуске — самая долгая часть (декодирование видео) не повторяется.
"""
import json
import os
import time

import numpy as np

from . import align, audio, ff, render, verify, video


def _say(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _cached(path, produce, what):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        _say(f'{what}: уже готово')
        return path
    _say(f'{what}…')
    produce()
    return path


def run(target, source, out_path, work, source_audio=0, title=None,
        codec='ac3', bitrate='448k', hwaccel=True, offset_range=(-10.0, 100.0),
        keep_pitch=False):
    """target — файл с нужной картинкой; source — файл с нужной озвучкой."""
    os.makedirs(work, exist_ok=True)
    report = {'target': target, 'source': source, 'output': out_path}

    tinfo = ff.video_info(target)
    sinfo = ff.video_info(source)
    report['target_info'] = tinfo
    report['source_info'] = sinfo
    _say(f'цель:    {tinfo["width"]}x{tinfo["height"]} {tinfo["rate"]:.3f} fps, '
         f'{tinfo["duration"]:.1f} с{"  (Dolby Vision)" if tinfo["dovi"] else ""}')
    _say(f'источник: {sinfo["width"]}x{sinfo["height"]} {sinfo["rate"]:.3f} fps, '
         f'{sinfo["duration"]:.1f} с')
    if abs(tinfo['rate'] - sinfo['rate']) > 0.01:
        _say('ВНИМАНИЕ: разные частоты кадров — карта всё равно построится, '
             'но проверьте результат внимательнее')

    f_t = os.path.join(work, 'frames_target.raw')
    f_s = os.path.join(work, 'frames_source.raw')
    _cached(f_t, lambda: ff.extract_frames(target, f_t, hwaccel=hwaccel),
            'кадры цели')
    _cached(f_s, lambda: ff.extract_frames(source, f_s, hwaccel=hwaccel),
            'кадры источника')

    fr_t = video.open_raw(f_t)
    fr_s = video.open_raw(f_s)
    _say(video.describe_geometry(fr_t, 'цель'))
    _say(video.describe_geometry(fr_s, 'источник'))
    _say('признаки видео…')
    F4 = video.features(fr_t)
    FH = video.features(fr_s)
    rate = tinfo['rate']

    _say('грубая карта…')
    ct, co, cq = align.coarse_map(F4, FH, rate,
                                  dmin=offset_range[0], dmax=offset_range[1])
    inserts = align.find_inserts(ct, co)
    _say(f'качество сопоставления: медиана {np.median(cq):.3f}; '
         f'найдено вставок: {len(inserts)}')

    _say('уточнение границ вставок…')
    edges = []
    for t_guess, ob, oa in inserts:
        e = align.refine_edge(F4, FH, rate, t_guess, ob, oa)
        # ложные вставки: крутой дрейф даёт скачок грубой карты, но при
        # уточнении оба края ложатся плохо или «реклама» выходит в секунду
        real = min(e['q_left'], e['q_right']) >= 0.6 and e['length'] >= 3.0
        _say(f'  склейка {e["cut"]:.3f} с: реклама в источнике '
             f'[{e["ad_start"]:.2f} .. {e["ad_end"]:.2f}] '
             f'({e["length"]:.2f} с), качество {e["q_left"]:.2f}/{e["q_right"]:.2f}'
             f'{"" if real else "  -> не вставка, отброшено"}')
        if real:
            edges.append(e)
    report['inserts'] = edges

    cuts = [e['cut'] for e in edges]

    def predict(t):
        return float(np.interp(t, ct, co))

    _say('точная карта…')
    # узлы каждую секунду: скорость источника может гулять на проценты
    # за десятки секунд, редкие узлы срезают такие повороты
    nodes = align.fine_map(F4, FH, rate, predict, cuts, node=1.0, win=4.0, search=2.0)
    np.save(os.path.join(work, 'nodes.npy'), nodes)
    ok = ~np.isnan(nodes[:, 1])
    _say(f'узлов {len(nodes)}, измерено {ok.sum()}, '
         f'качество медиана {np.median(nodes[ok, 2]):.3f}')

    a_t = os.path.join(work, 'audio_target_%d.wav')
    tracks = ff.audio_streams(target)[:2]
    a4 = []
    for k, _ in enumerate(tracks):
        p = a_t % k
        _cached(p, lambda p=p, k=k: ff.extract_audio(target, p, a_index=k),
                f'звук цели #{k}')
        a4.append(audio.features(p))
    a_s8 = os.path.join(work, 'audio_source_8k.wav')
    _cached(a_s8, lambda: ff.extract_audio(source, a_s8, a_index=source_audio),
            'звук источника (анализ)')
    ahd = audio.features(a_s8)

    map_t, map_d, cut_arr, noise = align.smooth_map(
        nodes, edges, tinfo['duration'], window=8.0)
    # сдвиг звук/картинка меряем относительно уже гладкой карты
    smooth_nodes = np.c_[map_t, map_d, np.ones_like(map_t), np.ones_like(map_t)]
    bias, nb = align.audio_bias(a4, ahd, smooth_nodes, cuts, audio.FPS)
    _say(f'сдвиг звука относительно картинки: {bias*1000:+.0f} мс (по {nb} замерам)')
    report['audio_bias_ms'] = bias * 1000
    map_d = map_d + bias
    _say(f'карта готова: смещение {map_d.min():+.2f}..{map_d.max():+.2f} с, '
         f'шум узлов {noise*1000:.0f} мс')
    report['offset_min'] = float(map_d.min())
    report['offset_max'] = float(map_d.max())
    report['node_noise_ms'] = noise * 1000
    np.save(os.path.join(work, 'map_t.npy'), map_t)
    np.save(os.path.join(work, 'map_d.npy'), map_d)

    a_full = os.path.join(work, 'source_full.wav')
    _cached(a_full,
            lambda: ff.extract_audio(source, a_full, a_index=source_audio,
                                     rate=48000, channels=2),
            'звук источника (полный)')

    synced = os.path.join(work, 'synced.wav')

    def do_render(d):
        if keep_pitch:
            render.stretch(a_full, synced, map_t, d, cut_arr, tinfo['duration'])
        else:
            render.resample(a_full, synced, map_t, d, cut_arr, tinfo['duration'])

    _say('перекладка звука…' + (' с сохранением тона (WSOLA)' if keep_pitch else ''))
    do_render(map_d)

    # замкнутый контур: поправку звук/картинка мерили по исходнику, а
    # теперь меряем то, что вышло. Если готовый звук всё ещё заметно
    # опережает или отстаёт, сдвигаем карту на остаток и рендерим ещё раз.
    chk8 = os.path.join(work, 'synced_8k.wav')
    ff.run(['ffmpeg', '-v', 'error', '-y', '-i', synced, '-ac', '1', '-ar',
            str(audio.SR), '-c:a', 'pcm_s16le', chk8])
    pre = verify.residual_sync(a4, audio.features(chk8), audio.FPS, cuts)
    _say(f'  рассинхрон после рендера: медиана {pre["median_ms"]:+.0f} мс '
         f'по {pre["n_good"]} замерам')
    if abs(pre['median_ms']) > 15 and pre['n_good'] >= 30:
        fix = pre['median_ms'] / 1000
        map_d = map_d + fix
        report['audio_bias_ms'] += fix * 1000
        np.save(os.path.join(work, 'map_d.npy'), map_d)
        _say(f'  поправка {fix*1000:+.0f} мс, рендер заново…')
        do_render(map_d)
    sp = render.splice_check(synced, cut_arr)
    for s in sp:
        _say(f'  стык {s["cut"]:.2f} с: скачок {s["jump"]:.0f} '
             f'(обычный перепад рядом {s["typical"]:.0f}) '
             f'{"— гладко" if s["ok"] else "— ВОЗМОЖЕН ЩЕЛЧОК"}')
    report['splices'] = sp

    enc = os.path.join(work, f'synced.{codec}')
    _say(f'кодирование {codec} {bitrate}…')
    ff.encode(synced, enc, codec=codec, bitrate=bitrate)

    _say('сборка MKV…')
    ff.mux(target, enc, out_path,
           title or 'Русская (синхронизирована)')

    _say('проверка результата…')
    a_chk = os.path.join(work, 'result_8k.wav')
    ff.extract_audio(out_path, a_chk, a_index=0)
    res = verify.residual_sync(a4, audio.features(a_chk), audio.FPS, cuts)
    _say(f'остаточный рассинхрон: медиана {res["median_ms"]:+.0f} мс, '
         f'в пределах 40 мс — {res["within_40ms"]*100:.0f}% замеров')
    dur = tinfo['duration']
    probes = [t for t in (10, 258, 1000, 2500, 5000, dur - 200) if 0 < t < dur - 5]
    il = verify.interleaving(out_path, probes)
    _say(f'чередование дорожек: худший разброс {il["worst_mb"]:.2f} МБ '
         f'({"норма" if il["ok"] else "НАРУШЕНО"})')
    report['sync'] = res
    report['interleaving'] = il

    with open(os.path.join(work, 'report.json'), 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _say(f'готово: {out_path}')
    return report
