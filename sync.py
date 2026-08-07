#!/usr/bin/env python3
"""Перенос звуковой дорожки одного релиза фильма на картинку другого.

Пример:
    python sync.py --target 4k.mkv --source 1080p.mkv --out result.mkv
"""
import argparse
import os
import sys

from dubsync import ff, pipeline


def main():
    p = argparse.ArgumentParser(
        description='Наложить звук из одного релиза на видео другого, '
                    'вырезав рекламные вставки и убрав дрейф.')
    p.add_argument('--target', required=True,
                   help='файл с нужной картинкой (например, 4K)')
    p.add_argument('--source', required=True,
                   help='файл с нужной озвучкой (например, 1080p)')
    p.add_argument('--out', help='куда сохранить результат '
                                 '(не нужен при --list-audio)')
    p.add_argument('--work', default=None,
                   help='папка для промежуточных файлов (по умолчанию рядом с --out)')
    p.add_argument('--source-audio', type=int, default=0,
                   help='номер аудиодорожки в источнике (по умолчанию 0)')
    p.add_argument('--title', default=None, help='название новой дорожки')
    p.add_argument('--codec', default='ac3', help='кодек новой дорожки (ac3, aac, flac)')
    p.add_argument('--bitrate', default='448k', help='битрейт новой дорожки')
    p.add_argument('--no-hwaccel', action='store_true',
                   help='отключить аппаратное декодирование при разборе кадров')
    p.add_argument('--list-audio', action='store_true',
                   help='только показать аудиодорожки обоих файлов и выйти')
    a = p.parse_args()

    for path in (a.target, a.source):
        if not os.path.exists(path):
            sys.exit(f'нет файла: {path}')

    if a.list_audio:
        for name, path in (('ЦЕЛЬ', a.target), ('ИСТОЧНИК', a.source)):
            print(f'=== {name}: {os.path.basename(path)}')
            for s in ff.audio_streams(path):
                print(f'  --source-audio {s["a_index"]}  {s["codec"]}, '
                      f'{s["channels"]} кан., язык {s["lang"]}, {s["title"] or ""}')
        return

    if not a.out:
        sys.exit('нужен --out (куда сохранить результат)')
    work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                  'dubsync_work')
    pipeline.run(a.target, a.source, a.out, work,
                 source_audio=a.source_audio, title=a.title,
                 codec=a.codec, bitrate=a.bitrate, hwaccel=not a.no_hwaccel)


if __name__ == '__main__':
    main()
