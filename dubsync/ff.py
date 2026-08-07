"""Вызовы ffmpeg/ffprobe."""
import json
import shutil
import subprocess


def _require(tool):
    if shutil.which(tool) is None:
        raise RuntimeError(f'{tool} не найден в PATH. Установите ffmpeg.')


def run(args, quiet=True):
    _require(args[0])
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'{args[0]} завершился с ошибкой:\n{res.stderr[-4000:]}')
    return res.stdout


def probe(path):
    out = run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
               '-of', 'json', path])
    return json.loads(out)


def video_info(path):
    """Частота кадров, размер и длительность видеопотока."""
    info = probe(path)
    v = next(s for s in info['streams'] if s['codec_type'] == 'video')
    num, den = (int(x) for x in v['r_frame_rate'].split('/'))
    return {
        'rate': num / den,
        'width': int(v['width']),
        'height': int(v['height']),
        'duration': float(info['format']['duration']),
        'dovi': any(sd.get('side_data_type', '').startswith('DOVI')
                    for sd in v.get('side_data_list', [])),
    }


def audio_streams(path):
    """Список аудиодорожек: индекс внутри типа, язык, название, каналы."""
    info = probe(path)
    out = []
    n = 0
    for s in info['streams']:
        if s['codec_type'] != 'audio':
            continue
        tags = s.get('tags', {})
        out.append({'a_index': n, 'index': s['index'], 'codec': s.get('codec_name'),
                    'lang': tags.get('language'), 'title': tags.get('title'),
                    'channels': s.get('channels')})
        n += 1
    return out


def extract_frames(path, out_path, width=48, height=27, hwaccel=True, fps=None):
    """Кадры в сыром виде: grayscale width x height, по кадру на каждый кадр видео."""
    vf = f'scale={width}:{height},format=gray'
    if fps:
        vf = f'fps={fps},' + vf
    args = ['ffmpeg', '-v', 'error', '-y']
    if hwaccel:
        args += ['-hwaccel', 'videotoolbox']
    args += ['-i', path, '-an', '-sn', '-vf', vf,
             '-f', 'rawvideo', '-pix_fmt', 'gray', out_path]
    run(args)


def extract_audio(path, out_path, a_index=0, rate=8000, channels=1, pan=None):
    args = ['ffmpeg', '-v', 'error', '-y', '-i', path, '-vn', '-sn',
            '-map', f'0:a:{a_index}']
    if pan:
        args += ['-af', pan]
    args += ['-ac', str(channels), '-ar', str(rate), '-c:a', 'pcm_s16le', out_path]
    run(args)


def encode(wav_path, out_path, codec='ac3', bitrate='448k'):
    run(['ffmpeg', '-v', 'error', '-y', '-i', wav_path,
         '-c:a', codec, '-b:a', bitrate, out_path])


def mux(video_src, new_audio, out_path, title, keep_audio=True, keep_subs=True):
    """Сборка итогового MKV: новая дорожка первой и по умолчанию.

    max_interleave_delta 0 обязателен: иначе ffmpeg раскладывает добавленную
    дорожку с отрывом в гигабайты от видео того же момента, и плеер захлёбывается.
    """
    args = ['ffmpeg', '-v', 'error', '-stats', '-y',
            '-i', video_src, '-i', new_audio,
            '-map', '0:v:0', '-map', '1:a:0']
    if keep_audio:
        args += ['-map', '0:a?']
    if keep_subs:
        args += ['-map', '0:s?']
    args += ['-map_chapters', '0', '-c', 'copy',
             '-max_interleave_delta', '0',
             '-metadata:s:a:0', 'language=rus',
             '-metadata:s:a:0', f'title={title}',
             '-disposition:a', '0', '-disposition:a:0', 'default',
             out_path]
    _require('ffmpeg')
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'ffmpeg (mux) завершился с ошибкой:\n{res.stderr[-4000:]}')
