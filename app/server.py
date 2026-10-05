"""Local, dependency-free HTTP interface for the existing Pupu SVC runtime."""
import json
import math
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_paths import ENGINE, MODEL_ROOT, python_path, hub_cache

APP = Path(__file__).resolve().parent
ROOT = APP.parent
WORKSPACE = ROOT
PYTHON = python_path()
PORT = 23335
LOCK = threading.RLock()
FILES = {}
JOBS = {}
ACTIVE = None
PROCESS = None
EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aiff', '.aif'}
for folder in ('data', 'outputs', 'logs'):
    (ROOT / folder).mkdir(exist_ok=True)


def recent_jobs():
    """Read completed results from disk; running/failed jobs exist only in memory."""
    jobs = {}
    for folder in (ROOT / 'outputs').glob('*'):
        if not folder.is_dir() or folder.name == ACTIVE:
            continue
        files = sorted(folder.glob('*.flac'))
        if not files:
            continue
        output = files[0]
        try:
            modified = output.stat().st_mtime
        except OSError:
            continue
        match = re.match(r'^(.*)_pupu_([+-]?[\d.]+)key$', output.stem)
        jobs[folder.name] = dict(
            id=folder.name, name=match[1] if match else output.stem,
            pitch=float(match[2]) if match else 0, status='done', stage='done',
            started=modified, ended=modified, percent=100, indeterminate=False,
            message='转换完成，可以试听或下载。', output=str(output), logs=[])
    for job in JOBS.values():
        if job['status'] != 'done' or job['id'] in jobs:
            jobs[job['id']] = job
    return sorted(jobs.values(), key=lambda j: j['started'], reverse=True)[:30]


def find_job(job_id):
    return next((j for j in recent_jobs() if j['id'] == job_id), None)


def register(path, name=None):
    import soundfile as sf
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        raise ValueError('请选择有效的音频文件（WAV、FLAC、MP3、OGG、M4A）。')
    try:
        info = sf.info(str(path))
        duration = info.duration
        rate = info.samplerate
        if duration <= 0:
            raise ValueError('音频为空')
    except Exception:
        # Some compressed formats need librosa/audioread during actual conversion.
        if path.suffix.lower() not in {'.m4a'}:
            raise ValueError('无法读取音频，请确认文件完整，或先转为 WAV / FLAC。')
        duration, rate = None, None
    file_id = uuid.uuid4().hex
    meta = dict(id=file_id, name=name or path.name, path=str(path), size=path.stat().st_size,
                duration=duration, sample_rate=rate, url='/api/audio/' + file_id)
    with LOCK:
        FILES[file_id] = meta
    return meta


def public_job(job):
    result = dict(job)
    result['elapsed'] = round((job.get('ended') or time.time()) - job['started'], 1)
    result['logs'] = job.get('logs', [])[-100:]
    if job.get('output') and Path(job['output']).is_file():
        result['audio_url'] = '/api/result/' + job['id']
        result['download_url'] = result['audio_url'] + '?download=1'
    return result


def run_job(job, settings):
    global ACTIVE, PROCESS
    request_path = ROOT / 'data' / (job['id'] + '.json')
    try:
        request_path.write_text(json.dumps(settings, ensure_ascii=False), encoding='utf-8')
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ENGINE), PYTHONUTF8='1', PYTHONUNBUFFERED='1',
                   HF_HOME=str(hub_cache().parent),
                   HF_HUB_CACHE=str(hub_cache()),
                   HF_HUB_OFFLINE='1')
        with LOCK:
            if job['status'] == 'cancelled':
                return
            proc = subprocess.Popen([str(PYTHON), '-u', str(APP / 'runner.py'), str(request_path)],
                                    cwd=str(ENGINE), env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            PROCESS = proc
        output = None
        with (ROOT / 'logs' / (job['id'] + '.log')).open('w', encoding='utf-8') as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                with LOCK:
                    if line.startswith('@@PUPU@@'):
                        event = json.loads(line[len('@@PUPU@@'):])
                        if job['status'] != 'cancelled':
                            event['percent'] = max(job['percent'], event.get('percent', job['percent']))
                            job.update({k: v for k, v in event.items() if k != 'output'})
                        output = event.get('output', output)
                    elif line.strip():
                        job['logs'].append(line.strip())
                        job['logs'] = job['logs'][-100:]
        code = proc.wait()
        with LOCK:
            if job['status'] != 'cancelled':
                if code == 0 and output and Path(output).is_file():
                    job.update(status='done', stage='done', percent=100, indeterminate=False,
                               message='转换完成，可以试听或下载。', output=output)
                else:
                    job.update(status='error', indeterminate=False,
                               message='转换失败。展开运行详情查看原因，然后重试。')
            job['ended'] = time.time()
    except Exception as exc:
        with LOCK:
            if job['status'] != 'cancelled':
                job.update(status='error', message=str(exc), ended=time.time(), indeterminate=False)
    finally:
        with LOCK:
            PROCESS = None
            ACTIVE = None
        request_path.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Audio players cancel requests when seeking or switching tracks.
            self.close_connection = True

    def json_response(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def json_body(self):
        length = int(self.headers.get('Content-Length', '0'))
        if length > 65536:
            raise ValueError('请求过大')
        return json.loads(self.rfile.read(length) or b'{}')

    def send_file(self, path, download=False):
        path = Path(path)
        if not path.is_file():
            self.json_response({'error': '文件不存在'}, 404)
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get('Range')
        if header and header.startswith('bytes='):
            try:
                left, right = header[6:].split('-')
                if left:
                    start = int(left)
                    end = min(int(right), end) if right else end
                else:
                    start = max(0, size - int(right))
                if start > end or start < 0:
                    raise ValueError()
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
        self.send_response(status)
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_header('Content-Type', mime)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        if download:
            from urllib.parse import quote
            self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(path.name))
        self.end_headers()
        try:
            with path.open('rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining:
                    data = f.read(min(256 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def do_GET(self):
        route = urlparse(self.path)
        path = route.path
        if path == '/api/state':
            with LOCK:
                self.json_response(dict(active=ACTIVE, jobs=[public_job(j) for j in (recent_jobs() if parse_qs(route.query).get('history', ['1'])[0] == '1' else JOBS.values()) if j['status'] != 'done' or (j.get('output') and Path(j['output']).is_file())],
                                        ready=PYTHON.is_file() and (MODEL_ROOT/'YingMusic-SVC-full.pt').is_file(),
                                        output_dir=str(ROOT/'outputs')))
        elif path.startswith('/api/audio/'):
            with LOCK:
                file = FILES.get(path.rsplit('/', 1)[-1])
            if file:
                self.send_file(file['path'])
            else:
                self.json_response({'error': '文件记录不存在，请重新选择。'}, 404)
        elif path.startswith('/api/result/'):
            with LOCK:
                job = find_job(path.rsplit('/', 1)[-1])
            if job and job.get('output'):
                self.send_file(job['output'], 'download' in parse_qs(route.query))
            else:
                self.json_response({'error': '结果尚未生成'}, 404)
        elif path == '/':
            self.send_file(ROOT/'web/index.html')
        else:
            self.json_response({'error': '未找到页面'}, 404)

    def do_POST(self):
        global ACTIVE
        try:
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://localhost:{PORT}', f'http://127.0.0.1:{PORT}'}:
                self.json_response({'error': '禁止跨站请求'}, 403)
                return
            route = urlparse(self.path)
            if route.path == '/api/upload':
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 512 * 1024 * 1024:
                    raise ValueError('单个文件请控制在 512 MB 以内。')
                name = parse_qs(route.query).get('name', ['audio.wav'])[0]
                name = Path(name.replace('\\', '/')).name
                if Path(name).suffix.lower() not in EXTENSIONS:
                    raise ValueError('不支持这个文件类型，请选择音频。')
                dest = ROOT/'data'/(uuid.uuid4().hex + Path(name).suffix.lower())
                try:
                    with dest.open('wb') as out:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError('文件上传中断，请重试。')
                            out.write(chunk)
                            remaining -= len(chunk)
                    meta = register(dest, name)
                except Exception:
                    dest.unlink(missing_ok=True)
                    raise
                self.json_response(meta)
            elif route.path == '/api/import':
                body = self.json_body()
                self.json_response(register(str(body.get('path', '')).strip().strip('"'), body.get('name')))
            elif route.path == '/api/start':
                body = self.json_body()
                with LOCK:
                    if ACTIVE:
                        self.json_response({'error': '已有任务正在转换，请等待完成或取消。'}, 409)
                        return
                    source, target = FILES.get(body.get('source')), FILES.get(body.get('target'))
                    if not source or not target:
                        raise ValueError('请先选择源人声和参考音色。')
                    pitch = float(body.get('pitch', 0))
                    steps = int(body.get('steps', 30))
                    cfg = float(body.get('cfg', 0.7))
                    if not all(math.isfinite(v) for v in (pitch, cfg)) or not -12 <= pitch <= 12 or not 5 <= steps <= 60 or not 0 <= cfg <= 1:
                        raise ValueError('参数超出有效范围。')
                    precision = body.get('precision', 'bf16')
                    if precision not in ('bf16', 'fp32'):
                        raise ValueError('无效精度设置')
                    output_root = ROOT / 'outputs'
                    output_root.mkdir(exist_ok=True)
                    base = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    job_id = base
                    suffix = 2
                    while (output_root / job_id).exists():
                        job_id = f'{base}_{suffix}'
                        suffix += 1
                    (output_root / job_id).mkdir()
                    job = dict(id=job_id, name=source['name'], reference=target['name'], status='running',
                               started=time.time(), percent=0, stage='loading', message='正在启动转换引擎…',
                               indeterminate=True, pitch=pitch, steps=steps, logs=[])
                    JOBS[job_id] = job
                    ACTIVE = job_id
                    settings = dict(id=job_id, source=source['path'], target=target['path'], source_name=source['name'],
                                    pitch=pitch, steps=steps, cfg=cfg, precision=precision,
                                    output=str(ROOT/'outputs'/job_id))
                    threading.Thread(target=run_job, args=(job, settings), daemon=True).start()
                    self.json_response(public_job(job))
            elif route.path == '/api/cancel':
                body = self.json_body()
                with LOCK:
                    job = JOBS.get(body.get('id'))
                    if not job or job['id'] != ACTIVE:
                        raise ValueError('该任务已经结束。')
                    job.update(status='cancelled', message='已取消，可以重新转换。', ended=time.time(), indeterminate=False)
                    if PROCESS and PROCESS.poll() is None:
                        PROCESS.terminate()
                self.json_response({'ok': True})
            elif route.path == '/api/open-results':
                body = self.json_body()
                with LOCK:
                    job = find_job(body.get('id'))
                directory = Path(job['output']).parent if job and job.get('output') else ROOT/'outputs'
                os.startfile(str(directory))
                self.json_response({'ok': True})
            else:
                self.json_response({'error': '未找到接口'}, 404)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            self.json_response({'error': str(exc)}, 400)


if __name__ == '__main__':
    import argparse
    import webbrowser
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    except OSError:
        import urllib.request
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/state', timeout=2) as response:
                state = json.load(response)
            if 'output_dir' not in state:
                raise RuntimeError('端口被其他应用占用')
            webbrowser.open(f'http://localhost:{PORT}')
            raise SystemExit(0)
        except Exception:
            raise SystemExit('23335 端口已被其他程序占用，请关闭占用程序后重新启动。')
    print(f'better-YingMusic: http://localhost:{PORT}', flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()
    server.serve_forever()
