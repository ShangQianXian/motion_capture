"""Explicit model downloads, using only the standard library and the manifest."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path, PureWindowsPath
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

BLOCK = 1024 * 1024


class DownloadError(RuntimeError):
    pass


def safe_path(root, relative):
    """Reject absolute paths, traversal, Windows streams and escaping symlinks."""
    value = str(relative).replace('\\', '/')
    parts = value.split('/')
    if (not value or PureWindowsPath(value).drive or value.startswith('/')
            or any(p in ('', '.', '..') or ':' in p or p.endswith((' ', '.'))
                   or PureWindowsPath(p).is_reserved() for p in parts)):
        raise DownloadError('Unsafe model path: ' + value)
    target = (Path(root) / value).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise DownloadError('Model path escapes Models Root: ' + value)
    return target


def checked_url(url):
    value = urlsplit(str(url))
    if value.scheme not in ('http', 'https') or not value.hostname or value.username or value.password:
        raise DownloadError('Expected an HTTP(S) download URL without embedded credentials.')
    return str(url)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def download_lock(root):
    """OS releases the lock even if a download process is killed."""
    state = safe_path(root, '.downloads')
    state.mkdir(parents=True, exist_ok=True)
    lock = safe_path(root, '.downloads/download.lock').open('a+b')
    try:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DownloadError('Another downloader is using this Models Root.') from exc
        yield
    finally:
        lock.close()


def select_artifacts(manifest, profiles=(), artifact_ids=(), include_optional=False, all_models=False):
    """Select requested profiles, never their preflight downgrade; include configs."""
    selected, visiting, by_path = {}, set(), {}
    for artifact in manifest.artifacts:
        name = artifact.get('id')
        if not isinstance(name, str) or not name or name in visiting:
            raise DownloadError('Manifest contains an empty or duplicate artifact ID.')
        visiting.add(name)
        path = safe_path(manifest.models_root, artifact.get('relative_path', ''))
        key = str(path).casefold()
        if key in by_path:
            raise DownloadError('Multiple artifacts use the same destination: ' + str(path))
        by_path[key] = name
    visiting.clear()

    def add(name):
        if name in visiting:
            raise DownloadError('Circular config_id dependency: ' + name)
        if name in selected:
            return
        artifact = manifest.artifact(name)
        if artifact is None:
            raise DownloadError('Unknown artifact ID: ' + str(name))
        checked_url(artifact.get('download_url', ''))
        digest = artifact.get('sha256')
        if digest is not None and not re.fullmatch(r'[0-9a-fA-F]{64}', str(digest)):
            raise DownloadError('Invalid SHA256 for ' + name)
        size = artifact.get('size_bytes')
        if size is not None and (type(size) is not int or size <= 0):
            raise DownloadError('Invalid size_bytes for ' + name)
        visiting.add(name)
        if artifact.get('config_id'):
            add(artifact['config_id'])
        visiting.remove(name)
        selected[name] = artifact

    seen_profiles = set()

    def profile(name):
        if name in seen_profiles:
            return
        if name not in manifest.profiles:
            raise DownloadError('Unknown profile: ' + name)
        seen_profiles.add(name)
        rules = manifest.rules_for(name)
        for artifact_id in rules.get('required_artifact_ids', []):
            add(artifact_id)
        alternatives = rules.get('required_any_of', [])
        if alternatives:
            # Existing alternatives still pass through integrity checking later.
            existing = [i for i in alternatives if manifest.artifact(i) and
                        safe_path(manifest.models_root, manifest.artifact(i)['relative_path']).is_file()]
            add(next(iter(existing or alternatives)))
        if include_optional or name == 'hand_enhanced':
            for status in manifest.statuses_for_profile(name):
                add(status.id)
        if rules.get('fallback_profile'):
            profile(rules['fallback_profile'])

    if all_models:
        for artifact in manifest.artifacts:
            add(artifact['id'])
    else:
        for name in profiles:
            profile(name)
        for name in artifact_ids:
            add(name)
    return list(selected.values())


class Downloader:
    def __init__(self, root, *, timeout=30, retries=3, force=False, verify_only=False,
                 log=print, opener=urlopen, sleep=time.sleep):
        self.root = Path(root).resolve()
        self.timeout, self.retries = timeout, retries
        self.force, self.verify_only = force, verify_only
        self.log, self.opener, self.sleep = log, opener, sleep
        self.package_bases = set()

    def state_paths(self, target):
        relative = target.relative_to(self.root).as_posix()
        key = hashlib.sha256(relative.encode()).hexdigest()
        return tuple(safe_path(self.root, '.downloads/' + key + suffix)
                     for suffix in ('.record.json', '.part', '.part.json'))

    def inspect(self, target, url, expected=None, size=None):
        if not target.is_file():
            return False
        record = read_json(self.state_paths(target)[0])
        digest = expected or (record.get('sha256') if record.get('url') == url else None)
        if not digest:
            raise DownloadError('Existing file has no trusted manifest hash or download receipt; '
                                'use --force to replace it: ' + str(target))
        if (target.stat().st_size == 0 or (size is not None and target.stat().st_size != size)
                or (record.get('url') == url and record.get('size_bytes') != target.stat().st_size)
                or sha256_file(target).lower() != digest.lower()):
            raise DownloadError('Existing file failed integrity checking; use --force: ' + str(target))
        return True

    def record(self, target, url):
        data = {'url': url, 'relative_path': target.relative_to(self.root).as_posix(),
                'size_bytes': target.stat().st_size, 'sha256': sha256_file(target)}
        atomic_json(self.state_paths(target)[0], data)

    def fetch(self, url, relative, *, expected=None, size=None, config=False):
        url, target = checked_url(url), safe_path(self.root, relative)
        if not self.force or self.verify_only:
            if self.inspect(target, url, expected, size):
                self.log('[verified] ' + relative)
                return target
        if self.verify_only:
            raise DownloadError('Missing file: ' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt, part, resume_file = self.state_paths(target)
        part.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.retries + 1):
            try:
                self.transfer(url, part, resume_file, size, config)
                if expected and sha256_file(part).lower() != expected.lower():
                    part.unlink(missing_ok=True)
                    resume_file.unlink(missing_ok=True)
                    raise DownloadError('SHA256 mismatch: ' + relative)
                head = part.read_bytes() if config else b''
                if config:
                    self.config_bases(head.decode('utf-8-sig'))
                else:
                    with part.open('rb') as stream:
                        head = stream.read(512).lstrip().lower()
                    if head.startswith((b'<html', b'<!doctype', b'<?xml', b'{"error')):
                        part.unlink(missing_ok=True)
                        raise DownloadError('Server returned an error document instead of a model: ' + relative)
                os.replace(part, target)
                self.record(target, url)
                resume_file.unlink(missing_ok=True)
                self.log('[downloaded] ' + relative)
                return target
            except HTTPError as exc:
                if exc.code == 416:
                    part.unlink(missing_ok=True)
                    resume_file.unlink(missing_ok=True)
                elif exc.code not in (408, 429) and exc.code < 500:
                    raise DownloadError('HTTP {0}: {1}'.format(exc.code, url)) from exc
                failure = exc
            except (URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as exc:
                failure = exc
            if attempt == self.retries:
                raise DownloadError('Download failed after retries: {0}: {1}'.format(url, failure)) from failure
            self.log('[retry {0}/{1}] {2}'.format(attempt + 1, self.retries, failure))
            self.sleep(min(2 ** attempt, 8))

    def transfer(self, url, part, resume_file, expected_size, config):
        previous = read_json(resume_file)
        validator = previous.get('etag') or previous.get('last_modified')
        offset = part.stat().st_size if part.exists() and previous.get('url') == url and validator else 0
        headers = {'User-Agent': 'MotionCapture-ModelDownloader/0.1', 'Accept-Encoding': 'identity'}
        if offset:
            headers.update(Range='bytes={0}-'.format(offset), **{'If-Range': validator})
            self.log('[resume] {0:.1f} MiB'.format(offset / BLOCK))
        with self.opener(Request(url, headers=headers), timeout=self.timeout) as response:
            checked_url(response.geturl())
            status = response.status
            if status not in (200, 206):
                raise DownloadError('Unexpected HTTP status: ' + str(status))
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise DownloadError('Server ignored Accept-Encoding: identity.')
            length = response.headers.get('Content-Length')
            length = int(length) if length is not None else None
            total = length
            if status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                    raise DownloadError('Invalid resume Content-Range; partial file kept.')
                total = int(match[3])
                if length is not None and length != total - offset:
                    raise DownloadError('Inconsistent resume Content-Length.')
                etag = response.headers.get('ETag')
                if offset and previous.get('etag') and etag and etag != previous['etag']:
                    part.unlink(missing_ok=True)
                    raise ConnectionError('Remote file changed; restarting download.')
            else:
                offset = 0  # Range ignored or If-Range failed: overwrite, never append.
            if expected_size is not None and total is not None and total != expected_size:
                raise DownloadError('Remote size differs from manifest size_bytes.')
            if config and total is not None and total > 5 * BLOCK:
                raise DownloadError('Config exceeds 5 MiB limit.')
            etag = response.headers.get('ETag')
            atomic_json(resume_file, {'url': url, 'etag': etag if etag and not etag.startswith('W/') else None,
                                     'last_modified': response.headers.get('Last-Modified'), 'total': total})
            received, last_log = offset, 0.
            with part.open('ab' if offset else 'wb') as output:
                while True:
                    chunk = response.read(BLOCK)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if config and received > 5 * BLOCK:
                        raise DownloadError('Config exceeds 5 MiB limit.')
                    if time.monotonic() - last_log >= 1:
                        self.log('  {0:.1f} MiB{1}'.format(received / BLOCK,
                                 ' / {0:.1f} MiB ({1:.0%})'.format(total / BLOCK, received / total) if total else ''))
                        last_log = time.monotonic()
            if received == 0 or (total is not None and received != total) or (expected_size is not None and received != expected_size):
                raise ConnectionError('Incomplete download ({0} / {1} bytes).'.format(received, total))

    @staticmethod
    def config_bases(source):
        """Inspect configuration syntax without importing/executing downloaded code."""
        try:
            tree = ast.parse(source)
            values = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_base_' for t in node.targets):
                    value = ast.literal_eval(node.value)
                    values = [value] if isinstance(value, str) else value
                if isinstance(node, ast.ImportFrom) and any(n.name == 'read_base' for n in node.names):
                    raise ValueError('read_base configs are not supported')
            if not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values):
                raise ValueError('_base_ must be a literal string or list')
            return values
        except (ValueError, TypeError, SyntaxError) as exc:
            raise DownloadError('Invalid or unsupported Python config: ' + str(exc)) from exc

    def config(self, artifact):
        url = checked_url(artifact['download_url'])
        target = safe_path(self.root, artifact['relative_path'])
        wrapper_url = 'config-wrapper:' + url
        if target.exists() and (not self.force or self.verify_only):
            self.inspect(target, wrapper_url)
        parsed = urlsplit(url)
        parts = parsed.path.lstrip('/').split('/')
        # Keep the pinned GitHub repository tree; generic servers use their URL root.
        prefix = '/' + '/'.join(parts[:3]) + '/' if parsed.hostname == 'raw.githubusercontent.com' else '/'
        origin = urlunsplit((parsed.scheme, parsed.netloc, prefix, '', ''))
        mirror = 'openmmlab/configs/_sources/' + hashlib.sha256(origin.encode()).hexdigest()[:24] + '/'
        seen, active = {}, set()

        def visit(current, expected=None, size=None):
            if current in active:
                raise DownloadError('Circular _base_ config dependency: ' + current)
            if current in seen:
                return seen[current]
            if len(seen) + len(active) >= 64:
                raise DownloadError('Config dependency graph exceeds 64 files.')
            parsed_current = urlsplit(current)
            if not current.startswith(origin) or parsed_current.query or parsed_current.fragment:
                raise DownloadError('Config dependency leaves its source repository: ' + current)
            relative = mirror + parsed_current.path[len(prefix):]
            active.add(current)
            path = self.fetch(current, relative, expected=expected, size=size, config=True)
            for base in self.config_bases(path.read_text(encoding='utf-8-sig')):
                if '::' in base:
                    self.package_bases.add(base.split('::', 1)[0])
                    continue  # mmengine resolves these from installed mmdet/mmpose.
                if base.startswith(('/', '\\')) or PureWindowsPath(base).drive or urlsplit(base).scheme:
                    raise DownloadError('Unsupported absolute config base: ' + base)
                visit(urljoin(current, base.replace('\\', '/')))
            active.remove(current)
            seen[current] = path
            return path

        source = visit(url, artifact.get('sha256'), artifact.get('size_bytes'))
        if self.verify_only:
            if not self.inspect(target, wrapper_url):
                raise DownloadError('Missing config entry: ' + str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            relative = os.path.relpath(source, target.parent).replace('\\', '/')
            wrapper = '# Generated by tools/download_models.py; keep the _sources directory.\n_base_ = ' + repr(relative) + '\n'
            temporary = safe_path(self.root, target.relative_to(self.root).as_posix() + '.tmp')
            temporary.write_text(wrapper, encoding='utf-8')
            os.replace(temporary, target)
            self.record(target, wrapper_url)
        self.log('[config ready] ' + artifact['relative_path'])
        return target

    def artifact(self, artifact):
        if artifact.get('kind') == 'config':
            return self.config(artifact)
        return self.fetch(artifact['download_url'], artifact['relative_path'],
                          expected=artifact.get('sha256'), size=artifact.get('size_bytes'))
