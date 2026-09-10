"""External worker process management (guide section 4.2).

Standard library only, so the same code runs inside Blender and from plain
Python tests. The Blender UI thread never blocks on the worker: stdout and
stderr are drained by reader threads into a queue that the modal operator polls.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

try:  # pragma: no cover - queue is always available on supported Pythons
    import queue
except ImportError:  # pragma: no cover
    queue = None  # type: ignore

from . import errors, job_schema, paths, progress

#: Windows flag that keeps a console window from flashing up.
_CREATE_NO_WINDOW = 0x08000000

#: Module name used for ``python -m`` (guide section 7.1).
WORKER_MODULE = "backend_worker.cli"

#: Seconds to wait for cooperative cancellation before escalating.
CANCEL_GRACE_SECONDS = 3.0

#: Seconds to wait after ``terminate()`` before ``kill()``.
TERMINATE_GRACE_SECONDS = 2.0

#: Exit code the worker uses for a cancelled run (guide section 7.1).
EXIT_CANCELLED = 130

_active_processes = []
_active_lock = threading.Lock()


def _popen_kwargs() -> dict:
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "cwd": paths.addon_root(),
        "universal_newlines": True,
        "bufsize": 1,
    }
    # ``encoding``/``errors`` require universal_newlines on all supported versions.
    kwargs["encoding"] = "utf-8"
    kwargs["errors"] = "replace"
    if os.name == "nt":
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    env = dict(os.environ)
    for key in list(env):
        if key.upper() in ("PYTHONHOME", "PYTHONPATH"):
            del env[key]
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    kwargs["env"] = env
    return kwargs


class WorkerProcess(object):
    """A running ``backend_worker.cli`` process with non-blocking event access."""

    def __init__(self, worker_python: str, job_path: str, output_dir: str = "", extra_args=()) -> None:
        self.worker_python = paths.normalize(worker_python)
        self.job_path = paths.normalize(job_path)
        self.output_dir = paths.normalize(output_dir) if output_dir else os.path.dirname(self.job_path)
        self.log_path = paths.worker_log_path(self.output_dir) if self.output_dir else ""
        self.events = queue.Queue() if queue is not None else None
        self.stderr_lines = []
        self.cancel_requested = False
        self._cancel_deadline = None
        self._terminate_deadline = None
        self._threads = []
        self._stderr_handle = None
        self._closed = False

        command = [self.worker_python, "-m", WORKER_MODULE, "--job", self.job_path]
        command.extend(extra_args)
        self.command = command

        # Remove a stale sentinel here, before the process exists. Doing it inside
        # the worker would race with an immediate cancel request and lose it.
        if self.output_dir:
            try:
                paths.ensure_dir(self.output_dir)
                stale = paths.cancel_flag_path(self.output_dir)
                if os.path.exists(stale):
                    os.remove(stale)
            except OSError:
                pass

        try:
            self.process = subprocess.Popen(command, **_popen_kwargs())
        except OSError as exc:
            raise errors.MocapError(
                errors.WORKER_PYTHON_NOT_FOUND,
                "无法启动 worker：{0}（{1}）".format(self.worker_python, exc),
                details={"command": command},
            )

        if self.log_path:
            try:
                paths.ensure_dir(os.path.dirname(self.log_path))
                self._stderr_handle = open(self.log_path, "a", encoding="utf-8", errors="replace")
                self._stderr_handle.write(
                    "\n=== worker start {0} ===\n{1}\n".format(
                        time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(command)
                    )
                )
                self._stderr_handle.flush()
            except OSError:
                self._stderr_handle = None

        self._start_reader(self.process.stdout, self._handle_stdout)
        self._start_reader(self.process.stderr, self._handle_stderr)

        with _active_lock:
            _active_processes.append(self)

    # -- reader plumbing ---------------------------------------------------------------

    def _start_reader(self, stream, handler) -> None:
        if stream is None:
            return
        thread = threading.Thread(target=self._read_loop, args=(stream, handler), daemon=True)
        thread.start()
        self._threads.append(thread)

    def _read_loop(self, stream, handler) -> None:
        try:
            for line in iter(stream.readline, ""):
                handler(line)
        except (ValueError, OSError):  # stream closed while reading
            pass
        finally:
            try:
                stream.close()
            except (ValueError, OSError):
                pass

    def _handle_stdout(self, line: str) -> None:
        event = progress.parse_progress_line(line)
        if self.events is not None:
            self.events.put(event)
        if self._stderr_handle is not None and not event.parsed:
            try:
                self._stderr_handle.write("UNPARSED STDOUT: {0}".format(line))
                self._stderr_handle.flush()
            except (OSError, ValueError):
                pass

    def _handle_stderr(self, line: str) -> None:
        text = line.rstrip("\r\n")
        if text:
            self.stderr_lines.append(text)
            del self.stderr_lines[:-200]
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.write(line)
                self._stderr_handle.flush()
            except (OSError, ValueError):
                pass

    # -- state -------------------------------------------------------------------------

    @property
    def returncode(self):
        return self.process.poll()

    def is_running(self) -> bool:
        return self.process.poll() is None

    def poll_events(self, limit: int = 200) -> list:
        """Drain up to ``limit`` pending events without blocking."""
        collected = []
        if self.events is None:
            return collected
        for _ in range(max(1, limit)):
            try:
                collected.append(self.events.get_nowait())
            except queue.Empty:
                break
        return collected

    def drain_finished(self) -> bool:
        """True when the process exited and every reader thread finished."""
        if self.is_running():
            return False
        return all(not thread.is_alive() for thread in self._threads)

    # -- cancellation ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cooperative cancellation, escalating over time."""
        now = time.time()
        if not self.cancel_requested:
            self.cancel_requested = True
            self._cancel_deadline = now + CANCEL_GRACE_SECONDS
            if self.output_dir:
                try:
                    paths.ensure_dir(self.output_dir)
                    with open(paths.cancel_flag_path(self.output_dir), "w", encoding="utf-8") as handle:
                        handle.write("cancel requested at {0}\n".format(time.time()))
                except OSError:
                    self._cancel_deadline = now  # cannot signal cooperatively
        self.tick_cancel()

    def tick_cancel(self) -> None:
        """Escalate a pending cancellation when the grace period elapsed."""
        if not self.cancel_requested or not self.is_running():
            return
        now = time.time()
        if self._terminate_deadline is not None:
            if now >= self._terminate_deadline:
                try:
                    self.process.kill()
                except OSError:
                    pass
            return
        if self._cancel_deadline is not None and now >= self._cancel_deadline:
            try:
                self.process.terminate()
            except OSError:
                pass
            self._terminate_deadline = now + TERMINATE_GRACE_SECONDS

    # -- teardown ----------------------------------------------------------------------

    def wait(self, timeout: float | None = None):
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def close(self) -> None:
        """Release handles and stop tracking this process."""
        if self._closed:
            return
        self._closed = True
        if self.is_running():
            try:
                self.process.kill()
            except OSError:
                pass
        for thread in self._threads:
            thread.join(timeout=0.5)
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except (OSError, ValueError):
                pass
            self._stderr_handle = None
        with _active_lock:
            if self in _active_processes:
                _active_processes.remove(self)

    def stderr_tail(self, max_lines: int = 20) -> str:
        return "\n".join(self.stderr_lines[-max_lines:])

    def exit_error(self) -> errors.MocapError:
        """Structured error describing a non-zero exit without a ``failed`` event."""
        code = self.returncode
        if code == EXIT_CANCELLED:
            return errors.MocapError(errors.CANCELLED, "worker 已取消。")
        detail = self.stderr_tail()
        if not detail and self.log_path:
            detail = paths.tail_text_file(self.log_path, 600)
        return errors.MocapError(
            errors.WORKER_EXIT_ERROR,
            "worker 异常退出（退出码 {0}）。".format(code),
            details={
                "returncode": code,
                "log_path": self.log_path,
                "stderr_tail": detail,
                "command": self.command,
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "WorkerProcess(pid={0}, running={1})".format(
            getattr(self.process, "pid", None), self.is_running()
        )


def run_worker(worker_python: str, job_path: str, output_dir: str = "", extra_args=()) -> WorkerProcess:
    """Start ``backend_worker.cli`` with the provided job path."""
    resolved = job_schema.validate_worker_python(worker_python)
    if not paths.is_file(job_path):
        raise errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "job 文件不存在：{0}".format(job_path),
            details={"job_path": str(job_path or "")},
        )
    return WorkerProcess(resolved, job_path, output_dir=output_dir, extra_args=extra_args)


def run_blocking(worker_python: str, job_path: str, output_dir: str = "", on_event=None, timeout: float | None = None):
    """Run a worker to completion, forwarding every event to ``on_event``.

    Returns ``(result_path, events)``. Raises :class:`errors.MocapError` when the
    worker reports a failure or exits non-zero. Used by background tests and by
    the operator when no window is available for a modal timer.
    """
    worker = run_worker(worker_python, job_path, output_dir=output_dir)
    events = []
    result_path = ""
    failure = None
    deadline = None if timeout is None else time.time() + timeout
    try:
        while True:
            for event in worker.poll_events():
                events.append(event)
                if on_event is not None:
                    on_event(event)
                if event.event == progress.EVENT_COMPLETED:
                    result_path = event.result_path
                elif event.event == progress.EVENT_FAILED:
                    failure = event.error or errors.MocapError(
                        errors.WORKER_EXIT_ERROR, "worker 报告失败但未提供错误详情。"
                    )
                elif event.event == progress.EVENT_CANCELLED:
                    failure = errors.MocapError(errors.CANCELLED, "worker 已取消。")
            if worker.drain_finished() and (worker.events is None or worker.events.empty()):
                break
            if deadline is not None and time.time() > deadline:
                worker.cancel()
                failure = errors.MocapError(
                    errors.WORKER_EXIT_ERROR,
                    "worker 超时（{0:.0f} 秒）。".format(timeout or 0.0),
                )
                break
            time.sleep(0.02)
        worker.wait(timeout=5.0)
        if failure is None and worker.returncode not in (0, None):
            failure = worker.exit_error()
    finally:
        worker.close()

    if failure is not None:
        raise failure
    return result_path, events


def probe(worker_python: str, args, timeout: float = 60.0) -> dict:
    """Run a short worker command (``--check-env`` / ``--check-cuda``).

    Returns the last JSON object printed on stdout. Raises
    :class:`errors.MocapError` when the interpreter is unusable or produced no
    JSON at all.
    """
    resolved = job_schema.validate_worker_python(worker_python)
    command = [resolved, "-m", WORKER_MODULE]
    command.extend(args)
    kwargs = _popen_kwargs()
    kwargs.pop("bufsize", None)
    try:
        completed = subprocess.run(command, timeout=timeout, **kwargs)
    except OSError as exc:
        raise errors.MocapError(
            errors.WORKER_PYTHON_NOT_FOUND,
            "无法运行 worker 命令：{0}（{1}）".format(" ".join(command), exc),
            details={"command": command},
        )
    except subprocess.TimeoutExpired:
        raise errors.MocapError(
            errors.WORKER_EXIT_ERROR,
            "worker 命令超时（{0:.0f} 秒）：{1}".format(timeout, " ".join(args)),
            details={"command": command},
        )

    payload = None
    for line in (completed.stdout or "").splitlines():
        event = progress.parse_progress_line(line)
        if event.parsed:
            payload = event.to_dict()
    if payload is None:
        raise errors.MocapError(
            errors.WORKER_EXIT_ERROR,
            "worker 未输出可解析的 JSON（退出码 {0}）。".format(completed.returncode),
            details={
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": (completed.stdout or "")[-600:],
                "stderr_tail": (completed.stderr or "")[-600:],
            },
        )
    payload["returncode"] = completed.returncode
    return payload


def check_env(worker_python: str, timeout: float = 120.0, profile=None) -> dict:
    """Run ``--check-env`` and return the parsed report."""
    args = ["--check-env"]
    if profile:
        args.extend(["--profile", profile])
    return probe(worker_python, args, timeout=timeout)


def check_cuda(worker_python: str, timeout: float = 120.0) -> dict:
    """Run ``--check-cuda`` and return the parsed report."""
    return probe(worker_python, ["--check-cuda"], timeout=timeout)


def active_processes() -> list:
    """Snapshot of tracked worker processes."""
    with _active_lock:
        return list(_active_processes)


def shutdown_all() -> int:
    """Terminate every tracked worker; called from the add-on's ``unregister``."""
    count = 0
    for worker in active_processes():
        try:
            worker.close()
            count += 1
        except Exception:  # pragma: no cover - defensive during teardown
            pass
    return count


def local_python() -> str:
    """Interpreter running this process; a sane default for the mock worker."""
    return sys.executable or ""
