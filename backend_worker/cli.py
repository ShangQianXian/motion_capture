"""Worker command line entry point (guide section 7.1).

Usage (run from the add-on root so ``core`` resolves as a top-level package)::

    python -m backend_worker.cli --job D:/path/job.json
    python -m backend_worker.cli --job D:/path/job.json --mock
    python -m backend_worker.cli --check-env
    python -m backend_worker.cli --check-cuda

Contract:

* stdout carries nothing but JSONL progress events,
* debug output goes to stderr and ``<output.dir>/worker.log``,
* a failure emits a ``failed`` event and exits non-zero,
* cancellation emits a ``cancelled`` event and exits with code 130.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

# Allow ``python backend_worker/cli.py`` and ``python -m backend_worker.cli`` from
# anywhere by putting the add-on root (the parent of this package) on sys.path.
_ADDON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ADDON_ROOT not in sys.path:
    sys.path.insert(0, _ADDON_ROOT)
if not __package__:
    __package__ = "backend_worker"

from ._core import errors, job_schema  # noqa: E402  (path bootstrap must run first)

from . import pipeline  # noqa: E402
from .reporter import CancellationToken, Reporter  # noqa: E402

#: Exit codes.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CANCELLED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend_worker.cli",
        description="Motion Capture for Rigify - external inference worker.",
    )
    parser.add_argument("--job", metavar="PATH", help="path to a job JSON file")
    parser.add_argument("--profile", choices=("preview", "fallback_cpu", "quality", "quality_plus", "quality_feet"),
                        help="validate required dependencies for this profile with --check-env")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="produce a synthetic result without loading any model",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="report the interpreter and optional dependency versions as JSON",
    )
    parser.add_argument(
        "--check-cuda",
        action="store_true",
        help="report CUDA availability as JSON",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_env or args.check_cuda:
        return _run_probe(args)

    if not args.job:
        parser.error("--job PATH is required unless --check-env or --check-cuda is given")

    reporter = None
    cancel_token = None
    try:
        job = job_schema.read_job(args.job)
        output_dir = str((job.get("output") or {}).get("dir") or "")
        reporter = Reporter(job_id=str(job.get("job_id") or ""), output_dir=output_dir)
        # The sentinel is cleared by the add-on *before* the worker starts, never
        # here: clearing it on this side would race with an immediate cancel and
        # silently discard the request.
        cancel_token = CancellationToken(output_dir)
        if job.get("mode") == job_schema.MODE_MEDIA_PREVIEW:
            from . import media_preview
            with contextlib.redirect_stdout(sys.stderr):
                media_preview.serve(job, reporter, cancel_token, sys.stdin)
            reporter.completed("")
            return EXIT_OK
        reporter.started(
            "Worker started (profile={0}, mock={1})".format(
                (job.get("model") or {}).get("profile"), bool(args.mock)
            )
        )
        with contextlib.redirect_stdout(sys.stderr):
            result_path = pipeline.run_job(job, reporter, cancel_token, mock=bool(args.mock))
        reporter.completed(result_path)
        return EXIT_OK
    except KeyboardInterrupt:
        _emit_cancelled(reporter)
        return EXIT_CANCELLED
    except errors.MocapError as exc:
        if exc.code == errors.CANCELLED:
            _emit_cancelled(reporter)
            return EXIT_CANCELLED
        _emit_failed(reporter, exc)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - a worker must never leak a traceback to stdout
        _emit_failed(reporter, errors.wrap_unexpected(exc, "worker main"))
        return EXIT_FAILED
    finally:
        if reporter is not None:
            reporter.close()


def _run_probe(args) -> int:
    """Handle ``--check-env`` / ``--check-cuda`` without a job file."""
    reporter = Reporter()
    try:
        if args.check_env:
            reporter.emit("started", message="check-env")
            with contextlib.redirect_stdout(sys.stderr):
                report = pipeline.check_env(profile=args.profile)
            if report.get("ok", True):
                reporter.emit("completed", progress=1.0, check="env", report=report)
            else:
                reporter.emit("failed", check="env", report=report, error=report["error"])
                return EXIT_FAILED
        if args.check_cuda:
            reporter.emit("started", message="check-cuda")
            with contextlib.redirect_stdout(sys.stderr):
                report = pipeline.check_cuda()
            if report.get("available"):
                reporter.emit("completed", progress=1.0, check="cuda", report=report)
            else:
                payload = report.get("error") or errors.MocapError(
                    errors.CUDA_UNAVAILABLE, "CUDA 不可用。"
                ).to_dict()
                reporter.emit("failed", error=payload, check="cuda", report=report)
                return EXIT_FAILED
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001
        _emit_failed(reporter, errors.wrap_unexpected(exc, "probe"))
        return EXIT_FAILED
    finally:
        reporter.close()


def _emit_failed(reporter, error) -> None:
    if reporter is not None:
        reporter.failed(error)
        return
    # No reporter yet (e.g. the job file itself was unreadable): still emit JSONL.
    fallback = Reporter()
    try:
        fallback.failed(error)
    finally:
        fallback.close()


def _emit_cancelled(reporter) -> None:
    if reporter is not None:
        reporter.cancelled()
        return
    fallback = Reporter()
    try:
        fallback.cancelled()
    finally:
        fallback.close()


if __name__ == "__main__":
    sys.exit(main())
