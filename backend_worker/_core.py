"""Dual-mode access to the framework-free ``core`` package.

The worker is documented to run as ``python -m backend_worker.cli`` from the
add-on root, where ``core`` is a top-level package. The Blender-side tests and
tools also import ``motion_capture.backend_worker.mock_source`` in-process, where
the same package is reachable only as ``motion_capture.core``.

Rather than guessing with ``try``/``except ImportError`` - which would mask a real
import error inside ``core`` - the invocation style is detected from
``__package__``:

* ``backend_worker._core``                    -> top-level, use ``core``
* ``motion_capture.backend_worker._core``     -> nested, use ``..core``

Importing through this one module also guarantees a single ``core`` instance per
process, so ``isinstance`` checks stay reliable.
"""

from __future__ import annotations

_NESTED = bool(__package__) and "." in __package__

if _NESTED:  # imported as motion_capture.backend_worker._core (inside Blender)
    from ..core import motion_processing
    from ..core import errors  # noqa: F401
    from ..core import job_schema  # noqa: F401
    from ..core import model_manifest  # noqa: F401
    from ..core import paths  # noqa: F401
    from ..core import progress  # noqa: F401
    from ..core import result_schema  # noqa: F401
    from ..core import retarget_math  # noqa: F401
    from ..core import skeleton  # noqa: F401
    from ..core import smoothing  # noqa: F401
    from ..core import preview  # noqa: F401
else:  # imported as backend_worker._core (worker process)
    from core import motion_processing
    from core import errors  # noqa: F401
    from core import job_schema  # noqa: F401
    from core import model_manifest  # noqa: F401
    from core import paths  # noqa: F401
    from core import progress  # noqa: F401
    from core import result_schema  # noqa: F401
    from core import retarget_math  # noqa: F401
    from core import skeleton  # noqa: F401
    from core import smoothing  # noqa: F401
    from core import preview  # noqa: F401

__all__ = [
    'motion_processing',
    "errors",
    "job_schema",
    "model_manifest",
    "paths",
    "progress",
    "result_schema",
    "retarget_math",
    "skeleton",
    "smoothing",
    "preview",
]
