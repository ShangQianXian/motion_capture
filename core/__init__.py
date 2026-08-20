"""Framework-free core logic shared by the Blender add-on and the external worker.

Every module in this package is standard-library only and must never import
``bpy``, ``torch``, ``mediapipe``, ``cv2`` or ``numpy``. Internal imports are
relative so the package works under both import names:

* ``core.*``               - external worker process (add-on root on ``sys.path``)
* ``motion_capture.core.*`` - inside Blender
"""

from __future__ import annotations

__all__ = [
    "errors",
    "paths",
    "model_manifest",
    "skeleton",
    "job_schema",
    "result_schema",
    "progress",
    "smoothing",
    "retarget_math",
    "worker_client",
]
