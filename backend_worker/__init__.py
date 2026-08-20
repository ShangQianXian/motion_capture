"""External inference worker.

Run as a module from the add-on root so ``core`` resolves as a top-level
package (guide section 7.1)::

    python -m backend_worker.cli --job D:/path/job.json
    python -m backend_worker.cli --job D:/path/job.json --mock
    python -m backend_worker.cli --check-env
    python -m backend_worker.cli --check-cuda

Heavy dependencies (torch, mmpose, mediapipe, cv2, numpy) are imported lazily by
the individual stage modules, never at package import time.
"""

from __future__ import annotations
