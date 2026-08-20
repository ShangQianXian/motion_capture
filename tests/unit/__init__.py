"""Dependency-free unit tests.

Written as ``unittest.TestCase`` classes so they run with either runner::

    python -m unittest discover -s tests/unit -t .
    python -m pytest tests/unit

Nothing here may import ``bpy``, ``torch``, ``mmpose``, ``mediapipe``, ``cv2`` or
``numpy``.
"""
