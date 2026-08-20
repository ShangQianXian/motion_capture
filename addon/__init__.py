"""Blender UI layer: preferences, scene properties, panels and operators.

Modules here import ``bpy`` but must never import ``torch``, ``mmpose``,
``mediapipe`` or ``cv2``, and must never run inference on Blender's main thread
(guide section 3.5). Long work always goes to the external worker.
"""

from __future__ import annotations

__all__ = ["preferences", "properties", "panels", "operators", "ui_text"]
