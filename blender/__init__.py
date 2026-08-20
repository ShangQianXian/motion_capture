"""Blender-side integration: Rigify detection, retargeting and Action baking.

Modules in this package are the only ones allowed to import ``bpy`` and
``mathutils`` for real work; they convert to and from the framework-free maths in
``core`` at the boundary.
"""

from __future__ import annotations

__all__ = ["rigify_adapter", "action_baker", "temp_data"]
