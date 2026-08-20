"""Add-on preferences (guide section 3.2).

Note: this module deliberately does **not** use ``from __future__ import
annotations``. Blender resolves string annotations through
``typing.get_type_hints``, which swaps the globals and locals namespaces, so
PEP 563 breaks ``register_class`` for property definitions.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty

from ..core import model_manifest, paths
from . import ui_text

#: Package name used to look the preferences up: this module's parent package.
#: Dropping the last component (rather than keeping the first) keeps this correct
#: whether the add-on is installed as ``motion_capture`` or nested under an
#: extension repository prefix.
ADDON_PACKAGE = ".".join(__package__.split(".")[:-1]) or __package__


def profile_items():
    """Enum items for the capture profiles."""
    return ui_text.capture_profile_items()


class MotionCapturePreferences(bpy.types.AddonPreferences):
    """Paths and defaults that are shared by every scene."""

    bl_idname = ADDON_PACKAGE

    worker_python: StringProperty(
        name=ui_text.PREF_WORKER_PYTHON,
        description=ui_text.PREF_WORKER_PYTHON_DESC,
        subtype="FILE_PATH",
        default="",
    )
    models_root: StringProperty(
        name=ui_text.PREF_MODELS_ROOT,
        description=ui_text.PREF_MODELS_ROOT_DESC,
        subtype="DIR_PATH",
        default="",
    )
    max_vram_gb: FloatProperty(
        name=ui_text.PREF_MAX_VRAM,
        description=ui_text.PREF_MAX_VRAM_DESC,
        default=7.0,
        min=1.0,
        max=24.0,
    )
    default_profile: EnumProperty(
        name=ui_text.PREF_DEFAULT_PROFILE,
        description=ui_text.PREF_DEFAULT_PROFILE_DESC,
        items=profile_items(),
        default="quality",
    )
    allow_auto_download: BoolProperty(
        name=ui_text.PREF_ALLOW_AUTO_DOWNLOAD,
        description=ui_text.PREF_ALLOW_AUTO_DOWNLOAD_DESC,
        default=False,
    )

    # -- resolved paths ------------------------------------------------------------------

    def resolved_worker_python(self) -> str:
        """Absolute worker interpreter path (``//relative`` expanded)."""
        return paths.normalize(bpy.path.abspath(self.worker_python)) if self.worker_python else ""

    def resolved_models_root(self) -> str:
        """Absolute models root path (``//relative`` expanded)."""
        return paths.normalize(bpy.path.abspath(self.models_root)) if self.models_root else ""

    # -- UI ------------------------------------------------------------------------------

    def draw(self, context) -> None:
        layout = self.layout

        box = layout.box()
        box.label(text=ui_text.PREF_PATHS_HEADER, icon="FILE_FOLDER")
        box.prop(self, "worker_python")
        box.prop(self, "models_root")
        row = box.row(align=True)
        row.prop(self, "max_vram_gb")
        row.prop(self, "default_profile", text="")

        hint_box = layout.box()
        hint_box.label(text=ui_text.PREF_NO_AUTO_DOWNLOAD_NOTE, icon="INFO")
        hint_box.label(text=ui_text.MSG_WORKER_HINT, icon="DOT")
        row = hint_box.row()
        row.enabled = False
        row.prop(self, "allow_auto_download")

        if not self.worker_python or not self.models_root:
            hints = _manifest_hints()
            if hints:
                suggestion = layout.box()
                suggestion.label(text="manifest 建议值", icon="QUESTION")
                for label, value in hints:
                    suggestion.label(text="{0}: {1}".format(label, value))

        test_box = layout.box()
        test_box.label(text=ui_text.PREF_TESTS_HEADER, icon="PLAY")
        row = test_box.row(align=True)
        row.operator("mocap.test_worker_python", icon="CONSOLE")
        row.operator("mocap.test_cuda", icon="MEMORY")
        row = test_box.row(align=True)
        row.operator("mocap.preflight", icon="CHECKMARK")
        row.operator("mocap.open_models_dir", icon="FILE_FOLDER")


def _manifest_hints() -> list:
    """Suggested paths from the bundled manifest, for first-time setup."""
    try:
        manifest = model_manifest.load_manifest("")
    except Exception:  # noqa: BLE001 - hints are best effort only
        return []
    hints = []
    if manifest.worker_python_hint:
        hints.append((ui_text.PREF_WORKER_PYTHON, manifest.worker_python_hint))
    if manifest.models_root_hint:
        hints.append((ui_text.PREF_MODELS_ROOT, manifest.models_root_hint))
    return hints


def get_preferences(context=None):
    """Return this add-on's preferences, or ``None`` when unavailable."""
    ctx = context or bpy.context
    addons = getattr(ctx.preferences, "addons", None)
    if addons is None:
        return None
    entry = addons.get(ADDON_PACKAGE)
    return entry.preferences if entry is not None else None


classes = (MotionCapturePreferences,)
