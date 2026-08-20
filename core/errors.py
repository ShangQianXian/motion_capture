"""Structured error objects shared by the add-on and the external worker.

This module is intentionally standard-library only so it can be imported both as
``core.errors`` (external worker process, add-on root on ``sys.path``) and as
``motion_capture.core.errors`` (inside Blender).

The serialised form follows ``docs/DEVELOPMENT_GUIDE.md`` section 5.4::

    {
      "code": "MODEL_MISSING",
      "message": "...",
      "suggestion": "...",
      "recoverable": true,
      "details": {...}
    }
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Error codes required by docs/DEVELOPMENT_GUIDE.md section 5.4
# --------------------------------------------------------------------------------------

WORKER_PYTHON_NOT_FOUND = "WORKER_PYTHON_NOT_FOUND"
MODELS_ROOT_NOT_FOUND = "MODELS_ROOT_NOT_FOUND"
MANIFEST_INVALID = "MANIFEST_INVALID"
MODEL_MISSING = "MODEL_MISSING"
CONFIG_MISSING = "CONFIG_MISSING"
CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
CUDA_OOM = "CUDA_OOM"
MEDIA_OPEN_FAILED = "MEDIA_OPEN_FAILED"
NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
RIGIFY_NOT_FOUND = "RIGIFY_NOT_FOUND"
RIGIFY_MAPPING_FAILED = "RIGIFY_MAPPING_FAILED"
RESULT_SCHEMA_INVALID = "RESULT_SCHEMA_INVALID"

#: Codes the documentation explicitly requires support for.
REQUIRED_ERROR_CODES = (
    WORKER_PYTHON_NOT_FOUND,
    MODELS_ROOT_NOT_FOUND,
    MANIFEST_INVALID,
    MODEL_MISSING,
    CONFIG_MISSING,
    CUDA_UNAVAILABLE,
    CUDA_OOM,
    MEDIA_OPEN_FAILED,
    NO_PERSON_DETECTED,
    RIGIFY_NOT_FOUND,
    RIGIFY_MAPPING_FAILED,
    RESULT_SCHEMA_INVALID,
)

# --------------------------------------------------------------------------------------
# Extension codes (documented in README.md as additions beyond the guide)
# --------------------------------------------------------------------------------------

JOB_SCHEMA_INVALID = "JOB_SCHEMA_INVALID"
DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
WORKER_EXIT_ERROR = "WORKER_EXIT_ERROR"
CANCELLED = "CANCELLED"
MEDIA_NOT_FOUND = "MEDIA_NOT_FOUND"
UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
INTERNAL_ERROR = "INTERNAL_ERROR"

EXTENSION_ERROR_CODES = (
    JOB_SCHEMA_INVALID,
    DEPENDENCY_MISSING,
    WORKER_EXIT_ERROR,
    CANCELLED,
    MEDIA_NOT_FOUND,
    UNSUPPORTED_MEDIA_TYPE,
    INTERNAL_ERROR,
)

ALL_ERROR_CODES = REQUIRED_ERROR_CODES + EXTENSION_ERROR_CODES

# --------------------------------------------------------------------------------------
# Default suggestions / recoverability
# --------------------------------------------------------------------------------------

#: Default remediation hints, shown to the user when a raiser gives no suggestion.
DEFAULT_SUGGESTIONS = {
    WORKER_PYTHON_NOT_FOUND: "在插件偏好设置中把 Worker Python 指向一个可用的 Python 解释器（例如 .venv/Scripts/python.exe）。",
    MODELS_ROOT_NOT_FOUND: "在插件偏好设置中把 Models Root 指向存在的模型根目录。",
    MANIFEST_INVALID: "检查 <models_root>/manifest.json 是否为合法 JSON；删除它可回退到内置 manifest.example.json。",
    MODEL_MISSING: "按预检面板给出的相对路径和下载链接手动下载模型权重后重新检查。",
    CONFIG_MISSING: "按预检面板给出的相对路径下载对应的 MMPose/MMDetection config 文件后重新检查。",
    CUDA_UNAVAILABLE: "确认 worker 环境安装了 CUDA 版 PyTorch；或改用 preview / fallback_cpu profile。",
    CUDA_OOM: "改用 quality profile、降低输入分辨率，或减小 Max VRAM GB 后重试。",
    MEDIA_OPEN_FAILED: "确认媒体文件未损坏且格式受支持（图片 png/jpg/jpeg/webp，视频 mp4/mov/avi/mkv）。",
    NO_PERSON_DETECTED: "换一段画面中人物完整可见的素材，或降低检测阈值后重试。",
    RIGIFY_NOT_FOUND: "选择由 Rigify 生成的人形 rig（不是 metarig）。先在 metarig 上点击 Generate Rig。",
    RIGIFY_MAPPING_FAILED: "确认目标 rig 由标准 Rigify Human metarig 生成，包含 spine_fk / upper_arm_fk / thigh_fk 等控制骨。",
    RESULT_SCHEMA_INVALID: "该 mocap_result.json 不符合 v0.1 结果规范，请重新运行捕捉生成结果。",
    JOB_SCHEMA_INVALID: "检查捕捉面板中的输入文件、帧范围和目标 FPS 设置。",
    DEPENDENCY_MISSING: "在 worker 虚拟环境中安装缺失的 Python 依赖（见 docs/INSTALL.md）。",
    WORKER_EXIT_ERROR: "查看 worker.log 获取详细堆栈；确认 Worker Python 环境完整。",
    CANCELLED: "任务已被用户取消。",
    MEDIA_NOT_FOUND: "确认 Source Media 路径存在。",
    UNSUPPORTED_MEDIA_TYPE: "改用受支持的格式：图片 png/jpg/jpeg/webp，视频 mp4/mov/avi/mkv。",
    INTERNAL_ERROR: "请把 worker.log 与操作步骤一起反馈给开发者。",
}

#: Codes that do not necessarily invalidate the whole session.
_NON_RECOVERABLE = frozenset({INTERNAL_ERROR, RESULT_SCHEMA_INVALID})


def default_suggestion(code: str) -> str:
    """Return the default remediation hint for ``code``."""
    return DEFAULT_SUGGESTIONS.get(code, "")


def default_recoverable(code: str) -> bool:
    """Return whether ``code`` is considered recoverable by default."""
    return code not in _NON_RECOVERABLE


class MocapError(Exception):
    """A structured, user-presentable error.

    Attributes mirror the JSON error object from the development guide so that
    the same instance can be reported to Blender's UI and serialised to worker
    stdout without a second representation.
    """

    __slots__ = ("code", "message", "suggestion", "recoverable", "details")

    def __init__(
        self,
        code: str,
        message: str,
        suggestion: str | None = None,
        recoverable: bool | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = default_suggestion(code) if suggestion is None else suggestion
        self.recoverable = default_recoverable(code) if recoverable is None else bool(recoverable)
        self.details = dict(details) if details else {}

    # -- serialisation ---------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the section 5.4 JSON representation."""
        payload = {
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
            "recoverable": self.recoverable,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "MocapError":
        """Rebuild an error from its serialised form; tolerates partial payloads."""
        if not isinstance(data, dict):
            return cls(INTERNAL_ERROR, "Malformed error payload.", details={"raw": repr(data)})
        return cls(
            code=str(data.get("code") or INTERNAL_ERROR),
            message=str(data.get("message") or "Unknown worker error."),
            suggestion=data.get("suggestion"),
            recoverable=data.get("recoverable"),
            details=data.get("details") if isinstance(data.get("details"), dict) else None,
        )

    # -- presentation ----------------------------------------------------------------

    def user_text(self) -> str:
        """Single-line text for ``Operator.report`` and the log panel."""
        text = "[{0}] {1}".format(self.code, self.message)
        if self.suggestion:
            text = "{0} 建议：{1}".format(text, self.suggestion)
        return text

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "MocapError(code={0!r}, message={1!r})".format(self.code, self.message)


def wrap_unexpected(exc: BaseException, context: str = "") -> MocapError:
    """Convert an arbitrary exception into a structured :class:`MocapError`."""
    if isinstance(exc, MocapError):
        return exc
    message = "{0}: {1}".format(type(exc).__name__, exc) if str(exc) else type(exc).__name__
    if context:
        message = "{0} ({1})".format(message, context)
    return MocapError(INTERNAL_ERROR, message, details={"context": context} if context else None)
