"""Model manifest loading and per-profile preflight checks.

Implements ``docs/DEVELOPMENT_GUIDE.md`` sections 4.1 and 6. The manifest is the
single source of truth for model files; this module never maintains a second
download list.

Lookup order (section 6.1):

1. ``<models_root>/manifest.json``   - user copy wins
2. ``<addon_root>/models/manifest.example.json`` - bundled fallback
"""

from __future__ import annotations

import json
import os

from . import errors, paths

#: Profiles selectable as a capture profile in the UI.
CAPTURE_PROFILES = ("preview", "quality", "quality_plus", "fallback_cpu")

#: Profiles that only describe optional model groups (preflight display only).
AUXILIARY_PROFILES = ("hand_enhanced",)

ALL_PROFILES = CAPTURE_PROFILES + AUXILIARY_PROFILES

#: Built-in preflight rules used when the manifest has no entry for a profile.
#:
#: ``manifest.example.json`` ships rules for preview/quality/quality_plus/
#: hand_enhanced but not for ``fallback_cpu``; guide section 6.2 states that
#: profile prefers MediaPipe Lite and otherwise MediaPipe Full, which is an
#: "any of" requirement rather than a hard list. A user supplied
#: ``manifest.json`` may override this by adding its own ``preflight_rules``.
PROFILE_DEFAULT_RULES = {
    "fallback_cpu": {
        "required_any_of": ["mediapipe_pose_lite", "mediapipe_pose_full"],
        "optional_artifact_ids": ["mediapipe_hand"],
    },
    "quality_plus": {
        "optional_artifact_ids": ["mediapipe_hand", "videopose3d_body3d"],
        "fallback_profile": "quality",
    },
}

_STATUS_OK = "OK"
_STATUS_MISSING_REQUIRED = "MISSING_REQUIRED"
_STATUS_MISSING_OPTIONAL = "MISSING_OPTIONAL"


class ArtifactStatus(object):
    """Presence information for a single manifest artifact."""

    __slots__ = (
        "id",
        "display_name",
        "purpose",
        "kind",
        "framework",
        "relative_path",
        "absolute_path",
        "download_url",
        "required",
        "exists",
        "config_id",
        "fallback",
        "profiles",
    )

    def __init__(self, artifact: dict, models_root: str, required: bool) -> None:
        self.id = str(artifact.get("id") or "")
        self.display_name = str(artifact.get("display_name") or self.id)
        self.purpose = str(artifact.get("purpose") or "")
        self.kind = str(artifact.get("kind") or "weights")
        self.framework = str(artifact.get("framework") or "")
        self.relative_path = str(artifact.get("relative_path") or "")
        self.download_url = str(artifact.get("download_url") or "")
        self.config_id = artifact.get("config_id") or None
        self.fallback = str(artifact.get("fallback") or "")
        self.profiles = tuple(artifact.get("profiles") or ())
        self.required = bool(required)
        self.absolute_path = (
            os.path.join(paths.normalize(models_root), *self.relative_path.split("/"))
            if models_root and self.relative_path
            else ""
        )
        self.exists = bool(self.absolute_path) and os.path.isfile(self.absolute_path)

    @property
    def status(self) -> str:
        """``OK`` / ``MISSING_REQUIRED`` / ``MISSING_OPTIONAL``."""
        if self.exists:
            return _STATUS_OK
        return _STATUS_MISSING_REQUIRED if self.required else _STATUS_MISSING_OPTIONAL

    @property
    def error_code(self) -> str:
        """Error code that best describes this artifact being absent."""
        return errors.CONFIG_MISSING if self.kind == "config" else errors.MODEL_MISSING

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "purpose": self.purpose,
            "kind": self.kind,
            "framework": self.framework,
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "download_url": self.download_url,
            "required": self.required,
            "exists": self.exists,
            "status": self.status,
            "config_id": self.config_id,
            "fallback": self.fallback,
            "profiles": list(self.profiles),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "ArtifactStatus({0!r}, {1})".format(self.id, self.status)


class ModelManifest(object):
    """Parsed manifest, preserving the original artifact list."""

    __slots__ = ("data", "source_path", "models_root", "artifacts_by_id")

    def __init__(self, data: dict, source_path: str, models_root: str) -> None:
        self.data = data
        self.source_path = source_path
        self.models_root = paths.normalize(models_root) if models_root else ""
        self.artifacts_by_id = {}
        for artifact in self.artifacts:
            artifact_id = str(artifact.get("id") or "")
            if artifact_id:
                self.artifacts_by_id[artifact_id] = artifact

    # -- raw access ------------------------------------------------------------------

    @property
    def schema_version(self) -> str:
        return str(self.data.get("schema_version") or "")

    @property
    def artifacts(self) -> list:
        """The original, unmodified artifact list from the manifest."""
        value = self.data.get("artifacts")
        return list(value) if isinstance(value, list) else []

    @property
    def profiles(self) -> dict:
        value = self.data.get("profiles")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def preflight_rules(self) -> dict:
        value = self.data.get("preflight_rules")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def models_root_hint(self) -> str:
        return str(self.data.get("models_root_hint") or "")

    @property
    def worker_python_hint(self) -> str:
        return str(self.data.get("worker_python_hint") or "")

    @property
    def manual_download_only(self) -> bool:
        return bool(self.data.get("manual_download_only", True))

    def profile_display_name(self, profile: str) -> str:
        entry = self.profiles.get(profile)
        if isinstance(entry, dict) and entry.get("display_name"):
            return str(entry["display_name"])
        return profile

    def profile_description(self, profile: str) -> str:
        entry = self.profiles.get(profile)
        if isinstance(entry, dict):
            return str(entry.get("description") or "")
        return ""

    def rules_for(self, profile: str) -> dict:
        """Merge manifest rules with the built-in defaults for ``profile``."""
        merged = dict(PROFILE_DEFAULT_RULES.get(profile) or {})
        merged.update(self.preflight_rules.get(profile) or {})
        return merged

    def artifact(self, artifact_id: str) -> dict | None:
        return self.artifacts_by_id.get(artifact_id)

    def artifact_status(self, artifact_id: str, required: bool = False) -> ArtifactStatus | None:
        artifact = self.artifact(artifact_id)
        if artifact is None:
            return None
        return ArtifactStatus(artifact, self.models_root, required)

    def statuses_for_profile(self, profile: str) -> list:
        """All artifacts relevant to ``profile``, required ones first."""
        rules = self.rules_for(profile)
        required_ids = list(rules.get("required_artifact_ids") or [])
        any_of_ids = list(rules.get("required_any_of") or [])
        optional_ids = list(rules.get("optional_artifact_ids") or [])

        seen = set()
        result = []
        for artifact_id in required_ids + any_of_ids + optional_ids:
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            status = self.artifact_status(artifact_id, required=artifact_id in required_ids)
            if status is not None:
                result.append(status)
        # Artifacts that name the profile but are not listed in the rules.
        for artifact in self.artifacts:
            artifact_id = str(artifact.get("id") or "")
            if artifact_id in seen or profile not in (artifact.get("profiles") or ()):
                continue
            seen.add(artifact_id)
            result.append(ArtifactStatus(artifact, self.models_root, required=False))
        return result


class PreflightReport(object):
    """Outcome of :func:`check_profile_requirements` (guide section 4.1)."""

    __slots__ = (
        "ok",
        "profile",
        "effective_profile",
        "missing_required",
        "missing_optional",
        "warnings",
        "download_urls",
        "statuses",
        "models_root",
        "manifest_path",
        "error",
    )

    def __init__(self, profile: str, effective_profile: str | None = None) -> None:
        self.ok = False
        self.profile = profile
        self.effective_profile = effective_profile or profile
        self.missing_required = []
        self.missing_optional = []
        self.warnings = []
        self.download_urls = []
        self.statuses = []
        self.models_root = ""
        self.manifest_path = ""
        self.error = None  # type: errors.MocapError | None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "profile": self.profile,
            "effective_profile": self.effective_profile,
            "missing_required": [s.to_dict() for s in self.missing_required],
            "missing_optional": [s.to_dict() for s in self.missing_optional],
            "warnings": list(self.warnings),
            "download_urls": list(self.download_urls),
            "models_root": self.models_root,
            "manifest_path": self.manifest_path,
            "error": self.error.to_dict() if self.error else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "PreflightReport(profile={0!r}, ok={1}, missing_required={2})".format(
            self.profile, self.ok, len(self.missing_required)
        )


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def manifest_search_paths(models_root: str | None) -> list:
    """Candidate manifest paths in priority order."""
    candidates = []
    if models_root:
        candidates.append(os.path.join(paths.normalize(models_root), "manifest.json"))
    candidates.append(paths.bundled_manifest_path())
    return candidates


def load_manifest(models_root: str | None) -> ModelManifest:
    """Load ``models/manifest.json`` when present, otherwise ``manifest.example.json``.

    Raises :class:`errors.MocapError` with ``MANIFEST_INVALID`` when a manifest
    exists but cannot be parsed, and with ``MANIFEST_INVALID`` when no manifest
    can be found at all (the bundled example is expected to always exist).
    """
    tried = []
    for candidate in manifest_search_paths(models_root):
        tried.append(candidate)
        if not os.path.isfile(candidate):
            continue
        try:
            # utf-8-sig tolerates the BOM that Notepad and PowerShell add on Windows.
            with open(candidate, "r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            raise errors.MocapError(
                errors.MANIFEST_INVALID,
                "无法解析 manifest：{0}（{1}）".format(candidate, exc),
                details={"manifest_path": candidate},
            )
        if not isinstance(data, dict) or not isinstance(data.get("artifacts"), list):
            raise errors.MocapError(
                errors.MANIFEST_INVALID,
                "manifest 结构不合法（缺少 artifacts 列表）：{0}".format(candidate),
                details={"manifest_path": candidate},
            )
        return ModelManifest(data, candidate, models_root or "")

    raise errors.MocapError(
        errors.MANIFEST_INVALID,
        "找不到任何 manifest 文件。",
        details={"searched": tried},
    )


def check_profile_requirements(profile: str, models_root: str | None) -> PreflightReport:
    """Return missing required and optional artifacts for ``profile``.

    Never raises: manifest problems and a missing models root are reported inside
    the returned :class:`PreflightReport` so the UI can always render something.
    """
    profile = str(profile or "")
    report = PreflightReport(profile)

    if profile not in ALL_PROFILES:
        report.error = errors.MocapError(
            errors.JOB_SCHEMA_INVALID,
            "未知 profile：{0}".format(profile),
            details={"known_profiles": list(ALL_PROFILES)},
        )
        report.warnings.append(report.error.user_text())
        return report

    try:
        manifest = load_manifest(models_root)
    except errors.MocapError as exc:
        report.error = exc
        report.warnings.append(exc.user_text())
        return report

    report.manifest_path = manifest.source_path
    report.models_root = manifest.models_root

    root_ok = bool(models_root) and os.path.isdir(paths.normalize(models_root))
    if not root_ok:
        report.error = errors.MocapError(
            errors.MODELS_ROOT_NOT_FOUND,
            "Models Root 未配置或不存在：{0}".format(models_root or "<empty>"),
            details={"models_root": models_root or "", "hint": manifest.models_root_hint},
        )
        report.warnings.append(report.error.user_text())

    effective_profile, fallback_warnings = _resolve_effective_profile(manifest, profile)
    report.effective_profile = effective_profile
    report.warnings.extend(fallback_warnings)

    report.statuses = manifest.statuses_for_profile(effective_profile)
    rules = manifest.rules_for(effective_profile)
    required_ids = list(rules.get("required_artifact_ids") or [])
    any_of_ids = list(rules.get("required_any_of") or [])
    optional_ids = [
        artifact_id
        for artifact_id in (rules.get("optional_artifact_ids") or [])
        if artifact_id not in required_ids and artifact_id not in any_of_ids
    ]

    for artifact_id in required_ids:
        status = manifest.artifact_status(artifact_id, required=True)
        if status is None:
            report.warnings.append(
                "manifest 中找不到 artifact id：{0}（profile {1}）".format(artifact_id, effective_profile)
            )
            continue
        if not status.exists:
            report.missing_required.append(status)

    if any_of_ids:
        any_statuses = [manifest.artifact_status(i, required=True) for i in any_of_ids]
        any_statuses = [s for s in any_statuses if s is not None]
        if any_statuses and not any(s.exists for s in any_statuses):
            report.missing_required.extend(any_statuses)
            report.warnings.append(
                "profile {0} 至少需要以下之一：{1}".format(
                    effective_profile, "、".join(s.display_name for s in any_statuses)
                )
            )

    for artifact_id in optional_ids:
        status = manifest.artifact_status(artifact_id, required=False)
        if status is not None and not status.exists:
            report.missing_optional.append(status)

    for status in report.missing_required + report.missing_optional:
        if status.download_url and status.download_url not in report.download_urls:
            report.download_urls.append(status.download_url)

    for status in report.missing_required:
        report.warnings.append(
            "[{0}] 缺少必需文件：{1} -> {2}".format(
                status.error_code, status.display_name, status.relative_path
            )
        )
    for status in report.missing_optional:
        detail = "；降级：{0}".format(status.fallback) if status.fallback else ""
        report.warnings.append(
            "[OPTIONAL] 缺少可选文件：{0} -> {1}{2}".format(
                status.display_name, status.relative_path, detail
            )
        )

    report.ok = root_ok and not report.missing_required and report.error is None
    return report


def _resolve_effective_profile(manifest: ModelManifest, profile: str) -> tuple:
    """Apply manifest fallbacks, e.g. ``quality_plus`` -> ``quality``."""
    warnings = []
    current = profile
    for _ in range(4):  # guard against manifest fallback cycles
        rules = manifest.rules_for(current)
        fallback = rules.get("fallback_profile")
        required_ids = list(rules.get("required_artifact_ids") or [])
        if not fallback or not required_ids:
            break
        missing = [
            artifact_id
            for artifact_id in required_ids
            if not _artifact_exists(manifest, artifact_id)
        ]
        if not missing:
            break
        warnings.append(
            "profile {0} 缺少必需文件（{1}），自动回退到 {2}。".format(
                current, "、".join(missing), fallback
            )
        )
        current = str(fallback)
    return current, warnings


def _artifact_exists(manifest: ModelManifest, artifact_id: str) -> bool:
    status = manifest.artifact_status(artifact_id)
    return bool(status is not None and status.exists)


def first_missing_error(report: PreflightReport) -> errors.MocapError | None:
    """Return the most relevant error for a failed report, or ``None``."""
    if report.error is not None:
        return report.error
    if not report.missing_required:
        return None
    status = report.missing_required[0]
    return errors.MocapError(
        status.error_code,
        "Required model is missing: {0}.".format(status.display_name),
        details={
            "artifact_id": status.id,
            "relative_path": status.relative_path,
            "download_url": status.download_url,
            "profile": report.effective_profile,
        },
    )


def format_missing_links(report: PreflightReport) -> str:
    """Clipboard text for missing artifacts, per guide section 6.3."""
    lines = []
    if report.missing_required:
        lines.append("Missing required models:")
        for status in report.missing_required:
            lines.append("- {0}".format(status.display_name))
            lines.append("  Save to: models/{0}".format(status.relative_path))
            lines.append("  URL: {0}".format(status.download_url))
    if report.missing_optional:
        if lines:
            lines.append("")
        lines.append("Missing optional models:")
        for status in report.missing_optional:
            lines.append("- {0}".format(status.display_name))
            lines.append("  Save to: models/{0}".format(status.relative_path))
            lines.append("  URL: {0}".format(status.download_url))
            if status.fallback:
                lines.append("  Fallback: {0}".format(status.fallback))
    if not lines:
        lines.append("No missing models for profile {0}.".format(report.effective_profile))
    return "\n".join(lines)


def resolve_artifact_path(models_root: str, artifact_id: str, manifest: ModelManifest | None = None):
    """Absolute path of ``artifact_id``, or ``None`` when unknown/absent.

    Used by the worker to locate weights and configs without duplicating the
    manifest layout.
    """
    active = manifest if manifest is not None else load_manifest(models_root)
    status = active.artifact_status(artifact_id)
    if status is None:
        return None
    return status.absolute_path or None


def require_artifact(models_root: str, artifact_id: str, manifest: ModelManifest | None = None) -> str:
    """Absolute path of ``artifact_id``; raises ``MODEL_MISSING``/``CONFIG_MISSING``."""
    active = manifest if manifest is not None else load_manifest(models_root)
    status = active.artifact_status(artifact_id)
    if status is None:
        raise errors.MocapError(
            errors.MANIFEST_INVALID,
            "manifest 中不存在 artifact id：{0}".format(artifact_id),
            details={"artifact_id": artifact_id},
        )
    if not status.exists:
        raise errors.MocapError(
            status.error_code,
            "Required model is missing: {0}.".format(status.display_name),
            details={
                "artifact_id": status.id,
                "relative_path": status.relative_path,
                "download_url": status.download_url,
            },
        )
    return status.absolute_path
