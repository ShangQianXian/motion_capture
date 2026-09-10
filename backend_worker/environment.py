"""Profile-specific dependency and CPU operator probes; no imports at load time."""
from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import struct
import subprocess
import sys

from ._core import errors, paths


def expected_versions(profile):
    """Read the same baseline used by the installer, without a second version matrix."""
    from pathlib import Path
    kind = "preview" if profile in ("preview", "fallback_cpu") else "quality"
    source = Path(paths.addon_root()) / "requirements" / (kind + ".in")
    result = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith(("#", "--")):
            name, version = line.strip().split("==", 1)
            result[name] = version
    return kind, result


def validate_environment(profile):
    kind, versions = expected_versions(profile)
    problems, modules = [], {}
    if sys.version_info[:3] != (3, 10, 0) or struct.calcsize("P") != 8:
        problems.append("此依赖锁要求 Python 3.10.0 64 位。")
    distributions = {d.metadata["Name"].lower().replace("_", "-"): d.version
                     for d in importlib.metadata.distributions() if d.metadata["Name"]}
    for name, expected in versions.items():
        actual = distributions.get(name.lower())
        modules[name] = {"available": actual is not None, "version": actual or "", "expected": expected}
        if actual != expected:
            problems.append("{0}: 期望 {1}，实际 {2}".format(name, expected, actual or "未安装"))
    cv_packages = sorted(name for name in distributions if name in (
        "opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"))
    if len(cv_packages) != 1:
        problems.append("每个环境必须且只能有一种 OpenCV 包；实际：{0}".format(cv_packages))
    imports = ["numpy", "scipy", "cv2"]
    imports += (["mediapipe.tasks.python.vision", "jax", "jaxlib"] if kind == "preview" else
                ["torch", "torchvision", "mmcv.ops", "mmdet.apis", "mmpose.apis", "chumpy", "xtcocotools.coco"])
    # Keep third-party banners out of the CLI's JSONL channel.
    with contextlib.redirect_stdout(sys.stderr):
        for name in imports:
            try:
                importlib.import_module(name)
            except Exception as exc:
                problems.append("{0} 导入失败：{1}: {2}".format(name, type(exc).__name__, exc))
        if not problems:
            try:
                import numpy as np
                import cv2
                image = np.zeros((8, 8, 3), dtype=np.uint8)
                ok, encoded = cv2.imencode(".png", image)
                assert ok and np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), image)
                if kind == "quality":
                    import torch
                    from torchvision.ops import nms as tv_nms
                    from mmcv.ops import nms as mmcv_nms
                    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
                    scores = torch.tensor([0.9, 0.8])
                    assert tv_nms(boxes, scores, 0.5).tolist() == [0]
                    assert mmcv_nms(boxes, scores, 0.5)[1].tolist() == [0]
                    assert np.array_equal(torch.from_numpy(np.ones(2)).numpy(), np.ones(2))
            except Exception as exc:
                problems.append("CPU 算子/图像解码测试失败：{0}: {1}".format(type(exc).__name__, exc))
    flags = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
    try:
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=60, **flags)
        if check.returncode:
            problems.append("pip check: " + (check.stdout + check.stderr).strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        problems.append("pip check 失败：{0}".format(exc))
    report = {"ok": not problems, "environment": kind, "modules": modules,
              "problems": problems, "opencv_packages": cv_packages}
    if problems:
        report["error"] = errors.MocapError(
            errors.DEPENDENCY_MISSING, "Worker 环境不完整或版本不兼容。",
            suggestion="使用 tools/bootstrap_worker_env.ps1 安装对应的锁定环境。",
            details={"environment": kind, "problems": problems},
        ).to_dict()
    return report
