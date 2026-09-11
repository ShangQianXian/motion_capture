# 打包与发布

> 当前实现目标：Blender 4.5.0，Python 3.10.0 双 worker 环境。依赖版本、安装与诊断以 [INSTALL.md](INSTALL.md) 和 `requirements/` 为准；下文早期 4.0 兼容记录仅作历史参考。

对应 `docs/DEVELOPMENT_GUIDE.md` §13。

## 1. 打包

```powershell
cd D:\blender_addons\motion_capture
.\tools\build_zip.ps1
# 输出：dist\motion_capture-0.2.0.zip
```

脚本行为：

- 版本号直接从 `__init__.py` 的 `bl_info["version"]` 读取，**不会**与代码脱节。
- zip 内唯一的顶层目录是 `motion_capture/`，这正是 Blender「Install from Disk」期望的结构。
- 打包前校验 `__init__.py` 与 `models/manifest.example.json` 存在，否则拒绝出包。
- 输出会报告跳过的权重文件数量。
- 根目录使用发布内容白名单，不把临时文件或未识别的本地文件打入安装包。

## 2. 包含 / 排除

**包含**

- 插件 Python 源码（`__init__.py`、`addon/`、`core/`、`blender/`、`backend_worker/`）
- `docs/`
- `models/manifest.example.json`
- `tools/`（安装、测试、打包、环境引导脚本）
- `requirements/`（两个环境的核心约束和完整版本锁）
- `tests/`（不含大体积媒体素材）
- `README.md`、`.gitignore`

**排除**

| 类别 | 规则 |
|---|---|
| 虚拟环境 | `.venv/`、`.venv-preview/`、`venv/` |
| 版本控制与编辑器 | `.git/`、`.idea/`、`.vscode/`、`.agents/`、`.codex/` |
| 缓存与下载状态 | `__pycache__/`、`.pytest_cache/`、`.cache/`、`.downloads/` |
| 下载的配置 | `models/**/*.py`、配置 `_sources/` 目录 |
| 用户模型配置 | `models/manifest.json` |
| 模型权重 | `*.pth`、`*.task`、`*.onnx`、`*.pt`、`*.ckpt` |
| 任务数据 | `.mocap_jobs/`、`*.log` |
| 构建产物 | `dist/`、`*.zip` |
| 测试媒体 | `*.mp4`、`*.mov`、`*.avi`、`*.mkv` |

> 第三方模型权重默认由用户自行下载，**不随插件包分发**，除非许可证明确允许（见 `docs/LICENSES.md`）。

## 3. 发布前检查表

```powershell
.\tools\run_tests.ps1
```

必须全部通过：

- [ ] `python -m unittest discover -s tests/unit -t .` 全绿（338 个用例，可用零依赖 Python 3.10.0 执行）
- [ ] `blender --background --factory-startup --python tests/blender/test_enable_addon.py` 退出码 0
- [ ] `blender --background --factory-startup --python tests/blender/test_mock_retarget.py` 退出码 0
- [ ] 上述两个 Blender 测试在目标版本 **Blender 4.5.0** 上通过
- [ ] 两个 worker 的版本、`pip check`、适配器契约和 Quality 的 CUDA 算子检查通过
- [ ] 从生成的 zip 全新安装一次，确认能启用、面板出现、mock 流程跑通
- [ ] `docs/ACCEPTANCE.md` 的手工验收清单已走完（真实模型链路）
- [ ] `docs/LICENSES.md` 的许可证核查已完成
- [ ] zip 内不含任何 `.pth` / `.task` 文件
- [ ] `README.md` 的「当前状态」表与实际一致

## 4. 版本兼容

当前最低版本及验收目标为 **Blender 4.5.0**。本次实测：

| Blender | Python | 状态 |
|---|---|---|
| 4.5.0 | 3.11.11 | 通过（54 + 63 + 16 + 36 项检查） |

兼容性注意点：

- 共用代码必须兼容外部 worker 的 **Python 3.10.0**；Blender 使用其内置 Python。
- 不要使用 `action.slots`：该 API 仅 4.4+ 存在，且 4.5.0 上没有 `slots.new_for_id`。
  统一用 `animation_data.action = act` + `keyframe_insert` + `action.fcurves`。
- `bone.collections`（骨骼集合）在 4.0 与 4.5 上都可用。
- `UILayout.progress()` 仅 4.2+ 有；4.0 上回退成文本标签。
- `addon/` 下的模块**不要**加 `from __future__ import annotations`，否则 `register_class` 会失败。

## 5. 升级版本号

改 `__init__.py` 里的 `bl_info["version"]` 一处即可，打包脚本与 zip 名会自动跟随。
同步更新 `README.md` 的「当前状态」表与 `docs/ACCEPTANCE.md` 的记录表头。
