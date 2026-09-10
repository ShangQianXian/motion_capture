# Blender Motion Capture 插件 v0.1 详细开发文档

> 当前实现目标：Blender 4.5.0，Python 3.10.0 双 worker 环境。依赖版本、安装与诊断以 [INSTALL.md](INSTALL.md) 和 `requirements/` 为准；下文早期 4.0 兼容记录仅作历史参考。

本文档是给后续 AI 或工程师执行开发用的施工手册。目标不是解释创意，而是把 Blender 动捕插件 v0.1 从空仓库开发到可验收版本所需的模块、接口、流程、算法、错误处理和测试标准全部固定下来。

总体技术方案见 `docs/TECHNICAL_DESIGN.md`。模型清单、下载地址、profile 和降级规则以 `models/manifest.example.json` 为准；不要在本文档里维护第二套模型数据源。

## 1. 项目目标

### 1.1 产品目标

开发一款 Blender 4.0+ 插件，允许用户选择视频或图片，从画面中捕捉单人角色动作，并把动作应用到 Rigify Human 生成的角色骨骼上，最终生成可编辑的 Blender Action。

### 1.2 默认运行环境

- 操作系统：Windows 10/11。
- Blender：4.0 或更高版本。
- GPU：NVIDIA RTX4060，8GB 显存。
- CPU：Intel i7-14650HX 或近似性能。
- 推理方式：本地离线推理。
- Python 依赖策略：深度学习依赖安装在外部 worker 虚拟环境，不安装到 Blender 内置 Python。

### 1.3 v0.1 必须完成

- 插件可以被 Blender 安装、启用、禁用。
- 插件偏好设置可以配置 `Worker Python`、`Models Root`、`Max VRAM GB`、`Default Profile`。
- 插件可以读取 `models/manifest.example.json` 并检查模型缺失情况。
- 插件提供模型环境检查 UI，并能展示缺失模型的文件名和下载链接。
- 插件可以启动外部 worker，并读取 JSONL 进度。
- worker 支持 `mock` 模式，即使没有真实模型也能输出合法 `mocap_result.json`。
- preview profile 支持 MediaPipe Pose，能处理单图和视频。
- quality profile 支持 RTMDet-m、RTMPose-m、MotionBERT 的接口封装。
- 插件能导入 worker 输出结果，并对 Rigify Human 生成 rig 写入动画关键帧。
- 插件能烘焙 Action，并清理临时对象。

### 1.4 v0.1 明确不做

- 不做多人动捕。检测到多人时默认使用面积最大且置信度最高的人。
- 不做实时摄像头直播。
- 不做云端推理。
- 不做面部表情捕捉。
- 不自动联网下载模型。
- 不直接写 Rigify 的 DEF 变形骨作为主输出。

## 2. 开发路线图

每个阶段都必须可独立验收。不要等所有 AI 模型接好之后才验证 Blender 插件主体。

### Phase 0：仓库结构与插件基础注册

目标：Blender 能发现插件，启用插件时不报错。

必须创建：

```text
__init__.py
addon/
  __init__.py
  preferences.py
  properties.py
  panels.py
  operators.py
core/
  __init__.py
  model_manifest.py
  job_schema.py
  errors.py
backend_worker/
  __init__.py
  cli.py
blender/
  __init__.py
  rigify_adapter.py
  action_baker.py
tests/
  unit/
  fixtures/
```

验收：

- Blender 启用插件不报错。
- `bpy.ops.preferences.addon_enable(module="motion_capture")` 可成功执行。
- 禁用插件后所有注册类、Scene 属性和临时 handler 被清理。

### Phase 1：偏好设置、manifest 与预检 UI

目标：用户还没下载模型时，也能知道缺什么、去哪下载、怎么配置。

必须实现：

- `MotionCapturePreferences`
- `MotionCaptureSceneProperties`
- `load_manifest(models_root)`
- `check_profile_requirements(profile, models_root)`
- `MOCAP_OT_preflight`
- `MOCAP_OT_copy_missing_model_links`
- `MOCAP_OT_open_models_dir`
- `MOCAP_OT_test_worker_python`
- `MOCAP_OT_test_cuda`

验收：

- `models_root` 为空时提示 `MODELS_ROOT_NOT_FOUND`。
- `models_root` 存在但模型缺失时提示 `MODEL_MISSING` 或 `CONFIG_MISSING`。
- 缺失报告必须包含 `id`、`display_name`、`relative_path`、`download_url`、`required`。
- `preview` 只强制要求 MediaPipe Pose Full。
- `quality` 强制要求 RTMDet-m、RTMPose-m、MotionBERT 及对应 config。

### Phase 2：worker CLI 与 mock 流程

目标：没有模型也能跑通 Blender 到 worker 再回到 Blender 的完整数据流。

必须实现：

- `backend_worker/cli.py`
- `--job PATH`
- `--check-env`
- `--check-cuda`
- `--mock`
- JSONL progress 输出。
- `mocap_result.json` 输出。

验收：

- Blender operator 能启动 worker。
- worker 每行 stdout 都是合法 JSON。
- mock 模式输出 30 帧简单挥手或走路动作。
- 取消任务时 worker 退出，Blender UI 不假死。

### Phase 3：MediaPipe preview 链路

目标：实现快速预览和低依赖兜底。

必须实现：

- `backend_worker/pose_mediapipe.py`
- 图片输入单帧推理。
- 视频输入逐帧或按目标 FPS 抽帧推理。
- MediaPipe landmark 到标准骨架的映射。
- 可选 MediaPipe Hand Landmarker。

验收：

- 单图能输出 1 帧 result。
- 短视频能输出多帧 result。
- 身体模型存在、手部模型缺失时，身体流程继续可用。
- 低置信度关节写入 `confidence`，不要伪装成高精度数据。

### Phase 4：Rigify mock result 重定向与 Action 烘焙

目标：先用 mock result 验证 Blender 侧动画输出，不依赖真实 AI。

必须实现：

- `detect_rigify_human(armature)`
- `build_rigify_mapping(armature)`
- `retarget_to_rigify(result, armature, options)`
- `bake_action(armature, frame_start, frame_end)`
- 临时源骨架或临时空对象管理。

验收：

- 标准 Rigify Human 生成 rig 能被检测。
- mock result 可以生成 Action。
- Action 中存在关键帧。
- 烘焙后动画可编辑。
- 临时对象和约束可清理。

### Phase 5：High Quality 链路

目标：接入高精度身体动捕主链路。

必须实现：

- `backend_worker/pose2d_mmpose.py`
- `backend_worker/pose3d_motionbert.py`
- RTMDet-m 人体检测。
- RTMPose-m 2D body keypoints。
- MotionBERT 3D pose lifting。
- `quality_plus` 使用 RTMPose-x，OOM 或模型缺失时回退 `quality`。

验收：

- 1080p 单人短视频能完成处理。
- CUDA 不可用时给出 `CUDA_UNAVAILABLE`。
- CUDA OOM 时给出 `CUDA_OOM`，并建议降低分辨率或回退 profile。
- 结果坐标能被 Rigify 流程消费。

### Phase 6：平滑、足底锁定与错误恢复

目标：提高动画可用性，降低抖动和脚滑。

必须实现：

- 位置低通滤波。
- 四元数 slerp 平滑。
- 低置信度帧插值。
- ankle/toe 接触检测。
- 足底锁定修正。
- per-frame warning 聚合。

验收：

- 走路视频脚滑明显减少。
- 遮挡帧不会出现大幅爆转。
- warning 不阻断结果导入。
- 错误日志能告诉用户下一步怎么处理。

### Phase 7：打包、验收和用户文档

目标：形成可分发 v0.1 包。

必须完成：

- 插件 zip 打包说明。
- worker 环境安装说明。
- 模型手动下载说明。
- 示例素材验收清单。
- 第三方许可证检查清单。

## 3. 推荐目录与模块职责

### 3.1 Blender 插件入口

`__init__.py` 必须包含 `bl_info`：

```python
bl_info = {
    "name": "Motion Capture for Rigify",
    "author": "Project Team",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Mocap",
    "description": "Capture body motion from video or image and retarget it to Rigify rigs.",
    "category": "Animation",
}
```

注册要求：

- 所有 `bpy.types.Operator`、`Panel`、`PropertyGroup`、`AddonPreferences` 必须集中注册。
- `register()` 中添加 `bpy.types.Scene.mocap_props`。
- `unregister()` 中删除 `bpy.types.Scene.mocap_props`。
- 禁止在 import 阶段启动 worker、读取模型或导入 torch。

### 3.2 `addon/preferences.py`

定义 `MotionCapturePreferences(bpy.types.AddonPreferences)`。

字段：

- `worker_python: StringProperty(subtype="FILE_PATH")`
- `models_root: StringProperty(subtype="DIR_PATH")`
- `max_vram_gb: FloatProperty(default=7.0, min=1.0, max=24.0)`
- `default_profile: EnumProperty(items=preview/quality/quality_plus/fallback_cpu)`
- `allow_auto_download: BoolProperty(default=False)`

UI 要求：

- 显示路径输入框。
- 显示环境测试按钮。
- 明确写出 v0.1 不自动下载模型。

### 3.3 `addon/properties.py`

定义 `MotionCaptureSceneProperties(bpy.types.PropertyGroup)`。

字段：

- `source_media: StringProperty(subtype="FILE_PATH")`
- `source_type: EnumProperty(items=image/video/auto)`
- `target_armature: PointerProperty(type=bpy.types.Object)`
- `capture_profile: EnumProperty(items=preview/quality/quality_plus/fallback_cpu)`
- `frame_start: IntProperty(default=1, min=1)`
- `frame_end: IntProperty(default=0, min=0)`，0 表示自动。
- `target_fps: IntProperty(default=30, min=1, max=120)`
- `include_hands: BoolProperty(default=True)`
- `smoothing_strength: FloatProperty(default=0.65, min=0.0, max=1.0)`
- `foot_lock_strength: FloatProperty(default=0.7, min=0.0, max=1.0)`
- `root_motion: EnumProperty(items=world/in_place)`
- `job_status: StringProperty(default="idle")`
- `last_result_path: StringProperty(subtype="FILE_PATH")`
- `last_error: StringProperty(default="")`

### 3.4 `addon/panels.py`

创建 View3D 侧边栏 Tab：`Mocap`。

面板：

- `MOCAP_PT_environment`
- `MOCAP_PT_capture`
- `MOCAP_PT_rigify`
- `MOCAP_PT_logs`

UI 原则：

- 环境未通过时，捕捉按钮禁用或显示警告。
- `preview` 可以在缺少 MMPose 模型时运行。
- `quality` 缺少必需模型时不允许启动。
- 日志面板显示最近 20 条 progress/warning/error。

### 3.5 `addon/operators.py`

Operator 命名：

- `mocap.preflight`
- `mocap.copy_missing_model_links`
- `mocap.open_models_dir`
- `mocap.test_worker_python`
- `mocap.test_cuda`
- `mocap.test_preview_model`
- `mocap.test_quality_model`
- `mocap.run_capture`
- `mocap.cancel_capture`
- `mocap.import_result`
- `mocap.apply_to_rigify`
- `mocap.bake_action`
- `mocap.clear_temp_data`

要求：

- operator 不直接导入 torch、mmpose、mediapipe。
- 长任务只通过外部 worker 执行。
- Blender 主线程只做 UI 更新、进度轮询和最终数据应用。
- 所有错误都通过 `self.report({"ERROR"}, message)` 和 `props.last_error` 返回给用户。

## 4. 稳定接口

实现者必须优先稳定这些接口。后续模块互相调用只依赖这些函数，不直接读取对方内部结构。

### 4.1 模型 manifest

```python
def load_manifest(models_root: str) -> "ModelManifest":
    """Load models/manifest.json when present, otherwise models/manifest.example.json."""
```

行为：

- 先找 `<models_root>/manifest.json`。
- 不存在时找仓库内 `models/manifest.example.json`。
- JSON 解析失败时返回 `RESULT_SCHEMA_INVALID` 风格错误，但错误码用 `MANIFEST_INVALID`。
- 返回对象必须保留原始 artifact 列表。

```python
def check_profile_requirements(profile: str, models_root: str) -> "PreflightReport":
    """Return missing required and optional artifacts for a profile."""
```

`PreflightReport` 字段：

- `ok: bool`
- `profile: str`
- `effective_profile: str`
- `missing_required: list[ArtifactStatus]`
- `missing_optional: list[ArtifactStatus]`
- `warnings: list[str]`
- `download_urls: list[str]`

### 4.2 Job 构建与 worker 调度

```python
def build_job(scene_props, preferences) -> dict:
    """Build a worker job from Blender scene properties and add-on preferences."""
```

必须校验：

- `source_media` 存在。
- `target_fps` 合法。
- `models_root` 存在。
- `profile` 可用。
- 输出目录可创建。

```python
def run_worker(worker_python: str, job_path: str) -> "WorkerProcess":
    """Start backend_worker.cli with the provided job path."""
```

要求：

- 使用 `subprocess.Popen`。
- `stdout` 按 JSONL 读取进度。
- `stderr` 写入日志文件。
- 不阻塞 Blender UI。

### 4.3 Worker progress 解析

```python
def parse_progress_line(line: str) -> "ProgressEvent":
    """Parse one JSONL progress line from worker stdout."""
```

非法 JSON 行：

- 不让插件崩溃。
- 记录为 warning。
- 保留原始文本，便于排查。

### 4.4 结果导入与重定向

```python
def load_mocap_result(path: str) -> "MocapResult":
    """Validate and load mocap_result.json."""
```

```python
def detect_rigify_human(armature) -> "RigifyDetection":
    """Detect whether the selected armature is a Rigify generated human rig."""
```

```python
def build_rigify_mapping(armature) -> "RigifyMapping":
    """Build source standard skeleton to Rigify control bone mapping."""
```

```python
def retarget_to_rigify(result, armature, options) -> "bpy.types.Action":
    """Insert keyframes on Rigify control bones from mocap result."""
```

```python
def bake_action(armature, frame_start: int, frame_end: int) -> "bpy.types.Action":
    """Bake visual transforms into an editable Blender Action."""
```

## 5. Worker 协议

### 5.1 Job JSON

worker 输入必须是一个 JSON 文件：

```json
{
  "job_id": "uuid-string",
  "mode": "capture",
  "input": {
    "path": "D:/path/source.mp4",
    "type": "video",
    "frame_start": 1,
    "frame_end": 0,
    "target_fps": 30
  },
  "model": {
    "profile": "quality",
    "device": "cuda:0",
    "max_vram_gb": 7.0,
    "models_root": "D:/blender_addons/motion_capture/models"
  },
  "options": {
    "single_person": true,
    "include_hands": true,
    "smoothing_strength": 0.65,
    "foot_lock_strength": 0.7,
    "root_motion": "world",
    "scale_mode": "target_rig_height"
  },
  "output": {
    "dir": "D:/project/.mocap_jobs/uuid-string",
    "result_filename": "mocap_result.json"
  }
}
```

字段规则：

- `frame_end = 0` 表示处理到媒体末尾。
- `input.type = auto` 时 worker 根据扩展名判断。
- `profile` 必须是 manifest 中定义的 profile。
- `device` 可为 `cuda:0` 或 `cpu`。
- `single_person` 在 v0.1 必须为 true。

### 5.2 Progress JSONL

worker stdout 每一行必须是独立 JSON：

```json
{"event":"started","job_id":"uuid-string","message":"Worker started"}
{"event":"loading_model","profile":"quality","model_id":"rtmpose_m_body","progress":0.15}
{"event":"processing_frame","frame":42,"total_frames":300,"progress":0.42}
{"event":"warning","code":"LOW_CONFIDENCE","message":"Left wrist confidence is low","frame":88}
{"event":"completed","result_path":"D:/project/.mocap_jobs/uuid-string/mocap_result.json","progress":1.0}
```

允许的 `event`：

- `started`
- `loading_model`
- `processing_frame`
- `warning`
- `completed`
- `failed`
- `cancelled`

`failed` 示例：

```json
{
  "event": "failed",
  "error": {
    "code": "CUDA_OOM",
    "message": "CUDA ran out of memory while running RTMPose-x.",
    "suggestion": "Switch to quality profile or reduce input resolution.",
    "recoverable": true
  }
}
```

### 5.3 Result JSON

worker 输出 `mocap_result.json`：

```json
{
  "version": "0.1",
  "source": {
    "path": "D:/path/source.mp4",
    "type": "video",
    "profile": "quality"
  },
  "fps": 30,
  "coordinate_system": "blender_world",
  "unit": "meter",
  "skeleton": "mocap_standard_v0",
  "frames": [
    {
      "frame": 1,
      "time": 0.0,
      "body3d": {
        "pelvis": [0.0, 1.0, 0.0],
        "spine": [0.0, 1.18, 0.0],
        "chest": [0.0, 1.38, 0.0],
        "neck": [0.0, 1.55, 0.0],
        "head": [0.0, 1.72, 0.0],
        "shoulder.L": [-0.18, 1.48, 0.0],
        "elbow.L": [-0.45, 1.35, 0.0],
        "wrist.L": [-0.68, 1.22, 0.0],
        "shoulder.R": [0.18, 1.48, 0.0],
        "elbow.R": [0.45, 1.35, 0.0],
        "wrist.R": [0.68, 1.22, 0.0],
        "hip.L": [-0.12, 0.98, 0.0],
        "knee.L": [-0.12, 0.55, 0.03],
        "ankle.L": [-0.12, 0.08, 0.0],
        "toe.L": [-0.12, 0.02, -0.18],
        "hip.R": [0.12, 0.98, 0.0],
        "knee.R": [0.12, 0.55, 0.03],
        "ankle.R": [0.12, 0.08, 0.0],
        "toe.R": [0.12, 0.02, -0.18]
      },
      "hands3d": {},
      "confidence": {
        "body_mean": 0.93,
        "wrist.L": 0.82,
        "wrist.R": 0.91
      },
      "contacts": {
        "foot.L": true,
        "foot.R": false
      }
    }
  ],
  "warnings": []
}
```

### 5.4 Error JSON

错误对象统一结构：

```json
{
  "code": "MODEL_MISSING",
  "message": "Required model is missing: RTMPose-m Body.",
  "suggestion": "Download the model from manifest.example.json and place it under models/openmmlab/body2d.",
  "recoverable": true,
  "details": {
    "artifact_id": "rtmpose_m_body",
    "relative_path": "openmmlab/body2d/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
  }
}
```

必须支持错误码：

- `WORKER_PYTHON_NOT_FOUND`
- `MODELS_ROOT_NOT_FOUND`
- `MANIFEST_INVALID`
- `MODEL_MISSING`
- `CONFIG_MISSING`
- `CUDA_UNAVAILABLE`
- `CUDA_OOM`
- `MEDIA_OPEN_FAILED`
- `NO_PERSON_DETECTED`
- `RIGIFY_NOT_FOUND`
- `RIGIFY_MAPPING_FAILED`
- `RESULT_SCHEMA_INVALID`

## 6. 模型与环境预检

### 6.1 manifest 读取策略

读取顺序：

1. `<models_root>/manifest.json`
2. `<addon_root>/models/manifest.example.json`

如果用户复制了 example 并修改为 `manifest.json`，以用户版本为准。

### 6.2 profile 规则

- `preview`：必需 `mediapipe_pose_full`。
- `quality`：必需 `rtmdet_m_person`、`config_rtmdet_m_person`、`rtmpose_m_body`、`config_rtmpose_m_body`、`motionbert_body3d`、`config_motionbert_body3d`。
- `quality_plus`：必需 RTMPose-x 及 config；缺失时回退 `quality`。
- `fallback_cpu`：优先 MediaPipe Lite，其次 MediaPipe Full。
- `hand_enhanced`：全部可选；缺失时只禁用增强手部。

### 6.3 预检 UI 输出

每条模型状态显示：

- 状态：OK、Missing Required、Missing Optional。
- 展示名。
- 相对路径。
- 下载 URL。
- 降级说明。

复制下载链接时，按以下格式写入剪贴板：

```text
Missing required models:
- RTMPose-m Body
  Save to: models/openmmlab/body2d/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
  URL: https://download.openmmlab.com/...
```

## 7. Worker 技术方案

### 7.1 CLI

`backend_worker/cli.py` 参数：

```text
python -m backend_worker.cli --job D:/path/job.json
python -m backend_worker.cli --job D:/path/job.json --mock
python -m backend_worker.cli --check-env
python -m backend_worker.cli --check-cuda
```

实现要求：

- 使用 `argparse`。
- 所有 progress 输出走 stdout JSONL。
- 调试日志走 stderr 或 job 输出目录中的 `worker.log`。
- 失败时 stdout 输出 `failed` event，进程返回非 0。
- 取消时输出 `cancelled` event，进程返回 130。

### 7.2 Mock 模式

mock 模式必须生成合法 result，供 Blender 侧开发使用。

数据要求：

- 默认 30 FPS。
- 默认 60 帧。
- pelvis 做轻微上下移动。
- 双脚交替 contact。
- 一只手臂做简单挥动。
- 所有标准骨骼点都存在。

### 7.3 媒体解码

`backend_worker/media_decode.py`：

- 图片：`png`、`jpg`、`jpeg`、`webp`。
- 视频：`mp4`、`mov`、`avi`、`mkv`。
- 默认使用 OpenCV。
- 解码失败返回 `MEDIA_OPEN_FAILED`。
- 视频按 `target_fps` 抽帧，不要无条件处理全部原始帧率。

### 7.4 MediaPipe preview

`backend_worker/pose_mediapipe.py`：

- 使用 `pose_landmarker_full.task` 作为默认模型。
- 用户选择 heavy 时使用 heavy。
- CPU fallback 可使用 lite。
- MediaPipe 输出需要转换到标准骨架。
- MediaPipe 世界坐标不可靠时，仍要输出 confidence 和 warning。

### 7.5 MMPose high quality

`backend_worker/pose2d_mmpose.py`：

- 使用 RTMDet-m 检测人体 bbox。
- 如果多人，选择最大面积且置信度最高 bbox。
- 使用 RTMPose-m 推理 body keypoints。
- `quality_plus` 使用 RTMPose-x。
- OOM 时捕捉异常并返回 `CUDA_OOM`，由调用层降级。

`backend_worker/pose3d_motionbert.py`：

- 输入连续 2D keypoint 序列。
- MotionBERT 使用 243-frame window。
- 短视频不足 window 时使用边界帧 padding。
- 输出每帧 3D 标准骨架。

## 8. 标准骨架与坐标系

### 8.1 标准骨架 v0

身体关键点：

```text
pelvis
spine
chest
neck
head
shoulder.L elbow.L wrist.L
shoulder.R elbow.R wrist.R
hip.L knee.L ankle.L heel.L toe.L
hip.R knee.R ankle.R heel.R toe.R
```

手部关键点可选：

```text
thumb.01.L thumb.02.L thumb.03.L thumb.tip.L
index.01.L index.02.L index.03.L index.tip.L
middle.01.L middle.02.L middle.03.L middle.tip.L
ring.01.L ring.02.L ring.03.L ring.tip.L
pinky.01.L pinky.02.L pinky.03.L pinky.tip.L
```

右手同理使用 `.R`。

### 8.2 坐标系

worker 输出统一为 Blender 世界坐标：

- X：角色左右。
- Y：角色前后。
- Z：向上。
- 单位：米。

如果上游模型坐标不同，必须在 worker 内转换，不要把模型坐标泄漏到 Blender retarget 层。

### 8.3 置信度

置信度范围 `0.0` 到 `1.0`。

- `>= 0.7`：正常使用。
- `0.4 - 0.7`：使用但参与平滑。
- `< 0.4`：优先插值，必要时写 warning。

## 9. 平滑与足底锁定

### 9.1 位置平滑

使用简单低通滤波作为 v0.1 默认方案：

```text
smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t - 1]
```

`alpha` 由 UI 的 `smoothing_strength` 转换：

```text
alpha = 1.0 - clamp(smoothing_strength, 0.0, 0.95)
```

### 9.2 旋转平滑

骨骼旋转必须用 quaternion slerp，不要对欧拉角直接平均。

规则：

- 连续帧四元数点积为负时翻转其中一个，避免走长弧。
- 低置信度帧降低当前帧权重。
- 最终写入 Rigify 前统一转换为目标骨局部空间。

### 9.3 低置信度插值

规则：

- 连续缺失不超过 10 帧：前后有效帧线性插值位置，slerp 插值旋转。
- 连续缺失超过 10 帧：保留上一可信姿态，并写 warning。
- 首尾缺失：使用最近有效帧。

### 9.4 足底锁定

接触判断：

- foot height 接近局部地面。
- ankle/toe 水平速度低。
- foot confidence >= 0.5。

修正策略：

- 接触开始时记录 foot world position。
- 接触期间保持 foot world position 稳定。
- 只微调脚部 IK 目标和 pelvis 高度。
- 不改写上半身姿态。

## 10. Rigify 适配方案

### 10.1 支持对象

v0.1 只支持 Rigify Human 生成后的 rig。不要对 meta-rig 执行动作写入。

检测逻辑：

- 对象类型必须是 `ARMATURE`。
- pose bones 中存在常见 Rigify 控制骨。
- 优先检查 `rig_id` 或 Rigify 生成属性。
- 检测失败返回 `RIGIFY_NOT_FOUND` 或 `RIGIFY_MAPPING_FAILED`。

### 10.2 默认控制骨映射

实际 Rigify 名称可能随 rig 类型略有差异，实现时要允许别名查找。

推荐映射：

```text
pelvis/root -> root 或 torso
spine -> spine_fk
chest -> chest
neck -> neck
head -> head
upper arm.L -> upper_arm_fk.L
forearm.L -> forearm_fk.L
hand.L -> hand_fk.L
thigh.L -> thigh_fk.L
shin.L -> shin_fk.L
foot.L -> foot_fk.L
toe.L -> toe.L 或 toe_fk.L
```

右侧 `.R` 同理。

### 10.3 重定向数学

每条骨链流程：

1. 从标准骨架取父子关节点，计算源方向向量。
2. 从 Rigify rest pose 取目标骨默认方向。
3. 计算 `delta_rotation = target_rest_direction.rotation_difference(source_direction)`。
4. 转换到 pose bone local space。
5. 设置 pose bone rotation mode 为 `QUATERNION`。
6. 插入 rotation keyframe。

注意：

- 目标角色比例由 Rigify rest pose 决定。
- 不直接缩放角色骨骼长度。
- root motion 只写 root 或 torso 控制骨位置。
- in-place 模式去除水平位移，只保留上下起伏和旋转。

### 10.4 Action 烘焙

要求：

- 新建 Action 名称：`Mocap_<source_name>_<profile>`。
- 所有关键帧插入到目标 rig。
- 临时约束烘焙为普通关键帧。
- 烘焙后删除临时源骨架、空对象和约束。
- 保留最终 Action 给用户继续编辑。

## 11. UI 用户流程

### 11.1 首次配置流程

1. 安装并启用插件。
2. 打开 Add-ons 偏好设置。
3. 填写 `Worker Python`。
4. 填写 `Models Root`。
5. 点击 `测试 Worker Python`。
6. 点击 `测试 CUDA`。
7. 点击 `检查模型环境`。
8. 缺模型时点击 `复制缺失模型下载链接`。
9. 下载模型后重新检查。

### 11.2 捕捉流程

1. 在 View3D 侧边栏打开 `Mocap`。
2. 选择视频或图片。
3. 选择 Rigify Armature。
4. 选择 profile。
5. 设置帧范围和目标 FPS。
6. 点击 `Run Capture`。
7. 等待 worker 输出完成。
8. 点击 `Import Result`。
9. 点击 `Apply to Rigify`。
10. 检查动画。
11. 点击 `Bake Action`。

### 11.3 UI 状态

`job_status` 枚举：

- `idle`
- `checking`
- `ready`
- `running`
- `completed`
- `failed`
- `cancelled`

按钮启用规则：

- `Run Capture`：环境通过且 source media 存在。
- `Cancel Capture`：只在 `running` 时启用。
- `Import Result`：`last_result_path` 存在时启用。
- `Apply to Rigify`：result 已导入且目标 rig 有效。
- `Bake Action`：已应用临时动画时启用。

## 12. 测试策略

### 12.1 无 Blender 的纯 Python 测试

必须能在普通 Python 中跑：

- manifest JSON 可解析。
- `check_profile_requirements` 对缺失模型返回正确报告。
- job schema 校验。
- progress JSONL 解析。
- result schema 校验。
- mock worker 输出合法 result。
- smoothing 插值逻辑。

推荐命令：

```powershell
python -m pytest tests/unit
```

### 12.2 Blender background 测试

有 Blender 可执行文件时：

```powershell
blender --background --factory-startup --python tests/blender/test_enable_addon.py
blender --background --factory-startup --python tests/blender/test_mock_retarget.py
```

验收：

- 插件启用不报错。
- 属性注册成功。
- mock result 能导入。
- 如果测试环境有 Rigify，标准 human rig 可以完成映射。

### 12.3 手工验收素材

素材：

- T-pose 或 A-pose 单图。
- 正面走路视频。
- 抬手挥手视频。
- 侧身短暂遮挡视频。
- 30 秒 1080p 单人视频。

验收标准：

- 没有左右反转。
- 骨盆高度稳定。
- 肩肘腕方向正确。
- 膝盖和肘部不明显反折。
- 脚滑比未锁定状态明显降低。
- Action 可编辑。

## 13. 打包与发布

### 13.1 插件包

打包内容：

- 插件 Python 源码。
- `docs/`
- `models/manifest.example.json`
- 示例配置文件。

不打包：

- `.venv/`
- 下载的 `.pth`、`.task` 模型权重。
- 用户生成的 `.mocap_jobs/`。
- 临时日志和缓存。

### 13.2 版本兼容

最低支持 Blender 4.0。每次发布前至少验证：

- Blender 4.0
- 当前 Blender LTS
- 最新稳定 Blender

### 13.3 许可证

发布前检查：

- PyTorch
- MediaPipe
- MMPose
- MMEngine
- MMCV
- MMDetection
- MotionBERT
- VideoPose3D
- InterNet
- 所有模型权重许可证

第三方权重默认由用户自行下载，不随插件包分发。

## 14. 实现顺序检查表

按此顺序开发，减少后期返工：

- 创建插件骨架和注册入口。
- 创建偏好设置和场景属性。
- 接入 manifest 读取和 profile 预检。
- 做环境检查面板。
- 做 worker CLI mock 模式。
- 做 Blender 到 worker 的异步任务调度。
- 做 result schema 和 mock result 导入。
- 做 Rigify 检测和骨骼映射。
- 做 mock Action 写入和烘焙。
- 接入 MediaPipe preview。
- 接入 MMPose high quality。
- 接入平滑、低置信度插值和足底锁定。
- 完成错误处理和日志面板。
- 完成打包和验收测试。

## 15. 给后续 AI 的开发约束

- 不要在 Blender import 阶段做重活。
- 不要在 Blender 主线程跑深度学习推理。
- 不要把模型权重提交进仓库。
- 不要跳过 mock 流程；真实模型接入前必须先完成 mock 闭环。
- 不要直接写 DEF 骨作为主要动画输出。
- 不要新增第二份模型下载清单；读取 `models/manifest.example.json`。
- 不要把 warning 当 fatal error，除非结果 schema 已无法保证。
- 每个阶段完成后都要补对应测试或验收脚本。
