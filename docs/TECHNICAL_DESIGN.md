# Blender Motion Capture 插件 v0.1 技术文档

本文档描述一款 Blender 4.0+ 动作捕捉插件的第一版实现方案。插件面向本地离线使用，目标硬件为 NVIDIA RTX4060 8GB 显存与 i7-14650HX CPU，核心能力是从视频或图片中捕捉单人角色动作，并将结果重定向到 Blender Rigify 生成的人形骨骼。

面向实现者的详细开发流程、模块职责、接口协议、算法方案和验收步骤见 `docs/DEVELOPMENT_GUIDE.md`。

## 1. 目标与范围

### 1.1 第一版目标

- 支持导入视频或图片作为动捕源。
- 支持单人全身动作捕捉，图片输出单帧姿态，视频输出可编辑 Blender Action。
- 支持基础手部捕捉；手部模型缺失时只禁用手部，不影响身体动捕。
- 支持 Rigify Human 生成 rig 的动作重定向与关键帧烘焙。
- 支持本地离线推理，不默认自动下载模型。
- 支持用户在插件偏好设置中配置外部 worker Python 与模型目录。

### 1.2 第一版不做

- 不支持多人角色选择和多人同时动捕。
- 不做实时动捕直播，优先离线高精度处理。
- 不做面部表情捕捉。
- 不把 PyTorch、MMPose、CUDA 等 AI 依赖安装到 Blender 内置 Python。
- 不内置第三方模型权重，用户自行下载并配置。

## 2. 总体架构

插件由两部分组成：

- Blender 插件本体：负责 UI、偏好设置、任务调度、Rigify 检测、动作重定向和 Action 烘焙。
- 外部 Python worker：负责视频/图片解码、人体检测、2D 姿态估计、3D 姿态提升、手部捕捉、时序平滑和结果导出。

推荐目录结构：

```text
motion_capture/
  __init__.py
  addon/
    panels.py
    operators.py
    properties.py
    preferences.py
  core/
    job_schema.py
    model_manifest.py
    skeleton.py
    smoothing.py
    retarget_math.py
  blender/
    rigify_adapter.py
    action_baker.py
    viewport_preview.py
  backend_worker/
    cli.py
    media_decode.py
    pose2d_mmpose.py
    pose3d_motionbert.py
    pose_mediapipe.py
    export_result.py
  models/
    manifest.example.json
  tests/
    unit/
    blender/
  docs/
    TECHNICAL_DESIGN.md
```

### 2.1 运行流程

1. 用户在 Blender 面板选择图片或视频、目标 Rigify Armature、捕捉质量和输出选项。
2. 插件读取偏好设置中的 `Worker Python` 和 `Models Root`。
3. 插件执行预检：Blender 版本、Rigify 状态、目标骨骼、worker Python、CUDA、模型文件。
4. 插件生成 JSON job 文件，并通过 `subprocess.Popen` 启动外部 Python worker。
5. worker 解码媒体并执行姿态估计，持续输出 JSONL 进度。
6. worker 输出标准化 `mocap_result.json`。
7. 插件读取结果并将动作重定向到 Rigify 控制骨。
8. 插件烘焙为 Blender Action，清理临时对象和约束。

## 3. 模型安装与配置

模型权重不包含在插件包内。用户需要手动下载模型，并在 Blender 插件偏好设置中指定模型目录。

### 3.1 模型目录约定

建议使用以下目录：

```text
D:/blender_addons/motion_capture/models/
  manifest.example.json
  mediapipe/
    pose_landmarker_full.task
    pose_landmarker_heavy.task
    pose_landmarker_lite.task
    hand_landmarker.task
  openmmlab/
    configs/
      rtmdet_m_640-8xb32_coco-person.py
      rtmpose-m_8xb256-420e_body8-256x192.py
      rtmpose-x_8xb256-700e_coco-384x288.py
      motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py
      rtmdet_nano_320-8xb32_hand.py
      rtmpose-m_8xb256-210e_hand5-256x256.py
      internet_res50_4xb16-20e_interhand3d-256x256.py
    detectors/
      rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
      rtmdet_nano_8xb32-300e_hand-267f9c8f.pth
    body2d/
      rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
      rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth
    body3d/
      motionbert_ft_h36m-d80af323_20230531.pth
      videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth
    hand2d/
      rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
    hand3d/
      res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth
```

Blender 插件偏好设置中填写：

```text
Worker Python:
D:/blender_addons/motion_capture/.venv/Scripts/python.exe

Models Root:
D:/blender_addons/motion_capture/models
```

### 3.2 外部 Python 环境安装

不要把 PyTorch、MMPose、MMCV、MMDetection 安装进 Blender 内置 Python。推荐创建独立的 worker 虚拟环境：

```powershell
cd D:\blender_addons\motion_capture
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

安装 PyTorch 时，以 PyTorch 官网当前选择器为准，选择 Windows、Pip、CUDA 12.x。RTX4060 推荐优先使用 CUDA 12.x wheel。示例：

```powershell
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

然后安装推理依赖。安装顺序必须是先 PyTorch，再 MMCV/MMDetection/MMPose：

```powershell
.\.venv\Scripts\python -m pip install -U openmim
.\.venv\Scripts\python -m mim install "mmengine" "mmcv" "mmdet" "mmpose"
.\.venv\Scripts\python -m pip install opencv-python mediapipe numpy scipy tqdm
```

安装完成后，用户可用以下命令检查 CUDA：

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 3.3 模型下载清单

| 用途 | 模型 | 必需性 | 保存路径 | 下载地址 |
|---|---|---:|---|---|
| 快速身体预览 | MediaPipe Pose Full | 推荐 | `models/mediapipe/pose_landmarker_full.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task |
| 高质量预览 | MediaPipe Pose Heavy | 可选 | `models/mediapipe/pose_landmarker_heavy.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task |
| 低配兜底 | MediaPipe Pose Lite | 可选 | `models/mediapipe/pose_landmarker_lite.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task |
| 基础手部 | MediaPipe Hand | 推荐 | `models/mediapipe/hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task |
| 人体检测 | RTMDet-m Person | 高精度必需 | `models/openmmlab/detectors/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` | https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth |
| 2D 身体 | RTMPose-m Body | 高精度必需 | `models/openmmlab/body2d/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth` | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth |
| 2D 身体增强 | RTMPose-x Body | 可选 | `models/openmmlab/body2d/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth` | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth |
| 3D 身体 | MotionBERT | 高精度必需 | `models/openmmlab/body3d/motionbert_ft_h36m-d80af323_20230531.pth` | https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth |
| 3D 身体兜底 | VideoPose3D | 可选 | `models/openmmlab/body3d/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth` | https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth |
| 手部检测 | RTMDet-nano Hand | 手部增强可选 | `models/openmmlab/detectors/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth` | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth |
| 2D 手部 | RTMPose-m Hand5 | 手部增强可选 | `models/openmmlab/hand2d/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth` | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth |
| 3D 手部 | InterNet InterHand3D | 手部增强可选 | `models/openmmlab/hand3d/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth` | https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth |

### 3.4 MMPose 配置文件下载清单

MMPose 模型需要同时提供 config 文件。用户可以下载完整 MMPose 源码，也可以把以下配置文件单独保存到 `models/openmmlab/configs/`。

| 用途 | 保存路径 | 下载地址 |
|---|---|---|
| RTMDet-m Person | `models/openmmlab/configs/rtmdet_m_640-8xb32_coco-person.py` | https://raw.githubusercontent.com/open-mmlab/mmpose/main/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py |
| RTMPose-m Body | `models/openmmlab/configs/rtmpose-m_8xb256-420e_body8-256x192.py` | https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py |
| RTMPose-x Body | `models/openmmlab/configs/rtmpose-x_8xb256-700e_coco-384x288.py` | https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-x_8xb256-700e_coco-384x288.py |
| MotionBERT | `models/openmmlab/configs/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py` | https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py |
| RTMDet-nano Hand | `models/openmmlab/configs/rtmdet_nano_320-8xb32_hand.py` | https://raw.githubusercontent.com/open-mmlab/mmpose/main/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py |
| RTMPose-m Hand5 | `models/openmmlab/configs/rtmpose-m_8xb256-210e_hand5-256x256.py` | https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py |
| InterNet Hand3D | `models/openmmlab/configs/internet_res50_4xb16-20e_interhand3d-256x256.py` | https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/hand_3d_keypoint/internet/interhand3d/internet_res50_4xb16-20e_interhand3d-256x256.py |

### 3.5 用户安装步骤

1. 创建 `models/` 目录和子目录。
2. 按“模型下载清单”下载 `.task` 和 `.pth` 文件。
3. 按“MMPose 配置文件下载清单”下载 `.py` config 文件。
4. 打开 Blender，启用本插件。
5. 进入插件偏好设置，填写 `Worker Python` 和 `Models Root`。
6. 点击 `检查模型环境`。
7. 如果缺少模型，点击 `复制缺失模型下载链接`，下载后重新检查。
8. 点击 `测试 Worker Python`。
9. 点击 `测试 CUDA`。
10. 先用 `测试 Preview 模型` 验证 MediaPipe 链路，再用 `测试 High Quality 模型` 验证 MMPose/MotionBERT 链路。

## 4. 模型 profile 与预检规则

### 4.1 profile

- `preview`：使用 MediaPipe Pose Full，可选 MediaPipe Hand。适合快速预览和低依赖测试。
- `quality`：使用 RTMDet-m Person、RTMPose-m Body、MotionBERT。适合最终动画生成。
- `quality_plus`：使用 RTMPose-x Body 替代 RTMPose-m Body。精度更高，速度更慢，RTX4060 8GB 上可能需要降低分辨率。
- `fallback_cpu`：优先使用 MediaPipe Lite 或降低分辨率后的 RTMPose-m。只保证基础可用。
- `hand_enhanced`：使用 RTMDet-nano Hand、RTMPose-m Hand5、InterNet InterHand3D。第一版作为实验增强。

### 4.2 预检规则

- `preview` 模式只要求 `models/mediapipe/pose_landmarker_full.task`。
- `quality` 模式要求 RTMDet-m、RTMPose-m、MotionBERT 及对应 config。
- MediaPipe Hand 缺失时禁用手部捕捉，但身体捕捉继续可用。
- RTMPose-x 缺失时，`quality_plus` 自动回退到 `quality`。
- VideoPose3D 缺失时不影响 MotionBERT 主链路。
- MMPose Hand/InterNet 缺失时不影响身体捕捉。
- Worker Python 配错时，Blender 插件必须保持可启动，只在执行 worker 相关功能时报告错误。
- CUDA 不可用时，允许用户切换 `preview` 或 `fallback_cpu`。

## 5. 插件 UI 与用户操作

插件偏好设置：

- `Worker Python`：外部 Python 解释器路径。
- `Models Root`：模型根目录。
- `Max VRAM GB`：默认 `7`，RTX4060 8GB 预留一部分给系统和 Blender。
- `Default Profile`：默认 `quality`。
- `Allow Auto Download`：第一版默认关闭。

插件面板按钮：

- `检查模型环境`
- `打开模型目录`
- `复制缺失模型下载链接`
- `测试 Worker Python`
- `测试 CUDA`
- `测试 Preview 模型`
- `测试 High Quality 模型`

执行面板：

- `Source Media`：选择图片或视频。
- `Target Rig`：选择 Rigify Armature。
- `Capture Profile`：选择 `preview`、`quality`、`quality_plus` 或 `fallback_cpu`。
- `Root Motion`：选择保留世界位移或固定原地。
- `Smoothing Strength`：平滑强度。
- `Foot Lock Strength`：足底锁定强度。
- `Run Capture`：启动捕捉。
- `Apply to Rigify`：应用到 Rigify。
- `Bake Action`：烘焙为 Action。

## 6. Worker 数据接口

### 6.1 Job 请求

```json
{
  "job_id": "uuid",
  "input": {
    "path": "D:/path/video.mp4",
    "type": "video",
    "frame_start": 1,
    "frame_end": null,
    "target_fps": 30
  },
  "model": {
    "profile": "quality",
    "device": "cuda:0",
    "max_vram_gb": 7,
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
    "dir": "D:/project/.mocap_jobs/job_id"
  }
}
```

### 6.2 Worker 输出

```json
{
  "version": "0.1",
  "fps": 30,
  "frames": [
    {
      "frame": 1,
      "time": 0.0,
      "body3d": {
        "pelvis": [0.0, 1.0, 0.0],
        "spine": [0.0, 1.2, 0.0],
        "neck": [0.0, 1.55, 0.0],
        "head": [0.0, 1.72, 0.0]
      },
      "hands3d": {},
      "confidence": {
        "body_mean": 0.93,
        "left_foot": 0.88,
        "right_foot": 0.91
      },
      "contacts": {
        "left_foot": true,
        "right_foot": false
      }
    }
  ],
  "warnings": []
}
```

## 7. Rigify 适配

第一版面向 Rigify Human 生成后的 rig，不直接操作 meta-rig。

检测规则：

- 目标对象必须是 Armature。
- 优先检查 Rigify 生成 rig 的自定义属性和常见控制骨。
- 检测失败时提示用户选择 Rigify Human 生成 rig 或加载自定义映射。

重定向策略：

- 默认将动作写入 Rigify 控制骨，而不是 DEF 变形骨。
- 躯干、头、手臂、腿优先使用 FK 控制骨烘焙。
- 足底锁定阶段可临时使用 IK 脚控制骨修正接触帧。
- 以目标 Rigify rest pose 的骨长作为最终比例，不直接套用模型骨长。
- 所有临时约束在烘焙后清理。

## 8. 精度与性能策略

- 视频优先使用时序模型，避免逐帧抖动。
- 对低置信度关键点做插值和平滑。
- 对膝盖、肘部使用骨链方向约束，减少异常翻折。
- 对脚跟和脚尖做接触检测，接触帧进行足底锁定。
- RTX4060 8GB 默认输入长边限制在 960 到 1280 之间。
- 出现 CUDA OOM 时自动降低 batch、降低输入分辨率或从 `quality_plus` 回退到 `quality`。

## 9. 模型 manifest

仓库提供 `models/manifest.example.json`。实现时复制为 `models/manifest.json` 或直接读取 example 作为默认模板。

manifest 必须记录：

- 模型 ID。
- 展示名称。
- 作用。
- 文件相对路径。
- 下载 URL。
- 所属 profile。
- 是否必需。
- 依赖的 config 文件。
- 缺失时的降级策略。

插件预检只检查本地文件存在性，不自动下载。缺失时 UI 显示具体文件名和下载链接。

## 10. 测试计划

### 10.1 模型安装测试

- 新机器无模型：插件能列出所有缺失模型和下载地址。
- 用户手动下载后：`检查模型环境` 逐项显示通过。
- Worker Python 配错：显示明确错误，不影响 Blender 启动。
- CUDA 不可用：允许切换 CPU/preview 模式。

### 10.2 动捕链路测试

- MediaPipe preview：单图和视频帧都能输出身体关键点。
- High Quality：RTMDet-m + RTMPose-m + MotionBERT 能完成 1080p 单人短视频。
- RTMPose-x 缺失或 OOM：自动回退 RTMPose-m。
- 手部模型缺失：身体动捕继续可用。
- 视频中短暂遮挡：输出 warning，动画不发生明显爆转。

### 10.3 Rigify 测试

- 加载标准 Rigify Human rig 后能检测目标骨骼。
- mock mocap result 能生成 Action 并插入关键帧。
- FK 烘焙后动画可编辑。
- 足底锁定开启后，走路视频脚滑明显减少。
- 临时源骨架、约束和缓存对象可清理。

## 11. 许可证与发布注意事项

- 插件发布前必须复核 PyTorch、MMPose、MMCV、MMDetection、MediaPipe、MotionBERT、VideoPose3D、InterNet 及各模型权重许可证。
- 不要把第三方模型权重直接打包进插件，除非许可证明确允许。
- 文档中应保留模型来源链接和下载说明。
- 如果未来支持自动下载，需要加入用户确认、下载进度、校验和、失败重试和许可证提示。

## 12. 参考来源

- PyTorch Start Locally: https://docs.pytorch.org/get-started/locally/
- MMPose 安装文档: https://github.com/open-mmlab/mmpose/blob/main/docs/en/installation.md
- MMPose 推理文档: https://mmpose.readthedocs.io/en/latest/user_guides/inference.html
- MMPose 3D Human Pose Demo: https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/3d_human_pose_demo.md
- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- MediaPipe Hand Landmarker: https://developers.google.cn/edge/mediapipe/solutions/vision/hand_landmarker/index
- Blender Python API: https://docs.blender.org/api/current/
- Blender Rigify Manual: https://docs.blender.org/manual/en/latest/addons/rigging/rigify/
