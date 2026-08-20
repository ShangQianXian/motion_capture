# 安装与模型下载

面向使用者。目标环境：Windows 10/11、Blender 4.0+、NVIDIA RTX4060 8GB。

模型清单的唯一数据源是 `models/manifest.example.json`；本文档只解释操作步骤，不重复维护第二份清单。

## 1. 安装插件

### 1.1 使用发布包

1. 用 `tools\build_zip.ps1` 生成（或从发布页下载）`motion_capture-0.1.0.zip`。
2. Blender → `Edit > Preferences > Add-ons > Install from Disk`，选择该 zip。
3. 在列表中勾选 **Motion Capture for Rigify**。
4. 侧边栏出现 `Mocap` 标签（View3D 中按 `N` 打开侧边栏）。

### 1.2 开发者模式（目录联接）

```powershell
cd D:\blender_addons\motion_capture
.\tools\dev_install.ps1                 # 默认链接 Blender 4.0 与 4.5
.\tools\dev_install.ps1 -BlenderVersion 4.5
.\tools\dev_install.ps1 -Remove         # 卸载链接
```

联接指向仓库根目录，改代码后重启 Blender（或禁用再启用插件）即可生效。Windows 的目录
联接不需要管理员权限。

## 2. 只想先跑通流程？不需要任何模型

mock 模式不加载任何模型，也不需要 torch / mediapipe / opencv：

1. 偏好设置 → `Worker Python` 填任意 Python 3.10+ 解释器，例如
   `E:\SoftWare\Python\Python3.12.3\python.exe`。
2. `Models Root` 填仓库的 `models` 目录，例如 `D:\blender_addons\motion_capture\models`。
3. 侧边栏 `Mocap > 捕捉` → 点 **运行 Mock 捕捉**。
4. **导入结果** → 在 `Rigify 应用` 面板选中 Rigify rig → **应用到 Rigify** → **烘焙 Action**。

这条路径可以完整验证 Blender ↔ worker 的数据流与 Rigify 输出，再去装真实依赖。

## 3. 创建 worker 虚拟环境

深度学习依赖**绝不能**装进 Blender 内置 Python。

```powershell
cd D:\blender_addons\motion_capture
.\tools\bootstrap_worker_env.ps1 -DryRun     # 先看要执行什么
.\tools\bootstrap_worker_env.ps1             # 真正安装（数 GB，需联网）
```

脚本按 `docs/TECHNICAL_DESIGN.md` §3.2 的顺序执行：

1. `py -3.12 -m venv .venv`
2. 升级 pip
3. **先** 安装 PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
4. `pip install opencv-python mediapipe numpy scipy tqdm`
5. **后** 安装 OpenMMLab：`pip install -U openmim` + `mim install mmengine mmcv mmdet mmpose`

CUDA 版本以 PyTorch 官网选择器为准；换版本用 `-CudaIndex`，纯 CPU 用 `-CpuOnly`，
只做 preview 用 `-SkipOpenMMLab`。

装完把 `Worker Python` 指向：

```text
D:\blender_addons\motion_capture\.venv\Scripts\python.exe
```

自检：

```powershell
.\.venv\Scripts\python.exe -m backend_worker.cli --check-env
.\.venv\Scripts\python.exe -m backend_worker.cli --check-cuda
```

两条命令都只输出单行 JSON；缺依赖时给出结构化错误码，不会抛 traceback。

## 4. 下载模型

在 `Models Root` 下建立如下目录（子目录不存在时先创建）：

```text
models/
  mediapipe/
  openmmlab/
    configs/
    detectors/
    body2d/
    body3d/
    hand2d/
    hand3d/
```

**最省事的做法**：先在 Blender 里点 **检查模型环境**，再点 **复制缺失模型下载链接**，
剪贴板里就是每个缺失文件的展示名、保存路径和下载地址：

```text
Missing required models:
- RTMPose-m Body
  Save to: models/openmmlab/body2d/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
  URL: https://download.openmmlab.com/...
```

各 profile 的最小需求：

| Profile | 必需 | 可选 |
|---|---|---|
| `preview` | MediaPipe Pose Full | MediaPipe Hand / Pose Heavy / Pose Lite |
| `quality` | RTMDet-m Person、RTMPose-m Body、MotionBERT 及三个对应 config | MediaPipe Hand、VideoPose3D |
| `quality_plus` | 同上但用 RTMPose-x 及其 config | 缺失时自动回退 `quality` |
| `fallback_cpu` | MediaPipe Pose **Lite 或 Full 之一** | MediaPipe Hand |
| `hand_enhanced` | 无 | 全部可选，缺失只禁用增强手部 |

完整地址见 `docs/TECHNICAL_DESIGN.md` §3.3 / §3.4，或直接读 `models/manifest.example.json`。

> MMPose 权重必须同时有对应的 `.py` config 文件，否则预检报 `CONFIG_MISSING`。

### 自定义清单

想改路径或加模型：把 `models/manifest.example.json` 复制成
`models/manifest.json` 再改。插件优先读取 `<models_root>/manifest.json`，
其次才用插件内置的 example。`manifest.json` 已在 `.gitignore` 中。

## 5. 首次配置检查表

1. 安装并启用插件。
2. `Edit > Preferences > Add-ons > Motion Capture for Rigify` 展开。
3. 填写 `Worker Python`。
4. 填写 `Models Root`。
5. 点 **测试 Worker Python**。
6. 点 **测试 CUDA**（不可用时改用 `preview` 或 `fallback_cpu`）。
7. 点 **检查模型环境**。
8. 缺模型 → **复制缺失模型下载链接** → 下载 → 再检查。
9. 点 **测试 Preview 模型** 验证 MediaPipe 链路。
10. 点 **测试 High Quality 模型** 验证 MMPose / MotionBERT 链路。

## 6. 捕捉流程

1. View3D 侧边栏（`N`）打开 `Mocap`。
2. `捕捉` 面板选 **源文件**（图片或视频）。
3. `Rigify 应用` 面板选 **目标 Rig**（Rigify 生成的 rig，**不是** metarig）。
4. 选 **捕捉 Profile**。
5. 设置起始帧 / 结束帧（0 = 到素材末尾）与目标 FPS。
6. 点 **运行捕捉**，进度条与日志面板会实时更新；`ESC` 或 **取消捕捉** 可中止。
7. 点 **导入结果**。
8. 点 **应用到 Rigify**。
9. 检查动画，必要时勾选 **左右镜像 (Flip X)** 后重新应用。
10. 点 **烘焙 Action**。

## 7. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `WORKER_PYTHON_NOT_FOUND` | `Worker Python` 未填或路径不存在。mock 模式下任意 Python 3.10+ 都可以 |
| `MODELS_ROOT_NOT_FOUND` | `Models Root` 未填或目录不存在 |
| `MODEL_MISSING` / `CONFIG_MISSING` | 按 **复制缺失模型下载链接** 给出的路径放好文件；MMPose 权重必须配套 config |
| `MANIFEST_INVALID` | `models/manifest.json` 不是合法 JSON；删掉它即可回退到内置 example |
| `CUDA_UNAVAILABLE` | worker 环境不是 CUDA 版 PyTorch，或驱动异常；也可改用 `preview` / `fallback_cpu` |
| `CUDA_OOM` | 换 `quality`、降低素材分辨率，或调小 `Max VRAM GB` |
| `MEDIA_OPEN_FAILED` | 文件损坏或格式不支持（图片 png/jpg/jpeg/webp，视频 mp4/mov/avi/mkv） |
| `RIGIFY_NOT_FOUND` | 选中的是 metarig 或不是 Armature；先执行 Rigify 的 Generate Rig |
| `RIGIFY_MAPPING_FAILED` | 目标 rig 不是标准 Rigify Human 生成的，缺少 `spine_fk` / `upper_arm_fk` 等控制骨 |
| 动作左右颠倒 | 勾选 **左右镜像 (Flip X)** 后重新 **应用到 Rigify** |
| 动画不驱动模型 | 确认 **切换四肢为 FK** 已勾选（Rigify 的 `IK_FK` 必须为 1.0） |
| 角色躺倒 | 结果是 Y-up 数据，日志里会有 `SUSPECT_COORDINATE_SYSTEM` 告警 |
| worker 崩溃 | 看任务目录下的 `worker.log`；路径在日志面板与错误详情里 |
