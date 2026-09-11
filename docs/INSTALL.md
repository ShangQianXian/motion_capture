# 安装与模型下载

面向使用者。目标环境：Windows 10/11、Blender 4.5.0、NVIDIA RTX4060 8GB。

模型清单的唯一数据源是 `models/manifest.example.json`；本文档只解释操作步骤，不重复维护第二份清单。

## 1. 安装插件

### 1.1 使用发布包

1. 用 `tools\build_zip.ps1` 生成（或从发布页下载）`motion_capture-0.2.0.zip`。
2. Blender → `Edit > Preferences > Add-ons > Install from Disk`，选择该 zip。
3. 在列表中勾选 **Motion Capture for Rigify**。
4. 侧边栏出现 `Mocap` 标签（View3D 中按 `N` 打开侧边栏）。

### 1.2 开发者模式（目录联接）

```powershell
cd D:\blender_addons\motion_capture
.\tools\dev_install.ps1                 # 默认链接 Blender 4.5
.\tools\dev_install.ps1 -BlenderVersion 4.5
.\tools\dev_install.ps1 -Remove         # 卸载链接
```

联接指向仓库根目录，改代码后重启 Blender（或禁用再启用插件）即可生效。Windows 的目录
联接不需要管理员权限。

## 2. 只想先跑通流程？不需要任何模型

Mock 生成不加载任何模型；v0.2 的素材对照预览需要 Preview worker 中的 OpenCV：

1. 偏好设置 → `Worker Python` 填任意 Python 3.10+ 解释器，例如
   `E:\SoftWare\Python\Python3.10.0\python.exe`。
2. `Models Root` 填仓库的 `models` 目录，例如 `D:\blender_addons\motion_capture\models`。
3. 配置 Preview worker（见下一节），选择一张图片，展开 `Mocap > 素材与生成 > 开发工具` → 点 **运行 Mock 捕捉**。
4. 打开 **对照预览** → 在 **应用** 面板选中 Rigify rig → **确认并应用**。

这条路径可以完整验证 Blender ↔ worker 的数据流与 Rigify 输出，再去装真实依赖。

## 3. 创建 Python 3.10.0 双环境

当前验收平台是 Windows x64、Blender 4.5.0、RTX 4060 8GB。两个 worker 均使用外部 Python **3.10.0**；Blender 4.5.0 自带的 Python 3.11.11 保持独立。

```powershell
.\tools\bootstrap_worker_env.ps1 -PythonExe 'E:\SoftWare\Python\Python3.10.0\python.exe' -DryRun
.\tools\bootstrap_worker_env.ps1 -PythonExe 'E:\SoftWare\Python\Python3.10.0\python.exe'
```

脚本默认创建两个环境，安装核心版本及所有已锁定的间接依赖，并执行环境检查。任一步失败都会返回非零退出码；已有环境必须是 Python 3.10.0 x64，否则停止并提示使用干净环境。不会自动删除已有环境。

| 用途 | 解释器 | 依赖基线 / 完整锁定文件 |
|---|---|---|
| Quality / Quality Plus、日常开发 | `.venv/Scripts/python.exe` | `requirements/quality.in` / `requirements/quality.txt` |
| Preview / CPU fallback | `.venv-preview/Scripts/python.exe` | `requirements/preview.in` / `requirements/preview.txt` |

核心版本从 `.in` 文件维护；`.txt` 文件记录通过检查的完整依赖。安装器优先使用 `.txt`。不要单独升级 torch、MMCV、NumPy 或 OpenCV；升级需要重新解析并通过对应环境和算子测试。

高质量环境使用 PyTorch 2.1.0+cu121、torchvision 0.16.0+cu121、MMCV 2.1.0、MMEngine 0.10.7、MMDetection 3.2.0、MMPose 1.3.2、NumPy 1.23.5、SciPy 1.10.1。预览环境使用 MediaPipe 0.10.21、NumPy 1.26.4、SciPy 1.11.4、protobuf 4.25.3、JAX/jaxlib 0.4.30。

MMCV 必须使用 cu121 / torch2.1.0 / cp310 / win_amd64 官方 wheel，不会自动源码编译。Chumpy 0.70 是纯 Python 老包，在预装 NumPy、SciPy、固定构建工具后单独安装。Python 3.10.0 自带旧 pip 的代理兼容问题由经过 SHA256 验证的官方 pip wheel 引导解决。

PyTorch wheel 自带 CUDA 运行库；`nvidia-smi` 的 CUDA 字段表示驱动支持能力，不是必须与 wheel 相同的本机 Toolkit 版本。此方案无需单独安装 CUDA Toolkit。保留当前驱动，通过实际 GPU 运算检查兼容性。

MediaPipe 使用 `opencv-contrib-python`，MMPose 使用 `opencv-python`；两者共享 `cv2`，因此必须分环境安装。预览环境没有 torch / OpenMMLab 是正常现象。

偏好设置：`Worker Python` 指向 `.venv/Scripts/python.exe`，新增的 `Preview Worker Python` 指向 `.venv-preview/Scripts/python.exe`。留空时自动查找插件根目录下对应环境；Preview 未配置且未找到独立环境时兼容旧的 Worker Python 路径。显式填写的路径优先。使用 ZIP 安装的插件时，填写开发目录中这两个解释器的绝对路径。

```powershell
.\.venv\Scripts\python.exe -m backend_worker.cli --check-env --profile quality
.\.venv\Scripts\python.exe -m backend_worker.cli --check-cuda
.\.venv-preview\Scripts\python.exe -m backend_worker.cli --check-env --profile preview
```

检查输出为 JSONL（started + completed/failed），标准输出不承载普通日志。带 profile 的检查要求对应依赖版本、唯一 OpenCV 包、必需 API、CPU 算子和 `pip check` 全部通过；CUDA 检查执行矩阵乘法以及 torchvision/MMCV NMS。省略 profile 的 `--check-env` 仅列出环境信息，允许无依赖 mock 开发。

仅安装预览：`-Environment preview`；只安装高质量：`-Environment quality`。旧 `-CpuOnly` / `-SkipOpenMMLab` 参数均映射到 preview-only；`-CudaIndex` 只接受 cu121，避免产生与 MMCV 不匹配的环境。

### IDE 与终端

PyCharm：Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Existing，选择 `D:/blender_addons/motion_capture/.venv/Scripts/python.exe`。预览模块调试使用 `.venv-preview/Scripts/python.exe` 的运行配置。项目中的 `.vscode/settings.json` 为 VS Code 指定开发解释器。

终端直接使用 `.\.venv\Scripts\python.exe -m pip ...`，避免裸 `pip` 与 `python` 指向不同版本。无需更改系统 PATH 或移除 Python 3.12。

## 4. 下载模型

### 4.1 使用下载工具（推荐）

在项目根目录运行。工具只需 Python 3.10+ 标准库，默认使用本项目的 `.venv`；不需要额外安装下载库。

```powershell
cd D:\blender_addons\motion_capture

# 先查看高质量模式需要下载的文件，不联网、不写文件
.\tools\download_models.ps1 -Profile quality -DryRun

# 下载预览模型，以及基础手部模型
.\tools\download_models.ps1 -Profile preview -Artifact mediapipe_hand

# 下载高质量模式：检测器、2D 姿态、MotionBERT 及配套配置
.\tools\download_models.ps1 -Profile quality

# 下载高质量增强模式，同时准备 quality 的回退模型
.\tools\download_models.ps1 -Profile quality_plus
```

默认保存到项目 `models/`。可用 `-ModelsRoot 'D:\Mocap Models'` 指定其他目录，随后在 Blender 偏好设置的 **Models Root** 填写同一目录。工具自动创建所需目录；优先读取该目录的 `manifest.json`，否则使用项目的 `models/manifest.example.json`。指定独立清单可用 `-Manifest 'D:\my-manifest.json'`。

也可直接使用 Python：

```powershell
.\.venv\Scripts\python.exe tools/download_models.py --profile quality
.\.venv\Scripts\python.exe tools/download_models.py --profile preview --artifact mediapipe_hand
```

| 参数（PowerShell / Python） | 作用 |
|---|---|
| `-Profile` / `--profile` | preview、quality、quality_plus、fallback_cpu、hand_enhanced；PowerShell 用数组、Python 重复参数可组合 |
| `-Artifact` / `--artifact` | 按 manifest 模型 ID 追加下载；权重会自动带上它的 `config_id` |
| `-IncludeOptional` / `--include-optional` | 同时选择该模式全部可选模型，包括当前管线尚未使用的扩展模型 |
| `-All` / `--all` | 下载清单全部 19 项，含可选模型；不能和 profile / artifact 合用 |
| `-ConfigsOnly` / `--configs-only` | 只准备所选配置及其相对依赖，不下载权重 |
| `-DryRun` / `--dry-run` | 显示 ID、目标路径和下载地址，不联网、不写文件 |
| `-VerifyOnly` / `--verify-only` | 离线校验所选文件和配置依赖，不发出下载请求 |
| `-Force` / `--force` | 重新下载；单个文件验证成功后才替换旧文件 |
| `-Timeout 30 -Retries 3` / `--timeout 30 --retries 3` | 单次网络超时秒数与每文件重试次数 |
| `-PythonExe` | PowerShell 入口指定 Python 解释器路径 |

不指定选择参数时只选择 Preview 身体模型。`fallback_cpu` 优先复用已有 Lite / Full，否则下载 Lite；`hand_enhanced` 选择其整组可选模型。**下载了可选模型不代表当前捕捉管线已实现该增强功能。**

下载中按 Ctrl+C 可以中断；再次执行相同命令会重用 `.downloads/` 下的部分文件。服务器提供 ETag 或 Last-Modified 时，使用 Range / If-Range 续传；服务器不支持续传或文件发生变化时从头下载，避免拼接出损坏文件。正在运行的下载器会锁定该 Models Root，防止两个进程同时修改部分文件。

下载完成后检查长度、拒绝明显的 HTML/XML 错误页，并记录文件 SHA256。manifest 可为每个原始文件增加 `sha256`（64 位十六进制字符串）与 `size_bytes`（正整数），工具会校验预期值。**内置清单已记录核实的文件大小，以及本次从官方 v1.3.2 下载的配置文件 SHA256；权重尚无发布者提供的完整 SHA256，本地下载记录能检测后续文件变化，不能替代发布者校验值。** 已有文件无校验记录、大小不符或哈希变化时会报错；确认需要替换后使用 `-Force`。

```powershell
.\tools\download_models.ps1 -Profile quality -VerifyOnly
.\tools\download_models.ps1 -Profile quality -Force
```

OpenMMLab 配置来源固定为 MMPose **v1.3.2**。工具通过语法解析递归下载相对 `_base_` 依赖，保留目录结构，并在 manifest 指定位置生成配置入口；下载时不执行这些 Python 文件。`openmmlab/configs/_sources/` 是配置的一部分，**移动模型目录时必须一起保留**。`mmdet::` / `mmpose::` 配置依赖由已安装的 Quality 环境解析，例如 [官方 RTMDet 配置](https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py)。自定义配置必须使用静态字符串或列表形式的 `_base_`；动态继承会明确报错。

结果报告保存在 `<Models Root>/.downloads/last_report.json`。某项失败时保留已完成的文件，最终返回非零退出码；修复网络或清单后重跑即可。工具沿用系统代理及证书验证，不修改系统代理。模型来源与许可见 [LICENSES.md](LICENSES.md)。运行命令是用户主动发起下载，插件启用与捕捉过程不会自行下载模型。

### 4.2 手动下载

也可在 `Models Root` 下手工建立目录：

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

先在 Blender 里点 **检查模型环境**，再点 **复制缺失模型下载链接**，
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

> MMPose 权重必须同时有配套 `.py` config 和继承依赖。手动保存单个配置文件可能缺少相对 `_base_` 文件；建议用下载工具准备配置。

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
8. 缺模型 → 使用 `download_models.ps1 -Profile <模式>` 下载 → 再检查。
9. 点 **测试 Preview 模型** 验证 MediaPipe 链路。
10. 点 **测试 High Quality 模型** 验证 MMPose / MotionBERT 链路。

## 6. 捕捉流程

1. View3D 侧边栏（`N`）打开 `Mocap`。
2. **素材与生成** 面板选 **源文件**，查看缩略图、分辨率，以及视频的时长和帧率。
3. **应用** 面板选 **目标 Rig**（Rigify 生成的 rig，**不是** metarig）。
4. 选 **捕捉 Profile**。
5. 设置起始帧 / 结束帧（0 = 到素材末尾）与目标 FPS。
6. 点 **生成动捕预览**，等待进度完成；`ESC` 或 **取消捕捉** 可中止。
7. 结果自动加载，点 **打开对照预览**，核对原素材、二维识别点与三维动作。
8. 播放、逐帧或跳到问题帧；必要时即时调整镜像和倾斜。平滑和足底锁定修改后需要重新生成。
9. 核对后点 **确认并应用**，新 Action 从第 1 帧开始。保留原 Action，支持 Ctrl+Z。
10. 如有高级需求，可展开 **高级选项与烘焙**。完整控件与旧版结果说明见 [v0.2 指南](V0_2_REVIEW.md)。

### 走路动画整体前倾或后仰

结果加载后，在 **预览与校正** 点击 **按站立姿态校准**，核对三维预览后再 **确认并应用**。插件会在 **倾斜校正（X）** 中填入整段动作共用的角度；可以手动微调，0° 表示保留原姿态。加载另一份结果时角度重置为 0°。

估算假设人物在整段素材中总体直立，适合站立、走路。弯腰、躺卧等动作请保留 0° 或手动设置。校准使用数据副本，保留肢体摆动，重复应用不会累计旋转。

2026-09-08 之前产生的高质量结果可能存在脚趾高度错误。更新插件后重新捕捉，再校准、应用和烘焙；仅调整已有 Action 无法补回正确的脚部数据。

## 7. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `WORKER_PYTHON_NOT_FOUND` | `Worker Python` 未填或路径不存在。mock 模式下任意 Python 3.10+ 都可以 |
| `MODELS_ROOT_NOT_FOUND` | `Models Root` 未填或目录不存在 |
| `MODEL_MISSING` / `CONFIG_MISSING` | 按 **复制缺失模型下载链接** 给出的路径放好文件；MMPose 权重必须配套 config |
| `MANIFEST_INVALID` | `models/manifest.json` 不是合法 JSON；删掉它即可回退到内置 example |
| `CUDA_UNAVAILABLE` | worker 环境不是 CUDA 版 PyTorch，或驱动异常；也可改用 `preview` / `fallback_cpu` |
| `CUDA_OOM` | 换 `quality`、降低素材分辨率，或调小 `Max VRAM GB` |
| MotionBERT 报 `(1,17,3)` 与 `(243,17,3)` 广播错误 | 旧版插件未对齐 MMPose 的单帧占位目标与 MotionBERT 时间窗口。更新插件（包含 `backend_worker/pose3d_motionbert.py` 修复）并重新运行捕捉；无需重装环境或下载模型 |
| `MEDIA_OPEN_FAILED` | 文件损坏或格式不支持（图片 png/jpg/jpeg/webp，视频 mp4/mov/avi/mkv） |
| `RIGIFY_NOT_FOUND` | 选中的是 metarig 或不是 Armature；先执行 Rigify 的 Generate Rig |
| `RIGIFY_MAPPING_FAILED` | 目标 rig 不是标准 Rigify Human 生成的，缺少 `spine_fk` / `upper_arm_fk` 等控制骨 |
| 动作左右颠倒 | 勾选 **左右镜像 (Flip X)** 后重新 **应用到 Rigify** |
| 动画不驱动模型 | 确认 **切换四肢为 FK** 已勾选（Rigify 的 `IK_FK` 必须为 1.0） |
| 角色躺倒 | 结果是 Y-up 数据，日志里会有 `SUSPECT_COORDINATE_SYSTEM` 告警 |
| worker 崩溃 | 看任务目录下的 `worker.log`；路径在日志面板与错误详情里 |

### 下载工具验证记录（2026-09-07）

23 项下载器回归测试、全部 326 项单元测试通过；7 个官方配置及相对依赖完成实际下载、离线复验，并通过 MMEngine 配置加载。12 个权重地址通过 HTTP HEAD 可用性与长度检查。本次工具开发验证没有下载权重或运行真实模型推理。
