# 依赖与代码修复报告

验证日期：2026-09-07。范围：Windows x64、Python 3.10.0 双 worker、Blender 4.5.0；按本次约定，没有下载模型权重或进行真实素材推理。

## 结果

本机两套环境均已安装并通过验证：未发现锁定依赖之间的版本冲突、已检查的二进制接口不兼容或 OpenCV 包覆盖问题。这个结论覆盖环境、CPU/CUDA 算子和无权重推理接口，不代表已经完成真实模型效果验收。

| 用途 | 本机解释器 | 完整版本锁 |
|---|---|---|
| Quality / Quality Plus、开发 | `D:/blender_addons/motion_capture/.venv/Scripts/python.exe` | [quality.txt](../requirements/quality.txt)，56 个包 |
| Preview / CPU fallback | `D:/blender_addons/motion_capture/.venv-preview/Scripts/python.exe` | [preview.txt](../requirements/preview.txt)，31 个包 |

两者均由 `E:/SoftWare/Python/Python3.10.0/python.exe` 创建，版本为 **3.10.0 / 64 位**。Blender 4.5.0 使用自身的 Python **3.11.11**，通过外部进程与 worker 通信。

Quality 核心版本保持已批准基线：PyTorch 2.1.0+cu121、torchvision 0.16.0+cu121、MMCV 2.1.0、MMEngine 0.10.7、MMDetection 3.2.0、MMPose 1.3.2。完整约束见 [quality.in](../requirements/quality.in) 和 [preview.in](../requirements/preview.in)。

在 RTX 4060 Laptop GPU / 8 GB 上，PyTorch 报告 CUDA 运行库 **12.1**。当前驱动可执行该组合；没有更换显卡驱动或额外安装 CUDA Toolkit。

## 已验证项目

| 检查 | 结果 |
|---|---|
| 两环境精确核心版本、Python 版本与位数 | 通过 |
| 两环境 `pip check` | 通过 |
| 每个环境仅一种 OpenCV；PNG 编解码 | 通过 |
| Quality NumPy ↔ torch 转换、torchvision/MMCV CPU NMS | 通过 |
| CUDA 矩阵乘法、torchvision CUDA NMS、MMCV CUDA NMS | 通过 |
| MMDetection / MMPose API、Chumpy、xtcocotools 导入 | 通过 |
| Preview MediaPipe Tasks API、JAX/jaxlib 导入 | 通过 |
| 按完整锁重复执行 Quality 与 Preview 安装器 | 通过，包版本保持不变 |
| 标准库单元测试 | 303 项通过 |
| Quality worker 适配器测试 | 8 项通过，含真实 MMPose API 的无权重契约测试 |
| Preview worker 适配器测试 | 7 项通过；1 项 MMPose 专属测试按设计跳过 |
| Blender 4.5.0 插件启停与注册 | 48 项通过 |
| Blender 4.5.0 mock 重定向、烘焙与数据保护 | 63 项通过 |

最终回归命令：

```powershell
.\tools\run_tests.ps1 -Blender 'E:\SoftWare\Blender\blender-4.5.0-windows-x64\blender.exe'
```

第三方库仍会发出弃用提示（如 matplotlib/pyparsing、pkg_resources）；它们没有导致本次检查失败。验证过程中曾因执行沙箱限制无法写入 YAPF 用户缓存，获准使用正常权限后，相关检查通过；没有修改第三方源码或关闭证书校验。

## 主要修复

- 安装器明确选择 Python 3.10.0，拒绝错误版本、32 位或不完整环境。固定构建工具与全部传递依赖，匹配 MMCV wheel，单独安装 Chumpy。安装失败、依赖冲突或算子失败均返回失败状态。
- Quality 与 Preview 分离，避免两种 OpenCV 发行包覆盖同一个 `cv2`。捕捉、模型自测、环境检查共用 profile 路由；预览任务记录 CPU 设备。
- 环境诊断区分预览与高质量依赖。CLI 兼容模块启动和直接脚本启动，保留 JSONL 输出，隔离父进程的 `PYTHONHOME` / `PYTHONPATH`。
- MotionBERT 输入改为 MMPose 要求的逐帧、逐人嵌套样本，补齐 bbox、track ID、gt_instances；按目标帧构造时间窗口，保留缺失帧位置，正确解释人与时间维度。异常输出直接报错，不再静默填零或重复调用失败接口。
- 修正 COCO → H36M 头部映射和检测器 pipeline 适配。Quality Plus 运行时降级后，结果记录实际 profile。
- 镜像操作使用副本，避免重复应用修改缓存；结果拒绝 NaN、无穷和布尔 FPS。
- 烘焙保留用户约束，仅处理目标 rig；恢复选择、模式、帧位置和所处理的可见性状态，只清理目标 rig 的插件临时约束。
- ZIP 排除双虚拟环境、缓存、IDE 配置、用户 manifest 和模型权重；包含版本锁与安装脚本。

## 本机开发配置

PyCharm 2024.1 已注册两个新解释器，本项目默认 SDK 为 `Python 3.10 (motion_capture)`，指向 `.venv`。预览调试可选择 `Python 3.10 (motion_capture preview)`。IDE 配置原件备份于项目 `.cache/ide-repair/`；原有 Python 3.12 SDK 保留。重新打开项目后由 IDE 加载这些配置。

本地 `.vscode/settings.json` 同样指向 `.venv/Scripts/python.exe`。IDE 配置属于本机文件，不进入 Git 或 ZIP。安装到 Blender 的另一目录后，应在插件偏好中填写上表的两个绝对解释器路径。

后续安装或维护请使用 [INSTALL.md](INSTALL.md) 中的脚本，不使用裸 `pip` 或单独升级核心包。系统 PATH 与 Python 3.12 安装保持原状。

## 代码质量评价与剩余验收

修复前约 **6.5/10**；完成本轮后，工程可靠性可评约 **7/10**。这是基于当前检查的主观评价：模块分层、外部 worker、延迟导入、结构化错误和 mock 测试是优点；本轮补上了依赖可复现性、真实接口契约和 Blender 用户数据保护。

仍属于开发版本。下一阶段最有价值的工作是用固定权重及完整配置文件验证真实推理，检查配置继承与 checkpoint 是否匹配，并用含遮挡、转身、快速运动的视频评估精度、时间连续性、显存和速度。单人选择、脚趾外推、扭转求解等现有限制见 README；本次没有把这些模型效果问题视为已解决。

## MotionBERT 真实推理修复（2026-09-07）

用户在 Quality / Quality Plus 捕捉中遇到 `(1,17,3)` 与 `(243,17,3)` 的 NumPy 广播错误。已用真实 MMPose 1.3.2 编码器复现：通用 `inference_pose_lifter_model` 为 243 帧输入创建单帧 `lifting_target`，而 `MotionBERTLabel.encode` 对它原地乘以 243 帧的缩放因子，导致失败。此前无权重契约测试使用空 pipeline，未覆盖这一步。

修复在插件的模型实例推理 pipeline 中，进入 `GenerateTarget` 前扩展占位目标及其可见性到输入时间长度。保留 243 帧窗口和中心帧输出，不改变预测数据、不修改第三方包或下载的配置文件，也无需调整依赖版本。

本轮验证：

- 新增真实编码器、数据打包、回归头解码测试，覆盖单帧视频片段及中间缺失帧；新增占位数组独立性与异常长度测试。修复前真实 pipeline 测试复现原报错，修复后通过。
- 使用用户报错视频和本机已下载的 RTMDet / RTMPose / MotionBERT 权重，在 CUDA 上完整执行 Quality 与 Quality Plus。两者均输出 192 帧、24 FPS，结果结构及坐标有限性校验通过；Quality Plus 未降级。24 FPS 为原视频帧率。
- 326 项单元测试通过；Quality worker 11 项通过；Preview worker 9 项通过、2 项 MMPose 专属测试跳过；Blender 4.5.0 注册 48 项、mock 重定向与烘焙 63 项通过；两环境检查与 CUDA 算子检查通过。
- 本机真实推理结果及日志保存在 `.cache/motionbert-capture-20260907-233808/`，回归摘要见 `.cache/motionbert-tests.log`。
- 已备份并同步 Blender 4.5 已安装目录中的 `backend_worker/pose3d_motionbert.py`，源文件与安装文件 SHA-256 一致；旧文件位于 `.cache/motionbert-installed-backup-20260907-234232/`。
- 使用已安装插件在 Blender 4.5.0 空白后台场景调用真实“导入结果”操作，两个模式均成功导入 192 帧、24 FPS；记录见上述结果目录中的 `blender-validation.json`。

结果仍会报告已有的 `HANDS_DISABLED` 提示：v0.1 高质量链路只捕捉身体，不提供手指捕捉。本轮证明该视频的完整推理链路可运行，尚未完成遮挡、转身、快速运动等素材的动作精度验收。

## 走路动画倾斜与脚部修复（2026-09-08）

已检查用户截图、原视频及最新任务结果。倾斜已经存在于 3D 关键点中：头部到骨盆的方向约前倾 13°，双脚到头部的整体方向约前倾 18°。默认未旋转的 Rigify 控制骨与这些源方向一致，因此轴交换或给 rig 对象随意添加旋转不能完整解决问题。

本轮处理了三个问题：

- MotionBERT 输出仍以骨盆为原点时，脚趾 Z 被 `max(0, ankle_z - 0.04)` 提前截断，导致落地平移后脚趾出现在骨盆高度。现改为先保留相对脚踝高度，再统一落地。
- 重定向误把含对象世界变换的矩阵直接赋给 `PoseBone.matrix`。现先把源方向转换到骨架对象空间，使用正确的局部静止矩阵，根位移也转换一次。旋转 30° 的对象在旧代码中产生约 60° 的方向偏差，修复后旋转、位移与均匀缩放测试中的偏差小于 0.1°。
- 增加用户主动触发的站立姿态校准及可编辑 X 轴角度。按整段动作中双脚到头部方向的中位数估算固定校正，保留帧间摆动、左右方向及原始数据；导入不同结果时重置角度。该估算依赖总体直立的假设，不是对单目相机真实外参的测量。

Quality 和 Quality Plus 已使用真实模型分别重跑全部 192 帧、24 FPS，再通过 Blender 实际的导入、校准和应用操作生成 Action。校正角分别约 −18.0°、−17.7°；校正后源身体整体俯仰中位数接近 0°，抽检控制骨与校正源方向的夹角小于 0.1°。每份 Action 有 18,244 个关键帧，完整示例保存于 `.cache/motionbert-capture-20260908-001544/walk_calibrated.blend`；该目录同时包含推理结果与 `animation-validation.json`。

回归验证：330 项单元测试通过、Quality worker 11 项通过、Preview worker 9 项通过（2 项按设计跳过）；Blender 注册 49 项、原有重定向/烘焙 63 项及新增坐标/校准 16 项检查通过。新增 Blender 测试的首次全量执行因测试自身重置场景时注销插件而中止；修正初始化顺序后单独执行通过，其余检查无需重复运行。

仍需按具体素材检查脚滑、遮挡和关节朝向；这次修复不代表单目模型恢复了真实深度或精确脚趾/手指数据。
