# 第三方许可证核查清单

## v0.3 核查记录（2026-09-12）

本节是 v0.3 当前状态，下面历史清单的空白项不代表已经获得授权。安装包不含模型权重及下载的第三方 Python 配置。

| 组件／权重 | 已核实的范围 | v0.3 状态 |
|---|---|---|
| RTMPose-m WholeBody，`rtmpose_m_wholebody` | MMPose 代码／配置 Apache-2.0；权重来自官方模型目录，训练使用 COCO-WholeBody／AIC／COCO | 特定权重的商业授权未确认，不能标注“可商用” |
| RTMDet-m Person、RTMPose-m／x Body | OpenMMLab 代码采用 Apache-2.0；使用清单固定的官方权重 | 代码许可不代替具体权重及训练数据授权，商业状态未确认 |
| MotionBERT H36M | 原项目和 MMPose 实现的代码 Apache-2.0；实际使用 OpenMMLab H36M 微调权重 | H36M 数据及衍生权重需单独核实，商业状态未确认 |
| MediaPipe Pose Full／Heavy／Lite、Hand | 保留 Google 官方模型来源，运行时未捆绑 | 本次未取得针对这些具体权重的独立商业授权确认 |
| VideoPose3D、Hand5、InterNet、手检测器等清单可选权重 | 保留原来源；v0.3 未新增或启用这些三维／手部链路 | 保持待核查，不作可商用承诺 |

官方依据：[MMPose 许可证](https://github.com/open-mmlab/mmpose/blob/v1.3.2/LICENSE)、[MMDetection 许可证](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE)、[MotionBERT 许可证](https://github.com/Walter0807/MotionBERT/blob/main/LICENSE)、[WholeBody 模型目录](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose_coco-wholebody.yml)。

[COCO-WholeBody 官方使用条款](https://github.com/jin-s13/COCO-WholeBody#terms-of-use)明确说明数据集用于研究及非商业用途，商业使用其标注需联系授权。这里记录数据集条款，不推断它自动等同于所有衍生权重的许可；因此新档位保持“商业授权未确认”。未接入 GVHMR、WHAM，也未接受它们的模型许可。

新 WholeBody 权重校验值来自本机从固定官方 URL 下载的文件，用于版本固定及后续完整性校验，不是发布者签名。运行时、旧权重和插件自身许可证仍需由发布者完成商业发行核查；本地开发安装包不构成商业授权证明。

对应 `docs/DEVELOPMENT_GUIDE.md` §13.3 与 `docs/TECHNICAL_DESIGN.md` §11。

## 1. 基本原则

1. 插件包**不分发**任何第三方模型权重；用户按 `docs/INSTALL.md` 自行下载。
2. 插件包**不捆绑** PyTorch / MMPose / MediaPipe 等运行时；它们装在用户自建的
   worker 虚拟环境里。
3. 文档中保留每个模型的来源链接与下载说明（`models/manifest.example.json` 的
   `download_url` 字段即为出处）。
4. 下载由用户主动运行 `tools/download_models.ps1` 发起，命令选择的 profile / ID 表示本次下载范围；插件不会在后台触发下载。工具提供进度、失败重试、SHA256 记录及清单校验，并显示本文件的位置。发布者预期 SHA256 未提供时，会明确说明校验的限制。

## 2. 发布前必须复核的项目

下表在每次发布前逐项确认；本仓库不代替法律意见，务必以各项目官方仓库的当前许可证为准。

### 2.1 运行时依赖（用户环境安装，不随包分发）

| 项目 | 用途 | 官方地址 | 许可证已核对 |
|---|---|---|---|
| PyTorch | 推理后端 | https://github.com/pytorch/pytorch | [ ] |
| TorchVision | 图像算子 | https://github.com/pytorch/vision | [ ] |
| MediaPipe | preview 身体 / 手部 | https://github.com/google-ai-edge/mediapipe | [ ] |
| MMEngine | OpenMMLab 基础库 | https://github.com/open-mmlab/mmengine | [ ] |
| MMCV | OpenMMLab 算子库 | https://github.com/open-mmlab/mmcv | [ ] |
| MMDetection | 人体 / 手部检测 | https://github.com/open-mmlab/mmdetection | [ ] |
| MMPose | 2D / 3D 姿态估计 | https://github.com/open-mmlab/mmpose | [ ] |
| OpenCV (opencv-python) | 媒体解码 | https://github.com/opencv/opencv-python | [ ] |
| NumPy / SciPy / tqdm | 数值与工具 | 各自官方仓库 | [ ] |

### 2.2 模型与算法（权重由用户下载）

| 模型 / 算法 | manifest id | 出处 | 许可证已核对 |
|---|---|---|---|
| MediaPipe Pose Landmarker (Full / Heavy / Lite) | `mediapipe_pose_full` / `_heavy` / `_lite` | Google MediaPipe Models | [ ] |
| MediaPipe Hand Landmarker | `mediapipe_hand` | Google MediaPipe Models | [ ] |
| RTMDet-m Person | `rtmdet_m_person` | OpenMMLab RTMPose 项目 | [ ] |
| RTMPose-m Body | `rtmpose_m_body` | OpenMMLab RTMPose v1 | [ ] |
| RTMPose-x Body | `rtmpose_x_body` | OpenMMLab RTMPose v1 | [ ] |
| MotionBERT (H36M 微调) | `motionbert_body3d` | MotionBERT + MMPose 移植版 | [ ] |
| VideoPose3D | `videopose3d_body3d` | Meta VideoPose3D + MMPose 移植版 | [ ] |
| RTMDet-nano Hand | `rtmdet_nano_hand` | OpenMMLab RTMPose v1 | [ ] |
| RTMPose-m Hand5 | `rtmpose_m_hand5` | OpenMMLab RTMPose v1 | [ ] |
| InterNet InterHand3D | `internet_hand3d` | InterHand2.6M / MMPose 移植版 | [ ] |

> **注意**：训练数据集的许可证可能比代码更严格（例如 Human3.6M、InterHand2.6M 都有
> 单独的使用条款，通常仅限学术研究）。基于这些数据集训练的权重用于商业项目前，
> 必须单独确认授权。这直接影响 `motionbert_body3d`、`videopose3d_body3d` 与
> `internet_hand3d`。

### 2.3 MMPose config 文件

`models/openmmlab/configs/` 下的 `.py` config 来自 MMPose / MMDetection 源码仓库，
沿用其项目许可证。插件不分发这些文件，由用户按 `docs/TECHNICAL_DESIGN.md` §3.4 下载。

## 3. 本插件自身

- 插件源码的许可证由项目维护者确定，发布前需在此处与仓库根目录补上 `LICENSE` 文件。
- 若最终选择的许可证与某个运行时依赖不兼容，需在文档中明确说明这些依赖属于
  「用户自备的外部环境」，而非本插件的衍生作品。

## 4. 发布前签核

- [ ] 上表所有条目已核对
- [ ] 生成的 zip 内不含任何 `.pth` / `.task` / `.onnx` / `.pt` / `.ckpt`
- [ ] `docs/INSTALL.md` 中每个模型都有可访问的官方下载地址
- [ ] 涉及学术研究限制的数据集来源已在发布说明中提示
- [ ] 仓库根目录已放置 `LICENSE`

核查人：______________  日期：____-__-__  版本：0.1.0
