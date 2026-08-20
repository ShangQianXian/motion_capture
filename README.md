# Motion Capture for Rigify (v0.1)

从视频或图片捕捉单人全身动作，并重定向到 Blender **Rigify Human 生成 rig**，输出可编辑的 Blender Action。

- 设计文档：`docs/TECHNICAL_DESIGN.md`
- 施工手册：`docs/DEVELOPMENT_GUIDE.md`
- 安装与模型下载：`docs/INSTALL.md`
- 打包发布：`docs/PACKAGING.md`
- 验收清单：`docs/ACCEPTANCE.md`
- 许可证核查：`docs/LICENSES.md`

## 1. 这个仓库就是插件包

仓库根目录 `motion_capture/` **本身**就是 Blender 插件包（`__init__.py` 在根目录），所以
`models/manifest.example.json` 与 `docs/` 天然位于 `<addon_root>` 下，符合开发文档 §6.1 的
manifest 查找顺序。

```text
motion_capture/                 <- 安装后即 <blender>/scripts/addons/motion_capture
  __init__.py                   bl_info + 集中 register/unregister
  addon/                        UI：preferences / properties / panels / operators / ui_text
  core/                         纯标准库逻辑，Blender 与 worker 共用
  blender/                      仅此目录做 bpy/mathutils 重活：Rigify 检测、重定向、烘焙
  backend_worker/               外部推理进程：CLI、解码、姿态估计、平滑、导出
  models/manifest.example.json  模型清单唯一数据源（不要再建第二份）
  tests/unit/  tests/blender/   零依赖单测 + Blender 后台测试
  tools/                        dev_install / run_tests / build_zip / bootstrap_worker_env
```

## 2. 三分钟跑通（不需要任何模型）

```powershell
# 1. 链接到 Blender 插件目录（目录联接，免管理员）
.\tools\dev_install.ps1

# 2. 跑全部自动化测试
.\tools\run_tests.ps1

# 3. 单独跑一次 mock worker
python -m backend_worker.cli --job <某个 job.json> --mock
```

在 Blender 中：启用 **Motion Capture for Rigify** → 偏好设置里把 `Worker Python` 指向任意
Python 3.10+（mock 模式不需要推理依赖），`Models Root` 指向本仓库 `models/` → 侧边栏
`Mocap` → 点 **运行 Mock 捕捉** → **导入结果** → 选中 Rigify rig → **应用到 Rigify** →
**烘焙 Action**。

真实推理请先执行 `.\tools\bootstrap_worker_env.ps1` 并按 `docs/INSTALL.md` 下载模型。

## 3. 架构约束（改代码前请先读）

| 约束 | 原因 |
|---|---|
| `core/` 只用标准库，模块间用相对导入 | 同一份代码要能以 `core.*`（worker 进程）和 `motion_capture.core.*`（Blender）两种包名工作 |
| `backend_worker/*` 一律 `from ._core import ...` | `_core.py` 按 `__package__` 判定两种调用方式，保证每个进程只有一份 `core` 实例 |
| `addon/*` 中 **不要**写 `from __future__ import annotations` | Blender 用 `typing.get_type_hints` 解析字符串注解，且会交换 globals/locals，PEP 563 会让 `register_class` 报 `NameError` |
| import 阶段不启动 worker、不读模型、不 import torch/mediapipe/cv2/numpy | 保证插件在任何环境下都能启用 |
| 深度学习推理只在外部 worker 进程里跑 | Blender 主线程只做 UI、进度轮询和数据应用 |
| 动画只写 Rigify **控制骨**，不写 `DEF-` 变形骨 | 开发文档 §15 |
| 代码需兼容 Python 3.10 | Blender 4.0 内置 Python 3.10.13 |
| 不使用 `action.slots` | 该 API 仅 Blender 4.4+ 存在，且 4.5.0 上没有 `slots.new_for_id` |

## 4. 与文档的偏差与裁定

文档内部存在若干互相矛盾之处，实现时的裁定如下（**代码以本表为准**）。

| 文档处 | 冲突内容 | 本实现的裁定 |
|---|---|---|
| §8.2 vs §5.3 示例 | §8.2 规定 Z 轴向上；§5.3 的示例 JSON 里 `pelvis:[0,1.0,0]`、`head:[0,1.72,0]` 是 **Y-up** | **以 §8.2 为准**：X=左右、Y=前后、Z=上、单位米。§5.3 数值仅为示意。`looks_y_up()` 会检出 Y-up 数据并告警 |
| §5.3 示例 | 示例把 `shoulder.L` 放在 **−X** | **`.L` = +X**（角色自身左侧），与 Blender/Rigify 实测一致。`detect_mirror()` 检出镜像并提示勾选 Flip X |
| Phase 2 验收 vs §7.2 | 一处写 mock 输出 30 帧，一处写默认 60 帧 | 默认 **60 帧**，同时严格遵守 job 的 `frame_start/frame_end` |
| §10.2 推荐映射 | `chest -> chest` | `hips`/`chest` 实测 rest 方向是 **+Y 水平**（widget 型控制骨），做 rest→源方向求解会算错，故躯干改用 `spine_fk`/`spine_fk.001`/`spine_fk.002`/`spine_fk.003` 链；`hips`/`chest` 保持 rest 供动画师后期手调 |
| §10.2 推荐映射 | `toe.L -> toe.L 或 toe_fk.L` | Rigify 0.6.10 只有 `toe_fk.L`，故 `toe_fk.L` 为主、`toe.L` 作别名回退 |
| §9.4 足底锁定 | 位置未明确在哪一侧实现 | 放在 **worker 侧的 3D 关节坐标上**（§2 规定 worker 负责时序平滑）：水平修正只作用于该侧 `ankle/heel/toe`（`knee` 取 50%），高度修正对全身做 Z 平移。Blender 侧不引入 IK 约束，符合 §9.4「只微调脚部与 pelvis 高度、不改写上半身姿态」 |
| `manifest.example.json` | `preflight_rules` 缺 `fallback_cpu`，`quality_plus` 缺 `optional_artifact_ids` | **不改 manifest**（§15 禁止第二份清单）。在 `core/model_manifest.py` 的 `PROFILE_DEFAULT_RULES` 里补全：`fallback_cpu` 用 `required_any_of: [mediapipe_pose_lite, mediapipe_pose_full]`（§6.2「优先 Lite，其次 Full」）。用户自建 `models/manifest.json` 可覆盖 |
| §4.1 profile 列表 | `hand_enhanced` 是 profile，但 §3.3 的捕捉枚举不含它 | `hand_enhanced` 只作**预检 profile**，不进捕捉 profile 枚举 |

## 5. 相对文档的新增项

### 5.1 新增错误码（§5.4 的 12 个必需码全部支持，另加）

`JOB_SCHEMA_INVALID`、`DEPENDENCY_MISSING`、`WORKER_EXIT_ERROR`、`CANCELLED`、
`MEDIA_NOT_FOUND`、`UNSUPPORTED_MEDIA_TYPE`、`INTERNAL_ERROR`。

### 5.2 新增场景属性（§3.3 列出的字段全部具备，另加）

`effective_profile`、`last_job_dir`、`progress`、`progress_text`、`switch_limbs_to_fk`、
`flip_x`、`clean_curves`、`frame_step`、`applied_action_name`、`result_frame_start`、
`result_frame_end`、`preflight_ok`、`preflight_done`、`preflight_summary`、`log_entries`。

其中 **`switch_limbs_to_fk`（默认开）是功能正确性所必需**：Rigify 的
`upper_arm_parent.*` / `thigh_parent.*` 带 `IK_FK` 自定义属性，不置为 FK(1.0) 时写入的
FK 关键帧不会驱动变形骨。

### 5.3 新增模块

- `core/result_schema.py`：结果校验与加载（文档只在 §4.4 给了函数签名）。
- `core/progress.py`、`core/worker_client.py`、`core/paths.py`：拆分自 §4.2/4.3 的接口。
- `blender/temp_data.py`：临时对象/约束注册表与幂等清理。
- `backend_worker/_core.py`、`reporter.py`、`pipeline.py`、`postprocess.py`、`mock_source.py`。
- `mocap.run_capture` 带 `mock` 布尔参数，面板上是独立的「运行 Mock 捕捉」按钮。
- job schema 增加 `mode: "self_test"`，供「测试 Preview/High Quality 模型」复用（未新增 CLI 参数）。

## 6. 当前状态

| 阶段 | 状态 | 验证方式 |
|---|---|---|
| Phase 0 插件注册 | 完成 | `tests/blender/test_enable_addon.py`（4.5.0 / 4.0.2 各 46 检查 PASS） |
| Phase 1 偏好 / manifest / 预检 UI | 完成 | `tests/unit/test_model_manifest.py` |
| Phase 2 worker CLI 与 mock 闭环 | 完成 | `tests/unit/test_mock_worker.py`（含真实子进程 CLI 契约测试） |
| Phase 3 MediaPipe preview | 代码完成，**本机未验证** | 需 `mediapipe` + `.task` 权重 |
| Phase 4 Rigify 重定向与烘焙 | 完成 | `tests/blender/test_mock_retarget.py`（4.5.0 / 4.0.2 各 53 检查 PASS） |
| Phase 5 High Quality (MMPose) | 代码完成，**本机未验证** | 需 torch/mmdet/mmpose + `.pth` 权重 |
| Phase 6 平滑 / 足底锁定 / 错误恢复 | 完成 | `tests/unit/test_smoothing.py` |
| Phase 7 打包与文档 | 完成 | `tools/build_zip.ps1`、本 README 与 docs/ 下四篇新文档 |

Phase 3/5 的推理模块采用**延迟导入 + 结构化错误**：缺少依赖返回 `DEPENDENCY_MISSING`，
缺少权重/config 返回 `MODEL_MISSING`/`CONFIG_MISSING`，CUDA 不可用返回 `CUDA_UNAVAILABLE`，
显存不足返回 `CUDA_OOM` 并给出降级建议。精度验收步骤见 `docs/ACCEPTANCE.md`。

## 7. 已知限制（v0.1）

- 单人；检测到多人时取面积最大且置信度最高的一个。
- 无面部捕捉；`head` 控制骨跟随 `neck`，不做独立头部朝向求解。
- 四肢不做扭转（twist）求解：`quat_from_two_vectors` 只约束骨骼指向，前臂/手腕自转未定义。
  躯干链用「指向 + 参考轴」双轴求解，因此角色朝向是稳定的。
- MediaPipe 世界坐标以髋部为原点，全局位移为近似值，运行时会发出
  `MEDIAPIPE_WORLD_APPROXIMATE` 告警。
- H36M-17 没有脚趾关节，`toe.L/R` 由脚踝外推合成，置信度标记为 0.3。
- 重定向速度约每帧 5–10 ms（900 帧约 25–35 秒）；长视频可用 `frame_step` 抽帧预览。
- 不自动下载模型；不把第三方权重打进插件包。
