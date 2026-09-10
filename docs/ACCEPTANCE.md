# 验收清单

> 当前实现目标：Blender 4.5.0，Python 3.10.0 双 worker 环境。依赖版本、安装与诊断以 [INSTALL.md](INSTALL.md) 和 `requirements/` 为准；下文早期 4.0 兼容记录仅作历史参考。

对应 `docs/DEVELOPMENT_GUIDE.md` §12 与 `docs/TECHNICAL_DESIGN.md` §10。

自动化部分（A 节）在本机已全部通过；手工部分（B/C 节）需要真实模型与素材。

---

## A. 自动化验收（已通过）

```powershell
.\tools\run_tests.ps1
```

### A.1 无 Blender 的纯 Python 测试

```powershell
python -m unittest discover -s tests/unit -t .
python -m pytest tests/unit          # pytest 已安装时同样可用
```

| 覆盖点（文档 §12.1） | 测试文件 |
|---|---|
| manifest JSON 可解析 | `test_model_manifest.py` |
| `check_profile_requirements` 缺失报告正确 | `test_model_manifest.py` |
| job schema 校验 | `test_job_schema.py` |
| progress JSONL 解析 | `test_progress.py` |
| result schema 校验 | `test_result_schema.py` |
| mock worker 输出合法 result | `test_mock_worker.py` |
| smoothing / 插值 / 足底锁定 | `test_smoothing.py` |
| 重定向数学 | `test_retarget_math.py` |
| 标准骨架与映射表 | `test_skeleton.py` |
| 错误对象与路径工具 | `test_errors_paths.py` |

结果：**326 个用例通过**。新增解释器误配、环境路由、CLI 失败退出、非有限 FPS 和降级 profile 回归，另含 23 项模型下载器测试。

### A.2 Blender 后台测试

```powershell
blender --background --factory-startup --python tests\blender\test_enable_addon.py
blender --background --factory-startup --python tests\blender\test_mock_retarget.py
```

| 验收项 | 结果 |
|---|---|
| 插件启用不报错 | 通过 |
| `addon_enable(module="motion_capture")` 成功 | 通过 |
| 属性注册成功 | 通过 |
| 禁用后类 / Scene 属性 / handler 全部清理 | 通过 |
| 标准 Rigify Human rig 可被检测 | 通过 |
| metarig 与非 Armature 被正确拒绝 | 通过 |
| mock result 能导入 | 通过 |
| 生成 Action 且含关键帧 | 通过（219 条曲线 / 6454 关键帧） |
| 烘焙后动画可编辑 | 通过（537 条曲线 / 15994 关键帧） |
| 临时对象与约束可清理 | 通过 |
| 无左右反转 | 通过（`.L` 骨骼世界 X 为正） |
| in-place 模式去掉水平位移、保留起伏 | 通过 |

本次已验证：**Blender 4.5.0（Python 3.11.11）**，48 + 63 项检查通过，包括用户约束保留、其他 rig 不受影响和连续镜像应用。4.0.2 为早期测试记录，不在本次支持范围内。

### A.3 Worker CLI 契约

`tests/unit/test_mock_worker.py::TestWorkerCLI` 以真实子进程验证：

- [x] stdout 每一行都是合法 JSON，且 `event` 在允许集合内
- [x] 首行 `started`、末行 `completed`
- [x] `completed.result_path` 指向一个能通过校验的 result
- [x] `worker.log` 被写入，stdout 保持纯净
- [x] job 文件缺失 → `failed` 事件 + 非 0 退出
- [x] 无模型的真实 profile → `failed` 且错误码是 `MODEL_MISSING` / `CONFIG_MISSING`
- [x] `cancel.request` 哨兵 → `cancelled` 事件 + 退出码 **130**
- [x] `--check-env` 返回 JSON 且退出码 0
- [x] `--check-cuda` 无 torch 时不泄露 traceback

---

## B. 手工验收：环境与降级

需要真实模型。逐项打勾。

### B.1 模型安装

- [ ] 全新机器、无任何模型：**检查模型环境** 能列出所有缺失项（展示名 + 相对路径 + 下载链接 + 降级说明）
- [ ] **复制缺失模型下载链接** 的剪贴板格式符合 §6.3
- [ ] 手动下载后逐项显示通过
- [ ] `Worker Python` 配错：报明确错误，且**不影响 Blender 启动与面板显示**
- [ ] CUDA 不可用：允许切到 `preview` / `fallback_cpu`
- [ ] 只装 MediaPipe Pose Full：`preview` 可运行，`quality` 的 **运行捕捉** 按钮置灰
- [ ] 删除 RTMPose-x：`quality_plus` 自动回退 `quality` 并给出告警
- [ ] 删除 MediaPipe Hand：身体捕捉照常，手部被禁用并给出告警
- [ ] 删除某个 `.py` config：报 `CONFIG_MISSING` 而不是 `MODEL_MISSING`

### B.2 取消与健壮性

- [ ] 长视频运行中点 **取消捕捉**：worker 退出，UI 不假死，状态变为「已取消」
- [ ] 运行中按 `ESC`：同上
- [ ] 运行中禁用插件：worker 被终止，无残留进程
- [ ] 日志面板显示最近 20 条 progress / warning / error

---

## C. 手工验收：动捕质量

### C.1 素材清单（§12.3）

| # | 素材 | 用途 |
|---|---|---|
| 1 | T-pose 或 A-pose 单人正面**图片** | 单帧姿态、左右方向 |
| 2 | 正面**走路**视频 | 步态、足底锁定 |
| 3 | **抬手挥手**视频 | 手臂链方向 |
| 4 | 侧身**短暂遮挡**视频 | 低置信度插值 |
| 5 | 30 秒 **1080p 单人**视频 | 性能与稳定性 |

### C.2 判定标准

对每个素材（尤其 #2 与 #5）确认：

- [ ] **没有左右反转**：角色左手对应画面中人物的左手
- [ ] **骨盆高度稳定**：无逐帧抖动或整体漂移
- [ ] **肩肘腕方向正确**：手臂不穿身体、不反向
- [ ] **膝盖与肘部不明显反折**
- [ ] **脚滑明显降低**：`Foot Lock Strength = 0` 与 `0.7` 对比，接触期脚部位移显著减少
- [ ] **Action 可编辑**：Dope Sheet / Graph Editor 中可选中并调整关键帧
- [ ] 遮挡帧**不出现大幅爆转**，日志给出 `LOW_CONFIDENCE` / `LONG_OCCLUSION` 告警
- [ ] warning **不阻断**结果导入

### C.3 各链路

- [ ] `preview`：单图输出 1 帧、短视频输出多帧
- [ ] `quality`：1080p 单人 30 秒视频能完整跑完
- [ ] `quality_plus`：RTMPose-x 缺失或 OOM 时回退 `quality`
- [ ] `fallback_cpu`：无 CUDA 环境下能出结果

### C.4 性能参考（RTX4060 8GB / i7-14650HX）

| 环节 | 期望 |
|---|---|
| Blender 侧重定向 | 约 5–10 ms/帧（900 帧约 25–35 秒） |
| 烘焙 | 30 帧约 0.1 秒 |
| `quality` 推理 | 取决于分辨率；长边限制在 960–1280 |

---

## D. 记录表

| 日期 | 版本 | Blender | 素材 | 结果 | 备注 |
|---|---|---|---|---|---|
| | 0.1.0 | 4.5.0 | 自动化 | PASS | 292 单测 + 46/53 后台检查 |
| | 0.1.0 | 4.0.2 | 自动化 | PASS | 46/53 后台检查 |
| | | | | | |
