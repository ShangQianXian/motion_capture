"""Centralised Chinese UI strings.

Identifiers stay English everywhere (``bl_idname``, property names, error codes);
only user-visible labels, descriptions and messages live here. That keeps the
panels readable and makes a future translation a single-file change.
"""

from __future__ import annotations

from ..core import model_manifest

# -- Panels -----------------------------------------------------------------------------

TAB_CATEGORY = "Mocap"

PANEL_ENVIRONMENT = "环境与模型"
PANEL_CAPTURE = "1 · 素材与生成"
PANEL_RIGIFY = "3 · 确认并应用"
PANEL_LOGS = "日志"

# -- Preferences ------------------------------------------------------------------------

PREF_WORKER_PYTHON = "Worker Python"
PREF_WORKER_PYTHON_DESC = "外部推理用的 Python 解释器路径，例如 .venv/Scripts/python.exe"
PREF_MODELS_ROOT = "模型根目录"
PREF_MODELS_ROOT_DESC = "存放 mediapipe/ 与 openmmlab/ 子目录的模型根目录"
PREF_MAX_VRAM = "最大显存 (GB)"
PREF_MAX_VRAM_DESC = "留给推理的显存上限，RTX4060 8GB 建议 7"
PREF_DEFAULT_PROFILE = "默认 Profile"
PREF_DEFAULT_PROFILE_DESC = "新场景默认使用的捕捉质量档位"
PREF_ALLOW_AUTO_DOWNLOAD = "允许自动下载"
PREF_ALLOW_AUTO_DOWNLOAD_DESC = "v0.1 不实现自动下载，此开关仅为后续版本预留"
PREF_NO_AUTO_DOWNLOAD_NOTE = "v0.1 不会自动下载模型，请按提示手动下载后填写模型根目录。"
PREF_PATHS_HEADER = "路径配置"
PREF_TESTS_HEADER = "环境测试"

# -- Scene properties -------------------------------------------------------------------

PROP_SOURCE_MEDIA = "源文件"
PROP_SOURCE_MEDIA_DESC = "用于动捕的视频或图片"
PROP_SOURCE_TYPE = "源类型"
PROP_SOURCE_TYPE_DESC = "按扩展名自动判断，或强制指定图片/视频"
PROP_TARGET_ARMATURE = "目标 Rig"
PROP_TARGET_ARMATURE_DESC = "Rigify Human 生成后的骨架对象"
PROP_CAPTURE_PROFILE = "捕捉 Profile"
PROP_CAPTURE_PROFILE_DESC = "推理质量档位"
PROP_FRAME_START = "起始帧"
PROP_FRAME_START_DESC = "从第几帧开始处理"
PROP_FRAME_END = "结束帧"
PROP_FRAME_END_DESC = "处理到第几帧结束，0 表示处理到素材末尾"
PROP_TARGET_FPS = "目标 FPS"
PROP_TARGET_FPS_DESC = "视频按此帧率抽帧，不是原始帧率"
PROP_INCLUDE_HANDS = "包含手部"
PROP_INCLUDE_HANDS_DESC = "在模型可用时捕捉手指动作"
PROP_SMOOTHING = "平滑强度"
PROP_SMOOTHING_DESC = "越大越平滑，也越迟滞"
PROP_FOOT_LOCK = "足底锁定强度"
PROP_FOOT_LOCK_DESC = "接触帧把脚拉回锁定点的强度，用于减少脚滑"
PROP_ROOT_MOTION = "根位移"
PROP_ROOT_MOTION_DESC = "保留世界位移，或固定在原地只保留起伏"
PROP_SWITCH_FK = "切换四肢为 FK"
PROP_SWITCH_FK_DESC = "把 Rigify 的 IK_FK 设为 FK，让写入的 FK 关键帧真正驱动变形骨"
PROP_FOOT_CORRECTION = "足部落点校正"
PROP_FOOT_CORRECTION_DESC = (
    "按素材的脚踝位置重新求解大小腿，补偿目标骨架与素材人物的骨长差异。"
    "实测把脚踝偏移从 11.6/4.8 cm 降到 3.2/2.8 cm；代价是腿的朝向会有约 6° 变化"
    "（髋和脚都对齐时，膝盖是唯一还能动的关节）。关掉则只传方向，脚落在骨架自己的腿长上"
)
PROP_FLIP_X = "左右镜像 (Flip X)"
PROP_FLIP_X_DESC = "当动作左右颠倒时勾选，导入时对 X 轴取反"
PROP_PITCH_CORRECTION = "倾斜校正（X）"
PROP_PITCH_CORRECTION_DESC = "给整段动作施加相同的 X 轴旋转，补偿整体前倾或后仰；0° 保留原始姿态"
PROP_CLEAN_CURVES = "烘焙后简化曲线"
PROP_CLEAN_CURVES_DESC = "烘焙时删除冗余关键帧，文件更小但精度略降"
PROP_FRAME_STEP = "帧步长"
PROP_FRAME_STEP_DESC = "大于 1 时抽帧写入关键帧，用于长视频快速预览"
PROP_INPUT_NORMALISATION = "二维输入尺度"
PROP_INPUT_NORMALISATION_DESC = (
    "提升网络看到的二维关键点尺度。跟随画面比例：人物占画面长边 50%–90% 时最准（推荐）。"
    "规范尺度：把人物重定基到训练时的标准比例，人物偏小时（占画面不足约 45%）明显更稳，"
    "人物偏大时避免输入超范围。改这项会让已有结果失效，需重新生成"
)

# -- Operator labels --------------------------------------------------------------------

OP_PREFLIGHT = "检查模型环境"
OP_PREFLIGHT_DESC = "检查当前 profile 需要的模型文件是否齐全"
OP_COPY_LINKS = "复制缺失模型下载链接"
OP_COPY_LINKS_DESC = "把缺失模型的保存路径和下载地址复制到剪贴板"
OP_OPEN_MODELS_DIR = "打开模型目录"
OP_OPEN_MODELS_DIR_DESC = "在文件管理器中打开模型根目录"
OP_TEST_WORKER = "测试 Worker Python"
OP_TEST_WORKER_DESC = "运行 worker 的 --check-env，检查解释器与依赖"
OP_TEST_CUDA = "测试 CUDA"
OP_TEST_CUDA_DESC = "运行 worker 的 --check-cuda，检查显卡可用性"
OP_TEST_PREVIEW = "测试 Preview 模型"
OP_TEST_PREVIEW_DESC = "加载 MediaPipe 模型验证 preview 链路"
OP_TEST_QUALITY = "测试 High Quality 模型"
OP_TEST_QUALITY_DESC = "加载 RTMDet/RTMPose/MotionBERT 验证高精度链路"
OP_RUN_CAPTURE = "生成动捕预览"
OP_RUN_CAPTURE_DESC = "启动外部 worker 执行动作捕捉"
OP_RUN_MOCK = "运行 Mock 捕捉"
OP_RUN_MOCK_DESC = "不加载任何模型，生成合成动作用于验证流程"
OP_CANCEL_CAPTURE = "取消捕捉"
OP_CANCEL_CAPTURE_DESC = "请求 worker 停止当前任务"
OP_IMPORT_RESULT = "导入结果"
OP_IMPORT_RESULT_DESC = "读取并校验 mocap_result.json"
OP_APPLY_RIGIFY = "确认并应用"
OP_APPLY_RIGIFY_DESC = "确认已核对当前预览，创建从第 1 帧开始的新 Action；保留原动画"
OP_CALIBRATE_PITCH = "按站立姿态校准"
OP_CALIBRATE_PITCH_DESC = "假设人物总体直立，根据整段动作估算 X 轴倾斜校正。适合站立、走路；弯腰、躺卧动作请手动调整"
OP_BAKE_ACTION = "烘焙 Action"
OP_BAKE_ACTION_DESC = "把可视变换烘焙成可编辑的 Action 并清理临时数据"
OP_CLEAR_TEMP = "清理临时数据"
OP_CLEAR_TEMP_DESC = "删除本插件创建的临时对象和约束"

# -- Status -----------------------------------------------------------------------------

STATUS_LABELS = {
    "applying": "正在应用动作",
    "idle": "空闲",
    "checking": "检查中",
    "ready": "就绪",
    "running": "运行中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}

PROFILE_LABELS = {
    'quality_feet': '身体与脚部增强（兼容 Quality）',
    "preview": "快速捕捉（MediaPipe）",
    "quality": "Quality（身体、脚掌与头部）",
    "quality_plus": "Quality Plus（增强身体、脚掌与头部）",
    "fallback_cpu": "CPU Fallback（低配兜底）",
}

# -- Messages ---------------------------------------------------------------------------

MSG_NO_MODELS_ROOT = "尚未配置模型根目录，请在插件偏好设置中填写。"
MSG_NO_WORKER_PYTHON = "尚未配置 Worker Python，请在插件偏好设置中填写。"
MSG_ENV_OK = "环境检查通过：{0}"
MSG_ENV_MISSING = "缺少 {0} 个必需文件，{1} 个可选文件。"
MSG_COPIED = "已复制到剪贴板（{0} 行）。"
MSG_PREFLIGHT_FIRST = "请先点击“检查模型环境”。"
MSG_CAPTURE_STARTED = "已启动 worker：{0}"
MSG_CAPTURE_DONE = "捕捉完成：{0}"
MSG_CAPTURE_CANCELLED = "已请求取消捕捉。"
MSG_NO_RUNNING_JOB = "当前没有正在运行的捕捉任务。"
MSG_RESULT_IMPORTED = "已导入 {0} 帧（{1} FPS，profile {2}）。"
MSG_NO_RESULT = "没有可导入的结果，请先运行捕捉。"
MSG_APPLIED = "已写入 Action “{0}”：{1} 条曲线，{2} 个关键帧。"
MSG_BAKED = "烘焙完成：Action “{0}”，{1} 条曲线，{2} 个关键帧。"
MSG_TEMP_CLEARED = "已清理 {0} 个临时对象、{1} 个临时约束。"
MSG_SELECT_ARMATURE = "请先选择 Rigify 生成的目标骨架。"
MSG_ENV_BLOCKED = "当前 profile 缺少必需模型，无法运行捕捉。"
MSG_PREVIEW_HINT = "preview 只需要 MediaPipe Pose Full，缺少 MMPose 模型也能运行。"
MSG_WORKER_HINT = "mock 模式不需要推理依赖，任何 Python 3.10+ 都可以作为 Worker Python。"
MSG_LOG_EMPTY = "暂无日志。"
MSG_SELF_TEST_OK = "模型自测通过：{0}"

#: Number of log entries kept for the log panel (guide section 3.4).
LOG_LIMIT = 20


def status_label(status: str) -> str:
    """Chinese label for a ``job_status`` value."""
    return STATUS_LABELS.get(str(status), str(status))


def profile_label(profile: str) -> str:
    """Chinese label for a capture profile."""
    return PROFILE_LABELS.get(str(profile), str(profile))


def capture_profile_items() -> list:
    """``EnumProperty`` items for the capture profiles.

    Built by a plain function on purpose: Blender resolves string annotations
    through ``typing.get_type_hints``, which swaps globals and locals, so a list
    comprehension written inline in an annotation cannot see its module's names.
    """
    items = []
    for profile in model_manifest.CAPTURE_PROFILES:
        label = profile_label(profile)
        items.append((profile, label, label))
    return items
