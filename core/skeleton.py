"""Standard skeleton v0 definition and source-model joint mappings.

Standard-library only. Defines:

* the joint names of ``mocap_standard_v0`` (guide section 8.1),
* the coordinate convention (guide section 8.2),
* MediaPipe Pose / MediaPipe Hand / H36M-17 landmark mappings,
* the retarget chain table consumed by ``blender/rigify_adapter.py``.

Coordinate convention (normative, guide section 8.2)::

    X = character left/right, +X is the character's own LEFT (matches Blender
        Rigify where ``.L`` bones live at +X)
    Y = character front/back
    Z = up
    unit = meter

Note: the sample result JSON in guide section 5.3 is Y-up and puts ``.L`` at
negative X. That sample contradicts section 8.2 and the Blender/Rigify
convention, so section 8.2 wins. See ``README.md`` for the full deviation list.
"""

from __future__ import annotations

#: Skeleton identifier written into ``mocap_result.json``.
SKELETON_ID = "mocap_standard_v0"

#: Coordinate system identifier written into ``mocap_result.json``.
COORDINATE_SYSTEM = "blender_world"

#: Length unit written into ``mocap_result.json``.
UNIT = "meter"

# --------------------------------------------------------------------------------------
# Joint names
# --------------------------------------------------------------------------------------

#: Torso joints, ordered from the pelvis upwards.
TORSO_JOINTS = ("pelvis", "spine", "chest", "neck", "head")

#: Arm joints per side.
ARM_JOINTS = ("shoulder", "elbow", "wrist")

#: Leg joints per side (``heel`` is optional, see :data:`OPTIONAL_BODY_JOINTS`).
LEG_JOINTS = ("hip", "knee", "ankle", "toe")

#: Body joints every frame of a valid result must provide (19 entries).
BODY_JOINTS = TORSO_JOINTS + tuple(
    "{0}.{1}".format(joint, side) for side in ("L", "R") for joint in ARM_JOINTS
) + tuple(
    "{0}.{1}".format(joint, side) for side in ("L", "R") for joint in LEG_JOINTS
)

#: Body joints that may be absent (guide section 8.1 lists them, section 5.3 omits them).
OPTIONAL_BODY_JOINTS = ("heel.L", "heel.R")

#: Every body joint name the standard skeleton knows about.
ALL_BODY_JOINTS = BODY_JOINTS + OPTIONAL_BODY_JOINTS

#: Finger names in the standard skeleton and their Rigify control prefixes.
FINGERS = (
    ("thumb", "thumb"),
    ("index", "f_index"),
    ("middle", "f_middle"),
    ("ring", "f_ring"),
    ("pinky", "f_pinky"),
)

#: Per-finger segment suffixes (guide section 8.1).
FINGER_SEGMENTS = ("01", "02", "03", "tip")


def hand_joints(side: str) -> tuple:
    """Optional hand joint names for ``side`` (``"L"`` or ``"R"``)."""
    return tuple(
        "{0}.{1}.{2}".format(finger, segment, side)
        for finger, _rig in FINGERS
        for segment in FINGER_SEGMENTS
    )


#: All optional hand joints (40 entries: 20 per hand).
HAND_JOINTS = hand_joints("L") + hand_joints("R")

# --------------------------------------------------------------------------------------
# Retarget chain table
# --------------------------------------------------------------------------------------

#: ``mode`` values used by :data:`RETARGET_CHAINS`.
MODE_ROOT = "root"          # translation only (root motion)
MODE_AIM = "aim"            # minimal-arc rotation from rest direction to source direction
MODE_AIM_REF = "aim_ref"    # aim plus a secondary reference axis (removes twist ambiguity)
MODE_IDENTITY = "identity"  # keep rest orientation, still keyframed for a complete Action

#: Reference axis sources for :data:`MODE_AIM_REF`.
REF_HIP_LINE = "hip_line"            # hip.L - hip.R
REF_SHOULDER_LINE = "shoulder_line"  # shoulder.L - shoulder.R


class ChainSpec(object):
    """One source-direction to Rigify-control-bone assignment."""

    __slots__ = ("target", "aliases", "source", "mode", "ref", "required", "group")

    def __init__(
        self,
        target: str,
        source=None,
        mode: str = MODE_AIM,
        ref=None,
        aliases=(),
        required: bool = True,
        group: str = "body",
    ) -> None:
        self.target = target
        self.aliases = tuple(aliases)
        self.source = tuple(source) if source else None
        self.mode = mode
        self.ref = ref
        self.required = bool(required)
        self.group = group

    @property
    def candidates(self) -> tuple:
        """Bone names to try in order when resolving against a real armature."""
        return (self.target,) + self.aliases

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "ChainSpec({0!r}, {1}, {2})".format(self.target, self.source, self.mode)


#: Torso, arm and leg chains.
#:
#: Rest directions were measured on a Rigify 0.6.10 human rig; ``hips`` and
#: ``chest`` are deliberately not driven because their rest direction is +Y
#: horizontal (widget-style controls), which makes rest-to-source aiming invalid.
#: They stay at rest so the animator can still tweak them afterwards.
BODY_CHAINS = (
    # Root motion: pelvis world position -> torso control location.
    ChainSpec("torso", source=("pelvis", None), mode=MODE_ROOT, aliases=("root",)),
    # Spine: 3 standard segments spread over the 4 Rigify spine FK controls.
    ChainSpec("spine_fk", source=("pelvis", "spine"), mode=MODE_AIM_REF, ref=REF_HIP_LINE),
    ChainSpec("spine_fk.001", source=("spine", "chest"), mode=MODE_AIM_REF, ref=REF_HIP_LINE),
    ChainSpec(
        "spine_fk.002",
        source=("spine", "chest"),
        mode=MODE_AIM_REF,
        ref=REF_SHOULDER_LINE,
        aliases=("chest",),
    ),
    ChainSpec("spine_fk.003", source=("chest", "neck"), mode=MODE_AIM_REF, ref=REF_SHOULDER_LINE),
    ChainSpec("neck", source=("neck", "head"), mode=MODE_AIM_REF, ref=REF_SHOULDER_LINE),
    # Legacy frames follow the neck; an optional anatomical orientation overrides this in v0.3.1.
    ChainSpec("head", mode=MODE_IDENTITY, ref=REF_SHOULDER_LINE),
    # Arms.
    ChainSpec("shoulder.L", source=("chest", "shoulder.L"), mode=MODE_AIM),
    ChainSpec("upper_arm_fk.L", source=("shoulder.L", "elbow.L"), mode=MODE_AIM),
    ChainSpec("forearm_fk.L", source=("elbow.L", "wrist.L"), mode=MODE_AIM),
    ChainSpec("hand_fk.L", source=("wrist.L", "middle.01.L"), mode=MODE_AIM, required=False),
    ChainSpec("shoulder.R", source=("chest", "shoulder.R"), mode=MODE_AIM),
    ChainSpec("upper_arm_fk.R", source=("shoulder.R", "elbow.R"), mode=MODE_AIM),
    ChainSpec("forearm_fk.R", source=("elbow.R", "wrist.R"), mode=MODE_AIM),
    ChainSpec("hand_fk.R", source=("wrist.R", "middle.01.R"), mode=MODE_AIM, required=False),
    # Legs.
    ChainSpec("thigh_fk.L", source=("hip.L", "knee.L"), mode=MODE_AIM),
    ChainSpec("shin_fk.L", source=("knee.L", "ankle.L"), mode=MODE_AIM),
    ChainSpec("foot_fk.L", source=("ankle.L", "toe.L"), mode=MODE_AIM, ref=REF_HIP_LINE),
    ChainSpec("toe_fk.L", mode=MODE_IDENTITY, aliases=("toe.L",)),
    ChainSpec("thigh_fk.R", source=("hip.R", "knee.R"), mode=MODE_AIM),
    ChainSpec("shin_fk.R", source=("knee.R", "ankle.R"), mode=MODE_AIM),
    ChainSpec("foot_fk.R", source=("ankle.R", "toe.R"), mode=MODE_AIM, ref=REF_HIP_LINE),
    ChainSpec("toe_fk.R", mode=MODE_IDENTITY, aliases=("toe.R",)),
)


def _finger_chains() -> tuple:
    chains = []
    for side in ("L", "R"):
        for standard, rig in FINGERS:
            for index, segment in enumerate(FINGER_SEGMENTS[:-1]):
                nxt = FINGER_SEGMENTS[index + 1]
                chains.append(
                    ChainSpec(
                        "{0}.{1}.{2}".format(rig, segment, side),
                        source=(
                            "{0}.{1}.{2}".format(standard, segment, side),
                            "{0}.{1}.{2}".format(standard, nxt, side),
                        ),
                        mode=MODE_AIM,
                        required=False,
                        group="hand",
                    )
                )
    return tuple(chains)


#: Optional finger chains, only used when the result carries ``hands3d`` data.
HAND_CHAINS = _finger_chains()

#: Complete chain table.
RETARGET_CHAINS = BODY_CHAINS + HAND_CHAINS

#: Rigify bones whose ``IK_FK`` custom property must be set to FK (1.0) so the
#: FK keyframes actually drive the deform bones. Verified on Rigify 0.6.10.
IK_FK_SWITCH_BONES = (
    "upper_arm_parent.L",
    "upper_arm_parent.R",
    "thigh_parent.L",
    "thigh_parent.R",
)

#: Property name of the Rigify IK/FK switch (1.0 == FK).
IK_FK_PROPERTY = "IK_FK"

#: Bones used to recognise a generated Rigify human rig.
RIGIFY_SIGNATURE_BONES = (
    "torso",
    "spine_fk",
    "neck",
    "head",
    "upper_arm_fk.L",
    "upper_arm_fk.R",
    "thigh_fk.L",
    "thigh_fk.R",
    "foot_fk.L",
    "foot_fk.R",
)

#: Bone whose rest head height is used as the rig's reference pelvis height.
RIG_PELVIS_REFERENCE_BONES = ("ORG-spine", "DEF-spine", "spine_fk", "torso")

# --------------------------------------------------------------------------------------
# MediaPipe Pose Landmarker
# --------------------------------------------------------------------------------------

#: MediaPipe Pose landmark index -> standard joint name (direct correspondences).
MEDIAPIPE_POSE_MAP = {
    11: "shoulder.L",
    12: "shoulder.R",
    13: "elbow.L",
    14: "elbow.R",
    15: "wrist.L",
    16: "wrist.R",
    23: "hip.L",
    24: "hip.R",
    25: "knee.L",
    26: "knee.R",
    27: "ankle.L",
    28: "ankle.R",
    29: "heel.L",
    30: "heel.R",
    31: "toe.L",
    32: "toe.R",
}

#: Landmarks averaged to build joints MediaPipe does not provide directly.
MEDIAPIPE_DERIVED = {
    "pelvis": ((23, 24), 0.0),
    "neck": ((11, 12), 0.0),
    "head": ((7, 8), 0.0),
}

#: Fraction along pelvis -> neck used for the two intermediate torso joints.
MEDIAPIPE_SPINE_FRACTION = 0.35
MEDIAPIPE_CHEST_FRACTION = 0.75

#: MediaPipe Hand landmark index -> standard finger joint suffix.
MEDIAPIPE_HAND_MAP = {
    1: "thumb.01",
    2: "thumb.02",
    3: "thumb.03",
    4: "thumb.tip",
    5: "index.01",
    6: "index.02",
    7: "index.03",
    8: "index.tip",
    9: "middle.01",
    10: "middle.02",
    11: "middle.03",
    12: "middle.tip",
    13: "ring.01",
    14: "ring.02",
    15: "ring.03",
    16: "ring.tip",
    17: "pinky.01",
    18: "pinky.02",
    19: "pinky.03",
    20: "pinky.tip",
}


def mediapipe_to_blender(x: float, y: float, z: float) -> tuple:
    """Convert a MediaPipe world landmark to the standard coordinate system.

    MediaPipe world landmarks are metric with the hip midpoint as origin,
    ``x`` to the image right, ``y`` down and ``z`` towards the camera (more
    negative == closer). A subject facing the camera has their own left on the
    image right, so subject-left maps to +X, matching Rigify ``.L`` bones.
    """
    return (float(x), float(z), -float(y))


# --------------------------------------------------------------------------------------
# H36M 17-keypoint layout (RTMPose -> MotionBERT pipeline)
# --------------------------------------------------------------------------------------

#: H36M joint index -> standard joint name.
H36M_17_MAP = {
    0: "pelvis",
    1: "hip.R",
    2: "knee.R",
    3: "ankle.R",
    4: "hip.L",
    5: "knee.L",
    6: "ankle.L",
    7: "spine",
    8: "chest",
    9: "neck",
    10: "head",
    11: "shoulder.L",
    12: "elbow.L",
    13: "wrist.L",
    14: "shoulder.R",
    15: "elbow.R",
    16: "wrist.R",
}

#: Joints the H36M layout cannot provide; they are synthesised with low confidence.
H36M_SYNTHESISED_JOINTS = ("toe.L", "toe.R")

#: Confidence written for synthesised joints (below the 0.4 "interpolate" threshold).
SYNTHESISED_CONFIDENCE = 0.3


def h36m_to_blender(x: float, y: float, z: float) -> tuple:
    """Convert an H36M/MotionBERT root-relative coordinate to the standard system.

    MotionBERT emits H36M-style coordinates with ``y`` pointing down and ``z``
    away from the camera; the worker converts once here so no model-specific
    axis convention ever reaches the Blender retarget layer.
    """
    return (float(x), float(z), -float(y))


# --------------------------------------------------------------------------------------
# Confidence thresholds (guide section 8.3)
# --------------------------------------------------------------------------------------

#: At or above this value a keypoint is used as-is.
CONFIDENCE_GOOD = 0.7

#: Below this value a keypoint is preferably interpolated.
CONFIDENCE_LOW = 0.4

#: Minimum confidence for a foot to be considered for contact detection.
CONFIDENCE_CONTACT = 0.5


def is_left(joint: str) -> bool:
    """True when ``joint`` belongs to the character's left side."""
    return str(joint).endswith(".L")


def is_right(joint: str) -> bool:
    """True when ``joint`` belongs to the character's right side."""
    return str(joint).endswith(".R")


def mirror_joint(joint: str) -> str:
    """Return the opposite-side joint name (``"wrist.L"`` -> ``"wrist.R"``)."""
    text = str(joint)
    if text.endswith(".L"):
        return text[:-2] + ".R"
    if text.endswith(".R"):
        return text[:-2] + ".L"
    return text
