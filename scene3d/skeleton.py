"""Parametric humanoid rig with forward kinematics and OpenPose BODY_25 output.

A human's pose is a small parameter dict (angles in degrees, lengths in m):

  pelvis_y      height of the pelvis center
  pitch, roll   whole-body orientation (pitch +90 = lying face down)
  spine_*       chest orientation relative to the pelvis
  head_*        head orientation relative to the chest
  legs / arms   per-limb chord direction (in parent frame), flexion, pole hint

The entity-level x, z position and yaw live on the Human object. Limb chords +
two-bone IK give exact bone lengths in every pose, so the same machinery drives
both preset poses and constraint solving.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from . import hands
from .vecmath import (V, UP, clamp, f_fwd, f_right, f_up, frame, horizontal,
                      look_frame, normalize, norm, rot_x, rot_y, rot_z,
                      two_bone_ik, chord_for_bend, yaw_of)

# ---------------------------------------------------------------- dimensions
# At scale 1.0 the model is ~1.75 m tall.
DIMS = {
    "hip_half": 0.095,      # pelvis center to hip joint (x)
    "thigh": 0.42,
    "shin": 0.42,
    "foot_len": 0.19,       # ankle to big toe (forward)
    "heel_len": 0.07,
    "ankle_h": 0.08,        # ankle height when standing flat
    "spine": 0.50,          # pelvis center to neck
    "shoulder_half": 0.18,
    "shoulder_drop": 0.03,  # shoulders sit slightly below the neck point
    "upper_arm": 0.28,
    "forearm": 0.26,
    "hand": 0.18,
    "neck_head": 0.15,      # neck joint to head center
    "head_r": 0.105,
}
LEG_LEN = DIMS["thigh"] + DIMS["shin"]
ARM_LEN = DIMS["upper_arm"] + DIMS["forearm"]
STAND_PELVIS_Y = DIMS["ankle_h"] + LEG_LEN  # 0.92

BODY_25_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow",
    "l_wrist", "mid_hip", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
    "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear", "l_big_toe", "l_small_toe",
    "l_heel", "r_big_toe", "r_small_toe", "r_heel",
]

# Approximate joint surface radii, used for ground clearance and collisions.
JOINT_RADII = {
    "nose": 0.02, "neck": 0.06, "mid_hip": 0.11,
    "r_shoulder": 0.06, "l_shoulder": 0.06, "r_elbow": 0.045, "l_elbow": 0.045,
    "r_wrist": 0.035, "l_wrist": 0.035, "r_hip": 0.09, "l_hip": 0.09,
    "r_knee": 0.06, "l_knee": 0.06, "r_ankle": 0.045, "l_ankle": 0.045,
    "r_eye": 0.02, "l_eye": 0.02, "r_ear": 0.02, "l_ear": 0.02,
    "l_big_toe": 0.02, "l_small_toe": 0.02, "l_heel": 0.03,
    "r_big_toe": 0.02, "r_small_toe": 0.02, "r_heel": 0.03,
}


def _limb(dirv, bend, pole):
    return {"dir": V(*dirv), "bend": float(bend), "pole": V(*pole)}


def default_params() -> dict:
    return {
        "pelvis_y": STAND_PELVIS_Y,
        "pitch": 0.0, "roll": 0.0,
        "spine_pitch": 0.0, "spine_yaw": 0.0, "spine_roll": 0.0,
        "head_pitch": 0.0, "head_yaw": 0.0,
        "legs": {
            "L": _limb((0.02, -1.0, 0.0), 0.0, (0, 0, 1)),
            "R": _limb((-0.02, -1.0, 0.0), 0.0, (0, 0, 1)),
        },
        "arms": {
            "L": _limb((-0.25, -1.0, 0.06), 12.0, (0, 0.2, 1)),
            "R": _limb((0.25, -1.0, 0.06), 12.0, (0, 0.2, 1)),
        },
        "feet": {"L": "auto", "R": "auto"},   # auto | flat | follow
        "hand_style": {"L": "relaxed", "R": "relaxed"},
        "hand_aim": {"L": None, "R": None},        # world-space aim override
        "hand_aim_local": {"L": None, "R": None},  # yaw-local aim (presets)
    }


# ---------------------------------------------------------------- pose presets
# Each preset is a partial parameter update; unspecified values keep defaults.
# Limb "dir" vectors are hip->ankle / shoulder->wrist chord directions in the
# pelvis / chest frame respectively.

def _presets():
    P = {}
    P["stand"] = {}
    P["tpose"] = {"arms": {
        "L": _limb((-1, 0, 0), 0, (0, 0, 1)), "R": _limb((1, 0, 0), 0, (0, 0, 1))}}
    P["sit"] = {  # on the ground, legs forward, knees eased
        "pelvis_y": 0.10, "spine_pitch": 8,
        "legs": {"L": _limb((0.06, -0.10, 1.0), 35, (0, 1, 0.2)),
                 "R": _limb((-0.06, -0.10, 1.0), 35, (0, 1, 0.2))},
    }
    P["sit_chair"] = {  # pelvis height set by the seat; thighs forward, shins down
        "spine_pitch": 4,
        "legs": {"L": _limb((0.05, -0.70, 0.70), 85, (0, 1, 0.5)),
                 "R": _limb((-0.05, -0.70, 0.70), 85, (0, 1, 0.5))},
        "arms": {"L": _limb((-0.2, -1.0, 0.25), 30, (0, 0.2, 1)),
                 "R": _limb((0.2, -1.0, 0.25), 30, (0, 0.2, 1))},
    }
    P["cross_legged"] = {
        "pelvis_y": 0.14, "spine_pitch": 10,
        "legs": {"L": _limb((0.85, -0.35, 0.55), 130, (0, 1, 0.6)),
                 "R": _limb((-0.85, -0.35, 0.55), 130, (0, 1, 0.6))},
    }
    P["squat"] = {
        "pelvis_y": 0.32, "spine_pitch": 22,
        "legs": {"L": _limb((0.25, -0.95, 0.18), 125, (0, 0.3, 1)),
                 "R": _limb((-0.25, -0.95, 0.18), 125, (0, 0.3, 1))},
        "arms": {"L": _limb((-0.15, -0.5, 0.85), 25, (0, 1, 0.3)),
                 "R": _limb((0.15, -0.5, 0.85), 25, (0, 1, 0.3))},
    }
    P["crouch"] = {
        "pelvis_y": 0.45, "spine_pitch": 38,
        "legs": {"L": _limb((0.15, -1.0, 0.10), 95, (0, 0.3, 1)),
                 "R": _limb((-0.15, -1.0, 0.10), 95, (0, 0.3, 1))},
    }
    P["kneel"] = {  # both knees down, torso upright, shins back along the ground
        "pelvis_y": DIMS["thigh"] + 0.06,
        "legs": {"L": _limb((0.10, -0.55, -0.80), 95, (0, -0.3, -1)),
                 "R": _limb((-0.10, -0.55, -0.80), 95, (0, -0.3, -1))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["kneel_one"] = {  # right knee down, left foot planted (proposal / start pose)
        "pelvis_y": DIMS["thigh"] + 0.08,
        "legs": {"L": _limb((0.30, -0.85, 0.42), 100, (0, 0.5, 1)),
                 "R": _limb((-0.10, -0.55, -0.80), 95, (0, -0.3, -1))},
        "feet": {"R": "follow"},
    }
    P["lie_back"] = {
        "pelvis_y": 0.13, "pitch": -90,
        "legs": {"L": _limb((0.05, -1.0, 0.0), 4, (0, 0, 1)),
                 "R": _limb((-0.05, -1.0, 0.0), 4, (0, 0, 1))},
        "arms": {"L": _limb((-0.35, -1.0, 0.0), 10, (0, 0, 1)),
                 "R": _limb((0.35, -1.0, 0.0), 10, (0, 0, 1))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["lie_front"] = dict(P["lie_back"], pitch=90, pelvis_y=0.13)
    P["lie_side_left"] = dict(P["lie_back"], pitch=0, roll=-90, pelvis_y=0.16)
    P["lie_side_right"] = dict(P["lie_back"], pitch=0, roll=90, pelvis_y=0.16)
    P["sprawl"] = {  # collapsed on the back, limbs flung out - defeat pose
        "pelvis_y": 0.13, "pitch": -90,
        "legs": {"L": _limb((0.45, -0.95, 0.0), 25, (0, 0.4, 1)),
                 "R": _limb((-0.35, -0.9, 0.15), 45, (0, 0.4, 1))},
        "arms": {"L": _limb((-0.9, -0.45, 0.0), 20, (0, 0.5, 1)),
                 "R": _limb((0.95, 0.25, 0.0), 35, (0, 0.5, 1))},
        "head_yaw": 20, "feet": {"L": "follow", "R": "follow"},
    }
    P["fetal"] = {
        "pelvis_y": 0.16, "roll": 90, "spine_pitch": 40,
        "legs": {"L": _limb((0.05, -0.55, 0.75), 120, (0, 1, 0.4)),
                 "R": _limb((-0.05, -0.55, 0.75), 120, (0, 1, 0.4))},
        "arms": {"L": _limb((-0.1, -0.6, 0.75), 110, (0, 0.4, 1)),
                 "R": _limb((0.1, -0.6, 0.75), 110, (0, 0.4, 1))},
        "head_pitch": 30, "feet": {"L": "follow", "R": "follow"},
    }
    P["all_fours"] = {  # hands and knees
        "hand_aim_local": {"L": ((0, 0, 1), (0, -1, 0)), "R": ((0, 0, 1), (0, -1, 0))},
        "hand_style": {"L": "open", "R": "open"},
        "pelvis_y": DIMS["thigh"] + 0.06, "pitch": 78,
        "legs": {"L": _limb((0.08, -0.55, -0.80), 95, (0, -0.3, -1)),
                 "R": _limb((-0.08, -0.55, -0.80), 95, (0, -0.3, -1))},
        "arms": {"L": _limb((-0.10, -1.0, 0.05), 4, (0, 0, 1)),
                 "R": _limb((0.10, -1.0, 0.05), 4, (0, 0, 1))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["plank"] = {
        "hand_aim_local": {"L": ((0, 0, 1), (0, -1, 0)), "R": ((0, 0, 1), (0, -1, 0))},
        "hand_style": {"L": "open", "R": "open"},
        "pelvis_y": 0.52, "pitch": 84,
        "legs": {"L": _limb((0.05, -0.15, -1.0), 4, (0, -1, 0)),
                 "R": _limb((-0.05, -0.15, -1.0), 4, (0, -1, 0))},
        "arms": {"L": _limb((-0.10, -1.0, 0.05), 4, (0, 0, 1)),
                 "R": _limb((0.10, -1.0, 0.05), 4, (0, 0, 1))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["handstand"] = {
        "hand_aim_local": {"L": ((0, 0, 1), (0, -1, 0)), "R": ((0, 0, 1), (0, -1, 0))},
        "hand_style": {"L": "open", "R": "open"},
        # Body pitched 180: local +Y points at the ground, so "arms overhead"
        # (local +Y) reach down to it and straight legs (local -Y) point up.
        "pelvis_y": ARM_LEN + DIMS["spine"] + 0.12, "pitch": 180,
        "legs": {"L": _limb((0.06, -1.0, 0.0), 4, (0, 0, 1)),
                 "R": _limb((-0.06, -1.0, 0.0), 4, (0, 0, 1))},
        "arms": {"L": _limb((-0.12, 1.0, 0.0), 2, (0, 0, 1)),
                 "R": _limb((0.12, 1.0, 0.0), 2, (0, 0, 1))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["bridge"] = {  # yoga wheel: belly up, hands and feet on the ground.
        # Pelvis pitched -90 (face up): local -Z maps to world "down" and the
        # torso (local +Y) runs along the ground toward -Z, so legs fold down
        # in front (-Z local) and arms plant beyond the head (+Y local).
        "hand_aim_local": {"L": ((0, 0, -1), (0, -1, 0)), "R": ((0, 0, -1), (0, -1, 0))},
        "hand_style": {"L": "open", "R": "open"},
        "pelvis_y": 0.50, "pitch": -90, "spine_pitch": -22,
        "legs": {"L": _limb((0.10, -0.72, -0.66), 75, (0, 0, 1)),
                 "R": _limb((-0.10, -0.72, -0.66), 75, (0, 0, 1))},
        "arms": {"L": _limb((-0.12, 0.42, -0.78), 20, (0, 0, 1)),
                 "R": _limb((0.12, 0.42, -0.78), 20, (0, 0, 1))},
        "head_pitch": -30, "feet": {"L": "follow", "R": "follow"},
    }
    P["lunge"] = {
        "pelvis_y": 0.55,
        "legs": {"L": _limb((0.10, -0.72, 0.60), 95, (0, 1, 0.5)),
                 "R": _limb((-0.10, -0.55, -0.85), 45, (0, -0.4, -1))},
        "feet": {"R": "follow"},
    }
    P["split"] = {
        "pelvis_y": 0.16,
        "legs": {"L": _limb((0.06, -0.18, 1.0), 4, (0, 1, 0)),
                 "R": _limb((-0.06, -0.18, -1.0), 4, (0, -1, 0))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["run"] = {
        "pelvis_y": STAND_PELVIS_Y - 0.04, "spine_pitch": 12,
        "legs": {"L": _limb((0.06, -0.75, 0.55), 65, (0, 1, 0.5)),
                 "R": _limb((-0.06, -0.80, -0.50), 80, (0, -0.5, -1))},
        "arms": {"L": _limb((-0.15, -0.55, -0.65), 95, (0, 0.3, 1)),
                 "R": _limb((0.15, -0.45, 0.70), 95, (0, 0.6, 1))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["walk"] = {
        "spine_pitch": 4,
        "legs": {"L": _limb((0.04, -0.95, 0.28), 18, (0, 1, 0.5)),
                 "R": _limb((-0.04, -0.95, -0.26), 30, (0, 0.5, 1))},
        "arms": {"L": _limb((-0.2, -0.95, -0.25), 20, (0, 0.2, 1)),
                 "R": _limb((0.2, -0.95, 0.25), 20, (0, 0.2, 1))},
        "feet": {"R": "follow"},
    }
    P["jump"] = {
        "pelvis_y": STAND_PELVIS_Y + 0.35, "spine_pitch": 6,
        "legs": {"L": _limb((0.08, -0.75, 0.30), 95, (0, 1, 0.4)),
                 "R": _limb((-0.08, -0.75, 0.30), 95, (0, 1, 0.4))},
        "arms": {"L": _limb((-0.55, 0.80, 0.10), 15, (-1, 0, 0.3)),
                 "R": _limb((0.55, 0.80, 0.10), 15, (1, 0, 0.3))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["cower"] = {
        "pelvis_y": 0.30, "spine_pitch": 55,
        "legs": {"L": _limb((0.18, -0.95, 0.12), 120, (0, 0.3, 1)),
                 "R": _limb((-0.18, -0.95, 0.12), 120, (0, 0.3, 1))},
        "arms": {"L": _limb((-0.15, 0.55, 0.65), 130, (0, 1, 0)),
                 "R": _limb((0.15, 0.55, 0.65), 130, (0, 1, 0))},
        "head_pitch": 35,
    }
    P["straddle"] = {  # legs astride something (broom, bench, horse-style)
        "spine_pitch": 12,
        "legs": {"L": _limb((0.55, -0.80, 0.28), 70, (0.6, 0.6, 0.5)),
                 "R": _limb((-0.55, -0.80, 0.28), 70, (-0.6, 0.6, 0.5))},
        "feet": {"L": "follow", "R": "follow"},
    }
    P["tree_pose"] = {  # yoga: stand on left leg, right sole to inner thigh
        "legs": {"L": _limb((0.02, -1.0, 0.0), 0, (0, 0, 1)),
                 "R": _limb((-0.95, -0.75, 0.05), 135, (-1, 0.3, 0.3))},
        "arms": {"L": _limb((-0.10, 0.98, 0.05), 18, (-1, 0, 0.2)),
                 "R": _limb((0.10, 0.98, 0.05), 18, (1, 0, 0.2))},
        "feet": {"R": "follow"},
    }
    P["warrior"] = {  # warrior II
        "pelvis_y": 0.72,
        "legs": {"L": _limb((0.72, -0.65, 0.0), 80, (1, 0.6, 0)),
                 "R": _limb((-0.80, -0.60, 0.0), 8, (0, 0, 1))},
        "arms": {"L": _limb((-1.0, 0.02, 0.0), 2, (0, 0, 1)),
                 "R": _limb((1.0, 0.02, 0.0), 2, (0, 0, 1))},
        "head_yaw": -50, "feet": {"L": "follow", "R": "follow"},
    }
    return P


POSE_PRESETS = _presets()


def merge_params(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if k in ("legs", "arms", "feet", "hand_style", "hand_aim",
                 "hand_aim_local") and isinstance(v, dict):
            for side, sv in v.items():
                out[k][side] = copy.deepcopy(sv)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Human:
    def __init__(self, name: str, scale: float = 1.0):
        self.name = name
        self.kind = "HUMAN"
        self.scale = scale
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.params = default_params()
        self.params["pelvis_y"] = STAND_PELVIS_Y * scale
        self.pose_name = "stand"
        self.ik_constraints = []   # applied in order after FK; see solver.py
        self.look_target = None    # ("entity"|"point"|"dir", value)
        self.placed = False
        self.grounded = True       # snap lowest point to the ground on evaluate
        self.joints: dict[str, np.ndarray] = {}
        self.frames: dict[str, np.ndarray] = {}
        self.hand_pts = {"L": None, "R": None}
        self.warnings: list[str] = []

    # -------------------------------------------------------------- helpers
    def d(self, key: str) -> float:
        return DIMS[key] * self.scale

    def set_pose(self, name: str):
        if name not in POSE_PRESETS:
            raise ValueError(f"unknown pose '{name}' (choose from {sorted(POSE_PRESETS)})")
        keep = self.params
        self.params = merge_params(default_params(), POSE_PRESETS[name])
        # Preserve hand styles/aims set by earlier commands only when the
        # preset does not define its own.
        for k in ("hand_style", "hand_aim", "hand_aim_local"):
            if k not in POSE_PRESETS[name]:
                self.params[k] = keep[k]
        if "pelvis_y" not in POSE_PRESETS[name]:
            self.params["pelvis_y"] = STAND_PELVIS_Y
        self.params["pelvis_y"] *= self.scale
        self.pose_name = name

    def pelvis_frame(self) -> np.ndarray:
        return rot_y(self.yaw) @ rot_x(self.params["pitch"]) @ rot_z(self.params["roll"])

    def forward(self) -> np.ndarray:
        return rot_y(self.yaw) @ V(0, 0, 1)

    def right(self) -> np.ndarray:
        return rot_y(self.yaw) @ V(1, 0, 0)

    def bounding_radius(self) -> float:
        return 0.26 * self.scale

    # ------------------------------------------------------------------- FK
    def fk(self):
        """Compute joints/frames from params (ignores IK constraints; the
        solver applies those on top)."""
        s = self.scale
        p = self.params
        pelvis = V(self.x, p["pelvis_y"], self.z)
        PF = self.pelvis_frame()
        CF = PF @ rot_y(p["spine_yaw"]) @ rot_x(p["spine_pitch"]) @ rot_z(p["spine_roll"])
        HF = CF @ rot_y(p["head_yaw"]) @ rot_x(p["head_pitch"])

        J = {}
        F = {"pelvis": PF, "chest": CF, "head": HF}
        J["mid_hip"] = pelvis
        neck = pelvis + CF @ V(0, DIMS["spine"] * s, 0)
        J["neck"] = neck
        for side, sgn in (("L", -1), ("R", 1)):
            pre = side.lower() + "_"
            hip = pelvis + PF @ V(sgn * DIMS["hip_half"] * s, -0.02 * s, 0)
            J[pre + "hip"] = hip
            leg = p["legs"][side]
            chord = chord_for_bend(self.d("thigh"), self.d("shin"), leg["bend"])
            target = hip + (PF @ normalize(leg["dir"])) * chord
            pole = PF @ normalize(leg["pole"])
            knee, ankle, _ = two_bone_ik(hip, target, self.d("thigh"), self.d("shin"), pole)
            J[pre + "knee"], J[pre + "ankle"] = knee, ankle

            sh = neck + CF @ V(sgn * DIMS["shoulder_half"] * s, -DIMS["shoulder_drop"] * s, 0)
            J[pre + "shoulder"] = sh
            arm = p["arms"][side]
            chord = chord_for_bend(self.d("upper_arm"), self.d("forearm"), arm["bend"])
            target = sh + (CF @ normalize(arm["dir"])) * chord
            pole = CF @ normalize(arm["pole"])
            elbow, wrist, _ = two_bone_ik(sh, target, self.d("upper_arm"), self.d("forearm"), pole)
            J[pre + "elbow"], J[pre + "wrist"] = elbow, wrist

        head_c = neck + HF @ V(0, DIMS["neck_head"] * s, 0.01 * s)
        F["head_center"] = head_c
        J["nose"] = head_c + HF @ V(0, -0.02 * s, 0.105 * s)
        J["r_eye"] = head_c + HF @ V(0.033 * s, 0.03 * s, 0.095 * s)
        J["l_eye"] = head_c + HF @ V(-0.033 * s, 0.03 * s, 0.095 * s)
        J["r_ear"] = head_c + HF @ V(0.075 * s, 0.015 * s, 0.005 * s)
        J["l_ear"] = head_c + HF @ V(-0.075 * s, 0.015 * s, 0.005 * s)

        self.joints, self.frames = J, F
        self._solve_feet()
        return J

    def _solve_feet(self):
        """Place heel/toe points once ankles are final. Called again by the
        solver after IK moves a leg."""
        s = self.scale
        J = self.joints
        fwd = self.forward()
        rightv = self.right()
        for side, sgn in (("L", -1), ("R", 1)):
            pre = side.lower() + "_"
            ankle = J[pre + "ankle"]
            knee = J[pre + "knee"]
            mode = self.params["feet"].get(side, "auto")
            near_ground = ankle[1] < (DIMS["ankle_h"] * s + 0.06)
            if mode == "flat" or (mode == "auto" and near_ground):
                f = normalize(horizontal(fwd), fallback=[0, 0, 1])
                down = V(0, -min(ankle[1], DIMS["ankle_h"] * s) + 0.02 * s, 0)
                J[pre + "big_toe"] = ankle + f * self.d("foot_len") + down
                J[pre + "heel"] = ankle - f * self.d("heel_len") + down
            else:
                shin = normalize(ankle - knee, fallback=[0, -1, 0])
                # Toes continue past the ankle, biased away from the shin.
                f = normalize(shin + horizontal(fwd) * 0.7 + V(0, -0.35, 0))
                J[pre + "big_toe"] = ankle + f * self.d("foot_len")
                J[pre + "heel"] = ankle - f * self.d("heel_len")
            out = rightv * (0.055 * s * sgn)
            J[pre + "small_toe"] = J[pre + "big_toe"] * 0.82 + (ankle + out) * 0.18 + out * 0.5

    def hand_frame(self, side: str) -> np.ndarray:
        """Current hand frame for a side, honoring any aim override."""
        pre = side.lower() + "_"
        wrist = self.joints[pre + "wrist"]
        elbow = self.joints[pre + "elbow"]
        aim = self.params["hand_aim"][side]
        aim_local = self.params["hand_aim_local"][side]
        if aim is not None:
            return hands.aimed_hand_frame(np.array(aim[0]), np.array(aim[1]))
        if aim_local is not None:
            R = rot_y(self.yaw)
            return hands.aimed_hand_frame(R @ np.array(aim_local[0]),
                                          R @ np.array(aim_local[1]))
        return hands.default_hand_frame(elbow, wrist, self.right(), side)

    def solve_hands(self):
        """Build 21-point hands from wrists + current hand styles/aims."""
        for side in ("L", "R"):
            pre = side.lower() + "_"
            wrist = self.joints[pre + "wrist"]
            hf = self.hand_frame(side)
            style = self.params["hand_style"][side]
            self.hand_pts[side] = hands.hand_keypoints(wrist, hf, side, style, self.scale)

    # ------------------------------------------------------------- reach data
    def limb_root(self, limb: str) -> np.ndarray:
        base = {"arm_L": "l_shoulder", "arm_R": "r_shoulder",
                "leg_L": "l_hip", "leg_R": "r_hip"}[limb]
        return self.joints[base]

    def limb_lengths(self, limb: str):
        if limb.startswith("arm"):
            return self.d("upper_arm"), self.d("forearm")
        return self.d("thigh"), self.d("shin")

    def body25(self) -> np.ndarray:
        return np.array([self.joints[n] for n in BODY_25_NAMES])

    def lowest_point(self) -> float:
        lows = [self.joints[n][1] - JOINT_RADII[n] * self.scale for n in BODY_25_NAMES]
        for side in ("L", "R"):
            if self.hand_pts[side] is not None:
                lows.append(float(np.min(self.hand_pts[side][:, 1])) - 0.008)
        return min(lows)

    # --------------------------------------------------------- named parts
    def part_point(self, part: str):
        """Resolve a body-part name to (center, surface_radius).

        Contact targets take the returned center +/- radius along the approach
        direction, so touches land on the skin, not inside it.
        """
        s = self.scale
        J = self.joints
        HF = self.frames["head"]
        hc = self.frames["head_center"]
        CF = self.frames["chest"]
        r = 0.02 * s
        simple = {
            "head": (hc, DIMS["head_r"] * s),
            "face": (hc + HF @ V(0, 0, 0.08 * s), 0.03 * s),
            "forehead": (hc + HF @ V(0, 0.05 * s, 0.085 * s), r),
            "nose": (J["nose"], 0.012 * s),
            "chin": (hc + HF @ V(0, -0.08 * s, 0.07 * s), r),
            "jaw": (hc + HF @ V(0, -0.07 * s, 0.06 * s), 0.03 * s),
            "left_cheek": (hc + HF @ V(-0.055 * s, -0.02 * s, 0.07 * s), r),
            "right_cheek": (hc + HF @ V(0.055 * s, -0.02 * s, 0.07 * s), r),
            "left_ear": (J["l_ear"], r),
            "right_ear": (J["r_ear"], r),
            "left_eye": (J["l_eye"], 0.012 * s),
            "right_eye": (J["r_eye"], 0.012 * s),
            "top_of_head": (hc + HF @ V(0, 0.10 * s, 0), r),
            "neck": (J["neck"] + CF @ V(0, 0.03 * s, 0), 0.05 * s),
            "throat": (J["neck"] + CF @ V(0, 0.04 * s, 0.05 * s), r),
            "chest": (J["neck"] + CF @ V(0, -0.14 * s, 0.10 * s), 0.03 * s),
            "back": (J["neck"] * 0.6 + J["mid_hip"] * 0.4 + CF @ V(0, 0, -0.11 * s), 0.03 * s),
            "upper_back": (J["neck"] + CF @ V(0, -0.06 * s, -0.10 * s), 0.03 * s),
            "lower_back": (J["mid_hip"] + self.pelvis_frame() @ V(0, 0.06 * s, -0.11 * s), 0.03 * s),
            "belly": (J["mid_hip"] * 0.55 + J["neck"] * 0.45 + CF @ V(0, 0, 0.11 * s), 0.03 * s),
            "stomach": (J["mid_hip"] * 0.55 + J["neck"] * 0.45 + CF @ V(0, 0, 0.11 * s), 0.03 * s),
            "waist": (J["mid_hip"] + self.pelvis_frame() @ V(0, 0.10 * s, 0), 0.13 * s),
            "hips": (J["mid_hip"], 0.13 * s),
            "butt": (J["mid_hip"] + self.pelvis_frame() @ V(0, -0.03 * s, -0.10 * s), 0.04 * s),
            "groin": (J["mid_hip"] + self.pelvis_frame() @ V(0, -0.08 * s, 0.03 * s), 0.04 * s),
        }
        if part in simple:
            return simple[part]
        # Side-qualified limb parts.
        for side, sgn, pre in (("left", -1, "l_"), ("right", 1, "r_")):
            if not part.startswith(side + "_"):
                continue
            sub = part[len(side) + 1:]
            side_key = "L" if side == "left" else "R"
            wrist = J[pre + "wrist"]
            handf = self.hand_frame(side_key)
            hand_center = wrist + handf[:, 2] * 0.09 * s
            m = {
                "shoulder": (J[pre + "shoulder"], 0.06 * s),
                "upper_arm": ((J[pre + "shoulder"] + J[pre + "elbow"]) / 2, 0.045 * s),
                "elbow": (J[pre + "elbow"], 0.045 * s),
                "forearm": ((J[pre + "elbow"] + wrist) / 2, 0.04 * s),
                "wrist": (wrist, 0.035 * s),
                "hand": (hand_center, 0.04 * s),
                "palm": (hand_center, 0.035 * s),
                "fist": (hand_center, 0.045 * s),
                "fingers": (wrist + handf[:, 2] * 0.155 * s, 0.015 * s),
                "fingertip": (wrist + handf[:, 2] * 0.165 * s, 0.01 * s),
                "thumb": (wrist + (handf[:, 2] * 0.06 + handf[:, 0] * sgn * 0.05) * s, 0.014 * s),
                "hip": (J[pre + "hip"], 0.09 * s),
                "thigh": ((J[pre + "hip"] + J[pre + "knee"]) / 2, 0.075 * s),
                "inner_thigh": (J[pre + "hip"] * 0.45 + J[pre + "knee"] * 0.55
                                - self.right() * sgn * 0.06 * s, 0.02 * s),
                "knee": (J[pre + "knee"], 0.055 * s),
                "shin": ((J[pre + "knee"] + J[pre + "ankle"]) / 2, 0.05 * s),
                "calf": ((J[pre + "knee"] + J[pre + "ankle"]) / 2, 0.055 * s),
                "ankle": (J[pre + "ankle"], 0.045 * s),
                "foot": ((J[pre + "ankle"] + J[pre + "big_toe"]) / 2, 0.045 * s),
                "toes": (J[pre + "big_toe"], 0.025 * s),
                "heel": (J[pre + "heel"], 0.03 * s),
                "sole": ((J[pre + "heel"] + J[pre + "big_toe"]) / 2 + V(0, -0.015 * s, 0),
                         0.01 * s),
            }
            if sub in m:
                return m[sub]
        raise ValueError(f"unknown body part '{part}' on {self.name}")


PLURAL_PARTS = {
    "hands": ("left_hand", "right_hand"),
    "palms": ("left_palm", "right_palm"),
    "feet": ("left_foot", "right_foot"),
    "knees": ("left_knee", "right_knee"),
    "arms": ("left_arm", "right_arm"),
    "shoulders": ("left_shoulder", "right_shoulder"),
}
