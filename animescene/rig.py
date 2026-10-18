"""Parametric anime-proportioned humanoid rig.

A pose is solved in the character's *local* frame (x = character's right,
y = up, z = forward, origin on the ground under the pelvis) from a compact
`PoseParams` description: torso angles plus world-space-like targets for the
hands and feet, solved with analytic two-bone IK. `Rig.solve` returns the
local joint positions; `Placement` maps them into the world.

Bone lengths are always exact: IK never stretches a limb, an unreachable
target just straightens the limb towards it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vec import UP, rot_axis, unit, v

# Build multipliers for limb / torso thickness.
BUILDS = {
    "petite": 0.86,
    "slim": 0.93,
    "athletic": 1.0,
    "muscular": 1.16,
    "heavy": 1.3,
}


@dataclass
class Body:
    height: float = 1.65
    build: str = "athletic"
    gender: str = "female"  # "female" | "male"

    def __post_init__(self):
        if self.build not in BUILDS:
            raise ValueError(f"unknown build {self.build!r}; choose from {sorted(BUILDS)}")
        H = self.height
        m = self.gender == "male"
        # Anime proportions: ~7 heads tall, long legs.
        self.hip_h = 0.515 * H
        self.hip_w = (0.058 if m else 0.062) * H
        self.thigh = 0.245 * H
        self.shin = 0.235 * H
        self.ankle_h = 0.035 * H
        self.foot = 0.13 * H
        self.spine = 0.18 * H  # pelvis centre -> chest joint
        self.upper = 0.11 * H  # chest joint -> neck base
        self.neck = 0.045 * H
        self.head_r = 0.066 * H
        self.shoulder_w = (0.115 if m else 0.098) * H
        self.upper_arm = 0.17 * H
        self.forearm = 0.15 * H
        self.hand = 0.095 * H
        k = BUILDS[self.build]
        self.thick = k
        self.r = {
            "neck": 0.024 * H * k,
            "chest": (0.075 if m else 0.068) * H * k,
            "waist": (0.062 if m else 0.052) * H * k,
            "pelvis": (0.064 if m else 0.068) * H * k,
            "upper_arm": 0.028 * H * k,
            "forearm": 0.023 * H * k,
            "hand": 0.024 * H * k,
            "thigh": 0.047 * H * k,
            "shin": 0.032 * H * k,
            "foot": 0.022 * H * k,
        }

    @property
    def arm_reach(self) -> float:
        return self.upper_arm + self.forearm

    @property
    def leg_len(self) -> float:
        return self.thigh + self.shin


@dataclass
class Limb:
    """Target for a hand or foot. `target` is local unless `world` is set."""

    target: np.ndarray
    pole: np.ndarray  # direction the elbow / knee should point to (local)
    tip_dir: np.ndarray | None = None  # direction of hand/foot (local); None = follow limb


@dataclass
class PoseParams:
    pelvis: np.ndarray = None  # local pelvis centre
    pelvis_rot: tuple = (0.0, 0.0, 0.0)  # yaw, pitch(+forward), roll(+to right side) degrees
    chest_rot: tuple = (0.0, 0.0, 0.0)  # relative to pelvis
    head_rot: tuple = (0.0, 0.0, 0.0)  # relative to chest
    look_at: np.ndarray | None = None  # local point the head turns to (overrides head yaw/pitch)
    hands: dict = field(default_factory=dict)  # "l"/"r" -> Limb
    feet: dict = field(default_factory=dict)
    grounded: bool = True
    fist: dict = field(default_factory=lambda: {"l": True, "r": True})


def euler(yaw, pitch, roll) -> np.ndarray:
    """Yaw about local up, then pitch about local right (+ bends forward), then roll."""
    Ry = rot_axis(UP, np.deg2rad(yaw))
    # +pitch should tilt the up vector toward +z (forward): rotate about +x by +pitch.
    Rx = rot_axis(v(1, 0, 0), np.deg2rad(pitch))
    # +roll tilts up toward the character's right (-x in local frame? no: local x IS right)
    Rz = rot_axis(v(0, 0, 1), -np.deg2rad(roll))
    return Ry @ Rx @ Rz


def two_bone_ik(root, target, la, lb, pole):
    """Return (mid, end) for a two-bone chain reaching for target with a pole hint."""
    d = target - root
    dist = np.linalg.norm(d)
    dirn = unit(d, fallback=(0, -1, 0))
    dist = float(np.clip(dist, abs(la - lb) + 1e-4, la + lb - 1e-5))
    # angle at root between chain direction and first bone
    cos_a = (la * la + dist * dist - lb * lb) / (2 * la * dist)
    a = np.arccos(np.clip(cos_a, -1, 1))
    pole_perp = pole - dirn * np.dot(pole, dirn)
    if np.linalg.norm(pole_perp) < 1e-6:
        alt = v(0, 0, 1) if abs(dirn[2]) < 0.9 else v(1, 0, 0)
        pole_perp = alt - dirn * np.dot(alt, dirn)
    pole_perp = unit(pole_perp)
    mid = root + la * (np.cos(a) * dirn + np.sin(a) * pole_perp)
    end = root + dirn * dist
    return mid, end


JOINTS = [
    "pelvis", "spine", "chest", "neck", "head", "head_top", "nose",
    "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "l_elbow", "l_wrist", "l_hand",
    "r_shoulder", "r_elbow", "r_wrist", "r_hand",
    "l_hip", "l_knee", "l_ankle", "l_toe",
    "r_hip", "r_knee", "r_ankle", "r_toe",
]


class Rig:
    def __init__(self, body: Body):
        self.b = body

    # ---- rest helpers used by actions -------------------------------------------------
    def rest_pelvis(self) -> np.ndarray:
        return v(0, self.b.hip_h + 0.03 * self.b.height, 0)

    def solve(self, p: PoseParams) -> dict:
        b = self.b
        J: dict[str, np.ndarray] = {}
        pel = self.rest_pelvis() if p.pelvis is None else np.asarray(p.pelvis, float)
        Rp = euler(*p.pelvis_rot)
        Rc = Rp @ euler(*p.chest_rot)
        J["pelvis"] = pel
        J["spine"] = pel + Rp @ v(0, b.spine * 0.5, 0)
        J["chest"] = J["spine"] + Rc @ v(0, b.spine * 0.5, 0)
        J["neck"] = J["chest"] + Rc @ v(0, b.upper, 0)
        if p.look_at is not None:
            d = np.asarray(p.look_at, float) - J["neck"]
            # express look direction in chest frame, clamp to a human range
            dl = Rc.T @ d
            yaw = np.rad2deg(np.arctan2(dl[0], dl[2]))
            pitch = -np.rad2deg(np.arctan2(dl[1], np.hypot(dl[0], dl[2])))
            yaw = float(np.clip(yaw, -75, 75))
            pitch = float(np.clip(pitch, -50, 60))
            Rh = Rc @ euler(yaw, pitch, p.head_rot[2])
        else:
            Rh = Rc @ euler(*p.head_rot)
        self.head_R = Rh
        J["head"] = J["neck"] + Rh @ v(0, b.neck + b.head_r * 0.85, 0.01 * b.height)
        J["head_top"] = J["head"] + Rh @ v(0, b.head_r, 0)
        J["nose"] = J["head"] + Rh @ v(0, -0.15 * b.head_r, 0.98 * b.head_r)
        # local x is the character's right
        J["r_eye"] = J["head"] + Rh @ v(0.38 * b.head_r, 0.1 * b.head_r, 0.85 * b.head_r)
        J["l_eye"] = J["head"] + Rh @ v(-0.38 * b.head_r, 0.1 * b.head_r, 0.85 * b.head_r)
        J["r_ear"] = J["head"] + Rh @ v(0.95 * b.head_r, 0.0, -0.05 * b.head_r)
        J["l_ear"] = J["head"] + Rh @ v(-0.95 * b.head_r, 0.0, -0.05 * b.head_r)

        for s, sx in (("r", 1.0), ("l", -1.0)):
            sh = J["neck"] + Rc @ v(sx * b.shoulder_w, -0.025 * b.height, -0.01 * b.height)
            J[f"{s}_shoulder"] = sh
            limb = p.hands.get(s)
            if limb is None:  # relaxed arm hanging, slightly out
                tgt = sh + Rc @ v(sx * 0.06 * b.height, -b.arm_reach * 0.97, 0.02 * b.height)
                limb = Limb(tgt, Rc @ v(0, 0, -1))
            el, wr = two_bone_ik(sh, np.asarray(limb.target, float), b.upper_arm, b.forearm,
                                 np.asarray(limb.pole, float))
            J[f"{s}_elbow"], J[f"{s}_wrist"] = el, wr
            hd = unit(wr - el) if limb.tip_dir is None else unit(np.asarray(limb.tip_dir, float))
            hl = b.hand * (0.55 if p.fist.get(s, True) else 1.0)
            J[f"{s}_hand"] = wr + hd * hl

            hip = pel + Rp @ v(sx * b.hip_w, -0.035 * b.height, 0)
            J[f"{s}_hip"] = hip
            foot = p.feet.get(s)
            if foot is None:
                tgt = v(sx * (b.hip_w + 0.01), b.ankle_h, 0.0)
                foot = Limb(tgt, v(0, 0, 1))
            kn, an = two_bone_ik(hip, np.asarray(foot.target, float), b.thigh, b.shin,
                                 np.asarray(foot.pole, float))
            J[f"{s}_knee"], J[f"{s}_ankle"] = kn, an
            if foot.tip_dir is not None:
                fd = unit(np.asarray(foot.tip_dir, float))
            else:
                shin_dir = unit(an - kn)
                fwd = Rp @ v(0, 0, 1)
                fd = unit(fwd - shin_dir * np.dot(fwd, shin_dir))  # perpendicular to shin
                if p.grounded:
                    fd = unit(v(fd[0], min(fd[1], 0.0) * 0.2, fd[2]))
            J[f"{s}_toe"] = an + fd * b.foot
        return J
