"""Action library: each action turns (actor, target, params) into a solved pose.

Everything here is in the actor's local frame: x = actor's right, y = up,
z = actor's forward, origin on the ground below the rest pelvis. The scene
solver hands every action the target's joints (and weapon) already mapped
into this frame, so "punch the head" is literally "put the fist on that
point" and the result is exact in 3D.

An action returns an `ActionOut`: the IK description of the pose, how
held weapons are oriented, whether the body is airborne, its direction of
motion (used for motion-consistent prompting), the contact/impact point and
booru tags that describe it to the image model.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .rig import Body, Limb, PoseParams, Rig
from .vec import unit, v
from .weapons import Weapon


@dataclass
class ActionCtx:
    body: Body
    rig: Rig
    weapon: Weapon
    params: dict
    target: dict | None = None  # target joints in actor-local frame
    target_body: Body | None = None
    target_weapon: tuple | None = None  # (grip, tip) of target's weapon, actor-local
    altitude: float | None = None
    # world-space view of the target, for actions that mount themselves onto the target's body
    target_world: dict | None = None
    target_yaw: float = 0.0
    target_origin: np.ndarray | None = None
    to_local: Callable | None = None  # world point -> actor-local (current placement)
    origin: np.ndarray | None = None  # actor's current world placement

    def tw(self, name: str) -> np.ndarray:
        return self.target_world[name]

    @property
    def t_fwd(self) -> np.ndarray:
        return v(np.sin(self.target_yaw), 0, np.cos(self.target_yaw))

    @property
    def t_right(self) -> np.ndarray:  # target's own right (world)
        f = self.t_fwd
        return v(-f[2], 0, f[0])

    @property
    def H(self) -> float:
        return self.body.height

    def p(self, key, default=None):
        return self.params.get(key, default)

    def tj(self, name: str) -> np.ndarray:
        """Target joint (local), or a virtual opponent 1.1 m in front."""
        if self.target is not None:
            return self.target[name]
        virt = {"head": 0.9, "neck": 0.82, "chest": 0.72, "spine": 0.62, "pelvis": 0.55,
                "l_knee": 0.27, "r_knee": 0.27}
        return v(0, virt.get(name, 0.7) * 1.65, 1.1)

    def aim(self, default="head") -> np.ndarray:
        a = self.p("aim", default)
        names = {"head": "head", "face": "head", "chest": "chest", "body": "spine",
                 "stomach": "spine", "gut": "spine", "legs": "l_knee", "neck": "neck"}
        if a not in names:
            raise ValueError(f"unknown aim {a!r}; choose from {sorted(names)}")
        return self.tj(names[a])

    def target_dir(self) -> np.ndarray:
        t = self.tj("pelvis")
        return unit(v(t[0], 0, t[2]), fallback=(0, 0, 1))


@dataclass
class ActionOut:
    pose: PoseParams
    weapon_dirs: dict = field(default_factory=dict)  # hand -> blade direction (local)
    tags: list = field(default_factory=list)
    airborne: bool = False
    motion: np.ndarray = field(default_factory=lambda: v(0, 0, 0))
    impact: np.ndarray | None = None
    # contact the solver should make exact by stepping in/out: (effector joint, desired point, local)
    reach: tuple | None = None
    # (origin world, yaw radians): put the actor exactly here relative to the target's body
    mount: tuple | None = None
    turn: float | None = None  # world yaw the action needs (e.g. back to the target), origin kept
    # booru tags for a glowing effect painted around the impact point (regional prompt)
    effect: str = ""


ACTIONS: dict[str, Callable[[ActionCtx], ActionOut]] = {}
# Preferred horizontal pelvis-to-pelvis distance to the target, as f(ctx).
DISTANCE: dict[str, Callable[[ActionCtx], float]] = {}
# Does the action need the target's posed joints (so the target is solved first)?
NEEDS_TARGET_POSE: set[str] = set()
DESCRIPTIONS: dict[str, str] = {}


def action(name, distance=None, needs_pose=False, doc=""):
    def deco(fn):
        ACTIONS[name] = fn
        DESCRIPTIONS[name] = doc or (fn.__doc__ or "").strip().split("\n")[0]
        if distance is not None:
            DISTANCE[name] = distance
        if needs_pose:
            NEEDS_TARGET_POSE.add(name)
        return fn
    return deco


# ----------------------------------------------------------------------------- helpers
def _feet(c: ActionCtx, l, r, lpole=(0, 0, 1), rpole=(0, 0, 1)):
    H, ah = c.H, c.body.ankle_h
    return {"l": Limb(v(l[0] * H, ah, l[1] * H), v(*lpole)),
            "r": Limb(v(r[0] * H, ah, r[1] * H), v(*rpole))}


def _pel(c: ActionCtx, dx=0.0, dy=0.0, dz=0.0):
    p = c.rig.rest_pelvis()
    return p + v(dx, dy, dz) * c.H


def _hand(x, y, z, H, pole, tip=None):
    return Limb(v(x, y, z) * H, v(*pole), None if tip is None else v(*tip))


def _arm_reach_clamp(c: ActionCtx, side: str, point: np.ndarray, chest: np.ndarray, frac=0.98):
    """Pull a hand target back inside reach of an approximate shoulder position."""
    sx = 1.0 if side == "r" else -1.0
    sh = chest + v(sx * c.body.shoulder_w, 0.08 * c.H, 0)
    d = point - sh
    reach = c.body.arm_reach * frac
    n = np.linalg.norm(d)
    return point if n <= reach else sh + d / n * reach


def _approx_chest(c: ActionCtx, pel: np.ndarray, pitch_deg=0.0):
    a = np.deg2rad(pitch_deg)
    return pel + v(0, np.cos(a), np.sin(a)) * c.body.spine


def _armed(c: ActionCtx) -> bool:
    return c.weapon.reach_class in ("blade", "pole")


def _grip(c: ActionCtx, pose: PoseParams, grip: np.ndarray, blade: np.ndarray, out: ActionOut,
          main="r"):
    """Put the main hand on `grip`, orient the weapon, place the off hand on the handle."""
    w = c.weapon
    off = "l" if main == "r" else "r"
    blade = unit(blade)
    side = 1.0 if main == "r" else -1.0
    pose.hands[main] = Limb(grip, v(side, -0.6, -0.3), blade)
    out.weapon_dirs[main] = blade
    if w.kind in ("twin_daggers", "twin_swords"):
        return
    if w.two_handed:
        offset = 0.42 if w.reach_class == "pole" else -0.17
        pose.hands[off] = Limb(grip + blade * offset, v(-side, -0.6, -0.3), blade)


def _base_tags(c: ActionCtx) -> list:
    return [c.weapon.tags] if c.weapon.tags else []


# ----------------------------------------------------------------------------- actions
@action("stand", doc="Neutral standing, looking at the target (if any).")
def stand(c: ActionCtx) -> ActionOut:
    pose = PoseParams(look_at=c.tj("head") if c.target else None)
    pose.feet = _feet(c, (-0.065, 0.01), (0.065, -0.01))
    out = ActionOut(pose, tags=["standing"] + _base_tags(c))
    if _armed(c):  # weapon lowered at the side
        _grip(c, pose, v(0.16, 0.42, 0.06) * c.H, v(0.1, -0.35, 1.0), out)
        pose.hands.pop("l", None)
    return out


@action("stance", doc="Ready fighting stance facing the target; weapon at guard.")
def stance(c: ActionCtx) -> ActionOut:
    H = c.H
    pose = PoseParams(pelvis=_pel(c, 0, -0.06, 0), pelvis_rot=(22, 4, 0), chest_rot=(-14, 6, 0),
                      look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.07, 0.15), (0.13, -0.13), (0, 0.2, 1), (0.5, 0.2, 1))
    out = ActionOut(pose, tags=["fighting stance"] + _base_tags(c))
    if c.weapon.kind in ("katana", "sword", "greatsword"):
        _grip(c, pose, v(0.02, 0.58, 0.24) * H, v(0, 0.78, 0.62), out)
    elif c.weapon.kind in ("twin_daggers", "twin_swords"):
        pose.hands["r"] = _hand(0.1, 0.72, 0.22, H, (1, -1, 0))
        pose.hands["l"] = _hand(-0.08, 0.8, 0.2, H, (-1, -1, 0))
        out.weapon_dirs = {"r": unit(v(0.2, 0.6, 0.8)), "l": unit(v(-0.3, -0.5, 0.8))}
    elif c.weapon.reach_class == "pole":
        _grip(c, pose, v(0.1, 0.55, -0.02) * H, v(-0.1, 0.2, 1.0), out)
    elif c.weapon.kind == "magic":
        pose.hands["r"] = _hand(0.1, 0.7, 0.3, H, (1, -1, 0), tip=(0, 0.3, 1))
        pose.hands["l"] = _hand(-0.08, 0.68, 0.18, H, (-1, -1, 0), tip=(0, 0.6, 0.7))
        pose.fist = {"l": False, "r": False}
        out.tags.append("glowing hand")
    else:
        pose.hands["l"] = _hand(-0.03, 0.83, 0.2, H, (-0.4, -1, 0))
        pose.hands["r"] = _hand(0.07, 0.8, 0.12, H, (0.4, -1, 0))
        out.tags.append("clenched hands")
    return out


def _punch_distance(c: ActionCtx) -> float:
    H = c.H
    names = {"head": 0.88, "face": 0.88, "neck": 0.8, "chest": 0.72, "body": 0.62, "stomach": 0.62}
    tb = c.target_body
    th = (tb.height if tb else 1.65) * names.get(c.p("aim", "head"), 0.8)
    dy = th - 0.74 * H  # lowered shoulder height in the lunge
    reach = c.body.arm_reach * 0.97
    horiz = np.sqrt(max(reach * reach - dy * dy, (0.1 * H) ** 2))
    return float(horiz + 0.1 * H + (tb.head_r if tb else 0.11))


@action("punch", distance=_punch_distance, needs_pose=True,
        doc="Straight punch landing on the target (aim: head|chest|stomach); style: straight|uppercut.")
def punch(c: ActionCtx) -> ActionOut:
    H = c.H
    style = c.p("style", "straight")
    pel = _pel(c, 0, -0.07, 0.06)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-22, 8, 0), chest_rot=(-16, 10, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.08, 0.2), (0.13, -0.2), (0, 0.2, 1), (0.4, 0.1, 1))
    aim = c.aim("head")
    head_r = c.target_body.head_r if c.target_body else 0.11
    toward = unit(aim - _approx_chest(c, pel, 10))
    fist = aim - toward * (head_r + 0.3 * c.body.hand)  # wrist target: knuckles on the surface
    pole = (1, -0.4, 0) if style == "straight" else (0.6, -1, 0.2)
    pose.hands["r"] = Limb(fist, v(*pole))
    pose.hands["l"] = _hand(-0.05, 0.8, 0.13, H, (-0.5, -1, 0))
    tags = ["punching", "motion lines", "impact"]
    if style == "uppercut":
        tags.append("uppercut")
    return ActionOut(pose, tags=tags + _base_tags(c), motion=toward, impact=fist + toward * head_r * 0.9,
                     reach=("r_wrist", fist))


def _kick_distance(c: ActionCtx) -> float:
    H = c.H
    hip_y = c.body.hip_h + 0.04 * H  # the rolled pelvis lifts the kicking hip
    names = {"head": 0.88, "face": 0.88, "neck": 0.8, "chest": 0.72, "body": 0.6, "stomach": 0.6}
    th = (c.target_body.height if c.target_body else 1.65) * names.get(c.p("aim", "head"), 0.7)
    dy = th - hip_y
    leg = c.body.leg_len * 0.97
    horiz = np.sqrt(max(leg * leg - dy * dy, (0.12 * H) ** 2))
    return float(np.clip(horiz + 0.1 * H, 0.32 * H, 0.62 * H))


@action("kick", distance=_kick_distance, needs_pose=True,
        doc="High roundhouse kick with the right leg (aim: head|chest|stomach); style: roundhouse|front.")
def kick(c: ActionCtx) -> ActionOut:
    H = c.H
    style = c.p("style", "roundhouse")
    pel = _pel(c, 0.03, -0.02, -0.02)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-35 if style == "roundhouse" else 0, -24, -22),
                      chest_rot=(10, -6, -14), look_at=c.tj("head"))
    pose.feet = {"l": Limb(v(-0.05 * H, c.body.ankle_h, 0.0), v(0.2, 0.2, 1))}
    aim = c.aim("head")
    hip = pel + v(0.06 * H, 0.03 * H, 0)
    d = aim - hip
    leg = c.body.leg_len * 0.99
    foot = aim if np.linalg.norm(d) <= leg else hip + unit(d) * leg
    knee_pole = (1, 0.4, 0.2) if style == "roundhouse" else (0, 1, 0.4)
    pose.feet["r"] = Limb(foot, v(*knee_pole), unit(d))
    pose.hands["l"] = _hand(-0.33, 0.78, 0.05, H, (-0.2, -1, -0.5))
    pose.hands["r"] = _hand(0.2, 0.74, -0.2, H, (0.5, -1, 0))
    out = ActionOut(pose, tags=["kicking", "high kick", "leg up", "motion lines", "impact"],
                    motion=unit(d), impact=foot + unit(d) * 0.04 * H, reach=("r_ankle", aim))
    if _armed(c):  # weapon swung out wide in the right hand for balance
        _grip(c, pose, v(0.3, 0.8, -0.1) * H, v(0.6, 0.6, -0.3), out)
        pose.hands["l"] = _hand(-0.33, 0.78, 0.05, H, (-0.2, -1, -0.5))
        out.tags += _base_tags(c)
    return out


@action("flying_kick", distance=lambda c: 0.55 * c.H, needs_pose=True,
        doc="Airborne jump kick, leading leg straight into the target (aim: chest|head).")
def flying_kick(c: ActionCtx) -> ActionOut:
    H = c.H
    aim = c.aim("chest")
    pel_y = max(aim[1] + 0.12 * H, 0.75 * H) + (c.altitude or 0.0)
    pel = v(0, pel_y, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-15, -22, 0), chest_rot=(0, 12, 0),
                      look_at=c.tj("head"), grounded=False)
    hip = pel + v(0.06 * H, -0.03 * H, 0)
    d = aim - hip
    leg = c.body.leg_len * 0.99
    foot = aim if np.linalg.norm(d) <= leg else hip + unit(d) * leg
    pose.feet["r"] = Limb(foot, v(0, 1, 0.2), unit(d))
    pose.feet["l"] = Limb(pel + v(-0.08, -0.22, -0.2) * H, v(0, 0.2, 1), v(0, -0.7, -0.6))
    pose.hands["l"] = _hand(-0.28, 0.0, -0.1, H, (-0.3, -1, 0))
    pose.hands["l"].target = pel + v(-0.25, 0.25, -0.1) * H
    pose.hands["r"] = Limb(pel + v(0.12, 0.38, 0.12) * H, v(0.5, -1, 0))
    return ActionOut(pose, tags=["flying kick", "jumping", "midair", "kicking", "motion lines", "impact"],
                     airborne=True, motion=unit(d), impact=foot + unit(d) * 0.04 * H, reach=("r_ankle", aim))


def _blade_distance(c: ActionCtx) -> float:
    return 0.36 * c.H + max(c.weapon.length, 0.25) * 0.8


@action("slash", distance=_blade_distance, needs_pose=True,
        doc="Weapon strike. style: diagonal|horizontal|overhead|rising; phase: impact|windup|follow; aim.")
def slash(c: ActionCtx) -> ActionOut:
    H = c.H
    style = c.p("style", "diagonal")
    phase = c.p("phase", "impact")
    pel = _pel(c, 0, -0.08, 0.05)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-12, 10, 0), chest_rot=(-10, 8, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.11, -0.17), (0.08, 0.21), (-0.3, 0.2, 1), (0.2, 0.2, 1))
    out = ActionOut(pose, tags=_base_tags(c))
    aim = c.aim("chest" if style == "horizontal" else "head")
    chest = _approx_chest(c, pel, 10)
    if not _armed(c):  # unarmed "slash" = chop
        return punch(c)
    if c.p("_clash"):  # the target is blocking: the blade stops on theirs, in front of them
        aim = c.tj("head") - c.target_dir() * 0.42 + v(0, 0.02, 0)
    if phase == "windup":
        grip = v(0.06, 1.08, -0.02) * H
        blade = v(0.15, 0.45, -0.88) if style == "overhead" else v(0.85, 0.35, -0.4)
        pose.pelvis_rot, pose.chest_rot = (10, -4, 0), (12, -8, 0)
        out.tags += ["raised weapon", "about to attack"]
        motion = unit(aim - chest)
    elif phase == "follow":
        grip = v(-0.2, 0.45, 0.2) * H
        blade = v(-0.7, -0.45, 0.4)
        pose.pelvis_rot, pose.chest_rot = (-30, 16, 0), (-24, 12, 0)
        out.tags += ["slashing", "sword trail", "motion blur", "motion lines"]
        motion = unit(aim - chest)
    else:
        # where the hands are at the moment of the cut, relative to the chest
        rel = {"overhead": v(0.04, 0.3, 0.16), "diagonal": v(0.14, 0.2, 0.2),
               "horizontal": v(0.2, 0.03, 0.22), "rising": v(0.12, -0.18, 0.22)}[style]
        toward = unit(aim - chest)
        if c.p("_clash"):
            # blade lock: hands above the contact so the edge comes down onto the guard
            flat = unit(v(toward[0], 0, toward[2]))
            drop = {"overhead": 0.45, "diagonal": 0.3, "horizontal": 0.05, "rising": -0.3}[style]
            grip = aim - flat * c.weapon.length * 0.62 + v(0, drop * c.weapon.length, 0)
            grip = _arm_reach_clamp(c, "r", grip, chest, 0.97)
            blade = unit(aim - grip)
        else:
            grip = chest + rel * H
            grip = _arm_reach_clamp(c, "r", grip, chest, 0.95)
            blade = unit(aim - grip)
            # tilt the blade off the line of attack so it reads as a swing, not a stab
            across = {"overhead": v(0, 0.25, 0), "diagonal": v(-0.25, 0.2, 0),
                      "horizontal": v(-0.3, 0, 0), "rising": v(-0.15, -0.25, 0)}[style]
            blade = unit(blade + across)
        out.tags += ["slashing", "sword swing", "motion blur", "motion lines"]
        out.impact = grip + blade * c.weapon.length * 0.7
        motion = toward
    _grip(c, pose, grip, blade, out)
    if c.weapon.kind in ("twin_daggers", "twin_swords"):
        pose.hands["l"] = Limb(chest + v(-0.2, 0.0, 0.12) * H, v(-1, -0.5, -0.3))
        out.weapon_dirs["l"] = unit(v(-0.4, -0.5, 0.5))
    out.motion = motion
    return out


@action("thrust", distance=lambda c: 0.4 * c.H + max(c.weapon.length, 0.25) * 0.85, needs_pose=True,
        doc="Lunging thrust/stab of the weapon point at the target (aim).")
def thrust(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.12, 0.08)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-25, 14, 0), chest_rot=(-12, 6, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.12, -0.25), (0.06, 0.25), (-0.3, 0.2, 1), (0.1, 0.3, 1))
    out = ActionOut(pose, tags=["thrusting", "lunge", "motion lines"] + _base_tags(c))
    aim = c.aim("chest")
    chest = _approx_chest(c, pel, 14)
    if c.weapon.reach_class == "pole":
        grip = chest + v(0.1, -0.12, 0.05) * H
    else:
        grip = chest + unit(aim - chest) * 0.3 * H + v(0.08, -0.05, 0) * H
        grip = _arm_reach_clamp(c, "r", grip, chest, 0.97)
    blade = unit(aim - grip)
    if c.weapon.reach_class == "pole":  # the point stops at the body surface, not through it
        r = c.target_body.r["chest"] if c.target_body else 0.1
        grip = aim - blade * (c.weapon.length + r)
    _grip(c, pose, grip, blade, out)
    if not c.weapon.two_handed:
        pose.hands["l"] = _hand(-0.22, 0.7, -0.15, H, (-0.5, -1, -0.3))
    out.motion, out.impact = blade, grip + blade * c.weapon.length
    return out


@action("block", needs_pose=True,
        doc="Parry/block the target's attack: weapon across the incoming blade, or arms guarding.")
def block(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.1, -0.03)
    pose = PoseParams(pelvis=pel, pelvis_rot=(10, -4, 0), chest_rot=(-8, -4, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.12, 0.08), (0.12, -0.16), (-0.3, 0.2, 1), (0.4, 0.2, 1))
    out = ActionOut(pose, tags=["blocking", "defending"] + _base_tags(c))
    chest = _approx_chest(c, pel, -4)
    if _armed(c) and c.target_weapon is not None and c.p("_incoming"):
        g, tip = c.target_weapon
        da = unit(tip - g)
        # meet the incoming blade at its point closest to the space in front of my face
        guard = chest + v(0, 0.3 * H, 0.26 * H)  # in front of and above the face
        s_ = np.clip(np.dot(guard - g, tip - g) / max(np.dot(tip - g, tip - g), 1e-9), 0.45, 0.95)
        contact = g + (tip - g) * s_
        # my blade: horizontal, perpendicular to the incoming one, tip to my left, slightly raised
        flat_in = v(da[0], 0, da[2])
        mine = np.cross(v(0, 1, 0), flat_in) if np.linalg.norm(flat_in) > 0.2 else v(-1, 0, 0)
        mine = unit(mine)
        if mine[0] > 0:
            mine = -mine
        mine = unit(mine + v(0, 0.3, 0))
        grip = contact - mine * min(0.32, c.weapon.length * 0.45)
        grip = _arm_reach_clamp(c, "r", grip, chest, 0.95)
        mine = unit(contact - grip)
        _grip(c, pose, grip, mine, out)
        if c.weapon.two_handed:  # brace the flat of the blade with the off hand
            pose.hands["l"] = Limb(grip + mine * min(0.45, c.weapon.length * 0.6), v(-1, -0.6, -0.3), mine)
        if c.weapon.kind in ("twin_daggers", "twin_swords"):  # X-guard with both blades
            pose.hands["l"] = Limb(grip + v(-0.12, -0.04, 0) * H, v(-1, -0.6, -0.3))
            out.weapon_dirs["l"] = unit(mine * v(-1, 1, 1))
        out.impact = contact
        out.tags += ["clashing blades", "sparks", "weapon clash"]
    elif _armed(c):
        _grip(c, pose, v(0.05, 0.85, 0.24) * H, v(-1, 0.25, 0.1), out)
    else:
        pose.hands["r"] = _hand(-0.04, 0.86, 0.17, H, (1, -1, 0))
        pose.hands["l"] = _hand(0.04, 0.82, 0.19, H, (-1, -1, 0))
        out.tags += ["crossed arms", "guarding"]
    out.motion = -c.target_dir() * 0.3
    return out


@action("knocked_back", needs_pose=False,
        doc="Hit and flung backwards through the air away from the target (the attacker).")
def knocked_back(c: ActionCtx) -> ActionOut:
    H = c.H
    alt = 0.18 * H if c.altitude is None else c.altitude
    pel = _pel(c, 0, 0, -0.04) + v(0, alt, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(8, -38, 12), chest_rot=(0, -22, 6), head_rot=(15, -30, 10),
                      grounded=False)
    pose.hands["l"] = Limb(pel + v(-0.32, 0.5, 0.25) * H, v(-0.6, -0.2, -0.6))
    pose.hands["r"] = Limb(pel + v(0.3, 0.42, 0.32) * H, v(0.6, -0.4, -0.6))
    pose.feet["l"] = Limb(pel + v(-0.1, -0.4, 0.18) * H, v(0, 0.5, 1))
    pose.feet["r"] = Limb(pel + v(0.12, -0.44, 0.05) * H, v(0, 0.2, 1))
    out = ActionOut(pose, airborne=True, motion=v(0, 0.25, -1),
                    tags=["being hit", "knocked back", "midair", "flying backwards", "pain",
                          "open mouth", "motion lines"])
    if _armed(c):
        _grip(c, pose, pose.hands["r"].target, v(0.5, -0.1, -0.85), out)  # weapon trails the fall
        pose.hands["l"] = Limb(pel + v(-0.32, 0.5, 0.25) * H, v(-0.6, -0.2, -0.6))
        out.tags.append(c.weapon.tags)
    return out


@action("staggered", needs_pose=False, doc="Just took a hit: grounded, reeling back, guard broken.")
def staggered(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.07, -0.05)
    pose = PoseParams(pelvis=pel, pelvis_rot=(10, -18, 8), chest_rot=(0, -14, 0), head_rot=(10, -18, 0))
    pose.feet = _feet(c, (-0.12, 0.12), (0.1, -0.26), (-0.3, 0.2, 1), (0.3, 0.2, 1))
    pose.hands["l"] = Limb(pel + v(-0.32, 0.32, 0.2) * H, v(-0.6, -0.4, -0.5))
    pose.hands["r"] = Limb(pel + v(0.28, 0.34, 0.22) * H, v(0.6, -0.4, -0.5))
    out = ActionOut(pose, motion=v(0, 0, -1), tags=["staggering", "being hit", "pain", "clenched teeth"])
    if _armed(c):
        _grip(c, pose, pose.hands["r"].target, v(0.4, -0.6, -0.5), out)
        pose.hands["l"] = Limb(pel + v(-0.32, 0.32, 0.2) * H, v(-0.6, -0.4, -0.5))
        out.tags.append(c.weapon.tags)
    return out


@action("leap_attack", distance=lambda c: 0.45 * c.H + max(c.weapon.length, 0.3), needs_pose=True,
        doc="Leaping in from above with the weapon raised overhead (phase: windup|impact).")
def leap_attack(c: ActionCtx) -> ActionOut:
    H = c.H
    alt = (0.55 * H) if c.altitude is None else c.altitude
    pel = _pel(c, 0, 0, 0) + v(0, alt, 0)
    phase = c.p("phase", "windup")
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, -6, 0), chest_rot=(0, -16 if phase == "windup" else 18, 0),
                      look_at=c.tj("head"), grounded=False)
    pose.feet["r"] = Limb(pel + v(0.09, -0.24, 0.12) * H, v(0, 0.3, 1), v(0, -0.8, 0.4))
    pose.feet["l"] = Limb(pel + v(-0.09, -0.42, -0.2) * H, v(0, 0.1, 1), v(0, -0.9, -0.3))
    out = ActionOut(pose, airborne=True, tags=["jumping", "midair", "leaping"] + _base_tags(c))
    aim = c.aim("head")
    if _armed(c):
        if phase == "windup":
            grip = pel + v(0.05, 0.62, -0.06) * H
            blade = v(0.1, 0.35, -0.93)
            out.tags += ["raised weapon", "overhead swing", "about to attack"]
        else:
            chest = pel + v(0, 0.3, 0.05) * H
            grip = chest + unit(aim - chest) * 0.3 * H
            blade = unit(aim - grip)
            out.tags += ["slashing", "downward slash", "motion blur"]
            out.impact = grip + blade * c.weapon.length * 0.7
        _grip(c, pose, grip, blade, out)
    else:
        pose.hands["r"] = Limb(pel + v(0.12, 0.65, 0.05) * H, v(1, -0.3, -0.5))
        pose.hands["l"] = Limb(pel + v(-0.25, 0.35, 0.2) * H, v(-1, -0.4, 0))
        out.tags += ["clenched hand", "raised fist"]
    out.motion = unit(aim - pel + v(0, -0.3, 0))
    return out


@action("cast", distance=lambda c: 3.0, needs_pose=False,
        doc="Casting a spell: arm outstretched toward the target, palm open; style: one_hand|two_hands.")
def cast(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.06, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-15, 4, 0), chest_rot=(-10, 4, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.1, 0.13), (0.13, -0.14), (-0.3, 0.2, 1), (0.4, 0.2, 1))
    chest = _approx_chest(c, pel, 4)
    dirn = unit(c.tj("chest") - chest)
    pose.hands["r"] = Limb(chest + v(0.15 * H, 0.1 * H, 0) + dirn * c.body.arm_reach * 0.95,
                           v(1, -0.6, -0.2), dirn + v(0, 0.35, 0))
    if c.p("style", "one_hand") == "two_hands":
        pose.hands["l"] = Limb(chest + v(-0.05 * H, 0.1 * H, 0) + dirn * c.body.arm_reach * 0.9,
                               v(-1, -0.6, -0.2), dirn + v(0, 0.35, 0))
    else:
        pose.hands["l"] = _hand(-0.12, 0.7, 0.05, H, (-1, -1, 0), tip=(0.2, 0.6, 0.5))
    pose.fist = {"l": False, "r": False}
    tags = ["casting spell", "outstretched arm", "open hand", "glowing hand", "magic circle", "magic"]
    out = ActionOut(pose, tags=tags, motion=dirn * 0.3)
    if c.weapon.reach_class == "pole":  # staff raised in the off hand
        pose.hands["l"] = Limb(chest + v(-0.25, 0.05, 0.12) * H, v(-1, -0.6, -0.2))
        out.weapon_dirs["l"] = unit(v(-0.1, 1, 0.15))
        out.tags.append(c.weapon.tags)
    return out


@action("dodge", needs_pose=False, doc="Leaning/jumping back and aside to evade the target's attack.")
def dodge(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, -0.04, -0.1, -0.06)
    pose = PoseParams(pelvis=pel, pelvis_rot=(15, -16, -14), chest_rot=(5, -14, -10), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.2, -0.1), (0.1, 0.12), (-0.5, 0.2, 1), (0.2, 0.2, 1))
    pose.hands["l"] = Limb(pel + v(-0.36, 0.25, -0.05) * H, v(-0.5, -1, 0))
    pose.hands["r"] = _hand(0.1, 0.78, 0.16, H, (0.6, -1, 0))
    out = ActionOut(pose, motion=v(-0.6, 0, -0.8), tags=["dodging", "leaning back", "evading"] + _base_tags(c))
    if _armed(c):
        _grip(c, pose, v(0.12, 0.75, 0.16) * H, v(0.2, 0.5, 0.85), out)
        pose.hands["l"] = Limb(pel + v(-0.36, 0.25, -0.05) * H, v(-0.5, -1, 0))
    return out


@action("charge", distance=lambda c: 1.6, needs_pose=False,
        doc="Running/charging at the target, body leaning into the sprint.")
def charge(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.04, 0.02)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 22, 0), chest_rot=(0, 10, 0), look_at=c.tj("head"))
    pose.feet["r"] = Limb(pel + v(0.08, -0.33, 0.18) * H, v(0, 0.2, 1))
    pose.feet["l"] = Limb(v(-0.07 * H, c.body.ankle_h + 0.03 * H, -0.3 * H), v(0, 0, 1), v(0, -0.6, 0.8))
    pose.hands["l"] = Limb(pel + v(-0.18, 0.42, 0.25) * H, v(-0.3, -1, 0))
    pose.hands["r"] = Limb(pel + v(0.2, 0.2, -0.22) * H, v(0.3, -0.2, 0.4))
    out = ActionOut(pose, motion=v(0, 0, 1), tags=["running", "charging", "motion lines", "motion blur"])
    if _armed(c):
        _grip(c, pose, pel + v(0.22, 0.12, -0.15) * H, v(0.2, -0.15, -1), out)
        pose.hands["l"] = Limb(pel + v(-0.18, 0.42, 0.25) * H, v(-0.3, -1, 0))
        out.tags.append(c.weapon.tags)
    return out


@action("kneel", needs_pose=False, doc="Down on one knee (wounded/catching breath), looking at the target.")
def kneel(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = v(0, 0.33 * H, -0.02 * H)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 12, 0), chest_rot=(0, 8, 0), look_at=c.tj("head"))
    pose.feet["l"] = Limb(v(-0.09 * H, c.body.ankle_h, 0.2 * H), v(0, 0.3, 1))
    pose.feet["r"] = Limb(v(0.09 * H, 0.03 * H, -0.27 * H), v(0, -0.6, 1), v(0, -0.2, -1))
    out = ActionOut(pose, tags=["kneeling", "on one knee", "injured", "panting"])
    if _armed(c):  # leaning on the weapon
        _grip(c, pose, v(0.16, 0.42, 0.2) * H, v(0.05, -1, 0.15), out)
        pose.hands.pop("l", None)
        out.tags.append(c.weapon.tags)
    pose.hands.setdefault("l", Limb(v(-0.1, 0.36, 0.2) * H, v(-1, 0, -0.3)))
    return out


@action("fallen", needs_pose=False, doc="Lying defeated on the ground (on back).")
def fallen(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = v(0, c.body.r["pelvis"] * 1.0, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, -90, 0), chest_rot=(0, -4, 6), head_rot=(25, 0, 0))
    pose.feet["l"] = Limb(v(-0.12 * H, 0.04 * H, 0.47 * H), v(0, 1, 0.3))
    pose.feet["r"] = Limb(v(0.05 * H, 0.15 * H, 0.38 * H), v(0, 1, 0.1))
    pose.hands["l"] = Limb(v(-0.38 * H, 0.03 * H, -0.3 * H), v(0, 1, 0))
    pose.hands["r"] = Limb(v(0.3 * H, 0.03 * H, -0.05 * H), v(0, 1, 0))
    return ActionOut(pose, tags=["lying", "on back", "defeated", "injured"])


@action("grab", distance=lambda c: 0.42 * c.H, needs_pose=True,
        doc="Grabbing the target by the collar with the right hand, left fist cocked back.")
def grab(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.04, 0.04)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-10, 6, 0), chest_rot=(-6, 6, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.09, 0.12), (0.1, -0.12))
    neck = c.tj("neck")
    pose.hands["r"] = Limb(_arm_reach_clamp(c, "r", neck, _approx_chest(c, pel, 6)), v(1, -0.5, 0))
    pose.hands["l"] = _hand(-0.18, 0.86, -0.12, H, (-0.5, -0.2, -1))
    return ActionOut(pose, tags=["grabbing", "collar grab", "angry", "clenched hand"], motion=v(0, 0, 0.4),
                     impact=neck)


@action("point", needs_pose=False, doc="Standing tall, pointing (or the weapon) at the target.")
def point(c: ActionCtx) -> ActionOut:
    H = c.H
    pose = PoseParams(look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.08, 0.06), (0.09, -0.06))
    pel = c.rig.rest_pelvis()
    chest = _approx_chest(c, pel)
    dirn = unit(c.tj("head") - chest)
    out = ActionOut(pose, tags=["pointing", "confident"] + _base_tags(c))
    if _armed(c):
        _grip(c, pose, chest + v(0.15 * H, 0.08 * H, 0) + dirn * c.body.arm_reach * 0.92, dirn, out)
        if not c.weapon.reach_class == "pole":
            pose.hands.pop("l", None)
        out.tags.append("pointing weapon")
    else:
        pose.hands["r"] = Limb(chest + v(0.15 * H, 0.08 * H, 0) + dirn * c.body.arm_reach, v(1, -0.5, 0))
        pose.fist["r"] = False
    return out


# ============================================================================ close combat
# Mounted actions place the attacker on the target's *solved* body (ctx.target_world): the
# victim's reaction pose is solved first, then the attacker is positioned and posed around it.

def _yaw(d: np.ndarray) -> float:
    return float(np.arctan2(d[0], d[2]))


def _fwd_right(yaw: float):
    f = v(np.sin(yaw), 0, np.cos(yaw))
    return f, v(-f[2], 0, f[0])


def _mount_local(c: ActionCtx, origin: np.ndarray, yaw: float, p: np.ndarray) -> np.ndarray:
    """World point -> local coordinates of an actor standing at (origin, yaw)."""
    f, r = _fwd_right(yaw)
    d = p - v(origin[0], 0, origin[2])
    return v(d @ r, d[1], d @ f)


def _limb_world(c, origin, yaw, side, point, pole_local, tip=None):
    return Limb(_mount_local(c, origin, yaw, point), v(*pole_local), None if tip is None else v(*tip))


@action("headlock", needs_pose=True,
        doc="Front headlock: the target (doubled over, use `headlocked`) has its head trapped under the right arm.")
def headlock(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return stance(c)
    head, neck = c.tw("head"), c.tw("neck")
    yaw = _yaw(-c.t_fwd)  # face the victim
    f, r = _fwd_right(yaw)
    armpit_h = 0.66 * H
    drop = float(np.clip(armpit_h - head[1], 0.0, 0.16 * H))
    origin = head - r * 0.1 * H - f * 0.05 * H
    origin = v(origin[0], 0, origin[2])
    pose = PoseParams(pelvis=_pel(c, 0, 0, -0.02) - v(0, drop, 0), pelvis_rot=(22, -10, 0), chest_rot=(10, -4, 0),
                      look_at=_mount_local(c, origin, yaw, head))
    pose.feet = _feet(c, (-0.12, 0.12), (0.14, -0.12), (-0.3, 0.3, 1), (0.3, 0.3, 1))
    grip = neck - v(0, 0.07, 0) - r * 0.1  # forearm under the throat, hand out the far side
    pose.hands["r"] = _limb_world(c, origin, yaw, "r", grip, (1, 0.3, -0.4))
    pose.hands["l"] = _limb_world(c, origin, yaw, "l", grip - r * 0.05 + v(0, 0.04, 0) - f * 0.04, (-1, -0.4, -0.3))
    out = ActionOut(pose, tags=["headlock", "grappling", "arm around neck", "clenched teeth", "angry"],
                    motion=v(0, 0, -0.3), impact=_mount_local(c, origin, yaw, neck), mount=(origin, yaw))
    return out


@action("headlocked", doc="Bent over at the waist with the head trapped in the target's headlock, struggling.")
def headlocked(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.08, -0.04)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 52, 0), chest_rot=(0, 26, 0), head_rot=(0, -48, 0))
    pose.feet = _feet(c, (-0.1, -0.16), (0.1, 0.06), (0, 0.4, 1), (0, 0.4, 1))
    head = c.rig.solve(pose)["head"]
    pose.hands["r"] = Limb(head + v(0.1, 0.04, -0.06) * H, v(1, -0.5, -0.4))
    pose.hands["l"] = Limb(head + v(-0.1, 0.06, -0.04) * H, v(-1, -0.5, -0.4))
    return ActionOut(pose, tags=["headlock", "bent over", "being choked", "struggling", "pain", "clenched teeth",
                                 "sweat", "grabbing another's arm"])


@action("rear_choke", needs_pose=True, doc="Chokehold from behind: forearm across the target's throat.")
def rear_choke(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return stance(c)
    yaw = c.target_yaw
    f, r = _fwd_right(yaw)
    neck, head, pel_t = c.tw("neck"), c.tw("head"), c.tw("pelvis")
    origin = v(pel_t[0], 0, pel_t[2]) - f * 0.17 * H
    lift = float(np.clip(head[1] + 0.02 - (c.body.hip_h + 0.45 * H), -0.15 * H, 0.0))
    pose = PoseParams(pelvis=_pel(c, 0, 0, 0) + v(0, lift, 0), pelvis_rot=(-10, -10, 0), chest_rot=(-6, -4, 0),
                      look_at=_mount_local(c, origin, yaw, head + f * 0.5))
    pose.feet = _feet(c, (-0.12, 0.06), (0.12, -0.12))
    front = neck + f * 0.06 - v(0, 0.01, 0)
    pose.hands["r"] = _limb_world(c, origin, yaw, "r", front - r * 0.11, (1, -0.3, -0.2))
    pose.hands["l"] = _limb_world(c, origin, yaw, "l", head - f * 0.1 + v(0, 0.02, 0), (-1, 0.0, -0.3))
    return ActionOut(pose, tags=["chokehold", "rear naked choke", "grappling", "from behind", "clenched teeth"],
                     impact=_mount_local(c, origin, yaw, front), mount=(origin, yaw))


@action("choked", doc="Being choked from behind: arched back, clawing at the forearm across the throat.")
def choked(c: ActionCtx) -> ActionOut:
    H = c.H
    pose = PoseParams(pelvis=_pel(c, 0, -0.02, 0.02), pelvis_rot=(0, -8, 0), chest_rot=(0, -10, 0),
                      head_rot=(-12, -22, 0))
    pose.feet = _feet(c, (-0.08, 0.1), (0.09, 0.02))
    J = c.rig.solve(pose)  # where the arched neck actually is
    neck_front = J["neck"] + unit(J["nose"] - J["head"]) * 0.05 * H
    pose.hands["r"] = Limb(neck_front + v(0.06, -0.04, 0.03) * H, v(1, -0.6, 0))
    pose.hands["l"] = Limb(neck_front + v(-0.06, -0.03, 0.04) * H, v(-1, -0.6, 0))
    return ActionOut(pose, tags=["being choked", "struggling", "tears", "open mouth", "pain", "hands on another's arm"])


@action("leg_lock", needs_pose=True,
        doc="Ankle lock on the ground: sitting at the target's feet (target uses `leg_locked`), ankle trapped, leaning back.")
def leg_lock(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return stance(c)
    ankle, tpel = c.tw("r_ankle"), c.tw("pelvis")
    d = unit(v(ankle[0] - tpel[0], 0, ankle[2] - tpel[2]))
    origin = v(ankle[0], 0, ankle[2]) + d * 0.13 * H
    yaw = _yaw(-d)
    f, r = _fwd_right(yaw)
    pose = PoseParams(pelvis=v(0, c.body.r["pelvis"] * 1.05, 0), pelvis_rot=(0, -26, 0), chest_rot=(0, -6, 0),
                      look_at=_mount_local(c, origin, yaw, c.tw("head")))
    ah = c.body.ankle_h
    pose.feet = {"l": Limb(v(-0.15 * H, ah, 0.4 * H), v(0, 1, 0.2)), "r": Limb(v(0.15 * H, ah, 0.38 * H), v(0, 1, 0.2))}
    wrap = ankle - r * 0.07 + v(0, -0.02, 0)
    pose.hands["r"] = _limb_world(c, origin, yaw, "r", wrap, (1, -0.6, -0.2))
    pose.hands["l"] = _limb_world(c, origin, yaw, "l", wrap - r * 0.05 + v(0, 0.05, 0) - f * 0.03, (-1, -0.6, -0.2))
    return ActionOut(pose, tags=["leg lock", "ankle lock", "grappling", "sitting", "on ground", "leaning back",
                                 "clenched teeth"], impact=_mount_local(c, origin, yaw, ankle), mount=(origin, yaw))


@action("leg_locked", doc="Flat on the back with one leg trapped straight up in the target's leg lock.")
def leg_locked(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = v(0, c.body.r["pelvis"] * 1.0, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, -90, 0), chest_rot=(0, 12, 6), head_rot=(10, 30, 0))
    pose.feet["r"] = Limb(v(0.06 * H, 0.33 * H, 0.42 * H), v(0, 1, 0.1))
    pose.feet["l"] = Limb(v(-0.12 * H, 0.04 * H, 0.3 * H), v(0, 1, 0.3))
    pose.hands["l"] = Limb(v(-0.36 * H, 0.03 * H, -0.12 * H), v(0, 1, 0))
    pose.hands["r"] = Limb(v(0.3 * H, 0.03 * H, -0.05 * H), v(0, 1, 0))
    return ActionOut(pose, tags=["lying", "on back", "leg lock", "pain", "screaming", "tears", "clenched hands"])


def _hip_reach(c: ActionCtx, hip_h: float, target_h: float, frac=0.95) -> float:
    leg = c.body.leg_len * frac
    dy = abs(hip_h - target_h)
    return float(np.sqrt(max(leg * leg - dy * dy, (0.1 * c.H) ** 2)))


@action("back_knee_kick", needs_pose=True,
        doc="Stomp kick into the back of the target's knee from behind (target: `knee_buckle`, facing away).")
def back_knee_kick(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return kick(c)
    side = c.p("leg", "l")  # which of the target's knees
    knee = c.tw(f"{side}_knee")
    yaw = c.target_yaw
    f, r = _fwd_right(yaw)
    point = knee - f * (c.target_body.r["shin"] + 0.02)  # the hollow behind the knee
    hip_h = c.body.hip_h - 0.02 * H
    horiz = _hip_reach(c, hip_h, point[1])
    origin = point - f * horiz - r * c.body.hip_w
    origin = v(origin[0], 0, origin[2])
    pose = PoseParams(pelvis=_pel(c, 0, -0.02, 0), pelvis_rot=(0, -14, 0), chest_rot=(0, 8, 0),
                      look_at=_mount_local(c, origin, yaw, c.tw("head")))
    pose.feet["l"] = Limb(v(-0.07 * H, c.body.ankle_h, 0.0), v(0, 0.2, 1))
    pose.feet["r"] = _limb_world(c, origin, yaw, "r", point, (0, 1, 0.3), (0.35, -0.15, 1))
    pose.hands["l"] = _hand(-0.1, 0.8, 0.14, H, (-0.5, -1, 0))
    pose.hands["r"] = _hand(0.18, 0.66, -0.1, H, (0.5, -1, 0))
    return ActionOut(pose, tags=["kicking", "low kick", "from behind", "sneak attack", "motion lines"],
                     motion=v(0, -0.3, 1), impact=_mount_local(c, origin, yaw, point), mount=(origin, yaw))


@action("knee_buckle", doc="The knee just got kicked out from behind: dropping toward one knee, off balance.")
def knee_buckle(c: ActionCtx) -> ActionOut:
    H = c.H
    side = c.p("leg", "l")
    sx = -1.0 if side == "l" else 1.0
    pel = _pel(c, 0, -0.14, 0.02)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, -12, 8 * -sx), chest_rot=(0, -6, 0), head_rot=(0, -18, 0))
    ah = c.body.ankle_h
    pose.feet[side] = Limb(v(sx * 0.08 * H, ah + 0.07 * H, -0.2 * H), v(0, -0.6, 1), v(0, -0.5, -0.8))
    other = "r" if side == "l" else "l"
    pose.feet[other] = Limb(v(-sx * 0.09 * H, ah, 0.15 * H), v(0, 0.2, 1))
    pose.hands["l"] = Limb(pel + v(-0.34, 0.42, -0.1) * H, v(-0.5, -0.5, -0.4))
    pose.hands["r"] = Limb(pel + v(0.3, 0.5, 0.15) * H, v(0.5, -0.5, -0.4))
    pose.fist = {"l": False, "r": False}
    return ActionOut(pose, motion=v(0, -1, 0.2),
                     tags=["falling", "off balance", "on one knee", "surprised", "pain", "open mouth", "flailing arms"])


@action("spinning_back_kick", distance=lambda c: 0.62 * c.H, needs_pose=True,
        doc="Turned-back high kick: back to the target, leg driving backward and up into the neck/head.")
def spinning_back_kick(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return kick(c)
    aim = c.tw({"head": "head", "neck": "neck", "chest": "chest"}.get(c.p("aim", "neck"), "neck"))
    tpel = c.tw("pelvis")
    # stay on the side the placement chose; turn the back to the target
    if c.origin is not None and np.hypot(c.origin[0] - tpel[0], c.origin[2] - tpel[2]) > 0.05:
        d = unit(v(c.origin[0] - tpel[0], 0, c.origin[2] - tpel[2]))
    else:
        d = c.t_fwd
    yaw = _yaw(d)  # facing away from the target
    f, r = _fwd_right(yaw)
    origin = c.origin if c.origin is not None else v(tpel[0], 0, tpel[2]) + d * 0.5 * H
    pose = PoseParams(pelvis=_pel(c, 0, -0.05, 0.05), pelvis_rot=(-10, 48, 10), chest_rot=(35, -12, 0),
                      look_at=_mount_local(c, origin, yaw, c.tw("head")))
    pose.feet["l"] = Limb(v(-0.06 * H, c.body.ankle_h, 0.1 * H), v(0, 0.2, 1))
    goal = _mount_local(c, origin, yaw, aim - d * (c.target_body.r["neck"] + 0.02))
    pose.feet["r"] = Limb(goal, v(0, -1, 0.1), v(0.2, -0.8, -0.5))
    pose.hands["l"] = _hand(-0.12, 0.62, 0.3, H, (-0.5, -1, 0))
    pose.hands["r"] = _hand(0.25, 0.55, 0.1, H, (0.5, -1, 0))
    return ActionOut(pose, tags=["spinning kick", "back kick", "high kick", "kicking", "looking back",
                                 "motion blur", "motion lines", "impact"],
                     motion=v(0, 0.1, -1), impact=_mount_local(c, origin, yaw, aim), turn=yaw,
                     reach=("r_ankle", goal))


def _palm_distance(c: ActionCtx) -> float:
    return _punch_distance(replace(c, params={**c.params, "aim": c.p("aim", "stomach")}))


@action("palm_strike", distance=_palm_distance, needs_pose=True,
        doc="Open palm driven into the target (default the stomach); style: energy (with an energy blast) | plain.")
def palm_strike(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.1, 0.07)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-24, 12, 0), chest_rot=(-14, 8, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.09, 0.22), (0.13, -0.2), (0, 0.2, 1), (0.4, 0.1, 1))
    aim = c.aim("stomach")
    r_t = c.target_body.r["waist"] if c.target_body else 0.09
    toward = unit(v(aim[0], 0, aim[2]) - v(0, 0, 0))
    wrist = aim - toward * (r_t + 0.035 * H) - v(0, 0.035 * H, 0)  # heel of the palm on the surface
    pose.hands["r"] = Limb(wrist, v(1, -0.6, 0), toward * 0.3 + v(0, 1, 0))
    pose.hands["l"] = _hand(-0.12, 0.66, 0.06, H, (-0.5, -1, 0))
    pose.fist = {"l": True, "r": False}
    energy = c.p("style", "energy") == "energy"
    out = ActionOut(pose, tags=["palm strike", "open hand", "outstretched arm", "shouting"]
                    + (["glowing hand", "energy"] if energy else []),
                    motion=toward, impact=aim - toward * r_t, reach=("r_wrist", wrist))
    if energy:
        out.effect = c.p("effect", "energy blast, shockwave, glowing, light particles, blue energy, lens flare")
    return out


@action("doubled_over", doc="Just took a blow to the gut: folded around it, lifted onto the toes, mouth open.")
def doubled_over(c: ActionCtx) -> ActionOut:
    H = c.H
    lift = c.p("lift", 0.03) * H
    pel = _pel(c, 0, -0.03, -0.04) + v(0, lift, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 30, 0), chest_rot=(0, 30, 0), head_rot=(0, -6, 0),
                      grounded=lift <= 0)
    ah = c.body.ankle_h
    pose.feet["l"] = Limb(v(-0.08 * H, ah + lift * 0.6, -0.14 * H), v(0, 0.3, 1), v(0, -0.85, 0.5))
    pose.feet["r"] = Limb(v(0.08 * H, ah + lift * 0.9, -0.04 * H), v(0, 0.3, 1), v(0, -0.85, 0.5))
    pose.hands["l"] = Limb(pel + v(-0.2, 0.18, 0.36) * H, v(-0.6, -0.3, -0.5))
    pose.hands["r"] = Limb(pel + v(0.22, 0.22, 0.34) * H, v(0.6, -0.3, -0.5))
    pose.fist = {"l": False, "r": False}
    return ActionOut(pose, airborne=lift > 0, motion=v(0, 0.15, -1),
                     tags=["doubled over", "stomach punch", "pain", "open mouth", "saliva", "wide-eyed",
                           "constricted pupils", "motion lines"])


@action("clutch_stomach", doc="Hunched over, both hands clutching the stomach after a nasty hit.")
def clutch_stomach(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.08, -0.03)
    pose = PoseParams(pelvis=pel, pelvis_rot=(8, 22, 0), chest_rot=(0, 22, 0), head_rot=(0, 2, 0),
                      look_at=c.tj("head") if c.target is not None else None)
    pose.feet = _feet(c, (-0.09, 0.08), (0.1, -0.12), (0, 0.4, 1), (0.2, 0.4, 1))
    J = c.rig.solve(pose)
    belly = J["spine"] + unit(J["spine"] - J["pelvis"]) * 0.0 + v(0, -0.01, 0.07) * H
    pose.hands["r"] = Limb(belly + v(0.05, 0.02, 0.02) * H, v(1, -0.3, -0.2), v(-1, -0.2, 0.2))
    pose.hands["l"] = Limb(belly + v(-0.04, -0.03, 0.02) * H, v(-1, -0.3, -0.2), v(1, -0.2, 0.2))
    pose.fist = {"l": False, "r": False}
    return ActionOut(pose, tags=["hand on own stomach", "hunched over", "pain", "clenched teeth", "sweat", "tears",
                                 "glaring"])


@action("stumble", doc="Stumbling backwards off balance, legs crossing, arms windmilling.")
def stumble(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0.02, -0.05, -0.04)
    pose = PoseParams(pelvis=pel, pelvis_rot=(12, -14, -12), chest_rot=(6, -10, -6), head_rot=(-10, -10, 8))
    ah = c.body.ankle_h
    pose.feet["l"] = Limb(v(0.04 * H, ah + 0.05 * H, -0.2 * H), v(0, 0.3, 1), v(0.2, -0.6, 0.7))
    pose.feet["r"] = Limb(v(0.1 * H, ah, 0.08 * H), v(0.2, 0.2, 1))
    pose.hands["l"] = Limb(pel + v(-0.4, 0.38, 0.06) * H, v(-0.3, -0.6, -1))
    pose.hands["r"] = Limb(pel + v(0.34, 0.3, -0.14) * H, v(0.3, -0.6, -1))
    pose.fist = {"l": False, "r": False}
    return ActionOut(pose, motion=v(0, 0, -1),
                     tags=["stumbling", "off balance", "flailing arms", "surprised", "open mouth"])


@action("knee_strike", needs_pose=True,
        doc="Clinch knee: hands pulling the target's head down while the right knee drives into the stomach.")
def knee_strike(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return kick(c)
    spine, head = c.tw("spine"), c.tw("head")
    yaw = _yaw(-c.t_fwd)
    f, r = _fwd_right(yaw)
    K = spine + c.t_fwd * (c.target_body.r["waist"] + 0.04)  # front of the target's stomach
    hip_h = c.body.hip_h - 0.035 * H
    dy = K[1] - hip_h
    horiz = float(np.sqrt(max(c.body.thigh ** 2 - dy ** 2, (0.06 * H) ** 2)))
    origin = K - f * horiz - r * c.body.hip_w
    origin = v(origin[0], 0, origin[2])
    hip = origin + v(0, hip_h, 0) + r * c.body.hip_w
    knee = hip + unit(K - hip) * c.body.thigh
    foot = knee + unit(v(0, -1, 0) - f * 0.7) * c.body.shin
    pose = PoseParams(pelvis=_pel(c, 0, 0, 0), pelvis_rot=(0, -8, 0), chest_rot=(0, 14, 0),
                      look_at=_mount_local(c, origin, yaw, head))
    pose.feet["l"] = Limb(v(-0.07 * H, c.body.ankle_h, -0.05 * H), v(0, 0.2, 1))
    pose.feet["r"] = _limb_world(c, origin, yaw, "r", foot, (0, 0.8, 1))
    back = head - c.t_fwd * 0.07 + v(0, 0.03, 0)
    pose.hands["r"] = _limb_world(c, origin, yaw, "r", back + r * 0.05, (1, -0.2, 0))
    pose.hands["l"] = _limb_world(c, origin, yaw, "l", back - r * 0.05, (-1, -0.2, 0))
    return ActionOut(pose, tags=["knee strike", "kneeing", "clinch", "grabbing another's head", "angry", "impact"],
                     motion=v(0, 0.6, 0.8), impact=_mount_local(c, origin, yaw, K), mount=(origin, yaw))
