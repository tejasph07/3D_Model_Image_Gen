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
        if a in ("stomach", "gut", "body"):  # the soft belly, between the navel and the bottom of the ribs
            # (the rig's "spine" joint sits at the lower ribs, "chest" at the bust, "pelvis" at the hips)
            return self.tj("spine") * 0.7 + self.tj("pelvis") * 0.3
        if names[a] == "head":  # blows to the head land on the face/jaw, not the crown
            hr = self.target_body.head_r if self.target_body else 0.11
            return self.tj("head") * 0.5 + self.tj("nose") * 0.5 - v(0, 0.3 * hr, 0)
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
    # soft-body presses on the target's mesh: {"point": effector surface (local), "dir": into the body
    # (local), "radius": m, "max_depth": m}; the renderer measures the real skin under the point and
    # dents it exactly as far as the effector went in. "depth" < 0 pulls the surface instead (pinch).
    press: list = field(default_factory=list)


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


# Arm shapes in the chest frame, as offsets from the shoulder in units of the arm's own reach
# (x = outward from the body, y = up along the spine, z = forward from the chest), with the
# direction the elbow points. Real arms are rarely straight: everything that isn't landing a blow
# stays well inside reach so the elbow visibly bends.
ARMS = {
    "hang": ((0.12, -0.87, 0.17), (0.15, 0, -1)),          # relaxed at the side
    "hang_back": ((0.16, -0.84, -0.12), (0.2, 0.1, -1)),
    "guard": ((-0.22, -0.16, 0.6), (0.3, -1, 0.15)),       # fist in front of the chin
    "guard_high": ((-0.2, -0.02, 0.5), (0.25, -1, 0.15)),  # peek-a-boo: glove at the cheek
    "guard_lead": ((0.0, -0.06, 0.62), (0.4, -1, -0.1)),   # lead hand out in front of the shoulder
    "chin": ((-0.2, 0.0, 0.46), (0.35, -1, 0.1)),          # rear hand tucked at the chin
    "cocked": ((0.2, -0.12, -0.32), (0.2, -1, -0.6)),      # fist drawn back to throw
    "chamber": ((0.08, -0.62, -0.04), (0.1, -0.3, -1)),    # fist pulled back to the hip
    "fist_side": ((0.2, -0.74, 0.08), (0.3, 0, -1)),       # clenched at the side, elbow soft
    "balance": ((0.6, -0.3, 0.32), (0.1, -0.7, -1)),       # out to the side for balance
    "balance_high": ((0.55, 0.12, 0.36), (0.2, -1, -0.5)),
    "swing_back": ((0.2, -0.5, -0.52), (0.15, 0.3, -1)),   # counter-swung behind (kicks, running)
    "swing_fwd": ((-0.05, -0.28, 0.5), (0.25, -1, -0.3)),
    "flail": ((0.52, 0.32, -0.05), (0.5, -1, 0.1)),        # thrown up and out, elbows bent
    "flail_fwd": ((0.18, 0.05, 0.68), (0.6, -1, 0)),       # trailing forward as the body is driven back
    "flail_low": ((0.45, -0.48, 0.3), (0.4, -0.3, -1)),
    "brace": ((0.3, -0.72, 0.36), (0.3, 0, -1)),           # reaching down to catch a fall
    "cover": ((-0.42, 0.14, 0.42), (0.5, -1, 0)),          # forearm across the face
    "reach": ((0.0, -0.08, 0.8), (0.5, -1, 0)),            # open hand reaching forward
}


def _arm(c: ActionCtx, side: str, shape: str, dx=0.0, dy=0.0, dz=0.0, pole=None, tip=None) -> Limb:
    """A natural arm from the ARMS library (offsets dx/dy/dz in arm-reach units, x outward)."""
    (x, y, z), pl = ARMS[shape]
    sx = 1.0 if side == "r" else -1.0
    off = v(x + dx, y + dy, z + dz)
    n = np.linalg.norm(off)
    if n > 0.93:
        off = off / n * 0.93
    pl = pl if pole is None else pole
    t = None if tip is None else v(sx * tip[0], tip[1], tip[2])
    return Limb(v(sx * off[0], off[1], off[2]) * c.body.arm_reach, v(sx * pl[0], pl[1], pl[2]), t, frame="chest")


def _arms(c: ActionCtx, pose: PoseParams, l: str, r: str, **kw):
    pose.hands["l"] = _arm(c, "l", l, **kw)
    pose.hands["r"] = _arm(c, "r", r, **kw)


def _own(c: ActionCtx, pose: PoseParams, side: str, point, pole, tip=None) -> Limb:
    """A hand on a point of the actor's own body (local, for the pose as it is now), kept in the
    chest frame so the contact survives re-grounding of the pelvis."""
    J = c.rig.solve(pose)
    Rc = c.rig.chest_R
    sh = J[f"{side}_shoulder"]
    return Limb(Rc.T @ (np.asarray(point, float) - sh), Rc.T @ np.asarray(pole, float),
                None if tip is None else Rc.T @ np.asarray(tip, float), frame="chest")


def _solve(c: ActionCtx, pose: PoseParams) -> dict:
    """Joints of the pose as it stands (arms not yet placed fall back to hanging)."""
    return c.rig.solve(pose)


def _pole_to(J: dict, side: str, elbow_at: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pole direction that puts the elbow on the side of `elbow_at` (local points)."""
    sh = J[f"{side}_shoulder"]
    d = unit(target - sh)
    e = elbow_at - sh
    return unit(e - d * (e @ d), fallback=(0, -1, 0))


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
    arms = c.p("arms")  # e.g. "hang,fist_side" (left,right) from the ARMS library
    if arms:
        l, r = (arms.split(",") + [arms])[:2] if isinstance(arms, str) else arms
        _arms(c, pose, l.strip(), r.strip())
    else:
        _arms(c, pose, "hang", "hang")
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
    elif c.p("style") == "boxing" or c.weapon.kind == "boxing_gloves":  # peek-a-boo guard
        _arms(c, pose, "guard_high", "guard_high")
        pose.pelvis_rot, pose.chest_rot = (18, 12, 0), (-10, 10, 0)
        out.tags += ["boxing", "fighting stance", "clenched hands"]
    elif c.weapon.kind == "magic":
        pose.hands["r"] = _arm(c, "r", "reach", dx=0.05, dy=-0.1, dz=-0.15, tip=(0, 0.4, 1))
        pose.hands["l"] = _arm(c, "l", "guard_lead", dx=0.1, dy=-0.15, dz=-0.1, tip=(0, 0.6, 0.7))
        pose.fist = {"l": "relaxed", "r": "relaxed"}
        out.tags.append("glowing hand")
    else:
        pose.hands["l"] = _arm(c, "l", "guard_lead")
        pose.hands["r"] = _arm(c, "r", "chin")
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
    """Styles: straight (rear cross), jab (lead hand), hook (from the side), uppercut, body (to the gut)."""
    H = c.H
    style = c.p("style", "straight")
    if style not in ("straight", "jab", "hook", "uppercut", "body"):
        raise ValueError(f"unknown punch style {style!r}")
    hand = "l" if style == "jab" else "r"
    other = "r" if hand == "l" else "l"
    low = style == "body" or c.p("aim") in ("stomach", "body", "gut")
    # body shots drop the level and slip the head to the outside of the target's (folding) body
    pel = _pel(c, 0, -0.1 if low else -0.07, 0.06)
    twist = {"straight": -22, "jab": 12, "hook": -34, "uppercut": -18, "body": -24}[style]
    sl = (-1 if hand == "r" else 1)  # lean away from the punching side
    pose = PoseParams(pelvis=pel, pelvis_rot=(twist, 10 if low else 8, 8 * sl if low else 0),
                      chest_rot=(twist * 0.7, 8 if low else 10, 10 * sl if low else 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.08, 0.2), (0.13, -0.2), (0, 0.2, 1), (0.4, 0.1, 1))
    aim = c.aim("stomach" if style == "body" else ("head" if not low else c.p("aim")))
    tb = c.target_body
    # the head aim is already the jaw, a third of a head radius under the skin
    surf = {"head": (tb.head_r if tb else 0.11) * 0.35, "face": (tb.head_r if tb else 0.11) * 0.35}.get(
        c.p("aim", "head"), None) if not low else None
    if surf is None:
        surf = (tb.r["waist"] if tb else 0.09) if (low or c.p("aim") in ("stomach", "body", "gut")) else \
            ((tb.r["chest"] if tb else 0.1) if c.p("aim") == "chest" else (tb.head_r if tb else 0.11))
    J = _solve(c, pose)
    # a much taller target: jump into the blow (anime-style) instead of punching straight up
    rise = aim[1] - J[f"{hand}_shoulder"][1] - 0.75 * c.body.arm_reach
    airborne = False
    if rise > 0.04 * H and style != "body":
        lift = min(rise, 0.3 * H)
        pose.pelvis = pose.pelvis + v(0, lift, 0)
        pose.grounded, airborne = False, True
        ah = c.body.ankle_h
        pose.feet = {"l": Limb(v(-0.08 * H, ah + lift * 0.8, 0.12 * H), v(0, 0.3, 1), v(0, -0.6, 0.8)),
                     "r": Limb(v(0.1 * H, ah + lift * 1.1, -0.2 * H), v(0, 0, 1), v(0, -0.8, -0.5))}
        J = _solve(c, pose)
    chest = J["chest"]
    toward = unit(aim - chest)
    body_shot = low or c.p("aim") in ("stomach", "body", "gut", "chest")
    if body_shot and style in ("straight", "jab", "body"):
        # knuckles sunk into the skin along the torso's own normal (a folding body's belly faces down)
        skin, n_in = _surface(c, aim, "chest" if c.p("aim") == "chest" else "stomach")
        knuckle = skin + n_in * 0.025
        travel = unit(knuckle - J[f"{hand}_shoulder"])
        fist = knuckle - travel * 0.5 * c.body.hand
        pose.hands[hand] = Limb(fist, v((-1 if hand == "l" else 1) * 0.6, -1, 0))
        pose.hands[other] = _arm(c, other, "chin")
        tags = ["punching", "motion lines", "impact", "body blow", "stomach punch"]
        return ActionOut(pose, tags=tags + _base_tags(c), motion=travel, impact=skin, reach=(f"{hand}_wrist", fist),
                         press=[{"point": knuckle, "dir": n_in, "radius": 0.06, "max_depth": 0.05}])
    if style == "hook":  # the fist arrives from the outside, travelling across; elbow up at ~90 degrees
        across = unit(v(-1, 0, 0.35))
        fist = aim - across * (surf + 0.3 * c.body.hand) * 0.6 - toward * (surf + 0.3 * c.body.hand) * 0.8
        pole, motion = (1, 0.35, -0.2), across
    elif style == "uppercut":  # elbow tucked under, forearm vertical
        fist = aim - unit(v(0, 1.2, 0) + toward * 0.6) * (surf + 0.55 * c.body.hand)  # rising under the chin
        pole, motion = (0.3, -1, 0.2), unit(v(0, 1, 0.4))
    else:
        fist = aim - toward * (surf + 0.3 * c.body.hand)  # wrist target: knuckles on the surface
        pole, motion = ((-1 if hand == "l" else 1) * 0.6, -1, 0), toward
    pose.hands[hand] = Limb(fist, v(*pole))
    pose.hands[other] = _arm(c, other, "chin")  # the other hand stays home, guarding the chin
    tags = ["punching", "motion lines", "impact"] + {"jab": ["jab"], "hook": ["hook punch"], "uppercut": ["uppercut"],
                                                     "body": ["body blow", "stomach punch"], "straight": []}[style]
    if airborne:
        tags += ["jumping", "midair"]
    reach = (f"{hand}_wrist", fist)
    if style in ("hook", "uppercut"):  # short, bent-arm blows: step in until the elbow is at ~90 degrees
        sh = J[f"{hand}_shoulder"]
        reach = (f"{hand}_shoulder", fist - unit(fist - sh) * 0.74 * c.body.arm_reach)
    return ActionOut(pose, tags=tags + _base_tags(c), motion=motion, impact=fist + toward * surf * 0.9,
                     reach=reach, airborne=airborne)


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
        doc="Kick with the right leg (aim: head|chest|stomach); style: roundhouse|front.")
def kick(c: ActionCtx) -> ActionOut:
    """Roundhouse: support foot pivoted so the heel points at the target, hips turned over, body
    leaning away, shin/instep arriving from the side, kicking-side arm swung down and back, the other
    hand guarding the face. Front (teep): hips square, lean back, ball of the foot driven straight in."""
    H = c.H
    style = c.p("style", "roundhouse")
    aim = c.aim("head")
    tb = c.target_body
    ah = c.body.ankle_h
    if style == "roundhouse":
        pel = _pel(c, 0.02, -0.02, -0.03)
        pose = PoseParams(pelvis=pel, pelvis_rot=(-62, -8, -22), chest_rot=(36, -2, 12), look_at=c.tj("head"))
        # support foot pivoted ~120 degrees: toes point away to the left-back, knee over the toes
        pose.feet = {"l": Limb(v(-0.05 * H, ah, 0.0), v(-0.8, 0.3, -0.5), v(-0.85, 0, -0.5))}
        hip = pel + v(0.05 * H, 0.0, 0)
        surf = (tb.head_r if tb else 0.11) * 0.4 if c.p("aim", "head") in ("head", "face") else \
            (tb.r["chest"] if tb else 0.1) * 0.8
        travel = unit(v(-1, 0.05, 0.2))  # the leg whips across from the kicker's right
        goal = aim - travel * surf * 1.1 - v(0, 0, 0.3 * surf)
        leg_len = c.body.leg_len
        jump = (goal[1] - hip[1]) - 0.8 * leg_len
        if jump > 0.01 * H:  # the head is out of reach from the floor: a jumping roundhouse
            pel = pel + v(0, jump, 0)
            hip = hip + v(0, jump, 0)
            pose.pelvis, pose.grounded = pel, False
            pose.feet["l"] = Limb(pel + v(-0.08, -0.4, 0.02) * H, v(-0.3, 0.3, 1), v(-0.3, -0.8, 0.3))
        d = goal - hip
        leg = c.body.leg_len * 0.985
        foot = goal if np.linalg.norm(d) <= leg else hip + unit(d) * leg
        # the front of the shin and the instep lead the swing: the knee faces along the travel, toes pointed
        pose.feet["r"] = Limb(foot, v(-1, 0.15, 0.25), unit(unit(d) + travel * 0.35))
        pose.hands["l"] = _arm(c, "l", "guard", dz=-0.05)
        pose.hands["r"] = _arm(c, "r", "swing_back", dx=0.15, dy=-0.15, dz=0.05)
        motion = unit(travel + v(0, 0, 0.8))  # sweeps across and drives through
        tags = ["kicking", "roundhouse kick", "high kick", "leg up", "motion lines", "impact"]
        if not pose.grounded:
            tags += ["jumping", "midair"]
    else:
        pel = _pel(c, 0, -0.02, -0.05)
        pose = PoseParams(pelvis=pel, pelvis_rot=(-12, -18, -6), chest_rot=(6, 4, 0), look_at=c.tj("head"))
        pose.feet = {"l": Limb(v(-0.06 * H, ah, -0.02 * H), v(-0.2, 0.2, 1), v(-0.35, 0, 1))}
        hip = pel + v(0.05 * H, 0.0, 0)
        surf = tb.r["waist"] if tb else 0.09
        goal = aim - unit(v(aim[0], 0, aim[2])) * (surf + 0.02 * H)
        d = goal - hip
        leg = c.body.leg_len * 0.97
        foot = goal if np.linalg.norm(d) <= leg else hip + unit(d) * leg
        pose.feet["r"] = Limb(foot, v(0, 1, 0.3), unit(v(0, 0.7, 0.7)))  # ball of the foot, toes pulled back
        pose.hands["l"] = _arm(c, "l", "guard")
        pose.hands["r"] = _arm(c, "r", "swing_back", dz=0.25, dy=0.05)
        motion = unit(v(d[0], 0, d[2]))
        tags = ["kicking", "front kick", "leg up", "motion lines", "impact"]
    # the solver steps the kicker in/out until the ankle is exactly on the goal
    out = ActionOut(pose, tags=tags, motion=motion, impact=foot + unit(d) * 0.04 * H, reach=("r_ankle", goal),
                    airborne=not pose.grounded)
    if _armed(c):  # weapon swung out wide in the right hand for balance
        _grip(c, pose, v(0.3, 0.8, -0.1) * H, v(0.6, 0.6, -0.3), out)
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
    pose.feet["l"] = Limb(pel + v(-0.08, -0.22, -0.2) * H, v(0, 0.2, 1), v(0, -0.7, -0.6))  # tucked under
    pose.hands["l"] = _arm(c, "l", "guard")
    pose.hands["r"] = _arm(c, "r", "swing_back", dx=0.1)
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
    else:  # forearms stacked in front of the face, elbows down
        pose.hands["r"] = _arm(c, "r", "cover")
        pose.hands["l"] = _arm(c, "l", "guard_high", dz=0.05, dy=-0.05)
        out.tags += ["crossed arms", "guarding"]
    out.motion = -c.target_dir() * 0.3
    return out


@action("knocked_back", needs_pose=False,
        doc="Hit and flung backwards through the air away from the target (the attacker).")
def knocked_back(c: ActionCtx) -> ActionOut:
    H = c.H
    alt = 0.18 * H if c.altitude is None else c.altitude
    pel = _pel(c, 0, 0, -0.04) + v(0, alt, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(8, -30, 12), chest_rot=(0, -14, 6), head_rot=(15, -16, 10),
                      grounded=False)
    # the body is driven back, the limbs lag behind: arms trail forward and up, legs trail forward
    pose.hands["l"] = _arm(c, "l", "flail_fwd", dx=0.15, dy=0.1)
    pose.hands["r"] = _arm(c, "r", "flail", dz=0.25)
    pose.feet["l"] = Limb(pel + v(-0.1, -0.4, 0.18) * H, v(0, 0.5, 1))
    pose.feet["r"] = Limb(pel + v(0.12, -0.42, 0.05) * H, v(0, 0.2, 1))
    pose.fist = {"l": "claw", "r": "relaxed"}
    out = ActionOut(pose, airborne=True, motion=v(0, 0.25, -1),
                    tags=["being hit", "knocked back", "midair", "flying backwards", "pain",
                          "open mouth", "motion lines"])
    if _armed(c):
        J = _solve(c, pose)
        _grip(c, pose, J["r_wrist"], v(0.5, -0.1, -0.85), out)  # weapon trails the fall
        out.tags.append(c.weapon.tags)
    return out


@action("staggered", needs_pose=False, doc="Just took a hit: grounded, reeling back, guard broken.")
def staggered(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.04, -0.05)
    pose = PoseParams(pelvis=pel, pelvis_rot=(10, -12, 8), chest_rot=(0, -8, 0), head_rot=(16, -16, 8))
    pose.feet = _feet(c, (-0.12, 0.12), (0.1, -0.26), (-0.3, 0.2, 1), (0.3, 0.2, 1))
    # guard blown open: one hand jerks up toward the struck face, the other flung out low
    pose.hands["r"] = _arm(c, "r", "chin", dx=0.15, dz=-0.12)
    pose.hands["l"] = _arm(c, "l", "flail_low", dy=-0.12, dz=-0.05)
    pose.fist = {"l": "relaxed", "r": "claw"}
    out = ActionOut(pose, motion=v(0, 0, -1), tags=["staggering", "being hit", "pain", "clenched teeth"])
    if _armed(c):
        J = _solve(c, pose)
        _grip(c, pose, J["r_wrist"], v(0.4, -0.6, -0.5), out)
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
    else:  # fist raised high to hammer down, the other arm out for balance
        pose.hands["r"] = _arm(c, "r", "flail", dx=-0.32, dy=0.42, dz=0.12)
        pose.hands["l"] = _arm(c, "l", "balance", dy=0.1)
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
    J = _solve(c, pose)
    dirn = unit(c.tj("chest") - J["r_shoulder"])
    pose.hands["r"] = Limb(J["r_shoulder"] + dirn * c.body.arm_reach * 0.9, v(0.6, -1, -0.2), dirn + v(0, 0.6, 0))
    if c.p("style", "one_hand") == "two_hands":
        dl = unit(c.tj("chest") - J["l_shoulder"])
        pose.hands["l"] = Limb(J["l_shoulder"] + dl * c.body.arm_reach * 0.86, v(-0.6, -1, -0.2), dl + v(0, 0.6, 0))
    else:  # the other hand braces the casting arm's wrist... or stays back, open
        pose.hands["l"] = _arm(c, "l", "guard_lead", dx=0.12, dy=-0.12, dz=-0.12, tip=(0.2, 0.6, 0.5))
    pose.fist = {"l": "relaxed", "r": "flat"}
    tags = ["casting spell", "outstretched arm", "open hand", "glowing hand", "magic circle", "magic"]
    out = ActionOut(pose, tags=tags, motion=dirn * 0.3)
    if c.weapon.reach_class == "pole":  # staff raised in the off hand
        pose.hands["l"] = Limb(J["l_shoulder"] + v(-0.12, -0.18, 0.15) * H, v(-1, -0.6, -0.2))
        out.weapon_dirs["l"] = unit(v(-0.1, 1, 0.15))
        out.tags.append(c.weapon.tags)
    return out


@action("dodge", needs_pose=False, doc="Leaning/jumping back and aside to evade the target's attack.")
def dodge(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, -0.04, -0.1, -0.06)
    pose = PoseParams(pelvis=pel, pelvis_rot=(15, -16, -14), chest_rot=(5, -14, -10), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.2, -0.1), (0.1, 0.12), (-0.5, 0.2, 1), (0.2, 0.2, 1))
    pose.hands["l"] = _arm(c, "l", "balance", dz=-0.15)
    pose.hands["r"] = _arm(c, "r", "guard")
    out = ActionOut(pose, motion=v(-0.6, 0, -0.8), tags=["dodging", "leaning back", "evading"] + _base_tags(c))
    if _armed(c):
        _grip(c, pose, v(0.12, 0.75, 0.16) * H, v(0.2, 0.5, 0.85), out)
    return out


@action("charge", distance=lambda c: 1.6, needs_pose=False,
        doc="Running/charging at the target, body leaning into the sprint.")
def charge(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.04, 0.02)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 22, 0), chest_rot=(0, 10, 0), look_at=c.tj("head"))
    pose.feet["r"] = Limb(pel + v(0.08, -0.33, 0.18) * H, v(0, 0.2, 1))
    pose.feet["l"] = Limb(v(-0.07 * H, c.body.ankle_h + 0.03 * H, -0.3 * H), v(0, 0, 1), v(0, -0.6, 0.8))
    # arms pump opposite to the legs, elbows at ~90 degrees
    pose.hands["l"] = _arm(c, "l", "swing_fwd", dy=0.05)
    pose.hands["r"] = _arm(c, "r", "swing_back")
    out = ActionOut(pose, motion=v(0, 0, 1), tags=["running", "charging", "motion lines", "motion blur"])
    if _armed(c):
        J = _solve(c, pose)
        _grip(c, pose, J["r_wrist"], v(0.2, -0.15, -1), out)
        out.tags.append(c.weapon.tags)
    return out


@action("kneel", needs_pose=False, doc="Down on one knee (wounded/catching breath), looking at the target.")
def kneel(c: ActionCtx) -> ActionOut:
    H, b = c.H, c.body
    # right knee on the floor under the hip, toes tucked behind; left foot planted, shin vertical
    knee_r = v(b.hip_w, b.r["shin"], -0.04 * H)
    pel = v(0, knee_r[1] + b.thigh * 0.96 + 0.035 * H, -0.01 * H)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 12, 0), chest_rot=(0, 8, 0), look_at=c.tj("head"))
    pose.feet["l"] = Limb(v(-b.hip_w * 1.2, b.ankle_h, b.thigh * 0.85), v(-0.1, 0.3, 1))
    pose.feet["r"] = Limb(knee_r + v(0, b.ankle_h - knee_r[1] + 0.02 * H, -b.shin * 0.97), v(0, -0.3, 1),
                          v(0, -0.5, -1))
    out = ActionOut(pose, tags=["kneeling", "on one knee", "injured", "panting"])
    if _armed(c):  # leaning on the weapon
        _grip(c, pose, v(0.16, 0.42, 0.2) * H, v(0.05, -1, 0.15), out)
        out.tags.append(c.weapon.tags)
    else:
        pose.hands["r"] = _arm(c, "r", "hang", dz=0.15, dy=0.1)
        pose.fist["r"] = "relaxed"
    J = _solve(c, pose)  # forearm resting on the raised knee, hand hanging past it
    pose.hands["l"] = _own(c, pose, "l", J["l_knee"] + v(0.0, 0.035, 0.05) * H, v(-1, 0.2, -0.4), v(0, -0.6, 0.6))
    pose.fist["l"] = "relaxed"
    return out


@action("fallen", needs_pose=False, doc="Lying defeated on the ground (on back).")
def fallen(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = v(0, c.body.r["pelvis"] * 1.0, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, -90, 0), chest_rot=(0, -4, 6), head_rot=(25, 0, 0))
    pose.feet["l"] = Limb(v(-0.12 * H, 0.04 * H, 0.47 * H), v(0, 1, 0.3))
    pose.feet["r"] = Limb(v(0.05 * H, 0.15 * H, 0.38 * H), v(0, 1, 0.1))
    # lying on the back the chest frame is (out, toward the head, up to the sky): arms flopped on the floor
    pose.hands["l"] = Limb(v(-0.5, -0.4, -0.1) * c.body.arm_reach, v(-1, 0.2, -0.2), frame="chest")
    pose.hands["r"] = Limb(v(0.55, 0.32, -0.1) * c.body.arm_reach, v(1, -0.3, -0.2), frame="chest")
    pose.fist = {"l": "relaxed", "r": "relaxed"}
    return ActionOut(pose, tags=["lying", "on back", "defeated", "injured"])


@action("grab", distance=lambda c: 0.42 * c.H, needs_pose=True,
        doc="Grabbing the target by the collar with the right hand, left fist cocked back.")
def grab(c: ActionCtx) -> ActionOut:
    pel = _pel(c, 0, -0.04, 0.04)
    pose = PoseParams(pelvis=pel, pelvis_rot=(-10, 6, 0), chest_rot=(-6, 6, 0), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.09, 0.12), (0.1, -0.12))
    neck = c.tj("neck")
    tb = c.target_body
    collar = neck - c.target_dir() * ((tb.r["neck"] if tb else 0.04) + 0.02) - v(0, 0.03 * c.H, 0)
    J = _solve(c, pose)
    pose.hands["r"] = Limb(_reach(J, "r", collar, c.body.arm_reach * 0.97), v(0.7, -1, -0.2))
    pose.hands["l"] = _arm(c, "l", "cocked")
    pose.fist = {"l": True, "r": "grip"}
    return ActionOut(pose, tags=["grabbing", "collar grab", "angry", "clenched hand"], motion=v(0, 0, 0.4),
                     impact=collar)


@action("point", needs_pose=False, doc="Standing tall, pointing (or the weapon) at the target.")
def point(c: ActionCtx) -> ActionOut:
    H = c.H
    pose = PoseParams(look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.08, 0.06), (0.09, -0.06))
    J = _solve(c, pose)
    dirn = unit(c.tj("head") - J["r_shoulder"])
    out = ActionOut(pose, tags=["pointing", "confident"] + _base_tags(c))
    if _armed(c):
        _grip(c, pose, J["r_shoulder"] + dirn * c.body.arm_reach * 0.85, dirn, out)
        out.tags.append("pointing weapon")
    else:
        pose.hands["r"] = Limb(J["r_shoulder"] + dirn * c.body.arm_reach * 0.93, v(1, -1, 0))
        pose.fist["r"] = False
    pose.hands.setdefault("l", _arm(c, "l", "hang"))
    pose.fist["l"] = "relaxed"
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
        doc="Front headlock / guillotine: the target (use `headlocked`) is bent over with its head trapped "
            "under the right armpit, forearm under the throat, hands clasped at the chest.")
def headlock(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return stance(c)
    head = c.tw("head")
    yaw = _yaw(-c.t_fwd)  # face the victim
    f, r = _fwd_right(yaw)
    up = v(0, 1, 0)
    hr = c.target_body.head_r
    # hips back and low, chest over the head, wide base (sprawl)
    pose = PoseParams(pelvis=_pel(c, 0, -0.04, -0.06), pelvis_rot=(12, 18, 0), chest_rot=(-6, 10, 0))
    pose.feet = _feet(c, (-0.13, 0.04), (0.14, -0.14), (-0.4, 0.3, 1), (0.4, 0.3, 1))
    # the trapped head sits pinned against the right side of the chest, under the armpit
    for _ in range(4):
        J = _solve(c, pose)
        Rc = c.rig.chest_R
        P = J["r_shoulder"] + Rc @ v(-0.01 * H, -0.06 * H - hr, c.body.r["chest"] + hr * 0.9)
        dy = head[1] - P[1]
        if abs(dy) < 0.004:
            break
        pose.pelvis = pose.pelvis + v(0, float(np.clip(dy, -0.18 * H, 0.03 * H)) * 0.9, 0)
    origin = head - r * P[0] - f * P[2]
    origin = v(origin[0], 0, origin[2])
    L = lambda p: _mount_local(c, origin, yaw, p)  # noqa: E731
    throat, face = _throat(c)
    # forearm blade under the throat: the wrist comes out past the far side of the neck, the elbow
    # wraps over the back of the neck
    wrist = throat - r * 0.045 * H + face * 0.01 * H
    elbow = c.tw("neck") - face * 0.07 * H + r * 0.07 * H + up * 0.03 * H
    J = _solve(c, pose)
    pose.hands["r"] = Limb(L(wrist), _pole_to(J, "r", L(elbow), L(wrist)), _mount_dir(yaw, unit(up - r * 0.6)))
    # the left hand clasps the right wrist (gable grip), elbow out, pulling up
    clasp = wrist - r * 0.035 * H + up * 0.03 * H - f * 0.02 * H
    pose.hands["l"] = Limb(L(clasp), v(-1, -0.5, -0.4), _mount_dir(yaw, unit(r * 0.8 + up * 0.3)))
    pose.fist = {"l": "grip", "r": "grip"}
    pose.look_at = L(c.tw("head") + up * 0.05)
    return ActionOut(pose, tags=["headlock", "guillotine choke", "grappling", "arm around neck", "clenched teeth",
                                 "angry"], motion=v(0, 0.2, -0.3), impact=L(throat), mount=(origin, yaw))


@action("headlocked", doc="Bent over at the waist with the head trapped in the target's headlock, struggling.")
def headlocked(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.08, -0.06)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 50, 0), chest_rot=(0, 26, 0), head_rot=(0, -30, 0))
    pose.feet = _feet(c, (-0.11, -0.2), (0.1, 0.04), (-0.2, 0.4, 1), (0.2, 0.4, 1))
    # folded far enough that the head ends up at the captor's chest (bend more, then sink the knees)
    tb = c.target_body
    want = (tb.hip_h + tb.spine * 0.75) if tb else 0.62 * H
    for _ in range(8):
        J = _solve(c, pose)
        dy = J["head"][1] - want
        if dy < 0.01:
            break
        y, p_, c_ = pose.pelvis_rot
        if p_ < 66:
            pose.pelvis_rot, pose.chest_rot = (y, min(66.0, p_ + 6), 0), (0, min(32.0, pose.chest_rot[1] + 2), 0)
        else:
            pose.pelvis = pose.pelvis - v(0, min(dy, 0.12 * H), 0)
    J = _solve(c, pose)
    face = unit(J["nose"] - J["head"])
    throat = J["neck"] * 0.7 + J["head"] * 0.3 + face * (c.body.r["neck"] + 0.03 * H)
    # both hands pry at the forearm under the throat
    pose.hands["r"] = _own(c, pose, "r", throat + v(0.035, -0.01, 0) * H, v(1, -0.4, -0.5))
    pose.hands["l"] = _own(c, pose, "l", throat + v(-0.04, 0.0, 0.01) * H, v(-1, -0.4, -0.5))
    pose.fist = {"l": "grip", "r": "grip"}
    return ActionOut(pose, tags=["headlock", "bent over", "being choked", "struggling", "pain", "clenched teeth",
                                 "sweat", "grabbing another's arm"])


@action("rear_choke", needs_pose=True,
        doc="Rear naked choke from behind: elbow under the target's chin, choking hand on the other biceps, "
            "other hand behind the head.")
def rear_choke(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return stance(c)
    yaw = c.target_yaw
    f, r = _fwd_right(yaw)
    up = v(0, 1, 0)
    tb = c.target_body
    neck, head, pel_t = c.tw("neck"), c.tw("head"), c.tw("pelvis")
    pose = PoseParams(pelvis=_pel(c, 0, 0, 0), pelvis_rot=(-8, -8, 0), chest_rot=(-4, 2, 0))
    pose.feet = _feet(c, (-0.12, 0.02), (0.12, -0.12))
    # chest glued to the victim's back, chin beside the victim's head: lower the hips to match
    for _ in range(4):
        J = _solve(c, pose)
        dy = (neck[1] + 0.02 * H) - J["neck"][1]
        if abs(dy) < 0.004:
            break
        pose.pelvis = pose.pelvis + v(0, float(np.clip(dy, -0.2 * H, 0.04 * H)), 0)
    J = _solve(c, pose)
    chest_t = c.tw("chest")
    gap = tb.r["chest"] + c.body.r["chest"] * 0.95
    # cheek to cheek: the captor's head sits beside the victim's (to its right), not behind it
    side = r * (tb.head_r + c.body.head_r) * 0.95
    origin = chest_t - f * gap - f * (J["chest"][2]) + side
    origin = v(origin[0], 0, origin[2])
    L = lambda p: _mount_local(c, origin, yaw, p)  # noqa: E731
    throat = c.tw("neck") * 0.7 + c.tw("head") * 0.3 + f * (tb.r["neck"] + 0.012 * tb.height)
    # choking arm: elbow under the chin, forearm along the left side of the neck, hand at the biceps
    elbow = throat + f * 0.035 * H - up * 0.015 * H
    wrist = neck - r * (tb.r["neck"] + 0.05 * H) - f * 0.01 * H + up * 0.01 * H
    J = _solve(c, pose)
    pose.hands["r"] = Limb(L(wrist), _pole_to(J, "r", L(elbow), L(wrist)))
    # support hand flat on the back of the head, elbow up beside the victim's left ear
    back = head - f * (tb.head_r + 0.02 * H) - up * 0.01 * H
    l_elbow = head - r * (tb.head_r + 0.06 * H) - up * 0.04 * H
    pose.hands["l"] = Limb(L(back), _pole_to(J, "l", L(l_elbow), L(back)), _mount_dir(yaw, unit(up + f * 0.3)))
    pose.fist = {"l": "flat", "r": "grip"}
    pose.look_at = L(head + f * 0.5)
    return ActionOut(pose, tags=["chokehold", "rear naked choke", "grappling", "from behind", "clenched teeth"],
                     impact=L(throat), mount=(origin, yaw))


@action("choked", doc="Being choked from behind: arched back, clawing at the forearm across the throat.")
def choked(c: ActionCtx) -> ActionOut:
    H = c.H
    pose = PoseParams(pelvis=_pel(c, 0, -0.02, 0.02), pelvis_rot=(0, -6, 0), chest_rot=(0, -6, 0),
                      head_rot=(-14, -12, 0))
    pose.feet = _feet(c, (-0.08, 0.1), (0.09, 0.02))
    # dragged down and back onto a shorter captor: knees sag until the neck is at their shoulder line
    tb = c.target_body
    if tb is not None:
        want = tb.hip_h + tb.spine + tb.upper * 1.05
        for _ in range(6):
            J = _solve(c, pose)
            dy = J["neck"][1] - want
            if dy < 0.01:
                break
            pose.pelvis = pose.pelvis - v(0, min(dy, 0.1 * H), 0.25 * min(dy, 0.1 * H))
            y, p_, r_ = pose.pelvis_rot
            pose.pelvis_rot = (y, max(-12.0, p_ - 3), r_)
    J = _solve(c, pose)
    face = unit(J["nose"] - J["head"])
    throat = J["neck"] * 0.7 + J["head"] * 0.3 + face * (c.body.r["neck"] + 0.03 * H)
    # both hands pulling down on the forearm across the throat
    pose.hands["r"] = _own(c, pose, "r", throat + v(0.03, -0.04, 0.01) * H, v(1, -0.7, 0))
    pose.hands["l"] = _own(c, pose, "l", throat + v(-0.05, -0.03, 0.0) * H, v(-1, -0.7, 0))
    pose.fist = {"l": "claw", "r": "grip"}
    return ActionOut(pose, tags=["being choked", "struggling", "tears", "open mouth", "pain", "hands on another's arm"])


@action("leg_lock", needs_pose=True,
        doc="Straight ankle lock from ashi garami: sitting at the target's feet (target uses `leg_locked`), "
            "the trapped leg across the lap between the thighs, knees pinched on it, outside foot pushing the "
            "hip, inside foot hooked under the far glute, ankle in the right armpit, leaning back.")
def leg_lock(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return stance(c)
    side = c.p("leg", "r")
    other = "l" if side == "r" else "r"
    ankle, hip_t, hip_o = c.tw(f"{side}_ankle"), c.tw(f"{side}_hip"), c.tw(f"{other}_hip")
    tb, b = c.target_body, c.body
    up = v(0, 1, 0)
    d = unit(v(ankle[0] - hip_t[0], 0, ankle[2] - hip_t[2]))  # along the trapped leg, toward its foot
    pr = b.r["pelvis"]
    pose = PoseParams(pelvis=v(0, pr * 1.05, 0), pelvis_rot=(0, -30, 0), chest_rot=(0, 2, 0))
    # the shin runs under the right armpit, along the ribs: outside the chest, under the upper arm
    pit_off = v(max(0.0, b.r["chest"] + tb.r["shin"] - b.shoulder_w), -(b.r["upper_arm"] + tb.r["shin"] + 0.01 * H), 0)
    best = None
    for lean in np.arange(-5, -66, -2.5):  # lean back until the armpit is at the height of the ankle
        pose.pelvis_rot = (0, float(lean), 0)
        J = _solve(c, pose)
        pit = J["r_shoulder"] + c.rig.chest_R @ pit_off
        if best is None or abs(pit[1] - ankle[1]) < best[0]:
            best = (abs(pit[1] - ankle[1]), float(lean))
    pose.pelvis_rot = (0, best[1], 0)
    pose.chest_rot = (0, max(0.0, -best[1] - 40) * 0.5, 0)
    J = _solve(c, pose)
    pit = J["r_shoulder"] + c.rig.chest_R @ pit_off

    def place(yaw):
        f, r = _fwd_right(yaw)
        o = ankle - r * pit[0] - f * pit[2]
        return v(o[0], 0, o[2])

    # angle the body so the leg runs from the armpit diagonally across the lap, crossing the pelvis just
    # right of centre (between the thighs) - solved by Newton on the yaw, pivoting about the ankle
    xc = 0.025 * H
    yaw = _yaw(-d)

    def err(yw):
        hl = _mount_local(c, place(yw), yw, hip_t)
        t = (0 - pit[2]) / max(hl[2] - pit[2], 1e-3)  # where the leg line crosses the pelvis plane
        return pit[0] + (hl[0] - pit[0]) * t - xc
    for _ in range(8):
        e = err(yaw)
        if abs(e) < 0.003:
            break
        de = (err(yaw + 0.01) - e) / 0.01
        if abs(de) < 1e-4:
            break
        yaw -= float(np.clip(e / de, -0.3, 0.3))
    origin = place(yaw)
    f, r = _fwd_right(yaw)
    L = lambda p: _mount_local(c, origin, yaw, p)  # noqa: E731
    D = lambda w: _mount_dir(yaw, w)  # noqa: E731

    # legs: the outside foot's sole on the trapped hip, the inside foot hooked under the far side of the
    # buttocks between the target's legs; knees up and squeezed together on the trapped thigh
    o = unit(v(hip_t[0] - hip_o[0], 0, hip_t[2] - hip_o[2]))  # outward from the trapped hip
    outside = "r" if D(o)[0] > 0 else "l"
    inside = "l" if outside == "r" else "r"
    foot_out = hip_t + o * 0.6 * tb.hip_w + up * (tb.r["pelvis"] * 0.8) + d * 0.02 * H
    foot_in = hip_t - o * 1.4 * tb.hip_w + d * 0.06 * H
    foot_in = v(foot_in[0], b.ankle_h + 0.01 * H, foot_in[2])
    # each knee rides up on its own side of the trapped leg (the thighs clamp it, they don't overlap it)
    al, hl = L(ankle), L(hip_t)

    def line_x(z):  # the trapped leg's x (attacker frame) where it passes depth z
        return al[0] + (hl[0] - al[0]) * (z - al[2]) / max(hl[2] - al[2], 1e-3)
    for s_, foot, tip in ((outside, foot_out, D(unit(-d + up * 0.5))), (inside, foot_in, D(unit(-d * 0.6 + o * 0.6)))):
        fl = L(foot)
        sgn = 1.0 if fl[0] > line_x(fl[2]) else -1.0
        pose.feet[s_] = Limb(fl, v(sgn * 0.3, 1, 0.1), tip)
    pose.grounded = True
    # arms: right forearm blade under the Achilles, wrist hugged to the chest; left hand clasps it (gable grip)
    shin_r = tb.r["shin"]
    wrist = ankle - up * (shin_r + b.r["forearm"]) - r * 0.08 * H + d * 0.01 * H
    wrist[1] = max(wrist[1], b.r["forearm"] + 0.005)  # never through the floor
    J = _solve(c, pose)
    elbow = J["r_shoulder"] + c.rig.chest_R @ v(0.05 * H, -0.15 * H, -0.01 * H)  # tight to the ribs, under the shin
    pose.hands["r"] = Limb(L(wrist), _pole_to(J, "r", elbow, L(wrist)), D(unit(-r + up * 0.2)), palm=D(unit(up - r * 0.3)))
    clasp = wrist - r * 0.035 * H + up * 0.025 * H - d * 0.01 * H
    pose.hands["l"] = Limb(L(clasp), v(-1, -0.6, -0.3), D(unit(r * 0.8 - up * 0.2)), palm=D(unit(-up + r * 0.2)))
    pose.fist = {"l": "grip", "r": "grip"}
    pose.look_at = L(c.tw("head"))
    return ActionOut(pose, tags=["leg lock", "ankle lock", "grappling", "sitting", "on ground", "leaning back",
                                 "clenched teeth"], impact=L(ankle), mount=(origin, yaw),
                     press=[{"point": L(ankle - up * shin_r * 0.6), "dir": up, "radius": 0.03, "max_depth": 0.015}])


@action("leg_locked", doc="Flat on the back with one leg trapped in the target's ankle lock, tapping the floor.")
def leg_locked(c: ActionCtx) -> ActionOut:
    H = c.H
    side = c.p("leg", "r")
    sx = 1.0 if side == "r" else -1.0
    pr = c.body.r["pelvis"]
    pel = v(0, pr, 0)
    # on the back, curling up in pain to look at the leg
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, -90, 0), chest_rot=(0, 22, -6 * sx), head_rot=(0, 26, 0))
    ah_t = 0.19 * (c.target_body.height if c.target_body else 1.6)  # the attacker's armpit, leaning back
    J = _solve(c, pose)
    hip = J[f"{side}_hip"]
    leg = c.body.leg_len * 0.985
    dy = ah_t - hip[1]
    dz = float(np.sqrt(max(leg * leg - dy * dy, 0.04)))
    pose.feet[side] = Limb(v(sx * 0.06 * H, ah_t, hip[2] + dz), v(0, 1, 0), v(0, 0.75, 0.6))
    other = "l" if side == "r" else "r"
    # the free leg splayed out of the way of the captor's inside leg, knee bent outward
    pose.feet[other] = Limb(v(-sx * 0.27 * H, c.body.ankle_h, 0.3 * H), v(-sx * 1.0, 0.7, 0.2))
    # chest frame lying down is (out, toward the head, up): one hand slaps the floor, one grabs at the thigh
    pose.hands[other] = Limb(v(-sx * 0.55, -0.3, -0.12) * c.body.arm_reach, v(-sx, 0.3, 0.2), v(-sx * 0.6, -0.8, 0),
                             frame="chest")
    pose.hands[side] = _own(c, pose, side, J[f"{side}_hip"] + v(sx * 0.03, 0.05, 0.12) * H, v(sx, 0.2, -0.5))
    pose.fist = {other: "flat", side: "claw"}
    return ActionOut(pose, tags=["lying", "on back", "leg lock", "pain", "screaming", "tears", "tapping out"])


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
    horiz = _hip_reach(c, hip_h, point[1], 0.97)
    origin = point - f * horiz - r * c.body.hip_w
    origin = v(origin[0], 0, origin[2])
    pose = PoseParams(pelvis=_pel(c, 0, -0.02, 0), pelvis_rot=(-8, -16, -4), chest_rot=(6, 6, 0),
                      look_at=_mount_local(c, origin, yaw, c.tw("knee" if False else f"{side}_knee")))
    pose.feet["l"] = Limb(v(-0.07 * H, c.body.ankle_h, 0.0), v(-0.2, 0.2, 1), v(-0.3, 0, 1))
    # the sole stamps flat into the hollow, toes turned out
    pose.feet["r"] = _limb_world(c, origin, yaw, "r", point, (0.3, 1, 0.3), (0.8, -0.1, 0.55))
    pose.hands["l"] = _arm(c, "l", "guard_lead", dy=-0.1)
    pose.hands["r"] = _arm(c, "r", "balance", dz=-0.1)
    pose.fist = {"l": True, "r": "relaxed"}
    return ActionOut(pose, tags=["kicking", "low kick", "from behind", "sneak attack", "motion lines"],
                     motion=v(0, -0.3, 1), impact=_mount_local(c, origin, yaw, point), mount=(origin, yaw))


@action("knee_buckle", doc="The knee just got kicked out from behind: dropping toward one knee, off balance.")
def knee_buckle(c: ActionCtx) -> ActionOut:
    H = c.H
    side = c.p("leg", "l")
    sx = -1.0 if side == "l" else 1.0
    pel = _pel(c, 0, -0.14, 0.04)
    # the kick drives the knee forward and down: the body pitches forward over it, head snapping back
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 10, 8 * -sx), chest_rot=(0, 6, 0), head_rot=(0, -24, 0))
    ah = c.body.ankle_h
    pose.feet[side] = Limb(v(sx * 0.08 * H, ah + 0.07 * H, -0.2 * H), v(0, -0.6, 1), v(0, -0.5, -0.8))
    other = "r" if side == "l" else "l"
    pose.feet[other] = Limb(v(-sx * 0.09 * H, ah, 0.15 * H), v(0, 0.2, 1))
    # arms fly up and out to catch the balance
    # falling toward the buckled knee: that hand reaches down to catch the fall, the other flails out
    pose.hands[side] = _arm(c, side, "brace", dz=0.1)
    pose.hands[other] = _arm(c, other, "balance", dy=0.15, dz=-0.1)
    pose.fist = {"l": "claw", "r": "relaxed"}
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
    # torso folds forward over the support leg as the heel drives back and up; head turned to sight
    pose = PoseParams(pelvis=_pel(c, 0, -0.05, 0.05), pelvis_rot=(-10, 48, 10), chest_rot=(35, -12, 0),
                      look_at=_mount_local(c, origin, yaw, c.tw("head")))
    pose.feet["l"] = Limb(v(-0.06 * H, c.body.ankle_h, 0.1 * H), v(0, 0.2, 1))
    goal = _mount_local(c, origin, yaw, aim - d * (c.target_body.r["neck"] + 0.02))
    pose.feet["r"] = Limb(goal, v(0, -1, 0.1), v(0.2, -0.8, -0.5))
    pose.hands["l"] = _arm(c, "l", "balance", dz=0.2, dy=-0.1)
    pose.hands["r"] = _arm(c, "r", "guard", dz=-0.1)
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
    pel = _pel(c, 0, -0.1, 0.07)  # set stance, head slipped outside the target's folding body
    pose = PoseParams(pelvis=pel, pelvis_rot=(-24, 10, -8), chest_rot=(-14, 6, -10), look_at=c.tj("head"))
    pose.feet = _feet(c, (-0.11, 0.22), (0.14, -0.2), (0, 0.2, 1), (0.4, 0.1, 1))
    aim = c.aim("stomach")
    part = "chest" if c.p("aim") == "chest" else "stomach"
    surface, toward = _surface(c, aim, part)  # the skin under the palm and the push direction into it
    sink = 0.03  # how far the palm drives into the soft belly (the renderer dents the mesh to match)
    up = v(0, 1, 0)
    tip = unit(up - toward * (toward @ up) + toward * 0.2)  # fingers up along the body, heel leading
    hand = c.body.hand
    palm_c = surface + toward * sink  # centre of the palm, pressed in
    wrist = palm_c - tip * 0.38 * hand - toward * 0.012
    pose.hands["r"] = Limb(wrist, v(0.6, -1, 0), tip, palm=toward)
    pose.hands["l"] = _arm(c, "l", "chamber")  # the other hand pulled back to the hip (hikite)
    pose.fist = {"l": True, "r": "flat"}
    energy = c.p("style", "energy") == "energy"
    out = ActionOut(pose, tags=["palm strike", "open hand", "outstretched arm", "shouting"]
                    + (["glowing hand", "energy"] if energy else []),
                    motion=toward, impact=surface, reach=("r_wrist", wrist),
                    press=[{"point": palm_c, "dir": toward, "radius": 0.085, "max_depth": 0.07}])
    if energy:
        out.effect = c.p("effect", "energy blast, shockwave, glowing, light particles, blue energy, lens flare")
    return out


@action("doubled_over", doc="Just took a blow to the gut: folded around it, lifted onto the toes, mouth open.")
def doubled_over(c: ActionCtx) -> ActionOut:
    H = c.H
    lift = c.p("lift", 0.0) * H
    # folded at the hips around the blow (references: torso near horizontal, head at waist height),
    # rear pushed back over the heels, knees soft and turned in, feet flat. "impact": still folding
    # (fist buried, ~60 degrees); "after": all the way over (~85 degrees)
    held = c.p("_by") in ("palm_strike", "punch", "press_stomach")
    phase = c.p("phase", "impact" if held else "after")
    fold = (40, 22, -30) if phase == "impact" else (52, 30, -40)
    pel = _pel(c, 0, -0.07, -0.09) + v(0, lift, 0)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, fold[0], 0), chest_rot=(0, fold[1], 0), head_rot=(0, fold[2], 0),
                      grounded=lift <= 0)
    ah = c.body.ankle_h
    # knees bent forward and slightly in (pigeon-toed buckle)
    pose.feet["l"] = Limb(v(-0.075 * H, ah + lift * 0.6, -0.02 * H), v(0.25, 0.1, 1), v(0.15, -0.2, 1))
    pose.feet["r"] = Limb(v(0.075 * H, ah + lift * 0.9, 0.04 * H), v(-0.25, 0.1, 1), v(-0.15, -0.2, 1))
    tb = c.target_body
    if c.p("_by") == "knee_strike" and tb is not None:
        # held in a clinch: the head is dragged down to the captor's chest, knees sagging
        pose.grounded, lift = True, 0.0
        pose.pelvis = _pel(c, 0, -0.06, -0.06)
        pose.feet = _feet(c, (-0.09, -0.1), (0.09, 0.0), (0, 0.4, 1), (0, 0.4, 1))
        want = tb.hip_h + tb.spine * 0.85
        for _ in range(8):
            J = _solve(c, pose)
            dy = J["head"][1] - want
            if dy < 0.01:
                break
            y_, p_, r_ = pose.pelvis_rot
            if p_ < 55:
                pose.pelvis_rot = (y_, min(55.0, p_ + 6), r_)
            else:
                pose.pelvis = pose.pelvis - v(0, min(dy, 0.1 * H), 0)
    # the arms fold in around the blow. If the striking arm is still there (a fist or palm sunk into
    # the gut), one hand clutches at that forearm; the other goes to the belly beside the blow
    J = _solve(c, pose)
    Rc = c.rig.chest_R
    belly = J["spine"] * 0.7 + J["pelvis"] * 0.3 + Rc @ v(0, 0, c.body.r["waist"] + 0.02 * H)
    by = c.p("_by")
    if c.target is not None and by in ("palm_strike", "punch", "press_stomach"):
        side = "r" if by == "palm_strike" else "r"
        wr, el = c.tj(f"{side}_wrist"), c.tj(f"{side}_elbow")
        fa = unit(el - wr)
        out_l = unit(np.cross(fa, v(0, 1, 0)))  # beside the forearm, on the victim's left
        if out_l[0] > 0:
            out_l = -out_l
        # wrapping the attacker's wrist from the side, palm on it
        grab = wr + fa * 0.09 * H + out_l * (c.body.r["forearm"] + 0.01 * H) + v(0, 0.012 * H, 0)
        pose.hands["l"] = Limb(_reach(J, "l", grab, c.body.arm_reach * 0.97), v(-1, -0.3, -0.4),
                               unit(-fa * 0.3 + v(0, -0.2, 0) - out_l), palm=-out_l)
        pose.hands["r"] = _own(c, pose, "r", belly + Rc @ v(0.1 * H, -0.02 * H, 0), Rc @ v(1, -0.3, -0.4),
                               Rc @ v(-0.3, -1, 0.2))
        pose.fist = {"l": "grip", "r": "claw"}
    else:
        pose.hands["l"] = _own(c, pose, "l", belly, v(-1, -0.3, -0.3), Rc @ v(1, -0.3, 0.2))
        pose.hands["r"] = _arm(c, "r", "hang", dx=0.05, dy=0.05, dz=0.3)  # dangling loose toward the ground
        pose.fist = {"l": "claw", "r": "relaxed"}
    return ActionOut(pose, airborne=lift > 0 and not pose.grounded, motion=v(0, 0.15, -1),
                     tags=["doubled over", "stomach punch", "pain", "open mouth", "saliva", "wide-eyed",
                           "constricted pupils", "motion lines"])


@action("clutch_stomach", doc="Hunched over, both hands clutching the stomach after a nasty hit.")
def clutch_stomach(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.08, -0.03)
    pose = PoseParams(pelvis=pel, pelvis_rot=(8, 22, 0), chest_rot=(0, 22, 0), head_rot=(0, 2, 0),
                      look_at=c.tj("head") if c.target is not None else None)
    pose.feet = _feet(c, (-0.09, 0.08), (0.1, -0.12), (0, 0.4, 1), (0.2, 0.4, 1))
    J = _solve(c, pose)
    Rc = c.rig.chest_R
    belly = J["spine"] + Rc @ v(0, -0.02 * H, c.body.r["waist"] + 0.02 * H)
    pose.hands["r"] = _own(c, pose, "r", belly + Rc @ v(0.04, 0.02, 0) * H, Rc @ v(1, -0.4, -0.3), Rc @ v(-1, -0.3, 0.1))
    pose.hands["l"] = _own(c, pose, "l", belly + Rc @ v(-0.03, -0.04, 0) * H, Rc @ v(-1, -0.4, -0.3), Rc @ v(1, -0.3, 0.1))
    pose.fist = {"l": "claw", "r": "claw"}
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
    pose.hands["l"] = _arm(c, "l", "flail", dy=-0.2, dz=0.15)
    pose.hands["r"] = _arm(c, "r", "balance", dz=0.1, dy=0.05)
    pose.fist = {"l": "claw", "r": "relaxed"}
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
    axis = unit(c.tw("neck") - c.tw("pelvis"))
    front = unit(c.t_fwd - axis * (c.t_fwd @ axis))  # the belly faces down when the target is folded
    K = spine + front * (c.target_body.r["waist"] + 0.03)  # front of the target's stomach
    hip_h = c.body.hip_h - 0.035 * H
    hv_r = c.target_body.head_r
    best = None
    # the folded head comes in under the chin: lean back from the waist and, if needed, give ground,
    # until it clears the chest and face (the knee still rises into the stomach)
    for back in np.arange(0.0, 0.16, 0.02):
        dy = K[1] - hip_h
        horiz = float(np.sqrt(max(c.body.thigh ** 2 - dy ** 2, (0.06 * H) ** 2))) + back
        origin = K - f * horiz - r * c.body.hip_w
        origin = v(origin[0], 0, origin[2])
        hip = origin + v(0, hip_h, 0) + r * c.body.hip_w
        knee = hip + unit(K - hip) * c.body.thigh
        foot = knee + unit(v(0, -1, 0) - f * 0.7) * c.body.shin
        L = lambda p, o=origin: _mount_local(c, o, yaw, p)  # noqa: E731
        hv = L(head)
        for lean in range(6, -31, -4):
            pose = PoseParams(pelvis=_pel(c, 0, 0, 0), pelvis_rot=(0, float(lean), 0), chest_rot=(0, 2, 0),
                              look_at=hv)
            pose.feet["l"] = Limb(v(-0.07 * H, c.body.ankle_h, -0.05 * H), v(0, 0.2, 1))
            pose.feet["r"] = _limb_world(c, origin, yaw, "r", foot, (0, 0.8, 1))
            J = _solve(c, pose)
            seg = J["neck"] - J["pelvis"]
            t = float(np.clip((hv - J["pelvis"]) @ seg / (seg @ seg), 0, 1))
            gap = min(np.linalg.norm(hv - (J["pelvis"] + seg * t)) - c.body.r["chest"] - hv_r,
                      np.linalg.norm(hv - J["head"]) - c.body.head_r - hv_r)
            if best is None or gap > best[0]:
                best = (gap, origin, pose)
            if gap > 0.005:
                break
        if best[0] > 0.005:
            break
    _, origin, pose = best
    # Thai clinch: both hands cupped over the back of the head, forearms on the collarbones
    # palms cupped on the back of the head, fingers reaching down over it toward the nape
    face = unit(c.tw("nose") - head)
    back = head - face * (c.target_body.head_r + 0.012)
    nape = unit(c.tw("neck") - head)
    nape = unit(nape - face * (nape @ face))
    for s, sx in (("r", 1.0), ("l", -1.0)):
        palm_c = back + r * sx * 0.04 * H
        wrist = palm_c - nape * 0.42 * c.body.hand
        pose.hands[s] = _limb_world(c, origin, yaw, s, wrist, (sx, -0.4, 0), _mount_dir(yaw, nape))
        pose.hands[s].palm = _mount_dir(yaw, face)
    pose.fist = {"l": "relaxed", "r": "relaxed"}
    Kl = _mount_local(c, origin, yaw, K)
    return ActionOut(pose, tags=["knee strike", "kneeing", "clinch", "grabbing another's head", "angry", "impact"],
                     motion=v(0, 0.6, 0.8), impact=Kl, mount=(origin, yaw),
                     press=[{"point": Kl + _mount_dir(yaw, -front) * 0.03, "dir": _mount_dir(yaw, -front),
                             "radius": 0.07, "max_depth": 0.05}])


def _reach(J: dict, side: str, point: np.ndarray, reach: float) -> np.ndarray:
    """`point`, pulled back inside `reach` of the actual shoulder."""
    sh = J[f"{side}_shoulder"]
    d = point - sh
    n = np.linalg.norm(d)
    return point if n <= reach else sh + d / n * reach


def _mount_dir(yaw: float, d: np.ndarray) -> np.ndarray:
    """World direction -> local direction of an actor facing `yaw`."""
    f, r = _fwd_right(yaw)
    return v(d @ r, d[1], d @ f)


def _throat(c: ActionCtx) -> tuple:
    """Target's throat (world) and the direction its face points."""
    face = unit(c.tw("nose") - c.tw("head"))
    neck = c.tw("neck") * 0.7 + c.tw("head") * 0.3
    return neck + face * (c.target_body.r["neck"] + 0.012 * c.target_body.height), face


@action("press_stomach", distance=lambda c: 0.36 * c.H, needs_pose=True,
        doc="Both open palms pressed flat into the target's stomach, sinking into the soft belly.")
def press_stomach(c: ActionCtx) -> ActionOut:
    H = c.H
    pel = _pel(c, 0, -0.03, 0.03)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 3, 0), chest_rot=(0, 5, 0))
    pose.feet = _feet(c, (-0.08, 0.05), (0.08, -0.04))
    tb = c.target_body
    # the bare belly: around the navel, just under the ribs
    belly = c.tj("spine") * 0.8 + c.tj("pelvis") * 0.2
    toward = unit(v(belly[0], 0, belly[2]))
    across = unit(np.cross(v(0, 1, 0), toward))  # the attacker's left, seen along `toward`
    r_t = tb.r["waist"] if tb else 0.08
    sink = c.p("sink", 0.035)
    hand = c.body.hand
    presses = []
    for s, sx in (("r", -1.0), ("l", 1.0)):
        # palms side by side over the navel, fingers fanned up and slightly inward
        centre = belly + across * sx * 0.045 * H - toward * (r_t * 0.85 - sink)
        tip = unit(v(0, 1, 0) * 0.9 - across * sx * 0.35 + toward * 0.15)
        wrist = centre - tip * 0.42 * hand - toward * 0.012
        pose.hands[s] = Limb(wrist, v(0.8 if s == "r" else -0.8, -1, -0.3), tip, palm=toward)
        # the heel/centre of the palm sinks deepest, the fingers a little less
        presses.append({"point": centre, "dir": toward, "radius": 0.06, "max_depth": 0.06})
        presses.append({"point": centre + tip * 0.45 * hand - toward * sink * 0.35, "dir": toward,
                        "radius": 0.045, "max_depth": 0.04})
    pose.fist = {"l": "flat", "r": "flat"}
    look = c.p("look", "face")
    pose.look_at = c.tj("head") if look == "face" else belly
    mid_w = (pose.hands["l"].target + pose.hands["r"].target) / 2
    R = c.body.arm_reach
    # a taller presser bends forward from the hips (then the knees) until the hands are in easy reach
    for _ in range(10):
        J = _solve(c, pose)
        sh_mid = (J["l_shoulder"] + J["r_shoulder"]) / 2
        dy = sh_mid[1] - mid_w[1]
        if dy < 0.62 * R:
            break
        y_, p_, r_ = pose.pelvis_rot
        if pose.pelvis[1] > c.rig.rest_pelvis()[1] - 0.14 * H:  # sink into the knees first
            pose.pelvis = pose.pelvis - v(0, min(dy - 0.62 * R, 0.03 * H), 0.01 * H)
            pose.feet = _feet(c, (-0.1, 0.1), (0.1, -0.06), (-0.3, 0.3, 1), (0.3, 0.3, 1))
        elif p_ < 24:
            pose.pelvis_rot = (y_, p_ + 5, r_)
        else:
            break
    # stand where the elbows are comfortably bent (shoulders ~0.8 arm lengths from the hands)
    horiz = np.sqrt(max((0.8 * R) ** 2 - dy ** 2, (0.5 * R) ** 2))
    want = mid_w - unit(v(mid_w[0] - sh_mid[0], 0, mid_w[2] - sh_mid[2])) * horiz
    return ActionOut(pose, tags=["hands on another's stomach", "pressing", "open hands", "close-up"],
                     motion=toward * 0.2, impact=belly - toward * r_t, press=presses,
                     reach=("r_shoulder", v(want[0], J["r_shoulder"][1], want[2]) + (J["r_shoulder"] - sh_mid)))


@action("flinch", doc="Being touched/pressed: torso pulling back from the contact, hands catching the wrists, looking down.")
def flinch(c: ActionCtx) -> ActionOut:
    H = c.H
    # the belly gives under the hands and the shoulders draw back from them; eyes down on the hands
    pel = _pel(c, 0, -0.015, -0.03)
    pose = PoseParams(pelvis=pel, pelvis_rot=(0, 2, 0), chest_rot=(0, -10, 0), head_rot=(0, 30, 0))
    pose.feet = _feet(c, (-0.07, 0.0), (0.07, -0.05))
    J = _solve(c, pose)
    if c.target is not None and c.p("hands", "grab") == "grab":  # catching the attacker's wrists
        for s, o in (("l", "r"), ("r", "l")):
            wr, el = c.tj(f"{o}_wrist"), c.tj(f"{o}_elbow")
            fa = unit(el - wr)
            # around the forearm just behind the wrist, from above, clear of the pressing hand
            g = wr + fa * 0.11 * H + v(0, c.target_body.r["forearm"] + 0.022 * H, 0)
            pose.hands[s] = Limb(_reach(J, s, g, c.body.arm_reach * 0.95), v(1 if s == "r" else -1, -0.6, -0.4),
                                 unit(fa * -0.2 + v(0, -0.3, 0) + v(-1 if s == "r" else 1, 0, 0) * 0.5),
                                 palm=v(0, -1, 0))
        pose.fist = {"l": "grip", "r": "grip"}
    else:  # hands flinching up and out of the way
        _arms(c, pose, "flail_low", "flail_low", dy=0.1)
        pose.fist = {"l": "relaxed", "r": "relaxed"}
    pose.look_at = J["spine"] + v(0, 0, 0.25 * H)  # down at the hands on the belly
    return ActionOut(pose, tags=["flinch", "embarrassed", "blush", "looking down", "grabbing another's wrist"])


@action("ear_pull", distance=lambda c: 0.3 * c.H, needs_pose=True,
        doc="Pinching the target's ear between thumb and finger and pulling it up and out (target: `ear_pulled`).")
def ear_pull(c: ActionCtx) -> ActionOut:
    H = c.H
    if c.target_world is None:
        return stance(c)
    pose = PoseParams(pelvis=_pel(c, 0, -0.01, 0), pelvis_rot=(-10, 2, 0), chest_rot=(-8, 4, 0))
    pose.feet = _feet(c, (-0.08, 0.04), (0.08, -0.04))
    ear = c.tj(f'{c.p("ear", "l")}_ear')  # by default the target's left ear, on my right-hand side
    head = c.tj("head")
    out_dir = unit(ear - head)  # outward from the skull through the ear
    hr = c.target_body.head_r
    pinch = ear + out_dir * 0.18 * hr + v(0, 0.12 * hr, 0)  # the rim of the ear, a little above the lobe
    hand = c.body.hand
    tip = unit(-out_dir * 0.6 + v(0, 0.45, 0) + v(0, 0, 0.2))
    wrist = pinch - tip * 0.62 * hand + out_dir * 0.01
    J = _solve(c, pose)
    pose.hands["r"] = Limb(wrist, v(0.8, -0.6, -0.3), tip, palm=unit(-out_dir * 0.2 + v(0, -1, 0) * 0.3 +
                                                                       unit(head - wrist)))
    pose.hands["l"] = _arm(c, "l", "fist_side", dy=0.15, dz=0.1)  # hand on the hip
    pose.fist = {"r": "pinch", "l": "relaxed"}
    pose.look_at = head
    sh = J["r_shoulder"]
    # arm's length (elbow soft), and never so close that the faces/chests meet
    horiz = max(np.sqrt(max((0.86 * c.body.arm_reach) ** 2 - (wrist[1] - sh[1]) ** 2, 0.0)),
                c.body.r["chest"] + hr + 0.06 * H - abs(wrist[0] - head[0]) * 0.5)
    want = wrist - unit(v(wrist[0] - sh[0], 0, wrist[2] - sh[2])) * horiz
    pull = unit(out_dir + v(0, 0.8, 0))
    return ActionOut(pose, tags=["ear pull", "pinching", "pulling another's ear", "smirk", "close-up"],
                     motion=pull * 0.3, impact=pinch,
                     press=[{"point": pinch, "dir": pull, "radius": 0.035, "depth": -0.012}],
                     reach=("r_shoulder", v(want[0], sh[1], want[2])))


@action("ear_pulled", doc="Ear being pinched and pulled: head tilted toward the pull, shoulder hunched, wincing.")
def ear_pulled(c: ActionCtx) -> ActionOut:
    H = c.H
    # which ear is pinched: by default the left (the one facing a right-handed puller's right hand)
    side = c.p("ear", "l")
    sx = 1.0 if side == "r" else -1.0
    # leaning and tilting toward the pull (+roll tilts toward the right in `euler`), rising onto that
    # side's toes, face turned a little toward the puller
    pose = PoseParams(pelvis=_pel(c, 0, -0.01, 0), pelvis_rot=(0, 0, 4 * sx), chest_rot=(-6 * sx, 0, 10 * sx),
                      head_rot=(-12 * sx, -16, 22 * sx))
    pose.feet = _feet(c, (-0.07, 0.0), (0.07, 0.0))
    if side == "l":
        pose.feet["l"] = Limb(v(-0.08 * c.H, c.body.ankle_h + 0.03 * H, 0.0), v(0, 0.3, 1), v(0, -0.6, 0.8))
    else:
        pose.feet["r"] = Limb(v(0.08 * c.H, c.body.ankle_h + 0.03 * H, 0.0), v(0, 0.3, 1), v(0, -0.6, 0.8))
    J = _solve(c, pose)
    other = "r" if side == "l" else "l"
    if c.target is not None:  # grabbing the pinching wrist
        wr, el = c.tj("r_wrist"), c.tj("r_elbow")
        g = wr + unit(el - wr) * 0.06 * H
        pose.hands[side] = Limb(_reach(J, side, g, c.body.arm_reach * 0.95), v(sx, -0.8, -0.2), palm=unit(wr - J["head"]))
        pose.fist[side] = "grip"
    else:
        pose.hands[side] = _arm(c, side, "flail", dy=0.1, dz=0.2)
    pose.hands[other] = _arm(c, other, "flail_low", dy=-0.05, dz=0.1)
    pose.fist[other] = "claw"
    return ActionOut(pose, tags=["ear pull", "wince", "one eye closed", "pain", "tears", "open mouth"])


def _surface(c: ActionCtx, aim: np.ndarray, part: str) -> tuple:
    """Where a blow aimed at `aim` meets the target's skin, and the inward normal there (attacker-local).
    The torso's front faces along its own front normal, not toward the attacker: on a body leaning or
    folding forward the belly faces down and the chest is in front of it."""
    tb = c.target_body
    if part == "head":
        n = unit(c.tj("nose") - c.tj("head"))
        n = unit(n * 0.6 + unit(-aim * v(1, 0, 1)) * 0.4)  # mostly the face, partly toward me
        return aim + n * (tb.head_r if tb else 0.11) * 0.35, -n
    axis = unit(c.tj("neck") - c.tj("pelvis"))
    me = unit(-v(aim[0], 0, aim[2]))  # from the target toward me
    if c.target_origin is not None and c.to_local is not None:
        # the front of the torso is where the target faces (a folded belly faces down-forward), not
        # whichever side happens to point at me
        o = np.asarray(c.target_origin, float)
        tf = c.to_local(o + c.t_fwd) - c.to_local(o)
        if tf @ me > 0.2:
            me = unit(me * 0.35 + unit(tf) * 0.65)
    n = me - axis * (me @ axis)
    n = unit(n) if np.linalg.norm(n) > 1e-6 else me
    r = (tb.r["chest"] if part == "chest" else tb.r["waist"]) if tb else 0.08
    return aim + n * r, -n
