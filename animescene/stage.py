"""Scene solver: placement, interaction-aware posing, grounding and collisions.

Input is a parsed scene (characters + one action each). Output is a
`SolvedScene`: every character's world-space joints, weapons and per-actor
metadata (tags, airborne, motion, impact point), ready for rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import ACTIONS, DISTANCE, NEEDS_TARGET_POSE, ActionCtx, ActionOut
from .characters import Character
from .rig import PoseParams, Rig
from .vec import frame_from, unit, v
from .weapons import get_weapon

REACTIONS = {"knocked_back", "staggered", "dodge", "fallen", "kneel", "headlocked", "choked", "leg_locked",
             "knee_buckle", "doubled_over", "clutch_stomach", "stumble", "flinch", "ear_pulled"}
# reactions whose body is moved so the attacker's blow actually lands on it
HIT_REACTIONS = {"knocked_back", "staggered", "knee_buckle", "doubled_over", "stumble"}
# attackers that hold on to the victim's actual (reacting) body rather than its pre-hit stance
HOLDS = {"headlock", "rear_choke", "leg_lock", "knee_strike", "palm_strike", "press_stomach", "ear_pull"}
ATTACKS = {"slash", "thrust", "leap_attack", "punch", "kick", "flying_kick", "charge", "palm_strike",
           "spinning_back_kick", "back_knee_kick", "knee_strike", "headlock", "rear_choke", "leg_lock"}


@dataclass
class Actor:
    char: Character
    action: str = "stance"
    target: str | None = None
    params: dict = field(default_factory=dict)
    at: tuple | None = None  # world (x, z) of the rest pelvis
    facing: float | str | None = None  # degrees (0 = +z) or another actor's name
    altitude: float | None = None
    bearing: float | None = None  # where to stand around the target, degrees
    distance: float | None = None
    # solved
    origin: np.ndarray = None
    yaw: float = 0.0
    joints: dict = None  # world
    local: dict = None
    out: ActionOut = None
    weapons: list = field(default_factory=list)  # [(pommel, grip, tip, half_width)]
    auto_placed: bool = False
    anchor: np.ndarray = None  # where the placement put the actor (before hit alignment)
    mounted: bool = False  # placed by its action relative to the target's body
    turned: bool = False  # its action chose its facing

    @property
    def name(self):
        return self.char.name

    @property
    def F(self) -> np.ndarray:
        return frame_from(v(np.sin(self.yaw), 0, np.cos(self.yaw)))

    def to_world(self, p):
        return self.origin + self.F @ p

    def to_local(self, p):
        return self.F.T @ (p - self.origin)


@dataclass
class SolvedScene:
    actors: list
    spec: dict

    def actor(self, name) -> Actor:
        for a in self.actors:
            if a.name == name:
                return a
        raise KeyError(name)

    def all_points(self) -> np.ndarray:
        pts = []
        for a in self.actors:
            pts += list(a.joints.values())
            for w in a.weapons:
                pts += [w[0], w[2]]
        return np.array(pts)


def _yaw_to(src: np.ndarray, dst: np.ndarray) -> float:
    d = dst - src
    return float(np.arctan2(d[0], d[2]))


def _seg_dist(p1, q1, p2, q2) -> float:
    """Closest distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-9 else 0.0
    t = (b * s + f) / e if e > 1e-9 else 0.0
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


class Stage:
    def __init__(self, actors: list[Actor], spec: dict | None = None):
        self.actors = actors
        self.by = {a.name: a for a in actors}
        self.spec = spec or {}
        az = np.deg2rad(self.spec.get("camera", {}).get("azimuth", 0.0))
        self.screen_right = v(np.cos(az), 0, -np.sin(az))  # world direction of screen-right
        for a in actors:
            if a.action not in ACTIONS:
                raise ValueError(f"{a.name}: unknown action {a.action!r}; choose from {sorted(ACTIONS)}")
            if a.target is not None and a.target not in self.by:
                raise ValueError(f"{a.name}: target {a.target!r} is not in the scene")
            if a.target == a.name:
                raise ValueError(f"{a.name} cannot target itself")

    # ------------------------------------------------------------------ placement
    def _ctx(self, a: Actor, with_pose=True) -> ActionCtx:
        rig = Rig(a.char.body)
        ctx = ActionCtx(body=a.char.body, rig=rig, weapon=get_weapon(a.char.weapon), params=a.params,
                        altitude=a.altitude, origin=a.origin)
        if a.target:
            t = self.by[a.target]
            ctx.target_body = t.char.body
            if t.action == "block" and t.target == a.name:
                ctx.params = {**a.params, "_clash": True}
            if t.target == a.name:  # reactions can shape themselves to what is being done to them
                ctx.params = {**ctx.params, "_by": t.action}
            if a.action == "block" and t.action in ATTACKS:
                ctx.params = {**a.params, "_incoming": True}
            if with_pose and t.joints is not None and a.origin is not None:
                joints = self._proxy(t, a) if self._is_hit_victim(t, a) else t.joints
                ctx.target = {k: a.to_local(p) for k, p in joints.items()}
                ctx.target_world = joints
                ctx.target_yaw = t.yaw
                ctx.target_origin = t.anchor if t.anchor is not None else t.origin
                ctx.to_local = a.to_local
                if t.weapons:
                    pom, grip, tip, _ = t.weapons[0]
                    ctx.target_weapon = (a.to_local(grip), a.to_local(tip))
        return ctx

    @staticmethod
    def _is_hit_victim(t: Actor, attacker: Actor) -> bool:
        # a body blow that the victim folds around is aimed at the folded body itself
        folded = t.action == "doubled_over" and attacker.action == "punch" and (
            attacker.params.get("style") == "body" or attacker.params.get("aim") in ("stomach", "body", "gut"))
        return (t.action in HIT_REACTIONS and attacker.target == t.name and attacker.action not in REACTIONS
                and attacker.action not in HOLDS and not folded)

    def _proxy(self, t: Actor, attacker: Actor) -> dict:
        """The victim standing (pre-hit) where it was struck, facing the attacker."""
        rig = Rig(t.char.body)
        local = rig.solve(PoseParams())
        yaw = t.yaw  # its facing (toward the attacker unless the scene turned it away)
        F = frame_from(v(np.sin(yaw), 0, np.cos(yaw)))
        base = v(t.anchor[0], 0, t.anchor[2])
        return {k: base + F @ p for k, p in local.items()}

    def _step_in(self):
        """Auto-placed strikers whose blow falls short (or overshoots) shift along their facing."""
        for a in self.actors:
            if (not a.auto_placed or a.mounted or a.out is None or a.out.reach is None
                    or a.distance is not None):
                continue
            joint, goal = a.out.reach
            miss = a.F @ (goal - a.local[joint])
            fwd = a.F @ v(0, 0, 1)
            along = float(np.clip(miss @ fwd, -0.3, 0.3))
            if abs(along) > 0.01:
                a.origin = a.origin + fwd * along
                a.anchor = a.anchor + fwd * along
                self._pose(a)

    def _align_victims(self):
        """Move each hit victim so the attacker's contact point lands on the struck body part."""
        for t in self.actors:
            if t.action not in HIT_REACTIONS or t.joints is None:
                continue
            att = next((a for a in self.actors if a.target == t.name and a.out is not None
                        and a.out.impact is not None and a.action not in REACTIONS and not a.mounted), None)
            if att is None:
                continue
            impact = att.to_world(att.out.impact)
            proxy = self._proxy(t, att)
            part = min(("head", "neck", "chest", "spine", "pelvis"),
                       key=lambda k: np.linalg.norm(proxy[k] - impact))
            motion = att.F @ att.out.motion
            # a moment after contact: the struck part has already travelled with the blow
            # a pressed contact (palm, fist sunk into the gut) keeps the surface on the effector
            r_part = {"head": t.char.body.head_r, "neck": t.char.body.r["neck"], "chest": t.char.body.r["chest"],
                      "spine": t.char.body.r["waist"], "pelvis": t.char.body.r["pelvis"]}[part]
            held = bool(att.out.press)
            goal = impact + unit(motion) * (0.28 if t.action == "knocked_back" else 0.08)
            # a contact made on the reacting body itself (palm/fist sunk in) is exact already: the
            # attacker stepped to it, so only resolve interpenetration
            delta = v(0, 0, 0) if held else goal - t.joints[part]
            if t.action != "knocked_back" or t.out is None or not t.out.airborne:
                delta[1] = 0.0
            else:  # lifted by the blow, not launched into orbit
                delta[1] = float(np.clip(delta[1], -0.05, 0.12))
            # a blow never drags its victim toward the attacker (beyond undoing its own recoil lean)
            flat = unit(v(motion[0], 0, motion[2]))
            back = delta @ flat
            if back < -0.3:
                delta = delta - flat * (back + 0.3)
            # keep flying along the blow until the bodies no longer pass through each other
            step = unit(v(motion[0], 0, motion[2])) * 0.05
            tol = -0.06 if t.action == "doubled_over" else 0.0  # a folded body may lean onto its attacker
            if held or t.action == "doubled_over":
                # folded over the arm: the head and shoulders pass outside the striking arm, not into the
                # attacker's chest (the blow came from that side)
                side = att.F @ v(1 if att.out.reach is None or att.out.reach[0].startswith("r") else -1, 0, 0)
                step = unit(side * 0.85 + unit(v(motion[0], 0, motion[2])) * 0.3) * 0.03
                tol = 0.0
            hr = t.char.body.head_r + att.char.body.head_r
            for _ in range(30):
                heads = np.linalg.norm(t.joints["head"] + delta - att.joints["head"]) - hr
                if self._body_clearance(t, att, delta) > tol and heads > 0.0:
                    break
                delta = delta + step
            t.origin = t.origin + delta
            t.joints = {k: p + delta for k, p in t.joints.items()}
            t.weapons = [tuple(x + delta if isinstance(x, np.ndarray) else x for x in w) for w in t.weapons]

    @staticmethod
    def _segments(a: Actor, offset=None, limbs=True):
        J, R = a.joints, a.char.body.r
        o = offset if offset is not None else 0.0
        segs = [("pelvis", "neck", R["chest"]), ("neck", "head", a.char.body.head_r)]
        if limbs:
            segs += [(f"{s}_hip", f"{s}_knee", R["thigh"]) for s in "lr"]
            segs += [(f"{s}_knee", f"{s}_ankle", R["shin"]) for s in "lr"]
        return [(J[p] + o, J[q] + o, r) for p, q, r in segs]

    def _body_clearance(self, victim: Actor, attacker: Actor, delta) -> float:
        """Smallest surface gap between the victim's and the attacker's torso, head and legs."""
        gap = np.inf
        for p1, q1, r1 in self._segments(victim, delta):
            for p2, q2, r2 in self._segments(attacker):
                gap = min(gap, _seg_dist(p1, q1, p2, q2) - r1 - r2)
        return gap

    def _distance(self, a: Actor) -> float:
        if a.distance is not None:
            return a.distance
        fn = DISTANCE.get(a.action)
        return fn(self._ctx(a, with_pose=False)) if fn else 1.3

    def place(self):
        placed = {}
        for a in self.actors:
            if a.at is not None:
                a.origin = v(a.at[0], 0, a.at[1])
                placed[a.name] = a

        def centroid():
            pts = [p.origin for p in placed.values()]
            return np.mean(pts, axis=0) if pts else v(0, 0, 0)

        def around(anchor: Actor, dist: float, bearing: float | None, other: Actor):
            if bearing is not None:
                d = v(np.sin(np.deg2rad(bearing)), 0, np.cos(np.deg2rad(bearing)))
            else:
                # side by side across the screen (readable silhouettes), on the side away
                # from everyone else; the first pair puts the attacker on the left.
                others = [p.origin for p in placed.values() if p is not anchor]
                side = -1.0
                if others:
                    off = (anchor.origin - np.mean(others, axis=0)) @ self.screen_right
                    side = 1.0 if off >= 0 else -1.0
                d = self.screen_right * side
            return anchor.origin + d * dist

        progress = True
        while progress and len(placed) < len(self.actors):
            progress = False
            for a in self.actors:
                if a.name in placed:
                    continue
                if a.target and a.target in placed:  # stand where my action needs me
                    a.origin = around(placed[a.target], self._distance(a), a.bearing, a)
                else:  # someone placed is acting on me: be where their action lands
                    src = next((p for p in placed.values() if p.target == a.name), None)
                    if src is None:
                        continue
                    a.origin = around(src, self._distance(src), a.bearing, a)
                a.auto_placed = True
                placed[a.name] = a
                progress = True
        for i, a in enumerate(self.actors):  # leftovers: spread on an arc
            if a.name not in placed:
                ang = np.deg2rad(-40 + 80 * i / max(1, len(self.actors) - 1))
                a.origin = v(3.0 * np.sin(ang), 0, -1.5 + 1.5 * np.cos(ang))
                a.auto_placed = True
                placed[a.name] = a
        for a in self.actors:
            a.anchor = a.origin.copy()
        self._face()

    def _face(self):
        c = np.mean([a.origin for a in self.actors], axis=0)
        for a in self.actors:
            if a.mounted or a.turned:
                continue
            if isinstance(a.facing, (int, float)):
                a.yaw = np.deg2rad(a.facing)
            elif a.facing == "away":  # back turned to whoever is acting on it (or its target)
                src = self.by.get(a.target) or next((p for p in self.actors if p.target == a.name), None)
                a.yaw = _yaw_to(a.origin, src.origin) + np.pi if src is not None else 0.0
            elif isinstance(a.facing, str):
                a.yaw = _yaw_to(a.origin, self.by[a.facing].origin)
            elif a.target:
                a.yaw = _yaw_to(a.origin, self.by[a.target].origin)
            else:
                src = next((p for p in self.actors if p.target == a.name), None)
                a.yaw = _yaw_to(a.origin, src.origin if src else c) if (src or len(self.actors) > 1) else 0.0

    # ------------------------------------------------------------------ posing
    def _pose(self, a: Actor, with_pose=True):
        ctx = self._ctx(a, with_pose)
        out = ACTIONS[a.action](ctx)
        if out.mount is not None and a.at is None:
            for _ in range(3):  # the action placed itself on the target's body: re-pose there
                origin, yaw = out.mount
                moved = np.linalg.norm(v(origin[0], 0, origin[2]) - a.origin) + abs(yaw - a.yaw)
                a.origin = v(origin[0], 0, origin[2])
                a.anchor = a.origin.copy()
                a.yaw = yaw
                a.mounted = True
                ctx = self._ctx(a, with_pose)
                out = ACTIONS[a.action](ctx)
                if moved < 1e-3:
                    break
        if out.turn is not None and abs(out.turn - a.yaw) > 1e-4:
            a.yaw = out.turn
            a.turned = True
            ctx = self._ctx(a, with_pose)
            out = ACTIONS[a.action](ctx)
        rig = ctx.rig
        b = a.char.body
        # Grounding moves the pelvis, then the limbs are re-solved so hand targets stay exact.
        for _ in range(3):
            local = rig.solve(out.pose)
            if out.pose.grounded and not out.airborne:
                low = min(local["l_ankle"][1] - b.ankle_h, local["r_ankle"][1] - b.ankle_h,
                          local["l_toe"][1] - b.r["foot"], local["r_toe"][1] - b.r["foot"],
                          local["l_knee"][1] - b.r["shin"], local["r_knee"][1] - b.r["shin"],
                          local["pelvis"][1] - b.r["pelvis"])
                shift = v(0, -low, 0)
            else:
                low = min(p[1] for p in local.values())
                shift = v(0, max(0.0, 0.06 - low), 0)
            if abs(shift[1]) < 1e-4:
                break
            base = rig.rest_pelvis() if out.pose.pelvis is None else np.asarray(out.pose.pelvis, float)
            out.pose.pelvis = base + shift
            if not out.pose.grounded or out.airborne:  # airborne: everything rides along
                for limb in list(out.pose.hands.values()) + list(out.pose.feet.values()):
                    if limb.frame == "local":
                        limb.target = np.asarray(limb.target, float) + shift
        a.local, a.out = local, out
        a.joints = {k: a.to_world(p) for k, p in local.items()}
        w = get_weapon(a.char.weapon)
        a.weapons = []
        if w.length > 0:
            for hand, d in out.weapon_dirs.items():
                dw = a.F @ d
                grip = (a.joints[f"{hand}_wrist"] + a.joints[f"{hand}_hand"]) / 2
                a.weapons.append((grip - dw * w.handle, grip, grip + dw * w.length, w.width))

    def _order(self):
        # reactions first, then actions needing the target's pose; blocks last
        def key(a: Actor):
            if a.action in REACTIONS:
                return 0
            if a.action == "block":
                return 2
            return 1
        return sorted(self.actors, key=key)

    def solve(self, iterations: int = 3) -> SolvedScene:
        self.place()
        for a in self.actors:  # initial guess
            self._pose(a, with_pose=False)
        for _ in range(2):
            for _ in range(iterations):
                for a in self._order():
                    self._pose(a, with_pose=a.action in NEEDS_TARGET_POSE or a.target is not None)
                self._step_in()
                self._align_victims()
            if not self._resolve_collisions():
                break
        self._free_arms()
        return SolvedScene(self.actors, self.spec)

    # ------------------------------------------------------------------ arm clearance
    @staticmethod
    def _closest(p, a, b):
        ab = b - a
        t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1))
        return a + ab * t

    def _free_arms(self, rounds: int = 4):
        """Arms that aren't making a deliberate contact (chest-frame shapes) are pushed out of any
        torso or head they sink into, their own included, and re-solved."""
        for _ in range(rounds):
            moved = False
            for a in self.actors:
                if a.out is None:
                    continue
                pose = a.out.pose
                rig = Rig(a.char.body)
                b = a.char.body
                for s in "lr":
                    limb = pose.hands.get(s)
                    if limb is not None and limb.frame != "chest":
                        continue
                    J = a.joints
                    pts = [J[f"{s}_wrist"], (J[f"{s}_wrist"] + J[f"{s}_hand"]) / 2,
                           (J[f"{s}_elbow"] + J[f"{s}_wrist"]) / 2]
                    push = v(0, 0, 0)
                    for o in self.actors:
                        if o.joints is None:
                            continue
                        R = o.char.body.r
                        caps = [(o.joints["pelvis"], o.joints["neck"], R["chest"] * (0.85 if o is a else 1.0)),
                                (o.joints["head"], o.joints["head"], o.char.body.head_r * 0.95)]
                        if o is not a:
                            caps += [(o.joints[f"{k}_hip"], o.joints[f"{k}_knee"], R["thigh"]) for k in "lr"]
                        for p in pts:
                            for c0, c1, rr in caps:
                                q = self._closest(p, c0, c1)
                                d = p - q
                                pen = rr + b.r["forearm"] - np.linalg.norm(d)
                                if pen > 0.005:
                                    away = unit(d, fallback=tuple(a.F @ v(1 if s == "r" else -1, 0, 0)))
                                    cand = away * (pen + 0.01)
                                    if np.linalg.norm(cand) > np.linalg.norm(push):
                                        push = cand
                    if np.linalg.norm(push) < 1e-4:
                        continue
                    if limb is None:  # the default hanging arm, made explicit so it can move
                        from .actions import _arm
                        from .actions import ActionCtx as _C
                        limb = _arm(_C(body=b, rig=rig, weapon=get_weapon("fists"), params={}), s, "hang")
                        pose.hands[s] = limb
                    rig.solve(pose)
                    local_shift = a.F.T @ push
                    limb.target = np.asarray(limb.target, float) + rig.chest_R.T @ local_shift
                    moved = True
                    local = rig.solve(pose)
                    a.local = local
                    a.joints = {k: a.to_world(p) for k, p in local.items()}
            if not moved:
                break

    # ------------------------------------------------------------------ collisions
    def _torso(self, a: Actor):
        J = a.joints
        return J["pelvis"], J["neck"], a.char.body.r["chest"]

    def _resolve_collisions(self) -> bool:
        moved = False
        for i, a in enumerate(self.actors):
            for b in self.actors[i + 1:]:
                pa, qa, ra = self._torso(a)
                pb, qb, rb = self._torso(b)
                gap = _seg_dist(pa, qa, pb, qb) - (ra + rb) - 0.04
                if gap < 0:
                    mover = b if b.auto_placed or not a.auto_placed else a
                    other = a if mover is b else b
                    d = mover.origin - other.origin
                    d[1] = 0
                    push = unit(d, (1, 0, 0)) * (-gap + 0.02)
                    mover.origin = mover.origin + push
                    mover.anchor = mover.anchor + push
                    moved = True
        if moved:
            self._face()
        return moved
