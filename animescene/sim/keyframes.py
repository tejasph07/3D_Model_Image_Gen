"""Keyframed contact choreography.

A position is written as a list of *touches* - which exact part of whom is on which exact spot of whom,
and how (a palm closing round it, a sole flat on it, a forearm blade under it, ...) - plus what each body
rests on (floor supports) and a posture hint. `solve_key` turns that description into both skeletons'
joint angles: deterministic whole-body inverse kinematics on the simulation's mesh-fitted bodies, with the
real joint limits, no body passing into another or into the floor, and the weight over the supports when
standing. The same description always gives the same pose.

A move is a sequence of keyframes; `inbetween` fills the frames between them (eased joint interpolation,
then each frame is repaired: contacts that hold in both keyframes stay held, supports stay on the floor,
nothing passes through anything).

Verbs (`grab`, `press`, `punch`, `clamp`, `rest`, `hook`, `squeeze`, `pinch`) each fix the effector
point, the direction it faces and the hand shape, so the hands and feet are right by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .control import Scene, posed

UP = np.array([0.0, 1.0, 0.0])

# local axes of the effector surfaces in each bone's rest frame (body rotations are rotations from rest)
PALM_AXIS = np.array([0.0, -1.0, 0.0])  # the palm faces -y in the T-pose
SOLE_AXIS = np.array([0.0, -1.0, 0.0])
BLADE_AXIS = np.array([0.0, -0.7, 0.7]) / np.linalg.norm([0.0, -0.7, 0.7])  # radius side of the forearm by the wrist


@dataclass
class Touch:
    """`who`'s `part` (a site) on `other`'s `spot` (a site), the effector surface facing into the spot.

    normal: which way the spot's surface faces - "surface" (away from the bone it sits on), a world
    vector, or a callable(S) -> vector. gap: distance off the surface (negative: pressed in). face:
    (bone, local axis) of the effector surface turned against the normal, or None."""
    who: str
    part: str
    other: str
    spot: str
    verb: str = "touch"
    normal: object = "surface"
    gap: float = 0.0
    face: tuple | None = None
    weight: float = 1.0
    shape: str | None = None  # hand shape for the renderer ("grip", "flat", "fist", "pinch", ...)
    mutual: float = 0.0  # how much the touched body moves to meet it (0: it is posed by its own description)

    def side(self):
        return self.part[-1] if self.part[-2:] in ("_l", "_r") else None


def _limb(part):
    side = part[-1]
    return "left" if side == "l" else "right"


# ----------------------------------------------------------------------------- verbs
def grab(who, side, other, spot, normal="surface", weight=1.0):
    """The hand closes round `spot`: palm on it, fingers wrapped (renderer grip shape)."""
    return Touch(who, f"palm_{side}", other, spot, "grab", normal, 0.0, (f"{_limb('_' + side)}Hand", PALM_AXIS), weight, "grip")


def press(who, side, other, spot, depth=0.02, normal="surface", weight=1.0):
    """Open palm pushed into a soft surface by `depth` (the renderer dents the mesh by that much)."""
    return Touch(who, f"palm_{side}", other, spot, "press", normal, -depth, (f"{_limb('_' + side)}Hand", PALM_AXIS), weight, "flat")


def pinch(who, side, other, spot, normal="surface", weight=1.0):
    return Touch(who, f"palm_{side}", other, spot, "pinch", normal, 0.0, (f"{_limb('_' + side)}Hand", PALM_AXIS), weight, "pinch")


def punch(who, side, other, spot, depth=0.02, normal="surface", weight=1.0):
    """Knuckles driven into `spot` along its normal (fist)."""
    return Touch(who, f"knuckles_{side}", other, spot, "punch", normal, -depth, None, weight, "fist")


def clamp(who, side, other, spot, normal, weight=1.0):
    """The blade of the forearm by the wrist hard against `spot` from the `normal` side (e.g. under an
    Achilles tendon: normal = down)."""
    return Touch(who, f"wristblade_{side}", other, spot, "clamp", normal, 0.0, (f"{_limb('_' + side)}LowerArm", BLADE_AXIS), weight,
                 "grip")


def rest(who, part, other, spot, normal="surface", weight=1.0, face=True):
    """A sole (or other flat part) set on `spot`."""
    f = (f"{_limb(part)}Foot", SOLE_AXIS) if face and part.startswith(("sole_", "heel_")) else None
    return Touch(who, part, other, spot, "rest", normal, 0.0, f, weight)


def hook(who, part, other, spot, normal="surface", weight=1.0):
    return Touch(who, part, other, spot, "hook", normal, 0.0, None, weight)


def tuck(who, part, other, spot, gap=0.04, weight=1.0):
    """`spot` held in against `part` (e.g. an ankle tucked under the armpit): `part` comes to it."""
    return Touch(who, part, other, spot, "tuck", lambda S: UP, gap, None, weight)


def squeeze(who, part, other, spot, normal="surface", depth=0.01, weight=0.25):
    return Touch(who, part, other, spot, "squeeze", normal, -depth, None, weight)


@dataclass
class Key:
    """One keyframe: touches, floor supports per actor (sites that rest on the floor), posture hints per
    actor (joint angles in degrees as for `posed`), root hints per actor (position xyz, yaw, pitch, roll in
    degrees: pitch -90 = lying on the back), and which actors must keep their weight over their supports."""
    t: float
    touches: list = field(default_factory=list)
    support: dict = field(default_factory=dict)
    posture: dict = field(default_factory=dict)
    root: dict = field(default_factory=dict)
    balance: tuple = ()
    order: tuple = ("V", "A")
    fixed: tuple = ()  # actors that keep the pose they come in with
    shapes: dict = field(default_factory=dict)  # extra hand shapes {actor: {"l": .., "r": ..}}
    stiff: dict = field(default_factory=dict)  # posture weight per actor (default 0.05): how firmly the hint holds
    lock: dict = field(default_factory=dict)  # joint name fragments per actor kept exactly at the posture hint
    soft: float = 0.01  # how far bodies in contact may press into each other (flesh), m
    name: str = ""


# ----------------------------------------------------------------------------- geometry
def _normal(S, t: Touch):
    """The outward normal of the touched surface at `spot`."""
    if callable(t.normal):
        n = np.asarray(t.normal(S), float)
    elif isinstance(t.normal, str):
        body, _ = S.specs[t.other].sites[t.spot]
        g = S.m.geom(f"{t.other}:{body}").id
        c = S.d.geom_xpos[g].copy()
        if S.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:  # closest point on the capsule's axis
            ax = S.d.geom_xmat[g].reshape(3, 3)[:, 2]
            h = S.m.geom_size[g][1]
            p = S.site(t.other, t.spot)
            c = c + ax * float(np.clip((p - c) @ ax, -h, h))
        n = S.site(t.other, t.spot) - c
    else:
        n = np.asarray(t.normal, float)
    return n / max(np.linalg.norm(n), 1e-9)


def _thickness(S, t: Touch):
    """How far the effector site sits inside its own surface (0 for surface sites)."""
    return 0.0


def _quat(yaw=0.0, pitch=0.0, roll=0.0):
    """Root orientation: yaw about up, then pitch about the body's left axis (-90: on the back, face up),
    then roll about the body's forward axis."""
    q = np.zeros(4)
    qs = []
    for ax, a in ((np.array([0, 1.0, 0]), yaw), (np.array([1.0, 0, 0]), pitch), (np.array([0, 0, 1.0]), roll)):
        qq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qq, ax, np.deg2rad(a))
        qs.append(qq)
    mujoco.mju_mulQuat(q, qs[0], qs[1])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, qs[2])
    return out


def _root_adr(S, actor):
    return S.m.jnt_qposadr[S.m.joint(actor + ":root").id]


def place_root(S, actor, pos=None, yaw=0.0, pitch=0.0, roll=0.0):
    adr = _root_adr(S, actor)
    if pos is not None:
        S.d.qpos[adr:adr + 3] = pos
    S.d.qpos[adr + 3:adr + 7] = _quat(yaw, pitch, roll)
    mujoco.mj_kinematics(S.m, S.d)


# ----------------------------------------------------------------------------- solving
def _tasks_for(S, key: Key, actor: str):
    tasks, allow, orient = [], [], []
    for t in key.touches:
        if t.who == actor:
            n = _normal(S, t)
            tasks.append((t.part, S.site(t.other, t.spot) + n * t.gap, t.weight))
            if t.face is not None:
                tasks.append((f"@axis:{t.face[0]}", (t.face[1], -n), 0.5 * t.weight))
        elif t.other == actor and t.mutual > 0:  # the touched body comes to meet it
            n = _normal(S, t)
            tasks.append((t.spot, S.site(t.who, t.part) - n * t.gap, t.mutual * t.weight))
        else:
            continue
        b1 = S.specs[t.who].sites[t.part][0]
        b2 = S.specs[t.other].sites[t.spot][0]
        allow.append((f"{t.who}:{b1}", f"{t.other}:{b2}"))
    for site in key.support.get(actor, ()):
        tasks.append((f"@y:{site}", np.array([0.0, _floor_height(S, actor, site), 0.0]), 1.5))
    r = key.root.get(actor, {})
    if r.get("pos") is not None:  # where the hips should be (a hint: the touches can move them a little)
        tasks.append(("pelvis", np.asarray(r["pos"], float), r.get("pos_w", 0.5)))
    if actor in key.balance:
        sup = [S.site(actor, s) for s in key.support.get(actor, ())]
        if sup:
            tasks.append(("@com", np.mean(sup, axis=0), 1.0))
    return tasks, tuple(allow)


def _floor_height(S, actor, site):
    """Height of a support site when its body lies on the floor: the site's distance from the body's
    lowest surface (soles and heels sit on the surface already; body-centre sites sit a radius up)."""
    body, _ = S.specs[actor].sites[site]
    if site.startswith(("sole_", "heel_", "toes_")):
        return 0.002
    g = S.m.geom(f"{actor}:{body}").id
    size = S.m.geom_size[g]
    if S.m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        return float(size[0])
    return float(min(size[:3]))


def solve_key(S: Scene, key: Key, rounds=4, iters=150, log=None, verbose=False):
    """Both bodies into the keyframe: alternate whole-body solves (each sees the other where it is now)."""
    postures = {a: posed(S, a, base=np.zeros(len(S.actors[a].jids)), **key.posture.get(a, {})) for a in S.actors}
    for a, r in key.root.items():
        pos = r.get("pos")
        place_root(S, a, None if pos is None else np.asarray(pos, float), r.get("yaw", 0.0), r.get("pitch", 0.0), r.get("roll", 0.0))
        S.d.qpos[S.actors[a].qadr] = postures[a]
    mujoco.mj_kinematics(S.m, S.d)
    # each body in turn (the receiver first: it is posed by its own description), each in two stages: without
    # collisions (the limbs find which side of which body they belong: an arm can only wrap round a leg if it
    # may pass it while searching), then with them (the result overlaps nothing)
    for a in key.order:
        if a in key.fixed:
            continue
        tasks0 = None
        for stage, avoid in enumerate((False, False, True, True)[:max(2, rounds)]):
            tasks, allow = _tasks_for(S, key, a)
            r = key.root.get(a, {})
            orient = [("hips", _quat(r.get("yaw", 0.0), r.get("pitch", 0.0), r.get("roll", 0.0)), r.get("hold", 0.1))] if r else None
            q, root, res = S.ik(a, tasks, posture=postures[a], w_post=key.stiff.get(a, 0.05), iters=iters, root=True,
                                root_rot=True, root_cost=1.0, avoid=avoid, allow=allow, orient=orient, upright=0.0,
                                q0=S.d.qpos.copy(), lock=key.lock.get(a, ()), soft=key.soft)
            adr = _root_adr(S, a)
            S.d.qpos[adr:adr + 7] = root
            S.d.qpos[S.actors[a].qadr] = q
            mujoco.mj_kinematics(S.m, S.d)
            if log and verbose:
                rr = report(S, key)
                log(f"   {a} stage {stage} avoid={avoid}: " + ", ".join(f"{k.split(':')[0]}.{k.split('->')[0].split('.')[-1]} {v:.3f}" for k, v in rr.items()))
    mujoco.mj_forward(S.m, S.d)
    rep = report(S, key)
    if log:
        log(f"[key {key.name or key.t}] " + ", ".join(f"{k} {v:.3f}" for k, v in rep.items()))
    return S.d.qpos.copy(), rep


def report(S: Scene, key: Key) -> dict:
    """How well each touch is met (m between the effector and its place) and the worst overlap between
    bodies (m; positive = interpenetration beyond the intended presses)."""
    out = {}
    for t in key.touches:
        n = _normal(S, t)
        out[f"{t.verb}:{t.who}.{t.part}->{t.other}.{t.spot}"] = float(np.linalg.norm(S.site(t.who, t.part) - (S.site(t.other, t.spot) + n * t.gap)))
    out["overlap"] = max_overlap(S, key)
    return out


def max_overlap(S: Scene, key: Key | None = None) -> float:
    intended = set()
    for t in (key.touches if key else []):
        intended.add(frozenset((f"{t.who}:{S.specs[t.who].sites[t.part][0]}", f"{t.other}:{S.specs[t.other].sites[t.spot][0]}")))
    mujoco.mj_collision(S.m, S.d)
    worst = 0.0
    for i in range(S.d.ncon):
        c = S.d.contact[i]
        n1 = S.m.body(S.m.geom_bodyid[c.geom1]).name
        n2 = S.m.body(S.m.geom_bodyid[c.geom2]).name
        if frozenset((n1, n2)) in intended:
            continue
        if "world" in (n1, n2) or n1.split(":")[0] == n2.split(":")[0]:
            worst = max(worst, -c.dist)
        else:
            worst = max(worst, -c.dist - (key.soft if key else 0.0))  # beyond the allowed flesh compression
    return worst


# ----------------------------------------------------------------------------- in-betweens
def _smooth(u):
    return u * u * u * (10 - 15 * u + 6 * u * u)


def inbetween(S: Scene, keys: list, poses: list, fps=24, repair=True):
    """Frames between solved keyframes: eased interpolation of every joint and root, then a repair solve
    per frame (touches present in both neighbouring keys stay met, supports stay down, no overlaps)."""
    frames = []
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        n = max(2, int(round((k1.t - k0.t) * fps)))
        common = [t for t in k1.touches if any((t.who, t.part, t.other, t.spot) == (u.who, u.part, u.other, u.spot) for u in k0.touches)]
        sup = {a: [s for s in k1.support.get(a, ()) if s in k0.support.get(a, ())] for a in S.actors}
        mid = Key(t=0, touches=common, support=sup, order=k1.order)
        for j in range(n):
            u = _smooth(j / n)
            q = _interp(S, poses[i], poses[i + 1], u)
            S.d.qpos[:] = q
            mujoco.mj_kinematics(S.m, S.d)
            if repair and j > 0:
                for a in mid.order:
                    tasks, allow = _tasks_for(S, mid, a)
                    qa, root, _ = S.ik(a, tasks, posture=S.q(a), w_post=0.3, iters=25, root=True, root_rot=True,
                                       avoid=True, allow=allow, upright=0.0, q0=S.d.qpos.copy())
                    adr = _root_adr(S, a)
                    S.d.qpos[adr:adr + 7] = root
                    S.d.qpos[S.actors[a].qadr] = qa
                    mujoco.mj_kinematics(S.m, S.d)
            frames.append((k0.t + (k1.t - k0.t) * j / n, S.d.qpos.copy(), k0 if u < 0.5 else k1))
    frames.append((keys[-1].t, poses[-1].copy(), keys[-1]))
    return frames


def _interp(S, q0, q1, u):
    q = q0 + (q1 - q0) * u
    for a in S.actors:  # roots: slerp the orientation
        adr = _root_adr(S, a)
        r0, r1 = q0[adr + 3:adr + 7], q1[adr + 3:adr + 7]
        if r0 @ r1 < 0:
            r1 = -r1
        th = np.arccos(np.clip(r0 @ r1, -1, 1))
        r = r0 if th < 1e-6 else (np.sin((1 - u) * th) * r0 + np.sin(u * th) * r1) / np.sin(th)
        q[adr + 3:adr + 7] = r / np.linalg.norm(r)
    return q


def hand_shapes(key: Key) -> dict:
    out = {}
    for t in key.touches:
        if t.shape and t.side():
            out.setdefault(t.who, {"l": "relaxed", "r": "relaxed"})[t.side()] = t.shape
    for a, sh in key.shapes.items():
        out.setdefault(a, {"l": "relaxed", "r": "relaxed"}).update(sh)
    return out
