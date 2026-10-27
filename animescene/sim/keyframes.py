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
    face_w: float = 0.5  # how firmly the facing holds, per unit weight
    roll: tuple | None = None  # (bone, local axis, world dir or callable(S, touch)) a second axis, fixing the turn about the normal

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
    """Thumb and index tip closed on `spot` (an ear, a cheek); the palm turned loosely toward it."""
    t = Touch(who, f"pinch_{side}", other, spot, "pinch", normal, 0.0, (f"{_limb('_' + side)}Hand", PALM_AXIS), weight, "pinch")
    t.face_w = 0.15
    return t


def punch(who, side, other, spot, depth=0.02, normal="surface", weight=1.0):
    """Knuckles driven into `spot` along its normal (fist)."""
    return Touch(who, f"knuckles_{side}", other, spot, "punch", normal, -depth, None, weight, "fist")


def twist(who, side, other, spot, angle=60.0, normal="surface", weight=1.0):
    """Pinch `spot` and wring it: the hand turns about the surface normal by `angle` degrees from fingers-up
    (an ear twisted, a cheek wrung). The fingers point along the rolled direction."""
    t = pinch(who, side, other, spot, normal, weight)
    fingers = np.array([1.0 if side == "l" else -1.0, 0.0, 0.0])  # hand bones point out along +-x in the T-pose

    def rolled(S, tt):  # tt: the touch as placed (renamed, turned) in the scene
        n = _normal(S, tt)
        u = UP - n * (UP @ n)
        u /= np.linalg.norm(u)
        a = np.deg2rad(angle)
        return u * np.cos(a) + np.cross(n, u) * np.sin(a)
    t.verb, t.roll = "twist", (f"{_limb('_' + side)}Hand", fingers, rolled)
    return t


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
    """`part` (an instep, a heel, the crook of a knee) hooked round `spot` from its `normal` side, e.g. a foot
    behind a glute."""
    return Touch(who, part, other, spot, "hook", normal, 0.0, None, weight)


def tuck(who, part, other, spot, gap=0.04, weight=1.0):
    """`spot` held in against `part` (e.g. an ankle tucked under the armpit): `part` comes to it."""
    return Touch(who, part, other, spot, "tuck", lambda S: UP, gap, None, weight)


def squeeze(who, part, other, spot, normal="surface", depth=0.01, weight=0.25):
    """`part` pressed lightly into `spot` as a secondary pressure (knees pinching a leg, a chest on a back);
    weaker than a main touch, so it gives way first."""
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


def _pos(S, actor, pos):
    """A root position hint; a None height means the actor's current hips height."""
    return np.array([pos[0], S.body(actor, "hips").xpos[1] if pos[1] is None else pos[1], pos[2]], float)


def place_root(S, actor, pos=None, yaw=0.0, pitch=0.0, roll=0.0):
    adr = _root_adr(S, actor)
    if pos is not None:
        S.d.qpos[adr:adr + 3] = [pos[0], S.d.qpos[adr + 1] if pos[1] is None else pos[1], pos[2]]
    S.d.qpos[adr + 3:adr + 7] = _quat(yaw, pitch, roll)
    mujoco.mj_kinematics(S.m, S.d)


# ----------------------------------------------------------------------------- solving
def _tasks_for(S, key: Key, actor: str, turn=1.0):
    tasks, allow, orient = [], [], []
    for t in key.touches:
        if t.who == actor:
            n = _normal(S, t)
            tasks.append((t.part, S.site(t.other, t.spot) + n * t.gap, t.weight))
            if t.face is not None:
                tasks.append((f"@axis:{t.face[0]}", (t.face[1], -n), t.face_w * t.weight * turn))
            if t.roll is not None:
                w = t.roll[2](S, t) if callable(t.roll[2]) else np.asarray(t.roll[2], float)
                w = w - n * (w @ n)  # only the turn about the normal
                tasks.append((f"@axis:{t.roll[0]}", (np.asarray(t.roll[1], float), w / np.linalg.norm(w)), 0.25 * t.weight * turn))
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
    if r.get("pos") is not None and actor not in key.balance:  # where the hips should be (a hint); height None:
        p = _pos(S, actor, r["pos"])  # whatever the legs and supports make it. (Standing bodies: the hint places
        # the feet and the weight instead, so they can bend.)
        tasks.append(("pelvis" if r["pos"][1] is not None else "@xz:pelvis", p, r.get("pos_w", 0.5)))
    if actor in key.balance:
        if r.get("pos") is not None:  # standing: feet under the hips, a stance apart, the weight between them
            base = _pos(S, actor, r["pos"]) * np.array([1.0, 0, 1.0])
            left = _quat(r.get("yaw", 0.0))
            m = np.zeros(9)
            mujoco.mju_quat2Mat(m, left)
            lat, fwd = m.reshape(3, 3)[:, 0], m.reshape(3, 3)[:, 2]
            off = r.get("stance", 0.10)
            for side, sx in (("l", 1.0), ("r", -1.0)):
                if f"sole_{side}" in key.support.get(actor, ()):
                    tasks.append((f"sole_{side}", base + lat * sx * off + fwd * r.get("step", {}).get(side, 0.0), 0.4))
            tasks.append(("@com", base, 1.0))
        else:
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


def score(rep: dict) -> float:
    """One number for how well a keyframe is met: main touches, squeezes (softer), overlaps (harder)."""
    sc = 0.0
    for k, v in rep.items():
        sc += 4.0 * v if k == "overlap" else (0.3 * v if k.startswith("squeeze") else v)
    return sc


def _seeds(key: Key, a: str):
    """Fixed starting variations for an actor's solve (the same list every time: results stay deterministic)."""
    r = key.root.get(a, {})
    out = [dict()]
    if r.get("pos") is not None:
        for dz in (-0.08, 0.08):
            out.append(dict(pos=[r["pos"][0], r["pos"][1], r["pos"][2] + dz]))
    out.append(dict(pitch=r.get("pitch", 0.0) - 12))
    out.append(dict(pitch=r.get("pitch", 0.0) + 12))
    return out


def solve_key(S: Scene, key: Key, rounds=4, iters=150, log=None, verbose=False, seeds=True):
    """Both bodies into the keyframe: each actor in key.order is solved against the others where they are
    now; an acting body is tried from a few fixed starting variations and the best-scoring result is kept."""
    postures = {a: posed(S, a, base=np.zeros(len(S.actors[a].jids)), **key.posture.get(a, {})) for a in S.actors}
    for a, r in key.root.items():
        pos = r.get("pos")
        place_root(S, a, pos, r.get("yaw", 0.0), r.get("pitch", 0.0), r.get("roll", 0.0))
        S.d.qpos[S.actors[a].qadr] = postures[a]
    mujoco.mj_kinematics(S.m, S.d)
    for a in key.order:
        if a in key.fixed:
            continue
        acting = any(t.who == a for t in key.touches)
        start = S.d.qpos.copy()
        best = None
        for sd in (_seeds(key, a) if seeds and acting else [dict()]):
            S.d.qpos[:] = start
            r = {**key.root.get(a, {}), **sd}
            if sd:
                place_root(S, a, r.get("pos"), r.get("yaw", 0.0), r.get("pitch", 0.0),
                           r.get("roll", 0.0))
            _solve_actor(S, key, a, postures[a], rounds, iters, log if verbose else None)
            sc = score({k: v for k, v in report(S, key).items()})
            if best is None or sc < best[0] - 1e-9:
                best = (sc, S.d.qpos.copy())
        S.d.qpos[:] = best[1]
        mujoco.mj_kinematics(S.m, S.d)
        # out of reach: she steps in (or back) by the horizontal shortfall of her touches and solves again
        r = key.root.get(a, {})
        for _ in range(3 if acting and r.get("pos") is not None and r.get("approach", True) else 0):
            miss = [(S.site(t.other, t.spot) + _normal(S, t) * t.gap) - S.site(t.who, t.part) for t in key.touches
                    if t.who == a and t.verb not in ("squeeze",) and t.other != a]
            if not miss:
                break
            m = np.mean(miss, axis=0) * np.array([1.0, 0, 1.0])
            if np.linalg.norm(m) < 0.03:
                break
            m *= min(1.0, 0.35 / np.linalg.norm(m))
            r["pos"] = [r["pos"][0] + m[0], r["pos"][1], r["pos"][2] + m[2]]
            S.d.qpos[:] = best[1]
            place_root(S, a, [S.body(a, "hips").xpos[0] + m[0], None, S.body(a, "hips").xpos[2] + m[2]],
                       r.get("yaw", 0.0), r.get("pitch", 0.0), r.get("roll", 0.0))
            _solve_actor(S, key, a, postures[a], rounds, iters, log if verbose else None)
            sc = score(report(S, key))
            if sc < best[0]:
                best = (sc, S.d.qpos.copy())
            S.d.qpos[:] = best[1]
            mujoco.mj_kinematics(S.m, S.d)
    mujoco.mj_forward(S.m, S.d)
    rep = report(S, key)
    if log:
        log(f"[key {key.name or key.t}] " + ", ".join(f"{k} {v:.3f}" for k, v in rep.items()))
    return S.d.qpos.copy(), rep


def _solve_actor(S, key, a, posture, rounds, iters, log=None):
    """Two stages: without collisions (the limbs find which side of which body they belong: an arm can only
    wrap round a leg if it may pass it while searching), then with them until it stops improving."""
    best, worse, kept, turn = np.inf, 0, (np.inf, None), 1.0
    for stage, avoid in enumerate([False, False] + [True] * max(2, rounds + 4)):
        if avoid and worse >= 2:
            break
        tasks, allow = _tasks_for(S, key, a, turn)
        r = key.root.get(a, {})
        orient = [("hips", _quat(r.get("yaw", 0.0), r.get("pitch", 0.0), r.get("roll", 0.0)), r.get("hold", 0.1))] if r else None
        q, root, res = S.ik(a, tasks, posture=posture, w_post=key.stiff.get(a, 0.05), iters=iters, root=True,
                            root_rot=True, root_cost=1.0, avoid=avoid, allow=allow, orient=orient, upright=0.0,
                            q0=S.d.qpos.copy(), lock=key.lock.get(a, ()), soft=key.soft)
        adr = _root_adr(S, a)
        S.d.qpos[adr:adr + 7] = root
        S.d.qpos[S.actors[a].qadr] = q
        mujoco.mj_kinematics(S.m, S.d)
        if avoid:
            rr = report(S, key)
            err = max([v for k, v in rr.items() if not k.startswith(("squeeze", "overlap"))] or [0.0])
            worse = worse + 1 if err > best - 0.002 else 0
            best = min(best, err)
            sc = score(rr)
            if sc < kept[0]:  # stages can trade one fault for another: keep the best one seen
                kept = (sc, S.d.qpos.copy())
            if err > 0.03:  # the place matters more than the hand's exact turn: give way on the facing
                turn *= 0.5
        if log:
            rr = report(S, key)
            log(f"   {a} stage {stage} avoid={avoid}: " + ", ".join(f"{k.split(':')[0]} {v:.3f}" for k, v in rr.items()))
    if kept[1] is not None:
        S.d.qpos[:] = kept[1]
        mujoco.mj_kinematics(S.m, S.d)


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


def place_keys(keys: list, names: dict, origin=(0.0, 0.0), yaw=0.0) -> list:
    """A move written for actors "A"/"V" with "A" at the origin facing +z, re-cast onto named characters,
    moved to `origin` (x, z) and turned by `yaw` degrees. Returns new Key objects."""
    import copy
    c, s_ = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))

    def rot(v):
        v = np.asarray(v, float)
        return np.array([c * v[0] + s_ * v[2], v[1], -s_ * v[0] + c * v[2]])

    def rn(n):
        return names.get(n, n)

    out = []
    for k in keys:
        k2 = copy.copy(k)
        k2.touches = []
        for t in k.touches:
            t2 = copy.copy(t)
            t2.who, t2.other = rn(t.who), rn(t.other)
            if not callable(t.normal) and not isinstance(t.normal, str):
                t2.normal = rot(t.normal)
            if t.roll is not None and not callable(t.roll[2]):
                t2.roll = (t.roll[0], t.roll[1], rot(t.roll[2]))
            k2.touches.append(t2)
        for f in ("support", "posture", "lock", "stiff", "shapes"):
            setattr(k2, f, {rn(a): copy.deepcopy(v) for a, v in getattr(k, f).items()})
        k2.balance, k2.order, k2.fixed = tuple(rn(a) for a in k.balance), tuple(rn(a) for a in k.order), tuple(rn(a) for a in k.fixed)
        k2.root = {}
        for a, r in k.root.items():
            r2 = dict(r)
            if r.get("pos") is not None:
                p = r["pos"]
                xz = rot([p[0], 0.0, p[2]])
                r2["pos"] = [xz[0] + origin[0], p[1], xz[2] + origin[1]]
            r2["yaw"] = r.get("yaw", 0.0) + yaw
            k2.root[rn(a)] = r2
        out.append(k2)
    return out
