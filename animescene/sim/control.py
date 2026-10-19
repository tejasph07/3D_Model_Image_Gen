"""Driving simulated characters: inverse kinematics for target poses, joint servos ("muscles") that
track them over time, grips that switch on when a hand actually touches something, and a balance assist
on the hips that can be released (a fall is then just gravity and contacts).

Everything here acts through the physics: servos apply torques within their force limits, grips are
equality constraints, contacts keep bodies apart. Nothing is teleported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .model import ActorSpec, compile_model
from .shapes import fit


# ----------------------------------------------------------------------------- sites
def standard_sites(model: str) -> dict:
    """Named contact points on a character, from its fitted shapes (bone, local position):
    palms, the blade of the forearm by the wrist, Achilles tendons, shins, soles, heels, hip bones,
    glutes, armpits, belly, sternum, back of the head..."""
    S = fit(model)
    B = {k: np.asarray(v) for k, v in S["_bones"].items()}
    out = {}
    for side, sx in (("l", 1.0), ("r", -1.0)):
        L = "left" if side == "l" else "right"
        hand = S[f"{L}Hand"]
        hp, hs = np.asarray(hand["pos"]), np.asarray(hand["size"])
        out[f"palm_{side}"] = (f"{L}Hand", (hp + np.array([0, -hs[1], 0])).tolist())  # palm faces -y at rest
        out[f"knuckles_{side}"] = (f"{L}Hand", (hp + np.array([sx * hs[0] * 0.3, hs[1], 0])).tolist())
        fa = S[f"{L}LowerArm"]
        to = np.asarray(fa["to"])
        # blade of the forearm near the wrist (the radius side faces forward in the T-pose)
        out[f"wristblade_{side}"] = (f"{L}LowerArm", (to * 0.92 + np.array([0, -fa["r"] * 0.7, fa["r"] * 0.7])).tolist())
        out[f"forearm_{side}"] = (f"{L}LowerArm", (to * 0.55).tolist())
        sh = S[f"{L}LowerLeg"]
        a, b = np.asarray(sh["from"]), np.asarray(sh["to"])
        out[f"achilles_{side}"] = (f"{L}LowerLeg", (a + (b - a) * 0.95 + np.array([0, 0, -sh["r"]])).tolist())
        out[f"ankle_{side}"] = (f"{L}LowerLeg", (a + (b - a) * 1.0).tolist())
        out[f"shin_{side}"] = (f"{L}LowerLeg", (a + (b - a) * 0.5 + np.array([0, 0, sh["r"]])).tolist())
        out[f"calf_{side}"] = (f"{L}LowerLeg", (a + (b - a) * 0.35 + np.array([0, 0, -sh["r"]])).tolist())
        th = S[f"{L}UpperLeg"]
        a, b = np.asarray(th["from"]), np.asarray(th["to"])
        out[f"knee_{side}"] = (f"{L}UpperLeg", (b + (b - a) / np.linalg.norm(b - a) * th["r"] * 0.5).tolist())
        out[f"innerthigh_{side}"] = (f"{L}UpperLeg", (a + (b - a) * 0.5 + np.array([-sx * th["r"], 0, 0])).tolist())
        out[f"outerthigh_{side}"] = (f"{L}UpperLeg", (a + (b - a) * 0.4 + np.array([sx * th["r"], 0, 0])).tolist())
        ft = S[f"{L}Foot"]
        fp, fs = np.asarray(ft["pos"]), np.asarray(ft["size"])
        out[f"sole_{side}"] = (f"{L}Foot", (fp + np.array([0, -fs[1], 0])).tolist())
        out[f"heel_{side}"] = (f"{L}Foot", (fp + np.array([0, -fs[1] * 0.5, -fs[2]])).tolist())
        out[f"toes_{side}"] = (f"{L}Foot", (fp + np.array([0, -fs[1] * 0.5, fs[2]])).tolist())
        hips = S["hips"]
        p, s = np.asarray(hips["pos"]), np.asarray(hips["size"])
        out[f"hipbone_{side}"] = ("hips", (p + np.array([sx * s[0] * 0.75, s[1] * 0.2, s[2] * 0.55])).tolist())
        out[f"glute_{side}"] = ("hips", (p + np.array([sx * s[0] * 0.45, -s[1] * 0.35, -s[2] * 0.9])).tolist())
        ch = S["upperChest"] if "upperChest" in S else S["chest"]
        cb = "upperChest" if "upperChest" in S else "chest"
        cp, cs = np.asarray(ch["pos"]), np.asarray(ch["size"])
        out[f"armpit_{side}"] = (cb, (cp + np.array([sx * cs[0] * 0.95, cs[1] * 0.25, -cs[2] * 0.05])).tolist())
        out[f"shoulder_{side}"] = (f"{L}UpperArm", [0.0, 0.0, 0.0])
        out[f"elbow_{side}"] = (f"{L}LowerArm", [0.0, 0.0, 0.0])
        out[f"hand_{side}"] = (f"{L}Hand", [0.0, 0.0, 0.0])
        out[f"hip_{side}"] = (f"{L}UpperLeg", [0.0, 0.0, 0.0])
        out[f"foot_{side}"] = (f"{L}Foot", [0.0, 0.0, 0.0])
    ch = S["upperChest"] if "upperChest" in S else S["chest"]
    cb = "upperChest" if "upperChest" in S else "chest"
    out["sternum"] = (cb, (np.asarray(ch["pos"]) + np.array([0, 0, ch["size"][2]])).tolist())
    sp = S["spine"]
    out["belly"] = ("spine", (np.asarray(sp["pos"]) + np.array([0, 0, sp["size"][2]])).tolist())
    out["lowback"] = ("spine", (np.asarray(sp["pos"]) + np.array([0, 0, -sp["size"][2]])).tolist())
    hd = S["head"]
    out["face"] = ("head", (np.asarray(hd["pos"]) + np.array([0, 0, hd["size"][2]])).tolist())
    out["crown"] = ("head", (np.asarray(hd["pos"]) + np.array([0, hd["size"][1], 0])).tolist())
    out["eyes"] = ("head", (np.asarray(hd["pos"]) + np.array([0, 0.01, hd["size"][2] * 0.8])).tolist())
    out["pelvis"] = ("hips", [0.0, 0.0, 0.0])
    out["chestc"] = (cb, np.asarray(ch["pos"]).tolist())
    return out


# ----------------------------------------------------------------------------- scene
GRIPPERS = ("leftHand", "rightHand", "leftLowerArm", "rightLowerArm", "leftUpperArm", "rightUpperArm",
            "leftLowerLeg", "rightLowerLeg", "leftUpperLeg", "rightUpperLeg", "leftFoot", "rightFoot")


@dataclass
class Actor:
    spec: ActorSpec
    jids: list = field(default_factory=list)  # hinge joint ids
    qadr: np.ndarray = None
    dadr: np.ndarray = None  # dof addresses (free root first)
    aids: np.ndarray = None  # actuators, same order as jids
    lo: np.ndarray = None
    hi: np.ndarray = None
    kp0: np.ndarray = None
    root_eq: int = -1
    mocap: int = -1


class Scene:
    """A compiled multi-character simulation with grips, balance assists and recording."""

    def __init__(self, specs: list[ActorSpec], timestep=0.002):
        for s in specs:
            sites = standard_sites(s.model)
            s.sites = {**sites, **s.sites}
        world = []
        eqs = []
        for s in specs:  # balance assist: a soft weld from the hips to a target the choreography moves
            world.append(f'<body name="{s.name}:target" mocap="true"><geom type="sphere" size="0.01" contype="0" '
                         f'conaffinity="0" rgba="0 0 0 0"/></body>')
            eqs.append(f'<weld name="{s.name}:assist" body1="{s.name}:hips" body2="{s.name}:target" active="false" '
                       f'solref="0.08 1"/>')
        # potential grips: any limb of a character onto any body of another (or its own other arm)
        self.grip_names = []
        for s in specs:
            for g in GRIPPERS:
                for o in specs:
                    for tb in fit(o.model)["_bones"]:
                        if tb in ("leftToes", "rightToes", "leftEye", "rightEye", "leftMiddleProximal", "rightMiddleProximal"):
                            continue
                        if o is s and not (g.endswith("Hand") and tb in ("leftLowerArm", "rightLowerArm", "leftHand", "rightHand")
                                           and tb[:4] != g[:4]):
                            continue
                        n = f"grip|{s.name}:{g}|{o.name}:{tb}"
                        eqs.append(f'<weld name="{n}" body1="{s.name}:{g}" body2="{o.name}:{tb}" active="false" solref="0.02 1"/>')
                        self.grip_names.append(n)
        self.m, self.xml = compile_model(specs, "\n".join(world), "\n".join(eqs), timestep)
        self.d = mujoco.MjData(self.m)
        self.specs = {s.name: s for s in specs}
        self.actors = {}
        m = self.m
        for s in specs:
            a = Actor(s)
            P = s.name + ":"
            for j in range(m.njnt):
                if m.joint(j).name.startswith(P) and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    a.jids.append(j)
            a.qadr = np.array([m.jnt_qposadr[j] for j in a.jids])
            root = m.joint(P + "root").id
            a.dadr = np.concatenate([np.arange(m.jnt_dofadr[root], m.jnt_dofadr[root] + 6),
                                     np.array([m.jnt_dofadr[j] for j in a.jids])])
            a.aids = np.array([m.actuator(m.joint(j).name).id for j in a.jids])
            a.lo, a.hi = m.jnt_range[a.jids, 0].copy(), m.jnt_range[a.jids, 1].copy()
            a.kp0 = m.actuator_gainprm[a.aids, 0].copy()
            a.root_eq = m.equality(P + "assist").id
            a.mocap = m.body_mocapid[m.body(P + "target").id]
            self.actors[s.name] = a
        mujoco.mj_forward(m, self.d)
        # planning copy: sees geoms coming within 3 cm (contact margin) without any effect on the physics
        self.m_ik = mujoco.MjModel.from_xml_string(self.xml)
        self.m_ik.geom_margin[:] = 0.03
        self.hold_current()
        self.frames = []
        self.t = 0.0
        self.log = []

    # ---- helpers
    def site(self, actor: str, name: str) -> np.ndarray:
        return self.d.site(f"{actor}:{name}").xpos.copy()

    def body(self, actor: str, name: str):
        return self.d.body(f"{actor}:{name}")

    def q(self, actor: str) -> np.ndarray:
        return self.d.qpos[self.actors[actor].qadr].copy()

    def hold_current(self, actors=None):
        for n, a in self.actors.items():
            if actors is None or n in actors:
                self.d.ctrl[a.aids] = self.d.qpos[a.qadr]

    def set_pose(self, actor: str, q: np.ndarray):
        a = self.actors[actor]
        self.d.ctrl[a.aids] = np.clip(q, a.lo, a.hi)

    def strength(self, actor: str, k: float, groups: tuple | None = None):
        """Scale the actor's muscles (1 = normal; a dazed or hurt body goes slack)."""
        a = self.actors[actor]
        for i, (j, aid) in enumerate(zip(a.jids, a.aids)):
            name = self.m.joint(j).name
            if groups is None or any(g in name for g in groups):
                self.m.actuator_gainprm[aid, 0] = a.kp0[i] * k
                self.m.actuator_biasprm[aid, 1] = -a.kp0[i] * k

    def assist(self, actor: str, on: bool, pos=None, quat=None, stiff=0.08, torque=25.0):
        """Balance assist: the hips are pulled softly toward (pos, quat). Off: the body is on its own."""
        a = self.actors[actor]
        self.d.eq_active[a.root_eq] = 1 if on else 0
        if on:
            hips = self.body(actor, "hips")
            self.d.mocap_pos[a.mocap] = hips.xpos if pos is None else pos
            self.d.mocap_quat[a.mocap] = hips.xquat if quat is None else quat
            self.m.eq_solref[a.root_eq] = (stiff, 1.0)
            # the target is the hips' own frame: zero relative pose
            self.m.eq_data[a.root_eq, :] = 0
            self.m.eq_data[a.root_eq, 6] = 1.0
            # the trunk's posture is held much harder than its position: the pelvis must not be tipped
            # over by a swinging leg (in a real body the whole posture muscles do that)
            self.m.eq_data[a.root_eq, 10] = torque

    def grip(self, who: str, limb: str, other: str, body: str, on=True, stiff=0.02):
        """Weld `who`'s limb to `other`'s body in their *current* relative pose (a hand closing on what it
        touches). Only call it when the limb is touching (see `touching`)."""
        n = f"grip|{who}:{limb}|{other}:{body}"
        eid = self.m.equality(n).id
        if on:
            b1, b2 = self.d.body(f"{who}:{limb}"), self.d.body(f"{other}:{body}")
            # relpose = pose of body2 in body1's frame (checked: the weld then holds without any jump)
            q1inv = np.zeros(4)
            mujoco.mju_negQuat(q1inv, b1.xquat)
            rel = np.zeros(3)
            mujoco.mju_rotVecQuat(rel, b2.xpos - b1.xpos, q1inv)
            rq = np.zeros(4)
            mujoco.mju_mulQuat(rq, q1inv, b2.xquat)
            self.m.eq_data[eid, 0:3] = 0
            self.m.eq_data[eid, 3:6] = rel
            self.m.eq_data[eid, 6:10] = rq
            self.m.eq_data[eid, 10] = 1.0
            self.m.eq_solref[eid] = (stiff, 1.0)
        self.d.eq_active[eid] = 1 if on else 0
        self.log.append((round(self.t, 3), "grip" if on else "release", who, limb, other, body))

    def contacts(self, a: str, b: str | None = None) -> list:
        """[(body of a, body of b, penetration m, world pos)] for touching geoms between actors a and b."""
        out = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            n1 = self.m.body(self.m.geom_bodyid[c.geom1]).name
            n2 = self.m.body(self.m.geom_bodyid[c.geom2]).name
            for x, y in ((n1, n2), (n2, n1)):
                if x.startswith(a + ":") and (b is None or y.startswith(b + ":")) and c.dist < 0.002:
                    out.append((x.split(":")[1], y, -c.dist, c.pos.copy()))
        return out

    def touching(self, a: str, limb: str, other: str, body: str, tol=0.004) -> bool:
        g1, g2 = self.m.geom(f"{a}:{limb}").id, self.m.geom(f"{other}:{body}").id
        dist = mujoco.mj_geomDistance(self.m, self.d, g1, g2, 0.05, None)
        return dist < tol

    def step(self, seconds: float, each=None, record_every: float | None = None):
        n = int(round(seconds / self.m.opt.timestep))
        for i in range(n):
            if each:
                each(self, i / max(n - 1, 1))
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep
            if record_every and (i % max(1, int(record_every / self.m.opt.timestep)) == 0):
                self.frames.append((self.t, self.d.qpos.copy()))

    # ---- inverse kinematics on a scratch copy (the plan for the muscles)
    def ik(self, actor: str, tasks: list, posture=None, w_post=0.02, iters=120, root=True, q0=None,
           orient: list | None = None, lock: tuple = (), root_cost: float = 1.0, root_rot: bool = True,
           avoid: bool = True, allow: tuple = (), clearance: float = 0.004) -> tuple:
        """Joint targets that put sites on points: tasks = [(site, target xyz, weight)], orient =
        [(body, target quat wxyz, weight)]. Respects joint limits. `root` lets the hips move/rotate too
        (the result's root pose is where the balance assist should take them). `lock`: joint name
        fragments that keep their current angle. Returns (hinge q, root qpos (7), residual m)."""
        m = self.m_ik if avoid else self.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.d.qpos if q0 is None else q0
        a = self.actors[actor]
        P = actor + ":"
        root_q = m.jnt_qposadr[m.joint(P + "root").id]
        cols = a.dadr if root else a.dadr[6:]
        locked = np.array([any(f in m.joint(j).name for f in lock) for j in a.jids]) if lock else np.zeros(len(a.jids), bool)
        post = self.q(actor) if posture is None else np.asarray(posture, float)
        if posture is not None:  # start from the intended shape: the solution stays in its family
            d.qpos[a.qadr] = np.clip(post, a.lo, a.hi)
        lam = 1e-3
        res = 0.0
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for it in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errs = [], []
            res = 0.0
            for site, target, w in tasks:
                sid = m.site(P + site).id
                e = np.asarray(target, float) - d.site_xpos[sid]
                res = max(res, float(np.linalg.norm(e)))
                mujoco.mj_jacSite(m, d, jacp, None, sid)
                rows.append(jacp[:, cols] * w)
                errs.append(e * w)
            for bname, tq, w in orient or []:
                bid = m.body(P + bname).id
                cur = d.xquat[bid]
                qe = np.zeros(4)
                ci = np.zeros(4)
                mujoco.mju_negQuat(ci, cur)
                mujoco.mju_mulQuat(qe, np.asarray(tq, float), ci)
                e = np.zeros(3)
                mujoco.mju_quat2Vel(e, qe, 1.0)
                mujoco.mj_jacBody(m, d, None, jacr, bid)
                rows.append(jacr[:, cols] * w)
                errs.append(e * w)
            if avoid:  # stay out of bodies: every near contact involving this actor must open to `clearance`
                mujoco.mj_collision(m, d)
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    n1, n2 = m.body(b1).name, m.body(b2).name
                    if not (n1.startswith(P) or n2.startswith(P)):
                        continue
                    want = clearance
                    if any({n1.split(":")[-1], n2.split(":")[-1]} >= {x.split(":")[-1], y.split(":")[-1]} and
                           ((n1.startswith(x.split(":")[0]) and n2.startswith(y.split(":")[0])) or
                            (n2.startswith(x.split(":")[0]) and n1.startswith(y.split(":")[0]))) for x, y in allow):
                        want = -0.002  # an intended contact: touching is the point
                    if c.dist >= want:
                        continue
                    nrm = c.frame[:3]  # from geom1 toward geom2
                    j1, j2 = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                    mujoco.mj_jac(m, d, j1, None, c.pos, b1)
                    mujoco.mj_jac(m, d, j2, None, c.pos, b2)
                    row = nrm @ (j2 - j1)
                    rows.append(row[cols][None, :] * 2.0)
                    errs.append(np.array([(want - c.dist) * 2.0]))
            J = np.vstack(rows)
            E = np.concatenate(errs)
            nh = len(a.jids)
            # posture pull on the hinges (keeps the solution natural and the null space quiet)
            Wp = np.zeros((nh, len(cols)))
            off = 6 if root else 0
            Wp[np.arange(nh), off + np.arange(nh)] = w_post
            qh = d.qpos[a.qadr]
            J = np.vstack([J, Wp])
            E = np.concatenate([E, (post - qh) * w_post])
            if root and not root_rot:  # the hips may shift, not tilt (bending happens in the joints)
                J[:, 3:6] = 0
            if root and root_cost != 1.0:  # moving the whole body costs more than moving a limb
                J[:, :6] /= root_cost
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), E)
            if root and root_cost != 1.0:
                dq[:6] /= root_cost
            dq = np.clip(dq, -0.2, 0.2)
            if lock:
                dq[off:][locked] = 0
            full = np.zeros(m.nv)
            full[cols] = dq
            mujoco.mj_integratePos(m, d.qpos, full, 1.0)
            d.qpos[a.qadr] = np.clip(d.qpos[a.qadr], a.lo, a.hi)
        return d.qpos[a.qadr].copy(), d.qpos[root_q:root_q + 7].copy(), res


def min_jerk(t: float) -> float:
    t = float(np.clip(t, 0, 1))
    return t * t * t * (10 - 15 * t + 6 * t * t)


# ----------------------------------------------------------------------------- timed motions
def _slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if q0 @ q1 < 0:
        q1 = -q1
    dot = float(np.clip(q0 @ q1, -1, 1))
    if dot > 0.9995:
        q = q0 + (q1 - q0) * t
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


class Director:
    """Runs a scene in time: each actor moves from its current muscle targets (and balance-assist
    target) to new ones along a minimum-jerk profile; watchers fire events when contacts happen."""

    def __init__(self, scene: Scene):
        self.S = scene
        self.moves = {}
        self.watch = []  # (predicate(scene) -> bool, action(scene), once)
        self.trackers = {}

    def move(self, actor: str, q=None, root=None, T=0.3, delay=0.0):
        S = self.S
        if q is not None:
            self.trackers.pop(actor, None)
        a = S.actors[actor]
        q_from = S.d.ctrl[a.aids].copy()
        mp = S.d.mocap_pos[a.mocap].copy()
        mq = S.d.mocap_quat[a.mocap].copy()
        self.moves[actor] = dict(q0=q_from, q1=q_from if q is None else np.clip(q, a.lo, a.hi),
                                 r0=(mp, mq), r1=None if root is None else (np.asarray(root[:3]), np.asarray(root[3:7])),
                                 t0=S.t + delay, T=T)

    def track(self, actor: str, tasks_fn, T: float, posture=None, every=0.02, speed=6.0, **ik):
        """Closed loop: every `every` seconds re-solve the IK for tasks_fn(scene) (targets read from where
        things are *now*) and move the muscle targets toward it at most `speed` rad/s."""
        S = self.S
        self.trackers[actor] = dict(fn=tasks_fn, until=S.t + T, every=every, next=S.t, posture=posture, ik=ik,
                                    goal=None, root=None, speed=speed)

    def when(self, pred, action, once=True):
        self.watch.append([pred, action, once])

    def _tick(self, S, _frac):
        for actor, mv in list(self.moves.items()):
            a = S.actors[actor]
            u = min_jerk((S.t - mv["t0"]) / mv["T"]) if S.t >= mv["t0"] else 0.0
            if actor not in self.trackers:  # a tracker owns the muscles; the move still carries the hips
                S.d.ctrl[a.aids] = mv["q0"] + (mv["q1"] - mv["q0"]) * u
            if mv["r1"] is not None:
                S.d.mocap_pos[a.mocap] = mv["r0"][0] + (mv["r1"][0] - mv["r0"][0]) * u
                S.d.mocap_quat[a.mocap] = _slerp(mv["r0"][1], mv["r1"][1], u)
        dt = S.m.opt.timestep
        for actor, tr in list(self.trackers.items()):
            if S.t > tr["until"]:
                del self.trackers[actor]
                continue
            a = S.actors[actor]
            if S.t >= tr["next"]:
                tr["next"] = S.t + tr["every"]
                q, r, _ = S.ik(actor, tr["fn"](S), posture=tr["posture"], iters=40, **tr["ik"])
                tr["goal"], tr["root"] = q, r
            if tr["goal"] is not None:
                cur = S.d.ctrl[a.aids]
                S.d.ctrl[a.aids] = cur + np.clip(tr["goal"] - cur, -tr["speed"] * dt, tr["speed"] * dt)
                if tr["ik"].get("root", True) and S.d.eq_active[a.root_eq]:
                    S.d.mocap_pos[a.mocap] += np.clip(tr["root"][:3] - S.d.mocap_pos[a.mocap], -1.0 * dt, 1.0 * dt)
        for w in list(self.watch):
            if w[0](S):
                w[1](S)
                if w[2]:
                    self.watch.remove(w)

    def run(self, seconds: float, record_every=1 / 30):
        self.S.step(seconds, self._tick, record_every)


def posed(scene: Scene, actor: str, base=None, **deg) -> np.ndarray:
    """Muscle targets from named joint angles in degrees, e.g. posed(S, "A", leftLowerLeg_flex=30)
    (bone_joint with '_' for '.'); unnamed joints keep `base` (default: current targets)."""
    a = scene.actors[actor]
    q = scene.d.ctrl[a.aids].copy() if base is None else np.array(base, float)
    names = [scene.m.joint(j).name.split(":")[1] for j in a.jids]
    for k, val in deg.items():
        key = k.replace("_", ".", 1) if "." not in k else k
        hit = [i for i, n in enumerate(names) if n == key]
        if not hit:
            raise KeyError(f"{actor} has no joint {key}")
        q[hit[0]] = np.deg2rad(val)
    return np.clip(q, a.lo, a.hi)


def sym(**deg) -> dict:
    """Same angles on both sides: sym(UpperLeg_flex=20) -> leftUpperLeg_flex=20, rightUpperLeg_flex=20."""
    out = {}
    for k, val in deg.items():
        out["left" + k] = val
        out["right" + k] = val
    return out


def fk_site(scene: Scene, actor: str, q: np.ndarray, site: str) -> np.ndarray:
    """Where `site` would be with the actor's hinges at q (root where it is now)."""
    d = mujoco.MjData(scene.m)
    d.qpos[:] = scene.d.qpos
    d.qpos[scene.actors[actor].qadr] = q
    mujoco.mj_kinematics(scene.m, d)
    return d.site(f"{actor}:{site}").xpos.copy()


def place(scene: Scene, actor: str, dx: float = 0.0, dz: float = 0.0, q: np.ndarray | None = None):
    """Shift an actor across the floor (and optionally set its joint angles) before the simulation runs."""
    m, d = scene.m, scene.d
    a = scene.actors[actor]
    adr = m.jnt_qposadr[m.joint(actor + ":root").id]
    d.qpos[adr] += dx
    d.qpos[adr + 2] += dz
    if q is not None:
        d.qpos[a.qadr] = q
        d.ctrl[a.aids] = q
    mujoco.mj_forward(m, d)
