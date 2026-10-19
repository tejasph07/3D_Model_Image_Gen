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
                        # a grip holds the point of contact but lets the limbs pivot (a ball joint, not a weld)
                        eqs.append(f'<connect name="{n}" body1="{s.name}:{g}" body2="{o.name}:{tb}" anchor="0 0 0" active="false" '
                                   f'solref="0.02 1"/>')
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
            mass = sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith(P))
            m.actuator_forcerange[a.aids] *= mass / 70.0  # a lighter body is weaker
            a.mass = mass
            a.kp0 = m.actuator_gainprm[a.aids, 0].copy()
            a.root_eq = m.equality(P + "assist").id
            a.mocap = m.body_mocapid[m.body(P + "target").id]
            self.actors[s.name] = a
        mujoco.mj_forward(m, self.d)
        # planning copy: sees geoms coming within 3 cm (contact margin) without any effect on the physics
        self.m_ik = mujoco.MjModel.from_xml_string(self.xml)
        from .model import set_masses
        set_masses(self.m_ik, specs)
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

    def assist(self, actor: str, on: bool, pos=None, quat=None, stiff=0.08, torque=25.0, max_force=0.6, max_torque=60.0,
               max_lift=0.25):
        """Balance assist: a force- and torque-limited pull of the hips toward (pos, quat), standing in for
        the balance reflexes a servo skeleton lacks. It can hold at most `max_force` x body weight and
        `max_torque` N m (scaled by mass), and pushes up with at most `max_lift` x body weight: the legs
        carry the body (and anything it lifts), the assist only steadies it.
        (`stiff`: response time in s; `torque` is kept for compatibility.) Off: the body is on its own."""
        a = self.actors[actor]
        self.d.eq_active[a.root_eq] = 0
        if not hasattr(self, "assists"):
            self.assists = {}
        if not on:
            self.assists.pop(actor, None)
            self.d.xfrc_applied[self.m.body(actor + ":hips").id] = 0
            return
        hips = self.body(actor, "hips")
        self.d.mocap_pos[a.mocap] = hips.xpos if pos is None else pos
        self.d.mocap_quat[a.mocap] = hips.xquat if quat is None else quat
        w = 2 * np.pi / max(stiff * 4, 0.05)  # natural frequency from the response time
        self.assists[actor] = dict(w=w, fmax=max_force * a.mass * 9.81, tmax=max_torque * a.mass / 70.0,
                                    lmax=max_lift * a.mass * 9.81)

    def apply_tone(self, Ti=0.25, leak=1.0):
        """Holding effort: a muscle under a steady load (its own limb's weight, a held leg) doesn't sag
        by load / stiffness the way a bare servo does; it raises its effort until the joint is where it
        was told to be. A leaky integral of each joint's angle error, scaled by the muscle's current gain
        (so a weakened muscle stays weak), applied as joint torque, with PD + effort clipped to the
        muscle's torque limit."""
        m, d = self.m, self.d
        dt = m.opt.timestep
        if not hasattr(self, "tone"):
            self.tone = {k: np.zeros(len(a.aids)) for k, a in self.actors.items()}
        for k, a in self.actors.items():
            g = m.actuator_gainprm[a.aids, 0]
            err = d.ctrl[a.aids] - d.qpos[a.qadr]
            I = self.tone[k]
            I += dt * (g * err / Ti - leak * I)
            fmax = m.actuator_forcerange[a.aids, 1]
            pd = d.actuator_force[a.aids]
            np.clip(I, -fmax - pd, fmax - pd, out=I)
            d.qfrc_applied[a.dadr[6:]] = I

    def apply_assists(self):
        m, d = self.m, self.d
        for actor, A in getattr(self, "assists", {}).items():
            a = self.actors[actor]
            bid = m.body(actor + ":hips").id
            tgt_p, tgt_q = d.mocap_pos[a.mocap], d.mocap_quat[a.mocap]
            # the whole body's mass is what the hips must steer
            M = a.mass
            w = A["w"]
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)  # (ang, lin) world frame
            # target and error in terms of the hips' own origin (the point every target is given for)
            F = M * (w * w * (tgt_p - d.xpos[bid]) - 2 * w * vel[3:])
            F[1] += M * 9.81 * 0.0
            n = np.linalg.norm(F)
            if n > A["fmax"]:
                F *= A["fmax"] / n
            F[1] = float(np.clip(F[1], -0.4 * A.get("lmax", np.inf), A.get("lmax", np.inf)))  # gravity pulls down, not the assist
            qe, ci = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(ci, d.xquat[bid])
            mujoco.mju_mulQuat(qe, tgt_q, ci)
            rv = np.zeros(3)
            mujoco.mju_quat2Vel(rv, qe, 1.0)
            I = M * 0.05  # rough trunk inertia (kg m2)
            T = I * (w * w * 4 * rv - 2 * w * 2 * vel[:3])
            n = np.linalg.norm(T)
            if n > A["tmax"]:
                T *= A["tmax"] / n
            d.xfrc_applied[bid, :3] = F
            d.xfrc_applied[bid, 3:] = T

    def grip(self, who: str, limb: str, other: str, body: str, on=True, stiff=0.02):
        """Weld `who`'s limb to `other`'s body in their *current* relative pose (a hand closing on what it
        touches). Only call it when the limb is touching (see `touching`)."""
        n = f"grip|{who}:{limb}|{other}:{body}"
        eid = self.m.equality(n).id
        if on:
            b1, b2 = self.d.body(f"{who}:{limb}"), self.d.body(f"{other}:{body}")
            # anchor: where the two actually touch (nearest points of their shapes), in each body's frame
            g1, g2 = self.m.geom(f"{who}:{limb}").id, self.m.geom(f"{other}:{body}").id
            ft = np.zeros(6)
            mujoco.mj_geomDistance(self.m, self.d, g1, g2, 0.2, ft)
            p = (ft[:3] + ft[3:]) / 2
            for k, b in ((0, b1), (3, b2)):
                qi = np.zeros(4)
                mujoco.mju_negQuat(qi, b.xquat)
                loc = np.zeros(3)
                mujoco.mju_rotVecQuat(loc, p - b.xpos, qi)
                self.m.eq_data[eid, k:k + 3] = loc
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
            self.apply_assists()
            self.apply_tone()
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep
            if record_every and (i % max(1, int(record_every / self.m.opt.timestep)) == 0):
                self.frames.append((self.t, self.d.qpos.copy()))

    # ---- inverse kinematics on a scratch copy (the plan for the muscles)
    def ik(self, actor: str, tasks: list, posture=None, w_post=0.02, iters=120, root=True, q0=None, **kw) -> tuple:
        """Joint targets that put sites on points (see _ik). Multi-start: if the first solve misses by
        more than 1.5 cm, retry from seeds with the knees and elbows partly bent (a straight limb is a
        singular point the gradient can't leave) and keep the best."""
        best = self._ik(actor, tasks, posture, w_post, iters, root, q0, **kw)
        if best[2] < 0.015:
            return best
        a = self.actors[actor]
        names = [self.m.joint(j).name for j in a.jids]
        base = self.q(actor) if posture is None else np.asarray(posture, float)
        for bend in (0.6, 1.2):
            seed = base.copy()
            for i, n in enumerate(names):
                if n.endswith(("LowerLeg.flex", "LowerArm.flex")):
                    seed[i] = max(seed[i], bend)
                if n.endswith("UpperLeg.flex"):
                    seed[i] = max(seed[i], bend * 0.7)
            r = self._ik(actor, tasks, seed, w_post * 0.5, iters, root, q0, **kw)
            if r[2] < best[2]:
                best = r
        return best

    def _ik(self, actor: str, tasks: list, posture=None, w_post=0.02, iters=120, root=True, q0=None,
           orient: list | None = None, lock: tuple = (), root_cost: float = 1.0, root_rot: bool = True,
           avoid: bool = True, allow: tuple = (), clearance: float = 0.004, upright: float = 1.0,
           hips_y: float | None = None) -> tuple:
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
        if posture is not None and q0 is None:  # start from the intended shape: the solution stays in its family
            d.qpos[a.qadr] = np.clip(post, a.lo, a.hi)
        root_q0 = d.qpos[root_q:root_q + 7].copy()
        if hips_y is not None:  # where the hips should be held vertically (rising, sitting down)
            root_q0[1] = hips_y
        lam = 4e-3  # damping: no huge steps near singular configurations (straight limbs)
        res = 0.0
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for it in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errs = [], []
            res = 0.0
            for site, target, w in tasks:
                if site == "@com":  # balance: the body's centre of mass over a floor point (horizontal only)
                    bid = m.body(P + "hips").id
                    e = np.asarray(target, float) - d.subtree_com[bid]
                    e[1] = 0.0
                    mujoco.mj_jacSubtreeCom(m, d, jacp, bid)
                    jc = jacp[:, cols].copy()
                    jc[1] = 0.0
                    rows.append(jc * w)
                    errs.append(e * w)
                    continue
                if site.startswith("@y:"):  # a height only: "@y:pelvis" -> that site's height to target[1]
                    sid = m.site(P + site[3:]).id
                    mujoco.mj_jacSite(m, d, jacp, None, sid)
                    rows.append(jacp[1:2, cols] * w)
                    errs.append(np.array([(float(np.ravel(target)[1]) - d.site_xpos[sid][1]) * w]))
                    continue
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
            off = 6 if root else 0
            # posture pull on the hinges (keeps the solution natural and the null space quiet)
            Wp = np.zeros((nh, len(cols)))
            off = 6 if root else 0
            Wp[np.arange(nh), off + np.arange(nh)] = w_post
            qh = d.qpos[a.qadr]
            J = np.vstack([J, Wp])
            E = np.concatenate([E, (post - qh) * w_post])
            if root:  # moving the whole body is a last resort: the hips stay near where they are, and upright
                Wr = np.zeros((6, len(cols)))
                Wr[np.arange(6), np.arange(6)] = w_post * 6.0
                rp = d.qpos[root_q:root_q + 3]
                E_r = np.zeros(6)
                E_r[:3] = (root_q0[:3] - rp) * w_post * 6.0
                if root_rot:  # pelvis rotation: prefer the current heading, upright
                    cur = d.qpos[root_q + 3:root_q + 7]
                    fwd = np.zeros(3)
                    mujoco.mju_rotVecQuat(fwd, np.array([0, 0, 1.0]), root_q0[3:7])
                    yw = np.arctan2(fwd[0], fwd[2])
                    tq = np.array([np.cos(yw / 2), 0, np.sin(yw / 2), 0])
                    qe, ci = np.zeros(4), np.zeros(4)
                    mujoco.mju_negQuat(ci, cur)
                    mujoco.mju_mulQuat(qe, tq, ci)
                    rv = np.zeros(3)
                    mujoco.mju_quat2Vel(rv, qe, 1.0)
                    # free-joint angular dofs are in the body frame
                    rv_local = np.zeros(3)
                    mujoco.mju_rotVecQuat(rv_local, rv, ci)
                    E_r[3:] = rv_local * w_post * 6.0 * upright
                J = np.vstack([J, Wr])
                E = np.concatenate([E, E_r])
            if root and not root_rot:  # the hips may shift, not tilt (bending happens in the joints)
                J[:, 3:6] = 0
            if root and root_cost != 1.0:  # moving the whole body costs more than moving a limb
                J[:, :6] /= root_cost
            # active set: a joint already against its limit and pushed further takes no part in this step
            qh_now = d.qpos[a.qadr]
            free = np.ones(J.shape[1], bool)
            for _ in range(4):
                Jf = J * free[None, :]
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam * np.eye(J.shape[0]), E)
                h = dq[off:]
                stuck = ((qh_now <= a.lo + 1e-4) & (h < 0)) | ((qh_now >= a.hi - 1e-4) & (h > 0))
                if not stuck.any():
                    break
                free[off:][stuck] = False
            if root and root_cost != 1.0:
                dq[:6] /= root_cost
            big = float(np.abs(dq).max())
            if big > 0.15:  # scale the whole step (clipping joints one by one would change its direction)
                dq *= 0.15 / big
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
        self._fk = mujoco.MjData(scene.m)
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

    def track(self, actor: str, tasks_fn, T: float, posture=None, every=0.02, speed=6.0, plant=(), balance=0.8, anchor=(),
              only=(), **ik):
        """Closed loop: every `every` seconds re-solve the IK for tasks_fn(scene) (targets read from where
        things are *now*) and move the muscle targets toward it at most `speed` rad/s.
        plant: sites (e.g. "sole_l", "heel_l") that stay where they are when tracking starts - feet on the
        floor. Without them the plan may bend a knee that, with the foot on the ground, would just drop
        the hips.
        anchor: sites on the ground (default: the planted ones) whose actual position fixes where the hips
        should be: the balance assist is sent to the planned hips shifted by (actual - planned) anchor
        position, so it never pulls the hips somewhere the standing leg can't carry them.
        only: joint name fragments - the plan drives just these muscles (the rest keep their own targets)."""
        S = self.S
        pinned = [(n, S.site(actor, n).copy()) for n in plant]
        anchor = tuple(anchor)
        if pinned:  # planted feet - where they actually are on the floor now - and the weight over them
            def fn(S):
                feet = [(n, np.array([S.site(actor, n)[0], p[1], S.site(actor, n)[2]])) for n, p in pinned]
                return list(tasks_fn(S)) + [(n, p, 1.5) for n, p in feet] + \
                    [("@com", np.mean([p for _, p in feet], axis=0), balance)]
        else:
            fn = tasks_fn
        self.trackers[actor] = dict(fn=fn, until=S.t + T, every=every, next=S.t, posture=posture, ik=ik,
                                    goal=None, root=None, speed=speed, anchor=anchor,
                                    mask=None if not only else np.array([any(f in S.m.joint(j).name for f in only)
                                                                         for j in S.actors[actor].jids]))

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
            if u >= 1.0 and S.t >= mv["t0"] + mv["T"]:
                del self.moves[actor]  # done: its end state stays as the targets, others may take over
        dt = S.m.opt.timestep
        for actor, tr in list(self.trackers.items()):
            if S.t > tr["until"]:
                del self.trackers[actor]
                continue
            a = S.actors[actor]
            if S.t >= tr["next"]:
                tr["next"] = S.t + tr["every"]
                adr = S.m.jnt_qposadr[S.m.joint(actor + ":root").id]
                tasks = tr["last_tasks"] = tr["fn"](S)
                if tr.get("plan") is None:  # first solve: from the intended shape, with restarts
                    q, r, _ = S.ik(actor, tasks, posture=tr["posture"], iters=120, **tr["ik"])
                else:  # then each solve continues the previous plan: a smooth trajectory, not a re-guess
                    q0 = S.d.qpos.copy()
                    if tr["ik"].get("root", True):  # the hips are part of the plan
                        q0[adr:adr + 7] = tr["plan"][0]
                    q0[a.qadr] = tr["plan"][1]
                    q, r, _ = S._ik(actor, tasks, tr["posture"], tr["ik"].get("w_post", 0.02), 30,
                                    tr["ik"].get("root", True), q0, **{k: v for k, v in tr["ik"].items() if k not in ("w_post", "root")})
                tr["plan"] = (r.copy(), q.copy())
                anc = tr["anchor"](S) if callable(tr["anchor"]) else tr["anchor"]
                if anc and tr["ik"].get("root", True):
                    d = self._fk
                    d.qpos[:] = S.d.qpos
                    d.qpos[adr:adr + 7] = r
                    d.qpos[a.qadr] = q
                    mujoco.mj_kinematics(S.m, d)
                    off = np.mean([S.site(actor, n) - d.site(f"{actor}:{n}").xpos for n in anc], axis=0)
                    r = r.copy()
                    r[[0, 2]] += off[[0, 2]]  # horizontal only: a vertical correction would feed back on a lifting foot
                tr["goal"], tr["root"] = q, r
            if tr["goal"] is not None:
                cur = S.d.ctrl[a.aids]
                step = np.clip(tr["goal"] - cur, -tr["speed"] * dt, tr["speed"] * dt)
                if tr["mask"] is not None:
                    step = step * tr["mask"]
                S.d.ctrl[a.aids] = cur + step
                if tr["ik"].get("root", True) and actor in getattr(S, "assists", {}):
                    S.d.mocap_pos[a.mocap] += np.clip(tr["root"][:3] - S.d.mocap_pos[a.mocap], -1.0 * dt, 1.0 * dt)
                    if tr["ik"].get("root_rot", True):
                        S.d.mocap_quat[a.mocap] = _slerp(S.d.mocap_quat[a.mocap], tr["root"][3:7], min(1.0, 3.0 * dt))
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


SNAP_EXTRA = ("toward", "right", "yaw", "ank_goal", "scoop", "over", "hs")  # scene attributes the phases keep


def snapshot(S: Scene) -> dict:
    """Everything needed to resume the simulation exactly here (state, muscles, grips, assists)."""
    d, m = S.d, S.m
    return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy(), eq_active=d.eq_active.copy(),
                eq_data=m.eq_data.copy(), eq_solref=m.eq_solref.copy(), mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(),
                gain=m.actuator_gainprm.copy(), bias=m.actuator_biasprm.copy(), t=S.t, held=set(getattr(S, "held", set())),
                log=list(S.log), assists={k: dict(v) for k, v in getattr(S, "assists", {}).items()}, frames=list(S.frames),
                tone={k: v.copy() for k, v in getattr(S, "tone", {}).items()},
                warm=d.qacc_warmstart.copy(), qfrc_applied=d.qfrc_applied.copy(), xfrc_applied=d.xfrc_applied.copy(),
                extra={k: v for k, v in vars(S).items() if k in SNAP_EXTRA})


def restore(S: Scene, snap: dict):
    d, m = S.d, S.m
    d.qpos[:], d.qvel[:], d.act[:], d.ctrl[:] = snap["qpos"], snap["qvel"], snap["act"], snap["ctrl"]
    d.eq_active[:] = snap["eq_active"]
    m.eq_data[:], m.eq_solref[:] = snap["eq_data"], snap["eq_solref"]
    d.mocap_pos[:], d.mocap_quat[:] = snap["mocap_pos"], snap["mocap_quat"]
    m.actuator_gainprm[:], m.actuator_biasprm[:] = snap["gain"], snap["bias"]
    S.t, S.held, S.log = snap["t"], set(snap["held"]), list(snap["log"])
    S.assists = {k: dict(v) for k, v in snap["assists"].items()}
    S.frames = list(snap["frames"])
    if "warm" in snap:  # the solver's warm start and the applied forces: resuming is then exact
        d.qacc_warmstart[:], d.qfrc_applied[:], d.xfrc_applied[:] = snap["warm"], snap["qfrc_applied"], snap["xfrc_applied"]
    if snap.get("tone"):
        S.tone = {k: v.copy() for k, v in snap["tone"].items()}
    for k, v in snap.get("extra", {}).items():
        setattr(S, k, v)
    mujoco.mj_forward(m, d)


class Reach:
    """A hand reaching for a surface point and gripping it, the way a hand does: it travels to a hover
    point `approach` metres off the surface with the palm already turned toward it, comes straight down,
    and closes only when the palm side touches. target(S) -> (point on the surface, outward normal)."""

    def __init__(self, scene: Scene, who: str, side: str, other: str, body: str, target, approach=0.06):
        self.S, self.who, self.side, self.other, self.body, self.target, self.approach = \
            scene, who, side, other, body, target, approach
        self.limb = ("left" if side == "l" else "right") + "Hand"
        self.stage = "far"  # far -> hover -> descend: the hand comes in from the open side, never across the body
        self.done = False

    def tasks(self, S):
        if self.done:
            return [(f"hand_{self.side}", S.site(self.who, f"hand_{self.side}"), 0.8)]
        p, n = self.target(S)
        n = np.asarray(n, float) / np.linalg.norm(n)
        thick = 0.036
        palm, knu = S.site(self.who, f"palm_{self.side}"), S.site(self.who, f"knuckles_{self.side}")
        if self.stage == "far":
            fp = p + n * (self.approach + 0.07)
            if np.linalg.norm(palm - fp) < 0.05:
                self.stage = "hover"
            return [(f"palm_{self.side}", fp, 1.0), (f"knuckles_{self.side}", fp + n * thick, 0.5)]
        if self.stage == "hover":
            hp = p + n * self.approach
            if np.linalg.norm(palm - hp) < 0.025 and self.facing(S, n) > 0.65:
                self.stage = "descend"
            return [(f"palm_{self.side}", hp, 1.0), (f"knuckles_{self.side}", hp + n * thick, 1.2)]
        dp = p + n * 0.002
        return [(f"palm_{self.side}", dp, 1.0), (f"knuckles_{self.side}", dp + n * thick, 0.8)]

    def facing(self, S, n) -> float:
        """cos of the angle between the back of the hand and n (1: palm flat on the surface). The palm
        faces the hand bone's -y in the rest frame; body rotations are rotations from rest."""
        back = S.d.body(f"{self.who}:{self.limb}").xmat.reshape(3, 3)[:, 1]
        return float(back @ n)

    def check(self, S):
        if self.done or self.stage != "descend":
            return self.done
        palm, knu = S.site(self.who, f"palm_{self.side}"), S.site(self.who, f"knuckles_{self.side}")
        p, _ = self.target(S)
        if S.touching(self.who, self.limb, self.other, self.body, tol=0.006) and \
                np.linalg.norm(palm - p) < np.linalg.norm(knu - p):
            S.grip(self.who, self.limb, self.other, self.body)
            if hasattr(S, "held"):
                S.held.add((self.limb, self.body))
            self.done = True
        return self.done

    def allow(self):
        return ((f"{self.who}:{self.limb}", f"{self.other}:{self.body}"),)


class Step:
    """One step: first the weight shifts over the stance foot, then the swing foot lifts, travels and comes
    down on `goal` (a floor point, or a point on a body - callable(S)) along a raised arc, the other foot
    planted throughout; the plan's free root carries the hips. tasks(S) merges with whatever else the
    body is doing."""

    SHIFT = 0.3  # fraction of T spent moving the weight before the foot leaves the ground

    def __init__(self, scene: Scene, who: str, side: str, goal, T=0.9, clear=0.07):
        self.S, self.who, self.side, self.T, self.clear = scene, who, side, T, clear
        self.goal = goal if callable(goal) else (lambda S, g=np.asarray(goal, float): g)
        o = "l" if side == "r" else "r"
        self.t0 = scene.t
        self.start = scene.site(who, f"sole_{side}").copy()
        self.heel_off = scene.site(who, f"heel_{side}") - self.start
        self.stance = [(f"sole_{o}", scene.site(who, f"sole_{o}").copy()), (f"heel_{o}", scene.site(who, f"heel_{o}").copy())]
        self.com0 = scene.d.subtree_com[scene.m.body(f"{who}:hips").id].copy()
        self.anchor = tuple(n for n, _ in self.stance)  # pass as Director.track(anchor=...)

    def phase(self, S):
        return min(1.0, max(0.0, (S.t - self.t0) / self.T))

    def u(self, S):
        return min_jerk((self.phase(S) - self.SHIFT) / (1 - self.SHIFT))

    def tasks(self, S):
        s, u = self.phase(S), self.u(S)
        g = self.goal(S)
        p = self.start + (g - self.start) * u + np.array([0, 1.0, 0]) * self.clear * np.sin(np.pi * u)
        over = np.mean([q for _, q in self.stance], axis=0)
        if s < self.SHIFT:  # weight onto the stance foot
            com = self.com0 + (over - self.com0) * min_jerk(s / self.SHIFT)
        else:  # over it while the foot is in the air; shared again as it lands
            both = (over + p + self.heel_off / 2) / 2
            com = over + (both - over) * float(np.clip((u - 0.85) / 0.15, 0, 1))
        return [(f"sole_{self.side}", p, 1.5), (f"heel_{self.side}", p + self.heel_off, 0.8)] + \
            [(n, q, 1.5) for n, q in self.stance] + [("@com", com, 1.0)]

    def done(self, S, tol=0.03):
        return self.phase(S) >= 1.0 and np.linalg.norm(S.site(self.who, f"sole_{self.side}") - self.goal(S)) < tol
