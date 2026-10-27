"""Straight ankle lock (ashi garami) reached through physics, from two people standing apart.

The defender (A) catches a front kick from the attacker (V), traps the leg against her side with her
elbow, drives through it so he falls, sits down at his trapped knee, entangles the leg - outside foot
on his hip bone, inside foot behind his buttock, knees squeezing his thigh - threads her forearm under
his Achilles with his foot in her armpit, closes a gable grip and arches back.

Every phase is closed loop: its targets are re-read from where the bodies actually are, and it ends on
a physical condition (the leg has arrived, the grip has formed, he is down...), not on a timer. Grips
form only on real contact, the right way round. Nothing is placed: muscles, gravity and contacts move
everything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .control import Director, Scene, fk_site, min_jerk, place, posed, sym
from .model import ActorSpec

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class Result:
    scene: Scene
    marks: dict = field(default_factory=dict)  # phase name -> sim time it was reached
    checks: dict = field(default_factory=dict)  # named contact checks at the end (metres)


def _gap(S, g1, g2):
    return float(mujoco.mj_geomDistance(S.m, S.d, S.m.geom(g1).id, S.m.geom(g2).id, 0.5, None))


def _back_off_scoop(S):
    S.scoop = max(0, S.scoop - 1)  # the arm comes back out a step and goes round again


def _retry_until(D, cond, progress, back_off, limit, stall=1.5, step=0.02):
    """Like _until, but a skill that stops making progress (progress(S) unchanged for `stall` s) backs off
    one step and tries again, the way a person re-adjusts a grip that isn't going in."""
    t_end, last, since = D.S.t + limit, progress(D.S), D.S.t
    while D.S.t < t_end:
        D.run(step)
        if cond(D.S):
            return True
        p = progress(D.S)
        if p != last:
            last, since = p, D.S.t
        elif D.S.t - since > stall:
            back_off(D.S)
            last, since = progress(D.S), D.S.t
    return False


def _step_to(D, side, goal, extra, posture, allow, T=1.2, clear=0.08, tries=3, tol=0.04):
    """A step onto `goal`; if the foot lands short, another (smaller) step from where it is."""
    from .control import Step
    S = D.S
    for _ in range(tries):
        st = Step(S, "A", side, goal, T=T, clear=clear)
        D.trackers.pop("A", None)
        D.track("A", lambda S, st=st: st.tasks(S) + extra(S), T=T + 0.4, posture=posture, root=True, root_rot=False,
                w_post=0.03, speed=4.0, allow=allow)
        if _until(D, lambda S, st=st: st.done(S, tol), T + 0.4):
            return True
    return False


def _until(D, cond, limit, step=0.02):
    """Run until cond(S) holds (or `limit` seconds pass). Returns whether it held."""
    t_end = D.S.t + limit
    while D.S.t < t_end:
        D.run(step)
        if cond(D.S):
            return True
    return False


def kick_catch(attacker="fem_vroid", victim="Sakurada_Fumiriya", log=print) -> Result:
    S = Scene([ActorSpec("A", attacker, (0, 0), 0), ActorSpec("V", victim, (0, 1.0), 180)])
    D = Director(S)
    R = Result(S)
    S.held = set()
    r_shin = S.m.geom("V:leftLowerLeg").size[0]
    r_fa = S.m.geom("A:rightLowerArm").size[0]

    # ------------------------------------------------------------------ stances, spacing from the kick
    stanceA = posed(S, "A", **sym(UpperLeg_flex=18, LowerLeg_flex=30, Foot_point=-12, UpperArm_raise=-62, UpperArm_flex=45,
                                  LowerArm_flex=115, Hand_flex=10), spine_flex=8, neck_flex=-6)
    stanceV = posed(S, "V", **sym(UpperLeg_flex=15, LowerLeg_flex=25, Foot_point=-10, UpperArm_raise=-60, UpperArm_flex=40,
                                  LowerArm_flex=110), spine_flex=6)
    kickV = posed(S, "V", base=stanceV, leftUpperLeg_flex=88, leftLowerLeg_flex=12, leftFoot_point=-20,
                  rightUpperLeg_flex=8, rightLowerLeg_flex=18, spine_flex=-12, chest_flex=-4)
    place(S, "A", q=stanceA)
    place(S, "V", q=stanceV)
    # a body kick at her lower ribs, sliding along her right flank (as she turns into the catch): his
    # ankle ends up beside her ribs, under her right arm
    side_w = abs(S.site("A", "armpit_r")[0])
    aim = np.array([-(side_w + r_shin - 0.005), S.site("A", "shoulder_r")[1] - 0.25, S.site("A", "chestc")[2] + 0.06])
    hf_i = [i for i, j in enumerate(S.actors["V"].jids) if S.m.joint(j).name == "V:leftUpperLeg.flex"][0]
    for _ in range(30):  # the hip angle sets the height of the kick
        dy = aim[1] - fk_site(S, "V", kickV, "ankle_l")[1]
        if abs(dy) < 0.004:
            break
        kickV[hf_i] = np.clip(kickV[hf_i] + dy / 0.8, 0.7, 2.1)
    ank = fk_site(S, "V", kickV, "ankle_l")
    place(S, "V", dz=aim[2] - ank[2], dx=aim[0] - ank[0])
    for n in "AV":
        S.assist(n, True, stiff=0.03)
    S.hold_current()
    D.run(0.3)

    # ------------------------------------------------------------------ the kick; her catching position
    S.strength("V", 3.0, ("leftUpperLeg", "leftLowerLeg"))
    D.move("V", kickV, np.r_[S.body("V", "hips").xpos + [0, 0, -0.03], S.body("V", "hips").xquat], T=0.32)
    ach_p, ank_p, kne_p = (fk_site(S, "V", kickV, n) for n in ("achilles_l", "ankle_l", "knee_l"))
    sd_p = (kne_p - ank_p) / np.linalg.norm(kne_p - ank_p)
    out_p = np.cross(sd_p, UP)
    out_p /= np.linalg.norm(out_p)
    if out_p @ (S.site("A", "chestc") - ank_p) > 0:
        out_p = -out_p
    postA = posed(S, "A", base=stanceA, spine_flex=4, rightUpperArm_raise=-55, rightUpperArm_flex=30, rightLowerArm_flex=80,
                  rightLowerArm_twist=-60)
    qr, _, res = S.ik("A", [("wristblade_r", ach_p + sd_p * 0.04 - UP * (r_shin + r_fa + 0.05) - out_p * 0.02, 1.0),
                            ("elbow_r", ach_p + out_p * 0.10 - UP * 0.10, 0.25)], posture=postA, root=False, w_post=0.03)
    D.move("A", qr, None, T=0.2)
    arrived = _until(D, lambda S: S.site("V", "ankle_l")[1] > aim[1] - 0.15 and np.linalg.norm(S.d.body("V:leftFoot").cvel[3:]) < 0.8, 0.9)
    R.marks["kick_landed"] = S.t
    log(f"[leg_lock] kick landed at t={S.t:.2f} ({'stopped' if arrived else 'timeout'}), ankle {S.site('V', 'ankle_l').round(3)}")

    # ------------------------------------------------------------------ the catch: elbow clamp + left palm on the shin
    allow_arm = (("A:leftHand", "V:leftLowerLeg"), ("A:rightLowerArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftLowerLeg"),
                 ("A:rightUpperArm", "V:leftFoot"), ("A:rightHand", "V:leftLowerLeg"), ("A:leftLowerArm", "V:leftLowerLeg"))

    S.scoop = 0  # her right forearm goes around the leg: out beside it, down past it, in under it

    def catch_tasks(S):
        ach, ank, kne = S.site("V", "achilles_l"), S.site("V", "ankle_l"), S.site("V", "knee_l")
        sd = (kne - ank) / np.linalg.norm(kne - ank)
        rt = _her_right(S)
        out = rt - sd * (rt @ sd)  # her right, square to his shin: the arm comes round the outside
        out /= np.linalg.norm(out)
        u = float(np.clip((S.site("A", "shoulder_l") - ank) @ sd, 0.06, np.linalg.norm(kne - ank) * 0.8))
        top = ank + sd * u + UP * r_shin  # the top of the shin where her left hand reaches it best
        c = ach + sd * 0.04 + UP * r_shin * 0.3  # the leg's axis by the Achilles
        way = [c + out * (r_shin + r_fa + 0.05), c + out * (r_shin + r_fa + 0.03) - UP * (r_shin + r_fa + 0.03),
               c - UP * (r_shin + r_fa) * 0.95 - out * 0.03]
        if S.scoop < 2 and np.linalg.norm(S.site("A", "wristblade_r") - way[S.scoop]) < 0.035:
            S.scoop += 1
        t = []
        if ("rightLowerArm", "leftLowerLeg") not in S.held:
            t.append(("wristblade_r", way[S.scoop], 1.0))
        else:
            t.append(("hand_r", S.site("A", "hand_r"), 1.0))
        if ("leftHand", "leftLowerLeg") not in S.held:
            t += [("palm_l", top, 0.8), ("knuckles_l", top + UP * 0.045, 0.5)]  # palm down onto the shin
        else:
            t.append(("hand_l", S.site("A", "hand_l"), 1.0))
        return t

    def catch_grips(S):
        clamp = _gap(S, "A:rightUpperArm", "V:leftLowerLeg") < 0.006 and _gap(S, "A:rightLowerArm", "V:leftLowerLeg") < 0.025
        scooped = S.scoop == 2 and _gap(S, "A:rightLowerArm", "V:leftLowerLeg") < 0.006 and \
            S.site("A", "wristblade_r")[1] < S.site("V", "achilles_l")[1]
        if ("rightLowerArm", "leftLowerLeg") not in S.held and (clamp or scooped):
            # the leg lies in the crook of her elbow, or on her forearm from above
            S.grip("A", "rightLowerArm", "V", "leftLowerLeg")
            S.held.add(("rightLowerArm", "leftLowerLeg"))
        if ("leftHand", "leftLowerLeg") not in S.held and S.touching("A", "leftHand", "V", "leftLowerLeg", tol=0.006) and \
                np.linalg.norm(S.site("A", "palm_l") - S.site("V", "shin_l")) < np.linalg.norm(S.site("A", "knuckles_l") - S.site("V", "shin_l")):
            S.grip("A", "leftHand", "V", "leftLowerLeg")  # palm side on the shin
            S.held.add(("leftHand", "leftLowerLeg"))
        return False

    D.when(catch_grips, lambda S: None, once=False)
    D.track("A", catch_tasks, T=1.0, posture=postA, root=False, w_post=0.03, speed=8.0, allow=allow_arm)
    caught = _until(D, lambda S: ("rightLowerArm", "leftLowerLeg") in S.held, 0.8)
    R.marks["caught"] = S.t
    log(f"[leg_lock] {'caught' if caught else 'NOT caught'} at t={S.t:.2f}; grips {S.log}")
    D.run(0.15)

    # ------------------------------------------------------------------ takedown: drive through the leg, he falls
    hA = S.body("A", "hips")
    S.assist("A", True, stiff=0.04, max_force=1.5, max_torque=150)  # she steps through (locomotion)
    D.move("A", None, np.r_[hA.xpos + [0, -0.05, 0.42], hA.xquat], T=0.45)
    S.assist("V", False)
    S.strength("V", 0.3, ("UpperLeg", "LowerLeg", "Foot"))
    S.strength("V", 0.08, ("rightUpperLeg", "rightLowerLeg", "rightFoot"))  # his standing leg buckles
    S.strength("V", 0.6, ("Arm", "Hand", "spine", "chest", "neck", "head"))
    fallV = posed(S, "V", **sym(UpperArm_raise=-40, UpperArm_flex=-35, LowerArm_flex=20), neck_flex=25, spine_flex=15,
                  rightUpperLeg_flex=40, rightLowerLeg_flex=60, leftLowerLeg_flex=10)
    D.move("V", fallV, None, T=0.25)
    down = _until(D, lambda S: S.body("V", "hips").xpos[1] < 0.25 and np.linalg.norm(S.d.body("V:hips").cvel[3:]) < 0.3, 2.5)
    R.marks["down"] = S.t
    log(f"[leg_lock] {'he is down' if down else 'NOT down'} at t={S.t:.2f}: V hips {S.body('V', 'hips').xpos.round(3)}")
    return R


def ground_finish(R: Result, log=print) -> Result:
    """From 'he is down with his leg trapped under her arm' to the finished straight ankle lock."""
    S = R.scene
    D = Director(S)
    D.trackers, D.watch = {}, []
    r_shin = S.m.geom("V:leftLowerLeg").size[0]
    r_fa = S.m.geom("A:rightLowerArm").size[0]
    r_th = S.m.geom("V:leftUpperLeg").size[0]

    # ------------------------------------------------------------------ she sits down, still holding the leg
    sitA = posed(S, "A", **sym(UpperLeg_flex=95, LowerLeg_flex=100, Foot_point=10), spine_flex=10, neck_flex=10)
    allow0 = (("A:rightLowerArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftFoot"),
              ("A:upperChest", "V:leftLowerLeg"), ("A:upperChest", "V:leftFoot"), ("A:chest", "V:leftLowerLeg"))
    fwdA = lambda S: S.d.body("A:hips").xmat.reshape(3, 3)[:, 2]  # noqa: E731
    lift = lambda S: [("wristblade_r", S.site("A", "armpit_r") + fwdA(S) * 0.10 - UP * 0.10, 1.0),  # noqa: E731
                      ("hand_l", S.site("A", "hand_l"), 0.2)]
    # on his back: he lies flat, head up looking at her; the trapped leg pivots freely at his hip (she
    # controls it), its knee kept straight
    lieV = posed(S, "V", **sym(UpperArm_raise=-70, UpperArm_flex=10, LowerArm_flex=30), neck_flex=30, spine_flex=5,
                 chest_flex=5, rightUpperLeg_flex=30, rightLowerLeg_flex=40, leftLowerLeg_flex=5)
    D.move("V", lieV, None, T=0.4)
    S.strength("V", 0.6, ("spine", "chest", "neck", "head", "Arm", "Hand"))
    S.strength("V", 0.03, ("leftUpperLeg",))
    # she recovers her stance first: upright, knees bent, facing his hips, still holding the leg
    hA = S.body("A", "hips").xpos.copy()
    to_v = (S.site("V", "hip_l") - hA) * np.array([1, 0, 1])
    yaw0 = np.arctan2(to_v[0], to_v[2])
    upright = posed(S, "A", **sym(UpperLeg_flex=25, LowerLeg_flex=40, Foot_point=-15), spine_flex=8)
    stand_h = S.d.body("A:hips").xpos[1] if S.d.body("A:hips").xpos[1] > 0.7 else 0.8 * S.body("A", "head").xpos[1] / 1.37 * 1.37 / 1.37
    stand_h = 0.55 * (S.site("A", "crown")[1] - 0.0) if False else 0.8 * (np.linalg.norm(S.site("A", "crown") - S.site("A", "sole_l")) / 1.45) * 0.97
    S.assist("A", True, stiff=0.06, max_force=1.5, max_torque=150)
    D.move("A", upright, np.r_[hA[0], stand_h, hA[2], np.cos(yaw0 / 2), 0, np.sin(yaw0 / 2), 0], T=0.5)
    D.track("A", lambda S: [("hand_r", S.site("A", "hand_r"), 1.0)], T=0.5, posture=upright, root=False, w_post=0.05, speed=5.0,
            allow=allow0)
    D.run(0.55)
    log(f"[leg_lock] stance recovered at t={S.t:.2f}: hips {S.body('A', 'hips').xpos.round(3)}")
    # still standing, she hugs his ankle up into her right armpit (his leg is in her grip)
    D.track("A", lift, T=2.5, posture=None, root=False, w_post=0.03, speed=5.0, allow=allow0)
    D.run(0.5)
    log(f"[leg_lock] ankle up at t={S.t:.2f}: ankle {S.site('V', 'ankle_l').round(3)} armpit {S.site('A', 'armpit_r').round(3)}")
    # she sits down with it there. The seated pose is solved from the contacts it needs: her pelvis on the
    # line of his leg a little past his knee (his knee between her thighs), her right armpit where his
    # ankle will be (on that line, at armpit height); the IK chooses how she turns and leans back for it
    hip_v, kn_v, an_v = S.site("V", "hip_l"), S.site("V", "knee_l"), S.site("V", "ankle_l")
    along = (S.body("A", "hips").xpos - hip_v) * np.array([1, 0, 1])  # from his hip toward her (his leg is vertical now)
    along /= np.linalg.norm(along)
    leg_len = np.linalg.norm(kn_v - hip_v) + np.linalg.norm(an_v - kn_v)
    seat = hip_v + along * leg_len * 0.68
    seat[1] = S.m.geom("A:hips").size[1] * 0.95
    pit_h = 0.36 * S.body("A", "head").xpos[1] / 1.37  # armpit height when sitting, leaning back (scaled to her size)
    pit_t = hip_v + along * np.sqrt(max(leg_len ** 2 - (pit_h - hip_v[1]) ** 2, 0.1)) * 0.95
    pit_t[1] = pit_h
    # her pelvis: facing his hips, turned ~12 degrees to her left (his leg rises from her crotch to her right
    # armpit), leaning back 25 degrees, sitting on her buttocks
    face = -along
    yaw = np.arctan2(face[0], face[2]) + np.deg2rad(12)
    qy = np.array([np.cos(yaw / 2), 0, np.sin(yaw / 2), 0])
    lateral = np.array([np.cos(yaw), 0, -np.sin(yaw)])  # her left (x is the character's left at rest)
    ql = np.zeros(4)
    mujoco.mju_axisAngle2Quat(ql, lateral, np.deg2rad(-25))  # pitch back about her left axis
    q_root = np.zeros(4)
    mujoco.mju_mulQuat(q_root, ql, qy)
    r_sit = np.r_[seat, q_root]
    q_sit, _, res = S.ik("A", [("armpit_r", pit_t, 0.5)], posture=sitA, root=False, w_post=0.1, avoid=False,
                         q0=np.r_[S.d.qpos[:0], S.d.qpos[0:]] if False else None)
    log(f"[leg_lock] seated pose: seat {seat.round(3)} yaw {np.rad2deg(yaw):.0f}")
    S.assist("A", True, pos=S.d.mocap_pos[S.actors["A"].mocap].copy(), quat=S.d.mocap_quat[S.actors["A"].mocap].copy(),
             stiff=0.06, max_force=1.5, max_torque=150)  # she moves her body with her legs (locomotion)
    D.move("A", None, r_sit, T=0.7)
    D.trackers.pop("A", None)
    D.track("A", lift, T=2.0, posture=sitA, root=False, w_post=0.05, speed=5.0, allow=allow0)
    for _ in range(12):
        D.run(0.1)
        Rp = S.d.body("A:hips").xmat.reshape(3, 3)
        Rc = S.d.body("A:upperChest").xmat.reshape(3, 3)
        tq = np.zeros(9); mujoco.mju_quat2Mat(tq, S.d.mocap_quat[S.actors["A"].mocap])
        log(f"   sit t={S.t:.2f} hips {S.body('A', 'hips').xpos.round(2)} pelvis up {Rp[:, 1].round(2)} target up {tq.reshape(3, 3)[:, 1].round(2)} "
            f"chest up {Rc[:, 1].round(2)} fwd {Rp[:, 2].round(2)}")
        if S.body("A", "hips").xpos[1] < seat[1] + 0.04:
            break
    D.run(0.1)
    R.marks["sat"] = S.t
    log(f"   after sit: his hip {S.site('V', 'hip_l').round(3)} ankle {S.site('V', 'ankle_l').round(3)} her armpit {S.site('A', 'armpit_r').round(3)} "
        f"mocap {S.d.mocap_pos[S.actors['A'].mocap].round(3)}")
    log(f"[leg_lock] sat at t={S.t:.2f}: A hips {S.body('A', 'hips').xpos.round(3)}")

    # ------------------------------------------------------------------ legs: ashi garami
    # outside (right) leg over the front of his thigh, sole on his hip bone; inside (left) leg under it,
    # foot behind his buttock between his legs; the two squeeze the thigh from above and below
    allow = (("A:leftHand", "V:leftLowerLeg"), ("A:rightLowerArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftLowerLeg"),
             ("A:rightUpperArm", "V:leftFoot"), ("A:rightHand", "V:leftLowerLeg"), ("A:leftLowerArm", "V:leftLowerLeg"),
             ("A:rightFoot", "V:hips"), ("A:leftFoot", "V:hips"), ("A:rightLowerLeg", "V:leftUpperLeg"),
             ("A:leftLowerLeg", "V:leftUpperLeg"), ("A:rightUpperLeg", "V:leftUpperLeg"), ("A:leftUpperLeg", "V:leftUpperLeg"),
             ("A:leftFoot", "V:leftUpperLeg"), ("A:hips", "V:leftLowerLeg"), ("A:hips", "V:leftUpperLeg"),
             ("A:leftHand", "A:rightLowerArm"), ("A:leftHand", "A:rightHand"), ("A:upperChest", "V:leftLowerLeg"),
             ("A:upperChest", "V:leftFoot"), ("A:chest", "V:leftLowerLeg"))

    def legs_tasks(S):
        hb = S.site("V", "hipbone_l")
        crotch = (S.site("V", "hip_l") + S.site("V", "hip_r")) / 2
        behind = S.site("V", "glute_l") * 0.6 + crotch * 0.4
        behind[1] = S.m.geom("A:leftFoot").size[1] + 0.005
        th_mid = S.site("V", "hip_l") * 0.45 + S.site("V", "knee_l") * 0.55
        outside = S.site("V", "outerthigh_l") - th_mid
        outside -= UP * (outside @ UP)
        outside /= np.linalg.norm(outside)
        r_ath = S.m.geom("A:rightUpperLeg").size[0]
        return [("sole_r", hb + UP * 0.02, 1.0), ("sole_l", behind, 0.8),
                ("knee_r", th_mid + UP * (r_th + r_ath + 0.02) + outside * 0.04, 0.4),   # over the thigh
                ("knee_l", th_mid - outside * (r_th + r_ath + 0.01), 0.4)]             # under/inside it

    # then the legs; the outside leg goes over his thigh: knee up first, then the sole down onto the hip
    S.over = False

    def legs_and_hold(S):
        t = legs_tasks(S)
        if not S.over:
            th_mid = S.site("V", "hip_l") * 0.45 + S.site("V", "knee_l") * 0.55
            t = [("knee_r", th_mid + UP * (r_th + 0.18), 1.0), t[1]]  # lift the knee clear above his thigh
            if S.site("A", "knee_r")[1] > th_mid[1] + r_th + 0.12:
                S.over = True
        return t + lift(S)
    D.track("A", legs_and_hold, T=1.4, posture=sitA, root=False, w_post=0.03, speed=6.0, allow=allow)
    ok = _until(D, lambda S: np.linalg.norm(S.site("A", "sole_r") - S.site("V", "hipbone_l")) < 0.06 and
                np.linalg.norm(S.site("A", "sole_l") - legs_tasks(S)[1][1]) < 0.08, 1.4)
    R.marks["legs"] = S.t
    log(f"[leg_lock] legs {'in' if ok else 'NOT in'} at t={S.t:.2f}: sole_r->hip {np.linalg.norm(S.site('A', 'sole_r') - S.site('V', 'hipbone_l')):.3f} "
        f"sole_l->behind {np.linalg.norm(S.site('A', 'sole_l') - legs_tasks(S)[1][1]):.3f}")

    # ------------------------------------------------------------------ arms: forearm under the Achilles, gable grip
    def lock_tasks(S):
        ach, ank, kne = S.site("V", "achilles_l"), S.site("V", "ankle_l"), S.site("V", "knee_l")
        sd = (kne - ank) / np.linalg.norm(kne - ank)
        med = S.site("A", "sternum") - ach
        med -= sd * (med @ sd)
        med /= np.linalg.norm(med)  # from the leg toward her chest, across it
        t = legs_tasks(S)
        if ("rightLowerArm", "leftLowerLeg") not in S.held:
            t += [("wristblade_r", ach + sd * 0.03 - UP * (r_shin + r_fa) * 0.9 + med * 0.02, 1.0),
                  ("hand_r", ach + sd * 0.03 + med * (r_shin + 0.05) + UP * 0.01, 0.4)]
        else:
            t.append(("hand_r", S.site("A", "hand_r"), 0.5))
        if ("leftHand", "rightLowerArm") not in S.held:
            t.append(("palm_l", S.site("A", "wristblade_r") + med * 0.05 + UP * 0.03, 0.8))
        return t

    def lock_grip(S):
        if ("rightLowerArm", "leftLowerLeg") not in S.held and S.touching("A", "rightLowerArm", "V", "leftLowerLeg", tol=0.006) \
                and S.site("A", "wristblade_r")[1] < S.site("V", "achilles_l")[1]:
            S.grip("A", "rightLowerArm", "V", "leftLowerLeg")
            S.held.add(("rightLowerArm", "leftLowerLeg"))
        if ("leftHand", "rightLowerArm") not in S.held and ("rightLowerArm", "leftLowerLeg") in S.held and \
                (S.touching("A", "leftHand", "A", "rightLowerArm", tol=0.008) or S.touching("A", "leftHand", "A", "rightHand", tol=0.008)):
            S.grip("A", "leftHand", "A", "rightLowerArm")
            S.held.add(("leftHand", "rightLowerArm"))
        return False

    for g in [x for x in list(S.held) if x[0] in ("rightLowerArm", "leftHand")]:  # let go to re-grip at the Achilles
        S.grip("A", g[0], "V", g[1], on=False)
        S.held.discard(g)
    D.when(lock_grip, lambda S: None, once=False)
    D.track("A", lock_tasks, T=1.2, posture=sitA, root=False, w_post=0.03, speed=6.0, allow=allow)
    ok = _until(D, lambda S: {("rightLowerArm", "leftLowerLeg"), ("leftHand", "rightLowerArm")} <= S.held, 1.2)
    R.marks["gripped"] = S.t
    log(f"[leg_lock] {'locked in' if ok else 'grips incomplete'} at t={S.t:.2f}: {S.log[-3:]}")

    # ------------------------------------------------------------------ finish: she arches back; he reacts
    archA = posed(S, "A", base=S.d.ctrl[S.actors["A"].aids], spine_flex=-20, chest_flex=-12, upperChest_flex=-8, neck_flex=-10)
    D.trackers.pop("A", None)
    hA = S.body("A", "hips")
    lean = np.zeros(4)
    mujoco.mju_axisAngle2Quat(lean, S.d.body("A:hips").xmat.reshape(3, 3)[:, 0], np.deg2rad(-20))  # pelvis tips back
    q2 = np.zeros(4)
    mujoco.mju_mulQuat(q2, lean, hA.xquat)
    D.move("A", archA, np.r_[hA.xpos, q2], T=0.5)
    S.assist("A", True, pos=hA.xpos, quat=hA.xquat, stiff=0.15, torque=5.0)  # soft: her legs and the floor carry her
    S.strength("V", 0.5)
    painV = posed(S, "V", neck_flex=35, spine_flex=20, leftUpperArm_raise=-30, leftUpperArm_flex=-10, leftLowerArm_flex=20,
                  rightUpperArm_raise=-75, rightUpperArm_flex=40, rightLowerArm_flex=60)
    D.move("V", painV, None, T=0.4)
    D.run(0.7)
    R.marks["finish"] = S.t
    R.checks = {
        "forearm_under_achilles": float(np.linalg.norm(S.site("A", "wristblade_r") - S.site("V", "achilles_l"))),
        "sole_r_on_hip": float(np.linalg.norm(S.site("A", "sole_r") - S.site("V", "hipbone_l"))),
        "inside_foot_behind": float(np.linalg.norm(S.site("A", "sole_l") - legs_tasks(S)[1][1])),
        "ankle_to_armpit": float(np.linalg.norm(S.site("V", "ankle_l") - S.site("A", "armpit_r"))),
        "knee_r_to_thigh": _gap(S, "A:rightUpperLeg", "V:leftUpperLeg"),
        "knee_l_to_thigh": _gap(S, "A:leftUpperLeg", "V:leftUpperLeg"),
    }
    log(f"[leg_lock] finish at t={S.t:.2f}: " + ", ".join(f"{k} {v:.3f}" for k, v in R.checks.items()))
    return R


def lying_start(attacker="fem_vroid", victim="Sakurada_Fumiriya", log=print) -> Result:
    """Second entry: he is on his back playing guard - his left foot pressed on the front of her right hip
    to keep her away, his right knee up and out; she stands at his hips, knees bent, facing him."""
    S = Scene([ActorSpec("A", attacker, (0, 0), 0), ActorSpec("V", victim, (0, 1.0), 180)])
    R = Result(S)
    S.held = set()
    m, d = S.m, S.d
    stanceA = posed(S, "A", **sym(UpperLeg_flex=22, LowerLeg_flex=32, Foot_point=-12, UpperArm_raise=-55, UpperArm_flex=45,
                                  LowerArm_flex=110), spine_flex=10)  # hands up in front of her chest
    place(S, "A", q=stanceA)
    # him: tipped onto his back, hips 0.60 in front of hers (his leg reaches her hip, nearly straight)
    adr = m.jnt_qposadr[m.joint("V:root").id]
    q_face = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_face, np.array([1.0, 0, 0]), np.deg2rad(90))
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_face, d.qpos[adr + 3:adr + 7].copy())
    d.qpos[adr + 3:adr + 7] = q
    d.qpos[adr + 1] = S.m.geom("V:hips").size[2] + 0.02
    lieV = posed(S, "V", **sym(UpperArm_raise=-70, UpperArm_flex=10, LowerArm_flex=30), neck_flex=35, spine_flex=8,
                 rightUpperLeg_flex=35, rightLowerLeg_flex=60, rightUpperLeg_abduct=30, leftUpperLeg_flex=60, leftLowerLeg_flex=70)
    place(S, "V", q=lieV)
    hA = S.body("A", "hips").xpos
    fw, rt = np.array([0, 0, 1.0]), np.array([-1.0, 0, 0])  # her front, her right
    # his foot frames on the outer front of her right hip (his leg along her right side); his hip in line with it
    foot_at = S.site("A", "hipbone_r") + fw * 0.04 + rt * 0.08
    place(S, "V", dz=hA[2] + 0.60 - S.site("V", "hip_l")[2], dx=foot_at[0] - S.site("V", "hip_l")[0])
    # his left leg: sole on her hip (solved for his leg joints only)
    others = ("spine", "chest", "neck", "head", "Shoulder", "Arm", "Hand", "right")
    # sole flat on her: toes up along her front, heel below
    flat = lambda p: [("sole_l", p, 1.0), ("toes_l", p + UP * 0.10 + fw * 0.01, 0.6), ("heel_l", p - UP * 0.07 + fw * 0.01, 0.6)]  # noqa: E731
    qV, _, res = S.ik("V", flat(foot_at), posture=lieV, root=False, lock=others, iters=300, avoid=False)
    qP, _, _ = S.ik("V", flat(foot_at - fw * 0.06), posture=qV, root=False, lock=others, iters=300, avoid=False)
    place(S, "V", q=qV)
    low = min(S.site("V", n)[1] for n in ("sole_r", "heel_r"))
    mujoco.mj_forward(m, d)
    pen = sorted(((c[0], c[1], round(c[2], 3)) for c in S.contacts("V", "A") if c[2] > 0.003), key=lambda c: -c[2])
    log(f"[leg_lock] guard: hip_l {S.site('V', 'hip_l').round(3)} her hip bone {S.site('A', 'hipbone_r').round(3)} interpenetration {pen[:4]}")
    log(f"[leg_lock] guard: his sole {S.site('V', 'sole_l').round(3)} on her hip {foot_at.round(3)} (res {res:.3f}), right foot y {low:.3f}")
    S.assist("A", True, stiff=0.12, max_force=0.6, max_torque=200, max_lift=0.25)
    S.hold_current()
    S.d.ctrl[S.actors["V"].aids] = qP  # his muscles aim the foot a little into her: a light, steady push
    S.strength("V", 0.8, ("leftUpperLeg", "leftLowerLeg", "leftFoot"))  # he holds the leg up, foot resting on her hip
    S.assist("V", True, stiff=0.2, max_force=0.2, max_torque=60, max_lift=0.0)  # he braces flat on his back (arms, trunk)
    D = Director(S)
    D.run(0.6)
    log(f"[leg_lock] start: V hips {S.body('V', 'hips').xpos.round(3)} sole_l {S.site('V', 'sole_l').round(3)} "
        f"A hips {S.body('A', 'hips').xpos.round(3)} contacts {sorted({(c[0], c[1]) for c in S.contacts('V', 'A')})}")
    return R


def _her_right(S):
    """Her right, level with the floor (the hips' -x: x is the character's left in the rest frame)."""
    r = -S.d.body("A:hips").xmat.reshape(3, 3)[:, 0] * np.array([1, 0, 1])
    return r / np.linalg.norm(r)


def _armpit_pocket(S):
    """Where his ankle sits in the lock: on her wrist blade in front of her right chest, his toes toward her
    right armpit - but never further from his hip than his straight leg reaches."""
    c = S.d.body("A:upperChest").xmat.reshape(3, 3)  # columns: her left, up, front (rest frame axes)
    p = S.site("A", "sternum") - c[:, 0] * 0.09 + c[:, 2] * 0.09 - c[:, 1] * 0.05
    hip = S.site("V", "hip_l")
    reach = 0.97 * (np.linalg.norm(S.site("V", "knee_l") - hip) + np.linalg.norm(S.site("V", "ankle_l") - S.site("V", "knee_l")))
    d = p - hip
    return hip + d * min(1.0, reach / np.linalg.norm(d))


def _draw_to_pocket(S, lead=0.06):
    """Forearm target that carries his ankle toward the pocket, a few cm ahead at a time."""
    step = _armpit_pocket(S) - S.site("V", "ankle_l")
    step *= min(1.0, lead / max(np.linalg.norm(step), 1e-6))
    return S.site("A", "wristblade_r") + step


def _toward_goal(S, lead=0.10):
    """How the holding hand should move to carry his ankle toward S.ank_goal: at most `lead` metres ahead
    of where the ankle is (closed loop on the leg, not on the hand)."""
    step = S.ank_goal - S.site("V", "ankle_l")
    return step * min(1.0, lead / max(np.linalg.norm(step), 1e-6))


def _hips_at(S, h, w=0.6):
    """Plan task: her hips toward height h, a few cm ahead at a time (the legs straighten or bend for it)."""
    pv = S.site("A", "pelvis")
    return ("@y:pelvis", pv + UP * np.clip(h - pv[1], -0.05, 0.05), w)


def _ground_height(S, actor, q):
    """Hips height that puts the lowest sole/heel point of pose q on the floor."""
    low = min(fk_site(S, actor, q, n)[1] for n in ("sole_l", "sole_r", "heel_l", "heel_r"))
    return S.body(actor, "hips").xpos[1] - low + 0.005


def _fwd(S):
    """Her facing direction, level with the floor."""
    f = S.d.body("A:hips").xmat.reshape(3, 3)[:, 2] * np.array([1, 0, 1])
    return f / np.linalg.norm(f)


FEET = ("sole_l", "sole_r", "heel_l", "heel_r")


def grab_and_lift(R: Result, log=print) -> Result:
    """His foot is on her right hip. She sinks a little (knees, hips), and her left palm comes down on top
    of his shin above the ankle: she holds the leg there; he goes slack in it as she takes control."""
    from .control import Reach
    S = R.scene
    D = Director(S)
    r_shin = S.m.geom("V:leftLowerLeg").size[0]
    base = posed(S, "A", **sym(UpperLeg_flex=35, LowerLeg_flex=45, Foot_point=-15), spine_flex=18)

    def shin_top(S):
        ank, kne = S.site("V", "ankle_l"), S.site("V", "knee_l")
        sd = (kne - ank) / np.linalg.norm(kne - ank)
        n = UP - sd * (UP @ sd)
        n /= np.linalg.norm(n)
        return ank + sd * 0.12 + n * r_shin, n

    left = Reach(S, "A", "l", "V", "leftLowerLeg", shin_top)
    D.when(left.check, lambda S: None, once=False)
    D.track("A", lambda S: left.tasks(S) + [("hand_r", S.site("A", "hip_r") + _her_right(S) * 0.22 + UP * 0.15, 0.3)], T=3.0,
            posture=base, root=True, root_rot=False, plant=FEET, w_post=0.03, speed=3.0, allow=left.allow())
    ok = _until(D, lambda S: left.done, 3.0)
    R.marks["grabbed"] = S.t
    S.strength("V", 0.05, ("leftUpperLeg", "leftLowerLeg", "leftFoot"))  # caught: his leg goes slack, she carries it
    S.ank_goal = S.site("V", "ankle_l").copy()
    R.marks["lifted"] = S.t
    log(f"[leg_lock] {'left palm on his shin' if ok else 'NOT gripped'} at t={S.t:.2f}: stage {left.stage} "
        f"his ankle {S.site('V', 'ankle_l').round(3)} her hips {S.body('A', 'hips').xpos.round(3)}")
    return R


def wrap_leg(R: Result, log=print) -> Result:
    """Standing with his ankle held up, she wraps her right forearm under his Achilles (around the leg:
    out beside it, down past it, in under it) and draws the foot into her right armpit."""
    S = R.scene
    D = Director(S)
    r_shin = S.m.geom("V:leftLowerLeg").size[0]
    r_fa = S.m.geom("A:rightLowerArm").size[0]
    S.scoop = 0
    allow = (("A:leftHand", "V:leftLowerLeg"), ("A:rightLowerArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftLowerLeg"),
             ("A:rightHand", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftFoot"), ("A:upperChest", "V:leftFoot"),
             ("A:upperChest", "V:leftLowerLeg"))

    def tasks(S):
        ach, ank, kne = S.site("V", "achilles_l"), S.site("V", "ankle_l"), S.site("V", "knee_l")
        sd = (kne - ank) / np.linalg.norm(kne - ank)
        rt = _her_right(S)
        out = rt - sd * (rt @ sd)  # her right, square to his shin: the arm comes round the outside
        out /= np.linalg.norm(out)
        c = ach + sd * 0.04 + UP * r_shin * 0.3
        way = [c + out * (r_shin + r_fa + 0.05), c + out * (r_shin + r_fa + 0.03) - UP * (r_shin + r_fa + 0.03),
               c - UP * (r_shin + r_fa) * 0.75 - out * 0.03]  # pressed up into the Achilles
        if S.scoop < 2 and np.linalg.norm(S.site("A", "wristblade_r") - way[S.scoop]) < 0.05:
            S.scoop += 1
        t = [("hand_l", S.site("A", "hand_l") + _toward_goal(S), 1.0)]  # the left hand keeps the leg up
        if ("rightLowerArm", "leftLowerLeg") not in S.held:
            t.append(("wristblade_r", way[S.scoop], 1.0))
            if S.scoop == 2:  # under the Achilles the forearm turns in across her front: elbow bent at her side
                t.append(("hand_r", _hand_across(S), 0.5))
        else:  # clamped: the forearm holds the leg where it is (it reaches her armpit as she goes down)
            t[0] = ("hand_l", S.site("A", "hand_l"), 0.5)
            t += [("wristblade_r", S.site("A", "wristblade_r"), 0.5), ("hand_r", _hand_across(S), 0.5)]
        return t

    def grip(S):
        if ("rightLowerArm", "leftLowerLeg") not in S.held and S.scoop == 2 and \
                _gap(S, "A:rightLowerArm", "V:leftLowerLeg") < 0.010 and S.site("A", "wristblade_r")[1] < S.site("V", "achilles_l")[1] \
                and np.linalg.norm(S.site("A", "wristblade_r") - S.site("V", "achilles_l")) < 0.07:  # under the Achilles itself
            S.grip("A", "rightLowerArm", "V", "leftLowerLeg", at=("wristblade_r", "achilles_l"))
            S.held.add(("rightLowerArm", "leftLowerLeg"))
        return False

    D.when(grip, lambda S: None, once=False)
    stand = posed(S, "A", **sym(UpperLeg_flex=30, LowerLeg_flex=40, Foot_point=-12), spine_flex=15)
    S.hs = _ground_height(S, "A", stand)
    D.track("A", tasks, T=8.0, posture=stand, root=True, root_rot=False, plant=FEET, w_post=0.04, speed=4.0, allow=allow)
    ok = _retry_until(D, lambda S: ("rightLowerArm", "leftLowerLeg") in S.held, lambda S: S.scoop, _back_off_scoop, 6.0)
    D.trackers.pop("A", None)
    D.track("A", tasks, T=0.5, posture=stand, root=True, root_rot=False, plant=FEET, w_post=0.04, speed=4.0, allow=allow)
    D.run(0.3)  # settle with the forearm clamped under the Achilles
    R.marks["wrapped"] = S.t
    log(f"[leg_lock] {'wrapped' if ok else 'NOT wrapped'} at t={S.t:.2f}: ankle {S.site('V', 'ankle_l').round(3)} "
        f"armpit {S.site('A', 'armpit_r').round(3)} scoop {S.scoop} her hips {S.body('A', 'hips').xpos.round(3)} "
        f"ankle off the armpit pocket {np.linalg.norm(S.site('V', 'ankle_l') - _armpit_pocket(S)):.3f} "
        f"forearm-shin gap {_gap(S, 'A:rightLowerArm', 'V:leftLowerLeg'):.3f} blade below achilles {S.site('V', 'achilles_l')[1] - S.site('A', 'wristblade_r')[1]:.3f} armpit-ankle {np.linalg.norm(S.site('A', 'armpit_r') - S.site('V', 'ankle_l')):.3f}")
    return R


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, np.sin(yaw / 2), 0])


def sit_into_lock(R: Result, log=print) -> Result:
    """Shared by both entries. From: she stands holding his left ankle, her right forearm under his
    Achilles, his foot at her chest; he is on his back. She sits straight down and back, the arm clamping
    the foot to her ribs (his knee straightens as she goes), then her legs go in: right sole on his left
    hip bone, left foot behind his buttock, knees pinching his thigh. Then the gable grip and the arch."""
    S = R.scene
    D = Director(S)
    D.trackers, D.watch = {}, []
    S.strength("V", 0.05, ("leftUpperLeg", "leftLowerLeg", "leftFoot"))
    r_th = S.m.geom("V:leftUpperLeg").size[0]
    seat_h = S.m.geom("A:hips").size[1] * 0.9
    allow = (("A:rightLowerArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftFoot"),
             ("A:rightHand", "V:leftLowerLeg"), ("A:leftHand", "V:leftLowerLeg"), ("A:upperChest", "V:leftFoot"),
             ("A:chest", "V:leftFoot"), ("A:upperChest", "V:leftLowerLeg"), ("A:chest", "V:leftLowerLeg"),
             ("A:rightFoot", "V:hips"), ("A:leftFoot", "V:hips"), ("A:rightLowerLeg", "V:leftUpperLeg"),
             ("A:leftLowerLeg", "V:leftUpperLeg"), ("A:rightUpperLeg", "V:leftUpperLeg"), ("A:leftUpperLeg", "V:leftUpperLeg"),
             ("A:leftFoot", "V:leftUpperLeg"), ("A:hips", "V:leftLowerLeg"), ("A:hips", "V:leftUpperLeg"),
             ("A:leftHand", "A:rightLowerArm"), ("A:leftHand", "A:rightHand"))

    def clamp(S):  # the arm keeps the foot against her ribs; the left hand stays on the shin
        return [("wristblade_r", S.site("A", "wristblade_r"), 0.3), ("hand_l", S.site("A", "hand_l"), 0.3)]

    # -------------------------------------------------------------- step in: left foot deep between his legs
    from .control import Step
    hipc = (S.site("V", "hip_l") + S.site("V", "hip_r")) / 2
    along = (S.site("A", "pelvis") - hipc) * np.array([1, 0, 1])
    along /= np.linalg.norm(along)  # from his hips toward her
    left_of = np.cross(UP, -along)  # her left as she faces him
    goal = hipc + along * 0.28 + left_of * 0.06
    goal[1] = S.site("A", "sole_l")[1]
    stand = posed(S, "A", **sym(UpperLeg_flex=35, LowerLeg_flex=45, Foot_point=-15), spine_flex=15)
    ok = _step_to(D, "l", goal, clamp, stand, allow)
    R.marks["stepped"] = S.t
    log(f"[leg_lock] {'stepped in' if ok else 'step NOT done'} at t={S.t:.2f}: sole_l off {np.round(S.site('A', 'sole_l') - goal, 3)} "
        f"hips {S.body('A', 'hips').xpos.round(3)}")

    # -------------------------------------------------------------- sit: squat to the bottom, then sit down behind the heels
    low = posed(S, "A", **sym(UpperLeg_flex=110, LowerLeg_flex=140, Foot_point=-20), spine_flex=25, neck_flex=10)
    sitA = posed(S, "A", **sym(UpperLeg_flex=75, LowerLeg_flex=90, Foot_point=10), spine_flex=10, neck_flex=10)
    D.trackers.pop("A", None)
    D.track("A", lambda S: clamp(S) + [_hips_at(S, 0.30, 0.8)], T=1.2, posture=low, root=True, root_rot=False, plant=FEET,
            w_post=0.03, speed=3.0, allow=allow)
    _until(D, lambda S: S.site("A", "pelvis")[1] < 0.34, 1.2)
    log(f"[leg_lock] squat bottom at t={S.t:.2f}: pelvis {S.site('A', 'pelvis').round(3)}")
    # she lets herself down onto her seat behind her heels: gravity takes her, the legs control it (the
    # assist gives almost no lift here, so nothing holds her in the air)
    heels = (S.site("A", "heel_l") + S.site("A", "heel_r")) / 2
    seat = heels - _fwd(S) * 0.12
    seat[1] = seat_h
    p0, t0, T = S.site("A", "pelvis").copy(), S.t, 0.8
    S.assists["A"]["lmax"] = 0.05 * S.actors["A"].mass * 9.81

    def sit(S):
        u = min_jerk((S.t - t0) / T)
        return clamp(S) + [("pelvis", p0 + (seat - p0) * u, 1.0)]

    D.trackers.pop("A", None)
    D.track("A", sit, T=T + 1.0, posture=sitA, root=True, root_rot=False, balance=0.0, w_post=0.03, speed=3.0,
            allow=allow)  # feet free: they slide out in front as the seat goes down
    on_seat = lambda S: any(c[0] == "hips" and c[1] == "world" for c in S.contacts("A"))  # noqa: E731
    ok = _until(D, on_seat, T + 1.0)
    D.run(0.3)
    S.assists["A"]["lmax"] = 0.25 * S.actors["A"].mass * 9.81
    # scoot in: feet on the floor in front, she lifts her seat a little and draws it toward him, sets it down
    dist = lambda S: np.linalg.norm((S.site("A", "pelvis") - hipc) * np.array([1, 0, 1]))  # noqa: E731
    for _ in range(6):  # close enough that, lying back, his straight leg reaches her armpit
        if dist(S) < 0.60:
            break
        p0 = S.site("A", "pelvis").copy()
        p1 = p0 + along * -min(0.12, dist(S) - 0.56)  # toward him (along points from him to her)
        t0 = S.t

        def scoot(S, p0=p0, p1=p1, t0=t0):
            u = min_jerk((S.t - t0) / 0.7)
            return clamp(S) + [("pelvis", p0 + (p1 - p0) * u + UP * 0.05 * np.sin(np.pi * u), 1.0)]

        D.trackers.pop("A", None)
        D.track("A", scoot, T=0.9, posture=sitA, root=True, root_rot=False, plant=FEET, balance=0.0, w_post=0.03, speed=3.0,
                allow=allow)
        D.run(0.9)
        log(f"   scoot at t={S.t:.2f}: pelvis to his hips {dist(S):.2f}")
    # seated: the outside (right) leg comes up and its sole goes onto his left hip bone
    on_hip = _foot_on_hip
    sr0, t0, seat_now = S.site("A", "sole_r").copy(), S.t, S.site("A", "pelvis").copy()

    def foot_up(S):
        u = min_jerk((S.t - t0) / 1.0)
        goal = _within_reach(S, "hip_r", on_hip(S), 0.70)
        return clamp(S) + [("sole_r", sr0 + (goal - sr0) * u + UP * 0.12 * np.sin(np.pi * u), 1.0),
                           ("pelvis", seat_now, 3.0)]

    D.trackers.pop("A", None)
    D.track("A", foot_up, T=1.6, posture=sitA, root=True, root_rot=False, plant=("sole_l", "heel_l"), balance=0.0, w_post=0.03,
            speed=3.0, allow=allow)
    _until(D, lambda S: S.t > t0 + 1.0 and np.linalg.norm(S.site("A", "sole_r") - on_hip(S)) < 0.05, 1.6)
    R.marks["sat"] = S.t
    log(f"[leg_lock] {'sat' if ok else 'NOT sat'} at t={S.t:.2f}: pelvis {S.site('A', 'pelvis').round(3)} "
        f"his ankle {S.site('V', 'ankle_l').round(3)} her armpit {S.site('A', 'armpit_r').round(3)} "
        f"pelvis to his hips {np.linalg.norm((S.site('A', 'pelvis') - hipc) * np.array([1, 0, 1])):.2f} "
        f"right sole to his hip {np.linalg.norm(S.site('A', 'sole_r') - on_hip(S)):.3f}")
    return R


def _pitched(S, deg, f=None):
    """Her pelvis orientation: heading f (default: hers now), tipped back `deg` degrees about her left-right axis."""
    f = _fwd(S) if f is None else f
    yaw = np.arctan2(f[0], f[2])
    qy = _yaw_quat(yaw)
    ql = np.zeros(4)
    mujoco.mju_axisAngle2Quat(ql, np.cross(UP, f), np.deg2rad(-deg))  # about her left axis: back
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, ql, qy)
    return q


LOCK_ALLOW = (("A:rightLowerArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftLowerLeg"), ("A:rightUpperArm", "V:leftFoot"),
              ("A:rightHand", "V:leftLowerLeg"), ("A:leftHand", "V:leftLowerLeg"), ("A:upperChest", "V:leftFoot"),
              ("A:chest", "V:leftFoot"), ("A:upperChest", "V:leftLowerLeg"), ("A:chest", "V:leftLowerLeg"),
              ("A:rightFoot", "V:hips"), ("A:leftFoot", "V:hips"), ("A:rightLowerLeg", "V:leftUpperLeg"),
              ("A:leftLowerLeg", "V:leftUpperLeg"), ("A:rightUpperLeg", "V:leftUpperLeg"), ("A:leftUpperLeg", "V:leftUpperLeg"),
              ("A:leftFoot", "V:leftUpperLeg"), ("A:hips", "V:leftLowerLeg"), ("A:hips", "V:leftUpperLeg"),
              ("A:leftHand", "A:rightLowerArm"), ("A:leftHand", "A:rightHand"), ("A:rightFoot", "V:leftUpperLeg"))


def lie_back(R: Result, log=print) -> Result:
    """Seated with his foot clamped under her right armpit: her left hand lets go of the shin and she lies
    back, square to him - the clamp goes back with her chest, so his knee straightens and his leg lies
    across her."""
    S = R.scene
    D = Director(S)
    D.trackers, D.watch = {}, []
    allow = LOCK_ALLOW
    # as she lies back the clamp brings his ankle into the pocket under her right armpit (and keeps it there,
    # moving with her chest): the forearm's place, in her upper chest's frame
    Rc = S.d.body("A:upperChest").xmat.reshape(3, 3)
    wb_goal = _armpit_pocket(S) + (S.site("A", "wristblade_r") - S.site("V", "ankle_l"))
    # a hold, not a crush: her arm holds the leg with part of its strength (the grip does the holding)
    S.strength("A", 0.5, ("rightShoulder", "rightUpperArm", "rightLowerArm"))
    # on the floor she needs no balance help: the assist only steers lightly (its torque could wedge her into him)
    S.assist("A", True, pos=S.d.mocap_pos[S.actors["A"].mocap].copy(), quat=S.d.mocap_quat[S.actors["A"].mocap].copy(),
             stiff=0.15, max_force=0.3, max_torque=40, max_lift=0.1)
    S.clamp_local = Rc.T @ (wb_goal - S.d.body("A:upperChest").xpos)  # see _clamp_hold
    seat = S.site("A", "pelvis").copy()  # she stays sitting where she is: the legs reach, the body doesn't slide
    seat[1] = min(seat[1], 0.2)
    # -------------------------------------------------------------- lie back: his leg straightens across her
    # her left hand lets go of his shin (it will close on her right wrist): the forearm alone holds the leg
    if ("leftHand", "leftLowerLeg") in S.held:
        S.grip("A", "leftHand", "V", "leftLowerLeg", on=False)
        S.held.discard(("leftHand", "leftLowerLeg"))
    hipc = (S.site("V", "hip_l") + S.site("V", "hip_r")) / 2
    to_him = (hipc - S.site("A", "pelvis")) * np.array([1, 0, 1])
    to_him /= np.linalg.norm(to_him)
    q_back = _pitched(S, 60, to_him)  # square to him, lying back: the leg straightens across her
    back = posed(S, "A", **sym(UpperLeg_flex=60, LowerLeg_flex=80, Foot_point=10), spine_flex=0, chest_flex=-5,
                 upperChest_flex=0, neck_flex=35)  # a long back, chin tucked to watch him
    D.track("A", lambda S: _clamp_hold(S) + [("pelvis", seat, 1.5), ("sole_r", _foot_on_hip(S), 1.0)], T=2.0, posture=back, root=True, root_rot=True, root_cost=1.0,
            plant=("sole_l", "heel_l"), balance=0.0, w_post=0.04, speed=1.5, allow=allow, orient=[("hips", q_back, 1.0), ("upperChest", q_back, 0.8)], upright=0.0)
    D.run(2.0)
    R.marks["reclined"] = S.t
    kne = S.site("V", "knee_l")
    log(f"[leg_lock] lay back at t={S.t:.2f}: right sole off his hip {np.linalg.norm(S.site('A', 'sole_r') - _foot_on_hip(S)):.3f} ankle off the armpit pocket {np.linalg.norm(S.site('V', 'ankle_l') - _armpit_pocket(S)):.3f} pelvis {S.site('A', 'pelvis').round(3)} pelvis up {S.d.body('A:hips').xmat.reshape(3, 3)[:, 1].round(2)} "
        f"his knee {kne.round(3)} (flex {np.rad2deg(S.q('V')[[S.m.joint(j).name for j in S.actors['V'].jids].index('V:leftLowerLeg.flex')]):.0f} deg)")
    S.seat, S.q_back = seat, q_back
    return R


def _within_reach(S, root_site, target, reach):
    """target, or the nearest point to it her limb can reach from root_site (it doesn't drag her body)."""
    o = S.site("A", root_site)
    d = target - o
    return o + d * min(1.0, reach / max(np.linalg.norm(d), 1e-6))


def _foot_on_hip(S):
    """Her outside (right) sole on the front of his left hip bone."""
    return S.site("V", "hipbone_l") + UP * 0.02 + _her_right(S) * 0.02


def _hand_across(S):
    """Her right hand in front of her sternum: the forearm runs across her front from under his Achilles,
    the elbow bent at her side (where the left hand will join it for the gable grip)."""
    c = S.d.body("A:upperChest").xmat.reshape(3, 3)
    return S.site("A", "sternum") + c[:, 2] * 0.14 - c[:, 1] * 0.06 + c[:, 0] * 0.02


def _clamp_hold(S):
    """The forearm stays where it is relative to her upper chest (the clamp moves with her trunk), the hand
    across her front."""
    b = S.d.body("A:upperChest")
    return [("wristblade_r", b.xpos + b.xmat.reshape(3, 3) @ S.clamp_local, 0.8), ("hand_r", _hand_across(S), 0.2)]


def finish_lock(R: Result, log=print) -> Result:
    """Lying back with his leg across her: her right leg comes over his thigh and the sole lands on his left hip bone, her left foot slides under the thigh to behind
    his buttock, knees pinch; her left hand leaves the shin and closes on her right wrist (gable grip);
    then she arches - hips forward, shoulders back - and his ankle is bent against her forearm."""
    S = R.scene
    D = Director(S)
    D.trackers, D.watch = {}, []
    r_th = S.m.geom("V:leftUpperLeg").size[0]
    r_ath = S.m.geom("A:rightUpperLeg").size[0]
    allow = LOCK_ALLOW
    lieA = posed(S, "A", **sym(UpperLeg_flex=70, LowerLeg_flex=70, Foot_point=10), spine_flex=10, neck_flex=25)

    def thigh_mid(S):
        return S.site("V", "hip_l") * 0.45 + S.site("V", "knee_l") * 0.55

    def outside(S):
        o = S.site("V", "outerthigh_l") - thigh_mid(S)
        o -= UP * (o @ UP)
        return o / np.linalg.norm(o)

    def behind(S):  # her inside foot: between his legs by his crotch, under the trapped thigh (within her reach)
        crotch = (S.site("V", "hip_l") + S.site("V", "hip_r")) / 2
        b = crotch + (S.site("V", "hip_r") - S.site("V", "hip_l")) * 0.15  # between his legs, under the trapped thigh
        reach = S.site("A", "hip_l") + (b - S.site("A", "hip_l")) * min(1.0, 0.62 / np.linalg.norm(b - S.site("A", "hip_l")))
        reach[1] = S.m.geom("A:leftFoot").size[1] + 0.01
        return reach

    sl0, t0 = S.site("A", "sole_l").copy(), S.t

    def legs(S):  # outside sole stays on his hip; the inside foot is lifted and placed under his thigh, behind
        tm = thigh_mid(S)  # it; the knees pinch the thigh
        u = min_jerk((S.t - t0) / 2.0)
        foot = sl0 + (behind(S) - sl0) * u + UP * 0.10 * np.sin(np.pi * u)
        return [("sole_r", _foot_on_hip(S), 1.0), ("knee_r", tm + outside(S) * (r_th + r_ath + 0.01) + UP * 0.03, 0.1),
                ("sole_l", foot, 0.8), ("knee_l", tm - UP * (r_th + r_ath + 0.01), 0.1 * u)]

    seat, q_back = S.seat, S.q_back
    hold = _clamp_hold
    D.track("A", lambda S: legs(S) + hold(S) + [("pelvis", seat, 1.5)], T=3.0, posture=lieA, root=True, root_rot=True,
            root_cost=2.0, w_post=0.03, speed=1.5, allow=allow, orient=[("hips", q_back, 1.0), ("upperChest", q_back, 0.8)], upright=0.0)
    ok = _until(D, lambda S: np.linalg.norm(S.site("A", "sole_r") - S.site("V", "hipbone_l")) < 0.07 and
                np.linalg.norm(S.site("A", "sole_l") - behind(S)) < 0.08, 3.0)
    R.marks["legs"] = S.t
    log(f"[leg_lock] legs {'in' if ok else 'NOT in'} at t={S.t:.2f}: sole_r->his hip {np.linalg.norm(S.site('A', 'sole_r') - S.site('V', 'hipbone_l')):.3f} "
        f"sole_l->behind {np.linalg.norm(S.site('A', 'sole_l') - behind(S)):.3f} pelvis {S.site('A', 'pelvis').round(3)}")
    return R
