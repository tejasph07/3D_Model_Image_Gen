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

from .control import Director, Scene, fk_site, place, posed, sym
from .model import ActorSpec

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class Result:
    scene: Scene
    marks: dict = field(default_factory=dict)  # phase name -> sim time it was reached
    checks: dict = field(default_factory=dict)  # named contact checks at the end (metres)


def _gap(S, g1, g2):
    return float(mujoco.mj_geomDistance(S.m, S.d, S.m.geom(g1).id, S.m.geom(g2).id, 0.5, None))


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
    """Second entry: he is already on his back; she stands a metre beyond his feet, facing him."""
    S = Scene([ActorSpec("A", attacker, (0, 0), 0), ActorSpec("V", victim, (0, 1.0), 180)])
    D = Director(S)
    R = Result(S)
    S.held = set()
    # lay him on his back: rotate his root 90 degrees (face up), feet toward her, then let him settle
    m, d = S.m, S.d
    adr = m.jnt_qposadr[m.joint("V:root").id]
    hip_h = S.m.geom("V:hips").size[2] + 0.02
    leg = np.linalg.norm(S.site("V", "hip_l") - S.site("V", "ankle_l"))
    q_face = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_face, np.array([1.0, 0, 0]), np.deg2rad(90))  # tip back: his front faces up
    q_yaw = d.qpos[adr + 3:adr + 7].copy()
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_face)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_face, q_yaw)
    d.qpos[adr:adr + 3] = [0.0, hip_h, 0.55 + leg + 0.05]
    d.qpos[adr + 3:adr + 7] = q
    lieV = posed(S, "V", **sym(UpperArm_raise=-70, UpperArm_flex=10, LowerArm_flex=30), neck_flex=30, spine_flex=5,
                 rightUpperLeg_flex=30, rightLowerLeg_flex=40)
    place(S, "V", q=lieV)
    stanceA = posed(S, "A", **sym(UpperLeg_flex=18, LowerLeg_flex=30, Foot_point=-12, UpperArm_raise=-62, UpperArm_flex=45,
                                  LowerArm_flex=115), spine_flex=8)
    place(S, "A", q=stanceA)
    S.assist("A", True, stiff=0.04)
    S.hold_current()
    D.run(0.6)  # he settles on the floor
    log(f"[leg_lock] start: V hips {S.body('V', 'hips').xpos.round(3)} ankle_l {S.site('V', 'ankle_l').round(3)} "
        f"A hips {S.body('A', 'hips').xpos.round(3)}")
    return R


def _her_right(S):
    """Her right, level with the floor (the hips' -x: x is the character's left in the rest frame)."""
    r = -S.d.body("A:hips").xmat.reshape(3, 3)[:, 0] * np.array([1, 0, 1])
    return r / np.linalg.norm(r)


def _toward_goal(S, lead=0.10):
    """How the holding hand should move to carry his ankle toward S.ank_goal: at most `lead` metres ahead
    of where the ankle is (closed loop on the leg, not on the hand)."""
    step = S.ank_goal - S.site("V", "ankle_l")
    return step * min(1.0, lead / max(np.linalg.norm(step), 1e-6))


def _ground_height(S, actor, q):
    """Hips height that puts the lowest sole/heel point of pose q on the floor."""
    low = min(fk_site(S, actor, q, n)[1] for n in ("sole_l", "sole_r", "heel_l", "heel_r"))
    return S.body(actor, "hips").xpos[1] - low + 0.005


def grab_and_lift(R: Result, log=print) -> Result:
    """She steps up to his feet, facing him, squats with his left foot just right of her centre, puts her
    left palm on top of his shin and draws the leg up as she rises: it pivots at his hip, foot to her chest."""
    S = R.scene
    D = Director(S)
    r_shin = S.m.geom("V:leftLowerLeg").size[0]
    S.strength("V", 0.03, ("leftUpperLeg",))  # his trapped leg pivots at the hip; she controls it
    from .control import Reach
    ank, kne = S.site("V", "ankle_l"), S.site("V", "knee_l")
    toward = (kne - ank) * np.array([1, 0, 1])
    toward /= np.linalg.norm(toward)  # from his foot up his leg: the way she faces
    right = np.cross(toward, UP)  # her right
    yaw = np.arctan2(toward[0], toward[2])
    S.toward, S.right, S.yaw = toward, right, yaw
    squat = posed(S, "A", **sym(UpperLeg_flex=75, LowerLeg_flex=95, Foot_point=-22, UpperArm_flex=35, UpperArm_raise=-50,
                                LowerArm_flex=60), spine_flex=30, chest_flex=10, neck_flex=-10)
    spot = ank - toward * 0.34 + right * 0.06  # his foot a little to her right
    h = _ground_height(S, "A", squat)
    S.assist("A", True, stiff=0.06, max_force=1.0, max_torque=260, max_lift=0.25)
    # walk-in is not the subject here: she sinks into the squat in front of his foot
    D.move("A", squat, np.r_[spot[0], h, spot[2], np.cos(yaw / 2), 0, np.sin(yaw / 2), 0], T=1.0)
    D.run(1.0)
    log(f"[leg_lock] squatting at his feet at t={S.t:.2f}: hips {S.body('A', 'hips').xpos.round(3)}")

    def shin_top(S):
        ank, kne = S.site("V", "ankle_l"), S.site("V", "knee_l")
        sd = (kne - ank) / np.linalg.norm(kne - ank)
        return ank + sd * 0.10 + UP * r_shin, UP

    left = Reach(S, "A", "l", "V", "leftLowerLeg", shin_top)

    def grab_tasks(S):
        t = left.tasks(S)
        t.append(("hand_r", S.site("A", "hip_r") + S.toward * 0.15 + UP * 0.10, 0.3))  # right hand ready by her hip
        return t

    D.when(left.check, lambda S: None, once=False)
    D.track("A", grab_tasks, T=3.0, posture=squat, root=False, w_post=0.03, speed=6.0, allow=left.allow())
    ok = False
    for _ in range(10):
        ok = _until(D, lambda S: left.done, 0.3)
        p, n = shin_top(S)
        pl, kn = S.site("A", "palm_l"), S.site("A", "knuckles_l")
        log(f"   t={S.t:.2f} left {left.stage} palm->hover {np.round(p + n * left.approach - pl, 3)} "
            f"facing {left.facing(S, n):.2f} hips {S.body('A', 'hips').xpos.round(3)}")
        if ok:
            break
    R.marks["grabbed"] = S.t
    log(f"[leg_lock] {'grabbed' if ok else 'NOT grabbed'} at t={S.t:.2f}: left {left.stage}")
    # rise with it. His leg is a pendulum hanging from his hip joint: the ankle can only travel on the
    # circle of radius |hip - ankle| around it, so the hand draws the ankle along that circle (closed
    # loop: hand target = hand + (point on the circle - where the ankle is)). Pulling off the circle
    # would drag his whole body instead.
    hip = S.site("V", "hip_l")
    L = np.linalg.norm(S.site("V", "ankle_l") - hip)
    h_lift = 0.80  # ankle at her lower ribs while she stands bent over it
    th = np.arcsin(np.clip((h_lift - hip[1]) / L, -1, 1))
    ank_goal = hip + L * (-np.cos(th) * toward + np.sin(th) * UP)
    ank_goal += right * ((S.site("V", "ankle_l") - hip) @ right)  # it stays in its own vertical plane
    S.ank_goal = ank_goal
    stand = posed(S, "A", **sym(UpperLeg_flex=40, LowerLeg_flex=40, Foot_point=-12), spine_flex=25)
    D.trackers.pop("A", None)  # rising is the plan's business: feet planted, posture standing

    def lift(S):
        return [("hand_l", S.site("A", "hand_l") + _toward_goal(S), 1.0)]

    D.track("A", lift, T=1.6, posture=stand, root=True, root_rot=False, root_cost=1.0, plant=("sole_l", "sole_r", "heel_l", "heel_r"), w_post=0.05, speed=4.0, allow=left.allow())
    _until(D, lambda S: np.linalg.norm(S.site("V", "ankle_l") - S.ank_goal) < 0.04, 1.6)
    R.marks["lifted"] = S.t
    log(f"[leg_lock] lifted at t={S.t:.2f}: ankle off goal {np.round(S.ank_goal - S.site('V', 'ankle_l'), 3)} his hips y {S.body('V', 'hips').xpos[1]:.3f} his ankle {S.site('V', 'ankle_l').round(3)} her hips {S.body('A', 'hips').xpos.round(3)}")
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
               c - UP * (r_shin + r_fa) * 0.95 - out * 0.03]
        if S.scoop < 2 and np.linalg.norm(S.site("A", "wristblade_r") - way[S.scoop]) < 0.035:
            S.scoop += 1
        t = [("hand_l", S.site("A", "hand_l") + _toward_goal(S), 1.0)]  # the left hand keeps the leg up
        if ("rightLowerArm", "leftLowerLeg") not in S.held:
            t.append(("wristblade_r", way[S.scoop], 1.0))
        else:  # close the armpit over his foot: she brings her armpit down onto the top of his ankle
            # (the leg can't come up to her: it is on its arc), the forearm stays clamped under the Achilles
            t.append(("armpit_r", S.site("V", "ankle_l") + UP * (r_shin + 0.03), 1.0))
        return t

    def grip(S):
        if ("rightLowerArm", "leftLowerLeg") not in S.held and S.scoop == 2 and \
                _gap(S, "A:rightLowerArm", "V:leftLowerLeg") < 0.006 and S.site("A", "wristblade_r")[1] < S.site("V", "achilles_l")[1]:
            S.grip("A", "rightLowerArm", "V", "leftLowerLeg")
            S.held.add(("rightLowerArm", "leftLowerLeg"))
        return False

    D.when(grip, lambda S: None, once=False)
    stand = posed(S, "A", **sym(UpperLeg_flex=30, LowerLeg_flex=40, Foot_point=-12), spine_flex=15)
    D.track("A", tasks, T=2.5, posture=stand, root=True, root_rot=False, root_cost=1.0, plant=("sole_l", "sole_r", "heel_l", "heel_r"), w_post=0.04, speed=6.0, allow=allow)
    ok = _until(D, lambda S: ("rightLowerArm", "leftLowerLeg") in S.held, 2.5)
    # now she folds over the foot: knees and hips bend, the armpit closes on top of his ankle
    D.trackers.pop("A", None)
    fold = posed(S, "A", **sym(UpperLeg_flex=50, LowerLeg_flex=60, Foot_point=-18), spine_flex=25, chest_flex=10)
    D.track("A", tasks, T=1.5, posture=fold, root=True, root_rot=False, root_cost=1.0, plant=("sole_l", "sole_r", "heel_l", "heel_r"), w_post=0.04, speed=4.0, allow=allow)
    _until(D, lambda S: np.linalg.norm(S.site("A", "armpit_r") - S.site("V", "ankle_l") - UP * (r_shin + 0.03)) < 0.04, 1.5)
    D.run(0.3)
    R.marks["wrapped"] = S.t
    log(f"[leg_lock] {'wrapped' if ok else 'NOT wrapped'} at t={S.t:.2f}: ankle {S.site('V', 'ankle_l').round(3)} "
        f"armpit {S.site('A', 'armpit_r').round(3)} scoop {S.scoop} her hips {S.body('A', 'hips').xpos.round(3)} "
        f"forearm-shin gap {_gap(S, 'A:rightLowerArm', 'V:leftLowerLeg'):.3f} armpit-ankle {np.linalg.norm(S.site('A', 'armpit_r') - S.site('V', 'ankle_l')):.3f}")
    return R
