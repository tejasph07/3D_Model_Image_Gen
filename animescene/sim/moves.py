"""Moves written as keyframed contact descriptions (see keyframes.py).

Straight ankle lock (ashi garami) on his left leg, two ways in:

- `leg_lock_guard`: he is on his back with his left foot framed on her right hip. She takes the shin with her
  left palm and the outside of the ankle with her right hand, wraps her right forearm under the Achilles
  with his ankle tucked under her armpit, steps in, sits, lies back with her right sole on his left hip
  bone and her left foot by his buttock, closes the gable grip and arches.
- `leg_lock_kick`: he throws a front kick at her ribs; she catches it (forearm under the calf, left palm on
  the shin), drives him over, and finishes the same way.
"""
from __future__ import annotations

import numpy as np

from .keyframes import UP, Key, clamp, grab, hook, rest, squeeze, tuck
from .model import ActorSpec

HER_RIGHT = np.array([-1.0, 0, 0])  # she faces +z; he lies with his feet toward her

STAND_A = dict(leftUpperLeg_flex=20, rightUpperLeg_flex=20, leftLowerLeg_flex=30, rightLowerLeg_flex=30,
               leftFoot_point=-10, rightFoot_point=-10, spine_flex=8)
GUARD_HANDS = dict(leftUpperArm_raise=-55, rightUpperArm_raise=-55, leftUpperArm_flex=45, rightUpperArm_flex=45,
                   leftLowerArm_flex=110, rightLowerArm_flex=110)
ON_BACK = dict(neck_flex=30, leftUpperArm_raise=-70, rightUpperArm_raise=-70, leftLowerArm_flex=25, rightLowerArm_flex=25)
FEET = ["sole_l", "sole_r", "heel_l", "heel_r"]
LYING = ["lowback", "glute_l", "glute_r"]


def specs(attacker="fem_vroid", victim="Sakurada_Fumiriya"):
    return [ActorSpec("A", attacker, (0, 0), 0), ActorSpec("V", victim, (0, 1.0), 180)]


def _final(t):
    """The finished lock: she lies back, his straight leg across her, ankle under her right armpit."""
    return Key(t=t, name="ashi garami", touches=[
        clamp("A", "r", "V", "achilles_l", normal=-UP),
        tuck("A", "armpit_r", "V", "ankle_l", gap=-0.06),
        grab("A", "l", "A", "wristblade_r"),
        rest("A", "sole_r", "V", "hipbone_l"),
        hook("A", "sole_l", "V", "glute_l"),
        squeeze("A", "knee_r", "V", "outerthigh_l"),
        squeeze("A", "knee_l", "V", "innerthigh_l"),
    ], support={"V": LYING + ["heel_r"], "A": ["glute_l", "glute_r"]},
        lock={"V": ("leftUpperLeg", "leftLowerLeg")}, stiff={"V": 0.2},
        posture={"V": dict(ON_BACK, rightUpperLeg_flex=10, rightLowerLeg_flex=15, rightUpperLeg_abduct=15, leftUpperLeg_flex=20),
                 "A": dict(leftUpperLeg_flex=60, leftLowerLeg_flex=70, rightUpperLeg_flex=50, rightLowerLeg_flex=40, spine_flex=10,
                           neck_flex=25)},
        root={"V": dict(pos=[0, 0.12, 1.0], yaw=180, pitch=-90, pos_w=1.5, hold=1.5),
              "A": dict(pos=[0.03, 0.17, 0.54], yaw=0, pitch=-50, pos_w=0.2, hold=0.3)})


def _break(t):
    """The finish: she arches (hips forward, shoulders back), his foot is bent back against her forearm; he
    clutches at the floor, head up, in pain."""
    k = _final(t)
    k.name = "break"
    k.posture["A"].update(spine_flex=-15, chest_flex=-10, neck_flex=30)
    k.root["A"].update(pitch=-65)
    k.posture["V"].update(leftFoot_point=-25, neck_flex=45, spine_flex=15, leftUpperArm_raise=-30, leftLowerArm_flex=60,
                          rightUpperArm_raise=-35, rightLowerArm_flex=70)
    k.lock["V"] = ("leftUpperLeg", "leftLowerLeg", "leftFoot")
    return k


def leg_lock_guard():
    """He is on his back, left foot on her right hip; she stands between his feet."""
    v_guard = dict(ON_BACK, leftUpperLeg_flex=60, leftLowerLeg_flex=70, rightUpperLeg_flex=50, rightLowerLeg_flex=100,
                   rightUpperLeg_abduct=20)
    keys = [
        Key(t=0.0, name="guard", order=("A", "V"), stiff={"A": 0.15},
            touches=[rest("V", "sole_l", "A", "hipbone_r")],
            support={"A": FEET, "V": LYING + ["sole_r"]}, balance=("A",),
            posture={"A": dict(STAND_A, **GUARD_HANDS), "V": v_guard},
            root={"A": dict(pos=[0, None, 0.38], yaw=0, pos_w=0.5, hold=1.0),
                  "V": dict(pos=[0, 0.12, 1.0], yaw=180, pitch=-90, pos_w=1.5, hold=1.5)}),
        Key(t=0.6, name="grab", order=("V", "A"),
            touches=[grab("A", "l", "V", "shin_l"), grab("A", "r", "V", "ankle_l", normal=HER_RIGHT),
                     rest("V", "sole_l", "A", "hipbone_r", weight=0.5)],
            support={"A": FEET, "V": LYING + ["sole_r"]}, balance=("A",),
            lock={"V": ("leftUpperLeg", "leftLowerLeg")}, stiff={"A": 0.15, "V": 0.3},
            posture={"A": dict(STAND_A, leftUpperLeg_flex=35, rightUpperLeg_flex=35, leftLowerLeg_flex=45, rightLowerLeg_flex=45,
                               spine_flex=25, chest_flex=10, neck_flex=25),
                     "V": v_guard},
            root={"A": dict(pos=[0, None, 0.38], yaw=0, pos_w=1.0, hold=1.0),
                  "V": dict(pos=[0, 0.12, 1.0], yaw=180, pitch=-90, pos_w=1.5, hold=1.5)}),
        Key(t=1.4, name="wrap", order=("V", "A"),
            touches=[clamp("A", "r", "V", "achilles_l", normal=-UP), tuck("A", "armpit_r", "V", "ankle_l", gap=-0.06),
                     grab("A", "l", "V", "shin_l")],
            support={"A": FEET, "V": LYING + ["sole_r"]}, balance=("A",),
            lock={"V": ("leftUpperLeg", "leftLowerLeg")}, stiff={"A": 0.15, "V": 0.3},
            posture={"A": dict(STAND_A, spine_flex=20, chest_flex=5, neck_flex=20),
                     "V": dict(v_guard, leftUpperLeg_flex=60, leftLowerLeg_flex=20)},
            root={"A": dict(pos=[0, None, 0.38], yaw=0, pos_w=0.5, hold=1.0),
                  "V": dict(pos=[0, 0.12, 1.0], yaw=180, pitch=-90, pos_w=1.5, hold=1.5)}),
        Key(t=2.4, name="sit", order=("V", "A"),
            touches=[clamp("A", "r", "V", "achilles_l", normal=-UP), tuck("A", "armpit_r", "V", "ankle_l", gap=-0.06),
                     grab("A", "l", "V", "shin_l")],
            support={"A": ["glute_l", "glute_r", "sole_l", "sole_r"], "V": LYING + ["heel_r"]},
            lock={"V": ("leftUpperLeg", "leftLowerLeg")}, stiff={"V": 0.3},
            posture={"A": dict(leftUpperLeg_flex=80, rightUpperLeg_flex=80, leftLowerLeg_flex=90, rightLowerLeg_flex=90, spine_flex=15),
                     "V": dict(ON_BACK, leftUpperLeg_flex=40, leftLowerLeg_flex=5, rightUpperLeg_flex=10, rightLowerLeg_flex=15)},
            root={"A": dict(pos=[0, 0.15, 0.55], yaw=0, pitch=-15, pos_w=0.5, hold=0.5),
                  "V": dict(pos=[0, 0.12, 1.0], yaw=180, pitch=-90, pos_w=1.5, hold=1.5)}),
        _final(3.4),
        _break(4.0),
    ]
    return keys


def leg_lock_kick():
    """Both standing; he throws a front kick at her right ribs, she catches it (forearm under the calf by the
    Achilles, ankle tucked under her armpit, left palm on his shin), drives forward so he falls on his back,
    and finishes as from the guard."""
    stance_v = dict(leftUpperLeg_flex=15, rightUpperLeg_flex=15, leftLowerLeg_flex=25, rightLowerLeg_flex=25,
                    leftFoot_point=-10, rightFoot_point=-10, spine_flex=6, **GUARD_HANDS)
    kick_v = dict(stance_v, leftUpperLeg_flex=85, leftLowerLeg_flex=10, leftFoot_point=-25, spine_flex=-5)
    caught = [clamp("A", "r", "V", "achilles_l", normal=-UP), tuck("A", "armpit_r", "V", "ankle_l", gap=-0.06),
              grab("A", "l", "V", "shin_l")]
    keys = [
        Key(t=0.0, name="stance", order=("A", "V"), stiff={"A": 0.15, "V": 0.15},
            support={"A": FEET, "V": FEET}, balance=("A", "V"),
            posture={"A": dict(STAND_A, **GUARD_HANDS), "V": stance_v},
            root={"A": dict(pos=[0, None, 0.0], yaw=0, pos_w=0.5, hold=1.0),
                  "V": dict(pos=[0, None, 1.15], yaw=180, pos_w=0.5, hold=1.0)}),
        Key(t=0.35, name="kick", order=("A", "V"), stiff={"A": 0.15, "V": 0.3},
            support={"A": FEET, "V": ["sole_r", "heel_r"]}, balance=("A", "V"),
            lock={"V": ("leftUpperLeg", "leftLowerLeg")},
            posture={"A": dict(STAND_A, **GUARD_HANDS), "V": kick_v},
            root={"A": dict(pos=[0, None, 0.0], yaw=0, pos_w=0.5, hold=1.0),
                  "V": dict(pos=[0.08, None, 1.10], yaw=180, pos_w=0.5, hold=1.0, stance=0.0)}),
        Key(t=0.7, name="catch", order=("V", "A"), stiff={"A": 0.15, "V": 0.3}, touches=caught,
            support={"A": FEET, "V": ["sole_r", "heel_r"]}, balance=("A", "V"),
            lock={"V": ("leftUpperLeg", "leftLowerLeg")},
            posture={"A": dict(STAND_A, spine_flex=15, neck_flex=15), "V": dict(kick_v, spine_flex=5, leftUpperLeg_flex=80)},
            root={"A": dict(pos=[0, None, 0.1], yaw=0, pos_w=0.5, hold=1.0),
                  "V": dict(pos=[0.08, None, 1.05], yaw=180, pos_w=0.5, hold=1.0, stance=0.0)}),
        Key(t=1.1, name="drive", order=("V", "A"), stiff={"A": 0.15, "V": 0.3}, touches=caught,
            support={"A": FEET}, balance=("A",),
            lock={"V": ("leftUpperLeg", "leftLowerLeg")},
            posture={"A": dict(STAND_A, spine_flex=25, neck_flex=15),
                     "V": dict(ON_BACK, leftUpperLeg_flex=80, leftLowerLeg_flex=10, rightUpperLeg_flex=30, rightLowerLeg_flex=40)},
            root={"A": dict(pos=[0, None, 0.35], yaw=0, pos_w=0.5, hold=1.0),
                  "V": dict(pos=[0.05, 0.62, 1.1], yaw=180, pitch=-50, pos_w=1.5, hold=1.5)}),
        Key(t=1.6, name="down", order=("V", "A"), stiff={"A": 0.15, "V": 0.3}, touches=caught,
            support={"A": FEET, "V": LYING + ["heel_r"]}, balance=("A",),
            lock={"V": ("leftUpperLeg", "leftLowerLeg")},
            posture={"A": dict(STAND_A, spine_flex=20, chest_flex=5, neck_flex=20),
                     "V": dict(ON_BACK, leftUpperLeg_flex=60, leftLowerLeg_flex=20, rightUpperLeg_flex=20, rightLowerLeg_flex=30)},
            root={"A": dict(pos=[0, None, 0.38], yaw=0, pos_w=0.5, hold=1.0),
                  "V": dict(pos=[0, 0.12, 1.0], yaw=180, pitch=-90, pos_w=1.5, hold=1.5)}),
    ]
    sit, fin, brk = leg_lock_guard()[3:]
    for k, t in ((sit, 2.6), (fin, 3.6), (brk, 4.2)):
        k.t = t
        keys.append(k)
    return keys
