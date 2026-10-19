"""Strikes, holds and reactions written as contact keyframes (see keyframes.py).

Every move is written for "A" (at the origin, facing +z) and "V" (facing her, -z); `place_keys` re-casts it
onto named characters anywhere. Distances are between the two hips.
"""
from __future__ import annotations

import numpy as np

from .keyframes import BLADE_AXIS as BLADE, UP, Key, Touch, clamp, grab, pinch, press, punch, squeeze

FEET = ["sole_l", "sole_r", "heel_l", "heel_r"]
GUARD = dict(leftUpperArm_raise=-55, rightUpperArm_raise=-55, leftUpperArm_flex=45, rightUpperArm_flex=45,
             leftLowerArm_flex=110, rightLowerArm_flex=110)
STAND = dict(leftUpperLeg_flex=18, rightUpperLeg_flex=18, leftLowerLeg_flex=28, rightLowerLeg_flex=28,
             leftFoot_point=-10, rightFoot_point=-10, spine_flex=6)
FIGHT = dict(STAND, **GUARD)
FWD = np.array([0.0, 0.0, 1.0])  # her forward; his forward is -z


def _std(name, t, d=1.0, a=None, v=None, touches=(), a_pos=(0.0, 0.0), v_pos=None, a_yaw=0.0, v_yaw=180.0,
         a_support=FEET, v_support=FEET, order=("V", "A"), lock=None, stiff=None, a_pitch=0.0, v_pitch=0.0, **kw):
    """A keyframe with both standing (unless told otherwise), A at a_pos (x, z), V at v_pos or d ahead of her."""
    vp = v_pos if v_pos is not None else (0.0, d)
    bal = tuple(x for x, sup in (("A", a_support), ("V", v_support)) if sup == FEET or (sup and "sole_l" in sup and "sole_r" in sup))
    return Key(t=t, name=name, touches=list(touches), order=order,
               support={"A": list(a_support), "V": list(v_support)}, balance=kw.pop("balance", bal),
               posture={"A": dict(FIGHT if a is None else a), "V": dict(FIGHT if v is None else v)},
               lock=lock or {}, stiff=stiff or {"A": 0.15, "V": 0.15},
               root={"A": dict(pos=[a_pos[0], None, a_pos[1]], yaw=a_yaw, pitch=a_pitch, pos_w=0.5, hold=1.0),
                     "V": dict(pos=[vp[0], None, vp[1]], yaw=v_yaw, pitch=v_pitch, pos_w=0.5, hold=1.0)}, **kw)


# ----------------------------------------------------------------------------- reactions (his poses)
DOUBLED = dict(STAND, spine_flex=40, chest_flex=20, upperChest_flex=10, neck_flex=-15, leftUpperLeg_flex=35, rightUpperLeg_flex=35,
               leftLowerLeg_flex=45, rightLowerLeg_flex=45, leftUpperArm_raise=-70, rightUpperArm_raise=-70, leftUpperArm_flex=30,
               rightUpperArm_flex=30, leftLowerArm_flex=100, rightLowerArm_flex=100)
HEAD_BACK = dict(STAND, spine_flex=-15, chest_flex=-10, neck_flex=-35, head_flex=-15, leftUpperArm_raise=-30, rightUpperArm_raise=-30,
                 leftLowerArm_flex=40, rightLowerArm_flex=40, leftUpperLeg_flex=5, rightUpperLeg_flex=25, rightLowerLeg_flex=35)
FLINCH = dict(STAND, spine_flex=20, chest_flex=10, neck_flex=10, leftUpperArm_raise=-60, rightUpperArm_raise=-60, leftUpperArm_flex=25,
              rightUpperArm_flex=25, leftLowerArm_flex=60, rightLowerArm_flex=60)
TURNED = dict(STAND, spine_twist=20, chest_twist=15, neck_twist=40, spine_bend=-15, neck_bend=-20, leftUpperArm_raise=-45,
              rightUpperArm_raise=-60, leftLowerArm_flex=50, rightLowerArm_flex=30)
REEL_SIDE = dict(STAND, spine_bend=20, chest_bend=12, neck_bend=15, leftUpperArm_raise=-20, rightUpperArm_raise=-70,
                 leftLowerArm_flex=30, rightLowerArm_flex=60, rightUpperLeg_flex=30, rightLowerLeg_flex=45)


# ----------------------------------------------------------------------------- strikes
def ready(d=1.15):
    return [_std("ready", 0.0, d)]


def stomach_punch(d=1.1):
    """Right fist driven into his solar plexus; he folds over it, the fist still buried."""
    fist = punch("A", "r", "V", "solarplexus", depth=0.035)
    a_punch = dict(STAND, spine_flex=12, leftUpperLeg_flex=45, leftLowerLeg_flex=45, rightUpperLeg_flex=-5, rightLowerLeg_flex=15,
                   leftUpperArm_raise=-55, leftUpperArm_flex=60, leftLowerArm_flex=120, spine_twist=15, chest_twist=10)
    return [_std("ready", 0.0, d),
            _std("punch", 0.25, d, a=a_punch, v=FLINCH, touches=[fist], a_pos=(0.0, 0.25)),
            _std("doubled over", 0.55, d, a=a_punch, v=DOUBLED, touches=[fist], a_pos=(0.0, 0.32), v_pos=(0.0, 0.95))]


def palm_strike(d=1.1):
    """Heel of the right palm up under his chin; his head snaps back."""
    hit = press("A", "r", "V", "chin", depth=0.02, normal=np.array([0, -0.6, 0.8]))
    a_strike = dict(STAND, spine_flex=5, leftUpperLeg_flex=40, leftLowerLeg_flex=40, rightUpperLeg_flex=-5, rightLowerLeg_flex=10,
                    leftUpperArm_raise=-55, leftUpperArm_flex=50, leftLowerArm_flex=110, chest_twist=12)
    return [_std("ready", 0.0, d),
            _std("palm strike", 0.22, d, a=a_strike, v=HEAD_BACK, touches=[hit], a_pos=(0.0, 0.3))]


def press_stomach(d=0.75):
    """Both open palms pressed flat into his stomach; he flinches back over them."""
    return [_std("ready", 0.0, d + 0.2),
            _std("press", 0.5, d, a=dict(STAND, spine_flex=15), v=FLINCH,
                 touches=[press("A", "l", "V", "belly", depth=0.03), press("A", "r", "V", "belly", depth=0.03)])]


def ear_pull(d=0.6):
    """She pinches his left ear between thumb and fingers and pulls up and toward her; his head follows it,
    he rises on his toes, wincing, hands at her wrist."""
    v = dict(STAND, neck_bend=-25, head_bend=-10, spine_bend=-8, leftFoot_point=25, rightFoot_point=25, leftUpperLeg_flex=5,
             rightUpperLeg_flex=5, leftLowerLeg_flex=10, rightLowerLeg_flex=10)
    return [_std("ready", 0.0, d + 0.3),
            _std("ear pull", 0.5, d, a=dict(STAND, spine_flex=0), v=v, order=("A", "V"),
                 touches=[pinch("A", "r", "V", "ear_l"), grab("V", "l", "A", "wristblade_r", weight=0.6),
                          grab("V", "r", "A", "forearm_r", weight=0.4)])]


def front_headlock(d=0.55):
    """He is bent over in front of her; her right forearm under his throat, his head trapped under her
    right armpit, left hand clasping her right wrist."""
    v = dict(DOUBLED, spine_flex=45, chest_flex=25, leftUpperLeg_flex=55, rightUpperLeg_flex=55, neck_flex=-25)
    return [_std("ready", 0.0, d + 0.4),
            _std("front headlock", 0.6, d, a=dict(STAND, spine_flex=20, leftUpperLeg_flex=30, rightUpperLeg_flex=30), v=v,
                 touches=[clamp("A", "r", "V", "throat", normal="surface"), Touch("A", "armpit_r", "V", "nape", "tuck", "surface", 0.02),
                          grab("A", "l", "A", "wristblade_r")])]


def rear_choke():
    """From behind: right forearm across his throat, right hand on her left forearm, left palm behind his
    head, her chest on his back. He claws at the arm."""
    v = dict(STAND, spine_flex=-10, neck_flex=-15, leftUpperArm_raise=-40, rightUpperArm_raise=-40, leftUpperArm_flex=70,
             rightUpperArm_flex=70, leftLowerArm_flex=120, rightLowerArm_flex=120, leftFoot_point=10, rightFoot_point=10)
    return [_std("rear choke", 0.0, a=dict(STAND, spine_flex=5), v=v, v_pos=(0.0, 0.32), v_yaw=0.0, order=("V", "A", "V"),
                 touches=[clamp("A", "r", "V", "throat", normal="surface"), grab("A", "r", "A", "forearm_l", weight=0.6),
                          press("A", "l", "V", "nape", depth=0.0, weight=0.6), squeeze("A", "sternum", "V", "upperback", depth=0.01, weight=0.5),
                          grab("V", "l", "A", "forearm_r", weight=0.4), grab("V", "r", "A", "wristblade_r", weight=0.4)])]


def knee_strike(d=0.6):
    """Both hands on the back of his head pull it down into her rising right knee."""
    v = dict(DOUBLED, spine_flex=45, neck_flex=-20)
    a = dict(STAND, rightUpperLeg_flex=100, rightLowerLeg_flex=110, rightFoot_point=30, spine_flex=10)
    return [_std("clinch", 0.0, d, a=dict(STAND, spine_flex=10), v=dict(STAND, spine_flex=25),
                 touches=[grab("A", "l", "V", "temple_r", weight=0.7), grab("A", "r", "V", "temple_l", weight=0.7)]),
            _std("knee", 0.3, d, a=a, v=v, a_support=["sole_l", "heel_l"],
                 touches=[grab("A", "l", "V", "temple_r", weight=0.7), grab("A", "r", "V", "temple_l", weight=0.7),
                          Touch("A", "knee_r", "V", "solarplexus", "knee", "surface", -0.03)])]


def hook(d=0.85):
    """Left hook to his right jaw; his head and shoulders turn away with it."""
    a = dict(STAND, spine_twist=-25, chest_twist=-15, leftUpperArm_raise=10, leftUpperArm_flex=80, leftLowerArm_flex=95,
             rightUpperArm_raise=-55, rightUpperArm_flex=50, rightLowerArm_flex=115, leftUpperLeg_flex=30, leftLowerLeg_flex=35)
    return [_std("ready", 0.0, d + 0.2),
            _std("hook", 0.22, d, a=a, v=TURNED, touches=[punch("A", "l", "V", "jaw_r", depth=0.02)])]


def uppercut(d=0.7):
    """Right uppercut under his chin from a crouch; he goes up on his toes, head thrown back."""
    a = dict(STAND, leftUpperLeg_flex=35, leftLowerLeg_flex=45, rightUpperLeg_flex=10, rightLowerLeg_flex=20, spine_flex=-5,
             spine_twist=15, rightUpperArm_flex=110, rightLowerArm_flex=80, rightUpperArm_raise=-40,
             leftUpperArm_raise=-55, leftUpperArm_flex=50, leftLowerArm_flex=115)
    v = dict(HEAD_BACK, leftFoot_point=25, rightFoot_point=25)
    return [_std("crouch", 0.0, d + 0.1, a=dict(STAND, leftUpperLeg_flex=45, rightUpperLeg_flex=45, leftLowerLeg_flex=60,
                                                   rightLowerLeg_flex=60, spine_flex=20, **GUARD)),
            _std("uppercut", 0.2, d, a=a, v=v, touches=[punch("A", "r", "V", "chin", depth=0.02, normal=np.array([0, -1.0, 0.2]))])]


def roundhouse(d=1.15):
    """Right roundhouse: her shin into his left ribs, standing on the left foot, turned over; he reels sideways."""
    a = dict(STAND, rightUpperLeg_flex=80, rightUpperLeg_abduct=55, rightLowerLeg_flex=15, rightFoot_point=30, spine_bend=25,
             leftUpperArm_raise=-40, leftUpperArm_flex=50, leftLowerArm_flex=110, rightUpperArm_raise=-70, rightUpperArm_flex=-30)
    return [_std("ready", 0.0, d),
            _std("roundhouse", 0.3, d, a=a, v=REEL_SIDE, a_support=["sole_l", "heel_l"], a_yaw=-35,
                 touches=[Touch("A", "shin_r", "V", "ribs_l", "kick", "surface", -0.02)])]


def spinning_back_kick(d=1.1):
    """Turned away, her right heel thrust straight back into his stomach; she looks over her shoulder."""
    a = dict(STAND, rightUpperLeg_flex=-25, rightLowerLeg_flex=10, rightFoot_point=-20, spine_flex=40, neck_flex=-40, neck_twist=40,
             leftUpperLeg_flex=40, leftLowerLeg_flex=35, leftUpperArm_raise=-40, rightUpperArm_raise=-40)
    return [_std("ready", 0.0, d),
            _std("spin", 0.25, d, a=dict(FIGHT, spine_flex=20), a_yaw=-120),
            _std("back kick", 0.5, d, a=a, v=DOUBLED, a_support=["sole_l", "heel_l"], a_yaw=180,
                 touches=[Touch("A", "heel_r", "V", "belly", "kick", "surface", -0.03)])]


def back_knee_kick():
    """Behind and to his side: her right sole stamps the back of his left knee; it buckles."""
    v = dict(STAND, leftUpperLeg_flex=40, leftLowerLeg_flex=90, leftFoot_point=30, spine_flex=-10, neck_flex=-20,
             leftUpperArm_raise=-20, rightUpperArm_raise=-30)
    a = dict(STAND, rightUpperLeg_flex=60, rightLowerLeg_flex=30, rightFoot_point=-10, spine_flex=-5)
    return [_std("behind", 0.0, a=FIGHT, v=FIGHT, v_pos=(0.15, 0.55), v_yaw=0.0),
            _std("stamp", 0.3, a=a, v=v, v_pos=(0.15, 0.55), v_yaw=0.0, a_support=["sole_l", "heel_l"],
                 v_support=["sole_r", "heel_r"], touches=[Touch("A", "sole_r", "V", "backknee_l", "stamp", "surface", -0.02)])]


def knee_buckle():
    """He drops to his knees."""
    v = dict(STAND, leftUpperLeg_flex=10, rightUpperLeg_flex=10, leftLowerLeg_flex=120, rightLowerLeg_flex=120, spine_flex=25,
             neck_flex=20, leftFoot_point=40, rightFoot_point=40, leftUpperArm_raise=-75, rightUpperArm_raise=-75)
    return [_std("buckle", 0.0, a=FIGHT, v=v, v_support=["knee_l", "knee_r", "toes_l", "toes_r"], d=0.9)]


def stumble(d=1.4):
    """He staggers back: one foot behind, arms out, trunk behind his hips."""
    v = dict(STAND, spine_flex=-20, neck_flex=15, leftUpperLeg_flex=-15, rightUpperLeg_flex=35, rightLowerLeg_flex=30,
             leftUpperArm_raise=-10, rightUpperArm_raise=-10, leftLowerArm_flex=30, rightLowerArm_flex=30)
    return [_std("stumble", 0.0, d, a=FIGHT, v=v)]


MOVES = {f.__name__: f for f in (ready, stomach_punch, palm_strike, press_stomach, ear_pull, front_headlock, rear_choke,
                                 knee_strike, hook, uppercut, roundhouse, spinning_back_kick, back_knee_kick, knee_buckle, stumble)}
