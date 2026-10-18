"""Close-combat geometry: locks, strikes from behind, gut shots and their reactions."""
import numpy as np
import pytest

from animescene.characters import Cast
from animescene.scene import stage_scene


def pair(att, vic, ap=None, vp=None, seeds=(6, 5)):
    v = {"name": "V", "gender": "male", "archetype": "brawler", "seed": seeds[1], "action": vic, "target": "A",
         "at": [0.0, 0.0], **(vp or {})}
    a = {"name": "A", "gender": "female", "archetype": "brawler", "seed": seeds[0], "action": att, "target": "V",
         **(ap or {})}
    s = stage_scene({"camera": {"azimuth": 20}, "characters": [v, a]}, Cast(), size=(160, 120))
    return s.scene.actor("A"), s.scene.actor("V")


def fwd(actor):
    return np.array([np.sin(actor.yaw), 0, np.cos(actor.yaw)])


@pytest.mark.parametrize("seeds", [(6, 5), (11, 12), (21, 3)])
def test_headlock_traps_the_head_under_the_arm(seeds):
    a, v = pair("headlock", "headlocked", seeds=seeds)
    # forearm under the throat, head next to the attacker's chest
    assert np.linalg.norm(a.joints["r_wrist"] - v.joints["neck"]) < 0.2
    assert np.linalg.norm(v.joints["head"] - a.joints["chest"]) < 0.4
    # the victim is bent over: head well in front of the pelvis
    assert np.dot(v.joints["head"] - v.joints["pelvis"], fwd(v)) > 0.25
    # they face each other
    assert np.dot(fwd(a), fwd(v)) < -0.8


def test_rear_choke_is_from_behind_with_the_arm_across_the_throat():
    a, v = pair("rear_choke", "choked")
    assert np.dot(a.origin - v.origin, fwd(v)) < -0.15  # attacker behind the victim
    assert np.dot(fwd(a), fwd(v)) > 0.95  # same facing
    assert np.linalg.norm(a.joints["r_wrist"] - v.joints["neck"]) < 0.2
    # victim's hands at its own throat, pulling at the arm
    assert np.linalg.norm(v.joints["r_wrist"] - v.joints["neck"]) < 0.25


def test_leg_lock_on_the_ground():
    a, v = pair("leg_lock", "leg_locked")
    assert a.joints["pelvis"][1] < 0.25 and v.joints["pelvis"][1] < 0.25  # both on the ground
    ankle = v.joints["r_ankle"]
    assert np.linalg.norm(a.joints["r_wrist"] - ankle) < 0.2
    assert ankle[1] > 0.35  # the trapped leg is held up
    assert np.linalg.norm(a.joints["chest"] - ankle) < 0.45  # hugged to the chest


def test_back_knee_kick_comes_from_behind_and_hits_the_hollow_of_the_knee():
    a, v = pair("back_knee_kick", "knee_buckle", vp={"facing": 100})
    assert np.dot(a.origin - v.origin, fwd(v)) < -0.2  # attacker behind
    # the kicking foot is at the back of the (proxy) knee: the victim then buckles
    assert a.joints["r_ankle"][1] < 0.7
    assert v.joints["l_knee"][1] < v.joints["r_knee"][1]  # the struck knee drops lower


def test_spinning_back_kick_turns_its_back_and_reaches_the_neck():
    a, v = pair("spinning_back_kick", "staggered")
    to_v = v.origin - a.origin
    assert np.dot(fwd(a), to_v) < 0  # back turned to the target
    assert np.linalg.norm(a.joints["r_ankle"] - v.joints["neck"]) < 0.45
    assert a.joints["r_ankle"][1] > 1.0  # it's a high kick
    gap = np.hypot(*(a.joints["pelvis"] - v.joints["pelvis"])[[0, 2]])
    assert gap > 0.35  # bodies don't overlap


def test_palm_strike_lands_on_the_stomach_and_folds_the_victim():
    a, v = pair("palm_strike", "doubled_over")
    palm = (a.joints["r_wrist"] + a.joints["r_hand"]) / 2
    d = np.linalg.norm(palm - v.joints["spine"]) - v.char.body.r["waist"]
    assert d < 0.15, d
    assert np.dot(v.joints["neck"] - v.joints["pelvis"], fwd(v)) > 0.15  # folded forward
    assert a.out.effect  # carries an energy effect region


def test_knee_strike_drives_the_knee_into_the_stomach():
    a, v = pair("knee_strike", "doubled_over")
    assert np.linalg.norm(a.joints["r_knee"] - v.joints["spine"]) < 0.3
    assert np.linalg.norm(a.joints["r_wrist"] - v.joints["head"]) < 0.25  # hands on the head


def test_clutching_hands_are_on_the_stomach():
    a, v = pair("kick", "clutch_stomach", ap={"aim": "stomach"})
    for s in "lr":
        assert np.linalg.norm(v.joints[f"{s}_wrist"] - v.joints["spine"]) < 0.3
