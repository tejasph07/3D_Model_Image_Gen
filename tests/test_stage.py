"""Physical-plausibility tests for the 3D stage (no image model needed)."""
import numpy as np
import pytest

from animescene.actions import ACTIONS
from animescene.characters import Cast, design, parse_description
from animescene.rig import Body, PoseParams, Rig
from animescene.scene import stage_scene
from animescene.stage import _seg_dist

BONES = [("l_shoulder", "l_elbow", "upper_arm"), ("l_elbow", "l_wrist", "forearm"),
         ("r_shoulder", "r_elbow", "upper_arm"), ("r_elbow", "r_wrist", "forearm"),
         ("l_hip", "l_knee", "thigh"), ("l_knee", "l_ankle", "shin"),
         ("r_hip", "r_knee", "thigh"), ("r_knee", "r_ankle", "shin")]


def duel(action, weapon="katana", target_action="stand", size=(256, 256), **params):
    spec = {"camera": {"shot": "wide", "azimuth": 20},
            "characters": [
                {"name": "T", "gender": "male", "archetype": "knight", "weapon": weapon, "action": target_action,
                 "target": "A", "at": [0, 0]},
                {"name": "A", "gender": "female", "archetype": "swordsman", "weapon": weapon, "action": action,
                 "target": "T", **params}]}
    return stage_scene(spec, Cast(), size=size)


def bone_errors(actor):
    b = actor.char.body
    J = actor.joints
    return [abs(np.linalg.norm(J[p] - J[q]) - getattr(b, L)) for p, q, L in BONES]


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_every_action_keeps_bone_lengths_and_stays_above_ground(action):
    s = duel(action)
    for a in s.scene.actors:
        assert max(bone_errors(a)) < 1e-6, (a.name, bone_errors(a))
        lowest = min(p[1] for p in a.joints.values())
        assert lowest > -0.02, f"{a.name} sinks below ground ({lowest:.3f})"


@pytest.mark.parametrize("action", ["stand", "stance", "punch", "slash", "block", "thrust", "cast"])
def test_grounded_actions_plant_their_feet(action):
    s = duel(action)
    a = s.scene.actor("A")
    b = a.char.body
    feet = [a.joints["l_ankle"][1] - b.ankle_h, a.joints["r_ankle"][1] - b.ankle_h]
    assert min(feet) == pytest.approx(0.0, abs=0.03)


@pytest.mark.parametrize("action", ["knocked_back", "flying_kick", "leap_attack"])
def test_airborne_actions_leave_the_ground(action):
    s = duel(action)
    a = s.scene.actor("A")
    b = a.char.body
    assert min(a.joints["l_ankle"][1], a.joints["r_ankle"][1]) - b.ankle_h > 0.05


def test_punch_lands_on_the_face():
    s = duel("punch", weapon="fists")
    a, t = s.scene.actor("A"), s.scene.actor("T")
    fist = (a.joints["r_wrist"] + a.joints["r_hand"]) / 2
    gap = np.linalg.norm(fist - t.joints["head"]) - t.char.body.head_r
    assert gap < 0.08, gap


def test_kick_reaches_the_head():
    s = duel("kick", weapon="fists")
    a, t = s.scene.actor("A"), s.scene.actor("T")
    d = np.linalg.norm(a.joints["r_ankle"] - t.joints["head"])
    assert d < 0.25, d
    # support foot planted
    assert a.joints["l_ankle"][1] - a.char.body.ankle_h == pytest.approx(0, abs=0.03)


def test_block_crosses_the_incoming_blade():
    s = duel("slash", target_action="block")
    att, blk = s.scene.actor("A"), s.scene.actor("T")
    _, g1, t1, _ = att.weapons[0]
    _, g2, t2, _ = blk.weapons[0]
    assert _seg_dist(g1, t1, g2, t2) < 0.06
    # the blades are crossed, not parallel
    d1, d2 = (t1 - g1) / np.linalg.norm(t1 - g1), (t2 - g2) / np.linalg.norm(t2 - g2)
    assert abs(d1 @ d2) < 0.7
    # and the attacker's blade stops before the defender's face
    assert _seg_dist(g1, t1, blk.joints["head"], blk.joints["head"]) > blk.char.body.head_r


def test_knocked_back_victim_is_where_the_blow_lands_and_flies_away():
    spec = {"camera": {"azimuth": 0},
            "characters": [
                {"name": "V", "gender": "male", "archetype": "brawler", "action": "knocked_back", "target": "K",
                 "at": [0, 0]},
                {"name": "K", "gender": "female", "archetype": "brawler", "action": "kick", "target": "V"}]}
    s = stage_scene(spec, Cast(), size=(256, 256))
    v, k = s.scene.actor("V"), s.scene.actor("K")
    foot = k.joints["r_ankle"]
    # a moment after impact: the struck head is just past the foot, and the bodies don't overlap
    assert min(np.linalg.norm(v.joints[p] - foot) for p in ("head", "neck", "chest")) < 0.8
    # the victim's torso leans away from the kicker
    away = v.joints["pelvis"] - k.joints["pelvis"]
    lean = v.joints["neck"] - v.joints["pelvis"]
    assert np.dot(away[[0, 2]], lean[[0, 2]]) > 0


def test_four_fighters_do_not_interpenetrate_and_are_framed():
    s = stage_scene("scenes/shrine_clash.json", Cast(), size=(608, 416))
    actors = s.scene.actors
    for i, a in enumerate(actors):
        for b in actors[i + 1:]:
            d = _seg_dist(a.joints["pelvis"], a.joints["neck"], b.joints["pelvis"], b.joints["neck"])
            assert d > a.char.body.r["chest"] + b.char.body.r["chest"] - 0.02, (a.name, b.name, d)
    uv = s.cam.project(s.scene.all_points())
    assert (uv[:, 0] >= 0).all() and (uv[:, 0] < 608).all()
    assert (uv[:, 1] >= 0).all() and (uv[:, 1] < 416).all()
    for i in range(len(actors)):
        assert (s.frame.ids == i).sum() > 200  # everyone is visible


def test_regions_partition_the_frame():
    s = stage_scene("scenes/shrine_clash.json", Cast(), size=(608, 416))
    total = s.masks.sum(0)
    assert total.max() <= 1.0 + 1e-3
    for i in range(len(s.scene.actors)):
        body = s.frame.ids == i
        assert s.masks[i][body].mean() > 0.9  # a character's own pixels belong to its region


def test_rig_ik_reaches_reachable_targets_exactly():
    rig = Rig(Body(1.7, "athletic", "male"))
    tgt = np.array([0.3, 1.2, 0.35])
    p = PoseParams()
    from animescene.rig import Limb
    p.hands["r"] = Limb(tgt, np.array([1.0, -1.0, 0]))
    J = rig.solve(p)
    assert np.linalg.norm(J["r_wrist"] - tgt) < 1e-6


def test_character_design_is_deterministic_and_palettes_differ():
    a = design("Kaze", seed=3, archetype="assassin")
    b = design("Kaze", seed=3, archetype="assassin")
    assert a == b
    cast = Cast()
    used = []
    for n in "ABCDE":
        c = cast.resolve({"name": n, "seed": 1}, used)
        used.append(c.palette)
    assert len(set(used)) == 5


def test_description_parsing():
    t = parse_description("a silver-haired swordswoman with red eyes wielding a spear, crimson... red cloak")
    assert t["gender"] == "female" and t["hair_color"] == "silver" and t["eyes"] == "red"
    assert t["weapon"] == "spear" and t["palette"] == "crimson"
    assert "weapon" not in parse_description("a swordswoman")


def test_cast_persists(tmp_path):
    p = tmp_path / "cast.json"
    cast = Cast(p)
    c = cast.resolve({"name": "Yui", "description": "pink-haired mage girl"}, [])
    cast.save()
    again = Cast(p).resolve({"name": "Yui"}, [])
    assert again == c and again.hair_color == "pink" and again.weapon == "magic"


def test_bad_specs_fail_loudly():
    with pytest.raises(ValueError):
        duel("moonwalk")
    with pytest.raises(ValueError):
        stage_scene({"characters": [{"name": "A", "action": "punch", "target": "Nobody"}]}, Cast())


def test_strikes_land_across_many_random_body_pairs():
    """Characters are designed on the fly, so contacts must hold for any pair of bodies."""
    kick, punch = [], []
    for sd in range(25):
        chars = [{"name": "T", "seed": sd, "action": "stand", "target": "A", "at": [0, 0]},
                 {"name": "A", "seed": sd + 100, "action": "kick", "target": "T"}]
        s = stage_scene({"characters": chars}, Cast(), size=(96, 96))
        a, t = s.scene.actor("A"), s.scene.actor("T")
        kick.append(np.linalg.norm(a.joints["r_ankle"] - t.joints["head"]))
        chars[1].update(action="punch", weapon="fists")
        s = stage_scene({"characters": chars}, Cast(), size=(96, 96))
        a, t = s.scene.actor("A"), s.scene.actor("T")
        fist = (a.joints["r_wrist"] + a.joints["r_hand"]) / 2
        punch.append(np.linalg.norm(fist - t.joints["head"]) - t.char.body.head_r)
    assert max(punch) < 0.06 and np.mean(punch) < 0.04
    assert max(kick) < 0.3 and np.mean(kick) < 0.1
