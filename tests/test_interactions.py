"""Two-human interactions, carries, and the poses image models fail at."""
import numpy as np
import pytest

from scene3d.vecmath import norm, horizontal, V

from helpers import all_ok, build, bone_lengths_ok, above_ground_ok, contact_dist


def test_punch_to_jaw():
    sc = build("create,hero,HUMAN\ncreate,thug,HUMAN\n"
               "infront,hero,thug\npunch,hero.right_fist,thug.jaw")
    hero = sc.entities["hero"]
    thug = sc.entities["thug"]
    for h in (hero, thug):
        bone_lengths_ok(h)
    d = contact_dist(sc, "hero", "right_fist", "thug", "jaw")
    assert d < 0.08, f"fist-jaw gap {d:.3f}"
    # Fist finger pose: all fingertips near the wrist.
    hand = hero.hand_pts["R"]
    for tip in (8, 12, 16, 20):
        assert norm(hand[tip] - hand[0]) < 0.11
    above_ground_ok(sc)


def test_high_kick_to_head():
    """A head-height kick - classic image-model failure."""
    sc = build("create,fighter,HUMAN\ncreate,victim,HUMAN\n"
               "infront,fighter,victim\nkick,fighter.right_foot,victim.head")
    f = sc.entities["fighter"]
    v = sc.entities["victim"]
    bone_lengths_ok(f)
    bone_lengths_ok(v)
    head_y = v.frames["head_center"][1]
    foot_y = f.joints["r_ankle"][1]
    assert foot_y > head_y - 0.45, f"kick foot at {foot_y:.2f}, head {head_y:.2f}"
    d = contact_dist(sc, "fighter", "right_toes", "victim", "head")
    assert d < 0.25
    # Support leg stays planted.
    assert f.joints["l_ankle"][1] < 0.15
    # Torso counter-leans back.
    assert f.params["spine_pitch"] < -5


def test_reach_far_target_steps_in():
    """Actor placed far away must walk/bend into range instead of stretching."""
    sc = build("create,a,HUMAN\ncreate,b,HUMAN\n"
               "at,a,0:0\nat,b,0:2.5\nface,a,b\nface,b,a\n"
               "touch,a.right_palm,b.left_shoulder")
    a = sc.entities["a"]
    d = contact_dist(sc, "a", "right_palm", "b", "left_shoulder")
    assert d < 0.08, f"palm-shoulder gap {d:.3f}"
    # a must have moved from its scripted spot.
    assert abs(a.z - 0.0) > 0.5
    all_ok(sc)


def test_touch_other_foot_means_low_reach():
    sc = build("create,a,HUMAN\ncreate,b,HUMAN\ninfront,a,b\n"
               "touch,a.right_palm,b.left_foot")
    a = sc.entities["a"]
    d = contact_dist(sc, "a", "right_palm", "b", "left_foot")
    assert d < 0.10, f"gap {d:.3f}"
    # Reaching someone's foot from standing requires major fold or crouch.
    assert a.joints["neck"][1] < 1.25
    bone_lengths_ok(a)


def test_piggyback():
    sc = build("create,carrier,HUMAN\ncreate,rider,HUMAN\n"
               "piggyback,carrier,rider")
    c = sc.entities["carrier"]
    r = sc.entities["rider"]
    bone_lengths_ok(c)
    bone_lengths_ok(r)
    # Rider is up on the carrier's back: hips high, feet off the ground.
    assert r.joints["mid_hip"][1] > 0.85
    assert r.joints["l_ankle"][1] > 0.25
    assert r.joints["r_ankle"][1] > 0.25
    # Rider is behind the carrier w.r.t. carrier facing.
    rel = np.array([r.x - c.x, 0, r.z - c.z])
    fwd = c.forward()
    assert float(rel @ fwd) < -0.1
    # Same facing.
    assert abs(((r.yaw - c.yaw) + 180) % 360 - 180) < 15
    above_ground_ok(sc)


def test_shoulder_sit():
    sc = build("create,dad,HUMAN\ncreate,kid,CHILD\nshoulder_sit,dad,kid")
    dad = sc.entities["dad"]
    kid = sc.entities["kid"]
    assert kid.joints["mid_hip"][1] > dad.joints["neck"][1] - 0.12
    assert kid.joints["l_ankle"][1] > 0.6
    bone_lengths_ok(dad)
    bone_lengths_ok(kid)
    above_ground_ok(sc)


def test_fireman_carry():
    sc = build("create,rescuer,HUMAN\ncreate,victim,HUMAN\n"
               "fireman_carry,rescuer,victim")
    res = sc.entities["rescuer"]
    vic = sc.entities["victim"]
    bone_lengths_ok(res)
    bone_lengths_ok(vic)
    # Victim draped at shoulder height, roughly horizontal torso.
    assert vic.joints["mid_hip"][1] > 1.15
    torso = vic.joints["neck"] - vic.joints["mid_hip"]
    assert abs(torso[1]) < 0.35 * norm(torso) + 0.15
    # Victim's body axis crosses the rescuer's facing (perpendicular-ish).
    dyaw = abs(((vic.yaw - res.yaw) + 180) % 360 - 180)
    assert 55 < dyaw < 125
    # Feet off the ground.
    assert vic.joints["l_ankle"][1] > 0.5
    above_ground_ok(sc)


def test_cradle_carry():
    sc = build("create,hero,HUMAN\ncreate,girl,HUMAN\nscale,girl,0.9\n"
               "cradle_carry,hero,girl")
    hero = sc.entities["hero"]
    girl = sc.entities["girl"]
    assert girl.joints["mid_hip"][1] > 0.8
    assert girl.joints["l_ankle"][1] > 0.35
    # Carried body is horizontal (face up).
    assert girl.params["pitch"] < -45
    bone_lengths_ok(hero)
    bone_lengths_ok(girl)


def test_defeat_scene():
    sc = build("create,villain,HUMAN\ncreate,hero,HUMAN\n"
               "pose,villain,sprawl\nnear,hero,villain\n"
               "look_at,hero,villain\nlook,villain,up")
    v = sc.entities["villain"]
    h = sc.entities["hero"]
    assert v.joints["mid_hip"][1] < 0.35          # down on the ground
    assert h.joints["mid_hip"][1] > 0.8           # standing over them
    # Hero's head pitched down toward the fallen villain.
    assert h.params["head_pitch"] > 10
    all_ok(sc)


def test_back_to_back():
    sc = build("create,a,HUMAN\ncreate,b,HUMAN\n"
               "at,a,0:0\nat,b,0:0.6\nface_away,a,b\nface_away,b,a")
    a, b = sc.entities["a"], sc.entities["b"]
    # Facing opposite directions.
    assert abs(((a.yaw - b.yaw) + 180) % 360 - 180) > 150
    # Collision resolution keeps them from overlapping.
    d = norm(horizontal(V(a.x - b.x, 0, a.z - b.z)))
    assert d > 0.28
    all_ok(sc)


def test_hug():
    sc = build("create,a,HUMAN\ncreate,b,HUMAN\nhug,a,b")
    a, b = sc.entities["a"], sc.entities["b"]
    # Bodies close, facing each other.
    d = norm(horizontal(V(a.x - b.x, 0, a.z - b.z)))
    assert d < 0.55
    assert abs(((a.yaw - b.yaw) + 180) % 360 - 180) > 150
    # Hands reach around toward the partner's back.
    gap = contact_dist(sc, "a", "right_palm", "b", "upper_back")
    assert gap < 0.16
    bone_lengths_ok(a)
    bone_lengths_ok(b)


def test_handshake():
    sc = build("create,a,HUMAN\ncreate,b,HUMAN\nhandshake,a,b")
    a, b = sc.entities["a"], sc.entities["b"]
    d = norm(a.joints["r_wrist"] - b.joints["r_wrist"])
    assert d < 0.15, f"wrists {d:.2f} apart"
    h = a.joints["r_wrist"][1]
    assert 0.7 < h < 1.3
    all_ok(sc)


def test_choke_lift_scene():
    """Villain grabs hero by the throat - both hands on the neck."""
    sc = build("create,villain,HUMAN\ncreate,hero,HUMAN\ninfront,villain,hero\n"
               "choke,villain.hands,hero")
    v = sc.entities["villain"]
    d1 = contact_dist(sc, "villain", "right_palm", "hero", "neck")
    d2 = contact_dist(sc, "villain", "left_palm", "hero", "neck")
    assert d1 < 0.14 and d2 < 0.14
    bone_lengths_ok(v)
