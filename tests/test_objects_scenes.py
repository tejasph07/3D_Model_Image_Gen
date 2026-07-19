"""Objects, props, riding, and full story-like scenes (the quidditch test)."""
import json
import os

import numpy as np
import pytest

from scene3d.render import render
from scene3d.camera import make_camera
from scene3d.openpose import scene_json
from scene3d.vecmath import norm, horizontal, V

from helpers import all_ok, build, bone_lengths_ok, above_ground_ok, contact_dist


def test_sit_on_chair():
    sc = build("create,a,HUMAN\ncreate,chair_1,CHAIR\nsit_on,a,chair_1")
    h = sc.entities["a"]
    c = sc.entities["chair_1"]
    seat_y = c.anchor("seat")[1]
    assert abs(h.joints["mid_hip"][1] - seat_y - 0.10) < 0.08
    # Thighs horizontal-ish, feet near the ground.
    assert h.joints["l_knee"][1] < h.joints["mid_hip"][1] + 0.1
    assert h.joints["l_ankle"][1] < 0.2
    bone_lengths_ok(h)


def test_stand_on_table():
    sc = build("create,a,HUMAN\ncreate,t,TABLE\nstand_on,a,t")
    h = sc.entities["a"]
    t = sc.entities["t"]
    assert h.joints["l_ankle"][1] > t.top_y_world() - 0.05
    assert h.joints["neck"][1] > 2.0
    bone_lengths_ok(h)


def test_ball_on_table():
    sc = build("create,t,TABLE\ncreate,b,BALL\non_top,b,t")
    b = sc.entities["b"]
    t = sc.entities["t"]
    assert abs(b.y - t.top_y_world()) < 0.02


def test_grab_door_knob():
    sc = build("create,a,HUMAN\ncreate,d,DOOR\ninfront,a,d\n"
               "grab,a.right_hand,d.knob")
    knob = sc.entities["d"].anchor("knob")
    wrist = sc.entities["a"].joints["r_wrist"]
    assert norm(wrist - knob) < 0.20
    all_ok(sc)


def test_hold_unplaced_ball_attaches_to_hand():
    sc = build("create,a,HUMAN\ncreate,ball_1,BALL\nhold,a.left_hand,ball_1")
    b = sc.entities["ball_1"]
    a = sc.entities["a"]
    center = b.anchor("center")
    wrist = a.joints["l_wrist"]
    assert norm(center - wrist) < 0.30
    assert center[1] > 0.4  # in the hand, not on the floor


def test_quidditch_seeker():
    """Flying broom rider 2 m up, one hand holding a ball. No training data
    for this - pure constraint math."""
    sc = build(
        "create,seeker,HUMAN\n"
        "create,broom,BROOM\n"
        "hover,broom,2.0\n"
        "ride,seeker,broom\n"
        "create,snitch,BALL\n"
        "scale,snitch,0.4\n"
        "hold,seeker.left_hand,snitch\n"
        "look,seeker,forward")
    s = sc.entities["seeker"]
    broom = sc.entities["broom"]
    snitch = sc.entities["snitch"]
    bone_lengths_ok(s)
    # Rider straddles the broom well above the ground.
    assert s.joints["mid_hip"][1] > 1.7
    assert s.joints["l_ankle"][1] > 1.0
    assert broom.y > 1.8
    # Legs on opposite sides of the broom axis.
    axis = broom.anchor("front") - broom.anchor("back")
    side = np.cross(axis / norm(axis), np.array([0, 1.0, 0]))
    lk = float((s.joints["l_knee"] - broom.anchor("middle")) @ side)
    rk = float((s.joints["r_knee"] - broom.anchor("middle")) @ side)
    assert lk * rk < 0, "knees must straddle the broom"
    # Right hand grips the shaft.
    grip = broom.anchor("middle") + axis / norm(axis) * 0.30
    assert norm(s.joints["r_wrist"] - grip) < 0.25
    # Snitch is in the left hand, airborne.
    assert snitch.anchor("center")[1] > 1.2
    above_ground_ok(sc)


def test_yoga_tree_pose():
    sc = build("create,a,HUMAN\npose,a,tree_pose")
    h = sc.entities["a"]
    bone_lengths_ok(h)
    # One foot planted, the other lifted to the standing knee/thigh.
    assert h.joints["l_ankle"][1] < 0.15
    assert h.joints["r_ankle"][1] > 0.35
    # Arms overhead.
    assert h.joints["l_wrist"][1] > h.joints["neck"][1]
    assert h.joints["r_wrist"][1] > h.joints["neck"][1]


def test_yoga_bridge():
    sc = build("create,a,HUMAN\npose,a,bridge")
    h = sc.entities["a"]
    bone_lengths_ok(h)
    above_ground_ok(sc)
    # Hips pushed up, hands and feet low.
    assert h.joints["mid_hip"][1] > 0.35
    assert h.joints["l_wrist"][1] < 0.30
    assert h.joints["l_ankle"][1] < 0.30


def test_weird_yoga_combo():
    """Warrior pose + head turned + one arm re-aimed: composed constraints."""
    sc = build("create,a,HUMAN\npose,a,warrior\npoint,a.left_arm,up\n"
               "look,a,right")
    h = sc.entities["a"]
    bone_lengths_ok(h)
    assert h.joints["l_wrist"][1] > h.joints["neck"][1]  # left arm now up
    assert h.joints["r_wrist"][1] < h.joints["neck"][1]  # right arm still out


def test_under_table_auto_lies_down():
    sc = build("create,a,HUMAN\ncreate,t,TABLE\nunder,a,t")
    h = sc.entities["a"]
    assert h.joints["neck"][1] < 0.75  # fits under the tabletop


def test_between_and_beside():
    sc = build("create,a,HUMAN\ncreate,b,HUMAN\ncreate,c,HUMAN\n"
               "at,a,-1:0\nat,b,1:0\nbetween,c,a:b")
    c = sc.entities["c"]
    assert abs(c.x) < 0.35 and abs(c.z) < 0.35


def test_json_output_format(tmp_path):
    sc = build("create,a,HUMAN\ncreate,chair_1,CHAIR\nsit_on,a,chair_1")
    cam = make_camera(sc)
    data = scene_json(sc, cam)
    assert data["canvas"] == [768, 768]
    person = data["people"][0]
    assert len(person["pose_keypoints_2d"]) == 25 * 3
    assert len(person["hand_left_keypoints_2d"]) == 21 * 3
    assert len(person["hand_right_keypoints_2d"]) == 21 * 3
    assert len(person["joints_3d"]) == 25
    assert data["objects"][0]["type"] == "CHAIR"
    # Round-trips as JSON.
    json.dumps(data)


def test_render_smoke(tmp_path):
    sc = build("create,a,HUMAN\ncreate,b,HUMAN\npiggyback,a,b\n"
               "camera,three_quarter,scene")
    img, cam = render(sc, 512, 384)
    assert img.size == (512, 384)
    arr = np.asarray(img)
    assert arr.max() > 100  # something actually drawn
    p = tmp_path / "out.png"
    img.save(p)
    assert p.stat().st_size > 1000


def test_cli(tmp_path):
    scene_file = tmp_path / "s.scene"
    scene_file.write_text("create,a,HUMAN\ncreate,c,CHAIR\nsit_on,a,c\n")
    from scene3d.cli import main
    rc = main([str(scene_file), "-o", str(tmp_path), "--width", "300",
               "--height", "300"])
    assert rc == 0
    assert (tmp_path / "s.png").exists()
    assert (tmp_path / "s.json").exists()
