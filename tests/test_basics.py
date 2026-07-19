"""Parser, presets, primitives, and single-human contact semantics."""
import numpy as np
import pytest

from scene3d.parser import parse, ParseError
from scene3d.skeleton import Human, POSE_PRESETS, STAND_PELVIS_Y
from scene3d.solver import SceneError
from scene3d.vecmath import norm, two_bone_ik, V

from helpers import all_ok, build, bone_lengths_ok, contact_dist


def test_parse_basic():
    cmds = parse("create,human_a,HUMAN\n# comment\n\ntouch,a.right_palm,b.forehead")
    assert len(cmds) == 2
    assert cmds[0].verb == "create"
    assert cmds[1].a.entity == "a" and cmds[1].a.part == "right_palm"


def test_parse_errors():
    with pytest.raises(ParseError):
        parse("create,human_a")           # 2 tokens
    with pytest.raises(ParseError):
        parse("create,human_a,HUMAN,extra")


def test_unknown_verb_and_entity():
    with pytest.raises(SceneError, match="unknown verb"):
        build("create,a,HUMAN\nfrobnicate,a,a")
    with pytest.raises(SceneError, match="unknown entity"):
        build("create,a,HUMAN\ntouch,a.right_palm,ghost.forehead")
    with pytest.raises(ValueError, match="unknown object type"):
        build("create,x,UNICORN")
    with pytest.raises(ValueError, match="unknown pose"):
        build("create,a,HUMAN\npose,a,levitate")


def test_two_bone_ik_exact():
    rng = np.random.default_rng(0)
    for _ in range(200):
        root = rng.normal(size=3)
        target = root + rng.normal(size=3) * 0.4
        mid, end, reached = two_bone_ik(root, target, 0.28, 0.26,
                                        rng.normal(size=3))
        assert abs(norm(mid - root) - 0.28) < 1e-6
        assert abs(norm(end - mid) - 0.26) < 1e-6
        if norm(target - root) <= 0.54:
            assert reached and norm(end - target) < 1e-6


@pytest.mark.parametrize("pose", sorted(POSE_PRESETS))
def test_all_presets_physical(pose):
    sc = build(f"create,a,HUMAN\npose,a,{pose}")
    h = sc.entities["a"]
    bone_lengths_ok(h)
    # Nothing under the floor in any preset.
    from helpers import above_ground_ok
    above_ground_ok(sc)


def test_lie_back_is_horizontal():
    sc = build("create,a,HUMAN\npose,a,lie_back")
    h = sc.entities["a"]
    assert h.joints["neck"][1] < 0.45
    assert h.joints["mid_hip"][1] < 0.35
    # Nose points up when lying on the back.
    assert h.joints["nose"][1] > h.joints["l_ear"][1] - 0.05


def test_handstand_hands_down_feet_up():
    sc = build("create,a,HUMAN\npose,a,handstand")
    h = sc.entities["a"]
    assert h.joints["l_wrist"][1] < 0.12 and h.joints["r_wrist"][1] < 0.12
    assert h.joints["l_ankle"][1] > 1.3 and h.joints["r_ankle"][1] > 1.3
    assert h.joints["mid_hip"][1] > h.joints["neck"][1]


def test_self_touch_forehead():
    sc = build("create,a,HUMAN\ntouch,a.right_palm,a.forehead")
    h = sc.entities["a"]
    all_ok(sc)
    # Palm effector ends up within a few cm of the forehead point.
    d = contact_dist(sc, "a", "right_palm", "a", "forehead")
    assert d < 0.06, f"palm-forehead gap {d:.3f}"
    # Still standing: no reach bend was needed.
    assert h.params["spine_pitch"] < 15
    assert h.joints["mid_hip"][1] > 0.8


def test_poke_own_cheek_finger_shape():
    sc = build("create,a,HUMAN\npoke,a.left_fingers,a.right_cheek")
    h = sc.entities["a"]
    all_ok(sc)
    hand = h.hand_pts["L"]
    cheek, cr = h.part_point("right_cheek")
    index_tip = hand[8]
    assert norm(index_tip - cheek) < 0.09
    # Poke = index extended, others curled: index tip far from wrist,
    # ring/pinky tips close to it.
    wrist = hand[0]
    assert norm(index_tip - wrist) > 0.12
    assert norm(hand[16] - wrist) < 0.10
    assert norm(hand[20] - wrist) < 0.10


def test_standing_toe_touch_bends_hips():
    sc = build("create,a,HUMAN\ntouch,a.right_fingers,a.right_toes")
    h = sc.entities["a"]
    bone_lengths_ok(h)
    d = contact_dist(sc, "a", "right_fingers", "a", "right_toes")
    assert d < 0.12, f"fingers-toes gap {d:.3f}"
    # Must have folded forward substantially to get there.
    assert h.params["spine_pitch"] > 55
    assert h.joints["neck"][1] < 1.1


def test_lying_toe_touch_keeps_lying():
    sc = build("create,a,HUMAN\npose,a,sit\ntouch,a.right_fingers,a.right_toes")
    h = sc.entities["a"]
    bone_lengths_ok(h)
    assert h.joints["mid_hip"][1] < 0.35   # still on the ground
    d = contact_dist(sc, "a", "right_fingers", "a", "right_toes")
    assert d < 0.15


def test_scale_child():
    sc = build("create,kid,CHILD")
    h = sc.entities["kid"]
    assert h.scale < 0.8
    assert h.joints["neck"][1] < 1.1
    bone_lengths_ok(h)


def test_determinism():
    text = ("create,a,HUMAN\ncreate,b,HUMAN\ninfront,a,b\n"
            "punch,a.right_fist,b.jaw\nkick,b.right_foot,a.left_thigh")
    j1 = build(text).entities["a"].body25()
    j2 = build(text).entities["a"].body25()
    assert np.allclose(j1, j2)
