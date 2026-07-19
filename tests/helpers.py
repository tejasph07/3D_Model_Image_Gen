"""Shared assertions for physical plausibility."""
import numpy as np

from scene3d.skeleton import BODY_25_NAMES, DIMS, JOINT_RADII
from scene3d.solver import Scene
from scene3d.vecmath import norm


def build(text: str) -> Scene:
    sc = Scene()
    sc.run_text(text)
    return sc


def bone_lengths_ok(h, tol=0.02):
    """Every rigid bone must keep its length after any solving."""
    J = h.joints
    s = h.scale
    checks = [
        ("l_hip", "l_knee", DIMS["thigh"]), ("l_knee", "l_ankle", DIMS["shin"]),
        ("r_hip", "r_knee", DIMS["thigh"]), ("r_knee", "r_ankle", DIMS["shin"]),
        ("l_shoulder", "l_elbow", DIMS["upper_arm"]),
        ("l_elbow", "l_wrist", DIMS["forearm"]),
        ("r_shoulder", "r_elbow", DIMS["upper_arm"]),
        ("r_elbow", "r_wrist", DIMS["forearm"]),
        ("mid_hip", "neck", DIMS["spine"]),
    ]
    bad = []
    for a, b, expect in checks:
        got = norm(J[a] - J[b])
        if abs(got - expect * s) > tol:
            bad.append(f"{a}-{b}: {got:.3f} vs {expect * s:.3f}")
    assert not bad, f"{h.name} bone lengths broken: {bad}"


def above_ground_ok(scene, tol=0.03):
    for h in scene.humans():
        for n in BODY_25_NAMES:
            y = h.joints[n][1] - JOINT_RADII[n] * h.scale
            assert y > -tol, f"{h.name}.{n} below ground: {y:.3f}"
    for p in scene.props():
        assert p.lowest_point() > -tol, f"{p.name} below ground"


def contact_dist(scene, name_a, part_a, name_b, part_b):
    a = scene.entities[name_a]
    b = scene.entities[name_b]
    ca, ra = a.part_point(part_a)
    cb, rb = b.part_point(part_b)
    return norm(ca - cb) - ra - rb


def all_ok(scene):
    for h in scene.humans():
        bone_lengths_ok(h)
    above_ground_ok(scene)
