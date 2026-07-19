"""Vector math, rotation frames, and analytic two-bone IK.

Conventions:
  * Right-handed, Y up, ground plane y = 0.
  * An entity with yaw = 0 faces +Z; its right side is +X.
  * Angles in degrees at the API surface, radians internally.
  * A "frame" is a 3x3 matrix whose columns are (right, up, forward).
"""
from __future__ import annotations

import math

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def V(x, y, z) -> np.ndarray:
    return np.array([float(x), float(y), float(z)])


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray, fallback=None) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array(fallback if fallback is not None else [0.0, 0.0, 1.0])
    return v / n


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def yaw_frame(yaw_deg: float) -> np.ndarray:
    """Frame for an upright entity facing `yaw` (0 = +Z)."""
    return rot_y(yaw_deg)


def frame(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Yaw about Y, then pitch about local X (forward bend +), then roll about local Z."""
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)


def f_right(F: np.ndarray) -> np.ndarray:
    return F[:, 0]


def f_up(F: np.ndarray) -> np.ndarray:
    return F[:, 1]


def f_fwd(F: np.ndarray) -> np.ndarray:
    return F[:, 2]


def look_frame(forward: np.ndarray, up_hint: np.ndarray = UP) -> np.ndarray:
    """Frame whose forward axis points along `forward`."""
    f = normalize(forward)
    r = np.cross(up_hint, f)
    if np.linalg.norm(r) < 1e-6:  # forward parallel to up hint
        r = np.cross(V(0, 0, 1), f)
        if np.linalg.norm(r) < 1e-6:
            r = V(1, 0, 0)
    r = normalize(r)
    u = normalize(np.cross(f, r))
    return np.column_stack([r, u, f])


def yaw_of(forward: np.ndarray) -> float:
    """Yaw (deg) whose facing best matches the horizontal part of `forward`."""
    return math.degrees(math.atan2(forward[0], forward[2]))


def horizontal(v: np.ndarray) -> np.ndarray:
    return V(v[0], 0.0, v[2])


def two_bone_ik(root: np.ndarray, target: np.ndarray, len_a: float, len_b: float,
                pole_hint: np.ndarray):
    """Solve a 2-segment chain (e.g. shoulder-elbow-wrist).

    Returns (mid, end, reached). Bone lengths are preserved exactly; if the
    target is out of reach the chain points at it fully extended and
    `reached` is False. `pole_hint` biases which way the joint bends.
    """
    to_t = target - root
    d = np.linalg.norm(to_t)
    max_d = len_a + len_b
    min_d = abs(len_a - len_b) + 1e-6
    reached = d <= max_d + 1e-6
    d_eff = clamp(d, min_d, max_d - 1e-6)
    dirv = normalize(to_t, fallback=[0, -1, 0])
    # Law of cosines for the angle at the root.
    cos_a = clamp((len_a * len_a + d_eff * d_eff - len_b * len_b) / (2 * len_a * d_eff), -1.0, 1.0)
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    pole = pole_hint - dirv * float(np.dot(pole_hint, dirv))
    if np.linalg.norm(pole) < 1e-6:
        pole = np.cross(dirv, UP)
        if np.linalg.norm(pole) < 1e-6:
            pole = np.cross(dirv, V(1, 0, 0))
    pole = normalize(pole)
    mid = root + dirv * (len_a * cos_a) + pole * (len_a * sin_a)
    end = root + dirv * min(d, max_d) if not reached else target.copy()
    # Keep the end exactly bone-length from mid (numerical safety).
    end = mid + normalize(end - mid, fallback=list(dirv)) * len_b
    return mid, end, reached


def chord_for_bend(len_a: float, len_b: float, bend_deg: float) -> float:
    """Root-to-end distance for a chain flexed by `bend_deg` (0 = straight)."""
    br = math.radians(bend_deg)
    c2 = len_a * len_a + len_b * len_b + 2 * len_a * len_b * math.cos(br)
    return math.sqrt(max(c2, 0.0))


def seg_seg_closest(p1, q1, p2, q2):
    """Closest points between segments p1-q1 and p2-q2. Returns (c1, c2, dist)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    if a < 1e-12 and e < 1e-12:
        return p1, p2, norm(p1 - p2)
    if a < 1e-12:
        s = 0.0
        t = clamp(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e < 1e-12:
            t = 0.0
            s = clamp(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = clamp((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = clamp((b - c) / a, 0.0, 1.0)
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return c1, c2, norm(c1 - c2)
