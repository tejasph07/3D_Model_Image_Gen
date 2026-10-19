"""Small vector / rotation helpers (numpy, right-handed, y up, metres)."""
from __future__ import annotations

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def v(*xs) -> np.ndarray:
    if len(xs) == 1:
        return np.asarray(xs[0], dtype=float)
    return np.array(xs, dtype=float)


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def unit(a: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    n = np.linalg.norm(a)
    if n < 1e-9:
        return np.asarray(fallback, dtype=float)
    return a / n


def flat(a: np.ndarray) -> np.ndarray:
    """Project onto the ground plane."""
    return np.array([a[0], 0.0, a[2]])


def rot_axis(axis, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    a = unit(np.asarray(axis, dtype=float))
    x, y, z = a
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    return rot_axis(UP, yaw_rad)


def yaw_of(direction: np.ndarray) -> float:
    """Yaw (about +y) that turns +z into the given horizontal direction."""
    return float(np.arctan2(direction[0], direction[2]))


def frame_from(forward: np.ndarray, up: np.ndarray = UP) -> np.ndarray:
    """Orthonormal frame as columns (right, up, forward). right = forward x up."""
    f = unit(forward)
    r = unit(np.cross(f, up), fallback=(1, 0, 0))
    u = np.cross(r, f)
    return np.stack([r, u, f], axis=1)


def deg(x: float) -> float:
    return float(np.deg2rad(x))


def lerp(a, b, t):
    return a + (b - a) * t
