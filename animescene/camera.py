"""Perspective camera with cinematic shot presets and automatic framing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vec import UP, rot_axis, unit, v

# shot -> (elevation deg, margin, fov deg); elevation < 0 means the camera looks up
SHOTS = {
    "wide": (8.0, 0.08, 40.0),
    "full": (5.0, 0.06, 38.0),
    "low_angle": (-14.0, 0.07, 48.0),
    "worms_eye": (-28.0, 0.06, 58.0),
    "high_angle": (28.0, 0.08, 40.0),
    "birds_eye": (55.0, 0.08, 40.0),
    "medium": (4.0, -0.25, 34.0),  # negative margin crops into the action
    "closeup": (3.0, 0.05, 30.0),  # use with camera.focus
    "extreme_closeup": (2.0, 0.0, 24.0),
}


@dataclass
class Camera:
    eye: np.ndarray
    fwd: np.ndarray
    right: np.ndarray
    up: np.ndarray
    width: int
    height: int
    fov: float  # vertical, degrees

    @property
    def f(self) -> float:
        return (self.height / 2) / np.tan(np.deg2rad(self.fov) / 2)

    def to_cam(self, pts: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(pts) - self.eye
        return np.stack([d @ self.right, d @ self.up, d @ self.fwd], axis=-1)

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points -> (u, v, depth) pixel coordinates."""
        c = self.to_cam(pts)
        z = np.maximum(c[:, 2], 1e-4)
        u = self.width / 2 + self.f * c[:, 0] / z
        vv = self.height / 2 - self.f * c[:, 1] / z
        return np.stack([u, vv, c[:, 2]], axis=-1)

    def rays(self, x0=0, y0=0, x1=None, y1=None) -> np.ndarray:
        """Unit ray directions for the pixel-centre grid of a sub-rectangle, (h, w, 3)."""
        x1 = self.width if x1 is None else x1
        y1 = self.height if y1 is None else y1
        xs = (np.arange(x0, x1) + 0.5 - self.width / 2) / self.f
        ys = -(np.arange(y0, y1) + 0.5 - self.height / 2) / self.f
        X, Y = np.meshgrid(xs, ys)
        d = X[..., None] * self.right + Y[..., None] * self.up + self.fwd
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _basis(fwd: np.ndarray, roll_deg: float):
    right = unit(np.cross(fwd, UP), fallback=(1, 0, 0))
    up = np.cross(right, fwd)
    if roll_deg:
        R = rot_axis(fwd, np.deg2rad(roll_deg))
        right, up = R @ right, R @ up
    return right, up


def frame_camera(points: np.ndarray, width: int, height: int, shot: str = "wide", azimuth: float = 0.0,
                 elevation: float | None = None, fov: float | None = None, roll: float = 0.0,
                 margin: float | None = None, min_eye_height: float = 0.25,
                 center_bias: tuple = (0.0, 0.0)) -> Camera:
    """Choose a camera that frames `points` for the given shot.

    azimuth: degrees around the scene; 0 = camera on the +z side looking toward -z.
    center_bias: (x, y) shift of the subject inside the frame, in fractions of the frame.
    """
    if shot not in SHOTS:
        raise ValueError(f"unknown shot {shot!r}; choose from {sorted(SHOTS)}")
    e0, m0, f0 = SHOTS[shot]
    elev = e0 if elevation is None else elevation
    fov = f0 if fov is None else fov
    margin = m0 if margin is None else margin
    center = (points.min(0) + points.max(0)) / 2
    for _ in range(6):
        cam, c = _fit(points, width, height, azimuth, elev, fov, roll, margin, center_bias, center)
        if cam.eye[1] >= min_eye_height - 1e-3:
            return cam
        # the lens would be under the floor: flatten the angle until it sits at min height
        horiz = np.hypot(*(cam.eye - c)[[0, 2]])
        elev = float(np.rad2deg(np.arctan2(min_eye_height - c[1], horiz)))
    return cam


def _fit(points, width, height, azimuth, elev, fov, roll, margin, center_bias, center):
    az, el = np.deg2rad(azimuth), np.deg2rad(elev)
    back = v(np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el))
    fwd = -back
    right, up = _basis(fwd, roll)

    def fit(dist, c):
        cam = Camera(c + back * dist, fwd, right, up, width, height, fov)
        return cam, cam.project(points)

    c = center.copy()
    lo, hi = 0.5, 80.0
    for _ in range(40):  # binary search on distance, re-centring as we go
        mid = (lo + hi) / 2
        cam, p = fit(mid, c)
        if (p[:, 2] <= 0.05).any():
            lo = mid
            continue
        umin, umax = p[:, 0].min(), p[:, 0].max()
        vmin, vmax = p[:, 1].min(), p[:, 1].max()
        # re-centre the look-at point on the projected bbox centre
        du = ((umin + umax) / 2 - width / 2 - center_bias[0] * width) / cam.f
        dv = ((vmin + vmax) / 2 - height / 2 - center_bias[1] * height) / cam.f
        depth = (c - cam.eye) @ fwd
        c = c + right * du * depth - up * dv * depth
        fits = (umax - umin) <= width * (1 - 2 * margin) and (vmax - vmin) <= height * (1 - 2 * margin)
        if fits:
            hi = mid
        else:
            lo = mid
    cam, _ = fit(hi, c)
    return cam, c
