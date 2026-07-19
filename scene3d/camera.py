"""Perspective camera with auto-framing of the scene."""
from __future__ import annotations

import math

import numpy as np

from .skeleton import Human
from .vecmath import V, UP, normalize, norm

VIEW_DIRS = {  # unit direction from target toward the camera
    "front": V(0, 0.18, 1), "back": V(0, 0.18, -1),
    "left": V(-1, 0.18, 0), "right": V(1, 0.18, 0),
    "three_quarter": V(0.75, 0.35, 1), "three_quarter_left": V(-0.75, 0.35, 1),
    "top": V(0.01, 1, 0.12), "low": V(0, -0.05, 1), "auto": V(0.55, 0.28, 1),
}


class Camera:
    def __init__(self, pos, target, fov_deg=50.0, width=768, height=768):
        self.pos = np.array(pos, dtype=float)
        self.target = np.array(target, dtype=float)
        self.fov = fov_deg
        self.width = width
        self.height = height
        f = normalize(self.target - self.pos)
        r = normalize(np.cross(f, UP), fallback=[1, 0, 0])
        u = np.cross(r, f)
        self._R = np.vstack([r, u, f])  # world -> camera rows
        self._fpx = (height / 2) / math.tan(math.radians(fov_deg) / 2)

    def project(self, p: np.ndarray):
        """Return (x_px, y_px, depth). depth <= 0 means behind the camera."""
        c = self._R @ (np.array(p, dtype=float) - self.pos)
        z = float(c[2])
        if z <= 0.05:
            return 0.0, 0.0, z
        x = self.width / 2 + self._fpx * float(c[0]) / z
        y = self.height / 2 - self._fpx * float(c[1]) / z
        return x, y, z

    def project_radius(self, p, r):
        _, _, z = self.project(p)
        if z <= 0.05:
            return 0.0
        return self._fpx * r / z


def _scene_points(scene):
    pts = []
    for h in scene.humans():
        pts.extend(h.body25())
    for p in scene.props():
        for shp in p.world_shapes():
            if shp[0] == "box":
                _, c, hlf, F = shp
                pts.append(c + F @ hlf)
                pts.append(c - F @ hlf)
            elif shp[0] == "sphere":
                pts.append(shp[1] + V(shp[2], shp[2], shp[2]))
                pts.append(shp[1] - V(shp[2], shp[2], shp[2]))
            else:
                pts.extend([shp[1], shp[2]])
    return np.array(pts) if pts else np.zeros((1, 3))


def make_camera(scene, width=768, height=768):
    view, target_name = scene.camera_cmd
    view = view if view in VIEW_DIRS else "auto"
    if target_name != "scene" and target_name in scene.entities:
        ent = scene.entities[target_name]
        if isinstance(ent, Human):
            pts = ent.body25()
        else:
            pts = _scene_points_single(ent)
    else:
        pts = _scene_points(scene)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    radius = max(float(norm(pts.max(axis=0) - pts.min(axis=0)) / 2), 0.6)
    fov = 50.0
    dist = radius / math.tan(math.radians(fov) / 2) * 1.15 + radius * 0.2
    d = normalize(VIEW_DIRS[view])
    pos = center + d * dist
    if pos[1] < 0.25:
        pos[1] = 0.25
    return Camera(pos, center, fov, width, height)


def _scene_points_single(prop):
    pts = []
    for shp in prop.world_shapes():
        if shp[0] == "box":
            _, c, hlf, F = shp
            pts.append(c + F @ hlf)
            pts.append(c - F @ hlf)
        elif shp[0] == "sphere":
            pts.append(shp[1] + V(shp[2], shp[2], shp[2]))
            pts.append(shp[1] - V(shp[2], shp[2], shp[2]))
        else:
            pts.extend([shp[1], shp[2]])
    return np.array(pts)
