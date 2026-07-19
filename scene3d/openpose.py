"""OpenPose-compatible JSON export (plus a scene3d extension block)."""
from __future__ import annotations

import json

import numpy as np

from .skeleton import BODY_25_NAMES


def _kp_list(cam, pts):
    out = []
    for p in pts:
        x, y, z = cam.project(p)
        visible = z > 0.05 and 0 <= x < cam.width and 0 <= y < cam.height
        out.extend([round(float(x), 2), round(float(y), 2),
                    1.0 if visible else 0.0])
    return out


def scene_json(scene, cam) -> dict:
    people = []
    for h in scene.humans():
        pts = h.body25()
        person = {
            "name": h.name,
            "pose_keypoints_2d": _kp_list(cam, pts),
            "hand_left_keypoints_2d": _kp_list(cam, h.hand_pts["L"])
                if h.hand_pts["L"] is not None else [],
            "hand_right_keypoints_2d": _kp_list(cam, h.hand_pts["R"])
                if h.hand_pts["R"] is not None else [],
            "joints_3d": {n: [round(float(v), 4) for v in h.joints[n]]
                          for n in BODY_25_NAMES},
        }
        people.append(person)
    objects = []
    for p in scene.props():
        objects.append({
            "name": p.name, "type": p.kind,
            "position": [round(p.x, 4), round(p.y, 4), round(p.z, 4)],
            "yaw_deg": round(p.yaw, 2), "pitch_deg": round(p.pitch, 2),
            "scale": p.scale,
            "anchors_3d": {k: [round(float(v), 4) for v in p.anchor(k)]
                           for k in p.spec.anchors},
        })
    return {
        "format": "scene3d-openpose-1.0",
        "canvas": [cam.width, cam.height],
        "camera": {
            "position": [round(float(v), 4) for v in cam.pos],
            "target": [round(float(v), 4) for v in cam.target],
            "fov_deg": cam.fov,
        },
        "people": people,
        "objects": objects,
        "warnings": scene.warnings,
    }


def write_json(scene, cam, path):
    with open(path, "w") as f:
        json.dump(scene_json(scene, cam), f, indent=1)
