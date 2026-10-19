"""Simulation state -> renderer jobs (bone rotations straight from the bodies)."""
from __future__ import annotations

import numpy as np

from ..vrm import ASSETS, model_path
from .model import ActorSpec


def actor_frame(m, d, spec: ActorSpec, **extra) -> dict:
    """One renderer actor: each bone body's world rotation (its rotation from rest) and the hips."""
    P = spec.name + ":"
    bones = {}
    for i in range(m.nbody):
        n = m.body(i).name
        if n.startswith(P):
            w, x, y, z = d.xquat[i]
            bones[n[len(P):]] = [float(x), float(y), float(z), float(w)]
    rel = model_path(spec.model).resolve().relative_to(ASSETS.resolve())
    return {"model": f"/assets/{rel.as_posix()}", "scale": spec.scale, "bones": bones,
            "root": d.body(P + "hips").xpos.tolist(), "expression": {}, "fist": {"l": "relaxed", "r": "relaxed"},
            "dents": [], "lookAt": None, "motion": None, **extra}


def camera(eye, target, fov=35.0, up=(0, 1, 0)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    return {"eye": eye.tolist(), "fwd": f.tolist(), "up": list(up), "fov": fov}


def job(out, actors: list, cam: dict, size=(640, 640), env=None, **kw) -> dict:
    return {"out": str(out), "width": size[0], "height": size[1], "camera": cam, "actors": actors,
            "environment": env or {"sky": ["#9cc7ef", "#dfe8f0"], "prims": [{"type": "plane", "size": [30, 30], "pos": [0, 0, 0], "color": "#b9b2a6"}]},
            "supersample": 2, "passes": [], **kw}
