"""Collision shapes fitted to a VRM model's real rest mesh, one rigid body per humanoid bone.

The renderer dumps the rest-pose skin vertices (labelled by the bone that drives them, skin vs. worn
clothing); every bone gets a capsule, ellipsoid or box that encloses its part of the body. Worn
clothing counts only up to LOOSE metres over the skin, so skirts and baggy sleeves don't become body.
Everything is in the model's rest frame (metres, y up, facing +z), relative to the bone's own origin.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..vrm import ASSETS, inspect, model_path

LOOSE = 0.02
VERSION = 1
# body part labels (from the renderer's bake) -> the humanoid bone that owns them
PART_BONE = {"hips": "hips", "spine": "spine", "chest": "chest", "neck": "neck", "head": "head",
             "l_thigh": "leftUpperLeg", "l_shin": "leftLowerLeg", "l_foot": "leftFoot",
             "r_thigh": "rightUpperLeg", "r_shin": "rightLowerLeg", "r_foot": "rightFoot",
             "l_upperarm": "leftUpperArm", "l_forearm": "leftLowerArm", "l_hand": "leftHand",
             "r_upperarm": "rightUpperArm", "r_forearm": "rightLowerArm", "r_hand": "rightHand"}


def rest_vertices(model: str) -> list:
    """[(part, x, y, z, kind)] of the model's rest-pose mesh (kind 0 skin, 3 clothing), cached."""
    path = model_path(model)
    cache = path.parent / ".measure" / f"{path.stem}.rest{VERSION}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rel = path.resolve().relative_to(ASSETS.resolve())
    tmp = Path(tempfile.mkdtemp())
    job = {"out": str(tmp / "r.png"), "width": 16, "height": 16, "springs": False, "dumpVerts": 1, "passes": [],
           "softContacts": False, "camera": {"eye": [0, 1, 4], "fwd": [0, 0, -1], "up": [0, 1, 0], "fov": 30},
           "environment": {"lights": []},
           "actors": [{"model": f"/assets/{rel.as_posix()}", "rest": True, "fist": {"l": "flat", "r": "flat"},
                       "expression": {}, "dents": []}]}
    (tmp / "jobs.json").write_text(json.dumps([job]))
    render = Path(__file__).resolve().parents[2] / "render3d" / "render.mjs"
    subprocess.run(["node", str(render), str(tmp / "jobs.json")], check=True, capture_output=True, timeout=300,
                   env={**os.environ, "ANIMESCENE_ASSETS": str(ASSETS)})
    verts = json.loads((tmp / "r_contacts.json").read_text())["verts"][0]
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(verts))
    return verts


def _combine(skin, wear, fn):
    """fn(points) for skin and for clothing; clothing extends the skin by at most LOOSE."""
    s = fn(skin) if len(skin) >= 8 else None
    w = fn(wear) if len(wear) >= 8 else None
    if s is None:
        return w
    if w is None:
        return s
    return np.maximum(s, np.minimum(w, s + LOOSE))


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip((P - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def fit(model: str) -> dict:
    """bone -> shape spec, in the bone's local rest frame:
    {"type": "capsule", "from": [..], "to": [..], "r": r} | {"type": "ellipsoid", "pos": [..], "size": [..]}
    | {"type": "box", "pos": [..], "size": [half extents]}; plus "_bones": rest world positions."""
    info = inspect(model_path(model))
    B = {k: np.asarray(p, float) for k, p in info.bones.items()}
    groups: dict = {}
    for part, x, y, z, k in rest_vertices(model):
        bone = PART_BONE.get(part)
        if bone:
            groups.setdefault(bone, ([], []))[1 if k == 3 else 0].append((x, y, z))
    G = {b: (np.array(s).reshape(-1, 3), np.array(w).reshape(-1, 3)) for b, (s, w) in groups.items()}
    if "upperChest" in B:  # the chest label covers chest + upperChest: split at the upperChest bone
        s, w = G["chest"]
        cut = B["upperChest"][1]
        G["upperChest"] = (s[s[:, 1] >= cut], w[w[:, 1] >= cut])
        G["chest"] = (s[s[:, 1] < cut], w[w[:, 1] < cut])
    out = {"_bones": {k: v.tolist() for k, v in B.items()}}
    child = {"leftUpperLeg": "leftLowerLeg", "leftLowerLeg": "leftFoot", "rightUpperLeg": "rightLowerLeg",
             "rightLowerLeg": "rightFoot", "leftUpperArm": "leftLowerArm", "leftLowerArm": "leftHand",
             "rightUpperArm": "rightLowerArm", "rightLowerArm": "rightHand", "neck": "head"}
    for bone, (skin, wear) in G.items():
        o = B[bone]
        if bone in child:  # limbs: capsule from joint to joint, radius around that axis
            a, b = B[bone], B[child[bone]]
            r = _combine(skin, wear, lambda P: np.array([np.percentile(_seg_dist(P, a, b), 75)]))[0]
            # stop short of the joints by the radius so neighbours don't overlap at the joint
            d = (b - a) / np.linalg.norm(b - a)
            L = np.linalg.norm(b - a)
            inset = min(r * 0.6, L * 0.2)
            out[bone] = {"type": "capsule", "from": (d * inset).tolist(), "to": (d * (L - inset)).tolist(), "r": float(r)}
        elif bone.endswith(("Hand", "Foot")):  # boxes: palm/fingers, shoe
            def box(P):
                lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
                return np.concatenate([lo, hi])
            lohi = _combine(skin, wear, box)
            lo, hi = lohi[:3], lohi[3:]
            if bone.endswith("Foot"):
                lo[1] = max(lo[1], 0.0)  # the sole sits on the floor at rest
            out[bone] = {"type": "box", "pos": ((lo + hi) / 2 - o).tolist(), "size": ((hi - lo) / 2).tolist()}
        else:  # torso segments and head: ellipsoids
            def ell(P):
                lo, hi = np.percentile(P, 3, axis=0), np.percentile(P, 97, axis=0)
                return np.concatenate([lo, hi])
            lohi = _combine(skin if bone != "head" else skin, wear if bone != "head" else np.zeros((0, 3)), ell)
            lo, hi = lohi[:3], lohi[3:]
            c, h = (lo + hi) / 2, (hi - lo) / 2
            # an ellipsoid inscribed in the box loses the corners: grow it toward the box diagonal
            k = 1.08 if bone != "head" else 1.0
            out[bone] = {"type": "ellipsoid", "pos": (c - o).tolist(), "size": (h * k).tolist()}
    return out
