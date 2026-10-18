"""VRM (glTF) model inspection: metadata, license and the humanoid rest skeleton.

Pure Python (no 3D engine): reads the glb JSON chunk, walks the node hierarchy to get the rest
world positions of the VRM humanoid bones, and derives a `Body` whose bone lengths are the
model's own, so the stage solves poses that fit this exact mesh.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rig import Body

HUMANOID = ["hips", "spine", "chest", "upperChest", "neck", "head", "leftUpperLeg", "leftLowerLeg", "leftFoot",
            "leftToes", "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes", "leftShoulder", "leftUpperArm",
            "leftLowerArm", "leftHand", "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
            "leftEye", "rightEye", "leftMiddleProximal", "rightMiddleProximal"]


def read_glb_json(path: str | Path) -> dict:
    with open(path, "rb") as f:
        magic, _version, _length = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            raise ValueError(f"{path} is not a glb/vrm file")
        clen, ctype = struct.unpack("<II", f.read(8))
        if ctype != 0x4E4F534A:
            raise ValueError("first glb chunk is not JSON")
        return json.loads(f.read(clen))


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    S = np.diag(n.get("scale", [1, 1, 1]))
    M[:3, :3] = _quat_mat(n.get("rotation", [0, 0, 0, 1])) @ S
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


@dataclass
class VRMInfo:
    path: str
    version: str  # "0" or "1"
    name: str
    authors: str
    license: str
    bones: dict  # humanoid bone -> rest world position (metres), in the model's own frame
    forward: np.ndarray  # rest facing direction in model space

    def body(self, build: str = "athletic", gender: str = "female") -> Body:
        """A Body with this model's bone lengths (the stage then poses the actual mesh exactly)."""
        B = self.bones
        d = lambda a, b: float(np.linalg.norm(B[a] - B[b]))  # noqa: E731
        top = self.head_top()
        H = float(top - min(B["leftFoot"][1], B["rightFoot"][1]) + 0.045)
        body = Body(height=H, build=build, gender=gender)
        body.thigh = (d("leftUpperLeg", "leftLowerLeg") + d("rightUpperLeg", "rightLowerLeg")) / 2
        body.shin = (d("leftLowerLeg", "leftFoot") + d("rightLowerLeg", "rightFoot")) / 2
        body.upper_arm = (d("leftUpperArm", "leftLowerArm") + d("rightUpperArm", "rightLowerArm")) / 2
        body.forearm = (d("leftLowerArm", "leftHand") + d("rightLowerArm", "rightHand")) / 2
        body.hip_w = abs(B["leftUpperLeg"][0] - B["rightUpperLeg"][0]) / 2
        body.shoulder_w = abs(B["leftUpperArm"][0] - B["rightUpperArm"][0]) / 2
        foot_y = min(B["leftFoot"][1], B["rightFoot"][1])
        body.ankle_h = max(0.03, foot_y)
        body.hip_h = (B["leftUpperLeg"][1] + B["rightUpperLeg"][1]) / 2
        chest = B.get("upperChest", B.get("chest", B["spine"]))
        body.spine = float(chest[1] - B["hips"][1]) * 1.0
        body.upper = float(B["neck"][1] - chest[1])
        body.neck = float(B["head"][1] - B["neck"][1])
        if "leftToes" in B:
            body.foot = d("leftFoot", "leftToes") + 0.03
        hand_tip = B.get("leftMiddleProximal")
        if hand_tip is not None:
            body.hand = d("leftHand", "leftMiddleProximal") * 2.0
        body.head_r = (top - float(B["head"][1])) / 1.85
        # VRoid-style meshes are slimmer than the generic capsule radii: contacts and floor
        # clearance are computed against the actual silhouette
        body.r = {k: val * 0.78 for k, val in body.r.items()}
        return body

    def head_top(self) -> float:
        """Crown height estimated from the head bone (VRM head bones sit at the skull base)."""
        B = self.bones
        neck_to_head = float(B["head"][1] - B["neck"][1])
        eye = B.get("leftEye")
        if eye is not None:
            return float(eye[1] + (eye[1] - B["head"][1]) * 1.3 + 0.03)
        return float(B["head"][1] + max(0.16, neck_to_head * 2.6))


def inspect(path: str | Path) -> VRMInfo:
    g = read_glb_json(path)
    ext = g.get("extensions", {})
    nodes = g["nodes"]
    if "VRMC_vrm" in ext:
        v = ext["VRMC_vrm"]
        version = "1"
        hb = {k: b["node"] for k, b in v["humanoid"]["humanBones"].items()}
        meta = v.get("meta", {})
        name, authors = meta.get("name", ""), ", ".join(meta.get("authors", []))
        lic = meta.get("licenseUrl", "") + (" (commercial: %s)" % meta.get("commercialUsage", "?"))
    elif "VRM" in ext:
        v = ext["VRM"]
        version = "0"
        hb = {b["bone"]: b["node"] for b in v["humanoid"]["humanBones"]}
        meta = v.get("meta", {})
        name, authors = meta.get("title", ""), meta.get("author", "")
        lic = meta.get("licenseName", "") + (" " + meta.get("otherLicenseUrl", "") if meta.get("otherLicenseUrl") else "")
    else:
        raise ValueError(f"{path} has no VRM extension")
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    cache = {}

    def world(i):
        if i in cache:
            return cache[i]
        M = _node_matrix(nodes[i])
        if i in parent:
            M = world(parent[i]) @ M
        cache[i] = M
        return M

    bones = {k: world(i)[:3, 3].copy() for k, i in hb.items() if k in HUMANOID}
    # VRM 0.x models face -Z, VRM 1.0 models face +Z
    forward = np.array([0, 0, -1.0]) if version == "0" else np.array([0, 0, 1.0])
    if version == "0":  # put everything in a +Z-facing frame like VRM 1.0 (three-vrm does the same)
        bones = {k: p * np.array([-1, 1, -1]) for k, p in bones.items()}
        forward = np.array([0, 0, 1.0])
    return VRMInfo(str(path), version, name, authors, lic, bones, forward)


def library(root: str | Path) -> dict:
    """All .vrm models under a directory (one level of subfolders allowed), by file stem."""
    out = {}
    for p in sorted(Path(root).glob("**/*.vrm")):
        try:
            out[p.stem] = inspect(p)
        except Exception as e:  # a broken file shouldn't hide the rest of the library
            print(f"[vrm] skipping {p.name}: {e}")
    return out


ASSETS = Path(__import__("os").environ.get("ANIMESCENE_ASSETS", "/home/user/assets"))
_INFO: dict = {}


def model_path(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".vrm" and p.exists():
        return p
    hits = sorted((ASSETS / "vrm").glob(f"**/{name}.vrm"))
    if not hits:
        raise FileNotFoundError(f"VRM model {name!r} not found under {ASSETS / 'vrm'}")
    return hits[0]


def model_info(name: str) -> VRMInfo:
    if name not in _INFO:
        _INFO[name] = inspect(model_path(name))
    return _INFO[name]
