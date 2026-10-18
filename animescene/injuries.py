"""Persistent injuries: stored on the character, re-projected through the 3D body, drawn by code.

An injury is a point on the body *surface* in bone coordinates — (bone, t along the bone,
angle around it) — plus a kind, a size and a seed. Because it lives on the skeleton, it
follows the body into any pose; because the pattern is generated from the seed, the cut,
bruise or bandage has exactly the same shape every time it is drawn. The image model never
invents or moves wounds: the prompts carry no injury tags and the decal is composited after
generation, occlusion-tested against the 3D depth buffer.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image

from .camera import Camera

# bone -> (start joint, end joint, radius key or callable)
BONES = {
    "head": ("neck", "head_top", lambda b: b.head_r * 0.95),
    "neck": ("chest", "neck", lambda b: b.r["neck"]),
    "chest": ("spine", "neck", lambda b: b.r["chest"] * 0.95),
    "stomach": ("pelvis", "chest", lambda b: b.r["waist"]),
    "l_upper_arm": ("l_shoulder", "l_elbow", lambda b: b.r["upper_arm"]),
    "r_upper_arm": ("r_shoulder", "r_elbow", lambda b: b.r["upper_arm"]),
    "l_forearm": ("l_elbow", "l_wrist", lambda b: b.r["forearm"]),
    "r_forearm": ("r_elbow", "r_wrist", lambda b: b.r["forearm"]),
    "l_thigh": ("l_hip", "l_knee", lambda b: b.r["thigh"]),
    "r_thigh": ("r_hip", "r_knee", lambda b: b.r["thigh"]),
    "l_shin": ("l_knee", "l_ankle", lambda b: b.r["shin"]),
    "r_shin": ("r_knee", "r_ankle", lambda b: b.r["shin"]),
}
# named spots: (bone, t, angle deg; 0 = front, +90 = the character's right side)
SPOTS = {
    "forehead": ("head", 0.72, 0), "l_brow": ("head", 0.68, -35), "r_brow": ("head", 0.68, 35),
    "l_cheek": ("head", 0.5, -55), "r_cheek": ("head", 0.5, 55), "lip": ("head", 0.37, 0),
    "chin": ("head", 0.3, 0), "nose": ("head", 0.5, 0), "jaw_l": ("head", 0.35, -70), "jaw_r": ("head", 0.35, 70),
    "neck": ("neck", 0.5, 0), "chest": ("chest", 0.5, 0), "stomach": ("stomach", 0.45, 0),
    "l_side": ("stomach", 0.5, -90), "r_side": ("stomach", 0.5, 90), "back": ("chest", 0.4, 180),
    "l_shoulder": ("l_upper_arm", 0.12, 0), "r_shoulder": ("r_upper_arm", 0.12, 0),
    "l_upper_arm": ("l_upper_arm", 0.5, -90), "r_upper_arm": ("r_upper_arm", 0.5, 90),
    "l_forearm": ("l_forearm", 0.5, 0), "r_forearm": ("r_forearm", 0.5, 0),
    "l_thigh": ("l_thigh", 0.5, 0), "r_thigh": ("r_thigh", 0.5, 0),
    "l_knee": ("l_shin", 0.04, 0), "r_knee": ("r_shin", 0.04, 0),
    "l_shin": ("l_shin", 0.5, 0), "r_shin": ("r_shin", 0.5, 0),
}
KINDS = ("bruise", "cut", "bleeding_cut", "scar", "bandage", "swelling")


def _bone_frame(J: dict, bone: str):
    a, b = BONES[bone][0], BONES[bone][1]
    p, q = J[a], J[b]
    axis = (q - p) / max(np.linalg.norm(q - p), 1e-9)
    # body forward from the shoulders (the local frame is mirrored: r_shoulder - l_shoulder = right)
    right = J["r_shoulder"] - J["l_shoulder"]
    up = J["neck"] - J["pelvis"]
    fwd = np.cross(up, right)
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    if bone == "head":
        fwd = J["nose"] - J["head"]
    f = fwd - axis * (fwd @ axis)
    if np.linalg.norm(f) < 1e-6:
        f = np.cross(axis, [1.0, 0, 0])
    f /= np.linalg.norm(f)
    side = np.cross(axis, f)  # sign fixed below against the body's right
    if side @ right < 0:
        side = -side
    return p, q, axis, f, side


def surface_point(actor, bone: str, t: float, angle_deg: float):
    """World position and outward normal of a bone-surface coordinate."""
    J = actor.joints
    p, q, axis, f, side = _bone_frame(J, bone)
    r = BONES[bone][2](actor.char.body)
    a = math.radians(angle_deg)
    n = f * math.cos(a) + side * math.sin(a)
    if bone == "head":
        c = J["head"]
        up = axis
        h = (t - 0.5) * 2.0  # -1 chin .. +1 crown
        n = n * math.sqrt(max(0.0, 1 - min(abs(h), 0.95) ** 2)) + up * h * 0.6
        n /= np.linalg.norm(n)
        return c + n * r, n
    return p + (q - p) * t + n * r, n


def anchor_from_point(actor, point: np.ndarray) -> tuple:
    """Nearest bone-surface coordinate to a world point (used when a blow leaves a mark)."""
    J = actor.joints
    best = None
    for bone, (a, b, _) in BONES.items():
        p, q, axis, f, side = _bone_frame(J, bone)
        seg = q - p
        t = float(np.clip((point - p) @ seg / max(seg @ seg, 1e-9), 0, 1))
        c = J["head"] if bone == "head" else p + seg * t
        d = point - c
        dist = np.linalg.norm(d)
        if best is None or dist < best[0]:
            dp = d - axis * (d @ axis)
            ang = math.degrees(math.atan2(dp @ side, dp @ f)) if np.linalg.norm(dp) > 1e-6 else 0.0
            if bone == "head":
                t = float(np.clip(0.5 + (d @ axis) / (2 * actor.char.body.head_r * 0.6), 0.05, 0.95))
            best = (dist, bone, t, ang)
    return best[1], round(best[2], 3), round(best[3], 1)


def make_injury(kind: str, spot: str | None = None, bone: str | None = None, t: float = 0.5, angle: float = 0.0,
                size: float = 0.05, seed: int | None = None, note: str = "") -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown injury kind {kind!r}; choose from {KINDS}")
    if spot is not None:
        if spot not in SPOTS:
            raise ValueError(f"unknown spot {spot!r}; choose from {sorted(SPOTS)}")
        bone, t, angle = SPOTS[spot]
    if bone not in BONES:
        raise ValueError(f"unknown bone {bone!r}")
    if seed is None:
        seed = random.Random(f"{kind}{bone}{t}{angle}").randrange(1 << 30)
    return {"kind": kind, "bone": bone, "t": t, "angle": angle, "size": size, "seed": seed, "note": note}


# ------------------------------------------------------------------ drawing
def _rgba(h, w):
    return np.zeros((h, w, 4), np.float32)


def _pattern(inj: dict, px: float) -> np.ndarray:
    """RGBA decal (float 0-1) in the injury's own frame: x along the bone, y across, gravity +y."""
    rng = random.Random(inj["seed"])
    s = max(6, int(px * 2.6))
    img = _rgba(s, s)
    c = s / 2
    kind = inj["kind"]
    if kind in ("bruise", "swelling"):
        for _ in range(7):
            ex, ey = rng.uniform(-0.18, 0.18) * s, rng.uniform(-0.12, 0.12) * s
            ax, ay = rng.uniform(0.12, 0.3) * s, rng.uniform(0.08, 0.2) * s
            col = rng.choice([(0.45, 0.2, 0.45), (0.55, 0.25, 0.4), (0.35, 0.25, 0.5), (0.6, 0.3, 0.3)])
            layer = np.zeros((s, s), np.float32)
            cv2.ellipse(layer, (int(c + ex), int(c + ey)), (int(ax), int(ay)), rng.uniform(0, 180), 0, 360, 1.0, -1)
            layer = cv2.GaussianBlur(layer, (0, 0), s * 0.06) * (0.45 if kind == "bruise" else 0.3)
            for k in range(3):
                img[..., k] = img[..., k] * (1 - layer) + col[k] * layer
            img[..., 3] = np.maximum(img[..., 3], layer)
        if kind == "swelling":
            img[..., :3] = img[..., :3] * 0.5 + np.array([0.85, 0.45, 0.45]) * 0.5
    elif kind in ("cut", "bleeding_cut", "scar"):
        n = 7
        pts = []
        for i in range(n):
            x = c + (i / (n - 1) - 0.5) * s * 0.75
            y = c + rng.uniform(-0.035, 0.035) * s + (i / (n - 1) - 0.5) * s * 0.18 * (1 if inj["seed"] % 2 else -1)
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        if kind == "scar":
            line = np.zeros((s, s), np.float32)
            cv2.polylines(line, [pts], False, 1.0, max(1, int(s * 0.035)), cv2.LINE_AA)
            img[..., :3] = np.array([0.93, 0.72, 0.68])
            img[..., 3] = line * 0.85
        else:
            line = np.zeros((s, s), np.float32)
            cv2.polylines(line, [pts], False, 1.0, max(1, int(s * 0.05)), cv2.LINE_AA)
            core = np.zeros((s, s), np.float32)
            cv2.polylines(core, [pts], False, 1.0, max(1, int(s * 0.02)), cv2.LINE_AA)
            drips = 2 if kind == "cut" else 4
            for _ in range(drips):
                i = rng.randrange(1, n - 1)
                x0, y0 = pts[i]
                length = rng.uniform(0.12, 0.25 if kind == "cut" else 0.42) * s
                wdt = max(1, int(s * rng.uniform(0.025, 0.04)))
                cv2.line(line, (int(x0), int(y0)), (int(x0 + rng.uniform(-0.02, 0.02) * s), int(y0 + length)), 1.0,
                         wdt, cv2.LINE_AA)
                cv2.circle(line, (int(x0), int(y0 + length)), int(wdt * 1.3), 1.0, -1, cv2.LINE_AA)
            img[..., 0], img[..., 1], img[..., 2] = 0.62, 0.04, 0.06
            img[..., 3] = np.clip(line, 0, 1) * 0.95
            img[..., :3] = img[..., :3] * (1 - core[..., None] * 0.6)  # darker core of the wound
    elif kind == "bandage":
        band = np.zeros((s, s), np.float32)
        cv2.rectangle(band, (int(c - s * 0.12), int(c - s * 0.42)), (int(c + s * 0.12), int(c + s * 0.42)), 1.0, -1)
        img[..., :3] = 0.97
        lines = np.zeros((s, s), np.float32)
        for k in range(-3, 4):
            y = int(c + k * s * 0.1)
            cv2.line(lines, (int(c - s * 0.12), y), (int(c + s * 0.12), y + int(s * 0.02)), 1.0, 1, cv2.LINE_AA)
        img[..., :3] = img[..., :3] * (1 - lines[..., None] * 0.18) - np.array([0, 0.02, 0.05]) * lines[..., None]
        edge = cv2.morphologyEx(band, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8))
        img[..., :3] *= 1 - edge[..., None] * 0.45
        img[..., 3] = band
        if inj.get("bloodied", True):
            spot = np.zeros((s, s), np.float32)
            cv2.circle(spot, (int(c + rng.uniform(-0.04, 0.04) * s), int(c)), int(s * 0.06), 1.0, -1, cv2.LINE_AA)
            spot = cv2.GaussianBlur(spot, (0, 0), s * 0.02) * band
            img[..., :3] = img[..., :3] * (1 - spot[..., None]) + np.array([0.7, 0.12, 0.12]) * spot[..., None]
    return img


def draw(image: Image.Image, actor, cam: Camera, depth: np.ndarray, ids: np.ndarray, actor_index: int,
         size_scale: tuple = (1.0, 1.0)) -> tuple[Image.Image, list]:
    """Composite all of an actor's stored injuries. Returns (image, list of drawn injuries)."""
    out = np.asarray(image).astype(np.float32) / 255
    H, W = out.shape[:2]
    sx, sy = size_scale  # render-resolution -> image-resolution scale
    drawn = []
    for inj in actor.char.injuries:
        if inj.get("healed"):
            continue
        p, n = surface_point(actor, inj["bone"], inj["t"], inj["angle"])
        u, vv, z = cam.project(p[None])[0]
        if z <= 0:
            continue
        x, y = int(u), int(vv)
        if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
            continue
        to_cam = cam.eye - p
        if n @ to_cam / np.linalg.norm(to_cam) < 0.15:  # facing away from the camera
            continue
        win = ids[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        dwin = depth[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        if not ((win == actor_index) & (np.abs(dwin - z) < 0.06)).any():
            continue  # hidden behind something
        px = inj["size"] * cam.f / z * sx
        if px < 2:
            continue
        dec = _pattern(inj, px)
        # orient: decal x along the projected bone, decal +y along projected gravity
        a, b = BONES[inj["bone"]][0], BONES[inj["bone"]][1]
        pa, pb = cam.project(np.array([actor.joints[a], actor.joints[b]]))[:, :2]
        ang = math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))
        down = cam.project(np.array([p, p - np.array([0, 0.05, 0])]))[:, :2]
        gang = math.degrees(math.atan2(down[1][1] - down[0][1], down[1][0] - down[0][0]))
        rot = gang - 90 if inj["kind"] in ("cut", "bleeding_cut") else ang
        # foreshortening: squash across the surface by how obliquely it faces us
        obl = float(np.clip(n @ to_cam / np.linalg.norm(to_cam), 0.3, 1.0))
        s = dec.shape[0]
        M = cv2.getRotationMatrix2D((s / 2, s / 2), -rot, 1.0)
        M[:, :2] = M[:, :2] @ np.diag([1.0, obl]) if inj["kind"] != "bandage" else M[:, :2]
        cx, cy = x * sx, y * sy
        M[:, 2] += np.array([cx - s / 2, cy - s / 2])
        warped = cv2.warpAffine(dec, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        # keep it on this character's visible body
        own = cv2.resize((ids == actor_index).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        own = cv2.GaussianBlur(cv2.dilate(own, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        alpha = warped[..., 3:4] * own[..., None]
        if inj["kind"] in ("bruise", "swelling"):  # multiply-like so shading underneath survives
            out = out * (1 - alpha) + out * warped[..., :3] * 1.6 * alpha
        else:
            out = out * (1 - alpha) + warped[..., :3] * alpha
        drawn.append(inj)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)), drawn
