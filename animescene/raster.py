"""Analytic ray caster for the stage: capsules + spheres + ground plane.

Bodies are built from capsules (limbs, torso) and spheres (head, hands);
weapons are thin capsules. Every primitive is intersected exactly per pixel
(numpy, restricted to the primitive's projected bounding box), producing a
depth buffer, an instance-id buffer and normals. That is all the control
images need: depth, occlusion-aware pose, and per-character masks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .stage import Actor, SolvedScene

GROUND_ID = -2
SKY_ID = -1


@dataclass
class Prim:
    kind: str  # "cap" | "sph"
    a: np.ndarray
    b: np.ndarray  # == a for spheres
    r: float
    owner: int  # actor index
    part: str


def body_prims(actor: Actor, idx: int) -> list[Prim]:
    J, b = actor.joints, actor.char.body
    H, R = b.height, b.r
    P = []

    def cap(p, q, r, part):
        P.append(Prim("cap", J[p] if isinstance(p, str) else p, J[q] if isinstance(q, str) else q, r, idx, part))

    def sph(p, r, part):
        c = J[p] if isinstance(p, str) else p
        P.append(Prim("sph", c, c, r, idx, part))

    # head: cranium + jaw
    hd, top, nose = J["head"], J["head_top"], J["nose"]
    up = (top - hd) / max(np.linalg.norm(top - hd), 1e-6)
    sph(hd, b.head_r, "head")
    sph(hd - up * b.head_r * 0.38 + (nose - hd) * 0.25, b.head_r * 0.68, "head")
    cap("neck", hd - up * b.head_r * 0.5, R["neck"], "neck")
    # torso
    cap("chest", "neck", R["chest"] * 0.8, "torso")
    cap("spine", "chest", R["chest"], "torso")
    cap("pelvis", "spine", R["waist"], "torso")
    cap("l_hip", "r_hip", R["pelvis"] * 0.85, "torso")
    cap("l_shoulder", "r_shoulder", R["upper_arm"] * 1.15, "torso")
    if b.gender == "female":
        chest_fwd = np.cross(J["r_shoulder"] - J["l_shoulder"], J["neck"] - J["spine"])
        chest_fwd /= max(np.linalg.norm(chest_fwd), 1e-6)
        # local frame is mirrored (x = right) so the cross product points backwards
        chest_fwd = -chest_fwd
        for s in ("l", "r"):
            c = J["chest"] * 0.75 + J[f"{s}_shoulder"] * 0.25 + chest_fwd * R["chest"] * 0.55
            c = c - (J["neck"] - J["chest"]) * 0.25
            sph(c, R["chest"] * 0.45, "torso")
    for s in ("l", "r"):
        cap(f"{s}_shoulder", f"{s}_elbow", R["upper_arm"], "arm")
        cap(f"{s}_elbow", f"{s}_wrist", R["forearm"], "arm")
        sph((J[f"{s}_wrist"] + J[f"{s}_hand"]) / 2, R["hand"], "hand")
        cap(f"{s}_hip", f"{s}_knee", R["thigh"], "leg")
        cap(f"{s}_knee", f"{s}_ankle", R["shin"], "leg")
        cap(f"{s}_ankle", f"{s}_toe", R["foot"], "foot")
    # Blades are drawn thicker than life: a real 3 cm blade is 2-3 px in a wide shot, too thin for
    # the depth ControlNet to register. Exaggerated, the weapon's line and crossing read clearly.
    for pom, grip, tip, wdt in actor.weapons:
        P.append(Prim("cap", pom, grip, max(wdt * 1.2, 0.03), idx, "weapon"))
        P.append(Prim("cap", grip, tip, max(wdt * 1.2, 0.034), idx, "weapon"))
    return P


def _hit_capsule(ro, rd, pa, pb, ra):
    """Vectorised ray/capsule intersection (iq). rd: (n,3). Returns t (inf = miss)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return _hit_capsule_impl(ro, rd, pa, pb, ra)


def _hit_capsule_impl(ro, rd, pa, pb, ra):
    ba = pb - pa
    oa = ro - pa
    baba = ba @ ba
    bard = rd @ ba
    baoa = oa @ ba
    rdoa = rd @ oa
    oaoa = oa @ oa
    a = baba - bard * bard
    b = baba * rdoa - baoa * bard
    c = baba * oaoa - baoa * baoa - ra * ra * baba
    h = b * b - a * c
    t = np.full(rd.shape[0], np.inf)
    ok = (h >= 0) & (a > 1e-12)
    tb = np.where(ok, (-b - np.sqrt(np.maximum(h, 0))) / np.where(a > 1e-12, a, 1), np.inf)
    y = baoa + tb * bard
    body = ok & (y > 0) & (y < baba) & (tb > 0)
    t[body] = tb[body]
    # caps (spheres at the ends) for everything not hit on the body
    for end, sel in ((pa, y <= 0), (pb, y >= baba)):
        m = ~body & ok & sel
        if not m.any():
            continue
        oc = ro - end
        bb = rd[m] @ oc
        cc = oc @ oc - ra * ra
        hh = bb * bb - cc
        tt = np.where(hh > 0, -bb - np.sqrt(np.maximum(hh, 0)), np.inf)
        tt[tt <= 0] = np.inf
        t[m] = np.minimum(t[m], tt)
    return t


def _hit_sphere(ro, rd, c, r):
    oc = ro - c
    b = rd @ oc
    cc = oc @ oc - r * r
    h = b * b - cc
    t = np.where(h > 0, -b - np.sqrt(np.maximum(h, 0)), np.inf)
    t[t <= 0] = np.inf
    return t


@dataclass
class Frame:
    depth: np.ndarray  # camera-space z (inf for sky)
    ids: np.ndarray  # actor index, GROUND_ID or SKY_ID
    parts: np.ndarray  # index into PART_NAMES
    normals: np.ndarray  # world normals
    cam: Camera


PART_NAMES = ["none", "head", "neck", "torso", "arm", "hand", "leg", "foot", "weapon", "ground"]


def render(scene: SolvedScene, cam: Camera, ground: bool = True) -> Frame:
    W, H = cam.width, cam.height
    depth = np.full((H, W), np.inf)
    tbuf = np.full((H, W), np.inf)
    ids = np.full((H, W), SKY_ID, dtype=np.int32)
    parts = np.zeros((H, W), dtype=np.int32)
    normals = np.zeros((H, W, 3))
    rays = cam.rays()
    flat_rays = rays.reshape(-1, 3)
    zfac = rays @ cam.fwd  # t -> camera z

    if ground:
        denom = flat_rays[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = np.where(denom < -1e-6, -cam.eye[1] / denom, np.inf)
        tg = tg.reshape(H, W)
        m = tg < tbuf
        tbuf[m] = tg[m]
        ids[m] = GROUND_ID
        parts[m] = PART_NAMES.index("ground")
        normals[m] = (0, 1, 0)

    prims = []
    for i, a in enumerate(scene.actors):
        prims += body_prims(a, i)
    for p in prims:
        # projected bounding box of the primitive
        pts = np.array([p.a, p.b])
        uvz = cam.project(pts)
        if (uvz[:, 2] <= 0.01).all():
            continue
        z = max(uvz[:, 2].min(), 0.05)
        rad = p.r * cam.f / z + 2
        x0 = int(max(0, np.floor(uvz[:, 0].min() - rad)))
        x1 = int(min(W, np.ceil(uvz[:, 0].max() + rad)))
        y0 = int(max(0, np.floor(uvz[:, 1].min() - rad)))
        y1 = int(min(H, np.ceil(uvz[:, 1].max() + rad)))
        if x0 >= x1 or y0 >= y1:
            continue
        rd = rays[y0:y1, x0:x1].reshape(-1, 3)
        if p.kind == "cap":
            t = _hit_capsule(cam.eye, rd, p.a, p.b, p.r)
        else:
            t = _hit_sphere(cam.eye, rd, p.a, p.r)
        t = t.reshape(y1 - y0, x1 - x0)
        sub = tbuf[y0:y1, x0:x1]
        m = t < sub
        if not m.any():
            continue
        sub[m] = t[m]
        ids[y0:y1, x0:x1][m] = p.owner
        parts[y0:y1, x0:x1][m] = PART_NAMES.index(p.part)
        hit = cam.eye + rd.reshape(y1 - y0, x1 - x0, 3)[m] * t[m][:, None]
        if p.kind == "cap":
            ba = p.b - p.a
            s = np.clip(((hit - p.a) @ ba) / max(ba @ ba, 1e-12), 0, 1)
            closest = p.a + s[:, None] * ba
        else:
            closest = np.broadcast_to(p.a, hit.shape)
        n = hit - closest
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        normals[y0:y1, x0:x1][m] = n
    finite = np.isfinite(tbuf)
    depth[finite] = (tbuf * zfac)[finite]
    return Frame(depth, ids, parts, normals, cam)


SOFT_TARGETS = {"spine", "pelvis", "chest"}  # body parts that visibly give under a blow


def apply_dents(scene: SolvedScene, frame: Frame, depth_m: float = 0.045, sigma_m: float = 0.075) -> list:
    """Push the victim's surface in around a blow's contact point (plus a small bulge ring).

    Strikes into soft tissue (stomach, sides) then show up in the depth control as a real
    deformation of the body under the fist/palm/knee instead of two shapes merely overlapping."""
    cam = frame.cam
    H, W = frame.depth.shape
    yy, xx = np.mgrid[0:H, 0:W]
    names = [a.name for a in scene.actors]
    dents = []
    for a in scene.actors:
        if a.out is None or a.out.impact is None or a.target not in names:
            continue
        v = scene.actors[names.index(a.target)]
        p = a.to_world(a.out.impact)
        part = min(("head", "neck", "chest", "spine", "pelvis"), key=lambda k: np.linalg.norm(v.joints[k] - p))
        if part not in SOFT_TARGETS:
            continue
        u, vv, z = cam.project(p[None])[0]
        if z <= 0:
            continue
        vi = names.index(v.name)
        r_px = sigma_m * cam.f / z
        d = np.hypot(xx - u, yy - vv) / r_px  # in units of sigma
        own = frame.ids == vi
        prof = np.exp(-d ** 2) - 0.25 * np.exp(-((d - 1.8) / 0.45) ** 2)
        frame.depth[own] += (depth_m * prof)[own]
        dents.append((v.name, part, (float(u), float(vv))))
    return dents


def export_stage(scene: SolvedScene, cam: Camera) -> dict:
    """The solved stage as plain data (for web viewers): primitives per actor + the camera."""
    actors = []
    for i, a in enumerate(scene.actors):
        prims = [{"k": p.kind, "a": [round(float(x), 4) for x in p.a], "b": [round(float(x), 4) for x in p.b],
                  "r": round(float(p.r), 4), "part": p.part} for p in body_prims(a, i)]
        actors.append({"name": a.name, "action": a.action, "target": a.target, "prims": prims,
                       "joints": {k: [round(float(x), 4) for x in v] for k, v in a.joints.items()}})
    return {"actors": actors,
            "camera": {"eye": [float(x) for x in cam.eye], "fwd": [float(x) for x in cam.fwd],
                       "up": [float(x) for x in cam.up], "fov": cam.fov, "aspect": cam.width / cam.height}}
