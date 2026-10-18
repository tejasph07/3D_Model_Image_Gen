"""Control images from a rendered stage: OpenPose, depth, regions, debug views."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .raster import GROUND_ID, SKY_ID, Frame
from .stage import SolvedScene

# COCO-18 order used by the OpenPose ControlNets
COCO = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
        "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def coco_points(actor) -> np.ndarray:
    J = actor.joints
    pts = []
    for k in COCO:
        if k == "neck":
            pts.append((J["l_shoulder"] + J["r_shoulder"]) / 2)
        elif k.endswith("_hip"):  # OpenPose hips sit a little wider / higher than the joint
            pts.append(J[k] + (J[k] - J["pelvis"]) * 0.25)
        else:
            pts.append(J[k])
    return np.array(pts)


def keypoints(scene: SolvedScene, frame: Frame, occlusion_tol: float = 0.12) -> list[np.ndarray]:
    """Per actor (18, 3) pixel keypoints with a visibility flag in column 2."""
    cam = frame.cam
    H, W = frame.depth.shape
    out = []
    for i, a in enumerate(scene.actors):
        pts = coco_points(a)
        uvz = cam.project(pts)
        vis = np.ones(len(pts), bool)
        to_cam = cam.eye - a.joints["head"]
        face = a.joints["nose"] - a.joints["head"]
        facing = np.dot(face / np.linalg.norm(face), to_cam / np.linalg.norm(to_cam))
        for k, (u, vv, z) in enumerate(uvz):
            if z <= 0 or not (0 <= u < W and 0 <= vv < H):
                vis[k] = False
                continue
            name = COCO[k]
            if name in ("nose", "r_eye", "l_eye") and facing < -0.15:
                vis[k] = False
                continue
            if name.endswith("_ear"):
                side = a.joints[name] - a.joints["head"]
                if np.dot(side / np.linalg.norm(side), to_cam / np.linalg.norm(to_cam)) < -0.35:
                    vis[k] = False
                    continue
            # occluded by another character?
            x, y = int(u), int(vv)
            win = frame.ids[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            dwin = frame.depth[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            other = (win >= 0) & (win != i) & (dwin < z - occlusion_tol)
            if other.sum() >= 5:
                vis[k] = False
        out.append(np.concatenate([uvz[:, :2], vis[:, None]], axis=1))
    return out


def draw_openpose(kps: list[np.ndarray], W: int, H: int) -> Image.Image:
    canvas = np.zeros((H, W, 3), np.uint8)
    scale = min(W, H) / 512
    stick = max(2, int(round(4 * scale)))
    rad = max(2, int(round(4 * scale)))
    for kp in kps:
        for li, (a, b) in enumerate(LIMBS):
            if not (kp[a, 2] and kp[b, 2]):
                continue
            (x1, y1), (x2, y2) = kp[a, :2], kp[b, :2]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            length = np.hypot(x1 - x2, y1 - y2)
            angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stick), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in COLORS[li]])
        for k in range(18):
            if kp[k, 2]:
                cv2.circle(canvas, (int(kp[k, 0]), int(kp[k, 1])), rad, COLORS[k], thickness=-1)
    return Image.fromarray(canvas)


def depth_image(frame: Frame, far_factor: float = 2.2) -> Image.Image:
    """Inverse depth, near = white (MiDaS / Depth-Anything convention). Sky = black."""
    d = frame.depth
    body = frame.ids >= 0
    finite = np.isfinite(d)
    if body.any():
        zb = d[body]
        near, far = zb.min(), zb.max() * far_factor
    else:
        near, far = d[finite].min(), d[finite].max()
    disp = np.zeros_like(d)
    disp[finite] = 1.0 / np.clip(d[finite], near, far)
    lo, hi = 1.0 / far, 1.0 / near
    img = np.clip((disp - lo) / (hi - lo), 0, 1)
    img[~finite] = 0
    img = img ** 0.8
    return Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")


def normal_image(frame: Frame) -> Image.Image:
    cam = frame.cam
    n = frame.normals
    # to camera space; BAE / normal-bae convention: x right, y up, z toward viewer
    nc = np.stack([n @ cam.right, n @ cam.up, -(n @ cam.fwd)], axis=-1)
    img = ((nc * 0.5 + 0.5) * 255).astype(np.uint8)
    img[frame.ids == SKY_ID] = (127, 127, 255)
    return Image.fromarray(img)


def regions(scene: SolvedScene, frame: Frame, grow: float = 0.09) -> np.ndarray:
    """Per-actor ownership masks (n, H, W) float in [0,1], grown to cover hair/clothes.

    Each pixel belongs to at most one actor: its own body pixels first, then the nearest
    actor whose grown silhouette reaches it.
    """
    H, W = frame.ids.shape
    n = len(scene.actors)
    dist = np.full((n, H, W), np.inf, np.float32)
    reach = np.zeros(n)
    for i in range(n):
        body = (frame.ids == i).astype(np.uint8)
        if body.sum() == 0:
            continue
        ys = np.nonzero(body.any(1))[0]
        height_px = ys.max() - ys.min() + 1
        reach[i] = max(6.0, grow * height_px)
        dist[i] = cv2.distanceTransform(1 - body, cv2.DIST_L2, 5)
    owner = np.argmin(dist, axis=0)
    best = np.take_along_axis(dist, owner[None], 0)[0]
    ok = best <= reach[owner]
    ids = frame.ids
    owner = np.where(ids >= 0, ids, owner)
    ok = ok | (ids >= 0)
    masks = np.zeros((n, H, W), np.float32)
    for i in range(n):
        m = ((owner == i) & ok).astype(np.float32)
        masks[i] = cv2.GaussianBlur(m, (0, 0), max(1.0, min(H, W) / 256))
    return masks


def debug_view(scene: SolvedScene, frame: Frame, kps: list[np.ndarray], masks: np.ndarray) -> Image.Image:
    depth = np.asarray(depth_image(frame)).astype(np.float32) / 255
    H, W = frame.ids.shape
    pal = np.array([(230, 60, 60), (60, 120, 240), (60, 200, 90), (220, 180, 40), (200, 80, 220), (40, 210, 210)],
                   np.float32) / 255
    col = np.zeros((H, W, 3), np.float32)
    for i in range(len(scene.actors)):
        col += masks[i][..., None] * pal[i % len(pal)]
    base = depth * 0.6 + col * 0.4
    pose = np.asarray(draw_openpose(kps, W, H)).astype(np.float32) / 255
    has = pose.max(-1, keepdims=True) > 0.05
    img = np.where(has, pose, base)
    img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = cv2.putText
    arr = np.asarray(img).copy()
    for i, a in enumerate(scene.actors):
        kp = kps[i]
        x, y = kp[1, :2] if kp[1, 2] else kp[kp[:, 2] > 0][0, :2] if (kp[:, 2] > 0).any() else (10, 20 + 20 * i)
        d(arr, f"{a.name}:{a.action}", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
          tuple(int(c * 255) for c in pal[i % len(pal)]), 1, cv2.LINE_AA)
    return Image.fromarray(arr)


# ----------------------------------------------------------------- colour blocking
COLOR_RGB = {
    "black": (28, 28, 34), "silver": (200, 202, 212), "white": (238, 238, 242), "blonde": (240, 214, 120),
    "red": (196, 36, 40), "pink": (240, 150, 185), "blue": (50, 90, 200), "light blue": (140, 190, 240),
    "purple": (120, 60, 170), "green": (50, 150, 80), "brown": (110, 70, 40), "orange": (235, 130, 40),
    "aqua": (60, 200, 200), "gold": (212, 175, 55), "yellow": (240, 210, 60), "grey": (130, 132, 140),
    "dark blue": (25, 40, 110), "golden": (220, 180, 60),
}
SKIN = {"": (250, 222, 200), "tan": (215, 165, 120), "dark skin": (140, 95, 65)}
BACKDROPS = [  # (keywords, sky colour, ground colour)
    (("fire", "burning", "lava", "inferno"), (90, 30, 25), (60, 35, 30)),
    (("night", "neon", "moon", "dark"), (25, 30, 60), (40, 42, 55)),
    (("rain", "storm"), (45, 55, 70), (50, 55, 62)),
    (("snow", "ice", "winter"), (200, 215, 235), (225, 230, 240)),
    (("sunset", "dusk", "evening"), (230, 140, 90), (120, 90, 80)),
    (("forest", "jungle"), (90, 140, 90), (80, 100, 60)),
    (("desert", "sand", "colosseum", "arena"), (170, 200, 235), (205, 180, 135)),
]


def _rgb(name: str, default=(128, 128, 128)):
    return np.array(COLOR_RGB.get(name.strip().lower(), default), np.float32)


def color_blocks(scene: SolvedScene, frame: Frame, chars, setting: str = "") -> Image.Image:
    """Flat-shaded 3D render in each character's own colours (hair, skin, outfit palette).

    Used as the init image of the base pass, so colours start out attached to the right body."""
    from .characters import PALETTES
    from .raster import PART_NAMES
    cam = frame.cam
    H, W = frame.ids.shape
    sky, ground = (128, 128, 128), (110, 110, 110)
    s = setting.lower()
    for keys, sk, gr in BACKDROPS:
        if any(k in s for k in keys):
            sky, ground = sk, gr
            break
    yy = np.linspace(0, 1, H)[:, None, None]
    img = np.broadcast_to(np.array(sky, np.float32) * (1.1 - 0.3 * yy), (H, W, 3)).copy()
    img[frame.ids == GROUND_ID] = ground
    rays = cam.rays()
    with np.errstate(invalid="ignore"):
        pts = cam.eye + rays * (frame.depth / (rays @ cam.fwd))[..., None]
    light = np.array([-0.4, 0.8, 0.45])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(frame.normals @ light, 0, 1)
    for i, (a, c) in enumerate(zip(scene.actors, chars)):
        own = frame.ids == i
        if not own.any():
            continue
        prim, sec = PALETTES.get(c.palette, ("grey", "black"))
        hair = _rgb(c.hair_color.replace(" hair", ""))
        skin = np.array(SKIN.get(c.skin, SKIN[""]), np.float32)
        cols = {"torso": _rgb(prim), "arm": _rgb(prim), "leg": _rgb(sec), "foot": _rgb(sec) * 0.6,
                "neck": skin, "hand": skin, "weapon": np.array((185, 190, 200), np.float32)}
        for part, col in cols.items():
            m = own & (frame.parts == PART_NAMES.index(part))
            img[m] = col
        head = own & (frame.parts == PART_NAMES.index("head"))
        if head.any():
            J = a.joints
            fwd = (J["nose"] - J["head"]) / np.linalg.norm(J["nose"] - J["head"])
            up = (J["head_top"] - J["head"]) / np.linalg.norm(J["head_top"] - J["head"])
            rel = pts[head] - J["head"]
            r = a.char.body.head_r
            face = (rel @ fwd > 0.3 * r) & (rel @ up < 0.3 * r)
            hc = np.where(face[:, None], skin, hair)
            img[head] = hc
        img[own] *= shade[own][:, None]
    img = cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), max(1.0, min(W, H) / 400))
    return Image.fromarray(img.astype(np.uint8))
