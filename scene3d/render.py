"""Render the scene as an OpenPose-style skeleton image (PNG).

Bodies use the standard OpenPose BODY_25 limb colors on black; hands are drawn
with per-finger colors; objects are drawn as gray wireframes so the downstream
image model gets full grounding. Elements are depth-sorted (painter's
algorithm).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .camera import make_camera
from .skeleton import BODY_25_NAMES

# OpenPose BODY_25 rendering pairs and per-keypoint colors.
BODY_25_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
    (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16),
    (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]
BODY_25_COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255),
    (255, 0, 170), (170, 0, 255), (255, 0, 255), (85, 0, 255), (0, 0, 255),
    (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
]
FINGER_COLORS = {
    0: (255, 60, 60), 1: (255, 160, 40), 2: (80, 220, 80),
    3: (60, 130, 255), 4: (200, 80, 255),
}
OBJECT_COLOR = (140, 140, 140)
GROUND_COLOR = (55, 55, 55)


def _draw_limb(draw, p1, p2, color, width):
    draw.line([p1, p2], fill=color, width=max(1, int(width)))


def render(scene, width=768, height=768, camera=None):
    cam = camera or make_camera(scene, width, height)
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    elements = []  # (depth, kind, payload)

    # Ground grid for spatial grounding.
    grid = []
    for i in range(-6, 7):
        grid.append(((i, 0, -6), (i, 0, 6)))
        grid.append(((-6, 0, i), (6, 0, i)))
    for a, b in grid:
        pa = cam.project(np.array(a, dtype=float))
        pb = cam.project(np.array(b, dtype=float))
        if pa[2] > 0.05 and pb[2] > 0.05:
            elements.append((max(pa[2], pb[2]) + 100.0, "line",
                             ((pa[0], pa[1]), (pb[0], pb[1]), GROUND_COLOR, 1)))

    for prop in scene.props():
        for shp in prop.world_shapes():
            _project_shape(cam, shp, elements)

    for h in scene.humans():
        pts = h.body25()
        proj = [cam.project(p) for p in pts]
        for idx, (i, j) in enumerate(BODY_25_PAIRS):
            pi, pj = proj[i], proj[j]
            if pi[2] <= 0.05 or pj[2] <= 0.05:
                continue
            depth = (pi[2] + pj[2]) / 2
            w = max(2.0, cam.project_radius((pts[i] + pts[j]) / 2, 0.028 * h.scale))
            elements.append((depth, "line",
                             ((pi[0], pi[1]), (pj[0], pj[1]),
                              BODY_25_COLORS[j], w)))
        for i, p in enumerate(proj):
            if p[2] <= 0.05:
                continue
            r = max(2.0, cam.project_radius(pts[i], 0.02 * h.scale))
            elements.append((p[2] - 0.01, "dot",
                             ((p[0], p[1]), r, BODY_25_COLORS[i])))
        for side in ("L", "R"):
            hp = h.hand_pts[side]
            if hp is None:
                continue
            hproj = [cam.project(p) for p in hp]
            for f in range(5):
                col = FINGER_COLORS[f]
                chain = [0, 1 + f * 4, 2 + f * 4, 3 + f * 4, 4 + f * 4]
                for k in range(4):
                    a, b = hproj[chain[k]], hproj[chain[k + 1]]
                    if a[2] <= 0.05 or b[2] <= 0.05:
                        continue
                    elements.append(((a[2] + b[2]) / 2 - 0.02, "line",
                                     ((a[0], a[1]), (b[0], b[1]), col,
                                      max(1, cam.project_radius(hp[chain[k]], 0.008)))))

    for depth, kind, payload in sorted(elements, key=lambda e: -e[0]):
        if kind == "line":
            p1, p2, color, w = payload
            _draw_limb(draw, p1, p2, color, w)
        elif kind == "dot":
            (x, y), r, color = payload
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        elif kind == "circle":
            (x, y), r, color, w = payload
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color,
                         width=max(1, int(w)))
    return img, cam


def _project_shape(cam, shp, elements):
    if shp[0] == "box":
        _, c, h, F = shp
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(c + F @ (h * np.array([sx, sy, sz])))
        idx = lambda sx, sy, sz: ((sx + 1) // 2) * 4 + ((sy + 1) // 2) * 2 + (sz + 1) // 2
        edges = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                edges.append((idx(sx, sy, -1), idx(sx, sy, 1)))
        for sx in (-1, 1):
            for sz in (-1, 1):
                edges.append((idx(sx, -1, sz), idx(sx, 1, sz)))
        for sy in (-1, 1):
            for sz in (-1, 1):
                edges.append((idx(-1, sy, sz), idx(1, sy, sz)))
        proj = [cam.project(p) for p in corners]
        for i, j in edges:
            a, b = proj[i], proj[j]
            if a[2] <= 0.05 or b[2] <= 0.05:
                continue
            elements.append(((a[2] + b[2]) / 2 + 0.5, "line",
                             ((a[0], a[1]), (b[0], b[1]), OBJECT_COLOR, 2)))
    elif shp[0] == "sphere":
        _, c, r = shp
        p = cam.project(c)
        if p[2] > 0.05:
            pr = cam.project_radius(c, r)
            elements.append((p[2] + 0.5, "circle", ((p[0], p[1]), pr,
                                                    OBJECT_COLOR, 2)))
    else:  # cyl
        _, c0, c1, r = shp
        a, b = cam.project(c0), cam.project(c1)
        if a[2] > 0.05 and b[2] > 0.05:
            w = max(2, cam.project_radius((c0 + c1) / 2, r) * 2)
            elements.append(((a[2] + b[2]) / 2 + 0.5, "line",
                             ((a[0], a[1]), (b[0], b[1]), OBJECT_COLOR, w)))
