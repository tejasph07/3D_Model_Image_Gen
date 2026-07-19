"""Common-object library: parametric props with named anchors and collision shapes.

Every object is a rigid body with position (x, y, z), yaw, and uniform scale.
Local frame: +Z forward (the side a user faces / approaches), +Y up.

Anchors are named local-space points used by the DSL (sit_on -> "seat",
hold -> "grip", on_top -> "top", ...). Shapes are simple primitives used for
rendering and collision: ("box", center, half_extents), ("sphere", center, r),
("cyl", p0, p1, r) - all in local space.
"""
from __future__ import annotations

import numpy as np

from .vecmath import V, rot_y, normalize


class ObjectSpec:
    def __init__(self, name, anchors, shapes, top_y, radius, sit_anchor=None,
                 grip_anchor=None, rest_y=0.0, ride_axis=None, portable=False):
        self.name = name
        self.portable = portable  # light enough to be held in one hand
        self.anchors = anchors          # name -> local point
        self.shapes = shapes
        self.top_y = top_y              # height of the top surface (for on_top)
        self.radius = radius            # bounding radius in XZ (for placement)
        self.sit_anchor = sit_anchor    # anchor name used by sit_on
        self.grip_anchor = grip_anchor  # default anchor for hold/grab
        self.rest_y = rest_y            # root y when resting on the ground
        self.ride_axis = ride_axis      # (anchor_a, anchor_b) straddle axis


def _box(cx, cy, cz, hx, hy, hz):
    return ("box", V(cx, cy, cz), V(hx, hy, hz))


def _specs():
    S = {}
    S["CHAIR"] = ObjectSpec(
        "CHAIR",
        anchors={
            "seat": V(0, 0.45, 0.0), "seat_front": V(0, 0.45, 0.14),
            "backrest": V(0, 0.72, -0.20), "back_top": V(0, 0.86, -0.21),
            "edge": V(0, 0.45, 0.21), "top": V(0, 0.86, -0.21),
            "left_side": V(-0.23, 0.45, 0), "right_side": V(0.23, 0.45, 0),
        },
        shapes=[
            _box(0, 0.425, 0, 0.22, 0.025, 0.22),
            _box(0, 0.66, -0.21, 0.22, 0.20, 0.02),
            _box(-0.19, 0.20, -0.19, 0.02, 0.20, 0.02),
            _box(0.19, 0.20, -0.19, 0.02, 0.20, 0.02),
            _box(-0.19, 0.20, 0.19, 0.02, 0.20, 0.02),
            _box(0.19, 0.20, 0.19, 0.02, 0.20, 0.02),
        ],
        top_y=0.45, radius=0.32, sit_anchor="seat", grip_anchor="back_top")
    S["STOOL"] = ObjectSpec(
        "STOOL",
        anchors={"seat": V(0, 0.45, 0), "top": V(0, 0.45, 0), "edge": V(0, 0.45, 0.16)},
        shapes=[_box(0, 0.435, 0, 0.17, 0.02, 0.17),
                _box(-0.13, 0.21, -0.13, 0.02, 0.21, 0.02),
                _box(0.13, 0.21, -0.13, 0.02, 0.21, 0.02),
                _box(-0.13, 0.21, 0.13, 0.02, 0.21, 0.02),
                _box(0.13, 0.21, 0.13, 0.02, 0.21, 0.02)],
        top_y=0.45, radius=0.24, sit_anchor="seat", grip_anchor="edge")
    S["BENCH"] = ObjectSpec(
        "BENCH",
        anchors={"seat": V(0, 0.45, 0), "seat_left": V(-0.5, 0.45, 0),
                 "seat_right": V(0.5, 0.45, 0), "top": V(0, 0.45, 0),
                 "edge": V(0, 0.45, 0.15)},
        shapes=[_box(0, 0.43, 0, 0.75, 0.025, 0.18),
                _box(-0.65, 0.2, 0, 0.03, 0.20, 0.15),
                _box(0.65, 0.2, 0, 0.03, 0.20, 0.15)],
        top_y=0.45, radius=0.78, sit_anchor="seat", grip_anchor="edge",
        ride_axis=("seat_left", "seat_right"))
    S["TABLE"] = ObjectSpec(
        "TABLE",
        anchors={"top": V(0, 0.75, 0), "edge": V(0, 0.75, 0.35),
                 "far_edge": V(0, 0.75, -0.35), "left_edge": V(-0.6, 0.75, 0),
                 "right_edge": V(0.6, 0.75, 0), "under": V(0, 0.35, 0),
                 "corner": V(0.6, 0.75, 0.35)},
        shapes=[_box(0, 0.735, 0, 0.60, 0.02, 0.35),
                _box(-0.55, 0.36, -0.30, 0.025, 0.36, 0.025),
                _box(0.55, 0.36, -0.30, 0.025, 0.36, 0.025),
                _box(-0.55, 0.36, 0.30, 0.025, 0.36, 0.025),
                _box(0.55, 0.36, 0.30, 0.025, 0.36, 0.025)],
        top_y=0.755, radius=0.70, grip_anchor="edge")
    S["DOOR"] = ObjectSpec(
        "DOOR",
        anchors={"knob": V(0.35, 1.0, 0.04), "handle": V(0.35, 1.0, 0.04),
                 "top": V(0, 2.0, 0), "center": V(0, 1.0, 0),
                 "edge": V(0.45, 1.0, 0)},
        shapes=[_box(0, 1.0, 0, 0.45, 1.0, 0.025),
                ("sphere", V(0.35, 1.0, 0.045), 0.035)],
        top_y=2.0, radius=0.48, grip_anchor="knob")
    S["BALL"] = ObjectSpec(
        "BALL",
        anchors={"center": V(0, 0.11, 0), "top": V(0, 0.22, 0)},
        shapes=[("sphere", V(0, 0.11, 0), 0.11)],
        top_y=0.22, radius=0.11, grip_anchor="center", rest_y=0.0, portable=True)
    S["STICK"] = ObjectSpec(  # broom / staff / pole: axis along local Z
        "STICK",
        anchors={"middle": V(0, 0.02, 0), "front": V(0, 0.02, 0.55),
                 "back": V(0, 0.02, -0.55), "tip": V(0, 0.02, 0.70),
                 "end": V(0, 0.02, -0.70), "grip": V(0, 0.02, 0.35)},
        shapes=[("cyl", V(0, 0.02, -0.70), V(0, 0.02, 0.70), 0.022)],
        top_y=0.04, radius=0.72, grip_anchor="grip",
        ride_axis=("back", "front"), portable=True)
    S["SWORD"] = ObjectSpec(
        "SWORD",
        anchors={"handle": V(0, 0.02, -0.42), "grip": V(0, 0.02, -0.42),
                 "middle": V(0, 0.02, 0.05), "tip": V(0, 0.02, 0.55)},
        shapes=[("cyl", V(0, 0.02, -0.50), V(0, 0.02, -0.34), 0.018),
                ("box", V(0, 0.02, -0.32), V(0.09, 0.012, 0.012)),
                ("cyl", V(0, 0.02, -0.30), V(0, 0.02, 0.55), 0.020)],
        top_y=0.05, radius=0.55, grip_anchor="handle", portable=True)
    S["ROPE"] = ObjectSpec(  # taut rope segment, axis along Z
        "ROPE",
        anchors={"middle": V(0, 0.01, 0), "end_a": V(0, 0.01, -1.0),
                 "end_b": V(0, 0.01, 1.0), "grip": V(0, 0.01, 0.0)},
        shapes=[("cyl", V(0, 0.01, -1.0), V(0, 0.01, 1.0), 0.015)],
        top_y=0.03, radius=1.02, grip_anchor="grip", ride_axis=("end_a", "end_b"))
    S["BED"] = ObjectSpec(
        "BED",
        anchors={"top": V(0, 0.50, 0), "pillow": V(0, 0.53, -0.75),
                 "foot_end": V(0, 0.50, 0.9), "edge": V(0.65, 0.50, 0)},
        shapes=[_box(0, 0.30, 0, 0.70, 0.20, 1.0),
                _box(0, 0.52, -0.78, 0.55, 0.05, 0.18),
                _box(0, 0.65, -0.98, 0.70, 0.25, 0.02)],
        top_y=0.50, radius=1.15, sit_anchor="edge")
    S["SOFA"] = ObjectSpec(
        "SOFA",
        anchors={"seat": V(0, 0.42, 0.05), "seat_left": V(-0.55, 0.42, 0.05),
                 "seat_right": V(0.55, 0.42, 0.05), "backrest": V(0, 0.70, -0.30),
                 "armrest_left": V(-0.85, 0.55, 0.05),
                 "armrest_right": V(0.85, 0.55, 0.05), "top": V(0, 0.75, -0.30)},
        shapes=[_box(0, 0.21, 0.05, 0.90, 0.21, 0.35),
                _box(0, 0.55, -0.32, 0.90, 0.28, 0.07),
                _box(-0.85, 0.38, 0.02, 0.07, 0.38, 0.38),
                _box(0.85, 0.38, 0.02, 0.07, 0.38, 0.38)],
        top_y=0.42, radius=0.95, sit_anchor="seat", grip_anchor="armrest_right")
    S["BOX"] = ObjectSpec(
        "BOX",
        anchors={"top": V(0, 0.5, 0), "edge": V(0, 0.5, 0.25), "center": V(0, 0.25, 0)},
        shapes=[_box(0, 0.25, 0, 0.25, 0.25, 0.25)],
        top_y=0.5, radius=0.36, sit_anchor="top", grip_anchor="edge")
    S["CUP"] = ObjectSpec(
        "CUP",
        anchors={"handle": V(0.05, 0.05, 0), "top": V(0, 0.10, 0), "center": V(0, 0.05, 0)},
        shapes=[("cyl", V(0, 0.0, 0), V(0, 0.10, 0), 0.04)],
        top_y=0.10, radius=0.06, grip_anchor="center", portable=True)
    S["BOOK"] = ObjectSpec(
        "BOOK",
        anchors={"top": V(0, 0.04, 0), "center": V(0, 0.02, 0), "spine": V(-0.09, 0.02, 0)},
        shapes=[_box(0, 0.02, 0, 0.09, 0.02, 0.12)],
        top_y=0.04, radius=0.15, grip_anchor="center", portable=True)
    S["WALL"] = ObjectSpec(
        "WALL",
        anchors={"center": V(0, 1.25, 0), "top": V(0, 2.5, 0), "base": V(0, 0, 0)},
        shapes=[_box(0, 1.25, 0, 1.5, 1.25, 0.10)],
        top_y=2.5, radius=1.55, grip_anchor="center")
    S["ROCK"] = ObjectSpec(
        "ROCK",
        anchors={"top": V(0, 0.55, 0), "center": V(0, 0.30, 0)},
        shapes=[("sphere", V(0, 0.28, 0), 0.30)],
        top_y=0.55, radius=0.32, sit_anchor="top", grip_anchor="top")
    S["LANTERN"] = ObjectSpec(
        "LANTERN",
        anchors={"handle": V(0, 0.28, 0), "top": V(0, 0.26, 0), "center": V(0, 0.13, 0)},
        shapes=[_box(0, 0.13, 0, 0.08, 0.11, 0.08),
                ("cyl", V(-0.08, 0.26, 0), V(0.08, 0.26, 0), 0.012)],
        top_y=0.26, radius=0.12, grip_anchor="handle", portable=True)
    return S


OBJECT_SPECS = _specs()
ALIASES = {
    "BROOM": "STICK", "STAFF": "STICK", "POLE": "STICK", "BAT": "STICK",
    "WAND": "STICK", "SPEAR": "STICK",
    "COUCH": "SOFA", "SEAT": "CHAIR", "DESK": "TABLE", "CRATE": "BOX",
    "STONE": "ROCK", "BOULDER": "ROCK", "FOOTBALL": "BALL", "QUAFFLE": "BALL",
    "MUG": "CUP", "TORCH": "LANTERN", "LAMP": "LANTERN",
}


def resolve_type(kind: str) -> str:
    k = kind.upper()
    k = ALIASES.get(k, k)
    if k not in OBJECT_SPECS:
        raise ValueError(
            f"unknown object type '{kind}' (choose from "
            f"{sorted(set(OBJECT_SPECS) | set(ALIASES))})")
    return k


class Prop:
    """An instance of an object in the scene."""

    def __init__(self, name: str, kind: str, scale: float = 1.0):
        self.name = name
        self.kind = resolve_type(kind)
        self.spec = OBJECT_SPECS[self.kind]
        self.scale = scale
        self.x = 0.0
        self.z = 0.0
        self.y = self.spec.rest_y * scale  # root height offset (hover lifts this)
        self.yaw = 0.0
        self.pitch = 0.0
        self.placed = False
        self.attached_to = None  # (human_name, side) when held with no placement
        self.warnings: list[str] = []

    def frame(self) -> np.ndarray:
        from .vecmath import rot_x
        return rot_y(self.yaw) @ rot_x(self.pitch)

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return V(self.x, self.y, self.z) + self.frame() @ (local * self.scale)

    def anchor(self, name: str) -> np.ndarray:
        if name not in self.spec.anchors:
            raise ValueError(
                f"unknown anchor '{name}' on {self.name} ({self.kind}); "
                f"has {sorted(self.spec.anchors)}")
        return self.to_world(self.spec.anchors[name])

    def default_grip(self) -> str:
        return self.spec.grip_anchor or "top"

    def part_point(self, part: str):
        """(center, radius) like Human.part_point, using anchors."""
        return self.anchor(part), 0.03 * self.scale

    def top_y_world(self) -> float:
        return self.y + self.spec.top_y * self.scale

    def bounding_radius(self) -> float:
        return self.spec.radius * self.scale

    def world_shapes(self):
        F = self.frame()
        origin = V(self.x, self.y, self.z)
        out = []
        for shp in self.spec.shapes:
            if shp[0] == "box":
                out.append(("box", origin + F @ (shp[1] * self.scale),
                            shp[2] * self.scale, F))
            elif shp[0] == "sphere":
                out.append(("sphere", origin + F @ (shp[1] * self.scale),
                            shp[2] * self.scale))
            else:
                out.append(("cyl", origin + F @ (shp[1] * self.scale),
                            origin + F @ (shp[2] * self.scale), shp[3] * self.scale))
        return out

    def lowest_point(self) -> float:
        lows = []
        for shp in self.world_shapes():
            if shp[0] == "box":
                _, c, h, F = shp
                corners = [c + F @ (h * np.array(sgn)) for sgn in
                           [(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]]
                lows.extend(p[1] for p in corners)
            elif shp[0] == "sphere":
                lows.append(shp[1][1] - shp[2])
            else:
                lows.append(min(shp[1][1], shp[2][1]) - shp[3])
        return min(lows)
