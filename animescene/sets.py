"""3D set dressing for the full-3D renderer, as plain data.

A set is a list of primitives (box / cylinder / plane / gltf prop), lights and a sky gradient,
laid out around the action and oriented to the camera so walls, ropes and signs frame the shot.
Add a set by writing one function here; add props by dropping .glb files in ASSETS/props and
referencing them as {"type": "gltf", "url": "/assets/props/<file>.glb", ...}.
"""
from __future__ import annotations

import numpy as np

SETS = {}


def set_builder(*keywords):
    def deco(fn):
        SETS[fn.__name__] = (keywords, fn)
        return fn
    return deco


def _frame(center, cam):
    f = np.array([cam.fwd[0], 0.0, cam.fwd[2]])
    f /= max(np.linalg.norm(f), 1e-6)
    r = np.array([-f[2], 0.0, f[0]])  # camera right on the ground
    c = np.array([center[0], 0.0, center[2]])
    return c, f, r


def _P(c, f, r, x, y, z):  # set-local (right, up, depth) -> world
    p = c + r * x + f * z
    return [float(p[0]), float(y), float(p[2])]


def _yaw(f):
    return float(np.arctan2(f[0], f[2]))


@set_builder("alley", "back alley", "street", "neon")
def alley(center, cam, spec):
    c, f, r = _frame(center, cam)
    yaw = _yaw(f)
    P = lambda x, y, z: _P(c, f, r, x, y, z)  # noqa: E731
    prims = [
        {"type": "plane", "size": [9, 60], "pos": P(0, 0.001, 10), "rotY": yaw, "color": "#555a68", "rough": 0.3},
        {"type": "box", "size": [0.4, 9, 60], "pos": P(-3.4, 4.5, 10), "rotY": yaw, "color": "#4a3f45"},
        {"type": "box", "size": [0.4, 9, 60], "pos": P(3.4, 4.5, 10), "rotY": yaw, "color": "#433a44"},
        {"type": "box", "size": [7, 9, 0.4], "pos": P(0, 4.5, 26), "rotY": yaw, "color": "#2b2a35"},
        # vending machines (lit panels), a dumpster, pipes, signs
        {"type": "box", "size": [0.8, 1.85, 0.75], "pos": P(-2.75, 0.93, 3.0), "rotY": yaw, "color": "#b8202c"},
        {"type": "box", "size": [0.05, 1.1, 0.6], "pos": P(-2.33, 1.15, 3.0), "rotY": yaw, "color": "#ffffff",
         "emissive": "#dff3ff", "emissiveIntensity": 1.6},
        {"type": "box", "size": [0.8, 1.85, 0.75], "pos": P(-2.75, 0.93, 3.85), "rotY": yaw, "color": "#2457b8"},
        {"type": "box", "size": [0.05, 1.1, 0.6], "pos": P(-2.33, 1.15, 3.85), "rotY": yaw, "color": "#ffffff",
         "emissive": "#fff2c8", "emissiveIntensity": 1.5},
        {"type": "box", "size": [1.0, 1.1, 1.7], "pos": P(2.7, 0.55, 5.5), "rotY": yaw, "color": "#2f5a3a"},
        {"type": "cylinder", "size": [0.07, 8.5], "pos": P(3.15, 4.25, 1.2), "color": "#5a5e66"},
        {"type": "cylinder", "size": [0.05, 8.5], "pos": P(-3.15, 4.25, 7.0), "color": "#5a5e66"},
        {"type": "box", "size": [0.08, 2.4, 0.7], "pos": P(-3.15, 3.6, 6.0), "rotY": yaw, "color": "#ff3fa8",
         "emissive": "#ff3fa8", "emissiveIntensity": 2.4},
        {"type": "box", "size": [0.08, 0.7, 1.9], "pos": P(3.15, 3.1, 8.5), "rotY": yaw, "color": "#38e0ff",
         "emissive": "#38e0ff", "emissiveIntensity": 2.4},
        {"type": "box", "size": [0.08, 1.6, 0.6], "pos": P(3.15, 4.4, 3.0), "rotY": yaw, "color": "#ffb030",
         "emissive": "#ffb030", "emissiveIntensity": 2.0},
        {"type": "plane", "size": [1.4, 0.9], "pos": P(0.6, 0.004, 1.5), "rotY": yaw, "color": "#1c2a44", "rough": 0.05},
        {"type": "plane", "size": [1.0, 1.6], "pos": P(-1.2, 0.004, 5.0), "rotY": yaw, "color": "#1c2a44", "rough": 0.05},
    ]
    lights = [
        {"type": "ambient", "color": "#9aa6d6", "intensity": 1.15},
        {"type": "dir", "color": "#e8ecff", "intensity": 1.6, "pos": P(-2, 6, -3)},
        {"type": "point", "color": "#ff3fa8", "intensity": 9, "distance": 9, "pos": P(-2.6, 3.4, 6.0)},
        {"type": "point", "color": "#38e0ff", "intensity": 9, "distance": 9, "pos": P(2.6, 3.0, 8.5)},
        {"type": "dir", "color": "#ff7ad9", "intensity": 1.2, "pos": P(1, 3, 8)},  # rim from behind
    ]
    return {"prims": prims, "lights": lights, "sky": ["#0a0d1c", "#2a2340"], "fog": ["#151629", 12, 34]}


@set_builder("ring", "boxing", "arena")
def ring(center, cam, spec):
    c, f, r = _frame(center, cam)
    yaw = _yaw(f)
    P = lambda x, y, z: _P(c, f, r, x, y, z)  # noqa: E731
    h = 3.2
    prims = [
        {"type": "plane", "size": [80, 80], "pos": P(0, -0.02, 0), "rotY": yaw, "color": "#121218"},
        {"type": "box", "size": [2 * h + 0.6, 0.06, 2 * h + 0.6], "pos": P(0, -0.03, 0), "rotY": yaw, "color": "#d9d4c6"},
        {"type": "box", "size": [2 * h + 1.2, 0.9, 2 * h + 1.2], "pos": P(0, -0.5, 0), "rotY": yaw, "color": "#1b2a6b"},
    ]
    for i, (x, z) in enumerate([(-h, -h), (h, -h), (h, h), (-h, h)]):
        prims.append({"type": "cylinder", "size": [0.09, 1.55], "pos": P(x, 0.78, z),
                      "color": "#c0272d" if i % 2 == 0 else "#2346c0"})
    for y, col in ((0.55, "#e8e8e8"), (0.95, "#c0272d"), (1.35, "#e8e8e8")):
        for (x0, z0), (x1, z1) in [((-h, -h), (h, -h)), ((h, -h), (h, h)), ((h, h), (-h, h)), ((-h, h), (-h, -h))]:
            a, b = np.array(P(x0, y, z0)), np.array(P(x1, y, z1))
            prims.append({"type": "rope", "a": a.tolist(), "b": b.tolist(), "radius": 0.025, "color": col})
    for k in range(40):  # crowd silhouettes beyond the ring
        ang = 2 * np.pi * k / 40
        rr = 7.5 + (k % 3) * 0.9
        prims.append({"type": "box", "size": [0.5, 1.3 + 0.2 * (k % 4), 0.4], "pos": P(rr * np.cos(ang), 0.7, rr * np.sin(ang)),
                      "rotY": float(ang), "color": "#1d1d26"})
    lights = [
        {"type": "ambient", "color": "#c9c3b8", "intensity": 1.2},
        {"type": "spot", "color": "#fff4e0", "intensity": 120, "pos": P(0, 9, -1), "target": P(0, 0, 0), "angle": 0.5},
        {"type": "dir", "color": "#ffffff", "intensity": 1.3, "pos": P(-3, 5, -4)},
        {"type": "dir", "color": "#8fb4ff", "intensity": 0.9, "pos": P(2, 3, 6)},
    ]
    return {"prims": prims, "lights": lights, "sky": ["#06060a", "#1b1822"], "fog": ["#0c0b12", 10, 30]}


@set_builder("plain")
def plain(center, cam, spec):
    c, f, r = _frame(center, cam)
    return {"prims": [{"type": "plane", "size": [80, 80], "pos": [float(c[0]), 0.0, float(c[2])], "color": "#8b8f99"}],
            "lights": [{"type": "ambient", "color": "#ffffff", "intensity": 1.4},
                       {"type": "dir", "color": "#ffffff", "intensity": 2.0, "pos": [-2, 5, 3]}],
            "sky": ["#9fc4ea", "#e6eef6"]}


def build_set(spec: dict, center, cam) -> dict:
    name = spec.get("set3d")
    if name is None:
        text = (spec.get("setting", "") + " " + spec.get("title", "")).lower()
        name = next((n for n, (kw, _) in SETS.items() if any(k in text for k in kw)), "plain")
    return SETS[name][1](center, cam, spec)
