"""Injuries on the 3D bodies: named spot + kind -> what the renderer needs to paint it on the skin.

The renderer finds the spot on the real skinned mesh and stores it in the mesh's rest pose, so a wound stays
on exactly the same patch of skin in every later pose and from every camera; its pattern comes from the seed
(the same bruise every time it is drawn). `age` heals it: 0 fresh .. 1 healed (bruises yellow, cuts close
to pale lines, swelling goes down).
"""
from __future__ import annotations

import numpy as np

from . import keyframes as K

KINDS = ("bruise", "cut", "bleeding_cut", "scar", "bandage", "swelling")
# a hit with this verb on this kind of spot leaves this (kind, radius m)
HEAD = ("chin", "jaw_l", "jaw_r", "temple_l", "temple_r", "nose", "ear_l", "ear_r", "eyes", "face")


def mark(S, actor, spot, kind="bruise", radius=0.03, seed=0, age=0.0, along=None):
    """Renderer injury for `actor`'s `spot` (any site name). Given on the skeleton's rest pose (bone, offset,
    outward normal, cut direction), so it means the same skin point in every pose; the current pose only
    supplies the surface normal there (turned back to rest)."""
    assert kind in KINDS, kind
    bone, off = S.specs[actor].sites[spot]
    n = K._normal(S, K.Touch(actor, spot, actor, spot))
    R = S.d.xmat[S.m.body(f"{actor}:{bone}").id].reshape(3, 3)  # body rotation from rest
    nr = R.T @ n
    if along is None:  # cuts run across the limb (round it), by default
        ax = np.asarray(off, float) if np.linalg.norm(off) > 0.05 else np.array([0.0, 1.0, 0.0])
        a = np.cross(ax / np.linalg.norm(ax), nr)
        along = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0, 0])
    else:
        along = R.T @ np.asarray(along, float)
    return {"bone": bone, "offset": [float(x) for x in off], "normal": nr.tolist(), "along": [float(x) for x in along],
            "down": (R.T @ np.array([0.0, -1.0, 0.0])).tolist(),  # where blood runs in this frame
            "kind": kind, "radius": radius, "seed": seed, "age": age}


def from_hit(verb, spot):
    """What a landed blow leaves: (kind, radius)."""
    if spot in ("chin", "nose"):
        return ("bleeding_cut", 0.018) if verb in ("punch", "kick") else ("bruise", 0.025)
    if spot in HEAD:
        return ("swelling", 0.03) if verb in ("punch", "kick", "knee") else ("bruise", 0.025)
    return ("bruise", 0.045 if verb in ("kick", "knee", "stamp") else 0.035)


class Ledger:
    """Who carries which injuries. Injuries persist across panels; `heal` ages them, `treat` bandages one."""

    def __init__(self):
        self.items = {}  # actor -> [dict(spot, kind, radius, seed, age)]

    def add(self, actor, spot, kind, radius=0.03, age=0.0):
        lst = self.items.setdefault(actor, [])
        for w in lst:  # hit again on the same spot: refresh it instead of stacking
            if w["spot"] == spot and w["kind"] == kind:
                w["age"] = 0.0
                return
        lst.append(dict(spot=spot, kind=kind, radius=radius, seed=len(lst) * 3 + hash(actor) % 97, age=age))

    def heal(self, amount):
        for lst in self.items.values():
            for w in lst:
                w["age"] = min(1.0, w["age"] + amount)

    def treat(self, actor, spot):
        for w in self.items.get(actor, []):
            if w["spot"] == spot:
                w["kind"], w["radius"] = "bandage", max(w["radius"], 0.035)

    def render(self, S, actor):
        return [mark(S, actor, w["spot"], w["kind"], w["radius"], w["seed"], w["age"]) for w in self.items.get(actor, [])]
