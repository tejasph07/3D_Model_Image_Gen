"""Procedural hands: verb-driven finger poses -> OpenPose 21-keypoint hands.

OpenPose hand layout: 0 wrist; then 4 points per finger
(thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20).

A hand is generated from a wrist position plus a hand frame
(right = across palm toward thumb side, up = palm normal, forward = wrist->fingers)
and a "finger style": per-finger extension in [0, 1] (0 = fully curled, 1 = straight).
"""
from __future__ import annotations

import math

import numpy as np

from .vecmath import normalize

# Segment lengths (m) per finger at scale 1: (metacarpal-ish base offset handled
# separately), proximal, middle, distal phalanges.
FINGER_SEGMENTS = {
    "thumb": (0.045, 0.035, 0.028),
    "index": (0.040, 0.026, 0.022),
    "middle": (0.044, 0.030, 0.024),
    "ring": (0.040, 0.028, 0.022),
    "pinky": (0.032, 0.020, 0.018),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
# Knuckle position across the palm: (right_offset, forward_offset) from wrist.
KNUCKLES = {
    "thumb": (0.035, 0.025),
    "index": (0.028, 0.085),
    "middle": (0.009, 0.090),
    "ring": (-0.010, 0.086),
    "pinky": (-0.028, 0.077),
}
HAND_LENGTH = 0.18  # wrist to middle fingertip, fully extended

# Finger styles: extension per finger, keyed by action verb family.
FINGER_STYLES = {
    "relaxed": {"thumb": 0.6, "index": 0.65, "middle": 0.6, "ring": 0.55, "pinky": 0.5},
    "open":    {"thumb": 0.95, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "touch":   {"thumb": 0.5, "index": 0.95, "middle": 1.0, "ring": 0.95, "pinky": 0.9},
    "poke":    {"thumb": 0.35, "index": 1.0, "middle": 0.15, "ring": 0.12, "pinky": 0.1},
    "pinch":   {"thumb": 0.65, "index": 0.6, "middle": 0.25, "ring": 0.2, "pinky": 0.15},
    "grab":    {"thumb": 0.45, "index": 0.45, "middle": 0.45, "ring": 0.45, "pinky": 0.45},
    "fist":    {"thumb": 0.15, "index": 0.08, "middle": 0.08, "ring": 0.08, "pinky": 0.08},
    "point":   {"thumb": 0.3, "index": 1.0, "middle": 0.1, "ring": 0.1, "pinky": 0.1},
}


def finger_style_for(verb: str) -> str:
    return {
        "touch": "touch", "press": "open", "palm": "open", "slap": "open",
        "push": "open", "high_five": "open", "poke": "poke", "point": "point",
        "pinch": "pinch", "grab": "grab", "grip": "grab", "hold": "grab",
        "pull": "grab", "squeeze": "grab", "choke": "grab", "handshake": "grab",
        "punch": "fist",
    }.get(verb, "relaxed")


def effector_offset(verb: str, part: str | None = None) -> float:
    """Distance from wrist to the part of the hand that makes contact (m).

    The named effector part wins (touching with the palm contacts at the
    palm, not the fingertips); the verb supplies a sensible default.
    """
    if part:
        sub = part.split("_", 1)[-1]
        by_part = {"fingers": 0.16, "fingertip": 0.165, "palm": 0.085,
                   "hand": 0.09, "fist": 0.09, "thumb": 0.07, "wrist": 0.02,
                   "forearm": 0.0}
        if sub in by_part:
            return by_part[sub]
    return {
        "poke": 0.165, "point": 0.165, "touch": 0.155, "pinch": 0.13,
        "grab": 0.09, "grip": 0.09, "hold": 0.09, "pull": 0.09,
        "squeeze": 0.09, "choke": 0.09, "handshake": 0.08,
        "punch": 0.09, "press": 0.08, "palm": 0.08, "slap": 0.09,
        "push": 0.08, "high_five": 0.08,
    }.get(verb, 0.12)


def hand_keypoints(wrist: np.ndarray, hand_frame: np.ndarray, side: str,
                   style: str = "relaxed", scale: float = 1.0) -> np.ndarray:
    """Return (21, 3) world-space hand keypoints.

    `hand_frame` columns: right (toward thumb for a RIGHT hand), palm normal,
    forward (wrist -> fingertips). For the left hand the thumb sits at -right.
    """
    ext = FINGER_STYLES.get(style, FINGER_STYLES["relaxed"])
    r, n, f = hand_frame[:, 0], hand_frame[:, 1], hand_frame[:, 2]
    thumb_sign = 1.0 if side == "R" else -1.0
    pts = [wrist.copy()]
    for name in FINGER_ORDER:
        koff_r, koff_f = KNUCKLES[name]
        base = wrist + (r * (thumb_sign * koff_r) + f * koff_f) * scale
        segs = FINGER_SEGMENTS[name]
        e = ext[name]
        if name == "thumb":
            # Thumb sweeps from across the palm (curled) to out+forward (open).
            open_dir = normalize(r * thumb_sign * 0.8 + f * 0.6)
            curl_dir = normalize(f * 0.5 - r * thumb_sign * 0.4 - n * 0.5)
            d0 = normalize(open_dir * e + curl_dir * (1 - e))
            curl_axis = normalize(np.cross(d0, n * -1.0 + f * 0.2))
        else:
            d0 = f.copy()
            curl_axis = r * thumb_sign  # fingers curl toward the palm (-n)
        # Per-segment curl: total flexion grows as extension drops. The curl
        # is applied before each segment (real fingers flex at the knuckle
        # first), so a fist folds tightly back toward the palm.
        total_curl = math.radians(10 + (1.0 - e) * 200.0)
        per_seg = total_curl / 3.0
        pts.append(base.copy())  # knuckle keypoint
        p = base
        d = d0
        k = normalize(curl_axis)
        c, s = math.cos(per_seg), math.sin(per_seg)
        for seg_len in segs:
            # Rodrigues rotation of d about the curl axis.
            d = normalize(d * c + np.cross(k, d) * s
                          + k * float(np.dot(k, d)) * (1 - c))
            p = p + d * seg_len * scale
            pts.append(p.copy())
    return np.array(pts)


def default_hand_frame(elbow: np.ndarray, wrist: np.ndarray, body_right: np.ndarray,
                       side: str) -> np.ndarray:
    """Relaxed hand frame: fingers continue the forearm, palm faces the body."""
    f = normalize(wrist - elbow, fallback=[0, -1, 0])
    inward = -body_right if side == "R" else body_right
    n = inward - f * float(np.dot(inward, f))
    n = normalize(n, fallback=[0, 1, 0])
    r = np.cross(n, f)
    return np.column_stack([normalize(r), n, f])


def aimed_hand_frame(aim: np.ndarray, palm_hint: np.ndarray) -> np.ndarray:
    """Hand frame with fingers along `aim` and palm normal near `palm_hint`."""
    f = normalize(aim, fallback=[0, 0, 1])
    n = palm_hint - f * float(np.dot(palm_hint, f))
    n = normalize(n, fallback=[0, 1, 0])
    r = np.cross(n, f)
    return np.column_stack([normalize(r), n, f])
