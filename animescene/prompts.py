"""Prompt assembly: global scene prompt, one prompt per character region, negative.

Tags follow danbooru conventions (what Illustrious/NoobAI-family models are
trained on). Facts the 3D stage knows are turned into tags too: how each
character is oriented relative to the camera (profile / from behind), who
is airborne, camera angle, and the effects the actions imply.
"""
from __future__ import annotations

import numpy as np

QUALITY = "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest"
NEGATIVE = ("lowres, worst quality, low quality, bad quality, normal quality, bad anatomy, bad hands, "
            "missing fingers, extra digits, fewer digits, extra arms, extra legs, extra limbs, fused bodies, "
            "conjoined, deformed, mutated, bad proportions, disembodied limb, floating limbs, duplicate, "
            "multiple views, text, signature, watermark, username, artist name, logo, jpeg artifacts, blurry, "
            "cropped, sketch, monochrome, greyscale, censored, nsfw, nude")
CAMERA_TAGS = {"low_angle": "from below, dynamic angle", "worms_eye": "from below, foreshortening, dynamic angle",
               "high_angle": "from above", "birds_eye": "from above, aerial view", "wide": "wide shot",
               "full": "full body", "medium": "cowboy shot"}
EFFECTS = ["sparks", "impact", "motion lines", "motion blur", "sword trail", "magic circle", "glowing hand",
           "speed lines", "energy"]


def _join(*parts) -> str:
    out, seen = [], set()
    for p in parts:
        if not p:
            continue
        for t in str(p).split(","):
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return ", ".join(out)


def orientation_tag(actor, cam) -> str:
    fwd = actor.F @ np.array([0, 0, 1.0])
    to_cam = cam.eye - actor.joints["head"]
    to_cam[1] = 0
    c = float(np.dot(fwd, to_cam) / (np.linalg.norm(to_cam) + 1e-9))
    if c < -0.55:
        return "from behind, back"
    if c < 0.35:
        return "from side, profile"
    return ""


def count_tags(chars) -> str:
    g = sum(c.gender == "female" for c in chars)
    b = len(chars) - g
    tags = []
    if g:
        tags.append("1girl" if g == 1 else f"{g}girls" if g < 6 else "6+girls")
    if b:
        tags.append("1boy" if b == 1 else f"{b}boys" if b < 6 else "6+boys")
    if g > 1 or (g and b):
        tags.append("multiple girls" if g > 1 else "")
    if b > 1:
        tags.append("multiple boys")
    return _join(*tags)


def build(staged) -> dict:
    spec = staged.spec
    scene, cam = staged.scene, staged.cam
    shot = spec.get("camera", {}).get("shot", "wide")
    roll = spec.get("camera", {}).get("roll", 0)
    effects = []
    fighting = False
    for a in scene.actors:
        effects += [t for t in a.out.tags if t in EFFECTS and t != "motion blur"]
        fighting |= a.action not in ("stand", "point", "kneel", "fallen")
    glob = _join(
        QUALITY, count_tags(staged.chars),
        "battle, fighting, action, dynamic pose" if fighting else "",
        CAMERA_TAGS.get(shot, ""), "dutch angle" if abs(roll) > 5 else "",
        spec.get("setting", ""), spec.get("lighting", "cinematic lighting, dramatic lighting"),
        ", ".join(effects), spec.get("style", ""),
    )
    regions = []
    for a, c in zip(scene.actors, staged.chars):
        # effects stay in the scene prompt; "motion blur" never goes on a person (it smears them away)
        action_tags = [t for t in a.out.tags if t not in EFFECTS]
        if a.out.airborne:
            action_tags.append("midair")
        hurt = any(t in a.out.tags for t in ("pain", "being hit", "injured", "defeated"))
        # region prompts carry no quality boilerplate: the tokens go to what makes this person unique
        regions.append(_join(
            c.count_tag, c.appearance_tags(), "" if hurt else c.expression, ", ".join(action_tags),
            orientation_tag(a, cam), spec.get("character_style", ""),
        ))
    background = _join(spec.get("setting", ""), spec.get("lighting", ""), "scenery, detailed background")
    negative = _join(NEGATIVE, spec.get("negative", ""))
    return {"global": glob, "regions": regions, "background": background, "negative": negative}
