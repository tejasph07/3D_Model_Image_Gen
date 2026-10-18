"""Scene spec (JSON) -> solved 3D stage -> camera -> control images."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import Camera, frame_camera
from .characters import Cast, Character
from .controls import color_blocks, debug_view, depth_image, draw_openpose, keypoints, normal_image, regions
from .raster import Frame, render
from .stage import Actor, SolvedScene, Stage

ASPECTS = {"landscape": (1216, 832), "portrait": (832, 1216), "square": (1024, 1024),
           "wide": (1344, 768), "tall": (768, 1344)}

ACTION_KEYS = {"action", "target", "at", "facing", "altitude", "bearing", "distance"}
PARAM_KEYS = {"style", "phase", "aim", "leg", "lift", "effect"}
UPPER = ["head", "head_top", "nose", "neck", "chest", "spine", "pelvis", "l_shoulder", "r_shoulder",
         "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"]


@dataclass
class Staged:
    spec: dict
    chars: list[Character]
    scene: SolvedScene
    cam: Camera
    frame: Frame
    kps: list
    masks: np.ndarray
    pose: Image.Image
    depth: Image.Image
    normal: Image.Image
    debug: Image.Image
    colors: Image.Image

    @property
    def size(self):
        return self.cam.width, self.cam.height


def load_spec(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return json.loads(Path(path_or_dict).read_text())


def build_actors(spec: dict, cast: Cast) -> tuple[list[Actor], list[Character]]:
    if not spec.get("characters"):
        raise ValueError("scene has no characters")
    names = [c["name"] for c in spec["characters"]]
    if len(set(names)) != len(names):
        raise ValueError("character names must be unique")
    actors, chars, used = [], [], []
    for cs in spec["characters"]:
        char = cast.resolve(cs, used)
        used.append(char.palette)
        chars.append(char)
        act = dict(cs.get("pose", {}))
        for k in ACTION_KEYS | PARAM_KEYS:
            if k in cs:
                act.setdefault(k, cs[k])
        params = {k: v for k, v in act.items() if k not in ACTION_KEYS}
        actors.append(Actor(
            char=char, action=act.get("action", "stance"), target=act.get("target"), params=params,
            at=tuple(act["at"]) if act.get("at") is not None else None, facing=act.get("facing"),
            altitude=act.get("altitude"), bearing=act.get("bearing"), distance=act.get("distance"),
        ))
    return actors, chars


def stage_scene(spec_in, cast: Cast | None = None, size: tuple | None = None) -> Staged:
    spec = load_spec(spec_in)
    cast = cast or Cast()
    actors, chars = build_actors(spec, cast)
    scene = Stage(actors, spec).solve()
    camspec = dict(spec.get("camera", {}))
    W, H = size or ASPECTS[camspec.pop("aspect", "landscape")]
    camspec.pop("aspect", None)
    focus = camspec.pop("focus", None)
    pts = focus_points(scene, focus) if focus else scene.all_points()
    cam = frame_camera(pts, W, H, **camspec)
    record_injuries(spec, scene, cast)
    frame = render(scene, cam, ground=spec.get("ground", True))
    kps = keypoints(scene, frame)
    masks = regions(scene, frame)
    return Staged(spec, chars, scene, cam, frame, kps, masks,
                  pose=draw_openpose(kps, W, H), depth=depth_image(frame), normal=normal_image(frame),
                  debug=debug_view(scene, frame, kps, masks),
                  colors=color_blocks(scene, frame, chars, spec.get("setting", "")))


def focus_points(scene, focus: list) -> np.ndarray:
    """Points a close-up must frame: "Name", "Name.upper", "Name.head", "Name.<joint>", "impact"."""
    pts = []
    for item in focus:
        if item == "impact":
            for a in scene.actors:
                if a.out.impact is not None:
                    c = a.to_world(a.out.impact)
                    pts += [c + d for d in np.array([[.25, 0, 0], [-.25, 0, 0], [0, .25, 0], [0, -.25, 0]])]
            continue
        name, _, part = item.partition(".")
        a = scene.actor(name)
        if not part:
            pts += list(a.joints.values())
        elif part == "upper":
            pts += [a.joints[k] for k in UPPER]
        elif part == "head":
            r = a.char.body.head_r
            pts += [a.joints["head"] + d * r * 1.6 for d in np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1.4, 0]])]
        else:
            pts += [a.joints[part] + d for d in np.array([[.15, 0, 0], [-.15, 0, 0], [0, .15, 0], [0, -.15, 0]])]
    if not pts:
        raise ValueError(f"camera focus {focus!r} selects nothing")
    return np.array(pts)


def record_injuries(spec: dict, scene, cast: Cast):
    """Apply each character's `inflict` list: wounds land at the blow's 3D contact point (or a named
    spot) and are stored on the character, so every later scene draws them in the same place."""
    from .injuries import anchor_from_point, make_injury
    title = spec.get("title", "")
    for cs in spec["characters"]:
        for k, ev in enumerate(cs.get("inflict", [])):
            a = scene.actor(cs["name"])
            key = f"{title}#{cs['name']}#{k}"
            if any(i.get("key") == key for i in a.char.injuries):
                continue  # already recorded when this scene was staged before
            if ev.get("spot") or ev.get("bone"):
                inj = make_injury(ev["kind"], spot=ev.get("spot"), bone=ev.get("bone"), t=ev.get("t", 0.5),
                                  angle=ev.get("angle", 0.0), size=ev.get("size", 0.05), seed=ev.get("seed"))
            else:
                src = scene.actor(ev.get("from") or next(x.name for x in scene.actors if x.target == a.name))
                if src.out.impact is None:
                    raise ValueError(f"{src.name}'s action has no impact point to wound {a.name} with")
                bone, t, ang = anchor_from_point(a, src.to_world(src.out.impact))
                inj = make_injury(ev["kind"], bone=bone, t=t, angle=ang, size=ev.get("size", 0.05), seed=ev.get("seed"))
            inj["key"] = key
            inj["scene"] = title
            a.char.injuries.append(inj)
            stored = cast.get(a.name)
            if stored is not None and stored is not a.char:
                stored.injuries.append(inj)
    cast.save()
