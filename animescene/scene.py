"""Scene spec (JSON) -> solved 3D stage -> camera -> control images."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import Camera, frame_camera
from .characters import Cast, Character
from .controls import debug_view, depth_image, draw_openpose, keypoints, normal_image, regions
from .raster import Frame, render
from .stage import Actor, SolvedScene, Stage

ASPECTS = {"landscape": (1216, 832), "portrait": (832, 1216), "square": (1024, 1024),
           "wide": (1344, 768), "tall": (768, 1344)}

ACTION_KEYS = {"action", "target", "at", "facing", "altitude", "bearing", "distance"}


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
        for k in ACTION_KEYS:
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
    cam = frame_camera(scene.all_points(), W, H, **camspec)
    frame = render(scene, cam, ground=spec.get("ground", True))
    kps = keypoints(scene, frame)
    masks = regions(scene, frame)
    return Staged(spec, chars, scene, cam, frame, kps, masks,
                  pose=draw_openpose(kps, W, H), depth=depth_image(frame), normal=normal_image(frame),
                  debug=debug_view(scene, frame, kps, masks))
