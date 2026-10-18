"""Character sheets: one turnaround image per character, generated once and kept in the cast.

The sheet (front / side / back, full body, plain background) is the identity anchor every
later render attends to through regional reference attention, so a character designed on the
fly looks the same in every panel: same face, hair, outfit, weapon.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .camera import frame_camera
from .characters import Cast, Character
from .controls import depth_image, draw_openpose, keypoints
from .prompts import QUALITY, _join
from .raster import render
from .stage import Actor, Stage

SHEET_TAGS = ("character sheet, reference sheet, multiple views, from front, from side, from behind, "
              "full body, standing, white background, simple background")
SHEET_NEGATIVE = ("lowres, worst quality, low quality, bad anatomy, bad hands, extra digits, fewer digits, "
                  "text, watermark, signature, blurry, cropped, out of frame, sketch, monochrome, "
                  "multiple girls, multiple boys, different outfits, inconsistent clothes, nsfw, nude")
SHEET_SIZE = (1216, 832)


def sheet_prompt(c: Character) -> str:
    return _join(QUALITY, c.count_tag, SHEET_TAGS, c.appearance_tags(), "neutral expression")


def reference_prompt(c: Character) -> str:
    """Prompt used when the sheet is replayed as a reference (describes the sheet image)."""
    return _join(c.count_tag, SHEET_TAGS, c.appearance_tags())


def stage_sheet(c: Character, size=SHEET_SIZE):
    """Three copies of the character standing: facing the camera, profile, back."""
    views = [("front", 0.0, -1.0), ("side", 90.0, 0.0), ("back", 180.0, 1.0)]
    actors = [Actor(char=c, action="stand", at=(x * 0.75 * c.height, 0.0), facing=yaw) for _, yaw, x in views]
    st = Stage(actors, {"camera": {"azimuth": 0}})
    st.by = {f"v{i}": a for i, a in enumerate(actors)}
    scene = st.solve()
    cam = frame_camera(scene.all_points(), size[0], size[1], shot="full", elevation=3.0, margin=0.04)
    frame = render(scene, cam, ground=False)
    kps = keypoints(scene, frame)
    return scene, frame, kps


def make_sheet(engine, c: Character, path: str | Path, seed: int = 0, steps: int = 30) -> Image.Image:
    W, H = SHEET_SIZE
    scene, frame, kps = stage_sheet(c)
    pose = draw_openpose(kps, W, H)
    depth = depth_image(frame)
    img = engine.sample(sheet_prompt(c), SHEET_NEGATIVE, W, H, steps=steps, cfg=6.0, seed=seed,
                        controls={"openpose": (pose, 1.0), "depth": (depth, 0.45)}, control_end=0.75,
                        progress=f"sheet {c.name}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return img


def ensure_sheets(engine, cast: Cast, chars: list[Character], seed: int = 0, log=print) -> dict:
    """Make (once) and return {name: sheet image} for the given characters."""
    out = {}
    base = cast.path.parent if cast.path else Path("cast_sheets")
    for c in chars:
        stored = cast.get(c.name)
        p = Path(stored.sheet) if stored is not None and stored.sheet else base / "sheets" / f"{c.name}.png"
        if not p.exists():
            log(f"[sheet] designing {c.name}")
            make_sheet(engine, c, p, seed=seed)
        c.sheet = str(p)
        if stored is not None:
            stored.sheet = str(p)
        out[c.name] = Image.open(p).convert("RGB")
    cast.save()
    return out
