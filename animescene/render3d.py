"""Full-3D rendering: solved stage -> VRM models posed in three.js (MToon toon shading) -> PNG.

The stage is the truth: the renderer only receives joint positions, camera and lighting.
Expressions, gaze and hand shapes are derived from the actions; hair/cloth springs settle with
the body arriving along its direction of motion so it trails like it should.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .vrm import ASSETS, model_path

RENDER_DIR = Path(__file__).resolve().parents[1] / "render3d"

# action tag -> VRM expression weights
EXPRESSIONS = [
    (("pain", "being choked", "screaming", "tears"), {"sad": 0.6, "aa": 0.7, "blink": 0.25}),
    (("surprised", "falling", "off balance", "stumbling", "wide-eyed"), {"surprised": 0.8, "aa": 0.5}),
    (("doubled over",), {"sad": 0.4, "surprised": 0.5, "aa": 0.9}),
    (("punching", "kicking", "slashing", "palm strike", "knee strike", "headlock", "chokehold", "leg lock",
      "thrusting", "shouting"), {"angry": 0.8, "aa": 0.35}),
    (("fighting stance", "blocking", "guarding"), {"angry": 0.55}),
    (("smirk", "grin", "confident"), {"happy": 0.45}),
]
EXPR_WORDS = {"angry": {"angry": 0.7}, "serious": {"angry": 0.35}, "smirk": {"happy": 0.5}, "grin": {"happy": 0.7},
              "determined": {"angry": 0.45}, "neutral": {}, "happy": {"happy": 0.8}, "sad": {"sad": 0.8}}

ENV_PRESETS = {  # keyword -> look of the simple set (sky, ground, lights)
    "night": dict(sky="#141826", ground="#2b2f3a", ambientIntensity=1.1, keyIntensity=1.7, keyColor="#cfd8ff",
                  rim={"color": "#ff4fd8", "intensity": 1.4, "dir": [3, 2, -4]}),
    "ring": dict(sky="#1b1b22", ground="#d8d2c4", ambientIntensity=1.3, keyIntensity=2.6, keyDir=[0, 6, 1]),
    "day": dict(sky="#9cc7ef", ground="#b9a98a", ambientIntensity=1.5, keyIntensity=2.4),
    "fire": dict(sky="#3a1410", ground="#3a2a26", ambientIntensity=1.1, keyIntensity=1.8, keyColor="#ffc28a",
                 rim={"color": "#ff6a2a", "intensity": 1.8, "dir": [-3, 2, -4]}),
}


def environment_for(spec: dict) -> dict:
    env = dict(spec.get("environment3d", {}))
    s = (spec.get("setting", "") + " " + spec.get("lighting", "")).lower()
    key = next((k for k, words in [("fire", ("fire", "burning", "inferno")), ("ring", ("ring", "arena")),
                                   ("night", ("night", "neon", "moon", "rain")), ("day", ("day", "sun"))]
                if any(w in s for w in words)), "night")
    return {**ENV_PRESETS[key], "grid": False, **env}


def expression_for(actor) -> dict:
    tags = set(actor.out.tags)
    for keys, ex in EXPRESSIONS:
        if tags & set(keys):
            return dict(ex)
    return dict(EXPR_WORDS.get(actor.char.expression, {"angry": 0.3}))


def set_for(staged) -> dict:
    from .sets import build_set
    center = np.mean([a.joints["pelvis"] for a in staged.scene.actors], axis=0)
    return build_set(staged.spec, center, staged.cam)


def build_job(staged, out_png: str | Path) -> dict:
    scene, cam = staged.scene, staged.cam
    actors = []
    for a in scene.actors:
        if not a.char.model:
            raise ValueError(f"{a.name} has no 3D model; set 'model' to one of the VRM library entries")
        rel = model_path(a.char.model).resolve().relative_to(ASSETS.resolve())
        look = None
        if a.target:
            look = scene.actor(a.target).joints["head"].tolist()
        actors.append({
            "model": f"/assets/{rel.as_posix()}",
            "scale": a.char.scale,
            "joints": {k: [float(x) for x in p] for k, p in a.joints.items()},
            "expression": expression_for(a),
            "lookAt": look,
            "fist": a.out.pose.fist,
            "motion": (a.F @ a.out.motion).tolist() if np.linalg.norm(a.out.motion) > 1e-6 else None,
        })
    return {"out": str(out_png), "width": cam.width, "height": cam.height,
            "camera": {"eye": cam.eye.tolist(), "fwd": cam.fwd.tolist(), "up": cam.up.tolist(), "fov": cam.fov},
            "actors": actors, "environment": set_for(staged), "supersample": 2, "passes": ["depth", "ids"]}


def render_jobs(jobs: list, log=print):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(jobs, f)
    r = subprocess.run(["node", str(RENDER_DIR / "render.mjs"), f.name], capture_output=True, text=True,
                       env={**__import__("os").environ, "ANIMESCENE_ASSETS": str(ASSETS)})
    for line in (r.stdout + r.stderr).splitlines():
        if line.strip():
            log(line)
    if r.returncode != 0:
        raise RuntimeError("3D render failed")
    from PIL import Image
    for j in jobs:  # downsample supersampled frames (ids with nearest: they are labels)
        if j.get("supersample", 1) > 1:
            for suffix, flt in (("", Image.LANCZOS), ("_depth", Image.LANCZOS), ("_ids", Image.NEAREST)):
                p = Path(str(j["out"]).replace(".png", f"{suffix}.png"))
                if p.exists():
                    Image.open(p).resize((j["width"], j["height"]), flt).save(p)
