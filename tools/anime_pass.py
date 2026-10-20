"""Anime-illustration pass over rendered fight panels with the SDXL anime engine (diffusion.py).

The render's depth pass and its traced line art go to ControlNet-Union, so positions and contacts hold even
when the model repaints most of the image; each character's prompt only speaks inside its own silhouette (the
renderer's ID pass), so four characters keep four designs. python -m tools.anime_pass PANEL_DIR... --story=S.json
[--strength=0.8 --depth=0.6 --line=0.6 --steps=24 --cfg=6 --width=1024 --tag=x] [--remote=URL --token=T]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

STYLE = ("masterpiece, best quality, amazing quality, very aesthetic, absurdres, anime screencap, anime coloring, "
         "cel shading, flat colors, sharp lineart, thick outlines, dramatic lighting, dynamic angle, manga panel")
NEGATIVE = ("3d, render, cgi, blender, realistic, photorealistic, doll, plastic, figure, lowres, worst quality, bad quality, "
            "blurry, jpeg artifacts, bad anatomy, bad hands, extra fingers, extra limbs, deformed, text, watermark, signature")


def id_masks(ids_png, n):
    import colorsys
    arr = np.asarray(Image.open(ids_png).convert("RGB")).astype(np.float32)
    out = []
    for i in range(n):
        r, g, b = colorsys.hls_to_rgb(((i * 67) % 360) / 360, 0.5, 1.0)
        out.append((np.abs(arr - np.array([r, g, b]) * 255).sum(-1) < 60).astype(np.float32))
    return np.stack(out)


def lineart(img: Image.Image) -> Image.Image:
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 130)
    return Image.fromarray(e).convert("RGB")


def run(engine, pdir: Path, spec: dict, strength=0.8, depth=0.6, line=0.6, steps=24, cfg=6.0, width=1024, seed=0,
        control_end=0.8, tag="", log=print):
    t0 = time.time()
    src = Image.open(pdir / "render.png" if (pdir / "render.png").exists() else pdir / "final.png").convert("RGB")
    W0, H0 = src.size
    W = width
    H = int(round(H0 * W / W0 / 64)) * 64
    img = src.resize((W, H), Image.LANCZOS)
    dep = Image.open(pdir / "final_depth.png").convert("RGB").resize((W, H), Image.LANCZOS)
    cast = spec["cast"]
    masks = id_masks(pdir / "final_ids.png", len(cast))
    masks = np.stack([cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in masks])
    grown = np.stack([cv2.GaussianBlur(cv2.dilate(m, np.ones((7, 7), np.uint8)), (0, 0), 2) for m in masks])
    grown = grown / np.maximum(1, grown.sum(0))[None]
    bg = np.clip(1 - grown.sum(0), 0, 1)
    setting = spec.get("setting", "")
    regions = []
    for c, m in zip(cast, grown):
        if m.sum() > 50:
            regions.append((f"{STYLE}, {c.get('tags', c.get('look', ''))}", m))
    regions.append((f"{STYLE}, no humans, scenery, {setting}, {spec.get('lighting', '')}", bg))
    who = ", ".join(c.get("tags", "").split(",")[0] for c, m in zip(cast, masks) if m.sum() > 50)
    glob = f"{STYLE}, {who}, fighting, {setting}"
    controls = {"depth": (dep, depth), "lineart": (lineart(img), line)}
    out = engine.sample(glob, NEGATIVE, W, H, steps=steps, cfg=cfg, seed=seed, controls=controls, control_end=control_end,
                        regions=regions, base_weight=0.3, init_image=img, strength=strength, progress=pdir.name)
    out = out.resize((W0, H0), Image.LANCZOS)
    dst = pdir / f"anime{tag}.png"
    out.save(dst)
    log(f"[anime] {dst} {time.time() - t0:.0f} s")
    return dst


if __name__ == "__main__":
    from animescene.diffusion import Engine
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    spec = json.loads(Path(kv.pop("story")).read_text())
    num = {k: float(v) for k, v in kv.items() if k not in ("tag", "remote", "token")}
    for k in ("steps", "width", "seed"):
        if k in num:
            num[k] = int(num[k])
    if "remote" in kv:  # a GPU elsewhere running animescene.gpu_server (e.g. the Colab notebook)
        from animescene.gpu_server import RemoteEngine
        eng = RemoteEngine(kv.pop("remote"), kv.pop("token"))
    else:
        eng = Engine()
    for d in args:
        run(eng, Path(d), spec, tag=kv.get("tag", ""), **num)
