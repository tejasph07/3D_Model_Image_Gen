"""End to end: scene spec -> 3D stage -> controls -> base -> hires -> per-character detail."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import prompts as P
from .characters import Cast
from .scene import Staged, stage_scene


@dataclass
class Quality:
    steps: int = 28
    cfg: float = 6.0
    pose_scale: float = 0.95
    depth_scale: float = 0.5
    control_end: float = 0.7
    base_weight: float = 0.3
    hires: float = 1.4  # 0/1 = off
    hires_strength: float = 0.45
    hires_steps: int = 24
    detail: bool = True
    detail_strength: float = 0.42
    detail_steps: int = 22
    detail_size: int = 1024
    color_init: float = 0.0  # >0: start the base pass from the 3D colour-block render at this strength
    references: bool = True  # identity via each character's turnaround sheet (reference attention)
    ref_weight: float = 2.2


PRESETS = {
    "draft": Quality(steps=18, hires=0, detail=False),
    "standard": Quality(hires=0, detail=True),
    "high": Quality(),
}


def _bbox(mask: np.ndarray, thr=0.5):
    ys, xs = np.nonzero(mask > thr)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _scaled(img: Image.Image, size) -> Image.Image:
    return img.resize(size, Image.BILINEAR)


def generate(spec, engine, out_dir: str | Path, quality: Quality | str = "high", seed: int | None = None,
             cast: Cast | None = None, log=print) -> dict:
    q = PRESETS[quality] if isinstance(quality, str) else quality
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st: Staged = stage_scene(spec, cast)
    W, H = st.size
    seed = st.spec.get("seed", 0) if seed is None else seed
    pr = P.build(st)
    st.pose.save(out / "control_pose.png")
    st.depth.save(out / "control_depth.png")
    st.debug.save(out / "stage_debug.png")
    st.colors.save(out / "control_colors.png")
    (out / "prompts.json").write_text(json.dumps(pr, indent=2))
    from .raster import export_stage
    (out / "stage3d.json").write_text(json.dumps(export_stage(st.scene, st.cam)))
    log(f"[stage] {len(st.chars)} characters solved in {time.time() - t0:.1f}s")

    masks = st.masks
    effects = effect_regions(st)
    if effects:  # glowing effects take over part of the region they sit in
        e_all = np.clip(sum(m for _, m in effects), 0, 1) * 0.5
        masks = masks * (1 - e_all)[None]
    bg = np.clip(1 - masks.sum(0) - (sum(m for _, m in effects) * 0.5 if effects else 0), 0, 1)
    regions = [(p, m) for p, m in zip(pr["regions"], masks)] + [(pr["background"], bg)]
    regions += [(p, m * 0.5) for p, m in effects]

    # identity: every character attends to its own turnaround sheet, inside its own region only
    refs = []
    if q.references:
        from .diffusion import Reference
        from .sheets import ensure_sheets, reference_prompt
        sheets = ensure_sheets(engine, cast or Cast(), st.chars, seed=seed, log=log)
        refs = [Reference(sheets[c.name], reference_prompt(c), mask=np.clip(m * 1.3, 0, 1), weight=q.ref_weight)
                for c, m in zip(st.chars, st.masks)]
    loras = [l for c in st.chars for l in c.lora]
    if loras or engine.loaded_loras:
        engine.set_loras(loras)

    controls = {"openpose": (st.pose, q.pose_scale), "depth": (st.depth, q.depth_scale)}
    img = engine.sample(pr["global"], pr["negative"], W, H, steps=q.steps, cfg=q.cfg, seed=seed,
                        controls=controls, control_end=q.control_end, regions=regions,
                        base_weight=q.base_weight, progress="base", references=refs,
                        **({"init_image": st.colors, "strength": q.color_init} if q.color_init > 0 else {}))
    img.save(out / "base.png")

    if q.hires and q.hires > 1.01:
        W2, H2 = int(W * q.hires) // 64 * 64, int(H * q.hires) // 64 * 64
        up = img.resize((W2, H2), Image.LANCZOS)
        m2 = [(p, cv2.resize(m, (W2, H2))) for p, m in regions]
        r2 = [replace(r, mask=cv2.resize(r.mask, (W2, H2))) for r in refs]
        c2 = {"openpose": (_scaled(st.pose, (W2, H2)), q.pose_scale * 0.6),
              "depth": (_scaled(st.depth, (W2, H2)), q.depth_scale * 0.6)}
        img = engine.sample(pr["global"], pr["negative"], W2, H2, steps=q.hires_steps, cfg=q.cfg - 0.5,
                            seed=seed + 1, controls=c2, control_end=0.6, regions=m2, base_weight=q.base_weight,
                            init_image=up, strength=q.hires_strength, references=r2, progress="hires")
        img.save(out / "hires.png")

    if q.detail:
        img = detail_pass(engine, st, pr, img, q, seed, out, refs)

    img = draw_injuries(st, img)
    img.save(out / "final.png")
    (out / "meta.json").write_text(json.dumps(panel_meta(st, img.size), indent=2))
    log(f"[done] {out / 'final.png'} in {time.time() - t0:.0f}s")
    return {"final": str(out / "final.png"), "prompts": pr}


def effect_regions(st: Staged) -> list:
    """(prompt, mask) discs around the contact point of actions that carry a visual effect."""
    out = []
    H, W = st.frame.ids.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for a in st.scene.actors:
        if not a.out.effect or a.out.impact is None:
            continue
        p = a.to_world(a.out.impact)
        u, v_, z = st.cam.project(p[None])[0]
        r = 0.32 * st.cam.f / max(z, 0.1)
        m = np.clip(1.3 - np.hypot(xx - u, yy - v_) / r, 0, 1).astype(np.float32)
        out.append((P._join(a.out.effect), m))
    return out


def panel_meta(st: Staged, size) -> dict:
    """Where things are in the final image (for speech bubbles / SFX placement)."""
    sx, sy = size[0] / st.size[0], size[1] / st.size[1]
    meta = {"size": list(size), "actors": {}, "impacts": []}
    for a in st.scene.actors:
        hu, hv, _ = st.cam.project(a.joints["head"][None])[0]
        tu, tv, _ = st.cam.project(a.joints["head_top"][None])[0]
        meta["actors"][a.name] = {"head": [hu * sx, hv * sy], "head_top": [tu * sx, tv * sy],
                                  "head_r_px": float(abs(hv - tv) * sy)}
        if a.out.impact is not None:
            iu, iv, _ = st.cam.project(a.to_world(a.out.impact)[None])[0]
            meta["impacts"].append([iu * sx, iv * sy])
    ys, xs = np.nonzero(st.frame.ids >= 0)
    if len(xs):
        meta["subjects"] = [float(xs.min() * sx), float(ys.min() * sy), float(xs.max() * sx), float(ys.max() * sy)]
    return meta


def draw_injuries(st: Staged, img: Image.Image) -> Image.Image:
    from .injuries import draw
    W, H = img.size
    scale = (W / st.size[0], H / st.size[1])
    for i, a in enumerate(st.scene.actors):
        if a.char.injuries:
            img, _ = draw(img, a, st.cam, st.frame.depth, st.frame.ids, i, scale)
    return img


def detail_pass(engine, st: Staged, pr: dict, img: Image.Image, q: Quality, seed: int, out: Path,
                refs: list | None = None) -> Image.Image:
    """Re-render each character in a zoomed crop (like ADetailer, but driven by the 3D masks)."""
    W, H = img.size
    sx, sy = W / st.size[0], H / st.size[1]
    for i, (a, mask) in enumerate(zip(st.scene.actors, st.masks)):
        m = cv2.resize(mask, (W, H))
        bb = _bbox(m, 0.3)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        bw, bh = x1 - x0, y1 - y0
        if max(bw / W, bh / H) > 0.85:  # already big in frame
            continue
        pad = 0.12 * max(bw, bh)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side_w, side_h = bw + 2 * pad, bh + 2 * pad
        ar = side_w / side_h
        # crop canvas: keep aspect, aim for ~1MP
        tw = int(round(np.sqrt(q.detail_size ** 2 * ar) / 64)) * 64
        th = int(round(np.sqrt(q.detail_size ** 2 / ar) / 64)) * 64
        tw, th = max(tw, 512), max(th, 512)
        side_w = max(side_w, side_h * tw / th)
        side_h = side_w * th / tw
        X0 = int(np.clip(cx - side_w / 2, 0, max(0, W - side_w)))
        Y0 = int(np.clip(cy - side_h / 2, 0, max(0, H - side_h)))
        X1, Y1 = int(min(W, X0 + side_w)), int(min(H, Y0 + side_h))
        crop = img.crop((X0, Y0, X1, Y1)).resize((tw, th), Image.LANCZOS)
        cm = cv2.resize(m[Y0:Y1, X0:X1], (tw, th))
        inpaint = np.clip(cv2.GaussianBlur((cm > 0.2).astype(np.float32), (0, 0), 6) * 1.3, 0, 1)
        box = (X0 / sx, Y0 / sy, X1 / sx, Y1 / sy)
        ctl = {"openpose": (st.pose.crop(box).resize((tw, th)), q.pose_scale * 0.7),
               "depth": (st.depth.crop(box).resize((tw, th)), q.depth_scale * 0.6)}
        prompt = P._join(pr["regions"][i], "solo focus")
        ref = [replace(refs[i], mask=None, weight=q.ref_weight)] if refs else None
        res = engine.sample(prompt, pr["negative"], tw, th, steps=q.detail_steps, cfg=q.cfg - 0.5, seed=seed + 10 + i,
                            controls=ctl, control_end=0.6, init_image=crop, strength=q.detail_strength,
                            inpaint_mask=inpaint, references=ref, progress=f"detail {a.name}")
        res = res.resize((X1 - X0, Y1 - Y0), Image.LANCZOS)
        alpha = cv2.resize(inpaint, (X1 - X0, Y1 - Y0))[..., None]
        base = np.asarray(img).astype(np.float32).copy()
        patch = base[Y0:Y1, X0:X1]
        base[Y0:Y1, X0:X1] = patch * (1 - alpha) + np.asarray(res).astype(np.float32) * alpha
        img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
        img.save(out / f"detail_{i}_{a.name}.png")
    return img
