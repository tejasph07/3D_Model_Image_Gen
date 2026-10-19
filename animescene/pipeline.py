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
    candidates: int = 2  # base passes to sample; the one that best matches the sheets is kept
    ref_weight: float = 2.2


PRESETS = {
    "draft": Quality(steps=18, hires=0, detail=False, candidates=1),
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
    sheets_img = {r.prompt: r.image for r in refs}
    best, best_score = None, -1.0
    for k in range(max(1, q.candidates)):
        cand = engine.sample(pr["global"], pr["negative"], W, H, steps=q.steps, cfg=q.cfg, seed=seed + 1000 * k,
                             controls=controls, control_end=q.control_end, regions=regions,
                             base_weight=q.base_weight, progress=f"base {k + 1}/{q.candidates}", references=refs,
                             **({"init_image": st.colors, "strength": q.color_init} if q.color_init > 0 else {}))
        score = identity_score(cand, st, [r.image for r in refs]) if refs else 0.0
        cand.save(out / f"base_{k}.png")
        log(f"[base] candidate {k}: identity score {score:.3f}")
        if score > best_score:
            best, best_score = cand, score
    img = best
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


def _hist(pixels: np.ndarray) -> np.ndarray:
    """Hue/saturation/value histogram of RGB pixels (n, 3) uint8, normalised."""
    if len(pixels) == 0:
        return np.zeros(12 * 4 * 4)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, _ = np.histogramdd(hsv.astype(float), bins=(12, 4, 4), range=((0, 180), (0, 256), (0, 256)))
    h = h.ravel()
    return h / max(h.sum(), 1)


def identity_score(img: Image.Image, st: Staged, sheets: list) -> float:
    """How well each character's region matches the colours of its own sheet (histogram
    intersection, averaged over characters). Cheap, but it catches swapped or lost designs."""
    arr = np.asarray(img.resize(st.size))
    scores = []
    for i, sheet in enumerate(sheets):
        body = (st.frame.ids == i) | (st.masks[i] > 0.85)
        if body.sum() < 200:
            continue
        s = np.asarray(sheet.convert("RGB").resize((608, 416)))
        bgc = np.median(np.concatenate([s[:8].reshape(-1, 3), s[-8:].reshape(-1, 3)]), axis=0)
        fg = np.abs(s.astype(int) - bgc).sum(-1) > 40  # character pixels on the plain sheet
        scores.append(np.minimum(_hist(arr[body]), _hist(s[fg])).sum())
    if not scores:
        return 0.0
    # one lost identity ruins a panel: weigh the worst-matching character as much as the average
    return float(0.5 * np.mean(scores) + 0.5 * np.min(scores))


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


def generate3d(spec, out_dir: str | Path, cast: Cast | None = None, log=print, render: bool = True) -> dict:
    """Full-3D panel: stage -> VRM models posed and toon-rendered -> code-drawn wounds. No diffusion.

    Returns the job (so a story can batch every panel into one renderer session) unless render=True."""
    from .raster import export_stage
    from .render3d import build_job, render_jobs
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st: Staged = stage_scene(spec, cast)
    st.debug.save(out / "stage_debug.png")
    st.pose.save(out / "control_pose.png")
    st.depth.save(out / "control_depth.png")
    (out / "stage3d.json").write_text(json.dumps(export_stage(st.scene, st.cam)))
    job = build_job(st, out / "render3d.png")
    finish = lambda: finish3d(st, out)  # noqa: E731
    if render:
        render_jobs([job], log=log)
        return finish()
    return {"job": job, "finish": finish}


def finish3d(st: Staged, out: Path) -> dict:
    img = Image.open(out / "render3d.png").convert("RGB")
    img = draw_injuries(st, img)
    img.save(out / "final.png")
    (out / "meta.json").write_text(json.dumps(panel_meta(st, img.size), indent=2))
    return {"final": str(out / "final.png")}


def id_masks(ids_png: str | Path, n: int) -> np.ndarray:
    """Per-actor masks from the renderer's flat-colour ID pass (actor i = hsl(i*67, 100%, 50%))."""
    import colorsys
    arr = np.asarray(Image.open(ids_png).convert("RGB")).astype(np.float32)
    masks = []
    for i in range(n):
        r, g, b = colorsys.hls_to_rgb(((i * 67) % 360) / 360, 0.5, 1.0)
        d = np.abs(arr - np.array([r, g, b]) * 255).sum(-1)
        masks.append((d < 60).astype(np.float32))
    return np.stack(masks)


def refine3d(engine, panel_dir: str | Path, spec, cast: Cast | None = None, strength: float = 0.42,
             steps: int = 28, seed: int = 0, tag: str = "", log=print) -> Path:
    """img2img over a 3D panel: the render fixes every position, the model only repaints surfaces.

    Constraints: ControlNet-Union with the meshes' own depth + line art traced from the render,
    per-character regional prompts restricted to each character's exact silhouette."""
    pd = Path(panel_dir)
    st: Staged = stage_scene(spec, cast)
    pr = P.build(st)
    src = Image.open(pd / "final.png").convert("RGB")
    W, H = src.size
    gray = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 140)
    lineart = Image.fromarray(edges).convert("RGB")
    lineart.save(pd / "control_lineart.png")
    masks = id_masks(pd / "render3d_ids.png", len(st.scene.actors))
    grown = np.stack([cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)), (0, 0), 3) for m in masks])
    grown = grown / np.maximum(1, grown.sum(0))[None]
    bg = np.clip(1 - grown.sum(0), 0, 1)
    regions = [(p, m) for p, m in zip(pr["regions"], grown)] + [(pr["background"], bg)]
    controls = {"depth": (Image.open(pd / "render3d_depth.png").convert("RGB"), 0.6), "lineart": (lineart, 0.55)}
    glob = P._join(pr["global"], "anime coloring, cel shading, detailed eyes, clean lineart")
    img = engine.sample(glob, P._join(pr["negative"], "3d, render, cgi, doll, plastic"), W, H, steps=steps, cfg=6.0,
                        seed=seed, controls=controls, control_end=0.85, regions=regions, base_weight=0.3,
                        init_image=src, strength=strength, progress=f"refine {pd.name}")
    out = pd / f"refined{tag}.png"
    img = draw_injuries(st, img)
    img.save(out)
    return out
