"""Optional anime-illustration pass over rendered 3D panels (img2img), free and local, CPU friendly.

Anything v5 (SD1.5 anime checkpoint) + ControlNet canny on the render's own edges + LCM-LoRA (8 steps).
The ControlNet holds every outline of the render, and a moderate strength keeps the composition, so poses,
contacts and injuries come through; the model redraws shading, lines and surface detail. About 3-6 min per
panel on 4 CPU cores at 768 px; weights (~5 GB) download from the Hugging Face Hub on first use, no account.
Run it on the clean render, before bubbles and sound effects are lettered.

`backend="space"`: the same pass on a free hosted GPU, no account - the public Hugging Face Space
linoyts/edit_friendly_ddpm_inversion (owner-paid A10G; "edit-friendly DDPM inversion": the render is inverted
and re-generated under the target prompt, which keeps the layout). ~15-30 s per panel, 512 x 512 working size
(the panel is padded to a square and cropped back). A third party's Space: it can be paused or changed at any
time, so the local path stays the fallback.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE = "stablediffusionapi/anything-v5"
CONTROL = "lllyasviel/control_v11p_sd15_canny"
LCM = "latent-consistency/lcm-lora-sdv1-5"
NEGATIVE = "3d, render, cgi, doll, plastic, lowres, blurry, bad anatomy, extra limbs, extra fingers, text, watermark"
_PIPE = None


def pipeline():
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetImg2ImgPipeline
        torch.set_num_threads(max(1, __import__("os").cpu_count() or 4))
        cn = ControlNetModel.from_pretrained(CONTROL, torch_dtype=torch.float32, use_safetensors=True)
        p = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(BASE, controlnet=cn, torch_dtype=torch.float32,
                                                                     safety_checker=None, use_safetensors=True)
        p.scheduler = LCMScheduler.from_config(p.scheduler.config)
        p.load_lora_weights(LCM)
        p.fuse_lora()
        _PIPE = p
    return _PIPE


def prompt_for(spec: dict, beat: dict, panel: dict) -> str:
    """Who is in the panel and what happens, in tag-style words the checkpoint knows."""
    look = {c["name"]: c.get("look", "") for c in spec.get("cast", [])}
    people = [look.get(beat["a"], ""), look.get(beat["v"], "")]
    act = beat["move"].replace("_", " ")
    return ", ".join(x for x in ["masterpiece, best quality, anime illustration, manga panel, cel shading, clean lineart",
                                  *people, f"{act}, fight scene, dynamic", spec.get("setting", ""), spec.get("lighting", "")] if x)


SPACE = "linoyts/edit_friendly_ddpm_inversion"


def stylize_space(src, dst, prompt, skip=50, cfg=12.0, seed=0, log=print):
    from gradio_client import Client, handle_file
    import tempfile
    t0 = time.time()
    im = Image.open(src).convert("RGB")
    W, H = im.size
    S = max(W, H)
    sq = Image.new("RGB", (S, S), (0, 0, 0))
    sq.paste(im, ((S - W) // 2, (S - H) // 2))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    sq.resize((512, 512), Image.LANCZOS).save(tmp)
    src_prompt = "3d render, cgi, " + prompt.split("anime illustration, ")[-1]
    out = Client(SPACE, verbose=False).predict(handle_file(tmp), src_prompt, prompt, 100, 3.5, cfg, skip, seed, False, api_name="/edit")
    res = Image.open(out).convert("RGB").resize((S, S), Image.LANCZOS)
    res.crop(((S - W) // 2, (S - H) // 2, (S - W) // 2 + W, (S - H) // 2 + H)).save(dst)
    log(f"[stylize/space] {Path(dst).name} {time.time() - t0:.0f} s")
    return dst


def stylize(src, dst, prompt, strength=0.5, steps=8, width=768, seed=1, control=0.85, log=print, backend="local"):
    if backend == "space":
        try:
            return stylize_space(src, dst, prompt, log=log)
        except Exception as e:  # paused / changed / busy: fall back to the local pipeline
            log(f"[stylize/space] unavailable ({e}); running locally")
    import cv2
    import torch
    t0 = time.time()
    im = Image.open(src).convert("RGB")
    W, H = im.size
    w = width
    h = int(round(H * w / W / 64)) * 64
    small = im.resize((w, h), Image.LANCZOS)
    edges = cv2.Canny(np.array(small), 70, 150)
    ctrl = Image.fromarray(np.stack([edges] * 3, -1))
    out = pipeline()(prompt, negative_prompt=NEGATIVE, image=small, control_image=ctrl, num_inference_steps=steps,
                     strength=strength, guidance_scale=1.5, controlnet_conditioning_scale=control,
                     generator=torch.Generator().manual_seed(seed)).images[0]
    out.resize((W, H), Image.LANCZOS).save(dst)
    log(f"[stylize] {Path(dst).name} {time.time() - t0:.0f} s")
    return dst
