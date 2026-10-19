"""Optional anime-illustration pass over rendered 3D panels (img2img), free and local, CPU friendly.

Anything v5 (SD1.5 anime checkpoint) + ControlNet canny on the render's own edges + LCM-LoRA (8 steps).
The ControlNet holds every outline of the render, and a moderate strength keeps the composition, so poses,
contacts and injuries come through; the model redraws shading, lines and surface detail. About 3-6 min per
panel on 4 CPU cores at 768 px; weights (~5 GB) download from the Hugging Face Hub on first use, no account.
Run it on the clean render, before bubbles and sound effects are lettered.
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


def stylize(src, dst, prompt, strength=0.5, steps=8, width=768, seed=1, control=0.85, log=print):
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
