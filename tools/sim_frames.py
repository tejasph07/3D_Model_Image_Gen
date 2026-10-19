"""Render frames of a recorded simulation (dev helper): python tools/sim_frames.py is used from code."""
from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from animescene.render3d import render_jobs
from animescene.sim.export import actor_frame, camera, job


def render_times(S, times, out, cams, size=(480, 480), extra=None):
    """Render recorded states nearest to `times` from each camera (eye, target) into one sheet."""
    jobs, labels = [], []
    d0 = S.d.qpos.copy()
    tmp = []
    for ti, t in enumerate(times):
        k = min(range(len(S.frames)), key=lambda i: abs(S.frames[i][0] - t)) if S.frames else None
        if k is not None:
            S.d.qpos[:] = S.frames[k][1]
        mujoco.mj_forward(S.m, S.d)
        acts = [actor_frame(S.m, S.d, s, **((extra or {}).get(n, {}))) for n, s in S.specs.items()]
        for ci, (eye, tgt) in enumerate(cams):
            p = f"{out[:-4]}_{ti}_{ci}.png"
            jobs.append(job(p, acts, camera(eye, tgt), size, softContacts=False))
            labels.append(f"t={t:.2f} cam{ci}")
            tmp.append(p)
    S.d.qpos[:] = d0
    mujoco.mj_forward(S.m, S.d)
    render_jobs(jobs, log=lambda s: None)
    W, H = size
    sheet = Image.new("RGB", (W * len(cams), H * len(times)), "white")
    for i, (p, lab) in enumerate(zip(tmp, labels)):
        im = Image.open(p).convert("RGB")
        ImageDraw.Draw(im).text((6, 6), lab, fill=(0, 0, 0))
        sheet.paste(im, ((i % len(cams)) * W, (i // len(cams)) * H))
    sheet.save(out)
    return out
