"""Solve a keyframed move and render it: python tools/move_sheet.py MOVE OUT.png [--fps=8] [--frames]

Prints how well every touch of every keyframe is met, renders each keyframe from three angles, and (with
--frames) a strip of the in-between frames."""
from __future__ import annotations

import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from animescene.render3d import render_jobs
from animescene.sim import keyframes as K
from animescene.sim import moves
from animescene.sim.control import Scene
from animescene.sim.export import actor_frame, camera, job


def solve(move, log=print):
    keys = getattr(moves, move)()
    S = Scene(moves.specs())
    poses = []
    for k in keys:
        t0 = time.time()
        q, rep = K.solve_key(S, k, log=log)
        poses.append(q)
        log(f"   ({time.time() - t0:.1f} s)")
    return S, keys, poses


def render(S, states, out, cams, size=(360, 360), labels=None):
    jobs, paths = [], []
    for i, (q, key) in enumerate(states):
        S.d.qpos[:] = q
        mujoco.mj_forward(S.m, S.d)
        sh = K.hand_shapes(key)
        acts = [actor_frame(S.m, S.d, s, **({"fist": sh[n]} if n in sh else {})) for n, s in S.specs.items()]
        pts = np.array([S.d.xpos[b] for b in range(S.m.nbody) if ":" in S.m.body(b).name and not S.m.body(b).name.endswith("target")])
        c = (pts.min(0) + pts.max(0)) / 2
        ext = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        for j, d in enumerate(cams):
            d = np.asarray(d, float) / np.linalg.norm(d)
            p = f"{out[:-4]}_{i}_{j}.png"
            jobs.append(job(p, acts, camera(c + d * ext * 1.35, c), size))
            paths.append(p)
    render_jobs(jobs, log=lambda s: None)
    W, H = size
    sheet = Image.new("RGB", (W * len(cams), H * len(states)), "white")
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        if labels:
            ImageDraw.Draw(im).text((6, 6), labels[k // len(cams)], fill=(0, 0, 0))
        sheet.paste(im, ((k % len(cams)) * W, (k // len(cams)) * H))
    sheet.save(out)
    return out


if __name__ == "__main__":
    move, out = sys.argv[1], sys.argv[2]
    fps = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--fps=")), 8.0)
    S, keys, poses = solve(move)
    cams = [(1.0, 0.45, 0.15), (-1.0, 0.45, 0.25), (0.25, 1.0, -0.35)]
    render(S, list(zip(poses, keys)), out, cams, labels=[f"{k.name} t={k.t:.1f}" for k in keys])
    print(out)
    if "--frames" in sys.argv:
        fr = K.inbetween(S, keys, poses, fps=fps)
        strip = out[:-4] + "_frames.png"
        render(S, [(q, k) for _, q, k in fr], strip, [cams[0]], size=(240, 240), labels=[f"t={t:.2f}" for t, _, _ in fr])
        print(strip)
