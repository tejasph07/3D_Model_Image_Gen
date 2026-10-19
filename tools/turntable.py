"""Proof the scenes are real 3D: one solved keyframe orbited by the camera (8 azimuths, 2 heights) plus the
renderer's depth pass. python -m tools.turntable MOVE KEYNAME OUT.png"""
from __future__ import annotations

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from animescene.render3d import render_jobs
from animescene.sim import keyframes as K
from animescene.sim import moves, strikes
from animescene.sim.control import Scene
from animescene.sim.export import actor_frame, camera, job


def main(move, keyname, out, size=320):
    keys = strikes.MOVES[move]() if move in strikes.MOVES else getattr(moves, move)()
    S = Scene(moves.specs())
    for k in keys:
        q, rep = K.solve_key(S, k)
        if k.name == keyname:
            break
    S.d.qpos[:] = q
    mujoco.mj_forward(S.m, S.d)
    sh = K.hand_shapes(k)
    acts = [actor_frame(S.m, S.d, s, **({"fist": sh[n]} if n in sh else {})) for n, s in S.specs.items()]
    c = np.mean([S.site(n, "chestc") for n in S.actors], axis=0)
    jobs, paths, labels = [], [], []
    for elev in (12, 55):
        for az in range(0, 360, 45):
            a, e = np.deg2rad(az), np.deg2rad(elev)
            d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
            p = f"{out[:-4]}_{len(paths)}.png"
            jobs.append(job(p, acts, camera(c + d * 2.6, c), (size, size), passes=["depth"] if (az, elev) == (90, 12) else []))
            paths.append(p)
            labels.append(f"az {az}  elev {elev}")
    render_jobs(jobs, log=lambda s: None)
    paths.append(paths[2][:-4] + "_depth.png")
    labels.append("depth pass (az 90)")
    sheet = Image.new("RGB", (size * 6, size * 3), "white")
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((size, size))
        ImageDraw.Draw(im).text((5, 5), labels[i], fill=(255, 255, 255) if "depth" in labels[i] else (0, 0, 0))
        sheet.paste(im, ((i % 6) * size, (i // 6) * size))
    sheet.save(out)
    print(out, rep)


if __name__ == "__main__":
    main(*sys.argv[1:4])
