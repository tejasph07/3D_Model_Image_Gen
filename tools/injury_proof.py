"""Proof that injuries are painted on the 3D body: the same wounds on one character in three different poses,
each from three cameras, plus one healed row. python -m tools.injury_proof OUT.png"""
from __future__ import annotations

import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from animescene.render3d import render_jobs
from animescene.sim import fight, strikes
from animescene.sim import keyframes as K
from animescene.sim import moves
from animescene.sim.control import Scene
from animescene.sim.export import actor_frame, camera, job
from animescene.sim.wounds import mark

WOUNDS = [("jaw_l", "bruise", 0.03), ("temple_r", "swelling", 0.03), ("forearm_l", "bleeding_cut", 0.045),
          ("forearm_r", "bandage", 0.04), ("chin", "cut", 0.015), ("shin_r", "scar", 0.05)]


def main(out):
    S = Scene(moves.specs())
    keys = strikes.stomach_punch()
    down = fight.BYSTANDER["down"]
    lying = K.Key(t=0, name="lying", order=("V",), posture={"V": dict(down["posture"])}, support={"V": list(down["support"])},
                  root={"V": dict(pos=[0.0, 0.12, 1.2], yaw=180, pitch=-90, pos_w=1.0, hold=1.5)},
                  stiff={"V": 0.3})
    states = []
    for k in (keys[0], keys[2], lying):
        q, _ = K.solve_key(S, k)
        states.append((k.name, q))
    jobs, paths, labels = [], [], []
    for age in (0.0, 0.9):
        for name, q in (states if age == 0 else states[:1]):
            S.d.qpos[:] = q
            mujoco.mj_forward(S.m, S.d)
            inj = [mark(S, "V", s, k, r, seed=i * 5, age=age) for i, (s, k, r) in enumerate(WOUNDS)]
            act = actor_frame(S.m, S.d, S.specs["V"], injuries=inj, expression=fight.PAIN)
            c = (S.site("V", "crown") + S.site("V", "heel_l")) / 2
            tag = "  HEALED (age 0.9)" if age else ""
            shots = [(c, 2.6, 200, f"{name}  full body"), (S.site("V", "jaw_l"), 0.55, None, f"{name}  face (jaw_l)"),
                     (S.site("V", "forearm_l"), 0.6, None, f"{name}  left forearm")]
            for tgt, dist, az, lab in shots:
                if az is None:  # close-up from straight off the skin there
                    spot = "jaw_l" if "face" in lab else "forearm_l"
                    d = K._normal(S, K.Touch("V", spot, "V", spot)) + np.array([0, 0.25, 0])
                else:
                    a = np.deg2rad(az)
                    d = np.array([np.sin(a), 0.25, np.cos(a)])
                p = f"{out[:-4]}_{len(paths)}.png"
                jobs.append(job(p, [act], camera(tgt + d / np.linalg.norm(d) * dist, tgt), (420, 420)))
                paths.append(p)
                labels.append(lab + tag)
    render_jobs(jobs, log=lambda s: None)
    W = 420
    sheet = Image.new("RGB", (W * 3, W * (len(paths) // 3)), "white")
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        ImageDraw.Draw(im).text((6, 6), labels[i], fill=(0, 0, 0))
        sheet.paste(im, ((i % 3) * W, (i // 3) * W))
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
