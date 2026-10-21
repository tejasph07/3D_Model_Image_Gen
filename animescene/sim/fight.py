"""A fight scene from keyframed moves -> rendered manga panels -> pages.

Scene file (JSON):
  title, setting, lighting, cast: [{name, model, scale?, expression?}], injuries?: [{who, spot, kind, radius?, age?}]
  beats: [{move, a, v, at: [x, z], yaw, injuries?, treat?: [{who, spot}], heal?: 0..1, others: {name: {pose: ready|down|kneel|doubled, at: [x, z], yaw}},
           panels: [{key: <keyframe name>, camera: {shot: wide|medium|close|extreme, az, elev, focus: pair|a|v|contact},
                     layout: wide|half|full|tall, caption, dialogue: [{who, text}], sfx: [{text, at: impact}],
                     expression: {name: tag}}]}]

Each beat places one move (strikes.MOVES or the ankle-lock sequences in moves.py) between two named characters;
everyone else holds a simple pose where they are. Every keyframe is solved deterministically (keyframes.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from ..manga import compose_pages
from ..render3d import ENV_PRESETS, EXPRESSIONS, render_jobs
from ..sets import build_set
from . import keyframes as K
from . import moves as M
from . import strikes as X
from .control import Scene
from .export import actor_frame
from .wounds import Ledger, from_hit
from .model import ActorSpec

SHOT = {"wide": 5.0, "medium": 3.0, "close": 1.7, "extreme": 0.95}

BYSTANDER = {
    "ready": dict(posture=X.FIGHT, support=X.FEET, balance=True),
    "down": dict(posture=dict(M.ON_BACK, rightUpperLeg_flex=20, rightLowerLeg_flex=30), support=M.LYING + ["heel_l", "heel_r"],
                 pitch=-90, height=0.12),
    "kneel": dict(posture=dict(X.STAND, leftUpperLeg_flex=10, rightUpperLeg_flex=10, leftLowerLeg_flex=120, rightLowerLeg_flex=120,
                               spine_flex=30, neck_flex=25, leftFoot_point=40, rightFoot_point=40, leftUpperArm_raise=-75,
                               rightUpperArm_raise=-75), support=["knee_l", "knee_r", "toes_l", "toes_r"]),
    "doubled": dict(posture=X.DOUBLED, support=X.FEET, balance=True),
}

PAIN = {"sad": 0.6, "aa": 0.7, "blink": 0.25}
FIERCE = {"angry": 0.8, "aa": 0.35}


def _tag_expression(tag):
    for keys, ex in EXPRESSIONS:
        if tag in keys:
            return dict(ex)
    return {"angry": 0.35}


def beat_keys(beat):
    move = beat["move"]
    keys = X.MOVES[move]() if move in X.MOVES else getattr(M, move)()
    keys = K.place_keys(keys, {"A": beat["a"], "V": beat["v"]}, origin=tuple(beat.get("at", (0.0, 0.0))), yaw=beat.get("yaw", 0.0))
    for name, o in beat.get("others", {}).items():  # everyone else holds a pose where they are
        b = BYSTANDER[o.get("pose", "ready")]
        for k in keys:
            k.posture[name] = dict(b["posture"])
            k.support[name] = list(b["support"])
            k.root[name] = dict(pos=[o["at"][0], b.get("height"), o["at"][1]], yaw=o.get("yaw", 0.0), pitch=b.get("pitch", 0.0),
                                pos_w=1.0, hold=1.5)
            if b.get("balance"):
                k.balance = tuple(k.balance) + (name,)
            k.order = (name,) + tuple(k.order)
            k.stiff[name] = 0.3
    return keys


class _Cam:
    def __init__(self, eye, target, fov, size):
        self.eye, self.target, self.fov, self.size = np.asarray(eye, float), np.asarray(target, float), fov, size
        f = self.target - self.eye
        self.fwd = f / np.linalg.norm(f)
        r = np.cross(self.fwd, [0, 1.0, 0])
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.fwd)

    def project(self, p):
        v = np.asarray(p, float) - self.eye
        z = v @ self.fwd
        f = (self.size[1] / 2) / np.tan(np.deg2rad(self.fov) / 2)
        return [self.size[0] / 2 + (v @ self.right) / z * f, self.size[1] / 2 - (v @ self.up) / z * f]

    def job_camera(self):
        return {"eye": self.eye.tolist(), "fwd": self.fwd.tolist(), "up": [0, 1, 0], "fov": self.fov}


OPENPOSE = ["nose", "neck", "shoulder_r", "elbow_r", "hand_r", "shoulder_l", "elbow_l", "hand_l", "hip_r", "knee_r",
            "ankle_r", "hip_l", "knee_l", "ankle_l", "eye_r", "eye_l", "ear_r", "ear_l"]


def openpose_points(S, n, cam):
    """The 18 OpenPose (COCO) keypoints of actor n, projected: [[x, y, depth] ...] (depth < 0: behind the camera)."""
    P = {k: S.site(n, k) for k in ("nose", "shoulder_r", "elbow_r", "hand_r", "shoulder_l", "elbow_l", "hand_l", "hip_r",
                                   "ankle_r", "hip_l", "ankle_l", "ear_r", "ear_l", "eyes")}
    P["neck"] = (P["shoulder_l"] + P["shoulder_r"]) / 2
    for side in ("l", "r"):  # knee joint (the kneecap site sits in front of it), eyes beside the eye-line centre
        P[f"knee_{side}"] = S.body(n, f"{'left' if side == 'l' else 'right'}LowerLeg").xpos.copy()
        P[f"eye_{side}"] = P["eyes"] + (P[f"ear_{side}"] - P["eyes"]) * 0.3
    out = []
    for k in OPENPOSE:
        x, y = cam.project(P[k])
        out.append([float(x), float(y), float((P[k] - cam.eye) @ cam.fwd)])
    return out


def panel_camera(S, beat, key, cam, size):
    a, v = beat["a"], beat["v"]
    focus = cam.get("focus", "pair")
    if focus == "contact" and key.touches:
        t = key.touches[0]
        target = (S.site(t.who, t.part) + S.site(t.other, t.spot)) / 2
    elif focus in (a, "a"):
        target = S.site(a, "chestc")
    elif focus in (v, "v"):
        target = S.site(v, "chestc")
    elif focus == "all":
        pts = [S.body(n, "hips").xpos for n in S.actors]
        target = np.mean(pts, axis=0) + [0, 0.5, 0]
    else:
        target = (S.body(a, "hips").xpos + S.body(v, "hips").xpos) / 2 + [0, 0.15, 0]
    el = np.deg2rad(cam.get("elev", 10.0))
    d = SHOT[cam.get("shot", "medium")] * cam.get("zoom", 1.0)
    others = [n for n in S.actors if n not in (a, v)] if focus != "all" else []
    pts_other = [S.d.xpos[i] for i in range(S.m.nbody) if S.m.body(i).name.split(":")[0] in others]
    pts_pair = []
    if focus == "contact" and key.touches:
        pts_pair = [S.d.xpos[i] for i in range(S.m.nbody) if S.m.body(i).name.split(":")[0] in (a, v)
                    and np.linalg.norm(S.d.xpos[i] - target) > 0.25]
    pts_all = [S.d.xpos[i] for i in range(S.m.nbody) if ":" in S.m.body(i).name and not S.m.body(i).name.endswith("target")]

    def blocked(eye):
        """Someone outside the pair between the camera and the action, or the camera inside a body."""
        seg = target - eye
        L = np.linalg.norm(seg)
        for p in pts_other:
            u = (p - eye) @ seg / L
            if 0.2 < u < L - 0.3 and np.linalg.norm((p - eye) - seg / L * u) < 0.3:
                return True
        if focus == "contact" and key.touches:  # nothing of the pair itself in front of the contact either
            for p in pts_pair:
                u = (p - eye) @ seg / L
                if 0.2 < u < L - 0.25 and np.linalg.norm((p - eye) - seg / L * u) < 0.13:
                    return True
        return any(np.linalg.norm(p - eye) < 0.35 for p in pts_all)

    for turn in (0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 180):
        base = np.deg2rad(beat.get("yaw", 0.0) + cam.get("az", 90.0) + turn)
        dirn = np.array([np.sin(base) * np.cos(el), np.sin(el), np.cos(base) * np.cos(el)])
        eye = target + dirn * d
        if not blocked(eye):
            break
    return _Cam(eye, target, cam.get("fov", 35.0), size)


def run(spec: dict, out_dir, size=(1200, 900), log=print, stylize=None, stylize_backend="local"):
    """stylize: None, "all" or panel indices to pass through the img2img illustration step (stylize.py)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cast = spec["cast"]
    specs = [ActorSpec(c["name"], c["model"], (i * 3.0, -6.0), 0.0, c.get("scale", 1.0)) for i, c in enumerate(cast)]
    S = Scene(specs)
    expr0 = {c["name"]: c.get("expression", "serious") for c in cast}
    panels, jobs, metas = [], [], []
    owners = []
    wounds = Ledger()  # injuries carried from beat to beat
    for w in spec.get("injuries", []):  # already hurt when the story starts
        wounds.add(w["who"], w["spot"], w.get("kind", "bruise"), w.get("radius", 0.03), w.get("age", 0.0))
    for bi, beat in enumerate(spec["beats"]):
        keys = beat_keys(beat)
        wanted = {p["key"] for p in beat.get("panels", [])}
        solved = {}
        for k in keys:
            if k.name not in wanted:
                continue  # only the keyframes a panel shows (in order: each solve starts from its hints)
            q, rep = K.solve_key(S, k)
            log(f"[beat {bi} {beat['move']} / {k.name}] " + ", ".join(f"{kk.split(':')[0]} {vv:.3f}" for kk, vv in rep.items()))
            solved[k.name] = (q, k)
        for pi, p in enumerate(beat.get("panels", [])):
            q, k = solved[p["key"]]
            S.d.qpos[:] = q
            mujoco.mj_forward(S.m, S.d)
            cam = panel_camera(S, beat, k, p.get("camera", {}), size)
            pdir = out / f"b{bi:02d}_p{pi}"
            pdir.mkdir(exist_ok=True)
            shapes = K.hand_shapes(k)
            acts = []
            for n, s in S.specs.items():
                ex = p.get("expression", {}).get(n)
                if ex:
                    e = _tag_expression(ex)
                elif n == beat["a"]:
                    e = FIERCE
                elif n == beat["v"] and k.touches:
                    e = PAIN
                else:
                    e = {"angry": 0.4} if expr0[n] in ("angry", "serious") else {"happy": 0.4}
                look = None
                if n == beat["a"]:
                    look = S.site(beat["v"], "eyes").tolist()
                elif n == beat["v"]:
                    look = S.site(beat["a"], "eyes").tolist()
                acts.append(actor_frame(S.m, S.d, s, expression=e, lookAt=look, injuries=wounds.render(S, n),
                                        **({"fist": shapes[n]} if n in shapes else {})))
            env = build_set(spec, np.mean([S.body(n, "hips").xpos for n in (beat["a"], beat["v"])], axis=0), cam)
            jobs.append({"out": str(pdir / "final.png"), "width": size[0], "height": size[1], "camera": cam.job_camera(),
                         "actors": acts, "environment": env, "supersample": 2, "passes": ["depth", "ids"], "softContacts": True})
            (pdir / "job.json").write_text(json.dumps(jobs[-1]))  # re-render later without re-solving
            meta = {"size": list(size), "actors": {}, "impacts": []}
            for n in S.actors:
                h = cam.project(S.site(n, "eyes"))
                top = cam.project(S.site(n, "crown"))
                if 0 <= h[0] <= size[0] and 0 <= h[1] <= size[1]:
                    meta["actors"][n] = {"head": h, "head_top": top, "head_r_px": float(abs(h[1] - top[1]))}
            meta["pose2d"] = {n: openpose_points(S, n, cam) for n in S.actors}  # for a skeleton ControlNet
            for t in k.touches:
                if t.verb in ("punch", "kick", "knee", "stamp", "press") and t.who == beat["a"]:
                    meta["impacts"].append(cam.project(S.site(t.other, t.spot)))
            pts = [cam.project(S.d.xpos[i]) for i in range(S.m.nbody) if S.m.body(i).name.split(":")[0] in (beat["a"], beat["v"])]
            xs, ys = [x for x, _ in pts], [y for _, y in pts]
            meta["subjects"] = [max(0, min(xs)), max(0, min(ys)), min(size[0], max(xs)), min(size[1], max(ys))]
            (pdir / "meta.json").write_text(json.dumps(meta))
            panels.append((p, pdir))
            owners.append(beat)
        for k in keys:  # the blows that landed in this beat mark the body from the next beat on
            for t in k.touches:
                if t.verb in ("punch", "kick", "knee", "stamp", "press", "twist", "pinch") and t.who == beat["a"]:
                    kind, r = from_hit(t.verb, t.spot)
                    wounds.add(t.other, t.spot, kind, r)
        for w in beat.get("injuries", []):  # story-given: {who, spot, kind, radius}
            wounds.add(w["who"], w["spot"], w.get("kind", "bruise"), w.get("radius", 0.03))
        for w in beat.get("treat", []):  # {who, spot}: bandaged
            wounds.treat(w["who"], w["spot"])
        if beat.get("heal"):  # time passes: 0..1
            wounds.heal(beat["heal"])
    log(f"rendering {len(jobs)} panels")
    render_jobs(jobs, log=lambda s: None)
    if stylize:
        from .. import stylize as ST
        pick = range(len(panels)) if stylize == "all" else stylize
        for i in pick:
            p, pdir = panels[i]
            (pdir / "final.png").replace(pdir / "render.png")
            ST.stylize(pdir / "render.png", pdir / "final.png", ST.prompt_for(spec, owners[i], p), log=log, backend=stylize_backend)
    pages = compose_pages(spec, panels, out)
    log("pages: " + ", ".join(str(p) for p in pages))
    return pages


if __name__ == "__main__":
    import sys
    kv = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--") and "=" in a)
    opt = kv.get("stylize")
    st = None if opt is None else ("all" if opt == "all" else [int(x) for x in opt.split(",")])
    run(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], stylize=st, stylize_backend=kv.get("stylize-backend", "local"))
