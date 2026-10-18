"""Render actions (solo or attacker/victim pairs) with VRM models from several angles into a sheet.

  python tools/pose_gallery.py OUT.png ITEM [ITEM...] [--az 20,110,200] [--size 520x520]

ITEM is `action` (solo), `action{json params}` or `attacker/victim` (pair, victim's params after `|`).
e.g.  stand{"arms":"guard,guard"}   leg_lock/leg_locked   back_knee_kick/knee_buckle|{"facing":-90}
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animescene.characters import Cast  # noqa: E402
from animescene.render3d import build_job, render_jobs  # noqa: E402
from animescene.scene import stage_scene  # noqa: E402


def _split(part):
    if "{" in part:
        name, js = part.split("{", 1)
        return name, json.loads("{" + js)
    return part, {}


def parse(item):
    """attacker{params}/victim{params}; a `|{...}` suffix also sets victim params."""
    vp = {}
    if "|" in item:
        item, vj = item.split("|", 1)
        vp = json.loads(vj)
    depth, cut = 0, None
    for i, ch in enumerate(item):
        depth += ch == "{"
        depth -= ch == "}"
        if ch == "/" and depth == 0:
            cut = i
            break
    a_part, v_part = (item, "") if cut is None else (item[:cut], item[cut + 1:])
    att, ap = _split(a_part)
    vic, vp2 = _split(v_part) if v_part else (None, {})
    return att, vic, ap, {**vp2, **vp}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    out, items = args[0], args[1:]
    azs = [float(x) for x in opts.get("az", "20,110").split(",")]
    W, Hh = (int(x) for x in opts.get("size", "480x480").split("x"))
    elev = float(opts.get("elev", 8))
    A, V = opts.get("a", "AvatarSample_B"), opts.get("v", "AvatarSample_C")
    tmp = Path(out).with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    jobs, labels, stages = [], [], []
    for i, item in enumerate(items):
        att, vic, ap, vp = parse(item)
        for az in azs:
            a = {"name": "A", "model": A, "gender": "female", "weapon": "fists", "action": att, **ap}
            chars = [a]
            if vic:
                a["target"] = "V"
                a.setdefault("bearing", -90)  # fixed layout (attacker on -x) so azimuth really orbits
                chars = [{"name": "V", "model": V, "gender": "male", "weapon": "fists", "action": vic, "target": "A", "at": [0.0, 0.0],
                          **vp}, a]
            else:
                a.setdefault("at", [0.0, 0.0])
                a.setdefault("facing", 0)
            spec = {"setting": "day", "camera": {"shot": opts.get("shot", "full"), "azimuth": az, "elevation": elev,
                                                 "margin": float(opts.get("margin", 0.04))}, "characters": chars}
            if opts.get("focus"):
                spec["camera"]["focus"] = opts["focus"].split(",")
            st = stage_scene(spec, Cast(), size=(W, Hh))
            jobs.append(build_job(st, tmp / f"{i}_{int(az)}.png"))
            stages.append(st)
            labels.append(f"{item} az{int(az)}")
    render_jobs(jobs, log=lambda s: None)
    cols = len(azs) * (1 if len(items) > 2 else 1)
    per_row = max(1, min(len(items), int(opts.get("cols", max(1, 6 // len(azs))))))
    rows = (len(items) + per_row - 1) // per_row
    sheet = Image.new("RGB", (W * len(azs) * per_row, Hh * rows), "white")
    for k, (job, lab) in enumerate(zip(jobs, labels)):
        i, j = divmod(k, len(azs))
        im = Image.open(job["out"]).convert("RGB")
        if "bones" in opts:  # overlay the solver's skeleton (truth) on the mesh
            st, d = stages[k], ImageDraw.Draw(im)
            for a in st.scene.actors:
                for s in "lr":
                    chain = [a.joints[f"{s}_{n}"] for n in ("shoulder", "elbow", "wrist", "hand")]
                    pts = st.cam.project(np.array(chain))
                    d.line([tuple(p[:2]) for p in pts], fill=(255, 0, 0) if s == "r" else (0, 90, 255), width=2)
        ImageDraw.Draw(im).text((6, 6), lab, fill=(0, 0, 0))
        sheet.paste(im, (((i % per_row) * len(azs) + j) * W, (i // per_row) * Hh))
    sheet.save(out)
    print(out, cols)


if __name__ == "__main__":
    main()
