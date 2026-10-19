"""Render the two-person combat pairs with real VRM models (fast 3D path) into one contact sheet."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animescene.characters import Cast  # noqa: E402
from animescene.render3d import build_job, render_jobs  # noqa: E402
from animescene.scene import stage_scene  # noqa: E402
from tools.pair_sheet import PAIRS  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "pairs3d.png"
A, V = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("AvatarSample_B", "AvatarSample_C")
jobs, labels = [], []
tmp = Path(out).with_suffix("")
tmp.mkdir(exist_ok=True)
for i, (att, vic, ap, vp) in enumerate(PAIRS):
    for az in (20, 110):
        v = {"name": "V", "model": V, "gender": "male", "weapon": "fists", "action": vic, "target": "A", "at": [0.0, 0.0], "facing": 90, **vp}
        a = {"name": "A", "model": A, "gender": "female", "weapon": "fists", "action": att, "target": "V", **ap}
        spec = {"setting": "day", "camera": {"shot": "full", "azimuth": az, "elevation": 8, "margin": 0.04},
                "characters": [v, a]}
        st = stage_scene(spec, Cast(), size=(480, 360))
        jobs.append(build_job(st, tmp / f"{i}_{az}.png"))
        labels.append(f"{att}/{vic} az{az}")
render_jobs(jobs)
sheet = Image.new("RGB", (960 * 3, 360 * ((len(PAIRS) + 2) // 3)), "white")
for k, (job, lab) in enumerate(zip(jobs, labels)):
    i, side = divmod(k, 2)
    im = Image.open(job["out"])
    ImageDraw.Draw(im).text((6, 6), lab, fill=(0, 0, 0))
    sheet.paste(im, ((i % 3) * 960 + side * 480, (i // 3) * 360))
sheet.save(out)
print(out)
