"""Side-by-side: reference image(s) | our 3D render(s) of the same interaction.

  python tools/compare_ref.py OUT.png "ref1.png,ref2.png" ITEM --az=.. [pose_gallery options]
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image

out, refs, rest = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
tmp = str(Path(out).with_suffix("")) + "_ours.png"
subprocess.run([sys.executable, str(Path(__file__).with_name("pose_gallery.py")), tmp, *rest], check=True,
               stdout=subprocess.DEVNULL)
ours = Image.open(tmp)
h = ours.height
ims = [Image.open(r).convert("RGB") for r in refs]
ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
W = sum(im.width for im in ims) + ours.width + 12
sheet = Image.new("RGB", (W, h), "red")
x = 0
for im in ims:
    sheet.paste(im, (x, 0))
    x += im.width
sheet.paste(ours, (x + 12, 0))
sheet.save(out)
print(out)
