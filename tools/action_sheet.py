"""Render every action (actor vs. a standing target) from two angles into one sheet."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animescene.actions import ACTIONS  # noqa: E402
from animescene.scene import stage_scene  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "action_sheet.png"
only = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(ACTIONS)
weapon = sys.argv[3] if len(sys.argv) > 3 else "katana"
tiles = []
for name in only:
    row = []
    for az in (90, 30):
        spec = {"camera": {"shot": "full", "azimuth": az, "elevation": 5},
                "characters": [
                    {"name": "T", "gender": "male", "archetype": "knight", "weapon": weapon,
                     "action": "block" if name == "slash" else "stand", "target": "A", "at": [0.0, 0.0], "facing": -90},
                    {"name": "A", "gender": "female", "archetype": "swordsman", "weapon": weapon,
                     "action": name, "target": "T"}]}
        s = stage_scene(spec, size=(384, 384))
        im = s.debug
        ImageDraw.Draw(im).text((4, 4), f"{name} az{az}", fill=(255, 255, 255))
        row.append(im)
    tiles.append(row)
W = 384 * 2
sheet = Image.new("RGB", (W * 3, 384 * ((len(tiles) + 2) // 3)))
for i, row in enumerate(tiles):
    x, y = (i % 3) * W, (i // 3) * 384
    sheet.paste(row[0], (x, y))
    sheet.paste(row[1], (x + 384, y))
sheet.save(out)
print(out)
