"""Stage one story panel from many camera azimuths (thumbnails) to pick the most readable angle."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animescene.characters import Cast  # noqa: E402
from animescene.scene import stage_scene  # noqa: E402
from animescene.story import panel_spec  # noqa: E402

story = json.loads(Path(sys.argv[1]).read_text())
pid, out = sys.argv[2], sys.argv[3]
cast = Cast()
used = []
for cs in story["cast"]:
    used.append(cast.resolve(cs, used).palette)
panel = next(p for p in story["panels"] if p["id"] == pid)
tiles = []
for az in range(-150, 181, 30):
    spec = panel_spec(story, panel)
    spec = json.loads(json.dumps(spec))
    spec["camera"]["azimuth"] = az
    for c in spec["characters"]:
        c.pop("inflict", None)
    st = stage_scene(spec, cast, size=(456, 312))
    im = st.debug
    ImageDraw.Draw(im).text((4, 4), f"az {az}", fill=(255, 255, 255))
    tiles.append(im)
sheet = Image.new("RGB", (456 * 4, 312 * 3))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % 4) * 456, (i // 4) * 312))
sheet.save(out)
