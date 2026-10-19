"""Render two-person interactions (attacker, victim reaction) from two angles into one sheet."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animescene.characters import Cast  # noqa: E402
from animescene.scene import stage_scene  # noqa: E402

PAIRS = [  # (attacker action, victim action, attacker params, victim params)
    ("headlock", "headlocked", {}, {}),
    ("rear_choke", "choked", {}, {}),
    ("leg_lock", "leg_locked", {}, {}),
    ("back_knee_kick", "knee_buckle", {}, {"facing": -90}),
    ("spinning_back_kick", "staggered", {}, {}),
    ("palm_strike", "doubled_over", {}, {}),
    ("knee_strike", "doubled_over", {}, {}),
    ("kick", "clutch_stomach", {"aim": "stomach", "style": "front"}, {}),
    ("punch", "stumble", {"style": "hook"}, {}),
]


def main(out, only=None):
    tiles = []
    for att, vic, ap, vp in PAIRS:
        if only and att not in only:
            continue
        row = []
        for az in (20, 110):
            v = {"name": "V", "gender": "male", "archetype": "brawler", "seed": 5, "action": vic, "target": "A",
                 "at": [0.0, 0.0], "facing": 90, **vp}
            a = {"name": "A", "gender": "female", "archetype": "brawler", "seed": 6, "action": att, "target": "V", **ap}
            spec = {"camera": {"shot": "full", "azimuth": az, "elevation": 8}, "characters": [v, a]}
            s = stage_scene(spec, Cast(), size=(384, 384))
            im = s.debug
            ImageDraw.Draw(im).text((4, 4), f"{att}/{vic} az{az}", fill=(255, 255, 255))
            row.append(im)
        tiles.append(row)
    W = 768
    sheet = Image.new("RGB", (W * 3, 384 * ((len(tiles) + 2) // 3)))
    for i, row in enumerate(tiles):
        x, y = (i % 3) * W, (i // 3) * 384
        sheet.paste(row[0], (x, y))
        sheet.paste(row[1], (x + 384, y))
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pairs.png", sys.argv[2].split(",") if len(sys.argv) > 2 else None)
