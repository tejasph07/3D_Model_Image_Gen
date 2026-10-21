"""Wrapper around anifun.ai's free Photo-to-Anime (no account): a headless browser uses the page as a visitor
would. Free guest allowance: 30 credits a day = 3 images (9 each); the wrapper stops when it is used up.
Faithful to the input (layout, poses, clothes), cleaner anime faces and lines; it can drift identity details
(skin tone, eye colour). ~30-60 s per image."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("Anime", "Kawaii anime", "Manga", "Webtoon", "Ghibli", "Cyberpunk", "Line art")


def photo_to_anime(src, dst, style="Anime", log=print) -> bool:
    tmp = Path(dst).with_suffix(".anifun.webp")
    r = subprocess.run(["node", str(ROOT / "render3d" / "anifun.mjs"), str(src), str(tmp), style], capture_output=True, text=True,
                       timeout=900, cwd=ROOT / "render3d")
    info = json.loads((r.stdout.strip().splitlines() or ["{}"])[-1])
    log(f"[anifun] {Path(dst).name}: {info}")
    if not info.get("ok"):
        return False
    W, H = Image.open(src).size
    Image.open(tmp).convert("RGB").resize((W, H), Image.LANCZOS).save(dst)
    tmp.unlink()
    return True
