"""Manga page layout: panels, narration boxes, speech bubbles and sound effects.

Bubbles and SFX are placed from the 3D stage (speaker head / impact point projected into
the panel), never painted by the image model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE_W = 1600
GUTTER = 22
MARGIN = 48
CELL_ASPECT = {"wide": 2.1, "half": 0.95, "full": 1.45, "tall": 0.7}
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]


def _font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rows(panels):
    rows, cur = [], []
    for p, d in panels:
        lay = p.get("layout", "wide")
        if lay == "half":
            cur.append((p, d))
            if len(cur) == 2:
                rows.append(cur)
                cur = []
        else:
            if cur:
                rows.append(cur)
                cur = []
            rows.append([(p, d)])
    if cur:
        rows.append(cur)
    return rows


def _crop(img: Image.Image, meta: dict, aspect: float) -> tuple[Image.Image, tuple]:
    """Crop to the cell aspect, keeping the subjects (and heads) in frame."""
    W, H = img.size
    if W / H > aspect:
        cw, ch = int(H * aspect), H
    else:
        cw, ch = W, int(W / aspect)
    x0s, y0s, x1s, y1s = meta.get("subjects", [0, 0, W, H])
    cx = (x0s + x1s) / 2
    tops = [a["head_top"][1] - 0.6 * a.get("head_r_px", 20) for a in meta.get("actors", {}).values()]
    cy = (y0s + y1s) / 2
    x0 = int(min(max(cx - cw / 2, 0), W - cw))
    y0 = int(min(max(cy - ch / 2, 0), H - ch))
    if tops:  # never cut a head: if heads stick out of the window, slide it up (feet go first)
        y0 = int(max(0, min(y0, min(tops) - 0.04 * ch)))
    return img.crop((x0, y0, x0 + cw, y0 + ch)), (x0, y0, cw / W)


def _wrap(text, font, width, draw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _bubble(draw, center, text, font, tail_to, max_w):
    lines = _wrap(text, font, max_w, draw)
    lh = font.size * 1.15
    tw = max(draw.textlength(l, font=font) for l in lines)
    th = lh * len(lines)
    rx, ry = tw / 2 + font.size * 0.9, th / 2 + font.size * 0.75
    cx, cy = center
    # tail
    ang = math.atan2(tail_to[1] - cy, tail_to[0] - cx)
    base = [(cx + rx * 0.5 * math.cos(ang + s), cy + ry * 0.5 * math.sin(ang + s)) for s in (-0.25, 0.25)]
    tip = (cx + (tail_to[0] - cx) * 0.8, cy + (tail_to[1] - cy) * 0.8)
    draw.polygon([base[0], tip, base[1]], fill="white", outline="black", width=3)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill="white", outline="black", width=3)
    draw.polygon([base[0], tip, base[1]], fill="white")
    for i, l in enumerate(lines):
        w = draw.textlength(l, font=font)
        draw.text((cx - w / 2, cy - th / 2 + i * lh), l, fill="black", font=font)
    return rx, ry


def _sfx(canvas: Image.Image, pos, text, size):
    font = _font(size)
    layer = Image.new("RGBA", (size * len(text) + size, size * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((size // 2, size // 3), text, font=font, fill=(255, 245, 220), stroke_width=max(3, size // 10),
           stroke_fill=(20, 10, 10))
    layer = layer.rotate(12, expand=True, resample=Image.BICUBIC)
    canvas.alpha_composite(layer, (int(pos[0] - layer.width / 2), int(pos[1] - layer.height / 2)))


def compose_pages(story: dict, panels: list, out: Path, max_rows: int = 4) -> list:
    rows = _rows(panels)
    pages, page_rows = [], [rows[i:i + max_rows] for i in range(0, len(rows), max_rows)]
    inner = PAGE_W - 2 * MARGIN
    for pi, prow in enumerate(page_rows):
        heights = []
        for row in prow:
            n = len(row)
            cw = (inner - GUTTER * (n - 1)) / n
            heights.append(int(cw / CELL_ASPECT[row[0][0].get("layout", "wide")]))
        page_h = int(sum(heights) + GUTTER * (len(prow) - 1) + 2 * MARGIN + (70 if pi == 0 else 0))
        page = Image.new("RGBA", (PAGE_W, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = MARGIN
        if pi == 0:
            draw.text((MARGIN, 18), story.get("title", ""), fill="black", font=_font(40))
            y += 70
        for row, h in zip(prow, heights):
            n = len(row)
            cw = int((inner - GUTTER * (n - 1)) / n)
            x = MARGIN
            for panel, pdir in row:
                img = Image.open(pdir / "final.png").convert("RGB")
                meta = json.loads((pdir / "meta.json").read_text()) if (pdir / "meta.json").exists() else {}
                crop, (ox, oy, _) = _crop(img, meta, cw / h)
                k = cw / crop.width
                cell = crop.resize((cw, h), Image.LANCZOS).convert("RGBA")

                def to_cell(p):
                    return ((p[0] - ox) * k, (p[1] - oy) * k)

                heads = [to_cell(a["head"]) for a in meta.get("actors", {}).values()]
                for s in panel.get("sfx", []):
                    pts = meta.get("impacts", [])
                    if s.get("at") == "impact" and pts:
                        px, py = to_cell(pts[0])
                        # beside the hit, never on it or on a face: farthest candidate from both
                        cands = [(px + dx * cw, py + dy * h) for dx in (-0.28, 0.28) for dy in (-0.22, 0.0, 0.22)]
                        cands = [(min(max(x_, cw * 0.15), cw * 0.85), min(max(y_, h * 0.12), h * 0.88)) for x_, y_ in cands]
                        best = max(cands, key=lambda q: min([np.hypot(q[0] - hx, q[1] - hy) for hx, hy in heads]
                                                            + [np.hypot(q[0] - px, q[1] - py) * 1.5]))
                        _sfx(cell, best, s["text"], int(h * 0.11))
                cd = ImageDraw.Draw(cell)
                font = _font(max(18, int(h * 0.045)))
                used = []
                for j, dlg in enumerate(panel.get("dialogue", [])):
                    a = meta.get("actors", {}).get(dlg["who"])
                    if not a:
                        continue
                    hx, hy = to_cell(a["head_top"])
                    bx = min(max(hx + (-1 if hx > cw / 2 else 1) * cw * 0.12, cw * 0.14), cw * 0.86)
                    by = max(hy - h * 0.12, h * 0.1)
                    for ux, uy in used:  # don't stack bubbles
                        if abs(ux - bx) < cw * 0.25 and abs(uy - by) < h * 0.15:
                            by = uy + h * 0.17
                    _bubble(cd, (bx, by), dlg["text"], font, (hx, hy), cw * 0.28)
                    used.append((bx, by))
                if panel.get("caption"):
                    cf = _font(max(16, int(h * 0.03)))
                    tw = cd.textlength(panel["caption"], font=cf)
                    cd.rectangle([8, 8, 8 + tw + 24, 8 + cf.size * 1.6], fill=(250, 248, 240), outline="black", width=3)
                    cd.text((20, 8 + cf.size * 0.25), panel["caption"], fill="black", font=cf)
                page.alpha_composite(cell, (x, y))
                draw.rectangle([x, y, x + cw, y + h], outline="black", width=4)
                x += cw + GUTTER
            y += h + GUTTER
        path = out / f"page_{pi + 1}.png"
        page.convert("RGB").save(path)
        pages.append(path)
    return pages
