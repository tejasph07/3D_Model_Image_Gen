"""Collect rendered stories into a static showcase site (index.html + assets).

usage: build_showcase.py OUT_DIR STORY_JSON:RENDER_DIR [STORY_JSON:RENDER_DIR ...]
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def jpg(src: Path, dst: Path, max_w: int, q: int = 86):
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=q, optimize=True, progressive=True)
    return dst


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    stories = []
    for arg in sys.argv[2:]:
        sj, rd = arg.split(":")
        story = json.loads(Path(sj).read_text())
        rd = Path(rd)
        key = Path(sj).stem
        cast = {c["name"]: c for c in json.loads((rd / "cast.json").read_text())}
        entry = {"key": key, "title": story["title"], "pages": [], "panels": [], "cast": []}
        for p in sorted(rd.glob("page_*.png")):
            entry["pages"].append(str(jpg(p, out / key / p.with_suffix(".jpg").name, 1600, 88).relative_to(out)))
        for name, c in cast.items():
            if c.get("sheet") and Path(c["sheet"]).exists():
                f = jpg(Path(c["sheet"]), out / key / "sheets" / f"{name}.jpg", 1216)
                entry["cast"].append({"name": name, "sheet": str(f.relative_to(out)),
                                      "tags": ", ".join(x for x in [c["hair_color"] + " hair", c["hair"], c["eyes"] + " eyes",
                                                                    c["outfit"]] if x),
                                      "injuries": [{k: i[k] for k in ("kind", "bone", "t", "angle", "scene") if k in i}
                                                   for i in c.get("injuries", [])]})
        for panel in story["panels"]:
            pd = rd / panel["id"]
            if not (pd / "final.png").exists():
                continue
            pe = {"id": panel["id"], "caption": panel.get("caption", ""),
                  "actions": [f"{c['name']}: {c.get('action', 'stance')}" + (f" → {c['target']}" if c.get("target") else "")
                              for c in panel["characters"]],
                  "camera": panel.get("camera", {})}
            pe["final"] = str(jpg(pd / "final.png", out / key / panel["id"] / "final.jpg", 1400).relative_to(out))
            for k in ("control_pose", "control_depth", "stage_debug"):
                if (pd / f"{k}.png").exists():
                    pe[k] = str(jpg(pd / f"{k}.png", out / key / panel["id"] / f"{k}.jpg", 760, 80).relative_to(out))
            if (pd / "stage3d.json").exists():
                dst = out / key / panel["id"] / "stage3d.json"
                dst.write_text((pd / "stage3d.json").read_text())
                pe["stage"] = str(dst.relative_to(out))
            entry["panels"].append(pe)
        stories.append(entry)
    (out / "data.json").write_text(json.dumps(stories, indent=1))
    tpl = (ROOT / "tools" / "showcase_template.html").read_text()
    (out / "index.html").write_text(tpl.replace("/*DATA*/null", json.dumps(stories)))
    print(out / "index.html")


if __name__ == "__main__":
    main()
