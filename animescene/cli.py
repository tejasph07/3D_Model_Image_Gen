"""Command line: stage / render / story / character."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .characters import Cast, design


def main(argv=None):
    ap = argparse.ArgumentParser(prog="animescene", description="Story scenes -> grounded 3D -> anime images")
    ap.add_argument("--cast", default="cast.json", help="persistent character file (default: cast.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage", help="solve the 3D scene and write control images only (no image model)")
    s.add_argument("scene")
    s.add_argument("-o", "--out", default="out")

    r = sub.add_parser("render", help="full pipeline: stage + SDXL/ControlNet generation")
    r.add_argument("scene", nargs="+")
    r.add_argument("-o", "--out", default="out")
    r.add_argument("-q", "--quality", default="high", choices=["draft", "standard", "high"])
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--models", default=None, help="model directory (default $ANIMESCENE_MODELS or /home/user/models)")

    st = sub.add_parser("story", help="direct a scene from story text with Claude, then stage (and render)")
    st.add_argument("text", help="story text or a path to a .txt file")
    st.add_argument("-o", "--out", default="out")
    st.add_argument("--render", action="store_true")
    st.add_argument("-q", "--quality", default="high", choices=["draft", "standard", "high"])

    m = sub.add_parser("comic", help="render a multi-panel story file (shared cast, persistent injuries) to manga pages")
    m.add_argument("story")
    m.add_argument("-o", "--out", default="out/comic")
    m.add_argument("-q", "--quality", default="high", choices=["draft", "standard", "high"])
    m.add_argument("--panels", default=None, help="comma-separated panel ids to (re)render")
    m.add_argument("--models", default=None)

    c = sub.add_parser("character", help="design a character on the fly and save it to the cast")
    c.add_argument("name")
    c.add_argument("description", nargs="?", default="")
    c.add_argument("--archetype")
    c.add_argument("--seed", type=int)

    a = ap.parse_args(argv)
    cast = Cast(a.cast)

    if a.cmd == "comic":
        from .diffusion import Engine
        from .story import run_story
        engine = Engine(a.models) if a.models else Engine()
        run_story(a.story, engine, a.out, a.quality, only=a.panels.split(",") if a.panels else None)
        return

    if a.cmd == "character":
        ch = design(a.name, seed=a.seed, archetype=a.archetype, description=a.description,
                    avoid_palettes=tuple(x.palette for x in cast.chars.values()))
        cast.add(ch)
        cast.save()
        print(json.dumps(ch.to_dict(), indent=2))
        return

    if a.cmd == "story":
        from .story import story_to_scene
        text = Path(a.text).read_text() if Path(a.text).is_file() else a.text
        summary = "\n".join(f"- {n}: {c.appearance_tags()}" for n, c in cast.chars.items())
        spec = story_to_scene(text, summary)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.json").write_text(json.dumps(spec, indent=2))
        print(f"scene -> {out / 'scene.json'}")
        if not a.render:
            _stage(spec, out, cast)
            return
        a.scene, a.seed, a.models = [str(out / "scene.json")], None, None
        a.cmd = "render"

    if a.cmd == "stage":
        _stage(a.scene, Path(a.out), cast)
        return

    if a.cmd == "render":
        from .diffusion import Engine
        from .pipeline import generate
        engine = Engine(a.models) if a.models else Engine()
        for sc in a.scene:
            name = Path(sc).stem
            out = Path(a.out) / name if len(a.scene) > 1 else Path(a.out)
            generate(sc, engine, out, a.quality, seed=a.seed, cast=cast)
        cast.save()


def _stage(spec, out: Path, cast: Cast):
    from .scene import stage_scene
    out.mkdir(parents=True, exist_ok=True)
    st = stage_scene(spec, cast)
    st.pose.save(out / "control_pose.png")
    st.depth.save(out / "control_depth.png")
    st.normal.save(out / "control_normal.png")
    st.debug.save(out / "stage_debug.png")
    for a in st.scene.actors:
        print(f"{a.name:>10}  {a.action:<13} at ({a.origin[0]:+.2f}, {a.origin[2]:+.2f})  tags: {', '.join(a.out.tags[:4])}")
    cast.save()
    print(f"controls -> {out}")


if __name__ == "__main__":
    sys.exit(main())
