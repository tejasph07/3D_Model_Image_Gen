"""Command line: scene file in, OpenPose PNG + JSON out.

    python -m scene3d examples/duel.scene -o out/
    echo "create,a,HUMAN" | python -m scene3d - -o out/ --name quick
"""
from __future__ import annotations

import argparse
import os
import sys

from .camera import make_camera
from .openpose import write_json
from .render import render
from .solver import build_scene


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="scene3d",
        description="Generate OpenPose skeleton images from 3-word scene scripts.")
    ap.add_argument("input", help="scene file, or '-' for stdin")
    ap.add_argument("-o", "--outdir", default="out")
    ap.add_argument("--name", default=None, help="output basename")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--camera", default=None,
                    help="override camera view (front/back/left/right/"
                         "three_quarter/three_quarter_left/top/low/auto)")
    args = ap.parse_args(argv)

    if args.input == "-":
        text = sys.stdin.read()
        name = args.name or "scene"
    else:
        with open(args.input) as f:
            text = f.read()
        name = args.name or os.path.splitext(os.path.basename(args.input))[0]

    scene = build_scene(text)
    if args.camera:
        scene.camera_cmd = (args.camera, scene.camera_cmd[1])
    cam = make_camera(scene, args.width, args.height)
    img, cam = render(scene, args.width, args.height, camera=cam)

    os.makedirs(args.outdir, exist_ok=True)
    png_path = os.path.join(args.outdir, name + ".png")
    json_path = os.path.join(args.outdir, name + ".json")
    img.save(png_path)
    write_json(scene, cam, json_path)

    for w in scene.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote {png_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
