"""Summarise the mesh contact report a 3D render writes next to its PNG (`*_contacts.json`).

  python tools/clip_report.py RENDER.png [--touch] [--min=0.004]

clip  = one actor's surface inside another's (depth in mm, vertex count, where)
floor = vertices under the ground; self = forearm/hand inside the own torso or legs.
"""
import json
import sys
from pathlib import Path


def load(png):
    return json.loads(Path(str(png).replace(".png", "_contacts.json")).read_text())


def worst(rep, min_depth=0.004, ignore=()):
    """Deepest clipping in the frame (m) and its label, for checks/tests."""
    best = (0.0, "")
    for p in rep["pairs"]:
        for e in p["clip"]:
            if e["depth"] >= min_depth and not any(k in e["pair"] for k in ignore) and e["depth"] > best[0]:
                best = (e["depth"], f"{p['a']}->{p['b']} {e['pair']}")
    for i, fl in enumerate(rep["floor"]):
        for e in fl:
            if e["depth"] >= min_depth and e["depth"] > best[0]:
                best = (e["depth"], f"{i} floor {e['part']}")
    for i, sl in enumerate(rep["self"]):
        for e in sl:
            if e["depth"] >= min_depth and e["depth"] > best[0]:
                best = (e["depth"], f"{i} self {e['pair']}")
    return best


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    mn = float(opts.get("min", 0.004))
    for png in args:
        rep = load(png)
        print(f"== {png}")
        for p in rep["pairs"]:
            for e in p["clip"]:
                if e["depth"] >= mn:
                    print(f"  clip  {p['a']}->{p['b']} {e['pair']:34s} {e['depth'] * 1000:5.1f}mm n={e['n']:5d} at={[round(x, 3) for x in e['at']]}")
            if "touch" in opts:
                for e in p["touch"][:12]:
                    print(f"  touch {p['a']}->{p['b']} {e['pair']:34s} n={e['n']}")
        for i, fl in enumerate(rep["floor"]):
            for e in fl:
                print(f"  floor {i} {e['part']:20s} {e['depth'] * 1000:5.1f}mm n={e['n']}")
        for i, sl in enumerate(rep["self"]):
            for e in sl:
                print(f"  self  {i} {e['pair']:30s} {e['depth'] * 1000:5.1f}mm n={e['n']}")
        print("  worst:", worst(rep, mn))


if __name__ == "__main__":
    main()
