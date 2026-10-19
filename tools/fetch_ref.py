"""Download a reference image into refs/<action>/ (resized to <= 800 px) and record its source.

  python tools/fetch_ref.py ACTION NAME IMAGE_URL SOURCE_PAGE_URL "what it shows"
"""
import io
import subprocess
import sys
from pathlib import Path

from PIL import Image

action, name, url, page, note = sys.argv[1:6]
root = Path(__file__).resolve().parents[1] / "refs"
dest = root / action / f"{name}.jpg"
dest.parent.mkdir(parents=True, exist_ok=True)
data = subprocess.run(["curl", "-fsSL", "--retry", "6", "--retry-delay", "5", "--retry-all-errors", "-A", "animescene-refs/0.1 (tejasph07@gmail.com)", url],
                      check=True, capture_output=True).stdout
im = Image.open(io.BytesIO(data))
im.seek(0)
im = im.convert("RGB")
im.thumbnail((800, 800))
im.save(dest, quality=88)
src = root / "SOURCES.md"
if not src.exists():
    src.write_text("# Reference image sources\n\nUsed only to compare poses against; each file links to its source page.\n\n"
                   "| file | shows | source |\n|---|---|---|\n")
line = f"| `{action}/{name}.jpg` | {note} | {page} |\n"
if line not in src.read_text():
    with src.open("a") as f:
        f.write(line)
print(dest, im.size)
