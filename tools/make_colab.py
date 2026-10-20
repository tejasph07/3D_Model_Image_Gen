"""Write colab/anime_gpu.ipynb: a self-contained Colab notebook that runs the anime engine on Colab's free GPU
and serves it through a Cloudflare quick tunnel (no Cloudflare account). The repo is private, so the notebook
carries its own copy of the engine code. python -m tools.make_colab"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def main():
    files = {"animescene/__init__.py": "", "animescene/diffusion.py": (ROOT / "animescene/diffusion.py").read_text(),
             "animescene/gpu_server.py": (ROOT / "animescene/gpu_server.py").read_text(),
             "fetch_models.py": (ROOT / "tools/fetch_models.py").read_text()}
    writer = "import os, json\nFILES = json.loads(%r)\nfor p, src in FILES.items():\n    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)\n    open(p, 'w').write(src)\nprint('wrote', list(FILES))" % json.dumps(files)
    cells = [
        cell("# Anime pass GPU server\n\n1. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.\n"
             "2. Wait for the last cell to print a URL and a token (about 5-8 min the first time: it downloads ~9 GB of models).\n"
             "3. Paste both into the chat. Keep this tab open while the panels render.", "markdown"),
        cell("!nvidia-smi -L\n!pip -q install diffusers==0.41.0 transformers accelerate safetensors opencv-python-headless"),
        cell(writer),
        cell("!python fetch_models.py --dest models --source hf"),
        cell("!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared"),
        cell("import subprocess, re, time\nfrom animescene.gpu_server import serve\ntoken = serve('models', 7860)\n"
             "p = subprocess.Popen(['./cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://localhost:7860'],\n"
             "                     stderr=subprocess.PIPE, text=True)\nurl = None\nwhile url is None:\n"
             "    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', p.stderr.readline())\n    url = m and m.group(0)\n"
             "print('URL  :', url)\nprint('TOKEN:', token)"),
        cell("# keeps the session busy so Colab doesn't idle it out while panels render\nimport time\nwhile True:\n    time.sleep(60)"),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "colab" / "anime_gpu.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(out)


if __name__ == "__main__":
    main()
