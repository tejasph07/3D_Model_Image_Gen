"""HTTP front for the diffusion Engine, to run it on a borrowed GPU (e.g. a free Colab T4).

POST /sample  {token, prompt, negative, width, height, steps, cfg, seed, strength, control_end, base_weight,
               init: png64, controls: {kind: [png64, scale]}, regions: [[prompt, png64 mask]]} -> {image: png64}
`RemoteEngine(url, token)` has the same `sample()` as Engine, so tools/anime_pass.py runs either way.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image


def _enc(im: Image.Image) -> str:
    b = io.BytesIO()
    im.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def _dec(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s)))


def _mask_enc(m: np.ndarray) -> str:
    return _enc(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)))


class RemoteEngine:
    def __init__(self, url: str, token: str, timeout=900):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def sample(self, prompt, negative, width, height, steps=26, cfg=6.0, seed=0, controls=None, control_end=0.75,
               regions=None, base_weight=0.3, init_image=None, strength=1.0, progress="", **_):
        body = {"token": self.token, "prompt": prompt, "negative": negative, "width": width, "height": height,
                "steps": steps, "cfg": cfg, "seed": seed, "control_end": control_end, "base_weight": base_weight,
                "strength": strength, "init": _enc(init_image) if init_image is not None else None,
                "controls": {k: [_enc(im), sc] for k, (im, sc) in (controls or {}).items()},
                "regions": [[p, _mask_enc(m)] for p, m in (regions or [])]}
        req = urllib.request.Request(self.url + "/sample", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return _dec(json.loads(r.read())["image"]).convert("RGB")


def serve(models="models", port=7860, token=None):
    import secrets
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from animescene.diffusion import Engine
    token = token or secrets.token_urlsafe(16)
    eng = Engine(models)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if q.get("token") != token:
                self.send_response(403)
                self.end_headers()
                return
            controls = {k: (_dec(v[0]).convert("RGB"), float(v[1])) for k, v in q["controls"].items()}
            regions = [(p, np.asarray(_dec(m).convert("L"), np.float32) / 255) for p, m in q["regions"]]
            with lock:  # one image at a time on the GPU
                im = eng.sample(q["prompt"], q["negative"], q["width"], q["height"], steps=q["steps"], cfg=q["cfg"],
                                seed=q["seed"], controls=controls, control_end=q["control_end"], regions=regions,
                                base_weight=q["base_weight"], strength=q["strength"],
                                init_image=_dec(q["init"]).convert("RGB") if q.get("init") else None, progress="remote")
            out = json.dumps({"image": _enc(im)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return token
