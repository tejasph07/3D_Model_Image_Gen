"""Fetch the model weights into a diffusers layout.

Weights used (all public on HuggingFace):
  sdxl/              Ine007/waiIllustriousSDXL_v150          (WAI-Illustrious-SDXL v15, anime SDXL)
  vae_fix/           madebyollin/sdxl-vae-fp16-fix
  controlnet_union/  xinsir/controlnet-union-sdxl-1.0 (promax)

`--source hf` downloads them from HuggingFace (needs huggingface_hub and access to huggingface.co).
`--source dockerhub` is for networks where HuggingFace is blocked but Docker Hub is not: it pulls
the HuggingFace-cache layers of public images that baked these exact repos in, extracts *only*
weight/config data (never code), and checks every LFS blob against the sha256 it is named by.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request

ACCEPT = ("application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json, "
          "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json")
# (image, layer digest prefix, kind)
DOCKER = {
    "sdxl": ("wxderxr/wai-illustrious-v150", "368076eb0f25", "hfcache"),
    "vae_fix": ("wxderxr/wai-illustrious-v150", "bb88cb87f8f8", "hfcache"),
    "controlnet_union": ("jsmorphogen/worker-controlnet-union", "b21b11d74ca7", "dir"),
}
HF = {"sdxl": "Ine007/waiIllustriousSDXL_v150", "vae_fix": "madebyollin/sdxl-vae-fp16-fix",
      "controlnet_union": ("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors")}
SAFE = (".safetensors", ".json", ".txt")


def _token(repo):
    u = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    return json.load(urllib.request.urlopen(u))["token"]


def _get(repo, path, tok, accept=True):
    h = {"Authorization": "Bearer " + tok}
    if accept:
        h["Accept"] = ACCEPT
    return urllib.request.urlopen(urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/{path}", headers=h))


def _layer(repo, prefix):
    tok = _token(repo)
    m = json.load(_get(repo, "manifests/latest", tok))
    if "manifests" in m:
        d = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"][0]["digest"]
        m = json.load(_get(repo, f"manifests/{d}", tok))
    digest = [l["digest"] for l in m["layers"] if l["digest"][7:].startswith(prefix)][0]
    return _get(repo, f"blobs/{digest}", tok, accept=False)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fetch_docker(name, dest):
    repo, prefix, kind = DOCKER[name]
    out = os.path.join(dest, name)
    os.makedirs(out, exist_ok=True)
    blobs, links = {}, []
    with _layer(repo, prefix) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
        for mem in tf:
            if kind == "hfcache":
                if mem.issym() and "/snapshots/" in mem.name:
                    links.append((mem.name.split("/snapshots/")[1].split("/", 1)[1], os.path.basename(mem.linkname)))
                elif mem.isfile() and "/blobs/" in mem.name:
                    p = os.path.join(out, ".blobs", os.path.basename(mem.name))
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    src = tf.extractfile(mem)
                    with open(p, "wb") as f:
                        while chunk := src.read(1 << 24):
                            f.write(chunk)
                    blobs[os.path.basename(mem.name)] = p
            elif mem.isfile() and mem.name.endswith(SAFE):
                p = os.path.join(out, os.path.basename(mem.name))
                src = tf.extractfile(mem)
                with open(p, "wb") as f:
                    while chunk := src.read(1 << 24):
                        f.write(chunk)
                print("  ", p)
    for rel, blob in links:
        if not rel.endswith(SAFE):
            continue
        if len(blob) == 64 and _sha256(blobs[blob]) != blob:
            raise SystemExit(f"checksum mismatch for {rel}")
        p = os.path.join(out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        os.replace(blobs[blob], p) if not os.path.exists(p) else None
        print("  ", p)


def fetch_hf(name, dest):
    from huggingface_hub import hf_hub_download, snapshot_download
    out = os.path.join(dest, name)
    src = HF[name]
    if isinstance(src, tuple):
        repo, fname = src
        os.makedirs(out, exist_ok=True)
        hf_hub_download(repo, "config_promax.json", local_dir=out)
        hf_hub_download(repo, fname, local_dir=out)
        os.replace(os.path.join(out, "config_promax.json"), os.path.join(out, "config.json"))
        os.replace(os.path.join(out, fname), os.path.join(out, "diffusion_pytorch_model.safetensors"))
    else:
        snapshot_download(src, local_dir=out, allow_patterns=["*.json", "*.txt", "*.safetensors"],
                          ignore_patterns=["*fp32*", "*.bin"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.environ.get("ANIMESCENE_MODELS", "models"))
    ap.add_argument("--source", choices=["hf", "dockerhub"], default="hf")
    a = ap.parse_args()
    for name in DOCKER:
        print(f"[{name}]")
        (fetch_hf if a.source == "hf" else fetch_docker)(name, a.dest)


if __name__ == "__main__":
    main()
