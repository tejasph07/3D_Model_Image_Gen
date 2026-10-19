"""Prose -> fight-scene file, written by an LLM inside the engine's vocabulary, then checked and repaired.

The LLM only *chooses*: which moves happen between whom, where people stand, which keyframes the panels
show, the cameras, captions, dialogue and sound effects. It never writes poses - every pose comes from the
keyframe library, so what it asks for is always something the solver can do. Its output is validated against
the catalog (moves, keyframe names, cast, models, poses, layouts, sites); errors go back to it to fix.

Backends (env):
  gemini  - Google AI Studio: GEMINI_API_KEY (model GEMINI_MODEL, default gemini-2.5-flash)
  openai  - any OpenAI-compatible chat endpoint: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
python -m animescene.storygen STORY.txt OUT.json [--backend gemini|openai] [--render OUTDIR]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "stories" / "alley_rin_fight.json"

SCHEMA = """{
 "title": str, "setting": str (words like alley/neon/street or ring/arena pick the set), "lighting": str,
 "cast": [{"name": str, "model": <one of models>, "scale"?: 0.9..1.15, "expression"?: <cast word>}],
 "injuries"?: [{"who": name, "spot": <site>, "kind": <injury kind>, "age"?: 0..1}],   // hurt before the story
 "beats": [{
   "move": <move name>, "a": name (does the move), "v": name (receives it),
   "at": [x, z] (metres, where the pair stands; keep 0,0 unless the fight travels), "yaw": degrees,
   "others": {name: {"pose": <bystander pose>, "at": [x, z], "yaw": degrees}},   // EVERY other cast member, 1.5-2.5 m away
   "injuries"?: [...], "treat"?: [{"who", "spot"}], "heal"?: 0..1,
   "panels": [{"key": <a keyframe name OF THIS MOVE>, "layout": wide|half|full|tall,
               "camera": {"shot": wide|medium|close|extreme, "az": deg, "elev": deg, "focus"?: pair|a|v|contact|all, "zoom"?: 0.8..1.3},
               "caption"?: str, "dialogue"?: [{"who": name, "text": str}], "sfx"?: [{"text": str, "at": "impact"}],
               "expression"?: {name: <expression tag>}}]
 }]
}"""

RULES = """Rules:
- Use only the moves, keyframe names, models, poses, sites and tags in the vocabulary. A panel's "key" must be one
  of the keys listed for that beat's move.
- Every cast member appears in every beat: as "a", "v" or in "others". Someone knocked out stays "down" in later beats.
- Bystanders stand 1.5-2.5 m from the pair and never at the same spot as each other.
- Panels: 1-3 per beat; consecutive half panels pair up side by side (reading order runs across beats); a page
  holds about 6 panels.
- "a" and "v" must differ. Keep dialogue short (manga bubbles: under 8 words).
- Close shots use focus "contact" for the moment of impact; wide shots use focus "all" to show everyone.
- Output ONLY the JSON object."""


def vocabulary() -> dict:
    from .sim.catalog import compact
    return compact()


def prompt(story_text: str, vocab: dict, errors: list | None = None, previous: str | None = None) -> str:
    ex = EXAMPLE.read_text() if EXAMPLE.exists() else ""
    p = ["You turn a short fight story into a manga scene file for a 3D posing engine.",
         "", "Vocabulary (the only things the engine can do):", json.dumps(vocab, indent=0),
         "", "Scene file format:", SCHEMA, "", RULES, "", "Example scene file:", ex, "", "Story:", story_text.strip()]
    if errors:
        p += ["", "Your previous answer:", previous or "", "", "It has these problems; return the corrected full JSON:"]
        p += [f"- {e}" for e in errors]
    return "\n".join(p)


# ----------------------------------------------------------------------------- backends
def _post(url, body, headers, timeout=180):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call_gemini(text: str, temperature=0.6) -> str:
    key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}
    r = _post(f"{base}/models/{model}:generateContent", body, {"x-goog-api-key": key})
    return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])


def call_openai(text: str, temperature=0.6) -> str:
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    body = {"model": os.environ.get("LLM_MODEL", "gemini-2.5-flash"), "temperature": temperature,
            "messages": [{"role": "user", "content": text}]}
    key = os.environ.get("LLM_API_KEY")
    r = _post(f"{base}/chat/completions", body, {"Authorization": f"Bearer {key}"} if key else {})
    return r["choices"][0]["message"]["content"]


BACKENDS = {"gemini": call_gemini, "openai": call_openai}


def parse(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])


# ----------------------------------------------------------------------------- checks
def validate(spec: dict, vocab: dict) -> list:
    """Everything the engine would trip over, as sentences the LLM can act on."""
    errs = []
    moves = vocab["moves"]
    cast = [c.get("name") for c in spec.get("cast", [])]
    if not cast:
        errs.append("cast is empty")
    if len(set(cast)) != len(cast):
        errs.append("cast names must be unique")
    for c in spec.get("cast", []):
        if c.get("model") not in vocab["models"]:
            errs.append(f"cast {c.get('name')}: model {c.get('model')!r} is not one of {vocab['models']}")
    sites = set(vocab["sites"])

    def site_ok(s):
        return s in sites or (s[-2:] in ("_l", "_r") and s[:-2] in sites)

    for w in spec.get("injuries", []):
        if w.get("who") not in cast or not site_ok(w.get("spot", "")) or w.get("kind", "bruise") not in vocab["injury_kinds"]:
            errs.append(f"story injury {w} has an unknown who/spot/kind")
    if not spec.get("beats"):
        errs.append("no beats")
    for i, b in enumerate(spec.get("beats", [])):
        tag = f"beat {i} ({b.get('move')})"
        mv = moves.get(b.get("move"))
        if mv is None:
            errs.append(f"{tag}: move {b.get('move')!r} does not exist; choose from {list(moves)}")
            continue
        if b.get("a") not in cast or b.get("v") not in cast:
            errs.append(f"{tag}: a/v must be cast names")
        if b.get("a") == b.get("v"):
            errs.append(f"{tag}: a and v are the same person")
        others = b.get("others", {})
        missing = [n for n in cast if n not in (b.get("a"), b.get("v")) and n not in others]
        if missing:
            errs.append(f"{tag}: {missing} missing from 'others' (everyone must be placed)")
        spots = []
        for n, o in others.items():
            if n not in cast:
                errs.append(f"{tag}: others has unknown person {n!r}")
            if n in (b.get("a"), b.get("v")):
                errs.append(f"{tag}: {n} is in the move and in others")
            if o.get("pose", "ready") not in vocab["bystander_poses"]:
                errs.append(f"{tag}: {n} pose {o.get('pose')!r} not in {vocab['bystander_poses']}")
            at = o.get("at")
            if not (isinstance(at, list) and len(at) == 2):
                errs.append(f"{tag}: {n} needs at: [x, z]")
                continue
            c = b.get("at", [0, 0])
            if (at[0] - c[0]) ** 2 + (at[1] - c[1]) ** 2 < 1.2 ** 2:
                errs.append(f"{tag}: {n} at {at} is too close to the fighting pair (keep 1.5-2.5 m away)")
            for s in spots:
                if (at[0] - s[0]) ** 2 + (at[1] - s[1]) ** 2 < 0.8 ** 2:
                    errs.append(f"{tag}: {n} stands on top of another bystander at {s}")
            spots.append(at)
        if not b.get("panels"):
            errs.append(f"{tag}: no panels")
        for j, p in enumerate(b.get("panels", [])):
            pt = f"{tag} panel {j}"
            if p.get("key") not in mv["keys"]:
                errs.append(f"{pt}: key {p.get('key')!r} is not a keyframe of {b.get('move')}; use one of {mv['keys']}")
            if p.get("layout", "wide") not in vocab["layouts"]:
                errs.append(f"{pt}: layout must be one of {vocab['layouts']}")
            cam = p.get("camera", {})
            if cam.get("shot", "medium") not in vocab["shots"]:
                errs.append(f"{pt}: shot must be one of {vocab['shots']}")
            if cam.get("focus", "pair") not in vocab["focus"] + cast:
                errs.append(f"{pt}: focus must be one of {vocab['focus']}")
            for d in p.get("dialogue", []):
                if d.get("who") not in cast:
                    errs.append(f"{pt}: dialogue by unknown {d.get('who')!r}")
    return errs


def lint_layout(spec: dict):
    """Fix what needs no judgement: half panels pair up in reading order (across beats); a half left without
    a partner becomes wide."""
    run = []
    for p in [p for b in spec.get("beats", []) for p in b.get("panels", [])] + [None]:
        if p is not None and p.get("layout") == "half":
            run.append(p)
            continue
        if len(run) % 2 == 1:
            run[-1]["layout"] = "wide"
        run = []
    return spec


def generate(story_text: str, backend="gemini", tries=3, log=print) -> dict:
    vocab = vocabulary()
    call = BACKENDS[backend]
    errors, raw = None, None
    for attempt in range(tries):
        t0 = time.time()
        raw = call(prompt(story_text, vocab, errors, raw))
        try:
            spec = lint_layout(parse(raw))
            errors = validate(spec, vocab)
        except (json.JSONDecodeError, ValueError) as e:
            spec, errors = None, [f"not valid JSON: {e}"]
        log(f"[storygen] attempt {attempt + 1}: {len(errors)} problems ({time.time() - t0:.1f} s)")
        if not errors:
            return spec
        for e in errors[:8]:
            log("   - " + e)
    raise RuntimeError("the LLM could not produce a valid scene: " + "; ".join(errors[:5]))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    spec = generate(Path(args[0]).read_text(), backend=opts.get("backend", "gemini"))
    Path(args[1]).write_text(json.dumps(spec, indent=1))
    print(args[1])
    if "render" in opts:
        from .sim.fight import run
        run(spec, opts["render"])
