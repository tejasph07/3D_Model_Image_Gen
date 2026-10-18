"""Story text -> scene spec, using Claude to do the directing.

The scene format reference (SCENE_FORMAT.md) is the system prompt; Claude
returns one JSON scene that the stage then solves. Requires the `anthropic`
package and credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FORMAT_DOC = Path(__file__).resolve().parents[1] / "SCENE_FORMAT.md"

INSTRUCTIONS = """You are the storyboard director for an anime illustration pipeline.
Turn the story moment you are given into ONE scene in the JSON format documented below.
Pick the single most dramatic instant, give every character one action from the action list,
make attackers and reactions point at each other with `target`, pin one member of each
fighting group with `at`, and choose a camera that keeps every fighter readable.
Use booru-style tags for setting, outfits and lighting. Reply with the JSON object only.

"""


def story_to_scene(story: str, cast_summary: str = "", model: str = "claude-opus-5-5") -> dict:
    import anthropic

    client = anthropic.Anthropic()
    system = INSTRUCTIONS + FORMAT_DOC.read_text()
    user = story if not cast_summary else f"Known characters (reuse their names):\n{cast_summary}\n\nStory:\n{story}"
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to direct this scene")
    text = "".join(b.text for b in response.content if b.type == "text")
    return parse_scene_json(text)


def parse_scene_json(text: str) -> dict:
    """Extract the first JSON object from a reply (tolerates code fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    scene = json.loads(m.group(0))
    if not isinstance(scene.get("characters"), list) or not scene["characters"]:
        raise ValueError("scene has no characters")
    return scene


# ------------------------------------------------------------------ multi-panel stories
def panel_spec(story: dict, panel: dict) -> dict:
    """A panel inherits the story's setting/lighting/seed; characters come from the cast by name."""
    spec = {k: story[k] for k in ("setting", "lighting", "style", "negative") if k in story}
    spec.update({k: v for k, v in panel.items() if k not in ("layout", "caption", "dialogue", "sfx")})
    spec["title"] = f"{story.get('title', 'story')}/{panel['id']}"
    spec.setdefault("seed", story.get("seed", 0) + 17 * (story["panels"].index(panel) + 1))
    return spec


def run_story(path, engine, out_dir, quality="high", only: list | None = None, log=print) -> list:
    """Render every panel in order with one shared cast (designs, sheets and injuries persist),
    then lay the panels out as manga pages."""
    from .characters import Cast
    from .manga import compose_pages
    from .pipeline import generate

    story = json.loads(Path(path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cast = Cast(out / "cast.json")
    used = [c.palette for c in cast.chars.values()]
    for cs in story.get("cast", []):
        if cs["name"] not in cast.chars:
            used.append(cast.resolve(cs, used).palette)
    cast.save()
    results = []
    for panel in story["panels"]:
        pdir = out / panel["id"]
        if only and panel["id"] not in only:
            if (pdir / "final.png").exists():
                results.append((panel, pdir))
            continue
        log(f"=== panel {panel['id']}")
        generate(panel_spec(story, panel), engine, pdir, quality, cast=cast, log=log)
        results.append((panel, pdir))
    pages = compose_pages(story, [(p, d) for p, d in results if (d / "final.png").exists()], out)
    log(f"[story] pages: {', '.join(str(p) for p in pages)}")
    return pages
