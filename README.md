# animescene — story scenes → grounded 3D → anime illustrations

Turns complicated story moments ("Akane brings her katana down on Ren, who
catches it on his blade; beside them Mira's roundhouse kick snaps into the
giant's jaw and sends him flying") into **one coherent, full-colour anime
illustration** where every blow lands where it should.

Image models alone can't do multi-character fights: limbs merge, punches
miss, colours and outfits swap between people. This pipeline makes the
*geometry* exact first and only then lets the image model paint:

```
story text ──(Claude, optional)──▶ scene JSON
scene JSON ──▶ 3D stage: rigs, IK, contacts, hit reactions, collisions, camera
           ──▶ controls: OpenPose (occlusion-aware) + depth + per-character masks
           ──▶ SDXL anime model + ControlNet-Union (pose+depth)
               + regional prompting (each character's prompt only paints its own 3D silhouette)
           ──▶ hires pass ──▶ per-character detail passes (zoomed re-render of each fighter)
```

## What the 3D stage guarantees

* **Contacts are real.** A punch puts the knuckles on the target's face, a
  kick puts the foot on the head (support foot planted), a block puts the
  defender's blade *across* the attacker's blade at the contact point, a
  spear point stops at the body surface.
* **Reactions follow the blow.** `knocked_back` / `staggered` victims are
  moved so the struck part is where the blow landed, a moment later along
  the attack's direction, never pulled toward the attacker and never
  intersecting the attacker's body.
* **Placement by intent.** Pin one fighter per group; everyone else is placed
  at their action's natural range (computed from both bodies' proportions),
  side by side across the screen so silhouettes stay readable.
* **Exact bone lengths, flat ground.** Analytic two-bone IK; grounded poses
  re-solve after grounding so hand targets stay exact; airborne poses stay
  airborne.
* **Characters on the fly, consistent forever.** Describe someone loosely or
  not at all; the designer fills in a coherent anime design (hair, eyes,
  archetype outfit, weapon, body). The first time a character appears it gets
  a generated **turnaround sheet** (front / side / back). Every later render
  runs the sheet through the UNet and lets each pixel of that character's 3D
  region attend to the sheet's features (*regional reference attention*), so
  face, hair, outfit and weapon stay on-model across panels and never bleed
  onto another fighter. Designs and sheets persist in `cast.json`; LoRAs can
  be attached per character as well.
* **Injuries are code, not prompts.** A wound is stored on the character as a
  point on the body surface (bone, position along it, angle around it) plus a
  seeded pattern. Blows record wounds at their exact 3D contact point; every
  later panel re-projects them through the posed body, tests visibility
  against the depth buffer and composites the identical cut / bruise /
  bandage. The image model is never asked to draw injuries.
* **Grappling and dirty fighting.** Headlocks, rear chokes, ankle locks,
  knee strikes in the clinch, back-of-the-knee kicks, spinning back kicks,
  palm strikes with an energy blast, every boxing punch — with reactions that
  follow the blow (doubling over, clutching the stomach, stumbling, knee
  buckling, being knocked flying). Soft-tissue hits dent the victim's body in
  the depth control.

## Quick start

```bash
pip install -r requirements.txt          # or a conda env with pytorch (CPU or CUDA)
python tools/fetch_models.py --dest models   # ~9 GB; --source dockerhub if HuggingFace is blocked
export ANIMESCENE_MODELS=$PWD/models

# 3D stage only (seconds, no model): control images + debug view
python -m animescene stage scenes/shrine_clash.json -o out/shrine

# full render
python -m animescene render scenes/shrine_clash.json -o out/shrine -q high

# design a character on the fly (saved to cast.json, reused by name)
python -m animescene character Yui "pink-haired petite mage girl in a violet robe"

# a whole multi-panel story -> manga pages (shared cast, persistent injuries)
python -m animescene comic stories/alley_rin.json -o out/alley

# story -> scene with Claude (needs ANTHROPIC_API_KEY), then render
python -m animescene story "Kaze drops out of the storm onto Lena..." -o out/ambush --render
```

The scene format is documented in [SCENE_FORMAT.md](SCENE_FORMAT.md) — it is
also the system prompt the story director uses, so an LLM can write scenes.

Each render writes `final.png` plus `base.png`, `hires.png`,
`detail_*.png`, the control images, `stage_debug.png` (skeletons over depth
and region masks) and `prompts.json`.

## Models

| role | model |
|---|---|
| base | WAI-Illustrious-SDXL v15 (anime SDXL, danbooru-tag prompting) |
| control | xinsir ControlNet-Union SDXL ProMax (OpenPose + depth in one net) |
| VAE | sdxl-vae-fp16-fix |

Quality presets (`-q`): `draft` (18 steps, no hires/detail), `standard`
(+ detail), `high` (28 steps + 1.4× hires + detail). On a 4-core CPU
(bf16, AMX) `high` takes ~30 min for a 4-character scene; a GPU does it in
about a minute.

## Layout

```
animescene/
  rig.py         anime-proportioned body, FK + two-bone IK
  actions.py     action library (punch, kick, flying_kick, slash, thrust, block, leap_attack,
                 knocked_back, staggered, dodge, cast, charge, grab, point, kneel, fallen, ...)
  stage.py       placement, interaction-aware solving, hit alignment, collisions
  camera.py      shot presets + automatic framing
  raster.py      analytic ray caster (depth, instance ids, normals)
  controls.py    OpenPose with occlusion, depth, region masks, debug view
  characters.py  character designer, description parser, persistent cast
  prompts.py     global / per-character / background prompts from the solved stage
  diffusion.py   SDXL + ControlNet-Union sampler with regional cross-attention, img2img, inpaint
  pipeline.py    base -> hires -> per-character detail
  sheets.py      character turnaround sheets (identity references)
  injuries.py    persistent wounds: body-surface anchors, seeded decals, occlusion tests
  manga.py       page layout, narration boxes, speech bubbles, SFX
  story.py       story text -> scene JSON via Claude; multi-panel story runner
scenes/          single-scene examples
stories/         multi-panel stories (alley fight, boxing title bout)
showcase/        rendered results
tests/           stage physics, prompts, regional-attention maths
tools/           action_sheet.py (pose inspection), fetch_models.py
```

## Tests

```bash
python -m pytest tests -q
```

The stage tests assert physics, not plumbing: bone lengths for every
action, feet planted / airborne, fist-on-face and foot-on-head distances,
crossed blades with the attacker's edge stopping before the defender's
face, hit victims landing where the blow lands, no interpenetration in the
four-fighter scene, everyone in frame, region masks partitioning the image.
The regional-attention tests check that each pixel attends only to its own
character's prompt and unowned pixels fall back to the scene prompt.
