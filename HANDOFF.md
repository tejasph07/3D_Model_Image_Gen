# Handoff: 3D pose-quality work in progress

Branch `ccr-6eb5bf0c-3h0b54`. These notes pick up where the last session stopped.

## Setup in a fresh container

```bash
pip install -r requirements.txt               # the 3D path needs numpy + pillow only; torch etc. are for diffusion
(cd render3d && npm install)                  # three, @pixiv/three-vrm, playwright-core
bash tools/fetch_vrm.sh                       # 15 VRM models -> $ANIMESCENE_ASSETS/vrm (default /home/user/assets)
# Chromium: render3d/render.mjs uses $CHROMIUM or the playwright headless_shell under /opt/pw-browsers
python -m pytest tests -q                     # 86 tests, all passing at handoff
```

The diffusion weights (`/home/user/models`, WAI-Illustrious + ControlNet-Union) are only needed for img2img
refinement and the old diffusion path. `tools/fetch_models.py` downloads them.

## Pipeline (current primary path: full 3D)

story JSON -> `animescene/stage.py` solves every pose from the action library (`animescene/actions.py`)
-> `animescene/render3d.py` builds jobs -> `render3d/stage.js` poses the real VRM meshes in headless
Chromium (two-bone IK on each model's own bones, clavicles, hand shapes, palm direction, soft-body dents,
eyes, expressions) -> manga page. img2img refinement (`pipeline.refine3d`) is optional. **Ask the user
before running it on a whole story.**

## Inspection tools

- `tools/pose_gallery.py OUT.png 'attacker{json}/victim{json}' --az=a,b,c --elev=.. --shot=closeup --focus=impact:0.15,V.head --a=MODEL --v=MODEL`
  renders any action or pair from several angles. Pairs use a fixed layout (attacker on -x), so
  azimuth orbits the scene. Az 0 or 10 is a profile view.
- `tools/compare_ref.py OUT.png ref1.png,ref2.png <pose_gallery args>` puts reference images next to our renders.
- Models with visible skin, which make contacts easy to see: `fem_vroid` (bare midriff), `Sakurada_Fumiriya`
  and `Sendagaya_Shino` (short sleeves).
- Each render also writes `*_bones.json`: the mesh's real bone positions plus the dents measured on it.

## Done in this round

- Arm-shape library in the chest frame (`ARMS`, `_arm`). Free arms stay bent, and a free-arm clearance pass runs in the stage.
- Hand shapes: fist, relaxed, flat, grip, claw, pinch. `Limb.palm` sets the palm direction.
- Soft-body dents (`render3d/stage.js`: `softBody`, `measureDents`, `setDents`). An action returns `press=[{point, dir,
  radius, max_depth}]`. The renderer raycasts the real skinned mesh and dents it by exactly how far the
  effector went in, adds a bulge ring and tilts the shading normals. Arms and hands are masked out of denting.
- New actions: `press_stomach` + `flinch`, `ear_pull` + `ear_pulled`.
- Ankle lock rebuilt as ashi garami. Front headlock, rear naked choke and Thai clinch built from contact geometry.
- `doubled_over` now matches anime stomach-punch references: deep hip fold, flat feet, folded over the fist,
  head passing outside the striking arm.
- Body blows land along the torso's own front normal (`_surface`). The eyes follow the pose's `look_at`.

## Open issues, in priority order (the user's last request)

The user wants every pose checked against **real online reference images**: photos, anime screenshots, manga
panels. Find what differs (contact, press, soft body, recoil, gaze, which body part is hit), fix it, and loop
until ours and the reference objectively show the same thing. The previous container could not reach
image hosts. References generated locally with SDXL were too broken to use.

1. Ankle lock (`leg_lock`): verify against real BJJ photos. The legs must visibly control the trapped leg:
   outside foot on the hip, knees pinched, inside foot behind the glute.
2. Gaze: check `look_at` and the eyes on every action.
3. Ear pull close-up: check the pinch contact and the victim's head tilt and wince against references.
4. Press stomach close-up: dent depth, palm orientation and finger spread against references.
5. Headlock, rear choke, knee strike, back-knee kick, spinning back kick, roundhouse, uppercut, hook,
   stumble, knee buckle: compare each with references from 2–3 angles plus a close-up.
6. Zero clipping in every frame. Check close-ups pixel by pixel.
7. Re-render `stories/alley_rin_3d.json` (`python -m animescene comic stories/alley_rin_3d.json -o OUT --mode 3d`),
   then commit and push. Hosting comes at the end (`tools/build_showcase.py`).
