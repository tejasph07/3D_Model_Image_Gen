# 3D Model Image Gen — text → grounded 3D scene → OpenPose

Turn dead-simple 3-word scene scripts into **physically consistent 3D
scenes** of humanoids and common objects, rendered as **OpenPose BODY_25
(+ hand) skeleton images** and JSON — ready to condition an image model
(ControlNet-style) so the *first* generation is already physically right.

The point is grounding: image models fail at high kicks to the head,
piggybacks, fireman's carries, riding a flying broom. Here those poses are
solved with bone-length-exact IK, surface-accurate contacts, collision
push-out, and a flat-ground constraint — so the language stays close to
text (a novel-to-visualization pipeline an LLM can drive) while the math
guarantees the physics.

```
create,human_a,HUMAN
create,chair_1,CHAIR
infront,human_a,chair_1
touch,human_a.right_palm,human_a.forehead
```

## Quick start

```bash
pip install -r requirements.txt
python -m scene3d examples/quidditch_seeker.scene -o out/
# -> out/quidditch_seeker.png (OpenPose skeleton render)
# -> out/quidditch_seeker.json (2D keypoints + 3D joints + objects + camera)
```

Or pipe a scene in: `echo "create,a,HUMAN" | python -m scene3d - -o out/`.

The full language reference is in [LANGUAGE.md](LANGUAGE.md) — it is written
to be pasted directly into an LLM prompt so a model can compile prose
("she carries him over her shoulders through the door") into scripts.

## What the engine guarantees

- **Exact bone lengths** in every pose (analytic two-bone IK, never scaled).
- **Surface contacts**: `poke,a.left_fingers,a.right_cheek` puts the index
  fingertip *on* the cheek surface — fingers pick their shape from the verb
  (poke = index, touch = four fingers, pinch = two, grab = wrap, punch = fist).
- **Reach realism**: out-of-reach targets make the actor step in, hinge at
  the hips, dip the knees — in that order, like a person.
- **Order-sensitive semantics**: standing + touch-toes = fold at the hips;
  sitting + touch-toes = reach along the ground.
- **Flat ground, no clipping**: nothing sinks below y = 0; bodies push out
  of each other; a hand-held sword tilts up instead of stabbing the floor.
- **Weird-pose coverage**: handstand, bridge, splits, high kicks, chokes,
  shoulder-sitting, piggyback, fireman/cradle carries, broom riding at
  altitude — all covered by tests in `tests/`.

## Output format

- **PNG**: OpenPose BODY_25 skeletons in the standard limb colors + 21-point
  hands, objects as gray wireframes, black background — ControlNet-friendly.
- **JSON**: per person `pose_keypoints_2d` (25×3), `hand_left/right_keypoints_2d`
  (21×3), full 3D joints; per object type/pose/anchors; camera intrinsics.

## Repository layout

```
scene3d/
  parser.py    3-token DSL parser
  skeleton.py  parametric human rig, FK, pose presets, body-part map
  hands.py     verb-driven finger poses, OpenPose hand keypoints
  objects.py   object library (chair, table, door, broom, sword, ...)
  solver.py    command execution, reach strategy, macros, collisions, ground
  ik.py        (in vecmath) analytic two-bone IK w/ pole vectors
  camera.py    auto-framing perspective camera
  render.py    OpenPose-style PNG renderer
  openpose.py  JSON export
  cli.py       command line
examples/      ready-to-render scenes
tests/         physical-plausibility test suite (pytest)
```

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Tests assert the physics, not just the plumbing: bone lengths after every
solve, no joint below the ground, contact gaps in centimeters, straddled
knees on opposite sides of the broom, the kicker's support foot planted.
