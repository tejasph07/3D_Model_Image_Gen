# Keyframe engine catalog

Generated from the code by `python -m animescene.sim.catalog` - everything listed here is what the engine
actually understands. A **move** is a list of **keyframes**; a keyframe is a list of **touches** (which part
of whom is on which spot of whom, with which **verb**), floor **supports**, a **posture** hint per actor
(joint angles) and a **root** hint (where the hips are, which way the body faces, standing/lying). The
solver turns that into both skeletons, within joint limits, without bodies passing through each other.
Characters A (attacker) and V (receiver) are renamed onto story characters by `place_keys`.

## 1. Contact sites (anchor points)

Every character has these named points, fitted to its own mesh. `_l`/`_r` = the character's left/right.

| site | sides | bone | where |
|---|---|---|---|
| `achilles` | l/r | LowerLeg | Achilles tendon, back of the ankle (ankle locks) |
| `ankle` | l/r | LowerLeg | ankle joint |
| `armpit` | l/r | upperChest | armpit, on the chest's side (tucking a leg/ankle under the arm) |
| `backknee` | l/r | LowerLeg | back of the knee, just below the joint (stamps, buckling kicks) |
| `belly` | - | spine | front of the stomach (presses, knees, kicks) |
| `calf` | l/r | LowerLeg | back of the calf, upper third |
| `chestc` | - | upperChest | centre of the chest volume (camera focus) |
| `chin` | - | head | point of the chin (uppercuts, palm strikes) |
| `crown` | - | head | top of the head |
| `ear` | l/r | head | outer ear (pulls, twists, slaps) |
| `elbow` | l/r | LowerArm | elbow joint (elbow strikes, being held) |
| `eyes` | - | head | between the eyes (gaze target) |
| `face` | - | head | front of the face |
| `foot` | l/r | Foot | foot bone origin |
| `forearm` | l/r | LowerArm | middle of the forearm (being grabbed, bars across throat) |
| `glute` | l/r | hips | buttock (sitting support, a foot behind it) |
| `hand` | l/r | Hand | the hand's bone origin (wrist joint) |
| `heel` | l/r | Foot | heel (support, back kicks, heel strikes) |
| `hip` | l/r | UpperLeg | hip joint (thigh bone origin) |
| `hipbone` | l/r | hips | front of the hip bone, pelvis crest (feet on hips in guard) |
| `innerthigh` | l/r | UpperLeg | inner side of the thigh, mid-length |
| `jaw` | l/r | head | jaw line below the cheek (hooks) |
| `knee` | l/r | UpperLeg | front of the kneecap (knee strikes, kneeling support) |
| `knuckles` | l/r | Hand | front of the closed fist (punches) |
| `lowback` | - | spine | lower back (lying support) |
| `nape` | - | neck | back of the neck (headlocks, a hand behind the head) |
| `nose` | - | head | tip of the nose |
| `outerthigh` | l/r | UpperLeg | outer side of the thigh (low kicks) |
| `palm` | l/r | Hand | centre of the palm (grabs, presses, palm strikes) |
| `pelvis` | - | hips | centre of the hips (position hints) |
| `pinch` | l/r | Hand | thumb-and-index fingertip (pinches, ear pulls/twists) |
| `ribs` | l/r | upperChest | lower ribs at the side (body hooks, roundhouse kicks) |
| `shin` | l/r | LowerLeg | front of the shin, mid-length (shin kicks, blocks) |
| `shoulder` | l/r | UpperArm | shoulder joint |
| `solarplexus` | - | upperChest | solar plexus, just under the breastbone (stomach punches) |
| `sole` | l/r | Foot | middle of the sole (standing support, pushes, stamps) |
| `sternum` | - | upperChest | breastbone, centre of the chest |
| `temple` | l/r | head | temple (side of the head above the eye) |
| `throat` | - | neck | front of the neck (chokes, forearm bars) |
| `toes` | l/r | Foot | ball of the foot / toes (kneeling support, front kicks) |
| `upperback` | - | upperChest | between the shoulder blades (chest-to-back in chokes) |
| `wristblade` | l/r | LowerArm | blade of the forearm just above the wrist, radius side (clamps, chokes, blocks) |

## 2. Touch verbs (actions)

Each verb fixes the effector point, which way its surface faces and the hand shape, so hands and feet
are right by construction. `normal` = which way the touched surface faces (`"surface"` = away from the
bone, or a world vector); negative gap = pressed into the flesh (the renderer dents the mesh that much).
Any touch can be `mutual` (0..1): the touched body moves to meet it (a head pulled by the ear).

- **`grab(who, side, other, spot, normal='surface', weight=1.0)`** - The hand closes round `spot`: palm on it, fingers wrapped (renderer grip shape).
- **`press(who, side, other, spot, depth=0.02, normal='surface', weight=1.0)`** - Open palm pushed into a soft surface by `depth` (the renderer dents the mesh by that much).
- **`pinch(who, side, other, spot, normal='surface', weight=1.0)`** - Thumb and index tip closed on `spot` (an ear, a cheek); the palm turned loosely toward it.
- **`twist(who, side, other, spot, angle=60.0, normal='surface', weight=1.0)`** - Pinch `spot` and wring it: the hand turns about the surface normal by `angle` degrees from fingers-up (an ear twisted, a cheek wrung). The fingers point along the rolled direction.
- **`punch(who, side, other, spot, depth=0.02, normal='surface', weight=1.0)`** - Knuckles driven into `spot` along its normal (fist).
- **`clamp(who, side, other, spot, normal, weight=1.0)`** - The blade of the forearm by the wrist hard against `spot` from the `normal` side (e.g. under an Achilles tendon: normal = down).
- **`rest(who, part, other, spot, normal='surface', weight=1.0, face=True)`** - A sole (or other flat part) set on `spot`.
- **`hook(who, part, other, spot, normal='surface', weight=1.0)`** - `part` (an instep, a heel, the crook of a knee) hooked round `spot` from its `normal` side, e.g. a foot behind a glute.
- **`tuck(who, part, other, spot, gap=0.04, weight=1.0)`** - `spot` held in against `part` (e.g. an ankle tucked under the armpit): `part` comes to it.
- **`squeeze(who, part, other, spot, normal='surface', depth=0.01, weight=0.25)`** - `part` pressed lightly into `spot` as a secondary pressure (knees pinching a leg, a chest on a back); weaker than a main touch, so it gives way first.

Hand shapes the renderer knows: `fist`, `relaxed`, `flat`, `grip`, `claw`, `pinch` (set by the verb, or
per actor/side in `Key.shapes`).

## 3. Keyframe fields

| field | meaning |
|---|---|
| `t` | time (s) in the move |
| `touches` | list of verbs above |
| `support` | per actor: sites resting on the floor (e.g. soles+heels standing; lowback+glutes lying; knees+toes kneeling) |
| `posture` | per actor: joint angles in degrees (section 4), a hint the solver bends from |
| `root` | per actor: pos [x, y|None, z] (None height = legs/supports decide), yaw (0 = facing +z), pitch (-90 = on the back), roll, pos_w/hold (how firmly), stance (foot spacing), step {l, r} (a foot forward), approach (walk in if out of reach) |
| `balance` | actors that must keep their weight over their feet |
| `order` | who is solved first (the receiver, then the actor; repeat names for a back-and-forth) |
| `fixed` | actors that keep the pose they come in with |
| `stiff` | per actor: how firmly the posture hint holds (default 0.05) |
| `lock` | per actor: joint-name fragments held exactly at the hint |
| `soft` | how far bodies in contact may press into each other (flesh), m |
| `shapes` | extra hand shapes {actor: {l, r}} |

## 4. Joints and limits (degrees)

| joint | min | max | sign |
|---|---|---|---|
| `spine_flex` | -25 | 45 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `spine_bend` | -20 | 20 | + leans to the character's left |
| `spine_twist` | -25 | 25 | + turns to the character's left |
| `chest_flex` | -15 | 25 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `chest_bend` | -15 | 15 | + leans to the character's left |
| `chest_twist` | -20 | 20 | + turns to the character's left |
| `upperChest_flex` | -10 | 15 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `upperChest_bend` | -10 | 10 | + leans to the character's left |
| `upperChest_twist` | -15 | 15 | + turns to the character's left |
| `neck_flex` | -35 | 45 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `neck_bend` | -30 | 30 | + leans to the character's left |
| `neck_twist` | -50 | 50 | + turns to the character's left |
| `head_flex` | -30 | 35 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `head_bend` | -20 | 20 | + leans to the character's left |
| `head_twist` | -30 | 30 | + turns to the character's left |
| `{left|right}Shoulder_raise` | -8 | 28 | + lifts (arm: abduction, -80 = arm down at the side; clavicle: shrug) |
| `{left|right}Shoulder_protract` | -15 | 20 | + shoulder forward |
| `{left|right}UpperArm_flex` | -45 | 135 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `{left|right}UpperArm_raise` | -88 | 95 | + lifts (arm: abduction, -80 = arm down at the side; clavicle: shrug) |
| `{left|right}UpperArm_twist` | -90 | 90 | + turns to the character's left |
| `{left|right}LowerArm_flex` | 0 | 150 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `{left|right}LowerArm_twist` | -90 | 70 | + turns to the character's left |
| `{left|right}Hand_flex` | -70 | 80 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `{left|right}Hand_dev` | -25 | 30 | + hand toward the thumb side |
| `{left|right}UpperLeg_flex` | -20 | 125 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `{left|right}UpperLeg_abduct` | -25 | 55 | + leg out to the side |
| `{left|right}UpperLeg_twist` | -40 | 40 | + turns to the character's left |
| `{left|right}LowerLeg_flex` | 0 | 150 | + bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward) |
| `{left|right}Foot_point` | -25 | 50 | + toes down |
| `{left|right}Foot_invert` | -20 | 20 | + sole inward |

## 5. Reaction / stance postures (strikes.py)

- **STAND**: leftUpperLeg_flex 18, rightUpperLeg_flex 18, leftLowerLeg_flex 28, rightLowerLeg_flex 28, leftFoot_point -10, rightFoot_point -10, spine_flex 6
- **GUARD**: leftUpperArm_raise -55, rightUpperArm_raise -55, leftUpperArm_flex 45, rightUpperArm_flex 45, leftLowerArm_flex 110, rightLowerArm_flex 110
- **FIGHT**: leftUpperLeg_flex 18, rightUpperLeg_flex 18, leftLowerLeg_flex 28, rightLowerLeg_flex 28, leftFoot_point -10, rightFoot_point -10, spine_flex 6, leftUpperArm_raise -55, rightUpperArm_raise -55, leftUpperArm_flex 45, rightUpperArm_flex 45, leftLowerArm_flex 110, rightLowerArm_flex 110
- **DOUBLED**: leftUpperLeg_flex 35, rightUpperLeg_flex 35, leftLowerLeg_flex 45, rightLowerLeg_flex 45, leftFoot_point -10, rightFoot_point -10, spine_flex 40, chest_flex 20, upperChest_flex 10, neck_flex -15, leftUpperArm_raise -70, rightUpperArm_raise -70, leftUpperArm_flex 30, rightUpperArm_flex 30, leftLowerArm_flex 100, rightLowerArm_flex 100
- **HEAD_BACK**: leftUpperLeg_flex 5, rightUpperLeg_flex 25, leftLowerLeg_flex 28, rightLowerLeg_flex 35, leftFoot_point -10, rightFoot_point -10, spine_flex -15, chest_flex -10, neck_flex -35, head_flex -15, leftUpperArm_raise -30, rightUpperArm_raise -30, leftLowerArm_flex 40, rightLowerArm_flex 40
- **FLINCH**: leftUpperLeg_flex 18, rightUpperLeg_flex 18, leftLowerLeg_flex 28, rightLowerLeg_flex 28, leftFoot_point -10, rightFoot_point -10, spine_flex 20, chest_flex 10, neck_flex 10, leftUpperArm_raise -60, rightUpperArm_raise -60, leftUpperArm_flex 25, rightUpperArm_flex 25, leftLowerArm_flex 60, rightLowerArm_flex 60
- **TURNED**: leftUpperLeg_flex 18, rightUpperLeg_flex 18, leftLowerLeg_flex 28, rightLowerLeg_flex 28, leftFoot_point -10, rightFoot_point -10, spine_flex 6, spine_twist 20, chest_twist 15, neck_twist 40, spine_bend -15, neck_bend -20, leftUpperArm_raise -45, rightUpperArm_raise -60, leftLowerArm_flex 50, rightLowerArm_flex 30
- **REEL_SIDE**: leftUpperLeg_flex 18, rightUpperLeg_flex 30, leftLowerLeg_flex 28, rightLowerLeg_flex 45, leftFoot_point -10, rightFoot_point -10, spine_flex 6, spine_bend 20, chest_bend 12, neck_bend 15, leftUpperArm_raise -20, rightUpperArm_raise -70, leftLowerArm_flex 30, rightLowerArm_flex 60

## 6. Moves

Keyframe names are what a story panel shows (`"key"`).

### `ready`



- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `stomach_punch`

Right fist driven into his solar plexus; he folds over it, the fist still buried.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"punch"** (t=0.25): punch: A.knuckles_r -> V.solarplexus (gap -0.035 m); supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"doubled over"** (t=0.55): punch: A.knuckles_r -> V.solarplexus (gap -0.035 m); supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `palm_strike`

Heel of the right palm up under his chin; his head snaps back.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"palm strike"** (t=0.22): press: A.palm_r -> V.chin (gap -0.020 m); supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `press_stomach`

Both open palms pressed flat into his stomach; he flinches back over them.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"press"** (t=0.50): press: A.palm_l -> V.belly (gap -0.030 m); press: A.palm_r -> V.belly (gap -0.030 m); supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `ear_pull`

She pinches his left ear between thumb and fingers and pulls up and toward her; his head follows it, he rises on his toes, wincing, hands at her wrist.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"ear pull"** (t=0.50): pinch: A.pinch_r -> V.ear_l; grab: V.palm_l -> A.wristblade_r; grab: V.palm_r -> A.forearm_r; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `ear_twist`

She pinches his left ear and wrings it, hand turned over; he is dragged down to her shoulder height, bent at the waist and knees, head bent away under the twist, both hands at her wrist, wincing. She stays upright.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"ear twist"** (t=0.50): twist: A.pinch_r -> V.ear_l; grab: V.palm_l -> A.wristblade_r; grab: V.palm_r -> A.forearm_r; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `front_headlock`

He is bent over in front of her; her right forearm under his throat, his head trapped under her right armpit, left hand clasping her right wrist.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"front headlock"** (t=0.60): clamp: A.wristblade_r -> V.throat; tuck: A.armpit_r -> V.nape (gap +0.020 m); grab: A.palm_l -> A.wristblade_r; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `rear_choke`

From behind: right forearm across his throat, right hand on her left forearm, left palm behind his head, her chest on his back. He claws at the arm.

- key **"rear choke"** (t=0.00): clamp: A.wristblade_r -> V.throat; grab: A.palm_r -> A.forearm_l; press: A.palm_l -> V.nape; squeeze: A.sternum -> V.upperback (gap -0.010 m); grab: V.palm_l -> A.forearm_r; grab: V.palm_r -> A.wristblade_r; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `knee_strike`

Both hands on the back of his head pull it down into her rising right knee.

- key **"clinch"** (t=0.00): grab: A.palm_l -> V.temple_r; grab: A.palm_r -> V.temple_l; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"knee"** (t=0.30): grab: A.palm_l -> V.temple_r; grab: A.palm_r -> V.temple_l; knee: A.knee_r -> V.solarplexus (gap -0.030 m); supports A: sole_l+heel_l, V: sole_l+sole_r+heel_l+heel_r

### `hook`

Left hook to his right jaw; his head and shoulders turn away with it.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"hook"** (t=0.22): punch: A.knuckles_l -> V.jaw_r (gap -0.020 m); supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `uppercut`

Right uppercut under his chin from a crouch; he goes up on his toes, head thrown back.

- key **"crouch"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"uppercut"** (t=0.20): punch: A.knuckles_r -> V.chin (gap -0.020 m); supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `roundhouse`

Right roundhouse: her shin into his left ribs, standing on the left foot, turned over; he reels sideways.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"roundhouse"** (t=0.30): kick: A.shin_r -> V.ribs_l (gap -0.020 m); supports A: sole_l+heel_l, V: sole_l+sole_r+heel_l+heel_r

### `spinning_back_kick`

Turned away, her right heel thrust straight back into his stomach; she looks over her shoulder.

- key **"ready"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"spin"** (t=0.25): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"back kick"** (t=0.50): kick: A.heel_r -> V.belly (gap -0.030 m); supports A: sole_l+heel_l, V: sole_l+sole_r+heel_l+heel_r

### `back_knee_kick`

Behind and to his side: her right sole stamps the back of his left knee; it buckles.

- key **"behind"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"stamp"** (t=0.30): stamp: A.sole_r -> V.backknee_l (gap -0.020 m); supports A: sole_l+heel_l, V: sole_r+heel_r

### `knee_buckle`

He drops to his knees.

- key **"buckle"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: knee_l+knee_r+toes_l+toes_r

### `stumble`

He staggers back: one foot behind, arms out, trunk behind his hips.

- key **"stumble"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r

### `leg_lock_guard`

He is on his back, left foot on her right hip; she stands between his feet.

- key **"guard"** (t=0.00): rest: V.sole_l -> A.hipbone_r; supports A: sole_l+sole_r+heel_l+heel_r, V: lowback+glute_l+glute_r+sole_r
- key **"grab"** (t=0.60): grab: A.palm_l -> V.shin_l; grab: A.palm_r -> V.ankle_l; rest: V.sole_l -> A.hipbone_r; supports A: sole_l+sole_r+heel_l+heel_r, V: lowback+glute_l+glute_r+sole_r
- key **"wrap"** (t=1.40): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> V.shin_l; supports A: sole_l+sole_r+heel_l+heel_r, V: lowback+glute_l+glute_r+sole_r
- key **"sit"** (t=2.40): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> V.shin_l; supports A: glute_l+glute_r+sole_l+sole_r, V: lowback+glute_l+glute_r+heel_r
- key **"ashi garami"** (t=3.40): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> A.wristblade_r; rest: A.sole_r -> V.hipbone_l; hook: A.sole_l -> V.glute_l; squeeze: A.knee_r -> V.outerthigh_l (gap -0.010 m); squeeze: A.knee_l -> V.innerthigh_l (gap -0.010 m); supports V: lowback+glute_l+glute_r+heel_r, A: glute_l+glute_r
- key **"break"** (t=4.00): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> A.wristblade_r; rest: A.sole_r -> V.hipbone_l; hook: A.sole_l -> V.glute_l; squeeze: A.knee_r -> V.outerthigh_l (gap -0.010 m); squeeze: A.knee_l -> V.innerthigh_l (gap -0.010 m); supports V: lowback+glute_l+glute_r+heel_r, A: glute_l+glute_r

### `leg_lock_kick`

Both standing; he throws a front kick at her right ribs, she catches it (forearm under the calf by the Achilles, ankle tucked under her armpit, left palm on his shin), drives forward so he falls on his back, and finishes as from the guard.

- key **"stance"** (t=0.00): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_l+sole_r+heel_l+heel_r
- key **"kick"** (t=0.35): no contact; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_r+heel_r
- key **"catch"** (t=0.70): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> V.shin_l; supports A: sole_l+sole_r+heel_l+heel_r, V: sole_r+heel_r
- key **"drive"** (t=1.10): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> V.shin_l; supports A: sole_l+sole_r+heel_l+heel_r
- key **"down"** (t=1.60): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> V.shin_l; supports A: sole_l+sole_r+heel_l+heel_r, V: lowback+glute_l+glute_r+heel_r
- key **"sit"** (t=2.60): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> V.shin_l; supports A: glute_l+glute_r+sole_l+sole_r, V: lowback+glute_l+glute_r+heel_r
- key **"ashi garami"** (t=3.60): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> A.wristblade_r; rest: A.sole_r -> V.hipbone_l; hook: A.sole_l -> V.glute_l; squeeze: A.knee_r -> V.outerthigh_l (gap -0.010 m); squeeze: A.knee_l -> V.innerthigh_l (gap -0.010 m); supports V: lowback+glute_l+glute_r+heel_r, A: glute_l+glute_r
- key **"break"** (t=4.20): clamp: A.wristblade_r -> V.achilles_l; tuck: A.armpit_r -> V.ankle_l (gap -0.060 m); grab: A.palm_l -> A.wristblade_r; rest: A.sole_r -> V.hipbone_l; hook: A.sole_l -> V.glute_l; squeeze: A.knee_r -> V.outerthigh_l (gap -0.010 m); squeeze: A.knee_l -> V.innerthigh_l (gap -0.010 m); supports V: lowback+glute_l+glute_r+heel_r, A: glute_l+glute_r

## 7. Bystander poses (everyone not in the beat's move)

- `ready`: supports sole_l, sole_r, heel_l, heel_r, balanced
- `down`: supports lowback, glute_l, glute_r, heel_l, heel_r, pitch -90
- `kneel`: supports knee_l, knee_r, toes_l, toes_r
- `doubled`: supports sole_l, sole_r, heel_l, heel_r, balanced

## 8. Story file (scene JSON)

```
title, setting, lighting, cast: [{name, model, scale?, expression?}], injuries?: [{who, spot, kind, radius?, age?}]
  beats: [{move, a, v, at: [x, z], yaw, injuries?, treat?: [{who, spot}], heal?: 0..1, others: {name: {pose: ready|down|kneel|doubled, at: [x, z], yaw}},
           panels: [{key: <keyframe name>, camera: {shot: wide|medium|close|extreme, az, elev, focus: pair|a|v|contact},
                     layout: wide|half|full|tall, caption, dialogue: [{who, text}], sfx: [{text, at: impact}],
                     expression: {name: tag}}]}]
```

- camera `shot`: `wide` (5.0 m), `medium` (3.0 m), `close` (1.7 m), `extreme` (0.95 m); `az` (deg round the pair, 0 = behind A), `elev`, `zoom`, `fov`; `focus`: `pair`, `a`, `v`, `contact`, `all`. The camera turns itself if anyone blocks the view.
- panel `layout`: `wide`, `half`, `full`, `tall`; `caption`, `dialogue` [{who, text}], `sfx` [{text, at: impact}], `expression` {name: tag}
- expression tags: wince, one eye closed; flinch, embarrassed; pain, being choked, screaming, tears; surprised, falling, off balance, stumbling, wide-eyed; doubled over; punching, kicking, slashing, palm strike, knee strike, headlock, chokehold, leg lock, thrusting, shouting; fighting stance, blocking, guarding; smirk, grin, confident; cast words: angry, serious, smirk, grin, determined, neutral, happy, sad

## 9. Injuries

Kinds: `bruise`, `cut`, `bleeding_cut`, `scar`, `bandage`, `swelling`. Placed on any site above; stored on the mesh's rest pose so
they stay on the same skin in every later pose and camera. `age` 0 fresh .. 1 healed. Landed blows add them
automatically: a punch or kick to the chin or nose leaves a bleeding cut; a punch, kick or knee to another
head spot (jaw, temple, ear, eyes, face) leaves swelling, a lighter blow there a bruise; anything else bruises
(larger for kicks, knees and stamps). The marks appear from the next beat on;
a story can add them (`injuries`), bandage them (`treat`) and let time pass (`heal`).

## 10. Characters (VRM models) and sets

Models: `AvatarSample_A`, `AvatarSample_B`, `AvatarSample_C`, `Avatar_Orion`, `Darkness_Shibu`, `HairSample_Male`, `Sakurada_Fumiriya`, `Seed-san`, `Sendagaya_Shibu`, `Sendagaya_Shino`, `Victoria_Rubin`, `Vita`, `Vivi`, `fem_vroid`, `masc_vroid`

Sets (picked by keywords in `setting`/`title`): `alley` (alley, back alley, street, neon); `ring` (ring, boxing, arena); `plain` (plain)
