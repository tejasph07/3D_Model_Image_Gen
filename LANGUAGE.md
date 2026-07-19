# scene3d language specification

This document is self-contained and designed to be pasted into an LLM prompt.
It describes a tiny scene-description language that compiles to a physically
consistent 3D scene (OpenPose skeletons + object wireframes).

## Format

One command per line, exactly three comma-separated tokens:

    verb,arg1,arg2

- `#` starts a comment. Blank lines are ignored. Verbs are case-insensitive.
- `entity.part` selects a body part or object anchor (`hero.right_palm`,
  `door_1.knob`).
- Some verbs take a number (`scale`, `turn`, `hover`) or a packed pair
  `a:b` (`at,hero,1.5:2.0`, `between,girl,hero:villain`).
- **Order matters.** Commands run top to bottom against the scene as it
  exists so far. `pose,a,stand` then `touch,a.right_fingers,a.right_toes`
  folds the body at the hips; with `pose,a,sit` first, the same touch keeps
  the person seated. Put placement before poses, poses before contacts.
- The ground is flat at y = 0. Nothing can go below it. Collisions are
  resolved automatically; contact commands land on body *surfaces*, never
  inside them. If a target is out of reach, the actor bends and/or steps
  closer automatically.

## Creation

| command | meaning |
|---|---|
| `create,name,TYPE` | add an entity. Human types: `HUMAN`, `PERSON`, `MAN`, `WOMAN`, `CHILD` (child is auto-scaled 0.62). Object types below. |
| `scale,name,0.8` | uniform scale (people and objects) |

Object types: `CHAIR STOOL BENCH TABLE DOOR BALL STICK SWORD ROPE BED SOFA
BOX CUP BOOK WALL ROCK LANTERN` with aliases `BROOM/STAFF/POLE/BAT/WAND/SPEAR
-> STICK`, `COUCH -> SOFA`, `DESK -> TABLE`, `SEAT -> CHAIR`, `CRATE -> BOX`,
`STONE/BOULDER -> ROCK`, `FOOTBALL/QUAFFLE -> BALL`, `MUG -> CUP`,
`TORCH/LAMP -> LANTERN`.

## Placement (moves arg1 relative to arg2)

| command | meaning |
|---|---|
| `at,a,x:z` or `at,a,x:y:z` | absolute position in meters (y lifts off the ground) |
| `infront,a,b` | a in front of b, facing b |
| `behind,a,b` | a behind b, facing b's back |
| `left_of,a,b` / `right_of,a,b` | at b's left/right side, facing b |
| `beside,a,b` (`next_to`) | shoulder to shoulder, facing the same way |
| `near,a,b` | close to b, facing b |
| `between,a,b:c` | midway between b and c |
| `on_top,a,b` | a on b's top surface (person on a human = shoulder-sit) |
| `above,a,b` | a floats 0.5 m above b |
| `under,a,b` (`below`) | a beneath b (a person auto-lies-down to fit) |
| `hover,a,2.0` | lift a so it floats at the given height (flying broom!) |
| `face,a,b` | turn a to face b |
| `face_away,a,b` | turn a to face directly away from b |
| `face_same,a,b` | a copies b's facing |
| `face_each,a,b` | mutual: a and b face each other |
| `turn,a,45` | rotate a by degrees (positive = clockwise from above) |

Back-to-back: `at` both, then `face_away` both. Passing strangers: `beside`
plus `turn,a,180`.

## Body poses

`pose,person,NAME` with NAME one of:

`stand sit sit_chair cross_legged squat crouch kneel kneel_one lie_back
lie_front lie_side_left lie_side_right sprawl fetal all_fours plank handstand
bridge lunge split run walk jump cower straddle tree_pose warrior tpose`

(`sprawl` = collapsed on the back, limbs flung — defeat. `kneel_one` = one
knee down. `bridge` = yoga wheel. `tree_pose`/`warrior` = yoga.)

Poses on objects:

| command | meaning |
|---|---|
| `sit_on,a,chair_1` | sit on the object's seat (works for chair/stool/bench/sofa/bed/box/rock; add `.anchor` to pick a spot, e.g. `sofa_1.seat_left`) |
| `stand_on,a,table_1` | stand on top of the object |
| `lie_on,a,bed_1` | lie on top of the object |
| `kneel_on,a,bench_1` | kneel on the object |
| `lean_on,a,table_1` | stand at the object, lean in, palm pressed on it |
| `ride,a,broom_1` / `straddle,a,b` | legs astride the object's axis; on a hovering object the rider flies with it; hands auto-grip the shaft. `ride,a,person` = piggyback. |

## Gaze and aiming

| command | meaning |
|---|---|
| `look_at,a,b` | turn a's head toward b (or `b.part`, or `ground`/`sky`) |
| `look,a,down` | direction: `up down left right forward` |
| `point,a.right_arm,up` | aim a limb: `left_arm right_arm arms left_leg right_leg`; direction (`up down forward back left right forward_up forward_down up_left up_right`) or any entity/part. Hand takes a pointing shape. |

## Contact actions (arg1 = actor's part, arg2 = target)

The verb sets the finger pose; the named effector part sets where contact
happens. Targets: `person.part`, `object`, or `object.anchor`.

| verb | hand/foot shape | typical use |
|---|---|---|
| `touch` | four fingers extended | `touch,a.right_palm,a.forehead` |
| `poke` | index finger only | `poke,a.left_fingers,b.right_cheek` |
| `pinch` | thumb + index | `pinch,a.right_fingers,b.left_ear` |
| `grab` / `grip` | full wrap | `grab,a.right_hand,b.left_wrist` |
| `hold` | wrap; an unplaced portable object (ball, cup, sword, stick, book, lantern) jumps into the hand | `hold,a.left_hand,cup_1` |
| `press` / `palm` | flat palm | `press,a.right_palm,door_1.center` |
| `slap` | open palm strike | `slap,a.right_palm,b.left_cheek` |
| `punch` | fist, torso twists into it | `punch,a.right_fist,b.jaw` |
| `push` | flat palms, braced stance, lean in | `push,a.hands,b.chest` |
| `pull` | grip, braced stance, lean away | `pull,a.hands,rope_1` |
| `squeeze` | hard grip | `squeeze,a.right_hand,b.right_forearm` |
| `choke` | grip on the neck | `choke,a.hands,b` |
| `kick` | foot strike, torso counter-leans (high kicks work: `kick,a.right_foot,b.head`) | `kick,a.right_foot,ball_1` |
| `stomp` | foot from above | `stomp,a.right_foot,b.chest` |
| `knee` | knee strike | `knee,a.right_knee,b.belly` |
| `elbow` | elbow strike | `elbow,a.right_elbow,b.face` |
| `headbutt` | forehead to target | `headbutt,a,b` |
| `high_five` | palm to palm | `high_five,a.right_palm,b.right_palm` |

Plural effectors expand to both sides: `hands palms feet knees arms
shoulders`. When both sides are plural (`touch,a.palms,b.shoulders`) they
pair mirrored, the natural way for people facing each other.

### Body part names

Head: `head face forehead nose chin jaw left_cheek right_cheek left_ear
right_ear left_eye right_eye top_of_head`
Torso: `neck throat chest back upper_back lower_back belly waist hips butt
groin`
Arms (`left_`/`right_`): `shoulder upper_arm elbow forearm wrist hand palm
fist fingers fingertip thumb`
Legs (`left_`/`right_`): `hip thigh inner_thigh knee shin calf ankle foot
toes heel sole`

### Object anchors (main ones)

chair: `seat backrest back_top edge` · table: `top edge far_edge left_edge
right_edge under corner` · door: `knob handle center top edge` · stick/broom:
`middle front back tip end grip` · sword: `handle middle tip` · bed: `top
pillow foot_end edge` · sofa: `seat seat_left seat_right backrest
armrest_left armrest_right` · ball: `center top` · rope: `middle end_a end_b`

## Two-person arrangements (macros)

| command | meaning |
|---|---|
| `piggyback,carrier,rider` | rider on the carrier's back, legs wrapped, carrier grips thighs |
| `shoulder_sit,carrier,rider` | rider sits on the shoulders, carrier holds shins |
| `fireman_carry,carrier,body` | body draped across the shoulders, arm locked over the knees |
| `cradle_carry,carrier,body` | bridal carry: arms under back and knees |
| `hug,a,b` | face to face, arms around each other's backs |
| `handshake,a,b` | right hands clasped between them |
| `kiss,a,b` | faces close, heads toward each other |

Macros position both people; you can still adjust afterwards
(`look_at`, `turn`, extra touches).

## Camera

`camera,VIEW,target` — VIEW: `front back left right three_quarter
three_quarter_left top low auto`; target: entity name or `scene`.
The camera auto-frames its target.

## Worked example

    # An impossible-for-diffusion scene: mid-air broom duel aftermath.
    create,seeker,HUMAN
    create,broom,BROOM
    hover,broom,2.0
    ride,seeker,broom
    create,snitch,BALL
    scale,snitch,0.4
    hold,seeker.left_hand,snitch
    create,rival,HUMAN
    pose,rival,sprawl
    at,rival,1.0:3.0
    look_at,seeker,rival
    camera,three_quarter,scene

## Guidance for scene authors (LLMs)

1. Create everything first, then place, then pose, then contacts, then gaze,
   then camera.
2. Prefer macros for standard two-person configurations; compose primitives
   for everything else (any unusual pose is expressible as pose + point +
   touch/grab constraints).
3. Don't specify what you don't care about — defaults are sensible.
4. One scene = one static moment. Verbs like `punch` freeze the instant of
   contact.
5. Distances are meters; people are ~1.75 m tall.
