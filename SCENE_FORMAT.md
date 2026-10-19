# Scene format

A scene is one JSON object. It describes **who** is in the shot, **what each
of them is doing to whom**, **where**, and **how the camera sees it**. The 3D
stage solves the exact poses and contacts; you never write coordinates
unless you want to.

```json
{
  "title": "Clash at the burning shrine",
  "setting": "burning shinto shrine courtyard at night, torii gate, stone lanterns, flying embers",
  "lighting": "fire light, rim light, dramatic lighting",
  "camera": {"shot": "low_angle", "azimuth": 28},
  "seed": 7,
  "characters": [
    {"name": "Ren", "description": "black-haired swordsman in a navy coat", "archetype": "knight",
     "action": "block", "target": "Akane", "at": [-1.0, 0.0]},
    {"name": "Akane", "description": "red-haired swordswoman", "archetype": "swordsman",
     "action": "slash", "style": "overhead", "target": "Ren"}
  ]
}
```

## Top level

| key | meaning |
|---|---|
| `setting` | background as booru tags / short phrases (place, time, weather, props) |
| `lighting` | optional lighting tags (default `cinematic lighting, dramatic lighting`) |
| `style` | optional global style tags |
| `camera` | see below |
| `seed` | integer seed (default 0) |
| `negative` | extra negative tags |
| `ground` | `false` for scenes with no visible floor (sky battles) |
| `characters` | list of 1–6 characters, each with its action |

## Camera

| key | values |
|---|---|
| `shot` | `wide`, `full`, `medium` (crops into the action), `low_angle`, `worms_eye`, `high_angle`, `birds_eye` |
| `azimuth` | degrees around the scene. 0 = straight on; 20–35 = three-quarter view (best for fights) |
| `elevation` | override the shot's pitch (negative = looking up) |
| `roll` | dutch angle in degrees |
| `fov` | vertical field of view |
| `margin` | framing margin as a fraction of the frame (negative crops) |
| `aspect` | `landscape` (default), `portrait`, `square`, `wide`, `tall` |
| `focus` | frame only these: `"Name"`, `"Name.upper"`, `"Name.head"`, `"Name.<joint>"`, `"impact"` (use with `closeup` / `extreme_closeup`) |

## Characters

Appearance (all optional; anything missing is designed from `seed`/`archetype`
so characters can be created on the fly, and the designer keeps every
character's colour palette distinct):

| key | meaning |
|---|---|
| `name` | unique; reusing a name from the cast file brings back the same design |
| `description` | loose English ("pink-haired petite mage girl in a violet robe"); parsed for gender, build, hair/eye colour, weapon, archetype, palette |
| `archetype` | `swordsman`, `knight`, `berserker`, `assassin`, `lancer`, `mage`, `brawler`, `gunslinger`, `priestess`, `reaper` |
| `gender` | `female` / `male` |
| `build` | `petite`, `slim`, `athletic`, `muscular`, `heavy` |
| `height` | metres |
| `hair`, `hair_color`, `eyes`, `skin`, `outfit`, `accessories`, `expression`, `extra` | booru tags |
| `weapon` | `katana`, `sword`, `greatsword`, `dagger`, `twin_daggers`, `twin_swords`, `spear`, `staff`, `scythe`, `fists`, `magic` |
| `palette` | `crimson`, `azure`, `emerald`, `violet`, `ivory`, `onyx`, `amber`, `teal`, `rose`, `navy`, `golden`, `slate` |
| `lora` | `[{"path": "file.safetensors", "scale": 0.8, "trigger": "trigger words"}]` |
| `seed` | design seed |

Action (on the character object):

| key | meaning |
|---|---|
| `action` | one of the actions below |
| `target` | name of the character the action is aimed at / reacting to |
| `aim` | `head`, `face`, `neck`, `chest`, `stomach`, `legs` |
| `style`, `phase` | action-specific (below) |
| `at` | `[x, z]` metres — pin a character; others are placed relative to their targets |
| `bearing` | degrees — which side of the target to stand on (default: side by side across the screen) |
| `distance` | metres from the target, overriding the action's natural range |
| `facing` | degrees or another character's name (default: the target) |
| `altitude` | metres of extra height for airborne actions |
| `inflict` | wounds this character receives in this scene: `[{"kind": "cut"}]` lands at the 3D contact point of the blow it takes (or `"from": "Name"`), or `[{"kind": "bruise", "spot": "l_cheek"}]` at a named spot. Kinds: `bruise`, `cut`, `bleeding_cut`, `scar`, `bandage`, `swelling`. Stored in the cast and drawn in every later scene at the same body location with the same pattern |

### Actions

| action | what the solver does |
|---|---|
| `stand` | neutral stance, weapon lowered |
| `stance` | fighting stance / weapon at guard (magic: hands glowing) |
| `punch` | fist placed on the target's `aim` point; `style`: `straight` / `uppercut` |
| `kick` | high kick, foot on the target's `aim`; support foot planted; `style`: `roundhouse` / `front` |
| `flying_kick` | airborne, leading leg straight into the target |
| `slash` | weapon cut; `style`: `diagonal`/`horizontal`/`overhead`/`rising`; `phase`: `impact`/`windup`/`follow`. Against a `block`ing target the blades lock in front of the defender |
| `thrust` | lunging stab / spear thrust at `aim` |
| `block` | defender's blade crosses the attacker's blade at the contact point (unarmed: crossed-arm guard) |
| `leap_attack` | airborne, weapon raised overhead (`phase`: `windup` / `impact`) |
| `knocked_back` | flung backwards through the air; automatically moved to where the attacker's blow landed |
| `staggered` | grounded hit reaction, also aligned to the blow |
| `dodge` | leaning/jumping back and aside |
| `cast` | spell: arm outstretched at the target, palm open; `style`: `one_hand` / `two_hands` |
| `palm_strike` | open palm pressed into the target (default `aim: stomach`); `style: energy` adds an energy-blast region at the contact |
| `knee_strike` | clinch: hands pull the target's head down, knee driven into the stomach |
| `spinning_back_kick` | back turned, heel driven backward and up into the neck/head (`aim`) |
| `back_knee_kick` | from behind, stomp into the hollow of the target's knee (`leg`: `l`/`r`); pair with `knee_buckle` |
| `headlock` / `headlocked` | front headlock: the victim bent over, head trapped under the arm, forearm under the throat |
| `rear_choke` / `choked` | chokehold from behind; the victim claws at the forearm |
| `leg_lock` / `leg_locked` | ground ankle lock: attacker sitting at the feet, ankle trapped, leaning back; victim flat on the back |
| `doubled_over` | gut-shot reaction: folded around the blow, onto the toes, mouth open |
| `clutch_stomach` | hunched, both hands on the stomach |
| `stumble` | stumbling back, legs crossing, arms windmilling |
| `knee_buckle` | knee kicked out: dropping toward one knee, off balance |

`punch` styles: `straight`, `jab` (lead hand), `hook`, `uppercut`, `body`. `stance` with `style: boxing` is a peek-a-boo
guard. Reactions (`knocked_back`, `staggered`, `doubled_over`, `stumble`, `knee_buckle`) are moved so the struck part is
where the blow actually landed; holds (`headlock`, `rear_choke`, `leg_lock`, `knee_strike`, `palm_strike`) attach to the
victim's reacting body. Gut shots dent the victim's surface in the depth control.
| `charge` | sprinting at the target |
| `grab` | collar grab with the right hand, left fist cocked |
| `point` | pointing (or the weapon) at the target |
| `kneel` | on one knee (leaning on the weapon if armed) |
| `fallen` | lying defeated on the ground |

## Tips for complex scenes

* Split a melee into **pairs or trios of interactions** (attacker → defender).
  Pin one member of each group with `at` (2–3 m apart); everyone else is
  placed by their action.
* Reactions point at the attacker: `{"action": "knocked_back", "target": "Mira"}`.
* A three-quarter camera (`azimuth` 20–35) reads better than a pure profile.
* Low angles (`low_angle`, `worms_eye`) make fights epic; `high_angle` for chaos overviews.

## Stories (multi-panel)

`python -m animescene comic story.json -o out/` renders a story file: a shared `cast`, story-wide `setting` /
`lighting` / `seed`, and `panels`, each a scene (characters referenced by name) plus page furniture:

```json
{"id": "p3", "layout": "half", "caption": "Too slow.",
 "camera": {"shot": "closeup", "azimuth": -30, "focus": ["Rin.upper", "Kaito.upper"]},
 "characters": [{"name": "Rin", "action": "stumble", "target": "Kaito", "at": [0, 0],
                 "inflict": [{"kind": "bleeding_cut"}]},
                {"name": "Kaito", "action": "punch", "aim": "face", "target": "Rin"}],
 "dialogue": [{"who": "Kaito", "text": "Too slow!"}],
 "sfx": [{"text": "CRACK", "at": "impact"}]}
```

`layout`: `wide` (full row), `half` (two per row), `full`, `tall`. Speech bubbles point at the speaker's head and SFX
sit beside the impact, both placed from the 3D stage. Each character gets a turnaround sheet the first time it appears;
every panel then attends to it for identity. See `stories/` for complete examples.
