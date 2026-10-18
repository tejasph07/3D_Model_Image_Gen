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
