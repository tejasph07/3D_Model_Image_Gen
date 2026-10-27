"""Everything the keyframe engine understands, read from the code itself: contact sites, touch verbs, hand
shapes, joints and their limits, reaction postures, moves (with their keyframes and touches), bystander poses,
camera and page options, expressions, injuries, characters and sets. `markdown()` is the human reference
(docs/CATALOG.md); `compact()` is the vocabulary handed to the story LLM."""
from __future__ import annotations

import inspect

import numpy as np

from . import fight, keyframes as K, moves as M, strikes as X, wounds
from .model import SKELETON
from .shapes import fit
from .control import standard_sites
from .. import manga, sets
from ..render3d import EXPRESSIONS, EXPR_WORDS
from ..vrm import ASSETS

SITE_INFO = {
    "palm": "centre of the palm (grabs, presses, palm strikes)",
    "pinch": "thumb-and-index fingertip (pinches, ear pulls/twists)",
    "knuckles": "front of the closed fist (punches)",
    "wristblade": "blade of the forearm just above the wrist, radius side (clamps, chokes, blocks)",
    "forearm": "middle of the forearm (being grabbed, bars across throat)",
    "hand": "the hand's bone origin (wrist joint)",
    "elbow": "elbow joint (elbow strikes, being held)",
    "shoulder": "shoulder joint",
    "armpit": "armpit, on the chest's side (tucking a leg/ankle under the arm)",
    "achilles": "Achilles tendon, back of the ankle (ankle locks)",
    "ankle": "ankle joint",
    "shin": "front of the shin, mid-length (shin kicks, blocks)",
    "calf": "back of the calf, upper third",
    "backknee": "back of the knee, just below the joint (stamps, buckling kicks)",
    "knee": "front of the kneecap (knee strikes, kneeling support)",
    "innerthigh": "inner side of the thigh, mid-length",
    "outerthigh": "outer side of the thigh (low kicks)",
    "hip": "hip joint (thigh bone origin)",
    "hipbone": "front of the hip bone, pelvis crest (feet on hips in guard)",
    "glute": "buttock (sitting support, a foot behind it)",
    "sole": "middle of the sole (standing support, pushes, stamps)",
    "heel": "heel (support, back kicks, heel strikes)",
    "toes": "ball of the foot / toes (kneeling support, front kicks)",
    "foot": "foot bone origin",
    "ear": "outer ear (pulls, twists, slaps)",
    "jaw": "jaw line below the cheek (hooks)",
    "temple": "temple (side of the head above the eye)",
    "ribs": "lower ribs at the side (body hooks, roundhouse kicks)",
    "sternum": "breastbone, centre of the chest",
    "solarplexus": "solar plexus, just under the breastbone (stomach punches)",
    "belly": "front of the stomach (presses, knees, kicks)",
    "lowback": "lower back (lying support)",
    "upperback": "between the shoulder blades (chest-to-back in chokes)",
    "chin": "point of the chin (uppercuts, palm strikes)",
    "nose": "tip of the nose",
    "eyes": "between the eyes (gaze target)",
    "face": "front of the face",
    "crown": "top of the head",
    "throat": "front of the neck (chokes, forearm bars)",
    "nape": "back of the neck (headlocks, a hand behind the head)",
    "pelvis": "centre of the hips (position hints)",
    "chestc": "centre of the chest volume (camera focus)",
}

VERB_FUNCS = ["grab", "press", "pinch", "twist", "punch", "clamp", "rest", "hook", "tuck", "squeeze"]
JOINT_DOC = {"flex": "+ bends forward (hip: thigh forward; knee/elbow: bends; spine/neck: curls forward)",
             "bend": "+ leans to the character's left", "twist": "+ turns to the character's left",
             "raise": "+ lifts (arm: abduction, -80 = arm down at the side; clavicle: shrug)",
             "protract": "+ shoulder forward", "abduct": "+ leg out to the side", "point": "+ toes down",
             "invert": "+ sole inward", "dev": "+ hand toward the thumb side"}


def sites():
    names = sorted(standard_sites("fem_vroid"))
    out = {}
    for n in names:
        base = n[:-2] if n[-2:] in ("_l", "_r") else n
        out.setdefault(base, {"sides": [], "info": SITE_INFO.get(base, ""), "bone": standard_sites("fem_vroid")[n][0].replace("left", "").replace("right", "")})
        if n[-2:] in ("_l", "_r"):
            out[base]["sides"].append(n[-1])
    return out


def joints():
    rows = []
    for bone, parent, js in SKELETON:
        for jn, ax, (lo, hi), grp in js:
            b = bone.replace("left", "{left|right}")
            rows.append((f"{b}_{jn}", lo, hi, JOINT_DOC.get(jn, "")))
    return rows


def verbs():
    out = []
    for f in VERB_FUNCS:
        fn = getattr(K, f)
        sig = str(inspect.signature(fn))
        out.append((f, sig, (inspect.getdoc(fn) or "").replace("\n", " ")))
    return out


def all_moves():
    out = {}
    for name, fn in X.MOVES.items():
        out[name] = (inspect.getdoc(fn) or "", fn())
    for name in ("leg_lock_guard", "leg_lock_kick"):
        fn = getattr(M, name)
        out[name] = (inspect.getdoc(fn) or "", fn())
    return out


def touch_text(t):
    s = f"{t.verb}: {t.who}.{t.part} -> {t.other}.{t.spot}"
    if t.gap:
        s += f" (gap {t.gap:+.3f} m)"
    return s


def models():
    return sorted(p.stem for p in (ASSETS / "vrm").glob("*.vrm"))


def markdown():
    L = ["# Keyframe engine catalog", "",
         "Generated from the code by `python -m animescene.sim.catalog` - everything listed here is what the engine",
         "actually understands. A **move** is a list of **keyframes**; a keyframe is a list of **touches** (which part",
         "of whom is on which spot of whom, with which **verb**), floor **supports**, a **posture** hint per actor",
         "(joint angles) and a **root** hint (where the hips are, which way the body faces, standing/lying). The",
         "solver turns that into both skeletons, within joint limits, without bodies passing through each other.",
         "Characters A (attacker) and V (receiver) are renamed onto story characters by `place_keys`.", ""]
    L += ["## 1. Contact sites (anchor points)", "",
          "Every character has these named points, fitted to its own mesh. `_l`/`_r` = the character's left/right.", "",
          "| site | sides | bone | where |", "|---|---|---|---|"]
    for n, v in sites().items():
        L.append(f"| `{n}` | {'/'.join(v['sides']) or '-'} | {v['bone']} | {v['info']} |")
    L += ["", "## 2. Touch verbs (actions)", "",
          "Each verb fixes the effector point, which way its surface faces and the hand shape, so hands and feet",
          "are right by construction. `normal` = which way the touched surface faces (`\"surface\"` = away from the",
          "bone, or a world vector); negative gap = pressed into the flesh (the renderer dents the mesh that much).",
          "Any touch can be `mutual` (0..1): the touched body moves to meet it (a head pulled by the ear).", ""]
    for f, sig, doc in verbs():
        L += [f"- **`{f}{sig}`** - {doc or '(see code)'}"]
    L += ["", "Hand shapes the renderer knows: `fist`, `relaxed`, `flat`, `grip`, `claw`, `pinch` (set by the verb, or",
          "per actor/side in `Key.shapes`).", "",
          "## 3. Keyframe fields", "",
          "| field | meaning |", "|---|---|"]
    for name, doc in [("t", "time (s) in the move"), ("touches", "list of verbs above"),
                      ("support", "per actor: sites resting on the floor (e.g. soles+heels standing; lowback+glutes lying; knees+toes kneeling)"),
                      ("posture", "per actor: joint angles in degrees (section 4), a hint the solver bends from"),
                      ("root", "per actor: pos [x, y|None, z] (None height = legs/supports decide), yaw (0 = facing +z), pitch (-90 = on the back), roll, pos_w/hold (how firmly), stance (foot spacing), step {l, r} (a foot forward), approach (walk in if out of reach)"),
                      ("balance", "actors that must keep their weight over their feet"),
                      ("order", "who is solved first (the receiver, then the actor; repeat names for a back-and-forth)"),
                      ("fixed", "actors that keep the pose they come in with"),
                      ("stiff", "per actor: how firmly the posture hint holds (default 0.05)"),
                      ("lock", "per actor: joint-name fragments held exactly at the hint"),
                      ("soft", "how far bodies in contact may press into each other (flesh), m"),
                      ("shapes", "extra hand shapes {actor: {l, r}}")]:
        L.append(f"| `{name}` | {doc} |")
    L += ["", "## 4. Joints and limits (degrees)", "", "| joint | min | max | sign |", "|---|---|---|---|"]
    for j, lo, hi, doc in joints():
        L.append(f"| `{j}` | {lo} | {hi} | {doc} |")
    L += ["", "## 5. Reaction / stance postures (strikes.py)", ""]
    for n in ("STAND", "GUARD", "FIGHT", "DOUBLED", "HEAD_BACK", "FLINCH", "TURNED", "REEL_SIDE"):
        L.append(f"- **{n}**: " + ", ".join(f"{k} {v}" for k, v in getattr(X, n).items()))
    L += ["", "## 6. Moves", "", "Keyframe names are what a story panel shows (`\"key\"`).", ""]
    for name, (doc, keys) in all_moves().items():
        L += [f"### `{name}`", "", doc.replace("\n", " "), ""]
        for k in keys:
            L.append(f"- key **\"{k.name}\"** (t={k.t:.2f}): " + ("; ".join(touch_text(t) for t in k.touches) or "no contact")
                     + f"; supports " + ", ".join(f"{a}: {'+'.join(s)}" for a, s in k.support.items()))
        L.append("")
    L += ["## 7. Bystander poses (everyone not in the beat's move)", ""]
    for n, b in fight.BYSTANDER.items():
        L.append(f"- `{n}`: supports {', '.join(b['support'])}" + (f", pitch {b['pitch']}" if b.get("pitch") else "") + (", balanced" if b.get("balance") else ""))
    L += ["", "## 8. Story file (scene JSON)", "", "```", inspect.getdoc(fight).split("Scene file (JSON):")[1].split("Each beat")[0].strip(), "```", "",
          "- camera `shot`: " + ", ".join(f"`{k}` ({v} m)" for k, v in fight.SHOT.items()) + "; `az` (deg round the pair, 0 = behind A), `elev`, `zoom`, `fov`;"
          " `focus`: `pair`, `a`, `v`, `contact`, `all`. The camera turns itself if anyone blocks the view.",
          "- panel `layout`: " + ", ".join(f"`{k}`" for k in manga.CELL_ASPECT) + "; `caption`, `dialogue` [{who, text}], `sfx` [{text, at: impact}], `expression` {name: tag}",
          "- expression tags: " + "; ".join(", ".join(k) for k, _ in EXPRESSIONS) + "; cast words: " + ", ".join(EXPR_WORDS), "",
          "## 9. Injuries", "",
          "Kinds: " + ", ".join(f"`{k}`" for k in wounds.KINDS) + ". Placed on any site above; stored on the mesh's rest pose so",
          "they stay on the same skin in every later pose and camera. `age` 0 fresh .. 1 healed. Landed blows add them",
          "automatically: a punch or kick to the chin or nose leaves a bleeding cut; a punch, kick or knee to another",
          "head spot (jaw, temple, ear, eyes, face) leaves swelling, a lighter blow there a bruise; anything else bruises",
          "(larger for kicks, knees and stamps). The marks appear from the next beat on;",
          "a story can add them (`injuries`), bandage them (`treat`) and let time pass (`heal`).", "",
          "## 10. Characters (VRM models) and sets", "",
          "Models: " + ", ".join(f"`{m}`" for m in models()), "",
          "Sets (picked by keywords in `setting`/`title`): " + "; ".join(f"`{n}` ({', '.join(kw) or 'default'})" for n, (kw, _) in sets.SETS.items()), ""]
    return "\n".join(L)


def compact():
    """Short vocabulary for an LLM writing story files."""
    mv = {name: [k.name for k in keys] for name, (_, keys) in all_moves().items()}
    return {"moves": {n: {"keys": ks, "about": (all_moves()[n][0].split("\n")[0])} for n, ks in mv.items()},
            "bystander_poses": list(fight.BYSTANDER), "shots": list(fight.SHOT), "focus": ["pair", "a", "v", "contact", "all"],
            "layouts": list(manga.CELL_ASPECT), "injury_kinds": list(wounds.KINDS), "sites": list(sites()),
            "models": models(), "expression_tags": [k[0] for k, _ in EXPRESSIONS], "sets": {n: list(kw) for n, (kw, _) in sets.SETS.items()}}


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/CATALOG.md"
    open(out, "w").write(markdown())
    print(out)
