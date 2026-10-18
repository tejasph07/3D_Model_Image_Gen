"""Anime character design: on-the-fly creation, prompt tags, a persistent cast.

A `Character` is the single source of truth for how someone looks: the body
the 3D rig uses (height, build, gender) and the booru tags the image model
uses (hair, eyes, outfit, palette). Characters can be

* written explicitly (every field),
* generated from a seed + archetype (`design(seed=..., archetype="swordsman")`),
* or described loosely ("silver-haired swordswoman in a red coat") and
  completed by the designer, which fills unspecified traits from the seed.

The designer keeps a cast's colour palettes distinct (two fighters never
share a dominant colour) so the generated image can't swap their colours,
and a `Cast` saves designs to JSON so the same character can come back in
any later scene exactly the same.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .rig import Body

HAIR_COLORS = ["black", "silver", "white", "blonde", "red", "pink", "blue", "light blue",
               "purple", "green", "brown", "orange", "aqua"]
EYE_COLORS = ["red", "blue", "green", "golden", "purple", "aqua", "brown", "grey", "pink", "orange"]
HAIR_F = ["long hair", "very long hair", "medium hair", "short hair", "ponytail", "high ponytail",
          "twintails", "bob cut", "hime cut", "braid", "side ponytail", "wavy hair"]
HAIR_M = ["short hair", "spiked hair", "messy hair", "medium hair", "long hair, ponytail",
          "undercut", "swept bangs", "hair between eyes"]
# Palette names map to outfit colour words; the cast keeps these distinct.
PALETTES = {
    "crimson": ("red", "black"), "azure": ("blue", "white"), "emerald": ("green", "gold"),
    "violet": ("purple", "black"), "ivory": ("white", "gold"), "onyx": ("black", "red"),
    "amber": ("orange", "brown"), "teal": ("aqua", "black"), "rose": ("pink", "white"),
    "navy": ("dark blue", "silver"), "golden": ("yellow", "white"), "slate": ("grey", "blue"),
}
ARCHETYPES = {
    "swordsman": dict(weapon="katana", outfits=["{a} haori, {b} hakama, white kimono", "{a} long coat, {b} pants, belt",
                                                 "{a} kimono, {b} obi, arm guards"]),
    "knight": dict(weapon="sword", outfits=["{b} armor, {a} cape, gauntlets, pauldrons, no headwear",
                                             "{a} tabard, {b} breastplate, {a} cape, no headwear"]),
    "berserker": dict(weapon="greatsword", outfits=["{a} torn cloak, {b} leather armor, bandages",
                                                     "{b} heavy armor, {a} scarf, spiked pauldrons"]),
    "assassin": dict(weapon="twin_daggers", outfits=["{b} hooded cloak, {a} scarf, bodysuit, mask",
                                                      "{a} ninja outfit, {b} arm wraps, mouth mask"]),
    "lancer": dict(weapon="spear", outfits=["{a} battle dress, {b} armored boots, gauntlets",
                                            "{b} chinese clothes, {a} sash, wide sleeves"]),
    "mage": dict(weapon="magic", outfits=["{a} robe, {b} hooded cape, wide sleeves",
                                          "{a} witch hat, {b} capelet, long skirt",
                                          "{a} coat, {b} shirt, magic runes, floating book"]),
    "brawler": dict(weapon="fists", outfits=["{a} jacket, {b} tank top, fingerless gloves, hand wraps",
                                             "{a} gi, {b} belt, bandaged hands",
                                             "{b} sleeveless shirt, {a} track pants, fingerless gloves"]),
    "gunslinger": dict(weapon="fists", outfits=["{a} trench coat, {b} vest, holster"]),
    "priestess": dict(weapon="staff", outfits=["{a} miko outfit, {b} hakama, detached sleeves",
                                               "{a} nun robe, {b} veil, cross"]),
    "reaper": dict(weapon="scythe", outfits=["{b} tattered cloak, {a} gothic dress, hood"]),
}


@dataclass
class Character:
    name: str
    gender: str = "female"  # female | male
    height: float = 1.65
    build: str = "athletic"
    hair: str = "long hair"
    hair_color: str = "black"
    eyes: str = "red"
    skin: str = ""  # "" (default), "dark skin", "tan"
    outfit: str = ""
    accessories: str = ""
    expression: str = "serious"
    weapon: str = "fists"
    palette: str = "crimson"
    extra: str = ""  # any additional tags (LoRA trigger words, scars, ...)
    lora: list = field(default_factory=list)  # [{"path": ..., "scale": 0.8, "trigger": "..."}]

    @property
    def body(self) -> Body:
        return Body(height=self.height, build=self.build, gender=self.gender)

    @property
    def count_tag(self) -> str:
        return "1girl" if self.gender == "female" else "1boy"

    def appearance_tags(self) -> str:
        parts = [self.hair_color + " hair" if "hair" not in self.hair_color else self.hair_color,
                 self.hair, f"{self.eyes} eyes", self.skin, self.outfit, self.accessories, self.extra]
        if self.build in ("muscular", "heavy"):
            parts.append("muscular" if self.gender == "male" else "toned")
        if self.gender == "male" and self.build == "muscular":
            parts.append("broad shoulders")
        for l in self.lora:
            if l.get("trigger"):
                parts.insert(0, l["trigger"])
        return ", ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# ------------------------------------------------------------------- loose descriptions
_KEYWORDS = {
    "gender": {r"\b(girl|woman|female|she|her|swordswoman|priestess|witch|heroine|lady|queen|princess)\b": "female",
               r"\b(boy|man|male|he|him|swordsman|guy|hero|king|prince|monk)\b": "male"},
    "build": {r"\b(muscular|burly|hulking|brawny)\b": "muscular", r"\b(petite|small|tiny|little)\b": "petite",
              r"\b(slim|slender|lithe|lean)\b": "slim", r"\b(heavy|giant|huge|massive)\b": "heavy"},
}


def _find_color(text: str, noun: str, choices: list) -> str | None:
    for c in sorted(choices, key=len, reverse=True):
        if re.search(rf"\b{c}[- ]?(haired|{noun})\b", text):
            return c
    return None


def parse_description(text: str) -> dict:
    """Pull the traits a short English description pins down; the rest is designed."""
    t = text.lower()
    out: dict = {}
    for key, pats in _KEYWORDS.items():
        for pat, val in pats.items():
            if re.search(pat, t):
                out[key] = val
                break
    hc = _find_color(t, "hair", HAIR_COLORS)
    if hc:
        out["hair_color"] = hc
    ec = _find_color(t, "eye", EYE_COLORS) or _find_color(t, "eyes", EYE_COLORS)
    if ec:
        out["eyes"] = ec
    from .weapons import WEAPONS
    for w in sorted(WEAPONS, key=len, reverse=True):
        if re.search(rf"\b{w.replace('_', ' ')}s?\b", t) or (w == "twin_daggers" and "daggers" in t):
            out["weapon"] = w
            break
    for arch in ARCHETYPES:
        if arch in t or (arch == "swordsman" and "swords" in t) or (arch == "mage" and ("wizard" in t or "sorcer" in t)):
            out["archetype"] = arch
            break
    garment = r"(coat|cloak|robe|armor|jacket|dress|kimono|outfit|scarf|cape|haori|gi|uniform)"
    for pal, (a, _b) in PALETTES.items():
        if re.search(rf"\b({a}|{pal})\b {garment}", t):
            out["palette"] = pal
            break
    if "dark skin" in t or "dark-skinned" in t:
        out["skin"] = "dark skin"
    elif "tan" in t.split():
        out["skin"] = "tan"
    return out


def design(name: str, seed: int | None = None, archetype: str | None = None, description: str = "",
           avoid_palettes: tuple = (), **overrides) -> Character:
    """Create a coherent new character. Anything not given is chosen from the seed."""
    rng = random.Random(seed if seed is not None else hash(name) & 0xFFFFFFFF)
    traits = parse_description(description) if description else {}
    traits.update({k: v for k, v in overrides.items() if v is not None})
    arch = archetype or traits.pop("archetype", None) or rng.choice(sorted(ARCHETYPES))
    traits.pop("archetype", None)
    if arch not in ARCHETYPES:
        raise ValueError(f"unknown archetype {arch!r}; choose from {sorted(ARCHETYPES)}")
    a = ARCHETYPES[arch]
    gender = traits.get("gender") or rng.choice(["female", "male"])
    build = traits.get("build") or (rng.choice(["athletic", "muscular", "slim"]) if gender == "male"
                                    else rng.choice(["slim", "athletic", "petite"]))
    if arch == "berserker" and "build" not in traits:
        build = "muscular"
    free = [p for p in sorted(PALETTES) if p not in avoid_palettes] or sorted(PALETTES)
    palette = traits.get("palette") or rng.choice(free)
    ca, cb = PALETTES[palette]
    outfit = rng.choice(a["outfits"]).format(a=ca, b=cb)
    height = {"female": 1.62, "male": 1.76}[gender] * {"petite": 0.92, "heavy": 1.06, "muscular": 1.04}.get(build, 1.0)
    height *= rng.uniform(0.97, 1.03)
    c = Character(
        name=name, gender=gender, height=round(height, 3), build=build,
        hair=traits.get("hair") or rng.choice(HAIR_F if gender == "female" else HAIR_M),
        hair_color=traits.get("hair_color") or rng.choice(HAIR_COLORS),
        eyes=traits.get("eyes") or rng.choice(EYE_COLORS),
        skin=traits.get("skin", ""),
        outfit=traits.get("outfit") or outfit,
        accessories=traits.get("accessories", ""),
        expression=traits.get("expression", "serious"),
        weapon=traits.get("weapon") or a["weapon"],
        palette=palette,
        extra=traits.get("extra", ""),
    )
    return c


class Cast:
    """A persistent set of characters (JSON on disk) so designs stay consistent across scenes."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.chars: dict[str, Character] = {}
        if self.path and self.path.exists():
            for d in json.loads(self.path.read_text()):
                c = Character.from_dict(d)
                self.chars[c.name] = c

    def get(self, name: str) -> Character | None:
        return self.chars.get(name)

    def add(self, c: Character) -> Character:
        self.chars[c.name] = c
        return c

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([c.to_dict() for c in self.chars.values()], indent=2))

    def resolve(self, spec: dict, used_palettes: list) -> Character:
        """Turn a scene's character entry into a Character (reuse, explicit, or designed)."""
        name = spec["name"]
        if spec.get("reuse", True) and name in self.chars and not spec.get("redesign"):
            base = self.chars[name].to_dict()
            base.update({k: v for k, v in spec.items() if k in Character.__dataclass_fields__})
            return Character.from_dict(base)
        fields = {k: v for k, v in spec.items() if k in Character.__dataclass_fields__ and k != "name"}
        c = design(name, seed=spec.get("seed"), archetype=spec.get("archetype"),
                   description=spec.get("description", ""), avoid_palettes=tuple(used_palettes), **fields)
        return self.add(c)
