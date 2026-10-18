"""Weapon library: geometry for the 3D render and tags for the prompt."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weapon:
    kind: str
    length: float  # grip point -> tip, metres
    handle: float  # grip point -> pommel (behind the hand)
    width: float  # blade half-width for the render
    two_handed: bool
    tags: str  # danbooru-style tags
    reach_class: str = "blade"  # blade | pole | none | ranged


WEAPONS = {
    "katana": Weapon("katana", 0.78, 0.24, 0.016, True, "katana, holding katana"),
    "sword": Weapon("sword", 0.8, 0.14, 0.024, False, "sword, holding sword"),
    "greatsword": Weapon("greatsword", 1.25, 0.3, 0.05, True, "greatsword, huge weapon, holding sword"),
    "dagger": Weapon("dagger", 0.3, 0.09, 0.016, False, "dagger, holding dagger"),
    "twin_daggers": Weapon("twin_daggers", 0.32, 0.09, 0.016, False, "dual wielding, dagger, holding dagger"),
    "twin_swords": Weapon("twin_swords", 0.72, 0.12, 0.02, False, "dual wielding, sword, holding sword"),
    "spear": Weapon("spear", 1.35, 0.6, 0.02, True, "spear, polearm, holding polearm", "pole"),
    "staff": Weapon("staff", 0.9, 0.75, 0.018, True, "staff, holding staff", "pole"),
    "scythe": Weapon("scythe", 1.1, 0.5, 0.02, True, "scythe, holding scythe", "pole"),
    "fists": Weapon("fists", 0.0, 0.0, 0.0, False, "", "none"),
    "boxing_gloves": Weapon("boxing_gloves", 0.0, 0.0, 0.0, False, "boxing gloves", "none"),
    "magic": Weapon("magic", 0.0, 0.0, 0.0, False, "magic", "ranged"),
}


def get_weapon(name: str | None) -> Weapon:
    if not name:
        return WEAPONS["fists"]
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in WEAPONS:
        raise ValueError(f"unknown weapon {name!r}; choose from {sorted(WEAPONS)}")
    return WEAPONS[key]
