"""MuJoCo model of one or more characters, built from their own VRM skeletons and meshes.

Frame: the models' rest frame - metres, y up, gravity -y; at rest every body is world-aligned and
the character faces +z, so a body's world rotation *is* the rotation of its VRM bone from rest and
the renderer can apply it 1:1. x is the character's LEFT.

Joints are anatomical hinges with human ranges of motion (defined for the left side and mirrored);
each hinge has a position servo ("muscle") whose stiffness says how hard the character holds or
drives that joint. Contacts are slightly soft (flesh gives a few millimetres).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shapes import fit

# (bone, parent, [(joint, axis for the LEFT side / centre, (lo, hi) degrees, servo group)])
# Signs: + is flexion/forward, raising, toes down, etc. (see each comment).
SKELETON = [
    ("hips", None, []),  # free joint
    ("spine", "hips", [("flex", (1, 0, 0), (-25, 45), "trunk"), ("bend", (0, 0, 1), (-20, 20), "trunk"),
                       ("twist", (0, 1, 0), (-25, 25), "trunk")]),
    ("chest", "spine", [("flex", (1, 0, 0), (-15, 25), "trunk"), ("bend", (0, 0, 1), (-15, 15), "trunk"),
                        ("twist", (0, 1, 0), (-20, 20), "trunk")]),
    ("upperChest", "chest", [("flex", (1, 0, 0), (-10, 15), "trunk"), ("bend", (0, 0, 1), (-10, 10), "trunk"),
                             ("twist", (0, 1, 0), (-15, 15), "trunk")]),
    ("neck", "upperChest", [("flex", (1, 0, 0), (-35, 45), "neck"), ("bend", (0, 0, 1), (-30, 30), "neck"),
                            ("twist", (0, 1, 0), (-50, 50), "neck")]),
    ("head", "neck", [("flex", (1, 0, 0), (-30, 35), "neck"), ("bend", (0, 0, 1), (-20, 20), "neck"),
                      ("twist", (0, 1, 0), (-30, 30), "neck")]),
    # clavicle: + raises the shoulder (rotation about +z lifts a +x bone), + brings it forward
    ("leftShoulder", "upperChest", [("raise", (0, 0, 1), (-8, 28), "shoulder"), ("protract", (0, -1, 0), (-15, 20), "shoulder")]),
    # shoulder: forward swing, raise (abduction; -80 = arm down at the side), humeral rotation
    ("leftUpperArm", "leftShoulder", [("flex", (0, -1, 0), (-45, 135), "shoulder"), ("raise", (0, 0, 1), (-88, 95), "shoulder"),
                                      ("twist", (1, 0, 0), (-90, 90), "shoulder")]),
    ("leftLowerArm", "leftUpperArm", [("flex", (0, -1, 0), (0, 150), "elbow"), ("twist", (1, 0, 0), (-90, 70), "forearm")]),
    ("leftHand", "leftLowerArm", [("flex", (0, 0, -1), (-70, 80), "wrist"), ("dev", (0, 1, 0), (-25, 30), "wrist")]),
    ("leftUpperLeg", "hips", [("flex", (-1, 0, 0), (-20, 125), "hip"), ("abduct", (0, 0, 1), (-25, 55), "hip"),
                              ("twist", (0, 1, 0), (-40, 40), "hip")]),
    ("leftLowerLeg", "leftUpperLeg", [("flex", (1, 0, 0), (0, 150), "knee")]),
    ("leftFoot", "leftLowerLeg", [("point", (1, 0, 0), (-25, 50), "ankle"), ("invert", (0, 0, 1), (-20, 20), "ankle")]),
]
MIRROR = {"left": "right"}
# servo stiffness (N m / rad per kg of the body it moves, roughly) and damping per group
GROUPS = {"trunk": (900, 10), "neck": (120, 1.5), "shoulder": (260, 3), "elbow": (160, 2), "forearm": (40, 0.5),
          "wrist": (25, 0.3), "hip": (900, 8), "knee": (700, 6), "ankle": (220, 2)}
# the strongest torque a muscle group can produce (N m, for a 70 kg adult; scaled by body mass): a servo
# saturates there, so nobody lifts what a human couldn't
MAX_TORQUE = {"trunk": 220, "neck": 35, "shoulder": 80, "elbow": 65, "forearm": 10, "wrist": 15, "hip": 220,
              "knee": 200, "ankle": 130}


# body segment mass fractions (de Leva 1996, adult average); a missing upperChest folds into the chest
MASS_FRACTION = {"head": 0.055, "neck": 0.014, "upperChest": 0.15, "chest": 0.10, "spine": 0.07, "hips": 0.11,
                 "Shoulder": 0.003, "UpperArm": 0.027, "LowerArm": 0.016, "Hand": 0.006,
                 "UpperLeg": 0.142, "LowerLeg": 0.043, "Foot": 0.014}


def body_mass(height: float, build: float = 1.0) -> float:
    """Total mass of a character from its height (BMI 21, x build)."""
    return 21.0 * height * height * build


def _mirror(axis):
    a = np.asarray(axis, float)
    return (a[0], -a[1], -a[2])  # rotation axes reflect across the x = 0 plane as pseudovectors


def skeleton():
    out = []
    for bone, parent, joints in SKELETON:
        out.append((bone, parent, joints))
        if bone.startswith("left"):
            mb = "right" + bone[4:]
            mp = ("right" + parent[4:]) if parent and parent.startswith("left") else parent
            out.append((mb, mp, [(j, _mirror(ax), rng, g) for j, ax, rng, g in joints]))
    return out


@dataclass
class ActorSpec:
    name: str
    model: str
    pos: tuple = (0.0, 0.0)  # x, z on the floor (the hips go above it)
    yaw: float = 0.0  # degrees, 0 = facing +z
    scale: float = 1.0
    sites: dict = field(default_factory=dict)  # extra named sites: name -> (bone, local pos)


def _f(v):
    return " ".join(f"{x:.5f}" for x in v)


def _quat_yaw(deg):
    a = np.deg2rad(deg) / 2
    return (np.cos(a), 0.0, np.sin(a), 0.0)  # w x y z, about +y


def actor_xml(spec: ActorSpec) -> tuple[str, list, list]:
    """(worldbody xml, actuator xml lines, body names) for one character."""
    shapes = fit(spec.model)
    k = spec.scale
    B = {b: np.asarray(p, float) * k for b, p in shapes["_bones"].items()}
    P = spec.name + ":"
    have = set(B)
    tree = [(b, p, j) for b, p, j in skeleton() if b in have]
    # a missing upperChest: its children hang on the chest
    parent_of = {}
    for b, p, j in tree:
        while p is not None and p not in have:
            p = {"upperChest": "chest"}.get(p)
        parent_of[b] = p
    children = {}
    for b, _, _ in tree:
        children.setdefault(parent_of[b], []).append(b)
    joints_of = {b: j for b, _, j in tree}
    acts = []

    def geom(bone):
        s = shapes.get(bone)
        if not s and bone.endswith("Shoulder"):  # clavicle: a short capsule toward the shoulder joint
            arm = B[bone.replace("Shoulder", "UpperArm")] - B[bone]
            s = {"type": "capsule", "from": [0, 0, 0], "to": (arm / k * 0.7).tolist(), "r": 0.03}
        if not s:
            return f'<inertial pos="0 0 0" mass="0.2" diaginertia="0.0005 0.0005 0.0005"/>'
        cls = f'class="{spec.name}"'
        if s["type"] == "capsule":
            return (f'<geom name="{P}{bone}" type="capsule" fromto="{_f(np.asarray(s["from"]) * k)} {_f(np.asarray(s["to"]) * k)}" '
                    f'size="{s["r"] * k:.5f}" {cls}/>')
        if s["type"] == "ellipsoid":
            return f'<geom name="{P}{bone}" type="ellipsoid" pos="{_f(np.asarray(s["pos"]) * k)}" size="{_f(np.asarray(s["size"]) * k)}" {cls}/>'
        return f'<geom name="{P}{bone}" type="box" pos="{_f(np.asarray(s["pos"]) * k)}" size="{_f(np.asarray(s["size"]) * k)}" {cls}/>'

    def body(bone, indent):
        par = parent_of[bone]
        rel = B[bone] - (B[par] if par else np.zeros(3))
        pad = "  " * indent
        lines = []
        if par is None:
            lines.append(f'{pad}<body name="{P}{bone}" pos="{_f(rel)}">')
            lines.append(f'{pad}  <freejoint name="{P}root"/>')
        else:
            lines.append(f'{pad}<body name="{P}{bone}" pos="{_f(rel)}">')
            for jn, ax, (lo, hi), grp in joints_of[bone]:
                kp, kd = GROUPS[grp]
                lines.append(f'{pad}  <joint name="{P}{bone}.{jn}" axis="{_f(ax)}" range="{lo} {hi}" damping="{kd:.2f}" '
                             f'armature="0.01" limited="true"/>')
                acts.append(f'<position name="{P}{bone}.{jn}" joint="{P}{bone}.{jn}" kp="{kp:.1f}" '
                            f'forcerange="-{MAX_TORQUE[grp]} {MAX_TORQUE[grp]}"/>')
        g = geom(bone)
        if g:
            lines.append(pad + "  " + g)
        for site, (sb, sp) in spec.sites.items():
            if sb == bone:
                lines.append(f'{pad}  <site name="{P}{site}" pos="{_f(np.asarray(sp) * k)}" size="0.008"/>')
        for c in children.get(bone, []):
            lines += body(c, indent + 1)
        lines.append(f"{pad}</body>")
        return lines

    root = [b for b, p, _ in tree if parent_of[b] is None][0]
    xml = body(root, 2)
    w, x, y, z = _quat_yaw(spec.yaw)
    # place the whole character: wrap in a frame at its floor position and yaw
    xml.insert(0, f'    <frame pos="{spec.pos[0]:.4f} 0 {spec.pos[1]:.4f}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}">')
    xml.append("    </frame>")
    return "\n".join(xml), acts, [b for b, _, _ in tree]


def build(actors: list[ActorSpec], extra_world: str = "", extra_equality: str = "", timestep=0.002,
          exclude: str = "") -> str:
    bodies, acts, defaults = [], [], []
    for a in actors:
        xml, ac, _ = actor_xml(a)
        bodies.append(xml)
        acts += ac
        # flesh: slightly soft contact (a few mm), skin/cloth friction
        defaults.append(f'<default class="{a.name}"><geom condim="4" friction="1.0 0.02 0.002" solref="0.008 1" '
                        f'solimp="0.92 0.99 0.004 0.5 2" density="1000" rgba="0.8 0.6 0.5 1"/></default>')
    return f"""<mujoco model="scene">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{timestep}" gravity="0 -9.81 0" iterations="60" ls_iterations="20" cone="elliptic" impratio="4"/>
  <default>
    {''.join(defaults)}
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" zaxis="0 1 0" friction="0.9 0.02 0.002" solref="0.004 1"/>
{chr(10).join(bodies)}
{extra_world}
  </worldbody>
  <contact>
{exclude}
  </contact>
  <equality>
{extra_equality}
  </equality>
  <actuator>
    {chr(10).join('    ' + x for x in acts)}
  </actuator>
</mujoco>"""


def compile_model(actors: list[ActorSpec], extra_world: str = "", extra_equality: str = "", timestep=0.002):
    """MjModel of the scene. Body pairs of the same character that already touch in the rest pose
    (hips and thighs, chest and upper arms, ...) are excluded from colliding with each other; every
    other pair - arms against the own torso, legs against each other - collides."""
    import mujoco
    xml = build(actors, extra_world, extra_equality, timestep)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        n1, n2 = m.body(b1).name, m.body(b2).name
        if n1.split(":")[0] == n2.split(":")[0] and c.dist < 0.004:
            pairs.add(tuple(sorted((n1, n2))))
    # torso neighbours two apart also overlap through the soft middle segment
    for a in actors:
        P = a.name + ":"
        for x, y in (("hips", "chest"), ("spine", "upperChest"), ("chest", "neck"), ("upperChest", "head"),
                     ("leftShoulder", "neck"), ("rightShoulder", "neck"), ("leftUpperArm", "upperChest"),
                     ("rightUpperArm", "upperChest"), ("leftUpperArm", "chest"), ("rightUpperArm", "chest"),
                     ("leftUpperLeg", "spine"), ("rightUpperLeg", "spine"),
                     # a person's own hands and forearms pass each other freely (crossed guards, clasped hands:
                     # a gable grip is a constraint, not a collision)
                     ("leftHand", "rightHand"), ("leftHand", "rightLowerArm"), ("rightHand", "leftLowerArm"),
                     ("leftLowerArm", "rightLowerArm")):
            pairs.add(tuple(sorted((P + x, P + y))))
    names = {m.body(i).name for i in range(m.nbody)}
    exclude = "\n".join(f'    <exclude body1="{a}" body2="{b}"/>' for a, b in sorted(pairs) if a in names and b in names)
    xml = build(actors, extra_world, extra_equality, timestep, exclude)
    m = mujoco.MjModel.from_xml_string(xml)
    set_masses(m, actors)
    return m, xml


def set_masses(m, actors: list[ActorSpec]):
    """Give every segment its anatomical share of the body's mass (geometry alone over-weights legs in
    trousers and under-weights the chest); inertias scale with the mass so the shapes' distribution stays."""
    for a in actors:
        P = a.name + ":"
        names = {m.body(i).name[len(P):]: i for i in range(m.nbody) if m.body(i).name.startswith(P)}
        shapes = fit(a.model)
        B = shapes["_bones"]
        height = (max(p[1] for p in B.values()) + 0.12) * a.scale
        total = body_mass(height)
        fr = {}
        for n in names:
            key = next((k for k in MASS_FRACTION if n == k or n.endswith(k)), None)
            if key:
                fr[n] = MASS_FRACTION[key]
        if "upperChest" not in names:
            fr["chest"] = fr.get("chest", 0) + MASS_FRACTION["upperChest"]
        for n, f in fr.items():
            i = names[n]
            new = total * f
            if m.body_mass[i] > 0:
                m.body_inertia[i] *= new / m.body_mass[i]
            m.body_mass[i] = new
