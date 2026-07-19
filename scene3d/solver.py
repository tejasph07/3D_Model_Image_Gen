"""Scene solver: executes DSL commands into a physically-consistent 3D scene.

Commands are applied strictly top to bottom; each contact command runs a
"reach strategy" against the pose as it exists at that moment (so
stand -> touch toes bends at the hips, while lie -> touch toes does not).

After all commands run, `finalize()` iterates: re-resolve constraints,
push interpenetrating bodies apart, snap everything onto the flat ground.
"""
from __future__ import annotations

import math

import numpy as np

from . import hands as hands_mod
from .objects import Prop
from .parser import Command, Ref, parse
from .skeleton import (ARM_LEN, DIMS, Human, JOINT_RADII, LEG_LEN, PLURAL_PARTS,
                       POSE_PRESETS, STAND_PELVIS_Y)
from .vecmath import (UP, V, clamp, horizontal, norm, normalize, rot_y,
                      seg_seg_closest, two_bone_ik, yaw_of)

CONTACT_VERBS = {
    "touch", "poke", "pinch", "grab", "grip", "hold", "press", "palm", "slap",
    "punch", "push", "pull", "squeeze", "choke", "kick", "stomp", "knee",
    "elbow", "headbutt", "high_five",
}
ARM_EFFECTORS = {"hand", "palm", "fist", "fingers", "fingertip", "wrist",
                 "thumb", "forearm"}
LEG_EFFECTORS = {"foot", "toes", "heel", "ankle", "shin", "sole"}
LOOK_DIRS = {"up": (0, -40), "down": (0, 48), "left": (-60, 0), "right": (60, 0),
             "forward": (0, 0), "ahead": (0, 0)}
CLEARANCE = 0.35          # default gap between placed entities
MIN_ROOT_SEP = 0.32       # how close two roots may get when stepping in to reach


class SceneError(ValueError):
    pass


class Constraint:
    """A persistent contact/aim goal on one human limb."""

    def __init__(self, human, limb, verb, target, approach=None, seq=0,
                 eff_part=None):
        self.human = human
        self.limb = limb          # arm_L / arm_R / leg_L / leg_R / knee_L /
                                  # knee_R / elbow_L / elbow_R / head
        self.verb = verb
        self.eff_part = eff_part  # named hand/foot part making contact
        self.target = target      # dict, see Scene._resolve_target
        self.approach = approach  # optional world-space approach override
        self.seq = seq
        self.error = 0.0


class Scene:
    def __init__(self):
        self.entities: dict[str, object] = {}
        self.order: list[str] = []
        self.constraints: list[Constraint] = []
        self.attachments: list[dict] = []   # rider/carried root followers
        self.exempt_pairs: set[frozenset] = set()
        self.camera_cmd = ("auto", "scene")
        self.warnings: list[str] = []
        self._next_x = 0.0
        self._seq = 0

    # ------------------------------------------------------------- utilities
    def ent(self, name: str, line=None):
        if name not in self.entities:
            raise SceneError(f"line {line}: unknown entity '{name}' "
                             f"(known: {sorted(self.entities)})")
        return self.entities[name]

    def humans(self):
        return [self.entities[n] for n in self.order
                if isinstance(self.entities[n], Human)]

    def props(self):
        return [self.entities[n] for n in self.order
                if isinstance(self.entities[n], Prop)]

    def exempt(self, a, b):
        self.exempt_pairs.add(frozenset((a.name, b.name)))

    def warn(self, msg):
        self.warnings.append(msg)

    # ------------------------------------------------------------- evaluation
    def evaluate(self):
        """Recompute the whole scene from parameters + constraints."""
        for att in self.attachments:
            self._apply_attachment(att)
        for h in self.humans():
            h.fk()
        for h in self.humans():
            self._resolve_look(h)
        for _pass in range(2):
            # Carried/riding humans re-root on their (possibly re-posed) base
            # first, then constraints solve on top of the refreshed FK.
            for att in self.attachments:
                self._apply_attachment(att, refresh=True)
            for c in sorted(self.constraints, key=lambda c: c.seq):
                self._apply_constraint(c)
            self._place_attached_props()
        for h in self.humans():
            h._solve_feet()
            h.solve_hands()
        self._place_attached_props()

    def _apply_attachment(self, att, refresh=False):
        kind = att["kind"]
        dep: Human = att["dep"]
        base = att["base"]
        if kind == "offset":  # dep's root rides at a yaw-local offset from base
            off = att["offset"]
            F = rot_y(base.yaw)
            world = F @ V(off[0], 0, off[2])
            dep.x = base.x + world[0]
            dep.z = base.z + world[2]
            base_y = base.params["pelvis_y"] if isinstance(base, Human) else base.y
            dep.params["pelvis_y"] = base_y + off[1]
            dep.yaw = base.yaw + att.get("dyaw", 0.0)
            if refresh:
                dep.fk()

    def _place_attached_props(self):
        for p in self.props():
            if p.attached_to is None:
                continue
            h, side, anchor = p.attached_to
            if not h.joints:
                h.fk()
            pre = side.lower() + "_"
            wrist = h.joints[pre + "wrist"]
            hf = h.hand_frame(side)
            palm = wrist + hf[:, 2] * 0.09 * h.scale - hf[:, 1] * 0.02
            if p.spec.ride_axis or p.kind == "SWORD":
                # Long objects align their axis with the fingers.
                p.yaw = yaw_of(hf[:, 2])
                p.pitch = math.degrees(math.asin(clamp(-float(hf[1, 2]), -1, 1)))
            else:
                p.yaw = h.yaw
                p.pitch = 0.0
            grip_local = p.spec.anchors[p.attached_to[2]] * p.scale
            origin = palm - p.frame() @ grip_local
            p.x, p.y, p.z = float(origin[0]), float(origin[1]), float(origin[2])
            # A held blade/staff must not dig into the floor: tilt its axis
            # toward horizontal (grip stays pinned in the palm) until clear.
            for _ in range(12):
                if p.lowest_point() >= -0.005:
                    break
                p.pitch -= math.copysign(9.0, p.pitch) if abs(p.pitch) > 1 else 9.0
                origin = palm - p.frame() @ grip_local
                p.x, p.y, p.z = (float(origin[0]), float(origin[1]),
                                 float(origin[2]))

    def _resolve_look(self, h: Human):
        if h.look_target is None:
            return
        kind, val = h.look_target
        if kind == "dir":
            dyaw, dpitch = LOOK_DIRS[val]
            h.params["head_yaw"] = dyaw
            h.params["head_pitch"] = dpitch
            h.fk()
            return
        if kind == "part":
            ent_name, part = val
            ent = self.entities.get(ent_name)
            if ent is None:
                return
            target = (ent.part_point(part)[0] if part else self._entity_focus(ent))
        else:
            target = np.array(val)
        # One refinement iteration: head center moves slightly with rotation.
        for _ in range(2):
            hc = h.frames["head_center"]
            CF = h.frames["chest"]
            d_local = CF.T @ (target - hc)
            yaw = math.degrees(math.atan2(d_local[0], d_local[2]))
            flat = math.hypot(d_local[0], d_local[2])
            pitch = -math.degrees(math.atan2(d_local[1], max(flat, 1e-6)))
            h.params["head_yaw"] = clamp(yaw, -80, 80)
            h.params["head_pitch"] = clamp(pitch, -55, 60)
            h.fk()

    def _entity_focus(self, ent):
        if isinstance(ent, Human):
            if not ent.joints:
                ent.fk()
            return ent.frames["head_center"]
        return ent.anchor("center") if "center" in ent.spec.anchors else \
            ent.to_world(V(0, ent.spec.top_y * 0.6, 0))

    # ------------------------------------------------- constraint application
    def _resolve_target(self, t):
        """Return (center, radius) in world space for a constraint target."""
        kind = t["kind"]
        if kind == "part":
            ent = self.entities[t["entity"]]
            return ent.part_point(t["part"])
        if kind == "anchor":
            p: Prop = self.entities[t["entity"]]
            return p.anchor(t["anchor"]), t.get("radius", 0.025 * p.scale)
        if kind == "point":
            return np.array(t["p"]), t.get("radius", 0.0)
        if kind == "rel":  # yaw-local offset from an entity's root
            ent = self.entities[t["entity"]]
            base_y = ent.params["pelvis_y"] if isinstance(ent, Human) else ent.y
            w = rot_y(ent.yaw) @ V(*t["offset"])
            return V(ent.x + w[0], base_y + w[1], ent.z + w[2]), t.get("radius", 0.0)
        if kind == "midgrip":  # handshake midpoint between two right shoulders
            a: Human = self.entities[t["a"]]
            b: Human = self.entities[t["b"]]
            m = (a.joints["r_shoulder"] + b.joints["r_shoulder"]) / 2
            m[1] = min(a.joints["r_shoulder"][1], b.joints["r_shoulder"][1]) - 0.38
            return m, 0.0
        if kind == "under_part":  # support from below (carry arms)
            ent = self.entities[t["entity"]]
            c, r = ent.part_point(t["part"])
            return c - UP * (r + 0.02), 0.0
        raise SceneError(f"bad constraint target {t}")

    def _apply_constraint(self, c: Constraint):
        h = c.human
        center, radius = self._resolve_target(c.target)
        if c.limb == "head":
            self._apply_headbutt(c, center, radius)
            return
        side = c.limb[-1]
        pre = side.lower() + "_"
        if c.limb.startswith("knee"):
            hip = h.joints[pre + "hip"]
            d = normalize(center - hip)
            knee = hip + d * (h.d("thigh"))
            fold = normalize(-d * 0.45 - UP * 0.9)
            ankle = knee + fold * h.d("shin")
            h.joints[pre + "knee"], h.joints[pre + "ankle"] = knee, ankle
            h.params["feet"][side] = "follow"
            c.error = max(0.0, norm(knee - center) - radius - 0.055 * h.scale)
            return
        if c.limb.startswith("elbow"):
            sh = h.joints[pre + "shoulder"]
            d = normalize(center - sh)
            elbow = sh + d * h.d("upper_arm")
            fold = normalize(-d + h.frames["chest"][:, 2] * 0.8)
            wrist = elbow + fold * h.d("forearm")
            h.joints[pre + "elbow"], h.joints[pre + "wrist"] = elbow, wrist
            c.error = max(0.0, norm(elbow - center) - radius - 0.045 * h.scale)
            return

        root = h.limb_root(c.limb)
        approach = (np.array(c.approach) if c.approach is not None
                    else normalize(center - root, fallback=[0, 0, 1]))
        surface = center - approach * radius
        if c.limb.startswith("arm"):
            eff = hands_mod.effector_offset(c.verb, c.eff_part) * h.scale
            target = surface - approach * eff
            la, lb = h.limb_lengths(c.limb)
            pole = self._arm_pole(h, side, target)
            mid, end, reached = two_bone_ik(root, target, la, lb, pole)
            h.joints[pre + "elbow"], h.joints[pre + "wrist"] = mid, end
            h.params["hand_aim"][side] = (approach.tolist(),
                                          self._palm_hint(c.verb, approach))
            h.params["hand_style"][side] = hands_mod.finger_style_for(c.verb)
            c.error = norm(end - target)
        else:  # leg
            eff = (0.13 if c.verb != "stomp" else 0.06) * h.scale
            target = surface - approach * eff
            la, lb = h.limb_lengths(c.limb)
            pole = self._leg_pole(h, target, root)
            mid, end, reached = two_bone_ik(root, target, la, lb, pole)
            h.joints[pre + "knee"], h.joints[pre + "ankle"] = mid, end
            h.params["feet"][side] = "follow"
            c.error = norm(end - target)

    def _apply_headbutt(self, c: Constraint, center, radius):
        h = c.human
        for _ in range(6):
            fh, fr = h.part_point("forehead")
            err = (center - normalize(center - h.frames["head_center"]) * radius) - fh
            if norm(err) < 0.015:
                break
            d_local = h.frames["chest"].T @ err
            h.params["spine_pitch"] = clamp(
                h.params["spine_pitch"] - d_local[1] * 60, -30, 75)
            h.params["head_pitch"] = clamp(
                h.params["head_pitch"] - d_local[1] * 30, -40, 50)
            h.x += err[0] * 0.6
            h.z += err[2] * 0.6
            h.fk()
        c.error = norm((h.part_point("forehead")[0]) - center) - radius

    def _arm_pole(self, h: Human, side, target):
        sh = h.joints[("l_" if side == "L" else "r_") + "shoulder"]
        out = h.right() * (1 if side == "R" else -1)
        if target[1] > sh[1] + 0.05:
            return normalize(out * 0.9 - h.forward() * 0.25 + UP * 0.1)
        return normalize(out * 0.45 - UP * 0.8 - h.forward() * 0.15)

    def _leg_pole(self, h: Human, target, hip):
        if target[1] > hip[1] - 0.25:      # high kick: knee points up/forward
            return normalize(h.forward() * 0.6 + UP * 0.8)
        return normalize(h.forward() * 0.9 + UP * 0.3)

    def _palm_hint(self, verb, approach):
        # Palm normal faces the contact for pressing verbs, sideways for others.
        if verb in ("press", "palm", "slap", "push", "touch", "high_five"):
            return list(np.array(approach))
        return list(normalize(np.cross(np.array(approach), UP),
                              fallback=[0, 1, 0]))

    # ------------------------------------------------------- reach strategy
    def _ensure_reachable(self, h: Human, limb: str, target_fn, verb: str,
                          eff_part=None):
        """Bend the spine / dip the knees / step closer until the limb can
        reach. Mutates pose parameters; returns residual shortfall (m)."""
        la, lb = h.limb_lengths(limb)
        reach = (la + lb) * 0.97

        def shortfall():
            h.fk()
            self._resolve_look(h)
            center, radius = target_fn()
            root = h.limb_root(limb)
            eff = hands_mod.effector_offset(verb, eff_part) * h.scale \
                if limb.startswith("arm") else 0.13 * h.scale
            need = norm(center - root) - radius - eff
            return need - reach, center

        def bend_toward():
            """Fold the spine (and dip the knees for low targets)."""
            gap, center = shortfall()
            if gap <= 0:
                return gap
            low_target = center[1] < h.params["pelvis_y"] * 0.75
            max_pitch = 100 if low_target else 48
            import copy as _copy
            for _ in range(30):
                gap, center = shortfall()
                if gap <= 0:
                    return gap
                before = (_copy.deepcopy(h.params), gap)
                root = h.limb_root(limb)
                if center[1] < root[1] - 0.05:
                    h.params["spine_pitch"] = clamp(
                        h.params["spine_pitch"] + 6, -25, max_pitch)
                elif center[1] > root[1] + 0.05:
                    h.params["spine_pitch"] = clamp(
                        h.params["spine_pitch"] - 4, -25, max_pitch)
                d_local = rot_y(-h.yaw) @ (center - root)
                if abs(d_local[0]) > 0.15:
                    h.params["spine_yaw"] = clamp(
                        h.params["spine_yaw"] + (5 if d_local[0] > 0 else -5),
                        -45, 45)
                if low_target and h.params["spine_pitch"] >= max_pitch - 1:
                    # Hip-hinge exhausted; dip with the knees.
                    for s in ("L", "R"):
                        h.params["legs"][s]["bend"] = min(
                            h.params["legs"][s]["bend"] + 7, 120)
                    h.params["pelvis_y"] = max(h.params["pelvis_y"] - 0.045,
                                               0.30 * h.scale)
                new_gap, _ = shortfall()
                if new_gap >= before[1] - 1e-4:
                    # Bending made it worse (target behind the back, etc.):
                    # revert and stop climbing.
                    h.params = before[0]
                    h.fk()
                    return before[1]
            return gap

        def step_in():
            # Step toward the target, but never inside another body.
            gap, center = shortfall()
            if gap <= 0:
                return gap
            d = horizontal(center - V(h.x, 0, h.z))
            dist = norm(d)
            if dist > 1e-6:
                step = min(gap + 0.02, max(0.0, dist - MIN_ROOT_SEP))
                d = d / dist
                h.x += d[0] * step
                h.z += d[2] * step
            g, _ = shortfall()
            return g

        gap, center = shortfall()
        if gap <= 0:
            return 0.0
        is_lying = abs(h.params["pitch"]) > 45 or abs(h.params["roll"]) > 45
        can_bend = limb.startswith("arm") and not is_lying
        if gap > 0.2:
            gap = step_in()   # cover most of the distance on foot first
        if gap > 0 and can_bend:
            gap = bend_toward()
        if gap > 0:
            gap = step_in()
            if gap > 0 and can_bend:
                gap = bend_toward()  # stepping changed the geometry; re-bend
        gap, _ = shortfall()
        if gap > 0.03:
            self.warn(f"{h.name}: target for '{verb}' is {gap:.2f} m beyond "
                      f"reach even after bending/stepping; clamped")
        return max(gap, 0.0)

    # --------------------------------------------------------- run commands
    def run(self, cmds: list[Command]):
        for cmd in cmds:
            for c in self._expand_plurals(cmd):
                self._dispatch(c)
        self.finalize()

    def run_text(self, text: str):
        self.run(parse(text))

    def _expand_plurals(self, cmd: Command):
        a_pl = cmd.a.part in PLURAL_PARTS if cmd.a.part else False
        b_pl = cmd.b.part in PLURAL_PARTS if cmd.b.part else False
        if not a_pl and not b_pl:
            return [cmd]
        out = []
        a_parts = PLURAL_PARTS.get(cmd.a.part, (cmd.a.part,) * 2) if cmd.a.part else (None, None)
        b_parts = PLURAL_PARTS.get(cmd.b.part, (cmd.b.part,) * 2) if cmd.b.part else (None, None)
        # When both sides are plural, pair mirrored (A.left touches B.right),
        # which is what facing bodies do naturally.
        pairs = [(0, 1), (1, 0)] if (a_pl and b_pl) else [(0, 0), (1, 1)]
        for ia, ib in pairs:
            a = Ref(raw=cmd.a.raw, entity=cmd.a.entity,
                    part=a_parts[ia] if a_pl else cmd.a.part)
            b = Ref(raw=cmd.b.raw, entity=cmd.b.entity,
                    part=b_parts[ib] if b_pl else cmd.b.part)
            out.append(Command(cmd.verb, a, b, cmd.line_no, cmd.raw))
        return out

    def _dispatch(self, cmd: Command):
        v = cmd.verb
        table = {
            "create": self._cmd_create, "scale": self._cmd_scale,
            "at": self._cmd_at, "turn": self._cmd_turn,
            "hover": self._cmd_hover, "pose": self._cmd_pose,
            "face": self._cmd_face, "face_away": self._cmd_face_away,
            "face_same": self._cmd_face_same, "face_each": self._cmd_face_each,
            "look_at": self._cmd_look_at, "look": self._cmd_look,
            "point": self._cmd_point, "raise": self._cmd_point,
            "camera": self._cmd_camera,
            "sit_on": self._cmd_sit_on, "stand_on": self._cmd_stand_on,
            "lie_on": self._cmd_lie_on, "kneel_on": self._cmd_kneel_on,
            "lean_on": self._cmd_lean_on,
            "straddle": self._cmd_ride, "ride": self._cmd_ride,
            "piggyback": self._cmd_piggyback,
            "shoulder_sit": self._cmd_shoulder_sit,
            "shoulder_carry": self._cmd_shoulder_sit,
            "fireman_carry": self._cmd_fireman_carry,
            "cradle_carry": self._cmd_cradle_carry, "carry": self._cmd_cradle_carry,
            "hug": self._cmd_hug, "kiss": self._cmd_kiss,
            "handshake": self._cmd_handshake,
        }
        placements = {"infront": (0, 0, 1, 180), "in_front": (0, 0, 1, 180),
                      "behind": (0, 0, -1, 0), "left_of": (-1, 0, 0, None),
                      "right_of": (1, 0, 0, None), "beside": (1, 0, 0, "same"),
                      "next_to": (1, 0, 0, "same"), "near": None,
                      "between": None, "on_top": None, "on": None,
                      "above": None, "under": None, "below": None}
        if v in table:
            table[v](cmd)
        elif v in placements:
            self._cmd_place(cmd)
        elif v in CONTACT_VERBS:
            self._cmd_contact(cmd)
        else:
            raise SceneError(
                f"line {cmd.line_no}: unknown verb '{v}'. Known verbs: "
                f"{sorted(set(table) | set(placements) | CONTACT_VERBS)}")
        # Contact & pose changes should be reflected before the next command
        # so ordering semantics hold.
        self.evaluate()

    # ---- creation / configuration
    def _cmd_create(self, cmd: Command):
        name = cmd.a.entity
        kind = cmd.b.raw.upper()
        if name in self.entities:
            raise SceneError(f"line {cmd.line_no}: '{name}' already exists")
        if kind in ("HUMAN", "PERSON", "MAN", "WOMAN", "CHILD"):
            ent = Human(name, scale=0.62 if kind == "CHILD" else 1.0)
        else:
            ent = Prop(name, kind)
        ent.x = self._next_x
        self._next_x += 1.3
        self.entities[name] = ent
        self.order.append(name)

    def _cmd_scale(self, cmd: Command):
        ent = self.ent(cmd.a.entity, cmd.line_no)
        if cmd.b.number is None:
            raise SceneError(f"line {cmd.line_no}: scale needs a number")
        f = cmd.b.number
        ent.scale = f
        if isinstance(ent, Human):
            ent.params["pelvis_y"] = STAND_PELVIS_Y * f

    # ---- placement
    def _cmd_at(self, cmd: Command):
        ent = self.ent(cmd.a.entity, cmd.line_no)
        if cmd.b.pair is None or len(cmd.b.pair) not in (2, 3):
            raise SceneError(f"line {cmd.line_no}: at needs x:z or x:y:z")
        vals = cmd.b.pair
        ent.x = float(vals[0])
        ent.z = float(vals[-1])
        if len(vals) == 3:
            if isinstance(ent, Human):
                ent.params["pelvis_y"] = float(vals[1]) + STAND_PELVIS_Y * ent.scale
                ent.grounded = float(vals[1]) <= 0.01
            else:
                ent.y = float(vals[1])
        ent.placed = True

    def _cmd_turn(self, cmd: Command):
        ent = self.ent(cmd.a.entity, cmd.line_no)
        if cmd.b.number is None:
            raise SceneError(f"line {cmd.line_no}: turn needs degrees")
        ent.yaw += cmd.b.number

    def _cmd_hover(self, cmd: Command):
        ent = self.ent(cmd.a.entity, cmd.line_no)
        if cmd.b.number is None:
            raise SceneError(f"line {cmd.line_no}: hover needs height in meters")
        hgt = cmd.b.number
        if isinstance(ent, Human):
            ent.params["pelvis_y"] = hgt + STAND_PELVIS_Y * ent.scale
            ent.grounded = False
        else:
            ent.y = hgt
            ent.placed = True

    def _cmd_place(self, cmd: Command):
        v = cmd.verb
        a = self.ent(cmd.a.entity, cmd.line_no)
        if v == "between":
            if cmd.b.pair is None or len(cmd.b.pair) != 2:
                raise SceneError(f"line {cmd.line_no}: between,X,b1:b2")
            b1 = self.ent(str(cmd.b.pair[0]), cmd.line_no)
            b2 = self.ent(str(cmd.b.pair[1]), cmd.line_no)
            a.x = (b1.x + b2.x) / 2
            a.z = (b1.z + b2.z) / 2
            a.yaw = yaw_of(V(b2.x - a.x, 0, b2.z - a.z))
            a.placed = True
            return
        b = self.ent(cmd.b.entity, cmd.line_no)
        d = a.bounding_radius() + b.bounding_radius() + CLEARANCE
        fb, rb = rot_y(b.yaw) @ V(0, 0, 1), rot_y(b.yaw) @ V(1, 0, 0)
        pos_b = V(b.x, 0, b.z)
        if v in ("infront", "in_front"):
            p = pos_b + fb * d
            a.yaw = b.yaw + 180
        elif v == "behind":
            p = pos_b - fb * d
            a.yaw = b.yaw
        elif v == "left_of":
            p = pos_b - rb * d
            a.yaw = yaw_of(pos_b - p)
        elif v == "right_of":
            p = pos_b + rb * d
            a.yaw = yaw_of(pos_b - p)
        elif v in ("beside", "next_to"):
            p = pos_b + rb * (d * 0.85)
            a.yaw = b.yaw
        elif v == "near":
            cur = V(a.x, 0, a.z)
            dirv = normalize(horizontal(cur - pos_b), fallback=[0, 0, 1])
            p = pos_b + dirv * (d * 0.75)
            a.yaw = yaw_of(pos_b - p)
        elif v in ("on_top", "on"):
            p = pos_b
            if isinstance(b, Prop):
                if isinstance(a, Human):
                    # Keep the current pose's ground-relative pelvis height,
                    # lifted onto the object's top surface.
                    a.params["pelvis_y"] += b.top_y_world()
                    a.grounded = False
                else:
                    a.y = b.top_y_world()
                self.exempt(a, b)
            else:  # on top of a human: route to shoulder-sit geometry
                self._shoulder_sit(b, a)
                return
        elif v == "above":
            p = pos_b
            top = b.top_y_world() if isinstance(b, Prop) else \
                b.params["pelvis_y"] + 0.9 * b.scale
            if isinstance(a, Human):
                a.params["pelvis_y"] = top + 0.5 + STAND_PELVIS_Y * a.scale
                a.grounded = False
            else:
                a.y = top + 0.5
            self.exempt(a, b)
        elif v in ("under", "below"):
            p = pos_b
            if isinstance(a, Human):
                clearance_y = (b.top_y_world() if isinstance(b, Prop) else 1.4)
                if clearance_y < 1.2 and abs(a.params["pitch"]) < 45:
                    a.set_pose("lie_back")
                    self.warn(f"{a.name}: lying down to fit under {b.name}")
            self.exempt(a, b)
        else:
            raise SceneError(f"line {cmd.line_no}: unhandled placement {v}")
        a.x, a.z = float(p[0]), float(p[2])
        a.placed = True

    def _cmd_face(self, cmd: Command):
        a = self.ent(cmd.a.entity, cmd.line_no)
        b = self.ent(cmd.b.entity, cmd.line_no)
        a.yaw = yaw_of(V(b.x - a.x, 0, b.z - a.z))

    def _cmd_face_away(self, cmd: Command):
        a = self.ent(cmd.a.entity, cmd.line_no)
        b = self.ent(cmd.b.entity, cmd.line_no)
        a.yaw = yaw_of(V(a.x - b.x, 0, a.z - b.z))

    def _cmd_face_same(self, cmd: Command):
        a = self.ent(cmd.a.entity, cmd.line_no)
        b = self.ent(cmd.b.entity, cmd.line_no)
        a.yaw = b.yaw

    def _cmd_face_each(self, cmd: Command):
        a = self.ent(cmd.a.entity, cmd.line_no)
        b = self.ent(cmd.b.entity, cmd.line_no)
        a.yaw = yaw_of(V(b.x - a.x, 0, b.z - a.z))
        b.yaw = yaw_of(V(a.x - b.x, 0, a.z - b.z))

    # ---- pose / gaze / aim
    def _cmd_pose(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        if not isinstance(h, Human):
            raise SceneError(f"line {cmd.line_no}: pose only applies to humans")
        name = cmd.b.raw.lower()
        h.set_pose(name)
        h.grounded = name != "jump"

    def _cmd_look_at(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        if not isinstance(h, Human):
            raise SceneError(f"line {cmd.line_no}: look_at needs a human")
        if cmd.b.raw.lower() in ("ground", "floor", "down"):
            h.look_target = ("dir", "down")
        elif cmd.b.raw.lower() in ("sky", "up"):
            h.look_target = ("dir", "up")
        else:
            self.ent(cmd.b.entity, cmd.line_no)
            h.look_target = ("part", (cmd.b.entity, cmd.b.part))

    def _cmd_look(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        d = cmd.b.raw.lower()
        if d not in LOOK_DIRS:
            raise SceneError(f"line {cmd.line_no}: look direction must be one of "
                             f"{sorted(LOOK_DIRS)}")
        h.look_target = ("dir", d)

    def _cmd_point(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        part = cmd.a.part or "right_arm"
        sides = []
        if part in ("arms", "both_arms"):
            sides = [("arm", "L"), ("arm", "R")]
        elif part in ("left_arm", "left_hand"):
            sides = [("arm", "L")]
        elif part in ("right_arm", "right_hand"):
            sides = [("arm", "R")]
        elif part == "left_leg":
            sides = [("leg", "L")]
        elif part == "right_leg":
            sides = [("leg", "R")]
        else:
            raise SceneError(f"line {cmd.line_no}: point works on arms/legs, "
                             f"got '{part}'")
        dirs = {"up": V(0, 1, 0), "down": V(0, -1, 0),
                "forward": None, "back": None, "left": None, "right": None,
                "forward_up": None, "forward_down": None}
        h.fk()
        token = cmd.b.raw.lower()
        f, r = h.forward(), h.right()
        world_dirs = {"up": V(0, 1, 0), "down": V(0, -1, 0), "forward": f,
                      "back": -f, "left": -r, "right": r,
                      "forward_up": normalize(f + UP),
                      "forward_down": normalize(f - UP),
                      "up_left": normalize(UP - r), "up_right": normalize(UP + r)}
        if token in world_dirs:
            wdir = world_dirs[token]
        else:
            tgt = self.ent(cmd.b.entity, cmd.line_no)
            focus = (tgt.part_point(cmd.b.part)[0] if cmd.b.part
                     else self._entity_focus(tgt))
            base = h.joints["r_shoulder" if sides[0][1] == "R" else "l_shoulder"] \
                if sides[0][0] == "arm" else h.joints["r_hip" if sides[0][1] == "R" else "l_hip"]
            wdir = normalize(focus - base)
        for chain, side in sides:
            local = (h.frames["chest"].T if chain == "arm"
                     else h.pelvis_frame().T) @ wdir
            key = "arms" if chain == "arm" else "legs"
            h.params[key][side]["dir"] = local
            h.params[key][side]["bend"] = 4.0
            if chain == "arm":
                h.params["hand_style"][side] = "point"
                h.params["hand_aim"][side] = (wdir.tolist(),
                                              list(normalize(np.cross(wdir, UP),
                                                             fallback=[0, 1, 0])))
            else:
                h.params["feet"][side] = "follow"

    def _cmd_camera(self, cmd: Command):
        self.camera_cmd = (cmd.a.raw.lower(), cmd.b.raw)

    # ---- support poses on objects
    def _sittable_anchor(self, prop: Prop, part, line):
        if part:
            return prop.anchor(part)
        if prop.spec.sit_anchor is None:
            raise SceneError(f"line {line}: {prop.kind} is not sittable; "
                             "give an anchor like obj.top")
        return prop.anchor(prop.spec.sit_anchor)

    def _cmd_sit_on(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        p = self.ent(cmd.b.entity, cmd.line_no)
        if isinstance(p, Human):  # sitting on a person = shoulder sit
            self._shoulder_sit(p, h)
            return
        seat = self._sittable_anchor(p, cmd.b.part, cmd.line_no)
        h.set_pose("sit_chair")
        h.yaw = p.yaw
        h.x, h.z = float(seat[0]), float(seat[2])
        # Scoot back slightly so thighs rest on the seat.
        back = rot_y(h.yaw) @ V(0, 0, -0.05)
        h.x += float(back[0])
        h.z += float(back[2])
        h.params["pelvis_y"] = float(seat[1]) + 0.10 * h.scale
        h.grounded = False
        h.placed = True
        self.exempt(h, p)

    def _cmd_stand_on(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        p = self.ent(cmd.b.entity, cmd.line_no)
        if not isinstance(p, Prop):
            raise SceneError(f"line {cmd.line_no}: stand_on needs an object")
        top = p.anchor(cmd.b.part) if cmd.b.part else p.to_world(
            V(0, p.spec.top_y, 0))
        h.set_pose("stand")
        h.x, h.z = float(top[0]), float(top[2])
        h.params["pelvis_y"] = float(top[1]) + STAND_PELVIS_Y * h.scale
        h.grounded = False
        h.placed = True
        self.exempt(h, p)

    def _cmd_lie_on(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        p = self.ent(cmd.b.entity, cmd.line_no)
        if not isinstance(p, Prop):
            raise SceneError(f"line {cmd.line_no}: lie_on needs an object")
        h.set_pose("lie_back")
        h.yaw = p.yaw
        h.x, h.z = p.x, p.z
        h.params["pelvis_y"] = p.top_y_world() + 0.13 * h.scale
        h.grounded = False
        h.placed = True
        self.exempt(h, p)

    def _cmd_kneel_on(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        p = self.ent(cmd.b.entity, cmd.line_no)
        h.set_pose("kneel")
        if isinstance(p, Prop):
            h.x, h.z = p.x, p.z
            h.params["pelvis_y"] = p.top_y_world() + (DIMS["thigh"] + 0.06) * h.scale
            h.grounded = False
            self.exempt(h, p)
        h.placed = True

    def _cmd_lean_on(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        p = self.ent(cmd.b.entity, cmd.line_no)
        if not isinstance(p, Prop):
            raise SceneError(f"line {cmd.line_no}: lean_on needs an object")
        anchor_name = cmd.b.part or p.default_grip()
        anchor = p.anchor(anchor_name)
        # Stand within arm reach of the anchor, lean in, press a palm on it.
        dirv = normalize(horizontal(V(h.x, 0, h.z) - V(p.x, 0, p.z)),
                         fallback=[0, 0, 1])
        stand = V(anchor[0], 0, anchor[2]) + dirv * (ARM_LEN * h.scale * 0.85)
        h.x, h.z = float(stand[0]), float(stand[2])
        h.yaw = yaw_of(-dirv)
        h.params["spine_pitch"] = 16
        h.placed = True
        self.exempt(h, p)
        self._add_contact(h, "right_palm", "press",
                          {"kind": "anchor", "entity": p.name,
                           "anchor": anchor_name}, cmd.line_no)

    def _cmd_ride(self, cmd: Command):
        h = self.ent(cmd.a.entity, cmd.line_no)
        p = self.ent(cmd.b.entity, cmd.line_no)
        if not isinstance(h, Human):
            raise SceneError(f"line {cmd.line_no}: ride needs a human rider")
        if isinstance(p, Human):  # riding a person = piggyback
            self._piggyback(p, h)
            return
        if p.spec.ride_axis is None and p.spec.sit_anchor is None:
            raise SceneError(f"line {cmd.line_no}: cannot straddle a {p.kind}")
        if p.spec.ride_axis:
            a0 = p.anchor(p.spec.ride_axis[0])
            a1 = p.anchor(p.spec.ride_axis[1])
            axis = a1 - a0
            mid = (a0 + a1) / 2
            h.set_pose("straddle")
            h.params["spine_pitch"] = 26  # racing lean toward the grip
            h.yaw = yaw_of(horizontal(axis))
            h.x, h.z = float(mid[0]), float(mid[2])
            h.params["pelvis_y"] = float(mid[1]) + 0.16 * h.scale
            h.grounded = False
            h.placed = True
            self.exempt(h, p)
            self.attachments.append({"kind": "offset", "dep": h, "base": p,
                                     "offset": (0.0,
                                                float(mid[1]) - p.y + 0.16 * h.scale,
                                                0.0),
                                     "dyaw": h.yaw - p.yaw})
            # Hands grip the shaft ahead of the rider.
            fwd = normalize(horizontal(axis), fallback=[0, 0, 1])
            for side, back in (("right", 0.28), ("left", 0.38)):
                grip = mid + normalize(axis) * back
                self._add_contact(h, f"{side}_hand", "hold",
                                  {"kind": "point", "p": grip.tolist(),
                                   "radius": 0.022 * p.scale}, cmd.line_no)
        else:  # straddle a chair-like object, facing its backrest
            seat = p.anchor(p.spec.sit_anchor)
            h.set_pose("straddle")
            h.yaw = p.yaw + 180
            h.x, h.z = float(seat[0]), float(seat[2])
            h.params["pelvis_y"] = float(seat[1]) + 0.10 * h.scale
            h.grounded = False
            h.placed = True
            self.exempt(h, p)

    # ---- contact commands
    def _effector_limb(self, h: Human, part, verb, line):
        if part is None:
            part = "right_foot" if verb in ("kick", "stomp") else \
                ("head" if verb == "headbutt" else "right_hand")
        if verb == "headbutt" or part in ("head", "forehead"):
            return "head", part
        if part in ("left_knee", "right_knee") and verb == "knee":
            return ("knee_L" if part.startswith("left") else "knee_R"), part
        if part in ("left_elbow", "right_elbow") and verb == "elbow":
            return ("elbow_L" if part.startswith("left") else "elbow_R"), part
        side = "L" if part.startswith("left_") else \
            ("R" if part.startswith("right_") else None)
        if side is None:
            raise SceneError(f"line {line}: effector '{part}' needs a side, "
                             "e.g. right_palm / left_foot")
        sub = part.split("_", 1)[1]
        if sub in ARM_EFFECTORS:
            return f"arm_{side}", part
        if sub in LEG_EFFECTORS:
            return f"leg_{side}", part
        if sub == "knee":
            return f"knee_{side}", part
        if sub == "elbow":
            return f"elbow_{side}", part
        raise SceneError(f"line {line}: '{part}' cannot be used as an effector")

    VERB_DEFAULT_PART = {"push": "chest", "punch": "chest", "choke": "neck",
                         "kick": "hips", "headbutt": "face", "slap": "face",
                         "stomp": "chest", "hold": "left_wrist",
                         "grab": "left_wrist", "pull": "left_wrist"}

    def _target_spec(self, ref: Ref, actor: Human, verb, line):
        ent = self.ent(ref.entity, line)
        if isinstance(ent, Human):
            part = ref.part or self.VERB_DEFAULT_PART.get(verb, "chest")
            if ent.joints:
                ent.part_point(part)  # validate the part name early
            return {"kind": "part", "entity": ent.name, "part": part}, ent
        anchor = ref.part or ent.default_grip()
        return {"kind": "anchor", "entity": ent.name, "anchor": anchor,
                "radius": 0.02 * ent.scale}, ent

    def _add_contact(self, h: Human, eff_part, verb, target, line,
                     approach=None):
        limb, eff = self._effector_limb(h, eff_part, verb, line)
        self._seq += 1
        c = Constraint(h, limb, verb, target, approach=approach, seq=self._seq,
                       eff_part=eff)
        self.constraints.append(c)
        self.evaluate()
        if limb != "head":
            # Knee/elbow strikes reach with the parent chain of the same side.
            chain = {"knee": "leg", "elbow": "arm"}.get(limb.split("_")[0],
                                                        limb.split("_")[0])
            self._ensure_reachable(h, f"{chain}_{limb[-1]}",
                                   lambda: self._resolve_target(target), verb,
                                   eff_part=eff)
        return c

    def _cmd_contact(self, cmd: Command):
        verb = cmd.verb
        h = self.ent(cmd.a.entity, cmd.line_no)
        if not isinstance(h, Human):
            raise SceneError(
                f"line {cmd.line_no}: '{verb}' needs a human effector like "
                f"{cmd.a.entity}.right_hand (objects are moved with hold/place)")
        target_ent = self.ent(cmd.b.entity, cmd.line_no)

        # hold on an unplaced object: attach it to the hand instead of reaching.
        if verb in ("hold", "grab", "grip") and isinstance(target_ent, Prop) \
                and target_ent.spec.portable and not target_ent.placed \
                and target_ent.attached_to is None:
            part = cmd.a.part or "right_hand"
            side = "L" if part.startswith("left") else "R"
            anchor = cmd.b.part or target_ent.default_grip()
            target_ent.anchor(anchor)  # validate
            target_ent.attached_to = (h, side, anchor)
            h.params["hand_style"][side] = "grab"
            self.exempt(h, target_ent)
            return

        target, tent = self._target_spec(cmd.b, h, verb, cmd.line_no)
        approach = None
        if verb == "stomp":
            approach = [0.0, -1.0, 0.0]
        if tent is not h:
            self.exempt(h, tent)

        if verb == "punch":
            side = "R" if (cmd.a.part or "right").startswith("right") else "L"
            h.params["spine_yaw"] = clamp(
                h.params["spine_yaw"] + (-16 if side == "R" else 16), -40, 40)
        if verb in ("push", "pull") and abs(h.params["pitch"]) < 45:
            h.params["spine_pitch"] += 10 if verb == "push" else -14
            brace, front = ("L", "R") if (cmd.a.part or "right").startswith("right") \
                else ("R", "L")
            h.params["legs"][brace]["dir"] = V(0.05 if brace == "L" else -0.05,
                                               -0.92, -0.38)
            h.params["legs"][brace]["bend"] = 12
            h.params["legs"][front]["bend"] = 25
            h.params["feet"][brace] = "follow"
        if verb == "kick":
            c_center, _ = self._resolve_target(target)
            lean = clamp((float(c_center[1]) - 0.7) * 40, 0, 38)
            h.params["spine_pitch"] = -lean
            h.params["arms"]["L"]["dir"] = V(-0.85, -0.15, -0.2)
            h.params["arms"]["L"]["bend"] = 20
            h.params["arms"]["R"]["dir"] = V(0.85, -0.15, -0.2)
            h.params["arms"]["R"]["bend"] = 20

        self._add_contact(h, cmd.a.part, verb, target, cmd.line_no,
                          approach=approach)

    # ---- two-person macros
    def _cmd_handshake(self, cmd: Command):
        a = self.ent(cmd.a.entity, cmd.line_no)
        b = self.ent(cmd.b.entity, cmd.line_no)
        self._face_pair(a, b, 0.86)
        for h, other in ((a, b), (b, a)):
            self._seq += 1
            self.constraints.append(Constraint(
                h, "arm_R", "handshake",
                {"kind": "midgrip", "a": a.name, "b": b.name}, seq=self._seq))
            h.params["hand_style"]["R"] = "grab"
        a.look_target = ("part", (b.name, None))
        b.look_target = ("part", (a.name, None))
        self.exempt(a, b)

    def _cmd_hug(self, cmd: Command):
        a = self.ent(cmd.a.entity, cmd.line_no)
        b = self.ent(cmd.b.entity, cmd.line_no)
        self._face_pair(a, b, 0.34)
        self.exempt(a, b)
        for h, other in ((a, b), (b, a)):
            fwd_o = rot_y(other.yaw) @ V(0, 0, 1)
            self._add_contact(h, "right_hand", "press",
                              {"kind": "part", "entity": other.name,
                               "part": "upper_back"}, cmd.line_no,
                              approach=list(-fwd_o))
            self._add_contact(h, "left_hand", "press",
                              {"kind": "part", "entity": other.name,
                               "part": "back"}, cmd.line_no,
                              approach=list(-fwd_o))
            h.params["head_yaw"] = 18
        a.look_target = None
        b.look_target = None

    def _cmd_kiss(self, cmd: Command):
        a = self.ent(cmd.a.entity, cmd.line_no)
        b = self.ent(cmd.b.entity, cmd.line_no)
        self._face_pair(a, b, 0.36)
        a.look_target = ("part", (b.name, None))
        b.look_target = ("part", (a.name, None))
        for h in (a, b):
            h.params["spine_pitch"] = max(h.params["spine_pitch"], 7)
        self.exempt(a, b)

    def _face_pair(self, a: Human, b: Human, dist: float):
        if not b.placed and a.placed:
            fa = rot_y(a.yaw) @ V(0, 0, 1)
            b.x, b.z = a.x + float(fa[0]) * dist, a.z + float(fa[2]) * dist
        elif not a.placed and b.placed:
            fbv = rot_y(b.yaw) @ V(0, 0, 1)
            a.x, a.z = b.x + float(fbv[0]) * dist, b.z + float(fbv[2]) * dist
        else:
            mid = V((a.x + b.x) / 2, 0, (a.z + b.z) / 2)
            d = normalize(horizontal(V(b.x - a.x, 0, b.z - a.z)),
                          fallback=[0, 0, 1])
            a.x, a.z = float(mid[0] - d[0] * dist / 2), float(mid[2] - d[2] * dist / 2)
            b.x, b.z = float(mid[0] + d[0] * dist / 2), float(mid[2] + d[2] * dist / 2)
        a.yaw = yaw_of(V(b.x - a.x, 0, b.z - a.z))
        b.yaw = a.yaw + 180
        a.placed = b.placed = True

    def _cmd_piggyback(self, cmd: Command):
        carrier = self.ent(cmd.a.entity, cmd.line_no)
        rider = self.ent(cmd.b.entity, cmd.line_no)
        self._piggyback(carrier, rider)

    def _piggyback(self, carrier: Human, rider: Human):
        line = 0
        carrier.params["spine_pitch"] = max(carrier.params["spine_pitch"], 16)
        for s in ("L", "R"):
            carrier.params["legs"][s]["bend"] = max(
                carrier.params["legs"][s]["bend"], 14)
        rider.set_pose("stand")
        rider.params["spine_pitch"] = 22   # chest pressed to the carrier's back
        for s, sgn in (("L", -1), ("R", 1)):
            rider.params["legs"][s] = {"dir": V(sgn * -0.55, -0.35, 0.72),
                                       "bend": 100, "pole": V(sgn * -0.4, 0.7, 0.5)}
            rider.params["feet"][s] = "follow"
        self.attachments.append({
            "kind": "offset", "dep": rider, "base": carrier,
            "offset": (0.0, 0.38 * rider.scale,
                       -(0.16 * carrier.scale + 0.08 * rider.scale)),
            "dyaw": 0.0})
        rider.grounded = False
        rider.placed = True
        self.exempt(carrier, rider)
        # Rider's hands drape over the carrier's collarbones.
        for side, part in (("left", "right_shoulder"), ("right", "left_shoulder")):
            fwd_c = rot_y(carrier.yaw) @ V(0, 0, 1)
            self._add_contact(rider, f"{side}_hand", "press",
                              {"kind": "part", "entity": carrier.name,
                               "part": part}, line, approach=list(-fwd_c * 0.4 - UP * 0.6))
        # Carrier's hands hook under the rider's thighs.
        for side, part in (("left", "left_thigh"), ("right", "right_thigh")):
            self._add_contact(carrier, f"{side}_hand", "grab",
                              {"kind": "under_part", "entity": rider.name,
                               "part": part}, line, approach=[0.0, 1.0, 0.0])

    def _cmd_shoulder_sit(self, cmd: Command):
        carrier = self.ent(cmd.a.entity, cmd.line_no)
        rider = self.ent(cmd.b.entity, cmd.line_no)
        self._shoulder_sit(carrier, rider)

    def _shoulder_sit(self, carrier: Human, rider: Human):
        line = 0
        rider.set_pose("stand")
        rider.params["spine_pitch"] = 6
        for s, sgn in (("L", -1), ("R", 1)):
            rider.params["legs"][s] = {"dir": V(sgn * -0.35, -0.72, 0.42),
                                       "bend": 62, "pole": V(sgn * -0.3, 0.5, 0.8)}
            rider.params["feet"][s] = "follow"
        neck_off = (DIMS["spine"] + 0.16) * carrier.scale
        self.attachments.append({
            "kind": "offset", "dep": rider, "base": carrier,
            "offset": (0.0, neck_off, -0.045 * carrier.scale), "dyaw": 0.0})
        rider.grounded = False
        rider.placed = True
        self.exempt(carrier, rider)
        for side, part in (("left", "left_shin"), ("right", "right_shin")):
            self._add_contact(carrier, f"{side}_hand", "grab",
                              {"kind": "part", "entity": rider.name,
                               "part": part}, line)
        for side in ("left", "right"):
            self._add_contact(rider, f"{side}_hand", "touch",
                              {"kind": "part", "entity": carrier.name,
                               "part": "top_of_head"}, line,
                              approach=[0.0, -1.0, 0.0])

    def _cmd_fireman_carry(self, cmd: Command):
        carrier = self.ent(cmd.a.entity, cmd.line_no)
        carried = self.ent(cmd.b.entity, cmd.line_no)
        line = cmd.line_no
        carrier.params["spine_pitch"] = 10
        for s in ("L", "R"):
            carrier.params["legs"][s]["bend"] = max(
                carrier.params["legs"][s]["bend"], 18)
        # Carried body lies across the shoulders, head toward carrier's left.
        carried.set_pose("stand")
        carried.params["pitch"] = 78
        carried.params["spine_pitch"] = 28
        carried.params["head_pitch"] = 30
        for s in ("L", "R"):
            carried.params["legs"][s] = {"dir": V(0, -0.9, 0.30), "bend": 55,
                                         "pole": V(0, 0.4, 1)}
            carried.params["feet"][s] = "follow"
            carried.params["arms"][s] = {"dir": V(-0.2 if s == "L" else 0.2, -0.95, 0.15),
                                         "bend": 15, "pole": V(0, 0, 1)}
        shoulder_h = (DIMS["spine"] - 0.02) * carrier.scale
        self.attachments.append({
            "kind": "offset", "dep": carried, "base": carrier,
            "offset": (0.14 * carrier.scale, shoulder_h + 0.14 * carried.scale,
                       -0.02 * carrier.scale),
            "dyaw": -90.0})
        carried.grounded = False
        carried.placed = True
        self.exempt(carrier, carried)
        # Carrier's right arm wraps over the carried knees; left hand locks
        # the carried person's right wrist across the chest.
        self._add_contact(carrier, "right_hand", "grab",
                          {"kind": "part", "entity": carried.name,
                           "part": "right_knee"}, line)
        self._add_contact(carrier, "left_hand", "grab",
                          {"kind": "part", "entity": carried.name,
                           "part": "right_wrist"}, line)

    def _cmd_cradle_carry(self, cmd: Command):
        carrier = self.ent(cmd.a.entity, cmd.line_no)
        carried = self.ent(cmd.b.entity, cmd.line_no)
        line = cmd.line_no
        carried.set_pose("stand")
        carried.params["pitch"] = -78
        carried.params["spine_pitch"] = 18
        carried.params["head_pitch"] = 20
        for s in ("L", "R"):
            carried.params["legs"][s] = {"dir": V(0, -0.85, -0.35), "bend": 55,
                                         "pole": V(0, 0.5, -0.8)}
            carried.params["feet"][s] = "follow"
        self.attachments.append({
            "kind": "offset", "dep": carried, "base": carrier,
            "offset": (0.06 * carrier.scale, 0.22 * carrier.scale,
                       0.28 * carrier.scale),
            "dyaw": 90.0})
        carried.grounded = False
        carried.placed = True
        self.exempt(carrier, carried)
        self._add_contact(carrier, "left_hand", "press",
                          {"kind": "under_part", "entity": carried.name,
                           "part": "upper_back"}, line, approach=[0.0, 1.0, 0.0])
        self._add_contact(carrier, "right_hand", "press",
                          {"kind": "under_part", "entity": carried.name,
                           "part": "left_knee"}, line, approach=[0.0, 1.0, 0.0])
        self._add_contact(carried, "right_hand", "touch",
                          {"kind": "part", "entity": carrier.name,
                           "part": "neck"}, line)
        carried.look_target = ("part", (carrier.name, None))

    # ------------------------------------------------------------- finalize
    def finalize(self):
        self.evaluate()
        for _ in range(4):
            moved = self._collision_pass()
            self._ground_pass()
            self.evaluate()
            if not moved:
                break
        self._ground_pass()
        self.evaluate()
        for c in self.constraints:
            if c.error > 0.04:
                self.warn(f"{c.human.name}: '{c.verb}' constraint is off by "
                          f"{c.error:.2f} m")

    def _ground_pass(self):
        for h in self.humans():
            if not h.grounded:
                low = h.lowest_point()
                if low < -0.005:  # never sink below the floor, even airborne
                    h.params["pelvis_y"] += -low
                continue
            low = h.lowest_point()
            if abs(low) > 0.004:
                h.params["pelvis_y"] -= low
        for p in self.props():
            if p.attached_to is not None:
                continue
            low = p.lowest_point()
            if low < -0.004:
                p.y += -low

    def _human_capsules(self, h: Human):
        J = h.joints
        s = h.scale
        caps = [
            (J["mid_hip"], J["neck"], 0.125 * s, "torso"),
            (h.frames["head_center"], h.frames["head_center"], 0.10 * s, "head"),
        ]
        for pre in ("l_", "r_"):
            caps.append((J[pre + "hip"], J[pre + "knee"], 0.07 * s, pre + "thigh"))
            caps.append((J[pre + "knee"], J[pre + "ankle"], 0.05 * s, pre + "shin"))
            caps.append((J[pre + "shoulder"], J[pre + "elbow"], 0.045 * s, pre + "uarm"))
            caps.append((J[pre + "elbow"], J[pre + "wrist"], 0.04 * s, pre + "farm"))
        return caps

    def _collision_pass(self):
        moved = False
        hums = self.humans()
        for i in range(len(hums)):
            for j in range(i + 1, len(hums)):
                a, b = hums[i], hums[j]
                if frozenset((a.name, b.name)) in self.exempt_pairs:
                    continue
                push = self._pair_penetration(a, b)
                if push is not None:
                    b.x += push[0]
                    b.z += push[2]
                    moved = True
        for h in hums:
            for p in self.props():
                if p.attached_to is not None:
                    continue
                if frozenset((h.name, p.name)) in self.exempt_pairs:
                    continue
                push = self._prop_penetration(h, p)
                if push is not None:
                    h.x += push[0]
                    h.z += push[2]
                    moved = True
        return moved

    def _pair_penetration(self, a: Human, b: Human):
        worst = 0.0
        for (p1, q1, r1, _) in self._human_capsules(a):
            for (p2, q2, r2, _) in self._human_capsules(b):
                _, _, d = seg_seg_closest(p1, q1, p2, q2)
                pen = (r1 + r2) - d
                if pen > worst:
                    worst = pen
        if worst < 0.015:
            return None
        d = horizontal(V(b.x - a.x, 0, b.z - a.z))
        if norm(d) < 1e-6:
            d = rot_y(a.yaw) @ V(0, 0, 1)
        d = normalize(d)
        return d * (worst + 0.01)

    def _prop_penetration(self, h: Human, p: Prop):
        worst = 0.0
        for (p1, q1, r1, _) in self._human_capsules(h):
            for shp in p.world_shapes():
                d = self._shape_seg_dist(shp, p1, q1)
                pen = r1 - d
                if pen > worst:
                    worst = pen
        if worst < 0.02:
            return None
        d = horizontal(V(h.x - p.x, 0, h.z - p.z))
        if norm(d) < 1e-6:
            d = rot_y(p.yaw) @ V(0, 0, 1)
        d = normalize(d)
        return d * (worst + 0.01)

    def _shape_seg_dist(self, shp, a, b):
        samples = [a + (b - a) * t for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        if shp[0] == "sphere":
            _, c, r = shp
            return min(norm(s - c) for s in samples) - r
        if shp[0] == "cyl":
            _, c0, c1, r = shp
            _, _, d = seg_seg_closest(a, b, c0, c1)
            return d - r
        _, c, h, F = shp
        best = 1e9
        for s in samples:
            local = F.T @ (s - c)
            q = np.maximum(np.abs(local) - h, 0.0)
            outside = norm(q)
            inside = min(0.0, float(np.max(np.abs(local) - h)))
            best = min(best, outside + inside)
        return best


def build_scene(text: str) -> Scene:
    sc = Scene()
    sc.run_text(text)
    return sc
