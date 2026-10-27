// Physics and contact checks on the real meshes.
//
// - bake(): the posed, skinned (and dented) surface of a model in world space, every vertex labelled
//   with the body part that drives it, plus a BVH for signed-distance queries.
// - contactReport(): where two actors interpenetrate (clipping), where they touch, what goes through
//   the floor, and arms through the own body. All in metres, measured on the meshes that are drawn.
// - fitColliders()/linkSprings(): capsules fitted to each body part of the actual mesh, so hair and
//   cloth springs collide with every body in the scene (not only the model's own few colliders) and
//   with the floor.
import * as THREE from "three";
import { MeshBVH } from "three-mesh-bvh";
import {
  VRMSpringBoneCollider, VRMSpringBoneColliderShapeCapsule, VRMSpringBoneColliderShapePlane,
  VRMSpringBoneColliderShapeSphere,
} from "@pixiv/three-vrm";

// body part segments: [part, bone at the start, bone at the end (or null: sphere at the start)]
export const PARTS = [
  ["hips", "hips", "spine"], ["spine", "spine", "chest"], ["chest", "chest", "neck"], ["neck", "neck", "head"],
  ["head", "head", "@crown"],  // skull: from the head bone up to the middle of the cranium
  ["l_thigh", "leftUpperLeg", "leftLowerLeg"], ["l_shin", "leftLowerLeg", "leftFoot"], ["l_foot", "leftFoot", "leftToes"],
  ["r_thigh", "rightUpperLeg", "rightLowerLeg"], ["r_shin", "rightLowerLeg", "rightFoot"], ["r_foot", "rightFoot", "rightToes"],
  ["l_upperarm", "leftUpperArm", "leftLowerArm"], ["l_forearm", "leftLowerArm", "leftHand"], ["l_hand", "leftHand", "leftMiddleProximal"],
  ["r_upperarm", "rightUpperArm", "rightLowerArm"], ["r_forearm", "rightLowerArm", "rightHand"], ["r_hand", "rightHand", "rightMiddleProximal"],
];
const BONE_PART = {
  hips: "hips", spine: "spine", chest: "chest", upperChest: "chest", neck: "neck", head: "head",
  leftEye: "head", rightEye: "head", jaw: "head",
  leftUpperLeg: "l_thigh", leftLowerLeg: "l_shin", leftFoot: "l_foot", leftToes: "l_foot",
  rightUpperLeg: "r_thigh", rightLowerLeg: "r_shin", rightFoot: "r_foot", rightToes: "r_foot",
  leftShoulder: "chest", rightShoulder: "chest",
  leftUpperArm: "l_upperarm", leftLowerArm: "l_forearm", leftHand: "l_hand",
  rightUpperArm: "r_upperarm", rightLowerArm: "r_forearm", rightHand: "r_hand",
};
export const ARM_PARTS = new Set(["l_upperarm", "l_forearm", "l_hand", "r_upperarm", "r_forearm", "r_hand"]);
// parts that give when pressed (flesh); the rest (head, hands, feet, shins, forearms) are treated as rigid
export const SOFT = new Set(["hips", "spine", "chest", "l_thigh", "r_thigh", "l_upperarm", "r_upperarm", "neck"]);

// rest-pose world ends of a part's segment (the skull runs from the head bone up 9 cm)
function segEnds(vrm, a, b) {
  const na = vrm.humanoid.getRawBoneNode(a);
  if (!na) return null;
  const pa = na.getWorldPosition(new THREE.Vector3());
  if (b === "@crown") return [na, pa, pa.clone().add(new THREE.Vector3(0, 0.09, 0.01))];
  const nb = b ? vrm.humanoid.getRawBoneNode(b) : null;
  return [na, pa, nb ? nb.getWorldPosition(new THREE.Vector3()) : null];
}

function boneTable(vrm) {
  // raw skeleton node -> body part (fingers map to the hand, springs to what they hang from)
  const raw = new Map();
  for (const name of Object.keys(vrm.humanoid.humanBones)) {
    const n = vrm.humanoid.getRawBoneNode(name);
    if (!n) continue;
    let part = BONE_PART[name];
    if (!part && /Thumb|Index|Middle|Ring|Little/.test(name)) part = name.startsWith("left") ? "l_hand" : "r_hand";
    if (part) raw.set(n, part);
  }
  const springs = new Set();
  if (vrm.springBoneManager) vrm.springBoneManager.joints.forEach((j) => springs.add(j.bone));
  return { raw, springs };
}

function partOf(bone, table, cache) {
  if (cache.has(bone)) return cache.get(bone);
  let spring = false, part = "hips";
  for (let p = bone; p; p = p.parent) {
    if (table.springs.has(p)) spring = true;
    if (table.raw.has(p)) { part = table.raw.get(p); break; }
  }
  const r = { part, spring };
  cache.set(bone, r);
  return r;
}

// Same falloff as the dent shader, applied on the CPU so the checks see the dented surface.
function applyDents(p, n, dents) {
  for (const d of dents || []) {
    const dir = d.dirV, rel = p.clone().sub(d.posV);
    const along = rel.dot(dir);
    const lat = rel.clone().addScaledVector(dir, -along);
    const x = lat.length() / d.radius;
    const depthFall = 1 - THREE.MathUtils.smoothstep(Math.abs(along), d.radius * 0.6, d.radius * 1.8);
    const facing = THREE.MathUtils.smoothstep(n.dot(dir) * -1, -0.35, 0.25);
    const f = x < 1 ? (1 - x * x) ** 2 : 0;
    const bulge = THREE.MathUtils.smoothstep(x, 0.75, 1.05) * (1 - THREE.MathUtils.smoothstep(x, 1.05, 1.7));
    p.addScaledVector(dir, d.depth * (f - (d.bulge ?? 0.32) * bulge) * depthFall * facing * d.mask);
  }
}

// World-space surface of one actor: positions, part label per vertex, hair/cloth flags, BVH.
export function bake(entry, dents) {
  const { vrm } = entry;
  entry.scene.updateMatrixWorld(true);
  const table = boneTable(vrm), cache = new Map();
  const pos = [], idx = [], part = [], kind = [];  // kind: 0 skin, 1 hair, 2 spring cloth, 3 worn clothes
  const D = (dents || []).map((d) => ({ ...d, posV: new THREE.Vector3(...d.pos), dirV: new THREE.Vector3(...d.dir).normalize() }));
  const v = new THREE.Vector3(), nrm = new THREE.Vector3();
  entry.scene.traverse((o) => {
    if (!o.isMesh || !o.visible) return;
    const g = o.geometry, P = g.getAttribute("position");
    if (!P) return;
    const base = pos.length / 3;
    const si = g.getAttribute("skinIndex"), sw = g.getAttribute("skinWeight"), N = g.getAttribute("normal");
    const mask = g.getAttribute("dentMask");
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    const hairMesh = /hair/i.test(o.name) || mats.every((m) => /hair/i.test(m.name || ""));
    const clothMesh = !hairMesh && mats.every((m) => /_CLOTH|cloth|shoe|tops|bottoms/i.test(m.name || ""));
    const nm = new THREE.Matrix3().getNormalMatrix(o.matrixWorld);
    for (let i = 0; i < P.count; i++) {
      o.getVertexPosition(i, v);
      v.applyMatrix4(o.matrixWorld);
      let lab = { part: "hips", spring: false };
      if (o.isSkinnedMesh && si && sw) {
        let best = -1, bw = -1;
        for (let k = 0; k < 4; k++) { const w = sw.getComponent(i, k); if (w > bw) { bw = w; best = si.getComponent(i, k); } }
        lab = partOf(o.skeleton.bones[best], table, cache);
      }
      if (D.length && N) {
        nrm.fromBufferAttribute(N, i).applyMatrix3(nm).normalize();  // rest normal: good enough for the facing term
        applyDents(v, nrm, D.map((d) => ({ ...d, mask: mask ? mask.getX(i) : 1 })));
      }
      pos.push(v.x, v.y, v.z);
      part.push(lab.part);
      kind.push(hairMesh ? 1 : (lab.spring ? (lab.part === "head" ? 1 : 2) : (clothMesh ? 3 : 0)));
    }
    if (g.index) {
      // per material group, so hair triangles can be told apart when the mesh is shared
      for (let i = 0; i < g.index.count; i++) idx.push(base + g.index.getX(i));
    } else for (let i = 0; i < P.count; i++) idx.push(base + i);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  const triPart = [], triKind = [];
  for (let t = 0; t < idx.length; t += 3) { triPart.push(part[idx[t]]); triKind.push(Math.max(kind[idx[t]], kind[idx[t + 1]], kind[idx[t + 2]])); }
  // the BVH reorders the index: keep a copy of the triangle labels keyed by the original order
  const bvh = new MeshBVH(geo, { indirect: true });
  geo.computeBoundingBox();
  return { geo, pos: new Float32Array(pos), part, kind, triPart, triKind, bvh, box: geo.boundingBox };
}

const KIND_MAX = (a, b) => (a === 3 && b === 0) || (a === 0 && b === 3) ? 3 : Math.max(a, b);
export function solid(k) { return k === 0 || k === 3; }  // skin or worn clothes: what bodies are made of

const _t = { point: new THREE.Vector3(), distance: 0, faceIndex: 0 };
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3(), _n = new THREE.Vector3();

// Signed distance from p to the baked surface (negative = inside), with the face's part label.
export function signed(B, p, maxD = 0.06) {
  const hit = B.bvh.closestPointToPoint(p, _t, 0, maxD);
  if (!hit) return null;
  const I = B.geo.index, P = B.geo.getAttribute("position");
  const f = hit.faceIndex;
  const ia = I.getX(f * 3), ib = I.getX(f * 3 + 1), ic = I.getX(f * 3 + 2);
  _a.fromBufferAttribute(P, ia); _b.fromBufferAttribute(P, ib); _c.fromBufferAttribute(P, ic);
  _n.subVectors(_c, _b).cross(_a.clone().sub(_b)).normalize();
  const s = p.clone().sub(hit.point).dot(_n) < 0 ? -1 : 1;
  return { d: s * hit.distance, point: hit.point.clone(), normal: _n.clone(), part: B.part[ia], kind: KIND_MAX(KIND_MAX(B.kind[ia], B.kind[ib]), B.kind[ic]) };
}

function bump(map, key, depth, p, n) {
  const e = map.get(key) || { n: 0, depth: 0, at: null, normal: null, pts: [] };
  e.n++;
  if (e.pts.length < 400) e.pts.push(p.x, p.y, p.z);
  if (depth > e.depth) { e.depth = depth; e.at = p.toArray(); e.normal = n ? n.toArray() : null; }
  map.set(key, e);
}

function spread(e) {  // size of the contact patch: how far the overlapping vertices reach from the deepest one
  if (!e.at) return 0;
  const d = [];
  for (let i = 0; i < e.pts.length; i += 3) d.push(Math.hypot(e.pts[i] - e.at[0], e.pts[i + 1] - e.at[1], e.pts[i + 2] - e.at[2]));
  d.sort((a, b) => a - b);
  return d.length ? d[Math.floor(d.length * 0.85)] : 0;
}

// Interpenetration and contact between two baked actors. A vertex of A counts as clipping when it is
// more than `tol` inside B's surface; the deepest point per part pair is kept (where the fix goes).
export function pairReport(A, B, tol = 0.004) {
  const box = B.box.clone().expandByScalar(0.06);
  const pen = new Map(), touch = new Map();
  const p = new THREE.Vector3();
  for (let i = 0; i < A.part.length; i++) {
    p.set(A.pos[i * 3], A.pos[i * 3 + 1], A.pos[i * 3 + 2]);
    if (!box.containsPoint(p)) continue;
    const s = signed(B, p, 0.05);
    if (!s) continue;
    const ka = A.kind[i] === 1 ? "hair" : A.kind[i] === 2 ? "cloth" : A.kind[i] === 3 ? "wear" : "";
    const kb = s.kind === 1 ? "hair" : s.kind === 2 ? "cloth" : s.kind === 3 ? "wear" : "";
    const key = `${ka ? ka + ":" : ""}${A.part[i]}>${kb ? kb + ":" : ""}${s.part}`;
    if (s.d < -tol) bump(pen, key, -s.d, p, s.normal);
    else if (s.d < 0.012) bump(touch, key, 0.012 - s.d, p, s.normal);
  }
  const out = (m) => [...m.entries()].map(([k, e]) => ({ pair: k, n: e.n, depth: +e.depth.toFixed(4), at: e.at, normal: e.normal,
    spread: +spread(e).toFixed(4) }))
    .sort((x, y) => y.depth * Math.sqrt(y.n) - x.depth * Math.sqrt(x.n));
  return { clip: out(pen).filter((e) => e.n >= 3), touch: out(touch) };
}

// Vertices below the floor, per part.
export function floorReport(A, floorY = 0, tol = 0.006) {
  const m = new Map();
  for (let i = 0; i < A.part.length; i++) {
    const y = A.pos[i * 3 + 1];
    if (y < floorY - tol) bump(m, (A.kind[i] === 1 ? "hair:" : A.kind[i] === 2 ? "cloth:" : A.kind[i] === 3 ? "wear:" : "") + A.part[i], floorY - y,
      new THREE.Vector3(A.pos[i * 3], y, A.pos[i * 3 + 2]));
  }
  return [...m.entries()].map(([k, e]) => ({ part: k, n: e.n, depth: +e.depth.toFixed(4), at: e.at })).filter((e) => e.n >= 3);
}

// Arms (and hair/cloth) through the actor's own torso, head and legs.
export function selfReport(A, tol = 0.006) {
  const I = A.geo.index;
  const core = [];
  for (let t = 0; t < I.count; t += 3) {
    const pa = A.part[I.getX(t)];
    if (!ARM_PARTS.has(pa) && solid(A.kind[I.getX(t)]) && pa !== "head") core.push(I.getX(t), I.getX(t + 1), I.getX(t + 2));
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", A.geo.getAttribute("position"));
  g.setIndex(core);
  const C = { geo: g, part: A.part, kind: A.kind, bvh: new MeshBVH(g, { indirect: true }) };
  const m = new Map(), p = new THREE.Vector3();
  for (let i = 0; i < A.part.length; i++) {
    const pa = A.part[i];
    if (!(pa.endsWith("forearm") || pa.endsWith("hand")) || !solid(A.kind[i])) continue;
    p.set(A.pos[i * 3], A.pos[i * 3 + 1], A.pos[i * 3 + 2]);
    const s = signed(C, p, 0.05);
    if (s && s.d < -tol) bump(m, `${pa}>${s.part}`, -s.d, p, s.normal);
  }
  return [...m.entries()].map(([k, e]) => ({ pair: k, n: e.n, depth: +e.depth.toFixed(4), at: e.at, normal: e.normal,
    spread: +spread(e).toFixed(4) })).filter((e) => e.n >= 4);
}

// ----------------------------------------------------------------------------- spring colliders
// Capsules fitted to the mesh in its rest pose: for every body part, the radius that encloses ~70% of
// its (non-hair, non-cloth) vertices around the bone segment, so colliders sit on the skin.
export function fitColliders(entry) {
  if (entry.colliders) return entry.colliders;
  const { vrm } = entry;
  vrm.humanoid.resetNormalizedPose();
  vrm.update(0);
  entry.scene.updateMatrixWorld(true);
  const B = bake(entry, []);
  const N = (b) => vrm.humanoid.getRawBoneNode(b);
  const out = [];
  for (const [name, a, b] of PARTS) {
    const ends = segEnds(vrm, a, b);
    if (!ends) continue;
    const [na, pa, pb] = ends;
    const ds = [];
    const seg = new THREE.Line3(pa, pb || pa), q = new THREE.Vector3(), p = new THREE.Vector3();
    for (let i = 0; i < B.part.length; i++) {
      if (B.part[i] !== name || !solid(B.kind[i])) continue;
      p.set(B.pos[i * 3], B.pos[i * 3 + 1], B.pos[i * 3 + 2]);
      if (seg.distanceSq() > 1e-10) seg.closestPointToPoint(p, true, q); else q.copy(seg.start);
      ds.push(p.distanceTo(q));
    }
    if (ds.length < 8) continue;
    ds.sort((x, y) => x - y);
    let r = ds[Math.floor(ds.length * 0.7)];
    if (name.endsWith("hand")) r = Math.min(r, 0.035);
    if (name === "head") r = ds[Math.floor(ds.length * 0.5)];
    const inv = new THREE.Matrix4().copy(na.matrixWorld).invert();
    out.push({ part: name, bone: a, radius: r, offset: pa.clone().applyMatrix4(inv), tail: pb ? pb.clone().applyMatrix4(inv) : null });
  }
  entry.colliders = out;
  return out;
}

function colliderObjects(entry, filter) {
  const { vrm } = entry;
  const cols = [];
  for (const c of fitColliders(entry)) {
    if (filter && !filter(c.part)) continue;
    const node = vrm.humanoid.getRawBoneNode(c.bone);
    const shape = c.tail
      ? new VRMSpringBoneColliderShapeCapsule({ offset: c.offset.clone(), tail: c.tail.clone(), radius: c.radius })
      : new VRMSpringBoneColliderShapeSphere({ offset: c.offset.clone(), radius: c.radius });
    const col = new VRMSpringBoneCollider(shape);
    node.add(col);
    cols.push(col);
  }
  return cols;
}

// Give every spring joint (hair, skirt, ribbons) the bodies of all actors and the floor to collide
// with. The model's own head is left out (hair roots start inside it); its own torso, arms and legs
// are in, which keeps long hair off the shoulders and skirts outside the thighs.
export function linkSprings(entries, floorY = 0) {
  const floor = new VRMSpringBoneCollider(new VRMSpringBoneColliderShapePlane({ offset: new THREE.Vector3(0, floorY + 0.004, 0), normal: new THREE.Vector3(0, 1, 0) }));
  floor.updateMatrixWorld(true);
  const groups = entries.map((e) => {
    (e.extraColliders || []).forEach((c) => c.parent && c.parent.remove(c));
    const own = colliderObjects(e, (p) => p !== "head" && p !== "neck");
    const all = colliderObjects(e, null);
    e.extraColliders = [...own, ...all];
    return { own: { name: "own-body", colliders: own }, all: { name: "body", colliders: all } };
  });
  const floorGroup = { name: "floor", colliders: [floor] };
  entries.forEach((e, i) => {
    const sm = e.vrm.springBoneManager;
    if (!sm) return;
    if (!e.baseGroups) e.baseGroups = new Map([...sm.joints].map((j) => [j, [...j.colliderGroups]]));
    for (const j of sm.joints) {
      const others = groups.filter((_, k) => k !== i).map((g) => g.all);
      j.colliderGroups = [...e.baseGroups.get(j), groups[i].own, ...others, floorGroup];
    }
  });
  return floor;
}


// Soft tissue gives where something presses into it: every overlap whose receiving surface is soft
// (belly, chest, hips, thighs, upper arms, clothing) becomes a dent sized to the contact patch, so the
// flesh is pushed in by the limb instead of the limb passing through it. Rigid-on-rigid overlaps are
// left for the pose solver (they are reported, not hidden).
export function contactDents(rep, maxDepth = 0.07) {
  const out = rep.self.map(() => []);
  const add = (who, e) => {
    const recv = e.pair.split(">")[1];
    const part = recv.replace(/^(cloth|hair|wear):/, "");
    if (!(SOFT.has(part) || recv.startsWith("cloth:")) || e.depth > maxDepth || !e.normal) return;
    const n = new THREE.Vector3(...e.normal).normalize();
    const at = new THREE.Vector3(...e.at).addScaledVector(n, e.depth);  // the receiving skin at the contact
    out[who].push({ pos: at.toArray(), dir: n.clone().negate().toArray(), depth: e.depth + 0.003, bulge: 0.12,
      radius: Math.min(0.09, Math.max(0.022, e.spread * 1.25 + e.depth)) });
  };
  for (const p of rep.pairs) for (const e of p.clip) add(p.b, e);
  rep.self.forEach((sl, i) => sl.forEach((e) => add(i, e)));
  return out.map((l) => l.sort((a, b) => b.depth * b.radius - a.depth * a.radius));
}
