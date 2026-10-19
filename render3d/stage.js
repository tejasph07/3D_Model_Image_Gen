// Browser side of the 3D renderer: loads VRM models, poses them from solved stage joints
// (direction retargeting + exact two-bone IK on the model's own limbs), and renders with MToon.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { bake, pairReport, floorReport, selfReport, fitColliders, linkSprings, contactDents } from "./physics.js";

const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));
const cache = new Map();

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const CHAIN = [
  // bone, our joint at its start, our joint at its end (direction), secondary-axis rule
  ["hips", "pelvis", "spine", "hips"],
  ["spine", "spine", "chest", "shoulders"],
  ["chest", "chest", "neck", "shoulders"],
  ["upperChest", null, null, "inherit"],
  ["neck", "neck", "head", "shoulders"],
  ["head", "head", "head_top", "face"],
  ["leftUpperLeg", "l_hip", "l_knee", "l_leg"], ["leftLowerLeg", "l_knee", "l_ankle", "l_leg"],
  ["leftFoot", "l_ankle", "l_toe", "l_foot"],
  ["rightUpperLeg", "r_hip", "r_knee", "r_leg"], ["rightLowerLeg", "r_knee", "r_ankle", "r_leg"],
  ["rightFoot", "r_ankle", "r_toe", "r_foot"],
  ["leftUpperArm", "l_shoulder", "l_elbow", "l_arm"], ["leftLowerArm", "l_elbow", "l_wrist", "l_arm"],
  ["leftHand", "l_wrist", "l_hand", "l_arm"],
  ["rightUpperArm", "r_shoulder", "r_elbow", "r_arm"], ["rightLowerArm", "r_elbow", "r_wrist", "r_arm"],
  ["rightHand", "r_wrist", "r_hand", "r_arm"],
];
const CHILD = { hips: "spine", spine: "chest", chest: "upperChest", upperChest: "neck", neck: "head",
  leftUpperLeg: "leftLowerLeg", leftLowerLeg: "leftFoot", leftFoot: "leftToes",
  rightUpperLeg: "rightLowerLeg", rightLowerLeg: "rightFoot", rightFoot: "rightToes",
  leftUpperArm: "leftLowerArm", leftLowerArm: "leftHand", leftHand: "leftMiddleProximal",
  rightUpperArm: "rightLowerArm", rightLowerArm: "rightHand", rightHand: "rightMiddleProximal" };

// Soft-body contacts: up to MAX_DENTS presses per mesh, applied after skinning. Each dent pushes the
// surface along `dir` by `depth` with a smooth falloff over `radius`, raises a small bulge ring
// around it (displaced flesh), and tilts the shading normals so the dent reads in toon shading.
// Negative depth pulls the surface (a pinched ear, a grabbed cheek).
const MAX_DENTS = 12;
const DENT_GLSL = `
#ifdef SOFT_DENTS
  for (int i = 0; i < ${MAX_DENTS}; i++) {
    if (i >= dentCount) break;
    vec3 rel = transformed - dentPos[i];
    vec3 dir = dentDir[i];
    float along = dot(rel, dir);
    vec3 lat = rel - dir * along;
    float x = length(lat) / dentRad[i];
    float depthFall = 1.0 - smoothstep(dentRad[i] * 0.6, dentRad[i] * 1.8, abs(along));
    float facing = smoothstep(-0.35, 0.25, dot(normalize(objectNormal), -dir));
    float w = depthFall * facing * dentMask;
    float f = x < 1.0 ? (1.0 - x * x) * (1.0 - x * x) : 0.0;
    float bulge = smoothstep(0.75, 1.05, x) * (1.0 - smoothstep(1.05, 1.7, x));
    transformed += dir * dentDepth[i] * (f - dentBulge[i] * bulge) * w;
    #if !defined(OUTLINE) && !defined(FLAT_SHADED)
      vec3 radial = length(lat) > 1e-6 ? lat / length(lat) : vec3(0.0);
      float slope = x < 1.0 ? dentDepth[i] / dentRad[i] * 4.0 * x * (1.0 - x * x) : 0.0;
      vNormal = normalize(vNormal - normalMatrix * radial * min(slope * 2.5, 0.55) * w);
    #endif
  }
#endif
`;


// ---------------------------------------------------------------- injuries
// An injury lives at a point of the mesh's bind (rest) pose, so it stays on exactly the same patch of skin
// in every pose, from every camera. Pattern = procedural noise from a seed: the same wound every time.
// kinds: 0 bruise, 1 cut, 2 bleeding cut, 3 scar, 4 bandage, 5 swelling. age 0 fresh .. 1 healed.
const MAX_INJ = 12;
const INJ_PARS = `
uniform int injCount;
uniform vec3 injPos[${MAX_INJ}];
uniform vec3 injDir[${MAX_INJ}];
uniform vec3 injNrm[${MAX_INJ}];
uniform vec3 injDown[${MAX_INJ}];
uniform float injRad[${MAX_INJ}];
uniform float injKind[${MAX_INJ}];
uniform float injSeed[${MAX_INJ}];
uniform float injAge[${MAX_INJ}];
uniform float injSkin;
varying vec3 vBindPos;
`;
const INJ_FRAG = INJ_PARS + `
float ihash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float inoise(vec3 x) {
  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(ihash(i), ihash(i + vec3(1, 0, 0)), f.x), mix(ihash(i + vec3(0, 1, 0)), ihash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(ihash(i + vec3(0, 0, 1)), ihash(i + vec3(1, 0, 1)), f.x), mix(ihash(i + vec3(0, 1, 1)), ihash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float ifbm(vec3 p) { return 0.55 * inoise(p) + 0.3 * inoise(p * 2.1) + 0.15 * inoise(p * 4.3); }
vec4 injOver(vec4 acc, vec3 c, float a) { a = clamp(a, 0.0, 1.0); return vec4(mix(acc.rgb, c, a), acc.a + a * (1.0 - acc.a)); }
vec4 injuryColor(vec3 p) {
  vec4 acc = vec4(0.0);
  for (int i = 0; i < ${MAX_INJ}; i++) {
    if (i >= injCount) break;
    vec3 rel = p - injPos[i];
    float r = injRad[i], age = injAge[i], k = injKind[i];
    if (abs(dot(rel, injNrm[i])) > r * 0.6) continue;  // only the near surface, not the other side of a limb
    float n = ifbm(rel / r * 3.0 + injSeed[i] * 7.13);
    float dd = length(rel) / r;
    vec3 dir = injDir[i];
    float along = dot(rel, dir);
    vec3 side3 = rel - dir * along - injNrm[i] * dot(rel, injNrm[i]);
    float perp = length(side3);
    if (k < 0.5) {  // bruise: dark purple-red core, red rim; yellow-green as it heals
      if (injSkin < 0.5) continue;
      float m = smoothstep(1.0, 0.3, dd + (n - 0.5) * 0.8);
      vec3 fresh = mix(vec3(0.36, 0.10, 0.28), vec3(0.66, 0.20, 0.26), smoothstep(0.0, 0.9, dd + (n - 0.5) * 0.3));
      vec3 old = mix(vec3(0.50, 0.42, 0.22), vec3(0.62, 0.60, 0.34), dd);
      acc = injOver(acc, mix(fresh, old, age), m * 0.8 * (1.0 - 0.55 * age));
    } else if (k < 2.5) {  // cut (and bleeding cut): a tapered dark-red slit, reddened lips, blood running down
      float len = r * (0.85 + 0.3 * inoise(vec3(injSeed[i] * 3.1)));
      float t = clamp(abs(along) / len, 0.0, 1.0);
      float w = r * 0.07 * (1.0 - t * t) * (1.0 - 0.6 * age);
      float jag = (inoise(vec3(along / r * 9.0 + injSeed[i], 0.0, 0.0)) - 0.5) * r * 0.05;
      float inside = step(abs(along), len);
      float slit = inside * smoothstep(w, w * 0.35, perp + jag);
      float lips = inside * smoothstep(w * 4.0, w, perp) * (1.0 - age);
      if (injSkin > 0.5) acc = injOver(acc, vec3(0.80, 0.33, 0.33), lips * 0.45);
      acc = injOver(acc, mix(vec3(0.42, 0.02, 0.05), vec3(0.85, 0.62, 0.60), age), slit);
      if (k > 1.5 && age < 0.6) {  // blood runs down from the cut (toward the ground in this frame)
        vec3 dn = injDown[i] - injNrm[i] * dot(injDown[i], injNrm[i]);  // downhill along the skin
        if (dot(dn, dn) < 0.04) continue;  // skin facing straight up or down: blood pools, no runs
        dn = normalize(dn);
        float below = dot(rel, dn);
        float lc = dot(rel, normalize(cross(injNrm[i], dn)));  // across the runs
        float u = lc / r * 2.2 + injSeed[i];
        float col = inoise(vec3(floor(u) * 1.7 + injSeed[i], 1.7, 0.0));
        float drip = step(abs(lc), r * 0.9) * step(0.0, below) * step(below, r * (0.3 + 1.7 * col));
        float edge = smoothstep(0.22, 0.08, abs(fract(u) - 0.5)) * (0.6 + 0.4 * step(below, r * (0.25 + 1.5 * col)));
        acc = injOver(acc, vec3(0.50, 0.02, 0.04), drip * edge * (1.0 - age / 0.6));
      }
    } else if (k < 3.5) {  // scar: pale raised line with stitch marks
      if (injSkin < 0.5) continue;
      float t = abs(along) / r;
      float m = step(t, 1.0) * smoothstep(r * 0.05, r * 0.02, perp);
      float stitch = step(t, 0.9) * smoothstep(0.08, 0.0, abs(fract(along / r * 4.0) - 0.5)) * smoothstep(r * 0.16, r * 0.1, perp);
      acc = injOver(acc, vec3(0.93, 0.78, 0.76), m * 0.85);
      acc = injOver(acc, vec3(0.70, 0.45, 0.45), stitch * 0.6 * (1.0 - age));
    } else if (k < 4.5) {  // bandage / plaster: a white pad with cross-weave, a blood spot through it
      if (injSkin < 0.5) continue;
      float sd = dot(rel, normalize(cross(injNrm[i], dir)));
      float m = step(abs(along), r) * step(abs(sd), r * 0.55);
      float weave = 0.92 + 0.08 * step(0.5, fract(along / r * 18.0)) * step(0.5, fract(sd / r * 18.0));
      acc = injOver(acc, vec3(0.96, 0.94, 0.90) * weave, m);
      float spot = smoothstep(0.45, 0.2, length(vec2(along, sd)) / r + (n - 0.5) * 0.4) * (1.0 - age);
      acc = injOver(acc, vec3(0.62, 0.10, 0.12), m * spot * 0.8);
    } else if (injSkin > 0.5) {  // swelling: hot red skin (the bump itself is in the vertex shader)
      float m = smoothstep(1.0, 0.2, dd + (n - 0.5) * 0.4);
      acc = injOver(acc, vec3(0.86, 0.42, 0.42), m * 0.45 * (1.0 - age));
    }
  }
  return acc;
}
`;
const INJ_VERT = `
  vBindPos = transformed;
  for (int i = 0; i < ${MAX_INJ}; i++) {
    if (i >= injCount) break;
    if (injKind[i] > 4.5 && injSkin > 0.5) {  // swelling: the skin rises
      float dd = length(transformed - injPos[i]) / injRad[i];
      float f = dd < 1.0 ? (1.0 - dd * dd) * (1.0 - dd * dd) : 0.0;
      transformed += normalize(normal) * injRad[i] * 0.22 * f * (1.0 - injAge[i]);
    }
  }
`;

function injuryMaterial(m) {  // add the injury uniforms and shader code to one material (once)
  if (m.userData.injU) return m.userData.injU;
  const U = {
    injCount: { value: 0 },
    injPos: { value: Array.from({ length: MAX_INJ }, () => new THREE.Vector3()) },
    injDir: { value: Array.from({ length: MAX_INJ }, () => new THREE.Vector3(1, 0, 0)) },
    injNrm: { value: Array.from({ length: MAX_INJ }, () => new THREE.Vector3(0, 0, 1)) },
    injDown: { value: Array.from({ length: MAX_INJ }, () => new THREE.Vector3(0, -1, 0)) },
    injRad: { value: new Array(MAX_INJ).fill(0.03) },
    injKind: { value: new Array(MAX_INJ).fill(0) },
    injSeed: { value: new Array(MAX_INJ).fill(0) },
    injAge: { value: new Array(MAX_INJ).fill(0) },
    injSkin: { value: /SKIN/i.test(m.name) ? 1 : 0 },
  };
  m.userData.injU = U;
  const prev = m.onBeforeCompile, prevKey = m.customProgramCacheKey;
  m.onBeforeCompile = (sh, r) => {
    if (prev) prev.call(m, sh, r);
    Object.assign(sh.uniforms, U);
    sh.vertexShader = sh.vertexShader.replace("void main() {", INJ_PARS + "\nvoid main() {")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\n" + INJ_VERT);
    sh.fragmentShader = sh.fragmentShader.replace("\nvoid main() {", "\n" + INJ_FRAG + "\nvoid main() {");
    if (sh.fragmentShader.includes("// #include <color_fragment>")) {  // MToon
      sh.fragmentShader = sh.fragmentShader
        .replace("// #include <color_fragment>", "vec4 injC = injuryColor(vBindPos);\n  diffuseColor.rgb = mix(diffuseColor.rgb, injC.rgb, injC.a);\n  // #include <color_fragment>")
        .replace("material.shadingShift = shadingShiftFactor;", "material.shadeColor = mix(material.shadeColor, injC.rgb * 0.62, injC.a);\n  material.shadingShift = shadingShiftFactor;");
    } else {
      sh.fragmentShader = sh.fragmentShader.replace("#include <color_fragment>",
        "#include <color_fragment>\n  vec4 injC = injuryColor(vBindPos);\n  diffuseColor.rgb = mix(diffuseColor.rgb, injC.rgb, injC.a);");
    }
  };
  m.customProgramCacheKey = () => (prevKey ? prevKey.call(m) : "") + "|inj";
  m.needsUpdate = true;
  return U;
}

const INJ_KINDS = { bruise: 0, cut: 1, bleeding_cut: 2, scar: 3, bandage: 4, swelling: 5 };

// An injury comes as {bone, offset, normal, along, kind, radius, seed, age}: a point on the character's own
// skeleton in the rest pose (bone + offset from it), the outward skin normal there and the cut's direction,
// all in the rest frame. It is found on the skin mesh in the rest (= bind) pose itself, so the same injury
// lands on the same skin point whatever the pose of the frame being drawn. The shader keeps it there.
function resolveInjury(entry, inj) {
  const key = JSON.stringify([inj.bone, inj.offset, inj.normal, inj.along]);
  entry.injCache = entry.injCache || new Map();
  if (entry.injCache.has(key)) return entry.injCache.get(key);
  const R = entry.rest[inj.bone];
  if (!R) return null;
  const n = V(inj.normal).normalize();
  const pt = R.pos.clone().add(V(inj.offset));
  let best = null;
  const tri = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()], hitP = new THREE.Vector3();
  entry.scene.traverse((o) => {
    if (!o.isMesh || !o.userData.restMatrix) return;
    const ms = Array.isArray(o.material) ? o.material : [o.material];
    if (!ms.some((m) => /SKIN/i.test(m.name))) return;
    const inv = o.userData.restMatrix.clone().invert();
    const ray = new THREE.Ray(pt.clone().addScaledVector(n, 0.12), n.clone().negate()).applyMatrix4(inv);
    ray.direction.normalize();
    const g = o.geometry, P = g.getAttribute("position"), I = g.index;
    const cnt = I ? I.count : P.count;
    for (let t = 0; t < cnt; t += 3) {
      const ia = I ? I.getX(t) : t, ib = I ? I.getX(t + 1) : t + 1, ic = I ? I.getX(t + 2) : t + 2;
      tri[0].fromBufferAttribute(P, ia); tri[1].fromBufferAttribute(P, ib); tri[2].fromBufferAttribute(P, ic);
      if (!ray.intersectTriangle(tri[0], tri[1], tri[2], false, hitP)) continue;
      const d = hitP.distanceTo(ray.origin);
      if (d > 0.3 || (best && d >= best.d)) continue;
      const nrm = tri[1].clone().sub(tri[0]).cross(tri[2].clone().sub(tri[0])).normalize();
      if (nrm.dot(ray.direction) > 0) nrm.negate();
      best = { d, bind: hitP.clone(), nrm, inv };
    }
  });
  if (!best) { entry.injCache.set(key, null); return null; }
  const tb = V(inj.along || [1, 0, 0]).transformDirection(best.inv);
  tb.addScaledVector(best.nrm, -tb.dot(best.nrm));
  if (tb.lengthSq() < 1e-10) tb.set(1, 0, 0);
  const res = { bind: best.bind, nrm: best.nrm, dir: tb.normalize(), inv: best.inv };
  entry.injCache.set(key, res);
  return res;
}

function setInjuries(entry, list) {
  const mats = [];
  entry.scene.traverse((o) => {
    if (!o.isMesh) return;
    (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { if (/SKIN|CLOTH/i.test(m.name)) mats.push(injuryMaterial(m)); });
  });
  const out = [];
  for (const inj of list || []) {
    const r = resolveInjury(entry, inj);
    if (r) out.push({ ...r, down: V(inj.down || [0, -1, 0]).transformDirection(r.inv), kind: INJ_KINDS[inj.kind] ?? 0, radius: inj.radius || 0.03, seed: inj.seed || 0, age: inj.age || 0 });
  }
  for (const U of mats) {
    U.injCount.value = Math.min(out.length, MAX_INJ);
    out.slice(0, MAX_INJ).forEach((j, i) => {
      U.injPos.value[i].copy(j.bind); U.injNrm.value[i].copy(j.nrm); U.injDir.value[i].copy(j.dir); U.injDown.value[i].copy(j.down);
      U.injRad.value[i] = j.radius; U.injKind.value[i] = j.kind; U.injSeed.value[i] = j.seed; U.injAge.value[i] = j.age;
    });
  }
  return out.map((j) => ({ bind: j.bind.toArray(), kind: j.kind }));
}

const ARM_BONES = ["Shoulder", "UpperArm", "LowerArm", "Hand", "Thumb", "Index", "Middle", "Ring", "Little"];

function dentMaskFor(o, vrm) {  // 1 = flesh that can be pressed in; arms/hands are rigid here
  const g = o.geometry;
  if (g.getAttribute("dentMask")) return;
  const n = g.getAttribute("position").count;
  const mask = new Float32Array(n).fill(1);
  const si = g.getAttribute("skinIndex"), sw = g.getAttribute("skinWeight");
  if (o.isSkinnedMesh && si && sw && vrm) {
    const arm = new Set();
    for (const name of Object.keys(vrm.humanoid.humanBones)) {
      if (ARM_BONES.some((k) => name.includes(k))) {
        const b = vrm.humanoid.getRawBoneNode(name);
        if (b) arm.add(b);
      }
    }
    const isArm = o.skeleton.bones.map((b) => {  // a bone under an arm bone (e.g. sleeve springs) counts too
      for (let p = b; p; p = p.parent) if (arm.has(p)) return true;
      return false;
    });
    for (let i = 0; i < n; i++) {
      let a = 0;
      for (let k = 0; k < 4; k++) if (isArm[si.getComponent(i, k)]) a += sw.getComponent(i, k);
      mask[i] = Math.max(0, 1 - a * 1.5);
    }
  }
  g.setAttribute("dentMask", new THREE.BufferAttribute(mask, 1));
}

function softBody(root, vrm) {  // install the dent hook on every material of a model instance
  root.traverse((o) => {
    if (!o.isMesh) return;
    dentMaskFor(o, vrm);
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    mats.forEach((m) => {
      if (m.userData.dentU) return;
      const U = {
        dentCount: { value: 0 },
        dentPos: { value: Array.from({ length: MAX_DENTS }, () => new THREE.Vector3()) },
        dentDir: { value: Array.from({ length: MAX_DENTS }, () => new THREE.Vector3(0, 0, 1)) },
        dentRad: { value: new Array(MAX_DENTS).fill(0.05) },
        dentDepth: { value: new Array(MAX_DENTS).fill(0) },
        dentBulge: { value: new Array(MAX_DENTS).fill(0.32) },
      };
      m.userData.dentU = U;
      const prev = m.onBeforeCompile, prevKey = m.customProgramCacheKey;
      m.onBeforeCompile = (sh, r) => {
        if (prev) prev.call(m, sh, r);
        Object.assign(sh.uniforms, U);
        sh.vertexShader = sh.vertexShader
          .replace("void main() {", `#define SOFT_DENTS\nattribute float dentMask;\nuniform int dentCount;\nuniform vec3 dentPos[${MAX_DENTS}];
uniform vec3 dentDir[${MAX_DENTS}];\nuniform float dentRad[${MAX_DENTS}];\nuniform float dentDepth[${MAX_DENTS}];\nuniform float dentBulge[${MAX_DENTS}];
void main() {`)
          .replace("#include <skinning_vertex>", "#include <skinning_vertex>\n" + DENT_GLSL);
      };
      m.customProgramCacheKey = () => (prevKey ? prevKey.call(m) : "") + "|dents";
      m.needsUpdate = true;
    });
  });
}

// A press is given as the attacker's effector surface point and push direction. Find the victim's
// actual (skinned) surface under it by raycasting, and dent exactly as deep as the effector went in.
function measureDents(entry, presses) {
  const out = [];
  const ray = new THREE.Raycaster();
  entry.scene.updateMatrixWorld(true);
  const meshes = [];
  entry.scene.traverse((o) => { if (o.isMesh && o.visible && !/hair/i.test(o.name + (o.parent ? o.parent.name : ""))) meshes.push(o); });
  for (const p of presses || []) {
    const pos = V(p.pos), dir = V(p.dir).normalize();
    const back = p.reach || 0.25;
    ray.set(pos.clone().addScaledVector(dir, -back), dir);
    ray.far = back + 0.25;
    const hit = ray.intersectObjects(meshes, false)[0];
    if (!hit) continue;
    const pen = pos.clone().sub(hit.point).dot(dir);  // > 0: the effector is inside the surface
    if (pen < -0.01 && p.depth >= 0) continue;  // not touching
    const depth = p.depth < 0 ? p.depth : Math.min(Math.max(pen, 0) + (p.extra ?? 0.006), p.maxDepth || 0.08);
    out.push({ pos: hit.point.toArray(), dir: dir.toArray(), depth, radius: p.radius });
  }
  return out;
}

function setDents(entry, dents) {  // dents in world space -> each mesh's local (post-skinning) space
  entry.scene.updateMatrixWorld(true);
  entry.scene.traverse((o) => {
    if (!o.isMesh) return;
    const inv = new THREE.Matrix4().copy(o.matrixWorld).invert();
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    mats.forEach((m) => {
      const U = m.userData.dentU;
      if (!U) return;
      const list = (dents || []).slice(0, MAX_DENTS);
      U.dentCount.value = list.length;
      list.forEach((d, i) => {
        U.dentPos.value[i].copy(V(d.pos)).applyMatrix4(inv);
        U.dentDir.value[i].copy(V(d.dir)).transformDirection(inv);
        const s = new THREE.Vector3(1, 0, 0).applyMatrix4(inv).sub(new THREE.Vector3(0, 0, 0).applyMatrix4(inv)).length();
        U.dentRad.value[i] = d.radius * s;
        U.dentDepth.value[i] = d.depth * s;
        U.dentBulge.value[i] = d.bulge ?? 0.32;
      });
    });
  });
}

async function loadVRM(url, key = url) {
  if (cache.has(key)) return cache.get(key);
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);  // every model faces +Z
  vrm.scene.traverse((o) => { o.frustumCulled = false; });
  softBody(vrm.scene, vrm);
  vrm.scene.updateMatrixWorld(true);
  vrm.scene.traverse((o) => { if (o.isMesh) o.userData.restMatrix = o.matrixWorld.clone(); });  // bind pose = rest pose
  const entry0 = { vrm, scene: vrm.scene };
  fitColliders(entry0);  // body capsules from the rest-pose mesh (hair/cloth collide with them)
  // rest pose in world space (normalized bones: identity local rotation at rest)
  const rest = {};
  for (const name of Object.keys(vrm.humanoid.humanBones)) {
    const n = vrm.humanoid.getNormalizedBoneNode(name);
    if (!n) continue;
    rest[name] = { pos: n.getWorldPosition(new THREE.Vector3()), quat: n.getWorldQuaternion(new THREE.Quaternion()) };
  }
  const entry = { vrm, rest, scene: vrm.scene, colliders: entry0.colliders };
  cache.set(key, entry);
  return entry;
}

function basis(dir, sec) {  // orthonormal frame from a primary direction and a secondary hint
  const x = dir.clone().normalize();
  let z = sec.clone().sub(x.clone().multiplyScalar(sec.dot(x)));
  if (z.lengthSq() < 1e-8) z = Math.abs(x.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  z.normalize();
  const y = new THREE.Vector3().crossVectors(z, x);
  return new THREE.Matrix4().makeBasis(x, y, z);
}

function deltaRotation(restDir, restSec, dir, sec) {
  const A = basis(restDir, restSec), B = basis(dir, sec);
  const R = B.multiply(A.transpose());
  return new THREE.Quaternion().setFromRotationMatrix(R);
}

function setWorldQuat(node, qWorld) {
  const pq = node.parent.getWorldQuaternion(new THREE.Quaternion());
  node.quaternion.copy(pq.invert().multiply(qWorld));
  node.updateMatrixWorld(true);
}

function twoBone(a, target, la, lb, pole) {  // returns [mid, end] positions
  const d = target.clone().sub(a);
  let dist = d.length();
  const dir = d.normalize();
  dist = Math.min(Math.max(dist, Math.abs(la - lb) + 1e-4), la + lb - 1e-5);
  const cosA = (la * la + dist * dist - lb * lb) / (2 * la * dist);
  const ang = Math.acos(Math.max(-1, Math.min(1, cosA)));
  let p = pole.clone().sub(a);
  p.sub(dir.clone().multiplyScalar(p.dot(dir)));
  if (p.lengthSq() < 1e-10) p = new THREE.Vector3(0, 0, 1).sub(dir.clone().multiplyScalar(dir.z));
  p.normalize();
  const mid = a.clone().add(dir.clone().multiplyScalar(la * Math.cos(ang))).add(p.multiplyScalar(la * Math.sin(ang)));
  return [mid, a.clone().add(dir.multiplyScalar(dist))];
}

function pose(entry, J, opts) {
  const { vrm } = entry;
  const sc = opts.scale || 1;
  vrm.scene.scale.setScalar(sc);
  const rest = {};
  for (const k in entry.rest) rest[k] = { pos: entry.rest[k].pos.clone().multiplyScalar(sc), quat: entry.rest[k].quat };
  const N = (b) => vrm.humanoid.getNormalizedBoneNode(b);
  vrm.humanoid.resetNormalizedPose();
  vrm.scene.updateMatrixWorld(true);
  const j = {};
  for (const k in J) j[k] = V(J[k]);
  const right = j.r_shoulder.clone().sub(j.l_shoulder);   // character's right (world)
  const hipRight = j.r_hip.clone().sub(j.l_hip);
  const fwdBody = new THREE.Vector3().crossVectors(j.neck.clone().sub(j.pelvis), right).negate();
  const face = j.nose.clone().sub(j.head);
  const restRight = new THREE.Vector3(-1, 0, 0);  // model faces +Z, so its right is -X
  const hinge = (u, l, fallback) => {
    const h = new THREE.Vector3().crossVectors(u, l);
    return h.lengthSq() > 1e-6 ? h.normalize() : fallback;
  };
  const restDir = (b) => {
    const c = CHILD[b];
    if (c && rest[c]) return rest[c].pos.clone().sub(rest[b].pos);
    if (b.endsWith("Hand")) return b.startsWith("left") ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(-1, 0, 0);
    if (b.endsWith("Foot")) return new THREE.Vector3(0, -0.3, 1);
    return new THREE.Vector3(0, 1, 0);
  };
  // hips: place so the leg roots coincide with the solved hip joints
  const hipsQ = deltaRotation(new THREE.Vector3(0, 1, 0), restRight, j.spine.clone().sub(j.pelvis), hipRight);
  const restLegMid = rest.leftUpperLeg.pos.clone().add(rest.rightUpperLeg.pos).multiplyScalar(0.5);
  const legMid = j.l_hip.clone().add(j.r_hip).multiplyScalar(0.5);
  const hipsPos = legMid.add(rest.hips.pos.clone().sub(restLegMid).applyQuaternion(hipsQ));
  const hipsNode = N("hips");
  hipsNode.position.copy(hipsNode.parent.worldToLocal(hipsPos.clone()));
  setWorldQuat(hipsNode, hipsQ.clone().multiply(rest.hips.quat));

  const sec = {
    shoulders: () => [restRight, right],
    face: () => [new THREE.Vector3(0, 0, 1), face],
  };
  for (const [bone, a, b, rule] of CHAIN.slice(1, 6)) {
    const node = N(bone);
    if (!node || !rest[bone]) continue;
    if (rule === "inherit") { node.quaternion.identity(); node.updateMatrixWorld(true); continue; }
    const [rs, ts] = sec[rule]();
    const rd = bone === "head" ? new THREE.Vector3(0, 1, 0) : restDir(bone);
    const q = deltaRotation(rd, rs, j[b].clone().sub(j[a]), ts);
    setWorldQuat(node, q.multiply(rest[bone].quat));
  }
  // limbs: exact two-bone IK on the model's own segments, hinge from the solved pose
  const limb = (side, up, low, end, jr, jm, je, jt, isArm) => {
    const nu = N(up), nl = N(low), ne = N(end);
    if (!nu || !nl) return;
    const root = nu.getWorldPosition(new THREE.Vector3());
    const la = rest[low].pos.distanceTo(rest[up].pos);
    const lb = rest[end].pos.distanceTo(rest[low].pos);
    const [mid, tip] = twoBone(root, j[je], la, lb, j[jm]);
    const ud = mid.clone().sub(root), ld = tip.clone().sub(mid);
    const restU = restDir(up), restL = restDir(low);
    let restH, fall;
    if (isArm) {
      restH = side === "l" ? new THREE.Vector3(0, -1, 0) : new THREE.Vector3(0, 1, 0);
      // straight arm: keep the elbow's hinge as it is at rest relative to the body
      fall = new THREE.Vector3().crossVectors(ud, fwdBody).normalize();
      if (side === "l") fall.negate();
    } else {
      restH = new THREE.Vector3(1, 0, 0);
      fall = hipRight.clone().normalize().negate();
    }
    const h = hinge(ud, ld, fall);
    setWorldQuat(nu, deltaRotation(restU, restH, ud, h).multiply(rest[up].quat));
    setWorldQuat(nl, deltaRotation(restL, restH, ld, h).multiply(rest[low].quat));
    if (ne && rest[end]) {
      const td = j[jt].clone().sub(j[je]);
      if (isArm && j[`${side}_palm`]) {  // explicit palm direction: rest palms face down in the T-pose
        const pm = j[`${side}_palm`].clone().sub(j[je]);
        setWorldQuat(ne, deltaRotation(restDir(end), new THREE.Vector3(0, -1, 0), td, pm).multiply(rest[end].quat));
      } else if (isArm) {
        setWorldQuat(ne, deltaRotation(restDir(end), restH, td, h).multiply(rest[end].quat));
      } else {
        const fu = j[`${side}_foot_up`];
        const shinUp = fu ? fu.clone().sub(j[je]) : j[jm].clone().sub(j[je]);
        setWorldQuat(ne, deltaRotation(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0),
          td, shinUp).multiply(rest[end].quat));
      }
    }
  };
  limb("l", "leftUpperLeg", "leftLowerLeg", "leftFoot", "l_hip", "l_knee", "l_ankle", "l_toe", false);
  limb("r", "rightUpperLeg", "rightLowerLeg", "rightFoot", "r_hip", "r_knee", "r_ankle", "r_toe", false);
  // clavicles: the shoulder rises as the arm lifts and slides forward as it reaches (a raised arm
  // on a fixed shoulder is what makes posed models look like mannequins)
  const torsoUp = j.neck.clone().sub(j.pelvis).normalize();
  const fwdN = fwdBody.clone().normalize();
  for (const [side, bone] of [["l", "leftShoulder"], ["r", "rightShoulder"]]) {
    const sn = N(bone);
    if (!sn || !rest[bone]) continue;
    const ud = j[`${side}_elbow`].clone().sub(j[`${side}_shoulder`]).normalize();
    const elev = Math.asin(Math.max(-1, Math.min(1, ud.dot(torsoUp))));
    const raise = Math.max(0, Math.min(0.4, (elev + 0.35) * 0.26));
    const prot = Math.max(-0.12, Math.min(0.24, ud.dot(fwdN) * 0.22));
    const lat = j[`${side}_shoulder`].clone().sub(j.neck);
    lat.sub(torsoUp.clone().multiplyScalar(lat.dot(torsoUp))).normalize();
    const qr = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3().crossVectors(lat, torsoUp).normalize(), raise);
    const qp = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3().crossVectors(lat, fwdN).normalize(), prot);
    setWorldQuat(sn, qr.multiply(qp).multiply(sn.getWorldQuaternion(new THREE.Quaternion())));
  }
  limb("l", "leftUpperArm", "leftLowerArm", "leftHand", "l_shoulder", "l_elbow", "l_wrist", "l_hand", true);
  limb("r", "rightUpperArm", "rightLowerArm", "rightHand", "r_shoulder", "r_elbow", "r_wrist", "r_hand", true);

  handsAndFace(entry, opts);
}

// Pose straight from a physics simulation: every humanoid bone's world rotation relative to its rest
// orientation (the simulator builds its bodies in the model's own rest frame) and the hips position.
const BONE_ORDER = ["hips", "spine", "chest", "upperChest", "neck", "head",
  "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand", "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
  "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes", "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes"];
function poseBones(entry, opts) {
  const { vrm } = entry;
  vrm.scene.scale.setScalar(opts.scale || 1);
  vrm.humanoid.resetNormalizedPose();
  vrm.scene.updateMatrixWorld(true);
  const N = (b) => vrm.humanoid.getNormalizedBoneNode(b);
  const hips = N("hips");
  hips.position.copy(hips.parent.worldToLocal(V(opts.root)));
  hips.updateMatrixWorld(true);
  for (const b of BONE_ORDER) {
    const n = N(b);
    if (!n || !entry.rest[b]) continue;
    const q = opts.bones[b];
    if (!q) continue;  // a bone the simulation doesn't drive rides on its parent
    setWorldQuat(n, new THREE.Quaternion(q[0], q[1], q[2], q[3]).multiply(entry.rest[b].quat));
  }
  handsAndFace(entry, opts);
}

function handsAndFace(entry, opts) {
  const { vrm } = entry;
  const N = (b) => vrm.humanoid.getNormalizedBoneNode(b);
  // fingers: hand shapes (fist | relaxed | flat | grip), curl per segment, outer fingers curl more
  const SHAPES = {
    fist: { f: [1.45, 1.45, 1.3], spread: 0, thumb: [0.6, 0.6, 0.6] },
    relaxed: { f: [0.32, 0.5, 0.36], spread: 0.1, thumb: [0.2, 0.25, 0.25] },
    flat: { f: [0.04, 0.06, 0.04], spread: 0.0, thumb: [0.12, 0.1, 0.05] },
    grip: { f: [1.0, 1.1, 0.8], spread: 0, thumb: [0.5, 0.45, 0.4] },
    claw: { f: [0.55, 1.0, 0.8], spread: 0.15, thumb: [0.35, 0.4, 0.4] },
    // thumb and index meet (pinching an ear, a cheek); the other fingers tucked
    pinch: { f: [1.25, 1.3, 1.0], index: [0.75, 0.85, 0.55], spread: 0, thumb: [0.75, 0.55, 0.45] },
  };
  for (const side of ["left", "right"]) {
    let shape = opts.fist ? opts.fist[side[0]] : true;
    if (shape === true || shape === undefined) shape = "fist";
    if (shape === false) shape = "relaxed";
    const S = SHAPES[shape] || SHAPES.relaxed;
    const s = side === "left" ? -1 : 1;
    ["Index", "Middle", "Ring", "Little"].forEach((f, fi) => {
      const extra = S === SHAPES.relaxed ? fi * 0.08 : 0;
      ["Proximal", "Intermediate", "Distal"].forEach((seg, si) => {
        const n = N(`${side}${f}${seg}`);
        if (!n) return;
        const curl = (f === "Index" && S.index) ? S.index[si] : S.f[si] + extra;
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), s * curl);
        if (si === 0 && S.spread) q.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), s * S.spread * (fi - 1.2) * 0.5));
        n.quaternion.copy(q);
      });
    });
    ["Metacarpal", "Proximal", "Distal"].forEach((seg, si) => {
      const n = N(`${side}Thumb${seg}`);
      if (n) n.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -s * S.thumb[si]);
    });
  }
  // expression + gaze
  const em = vrm.expressionManager;
  if (em) {
    for (const k of ["happy", "angry", "sad", "relaxed", "surprised", "aa", "ih", "ou", "ee", "oh", "blink", "blinkLeft", "blinkRight"]) em.setValue(k, 0);
    for (const [k, w] of Object.entries(opts.expression || {})) em.setValue(k, w);
  }
  if (vrm.lookAt) {
    if (opts.lookAt) { const t = new THREE.Object3D(); t.position.copy(V(opts.lookAt)); vrm.lookAt.target = t; }
    else vrm.lookAt.target = null;
  }
  vrm.scene.updateMatrixWorld(true);
}

function settleSprings(entries, motions, frames = 60) {
  // hair/cloth springs settle with every body (and the floor) in place; each body arrives along its
  // motion so hair trails behind it
  const st = entries.map((e, k) => {
    const hips = e.vrm.humanoid.getNormalizedBoneNode("hips");
    const end = hips.position.clone();
    const m = motions[k] ? V(motions[k]) : new THREE.Vector3();
    const back = hips.parent.worldToLocal(hips.getWorldPosition(new THREE.Vector3()).sub(m.clone().multiplyScalar(0.35)));
    if (e.vrm.springBoneManager) e.vrm.springBoneManager.reset();
    return { e, hips, end, back };
  });
  for (let i = 0; i < frames; i++) {
    const t = Math.min(1, i / (frames * 0.5));
    for (const s of st) { s.hips.position.lerpVectors(s.back, s.end, t); s.e.scene.updateMatrixWorld(true); }
    for (const s of st) s.e.vrm.update(1 / 60);
  }
  for (const s of st) { s.hips.position.copy(s.end); s.e.scene.updateMatrixWorld(true); }
  for (let i = 0; i < 20; i++) for (const s of st) s.e.vrm.update(1 / 60);  // come to rest at the final pose
}

function gradientSky(top, bottom) {
  const c = document.createElement("canvas"); c.width = 2; c.height = 256;
  const g = c.getContext("2d"), grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, top); grd.addColorStop(1, bottom); g.fillStyle = grd; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

async function environment(scene, env) {
  const sky = env.sky || ["#1a1e2a", "#3a3f4c"];
  scene.background = gradientSky(sky[0], sky[1]);
  if (env.fog) scene.fog = new THREE.Fog(env.fog[0], env.fog[1], env.fog[2]);
  for (const p of env.prims || []) {
    const mat = new THREE.MeshStandardMaterial({ color: p.color || "#777", roughness: p.rough ?? 0.85, metalness: 0.0 });
    if (p.emissive) { mat.emissive = new THREE.Color(p.emissive); mat.emissiveIntensity = p.emissiveIntensity || 1; }
    let m;
    if (p.type === "box") m = new THREE.Mesh(new THREE.BoxGeometry(...p.size), mat);
    else if (p.type === "plane") { m = new THREE.Mesh(new THREE.PlaneGeometry(p.size[0], p.size[1]), mat); }
    else if (p.type === "cylinder") m = new THREE.Mesh(new THREE.CylinderGeometry(p.size[0], p.size[0], p.size[1], 16), mat);
    else if (p.type === "rope") {
      const a = V(p.a), b = V(p.b);
      m = new THREE.Mesh(new THREE.CylinderGeometry(p.radius, p.radius, a.distanceTo(b), 8), mat);
      m.position.copy(a.clone().add(b).multiplyScalar(0.5));
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
      scene.add(m); continue;
    } else if (p.type === "gltf") {
      const g = await loader.loadAsync(p.url); m = g.scene;
      if (p.scale) m.scale.setScalar(p.scale);
    }
    if (!m) continue;
    m.receiveShadow = true;
    m.position.set(...p.pos);
    if (p.type === "plane") { m.rotation.set(-Math.PI / 2, 0, p.rotY || 0, "YXZ"); m.rotation.order = "YXZ"; m.rotation.y = p.rotY || 0; m.rotation.x = -Math.PI / 2; }
    else if (p.rotY) m.rotation.y = p.rotY;
    scene.add(m);
  }
  for (const l of env.lights || [{ type: "ambient", color: "#fff", intensity: 1.4 }, { type: "dir", color: "#fff", intensity: 2, pos: [-2, 5, 3] }]) {
    let L;
    if (l.type === "ambient") L = new THREE.AmbientLight(l.color, l.intensity);
    else if (l.type === "dir") {
      L = new THREE.DirectionalLight(l.color, l.intensity); L.position.set(...l.pos);
      if (!scene.userData.shadowLight) {  // first directional light casts the shadows
        scene.userData.shadowLight = L; L.castShadow = true;
        L.shadow.mapSize.set(2048, 2048); L.shadow.bias = -0.0005; L.shadow.normalBias = 0.02;
        L.shadow.radius = 3;
      }
    }
    else if (l.type === "point") { L = new THREE.PointLight(l.color, l.intensity, l.distance || 0, 1.5); L.position.set(...l.pos); }
    else if (l.type === "spot") {
      L = new THREE.SpotLight(l.color, l.intensity, 0, l.angle || 0.5, 0.4, 1.2); L.position.set(...l.pos);
      L.target.position.set(...l.target); scene.add(L.target);
    }
    if (L) scene.add(L);
  }
}

window.renderJob = async (job) => {
  const scene = new THREE.Scene();
  await environment(scene, job.environment || {});
  scene.userData.shadowLight = scene.userData.shadowLight || null;
  const seen = {}, entries = [];
  for (const a of job.actors) {
    seen[a.model] = (seen[a.model] || 0) + 1;  // the same model twice in a scene = two instances
    const entry = await loadVRM(a.model, `${a.model}#${seen[a.model]}`);
    entries.push(entry);
    if (a.rest) { entry.vrm.humanoid.resetNormalizedPose(); entry.vrm.scene.scale.setScalar(1); handsAndFace(entry, a); }
    else if (a.bones) poseBones(entry, a);
    else pose(entry, a.joints, a);
    entry.presses = a.dents;
    entry.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    scene.add(entry.scene);
  }
  if (job.springs !== false) {
    linkSprings(entries, 0);
    settleSprings(entries, job.actors.map((a) => a.motion));
  } else for (const e of entries) e.vrm.update(0);  // normalized pose -> the raw skeleton the mesh uses
  entries.forEach((e, i) => { e.lastInjuries = setInjuries(e, job.actors[i].injuries); });
  for (const e of entries) { e.lastDents = measureDents(e, e.presses); setDents(e, e.lastDents); }
  if (job.softContacts !== false && entries.length) {  // flesh gives where bodies press into it
    const auto = contactDents(contacts(entries), 0.03);
    entries.forEach((e, i) => {
      e.autoDents = auto[i].slice(0, MAX_DENTS - e.lastDents.length);
      e.lastDents = e.lastDents.concat(e.autoDents);
      setDents(e, e.lastDents);
    });
  }
  const sl = scene.userData.shadowLight;
  if (sl) {  // fit the shadow camera around the actors, light direction kept
    const box = new THREE.Box3(); entries.forEach((e) => box.expandByObject(e.scene));
    const ctr = box.getCenter(new THREE.Vector3()), rad = box.getSize(new THREE.Vector3()).length() * 0.75 + 1;
    const dir = sl.position.clone().normalize();
    sl.position.copy(ctr.clone().add(dir.multiplyScalar(rad * 2))); sl.target.position.copy(ctr); scene.add(sl.target);
    Object.assign(sl.shadow.camera, { left: -rad, right: rad, top: rad, bottom: -rad, near: 0.1, far: rad * 4 });
    sl.shadow.camera.updateProjectionMatrix();
  }
  const c = job.camera;
  const cam = new THREE.PerspectiveCamera(c.fov, job.width / job.height, 0.05, 200);
  cam.position.copy(V(c.eye)); cam.up.copy(V(c.up));
  cam.lookAt(V(c.eye).add(V(c.fwd)));
  renderer.setPixelRatio(1);
  const ss = job.supersample || 1;  // render big, the driver downsamples: clean anti-aliased lines
  renderer.setSize(job.width * ss, job.height * ss, false);
  renderer.render(scene, cam);
  const out = { beauty: renderer.domElement.toDataURL("image/png") };
  // where the mesh's bones actually ended up (raw skeleton), for checking against the solver
  out.bones = entries.map((e) => {
    const r = { _dents: e.lastDents, _presses: e.presses, _injuries: e.lastInjuries };
    for (const b of ["hips", "head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm",
      "rightHand", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]) {
      const n = e.vrm.humanoid.getRawBoneNode(b);
      if (n) r[b] = n.getWorldPosition(new THREE.Vector3()).toArray();
    }
    return r;
  });
  if (job.passes && job.passes.includes("depth")) {
    // depth of the real meshes, near = white, scaled to the actors' own depth range
    const box = new THREE.Box3(); entries.forEach((e) => box.expandByObject(e.scene));
    const corners = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z])
      corners.push(new THREE.Vector3(x, y, z).applyMatrix4(cam.matrixWorldInverse).z);
    const near = Math.max(0.05, -Math.max(...corners) - 0.2), far = -Math.min(...corners) + 0.8;
    const dcam = cam.clone(); dcam.near = near; dcam.far = far; dcam.updateProjectionMatrix();
    const bg = scene.background, fog = scene.fog;
    scene.background = new THREE.Color(0); scene.fog = null;
    scene.overrideMaterial = new THREE.MeshDepthMaterial();
    renderer.render(scene, dcam);
    out.depth = renderer.domElement.toDataURL("image/png");
    scene.overrideMaterial = null; scene.background = bg; scene.fog = fog;
  }
  if (job.passes && job.passes.includes("ids")) {
    // each actor in a flat colour (exact silhouettes, occlusion included), set hidden
    const saved = [];
    const hidden = [];
    scene.traverse((o) => { if (o.isMesh && !entries.some((e) => isInside(o, e.scene))) { hidden.push([o, o.visible]); o.visible = false; } });
    entries.forEach((e, i) => {
      const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(`hsl(${(i * 67) % 360}, 100%, 50%)`) });
      e.scene.traverse((o) => { if (o.isMesh) { saved.push([o, o.material]); o.material = mat; } });
    });
    const bg = scene.background, fog = scene.fog;
    scene.background = new THREE.Color(0); scene.fog = null;
    renderer.render(scene, cam);
    out.ids = renderer.domElement.toDataURL("image/png");
    saved.forEach(([o, m]) => { o.material = m; });
    hidden.forEach(([o, v]) => { o.visible = v; });
    scene.background = bg; scene.fog = fog;
  }
  if (job.contacts !== false) out.contacts = contacts(entries, job.dumpVerts);
  for (const e of entries) scene.remove(e.scene);
  return out;
};

function contacts(entries, dump) {
  const baked = entries.map((e) => bake(e, e.lastDents));
  const r = { pairs: [], floor: [], self: [] };
  if (dump) r.verts = baked.map((B) => {  // a sample of the skin (body/clothes only), for checking proxies
    const o = [];
    for (let i = 0; i < B.part.length; i += dump) if (B.kind[i] === 0 || B.kind[i] === 3) o.push([B.part[i], B.pos[i * 3], B.pos[i * 3 + 1], B.pos[i * 3 + 2], B.kind[i]]);
    return o;
  });
  baked.forEach((A, i) => {
    r.floor.push(floorReport(A, 0));
    r.self.push(selfReport(A));
    baked.forEach((B, k) => { if (k !== i) r.pairs.push({ a: i, b: k, ...pairReport(A, B) }); });
  });
  return r;
}

function isInside(o, root) { for (let p = o; p; p = p.parent) if (p === root) return true; return false; }

window.inspectModel = async (url) => {
  const { rest } = await loadVRM(url);
  const out = {};
  for (const k in rest) out[k] = rest[k].pos.toArray();
  return out;
};
window.ready = true;
