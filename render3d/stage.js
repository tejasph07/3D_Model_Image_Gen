// Browser side of the 3D renderer: loads VRM models, poses them from solved stage joints
// (direction retargeting + exact two-bone IK on the model's own limbs), and renders with MToon.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

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

async function loadVRM(url, key = url) {
  if (cache.has(key)) return cache.get(key);
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);  // every model faces +Z
  vrm.scene.traverse((o) => { o.frustumCulled = false; });
  vrm.scene.updateMatrixWorld(true);
  // rest pose in world space (normalized bones: identity local rotation at rest)
  const rest = {};
  for (const name of Object.keys(vrm.humanoid.humanBones)) {
    const n = vrm.humanoid.getNormalizedBoneNode(name);
    if (!n) continue;
    rest[name] = { pos: n.getWorldPosition(new THREE.Vector3()), quat: n.getWorldQuaternion(new THREE.Quaternion()) };
  }
  const entry = { vrm, rest, scene: vrm.scene };
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
      if (isArm) {
        setWorldQuat(ne, deltaRotation(restDir(end), restH, td, h).multiply(rest[end].quat));
      } else {
        const shinUp = j[jm].clone().sub(j[je]);
        setWorldQuat(ne, deltaRotation(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0),
          td, shinUp).multiply(rest[end].quat));
      }
    }
  };
  limb("l", "leftUpperLeg", "leftLowerLeg", "leftFoot", "l_hip", "l_knee", "l_ankle", "l_toe", false);
  limb("r", "rightUpperLeg", "rightLowerLeg", "rightFoot", "r_hip", "r_knee", "r_ankle", "r_toe", false);
  limb("l", "leftUpperArm", "leftLowerArm", "leftHand", "l_shoulder", "l_elbow", "l_wrist", "l_hand", true);
  limb("r", "rightUpperArm", "rightLowerArm", "rightHand", "r_shoulder", "r_elbow", "r_wrist", "r_hand", true);

  // fingers: fist or open hand
  for (const side of ["left", "right"]) {
    const fist = opts.fist ? opts.fist[side[0]] : true;
    const s = side === "left" ? -1 : 1;
    for (const f of ["Index", "Middle", "Ring", "Little"]) {
      for (const seg of ["Proximal", "Intermediate", "Distal"]) {
        const n = N(`${side}${f}${seg}`);
        if (n) { n.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), s * (fist ? 1.45 : 0.12)); }
      }
    }
    for (const seg of ["Metacarpal", "Proximal", "Distal"]) {
      const n = N(`${side}Thumb${seg}`);
      if (n) n.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -s * (fist ? 0.6 : 0.1));
    }
  }
  // expression + gaze
  const em = vrm.expressionManager;
  if (em) {
    for (const k of ["happy", "angry", "sad", "relaxed", "surprised", "aa", "ih", "ou", "ee", "oh", "blink"]) em.setValue(k, 0);
    for (const [k, w] of Object.entries(opts.expression || {})) em.setValue(k, w);
  }
  if (vrm.lookAt) {
    if (opts.lookAt) { const t = new THREE.Object3D(); t.position.copy(V(opts.lookAt)); vrm.lookAt.target = t; }
    else vrm.lookAt.target = null;
  }
  vrm.scene.updateMatrixWorld(true);
}

function settleSprings(entry, motion, frames = 45) {
  // let hair/cloth springs settle; the body arrives along its motion so hair trails behind it
  const { vrm } = entry;
  const hips = vrm.humanoid.getNormalizedBoneNode("hips");
  const end = hips.position.clone();
  const m = motion ? V(motion) : new THREE.Vector3();
  const back = hips.parent.worldToLocal(hips.getWorldPosition(new THREE.Vector3()).sub(m.clone().multiplyScalar(0.35)));
  if (vrm.springBoneManager) vrm.springBoneManager.reset();
  for (let i = 0; i < frames; i++) {
    const t = Math.min(1, i / (frames * 0.6));
    hips.position.lerpVectors(back, end, t);
    vrm.update(1 / 30);
  }
  hips.position.copy(end);
  vrm.update(1 / 60);
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
    pose(entry, a.joints, a);
    if (job.springs !== false) settleSprings(entry, a.motion);
    entry.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    scene.add(entry.scene);
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
  for (const e of entries) scene.remove(e.scene);
  return out;
};

function isInside(o, root) { for (let p = o; p; p = p.parent) if (p === root) return true; return false; }

window.inspectModel = async (url) => {
  const { rest } = await loadVRM(url);
  const out = {};
  for (const k in rest) out[k] = rest[k].pos.toArray();
  return out;
};
window.ready = true;
