#!/usr/bin/env bash
# Download the starter VRM library (VRoid sample characters, mostly CC0) into $ANIMESCENE_ASSETS/vrm.
# Add your own: drop any .vrm (VRoid Studio export, VRoid Hub download you have rights to, ...) in that folder
# and reference it by file name (without .vrm) in a character's "model" field.
set -euo pipefail
DEST="${ANIMESCENE_ASSETS:-/home/user/assets}/vrm"
mkdir -p "$DEST"
B=https://raw.githubusercontent.com/madjin/vrm-samples/master
for f in vroid/stable/AvatarSample_A.vrm vroid/stable/AvatarSample_B.vrm vroid/stable/AvatarSample_C.vrm \
         vroid/beta/Darkness_Shibu.vrm vroid/beta/Sakurada_Fumiriya.vrm vroid/beta/Sendagaya_Shibu.vrm \
         vroid/beta/Sendagaya_Shino.vrm vroid/beta/Victoria_Rubin.vrm vroid/beta/Vita.vrm vroid/beta/Vivi.vrm \
         vroid/beta/HairSample_Male.vrm vroid/fem_vroid.vrm vroid/masc_vroid.vrm Avatar_Orion.vrm; do
  curl -fsSL -o "$DEST/$(basename "$f")" "$B/$f"
done
curl -fsSL -o "$DEST/Seed-san.vrm" https://raw.githubusercontent.com/vrm-c/vrm-specification/master/samples/Seed-san/vrm/Seed-san.vrm
(cd "$(dirname "$0")/../render3d" && npm install --no-audit --no-fund)
echo "VRM library in $DEST"
