#!/bin/bash
# ./make.sh <hero|all> [step...]     steps: extract data base anims rig test (default: all of them)
# Builds heroes/<hero>/out/<hero>_rig.blend (+ <hero>_anims.blend, <hero>.blend/.glb/.fbx) from the game files.
# KEEP=1 keeps the animation glb and the decompiled dmx files (large, only needed to rebuild).
# OLD=4.4 builds with older-versions/blender-4.4*/ into heroes/<hero>/out-4.4/ (5.x .blend files don't open before 4.5).
set -eo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
B=${BLENDER:-~/.local/share/Steam/steamapps/common/Blender/blender}
OUT=out
if [ -n "$OLD" ]; then
  B=$(ls -d "$ROOT"/older-versions/blender-"$OLD"*/blender 2>/dev/null | sort -V | tail -1)
  [ -x "$B" ] || { echo "no Blender $OLD in older-versions/"; exit 1; }
  OUT=out-$OLD
fi
export DLRIG_OUT=$OUT
V=${S2V:-~/Documents/GitHub/deadlock-forge/tools/vrf/Source2Viewer-CLI}
PAK=${PAK:-~/.local/share/Steam/steamapps/common/Deadlock/game/citadel/pak01_dir.vpk}

field() { python3 -c "import json,sys; h=[h for h in json.load(open('heroes.json')) if h['slug']=='$1']; print(h[0]['$2'] if h else sys.exit('unknown hero $1'))"; }
has_face() { python3 -c "import json; h=[h for h in json.load(open('heroes.json')) if h['slug']=='$1'][0]; print(1 if h.get('face') else '')"; }

build() {
  local hero=$1; shift
  local steps=${*:-extract data base anims rig test}
  local model; model=$(field "$hero" model)
  local dir=$ROOT/heroes/$hero
  local stem=${model%.vmdl}
  mkdir -p "$dir"/{raw,anim,dec,data,$OUT}
  echo "== $hero ($model): $steps"
  for step in $steps; do
    case $step in
      extract)
        [ -f "$dir/raw/$stem.glb" ] || $V -i $PAK -f "${model}_c" -o "$dir/raw" --gltf_export_format glb \
          --gltf_export_materials --gltf_textures_adapt --gltf_export_extras > "$dir/extract.log" 2>&1
        [ -f "$dir/anim/$stem.glb" ] || [ -f "$dir/$OUT/${hero}_anims.blend" ] || $V -i $PAK -f "${model}_c" \
          -o "$dir/anim" --gltf_export_format glb --gltf_export_animations --gltf_compose_additive >> "$dir/extract.log" 2>&1
        if [ ! -f "$dir/dec/$model" ]; then
          $V -i $PAK -f "${model}_c" -o "$dir/dec" -d >> "$dir/extract.log" 2>&1
          [ -n "$KEEP" ] || find "$dir/dec" -type f ! -name '*.vmdl' -delete
        fi
        ;;
      data)
        python3 tools/vmdl_extract.py "$dir/dec/$model" "$dir/data/gamedata.json" > /dev/null
        python3 tools/gamerig.py "$dir/data/gamedata.json" "$dir/raw/$stem.glb" "$dir/data/gamerig.json"
        if [ -n "$(has_face "$hero")" ]; then
          $B -b -P tools/dump.py -- "$hero" 2>&1 | grep -E "^dumped|Error|Traceback" || true
          python3 tools/keys.py "$dir/data/head.npz" "$dir/data/keys.npz"
        fi
        ;;
      base)
        rm -f "$dir/$OUT/$hero.blend"
        $B -b -P tools/build.py -- "$hero" 2>&1 | grep -E "^(meshes|shape keys|wrote|mesh without|collar)|Error|Traceback" || true
        [ -f "$dir/$OUT/$hero.blend" ]
        ;;
      anims)
        if [ -f "$dir/anim/$stem.glb" ]; then
          rm -f "$dir/$OUT/${hero}_anims.blend"
          $B -b -P tools/anims.py -- "$hero" 2>&1 | grep -E "^clips|Error|Traceback" || true
          [ -f "$dir/$OUT/${hero}_anims.blend" ]
          [ -n "$KEEP" ] || rm -rf "$dir/anim"
        fi
        ;;
      rig)
        rm -f "$dir/$OUT/${hero}_rig.blend"
        $B -b "$dir/$OUT/$hero.blend" -P tools/rig.py -- "$hero" 2>&1 \
          | grep -E "^(IK|skipping|WARNING|wrote|twist|mirror)|Error|Traceback" || true
        [ -f "$dir/$OUT/${hero}_rig.blend" ]
        ;;
      test)
        $B -b "$dir/$OUT/${hero}_rig.blend" -P tools/test_rig.py 2>&1 | grep -E "^(PASS|FAIL|clips)|Error|Traceback" \
          | tee "$dir/test${OLD:+-$OLD}.log"
        ;;
      *) echo "unknown step $step"; exit 1 ;;
    esac
  done
}

if [ "$1" = all ]; then
  shift
  for h in $(python3 -c "import json; print(' '.join(h['slug'] for h in json.load(open('heroes.json'))))"); do
    build "$h" "$@" || echo "!! $h failed"
  done
else
  build "$@"
fi
