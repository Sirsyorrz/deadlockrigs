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
# Releases (up to 20.1) don't decompile the vmdl's cloth/physics nodes, so build a pinned master commit.
S2V_COMMIT=${S2V_COMMIT:-f030654ff81b352fd1bcb44d631c563f51c7dcd1}
V=${S2V:-$ROOT/s2v/bin/Source2Viewer-CLI}
PAK=${PAK:-~/.local/share/Steam/steamapps/common/Deadlock/game/citadel/pak01_dir.vpk}

field() { python3 -c "import json,sys; h=[h for h in json.load(open('heroes.json')) if h['slug']=='$1']; print(h[0]['$2'] if h else sys.exit('unknown hero $1'))"; }
need_s2v() {
  [ -x "$V" ] && return
  [ -n "$S2V" ] && { echo "S2V=$S2V is not executable"; exit 1; }
  echo "building Source2Viewer-CLI ${S2V_COMMIT:0:9}"
  local src=$ROOT/s2v/src
  [ -d "$src/.git" ] || git clone -q --filter=blob:none https://github.com/ValveResourceFormat/ValveResourceFormat "$src"
  git -C "$src" fetch -q origin "$S2V_COMMIT"
  git -C "$src" checkout -q --detach "$S2V_COMMIT"
  dotnet publish "$src/CLI/CLI.csproj" -c Release --self-contained -r linux-x64 -p:RunAnalyzersDuringBuild=false \
    -o "$(dirname "$V")" > "$ROOT/s2v/build.log" 2>&1 || { echo "build failed, see s2v/build.log"; exit 1; }
}

# prints the face builder: keys.py (from the head's geometry), morphs.py (game morphs) or bonekeys.py (face bones)
has_face() { python3 -c "import json; h=[h for h in json.load(open('heroes.json')) if h['slug']=='$1'][0]; f=h.get('face'); print(f.get('builder', 'keys.py') if f else '')"; }

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
        need_s2v
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
        need_s2v
        $V -i $PAK -f "${model}_c" -b PHYS > "$dir/data/phys.txt" 2>&1
        python3 tools/fe_extract.py "$dir/data/phys.txt" "$dir/data/cloth.json"
        rm -f "$dir/data/phys.txt"
        local builder; builder=$(has_face "$hero")
        if [ "$builder" = morphs.py ] || [ "$builder" = bonekeys.py ]; then
          $B -b -P "tools/$builder" -- "$hero" 2>&1 | grep -E "^(morph|bone) keys|Error|Traceback" || true
        elif [ -n "$builder" ]; then
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
