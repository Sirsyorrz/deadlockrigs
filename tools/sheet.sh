#!/bin/bash
# [HERO=violet] sheet.sh out.png keys_csv angle focus : render keys and tile them
set -e
cd "$(dirname "$0")/.."
B=${BLENDER:-blender}
tmp=$(mktemp -d)
$B -b -P tools/render_keys.py -- heroes/${HERO:=violet}/build/$HERO.blend "$tmp" "$2" "${3:-0}" "${4:-face}" > /dev/null 2>&1
files=("$tmp/neutral.png")
IFS=',' read -ra ks <<< "$2"
for k in "${ks[@]}"; do files+=("$tmp/$k.png"); done
montage -label '%t' "${files[@]}" -tile ${TILE:-4}x -geometry ${GEOM:-320}x${GEOM:-320}+2+2 -pointsize 14 "$1"
rm -rf "$tmp"
