#!/bin/bash
# tools/anim_sheet.sh blend clip frames view physics loop out.png
set -e
B=~/.local/share/Steam/steamapps/common/Blender/blender
tmp=$(mktemp -d)
$B -b "$1" -P "$(dirname "$0")/anim_views.py" -- "$tmp/a" "$2" "$3" "$4" "$5" "$6" 2>&1 | grep -E "FINISHED|CANCELLED|Error|rror|Traceback" || true
files=()
for f in ${3//,/ }; do files+=("$(printf '%s/a_%03d.png' "$tmp" "$f")"); done
magick "${files[@]}" +append -resize 70% "$7"
rm -rf "$tmp"
