#!/bin/bash
# tools/hand_sheet.sh blend side out.png pose...   renders tools/hand_views.py into one contact sheet
set -e
blend=$1; side=$2; out=$3; shift 3
B=${BLENDER:-blender}
tmp=$(mktemp -d)
$B -b "$blend" -P "$(dirname "$0")/hand_views.py" -- "$tmp/h" "$side" "$@" 2>&1 | grep -E "THUMB|Error|rror" || true
rows=()
for p in "$@"; do
  q=${p// /_}
  magick "$tmp/h_${q}_palm.png" "$tmp/h_${q}_back.png" "$tmp/h_${q}_thumb.png" "$tmp/h_${q}_pinky.png" +append -gravity NorthWest -pointsize 22 -fill white -annotate +8+6 "$p" "$tmp/row_$q.png"
  rows+=("$tmp/row_$q.png")
done
magick "${rows[@]}" -append -resize 60% "$out"
rm -rf "$tmp"
