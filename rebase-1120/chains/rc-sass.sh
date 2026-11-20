#!/bin/bash
# (11-20: from <the local records>/ with the names mb0b17 = main b0b17e8962, h50c66 = #3689 rebased onto
#  #3776 50c6634539, and rc-sass; <a local session>, <the local records>)
# On 03: rc-sass.sh -- the 0091 PR's two builds in tf-work/tp4 (here mb0b17 = main b0b17e8962, h50c66 = 50c6634539; first written 11-17 for e86a0bed1d and a4da5a4eb2,
# <a local session>): every fatbin name found in both builds' caches (src-N/.zig-cache/o and zc-N/o), its SASS by
# cuobjdump -sass in a throwaway tf3-build:20261014 container (no GPU, no network, two CPUs, nice 19) to
# /tmp/rc-sass/N/NAME.sass, then rc-sass.py beside this script compares them function by function into
# /tmp/rc-sass/summary.txt. The last line "== rc-sass finished" or "== rc-sass stopped".
set -u
P=$HOME/LLLLM/models/glm53-nvfp4/tf-work/tp4
O=/tmp/rc-sass
IMG=tf3-build:20261014
D=$(cd "$(dirname "$0")" && pwd)
stop() { echo "STOP: $*"; echo "== rc-sass stopped"; exit 1; }
[ -e "$O" ] && stop "$O exists"
mkdir -p "$O/mb0b17" "$O/h50c66" || stop "mkdir $O"
for n in mb0b17 h50c66; do
  find "$P/src-$n/.zig-cache/o" "$P/zc-$n/o" -name '*.fatbin' -size +1k -printf '%T@ %f %P\n' 2>/dev/null \
    | sed "s#^\([^ ]*\) \([^ ]*\) #\1 \2 #" > "$O/$n.all"
  # %P is relative to the find root; put the root back in front of it
  : > "$O/$n.list"
  for r in "src-$n/.zig-cache/o" "zc-$n/o"; do
    find "$P/$r" -name '*.fatbin' -size +1k -printf "%T@ %f $r/%P\n" 2>/dev/null
  done | sort -n | awk '{last[$2]=$3} END {for (k in last) print k, last[k]}' | sort > "$O/$n.list"
  echo "$n: $(wc -l < "$O/$n.list") fatbin names"
done
join "$O/mb0b17.list" "$O/h50c66.list" > "$O/both.list"
echo "both: $(wc -l < "$O/both.list") names"
awk '{print "mb0b17", $1, $2; print "h50c66", $1, $3}' "$O/both.list" > "$O/run.list"
docker run --rm --network none --cpus 2 --user "$(id -u):$(id -g)" -v "$P:/p:ro" -v "$O:/out" "$IMG" \
  bash -c 'while read -r n name rel; do nice -n 19 /usr/local/cuda/bin/cuobjdump -sass "/p/$rel" > "/out/$n/$name.sass" 2> "/out/$n/$name.err"; done < /out/run.list' \
  || stop "the cuobjdump container failed"
echo "dumps: mb0b17 $(ls "$O/mb0b17" | grep -c '\.sass$'), h50c66 $(ls "$O/h50c66" | grep -c '\.sass$'), errors $(cat "$O"/*/*.err | grep -c .)"
nice -n 19 python3 "$D/rc-sass.py" "$O" > "$O/summary.txt" 2>&1 || stop "rc-sass.py: $(tail -n 2 "$O/summary.txt")"
tail -n 40 "$O/summary.txt"
echo "== rc-sass finished"
