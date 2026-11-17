#!/bin/bash
# On 03: sass-0091.sh -- the 0091 PR's two builds in tf-work/tp4 (me86a = main e86a0bed1d, ha4da = a4da5a4eb2; 11-17,
# <a local session>): every fatbin name found in both builds' caches (src-N/.zig-cache/o and zc-N/o), its SASS by
# cuobjdump -sass in a throwaway tf3-build:20261014 container (no GPU, no network, two CPUs, nice 19) to
# /tmp/sass-0091/N/NAME.sass, then sass-0091.py beside this script compares them function by function into
# /tmp/sass-0091/summary.txt. The last line "== sass-0091 finished" or "== sass-0091 stopped".
set -u
P=$HOME/LLLLM/models/glm53-nvfp4/tf-work/tp4
O=/tmp/sass-0091
IMG=tf3-build:20261014
D=$(cd "$(dirname "$0")" && pwd)
stop() { echo "STOP: $*"; echo "== sass-0091 stopped"; exit 1; }
[ -e "$O" ] && stop "$O exists"
mkdir -p "$O/me86a" "$O/ha4da" || stop "mkdir $O"
for n in me86a ha4da; do
  find "$P/src-$n/.zig-cache/o" "$P/zc-$n/o" -name '*.fatbin' -size +1k -printf '%T@ %f %P\n' 2>/dev/null \
    | sed "s#^\([^ ]*\) \([^ ]*\) #\1 \2 #" > "$O/$n.all"
  # %P is relative to the find root; put the root back in front of it
  : > "$O/$n.list"
  for r in "src-$n/.zig-cache/o" "zc-$n/o"; do
    find "$P/$r" -name '*.fatbin' -size +1k -printf "%T@ %f $r/%P\n" 2>/dev/null
  done | sort -n | awk '{last[$2]=$3} END {for (k in last) print k, last[k]}' | sort > "$O/$n.list"
  echo "$n: $(wc -l < "$O/$n.list") fatbin names"
done
join "$O/me86a.list" "$O/ha4da.list" > "$O/both.list"
echo "both: $(wc -l < "$O/both.list") names"
awk '{print "me86a", $1, $2; print "ha4da", $1, $3}' "$O/both.list" > "$O/run.list"
docker run --rm --network none --cpus 2 --user "$(id -u):$(id -g)" -v "$P:/p:ro" -v "$O:/out" "$IMG" \
  bash -c 'while read -r n name rel; do nice -n 19 /usr/local/cuda/bin/cuobjdump -sass "/p/$rel" > "/out/$n/$name.sass" 2> "/out/$n/$name.err"; done < /out/run.list' \
  || stop "the cuobjdump container failed"
echo "dumps: me86a $(ls "$O/me86a" | grep -c '\.sass$'), ha4da $(ls "$O/ha4da" | grep -c '\.sass$'), errors $(cat "$O"/*/*.err | grep -c .)"
nice -n 19 python3 "$D/sass-0091.py" "$O" > "$O/summary.txt" 2>&1 || stop "sass-0091.py: $(tail -n 2 "$O/summary.txt")"
tail -n 40 "$O/summary.txt"
echo "== sass-0091 finished"
