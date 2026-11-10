#!/bin/bash
# On 04, an arm's HOOK while a server answers at 127.0.0.1:8095: grid2s.sh LABEL -- grid.py on that server at 4,096 and
# 65,536 tokens only, 6 reps a cell (the retake of T2's short cells, DECISIONS 11-09 17:5x). Its lines go to
# bench/LABEL.log (the last "grid LABEL end" or "grid LABEL STOP: ..."), its cells to bench/LABEL-grid.json, anything it
# prints to bench/LABEL.out. 3450 s at most, inside the arm's 3600 s for a hook (grid.py stops at 3300).
set -u
L=${1:?label}
T=$HOME/LLLLM/glm-work/tp4
B=$T/bench
echo "$(date '+%F %T') == grid2s $L begin: grid.py $(sha256sum "$T/grid.py" | cut -c1-16)" >> "$B/$L.log"
timeout 3450 python3 "$T/grid.py" http://127.0.0.1:8095 "$L" "$B" --contexts 4096,65536 --reps 6 >> "$B/$L.out" 2>&1
rc=$?
[ $rc = 0 ] || echo "$(date '+%F %T')   grid.py rc $rc" >> "$B/$L.log"
