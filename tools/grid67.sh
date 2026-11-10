#!/bin/bash
# On 06, an arm's HOOK while a server answers at 127.0.0.1:8095: grid67.sh LABEL -- grid.py on that server at 262,144
# and 1,040,384 tokens, 6 reps a cell: #2722's review, ask (3), TP=2's decode round of main against this head (DECISIONS
# 11-10 19:2x). 6 reps as e-chain.sh's short cells: at 3, TP=2's 256K x1 head moved by about main's run-to-run range
# (dz2a, dd2 and dz2b, 11-09). Its lines go to bench/LABEL.log (the last "grid LABEL end" or "grid LABEL STOP: ..."),
# its cells to bench/LABEL-grid.json, anything it prints to bench/LABEL.out. 3450 s at most, inside the arm's 3600 s
# for a hook (grid.py stops at 3300; 04-05's TP=2 grid of all four contexts at 3 reps took 2,264 s, dz2a).
set -u
L=${1:?label}
T=$(cd "$(dirname "$0")" && pwd)
B=$T/bench
echo "$(date '+%F %T') == grid67 $L begin: grid.py $(sha256sum "$T/grid.py" | cut -c1-16)" >> "$B/$L.log"
timeout 3450 python3 "$T/grid.py" http://127.0.0.1:8095 "$L" "$B" --contexts 262144,1040384 --reps 6 >> "$B/$L.out" 2>&1
rc=$?
[ $rc = 0 ] || echo "$(date '+%F %T')   grid.py rc $rc" >> "$B/$L.log"
