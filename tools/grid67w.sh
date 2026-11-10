#!/bin/bash
# On 06, an arm's HOOK while a server answers at 127.0.0.1:8095: grid67w.sh LABEL -- grid.py on that server at 262,144
# tokens only, 3 reps a cell: l2-chain.sh's uncounted first start (the first start of a chain ran fastest whichever
# binary it was, DECISIONS 11-09 12:1x), which also tries grid.py and its token list on 06 before the counted starts.
# Lines, cells and output as grid67.sh's. 3450 s at most, inside the arm's 3600 s for a hook (grid.py stops at 3300).
set -u
L=${1:?label}
T=$(cd "$(dirname "$0")" && pwd)
B=$T/bench
echo "$(date '+%F %T') == grid67w $L begin: grid.py $(sha256sum "$T/grid.py" | cut -c1-16)" >> "$B/$L.log"
timeout 3450 python3 "$T/grid.py" http://127.0.0.1:8095 "$L" "$B" --contexts 262144 >> "$B/$L.out" 2>&1
rc=$?
[ $rc = 0 ] || echo "$(date '+%F %T')   grid.py rc $rc" >> "$B/$L.log"
