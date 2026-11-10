#!/bin/bash
# On 06 (setsid): l2-chain.sh MAIN HEAD -- #2722's review, ask (3) (DECISIONS 11-10 19:2x): TP=2 on 06 (rank 0) and 07
# (rank 1) over their f0 link (tp67-arm.sh), main against 0067's head at 262,144 and 1,040,384 tokens, 1 and 4 streams,
# 6 reps a cell (grid67.sh as the hook), as l-chain.sh ran TP=4's long contexts. One start uncounted first (grid67w.sh,
# 262,144 tokens only: the first start of a chain ran fastest whichever binary it was, DECISIONS 11-09 12:1x), then
# three starts in turn, 60 s apart:
#   lw2   this head (h0067), grid67w.sh (uncounted)
#   lm2a  main (m1b16), grid    lh2  this head (h0067), grid    lm2b  main (m1b16), grid
# Every arm adds TF_GLM_PROMPT_TAIL=1 (grid.py's prefix resumes only with it) and nothing else (TP=2's default, as
# e-chain.sh). No arm sets TF_GLM_DSEL: rank 0's log must not name TF_GLM_DSEL=0's select, and each grid must end on
# the cells it was given, or the chain stops. The arguments are the sha256 (first 16) of bin/m1b16's and bin/h0067's
# tensorfold-native on 06; a mismatch stops before anything starts (tp67-arm.sh checks both hosts'), and an arm that
# does not serve or a grid that does not end stops the chain. Log logs/l2-chain.log; at the end the grids side by side
# (gsum.py), fsum.py's word per cell (gverdict.py), then "l2-chain end"; or "l2-chain STOP: ...". Never edit while it
# runs.
set -u
T=$(cd "$(dirname "$0")" && pwd)
C=$T/logs/l2-chain.log
A=$T/logs/arms.log
P="TF_GLM_PROMPT_TAIL=1"
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
SHORT="262144x1 262144x4"
LONG="262144x1 262144x4 1040384x1 1040384x4"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "l2-chain STOP: $*"; exit 1; }
[ $# = 2 ] || stop "give two sha16s: main's (bin/m1b16) and this head's (bin/h0067) native"
M=$1 H=$2
for pair in "m1b16:$M" "h0067:$H"; do
  b=${pair%%:*} want=${pair#*:}
  got=$(sha256sum "$T/bin/$b/tensorfold-native" 2>/dev/null | cut -c1-16)
  [ "$got" = "$want" ] || stop "bin/$b/tensorfold-native is '$got', not $want"
done
[ -s "$T/bench/grid-ids.json" ] || stop "no bench/grid-ids.json (04's token list)"
for f in tp67-arm.sh sx67.sh rank67.base.env tp2.env grid.py gsum.py gverdict.py fsum.py grid67.sh grid67w.sh; do
  [ -s "$T/$f" ] || stop "no $f"
done
log "l2-chain begin: main $M head $H grid.py $(sha256sum "$T/grid.py" | cut -c1-16)" \
  "ids $(sha256sum "$T/bench/grid-ids.json" | cut -c1-16) tp67-arm.sh $(sha256sum "$T/tp67-arm.sh" | cut -c1-16)"
arm() {   # label, bin, hook
  local label=$1 bin=$2 hook=$3 rc last n
  log "$label begin: tp67-arm.sh $bin serve $P $hook"
  bash "$T/tp67-arm.sh" "$label" "$bin" serve "$P" "$hook"
  rc=$?
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  n=$(grep -acF "$ZLINE" "$T/logs/serve-r0-$label.log" 2>/dev/null)
  log "$label rc=$rc $last; rank 0's TF_GLM_DSEL=0 lines: ${n:-none}"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ "${n:-0}" = 0 ] || stop "$label set no TF_GLM_DSEL, but rank 0's log names TF_GLM_DSEL=0"
}
grid() {   # label, cells: its grid ended on those cells, or the chain stops
  local cells
  grep -qx "grid $1 end" "$T/bench/$1.log" 2>/dev/null || stop "$1 grid: $(tail -n 2 "$T/bench/$1.log" 2>/dev/null | tr '\n' ' ')"
  cells=$(python3 -B -c 'import json, sys; print(" ".join("%dx%d" % (c["context"], c["streams"]) for c in json.load(open(sys.argv[1]))["cells"]))' "$T/bench/$1-grid.json" 2>/dev/null)
  [ "$cells" = "$2" ] || stop "$1 grid's cells are '$cells', not '$2'"
  log "$1 grid end: $(python3 -B "$T/gsum.py" "$T/bench" "$1" | tail -n 1)"
}
arm lw2 h0067 "$T/grid67w.sh"
grid lw2 "$SHORT"
arm lm2a m1b16 "$T/grid67.sh"
grid lm2a "$LONG"
arm lh2 h0067 "$T/grid67.sh"
grid lh2 "$LONG"
arm lm2b m1b16 "$T/grid67.sh"
grid lm2b "$LONG"
python3 -B "$T/gsum.py" "$T/bench" lm2a lh2 lm2b >> "$C" 2>&1
python3 -B "$T/gverdict.py" "$T/bench" lm2a,lm2b lh2 >> "$C" 2>&1
log "l2-chain end"
