#!/bin/bash
# On 04 (setsid): e-chain.sh D -- the retake of 0067's T2 at TP=2's short cells (DECISIONS 11-09 17:5x): d8 at TP=2 on
# 04-05 on its default (split), 4,096 and 65,536 tokens at 1 and 4 streams, 6 reps a cell (grid2s.sh as the hook). One
# uncounted start first (bench4.sh, as in d-chain.sh), then five starts in turn, 60 s apart:
#   ew2   d8  default, bench4.sh (uncounted)
#   ez2a  TF_GLM_DSEL=0    ed2a  default    ez2b  =0    ed2b  default    ez2c  =0    (grid2s.sh)
# Every arm adds TF_GLM_PROMPT_TAIL=1. After each arm, rank 0's log must name TF_GLM_DSEL=0's select exactly when the
# arm set it, and after each grid its cells must be the four short ones, or the chain stops. The argument is the sha256
# (first 16) of bin/d8's tensorfold-native on 04; a mismatch stops before anything starts, and an arm that does not
# serve or a grid that does not end stops the chain. Log logs/e-chain.log; last line "e-chain end" or
# "e-chain STOP: ...". Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
C=$T/logs/e-chain.log
A=$T/logs/arms.log
P="TF_GLM_PROMPT_TAIL=1"
Z="TF_GLM_DSEL=0"
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
SHORT="4096x1 4096x4 65536x1 65536x4"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "e-chain STOP: $*"; exit 1; }
[ $# = 1 ] || stop "give one sha16: d8's native"
got=$(sha256sum "$T/bin/d8/tensorfold-native" 2>/dev/null | cut -c1-16)
[ "$got" = "$1" ] || stop "bin/d8/tensorfold-native is '$got', not $1"
[ -s "$T/bench/grid-ids.json" ] || [ -s "$T/bench/corpus-grid.txt" ] || stop "neither bench/grid-ids.json nor bench/corpus-grid.txt"
[ -s "$T/grid2s.sh" ] || stop "no grid2s.sh"
log "e-chain begin: d8 $1 grid.py $(sha256sum "$T/grid.py" | cut -c1-16) grid2s.sh $(sha256sum "$T/grid2s.sh" | cut -c1-16)"
arm() {   # tp2-arm.sh, label, the arm's last word, then BIN EXPECT LINES [HOOK]
  local tool=$1 label=$2 want=$3 rc last n
  shift 3
  log "$label begin: $tool $*"
  bash "$T/$tool" "$label" "$@"
  rc=$?
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  n=$(grep -acF "$ZLINE" "$T/logs/serve-r0-$label.log" 2>/dev/null)
  log "$label rc=$rc $last; rank 0's TF_GLM_DSEL=0 lines: ${n:-none}"
  sleep 60
  case "$last" in *" $want") ;; *) stop "$label not $want" ;; esac
  case "|$3|" in
    *"|$Z|"*) [ "${n:-0}" -ge 1 ] || stop "$label set $Z, but rank 0's log does not name it" ;;
    *) [ "${n:-0}" = 0 ] || stop "$label left TF_GLM_DSEL unset, but rank 0's log names $Z" ;;
  esac
}
grid() {   # label: its grid ended on the four short cells, or the chain stops
  local cells
  grep -qx "grid $1 end" "$T/bench/$1.log" 2>/dev/null || stop "$1 grid: $(tail -n 2 "$T/bench/$1.log" 2>/dev/null | tr '\n' ' ')"
  cells=$(python3 -c 'import json, sys; print(" ".join("%dx%d" % (c["context"], c["streams"]) for c in json.load(open(sys.argv[1]))["cells"]))' "$T/bench/$1-grid.json" 2>/dev/null)
  [ "$cells" = "$SHORT" ] || stop "$1 grid's cells are '$cells', not '$SHORT'"
  log "$1 grid end: $(python3 "$T/gsum.py" "$T/bench" "$1" | tail -n 1)"
}
arm tp2-arm.sh ew2 served d8 serve "$P" "$T/bench4.sh"
arm tp2-arm.sh ez2a served d8 serve "$P|$Z" "$T/grid2s.sh"
grid ez2a
arm tp2-arm.sh ed2a served d8 serve "$P" "$T/grid2s.sh"
grid ed2a
arm tp2-arm.sh ez2b served d8 serve "$P|$Z" "$T/grid2s.sh"
grid ez2b
arm tp2-arm.sh ed2b served d8 serve "$P" "$T/grid2s.sh"
grid ed2b
arm tp2-arm.sh ez2c served d8 serve "$P|$Z" "$T/grid2s.sh"
grid ez2c
log "e-chain end"
