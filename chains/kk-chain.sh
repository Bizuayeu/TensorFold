#!/bin/bash
# On 04 (setsid): kk-chain.sh MAIN HEAD -- the retake of TP=4's 4K cells, main against 0067's head (DECISIONS 11-10 10:2x),
# as e-chain.sh retook TP=2's short cells: TP=4 on the ring with gather, as l-chain.sh ran it, 4,096 tokens at 1, 4 and
# 8 streams, 6 reps a cell (grid4k.sh as the hook). One start uncounted first (bench4.sh), then five in turn, 60 s apart:
#   kw4   this head (h0067), bench4.sh (uncounted)
#   km4a  main (m1b16)   kh4a  this head   km4b  main   kh4b  this head   km4c  main   (grid4k.sh)
# Every arm adds TF_GLM_PROMPT_TAIL=1. No arm sets TF_GLM_DSEL: rank 0's log must not name TF_GLM_DSEL=0's select, and
# after each grid its cells must be the three 4K ones, or the chain stops. The arguments are the sha256 (first 16) of
# bin/m1b16's and bin/h0067's tensorfold-native on 04; a mismatch stops before anything starts, and an arm that does not
# serve or a grid that does not end stops the chain. Log logs/kk-chain.log; last line "kk-chain end" or
# "kk-chain STOP: ...". Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
C=$T/logs/kk-chain.log
A=$T/logs/arms.log
P="TF_GLM_PROMPT_TAIL=1"
G="TF_GLM_PREFILL_REDUCE=gather|NCCL_CUMEM_ENABLE=1|$P"
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
CELLS="4096x1 4096x4 4096x8"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "kk-chain STOP: $*"; exit 1; }
[ $# = 2 ] || stop "give two sha16s: main's (bin/m1b16) and this head's (bin/h0067) native"
M=$1 H=$2
for pair in "m1b16:$M" "h0067:$H"; do
  b=${pair%%:*} want=${pair#*:}
  got=$(sha256sum "$T/bin/$b/tensorfold-native" 2>/dev/null | cut -c1-16)
  [ "$got" = "$want" ] || stop "bin/$b/tensorfold-native is '$got', not $want"
done
[ -s "$T/bench/grid-ids.json" ] || [ -s "$T/bench/corpus-grid.txt" ] || stop "neither bench/grid-ids.json nor bench/corpus-grid.txt"
[ -s "$T/grid4k.sh" ] || stop "no grid4k.sh"
log "kk-chain begin: main $M head $H grid.py $(sha256sum "$T/grid.py" | cut -c1-16) grid4k.sh $(sha256sum "$T/grid4k.sh" | cut -c1-16)"
arm() {   # label, bin, hook
  local label=$1 bin=$2 hook=$3 rc last n
  log "$label begin: tp4-arm.sh $bin serve $G $hook"
  bash "$T/tp4-arm.sh" "$label" "$bin" serve "$G" "$hook"
  rc=$?
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  n=$(grep -acF "$ZLINE" "$T/logs/serve-r0-$label.log" 2>/dev/null)
  log "$label rc=$rc $last; rank 0's TF_GLM_DSEL=0 lines: ${n:-none}"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ "${n:-0}" = 0 ] || stop "$label set no TF_GLM_DSEL, but rank 0's log names TF_GLM_DSEL=0"
}
grid() {   # label: its grid ended on the three 4K cells, or the chain stops
  local cells
  grep -qx "grid $1 end" "$T/bench/$1.log" 2>/dev/null || stop "$1 grid: $(tail -n 2 "$T/bench/$1.log" 2>/dev/null | tr '\n' ' ')"
  cells=$(python3 -c 'import json, sys; print(" ".join("%dx%d" % (c["context"], c["streams"]) for c in json.load(open(sys.argv[1]))["cells"]))' "$T/bench/$1-grid.json" 2>/dev/null)
  [ "$cells" = "$CELLS" ] || stop "$1 grid's cells are '$cells', not '$CELLS'"
  log "$1 grid end: $(python3 "$T/gsum.py" "$T/bench" "$1" | tail -n 1)"
}
arm kw4 h0067 "$T/bench4.sh"
arm km4a m1b16 "$T/grid4k.sh"
grid km4a
arm kh4a h0067 "$T/grid4k.sh"
grid kh4a
arm km4b m1b16 "$T/grid4k.sh"
grid km4b
arm kh4b h0067 "$T/grid4k.sh"
grid kh4b
arm km4c m1b16 "$T/grid4k.sh"
grid km4c
log "kk-chain end: $(python3 "$T/gsum.py" "$T/bench" km4a kh4a km4b kh4b km4c | tail -n 1)"
