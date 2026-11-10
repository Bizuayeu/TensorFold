#!/bin/bash
# On 04 (setsid): l-chain.sh MAIN HEAD -- 0067's long contexts, main against this head (DECISIONS 11-10 07:1x): TP=4 on
# the ring with gather, as d-chain.sh ran d8, but main's and this head's binaries in place of d8's switch. One start
# uncounted first (bench4.sh as the hook: the first start of a chain ran fastest whichever binary it was, DECISIONS 11-09
# 12:1x), then three starts with grid.py through grid4.sh as the hook. One after another, 60 s apart:
#   lw4   this head (h0067), bench4.sh (uncounted)
#   lm4a  main (m1b16), grid    lh4  this head (h0067), grid    lm4b  main (m1b16), grid
# Every arm adds TF_GLM_PROMPT_TAIL=1 (grid.py's prefix resumes only with it). No arm sets TF_GLM_DSEL: rank 0's log must
# not name TF_GLM_DSEL=0's select, or the chain stops. The arguments are the sha256 (first 16) of bin/m1b16's and
# bin/h0067's tensorfold-native on 04; a mismatch stops before anything starts (tp4-arm.sh checks every host's), and an arm
# that does not serve or a grid that does not end stops the chain. Log logs/l-chain.log; last line "l-chain end" or
# "l-chain STOP: ...". Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
C=$T/logs/l-chain.log
A=$T/logs/arms.log
P="TF_GLM_PROMPT_TAIL=1"
G="TF_GLM_PREFILL_REDUCE=gather|NCCL_CUMEM_ENABLE=1|$P"
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "l-chain STOP: $*"; exit 1; }
[ $# = 2 ] || stop "give two sha16s: main's (bin/m1b16) and this head's (bin/h0067) native"
M=$1 H=$2
for pair in "m1b16:$M" "h0067:$H"; do
  b=${pair%%:*} want=${pair#*:}
  got=$(sha256sum "$T/bin/$b/tensorfold-native" 2>/dev/null | cut -c1-16)
  [ "$got" = "$want" ] || stop "bin/$b/tensorfold-native is '$got', not $want"
done
[ -s "$T/bench/grid-ids.json" ] || [ -s "$T/bench/corpus-grid.txt" ] || stop "neither bench/grid-ids.json nor bench/corpus-grid.txt"
log "l-chain begin: main $M head $H grid.py $(sha256sum "$T/grid.py" | cut -c1-16)"
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
grid() {   # label: its grid ended, or the chain stops
  grep -qx "grid $1 end" "$T/bench/$1.log" 2>/dev/null || stop "$1 grid: $(tail -n 2 "$T/bench/$1.log" 2>/dev/null | tr '\n' ' ')"
  log "$1 grid end: $(python3 "$T/gsum.py" "$T/bench" "$1" | tail -n 1)"
}
arm lw4 h0067 "$T/bench4.sh"
arm lm4a m1b16 "$T/grid4.sh"
grid lm4a
arm lh4 h0067 "$T/grid4.sh"
grid lh4
arm lm4b m1b16 "$T/grid4.sh"
grid lm4b
log "l-chain end: $(python3 "$T/gsum.py" "$T/bench" lm4a lh4 lm4b | tail -n 1)"
