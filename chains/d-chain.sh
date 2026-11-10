#!/bin/bash
# On 04 (setsid): d-chain.sh D -- 0067's T2 (plans/MIA1104_ADOPTION_PLAN.md, "GPU の段の形"; DECISIONS 11-09 12:3x): d8
# (v8 with 0067) at TP=4 and then TP=2. Each configuration starts once uncounted first (bench4.sh as the hook: the first
# start of a chain ran fastest whichever binary it was, DECISIONS 11-09 12:1x), then three starts with grid.py through
# grid4.sh as the hook: the old one-CTA select (TF_GLM_DSEL=0), the default (0067's select on many CTAs), the old again.
# One after another, 60 s apart:
#   dw4   d8  TP=4 on the ring with gather, bench4.sh (uncounted)
#   dz4a  d8  likewise with TF_GLM_DSEL=0, grid    dd4  d8  likewise on the default, grid    dz4b  d8  as dz4a, grid
#   dw2   d8  TP=2 on 04-05 on its default (split), bench4.sh (uncounted)
#   dz2a  d8  likewise with TF_GLM_DSEL=0, grid    dd2  d8  likewise on the default, grid    dz2b  d8  as dz2a, grid
# Every arm adds TF_GLM_PROMPT_TAIL=1 (t2-chain.sh: grid.py's prefix resumes only with it). After each arm, rank 0's
# log must name TF_GLM_DSEL=0's select exactly when the arm set it (d8 logs it as its kernels load), or the chain
# stops. The argument is the sha256 (first 16) of bin/d8's tensorfold-native on 04; a mismatch stops before anything
# starts, and an arm that does not serve or a grid that does not end stops the chain. Log logs/d-chain.log; last line
# "d-chain end" or "d-chain STOP: ...". Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
C=$T/logs/d-chain.log
A=$T/logs/arms.log
P="TF_GLM_PROMPT_TAIL=1"
G="TF_GLM_PREFILL_REDUCE=gather|NCCL_CUMEM_ENABLE=1|$P"
Z="TF_GLM_DSEL=0"
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "d-chain STOP: $*"; exit 1; }
[ $# = 1 ] || stop "give one sha16: d8's native"
got=$(sha256sum "$T/bin/d8/tensorfold-native" 2>/dev/null | cut -c1-16)
[ "$got" = "$1" ] || stop "bin/d8/tensorfold-native is '$got', not $1"
[ -s "$T/bench/grid-ids.json" ] || [ -s "$T/bench/corpus-grid.txt" ] || stop "neither bench/grid-ids.json nor bench/corpus-grid.txt"
log "d-chain begin: d8 $1 grid.py $(sha256sum "$T/grid.py" | cut -c1-16)"
arm() {   # tp4-arm.sh or tp2-arm.sh, label, the arm's last word, then BIN EXPECT LINES [HOOK]
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
grid() {   # label: its grid ended, or the chain stops
  grep -qx "grid $1 end" "$T/bench/$1.log" 2>/dev/null || stop "$1 grid: $(tail -n 2 "$T/bench/$1.log" 2>/dev/null | tr '\n' ' ')"
  log "$1 grid end: $(python3 "$T/gsum.py" "$T/bench" "$1" | tail -n 1)"
}
arm tp4-arm.sh dw4 served d8 serve "$G" "$T/bench4.sh"
arm tp4-arm.sh dz4a served d8 serve "$G|$Z" "$T/grid4.sh"
grid dz4a
arm tp4-arm.sh dd4 served d8 serve "$G" "$T/grid4.sh"
grid dd4
arm tp4-arm.sh dz4b served d8 serve "$G|$Z" "$T/grid4.sh"
grid dz4b
arm tp2-arm.sh dw2 served d8 serve "$P" "$T/bench4.sh"
arm tp2-arm.sh dz2a served d8 serve "$P|$Z" "$T/grid4.sh"
grid dz2a
arm tp2-arm.sh dd2 served d8 serve "$P" "$T/grid4.sh"
grid dd2
arm tp2-arm.sh dz2b served d8 serve "$P|$Z" "$T/grid4.sh"
grid dz2b
log "d-chain end"
