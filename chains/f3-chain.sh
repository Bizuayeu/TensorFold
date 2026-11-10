#!/bin/bash
# On 01 (setsid): f3-chain.sh MAIN16 HEAD16 -- the F gate at TP=3 on 01-03 (plans/MIA1104_ADOPTION_PLAN.md 「損が無い」,
# DECISIONS 11-09 21:2x and 22:xx): main (bin/m1b16, upstream main 1b16a6e8d) and the head (bin/h0067, a4/glm-dsel4's
# last commit) through tp3-arm.sh on the release's TP=3 settings (8 streams), fbench.sh as the hook, holding 01's GPU
# lock (/tmp/glm-perf-01.lock, flock as dl0.sh) from before the first arm to the end. One uncounted start first, then
# five in turn, 60 s apart (DECISIONS 23:2x):
#   fw3  head, uncounted (the first load)   fm3a main  fh3a head  fm3b main  fh3b head  fm3c main
# The arguments are the sha256 (first 16) of WORK/bin/m1b16's and bin/h0067's tensorfold-native on 01; a mismatch stops
# before anything starts (tp3-arm.sh holds 02 and 03 to 01's). An arm that does not serve, a hook without its end
# line or a rank 0 log naming TF_GLM_DSEL=0 (no arm sets it) stops the chain. At the end fsum.py reads the five
# counted arms. Log WORK/logs/f3-chain.log; last line "f3-chain end" or "f3-chain STOP: ...". Never edit while it runs.
set -u
W=$HOME/LLLLM/glm-work/f0067
C=$W/logs/f3-chain.log
A=$W/logs/arms.log
B=$W/fbench
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "f3-chain STOP: $*"; exit 1; }
[ $# = 2 ] || stop "give two sha16: main's and the head's native"
MAIN16=$1 HEAD16=$2
got=$(sha256sum "$W/bin/m1b16/tensorfold-native" 2>/dev/null | cut -c1-16)
[ "$got" = "$MAIN16" ] || stop "bin/m1b16/tensorfold-native is '$got', not $MAIN16"
got=$(sha256sum "$W/bin/h0067/tensorfold-native" 2>/dev/null | cut -c1-16)
[ "$got" = "$HEAD16" ] || stop "bin/h0067/tensorfold-native is '$got', not $HEAD16"
for f in fbench.sh mkprompts_f.py csum.py fsum.py tp3-arm.sh cluster.tp3f.env; do [ -s "$W/$f" ] || stop "no $f"; done
for f in tools/bench_cache.py tools/bench_concurrent.py tools/bench_prompts.json b2-edit.json corpus.txt; do [ -s "$B/$f" ] || stop "no fbench/$f"; done
exec 9> /tmp/glm-perf-01.lock
log "f3-chain: waiting for 01's GPU lock"
flock 9
log "f3-chain begin, holding 01's GPU lock: m1b16 $MAIN16 h0067 $HEAD16 fbench.sh $(sha256sum "$W/fbench.sh" | cut -c1-16) fsum.py $(sha256sum "$W/fsum.py" | cut -c1-16) tp3-arm.sh $(sha256sum "$W/tp3-arm.sh" | cut -c1-16)"
arm() {   # label, bin
  local label=$1 bin=$2 rc last n
  log "$label begin: $bin"
  bash "$W/tp3-arm.sh" "$label" "$bin" "$W/fbench.sh"
  rc=$?
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  n=$(grep -acF "$ZLINE" "$W/logs/serve-r0-$label.log" 2>/dev/null)
  log "$label rc=$rc $last; rank 0's TF_GLM_DSEL=0 lines: ${n:-none}; $(grep -a "binary $bin " "$A" | tail -n 1 | cut -c21-)"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ "${n:-0}" = 0 ] || stop "$label's rank 0 log names TF_GLM_DSEL=0"
  grep -qx "fbench $label end" "$B/$label.log" 2>/dev/null || stop "$label's fbench: $(tail -n 2 "$B/$label.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  log "$label fbench: $(grep -a -E '  (cold|openai|decode|edit) ' "$B/$label.log" | cut -c21- | paste -sd'|' | cut -c1-1500)"
}
arm fw3 h0067
arm fm3a m1b16
arm fh3a h0067
arm fm3b m1b16
arm fh3b h0067
arm fm3c m1b16
python3 "$W/fsum.py" "$B" "$W/logs" fm3a,fm3b,fm3c fh3a,fh3b --json "$B/f3-fsum.json" > "$B/f3-fsum.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/f3-fsum.txt")"
log "f3-chain end"
