#!/bin/bash
# On 04 (setsid): f2-chain.sh MAIN16 HEAD16 -- the F gate at TP=2 on 04-05 (plans/MIA1104_ADOPTION_PLAN.md 「損が無い」,
# DECISIONS 11-09 21:2x and 22:xx): main (bin/m1b16, upstream main 1b16a6e8d) and the head (bin/h0067, a4/glm-dsel4's
# last commit), each on TP=2's release settings (tp2.env: the 1M window, 4 streams), fbench.sh as the hook. One
# uncounted start first, then five in turn, 60 s apart (DECISIONS 23:2x):
#   fw2  head, uncounted (the prompt files, the first load)   fm2a main  fh2a head  fm2b main  fh2b head  fm2c main
# The arguments are the sha256 (first 16) of bin/m1b16's and bin/h0067's tensorfold-native on 04; a mismatch stops
# before anything starts (tp2-arm.sh holds 05 to 04's). An arm that does not serve, a hook without its end line, or a
# rank 0 log naming TF_GLM_DSEL=0 (no arm sets it) stops the chain. At the end fsum.py reads the five counted arms.
# Log logs/f2-chain.log; last line "f2-chain end" or "f2-chain STOP: ...". Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
C=$T/logs/f2-chain.log
A=$T/logs/arms.log
B=$T/fbench
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "f2-chain STOP: $*"; exit 1; }
[ $# = 2 ] || stop "give two sha16: main's and the head's native"
MAIN16=$1 HEAD16=$2
got=$(sha256sum "$T/bin/m1b16/tensorfold-native" 2>/dev/null | cut -c1-16)
[ "$got" = "$MAIN16" ] || stop "bin/m1b16/tensorfold-native is '$got', not $MAIN16"
got=$(sha256sum "$T/bin/h0067/tensorfold-native" 2>/dev/null | cut -c1-16)
[ "$got" = "$HEAD16" ] || stop "bin/h0067/tensorfold-native is '$got', not $HEAD16"
for f in fbench.sh mkprompts_f.py csum.py fsum.py tp2-arm.sh; do [ -s "$T/$f" ] || stop "no $f"; done
for f in tools/bench_cache.py tools/bench_concurrent.py tools/bench_prompts.json b2-edit.json corpus.txt; do [ -s "$B/$f" ] || stop "no fbench/$f"; done
log "f2-chain begin: m1b16 $MAIN16 h0067 $HEAD16 fbench.sh $(sha256sum "$T/fbench.sh" | cut -c1-16) fsum.py $(sha256sum "$T/fsum.py" | cut -c1-16) tp2-arm.sh $(sha256sum "$T/tp2-arm.sh" | cut -c1-16)"
arm() {   # label, bin: tp2-arm.sh with fbench.sh as the hook, on the release settings
  local label=$1 bin=$2 rc last n
  log "$label begin: $bin"
  bash "$T/tp2-arm.sh" "$label" "$bin" serve "" "$T/fbench.sh"
  rc=$?
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  n=$(grep -acF "$ZLINE" "$T/logs/serve-r0-$label.log" 2>/dev/null)
  log "$label rc=$rc $last; rank 0's TF_GLM_DSEL=0 lines: ${n:-none}; $(grep -a "binary $bin " "$A" | tail -n 1 | cut -c21-)"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ "${n:-0}" = 0 ] || stop "$label's rank 0 log names TF_GLM_DSEL=0"
  grep -qx "fbench $label end" "$B/$label.log" 2>/dev/null || stop "$label's fbench: $(tail -n 2 "$B/$label.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  log "$label fbench: $(grep -a -E '  (cold|openai|decode|edit) ' "$B/$label.log" | cut -c21- | paste -sd'|' | cut -c1-1500)"
}
arm fw2 h0067
arm fm2a m1b16
arm fh2a h0067
arm fm2b m1b16
arm fh2b h0067
arm fm2c m1b16
python3 "$T/fsum.py" "$B" "$T/logs" fm2a,fm2b,fm2c fh2a,fh2b --json "$B/f2-fsum.json" > "$B/f2-fsum.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/f2-fsum.txt")"
log "f2-chain end"
