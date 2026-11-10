#!/bin/bash
# On 04 (TP=2) or 01 (TP=3), setsid, started once the F gate's chain writes its own lines: fr-chain.sh TP MAIN16 HEAD16
# -- the F gate's one retake (DECISIONS 11-09 23:3x, the user's answer): waits for f2-chain.log's (f3-chain.log's)
# "end" or "STOP" line (STOP: nothing runs), reads the chain's fsum line, and only when it counts slower gate figures
# runs head -> main once more (fhNc, fmNd, the alternation going on), then fsum.py over main's four and the head's
# three. Six hours of waiting at most. TP=3 takes 01's GPU lock before its arms. The arm checks are f2-chain.sh's and
# f3-chain.sh's. Log logs/frN-chain.log; last line "frN-chain end", "frN-chain skip: ..." or "frN-chain STOP: ...".
set -u
N=${1:?usage: fr-chain.sh TP MAIN16 HEAD16} MAIN16=${2:?} HEAD16=${3:?}
case $N in
  2) W=$HOME/LLLLM/glm-work/tp4; ARM=tp2-arm.sh ;;
  3) W=$HOME/LLLLM/glm-work/f0067; ARM=tp3-arm.sh ;;
  *) echo "TP is 2 or 3"; exit 2 ;;
esac
B=$W/fbench
CH=$W/logs/f$N-chain.log
C=$W/logs/fr$N-chain.log
A=$W/logs/arms.log
ZLINE="TF_GLM_DSEL=0: a decode window's DSA select keeps one 1,024-thread CTA a row"
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "fr$N-chain STOP: $*"; exit 1; }
log "fr$N-chain begin: waiting for f$N-chain's end"
end=$(( $(date +%s) + 21600 ))
while :; do
  last=$(tail -n 1 "$CH" 2>/dev/null)
  case "$last" in
    *"f$N-chain end") break ;;
    *"f$N-chain STOP"*) log "fr$N-chain skip: the chain stopped ($last)"; exit 0 ;;
  esac
  [ "$(date +%s)" -lt "$end" ] || stop "no end line in six hours"
  sleep 60
done
n=$(sed -n 's/.*fsum: \([0-9][0-9]*\) slower gate figures.*/\1/p' "$B/f$N-fsum.txt" | tail -n 1)
[[ "$n" =~ ^[0-9]+$ ]] || stop "no fsum line in f$N-fsum.txt"
[ "$n" -gt 0 ] || { log "fr$N-chain skip: $n slower gate figures"; exit 0; }
log "fr$N-chain: $n slower gate figures, one more head -> main"
for pair in "m1b16 $MAIN16" "h0067 $HEAD16"; do
  set -- $pair
  [ "$(sha256sum "$W/bin/$1/tensorfold-native" 2>/dev/null | cut -c1-16)" = "$2" ] || stop "bin/$1 is not $2"
done
if [ "$N" = 3 ]; then exec 9> /tmp/glm-perf-01.lock; log "waiting for 01's GPU lock"; flock 9; log "holding 01's GPU lock"; fi
arm() {   # label, bin
  local label=$1 bin=$2 rc last z
  log "$label begin: $bin"
  if [ "$N" = 2 ]; then bash "$W/$ARM" "$label" "$bin" serve "" "$W/fbench.sh"; else bash "$W/$ARM" "$label" "$bin" "$W/fbench.sh"; fi
  rc=$?
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  z=$(grep -acF "$ZLINE" "$W/logs/serve-r0-$label.log" 2>/dev/null)
  log "$label rc=$rc $last; rank 0's TF_GLM_DSEL=0 lines: ${z:-none}; $(grep -a "binary $bin " "$A" | tail -n 1 | cut -c21-)"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ "${z:-0}" = 0 ] || stop "$label's rank 0 log names TF_GLM_DSEL=0"
  grep -qx "fbench $label end" "$B/$label.log" 2>/dev/null || stop "$label's fbench: $(tail -n 2 "$B/$label.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
}
arm fh${N}c h0067
arm fm${N}d m1b16
python3 "$W/fsum.py" "$B" "$W/logs" fm${N}a,fm${N}b,fm${N}c,fm${N}d fh${N}a,fh${N}b,fh${N}c --json "$B/f$N-fsum-retake.json" > "$B/f$N-fsum-retake.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/f$N-fsum-retake.txt")"
log "fr$N-chain end"
