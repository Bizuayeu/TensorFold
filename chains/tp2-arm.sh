#!/bin/bash
# On 04 (setsid): tp2-arm.sh LABEL BIN EXPECT [lines] [HOOK] — tp4-arm.sh at TP=2 on 04 (rank 0) and 05 (rank 1) over their
# link (rank2.base.env: the mia pair's f1 lines, MASTER <the pair's link address>), for comparisons with the TP=4 ring on the same binary.
#   BIN     /work/bin/BIN/tensorfold-native (main, base, head), the same sha on every host
#   EXPECT  serve: health, one short greedy chat, the startup lines, stop; refuse: the start must fail on every rank
#   lines   appended to rank.base.env on every host (PATH to BIN is added first)
# Each rank: serve4.sh 4 RANK /work/rank.env in tp4-tf3, logs/serve-rR-LABEL.log; hostwatch.sh beside it (stops the
# engine under 5 GiB free). Ranks start 3, 2, 1, then 0. 20 minutes at most. Ends with "== arm LABEL served|refused|
# stopped" in logs/arms.log; the engines stopped and rank.env back to rank.base.env either way.
set -u
LABEL=${1:?label} BIN=${2:?bin} EXPECT=${3:?serve|refuse} LINES=${4:-} HOOK=${5:-}   # HOOK: a script run with LABEL on 04 after the chat, while serving
T=$HOME/LLLLM/glm-work/tp4
X="bash $T/sx4.sh"
LOG=$T/logs/arms.log
HOSTS=(host04 host05)   # index = rank
DEADLINE=$(($(date +%s) + 1200))
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
engines() { $X "$1" "docker exec tp4-tf3 pgrep -fc '^tensorfold-native serve' 2>/dev/null || echo 0" 2>/dev/null | tail -1; }
stopall() {
  local h
  for h in "${HOSTS[@]}"; do $X $h "docker exec tp4-tf3 pkill -TERM -f '^tensorfold-native serve' 2>/dev/null; true"; done
  for i in $(seq 1 12); do
    local n=0
    for h in "${HOSTS[@]}"; do n=$((n + $(engines $h))); done
    [ "$n" = 0 ] && break
    sleep 5
  done
  for h in "${HOSTS[@]}"; do $X $h "docker exec tp4-tf3 pkill -KILL -f '^tensorfold-native serve' 2>/dev/null; cp $T/rank2.base.env $T/rank.env"; done
}
finish() { stopall; log "== arm $LABEL $1"; exit 0; }
case $EXPECT in serve | refuse) ;; *) echo "EXPECT is serve or refuse"; exit 2 ;; esac
[[ "$LABEL" =~ ^[a-z0-9-]+$ ]] || { echo "bad label"; exit 2; }
log "== arm $LABEL begin: bin $BIN, expect $EXPECT, rank.env +[$LINES]"
want=""
for r in 0 1; do
  h=${HOSTS[$r]}
  line=$($X $h "echo \$(sha256sum $T/bin/$BIN/tensorfold-native | cut -c1-16) \$(docker ps --format '{{.Names}}' | grep -vx tp4-tf3 | wc -l) \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) \$(ls $T/logs/serve-r$r-$LABEL.log 2>/dev/null | wc -l)")
  set -- $line
  [ -z "$want" ] && want=$1
  [ "${1:-}" = "$want" ] && [ "${2:-}" = 0 ] && [ "${3:-}" = 0 ] && [ "${4:-0}" -ge 100 ] && [ "${5:-}" = 0 ] \
    || { log "STOP: $h not ready (bin sha, other containers, GPU apps, MemAvailable GiB, old log): $line"; log "== arm $LABEL stopped"; exit 1; }
done
log "  binary $BIN $want on every host"
for r in 0 1; do
  h=${HOSTS[$r]}
  { echo "PATH=/work/bin/$BIN:\$PATH"; [ -z "$LINES" ] || tr '|' '\n' <<<"$LINES"; } \
    | $X $h "docker start tp4-tf3 > /dev/null && cp $T/rank2.base.env $T/rank.env && cat >> $T/rank.env" \
    || { log "STOP: rank.env or container on $h"; finish stopped; }
  log "  $h rank.env adds: $($X $h "diff $T/rank2.base.env $T/rank.env | grep '^>' | paste -sd' '" < /dev/null)"
done
for r in 1 0; do
  h=${HOSTS[$r]}
  $X $h "docker exec -d tp4-tf3 bash -c 'bash /work/serve4.sh 2 $r /work/rank.env > /work/logs/serve-r$r-$LABEL.log 2>&1' && (setsid bash $T/checkout/v3/scripts/hostwatch.sh $T/logs/hostwatch-$LABEL.log tp4-tf3 < /dev/null > /dev/null 2>&1 &)" \
    || { log "STOP: start on $h"; finish stopped; }
  sleep 2
done
log "  started ranks 1 0"
sleep 30   # before the first look: serve4.sh execs the engine a moment after docker exec returns
state=""
while :; do
  [ "$(date +%s)" -lt "$DEADLINE" ] || { log "  past the deadline"; state=timeout; break; }
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8095/health)
  [ "$code" = 200 ] && { state=health; break; }
  gone=0
  for h in "${HOSTS[@]}"; do [ "$(engines $h)" = 0 ] && gone=$((gone + 1)); done
  [ "$gone" -gt 0 ] && { sleep 10; state="gone$gone"; break; }
  sleep 15
done
s=$(( $(date +%s) - DEADLINE + 1200 ))
log "  after ${s} s: $state"
for r in 0 1; do
  h=${HOSTS[$r]}
  log "  rank $r ($h) lines:"
  $X $h "grep -aE '^\[serve.sh\]|^\[tensorfold\]|error|Error|refus|warm|probe|NET/IB : Using|via NET|ibv_modify_qp|WARN|panic' $T/logs/serve-r$r-$LABEL.log | grep -avE 'dlvsym|ncclOsDlopen|Could not find' | tail -n 40 | cut -c1-400" >> "$LOG" 2>&1
done
case $EXPECT-$state in
  serve-health)
    for r in 0 1; do
      h=${HOSTS[$r]}
      log "  $h engine: $($X $h "docker exec tp4-tf3 sh -c 'p=\$(pgrep -f \"^tensorfold-native serve\" | head -1); readlink /proc/\$p/exe; sha256sum /proc/\$p/exe | cut -c1-16'" | tr '\n' ' ') MemAvailable $($X $h "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo") GiB"
    done
    curl -s -m 5 http://127.0.0.1:8095/health > "$T/logs/health-$LABEL.json"
    curl -s -m 300 http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"glm-tf","messages":[{"role":"user","content":"What is 17 times 23? Answer with the number only."}],"max_tokens":1024,"temperature":0}' \
      > "$T/logs/chat-$LABEL.json"
    log "  chat: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c=d['choices'][0]; print(repr(c['message'].get('content'))[:200], c.get('finish_reason'), d.get('usage'))" "$T/logs/chat-$LABEL.json" 2>&1 | tail -n 1 | cut -c1-500)"
    if [ -n "$HOOK" ]; then
      log "  hook $HOOK begin"
      timeout 3600 bash "$HOOK" "$LABEL" >> "$T/logs/hook-$LABEL.out" 2>&1; log "  hook rc=$?"
      gone=0; for h in "${HOSTS[@]}"; do [ "$(engines $h)" = 0 ] && gone=$((gone + 1)); done
      [ "$gone" = 0 ] || log "  after the hook: $gone ranks' engines are gone"
    fi
    finish served ;;
  refuse-gone*) finish refused ;;
  *) log "  unexpected: expect $EXPECT, got $state"; finish stopped ;;
esac
