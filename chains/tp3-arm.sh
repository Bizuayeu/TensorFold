#!/bin/bash
# On 01 (through f3-chain.sh): tp3-arm.sh LABEL BIN [HOOK] -- an arm of the F gate at TP=3 on 01-03, rank 0 on 01
# (DECISIONS 11-09 22:xx). The ranks run in this session's container tp3f on each host (c124's create_container.sh on
# glm53-tf3:3x-e11482a, the rc's image; WORK ~/LLLLM/glm-work/f0067 at /work) on the image's serve.sh, so the release's
# TP=3 settings (its tp3.env: the 1M window, 8 streams), and each host's tf-work/v3/rank.tp3.env (read, never written)
# copied to WORK/rank.env with PATH=/work/bin/BIN appended in front of the image's. c124's cluster.sh starts them (the
# three ranks together, rank 0 last, the memory guard on each host) and stops them, on cluster.tp3f.env.
#   BIN   WORK/bin/BIN/tensorfold-native, the same sha on every host
#   HOOK  a script run on 01 with LABEL while it serves (an hour at most)
# Before the start every host must hold BIN's sha, have no GPU compute app, no engine in tp3f, tp3f running, at least
# 100 GiB MemAvailable and no log of LABEL. Ends with "== arm LABEL served" or "== arm LABEL stopped" in
# WORK/logs/arms.log; the engines are stopped either way. 20 minutes to serve (cluster.sh's 15).
set -u
LABEL=${1:?label} BIN=${2:?bin} HOOK=${3:-}
G=$HOME/LLLLM/models/glm53-nvfp4
W=$HOME/LLLLM/glm-work/f0067
X="bash $G/tf-work/v292/sx.sh"
CL="bash $G/tf-work/v3/c124/v3/scripts/cluster.sh $W/cluster.tp3f.env"
LOG=$W/logs/arms.log
HOSTS=(host01 host02 host03)
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
finish() { $CL stop >> "$LOG" 2>&1; log "== arm $LABEL $1"; exit 0; }
[[ "$LABEL" =~ ^[a-z0-9-]+$ ]] || { echo "bad label"; exit 2; }
[ -s "$W/cluster.tp3f.env" ] || { echo "no cluster.tp3f.env"; exit 2; }
log "== arm $LABEL begin: bin $BIN, hook ${HOOK:-none}"
want=""
for r in 0 1 2; do
  h=${HOSTS[$r]}
  line=$($X $h "echo \$(sha256sum $W/bin/$BIN/tensorfold-native | cut -c1-16) \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) \$(docker inspect -f '{{.State.Running}}' tp3f 2>/dev/null) \$(docker exec tp3f ps -eo args 2>/dev/null | grep -c '[t]ensorfold-native serve') \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) \$(ls $W/logs/serve-r$r-$LABEL.log 2>/dev/null | wc -l)" < /dev/null)
  set -- $line
  [ -z "$want" ] && want=$1
  [ "${1:-}" = "$want" ] && [ "${2:-}" = 0 ] && [ "${3:-}" = true ] && [ "${4:-}" = 0 ] && [ "${5:-0}" -ge 100 ] && [ "${6:-}" = 0 ] \
    || { log "STOP: $h not ready (bin sha, GPU apps, tp3f running, its engines, MemAvailable GiB, old log): $line"; log "== arm $LABEL stopped"; exit 1; }
done
log "  binary $BIN $want on every host"
for r in 0 1 2; do
  h=${HOSTS[$r]}
  $X $h "cp $G/tf-work/v3/rank.tp3.env $W/rank.env.new && printf '\nPATH=/work/bin/$BIN:\$PATH\n' >> $W/rank.env.new && mv $W/rank.env.new $W/rank.env" < /dev/null \
    || { log "STOP: rank.env on $h"; finish stopped; }
  log "  $h rank.env: rank.tp3.env $($X $h "md5sum < $G/tf-work/v3/rank.tp3.env | cut -c1-8" < /dev/null) and $($X $h "diff $G/tf-work/v3/rank.tp3.env $W/rank.env | grep '^>' | paste -sd' '" < /dev/null)"
done
out=$(timeout 1200 $CL start "$LABEL" 2>&1)
rc=$?
echo "$out" | tail -n 12 >> "$LOG"
[ $rc = 0 ] || { log "  cluster.sh start rc=$rc"; finish stopped; }
for r in 0 1 2; do
  h=${HOSTS[$r]}
  log "  $h engine: $($X $h "docker exec tp3f sh -c 'p=\$(pgrep -f \"^tensorfold-native serve\" | head -1); readlink /proc/\$p/exe; sha256sum /proc/\$p/exe | cut -c1-16'" < /dev/null | tr '\n' ' ') MemAvailable $($X $h "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" < /dev/null) GiB"
done
curl -s -m 5 http://127.0.0.1:8095/health > "$W/logs/health-$LABEL.json"
model=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$W/logs/health-$LABEL.json" 2>/dev/null)
curl -s -m 300 http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"'"${model:-local-model}"'","messages":[{"role":"user","content":"What is 17 times 23? Answer with the number only."}],"max_tokens":1024,"temperature":0}' \
  > "$W/logs/chat-$LABEL.json"
log "  chat: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c=d['choices'][0]; print(repr(c['message'].get('content'))[:200], c.get('finish_reason'), d.get('usage'))" "$W/logs/chat-$LABEL.json" 2>&1 | tail -n 1 | cut -c1-500)"
if [ -n "$HOOK" ]; then
  log "  hook $HOOK begin"
  timeout 3600 bash "$HOOK" "$LABEL" >> "$W/logs/hook-$LABEL.out" 2>&1
  log "  hook rc=$?"
  for r in 0 1 2; do
    n=$($X ${HOSTS[$r]} "docker exec tp3f ps -eo args | grep -c '[t]ensorfold-native serve'" < /dev/null)
    [ "$n" = 1 ] || log "  after the hook: rank $r runs $n engines"
  done
fi
finish served
