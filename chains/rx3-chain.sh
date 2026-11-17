#!/bin/bash
# rx3: the 0091 PR's TP=4 receipt, its draft-off cells at 256K and 1016K again (<the local ledger> 11-17 02:2x: rx2's grid had
# 256K x8, 1016K x4 and x8 slower than main's two arms, whose range there was 0.00 to 0.10%; a local decision). Made by <the local records> from rx2-chain.sh. On 04 (setsid): rx3-chain.sh M16 H16 --
# bin/me86a (main e86a0bed1d) and bin/hcc45 (the PR head cc457382c9). As rx2: main's record from rx's timed start is seeded
# into the chain's copy of 04's price cache under both binaries' keys (window-costs-rxseed), so every arm is kept on
# main's window and draft prices and its expert form, and only the binaries' code differs. Arms, 60 s apart, each
# grid.py at 262,144 and 1,040,384 tokens at 1, 4 and 8 streams through grid4l.sh (rx3hook.sh), no drafted reps:
#   rx3ma main  rx3ha head  rx3mb main  rx3hb head  rx3mc main
# rx3/stop-now; tf-preflight on every host before each arm; 04's GPU lock; heat lines to rx3/heat-LABEL.txt; replies
# against rx3ma's (fxcheck.py); gsum.py of the five into rx3/gsum.txt. 04's cache moves aside to window-costs-rx3pre and
# comes back after; the chain's copy stays as window-costs-rx3. No arm starts after 06:50 (04-07 are lent to about 06:30
# by <another local session>, past it with a word). Log logs/rx3-chain.log; the last line "== rx3-chain finished" or
# "== rx3-chain stopped" (the line before says why). Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
F=$T/rx3
C=$T/logs/rx3-chain.log
A=$T/logs/arms.log
G="TF_GLM_PREFILL_REDUCE=gather|NCCL_CUMEM_ENABLE=1|TF_GLM_PROMPT_TAIL=1"
M=me86a
H=hcc45
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp4-tf3 python3 /work/kzt.py)
HOSTS="host04 host05 host07 host06"
LABELS="rx3ma rx3ha rx3mb rx3hb rx3mc"
LAST=$(date -d "2026-11-17 06:50" +%s)   # no arm starts after this: 04-07 are lent to about 06:30, past it with a word
HEAT=/var/log/gb10-telemetry/heat-cap.log
moved=""
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() {
  local why=$*
  [ -n "$moved" ] && why="$why (the cache is aside: $moved; to put the first tables back, in tp4-tf3: mv $WC $WC-rx3stop && mv $WC-rx3pre $WC)"
  log "rx3-chain STOP: $why"
  log "== rx3-chain stopped"
  exit 1
}
on() {   # HOST CMD: CMD on HOST (04 here, the others over sx4.sh)
  local h=$1
  shift
  if [ "$h" = host04 ]; then bash -c "$*"; else bash "$T/sx4.sh" "$h" "$*" < /dev/null; fi
}
[ $# = 2 ] || stop "give two sha16s: bin/$M's native, then bin/$H's"
for p in "$M $1" "$H $2"; do
  b=${p% *} want=${p#* }
  got=$(sha256sum "$T/bin/$b/tensorfold-native" 2>/dev/null | cut -c1-16)
  [ "$got" = "$want" ] || stop "bin/$b/tensorfold-native is '$got', not $want"
done
[ -e "$F" ] && stop "$F exists"
for l in $LABELS; do
  for f in "$T/logs/serve-r0-$l.log" "$T/bench/$l.log" "$T/bench/$l-grid.json"; do
    [ -e "$f" ] && stop "$f exists"
  done
done
for f in tp4-arm.sh grid4l.sh grid.py gsum.py fxcheck.py rx3hook.sh heatcut.py kzt.py sx4.sh; do
  [ -s "$T/$f" ] || stop "no $T/$f"
done
[ -s "$T/bench/grid-ids.json" ] || [ -s "$T/bench/corpus-grid.txt" ] || stop "neither bench/grid-ids.json nor bench/corpus-grid.txt"
img="" s4=""
for h in $HOSTS; do
  i=$(on "$h" "docker inspect -f '{{.Image}}' tp4-tf3 2>/dev/null | cut -c1-19")
  m=$(on "$h" "md5sum \$HOME/LLLLM/glm-work/tp4/serve4.sh 2>/dev/null | cut -c1-12")
  n=$(on "$h" "grep -a '^NCCL' \$HOME/LLLLM/glm-work/tp4/rank.base.env 2>/dev/null | paste -sd' ' | cut -c1-300")
  log "host $h: tp4-tf3 image ${i:-none}, serve4.sh ${m:-none}, rank.base.env NCCL: ${n:-none}"
  [ -n "$i" ] && [ -n "$m" ] || stop "$h: no tp4-tf3 image or no serve4.sh"
  [ -z "$img" ] && img=$i s4=$m
  [ "$i" = "$img" ] && [ "$m" = "$s4" ] || stop "$h's tp4-tf3 image or serve4.sh is not host04's"
done
docker exec tp4-tf3 sh -c "[ -d $WC ] && [ ! -e $WC-rx3pre ] && [ ! -e $WC-rx3 ] && [ ! -e $WC-rx3stop ]" \
  || stop "tp4-tf3: no $WC, or window-costs-rx3pre, -rx3 or -rx3stop is there"
mkdir -p "$F"
exec 9> /tmp/glm-perf-04.lock
flock -n 9 || stop "04's GPU lock (/tmp/glm-perf-04.lock) is held"
log "rx3-chain begin, holding 04's GPU lock: $M $1, $H $2; tp4-arm $(sha256sum "$T/tp4-arm.sh" | cut -c1-16) grid4l $(sha256sum "$T/grid4l.sh" | cut -c1-16) grid.py $(sha256sum "$T/grid.py" | cut -c1-16) rx3hook $(sha256sum "$T/rx3hook.sh" | cut -c1-16) heatcut $(sha256sum "$T/heatcut.py" | cut -c1-16) kzt $(sha256sum "$T/kzt.py" | cut -c1-16)"
ready() {   # LABEL: every host's tf-preflight; heat or an unread look waits
  local l=$1 i h out rc why hot
  for i in $(seq 1 30); do
    why="" hot=""
    for h in $HOSTS; do
      if [ "$h" = host04 ]; then   # the chain holds 04's lock itself (taken free at the begin line)
        out=$(python3 "$HOME/LLLLM/ops/tf-rules/hosts/tf-preflight" --label "$l" --skip lock 2>&1)
      else
        out=$(bash "$T/sx4.sh" "$h" "python3 ~/LLLLM/ops/tf-rules/hosts/tf-preflight --label $l" < /dev/null 2>&1)
      fi
      rc=$?
      echo "$(date '+%F %T') $l $h rc=$rc $(echo "$out" | tail -n 1)" >> "$F/preflight.log"
      [ $rc = 0 ] && continue
      out=$(echo "$out" | tail -n 1 | python3 -c 'import json,sys; print(",".join(json.load(sys.stdin)["failed"]) or "none")' 2>/dev/null)
      case "${out:-unread}" in heat) hot="$hot $h" ;; unread) hot="$hot $h(unread)" ;; *) why="$why $h:$out" ;; esac
    done
    [ -n "$why" ] && stop "$l: not ready:$why (rx3/preflight.log)"
    [ -z "$hot" ] && return 0
    log "$l: heat on$hot (look $i of 30); again in 60 s"
    sleep 60
  done
  stop "$l: still hot or unread on$hot after 30 looks"
}
heat() {   # LABEL T0: every host's heat-cap lines since T0
  local l=$1 t0=$2 h s sums=""
  for h in $HOSTS; do
    s=$(on "$h" "tail -n 400 $HEAT 2>/dev/null" | python3 "$T/heatcut.py" "$t0" "$h" | tee -a "$F/heat-$l.txt" | tail -n 1)
    sums="$sums; $s"
  done
  log "$l heat${sums}"
}
arm() {   # LABEL BIN LINES, then the want: MODE HOW
  local label=$1 bin=$2 lines=$3 rc last n0 t0
  shift 3
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rx3/stop-now: $(head -c 200 "$F/stop-now"))"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after 06:50: 04-07 are lent to about 06:30"
  ready "$label"
  echo "$* $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp4-arm.sh $bin serve +[$lines], want $* $bin"
  bash "$T/tp4-arm.sh" "$label" "$bin" serve "$lines" "$T/rx3hook.sh"
  rc=$?
  heat "$label" "$t0"
  [ "$(grep -a -c "== arm $label " "$A" 2>/dev/null)" -gt "${n0:-0}" ] 2>/dev/null || stop "$label: no new line in logs/arms.log"
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  log "$label rc=$rc $last; $(grep -a " $label want " "$F/form.log" 2>/dev/null | tail -n 1 | cut -c21- | cut -c1-300)"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ -e "$F/stop-$label" ] && stop "$label: rank 0's look is not the arm's ($* $bin): $(cat "$F/stop-$label")"
}
grid() {   # LABEL [REF]: the grid ended, and its replies are REF's
  local label=$1 ref=${2:-}
  grep -qx "grid $label end" "$T/bench/$label.log" 2>/dev/null || stop "$label grid: $(tail -n 2 "$T/bench/$label.log" 2>/dev/null | tr '\n' ' ')"
  if [ -n "$ref" ]; then
    python3 "$T/fxcheck.py" "$T/bench" "$ref" "$label" > "$F/replies-$label" 2>&1 || stop "$(tail -n 1 "$F/replies-$label")"
  fi
  log "$label grid end: $(python3 "$T/gsum.py" "$T/bench" "$label" | tail -n 1)${ref:+; $(tail -n 1 "$F/replies-$label" 2>/dev/null)}"
}
tables() { "${K[@]}" "$WC" > "$F/tables-$1" 2>&1 || stop "kzt.py: $(tail -n 1 "$F/tables-$1")"; log "tables-$1: $(grep -c . "$F/tables-$1") lines"; }
tables before
docker exec tp4-tf3 sh -c "[ ! -e $WC-rx3pre ] && mv $WC $WC-rx3pre && cp -a $WC-rx3pre $WC" || stop "setting the cache aside failed"
moved="window-costs-rx3pre"
log "window-costs to window-costs-rx3pre ($(docker exec tp4-tf3 sh -c "ls -A $WC-rx3pre | wc -l") files); window-costs now a copy of it ($(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l") files)"
tables start
# rx3, as rx2: the counted arms read main's record under both keys (<the local ledger> 11-11 12:1x (b), the 11:5x form)
read -r KM KH < <(docker exec tp4-tf3 sh -c "cat $WC-rxseed/names.txt")
case "${KM:-}/${KH:-}" in ?*.bin/?*.bin) ;; *) stop "window-costs-rxseed/names.txt names no two records: '${KM:-}' '${KH:-}'" ;; esac
own=$(docker exec tp4-tf3 sh -c "cat $WC-rxseed/words.txt")
case "$own" in *"; experts "*) ;; *) stop "window-costs-rxseed/words.txt holds no look words: '$own'" ;; esac
docker exec tp4-tf3 sh -c "cd $WC-rxseed && [ -s main.bin ] && for x in main.*; do cp -p \$x $WC/${KM%.bin}\${x#main} && cp -p \$x $WC/${KH%.bin}\${x#main} || exit 1; done" \
  || stop "seeding the copy with main's record failed"
echo "$own" > "$F/own-$M"
echo "$own" > "$F/own-$H"
log "seeded the copy with main's record as $M's ($KM) and $H's ($KH); both binaries' words: $own"
tables seeded
arm rx3ma "$M" "$G" grid kept
grid rx3ma
arm rx3ha "$H" "$G" grid kept
grid rx3ha rx3ma
arm rx3mb "$M" "$G" grid kept
grid rx3mb rx3ma
arm rx3hb "$H" "$G" grid kept
grid rx3hb rx3ma
arm rx3mc "$M" "$G" grid kept
grid rx3mc rx3ma
python3 "$T/gsum.py" "$T/bench" rx3ma rx3ha rx3mb rx3hb rx3mc > "$F/gsum.txt" 2>&1
log "grid arms end: gsum of rx3ma rx3ha rx3mb rx3hb rx3mc in rx3/gsum.txt ($(wc -l < "$F/gsum.txt") lines)"
tables end
docker exec tp4-tf3 sh -c "[ ! -e $WC-rx3 ] && mv $WC $WC-rx3 && mv $WC-rx3pre $WC" || stop "putting the first tables back failed"
moved=""
log "the chain's tables to window-costs-rx3 ($(docker exec tp4-tf3 sh -c "ls -A $WC-rx3 | wc -l") files); window-costs back with $(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l")"
log "rx3-chain end: $LABELS (rx3/gsum.txt, bench/LABEL-grid.json); the cache as it began, the chain's beside it"
log "== rx3-chain finished"
