#!/bin/bash
# rf2d: rf2c's five counted arms, after rf2c stopped at 12:19:49 on its new-record count (11-17 12:2x, TP=4 session
# [441daa]): the head's timed start (rf2cwh, uncounted, done) left two new records in the chain's copy, the head's key and
# the price reference's (602963cd..., which 04's first tables lacked; rf2c assumed it was there). Made by
# <the local records> from rf2c-chain.sh. On 04 (setsid): rf2d-chain.sh M16 H16 -- bin/me86a and
# bin/ha4da, as rf2c. It starts from the state rf2c left: logs/rf2c-chain.log ending "== rf2c-chain stopped" on that
# count, the first tables aside as window-costs-rf2cpre and window-costs rf2c's copy (main's record under me86a's key,
# the head's timed record under its key and the reference's). The head's key is the one record new since
# rf2c/tables-mseeded that is not the reference; it and the reference must hold rf2c/own-ha4da's words (flwords.py),
# me86a's key main's (window-costs-rf2seed/words.txt). Then main's record over the head's key, read back by sha, and:
#   rf2dma main  rf2dha head  rf2dmb main  rf2dhb head  rf2dmc main
# The rest is rf2c-chain.sh's: rf2dhook.sh checks rank 0's look (kept, rf2d/own-BIN's words: main's); rf2d/stop-now;
# 04-05's tf-preflight before each arm; 04's GPU lock; heat lines to rf2d/heat-LABEL.txt; no arm starts after 14:00
# (04-05 are lent until about 14:15; checked before and after its heat waits); fsum.py on the five into
# fbench/rf2d-fsum.txt and .json; at the end the copy goes beside as window-costs-rf2d and window-costs-rf2cpre comes
# back as window-costs. Log logs/rf2d-chain.log; the last line "== rf2d-chain finished" or "== rf2d-chain stopped" (the
# line before says why). Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
F=$T/rf2d
B=$T/fbench
C=$T/logs/rf2d-chain.log
A=$T/logs/arms.log
M=me86a
H=ha4da
FSUM=db74650034aa894d
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp4-tf3 python3 /work/kzt.py)
HOSTS="host04 host05"
LABELS="rf2dma rf2dha rf2dmb rf2dhb rf2dmc"
LAST=$(date -d "2026-11-17 14:00" +%s)   # no arm starts after this: 04-05 are lent until about 14:15 (<another local session>, 11-17 12:22)
HEAT=/var/log/gb10-telemetry/heat-cap.log
moved=""
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() {
  local why=$*
  [ -n "$moved" ] && why="$why (the cache is aside: $moved; to put the first tables back, in tp4-tf3: mv $WC $WC-rf2dstop && mv $WC-rf2cpre $WC)"
  log "rf2d-chain STOP: $why"
  log "== rf2d-chain stopped"
  exit 1
}
on() {   # HOST CMD: CMD on HOST (04 here, 05 over sx4.sh)
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
got=$(sha256sum "$T/fsum.py" 2>/dev/null | cut -c1-16)
[ "$got" = "$FSUM" ] || stop "fsum.py is '$got', not the 11-15 fix $FSUM"
[ -e "$F" ] && stop "$F exists"
for l in $LABELS; do
  for f in "$T/logs/serve-r0-$l.log" "$B/$l.log" "$B/$l-decode.json"; do
    [ -e "$f" ] && stop "$f exists"
  done
done
for f in tp2-arm.sh fbench.sh mkprompts_f.py csum.py fsum.py rf2dhook.sh heatcut.py kzt.py flwords.py sx4.sh rank2.base.env; do
  [ -s "$T/$f" ] || stop "no $T/$f"
done
for f in tools/bench_cache.py tools/bench_openai.py tools/bench_concurrent.py tools/bench_prompts.json b2-edit.json corpus.txt; do
  [ -s "$B/$f" ] || stop "no fbench/$f"
done
img="" s4=""
for h in $HOSTS; do
  i=$(on "$h" "docker inspect -f '{{.Image}}' tp4-tf3 2>/dev/null | cut -c1-19")
  m=$(on "$h" "md5sum \$HOME/LLLLM/glm-work/tp4/serve4.sh 2>/dev/null | cut -c1-12")
  n=$(on "$h" "grep -a '^NCCL' \$HOME/LLLLM/glm-work/tp4/rank2.base.env 2>/dev/null | paste -sd' ' | cut -c1-300")
  log "host $h: tp4-tf3 image ${i:-none}, serve4.sh ${m:-none}, rank2.base.env NCCL: ${n:-none}"
  [ -n "$i" ] && [ -n "$m" ] || stop "$h: no tp4-tf3 image or no serve4.sh"
  [ -z "$img" ] && img=$i s4=$m
  [ "$i" = "$img" ] && [ "$m" = "$s4" ] || stop "$h's tp4-tf3 image or serve4.sh is not host04's"
done
last=$(tail -n 1 "$T/logs/rf2c-chain.log" 2>/dev/null | cut -c21-)
why=$(tail -n 2 "$T/logs/rf2c-chain.log" 2>/dev/null | head -n 1 | cut -c21-)
[ "$last" = "== rf2c-chain stopped" ] || stop "logs/rf2c-chain.log ends '$last', not '== rf2c-chain stopped'"
case "$why" in "rf2c-chain STOP: the head's timed start left 2 new records, not one: "*) ;; *) stop "rf2c's stop is not the new-record count: $why" ;; esac
for f in own-$H tables-mseeded; do [ -s "$T/rf2c/$f" ] || stop "no rf2c/$f"; done
docker exec tp4-tf3 sh -c "[ -d $WC ] && [ -d $WC-rf2cpre ] && [ ! -e $WC-rf2d ] && [ ! -e $WC-rf2dstop ] && [ ! -e $WC-rf2c ] && [ -s $WC-rf2seed/main.bin ] && [ -s $WC-rf2seed/names.txt ] && [ -s $WC-rf2seed/words.txt ]" \
  || stop "tp4-tf3: no $WC (rf2c's copy), no window-costs-rf2cpre (the first tables) or no window-costs-rf2seed, or window-costs-rf2d, -rf2dstop or -rf2c is there"
moved="window-costs-rf2cpre"
mkdir -p "$F"
exec 9> /tmp/glm-perf-04.lock
flock -n 9 || stop "04's GPU lock (/tmp/glm-perf-04.lock) is held"
log "rf2d-chain begin, holding 04's GPU lock: $M $1, $H $2; tp2-arm $(sha256sum "$T/tp2-arm.sh" | cut -c1-16) fbench $(sha256sum "$T/fbench.sh" | cut -c1-16) fsum $(sha256sum "$T/fsum.py" | cut -c1-16) rf2dhook $(sha256sum "$T/rf2dhook.sh" | cut -c1-16) heatcut $(sha256sum "$T/heatcut.py" | cut -c1-16) kzt $(sha256sum "$T/kzt.py" | cut -c1-16) rank2.base.env $(sha256sum "$T/rank2.base.env" | cut -c1-16)"
ready() {   # LABEL: 04's and 05's tf-preflight; heat or an unread look waits
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
    [ -n "$why" ] && stop "$l: not ready:$why (rf2d/preflight.log)"
    [ -z "$hot" ] && return 0
    log "$l: heat on$hot (look $i of 30); again in 60 s"
    sleep 60
  done
  stop "$l: still hot or unread on$hot after 30 looks"
}
heat() {   # LABEL T0: 04's and 05's heat-cap lines since T0
  local l=$1 t0=$2 h s sums=""
  for h in $HOSTS; do
    s=$(on "$h" "tail -n 400 $HEAT 2>/dev/null" | python3 "$T/heatcut.py" "$t0" "$h" | tee -a "$F/heat-$l.txt" | tail -n 1)
    sums="$sums; $s"
  done
  log "$l heat${sums}"
}
arm() {   # LABEL BIN HOW: tp2-arm.sh with rf2dhook.sh (then fbench.sh) on the release settings
  local label=$1 bin=$2 how=$3 rc last n0 t0
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rf2d/stop-now: $(head -c 200 "$F/stop-now"))"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after 14:00: 04-05 are lent until about 14:15"
  ready "$label"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after 14:00 (after its heat waits): 04-05 are lent until about 14:15"
  echo "fbench $how $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp2-arm.sh $bin serve, want fbench $how $bin"
  bash "$T/tp2-arm.sh" "$label" "$bin" serve "" "$T/rf2dhook.sh"
  rc=$?
  heat "$label" "$t0"
  [ "$(grep -a -c "== arm $label " "$A" 2>/dev/null)" -gt "${n0:-0}" ] 2>/dev/null || stop "$label: no new line in logs/arms.log"
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  log "$label rc=$rc $last; $(grep -a " $label want " "$F/form.log" 2>/dev/null | tail -n 1 | cut -c21- | cut -c1-300)"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ -e "$F/stop-$label" ] && stop "$label: rank 0's look is not the arm's (fbench $how $bin): $(cat "$F/stop-$label")"
  grep -qx "fbench $label end" "$B/$label.log" 2>/dev/null || stop "$label's fbench: $(tail -n 2 "$B/$label.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  log "$label fbench: $(grep -a -E '  (cold|openai|decode|edit) ' "$B/$label.log" | cut -c21- | paste -sd'|' | cut -c1-1500)"
}
tables() { "${K[@]}" "$WC" > "$F/tables-$1" 2>&1 || stop "kzt.py: $(tail -n 1 "$F/tables-$1")"; log "tables-$1: $(grep -c . "$F/tables-$1") lines"; }
# rf2d: rf2c's copy as it stopped; the head's key is the new record that is not the price reference
P=$T/rf2c
REF=602963cdf94637bd04a0dab3608f390ad31b578d8c923b6b920b3b7d73de419a.bin
tables start
read -r KM rest < <(docker exec tp4-tf3 sh -c "cat $WC-rf2seed/names.txt")
case "${KM:-}" in ?*.bin) ;; *) stop "window-costs-rf2seed/names.txt names no record: '${KM:-}'" ;; esac
own=$(docker exec tp4-tf3 sh -c "cat $WC-rf2seed/words.txt")
case "$own" in *" ms at 1 row, "*"; experts "*) ;; *) stop "window-costs-rf2seed/words.txt holds no look words: '$own'" ;; esac
python3 "$T/flwords.py" "$F/tables-start" "$KM" "$own" > "$F/words-main.txt" 2>&1 \
  || stop "main's record: $(tail -n 1 "$F/words-main.txt")"
timed=$(cat "$P/own-$H" 2>/dev/null)
case "$timed" in *" ms at 1 row, "*"; experts "*) ;; *) stop "rf2c/own-$H holds no look words: '$timed'" ;; esac
KH=$(comm -13 <(cut -d' ' -f1 "$P/tables-mseeded" | sort) <(cut -d' ' -f1 "$F/tables-start" | sort) | grep '\.bin$' | grep -v -x -F "$REF")
[ "$(printf '%s\n' "$KH" | grep -c .)" = 1 ] || stop "rf2c's copy holds $(printf '%s\n' "$KH" | grep -c .) new records besides the reference, not one: $(echo $KH)"
python3 "$T/flwords.py" "$F/tables-start" "$KH" "$timed" > "$F/words-head.txt" 2>&1 \
  || stop "the head's record: $(tail -n 1 "$F/words-head.txt")"
python3 "$T/flwords.py" "$F/tables-start" "$REF" "$timed" > "$F/words-ref.txt" 2>&1 \
  || stop "the reference: $(tail -n 1 "$F/words-ref.txt")"
echo "$timed" > "$F/own-$H.timed"
log "rf2c's copy: $H's timed start saved $KH (and the reference $REF): $timed; $M's (main's record $KM): $own"
docker exec tp4-tf3 sh -c "cp -p $WC-rf2seed/main.bin $WC/$KH" || stop "seeding $H's key with main's record failed"
[ "$(docker exec tp4-tf3 sh -c "cat $WC/$KM" | sha256sum)" = "$(docker exec tp4-tf3 sh -c "cat $WC/$KH" | sha256sum)" ] \
  || stop "$H's key does not read back as main's record"
echo "$own" > "$F/own-$M"
echo "$own" > "$F/own-$H"
log "seeded the copy with main's record as $H's ($KH); both binaries' words: $own"
tables seeded
arm rf2dma "$M" kept
arm rf2dha "$H" kept
arm rf2dmb "$M" kept
arm rf2dhb "$H" kept
arm rf2dmc "$M" kept
python3 "$T/fsum.py" "$B" "$T/logs" rf2dma,rf2dmb,rf2dmc rf2dha,rf2dhb --json "$B/rf2d-fsum.json" > "$B/rf2d-fsum.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/rf2d-fsum.txt" | cut -c1-400)"
tables end
docker exec tp4-tf3 sh -c "[ ! -e $WC-rf2d ] && mv $WC $WC-rf2d && mv $WC-rf2cpre $WC" || stop "putting the first tables back failed"
moved=""
log "the chain's tables to window-costs-rf2d ($(docker exec tp4-tf3 sh -c "ls -A $WC-rf2d | wc -l") files); window-costs back with $(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l")"
log "rf2d-chain end: $LABELS (fbench/rf2d-fsum.txt); the cache as it began, the chain's beside it"
log "== rf2d-chain finished"
