#!/bin/bash
# rf2c: the 0091 PR's TP=2 gate again, for the head with the takesDecode floor (<the local ledger> 11-17 09:0x: 8 rows or more
# take glm_index_scores from 64 blocks; TP=2 and TP=3 again on the new head, aligned on main's prices). Made by
# <the local records> from rx/f2/rf2-chain.sh. On 04 (setsid): rf2c-chain.sh M16 H16 -- bin/me86a (main
# e86a0bed1d) and bin/ha4da (the PR head a4da5a4eb2) on TP=2's release settings (tp2-arm.sh on 04-05, rank2.base.env),
# fbench.sh as the work through rf2chook.sh. 04's window-costs moves aside to window-costs-rf2cpre with a copy in its
# place, and comes back at the end with the chain's beside it as window-costs-rf2c (a stop after the move says how to put
# it back). In the copy, main's record from rf2's timed start (window-costs-rf2seed: main.bin, names.txt's first name,
# words.txt; rf2b's seed) goes under me86a's key and must hold words.txt's words (flwords.py). The head times its
# windows once, uncounted (rf2cwh, also the first load): its start saves a record under its own key, which must be the
# one new record in the copy and hold the timed look's words. Then main's record is copied over that key and read back
# by sha, so every counted start is kept on main's window and draft prices and its expert form (classic), and only the
# binaries' code differs (<the local ledger> 11-11 12:1x (b), as rf2b). rf2c/own-ha4da.timed keeps the head's own words. One
# after another, 60 s apart:
#   rf2cwh head, timed, uncounted   rf2cma main  rf2cha head  rf2cmb main  rf2chb head  rf2cmc main
# The rest is rf2-chain.sh's: rf2chook.sh checks rank 0's look (timed, or kept with rf2c/own-BIN's words); rf2c/stop-now;
# 04-05's tf-preflight before each arm; 04's GPU lock; heat lines to rf2c/heat-LABEL.txt; no arm starts after 13:35
# (04-05 are lent until about 13:50; checked before and after its heat waits); fsum.py on the five counted arms (A =
# main's three, B = the head's two) into fbench/rf2c-fsum.txt and .json. Log logs/rf2c-chain.log; the last line
# "== rf2c-chain finished" or "== rf2c-chain stopped" (the line before says why). Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
F=$T/rf2c
B=$T/fbench
C=$T/logs/rf2c-chain.log
A=$T/logs/arms.log
M=me86a
H=ha4da
FSUM=db74650034aa894d
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp4-tf3 python3 /work/kzt.py)
HOSTS="host04 host05"
LABELS="rf2cwh rf2cma rf2cha rf2cmb rf2chb rf2cmc"
LAST=$(date -d "2026-11-17 13:35" +%s)   # no arm starts after this: 04-05 are lent until about 13:50 (<another local session>, 11-17 12:0x)
HEAT=/var/log/gb10-telemetry/heat-cap.log
moved=""
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() {
  local why=$*
  [ -n "$moved" ] && why="$why (the cache is aside: $moved; to put the first tables back, in tp4-tf3: mv $WC $WC-rf2cstop && mv $WC-rf2cpre $WC)"
  log "rf2c-chain STOP: $why"
  log "== rf2c-chain stopped"
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
for f in tp2-arm.sh fbench.sh mkprompts_f.py csum.py fsum.py rf2chook.sh heatcut.py kzt.py flwords.py sx4.sh rank2.base.env; do
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
docker exec tp4-tf3 sh -c "[ -d $WC ] && [ ! -e $WC-rf2cpre ] && [ ! -e $WC-rf2c ] && [ ! -e $WC-rf2cstop ] && [ -s $WC-rf2seed/main.bin ] && [ -s $WC-rf2seed/names.txt ] && [ -s $WC-rf2seed/words.txt ]" \
  || stop "tp4-tf3: no $WC or no window-costs-rf2seed (main.bin, names.txt, words.txt), or window-costs-rf2cpre, -rf2c or -rf2cstop is there"
mkdir -p "$F"
exec 9> /tmp/glm-perf-04.lock
flock -n 9 || stop "04's GPU lock (/tmp/glm-perf-04.lock) is held"
log "rf2c-chain begin, holding 04's GPU lock: $M $1, $H $2; tp2-arm $(sha256sum "$T/tp2-arm.sh" | cut -c1-16) fbench $(sha256sum "$T/fbench.sh" | cut -c1-16) fsum $(sha256sum "$T/fsum.py" | cut -c1-16) rf2chook $(sha256sum "$T/rf2chook.sh" | cut -c1-16) heatcut $(sha256sum "$T/heatcut.py" | cut -c1-16) kzt $(sha256sum "$T/kzt.py" | cut -c1-16) rank2.base.env $(sha256sum "$T/rank2.base.env" | cut -c1-16)"
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
    [ -n "$why" ] && stop "$l: not ready:$why (rf2c/preflight.log)"
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
arm() {   # LABEL BIN HOW: tp2-arm.sh with rf2chook.sh (then fbench.sh) on the release settings
  local label=$1 bin=$2 how=$3 rc last n0 t0
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rf2c/stop-now: $(head -c 200 "$F/stop-now"))"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after 13:35: 04-05 are lent until about 13:50"
  ready "$label"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after 13:35 (after its heat waits): 04-05 are lent until about 13:50"
  echo "fbench $how $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp2-arm.sh $bin serve, want fbench $how $bin"
  bash "$T/tp2-arm.sh" "$label" "$bin" serve "" "$T/rf2chook.sh"
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
tables before
docker exec tp4-tf3 sh -c "[ ! -e $WC-rf2cpre ] && mv $WC $WC-rf2cpre && cp -a $WC-rf2cpre $WC" || stop "setting the cache aside failed"
moved="window-costs-rf2cpre"
log "window-costs to window-costs-rf2cpre ($(docker exec tp4-tf3 sh -c "ls -A $WC-rf2cpre | wc -l") files); window-costs now a copy of it ($(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l") files)"
tables start
# rf2c: main's record (rf2wm's, window-costs-rf2seed) under me86a's key; the new head times its own once, then gets it
read -r KM rest < <(docker exec tp4-tf3 sh -c "cat $WC-rf2seed/names.txt")
case "${KM:-}" in ?*.bin) ;; *) stop "window-costs-rf2seed/names.txt names no record: '${KM:-}'" ;; esac
own=$(docker exec tp4-tf3 sh -c "cat $WC-rf2seed/words.txt")
case "$own" in *" ms at 1 row, "*"; experts "*) ;; *) stop "window-costs-rf2seed/words.txt holds no look words: '$own'" ;; esac
docker exec tp4-tf3 sh -c "cp -p $WC-rf2seed/main.bin $WC/$KM" || stop "seeding me86a's key with main's record failed"
tables mseeded
python3 "$T/flwords.py" "$F/tables-mseeded" "$KM" "$own" > "$F/words-main.txt" 2>&1 \
  || stop "main's record: $(tail -n 1 "$F/words-main.txt")"
echo "$own" > "$F/own-$M"
log "seeded the copy with main's record as $M's ($KM): $own"
arm rf2cwh "$H" timed
tables timed
KH=$(comm -13 <(cut -d' ' -f1 "$F/tables-mseeded" | sort) <(cut -d' ' -f1 "$F/tables-timed" | sort) | grep '\.bin$')
[ "$(printf '%s\n' "$KH" | grep -c .)" = 1 ] || stop "the head's timed start left $(printf '%s\n' "$KH" | grep -c .) new records, not one: $(echo $KH)"
timed=$(cat "$F/own-$H" 2>/dev/null)
python3 "$T/flwords.py" "$F/tables-timed" "$KH" "$timed" > "$F/words-head.txt" 2>&1 \
  || stop "the head's record: $(tail -n 1 "$F/words-head.txt")"
cp "$F/own-$H" "$F/own-$H.timed"
log "$H's timed start saved $KH: $timed; $M's (main's record): $own"
docker exec tp4-tf3 sh -c "cp -p $WC-rf2seed/main.bin $WC/$KH" || stop "seeding $H's key with main's record failed"
[ "$(docker exec tp4-tf3 sh -c "cat $WC/$KM" | sha256sum)" = "$(docker exec tp4-tf3 sh -c "cat $WC/$KH" | sha256sum)" ] \
  || stop "$H's key does not read back as main's record"
echo "$own" > "$F/own-$H"
log "seeded the copy with main's record as $H's ($KH); both binaries' words: $own"
tables seeded
arm rf2cma "$M" kept
arm rf2cha "$H" kept
arm rf2cmb "$M" kept
arm rf2chb "$H" kept
arm rf2cmc "$M" kept
python3 "$T/fsum.py" "$B" "$T/logs" rf2cma,rf2cmb,rf2cmc rf2cha,rf2chb --json "$B/rf2c-fsum.json" > "$B/rf2c-fsum.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/rf2c-fsum.txt" | cut -c1-400)"
tables end
docker exec tp4-tf3 sh -c "[ ! -e $WC-rf2c ] && mv $WC $WC-rf2c && mv $WC-rf2cpre $WC" || stop "putting the first tables back failed"
moved=""
log "the chain's tables to window-costs-rf2c ($(docker exec tp4-tf3 sh -c "ls -A $WC-rf2c | wc -l") files); window-costs back with $(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l")"
log "rf2c-chain end: $LABELS (fbench/rf2c-fsum.txt); the cache as it began, the chain's beside it"
log "== rf2c-chain finished"
