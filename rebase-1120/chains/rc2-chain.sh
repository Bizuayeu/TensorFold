#!/bin/bash
# rc2: #3689's TP=2 receipts again, for the head rebased onto #3776 (GLM decode kernels as dependent launches; <the local ledger>
# 11-20 07:1x, 07:2x and 08:1x; <the local records>). Made by <the local records> from
# <the local records>. On 04 (setsid): RC2_LAST="YYYY-MM-DD HH:MM" rc2-chain.sh M16 H16 --
# bin/mb0b17 (main b0b17e8962) and bin/h50c66 (the rebased head 50c6634539) in ~/LLLLM/glm-work/tp4 on TP=2's release
# settings (tp2-arm.sh on 04-05, rank2.base.env), fbench.sh as the work through rc2hook.sh. 04's window-costs moves aside
# to window-costs-rc2pre with a copy in its place, and comes back at the end with the chain's beside it as
# window-costs-rc2. First each binary times its windows once, uncounted, the head first; the two looks may name
# different expert forms (a load-time split, logged; the seed aligns them); then seedpick.py (--same-form only when the
# forms match) names main's record and the head's by rc2/own-*'s words, the names
# the cache held before the chain (rc2/tables-start) told apart from the new ones, and main's record files are copied
# over the head's key and read back by sha, so every counted start is kept on main's window and draft prices and only
# the binaries' code differs (<the local ledger> 11-11 12:1x (b)); rc2/own-BIN.timed keep each binary's own words. One after
# another, 60 s apart:
#   rc2wh head, timed  rc2wm main, timed  rc2ma main  rc2ha head  rc2mb main  rc2hb head  rc2mc main  rc2hc head  rc2md main
# The head starts four times, each with fbench (<a local note> §3). The rest is rf2-chain.sh's: rc2/stop-now; 04's
# and 05's tf-preflight before each arm; 04's GPU lock; heat lines to rc2/heat-LABEL.txt; fsum.py on the seven counted
# arms (A = main's four, B = the head's three) into fbench/rc2-fsum.txt and .json, and on all nine for the tokens into
# fbench/rc2-fsum-all.*; no arm starts after RC2_LAST. Log logs/rc2-chain.log; the last line "== rc2-chain finished" or
# "== rc2-chain stopped". Never edit while it runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
F=$T/rc2
B=$T/fbench
C=$T/logs/rc2-chain.log
A=$T/logs/arms.log
M=mb0b17
H=h50c66
FSUM=db74650034aa894d
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp4-tf3 python3 /work/kzt.py)
HOSTS="host04 host05"
LABELS="rc2wh rc2wm rc2ma rc2ha rc2mb rc2hb rc2mc rc2hc rc2md"
LAST=$(date -d "${RC2_LAST:-}" +%s 2>/dev/null) || LAST=0   # no arm starts after RC2_LAST (the night chain sets it)
HEAT=/var/log/gb10-telemetry/heat-cap.log
moved=""
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() {
  local why=$*
  [ -n "$moved" ] && why="$why (the cache is aside: $moved; to put the first tables back, in tp4-tf3: mv $WC $WC-rc2stop && mv $WC-rc2pre $WC)"
  log "rc2-chain STOP: $why"
  log "== rc2-chain stopped"
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
for f in tp2-arm.sh fbench.sh mkprompts_f.py csum.py fsum.py seedpick.py rc2hook.sh heatcut.py kzt.py sx4.sh rank2.base.env; do
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
docker exec tp4-tf3 sh -c "[ -d $WC ] && [ ! -e $WC-rc2pre ] && [ ! -e $WC-rc2 ] && [ ! -e $WC-rc2stop ]" \
  || stop "tp4-tf3: no $WC, or window-costs-rc2pre, -rc2 or -rc2stop is there"
mkdir -p "$F"
exec 9> /tmp/glm-perf-04.lock
flock -n 9 || stop "04's GPU lock (/tmp/glm-perf-04.lock) is held"
log "rc2-chain begin, holding 04's GPU lock: $M $1, $H $2; tp2-arm $(sha256sum "$T/tp2-arm.sh" | cut -c1-16) fbench $(sha256sum "$T/fbench.sh" | cut -c1-16) fsum $(sha256sum "$T/fsum.py" | cut -c1-16) rc2hook $(sha256sum "$T/rc2hook.sh" | cut -c1-16) heatcut $(sha256sum "$T/heatcut.py" | cut -c1-16) kzt $(sha256sum "$T/kzt.py" | cut -c1-16) rank2.base.env $(sha256sum "$T/rank2.base.env" | cut -c1-16)"
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
    [ -n "$why" ] && stop "$l: not ready:$why (rc2/preflight.log)"
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
arm() {   # LABEL BIN HOW: tp2-arm.sh with rc2hook.sh (then fbench.sh) on the release settings
  local label=$1 bin=$2 how=$3 rc last n0 t0
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rc2/stop-now: $(head -c 200 "$F/stop-now"))"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after RC2_LAST (${RC2_LAST:-unset})"
  ready "$label"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after RC2_LAST (${RC2_LAST:-unset}, after its heat waits)"
  echo "fbench $how $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp2-arm.sh $bin serve, want fbench $how $bin"
  bash "$T/tp2-arm.sh" "$label" "$bin" serve "" "$T/rc2hook.sh"
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
docker exec tp4-tf3 sh -c "[ ! -e $WC-rc2pre ] && mv $WC $WC-rc2pre && cp -a $WC-rc2pre $WC" || stop "setting the cache aside failed"
moved="window-costs-rc2pre"
log "window-costs to window-costs-rc2pre ($(docker exec tp4-tf3 sh -c "ls -A $WC-rc2pre | wc -l") files); window-costs now a copy of it ($(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l") files)"
tables start
arm rc2wh "$H" timed
arm rc2wm "$M" timed
fh=$(sed -n 's/^.*; experts //p' "$F/own-$H" 2>/dev/null) fm=$(sed -n 's/^.*; experts //p' "$F/own-$M" 2>/dev/null)
log "expert forms: $M '${fm:-none}', $H '${fh:-none}'"
[ -n "$fm" ] && [ -n "$fh" ] || stop "a timed look names no expert form: $M '${fm:-none}', $H '${fh:-none}'"
sf=""
[ "$fm" = "$fh" ] && sf=--same-form
log "seedpick ${sf:-plain (a split: the seed aligns the counted arms on the form of main)}"
cp "$F/own-$M" "$F/own-$M.timed" && cp "$F/own-$H" "$F/own-$H.timed" || stop "keeping the timed words failed"
tables timed
# rc2: aligned on main's TP=2 record (<the local ledger> 11-11 12:1x (b), form and prices): the head's key gets main's record
# (rc2wm's), main's own key stays; both binaries' words are main's
P=$F/window-costs-pick
mkdir -p "$P" && docker exec tp4-tf3 sh -c "cd $WC && tar cf - ." | tar xf - -C "$P" || stop "copying the cache out failed"
picked=$(python3 "$T/seedpick.py" $sf "$P" "$F/own-$M.timed" "$F/own-$H.timed" "$F/tables-start" 2> "$F/seedpick.txt") \
  || stop "seedpick: $(tail -n 1 "$F/seedpick.txt")"
read -r KM KH <<< "$picked"
case "$KM/$KH" in ?*.bin/?*.bin) ;; *) stop "seedpick named '$picked'" ;; esac
docker exec tp4-tf3 sh -c "cd $WC && for x in ${KM%.bin}*; do cp -p \$x ${KH%.bin}\${x#${KM%.bin}} || exit 1; done" \
  || stop "seeding $H's key with main's record failed"
[ "$(docker exec tp4-tf3 sh -c "cat $WC/$KM" | sha256sum)" = "$(docker exec tp4-tf3 sh -c "cat $WC/$KH" | sha256sum)" ] \
  || stop "$H's key does not read back as main's record"
own=$(cat "$F/own-$M.timed" 2>/dev/null)
case "$own" in *" ms at 1 row, "*"; experts "*) ;; *) stop "rc2/own-$M.timed holds no look words: '$own'" ;; esac
echo "$own" > "$F/own-$M"
echo "$own" > "$F/own-$H"
log "seeded the copy: main's record $KM over $H's key $KH; both binaries' words: $own"
tables seeded
arm rc2ma "$M" kept
arm rc2ha "$H" kept
arm rc2mb "$M" kept
arm rc2hb "$H" kept
arm rc2mc "$M" kept
arm rc2hc "$H" kept
arm rc2md "$M" kept
python3 "$T/fsum.py" "$B" "$T/logs" rc2ma,rc2mb,rc2mc,rc2md rc2ha,rc2hb,rc2hc --json "$B/rc2-fsum.json" > "$B/rc2-fsum.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/rc2-fsum.txt" | cut -c1-400)"
# all nine starts (the timed two on their own prices) for the greedy tokens; its speed figures are not read
python3 "$T/fsum.py" "$B" "$T/logs" rc2wm,rc2ma,rc2mb,rc2mc,rc2md rc2wh,rc2ha,rc2hb,rc2hc --json "$B/rc2-fsum-all.json" > "$B/rc2-fsum-all.txt" 2>&1
log "fsum all nine rc=$?: $(tail -n 1 "$B/rc2-fsum-all.txt" | cut -c1-400)"
tables end
docker exec tp4-tf3 sh -c "[ ! -e $WC-rc2 ] && mv $WC $WC-rc2 && mv $WC-rc2pre $WC" || stop "putting the first tables back failed"
moved=""
log "the chain's tables to window-costs-rc2 ($(docker exec tp4-tf3 sh -c "ls -A $WC-rc2 | wc -l") files); window-costs back with $(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l")"
log "rc2-chain end: $LABELS (fbench/rc2-fsum.txt); the cache as it began, the chain's beside it"
log "== rc2-chain finished"
