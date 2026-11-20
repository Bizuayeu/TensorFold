#!/bin/bash
# rc3b: the rest of rc3 (#3689's TP=3 receipts after the rebase onto #3776), after rc3 stopped at 09:43:19 on split expert
# forms: its two timed starts ran (rc3wh the head, classic: 29.49 ms at 1 row, 44.47 at 4, 129.97 at 32, a draft 1.49 ms;
# rc3wm main, factored: 28.81, 43.84, 129.58, 1.48), and fsum over the two found every cell, greedy row and cold prompt
# the same. As the 11-16 rx and rf2 splits went on (<the local ledger> 11-11 12:1x (b), the 11:5x form), every counted start is
# kept on main's record, form and prices. Made by <the local records> from
# <the local records>. On 01 (setsid): rc3b-chain.sh M16 H16 -- WORK/bin/mb0b17 (main
# b0b17e8962) and WORK/bin/h50c66 (the rebased head 50c6634539), WORK ~/LLLLM/glm-work/f0091, through tp3-arm.sh in tp3f
# with rc3bhook.sh, then fbench.sh. tp3f's table must hold exactly the three records rc3's timed starts saved; rc3/own-*
# must hold their looks' words. Then the table is copied out to rc3b/window-costs-pick, seedpick.py (--same-form only
# when the two forms match) names main's record and the head's, tp3f's tables are kept as window-costs-rc3bpre, main's
# record is copied over the head's key and read back by sha; rc3b/own-BIN.timed keep rc3's words. One after another, 60 s
# apart: rc3ma main  rc3ha head  rc3mb main  rc3hb head  rc3mc main  rc3hc head  rc3md main
# The rest is rf3c-chain.sh's: rc3b/stop-now; 01-03's tf-preflight before each arm; 01's GPU lock; heat lines to
# rc3b/heat-LABEL.txt; fsum.py on the seven counted arms (A = main's four, B = the head's three) into fbench/rc3-fsum.txt
# and .json, and on all nine (with rc3wm and rc3wh) into fbench/rc3-fsum-all.*; tp3f's window-costs copied out to
# rc3b/window-costs-end. Log WORK/logs/rc3b-chain.log; the last line "== rc3b-chain finished" or "== rc3b-chain stopped".
# Never edit while it runs.
set -u
G=$HOME/LLLLM/models/glm53-nvfp4
W=$HOME/LLLLM/glm-work/f0091
F=$W/rc3b
B=$W/fbench
C=$W/logs/rc3b-chain.log
A=$W/logs/arms.log
M=mb0b17
H=h50c66
FSUM=db74650034aa894d
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp3f python3 /work/kzt.py)
X="bash $G/tf-work/v292/sx.sh"
HOSTS="host01 host02 host03"
LABELS="rc3ma rc3ha rc3mb rc3hb rc3mc rc3hc rc3md"
HEAT=/var/log/gb10-telemetry/heat-cap.log
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "rc3b-chain STOP: $*"; log "== rc3b-chain stopped"; exit 1; }
on() {   # HOST CMD: CMD on HOST (01 here, 02 and 03 over sx.sh)
  local h=$1
  shift
  if [ "$h" = host01 ]; then bash -c "$*"; else $X "$h" "$*" < /dev/null; fi
}
mkdir -p "$W/logs"
[ $# = 2 ] || stop "give two sha16s: WORK/bin/$M's native, then WORK/bin/$H's"
for p in "$M $1" "$H $2"; do
  b=${p% *} want=${p#* }
  got=$(sha256sum "$W/bin/$b/tensorfold-native" 2>/dev/null | cut -c1-16)
  [ "$got" = "$want" ] || stop "WORK/bin/$b/tensorfold-native is '$got', not $want"
done
got=$(sha256sum "$W/fsum.py" 2>/dev/null | cut -c1-16)
[ "$got" = "$FSUM" ] || stop "fsum.py is '$got', not the 11-15 fix $FSUM"
[ -e "$F" ] && stop "$F exists"
for l in $LABELS; do
  for f in "$W/logs/serve-r0-$l.log" "$B/$l.log" "$B/$l-decode.json"; do
    [ -e "$f" ] && stop "$f exists"
  done
done
for f in tp3-arm.sh fbench.sh mkprompts_f.py csum.py fsum.py rc3bhook.sh heatcut.py kzt.py cluster.tp3f.env seedpick.py rc3/own-mb0b17 rc3/own-h50c66; do
  [ -s "$W/$f" ] || stop "no WORK/$f"
done
for f in tools/bench_cache.py tools/bench_openai.py tools/bench_concurrent.py tools/bench_prompts.json b2-edit.json corpus.txt; do
  [ -s "$B/$f" ] || stop "no fbench/$f"
done
img=""
for h in $HOSTS; do
  i=$(on "$h" "docker inspect -f '{{.State.Running}} {{.Image}}' tp3f 2>/dev/null | cut -c1-24")
  log "host $h: tp3f ${i:-none}"
  case "$i" in "true sha256:"*) ;; *) stop "$h: tp3f is not running ('${i:-none}')" ;; esac
  [ -z "$img" ] && img=$i
  [ "$i" = "$img" ] || stop "$h's tp3f image is not host01's"
done
docker exec tp3f sh -c "if [ -d $WC ]; then echo there; else mkdir -p $WC && echo made; fi" > "$W/logs/rc3b-wc.txt" 2>&1 \
  || stop "tp3f: $WC: $(tail -n 1 "$W/logs/rc3b-wc.txt")"
mkdir -p "$F"
exec 9> /tmp/glm-perf-01.lock
flock -n 9 || stop "01's GPU lock (/tmp/glm-perf-01.lock) is held"
log "rc3b-chain begin, holding 01's GPU lock: $M $1, $H $2; tp3f's window-costs $(cat "$W/logs/rc3b-wc.txt"); tp3-arm $(sha256sum "$W/tp3-arm.sh" | cut -c1-16) fbench $(sha256sum "$W/fbench.sh" | cut -c1-16) fsum $(sha256sum "$W/fsum.py" | cut -c1-16) rc3bhook $(sha256sum "$W/rc3bhook.sh" | cut -c1-16) heatcut $(sha256sum "$W/heatcut.py" | cut -c1-16) kzt $(sha256sum "$W/kzt.py" | cut -c1-16) cluster.tp3f.env $(sha256sum "$W/cluster.tp3f.env" | cut -c1-16)"
ready() {   # LABEL: 01-03's tf-preflight; heat or an unread look waits
  local l=$1 i h out rc why hot
  for i in $(seq 1 30); do
    why="" hot=""
    for h in $HOSTS; do
      if [ "$h" = host01 ]; then   # the chain holds 01's lock itself (taken free at the begin line)
        out=$(python3 "$HOME/LLLLM/ops/tf-rules/hosts/tf-preflight" --label "$l" --skip lock 2>&1)
      else
        out=$($X "$h" "python3 ~/LLLLM/ops/tf-rules/hosts/tf-preflight --label $l" < /dev/null 2>&1)
      fi
      rc=$?
      echo "$(date '+%F %T') $l $h rc=$rc $(echo "$out" | tail -n 1)" >> "$F/preflight.log"
      [ $rc = 0 ] && continue
      out=$(echo "$out" | tail -n 1 | python3 -c 'import json,sys; print(",".join(json.load(sys.stdin)["failed"]) or "none")' 2>/dev/null)
      case "${out:-unread}" in heat) hot="$hot $h" ;; unread) hot="$hot $h(unread)" ;; *) why="$why $h:$out" ;; esac
    done
    [ -n "$why" ] && stop "$l: not ready:$why (rc3b/preflight.log)"
    [ -z "$hot" ] && return 0
    log "$l: heat on$hot (look $i of 30); again in 60 s"
    sleep 60
  done
  stop "$l: still hot or unread on$hot after 30 looks"
}
heat() {   # LABEL T0: 01-03's heat-cap lines since T0
  local l=$1 t0=$2 h s sums=""
  for h in $HOSTS; do
    s=$(on "$h" "tail -n 400 $HEAT 2>/dev/null" | python3 "$W/heatcut.py" "$t0" "$h" | tee -a "$F/heat-$l.txt" | tail -n 1)
    sums="$sums; $s"
  done
  log "$l heat${sums}"
}
arm() {   # LABEL BIN HOW: tp3-arm.sh with rc3bhook.sh (then fbench.sh) on the release's TP=3 settings
  local label=$1 bin=$2 how=$3 rc last n0 t0
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rc3b/stop-now: $(head -c 200 "$F/stop-now"))"
  ready "$label"
  echo "fbench $how $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp3-arm.sh $bin, want fbench $how $bin"
  bash "$W/tp3-arm.sh" "$label" "$bin" "$W/rc3bhook.sh"
  rc=$?
  heat "$label" "$t0"
  [ "$(grep -a -c "== arm $label " "$A" 2>/dev/null)" -gt "${n0:-0}" ] 2>/dev/null || stop "$label: no new line in WORK/logs/arms.log"
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  log "$label rc=$rc $last; $(grep -a " $label want " "$F/form.log" 2>/dev/null | tail -n 1 | cut -c21- | cut -c1-300)"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ -e "$F/stop-$label" ] && stop "$label: rank 0's look is not the arm's (fbench $how $bin): $(cat "$F/stop-$label")"
  grep -qx "fbench $label end" "$B/$label.log" 2>/dev/null || stop "$label's fbench: $(tail -n 2 "$B/$label.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  log "$label fbench: $(grep -a -E '  (cold|openai|decode|edit) ' "$B/$label.log" | cut -c21- | paste -sd'|' | cut -c1-1500)"
}
tables() { "${K[@]}" "$WC" > "$F/tables-$1" 2>&1 || stop "kzt.py: $(tail -n 1 "$F/tables-$1")"; log "tables-$1: $(grep -c . "$F/tables-$1") lines"; }
tables start
# rc3b: rc3's two timed starts left exactly three records (the head's, main's and the reference key's); their looks'
# words are rc3/own-*. Aligned on main's TP=3 record (<the local ledger> 11-11 12:1x (b)): the head's key gets main's record
nb=$(grep -c '[.]bin' "$F/tables-start")
[ "$nb" = 3 ] || stop "tp3f's table holds $nb records, not rc3's three (rc3b/tables-start)"
for b in "$M" "$H"; do
  case "$(cat "$W/rc3/own-$b" 2>/dev/null)" in *" ms at 1 row, "*"; experts "*) ;; *) stop "rc3/own-$b holds no look words" ;; esac
  cp "$W/rc3/own-$b" "$F/own-$b.timed" || stop "keeping rc3/own-$b failed"
done
fm=$(sed -n 's/^.*; experts //p' "$F/own-$M.timed") fh=$(sed -n 's/^.*; experts //p' "$F/own-$H.timed")
sf=""
[ "$fm" = "$fh" ] && sf=--same-form
log "rc3's timed looks: $M '$fm', $H '$fh'; seedpick ${sf:-plain (a split)}"
P=$F/window-costs-pick
mkdir -p "$P" && docker exec tp3f sh -c "cd $WC && tar cf - ." | tar xf - -C "$P" || stop "copying tp3f's window-costs out failed"
picked=$(python3 "$W/seedpick.py" $sf "$P" "$F/own-$M.timed" "$F/own-$H.timed" 2> "$F/seedpick.txt") \
  || stop "seedpick: $(tail -n 1 "$F/seedpick.txt")"
read -r KM KH <<< "$picked"
case "$KM/$KH" in ?*.bin/?*.bin) ;; *) stop "seedpick named '$picked'" ;; esac
docker exec tp3f sh -c "[ ! -e $WC-rc3bpre ] && cp -a $WC $WC-rc3bpre && cp -p $WC/$KM $WC/$KH" || stop "seeding $H's key with main's record failed"
[ "$(docker exec tp3f sh -c "cat $WC/$KM" | sha256sum)" = "$(docker exec tp3f sh -c "cat $WC/$KH" | sha256sum)" ] \
  || stop "$H's key does not read back as main's record"
own=$(cat "$F/own-$M.timed")
echo "$own" > "$F/own-$M"
echo "$own" > "$F/own-$H"
log "seeded tp3f's table: main's record $KM over $H's key $KH (the tables before kept as window-costs-rc3bpre); both binaries' words: $own"
tables seeded
arm rc3ma "$M" kept
arm rc3ha "$H" kept
arm rc3mb "$M" kept
arm rc3hb "$H" kept
arm rc3mc "$M" kept
arm rc3hc "$H" kept
arm rc3md "$M" kept
python3 "$W/fsum.py" "$B" "$W/logs" rc3ma,rc3mb,rc3mc,rc3md rc3ha,rc3hb,rc3hc --json "$B/rc3-fsum.json" > "$B/rc3-fsum.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/rc3-fsum.txt" | cut -c1-400)"
# all nine starts (rc3's timed two on their own prices and forms) for the greedy tokens; its speed figures are not read
python3 "$W/fsum.py" "$B" "$W/logs" rc3wm,rc3ma,rc3mb,rc3mc,rc3md rc3wh,rc3ha,rc3hb,rc3hc --json "$B/rc3-fsum-all.json" > "$B/rc3-fsum-all.txt" 2>&1
log "fsum all nine rc=$?: $(tail -n 1 "$B/rc3-fsum-all.txt" | cut -c1-400)"
tables end
mkdir -p "$F/window-costs-end" && docker exec tp3f sh -c "cd $WC && tar cf - ." | tar xf - -C "$F/window-costs-end" \
  || stop "copying tp3f's window-costs out failed"
log "tp3f's window-costs copied out to rc3b/window-costs-end ($(ls -A "$F/window-costs-end" | wc -l) files)"
log "rc3b-chain end: $LABELS (fbench/rc3-fsum.txt, rc3-fsum-all.txt); tp3f stays until rc3-done.sh"
log "== rc3b-chain finished"
