#!/bin/bash
# rc3: #3689's TP=3 receipts again, for the head rebased onto #3776 (GLM decode kernels as dependent launches; <the local ledger>
# 11-20 07:1x and 07:2x, local choices; <the local records>). Made by
# <the local records> from <the local records>. On 01 (setsid): rc3-chain.sh M16
# H16 -- WORK/bin/mb0b17 (main b0b17e8962) and WORK/bin/h50c66 (the rebased head 50c6634539), WORK ~/LLLLM/glm-work/f0091,
# through tp3-arm.sh in tp3f (made fresh by rc3-setup.sh: its window-costs starts empty) with rc3hook.sh, then fbench.sh.
# First each binary times its windows once, uncounted, the head first (rf3-chain.sh's order: the reference key a timed
# start saves after its own then holds main's bytes), and the two timed looks must name one expert form. Then tp3f's
# window-costs is copied out to rc3/window-costs-pick, seedpick.py --same-form names main's record and the head's by
# rc3/own-*'s words (main's own key, not the reference key: the rf3d stop of 11-17 12:3x), tp3f's tables are kept as
# window-costs-rc3pre, main's record is copied over the head's key and read back by sha, so every counted start is kept
# on main's window and draft prices and only the binaries' code differs (<the local ledger> 11-11 12:1x (b)); rc3/own-BIN.timed
# keep each binary's own words. One after another, 60 s apart:
#   rc3wh head, timed  rc3wm main, timed  rc3ma main  rc3ha head  rc3mb main  rc3hb head  rc3mc main  rc3hc head  rc3md main
# The head starts four times and every start runs fbench, so the greedy tokens are compared over four of its starts
# (<a local note> §3). The rest is rf3c-chain.sh's: rc3/stop-now; 01-03's tf-preflight before each arm; 01's GPU
# lock; heat lines to rc3/heat-LABEL.txt; fsum.py on the seven counted arms (A = main's four, B = the head's three) into
# fbench/rc3-fsum.txt and .json, and on all nine (A = main's five, B = the head's four) into fbench/rc3-fsum-all.txt and
# .json for the tokens; tp3f's window-costs copied out to rc3/window-costs-end. Log WORK/logs/rc3-chain.log; the last
# line "== rc3-chain finished" or "== rc3-chain stopped" (the line before says why). Never edit while it runs.
set -u
G=$HOME/LLLLM/models/glm53-nvfp4
W=$HOME/LLLLM/glm-work/f0091
F=$W/rc3
B=$W/fbench
C=$W/logs/rc3-chain.log
A=$W/logs/arms.log
M=mb0b17
H=h50c66
FSUM=db74650034aa894d
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp3f python3 /work/kzt.py)
X="bash $G/tf-work/v292/sx.sh"
HOSTS="host01 host02 host03"
LABELS="rc3wh rc3wm rc3ma rc3ha rc3mb rc3hb rc3mc rc3hc rc3md"
HEAT=/var/log/gb10-telemetry/heat-cap.log
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "rc3-chain STOP: $*"; log "== rc3-chain stopped"; exit 1; }
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
for f in tp3-arm.sh fbench.sh mkprompts_f.py csum.py fsum.py rc3hook.sh heatcut.py kzt.py cluster.tp3f.env seedpick.py; do
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
docker exec tp3f sh -c "if [ -d $WC ]; then echo there; else mkdir -p $WC && echo made; fi" > "$W/logs/rc3-wc.txt" 2>&1 \
  || stop "tp3f: $WC: $(tail -n 1 "$W/logs/rc3-wc.txt")"
mkdir -p "$F"
exec 9> /tmp/glm-perf-01.lock
flock -n 9 || stop "01's GPU lock (/tmp/glm-perf-01.lock) is held"
log "rc3-chain begin, holding 01's GPU lock: $M $1, $H $2; tp3f's window-costs $(cat "$W/logs/rc3-wc.txt"); tp3-arm $(sha256sum "$W/tp3-arm.sh" | cut -c1-16) fbench $(sha256sum "$W/fbench.sh" | cut -c1-16) fsum $(sha256sum "$W/fsum.py" | cut -c1-16) rc3hook $(sha256sum "$W/rc3hook.sh" | cut -c1-16) heatcut $(sha256sum "$W/heatcut.py" | cut -c1-16) kzt $(sha256sum "$W/kzt.py" | cut -c1-16) cluster.tp3f.env $(sha256sum "$W/cluster.tp3f.env" | cut -c1-16)"
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
    [ -n "$why" ] && stop "$l: not ready:$why (rc3/preflight.log)"
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
arm() {   # LABEL BIN HOW: tp3-arm.sh with rc3hook.sh (then fbench.sh) on the release's TP=3 settings
  local label=$1 bin=$2 how=$3 rc last n0 t0
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rc3/stop-now: $(head -c 200 "$F/stop-now"))"
  ready "$label"
  echo "fbench $how $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp3-arm.sh $bin, want fbench $how $bin"
  bash "$W/tp3-arm.sh" "$label" "$bin" "$W/rc3hook.sh"
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
# rc3: each binary times its windows once on tp3f's fresh table, uncounted, the head first (the order rf3 ran them)
arm rc3wh "$H" timed
arm rc3wm "$M" timed
fh=$(sed -n 's/^.*; experts //p' "$F/own-$H" 2>/dev/null) fm=$(sed -n 's/^.*; experts //p' "$F/own-$M" 2>/dev/null)
log "expert forms: $M '${fm:-none}', $H '${fh:-none}'"
[ -n "$fm" ] && [ "$fm" = "$fh" ] || stop "the timed looks name different expert forms: $M '${fm:-none}', $H '${fh:-none}'"
cp "$F/own-$M" "$F/own-$M.timed" && cp "$F/own-$H" "$F/own-$H.timed" || stop "keeping the timed words failed"
tables timed
# rc3: aligned on main's TP=3 record (<the local ledger> 11-11 12:1x (b), form and prices): the head's key gets main's
# record (rc3wm's), main's own key stays; both binaries' words are main's
P=$F/window-costs-pick
mkdir -p "$P" && docker exec tp3f sh -c "cd $WC && tar cf - ." | tar xf - -C "$P" || stop "copying tp3f's window-costs out failed"
picked=$(python3 "$W/seedpick.py" --same-form "$P" "$F/own-$M.timed" "$F/own-$H.timed" 2> "$F/seedpick.txt") \
  || stop "seedpick: $(tail -n 1 "$F/seedpick.txt")"
read -r KM KH <<< "$picked"
case "$KM/$KH" in ?*.bin/?*.bin) ;; *) stop "seedpick named '$picked'" ;; esac
docker exec tp3f sh -c "[ ! -e $WC-rc3pre ] && cp -a $WC $WC-rc3pre && cp -p $WC/$KM $WC/$KH" || stop "seeding $H's key with main's record failed"
[ "$(docker exec tp3f sh -c "cat $WC/$KM" | sha256sum)" = "$(docker exec tp3f sh -c "cat $WC/$KH" | sha256sum)" ] \
  || stop "$H's key does not read back as main's record"
own=$(cat "$F/own-$M.timed" 2>/dev/null)
case "$own" in *" ms at 1 row, "*"; experts "*) ;; *) stop "rc3/own-$M.timed holds no look words: '$own'" ;; esac
echo "$own" > "$F/own-$M"
echo "$own" > "$F/own-$H"
log "seeded tp3f's table: main's record $KM over $H's key $KH (the first tables kept as window-costs-rc3pre); both binaries' words: $own"
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
# all nine starts (the timed two on their own prices) for the greedy tokens; its speed figures are not read
python3 "$W/fsum.py" "$B" "$W/logs" rc3wm,rc3ma,rc3mb,rc3mc,rc3md rc3wh,rc3ha,rc3hb,rc3hc --json "$B/rc3-fsum-all.json" > "$B/rc3-fsum-all.txt" 2>&1
log "fsum all nine rc=$?: $(tail -n 1 "$B/rc3-fsum-all.txt" | cut -c1-400)"
tables end
mkdir -p "$F/window-costs-end" && docker exec tp3f sh -c "cd $WC && tar cf - ." | tar xf - -C "$F/window-costs-end" \
  || stop "copying tp3f's window-costs out failed"
log "tp3f's window-costs copied out to rc3/window-costs-end ($(ls -A "$F/window-costs-end" | wc -l) files)"
log "rc3-chain end: $LABELS (fbench/rc3-fsum.txt); tp3f stays until rc3-done.sh"
log "== rc3-chain finished"
