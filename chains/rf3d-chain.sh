#!/bin/bash
# rf3d: the 0091 PR's TP=3 gate again, for the head with the takesDecode floor (<the local ledger> 11-17 09:0x: 8 rows or more
# take glm_index_scores from 64 blocks; TP=2 and TP=3 again on the new head, aligned on main's prices). Made by
# <the local records> from rx/f3c/rf3c-chain.sh. On 01 (setsid): rf3d-chain.sh M16 H16 -- WORK/bin/me86a
# (main e86a0bed1d) and WORK/bin/ha4da (the PR head a4da5a4eb2), WORK ~/LLLLM/glm-work/f0091, through tp3-arm.sh in tp3f
# with rf3dhook.sh, then fbench.sh. First tp3f's tables are kept as window-costs-rf3dpre and main's record KM (rf3wm's)
# must hold rf3/own-me86a's words (flwords.py). The head times its windows once, uncounted (rf3dwh, also the first
# load): its start saves a record under its own key, which must be the one new record in tp3f's table and hold the
# timed look's words. Then main's record is copied over that key and read back by sha, so every counted start is kept on
# main's window and draft prices (both factored) and only the binaries' code differs (<the local ledger> 11-11 12:1x (b), as
# rf3c). rf3d/own-ha4da.timed keeps the head's own words. One after another, 60 s apart:
#   rf3dwh head, timed, uncounted   rf3dma main  rf3dha head  rf3dmb main  rf3dhb head  rf3dmc main
# The rest is rf3c-chain.sh's: rf3d/stop-now; 01-03's tf-preflight before each arm; 01's GPU lock; heat lines to
# rf3d/heat-LABEL.txt; fsum.py on the five counted arms (A = main's three, B = the head's two) into fbench/rf3d-fsum.txt
# and .json; tp3f's window-costs copied out to rf3d/window-costs-end. Log WORK/logs/rf3d-chain.log; the last line
# "== rf3d-chain finished" or "== rf3d-chain stopped" (the line before says why). Never edit while it runs.
set -u
G=$HOME/LLLLM/models/glm53-nvfp4
W=$HOME/LLLLM/glm-work/f0091
F=$W/rf3d
B=$W/fbench
C=$W/logs/rf3d-chain.log
A=$W/logs/arms.log
M=me86a
H=ha4da
FSUM=db74650034aa894d
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp3f python3 /work/kzt.py)
X="bash $G/tf-work/v292/sx.sh"
HOSTS="host01 host02 host03"
LABELS="rf3dwh rf3dma rf3dha rf3dmb rf3dhb rf3dmc"
HEAT=/var/log/gb10-telemetry/heat-cap.log
kept=""
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "rf3d-chain STOP: $*${kept:+ ($kept)}"; log "== rf3d-chain stopped"; exit 1; }
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
for f in tp3-arm.sh fbench.sh mkprompts_f.py csum.py fsum.py rf3dhook.sh heatcut.py kzt.py cluster.tp3f.env flwords.py; do
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
docker exec tp3f sh -c "if [ -d $WC ]; then echo there; else mkdir -p $WC && echo made; fi" > "$W/logs/rf3d-wc.txt" 2>&1 \
  || stop "tp3f: $WC: $(tail -n 1 "$W/logs/rf3d-wc.txt")"
mkdir -p "$F"
exec 9> /tmp/glm-perf-01.lock
flock -n 9 || stop "01's GPU lock (/tmp/glm-perf-01.lock) is held"
log "rf3d-chain begin, holding 01's GPU lock: $M $1, $H $2; tp3f's window-costs $(cat "$W/logs/rf3d-wc.txt"); tp3-arm $(sha256sum "$W/tp3-arm.sh" | cut -c1-16) fbench $(sha256sum "$W/fbench.sh" | cut -c1-16) fsum $(sha256sum "$W/fsum.py" | cut -c1-16) rf3dhook $(sha256sum "$W/rf3dhook.sh" | cut -c1-16) heatcut $(sha256sum "$W/heatcut.py" | cut -c1-16) kzt $(sha256sum "$W/kzt.py" | cut -c1-16) cluster.tp3f.env $(sha256sum "$W/cluster.tp3f.env" | cut -c1-16)"
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
    [ -n "$why" ] && stop "$l: not ready:$why (rf3d/preflight.log)"
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
arm() {   # LABEL BIN HOW: tp3-arm.sh with rf3dhook.sh (then fbench.sh) on the release's TP=3 settings
  local label=$1 bin=$2 how=$3 rc last n0 t0
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rf3d/stop-now: $(head -c 200 "$F/stop-now"))"
  ready "$label"
  echo "fbench $how $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp3-arm.sh $bin, want fbench $how $bin"
  bash "$W/tp3-arm.sh" "$label" "$bin" "$W/rf3dhook.sh"
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
# rf3d: main's record (rf3wm's) stays under me86a's key; the new head times its own once, then gets main's record
KM=1ef15d341d0a59f9776c2ea07c0cfd50c5cf71184fc04b2f29387f13e1943588.bin
own=$(cat "$W/rf3/own-$M" 2>/dev/null)
case "$own" in *" ms at 1 row, "*"; experts "*) ;; *) stop "rf3/own-$M holds no look words: '$own'" ;; esac
python3 "$W/flwords.py" "$F/tables-start" "$KM" "$own" > "$F/words-main.txt" 2>&1 \
  || stop "main's record: $(tail -n 1 "$F/words-main.txt")"
docker exec tp3f sh -c "[ ! -e $WC-rf3dpre ] && cp -a $WC $WC-rf3dpre" || stop "keeping tp3f's tables as window-costs-rf3dpre failed (or it is there)"
kept="tp3f's tables before the chain are window-costs-rf3dpre"
echo "$own" > "$F/own-$M"
log "main's record $KM holds $M's words ($own); tp3f's tables kept as window-costs-rf3dpre"
arm rf3dwh "$H" timed
tables timed
KH=$(comm -13 <(cut -d' ' -f1 "$F/tables-start" | sort) <(cut -d' ' -f1 "$F/tables-timed" | sort) | grep '\.bin$')
[ "$(printf '%s\n' "$KH" | grep -c .)" = 1 ] || stop "the head's timed start left $(printf '%s\n' "$KH" | grep -c .) new records, not one: $(echo $KH)"
timed=$(cat "$F/own-$H" 2>/dev/null)
python3 "$W/flwords.py" "$F/tables-timed" "$KH" "$timed" > "$F/words-head.txt" 2>&1 \
  || stop "the head's record: $(tail -n 1 "$F/words-head.txt")"
cp "$F/own-$H" "$F/own-$H.timed"
log "$H's timed start saved $KH: $timed; $M's (main's record): $own"
docker exec tp3f sh -c "cp -p $WC/$KM $WC/$KH" || stop "seeding $H's key with main's record failed"
[ "$(docker exec tp3f sh -c "cat $WC/$KM" | sha256sum)" = "$(docker exec tp3f sh -c "cat $WC/$KH" | sha256sum)" ] \
  || stop "$H's key does not read back as main's record"
echo "$own" > "$F/own-$H"
log "seeded tp3f's table: main's record $KM over $H's key $KH; both binaries' words: $own"
tables seeded
arm rf3dma "$M" kept
arm rf3dha "$H" kept
arm rf3dmb "$M" kept
arm rf3dhb "$H" kept
arm rf3dmc "$M" kept
python3 "$W/fsum.py" "$B" "$W/logs" rf3dma,rf3dmb,rf3dmc rf3dha,rf3dhb --json "$B/rf3d-fsum.json" > "$B/rf3d-fsum.txt" 2>&1
log "fsum rc=$?: $(tail -n 1 "$B/rf3d-fsum.txt" | cut -c1-400)"
tables end
mkdir -p "$F/window-costs-end" && docker exec tp3f sh -c "cd $WC && tar cf - ." | tar xf - -C "$F/window-costs-end" \
  || stop "copying tp3f's window-costs out failed"
log "tp3f's window-costs copied out to rf3d/window-costs-end ($(ls -A "$F/window-costs-end" | wc -l) files)"
log "rf3d-chain end: $LABELS (fbench/rf3d-fsum.txt); tp3f stays until rf3-done.sh"
log "== rf3d-chain finished"
