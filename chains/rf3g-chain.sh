#!/bin/bash
# rf3g: the 0091 PR's TP=3 retake of bench_openai's code at t=0 (<the local ledger> 11-17 17:0x; <a local session>), made by
# <the local records> from fl/rf3e-chain.sh. On 01 (setsid): rf3g-chain.sh M16 H16 -- WORK/bin/me86a
# (main e86a0bed1d) and WORK/bin/ha4da (the PR head a4da5a4eb2), WORK ~/LLLLM/glm-work/f0091, through tp3-arm-g.sh in a
# new tp3f with rf3ghook.sh, then obench.sh (fbench.sh's cold prompts and bench_openai, in its order and with its
# arguments). No price is timed: tp3f's table, empty, gets rf3e's window-costs-end (WORK/rg-wc: me86a's key, the head's
# and one more holding main's record, the reference the head's timed one), read back by sha, so every start is kept on
# main's window and draft prices; after every arm the table must read back the same. One after another, 60 s apart:
#   rf3gma main  rf3gha head  rf3gza z  rf3gmb main  rf3ghb head  rf3gmc main  rf3gzb z  rf3ghc head  rf3gmd main
# z: the head with TF_GLM_DSCORES=0 in every rank's rank.env (uncounted, a diagnostic). The rest is rf3e-chain.sh's:
# rf3g/stop-now; 01-03's tf-preflight before each arm; 01's GPU lock; heat lines to rf3g/heat-LABEL.txt. Then fsum.py on
# main's four (A) against the head's three (B) into fbench/rf3g-fsum.txt and .json, and against z's two into
# fbench/rf3g-zsum.*; tp3f's window-costs copied out to rf3g/window-costs-end. Log WORK/logs/rf3g-chain.log; the last
# line "== rf3g-chain finished" or "== rf3g-chain stopped" (the line before says why). Never edit while it runs.
set -u
G=$HOME/LLLLM/models/glm53-nvfp4
W=$HOME/LLLLM/glm-work/f0091
F=$W/rf3g
B=$W/fbench
C=$W/logs/rf3g-chain.log
A=$W/logs/arms.log
M=me86a
H=ha4da
FSUM=db74650034aa894d
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp3f python3 /work/kzt.py)
X="bash $G/tf-work/v292/sx.sh"
HOSTS="host01 host02 host03"
LABELS="rf3gma rf3gha rf3gza rf3gmb rf3ghb rf3gmc rf3gzb rf3ghc rf3gmd"
HEAT=/var/log/gb10-telemetry/heat-cap.log
kept="tp3f is this chain's, its table restored from WORK/rg-wc"
RWC=$W/rg-wc
KM=22a23542f52bd3df45bc582f9bbe5788f5279a2d4d852617244e5ac346e35a65.bin
KH=29e7d711db3c90ae9021b0dc28b2834c50435793704a1aa2c64067b75895fe44.bin
KX=40b2ce805561c7a4a5b959dd6b4ad1c7c39af441d1cce86f65bd8b8d41bbfab4.bin
REF=1ef15d341d0a59f9776c2ea07c0cfd50c5cf71184fc04b2f29387f13e1943588.bin
SM=28ca3be5bc985d31c934392d3d65c5291e16399c1eb581056e1004e084165112
SR=618e466e23088f9952733f61435e0b8b0794a7dcd8fa9c6ac3dad4f1906173db
WANT_WC=$(printf '%s  %s\n' "$SM" "$KM" "$SM" "$KH" "$SM" "$KX" "$SR" "$REF" | sort -k2)
sums() { awk '{sub(/^\*/, "", $2); print $1 "  " $2}' | sort -k2; }   # sha256sum's lines, its binary mark dropped, by name
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "rf3g-chain STOP: $*${kept:+ ($kept)}"; log "== rf3g-chain stopped"; exit 1; }
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
  for f in "$W/logs/serve-r0-$l.log" "$B/$l.log" "$B/$l-decode.json" "$B/$l-openai.json"; do
    [ -e "$f" ] && stop "$f exists"
  done
done
for f in tp3-arm-g.sh obench.sh mkprompts_f.py csum.py fsum.py rf3ghook.sh heatcut.py kzt.py cluster.tp3f.env flwords.py; do
  [ -s "$W/$f" ] || stop "no WORK/$f"
done
for f in tools/bench_cache.py tools/bench_openai.py tools/bench_concurrent.py tools/bench_prompts.json b2-edit.json corpus.txt; do
  [ -s "$B/$f" ] || stop "no fbench/$f"
done
got=$(cd "$RWC" 2>/dev/null && sha256sum * 2>&1 | sums)
[ "$got" = "$WANT_WC" ] || stop "WORK/rg-wc is not rf3e's window-costs-end: $(echo $got | cut -c1-300)"
[ "$(cat "$W/rf3e/own-$M" 2>/dev/null)" = "28.59 ms at 1 row, 44.64 at 4, 129.35 at 32, a draft 1.50 ms; experts factored" ] \
  || stop "WORK/rf3e/own-$M is not main's words"
img=""
for h in $HOSTS; do
  i=$(on "$h" "docker inspect -f '{{.State.Running}} {{.Image}}' tp3f 2>/dev/null | cut -c1-24")
  log "host $h: tp3f ${i:-none}"
  case "$i" in "true sha256:"*) ;; *) stop "$h: tp3f is not running ('${i:-none}')" ;; esac
  [ -z "$img" ] && img=$i
  [ "$i" = "$img" ] || stop "$h's tp3f image is not host01's"
done
docker exec tp3f sh -c "if [ -d $WC ]; then echo there; else mkdir -p $WC && echo made; fi" > "$W/logs/rf3g-wc.txt" 2>&1 \
  || stop "tp3f: $WC: $(tail -n 1 "$W/logs/rf3g-wc.txt")"
[ "$(docker exec tp3f sh -c "ls -A $WC | wc -l")" = 0 ] || stop "tp3f's $WC is not empty: the new tp3f must start with no table"
mkdir -p "$F"
exec 9> /tmp/glm-perf-01.lock
flock -n 9 || stop "01's GPU lock (/tmp/glm-perf-01.lock) is held"
log "rf3g-chain begin, holding 01's GPU lock: $M $1, $H $2; tp3f's window-costs $(cat "$W/logs/rf3g-wc.txt"); tp3-arm-g $(sha256sum "$W/tp3-arm-g.sh" | cut -c1-16) obench $(sha256sum "$W/obench.sh" | cut -c1-16) fsum $(sha256sum "$W/fsum.py" | cut -c1-16) rf3ghook $(sha256sum "$W/rf3ghook.sh" | cut -c1-16) heatcut $(sha256sum "$W/heatcut.py" | cut -c1-16) kzt $(sha256sum "$W/kzt.py" | cut -c1-16) cluster.tp3f.env $(sha256sum "$W/cluster.tp3f.env" | cut -c1-16)"
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
    [ -n "$why" ] && stop "$l: not ready:$why (rf3g/preflight.log)"
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
arm() {   # LABEL BIN HOW [z]: tp3-arm-g.sh with rf3ghook.sh (then obench.sh) on the release's TP=3 settings; z adds TF_GLM_DSCORES=0
  local label=$1 bin=$2 how=$3 z=${4:-} extra="" rc last n0 t0
  [ "$z" = z ] && extra=TF_GLM_DSCORES=0
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rf3g/stop-now: $(head -c 200 "$F/stop-now"))"
  ready "$label"
  echo "obench $how $bin $z" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp3-arm-g.sh $bin${extra:+ $extra}, want obench $how $bin${z:+ z}"
  bash "$W/tp3-arm-g.sh" "$label" "$bin" "$W/rf3ghook.sh" "$extra"
  rc=$?
  heat "$label" "$t0"
  [ "$(grep -a -c "== arm $label " "$A" 2>/dev/null)" -gt "${n0:-0}" ] 2>/dev/null || stop "$label: no new line in WORK/logs/arms.log"
  last=$(grep -a "== arm $label " "$A" | tail -n 1 | cut -c21-)
  log "$label rc=$rc $last; $(grep -a " $label want " "$F/form.log" 2>/dev/null | tail -n 1 | cut -c21- | cut -c1-300)"
  sleep 60
  case "$last" in *" served") ;; *) stop "$label not served" ;; esac
  [ -e "$F/stop-$label" ] && stop "$label: rank 0's look is not the arm's (obench $how $bin${z:+ z}): $(cat "$F/stop-$label")"
  grep -qx "fbench $label end" "$B/$label.log" 2>/dev/null || stop "$label's obench: $(tail -n 2 "$B/$label.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  log "$label obench: $(grep -a -E '  (cold|openai) ' "$B/$label.log" | cut -c21- | paste -sd'|' | cut -c1-1500)"
  same "$label"
}
tables() { "${K[@]}" "$WC" > "$F/tables-$1" 2>&1 || stop "kzt.py: $(tail -n 1 "$F/tables-$1")"; log "tables-$1: $(grep -c . "$F/tables-$1") lines"; }
same() {   # LABEL: tp3f's table must read back as restored (the same files, the same sha256)
  local now
  now=$(docker exec tp3f sh -c "cd $WC && sha256sum *" 2>&1 | sums)
  echo "$now" > "$F/table-$1.sha"
  [ "$now" = "$base" ] || stop "$1: tp3f's table is not the restored one (rf3g/table-$1.sha)"
}
tables start
# rf3g: tp3f's table, empty, gets rf3e's window-costs-end from WORK/rg-wc, read back by sha; nothing is timed again
tar cf - -C "$RWC" . | docker exec -i tp3f sh -c "cd $WC && tar xf - --no-same-owner" || stop "restoring tp3f's table from WORK/rg-wc failed"
base=$(docker exec tp3f sh -c "cd $WC && sha256sum *" 2>&1 | sums)
[ "$base" = "$WANT_WC" ] || stop "tp3f's table does not read back as rf3e's window-costs-end: $(echo $base | cut -c1-300)"
echo "$base" > "$F/table-restored.sha"
tables seeded
own=$(cat "$W/rf3e/own-$M")
python3 "$W/flwords.py" "$F/tables-seeded" "$KM" "$own" > "$F/words-main.txt" 2>&1 \
  || stop "main's key: $(tail -n 1 "$F/words-main.txt")"
python3 "$W/flwords.py" "$F/tables-seeded" "$KH" "$own" > "$F/words-head.txt" 2>&1 \
  || stop "the head's key: $(tail -n 1 "$F/words-head.txt")"
echo "$own" > "$F/own-$M"
echo "$own" > "$F/own-$H"
log "restored tp3f's table from rf3e's window-costs-end (4 files, read back by sha): $M's key $KM and $H's key $KH hold main's words ($own)"
arm rf3gma "$M" kept
arm rf3gha "$H" kept
arm rf3gza "$H" kept z
arm rf3gmb "$M" kept
arm rf3ghb "$H" kept
arm rf3gmc "$M" kept
arm rf3gzb "$H" kept z
arm rf3ghc "$H" kept
arm rf3gmd "$M" kept
python3 "$W/fsum.py" "$B" "$W/logs" rf3gma,rf3gmb,rf3gmc,rf3gmd rf3gha,rf3ghb,rf3ghc --json "$B/rf3g-fsum.json" > "$B/rf3g-fsum.txt" 2>&1
log "fsum rc=$?: $(grep -a -F 'openai fibonacci-raw t=0.0' "$B/rf3g-fsum.txt" | tr -s ' ' | cut -c1-300); $(tail -n 1 "$B/rf3g-fsum.txt" | cut -c1-300)"
python3 "$W/fsum.py" "$B" "$W/logs" rf3gma,rf3gmb,rf3gmc,rf3gmd rf3gza,rf3gzb --json "$B/rf3g-zsum.json" > "$B/rf3g-zsum.txt" 2>&1
log "zsum rc=$?: $(grep -a -F 'openai fibonacci-raw t=0.0' "$B/rf3g-zsum.txt" | tr -s ' ' | cut -c1-300)"
tables end
mkdir -p "$F/window-costs-end" && docker exec tp3f sh -c "cd $WC && tar cf - ." | tar xf - -C "$F/window-costs-end" \
  || stop "copying tp3f's window-costs out failed"
log "tp3f's window-costs copied out to rf3g/window-costs-end ($(ls -A "$F/window-costs-end" | wc -l) files)"
log "rf3g-chain end: $LABELS (fbench/rf3g-fsum.txt, rf3g-zsum.txt); tp3f stays until rg-done.sh"
log "== rf3g-chain finished"
