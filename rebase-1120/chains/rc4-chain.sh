#!/bin/bash
# rc4: #3689's TP=4 receipt again, for the head rebased onto #3776 (GLM decode kernels as dependent launches; <the local ledger>
# 11-20 07:1x, 07:2x and 08:1x; <the local records>). Made by <the local records> from
# <the local records>. On 04 (setsid): RC4_LAST="YYYY-MM-DD HH:MM" rc4-chain.sh M16 H16 -- bin/mb0b17
# (main b0b17e8962) and bin/h50c66 (the rebased head 50c6634539) in ~/LLLLM/glm-work/tp4, through tp4-arm.sh in tp4-tf3
# with rc4hook.sh, on the ring (TF_GLM_PREFILL_REDUCE=gather, NCCL_CUMEM_ENABLE=1, TF_GLM_PROMPT_TAIL=1). 04's cache is
# set aside as window-costs-rc4pre and the chain runs on a copy of it, put back at the end (the chain's own beside it as
# window-costs-rc4). First each binary times its windows once, uncounted, the head first (the reference key a timed start
# saves after its own then holds main's bytes). The two timed looks may name different expert forms (a load-time split,
# as rx on 11-16 and rc3 at 09:43 on 11-20): the forms are logged and the seed aligns them. Then the copy is copied out
# to rc4/window-costs-pick, seedpick.py (--same-form only when the forms match) names main's record and the head's by
# rc4/own-*'s words,
# the names the cache held before the chain (rc4/tables-start) told apart from the new ones (main's own key, not the
# reference key), and main's record files are copied over the head's key and read back by sha, so every counted start
# is kept on main's window and draft prices and only the binaries' code differs (<the local ledger> 11-11 12:1x (b));
# rc4/own-BIN.timed keep each binary's own words. Each arm, 60 s apart: grid.py's 4K, 64K, 256K and 1016K at 1, 4 and 8
# streams through grid4.sh, then m2draft.py's three drafted streams on the kept 1,040,384 prefix:
#   rc4wh head, timed  rc4wm main, timed  rc4ma main  rc4ha head  rc4mb main  rc4hb head  rc4mc main  rc4hc head  rc4md main
# Every counted arm's grid replies (fxcheck.py) and drafted tokens are rc4ma's (the first counted main arm, as rx2 took
# rxma); the timed arms' are compared with rc4ma's at the end and only logged (a timed arm may run another form). Four
# starts of the head and five of main (<a local note> §3). The rest is rx2-chain.sh's: rc4hook.sh checks rank 0's look; rc4/stop-now; tf-preflight
# before each arm; 04's GPU lock; heat lines; gsum.py over the seven counted arms into rc4/gsum.txt; no arm starts after
# RC4_LAST. Log logs/rc4-chain.log; the last line "== rc4-chain finished" or "== rc4-chain stopped". Never edit while it
# runs.
set -u
T=$HOME/LLLLM/glm-work/tp4
F=$T/rc4
C=$T/logs/rc4-chain.log
A=$T/logs/arms.log
G="TF_GLM_PREFILL_REDUCE=gather|NCCL_CUMEM_ENABLE=1|TF_GLM_PROMPT_TAIL=1"
M=mb0b17
H=h50c66
WC=/root/.cache/tensorfold/window-costs
K=(docker exec tp4-tf3 python3 /work/kzt.py)
HOSTS="host04 host05 host07 host06"
LABELS="rc4wh rc4wm rc4ma rc4ha rc4mb rc4hb rc4mc rc4hc rc4md"
LAST=$(date -d "${RC4_LAST:-}" +%s 2>/dev/null) || LAST=0   # no arm starts after RC4_LAST (the night chain sets it)
HEAT=/var/log/gb10-telemetry/heat-cap.log
moved=""
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() {
  local why=$*
  [ -n "$moved" ] && why="$why (the cache is aside: $moved; to put the first tables back, in tp4-tf3: mv $WC $WC-rc4stop && mv $WC-rc4pre $WC)"
  log "rc4-chain STOP: $why"
  log "== rc4-chain stopped"
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
for f in tp4-arm.sh grid4.sh bench4.sh grid.py gsum.py fxcheck.py rc4hook.sh m2draft.py heatcut.py kzt.py sx4.sh seedpick.py; do
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
docker exec tp4-tf3 sh -c "[ -d $WC ] && [ ! -e $WC-rc4pre ] && [ ! -e $WC-rc4 ] && [ ! -e $WC-rc4stop ]" \
  || stop "tp4-tf3: no $WC, or window-costs-rc4pre, -rc4 or -rc4stop is there"
mkdir -p "$F"
exec 9> /tmp/glm-perf-04.lock
flock -n 9 || stop "04's GPU lock (/tmp/glm-perf-04.lock) is held"
log "rc4-chain begin, holding 04's GPU lock: $M $1, $H $2; tp4-arm $(sha256sum "$T/tp4-arm.sh" | cut -c1-16) grid4 $(sha256sum "$T/grid4.sh" | cut -c1-16) grid.py $(sha256sum "$T/grid.py" | cut -c1-16) rc4hook $(sha256sum "$T/rc4hook.sh" | cut -c1-16) m2draft $(sha256sum "$T/m2draft.py" | cut -c1-16) heatcut $(sha256sum "$T/heatcut.py" | cut -c1-16) kzt $(sha256sum "$T/kzt.py" | cut -c1-16)"
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
    [ -n "$why" ] && stop "$l: not ready:$why (rc4/preflight.log)"
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
  [ -e "$F/stop-now" ] && stop "asked to stop before $label (rc4/stop-now: $(head -c 200 "$F/stop-now"))"
  [ "$(date +%s)" -lt "$LAST" ] || stop "$label would start after RC4_LAST (${RC4_LAST:-unset})"
  ready "$label"
  echo "$* $bin" > "$F/want-$label"
  n0=$(grep -a -c "== arm $label " "$A" 2>/dev/null)
  t0=$(date +%s)
  log "$label begin: tp4-arm.sh $bin serve +[$lines], want $* $bin"
  bash "$T/tp4-arm.sh" "$label" "$bin" serve "$lines" "$T/rc4hook.sh"
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
drafted() {   # LABEL [REF]: the drafted reps ended with 512 tokens each, one token list among them; with REF, the same
  # list as REF's, and whether each rep's rounds and accepted drafts are REF's (the speeds compare only where they are)
  local label=$1 ref=${2:-} out
  out=$(python3 - "$F/draft-$label.json" ${ref:+"$F/draft-$ref.json"} <<'PY'
import json, sys
def reps(p):
    return [json.loads(l) for l in open(p) if l.startswith("{")]
try:
    a = reps(sys.argv[1])
    bad = [r.get("rep") for r in a if not r.get("ok") or r.get("tokens") != 512 or not r.get("tensorfold") or not r.get("speculative")]
    if len(a) < 3 or bad:
        print("STOP %d reps, bad %s" % (len(a), bad))
        sys.exit(0)
    sha = [r["tensorfold"]["token_sha"] for r in a]
    line = "reps %d, tok/s %s, s %s, rounds %s, accepted %s, cached %s, sha %s" % (
        len(a), ",".join("%.2f" % r["tensorfold"]["tokens_per_second"] for r in a), ",".join(str(r["s"]) for r in a),
        ",".join(str(r["speculative"]["rounds"]) for r in a), ",".join(str(r["speculative"]["accepted"]) for r in a),
        ",".join(str(r["cached"]) for r in a), ",".join(sha))
    if len(set(sha)) != 1:
        print("STOP the reps' tokens differ: " + line)
        sys.exit(0)
    if len(sys.argv) > 2:
        b = reps(sys.argv[2])
        if {r["tensorfold"]["token_sha"] for r in b} != set(sha):
            print("STOP tokens other than the reference's: " + line)
            sys.exit(0)
        same = len(a) == len(b) and all(x["speculative"]["rounds"] == y["speculative"]["rounds"] and x["speculative"]["accepted"] == y["speculative"]["accepted"] for x, y in zip(a, b))
        line += "; drafts as the reference's: " + ("yes" if same else "no (the speeds do not compare)")
    print(line)
except Exception as e:
    print("STOP unread: %s: %s" % (type(e).__name__, e))
PY
)
  log "$label drafted: $out"
  case "$out" in STOP*) stop "$label drafted: $out" ;; esac
}
peek() {   # LABEL REF: a timed arm's grid replies and drafted tokens against REF's, logged, not judged (its own form)
  local l=$1 r=$2 g d
  g=$(python3 "$T/fxcheck.py" "$T/bench" "$r" "$l" 2>&1 | tail -n 1)
  d=$(python3 - "$F/draft-$l.json" "$F/draft-$r.json" <<'PY'
import json, sys
def shas(p):
    return {json.loads(x)["tensorfold"]["token_sha"] for x in open(p) if x.startswith("{")}
try:
    a, b = shas(sys.argv[1]), shas(sys.argv[2])
    print("the same token_sha" if a == b else "token_sha differ: %s against %s" % (sorted(a), sorted(b)))
except Exception as e:
    print("unread: %s: %s" % (type(e).__name__, e))
PY
)
  log "$l against $r (logged, not judged): replies: $g; drafted: $d"
}
tables() { "${K[@]}" "$WC" > "$F/tables-$1" 2>&1 || stop "kzt.py: $(tail -n 1 "$F/tables-$1")"; log "tables-$1: $(grep -c . "$F/tables-$1") lines"; }
tables before
docker exec tp4-tf3 sh -c "[ ! -e $WC-rc4pre ] && mv $WC $WC-rc4pre && cp -a $WC-rc4pre $WC" || stop "setting the cache aside failed"
moved="window-costs-rc4pre"
log "window-costs to window-costs-rc4pre ($(docker exec tp4-tf3 sh -c "ls -A $WC-rc4pre | wc -l") files); window-costs now a copy of it ($(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l") files)"
tables start
# rc4: each binary times its windows once on the copy, uncounted, the head first, with the grid and the drafted reps
arm rc4wh "$H" "$G" grid timed
grid rc4wh
drafted rc4wh
arm rc4wm "$M" "$G" grid timed
grid rc4wm
drafted rc4wm
fh=$(sed -n 's/^.*; experts //p' "$F/own-$H" 2>/dev/null) fm=$(sed -n 's/^.*; experts //p' "$F/own-$M" 2>/dev/null)
log "expert forms: $M '${fm:-none}', $H '${fh:-none}'"
[ -n "$fm" ] && [ -n "$fh" ] || stop "a timed look names no expert form: $M '${fm:-none}', $H '${fh:-none}'"
sf=""
[ "$fm" = "$fh" ] && sf=--same-form
log "seedpick ${sf:-plain (a split: the seed aligns the counted arms on the form of main)}"
cp "$F/own-$M" "$F/own-$M.timed" && cp "$F/own-$H" "$F/own-$H.timed" || stop "keeping the timed words failed"
tables timed
# rc4: aligned on main's TP=4 record (<the local ledger> 11-11 12:1x (b), form and prices): the head's key gets main's record
# (rc4wm's), main's own key stays; both binaries' words are main's
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
case "$own" in *" ms at 1 row, "*"; experts "*) ;; *) stop "rc4/own-$M.timed holds no look words: '$own'" ;; esac
echo "$own" > "$F/own-$M"
echo "$own" > "$F/own-$H"
log "seeded the copy: main's record $KM over $H's key $KH; both binaries' words: $own"
tables seeded
arm rc4ma "$M" "$G" grid kept
grid rc4ma
drafted rc4ma
arm rc4ha "$H" "$G" grid kept
grid rc4ha rc4ma
drafted rc4ha rc4ma
arm rc4mb "$M" "$G" grid kept
grid rc4mb rc4ma
drafted rc4mb rc4ma
arm rc4hb "$H" "$G" grid kept
grid rc4hb rc4ma
drafted rc4hb rc4ma
arm rc4mc "$M" "$G" grid kept
grid rc4mc rc4ma
drafted rc4mc rc4ma
arm rc4hc "$H" "$G" grid kept
grid rc4hc rc4ma
drafted rc4hc rc4ma
arm rc4md "$M" "$G" grid kept
grid rc4md rc4ma
drafted rc4md rc4ma
peek rc4wh rc4ma
peek rc4wm rc4ma
python3 "$T/gsum.py" "$T/bench" rc4ma rc4ha rc4mb rc4hb rc4mc rc4hc rc4md > "$F/gsum.txt" 2>&1
log "grid arms end: gsum of rc4ma rc4ha rc4mb rc4hb rc4mc rc4hc rc4md in rc4/gsum.txt ($(wc -l < "$F/gsum.txt") lines)"
tables end
docker exec tp4-tf3 sh -c "[ ! -e $WC-rc4 ] && mv $WC $WC-rc4 && mv $WC-rc4pre $WC" || stop "putting the first tables back failed"
moved=""
log "the chain's tables to window-costs-rc4 ($(docker exec tp4-tf3 sh -c "ls -A $WC-rc4 | wc -l") files); window-costs back with $(docker exec tp4-tf3 sh -c "ls -A $WC | wc -l")"
log "rc4-chain end: $LABELS (rc4/gsum.txt, rc4/draft-LABEL.json, bench/LABEL-grid.json); the cache as it began, the chain's beside it"
log "== rc4-chain finished"
