#!/bin/bash
# On 04 (setsid): rcnight.sh M16 H16 -- tonight's chain for #3689's receipts after the rebase onto #3776 (<a local session>, 11-20; <the local records>): rc4-chain.sh (TP=4 on 04-07) with RC4_LAST "2026-11-21 03:00",
# then, only when its last line is "== rc4-chain finished", rc2-chain.sh (TP=2 on 04+05) with RC2_LAST "2026-11-21 06:40":
# the window lent by <another local session> (04+05) and <another local session> (06+07) ends at 07:00 (<the local ledger> 11-20 08:1x). One chain on the host,
# so the PC may sleep (<a local note> §2, the overnight rule). Each chain puts 04's cache back itself. Log
# logs/rcnight.log; the last line "== rcnight finished" (both finished) or "== rcnight stopped" (the line before says
# which chain and why).
set -u
T=$HOME/LLLLM/glm-work/tp4
L=$T/logs/rcnight.log
log() { echo "$(date '+%F %T') $*" >> "$L"; }
[ $# = 2 ] || { log "rcnight: give M16 H16"; log "== rcnight stopped"; exit 1; }
for f in rc4-chain.sh rc2-chain.sh; do
  [ -s "$T/$f" ] || { log "rcnight: no $T/$f"; log "== rcnight stopped"; exit 1; }
done
log "== rcnight begin: rc4-chain $(sha256sum "$T/rc4-chain.sh" | cut -c1-16), rc2-chain $(sha256sum "$T/rc2-chain.sh" | cut -c1-16), $1 $2"
RC4_LAST="2026-11-21 03:00" bash "$T/rc4-chain.sh" "$1" "$2"
e4=$(tail -n 1 "$T/logs/rc4-chain.log" 2>/dev/null)
log "rc4: $(tail -n 2 "$T/logs/rc4-chain.log" 2>/dev/null | head -n 1 | cut -c21- | cut -c1-300) / $e4"
case "$e4" in *"== rc4-chain finished") ;; *) log "rc2 not started: rc4 did not finish"; log "== rcnight stopped"; exit 1 ;; esac
RC2_LAST="2026-11-21 06:40" bash "$T/rc2-chain.sh" "$1" "$2"
e2=$(tail -n 1 "$T/logs/rc2-chain.log" 2>/dev/null)
log "rc2: $(tail -n 2 "$T/logs/rc2-chain.log" 2>/dev/null | head -n 1 | cut -c21- | cut -c1-300) / $e2"
case "$e2" in *"== rc2-chain finished") log "== rcnight finished" ;; *) log "rc2 did not finish"; log "== rcnight stopped"; exit 1 ;; esac
