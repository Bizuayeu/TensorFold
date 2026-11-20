#!/bin/bash
# (11-20: fl-gpu01.sh with the names rc-gpu and rcgpu, by hand, for #3689 rebased onto #3776: the head h50c66's
#  tf-cuda-test; <the local records>)
# On 01 (setsid): rc-gpu01.sh CUDA16 -- the floor head's GPU checks (11-17, <a local session>; <the local ledger> 11-17 09:0x),
# as <the local records> ran them and sbench-run01.sh holds 01: tf-cuda-test (fl-build.sh's, 03's
# tf3-build image, nvcc of that image, in glm-work/f0091/rcgpu/bin) runs kernel-contract filtered to "decode scores"
# (the decode window's index scores at its served shapes, 8 x 4,096 pools being 64 blocks, and the planted control),
# "tf_glm_index_scores" and "decode select", then glm-index-check (its decode windows counted by takesDecode with
# selectTokens' blocks; 2 of 7 take glm_index_scores on the head before the floor too). Holds /tmp/glm-perf-01.lock,
# after tf-preflight --skip lock (heat waits 60 s and looks again, up to 20 times; any other item stops). The
# gb10-heat-cap lines written during the run are copied into the log. Outputs in glm-work/f0091/rcgpu; log
# rcgpu/run.log: "== rc-gpu finished" or "== rc-gpu stopped". The lines are read by hand, not judged here.
set -u
P=$HOME/LLLLM/glm-work/f0091/rcgpu
LOG=$P/run.log
HEAT=/var/log/gb10-telemetry/heat-cap.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
halt() { log "STOP: $*"; log "== rc-gpu stopped"; exit 1; }
log "== rc-gpu begin"
[ "$(hostname)" = host01 ] || halt "not 01"
[ -x "$P/bin/tf-cuda-test" ] || halt "no tf-cuda-test in $P/bin"
got=$(sha256sum "$P/bin/tf-cuda-test" | cut -c1-16)
[ "$got" = "${1:-}" ] || halt "tf-cuda-test is $got, not '${1:-}'"
log "  tf-cuda-test $got"
exec 9> /tmp/glm-perf-01.lock
flock -n 9 || halt "01's GPU lock (/tmp/glm-perf-01.lock) is held"
ok=
for i in $(seq 1 20); do
  out=$(python3 "$HOME/LLLLM/ops/tf-rules/hosts/tf-preflight" --label rcgpu --skip lock 2>&1)
  rc=$?
  log "  preflight $i rc=$rc $(echo "$out" | tail -n 1 | cut -c1-700)"
  [ "$rc" -eq 0 ] && { ok=1; break; }
  echo "$out" | tail -n 1 | grep -q '"failed": \["heat"\]' || halt "preflight is not ready (not only heat)"
  sleep 60
done
[ -n "$ok" ] || halt "still not ready after 20 looks"
h0=$(wc -l < "$HEAT" 2>/dev/null || echo 0)
for f in "decode scores" "tf_glm_index_scores" "decode select"; do
  t=${f// /-}
  timeout 2700 "$P/bin/tf-cuda-test" kernel-contract "$f" > "$P/contract-$t.txt" 2>&1
  log "  kernel-contract \"$f\" rc=$?: $(grep -a 'SUMMARY' "$P/contract-$t.txt" | tail -n 1 | cut -c1-400)"
  log "  \"$f\" controls: $(grep -a -E '^CONTROL' "$P/contract-$t.txt" | cut -c1-200 | paste -sd'|')"
done
timeout 1800 "$P/bin/tf-cuda-test" glm-index-check > "$P/index.txt" 2>&1
log "  glm-index-check rc=$?: $(tail -n 1 "$P/index.txt" | cut -c1-240)"
log "  $(grep -a 'decode scores on glm_index_scores' "$P/index.txt" | tail -n 1 | cut -c1-160)"
h1=$(wc -l < "$HEAT" 2>/dev/null || echo 0)
log "  heat-cap lines during the run: $((h1 - h0))"
[ "$h1" -gt "$h0" ] && sed -n "$((h0 + 1)),${h1}p" "$HEAT" | cut -c1-300 >> "$LOG"
log "== rc-gpu finished"
