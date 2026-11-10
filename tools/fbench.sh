#!/bin/bash
# An arm's HOOK on rank 0's host, while a server answers at 127.0.0.1:8095: fbench.sh LABEL -- the F gate's figures
# (plans/MIA1104_ADOPTION_PLAN.md, 「損が無い」; DECISIONS 11-09 22:5x) with upstream's tools in fbench/tools beside this
# script, bench4.sh's receipt and the plan's cells:
#   cold    bench_cache.py --require-cold, 16 tokens, 30 s apart, on mkprompts_f.py's eight disjoint prompts: 2,048,
#           8,192, 32,768 and 65,536 tokens (bench4.sh's slices), then 9,600 and 38,400 (two of each)
#   openai  bench_openai.py, 256 tokens, 3 reps, temperatures 1.0 and 0 (its fibonacci-raw and gpu-chat-no-think)
#   decode  bench_concurrent.py on tools/bench_prompts.json (item i to stream i, distinct) at 1 and 4 streams, and 8
#           where /health's max_batch_size holds them (TP=3's release serves 8, TP=2's 4), greedy, 256 tokens, 3 reps,
#           --alone --serial (each reply against its request alone, each alone run against "draft": false)
#   edit    the same on fbench/b2-edit.json (B2's edit list: the reply copies a tools/ file) at 1 and 4 streams, 1 rep
# The model's name and max_batch_size come from /health. Lines to fbench/LABEL.log, the last "fbench LABEL end"; what the
# clients print in fbench/LABEL.out; reports in fbench/LABEL-*.json. 3300 s at most, inside an arm's 3600 s for a hook.
set -u
L=${1:?label}
T=$(cd "$(dirname "$0")" && pwd)
B=$T/fbench
BASE=http://127.0.0.1:8095
END=$(( $(date +%s) + 3300 ))
log() { echo "$(date '+%F %T') $*" >> "$B/$L.log"; }
left() { echo $(( END - $(date +%s) )); }
client() { local need=$1 limit; shift; limit=$(left); [ "$limit" -gt "$need" ] || { log "fbench $L STOP: $limit s left"; exit 1; }; timeout "$limit" "$@" >> "$B/$L.out" 2>&1; }
read -r MODEL BATCH < <(python3 -c 'import json, sys, urllib.request; h = json.load(urllib.request.urlopen(sys.argv[1] + "/health", timeout=10)); print(h["model"], h["max_batch_size"])' "$BASE" 2>/dev/null)
[ -n "${MODEL:-}" ] && [[ "${BATCH:-}" =~ ^[0-9]+$ ]] || { log "fbench $L STOP: no model name or max_batch_size from /health"; exit 1; }
LEVELS=1,4
[ "$BATCH" -ge 8 ] && LEVELS=1,4,8
log "== fbench $L begin: model $MODEL, max_batch_size $BATCH, decode levels $LEVELS, tools $(cd "$B/tools" && sha256sum bench_cache.py bench_openai.py bench_concurrent.py bench_prompts.json | cut -c1-16 | paste -sd' '), edit $(sha256sum "$B/b2-edit.json" | cut -c1-16), mkprompts_f $(sha256sum "$T/mkprompts_f.py" | cut -c1-16)"
client 60 python3 "$T/mkprompts_f.py" "$BASE" "$B/corpus.txt" "$B" || { log "fbench $L STOP: the prompt files"; exit 1; }
log "  prompts $(cd "$B" && sha256sum c2048.json c8192.json c32768.json c65536.json f9600a.json f9600b.json f38400a.json f38400b.json | cut -c1-16 | paste -sd' ')"
for p in c2048 c8192 c32768 c65536 f9600a f38400a f9600b f38400b; do
  n=${p#?}; n=${n%[ab]}
  client 300 python3 "$B/tools/bench_cache.py" "$BASE" "$MODEL" --prompt-file "$B/$p.json" --expected-prompt-tokens "$n" \
    --require-cold --repeats 1 --streams 1 --tokens 16 --output "$B/$L-cold-$p.json"
  rc=$?
  log "  cold $p rc=$rc: $(python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); f = r["rows"][0]; print(f["cache_state"], f["prompt_tokens"], "ttft", f["ttft_s"], "qualified", r["qualified"])' "$B/$L-cold-$p.json" 2>&1 | tail -n 1)"
  sleep 30
done
client 300 python3 "$B/tools/bench_openai.py" "$BASE" "$MODEL" --tokens 256 --reps 3 --temperatures 1.0,0 --label "$L" \
  --output "$B/$L-openai.json"
log "  openai rc=$?: $(python3 -c 'import json, sys; print(" | ".join("%s t=%s %s" % (r["prompt"], r["temperature"], r["decode_tps_median"]) for r in json.load(open(sys.argv[1]))))' "$B/$L-openai.json" 2>&1 | tail -n 1)"
sleep 30
client 600 python3 "$B/tools/bench_concurrent.py" "$BASE" "$MODEL" --prompts "$B/tools/bench_prompts.json" --levels "$LEVELS" \
  --alone --serial --temperatures 0 --tokens 256 --reps 3 --label "$L-decode" --output "$B/$L-decode.json"
log "  decode rc=$?: $(python3 "$T/csum.py" --line "$B/$L-decode.json" 2>&1 | tail -n 1)"
sleep 30
client 300 python3 "$B/tools/bench_concurrent.py" "$BASE" "$MODEL" --prompts "$B/b2-edit.json" --levels 1,4 \
  --alone --serial --temperatures 0 --tokens 256 --reps 1 --label "$L-edit" --output "$B/$L-edit.json"
log "  edit rc=$?: $(python3 "$T/csum.py" --line "$B/$L-edit.json" 2>&1 | tail -n 1)"
echo "fbench $L end" >> "$B/$L.log"
