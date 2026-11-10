#!/bin/bash
# On 04, while a server answers at 127.0.0.1:8095: bench4.sh LABEL — upstream's tools (main 9fb5d2096's tools/ in bench/tools)
# as records/20261105-mia1104/b2/rc2-chain.sh ran them at TP=2, on prompts of this directory's corpus (upstream's tools/**/*.py
# at c085aab04, joined in path order, sha256 ed6ce946...): bench_cache.py --require-cold at 2K, 8K, 32K and 64K (30 s apart),
# bench_openai.py 256 tokens 3 reps at temperature 1.0 and 0, bench_concurrent.py --levels 1,4 --alone --serial at 0.
# Results in bench/LABEL-*.json, one line each in bench/LABEL.log. Last line: "bench4 LABEL end".
set -u
L=${1:?label}
B=$HOME/LLLLM/glm-work/tp4/bench
T=$B/tools
BASE=http://127.0.0.1:8095
log() { echo "$(date '+%F %T') $*" >> "$B/$L.log"; }
run() { local limit=$1; shift; timeout "$limit" "$@" >> "$B/$L.log" 2>&1 || log "  client rc $? ($*)"; }
log "== bench4 $L begin"
[ -s "$B/prompt-65536.json" ] || run 600 python3 "$B/mkprompts.py" "$BASE" "$B/corpus.txt" "$B"
for n in 2048 8192 32768 65536; do
  run 1500 python3 "$T/bench_cache.py" "$BASE" glm-tf --prompt-file "$B/prompt-$n.json" --expected-prompt-tokens "$n" --require-cold --output "$B/$L-cache-$n.json"
  log "  cache $n: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); f=r['rows'][0]; print(f['cache_state'], f['ttft_s'], 'qualified', r['qualified'])" "$B/$L-cache-$n.json" 2>&1 | tail -n 1)"
  sleep 30
done
run 1200 python3 "$T/bench_openai.py" "$BASE" glm-tf --tokens 256 --reps 3 --temperatures 1.0,0 --label "$L" --output "$B/$L-decode.json"
log "  decode: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r.get('summary', r))[:600])" "$B/$L-decode.json" 2>&1 | tail -n 1)"
sleep 30
run 1800 python3 "$T/bench_concurrent.py" "$BASE" glm-tf --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output "$B/$L-concurrent.json"
log "  concurrent: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k: r[k] for k in r if k not in ('rows', 'cells', 'results')})[:600])" "$B/$L-concurrent.json" 2>&1 | tail -n 1)"
echo "bench4 $L end" >> "$B/$L.log"
