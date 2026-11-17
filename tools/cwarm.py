"""cwarm.py BENCH ARM,ARM,... -- in each arm's eight cold reports (bench_cache.py), every row's cache_state, cached
tokens and token_sha256: whether each prompt's resumed (warm) rows give the cold row's tokens, in every arm."""
import json
import os
import sys

COLD = ("c2048", "c8192", "f9600a", "f9600b", "c32768", "f38400a", "f38400b", "c65536")
bench, arms = sys.argv[1], sys.argv[2].split(",")
states, bad, warm_rows = {}, 0, 0
for p in COLD:
    for a in arms:
        rows = json.load(open(os.path.join(bench, "%s-cold-%s.json" % (a, p)), encoding="utf-8"))["rows"]
        cold = [r for r in rows if r.get("cache_state") == "cold"]
        for r in rows:
            states[r.get("cache_state")] = states.get(r.get("cache_state"), 0) + 1
            if r.get("cache_state") != "cold":
                warm_rows += 1
                if not cold or r.get("token_sha256") != cold[0].get("token_sha256"):
                    bad += 1
                    print("%s %s: %s row (cached %s) differs from the cold row" % (a, p, r.get("cache_state"), r.get("cached_tokens")))
    print("%-8s rows per arm %s, cached tokens %s" % (p, len(rows), [r.get("cached_tokens") for r in rows]))
print("cwarm: states %s; %d resumed rows, %d differ from their cold row" % (states, warm_rows, bad))
