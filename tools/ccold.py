"""The cold prompts' greedy replies across the F gate's arms: python ccold.py BENCH ARM,ARM,... -- for each of
fbench.sh's eight cold prompts, the 16 tokens' token_sha256 in BENCH/ARM-cold-PROMPT.json (bench_cache.py, temperature 0,
seed 1234) of every arm, and whether they are one hash. Their decode runs past the prompt, so from 8,192 tokens on it
reaches the decode select (a bucket of 4,096 pools or more: the merge's whole-row path to 16,384 tokens, the scan and
the merge's other paths past it). Ends 1 when a prompt's hashes differ or a report is missing or not cold."""
import json
import os
import sys

COLD = ("c2048", "c8192", "f9600a", "f9600b", "c32768", "f38400a", "f38400b", "c65536")


def main(argv):
    bench, arms = argv[0], argv[1].split(",")
    bad = 0
    for p in COLD:
        got = []
        for a in arms:
            path = os.path.join(bench, "%s-cold-%s.json" % (a, p))
            try:
                with open(path, encoding="utf-8") as f:
                    r = json.load(f)["rows"][0]
                ok = r.get("cache_state") == "cold" and r.get("cached_tokens") == 0 and r.get("tokens") == 16
                got.append((r.get("token_sha256") or "") if ok else "not cold: %s" % {k: r.get(k) for k in ("cache_state", "cached_tokens", "tokens")})
            except (OSError, ValueError, KeyError, IndexError) as e:
                got.append("missing: %s" % e)
        one = len(set(got)) == 1 and len(got[0]) == 64
        bad += not one
        print("%-8s %s %s" % (p, "one hash" if one else "DIFFER", got[0][:16] if one else " ".join(x[:16] for x in got)))
    print("ccold: %d of %d cold prompts with one reply over %d arms" % (len(COLD) - bad, len(COLD), len(arms)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
