"""The F gate's arms side by side (fbench.sh's reports, the arms' rank 0 logs), on the host or the PC:

    python3 fsum.py BENCH LOGS MAIN,MAIN,... HEAD,HEAD,... [--json OUT]

The As are main's arms and the Bs the head's, alternated, each its own start (main -> head -> main -> head -> main).
For every figure: each arm's value, B's mean against A's mean (a loss is positive: tok/s below it, seconds above it),
and the As' range, (max - min) / mean. The verdict is the rule DECISIONS fixed before the numbers (11-09 22:4x, 23:2x):
  slower  the loss is past max(the As' range, 0.25%)
  within  B's mean lies between the lowest and the highest A
  faster  B's mean is better than every A
  open    none of these (past the As on the slow side, inside the threshold)
The gate's figures (MIA1104's F gate, marked *) are bench_concurrent's decode cells (per stream and aggregate at each
stream count) and the cold 9,600 and 38,400 prompts (each arm's mean of two); a "slower" there counts. The receipt's
other figures (cold 2,048 to 65,536, one prompt each; bench_openai's rows; the edit list's cells) are verdicted and
reported alone. Exactness, its own lines: in every bench_concurrent cell of every arm each reply equals its request
alone and each alone run its "draft": false run, nothing failed, each stream's token_sha256 is one hash over the reps
and the same in every arm; bench_openai's greedy rows hold one token_sha over their reps, the same in every arm (its
sampled rows' hashes are compared as a note); every cold prompt is cold at its expected tokens. Memory: rank 0's load
line in each arm (model GiB, pages and their GiB, MiB fixed a stream, GiB left after the reserve, kept slots, load
seconds). The last line counts the slower gate figures, the slower receipt figures and the exactness failures.
"""
import json
import os
import re
import sys

COLD = ("c2048", "c8192", "c32768", "c65536", "f9600a", "f9600b", "f38400a", "f38400b")
LISTS = ("decode", "edit")
FLOOR = 0.0025
LOAD = re.compile(r"model (?P<model>[\d.]+) GiB; (?P<streams>\d+) streams? at once.*? taking (?P<pages>\d+) pages of 512 rows "
                  r"\((?P<pages_gib>[\d.]+) GiB\).*?(?P<fixed>\d+) MiB fixed each, of (?P<left>[\d.]+) GiB left")
KEPT = re.compile(r"(\d+) kept states' slots")
LOADED = re.compile(r"loaded in ([\d.]+)s")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def size(p):
    return int(p[1:].rstrip("ab"))


def cold(bench, label):
    out = {}
    for p in COLD:
        r = load(os.path.join(bench, "%s-cold-%s.json" % (label, p)))
        if r is None:
            out[p] = None
            continue
        row = r["rows"][0]
        out[p] = {"ttft": row.get("ttft_s"), "state": row.get("cache_state"), "tokens": row.get("prompt_tokens"),
                  "qualified": bool(r.get("qualified"))}
    return out


def openai(bench, label):
    r = load(os.path.join(bench, "%s-openai.json" % label))
    if r is None:
        return None
    return {(x["prompt"], x["temperature"]): {"tps": x.get("decode_tps_median"),
                                              "hashes": [e.get("token_sha") for e in x.get("evidence", [])]} for x in r}


def cells(bench, label, name):
    r = load(os.path.join(bench, "%s-%s.json" % (label, name)))
    if r is None:
        return None
    out = {}
    for c in r.get("cells", []):
        if c.get("temperature") != 0:
            continue
        streams = c["streams"]
        hashes = []
        for i in range(streams):
            hashes.append(sorted({rep["evidence"][i].get("token_sha256") for rep in c.get("reps", [])
                                  if len(rep.get("evidence") or []) > i}))
        out[streams] = {"tps": c.get("per_stream_tps"), "agg": c.get("aggregate_tps"), "failed": c.get("failed", 0),
                        "alone": c.get("alone") or {}, "serial": c.get("serial") or {}, "hashes": hashes}
    return out


def startup(logs, label):
    try:
        with open(os.path.join(logs, "serve-r0-%s.log" % label), encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    m = LOAD.search(text)
    if not m:
        return None
    out = {k: float(v) if "." in v else int(v) for k, v in m.groupdict().items()}
    k = KEPT.search(text)
    out["kept"] = int(k.group(1)) if k else None
    s = LOADED.search(text)
    out["load_s"] = float(s.group(1)) if s else None
    return out


def verdict(As, Bs, higher):
    if not As or not Bs or None in As or None in Bs:
        return None
    mean = sum(As) / len(As)
    b = sum(Bs) / len(Bs)
    loss = (1 - b / mean) if higher else (b / mean - 1)
    spread = (max(As) - min(As)) / mean
    lo, hi = min(As), max(As)
    if loss > max(spread, FLOOR):
        word = "slower"
    elif lo <= b <= hi:
        word = "within"
    elif (b > hi) if higher else (b < lo):
        word = "faster"
    else:
        word = "open"
    return {"loss": round(loss * 100, 2), "spread": round(spread * 100, 2), "verdict": word}


def main(argv):
    if len(argv) < 4:
        sys.exit(__doc__)
    bench, logs = argv[:2]
    mains, heads = argv[2].split(","), argv[3].split(",")
    arms = tuple(mains + heads)
    out = {"main": mains, "head": heads, "figures": [], "exact": [], "notes": [], "memory": {}}
    data = {x: {"cold": cold(bench, x), "openai": openai(bench, x), **{n: cells(bench, x, n) for n in LISTS}} for x in arms}
    rows = []
    for n in sorted({size(p) for p in COLD}):
        vals = []
        for x in arms:
            c = data[x]["cold"]
            want = [p for p in COLD if size(p) == n]
            got = [c[p]["ttft"] for p in want if c.get(p) and c[p]["ttft"] is not None]
            vals.append(sum(got) / len(got) if len(got) == len(want) else None)
        rows.append(("cold %d s%s" % (n, " (mean of two)" if n in (9600, 38400) else ""), vals, False, n in (9600, 38400)))
    keys = sorted(set().union(*[set(data[x]["openai"] or {}) for x in arms]), key=lambda k: (-k[1], k[0]))
    for key in keys:
        vals = [((data[x]["openai"] or {}).get(key) or {}).get("tps") for x in arms]
        rows.append(("openai %s t=%s tok/s" % key, vals, True, False))
    for name in LISTS:
        levels = sorted(set().union(*[set((data[x][name] or {}).keys()) for x in arms]))
        for n in levels:
            for key, what in (("tps", "per stream"), ("agg", "aggregate")):
                if n == 1 and key == "agg":
                    continue
                vals = [((data[x][name] or {}).get(n) or {}).get(key) for x in arms]
                rows.append(("%s x%d %s tok/s" % (name, n, what), vals, True, name == "decode"))
    slower, slower_r = 0, 0
    print("%-40s %s" % ("figure (gate: *)", "  ".join("%9s" % x for x in arms)))
    for what, vals, higher, gate in rows:
        v = verdict(vals[:len(mains)], vals[len(mains):], higher)
        if v and v["verdict"] == "slower":
            if gate:
                slower += 1
            else:
                slower_r += 1
        out["figures"].append({"figure": what, "gate": gate, "values": dict(zip(arms, vals)), **(v or {})})
        print("%-40s %s  %s" % (("* " if gate else "  ") + what, "  ".join("%9s" % ("-" if x is None else round(x, 3)) for x in vals),
                                "loss %+.2f%% range %.2f%% %s" % (v["loss"], v["spread"], v["verdict"]) if v else "missing"))
    bad, notes = [], []
    for x in arms:
        c = data[x]["cold"]
        for p in COLD:
            r = c.get(p)
            if not r or not r["qualified"] or r["state"] != "cold" or r["tokens"] != size(p):
                bad.append("%s cold %s: %s" % (x, p, r))
        o = data[x]["openai"]
        if not o:
            bad.append("%s openai: no report" % x)
        else:
            for key, r in sorted(o.items()):
                if key[1] == 0 and (len(set(r["hashes"])) != 1 or not r["hashes"][0]):
                    bad.append("%s openai %s t=0: %d hashes over the reps" % (x, key[0], len(set(r["hashes"]))))
        for name in LISTS:
            got = data[x][name]
            if not got:
                bad.append("%s %s: no report" % (x, name))
                continue
            for n, c2 in sorted(got.items()):
                if c2["failed"] or c2["alone"].get("unequal") or c2["alone"].get("failed") or c2["serial"].get("unequal") \
                        or c2["serial"].get("failed") or not c2["alone"].get("equal") or not c2["serial"].get("equal"):
                    bad.append("%s %s x%d: failed %s alone %s serial %s" % (x, name, n, c2["failed"], c2["alone"], c2["serial"]))
                for i, h in enumerate(c2["hashes"]):
                    if len(h) != 1 or not h[0]:
                        bad.append("%s %s x%d stream %d: %d hashes over the reps" % (x, name, n, i, len(h)))
    for key in keys:
        hs = [((data[x]["openai"] or {}).get(key) or {}).get("hashes") for x in arms]
        same = bool(all(hs)) and all(h == hs[0] for h in hs)
        if key[1] == 0 and not same:
            bad.append("openai %s t=0: the arms' token_sha differ (%s)" % (key[0], "; ".join("%s %s" % (x, h) for x, h in zip(arms, hs))))
        elif key[1] != 0:
            notes.append("openai %s t=%s: the arms' token_sha %s" % (key[0], key[1], "agree" if same else "differ: %s" % hs))
    for name in LISTS:
        for n in sorted((data[arms[0]][name] or {}).keys()):
            hs = [((data[x][name] or {}).get(n) or {}).get("hashes") for x in arms]
            if not all(hs) or not all(h == hs[0] for h in hs):
                bad.append("%s x%d: the arms' token hashes differ (%s)" % (name, n, "; ".join(
                    "%s %s" % (x, [y[0][:12] if len(y) == 1 and y[0] else y for y in (h or [])]) for x, h in zip(arms, hs))))
    out["exact"], out["notes"] = bad, notes
    print("exactness: %s" % ("every cell, greedy row and cold prompt holds" if not bad else "%d failures" % len(bad)))
    for line in bad:
        print("  " + line)
    for line in notes:
        print("note: " + line)
    for x in arms:
        s = startup(logs, x)
        out["memory"][x] = s
        print("memory %-8s %s" % (x, "no load line" if s is None else
                                  "model %(model)s GiB, %(streams)s streams, %(pages)s pages (%(pages_gib)s GiB), %(fixed)s MiB fixed, "
                                  "%(left)s GiB left, kept %(kept)s, loaded in %(load_s)s s" % s))
    print("fsum: %d slower gate figures, %d slower receipt figures, %d exactness failures, %d figures"
          % (slower, slower_r, len(bad), len(rows)))
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
