"""On 04, while a server answers at BASE: grid.py BASE LABEL BENCH [--tokens N] [--reps N] [--deadline S]
[--contexts C,C] -- T2 of plans/MEASURE_ORDER_PLAN.md: the engine's step time without drafts ("draft": false,
ignore_eos, greedy) at 1 and 4 streams, and 8 where /health's max_batch_size holds them, on prompts of 4,096, 65,536,
262,144 and 1,040,384 tokens (1,048,576 less 8,192 for the streams' own tokens and replies), or on the --contexts named
(a rising list out of those four; empty or blank keeps all four, anything else stops the grid before its first request).

Every launch runs the same prompts: one token list, BENCH/grid-ids.json, made the first time from BENCH/corpus-grid.txt
by the server's /tokenize. The shared prefix grows in raw /v1/completions requests of at most 65,536 new tokens, 30 s
apart, each one token past a chunk start so the kept prompt sits there (TF_KV_SHARE, and TF_GLM_PROMPT_TAIL=1 on the
server: without it a raw prompt keeps no state); each request after the first must resume within a chunk of where the
one before ended, or the grid stops (a lost state would rebuild the prefix).
Each context's cells run as soon as the prefix reaches it. A stream's prompt is the prefix, the token after it (a kept
state resumes only on the token it was kept with, prompt_radix.find) and 31 tokens of its own.

A rep starts its streams together. A round gap is the time between two text chunks of one stream inside the window
where every stream of the rep is decoding; the rep's round time is the nearest-rank median of them all (bench_metrics'
rank). A rep counts only when every stream completed its tokens with drafts off, resumed the prefix, sent about a chunk
a token and, beside other streams, decoded at least 64 rounds inside the window. A cell is the median of its counted
reps (two at least). Writes BENCH/LABEL-grid.json after every rep and lines to BENCH/LABEL.log; the log's last line is
"grid LABEL end" or "grid LABEL STOP: why".
"""
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

MODEL = "glm-tf"
CONTEXTS = (4096, 65536, 262144, 1040384)
CHUNK = 2048              # GLM's prompt pass and kept-state grid (docs/recipes/glm-5.3-flash-cuda.md)
STEP = 65536              # new prompt tokens a request prefills at most: bench4.sh's longest cold prompt on these hosts
OWN = 32                  # a stream's tokens after the prefix: the prefix's next token, then 31 of its own
OWN_AT = 1048576          # where the streams' own tokens start in the token list, 64 apart
MIN_GAPS = 64             # rounds a stream must decode inside the window when it runs beside others
REST_STEP, REST_CELL, REST_REP = 30, 30, 10


class Stop(Exception):
    pass


def rank(values, q):
    """Nearest-rank percentile of an ascending list (bench_metrics.rank); None for an empty one."""
    return values[max(0, math.ceil(len(values) * q) - 1)] if values else None


def window(runs):
    """(lo, hi): from the last stream's first chunk to the first stream's last one; None when a stream sent none."""
    if not runs or any(not r["pieces"] for r in runs):
        return None
    return max(r["pieces"][0] for r in runs), min(r["pieces"][-1] for r in runs)


def gaps(pieces, lo, hi):
    """A stream's gaps in ms between consecutive chunks that both arrived inside [lo, hi]."""
    return [(b - a) * 1000 for a, b in zip(pieces, pieces[1:]) if lo <= a and b <= hi]


def judge(runs, tokens, context):
    """A rep's figures and the reasons it does not count (an empty list: it counts)."""
    why = []
    for i, r in enumerate(runs):
        u, rt = r.get("usage") or {}, r.get("runtime") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if r.get("error") or not r.get("complete"):
            why.append("stream %d: %s" % (i, r.get("error") or "incomplete"))
        elif u.get("completion_tokens") != tokens:
            why.append("stream %d: %s tokens" % (i, u.get("completion_tokens")))
        elif rt.get("drafts") is not False:
            why.append("stream %d: drafts %s" % (i, rt.get("drafts")))
        elif not isinstance(cached, int) or cached < context - CHUNK:
            why.append("stream %d: cached %s of a %d prefix" % (i, cached, context))
        elif len(r["pieces"]) < 0.9 * tokens:
            why.append("stream %d: %d chunks for %d tokens" % (i, len(r["pieces"]), tokens))
    w = window(runs)
    per = [gaps(r["pieces"], *w) for r in runs] if w else [[] for _ in runs]
    if w and len(runs) > 1 and min(len(g) for g in per) < MIN_GAPS:
        why.append("streams overlap for %d rounds" % min(len(g) for g in per))
    pooled = sorted(g for p in per for g in p)
    shas = [(r.get("runtime") or {}).get("token_sha") for r in runs]
    ttft = [r["pieces"][0] - r["sent"] for r in runs if r["pieces"] and r.get("sent")]
    fig = {"round_ms": rank(pooled, 0.5), "round_p95_ms": rank(pooled, 0.95),
           "round_mean_ms": statistics.fmean(pooled) if pooled else None, "gaps": len(pooled),
           "window_s": (w[1] - w[0]) if w else None, "ttft_max_s": max(ttft) if ttft else None,
           "cached": [((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") for r in runs],
           "token_sha": shas, "distinct_replies": len(set(shas)),
           "chunks_a_token": (sum(len(r["pieces"]) for r in runs) / (tokens * len(runs))) if runs else None,
           "runtime": [{k: v for k, v in (r.get("runtime") or {}).items() if k not in ("prefill_widths", "prefill_raised")}
                       for r in runs]}
    if len(runs) > 1 and fig["distinct_replies"] < len(runs):
        why.append("%d streams, %d distinct replies" % (len(runs), fig["distinct_replies"]))
    return fig, why


def cell_value(reps):
    """The median round time of a cell's counted reps; None with fewer than two."""
    good = [r["round_ms"] for r in reps if not r["why"] and r["round_ms"] is not None]
    return statistics.median(good) if len(good) >= 2 else None


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream(base, prompt, tokens, gate):
    """One streamed raw completion: every text chunk's arrival (perf_counter), the usage and the runtime fields."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True, "draft": False}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = {"pieces": [], "usage": {}, "runtime": {}, "complete": False, "error": None, "sent": None}
    try:
        gate.wait(timeout=60)
        out["sent"] = time.perf_counter()
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    out["complete"] = True
                    break
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    out["error"] = "server: " + json.dumps(chunk["error"])[:200]
                    break
                out["usage"] = chunk.get("usage") or out["usage"]
                out["runtime"] = chunk.get("tensorfold") or out["runtime"]
                if any(c.get("text") for c in chunk.get("choices", [])):
                    out["pieces"].append(time.perf_counter())
    except Exception as e:  # noqa: BLE001 - a failed stream is reported in the rep
        out["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        gate.abort()
    return out


def together(base, prompts, tokens):
    gate = threading.Barrier(len(prompts))
    runs = [None] * len(prompts)

    def one(i):
        runs[i] = stream(base, prompts[i], tokens, gate)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return runs


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return "nvidia-smi: %s" % type(e).__name__


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def token_list(base, bench, need):
    path = bench + "/grid-ids.json"
    if not os.path.exists(path):
        with open(bench + "/corpus-grid.txt", encoding="utf-8") as f:
            text = f.read()
        ids = post(base, "/tokenize", {"prompt": text, "add_special_tokens": False}, 600)["tokens"]
        if len(ids) < need:
            raise Stop("corpus-grid.txt has %d tokens, short of %d" % (len(ids), need))
        with open(path + ".tmp", "w") as f:
            json.dump(ids, f)
        os.replace(path + ".tmp", path)
    with open(path) as f:
        ids = json.load(f)
    if len(ids) < need:
        raise Stop("grid-ids.json has %d tokens, short of %d" % (len(ids), need))
    return ids, sha16(path)


def parse_contexts(text):
    """--contexts: a rising comma list out of CONTEXTS; empty or blank keeps all of them."""
    if not text.strip():
        return CONTEXTS
    try:
        got = tuple(int(x) for x in text.split(","))
    except ValueError:
        got = ()
    if not got or any(c not in CONTEXTS for c in got) or list(got) != sorted(set(got)):
        raise ValueError("--contexts %r: give a rising comma list out of %s" % (text, ",".join(map(str, CONTEXTS))))
    return got


def main(argv):
    base, label, bench = argv[:3]
    opt = dict(zip(argv[3::2], argv[4::2]))
    tokens, nreps, deadline = int(opt.get("--tokens", 256)), int(opt.get("--reps", 3)), int(opt.get("--deadline", 3300))
    t0 = time.time()
    logf = open("%s/%s.log" % (bench, label), "a")

    def log(msg):
        logf.write("%s %s\n" % (time.strftime("%F %T"), msg))
        logf.flush()

    try:
        contexts = parse_contexts(opt.get("--contexts", ""))
    except ValueError as e:
        logf.write("grid %s STOP: %s\n" % (label, e))
        logf.close()
        return 1
    with urllib.request.urlopen(base + "/health", timeout=30) as r:
        health = json.load(r)
    levels = [1, 4] + ([8] if (health.get("max_batch_size") or 0) >= 8 else [])
    out = {"label": label, "client": sha16(__file__), "base": base, "health": health, "tokens": tokens, "reps": nreps,
           "contexts": list(contexts), "levels": levels, "prefix": [], "cells": [], "end": None}

    def save():
        with open("%s/%s-grid.json.tmp" % (bench, label), "w") as f:
            json.dump(out, f)
        os.replace("%s/%s-grid.json.tmp" % (bench, label), "%s/%s-grid.json" % (bench, label))

    def room(need_s, what):
        if time.time() - t0 + need_s > deadline:
            raise Stop("deadline: %d s gone, %s needs about %d more" % (time.time() - t0, what, need_s))

    log("== grid %s begin: levels %s, contexts %s, %d tokens, %d reps, client %s" % (
        label, levels, list(contexts), tokens, nreps, out["client"]))
    try:
        ids, out["ids"] = token_list(base, bench, OWN_AT + 64 * (max(levels) - 1) + OWN)
        log("  token list %s: %d tokens" % (out["ids"], len(ids)))
        own = [ids[OWN_AT + 64 * i:OWN_AT + 64 * i + OWN] for i in range(max(levels))]
        have = 0
        for context in contexts:
            while have < context:
                end = min(context, have + STEP)
                room(240, "the prefix to %d" % end)
                t = time.perf_counter()
                r = post(base, "/v1/completions", {"model": MODEL, "prompt": ids[:end] + [ids[end]], "max_tokens": 1,
                                                   "temperature": 0, "draft": False}, 1800)
                u = r.get("usage") or {}
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                step = {"to": end, "s": time.perf_counter() - t, "prompt_tokens": u.get("prompt_tokens"),
                        "cached": cached, "gpu": gpu()}
                out["prefix"].append(step)
                save()
                log("  prefix %d: %.1f s, prompt %s, cached %s, gpu %s" % (end, step["s"], u.get("prompt_tokens"),
                                                                           cached, step["gpu"]))
                if have and (not isinstance(cached, int) or cached < have - CHUNK):
                    raise Stop("the prefix to %d resumed at %s, not near %d: the kept prompt is gone" % (end, cached, have))
                have = end
                time.sleep(REST_STEP)
            for n in levels:
                cell = {"context": context, "streams": n, "reps": [], "round_ms": None}
                out["cells"].append(cell)
                for k in range(nreps):
                    room(150, "%d x%d rep %d" % (context, n, k + 1))
                    heat = gpu()
                    runs = together(base, [ids[:context + 1] + own[i][:OWN - 1] for i in range(n)], tokens)
                    fig, why = judge(runs, tokens, context)
                    fig.update(why=why, gpu=heat)
                    cell["reps"].append(fig)
                    cell["round_ms"] = cell_value(cell["reps"])
                    save()
                    log("  %d x%d rep %d: round %s ms p95 %s (gaps %d, window %s s), ttft max %s s, cached %s, "
                        "chunks a token %.2f, replies %d, gpu %s%s" % (
                            context, n, k + 1, _f(fig["round_ms"]), _f(fig["round_p95_ms"]), fig["gaps"],
                            _f(fig["window_s"], 1), _f(fig["ttft_max_s"]), fig["cached"], fig["chunks_a_token"] or 0,
                            fig["distinct_replies"], heat, (" -- does not count: " + "; ".join(why)) if why else ""))
                    if any("cached" in w for w in why):
                        raise Stop("%d x%d: a stream did not resume the prefix (%s)" % (context, n, "; ".join(why)))
                    if any(r.get("error") for r in runs):
                        raise Stop("%d x%d: a stream failed (%s)" % (context, n, "; ".join(why)))
                    time.sleep(REST_REP)
                log("  cell %d x%d: %s ms" % (context, n, _f(cell["round_ms"])))
                time.sleep(REST_CELL)
        out["end"] = "end"
    except Stop as e:
        out["end"] = "STOP: %s" % e
    except Exception as e:  # noqa: BLE001 - the log's last line says why
        out["end"] = "STOP: %s: %s" % (type(e).__name__, str(e)[:300])
    save()
    logf.write("grid %s %s\n" % (label, out["end"]))
    logf.close()
    return 0 if out["end"] == "end" else 1


def _f(v, digits=3):
    return "-" if v is None else ("%." + str(digits) + "f") % v


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
