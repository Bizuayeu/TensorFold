"""m2draft.py BASE IDS CONTEXT REPS (on 04, after m2hook.sh's grid kept the prefix; <the local records>, <a local session>, 11-16, M2 of <the local ledger> 11-16 18:4x): REPS streamed raw completions one after another with drafts on
(the server's MTP and copies), each the grid's stream 0 prompt at CONTEXT (IDS' first CONTEXT + 1 tokens, then the 31
tokens from 1,048,576; grid.py's OWN), 512 tokens, ignore_eos, greedy. prfdraft.py (<the local records>) without
its nsys window. One JSON line a rep: rep, ok, tokens, cached, ttft, seconds, the reply's speculative and tensorfold
fields. Exits 0 when every rep ended with [DONE] and no error."""

import json
import sys
import time
import urllib.request

OWN, OWN_AT = 32, 1048576


def one(base: str, prompt: list) -> dict:
    body = {"model": "glm-tf", "prompt": prompt, "max_tokens": 512, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    last = {}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, done, err = time.perf_counter(), None, False, None
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if line == "data: [DONE]":
                    done = True
                    break
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                err = chunk.get("error") or err
                for k in ("usage", "speculative", "tensorfold"):
                    if chunk.get(k):
                        last[k] = chunk[k]
                if first is None and any(c.get("text") for c in chunk.get("choices") or []):
                    first = time.perf_counter()
    except Exception as e:  # noqa: BLE001 - the record keeps whatever stopped the request
        err = f"{type(e).__name__}: {e}"
    usage = last.get("usage") or {}
    return {"ok": done and err is None, "tokens": usage.get("completion_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft_s": round(first - t0, 3) if first else None, "s": round(time.perf_counter() - t0, 2),
            "speculative": last.get("speculative"), "tensorfold": last.get("tensorfold"), "error": err}


def main(base: str, ids_path: str, context: str, reps: str) -> None:
    ids = json.load(open(ids_path, encoding="utf-8"))
    ids = ids["ids"] if isinstance(ids, dict) else ids
    n = int(context)
    prompt = ids[:n + 1] + ids[OWN_AT:OWN_AT + OWN - 1]
    good = True
    for k in range(int(reps)):
        rec = {"rep": k + 1, **one(base, prompt)}
        good = good and rec["ok"]
        print(json.dumps(rec), flush=True)
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
