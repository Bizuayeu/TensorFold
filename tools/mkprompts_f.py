"""The F gate's cold prompts (plans/MIA1104_ADOPTION_PLAN.md, 「損が無い」; DECISIONS 11-09 22:5x), tokenized by the
server's /tokenize from bench4.sh's corpus, eight disjoint slices of its tokens in this order:

    python3 mkprompts_f.py BASE CORPUS OUTDIR
    c65536 c32768 c8192 c2048    bench4.sh's mkprompts.py slices (the same tokens as its prompt-N.json)
    f38400a f38400b f9600a f9600b  the plan's two sizes, two of each, after them

No slice shares a token position with another, so no prompt resumes another's kept state. When all eight files are in
OUTDIR they are kept as they are (the gate's arms must send the same tokens); some but not all of them stop the
script, as does a short corpus, before anything is written.
"""

import json
import os
import sys
import urllib.request

SLICES = (("c65536", 65536), ("c32768", 32768), ("c8192", 8192), ("c2048", 2048),
          ("f38400a", 38400), ("f38400b", 38400), ("f9600a", 9600), ("f9600b", 9600))


def main():
    base, corpus, out = sys.argv[1:4]
    have = [os.path.exists(os.path.join(out, name + ".json")) for name, _ in SLICES]
    if all(have):
        print("kept: every prompt file is in", out)
        return 0
    if any(have):
        sys.exit("some prompt files are in %s and some are not: %s" % (out, [n for (n, _), h in zip(SLICES, have) if h]))
    text = open(corpus, encoding="utf-8").read()
    body = json.dumps({"prompt": text, "add_special_tokens": False}).encode()
    req = urllib.request.Request(base + "/tokenize", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        ids = json.load(r)["tokens"]
    need = sum(n for _, n in SLICES)
    if need > len(ids):
        sys.exit(f"the corpus has {len(ids)} tokens, short of {need}")
    start = 0
    for name, n in SLICES:
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump(ids[start:start + n], f)
        print(f"{name}.json: tokens {start} to {start + n} of {len(ids)}", flush=True)
        start += n
    return 0


if __name__ == "__main__":
    sys.exit(main())
