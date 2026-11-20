"""The PR's TP=4 tables: python3 gtable.py DIR RUN_ORDER MAIN,MAIN,... HEAD,HEAD,... -- DIR holds each arm's
draft-ARM.json (m2draft.py) and ARM-grid.json (grid.py); RUN_ORDER the counted arms in the order they ran.

Prints three markdown tables: the drafted reps in run order (tok/s, 3 reps of 512 tokens an arm); the grid's decode
round in ms, main's mean against this head's per cell with the change (+ longer); the grid in run order. Then, under
"verdicts:", each grid cell's word under fsum.py's rule (gverdict.py's) and dsum.py's checks and figures for the
drafted reps. Exits 0 when every grid cell is within or faster and dsum.py's checks hold, 2 otherwise (a cell slower
or open, or drafts that differ), so that what follows can stop for words."""
import io
import math
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsum  # noqa: E402
import fsum  # noqa: E402
from gverdict import grid  # noqa: E402

CTX = [(4096, "4K"), (65536, "64K"), (262144, "256K"), (1040384, "1016K")]
LEVELS = [1, 4, 8]


def kind(x, mains):
    return "main" if x in mains else "this head"


def main(argv):
    if len(argv) != 4:
        sys.exit(__doc__)
    d, order, mains, heads = argv[0], argv[1].split(","), argv[2].split(","), argv[3].split(",")
    if sorted(order) != sorted(mains + heads):
        sys.exit("RUN_ORDER names other arms than MAIN and HEAD")
    R = {x: dsum.reps(d, x) for x in order}
    G = {x: grid(d, x) for x in order}
    print("| TP=4 (ring, gather), drafted decode after a 1,040,384-token prompt, tok/s | %s |" % " | ".join(kind(x, mains) for x in order))
    print("|---|%s" % ("---|" * len(order)))
    print("| %d reps of 512 tokens | %s |" % (len(R[order[0]]), " | ".join(
        " / ".join("%.2f" % r["tensorfold"]["tokens_per_second"] for r in R[x]) for x in order)))
    print()
    print("| TP=4 (ring, gather), decode round in ms, mean of %d runs of main and %d of this head | main | this head | "
          "this head against main (+ longer, − shorter) |" % (len(mains), len(heads)))
    print("|---|---|---|---|")
    words, gm = [], {}
    for ctx, name in CTX:
        a, b, c = [], [], []
        for n in LEVELS:
            ma = [G[x][(ctx, n)]["round_ms"] for x in mains]
            hb = [G[x][(ctx, n)]["round_ms"] for x in heads]
            if None in ma or None in hb:
                a.append("-"), b.append("-"), c.append("-")
                words.append("%s x%d: no verdict, a cell did not count" % (name, n))
                continue
            v = fsum.verdict(ma, hb, higher=False)
            a.append("%.3f" % (sum(ma) / len(ma)))
            b.append("%.3f" % (sum(hb) / len(hb)))
            c.append(("%+.2f%%" % v["loss"]).replace("-", "−"))
            words.append("%s x%d: %+.2f%%, main's range %.2f%%, %s" % (name, n, v["loss"], v["spread"], v["verdict"]))
        print("| %s, 1 / 4 / 8 streams | %s | %s | %s |" % (name, " / ".join(a), " / ".join(b), " / ".join(c)))
    print()
    print("| TP=4 (ring, gather), decode round in ms, in run order | %s |" % " | ".join(kind(x, mains) for x in order))
    print("|---|%s" % ("---|" * len(order)))
    for ctx, name in CTX:
        print("| %s, 1 / 4 / 8 streams | %s |" % (name, " | ".join(" / ".join(
            "-" if G[x][(ctx, n)]["round_ms"] is None else "%.3f" % G[x][(ctx, n)]["round_ms"] for n in LEVELS) for x in order)))
    for x in order:
        v = [G[x][k]["round_ms"] for k in G[x] if G[x][k]["round_ms"] is not None]
        gm[x] = math.exp(sum(math.log(t) for t in v) / len(v)) if v else None
    print()
    print("verdicts:")
    for w in words:
        print("  grid " + w)
    print("  grid geometric means over the cells each arm counted: " + ", ".join(
        "%s %.3f ms" % (x, gm[x]) if gm[x] else "%s -" % x for x in order))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = dsum.main([d, ",".join(mains), ",".join(heads), "--order", ",".join(order)])
    for line in buf.getvalue().splitlines():
        print("  drafted " + line)
    clean = rc == 0 and all(w.endswith(("within", "faster")) for w in words)
    print("verdicts: " + ("every grid cell within or faster, the drafted reps' checks hold" if clean else "STOP: words needed"))
    return 0 if clean else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
