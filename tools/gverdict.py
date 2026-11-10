"""python3 gverdict.py BENCH MAIN,MAIN,... HEAD,HEAD,... -- grid.py's cells (BENCH/LABEL-grid.json, the decode round in
ms) under fsum.py's rule: per cell each arm's round, the head's mean against main's (+ longer), main's range
((max - min) / mean) and the word (slower, within, faster, open; fsum.py's verdict, lower is better). A cell some arm
did not count (round_ms None, as TP=2's 1016K x4 whose streams do not overlap) shows "-" and gets no verdict."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsum  # noqa: E402


def grid(bench, label):
    with open(os.path.join(bench, "%s-grid.json" % label), encoding="utf-8") as f:
        return {(c["context"], c["streams"]): c for c in json.load(f)["cells"]}


def main(argv):
    bench, mains, heads = argv[0], argv[1].split(","), argv[2].split(",")
    A = [grid(bench, x) for x in mains]
    B = [grid(bench, x) for x in heads]
    for k in A[0]:
        a = [g[k]["round_ms"] for g in A]
        b = [g[k]["round_ms"] for g in B]
        v = fsum.verdict(a, b, higher=False)
        word = ("%+.2f%%  main's range %.2f%%  %s" % (v["loss"], v["spread"], v["verdict"]) if v
                else "no verdict: a cell did not count")
        print("%dK x%d  main %s  head %s  %s" % (k[0] // 1024, k[1], ms(a), ms(b), word))


def ms(values):
    return " ".join("-" if x is None else "%.3f" % x for x in values)


if __name__ == "__main__":
    main(sys.argv[1:])
