"""python3 dsum.py DIR MAIN,MAIN,... HEAD,HEAD,... [--extra LABEL,...] [--json OUT] -- the drafted reps side by side
(DIR/draft-LABEL.json, m2draft.py: one JSON line a rep, 512 tokens with drafts on after the kept 1,040,384-token
prompt). For every arm, in the order given (mains and heads alternated as they ran: pass them in run order with
--order, or the mains then the heads): each rep's tok/s, rounds, drafted and accepted tokens, and token_sha.

Checks, before any figure: every rep ok with 512 tokens, the reply's speculative and tensorfold fields; one token_sha
over every rep of every arm (--extra adds arms, such as the uncounted timed starts, to this check only); and the same
rounds, drafted and accepted tokens at each rep in every counted arm. A rep's tok/s compares with another's only where
the drafts are the same, so when they differ the arms' drafts are printed and no figure is (exit 2).

Figures: a run's tok/s is the median of its reps. This head against main: the median of all the head's reps against
the median of all main's reps (+ faster), and the rule fsum.py states (fsum.verdict over the runs' medians, tok/s higher
is better; slower, within, faster or open). Exit 0 when every check holds, 2 when one does not, 1 on unreadable input."""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsum  # noqa: E402


def reps(d, label):
    with open(os.path.join(d, "draft-%s.json" % label), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.startswith("{")]


def bad_rep(r):
    tf, sp = r.get("tensorfold") or {}, r.get("speculative") or {}
    return (not r.get("ok") or r.get("tokens") != 512 or not tf.get("token_sha") or "tokens_per_second" not in tf
            or not {"rounds", "drafted", "accepted"} <= set(sp))


def drafts(r):
    sp = r["speculative"]
    return (sp["rounds"], sp["drafted"], sp["accepted"])


def main(argv):
    args, opts = [], {}
    i = 0
    while i < len(argv):
        if argv[i] in ("--extra", "--json", "--order"):
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    if len(args) != 3:
        sys.exit(__doc__)
    d, mains, heads = args[0], args[1].split(","), args[2].split(",")
    extra = opts["--extra"].split(",") if "--extra" in opts else []
    order = opts["--order"].split(",") if "--order" in opts else mains + heads
    if sorted(order) != sorted(mains + heads):
        sys.exit("--order names other arms than MAIN and HEAD")
    try:
        R = {x: reps(d, x) for x in mains + heads + extra}
    except (OSError, ValueError) as e:
        print("unreadable: %s: %s" % (type(e).__name__, e))
        return 1
    stop = []
    for x, rs in R.items():
        b = [r.get("rep") for r in rs if bad_rep(r)]
        if len(rs) < 3 or b:
            stop.append("%s: %d reps, bad %s" % (x, len(rs), b))
    if stop:
        print("STOP the reps: " + "; ".join(stop))
        return 2
    for x in order + extra:
        rs = R[x]
        kind = "main" if x in mains else "head" if x in heads else "extra (tokens only)"
        print("%-8s %-19s tok/s %s  rounds %s  drafted %s  accepted %s  token_sha %s" % (
            x, kind, " ".join("%.2f" % r["tensorfold"]["tokens_per_second"] for r in rs),
            ",".join(str(r["speculative"]["rounds"]) for r in rs), ",".join(str(r["speculative"]["drafted"]) for r in rs),
            ",".join(str(r["speculative"]["accepted"]) for r in rs), ",".join(sorted({r["tensorfold"]["token_sha"] for r in rs}))))
    shas = {r["tensorfold"]["token_sha256"] if r["tensorfold"].get("token_sha256") else r["tensorfold"]["token_sha"]
            for rs in R.values() for r in rs}
    nrep = sum(len(rs) for rs in R.values())
    out = {"main": mains, "head": heads, "extra": extra, "order": order, "reps": nrep, "token_shas": sorted(shas)}
    ok = True
    if len(shas) == 1:
        print("tokens: one token_sha over the %d reps of %d arms" % (nrep, len(R)))
    else:
        print("STOP tokens: %d token_shas over the %d reps of %d arms" % (len(shas), nrep, len(R)))
        ok = False
    counted = mains + heads
    n = min(len(R[x]) for x in counted)
    if any(len(R[x]) != n for x in counted):
        print("STOP the counted arms ran %s reps" % sorted({len(R[x]) for x in counted}))
        ok = False
    same = all(len({drafts(R[x][k]) for x in counted}) == 1 for k in range(n))
    out["drafts"] = [[drafts(R[x][k]) for x in order] for k in range(n)]
    if same:
        print("drafts: " + "; ".join("rep %d took %d rounds, drafted %d and accepted %d tokens in every counted arm" % (
            (k + 1,) + drafts(R[counted[0]][k])) for k in range(n)))
    else:
        print("STOP drafts differ between the counted arms (the speeds do not compare):")
        for k in range(n):
            print("  rep %d: %s" % (k + 1, ", ".join("%s %d/%d/%d" % ((x,) + drafts(R[x][k])) for x in order)))
        ok = False
    out["ok"] = ok
    if ok:
        med = {x: statistics.median(r["tensorfold"]["tokens_per_second"] for r in R[x]) for x in counted}
        ma = statistics.median(r["tensorfold"]["tokens_per_second"] for x in mains for r in R[x])
        hb = statistics.median(r["tensorfold"]["tokens_per_second"] for x in heads for r in R[x])
        v = fsum.verdict([round(med[x], 2) for x in mains], [round(med[x], 2) for x in heads], higher=True)
        print("runs' medians: " + ", ".join("%s %.2f" % (x, med[x]) for x in order))
        print("head's %d reps' median %.3f tok/s against main's %d reps' median %.3f: %+.1f%%" % (
            sum(len(R[x]) for x in heads), hb, sum(len(R[x]) for x in mains), ma, (hb / ma - 1) * 100))
        print("rule (fsum.verdict over the runs' medians, higher is better): loss %+.2f%%, main's range %.2f%%, %s" % (
            v["loss"], v["spread"], v["verdict"]))
        out.update({"run_median": {x: round(med[x], 2) for x in counted}, "main_median": round(ma, 3),
                    "head_median": round(hb, 3), "gain_pct": round((hb / ma - 1) * 100, 1), "verdict": v,
                    "tok_s": {x: [round(r["tensorfold"]["tokens_per_second"], 2) for r in R[x]] for x in order}})
    if "--json" in opts:
        with open(opts["--json"], "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
