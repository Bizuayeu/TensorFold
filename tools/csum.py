"""B2's reports (copies4.sh's BENCH/LABEL-LIST.json, in bench_concurrent.py's form), on 04 or the PC:
  csum.py --line REPORT             one line: each cell's per-stream and aggregate tok/s, round gap p50, tokens a chunk,
                                    and the equal/unequal/failed counts against alone and against "draft": false
  csum.py --check BENCH LABEL WANT  the arm passes: copies4.sh's end line, rank 0's engine environment holding
                                    TF_GLM_COPIES=1 WANT times (0 or 1), every list's report with a cell at 1 and at 4
                                    streams, no failed request, every reply equal to its request alone and every alone
                                    run to "draft": false. Prints what fails and ends 1; notes (a cell left untimed, a
                                    tool list without tool calls) do not fail
  csum.py BENCH OFF ON ON OFF       side by side: per list and streams each arm's figures, the OFF pair's and the ON
                                    pair's means, ON/OFF and each pair's spread, and the geometric mean of ON/OFF
"""
import json
import math
import os
import re
import sys

LISTS = ("prose", "code", "edit", "tool", "lprose", "ledit")
LEVELS = (1, 4)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cells(report):
    return {c["streams"]: c for c in report.get("cells", []) if c.get("temperature") == 0}


def figures(cell):
    """A cell's per-stream and aggregate tok/s, round gap and tokens a chunk; the tool list's toolbench.py also has the
    server's own tok/s (the reply's tokens_per_second), which stands in when the client could not time the stream."""
    pooled = cell.get("pooled") or {}
    out = {"tps": cell.get("per_stream_tps"), "agg": cell.get("aggregate_tps"), "gap": pooled.get("round_gap_p50_ms"),
           "tpc": pooled.get("tokens_per_chunk")}
    srv = (cell.get("request") or {}).get("server_tps")
    if srv is not None:
        out["srv"] = srv
    return out


def counts(cell, key):
    c = cell.get(key) or {}
    return c.get("equal", 0), c.get("unequal", 0), c.get("failed", 0)


def line(report):
    out = []
    for n, cell in sorted(cells(report).items()):
        f = figures(cell)
        out.append("x%d %s/s agg %s gap %s ms tpc %s alone %d/%d/%d serial %d/%d/%d" % (
            n, f["tps"], f["agg"], f["gap"], f["tpc"], *counts(cell, "alone"), *counts(cell, "serial")))
    return " | ".join(out) or "no cells"


def check(bench, label, want):
    bad, notes = [], []
    try:
        with open(os.path.join(bench, label + ".log"), encoding="utf-8") as f:
            log = f.read().splitlines()
    except OSError as e:
        return ["no %s.log (%s)" % (label, type(e).__name__)], notes
    if "copies4 %s end" % label not in log:
        bad.append("%s.log has no 'copies4 %s end'" % (label, label))
    env = [m.group(1) for m in (re.search(r"engine env TF_GLM_COPIES=1: (\S+)$", x) for x in log) if m]
    if env != [str(want)]:
        bad.append("rank 0's engine environment: TF_GLM_COPIES=1 counted %s, not %d" % (env or "nowhere", want))
    for name in LISTS:
        path = os.path.join(bench, "%s-%s.json" % (label, name))
        try:
            report = load(path)
        except (OSError, ValueError) as e:
            bad.append("%s: no report (%s)" % (name, type(e).__name__))
            continue
        got = cells(report)
        for n in LEVELS:
            cell = got.get(n)
            if cell is None:
                bad.append("%s: no cell at %d streams" % (name, n))
                continue
            if cell.get("failed"):
                bad.append("%s x%d: %d failed requests %s" % (name, n, cell["failed"], cell.get("errors", "")))
            for key in ("alone", "serial"):
                eq, ne, fail = counts(cell, key)
                if ne or fail or not eq:
                    bad.append("%s x%d against %s: %d equal, %d unequal, %d failed" % (
                        name, n, "alone" if key == "alone" else '"draft": false', eq, ne, fail))
            if cell.get("unmeasured"):
                notes.append("%s x%d: %d requests untimed (fewer than 4 text pieces)" % (name, n, cell["unmeasured"]))
            if name == "tool" and not (cell.get("request") or {}).get("tool_call"):
                notes.append("tool x%d: the replies' median has no tool call" % n)
    return bad, notes


def summary(bench, labels):
    off, on = (labels[0], labels[3]), (labels[1], labels[2])
    ratios = []
    print("list  x  | " + " | ".join("%s tok/s agg gap tpc" % x for x in labels) + " | OFF mean  ON mean  ON/OFF  "
          "spread OFF/ON | equal")
    for name in LISTS:
        got = {x: cells(load(os.path.join(bench, "%s-%s.json" % (x, name)))) for x in labels}
        for n in LEVELS:
            fig = {x: figures(got[x][n]) if n in got[x] else {} for x in labels}
            tps = {x: fig[x].get("tps") for x in labels}
            by = ""
            if not all(tps.values()) and all(fig[x].get("srv") for x in labels):
                tps, by = {x: fig[x]["srv"] for x in labels}, " (server tok/s)"
            row = " | ".join("%s %s %s %s%s" % (fig[x].get("tps"), fig[x].get("agg"), fig[x].get("gap"), fig[x].get("tpc"),
                                                " srv %s" % fig[x]["srv"] if "srv" in fig[x] else "") for x in labels)
            same = all(got[x].get(n) and not any(counts(got[x][n], k)[1:]) for x in labels for k in ("alone", "serial"))
            if all(tps[x] for x in labels):
                mo, mn = sum(tps[x] for x in off) / 2, sum(tps[x] for x in on) / 2
                ratios.append(mn / mo)
                tail = "%.1f  %.1f  %.4f  %.2f%%/%.2f%%" % (mo, mn, mn / mo, abs(tps[off[0]] - tps[off[1]]) / mo * 100,
                                                           abs(tps[on[0]] - tps[on[1]]) / mn * 100)
            else:
                tail = "-"
            print("%-6s x%d | %s | %s%s | %s" % (name, n, row, tail, by, "yes" if same else "NO"))
    if ratios:
        print("geometric mean of ON/OFF per-stream tok/s over %d cells: %.4f" % (
            len(ratios), math.exp(sum(math.log(r) for r in ratios) / len(ratios))))


def main(argv):
    if argv[:1] == ["--line"]:
        print(line(load(argv[1])))
        return 0
    if argv[:1] == ["--check"]:
        bad, notes = check(argv[1], argv[2], int(argv[3]))
        for x in notes:
            print("csum note: %s" % x)
        for x in bad:
            print("csum FAIL: %s" % x)
        print("csum %s: %s" % (argv[2], "pass" if not bad else "FAIL (%d)" % len(bad)))
        return 1 if bad else 0
    summary(argv[0], argv[1:5])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
