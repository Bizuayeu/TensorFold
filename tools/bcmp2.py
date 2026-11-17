"""bcmp2.py RUN_DIR ...: candidate (3)'s glm-score-bench lines (RUN_DIR/bench.txt from 9f51e1cd1d's bench, whose sweep
has "(decodeLoop)" for the new rule and "(376)" for the former nb / 376 rule) side by side. Per context and row count
from 4 rows: tf_glm_scores (A), the new rule, the former rule and the fastest fixed loop, each as "run1 / run2 / ...",
then the new rule against tf_glm_scores and against the former rule (medians over the runs, + slower, - faster)."""
import re
import statistics
import sys

runs = []
for d in sys.argv[1:]:
    cells = {}
    for line in open(f"{d}/bench.txt", encoding="utf-8"):
        m = re.match(r"ctx (\d+) rows (\d+) \(np \d+, nb (\d+), decodeLoop (\d+)\): tf_glm_scores ([\d.]+) / ([\d.]+)", line)
        if not m:
            continue
        ctx, rows, nb, dl, oa, ob = m.groups()
        c = {"oa": float(oa), "dl": int(dl), "fixed": {}}
        for lm in re.finditer(r"new loop (\d+)(?: \((decodeLoop|376)\))? ([\d.]+)", line):
            loop, tag, v = int(lm.group(1)), lm.group(2), float(lm.group(3))
            if tag == "decodeLoop":
                c["rule"] = v
            elif tag == "376":
                c["old"], c["old_loop"] = v, loop
            else:
                c["fixed"][loop] = v
        cells[(int(ctx), int(rows))] = c
    runs.append(cells)
fmt = lambda xs: " / ".join(f"{x:.3f}" for x in xs)
for k in sorted(runs[0]):
    if k[1] < 4:
        continue
    cs = [r[k] for r in runs if k in r]
    base = [c["oa"] for c in cs]
    rule = [c["rule"] for c in cs]
    old = [c["old"] for c in cs]
    best = [min(c["fixed"].values()) for c in cs]
    bl = sorted({min(c["fixed"], key=c["fixed"].get) for c in cs})
    vs_k = statistics.median(rule) / statistics.median(base) - 1
    vs_o = statistics.median(rule) / statistics.median(old) - 1
    print(f"ctx {k[0]:>7} rows {k[1]:>2}: tf_glm_scores {fmt(base)}; rule (loop {cs[0]['dl']}) {fmt(rule)}; "
          f"376 (loop {cs[0]['old_loop']}) {fmt(old)}; best fixed {bl} {fmt(best)}; rule {vs_k:+.1%} vs tf_glm_scores, "
          f"{vs_o:+.1%} vs 376")
