"""python3 gsum.py BENCH LABEL [LABEL ...] -- T2's grids side by side (BENCH/LABEL-grid.json, grid.py): each cell's
round time in ms for every label and its ratio to the first label's; the spread of the counted reps inside a cell
((max - min) / median); whether a cell's replies are the first label's (token_sha of every stream in every rep); and
per label the geometric mean over the cells every label counted, its ratio to the first label's, the GPU's temperature
and SM clock range at the reps' starts (rank 0) and any runtime field naming heat that is not zero.
"""
import json
import math
import sys


def gm(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def counted(cell):
    return [r for r in cell["reps"] if not r["why"] and r["round_ms"] is not None]


def spread(cell):
    v = sorted(r["round_ms"] for r in counted(cell))
    return (v[-1] - v[0]) / v[len(v) // 2] if len(v) >= 2 else None


def replies(cell):
    return [tuple(r["token_sha"]) for r in cell["reps"]]


def gpu_range(grid):
    temps, clocks = [], []
    for c in grid["cells"]:
        for r in c["reps"]:
            parts = [p.strip() for p in str(r.get("gpu", "")).split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                temps.append(int(parts[0]))
                clocks.append(int(parts[1]))
    return "%s-%s C, %s-%s MHz" % (min(temps), max(temps), min(clocks), max(clocks)) if temps else "-"


def heat(grid):
    seen = set()
    for c in grid["cells"]:
        for r in c["reps"]:
            for rt in r.get("runtime") or []:
                for k, v in rt.items():
                    if "heat" in k and v not in (None, 0, 0.0, False):
                        seen.add("%d x%d %s=%s" % (c["context"], c["streams"], k, v))
    return sorted(seen)


def main(argv):
    bench, labels = argv[0], argv[1:]
    grids = []
    for label in labels:
        with open("%s/%s-grid.json" % (bench, label)) as f:
            grids.append(json.load(f))
    cells = [{(c["context"], c["streams"]): c for c in g["cells"]} for g in grids]
    keys = []
    for g in grids:
        for c in g["cells"]:
            if (c["context"], c["streams"]) not in keys:
                keys.append((c["context"], c["streams"]))
    common = [k for k in keys if all(cs.get(k) and cs[k]["round_ms"] is not None for cs in cells)]
    for g in grids:
        print("%s: %s, client %s, ids %s, levels %s, %s tokens, %s reps, version %s" % (
            g["label"], g["end"], g["client"], g.get("ids"), g["levels"], g["tokens"], g["reps"],
            (g.get("health") or {}).get("version")))
    print("%-16s" % "cell" + "".join("%12s" % l for l in labels) + "".join("%10s" % ("/" + labels[0]) for l in labels[1:])
          + "   spread %   same replies")
    for k in keys:
        vals = [cs.get(k, {}).get("round_ms") for cs in cells]
        row = "%-16s" % ("%dK x%d" % (k[0] // 1024, k[1]))
        row += "".join("%12s" % ("-" if v is None else "%.3f" % v) for v in vals)
        row += "".join("%10s" % ("-" if None in (v, vals[0]) else "%+.2f%%" % (100 * (v / vals[0] - 1))) for v in vals[1:])
        sp = [spread(cs[k]) if cs.get(k) else None for cs in cells]
        row += "   " + " ".join("-" if s is None else "%.2f" % (100 * s) for s in sp)
        first = replies(cells[0][k]) if cells[0].get(k) else None
        row += "   " + " ".join("-" if not cs.get(k) else ("yes" if replies(cs[k]) == first else "no") for cs in cells[1:])
        print(row)
    means = [gm([cs[k]["round_ms"] for k in common]) for cs in cells]
    print("geometric mean over %d cells every label counted:" % len(common))
    for label, g, m in zip(labels, grids, means):
        print("  %-8s %s ms%s, gpu %s%s" % (label, "-" if m is None else "%.3f" % m,
                                            "" if m is None or means[0] is None or label == labels[0]
                                            else " (%+.2f%%)" % (100 * (m / means[0] - 1)), gpu_range(g),
                                            ("; heat " + ", ".join(heat(g))) if heat(g) else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
