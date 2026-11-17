"""x4pool.py FBENCH MAINS HEADS (11-17, <a local session>): the decode cells' pooled round, by stream count, from
fbench's LABEL-decode.json (MAINS and HEADS comma lists of labels), as the ledger's 08:4x breakdown read rf3c: for each
cell the arms' mean of pooled tokens_per_chunk and round_gap_p50_ms, aggregate_tps and per_stream_tps, then the heads'
mean against the mains' (+ longer or more, - shorter or less). Reads only; prints one line a stream count."""
import json
import os
import sys

fb, mains, heads = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")


def cells(label):
    with open(os.path.join(fb, f"{label}-decode.json"), encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for c in d["cells"]:
        p = c["pooled"]
        out[c["streams"]] = (p["tokens_per_chunk"], p["round_gap_p50_ms"], c["aggregate_tps"], c["per_stream_tps"])
    return out


def mean(xs):
    return sum(xs) / len(xs)


m = [cells(x) for x in mains]
h = [cells(x) for x in heads]
for n in sorted(m[0]):
    mv = [mean([a[n][i] for a in m]) for i in range(4)]
    hv = [mean([a[n][i] for a in h]) for i in range(4)]
    pct = [100 * (b / a - 1) for a, b in zip(mv, hv)]
    print(f"{n} streams: tokens a round {mv[0]:.3f} / {hv[0]:.3f} ({pct[0]:+.2f}%), round p50 {mv[1]:.2f} / {hv[1]:.2f} ms "
          f"({pct[1]:+.2f}%), all streams {mv[2]:.2f} / {hv[2]:.2f} tok/s ({pct[2]:+.2f}%), a stream {mv[3]:.2f} / {hv[3]:.2f} "
          f"({pct[3]:+.2f}%); main's rounds {', '.join(f'{a[n][1]:.2f}' for a in m)}, the head's {', '.join(f'{a[n][1]:.2f}' for a in h)}")
