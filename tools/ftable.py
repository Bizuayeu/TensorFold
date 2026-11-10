"""The PR's F-gate tables from fsum.py's JSON: python ftable.py TP FSUM.json [RUN_ORDER] -- RUN_ORDER the counted
arms in the order they ran, comma-separated (default: the chain's alternation, main and head in turn from main)."""
import json
import sys

ROWS = [  # (label, figures in the cell, unit)
    ("decode, a stream's tok/s, {ns} streams", ["decode x%d per stream tok/s"], "tok/s"),
    ("decode, all streams' tok/s, {nm} streams", ["decode x%d aggregate tok/s"], "tok/s"),
    ("cold 9,600 / 38,400, `ttft_s` (two prompts' mean)", ["cold 9600 s (mean of two)", "cold 38400 s (mean of two)"], "s"),
    ("cold 2K / 8K / 32K / 64K, `ttft_s`", ["cold 2048 s", "cold 8192 s", "cold 32768 s", "cold 65536 s"], "s"),
    ("`bench_openai`, t=0, code / chat (tok/s)", ["openai fibonacci-raw t=0.0 tok/s", "openai gpu-chat-no-think t=0.0 tok/s"], "tok/s"),
    ("`bench_openai`, t=1, code / chat (tok/s)", ["openai fibonacci-raw t=1.0 tok/s", "openai gpu-chat-no-think t=1.0 tok/s"], "tok/s"),
]


def fmt(v, unit):
    return "-" if v is None else ("%.3f" % v if unit == "s" else "%.1f" % v)


def main(argv):
    tp, path = argv[0], argv[1]
    d = json.load(open(path, encoding="utf-8"))
    figs = {f["figure"]: f for f in d["figures"]}
    mains, heads = d["main"], d["head"]
    order = argv[2].split(",") if len(argv) > 2 else [x for pair in zip(mains, heads + [None]) for x in pair if x]
    streams = sorted({int(k.split()[1][1:]) for k in figs if k.startswith("decode x")})
    multi = [n for n in streams if n > 1]
    rows = []
    for label, names, unit in ROWS:
        if "{ns}" in label:
            names = [names[0] % n for n in streams]
            label = label.format(ns=" / ".join(map(str, streams)))
        elif "{nm}" in label:
            names = [names[0] % n for n in multi]
            label = label.format(nm=" / ".join(map(str, multi)))
        got = [figs.get(n) for n in names]
        rows.append((label, got, unit))
    mean = lambda f, arms: sum(f["values"][x] for x in arms) / len(arms)
    print("| TP=%s, mean of %d runs of main and %d of this head | main | this head | this head against main (+ slower, − faster) |" % (tp, len(mains), len(heads)))
    print("|---|---|---|---|")
    for label, got, unit in rows:
        if not all(got):
            print("| %s | missing | | |" % label)
            continue
        a = " / ".join(fmt(mean(f, mains), unit) for f in got)
        b = " / ".join(fmt(mean(f, heads), unit) for f in got)
        c = " / ".join(("%+.2f%%" % f["loss"]).replace("-", "−") for f in got)  # fsum's loss: tok/s lost or seconds gained
        print("| %s | %s | %s | %s |" % (label, a, b, c))
    print()
    print("| TP=%s, in run order | %s |" % (tp, " | ".join("main" if x in mains else "this head" for x in order)))
    print("|---|%s" % ("---|" * len(order)))
    for label, got, unit in rows:
        if all(got):
            print("| %s | %s |" % (label, " | ".join(" / ".join(fmt(f["values"][x], unit) for f in got) for x in order)))
    print()
    for f in d["figures"]:
        print("%s%-40s loss %+6.2f%% main's range %5.2f%% %s" % ("* " if f["gate"] else "  ", f["figure"], f["loss"], f["spread"], f["verdict"]))
    gate = [f for f in d["figures"] if f["gate"]]
    print("gate figures: %d, main's widest range %.2f%% (%s); verdicts %s; exactness failures %d" % (
        len(gate), max(f["spread"] for f in gate), max(gate, key=lambda f: f["spread"])["figure"],
        {v: sum(f["verdict"] == v for f in d["figures"]) for v in ("faster", "within", "open", "slower")}, len(d["exact"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
