"""Rep-level means (3 reps after the warm-up) of a bench_openai t=0 row, by binary: ARGS = LOGDIR ROW(code|chat) A-arms B-arms."""
import re, statistics as st, sys, pathlib
D, row, A, B = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
pat = re.compile(r"done (\w+)-\w+ prompt=(\d+) .*?tokens=256 sha=(\w+) .*?rounds=(\d+) .*?decode=([\d.]+) .*?tok/s=([\d.]+)(.*)$")
def reps(arm):
    rows = [m for m in (pat.search(l) for l in (D / f"serve-r0-{arm}.log").read_text(encoding="utf-8", errors="replace").splitlines()) if m]
    i = max(k for k, m in enumerate(rows[:20]) if "seed=" in m[7]) + 1 + (4 if row == "chat" else 0)
    g = rows[i:i + 4]
    assert len({m[3] for m in g}) == 1
    return [float(m[6]) for m in g[1:]]
for name, arms in (("A", A), ("B", B)):
    v = [x for a in arms for x in reps(a)]
    print(f"{name} {row} t=0: {len(v)} reps, mean {st.mean(v):.3f}, min {min(v)}, max {max(v)}; run medians {[st.median(reps(a)) for a in arms]}")
