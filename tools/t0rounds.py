"""bench_openai's t=0 code and chat requests in rank 0's done lines: per run, the 3 reps after the warm-up (rounds, accepted,
decode s, ms a round = decode / rounds, tokens a round = 256 / rounds). The t=0 groups are the first two runs of four
seedless 256-token requests with one sha each, after the seeded (t=1) ones. Usage: t0rounds.py LOGDIR ARM..."""
import re, statistics as st, sys, pathlib
D = pathlib.Path(sys.argv[1])
pat = re.compile(r"done (\w+)-\w+ prompt=(\d+) .*?tokens=256 sha=(\w+) .*?rounds=(\d+) accepted=(\d+)/(\d+) .*?decode=([\d.]+) .*?tok/s=([\d.]+)(.*)$")
for arm in sys.argv[2:]:
    lines = (D / f"serve-r0-{arm}.log").read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for ln in lines:
        m = pat.search(ln)
        if m:
            rows.append(dict(kind=m[1], prompt=int(m[2]), sha=m[3], rounds=int(m[4]), acc=int(m[5]), dr=int(m[6]), dec=float(m[7]), tps=float(m[8]), seed="seed=" in m[9]))
    i = max(k for k, r in enumerate(rows[:20]) if r["seed"]) + 1   # after the seeded t=1 requests
    out = []
    for name in ("code", "chat"):
        g = rows[i:i + 4]
        assert len({r["sha"] for r in g}) == 1 and not any(r["seed"] for r in g), (arm, name, g)
        reps = g[1:]
        ms = [1000 * r["dec"] / r["rounds"] for r in reps]
        tpr = [256 / r["rounds"] for r in reps]
        out.append(f"{name}: tps {[r['tps'] for r in reps]} rounds {[r['rounds'] for r in reps]} acc {[r['acc'] for r in reps]} "
                   f"ms/round {[round(x, 2) for x in ms]} mean {st.mean(ms):.2f} tok/round {st.mean(tpr):.3f}")
        i += 4
    print(f"{arm:7s} " + " | ".join(out))
