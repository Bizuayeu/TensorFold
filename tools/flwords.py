"""flwords.py TABLES KEY WORDS (11-17, the 0091 PR's floor gates rf2c and rf3d, <a local session>): whether kzt.py's
line for KEY in TABLES holds WORDS, a look's "A ms at 1 row, B at 4, C at 32, a draft D ms; experts W". The 1-, 4- and
32-row prices (kzt.py's ms fields 0, 3 and 9: widths 1, 2, 3, 4, 6, 8, 12, 16, 24, 32) and the draft price agree to
0.0051 ms (seedpick.py's: the look rounds to 2 places, kzt.py to 3), and W is "classic" when fact is 0, else
"factored". Prints "ok KEY: LINE" and ends 0, or prints why and ends 1."""
import re
import sys

tables, key, words = sys.argv[1:4]
m = re.fullmatch(r"([0-9.]+) ms at 1 row, ([0-9.]+) at 4, ([0-9.]+) at 32, a draft ([0-9.]+) ms; experts (\w+)", words.strip())
if not m:
    print(f"not look words: {words!r}")
    sys.exit(1)
want = [float(x) for x in m.groups()[:4]]
with open(tables, encoding="utf-8") as f:
    lines = [line for line in f if line.split(" ", 1)[0] == key]
if len(lines) != 1:
    print(f"{len(lines)} lines for {key} in {tables}")
    sys.exit(1)
fields = dict(x.split("=", 1) for x in lines[0].split()[1:] if "=" in x)
try:
    ms = [float(x) for x in fields["ms"].split(",")]
    got = [ms[0], ms[3], ms[9], float(fields["draft"])]
    form = "classic" if int(fields["fact"]) == 0 else "factored"
except (KeyError, IndexError, ValueError) as e:
    print(f"{key}: not a table line ({e!r}): {lines[0].strip()[:200]}")
    sys.exit(1)
bad = [f"{n} {g} against {w}" for n, g, w in zip(("1 row", "4 rows", "32 rows", "draft"), got, want) if abs(g - w) > 0.0051]
if form != m.group(5):
    bad.append(f"experts {form} against {m.group(5)}")
if bad:
    print(f"{key}: " + "; ".join(bad))
    sys.exit(1)
print(f"ok {key}: {lines[0].strip()[:200]}")
