"""seedpick.py [--same-form] DIR OWN_M OWN_H [BEFORE]: the 0091 PR's receipts (<the local records>; <a local session>, 11-17). After a
chain stopped on split expert forms, names main's and the head's window-price records in DIR (a copy of the stopped
chain's window-costs) for the seed that aligns the counted arms on main's record (<the local ledger> 11-11 12:1x (b), the 11:5x
form). OWN_M and OWN_H hold each binary's timed look words, "A ms at 1 row, B at 4, C at 32, a draft D ms; experts W".
A record (kzt.py's layout) is a binary's when its 1-, 4- and 32-row prices and its draft price are those words to 0.005
ms and its form is W (classic when fact is 0, as cuda_lane_costs.zig's report prints). The head's must be one record;
main's one, or two with the same bytes (its key's and the reference key's, which a timed start saves after its own):
of two, the one whose name is not in BEFORE (the names the cache held before the chain; the binaries are new, the
reference may not be), else the earlier written. With --same-form (11-17, rf3c: TP=3 aligned on main's record though
both forms matched) the two looks may name one form; their prices must then tell the records apart. Prints "KM KH" on
stdout and every match on stderr; exits 1 otherwise."""
import os
import re
import struct
import sys

MAGIC = b"TFWC2\n"
SIZE = len(MAGIC) + 12 * 8
WORDS = re.compile(r"^([0-9.]+) ms at 1 row, ([0-9.]+) at 4, ([0-9.]+) at 32, a draft ([0-9.]+) ms; experts (factored|classic)$")


def words(path):
    s = open(path, encoding="utf-8").read().strip()
    m = WORDS.match(s)
    if not m:
        sys.exit("seedpick: %s holds no look words: %r" % (path, s[:200]))
    return [float(x) for x in m.groups()[:4]], m.group(5)


def tables(d):
    out = []
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if not n.endswith(".bin") or not os.path.isfile(p):
            continue
        b = open(p, "rb").read()
        if len(b) != SIZE or not b.startswith(MAGIC):
            continue
        ms = struct.unpack_from("<10d", b, len(MAGIC))
        (draft,) = struct.unpack_from("<d", b, len(MAGIC) + 80)
        (fact,) = struct.unpack_from("<Q", b, len(MAGIC) + 88)
        out.append((n, (ms[0], ms[3], ms[9], draft), "factored" if fact else "classic", b, os.stat(p).st_mtime_ns))
    return out


def main(argv):
    same = argv[:1] == ["--same-form"]
    argv = argv[1:] if same else argv
    if len(argv) not in (3, 4):
        sys.exit("usage: seedpick.py [--same-form] DIR OWN_M OWN_H [BEFORE]")
    before = set(open(argv[3], encoding="utf-8").read().split()) if len(argv) == 4 else set()
    (wm, fm), (wh, fh) = words(argv[1]), words(argv[2])
    if fm == fh and not same:
        sys.exit("seedpick: both looks name the %s form: not a split" % fm)
    t = tables(argv[0])
    near = lambda a, b: all(abs(x - y) <= 0.0051 for x, y in zip(a, b))
    cm = [r for r in t if near(r[1], wm) and r[2] == fm]
    ch = [r for r in t if near(r[1], wh) and r[2] == fh]
    for tag, c in (("main", cm), ("head", ch)):
        for r in c:
            print("%s: %s %s %s mtime_ns %d" % (tag, r[0], ",".join("%.3f" % x for x in r[1]), r[2], r[4]), file=sys.stderr)
    print("%d tables in %s" % (len(t), argv[0]), file=sys.stderr)
    if len(ch) != 1:
        sys.exit("seedpick: %d records hold the head's words, not 1" % len(ch))
    if len(cm) not in (1, 2) or len({r[3] for r in cm}) != 1:
        sys.exit("seedpick: %d records hold main's words (two must have one content)" % len(cm))
    new = [r for r in cm if r[0] not in before]
    km = new[0] if len(new) == 1 else min(cm, key=lambda r: (r[4], r[0]))
    print("main's of %d: %s (%s)" % (len(cm), km[0], "the one new to the cache" if len(new) == 1 else "the earlier written"), file=sys.stderr)
    print(km[0], ch[0][0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
