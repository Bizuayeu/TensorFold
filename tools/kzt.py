"""In tp4-tf3 on 04, for kz-chain.sh (11-13, <a local session>; <another local session>'s ① receipt): kzt.py DIR -- every
NAME.bin under DIR, one line each, as "NAME fact=N ms=M1,M2,...,M10 draft=D" with N the table's last u64 as it is (on
①'s build bit i is the i-th of the 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32-row windows, 1 is that width factored; kept4k.py's
"experts classic|factored" words would misread it) and the ten window prices and the draft's in ms. A table is "TFWC2\\n"
and the Kept of zig/src/families/glm/cuda_lane_cost_measure.zig (ten f64, the draft's f64, the u64; 102 bytes, kept4k.py's
reading); a file of another size or magic prints "NAME size=S magic=HEX" instead, so a changed layout shows. An empty or
missing DIR prints nothing and exits 0 (an aside dir starts empty)."""
import os
import struct
import sys

MAGIC = b"TFWC2\n"
SIZE = len(MAGIC) + 12 * 8


def line(name, b):
    if len(b) != SIZE or not b.startswith(MAGIC):
        return "%s size=%d magic=%s" % (name, len(b), b[: len(MAGIC)].hex())
    ms = struct.unpack_from("<10d", b, len(MAGIC))
    (draft,) = struct.unpack_from("<d", b, len(MAGIC) + 80)
    (fact,) = struct.unpack_from("<Q", b, len(MAGIC) + 88)
    return "%s fact=%d ms=%s draft=%.3f" % (name, fact, ",".join("%.3f" % m for m in ms), draft)


def main(argv):
    if len(argv) != 1:
        raise SystemExit("usage: kzt.py DIR")
    d = argv[0]
    if not os.path.isdir(d):
        return 0
    for n in sorted(n for n in os.listdir(d) if n.endswith(".bin")):
        with open(os.path.join(d, n), "rb") as f:
            print(line(n, f.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
