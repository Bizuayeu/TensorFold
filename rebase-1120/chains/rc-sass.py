"""rc-sass.py OUT: compare OUT/mb0b17/NAME.sass with OUT/h50c66/NAME.sass function by function (cuobjdump -sass text).
# (11-20: from <the local records>/ with the names mb0b17 = main b0b17e8962, h50c66 = #3689 rebased onto
#  #3776 50c6634539, and rc-sass; <a local session>, <the local records>)
A function's text is its lines after 'Function : NAME' up to the next header, whitespace-collapsed. Prints per fatbin
the functions in each, the same and the different ones, and for each different function its first differing line."""
import hashlib, os, re, sys
O = sys.argv[1]
HDR = re.compile(r"^\s*Function : (\S+)\s*$")
def funcs(path):
    out, name, buf = {}, None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = HDR.match(line)
            if m:
                if name is not None:
                    out.setdefault(name, []).append(buf)
                name, buf = m.group(1), []
            elif name is not None:
                s = " ".join(line.split())
                if s:
                    buf.append(s)
    if name is not None:
        out.setdefault(name, []).append(buf)
    return out
names = sorted(f[:-5] for f in os.listdir(os.path.join(O, "mb0b17")) if f.endswith(".sass"))
tot = {"same": 0, "diff": 0, "onlyA": 0, "onlyB": 0}
diffs = []
for n in names:
    a = funcs(os.path.join(O, "mb0b17", n + ".sass"))
    b = funcs(os.path.join(O, "h50c66", n + ".sass"))
    same = [k for k in a if k in b and a[k] == b[k]]
    diff = [k for k in a if k in b and a[k] != b[k]]
    oa = [k for k in a if k not in b]
    ob = [k for k in b if k not in a]
    tot["same"] += len(same); tot["diff"] += len(diff); tot["onlyA"] += len(oa); tot["onlyB"] += len(ob)
    print(f"{n}: functions A {len(a)} B {len(b)}; same {len(same)}, different {len(diff)}, only A {len(oa)}, only B {len(ob)}")
    for k in diff:
        la = [l for blk in a[k] for l in blk]; lb = [l for blk in b[k] for l in blk]
        i = next((j for j in range(min(len(la), len(lb))) if la[j] != lb[j]), min(len(la), len(lb)))
        diffs.append(f"  diff {n} {k[:90]}: lines A {len(la)} B {len(lb)}; first at {i}: A '{la[i][:110] if i < len(la) else ''}' B '{lb[i][:110] if i < len(lb) else ''}'")
    for k in oa[:5]:
        diffs.append(f"  only A {n} {k[:100]}")
    for k in ob[:5]:
        diffs.append(f"  only B {n} {k[:100]}")
print("\n".join(diffs[:80]))
print(f"total: {len(names)} fatbins; functions same {tot['same']}, different {tot['diff']}, only in main {tot['onlyA']}, only in the head {tot['onlyB']}")
