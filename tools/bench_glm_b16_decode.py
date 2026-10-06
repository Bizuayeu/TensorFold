"""Time GLM-5.3-Flash's BF16 decode matmuls (``qmm._bmm`` at 1..8 rows, the 16-row bucket) for every rank's weight
shape at TP=2 and TP=3 across a grid of tiles (BLOCK_N, warps, stages), each tile's outputs checked bit for bit against
today's at every decode row count, fp32 and bf16 out. The K slices and step never follow the tile
(``qmm.b16_split_k``), so a tile only moves time. One GPU; prints a JSON line per shape, then the picks.

    python tools/bench_glm_b16_decode.py > sweep.jsonl

Method: each call reads another copy of the weight, copies spanning twice the L2 cache, as a decode step reads each
weight once; a burst (one call per copy) is a CUDA graph's replay, as decode replays its steps; tiles' bursts
alternate with today's tile's, bursts' medians compared.
A second stream of today's tile (A/A) gives the noise floor: a tile is picked only where it beats today's by more
than the largest A/A gap of the whole sweep at every row count timed. A screen of every tile picks the fastest few
for a longer final run."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "cuda"))   # the shapes' derivation

from glm_b16_shapes import DECODE_ROWS, WORLDS, decode_shapes, matmuls

from tensorfold.families.glm5_next.cuda import qmm

TODAY = (qmm.B16_BN, *qmm.B16_CONFIG[16])
GRID = [t for t in itertools.product((32, 64, 128, 256), (2, 4, 8), (2, 3, 4)) if t != TODAY]


@contextmanager
def tile(t: tuple[int, int, int]):
    keep = qmm.b16_tile
    qmm.b16_tile = lambda n, k, bm: t
    try:
        yield
    finally:
        qmm.b16_tile = keep


def inputs(n: int, k: int, gen: torch.Generator):
    x = torch.randn((max(DECODE_ROWS), k), generator=gen)
    x[torch.rand(x.shape, generator=gen) < 0.3] = 0.0
    x[torch.rand(x.shape, generator=gen) < 0.05] = -0.0
    x[2] = -0.0
    w = torch.randn((n, k), generator=gen) * 0.02
    w[torch.rand((n, k), generator=gen) < 0.2] = -0.0
    return x.to(torch.bfloat16).cuda(), qmm.make_b16(w.cuda())


def outputs(x, q, t) -> list[torch.Tensor]:
    with tile(t):
        return [qmm.matmul(x[:m], q, f32=f32).view(torch.int32 if f32 else torch.int16)
                for f32 in (True, False) for m in DECODE_ROWS]


def same_bits(x, q, t, want) -> bool | str:
    try:
        got = outputs(x, q, t)
    except Exception as e:                         # noqa: BLE001 (a tile past the shared memory)
        return f"{type(e).__name__}: {str(e)[:80]}"
    return all(torch.equal(a, b) for a, b in zip(want, got))


def graph(x, copies, out, part, t) -> torch.cuda.CUDAGraph:
    """One call per copy with tile ``t``, captured as decode captures its steps (no launch overhead in the times)."""

    with tile(t):
        for q in copies:                           # compile and warm outside the capture
            qmm.matmul(x, q, out=out, part=part)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for q in copies:
                qmm.matmul(x, q, out=out, part=part)
    return g


def burst(g: torch.cuda.CUDAGraph, calls: int) -> float:
    """ms per call of one replay."""

    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / calls


def race(x, copies, tiles, rounds) -> dict:
    """Median ms per call of each tile, each burst right after one of today's tile; "aa" is today's second stream."""

    m = x.shape[0]
    out = torch.empty((m, copies[0].n), dtype=torch.bfloat16, device="cuda")
    part = torch.empty((8 * m * copies[0].n,), dtype=torch.float32, device="cuda")
    graphs = {t: graph(x, copies, out, part, t) for t in (TODAY, *tiles)}
    graphs["aa"] = graphs[TODAY]
    runs: dict = {t: [] for t in ("today", "aa", *tiles)}
    for _ in range(rounds):
        for t in ("aa", *tiles):
            runs["today"].append(burst(graphs[TODAY], len(copies)))
            runs[t].append(burst(graphs[t], len(copies)))
    del graphs
    return {t: statistics.median(v) for t, v in runs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,4,8", help="decode row counts timed")
    ap.add_argument("--screen", type=int, default=10, help="rounds of the screen (every tile with the same bits)")
    ap.add_argument("--final", type=int, default=100, help="rounds of the final run of the fastest tiles")
    ap.add_argument("--top", type=int, default=3, help="tiles of the screen kept for the final run")
    ap.add_argument("--shapes", default="", help="only these NxK (comma separated)")
    a = ap.parse_args()
    rows = [int(r) for r in a.rows.split(",")]
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    shapes = decode_shapes()
    if a.shapes:
        shapes = [s for s in shapes if f"{s[0]}x{s[1]}" in a.shapes.split(",")]
    print(json.dumps({"device": torch.cuda.get_device_name(0), "l2_bytes": l2, "today": TODAY, "grid": len(GRID),
                      "rows": rows, "screen": a.screen, "final": a.final, "top": a.top}), flush=True)
    results = {}
    for n, k in shapes:
        gen = torch.Generator().manual_seed(n * 7 + k)
        x, q = inputs(n, k, gen)
        want = outputs(x, q, TODAY)
        bits = {t: same_bits(x, q, t, want) for t in GRID}
        ok = [t for t in GRID if bits[t] is True]
        copies = [q] + [qmm.B16(q.weight.clone(), n, k) for _ in range(max(1, -(-2 * l2 // (n * k * 2))))]
        screen = {}
        for m in rows:
            screen[m] = race(x[:m], copies, ok, a.screen)
        mean = {t: statistics.mean(screen[m][t] / screen[m]["today"] for m in rows) for t in ok}
        best = sorted(ok, key=mean.get)[:a.top]
        final = {m: race(x[:m], copies, best, a.final) for m in rows}
        results[(n, k)] = final
        print(json.dumps({"n": n, "k": k, "sk": qmm.b16_split_k(n, k), "copies": len(copies),
                          "bits_differ": [t for t in GRID if bits[t] is False],
                          "failed": {str(t): bits[t] for t in GRID if isinstance(bits[t], str)},
                          "screen_best": [(t, round(mean[t], 4)) for t in best],
                          "final_ms": {m: {str(t): round(v, 5) for t, v in f.items()} for m, f in final.items()}}),
              flush=True)
        del copies, q
        torch.cuda.empty_cache()
    floor = max(abs(f["aa"] / f["today"] - 1) for fin in results.values() for f in fin.values())
    picks = {}
    for (n, k), fin in results.items():
        gains = {t: min(1 - f[t] / f["today"] for f in fin.values()) for t in next(iter(fin.values()))
                 if t not in ("today", "aa")}
        good = {t: g for t, g in gains.items() if g > floor}
        if good:
            t = max(good, key=good.get)
            picks[f"{n}x{k}"] = {"tile": t, "least_gain": round(good[t], 4),
                                 "ms": {m: (round(f["today"], 5), round(f[t], 5)) for m, f in fin.items()}}
    print(json.dumps({"aa_floor": round(floor, 4), "picks": picks}), flush=True)
    for w in WORLDS:
        for r in range(w):
            calls = matmuls(w, r)
            for m in rows:
                before = sum(results[(n, k)][m]["today"] for _, n, k in calls if (n, k) in results)
                after = sum(results[(n, k)][m][picks[f"{n}x{k}"]["tile"]] if f"{n}x{k}" in picks
                            else results[(n, k)][m]["today"] for _, n, k in calls if (n, k) in results)
                print(json.dumps({"step": f"TP={w} rank {r}", "rows": m, "calls": len(calls),
                                  "today_ms": round(before, 3), "picked_ms": round(after, 3)}), flush=True)


if __name__ == "__main__":
    main()
