"""BF16 decode matmuls (``qmm._bmm``, 1..8 rows) give the same bits under any tile: column width, warps and stages
are speed only. The K slices and K step (the sums' order) must not follow the tile, so a per-shape tile table can
never change a reply."""

from __future__ import annotations

import pytest
import torch
from glm_b16_shapes import DECODE_ROWS, decode_shapes, matmuls

from tensorfold.families.glm5_next.cuda import qmm

SHAPES = decode_shapes()
# (BLOCK_N, warps, stages): every column width a table could take, besides today's (64, 4, 3)
CANDIDATES = ((32, 2, 4), (128, 4, 3), (256, 8, 2), (64, 8, 4))


def test_shapes_hold_the_two_rank_projections():
    """The derivation reproduces the per-rank shapes the prompt tile table was tuned on (TP=2)."""

    two = {(n, k) for r in range(2) for _, n, k in matmuls(2, r)}
    assert {(12576, 4096), (4096, 128), (4096, 4096), (2048, 4096), (4096, 1024), (8192, 1536), (4096, 8192),
            (160, 4096), (4096, 1536), (77440, 4096)} == two
    assert all(k % qmm.B16_BK == 0 for _, k in SHAPES)


def _inputs(n: int, k: int):
    gen = torch.Generator().manual_seed(n * 7 + k)
    x = torch.randn((max(DECODE_ROWS), k), generator=gen)
    x[torch.rand(x.shape, generator=gen) < 0.3] = 0.0
    x[torch.rand(x.shape, generator=gen) < 0.05] = -0.0
    x[2] = -0.0                                                     # a row that sums signed zeros only
    w = torch.randn((n, k), generator=gen) * 0.02
    w[torch.rand((n, k), generator=gen) < 0.2] = -0.0
    return x.to(torch.bfloat16).cuda(), qmm.make_b16(w.cuda())


def _decode(x, q, f32: bool, tile=None, table: bool = False) -> list[torch.Tensor]:
    """Each decode row count's output: today's tile, ``tile`` (BLOCK_N, warps, stages) for the 16-row bucket, or with
    ``table`` B16_DECODE_SHAPES' (TF_GLM_B16_DECODE_TABLE)."""

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(qmm, "B16_DECODE_TABLE", table)
        if tile is not None:
            mp.setattr(qmm, "B16_BN", tile[0])
            mp.setattr(qmm, "B16_CONFIG", {**qmm.B16_CONFIG, 16: tile[1:]})
        return [qmm.matmul(x[:m], q, f32=f32).view(torch.int32 if f32 else torch.int16) for m in DECODE_ROWS]


@pytest.mark.parametrize("n,k", SHAPES, ids=[f"{n}x{k}" for n, k in SHAPES])
def test_split_k_does_not_follow_the_tile(n, k):
    today = qmm.b16_split_k(n, k)
    for tile in CANDIDATES:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(qmm, "B16_BN", tile[0])
            assert qmm.b16_split_k(n, k) == today, (n, k, tile)


@pytest.mark.parametrize("n,k", SHAPES, ids=[f"{n}x{k}" for n, k in SHAPES])
def test_decode_tiles_keep_the_bits(n, k):
    x, q = _inputs(n, k)
    for f32 in (True, False):
        want = _decode(x, q, f32)
        for tile in CANDIDATES:
            got = _decode(x, q, f32, tile)
            for m, a, b in zip(DECODE_ROWS, want, got):
                assert torch.equal(a, b), (n, k, qmm.b16_split_k(n, k), tile, m, f32)
        for m, a, b in zip(DECODE_ROWS, want, _decode(x, q, f32, table=True)):
            assert torch.equal(a, b), (n, k, qmm.B16_DECODE_SHAPES.get(f"{n}x{k}"), m, f32)


def test_decode_table_covers_decode_shapes_only():
    """Every entry is a decode shape at the 16-row bucket; with the switch off every shape takes today's tile."""

    assert {tuple(int(v) for v in s.split("x")) for s in qmm.B16_DECODE_SHAPES} <= set(SHAPES)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(qmm, "B16_DECODE_TABLE", False)
        assert {qmm.b16_tile(n, k, 16) for n, k in SHAPES} == {(64, 4, 3)}
    for n, k in SHAPES:
        assert qmm.b16_tile(n, k, 32) == (qmm.B16_BN, *qmm.B16_CONFIG[32])
