"""TF_GLM_PREFILL_REDUCE=scatter, the exact reduce-scatter of a prompt chunk's partials, on ranks that are threads of
one GPU (``threadcomm``): every rank's hc_post and MTP residual add over the scattered sums (in slot 0 of gather's
layout, -0.0 in the others) leave the bits they leave over the gathered partials, for any row count and two or three
ranks, while a rank receives what the module docstring counts (half of the all-gather's bytes at three ranks, three
quarters at two)."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from threadcomm import run_ranks  # noqa: E402

from tensorfold.families.glm5_next.cuda import glue, reduce  # noqa: E402

D = 512


def test_the_setting_is_gather_or_scatter():
    assert reduce.settings({}) == "gather"
    assert reduce.settings({"TF_GLM_PREFILL_REDUCE": ""}) == "gather"
    assert reduce.settings({"TF_GLM_PREFILL_REDUCE": " scatter "}) == "scatter"
    with pytest.raises(ValueError, match="TF_GLM_PREFILL_REDUCE: gather or scatter, not 'ring'"):
        reduce.settings({"TF_GLM_PREFILL_REDUCE": "ring"})


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("rows", [2, 3, 4, 7, 128, 500])
def test_row_shares_cover_the_rows_in_rank_order(world, rows):
    cut = reduce.shares(rows, world)
    assert cut[0][0] == 0 and cut[-1][1] == rows and all(a[1] == b[0] for a, b in pairwise(cut))
    sizes = [hi - lo for lo, hi in cut]
    assert max(sizes) - min(sizes) <= 1 and sizes == sorted(sizes, reverse=True)


def _partial(rank: int, rows: int) -> torch.Tensor:
    """fp32 partials whose sum depends on its order: magnitudes 2^-12 .. 2^12, signs mixed, so (p0 + p1) + p2 and
    p0 + (p1 + p2) differ in many places."""

    g = torch.Generator(device="cuda").manual_seed(100 + rank)
    mant = torch.rand((rows, D), generator=g, device="cuda") * 2 - 1
    exp = torch.randint(-12, 13, (rows, D), generator=g, device="cuda").float()
    return (mant * torch.exp2(exp)).contiguous()


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 7, 128, 500, 2048])
def test_scattered_sums_leave_the_gathered_bits(world, rows):
    """For each rank: scatter's [world, R, D] holds bf16(gather's partials summed rank 0 first) in slot 0 and -0.0 in
    the others, and hc_post and residual_add over it give gather's bytes; the received bytes are the module
    docstring's (fewer rows than ranks gather)."""

    def fn(rank, comm):
        part = _partial(rank, rows)
        g = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn((rows, 4 * D), generator=g, device="cuda").to(torch.bfloat16)
        post = torch.rand((rows, 4), generator=g, device="cuda")
        comb = torch.rand((rows, 16), generator=g, device="cuda")
        res = torch.randn((rows, D), generator=g, device="cuda").to(torch.bfloat16)
        out = {}
        for mode in reduce.MODES:
            room = torch.full((world * rows * D,), float("nan"), device="cuda")
            comm.received = 0
            got = reduce.sums(comm, part, room, mode == "scatter")
            h = torch.empty_like(x)
            glue.hc_post(x, h, got, post, comb)
            r = torch.empty_like(res)
            glue.residual_add(res, r, got)
            out[mode] = (got.clone(), comm.received, h, r)
        return out

    for rank, out in enumerate(run_ranks(fn, world)):
        (gg, gb, gh, gr), (sg, sb, sh, sr) = out["gather"], out["scatter"]
        assert gg.shape == sg.shape == (world, rows, D) and sg.dtype == torch.float32
        assert torch.equal(gh, sh) and torch.equal(gr, sr), rank
        assert gb == (world - 1) * rows * D * 4
        if rows < world:
            assert torch.equal(gg, sg) and sb == gb
            continue
        want = gg[0]
        for k in range(1, world):
            want = want + gg[k]
        assert torch.equal(sg[0], want.to(torch.bfloat16).float()), rank
        assert bool((sg[1:].view(torch.int32) == -2 ** 31).all()), rank          # -0.0: x + (-0.0) is x, bit for bit
        mine = reduce.shares(rows, world)[rank]
        mine = mine[1] - mine[0]
        assert sb == (world - 1) * mine * D * 4 + (rows - mine) * D * 2, rank


def test_the_partials_test_the_order():
    """The partials above make the summation order visible: another order changes the bf16 branch."""

    p = [_partial(r, 128) for r in range(3)]
    ours = ((p[0] + p[1]) + p[2]).to(torch.bfloat16)
    other = (p[0] + (p[1] + p[2])).to(torch.bfloat16)
    assert not torch.equal(ours, other)
