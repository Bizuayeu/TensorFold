"""TF_GLM_PREFILL_REDUCE=scatter, the exact reduce-scatter of a prompt chunk's partials, on ranks that are threads of
one GPU (``threadcomm``): every rank's hc_post and MTP residual add over the scattered sums (in slot 0 of gather's
layout, -0.0 in the others) leave the bits they leave over the gathered partials, for any row count and two or three
ranks, while a rank receives what the module docstring counts (half of the all-gather's bytes at three ranks, three
quarters at two). TF_GLM_PREFILL_REDUCE=split: hc_post over a rank's own rows of every rank's partials, then those
rows handed to the others, leaves the gathered hc_post's bytes on every row."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from threadcomm import run_ranks  # noqa: E402

from tensorfold.families.glm5_next.cuda import glue, reduce  # noqa: E402

D = 512


def test_the_setting_is_gather_scatter_or_split():
    class Peers:                                  # a communicator that can send and receive between ranks
        def send_recv(self, sends, recvs):
            pass

    # unset: split where the communicator has send_recv (NCCL, threads), gather where it has not
    assert reduce.settings({}) == "gather"
    assert reduce.settings({}, object()) == "gather"
    assert reduce.settings({}, Peers()) == "split"
    assert reduce.settings({"TF_GLM_PREFILL_REDUCE": ""}, Peers()) == "split"
    assert reduce.settings({"TF_GLM_PREFILL_REDUCE": "gather"}, Peers()) == "gather"
    assert reduce.settings({"TF_GLM_PREFILL_REDUCE": " scatter "}) == "scatter"
    assert reduce.settings({"TF_GLM_PREFILL_REDUCE": "split"}) == "split"
    with pytest.raises(ValueError, match="TF_GLM_PREFILL_REDUCE: gather, scatter or split, not 'ring'"):
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
        for mode in ("gather", "scatter"):
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


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("rows", [3, 4, 7, 128, 500, 2048])
def test_split_glue_leaves_the_gathered_bits(world, rows):
    """split: ``owned`` gives each rank every rank's partials of its own rows ([world, n, D], rank order); hc_post over
    them writes the gathered hc_post's bytes on those rows, and ``share`` fills every rank's other rows with their
    owners' bytes. A rank owning n rows receives (world - 1) x n x D x 4 bytes, then (R - n) x 4D x 2 (the streams;
    the engine shares the D-wide normed rows)."""

    def fn(rank, comm):
        part = _partial(rank, rows)
        g = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn((rows, 4 * D), generator=g, device="cuda").to(torch.bfloat16)
        post = torch.rand((rows, 4), generator=g, device="cuda")
        comb = torch.rand((rows, 16), generator=g, device="cuda")
        want = torch.empty_like(x)
        room = torch.full((world * rows * D,), float("nan"), device="cuda")
        glue.hc_post(x, want, reduce.sums(comm, part, room, False), post, comb)
        lo, hi = reduce.shares(rows, world)[rank]
        room = torch.full((world * rows * D,), float("nan"), device="cuda")
        comm.received = 0
        got = reduce.owned(comm, part, room)
        owned_bytes = comm.received
        h = torch.full_like(x, float("nan"))
        glue.hc_post(x[lo:hi], h[lo:hi], got, post[lo:hi], comb[lo:hi])
        comm.received = 0
        reduce.share(comm, [h])
        return want, h, got.clone(), (lo, hi), owned_bytes, comm.received

    for rank, (want, h, got, (lo, hi), owned_bytes, shared_bytes) in enumerate(run_ranks(fn, world)):
        n = hi - lo
        assert got.shape == (world, n, D), rank
        for k in range(world):          # rank k's partial of these rows, in slot k
            assert torch.equal(got[k], _partial(k, rows)[lo:hi]), (rank, k)
        assert torch.equal(h, want), rank
        assert owned_bytes == (world - 1) * n * D * 4, rank
        assert shared_bytes == (rows - n) * 4 * D * 2, rank
