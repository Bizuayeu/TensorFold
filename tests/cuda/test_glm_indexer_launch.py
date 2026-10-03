"""The indexer's launch shapes (rows a scoring program) never change a row's pool scores, chosen
pools or tokens: bf16 and TF_GLM_KV=fp8 pooled keys, decode windows and prompt chunks."""

import pytest
import torch
import triton

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

H, D = 32, 128


def _inputs(pos, R, fp8, pools=None):
    """Index queries and weights of rows pos .. pos + R - 1, and ``pools`` pooled keys (default: the visible ones)."""
    from tensorfold.families.glm5_next.cuda import kv8

    gen = torch.Generator(device="cuda").manual_seed(pos + R)
    qi = torch.randn((R, H * D), generator=gen, device="cuda").to(torch.bfloat16)
    wts = torch.randn((R, H), generator=gen, device="cuda").to(torch.bfloat16)
    keys = torch.randn((pools or (pos + R) // 4 + 2, D), generator=gen, device="cuda")
    pk = kv8.quantize_rows(keys) if fp8 else keys.to(torch.bfloat16)
    return qi, wts, pk, torch.tensor([pos], dtype=torch.int32, device="cuda")


def _scores(qi, wts, pk, pos_dev, R, rb, num_warps=4):
    """Every row's scores over all of pk's pools, rb rows a program."""
    from tensorfold.families.glm5_next.cuda import kv8, sparse

    pkc, fp8 = kv8.view(pk)
    NP = pk.shape[0] - 2
    out = torch.full((R, NP), float("nan"), dtype=torch.float32, device="cuda")
    sparse._scores[(triton.cdiv(R, rb), triton.cdiv(NP, 64))](qi, wts, wts.stride(0), pkc, out, pos_dev, R, NP,
                                                              D ** -0.5, 1.0 / 5.656854249492381, H=H, HP=H, D=D,
                                                              BP=64, RB=rb, FP8=fp8, num_warps=num_warps)
    return out


@cuda
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pos, R", [(2047, 64), (9000, 300), (130000, 2048)])
def test_rows_a_scoring_program_keep_every_rows_bits(fp8, pos, R):
    """SCORE_RB rows a program (prompt chunks) score each row as one row a program does (decode windows)."""
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev = _inputs(pos, R, fp8)
    one = _scores(qi, wts, pk, pos_dev, R, 1)
    assert torch.equal(_scores(qi, wts, pk, pos_dev, R, sparse.SCORE_RB), one)


@cuda
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pos, R", [(2047, 64), (9000, 300), (262144 - 2048, 2048)])
def test_select_tokens_lists_do_not_depend_on_rows_a_program(monkeypatch, fp8, pos, R):
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev = _inputs(pos, R, fp8)
    got = sparse.select_tokens(qi, wts, pk, pos, R, pk.shape[0] - 2, pos_dev)
    monkeypatch.setattr(sparse, "SCORE_RB", 1)
    want = sparse.select_tokens(qi, wts, pk, pos, R, pk.shape[0] - 2, pos_dev)
    assert torch.equal(got[1], want[1]) and bool((got[1] > 0).any())
    assert torch.equal(got[0], want[0])

