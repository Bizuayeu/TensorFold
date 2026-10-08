"""The indexer's launch shapes (pool blocks a scoring program, pool columns scored) and the selection's list never change a
row's pool scores, chosen pools or tokens: bf16 and TF_GLM_KV=fp8 pooled keys, decode windows and prompt chunks."""

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


def _scores(qi, wts, pk, pos_dev, R, loop, num_warps=4):
    """Every row's scores over all of pk's pools, loop pool blocks a program."""
    from tensorfold.families.glm5_next.cuda import kv8, sparse

    pkc, fp8 = kv8.view(pk)
    NP = pk.shape[0] - 2
    NB = triton.cdiv(NP, 64)
    out = torch.full((R, NP), float("nan"), dtype=torch.float32, device="cuda")
    sparse._scores[(R, triton.cdiv(NB, loop))](qi, wts, wts.stride(0), pkc, out, pos_dev, NP, NB, D ** -0.5,
                                               1.0 / 5.656854249492381, H=H, HP=H, D=D, BP=64, L=loop, FP8=fp8,
                                               num_warps=num_warps)
    return out


@cuda
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pos, R", [(2047, 64), (9000, 300), (130000, 2048), (200000, 8)])
def test_blocks_a_scoring_program_keep_every_rows_bits(fp8, pos, R):
    """SCORE_LOOP pool blocks a program (prompt chunks) score each block as one block a program does (decode
    windows), every column written."""
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev = _inputs(pos, R, fp8)
    one = _scores(qi, wts, pk, pos_dev, R, 1)
    assert not bool(one.isnan().any())
    assert torch.equal(_scores(qi, wts, pk, pos_dev, R, sparse.SCORE_LOOP), one)


@cuda
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pos, R", [(2047, 64), (9000, 300), (262144 - 2048, 2048)])
def test_select_tokens_lists_do_not_depend_on_blocks_a_program(monkeypatch, fp8, pos, R):
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev = _inputs(pos, R, fp8)
    got = sparse.select_tokens(qi, wts, pk, pos, R, pk.shape[0] - 2, pos_dev)
    monkeypatch.setattr(sparse, "SCORE_LOOP", 1)
    want = sparse.select_tokens(qi, wts, pk, pos, R, pk.shape[0] - 2, pos_dev)
    assert torch.equal(got[1], want[1]) and bool((got[1] > 0).any())
    assert torch.equal(got[0], want[0])


@pytest.mark.parametrize("pos, rows, np_max, want", [
    (None, 512, 65536, 65536),          # a captured graph's rows: positions on the device, every bucket column
    (2047, 64, 1024, 527),              # the last row's complete pools
    (1000, 64, 1024, 512),              # selection reads at least TOPK_POOLS columns
    (100, 4, 300, 300),                 # never past the bucket
    (260096 + 1536, 512, 131072, 65536),
])
def test_score_columns_are_the_columns_selection_reads(pos, rows, np_max, want):
    from tensorfold.families.glm5_next.cuda import sparse

    assert sparse.score_columns(pos, rows, np_max) == want


@cuda
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pos, R", [(2047, 64), (9000, 300), (262144 - 2048, 2048), (262144 - 1024, 1024)])
def test_selection_reads_only_scored_columns(monkeypatch, fp8, pos, R):
    """Each block's pool columns past score_columns' hold NaN (the highest key, were they read): every row's tokens
    and count are the ones of scoring the bucket's every column."""
    from tensorfold.families.glm5_next.cuda import sparse

    qi, wts, pk, pos_dev = _inputs(pos, R, fp8, pools=1 << 18)
    np_max = pk.shape[0] - 2
    assert sparse.pool_bucket(pos, R, np_max) > (pos + R) // 4 + 1           # columns past the visible pools
    with monkeypatch.context() as m:
        m.setattr(sparse, "score_columns", lambda pos, rows, np_max: np_max)
        want = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    empty = torch.empty

    def nan_empty(*a, **k):
        t = empty(*a, **k)
        return t.fill_(float("nan")) if t.dtype == torch.float32 else t
    with monkeypatch.context() as m:
        m.setattr(torch, "empty", nan_empty)
        got = sparse.select_tokens(qi, wts, pk, pos, R, np_max, pos_dev)
    assert torch.equal(got[1], want[1]) and bool((got[1] > 0).any())
    assert torch.equal(got[0], want[0])


def _hard_rows(np_):
    """test_glm_kernels' radix rows: distinct, heavy ties, -inf past the visible pools, -0 beside +0, fewer than 512
    finite, all equal, tiny rounded scores."""
    g = torch.Generator(device="cpu").manual_seed(np_)
    rows = [torch.randn(np_, generator=g), torch.randint(-3, 4, (np_,), generator=g).float()]
    r = torch.randn(np_, generator=g)
    r[np_ // 3:] = float("-inf")
    rows.append(r)
    r = torch.zeros(np_)
    r[::2] = -0.0
    r[5::7] = 1.0
    rows.append(r)
    r = torch.full((np_,), float("-inf"))
    r[:100] = torch.randn(100, generator=g)
    rows += [r, torch.full((np_,), 2.5), (torch.randn(np_, generator=g) * 1e-30).to(torch.bfloat16).float()]
    return torch.stack(rows).to("cuda").contiguous()


@cuda
@pytest.mark.parametrize("cap", [256, 1024, 8192, 65536])
@pytest.mark.parametrize("np_", [700, 5003, 65536])
@pytest.mark.parametrize("visible", [False, True])
def test_selection_list_keeps_the_sorted_top_k(monkeypatch, cap, np_, visible):
    """Whatever the list holds (SELECT_LIST 256: never 512 pools, the row is read to the end; at least the row's
    pools: no list) the pools are a stable descending sort's first 512, ties to the lower pool; past each row's
    visible pools (-inf) too."""
    from tensorfold.families.glm5_next.cuda import sparse

    scores = _hard_rows(np_)
    position = None
    if visible:
        pos = 2 * np_
        position = torch.tensor([pos], dtype=torch.int32, device="cuda")
        seen = (pos + torch.arange(scores.shape[0], device="cuda") + 1) // 4
        scores.masked_fill_(torch.arange(np_, device="cuda")[None, :] >= seen[:, None], float("-inf"))
    monkeypatch.setattr(sparse, "SELECT_LIST", cap)
    assert torch.equal(sparse.top_pools(scores, 512, position), sparse._top_pools(scores, 512))
