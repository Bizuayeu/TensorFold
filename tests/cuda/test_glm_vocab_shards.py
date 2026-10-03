"""The vocabulary gathers of GLM's CUDA engine over shards of unequal width (three ranks of 1,280 ids: 448/448/384,
the shape of GLM-5.3-Flash's 51,648/51,648/51,584), ranks as threads (``threadcomm``): sampled tokens and teacher-forced
prompt rows equal one rank's over the whole vocabulary, for every sampling path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from threadcomm import run_ranks  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

V, ROWS = 1280, 5
SPANS = [(0, 448), (448, 896), (896, 1280)]


def _logits() -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return (torch.randn((ROWS, V), generator=g) * 3).to(torch.bfloat16).cuda()      # bf16: ties among the values


def _weights(rank: int, comm, spans=SPANS):
    return SimpleNamespace(comm=comm, world=len(spans), vocab_offset=spans[rank][0], vocab_spans=spans)


@pytest.mark.parametrize("sampling", [None, Sampling(5, 1.0, 20, 0.95), Sampling(5, 1.0, 450, 0.98),
                                      Sampling(5, 0.8, 0, 0.9), Sampling(5, 1.0, 0, 0.9, 0.05)],
                         ids=["greedy", "top_k", "top_k past a shard", "nucleus", "min_p"])
def test_samples_over_unequal_shards_equal_one_rank(sampling):
    from tensorfold.families.glm5_next.cuda.decode import sample_rows

    full = _logits()
    positions = list(range(40, 40 + ROWS))
    want_p: list[float] = []
    want = sample_rows(_weights(0, None, [(0, V)]), full, positions, sampling, probs=want_p)

    def rank(r, comm):
        lo, hi = SPANS[r]
        probs: list[float] = []
        return sample_rows(_weights(r, comm), full[:, lo:hi].contiguous(), positions, sampling, probs=probs), probs

    for got, probs in run_ranks(rank, 3):
        assert got == want
        if sampling is not None:      # greedy: a draft's confidence among the gathered candidates, more on more ranks
            assert probs == pytest.approx(want_p, abs=1e-9)


@pytest.mark.parametrize("top", [0, 5, 400])
def test_prompt_rows_over_unequal_shards_equal_one_rank(top):
    from tensorfold.cuda.logprobs import prompt_rows
    from tensorfold.cuda.sampling import comm_gather, one_rank

    full = _logits()
    targets = [0, 447, 448, 1000, 1279]                     # the first and last id of shards, and between
    want = prompt_rows(full, targets, top, one_rank, [(0, V)])

    def rank(r, comm):
        lo, hi = SPANS[r]
        return prompt_rows(full[:, lo:hi].contiguous(), targets, top, comm_gather(comm), SPANS, r)

    for got in run_ranks(rank, 3):
        for (lp, rk, best), (want_lp, want_rk, want_best) in zip(got, want):
            assert lp == pytest.approx(want_lp, abs=1e-5) and rk == want_rk
            assert [i for i, _ in best] == [i for i, _ in want_best]
            assert [v for _, v in best] == pytest.approx([v for _, v in want_best], abs=1e-5)


def test_prompt_rows_of_equal_shards_keep_the_two_rank_layout():
    """Without spans, as before: equal shards, slot r from id r * shard."""

    from tensorfold.cuda.logprobs import prompt_rows
    from tensorfold.cuda.sampling import comm_gather

    full = _logits()
    targets = [3, 700, 1279, 640, 639]

    def rank(r, comm, spans=None):
        return prompt_rows(full[:, r * 640:(r + 1) * 640].contiguous(), targets, 5, comm_gather(comm), spans, r)

    assert run_ranks(rank, 2) == run_ranks(lambda r, c: rank(r, c, [(0, 640), (640, V)]), 2)
