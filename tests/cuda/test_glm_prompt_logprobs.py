"""GLM-5.3-Flash's teacher-forced prompt rows (``prompt_logprobs``) on the tiny synthetic checkpoints of
``test_glm_engine``, MLX 4-bit and NVFP4: they match a plain FP32 head over the prompt's final-normed rows, do not
depend on the prompt chunking or a repeat, leave the reply and the caches as a request without them does, and read as
vLLM's completions field.

One GPU plays rank 0 of two (``_TwoCopies``): both gather slots hold this rank's shard, so the vocabulary the rows rank
is that shard twice, ids r * shard onward in slot r; the reference builds the same one."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.probabilities import PromptProbabilities  # noqa: E402

from test_glm_engine import _TwoCopies, _checkpoint, _forget, _generate, _state  # noqa: E402

PROMPT = [int(t) for t in np.random.default_rng(7).integers(0, 1000, size=99)]
TOP = 5


@pytest.fixture(scope="module", params=["mlx", "nvfp4"])
def engine(request, tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp(f"glm_rows_{request.param}")
    _checkpoint(path, nvfp4=request.param == "nvfp4")
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies())


def _rows(engine, prompt=PROMPT, *, sampling=None, tokens=1, top=TOP):
    _forget(engine)
    rows = PromptProbabilities(top, prompt)
    engine.request.policy, engine.request.stop_eos = None, False
    out: list[int] = []
    stats = engine.generate(list(prompt), tokens, sampling, out.extend, prompt_logprobs=rows)
    return rows.emitted(), out, stats


def _reference(engine, prompt):
    """FP32 log-softmax of every row but the last: the prompt in one chunk, the head dequantized, the shard twice."""

    from tensorfold.families.glm5_next.cuda.decode import Engine
    from tensorfold.families.glm5_next.cuda.forward import chunks_for, compute, stage
    from tensorfold.families.glm5_next.cuda.qmm import B16, dequantize_q4

    w = engine.w
    e = Engine(w, capacity=2560, max_rows=8, prefill_rows=len(prompt))
    e.reset()
    with torch.no_grad():
        compute(w, e.st, e.pbuf, stage(w, e.st, e.pbuf, prompt), nch=chunks_for(e.st, len(prompt)), host_pos=0)
        hidden = e.pbuf.fnormed[:len(prompt) - 1].float()
        head = w.head.weight.float() if isinstance(w.head, B16) else dequantize_q4(w.head)
        logits = hidden @ head.T
    e.reset()
    return torch.cat([logits, logits], dim=1).double()


def test_rows_match_a_plain_fp32_head(engine):
    rows, _, _ = _rows(engine)
    ref = _reference(engine, PROMPT)
    lsm = torch.log_softmax(ref, dim=1)
    # bf16 logits from weights rounded once to bf16: each logit within two bf16 half-steps of its size, and a log
    # probability moves by its logit's error plus the log-sum-exp's (at most the largest logit error)
    tol = 2 * 2 * 2.0 ** -8 * float(ref.abs().max())
    assert len(rows) == len(PROMPT) - 1
    for p, row in enumerate(rows, start=1):
        want = lsm[p - 1]
        token = PROMPT[p]
        assert row["id"] == token
        assert abs(row["logprob"] - float(want[token])) <= tol, (p, row["logprob"], float(want[token]))
        higher = int((want > want[token] + tol).sum())
        close = int((want >= want[token] - tol).sum())
        assert higher < row["rank"] <= close, (p, row["rank"], higher, close)
        assert len(row["top"]) == TOP
        values = [value for _, value in row["top"]]
        assert values == sorted(values, reverse=True)
        for token_id, value in row["top"]:
            assert abs(value - float(want[token_id])) <= tol
        assert float(want.max()) - values[0] <= tol


@pytest.mark.parametrize("chunk", [7, 16, 98])
def test_rows_do_not_depend_on_the_chunking(engine, chunk):
    """A 99-token prompt in chunks of 7 (the last of 1 row), 16, 98 (a last chunk with no row to score) or one."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    want, _, _ = _rows(engine)
    e = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=chunk)
    got = PromptProbabilities(TOP, PROMPT)
    prefill(e, PROMPT, None, prompt_logprobs=got)
    assert got.emitted() == want


def test_a_repeat_gives_the_same_rows(engine):
    first, out, _ = _rows(engine)
    again, out2, _ = _rows(engine)
    assert again == first and out2 == out


@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)], ids=["greedy", "sampled"])
def test_rows_leave_the_reply_and_the_caches_as_without(engine, sampling):
    _forget(engine)
    plain, _ = _generate(engine, PROMPT, sampling)
    want = [t.clone() for t in _state(engine.e)]
    _, out, _ = _rows(engine, sampling=sampling, tokens=24)
    assert out == plain
    assert all(torch.equal(a, b) for a, b in zip(_state(engine.e), want))


def test_rows_never_resume_a_kept_prefix(engine):
    from tensorfold.families.glm5_next.cuda.decode import prefill

    want, _, _ = _rows(engine)
    engine.request.policy, engine.request.stop_eos = None, False
    engine.generate(PROMPT[:60], 1, None, lambda new: None)                # keeps a prefix of the prompt
    kept = engine._resume(PROMPT, [1, 3, 0, 0])
    assert kept is not None
    rows = PromptProbabilities(TOP, PROMPT)
    stats = engine.generate(list(PROMPT), 1, None, lambda new: None, prompt_logprobs=rows)
    assert stats["cached"] == 0 and rows.emitted() == want
    with pytest.raises(ValueError, match="resumed"):
        prefill(engine.e, PROMPT, None, resume=kept, prompt_logprobs=PromptProbabilities(TOP, PROMPT))


def test_rows_read_as_vllm_completions_prompt_logprobs(engine):
    agreement = pytest.importorskip("glm53_setup.agreement")
    from tensorfold.server.probabilities import prompt_entries

    rows, _, _ = _rows(engine)
    entries = prompt_entries(rows, str)
    parsed = agreement.parse_prompt_logprobs(entries, PROMPT)
    forced = agreement.teacher_forced(parsed, PROMPT)
    assert forced["positions"] == len(PROMPT) - 1
    assert forced["mean_nll"] == pytest.approx(-sum(row["logprob"] for row in rows) / len(rows))
    fields = {"logprob", "rank", "decoded_token"}
    assert all(set(detail) == fields for entry in entries[1:] for detail in entry.values())


def test_a_one_token_prompt_has_no_rows(engine):
    rows, out, _ = _rows(engine, PROMPT[:1])
    assert rows == [] and len(out) == 1
