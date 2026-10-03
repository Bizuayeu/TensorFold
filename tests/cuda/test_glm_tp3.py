"""GLM-5.3-Flash's engine on three ranks on one GPU, each rank a thread (``threadcomm``): a tiny checkpoint whose split
axes cut over three ranks the way GLM-5.3-Flash's do (heads 2/1/1, expert width 192/192/128, vocabulary 448/448/384
of 1,280) while two ranks take halves, MLX 4-bit and NVFP4. Rank 0 serves, the others follow through the doorbell.

Inside three ranks: drafted replies equal serial ones and leave the same caches on every rank, a resumed prompt
equals a fresh prefill, and any prompt chunking (pieced exchanges included) leaves the same bits; dense and past the
dense limit, bf16 and fp8 latents, a chunk's partials gathered or reduce-scattered (TF_GLM_PREFILL_REDUCE),
whose bits are the same. Three ranks and two cut the sums differently, so their bits differ: each is held
to an FP32 head over its own final rows (every vocabulary id gathered into its place) and to the other by how far
their final rows differ. Decode windows run eager here (the exchanges wait on the host);
``test_three_rank_shapes_capture_their_graphs`` captures them on rank 0's and rank 2's shapes, one rank standing in
for three."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402
from threadcomm import Closed, Hub, Store, run_ranks  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.engine.probabilities import PromptProbabilities  # noqa: E402

SHAPE = {"heads": 4, "lin_heads": 4, "moe_width": 512, "dense_width": 384, "vocab": 1280}
LONG = 2600                                  # a context past the dense limit (2,051 tokens)
PROMPT = {None: 37, LONG: 2100}              # prompt lengths, dense and past the dense limit
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(77, 1.0, 400, 0.98), Sampling(1234, 1.0, 0, 0.9, 0.02)]
SAMPLING_IDS = ["greedy", "top_k", "top_k past a shard", "nucleus"]


class Ranks:
    """``world`` engines on one GPU, rank r a thread: ``serve(fn)`` runs fn(rank 0's engine) while the others follow,
    ``each(fn)`` fn(rank, engine) on every rank at once."""

    def __init__(self, path, world: int, **options) -> None:
        from tensorfold.families.glm5_next.cuda.engine import GlmEngine

        self.world, self.hub, self.store = world, Hub(world), Store()
        self.engines = run_ranks(lambda r, comm: GlmEngine(path, rank=r, master="", port=0, world=world, comm=comm,
                                                           graphs=False, **options), world, self.hub, self.store)

    def each(self, fn) -> list:
        if self.hub.barrier.broken:                  # an earlier failure: let the next test run
            self.hub.barrier.reset()
        return run_ranks(lambda r, _: fn(r, self.engines[r]), self.world, self.hub, self.store)

    def serve(self, fn):
        self.store.open()

        def body(r, e):
            if r:
                try:
                    e.follow()
                except Closed:                       # rank 0 is done: every bell it rang was answered
                    return None
            try:
                return fn(e)
            finally:
                self.store.close()

        return self.each(body)[0]

    def generate(self, prompt, sampling, **kw):
        return self.serve(lambda e: _generate(e, prompt, sampling, **kw))

    def forget(self) -> None:
        for e in self.engines:
            _forget(e)


def _caches(e) -> list[torch.Tensor]:
    """What a request leaves on a rank: KDA states and conv windows, attention (and indexer) rows below the position;
    the MTP head's caches are left out (serial decoding does not run it)."""

    st = e.st
    out = [st.rec[st.cur[0]], st.conv] + [x[:st.pos] for x in st.kc + st.vc if x is not None]
    for ik, ig, pk in (st.index or [])[:len(st.kc)]:
        out += [ik[:st.pos], ig[:st.pos], pk[:st.pos // 4]]
    return [t.clone() for t in out]


def _equal(a: list, b: list) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    out = {}
    for quant in ("mlx", "nvfp4"):
        path = tmp_path_factory.mktemp(f"glm_tp3_{quant}")
        _checkpoint(path, nvfp4=quant == "nvfp4", **SHAPE)
        out[quant] = path
        # one thread loads and compiles the kernels first, not three at once
        e = GlmEngine(path, rank=0, master="", port=0, world=3, comm=_TwoCopies(3), context=LONG, graphs=False)
        _generate(e, list(range(PROMPT[LONG])), None, tokens=4)
        del e
    torch.cuda.empty_cache()
    return out


CASES = [(q, c, kv, red) for q in ("mlx", "nvfp4") for c in (None, LONG) for kv in ("bf16", "fp8")
         for red in ("gather", "scatter")]


@pytest.fixture(scope="module", params=CASES,
                ids=[f"{q}-{'long' if c else 'dense'}-{kv}-{red}" for q, c, kv, red in CASES])
def three(request, checkpoints):
    quant, context, kv, red = request.param
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TF_GLM_KV", kv)
        mp.setenv("TF_GLM_PREFILL_REDUCE", red)
        ranks = Ranks(checkpoints[quant], 3, **({"context": context} if context else {}))
    assert [e.w.vocab_spans for e in ranks.engines] == [[(0, 448), (448, 896), (896, 1280)]] * 3
    assert [e.e.pbuf.scatter for e in ranks.engines] == [red == "scatter"] * 3
    ranks.context, ranks.reduce = context, red
    yield ranks
    del ranks
    torch.cuda.empty_cache()


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=SAMPLING_IDS)
def test_drafted_replies_equal_serial_and_leave_its_caches(three, sampling):
    prompt = list(np.random.default_rng(5).integers(0, 1000, size=PROMPT[three.context]))
    tokens = 32 if three.context else 24

    def run(**kw):
        three.forget()
        out, stats = three.generate(prompt, sampling, tokens=tokens, **kw)
        return out, stats, three.each(lambda r, e: _caches(e.e))

    serial, _, want = run(draft=False)
    assert len(serial) == tokens
    for policy in (None, "2", "c3:0.35"):
        drafted, stats, got = run(policy=policy)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1 and 0 <= stats["accepted"] <= stats["drafted"], (policy, stats)
        assert all(_equal(w, g) for w, g in zip(want, got)), policy
    again, _, _ = run(draft=False)
    assert again == serial


@pytest.mark.parametrize("sampling", [SAMPLINGS[1], None], ids=["sampled", "greedy"])
def test_resumed_prompts_equal_fresh_prefills(three, sampling):
    first = list(np.random.default_rng(9).integers(0, 1000, size=70 if three.context is None else 2100))
    three.forget()
    reply, _ = three.generate(first, sampling)
    after = first + reply + [5, 6, 7]
    warm, stats = three.generate(after, sampling)
    assert stats["cached"] == len(first) - 1                # the reply prefills again
    resumed = three.each(lambda r, e: _caches(e.e))
    three.forget()
    cold, stats = three.generate(after, sampling)
    assert stats["cached"] == 0 and warm == cold
    assert all(_equal(a, b) for a, b in zip(resumed, three.each(lambda r, e: _caches(e.e))))


def test_prompt_chunks_leave_the_same_state(three, monkeypatch):
    """Any chunking, on every rank: a 300-row chunk's exchanges go in three pieces (TF_GLM_PREFILL_OVERLAP), 64- and
    7-row ones in one; past the dense limit 2,048-row chunks (in four pieces, then three) and 64-row ones."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    size, chunks = (300, (300, 64, 7)) if three.context is None else (2400, (2048, 64))
    prompt = [int(t) for t in np.random.default_rng(53).integers(0, 1000, size=size)]
    monkeypatch.setenv("TF_GLM_PREFILL_REDUCE", three.reduce)

    def run(rows):
        def fn(r, e):
            d = Engine(e.w, capacity=e.capacity_plan["cache_slots"], max_rows=8, prefill_rows=rows,
                       long_context=e.w.meta["long_context"], kv=e.kv)
            first = prefill(d, prompt, None)
            pieces = len(d.pbuf.overlap.cut)          # the last chunk's
            out = first, pieces, _caches(d)
            del d
            return out
        return three.each(fn)

    runs = [run(rows) for rows in chunks]
    assert [p for _, p, _ in runs[0]] == [3] * 3          # the last chunk (300 rows, or 352 after 2,048) in pieces
    (want, *others) = runs
    for got in others:
        for (a, _, x), (b, _, y) in zip(want, got):
            assert a == b and _equal(x, y)


def _prefill_bits(ranks: Ranks, prompt, rows: int, monkeypatch, reduce: str, on: str, pieces: str) -> list:
    """Every rank's first token, caches (the MTP head's too), its last chunk's pieces and the point-to-point exchanges
    of a fresh prompt buffer of ``rows`` rows prefilling ``prompt`` under the given TF_GLM_PREFILL_* settings."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    monkeypatch.setenv("TF_GLM_PREFILL_REDUCE", reduce)
    monkeypatch.setenv("TF_GLM_PREFILL_OVERLAP", on)
    monkeypatch.setenv("TF_GLM_OVERLAP_PIECES", pieces)

    def fn(r, e):
        d = Engine(e.w, capacity=e.capacity_plan["cache_slots"], max_rows=8, prefill_rows=rows,
                   long_context=e.w.meta["long_context"], kv=e.kv)
        e.w.comm.calls.clear()
        first = prefill(d, prompt, None)
        st = d.st
        cut = len(d.pbuf.overlap.cut) if d.pbuf.overlap is not None else 0
        out = first, _caches(d) + [st.mtp_kc[:st.mtp_len].clone()], cut, e.w.comm.calls.get("send_recv", 0)
        del d
        return out

    return ranks.each(fn)


SETTINGS = [("scatter", "0", "4"), ("scatter", "1", "1"), ("scatter", "1", "4"), ("gather", "1", "4")]


def _same_bits_both_ways(ranks: Ranks, prompt, chunks, monkeypatch) -> None:
    from tensorfold.families.glm5_next.cuda.overlap import ranges

    want = _prefill_bits(ranks, prompt, chunks[0], monkeypatch, "gather", "0", "4")
    assert all(n == 0 for *_, n in want)
    for rows in chunks:
        last = len(prompt) - (len(prompt) - 1) // rows * rows
        for red, on, pieces in SETTINGS:
            got = _prefill_bits(ranks, prompt, rows, monkeypatch, red, on, pieces)
            for (a, x, _, _), (b, y, cut, p2p) in zip(want, got):
                assert a == b and _equal(x, y), (rows, red, on, pieces)
                assert (p2p > 0) == (red == "scatter"), (rows, red, p2p)
                assert cut == (len(ranges(last, int(pieces))) if on == "1" else 0), (rows, on, pieces, cut)


def test_scattered_prompt_sums_leave_the_gathered_bits(three, monkeypatch):
    """TF_GLM_PREFILL_REDUCE=scatter against gather, unpieced and in one or four pieces, on every rank: the first
    token and every cache (the MTP head's too) bit for bit. Chunks of 500 rows (four pieces, 500 = 3 x 166 + 2), 499
    and 498 (a last chunk of 1 and 2 rows: fewer rows than ranks gather), 7; past the dense limit 2,048 (then 352, in
    three pieces) and 2,399 (then 1)."""

    if three.reduce == "scatter":
        pytest.skip("the gather instance builds both")
    size, chunks = (500, (500, 499, 498, 7)) if three.context is None else (2400, (2048, 2399))
    prompt = [int(t) for t in np.random.default_rng(61).integers(0, 1000, size=size)]
    _same_bits_both_ways(three, prompt, chunks, monkeypatch)


# -- three ranks against two ---------------------------------------------------------------------------------------
TOP = 5
ROWS_PROMPT = [int(t) for t in np.random.default_rng(7).integers(0, 1000, size=99)]


@pytest.fixture(scope="module", params=["mlx", "nvfp4"])
def worlds(request, checkpoints):
    from tensorfold.families.glm5_next.cuda.qmm import B16, dequantize_q4
    from tensorfold.families.glm5_next.cuda.weights import load

    path = checkpoints[request.param]
    one = load(path, rank=0, world=1, mtp=False)        # the whole head, read without any split
    head = one.head.weight.float() if isinstance(one.head, B16) else dequantize_q4(one.head)
    del one
    ranks = {world: Ranks(path, world) for world in (3, 2)}
    yield request.param, head, ranks
    del ranks
    torch.cuda.empty_cache()


def _logits(ranks: Ranks, prompt, head) -> torch.Tensor:
    """FP32 logits of every row of ``prompt``: the final-normed rows (the same on every rank) through the whole head."""

    from tensorfold.families.glm5_next.cuda.decode import Engine
    from tensorfold.families.glm5_next.cuda.forward import chunks_for, compute, stage

    def fn(r, e):
        d = Engine(e.w, capacity=2560, max_rows=8, prefill_rows=len(prompt))
        d.reset()
        compute(e.w, d.st, d.pbuf, stage(e.w, d.st, d.pbuf, prompt), nch=chunks_for(d.st, len(prompt)), host_pos=0)
        out = d.pbuf.fnormed[:len(prompt)].float().clone()
        d.reset()
        del d
        return out

    hidden = ranks.each(fn)
    assert all(torch.equal(h, hidden[0]) for h in hidden)
    return (hidden[0] @ head.T).double()


def _rows(ranks: Ranks, prompt):
    def fn(e):
        rows = PromptProbabilities(TOP, prompt)
        e.request.policy, e.request.stop_eos = None, False
        out: list[int] = []
        e.generate(list(prompt), 24, None, out.extend, prompt_logprobs=rows)
        return rows.emitted(), out

    ranks.forget()
    return ranks.serve(fn)


def _tolerance(logits: torch.Tensor) -> float:
    # test_glm_prompt_logprobs: bf16 logits from weights rounded once to bf16, each within two bf16 half-steps of its
    # size; a log probability moves by its logit's error plus the log-sum-exp's
    return 2 * 2 * 2.0 ** -8 * float(logits.abs().max())


def test_prompt_rows_put_every_id_in_its_place(worlds):
    """Each rank count's prompt rows (log probability, rank, top entries) against an FP32 head over its own rows."""

    _, head, ranks = worlds
    for world, rk in ranks.items():
        rows, _ = _rows(rk, ROWS_PROMPT)
        logits = _logits(rk, ROWS_PROMPT, head)[:-1]
        lsm = torch.log_softmax(logits, dim=1)
        tol = _tolerance(logits)
        for p, row in enumerate(rows, start=1):
            want, token = lsm[p - 1], ROWS_PROMPT[p]
            assert abs(row["logprob"] - float(want[token])) <= tol, (world, p)
            higher, close = int((want > want[token] + tol).sum()), int((want >= want[token] - tol).sum())
            assert higher < row["rank"] <= close, (world, p, row["rank"], higher, close)
            assert len(row["top"]) == TOP
            assert all(abs(lp - float(want[i])) <= tol for i, lp in row["top"]), (world, p, row["top"])


def test_three_ranks_and_two_differ_by_what_their_final_rows_differ(worlds):
    """Prompt rows of three ranks and of two: each log probability apart by no more than an FP32 head puts their
    final rows apart, plus each side's bf16 rounding; the replies' greedy tokens are reported."""

    quant, head, ranks = worlds
    (rows3, reply3), (rows2, reply2) = _rows(ranks[3], ROWS_PROMPT), _rows(ranks[2], ROWS_PROMPT)
    l3, l2 = _logits(ranks[3], ROWS_PROMPT, head)[:-1], _logits(ranks[2], ROWS_PROMPT, head)[:-1]
    s3, s2 = torch.log_softmax(l3, dim=1), torch.log_softmax(l2, dim=1)
    tol = _tolerance(l3) + _tolerance(l2)
    seen, apart = 0.0, 0.0
    for p, (a, b) in enumerate(zip(rows3, rows2), start=1):
        token = ROWS_PROMPT[p]
        rows_apart = abs(float(s3[p - 1, token] - s2[p - 1, token]))
        assert abs(a["logprob"] - b["logprob"]) <= rows_apart + tol, (p, a["logprob"], b["logprob"], rows_apart)
        seen, apart = max(seen, abs(a["logprob"] - b["logprob"])), max(apart, rows_apart)
    same = next((i for i, (x, y) in enumerate(zip(reply3, reply2)) if x != y), len(reply3))
    split = ""
    if same < len(reply3):         # where the replies part, the two tokens are within the rounding of their logits
        context = ROWS_PROMPT + reply3[:same]
        a3, a2 = _logits(ranks[3], context, head)[-1], _logits(ranks[2], context, head)[-1]
        x, y = reply3[same], reply2[same]
        lead3, lead2 = float(a3[x] - a3[y]), float(a2[x] - a2[y])
        assert max(abs(lead3), abs(lead2)) <= float((a3 - a2).abs().max()) + _tolerance(a3) + _tolerance(a2)
        step = 2.0 ** (np.floor(np.log2(abs(float(a3[x])))) - 7)            # a bf16 step at that logit
        split = (f"; at token {same} three ranks take {x} and two take {y}: an FP32 head puts {x} ahead by {lead3:.4g} "
                 f"(three) and {lead2:.4g} (two), a bf16 step there {step:.4g}")
    print(f"\n[tp3 vs tp2 {quant}] prompt rows: max |logprob 3 - 2| {seen:.4g}; final rows through an FP32 head "
          f"{apart:.4g} apart (max |logit 3 - 2| {float((l3 - l2).abs().max()):.4g}); bf16 tolerance {tol:.4g} "
          f"(max |logit| {float(l3.abs().max()):.4g}); greedy replies agree on {same} of {len(reply3)} tokens{split}",
          flush=True)


def test_decisions_score_every_label_in_its_place(worlds):
    """``score_labels`` over every id on three ranks: the gathered label logits and log-sum-exp match an FP32 head."""

    _, head, ranks = worlds
    rk = ranks[3]
    labels = list(range(SHAPE["vocab"]))
    got, lse = rk.serve(lambda e: e.score_labels(ROWS_PROMPT, labels))
    want = _logits(rk, ROWS_PROMPT, head)[-1]
    tol = _tolerance(want)
    assert len(got) == len(labels)
    assert float((torch.tensor(got, dtype=torch.float64, device=want.device) - want).abs().max()) <= tol
    assert abs(lse - float(torch.logsumexp(want, 0))) <= tol


def test_two_ranks_scattered_prompt_sums_leave_the_gathered_bits(worlds, monkeypatch):
    _, _, ranks = worlds
    prompt = [int(t) for t in np.random.default_rng(67).integers(0, 1000, size=500)]
    _same_bits_both_ways(ranks[2], prompt, (500, 499), monkeypatch)


def test_sampled_replies_repeat(worlds):
    _, _, ranks = worlds
    rk = ranks[3]
    rk.forget()
    first, _ = rk.generate(ROWS_PROMPT, SAMPLINGS[1], draft=False)
    rk.forget()
    again, _ = rk.generate(ROWS_PROMPT, SAMPLINGS[1])
    assert again == first


# -- the startup estimate ------------------------------------------------------------------------------------------
def test_each_rank_holds_its_startup_estimate(checkpoints):
    """Each rank's weights as loaded are the bytes its startup estimate counts, the 4-bit draft head of the BF16 head
    included: ``qmm.pack`` pads its rows to 128 (ranks 0 and 1 hold 512 rows for their 448)."""

    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.glm5_next.cuda import split
    from tensorfold.families.glm5_next.cuda.weights import load

    path = checkpoints["nvfp4"]
    for rank in range(3):
        w = load(path, rank=rank, world=3)
        assert w.draft_head is not None
        transform = split_weights(split.rule, w.plan)
        assert sum(transform(name, info)[0] for name, info in headers(path).items()) == w.nbytes(), rank
        del w
    torch.cuda.empty_cache()


# -- refusals and graphs -------------------------------------------------------------------------------------------
def test_a_rank_started_otherwise_is_named(checkpoints):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    def start(r, comm):
        GlmEngine(checkpoints["mlx"], rank=r, master="", port=0, world=3, comm=comm, graphs=False,
                  prefill_rows=1024 if r == 2 else None)

    with pytest.raises(RuntimeError, match=r"different settings.*rank 0 \[.*\], rank 2 \["):
        run_ranks(start, 3)


def test_a_rank_reducing_otherwise_is_named(checkpoints, monkeypatch):
    """TF_GLM_PREFILL_REDUCE scatter on rank 2 only (the environment is the process's: the setting is patched per
    thread) is refused at startup, before any prompt exchange could pair a send with an all-gather."""

    import threading

    from tensorfold.families.glm5_next.cuda import reduce
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    odd = threading.local()
    real = reduce.settings
    monkeypatch.setattr(reduce, "settings", lambda env=None: "scatter" if getattr(odd, "on", False) else real(env))

    def start(r, comm):
        odd.on = r == 2
        GlmEngine(checkpoints["mlx"], rank=r, master="", port=0, world=3, comm=comm, graphs=False)

    with pytest.raises(RuntimeError, match=r"different settings.*TF_GLM_PREFILL_REDUCE\): rank 0 \[.*\], rank 2 \["):
        run_ranks(start, 3)


def test_scatter_needs_point_to_point(checkpoints, monkeypatch):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    monkeypatch.setenv("TF_GLM_PREFILL_REDUCE", "scatter")
    with pytest.raises(ValueError, match="TF_GLM_PREFILL_REDUCE=scatter: the communicator has no send_recv"):
        GlmEngine(checkpoints["mlx"], rank=0, master="", port=0, world=3, comm=_TwoCopies(3), graphs=False)


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("quant, context", [("mlx", None), ("nvfp4", None), ("mlx", LONG)],
                         ids=["mlx", "nvfp4", "long"])
def test_three_rank_shapes_capture_their_graphs(checkpoints, rank, quant, context):
    """One rank stands in for three (its own partials thrice): rank 0's shapes (the widest shares) and rank 2's (the
    narrowest) capture their decode windows as CUDA graphs, and drafted replies equal serial ones through them."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    e = GlmEngine(checkpoints[quant], rank=rank, master="", port=0, world=3, comm=_TwoCopies(3, rank),
                  **({"context": context} if context else {}))
    assert e.e.graphs is not None and e.e.graphs.main
    prompt = list(np.random.default_rng(11).integers(0, 1000, size=PROMPT[context]))
    for sampling in (None, SAMPLINGS[2]):
        _forget(e)
        serial, _ = _generate(e, prompt, sampling, draft=False, tokens=32)
        for policy in (None, "2", "c3:0.35"):
            _forget(e)
            drafted, _ = _generate(e, prompt, sampling, policy=policy, tokens=32)
            assert drafted == serial, (sampling, policy)
    del e
    torch.cuda.empty_cache()
