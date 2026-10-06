"""TF_GLM_KV=fp8 (families/glm5_next/cuda/kv8.py): the DSA latent cache and the indexer's pooled keys as e4m3 rows
with a power-of-two scale each. The kernels store exactly ``kv8.quantize_rows`` of the bf16 rows, read them as the
bf16 kernels read the dequantized rows (the same values up to fp32 summation order), and the engine keeps its contract on such caches: drafted
replies equal serial ones, steps replayed as CUDA graphs equal eager ones, a resumed prompt equals a fresh prefill,
and a prompt's cache rows do not depend on its chunking. The startup estimate counts what the caches allocate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import CONFIG, _checkpoint, _forget, _generate, _TwoCopies  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

PROMPT = 2100              # past the dense limit: the first reply token is already a sparse row
CONTEXT = 2600


def _rows(n, width, gen):
    """bf16 rows of varied magnitude (amax from 2^-12 to 2^9), a zero row and a row with one large value."""
    x = torch.randn((n, width), generator=gen) * torch.exp2(torch.randint(-12, 10, (n, 1), generator=gen).float())
    x[0] = 0.0
    x[1, 7] = 300.0
    return x.to(torch.bfloat16).cuda()


# -- the format ------------------------------------------------------------------------------------------------------

def test_rows_take_the_bytes_the_estimate_counts():
    from tensorfold.cuda.geometry import mla_row_bytes
    from tensorfold.families.glm5_next.cuda import kv8

    for width in (128, 512):
        assert kv8.quantize_rows(torch.zeros((1, width), device="cuda")).shape[1] == mla_row_bytes(width, "fp8")


def test_dequantize_is_codes_times_a_bounding_power_of_two():
    from tensorfold.families.glm5_next.cuda import kv8

    x = _rows(64, 512, torch.Generator().manual_seed(0))
    q = kv8.quantize_rows(x)
    s = q[:, 512:516].contiguous().view(torch.float32)[:, 0]
    assert torch.equal(torch.frexp(s).mantissa, torch.full_like(s, 0.5))               # powers of two
    codes = q[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    assert torch.equal(kv8.dequantize(q), codes * s[:, None])                            # exactly
    assert bool((codes.abs().amax(1)[1:] >= 128).all() and (codes.abs().amax(1) <= 256).all())  # no saturation
    assert torch.equal(kv8.dequantize(q[:1]), torch.zeros((1, 512), device="cuda"))      # the zero row
    assert not q[:, 516:].any()                                                          # the pad stays zero
    xf = x.float()
    err = (kv8.dequantize(q) - xf).abs()
    assert bool((err <= xf.abs() * 2.0 ** -4 + s[:, None] * 2.0 ** -10).all())          # e4m3: 3 mantissa bits


@pytest.mark.parametrize("width", [128, 512])
def test_latent_write_stores_quantize_rows(width):
    from tensorfold.families.glm5_next.cuda import kv8, latent

    x = _rows(40, width, torch.Generator().manual_seed(width))
    cache = torch.zeros((100, width + kv8.PAD), dtype=torch.uint8, device="cuda")
    latent.latent_write(x[:, :], cache, torch.tensor([33], dtype=torch.int32, device="cuda"))
    assert torch.equal(cache[33:73], kv8.quantize_rows(x))
    assert not cache[:33].any() and not cache[73:].any()


def _index_inputs(R, gen):
    k_raw = torch.randn((R, 128), generator=gen).to(torch.bfloat16).cuda()
    gate = torch.randn((R, 128), generator=gen).cuda()
    ln_w = (1.0 + 0.05 * torch.randn(128, generator=gen)).to(torch.bfloat16).cuda()
    ln_b = (0.05 * torch.randn(128, generator=gen)).to(torch.bfloat16).cuda()
    ape = torch.randn((4, 128), generator=gen).to(torch.bfloat16).cuda()
    return k_raw, gate, ln_w, ln_b, ape


@pytest.mark.parametrize("pos, R", [(0, 64), (6, 17), (1021, 5)])
def test_pool_keys_store_quantize_rows_of_the_bf16_pools(pos, R):
    """index_update on an FP8 pooled-key cache stores quantize_rows of the bf16 pools, the same windows."""
    from tensorfold.families.glm5_next.cuda import kv8, sparse

    gen = torch.Generator().manual_seed(pos + R)
    first = _index_inputs(pos, gen) if pos else None
    window = _index_inputs(R, gen)
    cap = 1100
    out = {}
    for kind in ("bf16", "fp8"):
        ik = torch.zeros((cap, 128), dtype=torch.bfloat16, device="cuda")
        ig = torch.zeros_like(ik)
        pk = (torch.zeros((cap // 4 + 2, 128), dtype=torch.bfloat16, device="cuda") if kind == "bf16" else
              torch.zeros((cap // 4 + 2, 128 + kv8.PAD), dtype=torch.uint8, device="cuda"))
        if first is not None:
            sparse.index_update(*first, ik, ig, pk, torch.tensor([0], dtype=torch.int32, device="cuda"))
        sparse.index_update(*window, ik, ig, pk, torch.tensor([pos], dtype=torch.int32, device="cuda"))
        out[kind] = pk
    pools = (pos + R) // 4
    assert out["bf16"][:pools].any()
    assert torch.equal(out["fp8"][:pools], kv8.quantize_rows(out["bf16"][:pools]))
    assert not out["fp8"][pools:].any()


# -- the readers: the bf16 kernels' values on the dequantized rows -----------------------------------------------------

def _caches(n, width, gen):
    from tensorfold.families.glm5_next.cuda import kv8

    q = kv8.quantize_rows(_rows(n, width, gen))
    return q, kv8.dequantize(q).to(torch.bfloat16)                 # exact: e4m3 times a power of two fits bf16


def _close(got, want):
    """The bf16 kernel's values up to its fp32 sums' rounding (the tensor cores take the codes' tile in their own
    order): a handful of elements a bf16 step or two apart."""
    got, want = got.float(), want.float()
    assert (got != want).float().mean().item() < 1e-3
    torch.testing.assert_close(got, want, rtol=2 ** -6, atol=2 ** -6 * want.abs().amax().item())


@pytest.mark.parametrize("P, R", [(1500, 8), (300, 72)])
def test_dense_latent_attention_reads_fp8_as_its_dequantized_rows(P, R):
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(P + R)
    heads, L = 32, 512
    q8, deq = _caches(P + R, L, gen)
    qa = (torch.randn((R, heads, L), generator=gen) * 0.05).to(torch.bfloat16).cuda()
    nch = latent.chunks_for(P + R)
    s = latent.LatentScratch(R, heads, nch, "cuda")
    pos = torch.tensor([P], dtype=torch.int32, device="cuda")
    out = {}
    for name, cache in (("fp8", q8), ("bf16", deq)):
        out[name] = latent.attention(qa, cache, pos, s, scale=0.07, nch=nch,
                                     out=torch.empty((R, heads, L), dtype=torch.bfloat16, device="cuda")).clone()
    _close(out["fp8"], out["bf16"])
    for r in (0, R - 1):                                          # a window row is its serial step, bit for bit
        one = latent.attention(qa[r:r + 1].contiguous(), q8, pos + r, s, scale=0.07, nch=nch,
                               out=torch.empty((1, heads, L), dtype=torch.bfloat16, device="cuda"))
        assert torch.equal(out["fp8"][r], one[0]), f"row {r}"


@pytest.mark.parametrize("R", [4, 72])
def test_sparse_latent_attention_reads_fp8_as_its_dequantized_rows(R):
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(R)
    heads, L, n, W = 32, 512, 6000, 2051
    q8, deq = _caches(n, L, gen)
    qa = (torch.randn((R, heads, L), generator=gen) * 0.05).to(torch.bfloat16).cuda()
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = []
    for r in range(R):
        sel = torch.sort(torch.randperm(n, generator=gen)[:2048 - r % 5]).values
        tokens[r, :sel.numel()] = sel.to(torch.int32)
        counts.append(sel.numel())
    tokens, counts = tokens.cuda(), torch.tensor(counts, dtype=torch.int32, device="cuda")
    out = {}
    for name, cache in (("fp8", q8), ("bf16", deq)):
        out[name] = torch.zeros((R, heads, L), dtype=torch.bfloat16, device="cuda")
        latent.sparse_attention(qa, cache, tokens, counts, out[name], 0.07)
    _close(out["fp8"], out["bf16"])
    for r in (0, R - 1):
        one = torch.zeros((1, heads, L), dtype=torch.bfloat16, device="cuda")
        latent.sparse_attention(qa[r:r + 1].contiguous(), q8, tokens[r:r + 1].contiguous(),
                                counts[r:r + 1].contiguous(), one, 0.07)
        assert torch.equal(out["fp8"][r], one[0]), f"row {r}"


@pytest.mark.parametrize("pos, R", [(2047, 64), (9000, 5)])
def test_token_selection_reads_fp8_pools_as_their_dequantized_rows(pos, R):
    from tensorfold.families.glm5_next.cuda import sparse

    gen = torch.Generator().manual_seed(pos + R)
    H, D = 32, 128
    npool = (pos + R) // 4 + 2
    q8, deq = _caches(npool, D, gen)
    qi = torch.randn((R, H * D), generator=gen).to(torch.bfloat16).cuda()
    wts = torch.randn((R, H), generator=gen).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    got = sparse.select_tokens(qi, wts, q8, pos, R, npool - 2, pos_dev)
    want = sparse.select_tokens(qi, wts, deq, pos, R, npool - 2, pos_dev)
    assert torch.equal(got[1], want[1]) and bool((got[1] > 0).any())
    assert torch.equal(got[0], want[0])


# -- the engine on FP8 caches ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine_kv8(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_kv8")
    _checkpoint(path)
    env = pytest.MonkeyPatch()
    env.setenv("TF_GLM_KV", "fp8")
    try:
        engine = GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=CONTEXT, prefill_rows=256)
    finally:
        env.undo()
    st = engine.e.st
    assert engine.kv == "fp8" and st.kc[0].dtype == st.mtp_kc.dtype == st.index[0][2].dtype == torch.uint8
    assert st.index[0][0].dtype == torch.bfloat16                 # index keys and gates stay bf16
    return engine


def _prompt(seed=11, n=PROMPT):
    return list(np.random.default_rng(seed).integers(0, 1000, size=n))


def test_estimate_counts_what_the_caches_allocate(engine_kv8):
    """mla_cache_bytes (the startup estimate's caches) is what forward.State allocates, fp8 or bf16."""
    from tensorfold.cuda.geometry import mla_cache_bytes
    from tensorfold.families.glm5_next.cuda.forward import State

    text = CONFIG["text_config"]
    for kv in ("fp8", "bf16"):
        st = engine_kv8.e.st if kv == "fp8" else State(engine_kv8.w, engine_kv8.e.st.capacity, 8, kv=kv)
        held = st.kc + [st.mtp_kc] + [x for trio in st.index for x in trio]
        got = sum(t.numel() * t.element_size() for t in held)
        assert got == mla_cache_bytes(text, 2, st.capacity, latent=True, mtp=True, kv=kv), kv
        del st


@pytest.mark.parametrize("n", [300, PROMPT], ids=["dense", "sparse"])
@pytest.mark.parametrize("sampling", [Sampling(99, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_fp8_drafted_equals_serial(engine_kv8, sampling, n):
    """Within the dense limit (dense decode graphs) and past it (sparse graphs)."""
    prompt = _prompt(n=n)
    kind = "main" if n < 2051 else "sparse"
    before = engine_kv8.e.replays[kind]
    serial, stats = _generate(engine_kv8, prompt, sampling, draft=False, tokens=32)
    assert engine_kv8.e.replays[kind] > before
    assert len(serial) == 32 and stats["drafts"] is False
    for policy in (None, "2", "c3:0.35"):
        drafted, _ = _generate(engine_kv8, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy


def test_fp8_graphs_equal_eager_steps(engine_kv8):
    sampling = Sampling(5, 1.0, 20, 0.95)
    prompt = _prompt(seed=12)
    before = dict(engine_kv8.e.replays)
    with_graphs, _ = _generate(engine_kv8, prompt, sampling, tokens=32)
    assert engine_kv8.e.replays["sparse"] > before["sparse"]
    graphs, engine_kv8.e.graphs = engine_kv8.e.graphs, None
    try:
        eager, _ = _generate(engine_kv8, prompt, sampling, tokens=32)
    finally:
        engine_kv8.e.graphs = graphs
    assert with_graphs == eager


def test_fp8_resumed_prompt_equals_a_fresh_prefill(engine_kv8):
    sampling = Sampling(21, 1.0, 20, 0.95)
    first = _prompt(seed=13, n=2080)                    # crosses the dense limit inside a 256-row chunk
    reply, _ = _generate(engine_kv8, first, sampling, tokens=16)
    follow = first + reply + _prompt(seed=14, n=40)
    warm, stats = _generate(engine_kv8, follow, sampling, tokens=16)
    assert stats["cached"] == len(first) - 1
    _forget(engine_kv8)
    cold, stats = _generate(engine_kv8, follow, sampling, tokens=16)
    assert stats["cached"] == 0 and warm == cold


def test_fp8_switching_conversations_resumes_like_a_fresh_prefill(engine_kv8):
    """A, then B (which takes the live caches), then A again: A resumes from its saved FP8 rows as a fresh prefill."""
    sampling = Sampling(31, 1.0, 20, 0.95)
    a, b = _prompt(seed=40, n=2080), _prompt(seed=41, n=2097)
    reply_a, _ = _generate(engine_kv8, a, sampling, tokens=12)
    _generate(engine_kv8, b, sampling, tokens=12)
    next_a = a + reply_a + _prompt(seed=42, n=9)
    warm, stats = _generate(engine_kv8, next_a, sampling, tokens=12)
    assert stats["cached"] == len(a) - 1, stats
    cold, _ = _generate(engine_kv8, next_a, sampling, draft=False, tokens=12)
    assert warm == cold


def test_fp8_prompt_rows_do_not_depend_on_the_chunking(engine_kv8):
    """A prompt prefilled in 256-, 100- and 37-row chunks leaves the same FP8 latent, MTP and pooled-key rows."""
    from tensorfold.families.glm5_next.cuda.decode import prefill

    e = engine_kv8.e
    prompt = _prompt(seed=15, n=2100)
    rows = []
    try:
        for chunk in (256, 100, 37):
            _forget(engine_kv8)
            e.prefill_rows = chunk
            prefill(e, prompt, None)
            st = e.st
            n, m = st.pos, st.mtp_len
            assert n == len(prompt) and m > 0
            rows.append([st.kc[0][:n].clone(), st.mtp_kc[:m].clone()] +
                        [pk[:(n if i < len(st.kc) else m) // 4].clone() for i, (_, _, pk) in enumerate(st.index)])
    finally:
        e.prefill_rows = 256
        _forget(engine_kv8)
    for other in rows[1:]:
        assert all(torch.equal(x, y) for x, y in zip(rows[0], other))
