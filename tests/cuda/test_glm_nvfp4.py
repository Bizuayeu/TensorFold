"""GLM-5.3-Flash from a ModelOpt NVFP4 checkpoint (nvidia/GLM-5.3-Flash-NVFP4's layout) on the tiny synthetic model:
each rank holds the checkpoint's own values (routed experts in ``Experts4`` blocks, the dense MLP as ``Fp4Linear``),
the MTP layer's BF16 experts are packed to NVFP4 for drafting, the two ranks' partials of a layer add up to the whole
layer's dequantized reference, the engine keeps its contract (drafted replies equal serial ones, CUDA graphs equal
eager steps, a resumed prompt equals a fresh prefill, prompt chunking leaves the same state), and the startup estimate
counts what the weights hold."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import MOE, D, _checkpoint, _forget, _generate, _state, _TwoCopies  # noqa: E402

from tensorfold.cuda.nvfp4 import experts as nvx  # noqa: E402
from tensorfold.cuda.nvfp4 import format as fmt  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import latent, split  # noqa: E402

L = "model.language_model."
FILE = "model-00001-of-00001.safetensors"


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    p = tmp_path_factory.mktemp("glm_nvfp4")
    _checkpoint(p, nvfp4=True)
    return p


@pytest.fixture(scope="module")
def ranks(path):
    from tensorfold.families.glm5_next.cuda.weights import load

    return [load(path, rank=r) for r in (0, 1)]


@pytest.fixture(scope="module")
def engine(path):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies())


def _stored(path, name: str) -> np.ndarray:
    """A tensor's bytes as the checkpoint stores them, shaped (uint8 for the NVFP4 codes and e4m3 scales)."""

    header, base = split.read_header(path / FILE)
    info = header[name]
    a, b = info["data_offsets"]
    raw = np.fromfile(path / FILE, dtype=np.uint8, count=b - a, offset=base + a)
    if info["dtype"] == "F32":
        return raw.view(np.float32).reshape(info["shape"])
    if info["dtype"] == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
    return raw.reshape(info["shape"])


def _dequant(path, base: str) -> np.ndarray:
    """[N, K] fp32 of a stored NVFP4 linear: codes x e4m3 x the fp32 weight scale."""

    return fmt.dequant("nvfp4", _stored(path, base + ".weight"), _stored(path, base + ".weight_scale"),
                       float(_stored(path, base + ".weight_scale_2")))


def _rank_part(full: np.ndarray, proj: str, rank: int) -> np.ndarray:
    """Rank ``rank``'s rows of gate/up, columns of down (split.py's ROW and COL)."""

    if proj == "down_proj":
        k = full.shape[1] // 2
        return full[:, rank * k:(rank + 1) * k]
    n = full.shape[0] // 2
    return full[rank * n:(rank + 1) * n]


# -- (a) each rank holds the checkpoint's values ---------------------------------------------------------------------

def test_config_names_the_nvfp4_checkpoint(path, ranks):
    from tensorfold.cuda.nvfp4.linear import Fp4Linear
    from tensorfold.families.glm5_next.cuda.qmm import B16
    from tensorfold.families.glm5_next.cuda.weights import Config

    assert Config.read(path).quant == "nvfp4"
    assert split.read_header(path / FILE)[0][f"{L}layers.0.self_attn.q_conv1d.weight"]["dtype"] == "F32"
    w = ranks[0]
    assert isinstance(w.layers[1].moe.experts, nvx.Experts4) and w.layers[1].moe.experts.limit == 10.0
    assert isinstance(w.layers[1].moe.shared.gu, B16)                          # the shared expert stays BF16
    assert all(isinstance(lin, Fp4Linear) for lin in (*w.layers[0].mlp.gu, w.layers[0].mlp.down))
    assert isinstance(w.layers[1].dsa.proj, B16) and isinstance(w.mtp.eh, B16) and isinstance(w.embed, torch.Tensor)
    conv = torch.from_numpy(np.concatenate([_stored(path, f"{L}layers.0.self_attn.{x}_conv1d.weight")[:128]
                                            for x in "qkv"])).reshape(-1, 4)
    assert torch.equal(w.layers[0].kda.conv.cpu(), conv)                         # the fp32 taps as stored


def test_routed_experts_are_the_checkpoints_values(path, ranks):
    """Every expert matrix of each rank, decoded from its blocks, equals the stored codes x e4m3 x weight_scale_2."""

    for rank, w in enumerate(ranks):
        ex = w.layers[1].moe.experts
        assert ex.count == 8 and ex.width == MOE // 2 and ex.dims == D
        for e in range(8):
            for which in ("gate", "up", "down"):
                proj = f"{which}_proj"
                want = _rank_part(_dequant(path, f"{L}layers.1.mlp.experts.{e}.{proj}"), proj, rank)
                got = nvx.dense(ex, e, which).cpu().numpy()
                assert np.array_equal(got, want), (rank, e, which)


def test_dense_mlp_is_the_checkpoints_values(path, ranks):
    """The dense MLP's projections times the identity: the stored values (an e2m1 code times an e4m3 scale is exact in
    bf16, the weight scale one fp32 product), on the prompt GEMM and on the decode matmul."""

    for rank, w in enumerate(ranks):
        m = w.layers[0].mlp
        for lin, proj in ((m.gu[0], "gate_proj"), (m.gu[1], "up_proj"), (m.down, "down_proj")):
            want = torch.from_numpy(_rank_part(_dequant(path, f"{L}layers.0.mlp.{proj}"), proj, rank)).t().cuda()
            eye = torch.eye(lin.k, dtype=torch.bfloat16, device="cuda")
            torch.testing.assert_close(lin.prefill(eye, f32=True), want, rtol=2 ** -20, atol=0)
            torch.testing.assert_close(lin(eye[:8], f32=True), want[:8], rtol=2 ** -20, atol=0)


def test_mtp_experts_are_drafting_nvfp4_of_the_bf16_weights(path, ranks):
    """The MTP layer's BF16 routed experts, each rank's part packed by ModelOpt's recipe (``nvx.quantize``)."""

    for rank, w in enumerate(ranks):
        ex = w.mtp.layer.moe.experts
        assert isinstance(ex, nvx.Experts4)
        for e in (0, 7):
            for which in ("gate", "up", "down"):
                proj = f"{which}_proj"
                bf16 = _rank_part(_stored(path, f"{L}layers.2.mlp.experts.{e}.{proj}.weight"), proj, rank)
                words, scales, g = nvx.quantize(torch.from_numpy(np.ascontiguousarray(bf16))[None].cuda())
                want = fmt.dequant("nvfp4", words[0].cpu().numpy(), scales[0].cpu().numpy(), float(g[0]))
                assert np.array_equal(nvx.dense(ex, e, which).cpu().numpy(), want), (rank, e, which)


def _tensors(x) -> list[torch.Tensor]:
    """Every tensor a loaded part holds, in attribute order."""

    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _tensors(v)]
    if hasattr(x, "__dict__") and type(x).__module__.startswith("tensorfold"):
        return [t for v in vars(x).values() for t in _tensors(v)]
    return []


@pytest.mark.parametrize("chosen", [[0], [1]], ids=["kda dense", "dsa moe"])
def test_chosen_layers_load_as_in_the_whole_model(path, ranks, chosen):
    """``layers=`` builds only those layers (the MTP layer, head and embedding as ever), each the whole load's, so a
    test can hold a few real layers of several ranks on one GPU."""

    from tensorfold.families.glm5_next.cuda.weights import load

    for rank, whole in enumerate(ranks):
        w = load(path, rank=rank, layers=chosen)
        assert [lw.index for lw in w.layers] == chosen
        pairs = [(w.layers[0], whole.layers[chosen[0]]), (w.mtp, whole.mtp), (w.head, whole.head),
                 (w.embed, whole.embed)]
        for got, want in pairs:
            a, b = _tensors(got), _tensors(want)
            assert a and len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="layer"):
        load(path, rank=0, layers=[2])


def test_precision_checkpoint_is_refused_by_name(path):
    from tensorfold.cuda import precision
    from tensorfold.families.glm5_next.cuda.weights import load

    with precision.using(precision.CHECKPOINT, asked=True), pytest.raises(ValueError, match="--precision full"):
        load(path, rank=0)


# -- (b) the two ranks add up to the whole layer -----------------------------------------------------------------------

class _Capture:
    """One rank's all-gather: keeps what the rank sends (its fp32 partial)."""

    world = 2

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.sent = send.clone()
        n = send.numel()
        recv.view(-1)[:n].copy_(send.reshape(-1))
        recv.view(-1)[n:2 * n].copy_(send.reshape(-1))


def _partials(w, block, layer, x: torch.Tensor, prefill: bool):
    from tensorfold.families.glm5_next.cuda.forward import Buffers
    from tensorfold.families.glm5_next.cuda.qmm import group_sums

    R = x.shape[0]
    w.comm = _Capture()
    b = Buffers(w, 64 if prefill else 8, prefill=prefill)
    b.normed[:R].copy_(x)
    group_sums(x, b.xs[:R])
    block(layer, w, b, R)
    torch.cuda.synchronize()
    return w.comm.sent.view(R, D).double(), b


def _bf16(v: torch.Tensor) -> torch.Tensor:
    return v.float().to(torch.bfloat16).double()


def _swiglu(g: torch.Tensor, u: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    g, u = _bf16(g).clamp(max=limit), _bf16(u).clamp(-limit, limit)
    return _bf16(_bf16(g / (1 + torch.exp(-g))) * u)


def _mlp_reference(x: torch.Tensor, gate, up, down, rank: int | None = None) -> torch.Tensor:
    """The MLP in float64 with the kernels' bf16 roundings; ``rank``: that rank's half of the intermediate width."""
    if rank is not None:
        half = slice(rank * gate.shape[0] // 2, (rank + 1) * gate.shape[0] // 2)
        gate, up, down = gate[half], up[half], down[:, half]
    return _swiglu(x @ gate.t(), x @ up.t()) @ down.t()


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    """Within the kernels' fp32 sums (the bf16 roundings are the reference's own): measured 5e-8 to 1e-7."""
    assert ((got - want).norm() / want.norm()).item() < 1e-5


@pytest.mark.parametrize("prefill", [False, True], ids=["window", "prompt"])
@pytest.mark.parametrize("R", [1, 5])
def test_two_ranks_add_up_to_the_dense_layer(path, ranks, R, prefill):
    from tensorfold.families.glm5_next.cuda.forward import mlp_block

    x = torch.randn((R, D), generator=torch.Generator().manual_seed(R)).to(torch.bfloat16).cuda()
    total = sum(_partials(w, mlp_block, w.layers[0], x, prefill)[0] for w in ranks)
    full = [torch.from_numpy(_dequant(path, f"{L}layers.0.mlp.{p}")).double().cuda()
            for p in ("gate_proj", "up_proj", "down_proj")]
    _close(total, _mlp_reference(x.double(), *full))


@pytest.mark.parametrize("prefill", [False, True], ids=["window", "prompt"])
@pytest.mark.parametrize("R", [1, 5])
def test_two_ranks_add_up_to_the_moe_layer(path, ranks, R, prefill):
    """Rank 0's partial plus rank 1's: the routed experts (dequantized from the whole checkpoint tensors) and the
    shared expert (BF16), weighted by the routing both ranks compute alike; a prompt chunk keeps each rank's slot
    outputs in bf16, and so does the reference."""

    from tensorfold.families.glm5_next.cuda.forward import moe_block

    x = torch.randn((R, D), generator=torch.Generator().manual_seed(10 + R)).to(torch.bfloat16).cuda()
    runs = [_partials(w, moe_block, w.layers[1], x, prefill) for w in ranks]
    (p0, b0), (p1, b1) = runs
    assert torch.equal(b0.pick[:R], b1.pick[:R]) and torch.equal(b0.wts[:R], b1.wts[:R])
    xd = x.double()

    def expert(name: str, stored) -> tuple:
        return tuple(torch.from_numpy(stored(path, f"{L}layers.1.mlp.{name}.{p}")).double().cuda()
                     for p in ("gate_proj", "up_proj", "down_proj"))

    routed = [expert(f"experts.{e}", _dequant) for e in range(8)]
    shared = expert("shared_experts", lambda p, n: _stored(p, n + ".weight"))
    want = torch.zeros((R, D), dtype=torch.float64, device="cuda")
    for r in range(R):
        for s, e in enumerate(b0.pick[r].tolist()):
            for rank in (0, 1):
                y = _mlp_reference(xd[r:r + 1], *(shared if e == 8 else routed[e]), rank=rank)[0]
                want[r] += float(b0.wts[r, s]) * (_bf16(y) if prefill else y)
    _close(p0 + p1, want)


# -- (c) the engine's contract ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("sampling", [Sampling(4321, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafted_replies_equal_serial(engine, sampling):
    prompt = list(np.random.default_rng(8).integers(0, 1000, size=45))
    before = engine.e.replays["main"]
    serial, stats = _generate(engine, prompt, sampling, draft=False, tokens=32)
    assert engine.e.replays["main"] > before and stats["drafts"] is False
    for policy in (None, "auto", "1", "2", "3", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(engine, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1 and stats["min_rows"] >= 2, (policy, stats)


def test_graphs_equal_eager_steps(engine):
    sampling = Sampling(5, 1.0, 20, 0.95)
    prompt = list(np.random.default_rng(12).integers(0, 1000, size=50))
    before = dict(engine.e.replays)
    with_graphs, _ = _generate(engine, prompt, sampling, tokens=32)
    assert engine.e.replays["main"] > before["main"] and engine.e.replays["mtp"] > before["mtp"]
    graphs, engine.e.graphs = engine.e.graphs, None
    try:
        eager, _ = _generate(engine, prompt, sampling, tokens=32)
    finally:
        engine.e.graphs = graphs
    assert with_graphs == eager


@pytest.mark.parametrize("sampling", [Sampling(21, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_resumed_prompt_equals_a_fresh_prefill(engine, sampling):
    rng = np.random.default_rng(22)
    first = list(rng.integers(0, 1000, size=70))
    reply, _ = _generate(engine, first, sampling, tokens=20)
    after = first + reply + [31, 32]
    warm, stats = _generate(engine, after, sampling)
    assert stats["cached"] == len(first) - 1
    _forget(engine)
    cold, stats = _generate(engine, after, sampling)
    assert stats["cached"] == 0 and warm == cold
    serial, _ = _generate(engine, after, sampling, draft=False)
    assert serial == cold


def test_prompt_chunks_leave_the_same_state(engine):
    """A 300-row prompt in 2,048-, 100-, 16- and 7-row chunks: the same first token and bit-identical states (the
    grouped NVFP4 kernel's prompt items and the dense prompt GEMM never depend on the chunk)."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    prompt = [int(t) for t in np.random.default_rng(128).integers(0, 1000, size=300)]
    runs = []
    for rows in (2048, 100, 16, 7):
        e = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=rows)
        runs.append((prefill(e, prompt, None), [t.clone() for t in _state(e)]))
        del e
    first, want = runs[0]
    for rows, (token, got) in zip((100, 16, 7), runs[1:]):
        assert token == first and all(torch.equal(x, y) for x, y in zip(want, got)), rows


# -- (d) the startup estimate ----------------------------------------------------------------------------------------

def test_estimate_counts_what_the_weights_hold(path, engine):
    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.geometry import split_weights

    transform = split_weights(split.rule, engine.w.plan)
    estimate = sum(transform(name, info)[0] for name, info in headers(path).items())
    assert engine.w.mtp is not None
    assert estimate == engine.w.nbytes()
    assert engine.capacity_plan["weight_bytes_estimate"] == estimate


def test_weights_count_the_latent_paths_kv_b(engine):
    """Each DSA layer's ``latent.AbsorbW`` (kv_b split per head) is in ``nbytes``, and is its only copy of kv_b: the
    key and value rows kv_k / kv_v, which only TF_GLM_LATENT=0 reads, are not held."""

    dsa = [L.dsa for L in engine.w.layers if L.dsa is not None] + [engine.w.mtp.layer.dsa]
    held = [a.absorb for a in dsa]
    assert all(isinstance(h, latent.AbsorbW) for h in held)
    assert all(a.kv_k is None and a.kv_v is None for a in dsa)
    full = engine.w.nbytes()
    try:
        for a in dsa:
            a.absorb = None
        bare = engine.w.nbytes()
    finally:
        for a, h in zip(dsa, held):
            a.absorb = h
    assert full - bare == sum(h.nbytes() for h in held) > 0
