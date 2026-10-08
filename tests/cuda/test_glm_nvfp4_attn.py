"""GLM-5.3-Flash with its attention projections and lm_head in W4A16 NVFP4 (Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-
W4A16's layout) on the tiny synthetic model: each projection that stores a ``.weight_scale`` loads as ``Fp4Linear``
(KDA's six input projections and DSA's q_a and kv_a stacked under their one weight scale, the stack's rows zero-padded
to a multiple of 8 so every row of a window starts 16-byte aligned), kv_b dequantized for the latent path, the head
drafting itself. Its twin is the same checkpoint with those projections BF16 of their dequantized values (exact: the
synthetic weight scales are powers of two), so the two differ only by their matmul kernels: the ranks' sums (two
ranks, and three with rank shapes whose stacks need the padding) give the twin's rows within bf16 rounding, and the
engine keeps its contract (drafted replies equal serial ones with both drafters, CUDA graphs equal eager steps, a
resumed prompt equals a fresh prefill, prompt chunking leaves the same state); the startup estimate counts what the
weights hold, and the ranks refuse to serve different checkpoints."""

from __future__ import annotations

import json
import shutil
import struct

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import D, _checkpoint, _drafter, _forget, _generate, _state, _TwoCopies  # noqa: E402
from test_glm_tp3 import LONG, PROMPT, SHAPE, Ranks, _tolerance  # noqa: E402
from threadcomm import run_ranks  # noqa: E402

from tensorfold.cuda.nvfp4 import format as fmt  # noqa: E402
from tensorfold.cuda.nvfp4.linear import ROW_ALIGN, Fp4Linear  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import latent, split  # noqa: E402
from tensorfold.families.glm5_next.cuda.qmm import B16  # noqa: E402

L = "model.language_model."
FILE = "model-00001-of-00001.safetensors"


def _twin(src, dst) -> None:
    """``src`` with every NVFP4 projection outside the MLPs as BF16 of its dequantized values (asserted exact) under
    nvidia/GLM-5.3-Flash-NVFP4's config: the pinned checkpoint's layout holding the same numbers."""

    header, base = split.read_header(src / FILE)
    header.pop("__metadata__", None)
    raw = np.fromfile(src / FILE, dtype=np.uint8)

    def stored(name: str) -> np.ndarray:
        a, b = header[name]["data_offsets"]
        return raw[base + a:base + b]

    tensors = []
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        info = header[name]
        outside = ".mlp." not in name
        if outside and name.endswith((".weight_scale", ".weight_scale_2")):
            continue
        if outside and info["dtype"] == "U8":
            p = name[:-len(".weight")]
            n, half = info["shape"]
            w = fmt.dequant("nvfp4", stored(name).reshape(n, half), stored(p + ".weight_scale").reshape(n, half // 8),
                            float(stored(p + ".weight_scale_2").view(np.float32)[0]))
            bf = torch.from_numpy(w).to(torch.bfloat16)
            assert torch.equal(bf.float(), torch.from_numpy(w)), name
            tensors.append((name, "BF16", [n, 2 * half], bf.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))
            continue
        tensors.append((name, info["dtype"], info["shape"], stored(name)))
    dst.mkdir(parents=True, exist_ok=True)
    split.write(str(dst / FILE), tensors, {"format": "mlx"})
    config = json.loads((src / "config.json").read_text())
    q = config["quantization_config"]
    q["quant_algo"] = "NVFP4"
    del q["quantized_layers"]
    (dst / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("glm_attn")
    out = {}
    for key, shape in (("two", {}), ("three", SHAPE)):
        _checkpoint(root / key, nvfp4=True, attention=True, **shape)
        _twin(root / key, root / f"{key}_twin")
        out[key], out[f"{key}_twin"] = root / key, root / f"{key}_twin"
    _drafter(root / "dflash2")
    out["dflash2"] = root / "dflash2"
    return out


@pytest.fixture(scope="module")
def ranks(paths):
    from tensorfold.families.glm5_next.cuda.weights import load

    return [load(paths["two"], rank=r) for r in (0, 1)]


@pytest.fixture(scope="module")
def twins(paths):
    from tensorfold.families.glm5_next.cuda.weights import load

    return [load(paths["two_twin"], rank=r) for r in (0, 1)]


@pytest.fixture(scope="module")
def engine(paths):
    """Rank 0 of two standing in for both, with the DFlash2 drafter, whose head is the model's NVFP4 head."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    return GlmEngine(paths["two"], rank=0, master="", port=0, drafter=paths["dflash2"], comm=_TwoCopies())


# -- (a) what each rank holds ----------------------------------------------------------------------------------------

def test_attention_and_the_head_load_as_nvfp4(paths, ranks):
    from tensorfold.families.glm5_next.cuda.weights import Config

    assert Config.read(paths["two"]).quant == "nvfp4"
    for w in ranks:
        k, a = w.layers[0].kda, w.layers[1].dsa
        assert all(isinstance(x, Fp4Linear) for x in (k.proj, k.fb, k.gb, k.o, a.proj, a.q_b, a.o, a.index.qb, w.head))
        assert k.b_off + k.heads == 641 and k.proj.n == 648            # q k v f_a g_a b of one head, padded to 8
        assert isinstance(a.index.kw, B16) and isinstance(a.absorb, latent.AbsorbW) and a.kv_k is None
        assert w.draft_head is None                                    # the NVFP4 head drafts itself
        m = w.mtp.layer.dsa                                            # the MTP layer stays BF16
        assert isinstance(m.proj, B16) and isinstance(m.q_b, B16) and isinstance(w.mtp.eh, B16)


def _pairs(w, t) -> dict:
    k, kt = w.layers[0].kda, t.layers[0].kda
    a, at = w.layers[1].dsa, t.layers[1].dsa
    return {"kda proj": (k.proj, kt.proj), "f_b": (k.fb, kt.fb), "g_b": (k.gb, kt.gb), "kda o": (k.o, kt.o),
            "dsa proj": (a.proj, at.proj), "q_b": (a.q_b, at.q_b), "dsa o": (a.o, at.o),
            "wq_b": (a.index.qb, at.index.qb), "head": (w.head, t.head)}


@pytest.mark.parametrize("world", [2, 3])
def test_projections_hold_the_dequantized_values(paths, world):
    """Each rank's NVFP4 projections times the identity, on the prompt GEMM and the lane matmul: its twin's BF16
    weights, the stack's padding rows zero; kv_b absorbed as the twin's."""

    from tensorfold.families.glm5_next.cuda.weights import load

    key = {2: "two", 3: "three"}[world]
    for rank in range(world):
        w, t = (load(paths[k], rank=rank, world=world, mtp=False) for k in (key, f"{key}_twin"))
        for what, (lin, ref) in _pairs(w, t).items():
            assert lin.k == ref.k and lin.n == -(-ref.n // ROW_ALIGN) * ROW_ALIGN, what
            want = ref.weight.float().t()
            eye = torch.eye(lin.k, dtype=torch.bfloat16, device="cuda")
            for got in (lin.prefill(eye, f32=True), lin(eye[:8], f32=True)):
                torch.testing.assert_close(got[:, :ref.n], want[:got.shape[0]], rtol=2 ** -20, atol=0,
                                           msg=f"{what} rank {rank}")
                assert not got[:, ref.n:].any(), what
        a, at = w.layers[1].dsa.absorb, t.layers[1].dsa.absorb
        assert torch.equal(a.wk, at.wk) and torch.equal(a.wv, at.wv), rank
        del w, t
    torch.cuda.empty_cache()


def test_dequant_follows_the_format():
    """``latent.dequant_nvfp4``: codes (low nibble first) x e4m3 scale, then x the weight scale, in fp32: numpy's."""

    rng = np.random.default_rng(5)
    codes = rng.integers(0, 256, size=(96, 64), dtype=np.uint8)
    scales = rng.integers(0, 0x7F, size=(96, 8), dtype=np.uint8) | rng.choice([0, 0x80], size=(96, 8)).astype(np.uint8)
    for g in (0.0123, 2.0 ** -9, 3.7e-4):
        want = fmt.dequant("nvfp4", codes, scales, g)
        got = latent.dequant_nvfp4(torch.from_numpy(codes).cuda(), torch.from_numpy(scales).cuda(), g)
        assert got.dtype == torch.float32 and np.array_equal(got.cpu().numpy(), want), g


def test_the_per_head_path_reads_kv_b_as_nvfp4(paths, twins, monkeypatch):
    """TF_GLM_LATENT=0: kv_b's key and value rows as ``Fp4Linear``, the twin's rows."""

    from tensorfold.families.glm5_next.cuda.weights import load

    monkeypatch.setattr(latent, "ENABLED", False)
    w = load(paths["two"], rank=0, layers=[1], mtp=False)
    a, t = w.layers[0].dsa, twins[0].layers[1].dsa.absorb
    assert a.absorb is None
    for lin, ref in ((a.kv_k, t.wk), (a.kv_v, t.wv)):
        assert isinstance(lin, Fp4Linear)
        want = ref.reshape(-1, ref.shape[-1]).float().t()
        got = lin.prefill(torch.eye(lin.k, dtype=torch.bfloat16, device="cuda"), f32=True)
        torch.testing.assert_close(got, want, rtol=2 ** -20, atol=0)


def test_a_stack_of_unequal_weight_scales_is_refused(paths, tmp_path):
    """KDA's six input projections become one matmul under one weight scale: a checkpoint whose k_proj has another
    is refused by name, not served with the wrong scale."""

    from tensorfold.families.glm5_next.cuda.weights import load

    odd = tmp_path / "odd"
    shutil.copytree(paths["two"], odd)
    header, base = split.read_header(odd / FILE)
    a, _ = header[f"{L}layers.0.self_attn.k_proj.weight_scale_2"]["data_offsets"]
    with open(odd / FILE, "r+b") as f:
        f.seek(base + a)
        (v,) = struct.unpack("<f", f.read(4))
        f.seek(base + a)
        f.write(struct.pack("<f", 2 * v))
    with pytest.raises(ValueError, match="weight_scale_2"):
        load(odd, rank=0, layers=[0], mtp=False)


# -- (b) the ranks' sums against the twin ----------------------------------------------------------------------------

def _rows(rk: Ranks, prompt, windows) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Every rank's (rows, the MoE layer's expert picks of each row): its final-normed prompt rows (one chunk, before
    the MTP head's absorb reuses them: ``_logits`` of test_glm_tp3), then after a prefill its logits (its vocabulary
    share) of each decode window, each window kept."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill
    from tensorfold.families.glm5_next.cuda.forward import chunks_for, commit, compute, stage

    def fn(r, e):
        n = len(prompt)
        d = Engine(e.w, capacity=2560, max_rows=8, prefill_rows=n)
        d.reset()
        compute(e.w, d.st, d.pbuf, stage(e.w, d.st, d.pbuf, prompt), nch=chunks_for(d.st, n), host_pos=0)
        out = [(d.pbuf.fnormed[:n].float().clone(), d.pbuf.pick[:n].clone())]
        prefill(d, prompt, None)
        for ids in windows:
            logits = d.forward(ids).float().clone()
            out.append((logits, d.buf.pick[:len(ids)].clone()))
            commit(e.w, d.st, d.buf, len(ids), len(ids))
        del d
        return out

    return rk.each(fn)


@pytest.mark.parametrize("world", [2, 3])
def test_the_ranks_sums_give_the_twins_rows(paths, world):
    """The ranks' prompt rows (the prompt GEMM) and decode windows of 1 and 4 rows (the lane matmul) against the
    twin's: within two bf16 half-steps a side of each value's size (``test_glm_tp3._tolerance``) on every row both
    route to the same experts. The two sum in different orders, so a router's near tie may break apart (measured:
    one prompt row of 45 at three ranks, weights 1.101 and 1.099); the rows compared are most of them."""

    key = {2: "two", 3: "three"}[world]
    rng = np.random.default_rng(40 + world)
    prompt = [int(t) for t in rng.integers(0, 1000, size=45)]
    windows = [[int(rng.integers(0, 1000))], [int(t) for t in rng.integers(0, 1000, size=4)]]
    got = _rows(Ranks(paths[key], world), prompt, windows)
    want = _rows(Ranks(paths[f"{key}_twin"], world), prompt, windows)
    assert all(torch.equal(x[0][0], got[0][0][0]) for x in got)       # every rank glues the same rows
    over, compared, total = [], 0, 0
    for r, (a, b) in enumerate(zip(got, want)):
        for what, (x, px), (y, py) in zip(("prompt rows", "1 row", "4 rows"), a, b):
            same = (px == py).all(dim=1)
            apart, tol = float((x[same] - y[same]).abs().max()) if same.any() else 0.0, _tolerance(y)
            print(f"\n[attn nvfp4 vs twin, {world} ranks] rank {r} {what}: max |diff| {apart:.4g} on {int(same.sum())} "
                  f"of {len(same)} rows routed alike, tolerance {tol:.4g}")
            compared, total = compared + int(same.sum()), total + len(same)
            if apart > tol:
                over.append((r, what, apart, tol))
    assert not over and 2 * compared > total
    torch.cuda.empty_cache()


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_three_rank_windows_never_depend_on_their_rows(paths, rank):
    """Rank shapes of three (stacks of 1,026 and 641 rows, padded to 1,032 and 648): KDA's projections of a 4-row
    window, f_b and g_b reading their columns of it in place, give each row's bits alone."""

    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(paths["three"], rank=rank, world=3, layers=[0], mtp=False)
    k = w.layers[0].kda
    assert k.proj.n % ROW_ALIGN == 0 and 0 <= k.proj.n - (k.b_off + k.heads) < ROW_ALIGN
    x = torch.randn((4, D), generator=torch.Generator().manual_seed(rank)).to(torch.bfloat16).cuda()
    p = k.proj(x)
    assert torch.equal(p, torch.cat([k.proj(x[i:i + 1]) for i in range(4)]))
    for lin, off in ((k.fb, k.fa_off), (k.gb, k.ga_off)):
        cols = p[:, off:off + 128]
        want = torch.cat([lin(cols[i:i + 1].contiguous()) for i in range(4)])
        assert torch.equal(lin(cols), want) and torch.equal(lin.prefill(cols), lin.prefill(cols.contiguous()))
    del w
    torch.cuda.empty_cache()


# -- (c) the engine's contract ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("sampling", [Sampling(4321, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafted_replies_equal_serial(engine, sampling):
    """Every policy, MTP drafts and DFlash2's (both read the NVFP4 head)."""

    prompt = list(np.random.default_rng(8).integers(0, 1000, size=45))
    serial, stats = _generate(engine, prompt, sampling, draft=False, tokens=40)
    assert stats["drafts"] is False
    seen = set()
    for policy in (None, "auto:1:1:0", "f3", "fc5:0.3", "1", "2", "3", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(engine, prompt, sampling, policy=policy, tokens=40)
        assert drafted == serial, policy
        seen.update(stats.get("drafters", ""))
    assert seen == {"m", "f"}


def test_graphs_equal_eager_steps(engine):
    sampling = Sampling(5, 1.0, 20, 0.95)
    prompt = list(np.random.default_rng(12).integers(0, 1000, size=50))
    before = dict(engine.e.replays)
    with_graphs, _ = _generate(engine, prompt, sampling, policy="2", tokens=32)
    assert engine.e.replays["main"] > before["main"] and engine.e.replays["mtp"] > before["mtp"]
    graphs, engine.e.graphs = engine.e.graphs, None
    try:
        eager, _ = _generate(engine, prompt, sampling, policy="2", tokens=32)
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
    """A 300-row prompt in 2,048-, 100-, 16- and 7-row chunks: the same first token and bit-identical states."""

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


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("context", [None, LONG], ids=["dense", "long"])
def test_three_rank_shapes_capture_their_graphs(paths, rank, context):
    """One rank standing in for three on rank 0's and rank 1's shapes (both stacks padded): decode windows captured
    as CUDA graphs, and drafted replies equal serial ones through them, past the dense limit too (the indexer's
    NVFP4 wq_b)."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    e = GlmEngine(paths["three"], rank=rank, master="", port=0, world=3, comm=_TwoCopies(3, rank),
                  **({"context": context} if context else {}))
    assert e.e.graphs is not None and e.e.graphs.main
    prompt = list(np.random.default_rng(11).integers(0, 1000, size=PROMPT[context]))
    for sampling in (None, Sampling(77, 1.0, 400, 0.98)):
        _forget(e)
        serial, _ = _generate(e, prompt, sampling, draft=False, tokens=32)
        for policy in (None, "2", "c3:0.35"):
            _forget(e)
            drafted, _ = _generate(e, prompt, sampling, policy=policy, tokens=32)
            assert drafted == serial, (sampling, policy)
    del e
    torch.cuda.empty_cache()


# -- (d) the startup estimate and agreement --------------------------------------------------------------------------

@pytest.mark.parametrize("world", [2, 3])
def test_each_rank_holds_its_startup_estimate(paths, world):
    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.glm5_next.cuda.weights import load

    path = paths[{2: "two", 3: "three"}[world]]
    for rank in range(world):
        w = load(path, rank=rank, world=world)
        transform = split_weights(split.rule, w.plan)
        assert sum(transform(name, info)[0] for name, info in headers(path).items()) == w.nbytes(), rank
        del w
    torch.cuda.empty_cache()


@pytest.mark.parametrize("world", [2, 3])
def test_rank_folders_load_as_the_checkpoint(paths, tmp_path, world):
    """Each rank's folder written by the split (codes and scales cut alike, weight scales and the whole head kept) loads
    what the checkpoint read in place loads, and counts the same NVFP4 projections for the startup agreement."""

    from test_glm_nvfp4 import _tensors

    from tensorfold.cuda.capacity import headers
    from tensorfold.families.glm5_next import nvfp4_attention
    from tensorfold.families.glm5_next.cuda.weights import load

    path = paths[{2: "two", 3: "three"}[world]]
    for rank in range(world):
        out = tmp_path / f"rank{rank}"
        split.main([str(path), "--world", str(world), "--rank", str(rank), str(out)])
        assert nvfp4_attention(headers(out, rank=rank)) == nvfp4_attention(headers(path)) == 16
        a, b = (_tensors(load(p, rank=rank, world=world)) for p in (path, out))
        assert a and len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b)), rank
        del a, b
    torch.cuda.empty_cache()


def test_a_rank_serving_another_checkpoint_is_named(paths):
    """Rank 0 on the NVFP4 attention checkpoint (16 NVFP4 projections outside the MLPs: the head, KDA's nine, DSA's
    six), rank 1 on its BF16 twin: refused at startup."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    def start(r, comm):
        GlmEngine(paths["two" if r == 0 else "two_twin"], rank=r, master="", port=0, world=2, comm=comm,
                  graphs=False)

    with pytest.raises(RuntimeError, match=r"different settings.*NVFP4 attention.*rank 0 \[[^]]*, 16\], "
                                           r"rank 1 \[[^]]*, 0\]"):
        run_ranks(start, 2)
