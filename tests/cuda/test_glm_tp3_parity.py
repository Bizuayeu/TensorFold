"""GLM-5.3-Flash's layers cut over three ranks against the uncut layer: each rank's partial of a layer, from its own
share of the checkpoint, summed in rank order in fp32 as ``hc_post`` sums the gathered partials, held to a float64
reference of the whole layer and set beside the same layer cut over two ranks. On the tiny checkpoint of
``test_glm_tp3`` always; on nvidia/GLM-5.3-Flash-NVFP4 when TENSORFOLD_GLM_NVFP4 names its directory, a few real
layers of every rank on one GPU (``weights.load(layers=)``): 22/21/21 attention heads (``latent``'s 16-head window
tiles: two, the second masked; its 32-head prompt tiles: one, 10 or 11 heads masked), routed widths 704/704/640,
vocabulary 51,648/51,648/51,584.

Two references. Exact: the kernels' bf16 roundings at the same places (the MLPs, as test_glm_nvfp4 holds the tiny
ones; a prompt chunk's slot outputs are bf16 a rank, so that reference follows the cut), leaving only fp32 summation
order. Plain: float64 throughout, nothing rounded (every block, attention included): its error is the kernels' bf16
roundings and is held to be no larger on three ranks than on two. Heads stay apart up to the output projection, so
each head's attention output on three ranks is also held to the same head on two."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import _checkpoint  # noqa: E402
from test_glm_tp3 import SHAPE  # noqa: E402

from tensorfold.cuda.nvfp4 import experts as nvx  # noqa: E402
from tensorfold.cuda.nvfp4 import format as fmt  # noqa: E402
from tensorfold.engine.exact_sampling import MARGIN, Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import forward, kv8, sparse  # noqa: E402
from tensorfold.families.glm5_next.cuda.attention import CHUNK  # noqa: E402
from tensorfold.families.glm5_next.cuda.decode import sample_rows  # noqa: E402
from tensorfold.families.glm5_next.cuda.qmm import group_sums  # noqa: E402
from tensorfold.families.glm5_next.cuda.split import UNIT, RankReader, ShardPlan  # noqa: E402
from tensorfold.families.glm5_next.cuda.weights import PREFIX, Config, load  # noqa: E402

REAL = os.environ.get("TENSORFOLD_GLM_NVFP4", "")
LAYERS = {"tiny": [0, 1], "real": [0, 3, 4]}       # dense MLP + KDA, DSA + MoE (, KDA + MoE)
CAP = 2560                                         # cache rows: a 2,104-token context, past the dense limit
LONG = [(0, 2048, True), (2048, 2100, True), (2100, 2104, False)]   # prompt chunks, then a window: sparse rows
E2M1 = torch.tensor(fmt.E2M1, dtype=torch.float64)


class Model:
    """The checkpoint, every rank of three and of two (a few layers each), and the whole tensors for references."""

    def __init__(self, name: str, path: Path) -> None:
        self.name, self.path, self.cfg = name, path, Config.read(path)
        layers = LAYERS[name]
        self.ranks = {world: [load(path, rank=r, world=world, layers=layers) for r in range(world)]
                      for world in (3, 2)}
        self.reader = RankReader(path, ShardPlan(self.cfg, 1, 0))
        kinds = [self.cfg.kinds[i] for i in layers]
        self.at = {"dense": layers[0], "dsa": layers[kinds.index("dsa")],
                   "kda": layers[len(kinds) - 1 - kinds[::-1].index("kda")], "moe": layers[-1]}

    def full(self, name: str) -> torch.Tensor:
        """A whole checkpoint tensor (language-model names without their prefix) on the GPU, float64."""

        return self.raw(name).to("cuda", torch.float64)

    def raw(self, name: str) -> torch.Tensor:
        return self.reader.get(name if name.startswith("lm_head.") else PREFIX + name)

    def nvfp4(self, base: str) -> torch.Tensor:
        """[N, K] float64 of a stored NVFP4 linear: e2m1 code x e4m3 scale per 16 x the fp32 weight scale."""

        w = self.raw(base + ".weight").cuda()
        s = self.raw(base + ".weight_scale").cuda()
        s = s.view(torch.float8_e4m3fn) if s.dtype == torch.uint8 else s
        codes = torch.stack((w & 0xF, w >> 4), dim=-1).reshape(w.shape[0], -1).long()
        return E2M1.cuda()[codes] * s.double().repeat_interleave(16, dim=1) * float(self.raw(base + ".weight_scale_2"))

    def weight(self, base: str) -> torch.Tensor:
        """A projection's [N, K] float64: NVFP4 as dequantized, BF16 as stored."""

        if PREFIX + base + ".weight_scale" in self.reader.index:
            return self.nvfp4(base)
        return self.full(base + ".weight")


@pytest.fixture(scope="module", params=["tiny", "real"])
def model(request, tmp_path_factory):
    if request.param == "real":
        if not REAL:
            pytest.skip("TENSORFOLD_GLM_NVFP4: nvidia/GLM-5.3-Flash-NVFP4's directory")
        path = Path(REAL)
    else:
        path = tmp_path_factory.mktemp("glm_tp3_parity")
        _checkpoint(path, nvfp4=True, **SHAPE)
    m = Model(request.param, path)
    yield m
    m.reader.close()
    del m
    torch.cuda.empty_cache()


# -- running a block on every rank ------------------------------------------------------------------------------------

class _Capture:
    """One rank's all-gather: keeps what the rank sends (its fp32 partial), hands its own copy back in every slot."""

    def __init__(self, world: int) -> None:
        self.world, self.sent = world, []

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.sent.append(send.clone())
        n = send.numel()
        for r in range(self.world):
            recv.view(-1)[r * n:(r + 1) * n].copy_(send.reshape(-1))


def _each(ranks: list, fn) -> list[dict]:
    """fn(w) on every rank: its partials in call order (``sent``), what each output projection read (``o``: the
    attention output before it, per head) and fn's own result."""

    seen: list[torch.Tensor] = []
    inner = forward.out_proj

    def spy(w, b, x, q, xs, R, site):
        seen.append(x[:R].clone())
        return inner(w, b, x, q, xs, R, site)

    out = []
    forward.out_proj = spy
    try:
        for w in ranks:
            w.comm, seen[:] = _Capture(w.world), []
            got = fn(w)
            torch.cuda.synchronize()
            out.append({"sent": w.comm.sent, "o": list(seen), "got": got})
            w.comm = None
    finally:
        forward.out_proj = inner
    return out


def _summed(runs: list[dict], call: int, R: int) -> torch.Tensor:
    """The ranks' partials of one call added in rank order in fp32 (``hc_post``'s order), as float64 [R, D]."""

    total = runs[0]["sent"][call].float().clone()
    for run in runs[1:]:
        total += run["sent"][call].float()
    return total.view(R, -1).double()


def _heads(runs: list[dict], call: int, width: int) -> torch.Tensor:
    """Every rank's attention output of one call, heads in model order: float64 [R, heads, width]."""

    parts = [run["o"][call].double() for run in runs]
    return torch.cat([p.view(p.shape[0], -1, width) for p in parts], dim=1)


def _layer(w, i: int):
    return w.mtp.layer if i == w.cfg.layers else next(lw for lw in w.layers if lw.index == i)


def _feed(b, x: torch.Tensor) -> int:
    R = x.shape[0]
    b.normed[:R].copy_(x)
    group_sums(x, b.xs[:R])
    return R


def _x(rows: int, dims: int, seed: int) -> torch.Tensor:
    return torch.randn((rows, dims), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).cuda()


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).norm() / want.norm())


def _report(model, what: str, **values) -> None:
    text = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items())
    print(f"\nPARITY {model.name} {what} {text}", flush=True)


# -- references -------------------------------------------------------------------------------------------------------

def _bf16(v: torch.Tensor) -> torch.Tensor:
    return v.float().to(torch.bfloat16).double()


def _mlp(x, gate, up, down, limit: float, exact: bool) -> torch.Tensor:
    """SwiGLU MLP in float64; ``exact`` rounds where the kernels do (test_glm_nvfp4's _mlp_reference)."""

    g, u = x @ gate.t(), x @ up.t()
    if exact:
        g, u = _bf16(g), _bf16(u)
    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    a = _bf16(_bf16(g / (1 + torch.exp(-g))) * u) if exact else g / (1 + torch.exp(-g)) * u
    return a @ down.t()


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _kda_plain(m: Model, i: int, x: torch.Tensor) -> torch.Tensor:
    """KDA (kda.cu's definition) over rows x from a zero state, float64: each head's gated, normed read-out
    [rows, heads, 128] before the output projection."""

    c, p = m.cfg, f"layers.{i}.self_attn."
    X, N, H = x.double(), x.shape[0], c.lin_heads
    mixed = torch.cat([X @ m.full(p + f"{n}_proj.weight").t() for n in "qkv"], dim=1)
    conv = torch.cat([m.full(p + f"{n}_conv1d.weight").reshape(H * 128, -1) for n in "qkv"])
    taps = conv.shape[1]
    window = torch.cat([mixed.new_zeros((taps - 1, mixed.shape[1])), mixed])
    acc = sum(window[t:t + N] * conv[:, t] for t in range(taps))
    q, k, v = (acc * torch.sigmoid(acc)).view(N, 3, H, 128).unbind(1)
    q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
    k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    a = ((X @ m.full(p + "f_a_proj.weight").t()) @ m.full(p + "f_b_proj.weight").t()).view(N, H, 128)
    rate = torch.exp(m.full(p + "A_log").reshape(H, 1))
    g = torch.exp(c.lower * torch.sigmoid(rate * (a + m.full(p + "dt_bias").reshape(H, 128))))
    beta = torch.sigmoid(X @ m.full(p + "b_proj.weight").t())
    gate = ((X @ m.full(p + "g_a_proj.weight").t()) @ m.full(p + "g_b_proj.weight").t()).view(N, H, 128)
    s = X.new_zeros((H, 128, 128))                   # [head, value, key]
    ys = []
    for r in range(N):
        s = s * g[r][:, None, :]
        delta = (v[r] - (s * k[r][:, None, :]).sum(-1)) * beta[r][:, None]
        s = s + delta[:, :, None] * k[r][:, None, :]
        ys.append((s * q[r][:, None, :]).sum(-1))
    y = torch.stack(ys)
    return _rms(y, m.full(p + "o_norm.weight"), c.eps) * torch.sigmoid(gate)


def _dsa_plain(m: Model, i: int, x: torch.Tensor, rows: list[int], picks: dict[int, torch.Tensor]) -> torch.Tensor:
    """DSA over the latent of every row of x, float64: the attention output [len(rows), heads, v_dim] of the rows
    asked for, each over the keys up to it, or over the tokens ``picks`` names for it (the sparse rows)."""

    c, p = m.cfg, f"layers.{i}.self_attn."
    X = x.double()
    lat = _rms(X @ m.full(p + "kv_a_proj_with_mqa.weight").t(), m.full(p + "kv_a_layernorm.weight"), c.eps)
    qr = _rms(X[rows] @ m.full(p + "q_a_proj.weight").t(), m.full(p + "q_a_layernorm.weight"), c.eps)
    q = (qr @ m.full(p + "q_b_proj.weight").t()).view(len(rows), c.heads, c.qk_dim)
    kvb = m.full(p + "kv_b_proj.weight").view(c.heads, c.qk_dim + c.v_dim, -1)
    keys = torch.einsum("hdl,nl->nhd", kvb[:, :c.qk_dim], lat)
    vals = torch.einsum("hdl,nl->nhd", kvb[:, c.qk_dim:], lat)
    out = []
    for j, row in enumerate(rows):
        sel = picks.get(row)
        sel = torch.arange(row + 1, device=lat.device) if sel is None else sel.long()
        s = torch.einsum("hd,nhd->hn", q[j], keys[sel]) * c.qk_dim ** -0.5
        out.append(torch.einsum("hn,nhd->hd", torch.softmax(s, dim=1), vals[sel]))
    return torch.stack(out)


# -- (a) the dense MLP and the MoE ------------------------------------------------------------------------------------

def _mlp_run(w, i: int, x: torch.Tensor, prefill: bool):
    b = forward.Buffers(w, 64 if prefill else 8, prefill=prefill)
    R = _feed(b, x)
    layer = _layer(w, i)
    (forward.mlp_block if layer.mlp is not None else forward.moe_block)(layer, w, b, R)
    return b.pick[:R].clone(), b.wts[:R].clone()


@pytest.mark.parametrize("prefill", [False, True], ids=["window", "prompt"])
def test_dense_mlp_on_three_ranks_adds_up_to_the_layer(model, prefill):
    m, i, R = model, model.at["dense"], 5
    x = _x(R, m.cfg.hidden, 1)
    full = [m.weight(f"layers.{i}.mlp.{p}_proj") for p in ("gate", "up", "down")]
    exact, plain = (_mlp(x.double(), *full, m.cfg.limit, e) for e in (True, False))
    err = {}
    for world, ranks in m.ranks.items():
        got = _summed(_each(ranks, lambda w: _mlp_run(w, i, x, prefill)), 0, R)
        err[world] = (_rel(got, exact), _rel(got, plain))
    _report(m, f"dense-mlp L{i} {'prompt' if prefill else 'window'}", exact3=err[3][0], exact2=err[2][0],
            plain3=err[3][1], plain2=err[2][1])
    _explained(err)


def _explained(err: dict) -> None:
    """The exact reference accounts for the kernels' error: what is left (fp32 summation order, and an element whose
    fp32 value and the reference's fall either side of a bf16 rounding boundary, which then differ by a step) is under
    a tenth of the error against the plain reference. A missing, doubled or misplaced share, or a wrong scale, leaves
    both errors alike (a third of the layer and more). The tiny model leaves at most 1.4%; one flip spreads over a
    row through the next projection, so the bound is a ratio, not test_glm_nvfp4's 1e-5."""

    for world, (exact, plain) in err.items():
        assert exact < 0.1 * plain, (world, exact, plain)


def _experts(m: Model, i: int, world: int, rank: int):
    """Rank ``rank`` of ``world``'s gate/up rows and down columns of every expert, as float64 getters: routed
    experts the checkpoint's NVFP4 (the MTP layer's: that rank's own NVFP4 packing of its BF16 slice), the shared
    expert (id ``experts``) BF16."""

    c = m.cfg
    p = f"layers.{i}.mlp."
    mtp = i == c.layers

    def get(e: int):
        width = c.shared_width if e == c.experts else c.moe_width
        lo, hi = ShardPlan(c, world, rank).span(width, UNIT)
        if mtp and e < c.experts:
            ex = _layer(m.ranks[world][rank], i).moe.experts
            return tuple(nvx.dense(ex, e, n).cuda().double() for n in ("gate", "up", "down"))
        name = "shared_experts" if e == c.experts else f"experts.{e}"
        g, u, d = (m.weight(p + f"{name}.{n}_proj") for n in ("gate", "up", "down"))
        return g[lo:hi], u[lo:hi], d[:, lo:hi]

    return get


def _moe_want(m, i, x, pick, wts, world: int | None, prefill: bool) -> torch.Tensor:
    """The routed and shared experts weighted as routed, float64. ``world``: the exact reference of that cut (each
    rank's share rounded as the kernels round it, a prompt chunk's slot outputs bf16 a rank); None: the plain one
    (the whole checkpoint experts, MTP's BF16 ones, nothing rounded)."""

    c, xd = m.cfg, x.double()
    want = torch.zeros((x.shape[0], c.hidden), dtype=torch.float64, device="cuda")
    p = f"layers.{i}.mlp."
    for e in sorted(set(pick.flatten().tolist())):
        if world is None:
            name = "shared_experts" if e == c.experts else f"experts.{e}"
            shares = [tuple(m.weight(p + f"{name}.{n}_proj") for n in ("gate", "up", "down"))]
        else:
            shares = [_experts(m, i, world, r)(e) for r in range(world)]
        for r, s in zip(*torch.nonzero(pick == e, as_tuple=True)):
            for g, u, d in shares:
                y = _mlp(xd[r:r + 1], g, u, d, c.limit, world is not None)[0]
                want[r] += float(wts[r, s]) * (_bf16(y) if prefill and world is not None else y)
    return want


@pytest.mark.parametrize("prefill", [False, True], ids=["window", "prompt"])
@pytest.mark.parametrize("which", ["moe", "mtp"])
def test_moe_on_three_ranks_adds_up_to_the_layer(model, which, prefill):
    """Routing is replicated, so every rank of both cuts picks the same experts with the same weights."""

    m, R = model, 5
    i = m.cfg.layers if which == "mtp" else m.at["moe"]
    x = _x(R, m.cfg.hidden, 11)
    err, picks = {}, []
    for world, ranks in m.ranks.items():
        runs = _each(ranks, lambda w: _mlp_run(w, i, x, prefill))
        picks += [run["got"] for run in runs]
        pick, wts = runs[0]["got"]
        got = _summed(runs, 0, R)
        err[world] = (_rel(got, _moe_want(m, i, x, pick, wts, world, prefill)),
                      _rel(got, _moe_want(m, i, x, pick, wts, None, prefill)))
    assert all(torch.equal(a, picks[0][0]) and torch.equal(b, picks[0][1]) for a, b in picks)
    _report(m, f"{which} L{i} {'prompt' if prefill else 'window'}", exact3=err[3][0], exact2=err[2][0],
            plain3=err[3][1], plain2=err[2][1])
    _explained(err)


# -- (b) attention: KDA and DSA -----------------------------------------------------------------------------------

def _close_heads(heads: dict, want: torch.Tensor, spans: list[tuple[int, int]]) -> dict:
    """Each cut's per-head outputs against the plain reference, and three ranks' heads against two's: the largest
    difference among the heads a three-rank rank holds past its 16th (``spans``; the masked second tile of a window)
    and among the rest."""

    a, b = heads[3], heads[2]
    per_head = (a - b).abs().amax(dim=(0, 2))
    late = torch.zeros(per_head.shape[0], dtype=torch.bool, device=per_head.device)
    for lo, hi in spans:
        late[lo + 16:hi] = True
    return {"heads_plain3": _rel(a, want), "heads_plain2": _rel(b, want), "heads_3v2_max": float(per_head.max()),
            "past16_max": float(per_head[late].max()) if late.any() else 0.0,
            "first16_max": float(per_head[~late].max()), "heads_3v2_diff": int((a != b).sum()), "of": a.numel(),
            "heads_max": float(b.abs().max())}


def _same_size(e3: float, e2: float) -> bool:
    """Three ranks and two round different fp32 sums (each projection's K slices follow its shape) but as many of
    them, so their errors against the plain reference agree to within a tenth (measured: to three digits)."""

    return abs(e3 - e2) <= 0.1 * e2


def test_kda_on_three_ranks_adds_up_to_the_layer(model):
    """A 80-row prompt chunk (the wide chain) from a zero state, then a 5-row window from where it left off."""

    m, i = model, model.at["kda"]
    x = _x(85, m.cfg.hidden, 21)

    def run(w):
        st = forward.State(w, CAP, 8)
        b = forward.Buffers(w, 128, CAP, prefill=True)
        forward.kda_block(_layer(w, i), w, st, b, _feed(b, x[:80]))
        b = forward.Buffers(w, 8, CAP)
        forward.kda_block(_layer(w, i), w, st, b, _feed(b, x[80:]))

    o = _kda_plain(m, i, x)
    y = o.reshape(85, -1) @ m.full(f"layers.{i}.self_attn.o_proj.weight").t()
    heads, err = {}, {}
    for world, ranks in m.ranks.items():
        runs = _each(ranks, run)
        heads[world] = torch.cat([_heads(runs, 0, 128), _heads(runs, 1, 128)])
        got = torch.cat([_summed(runs, 0, 80), _summed(runs, 1, 5)])
        err[world] = _rel(got, y)
    found = _close_heads(heads, o, ShardPlan(m.cfg, 3, 0).spans(m.cfg.lin_heads))
    _report(m, f"kda L{i}", plain3=err[3], plain2=err[2], **found)
    assert _same_size(err[3], err[2]) and _same_size(found["heads_plain3"], found["heads_plain2"])


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
@pytest.mark.parametrize("which", ["dsa", "mtp"])
def test_dsa_on_three_ranks_adds_up_to_the_layer(model, which, kv):
    """2,104 rows: a 2,048-row prompt chunk (dense), a chunk across the dense limit (sparse from row 2,051), a
    4-row window (all sparse); every rank's indexer picks the same tokens, the reference attends to those."""

    m = model
    i = m.cfg.layers if which == "mtp" else m.at["dsa"]
    c = m.cfg
    x = _x(LONG[-1][1], c.hidden, 31)
    chosen: list = []
    inner = sparse.select_tokens

    def spy(*a, **k):
        tokens, counts = inner(*a, **k)
        chosen.append((tokens.clone(), counts.clone()))
        return tokens, counts

    def rows_of(n: int, width: int) -> torch.Tensor:
        if kv == "fp8":
            return torch.zeros((n, width + kv8.PAD), dtype=torch.uint8, device="cuda")
        return torch.zeros((n, width), dtype=torch.bfloat16, device="cuda")

    def run(w):
        lc = rows_of(CAP, c.kv_lora)
        index = (torch.zeros((CAP, c.index_dim), dtype=torch.bfloat16, device="cuda"),
                 torch.zeros((CAP, c.index_dim), dtype=torch.bfloat16, device="cuda"),
                 rows_of(CAP // 4 + 2, c.index_dim))
        pos = torch.zeros((1,), dtype=torch.int32, device="cuda")
        prompt, window = forward.Buffers(w, 2048, CAP, prefill=True), forward.Buffers(w, 8, CAP)
        for lo, hi, pre in LONG:
            b = prompt if pre else window
            pos.fill_(lo)
            forward.dsa_block(_layer(w, i), w, lc, None, pos, b, _feed(b, x[lo:hi]), -(-hi // CHUNK), index, lo)

    sparse.select_tokens = spy
    try:
        found = {world: _each(ranks, run) for world, ranks in m.ranks.items()}
    finally:
        sparse.select_tokens = inner
    calls = len(chosen) // 5
    assert calls == 2 and all(torch.equal(t, chosen[k % calls][0]) and torch.equal(n, chosen[k % calls][1])
                              for k, (t, n) in enumerate(chosen))       # both chunks with sparse rows, every rank
    picks = {}
    for (tokens, counts), (lo, hi, _) in zip(chosen[:calls], LONG[1:]):
        for j in range(hi - lo):
            if int(counts[j]):
                picks[lo + j] = tokens[j, :int(counts[j])]
    assert min(picks) == c.dense_limit and max(picks) == LONG[-1][1] - 1
    rows = list(range(0, 2048, 97)) + list(range(2040, LONG[-1][1]))
    o = _dsa_plain(m, i, x, rows, picks)
    y = o.reshape(len(rows), -1) @ m.full(f"layers.{i}.self_attn.o_proj.weight").t()
    heads, err = {}, {}
    for world, runs in found.items():
        every = torch.cat([_heads(runs, k, c.v_dim) for k in range(3)])
        got = torch.cat([_summed(runs, k, hi - lo) for k, (lo, hi, _) in enumerate(LONG)])
        heads[world], err[world] = every[rows], _rel(got[rows], y)
    close = _close_heads(heads, o, ShardPlan(c, 3, 0).spans(c.heads))
    _report(m, f"{which} L{i} kv={kv} sparse_rows={len(picks)}", plain3=err[3], plain2=err[2], **close)
    assert _same_size(err[3], err[2]) and _same_size(close["heads_plain3"], close["heads_plain2"])


# -- (c) the head and the vocabulary ----------------------------------------------------------------------------------

def _sample(ranks, logits: list[torch.Tensor], sampling) -> list[list[int]]:
    """``sample_rows`` on every rank: a first pass keeps what each rank sends, the second hands every rank them all."""

    sends: list[torch.Tensor] = []

    class Keep:
        def __init__(self, world: int) -> None:
            self.world = world

        def all_gather(self, send, recv):
            sends.append(send.clone())
            recv.zero_()

    class Give(Keep):
        def all_gather(self, send, recv):
            recv.copy_(torch.cat(sends))

    out = []
    for comm in (Keep, Give):
        out = []
        for w, lg in zip(ranks, logits):
            w.comm = comm(w.world)
            out.append(sample_rows(w, lg, list(range(lg.shape[0])), sampling))
            w.comm = None
    return out


def test_head_puts_every_id_in_its_place(model):
    """Each rank's logits over its vocabulary span, put side by side, against a float64 head (two bf16 half-steps
    of the largest logit, test_glm_tp3's tolerance); greedy and a top-k past the narrowest shard (rank 2 pads its
    candidates with -inf) choose alike on every rank and on both cuts."""

    m, R = model, 4
    x = _x(R, m.cfg.hidden, 41)
    head = m.raw("lm_head.weight").cuda()
    want = torch.cat([x.double() @ head[a:a + 16384].double().t() for a in range(0, head.shape[0], 16384)], dim=1)
    del head
    tol = 2 * 2.0 ** -8 * float(want.abs().max())
    logits, err, chosen = {}, {}, {}
    narrow = min(hi - lo for lo, hi in m.ranks[3][0].vocab_spans)
    samplings = {"greedy": None, "top_k past a shard": Sampling(77, 1.0, narrow - MARGIN + 16, 0.98)}
    for world, ranks in m.ranks.items():
        parts = []
        for w in ranks:
            b = forward.Buffers(w, 8)
            parts.append(forward.mm(b, x, w.head, group_sums(x), b.logits[:R]).clone())
        logits[world] = torch.cat(parts, dim=1).double()
        assert [p.shape[1] for p in parts] == [hi - lo for lo, hi in ranks[0].vocab_spans]
        err[world] = float((logits[world] - want).abs().max())
        for name, sampling in samplings.items():
            picks = _sample(ranks, parts, sampling)
            assert all(p == picks[0] for p in picks), (world, name)
            chosen[(world, name)] = picks[0]
    _report(m, "head", max3=err[3], max2=err[2], tol=tol, diff_3v2=int((logits[3] != logits[2]).sum()),
            greedy=chosen[(3, "greedy")], fp64_argmax=want.argmax(1).tolist())
    assert err[3] <= tol and err[2] <= tol
    for name in samplings:
        assert chosen[(3, name)] == chosen[(2, name)], name
    assert chosen[(3, "greedy")] == logits[3].argmax(1).tolist()


def test_reference_dequant_is_the_formats(model):
    """The references' NVFP4 decode (on the GPU) is ``format.dequant``'s."""

    base = f"layers.{model.at['dense']}.mlp.down_proj"
    want = fmt.dequant("nvfp4", model.raw(base + ".weight").numpy(),
                       model.raw(base + ".weight_scale").view(torch.uint8).numpy(),
                       float(model.raw(base + ".weight_scale_2")))
    assert np.array_equal(model.nvfp4(base).float().cpu().numpy(), want)       # one fp32 rounding, as numpy's


# -- (d) the kernels on a rank's 22 or 21 heads -----------------------------------------------------------------------

SLICES = [(22, 0), (21, 0), (21, 11)]          # a rank's heads among 32: 22 or 21 of them, at the start or past it


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
@pytest.mark.parametrize("R", [4, 72], ids=["window", "prompt"])
def test_latent_attention_gives_22_and_21_heads_the_bits_they_get_among_32(R, kv):
    """DSA's latent path on three ranks of GLM-5.3-Flash, at its sizes (latent 512, heads 256 wide, a 2,104-token
    context): absorb, dense and sparse attention and expand over 22 or 21 heads (a window's two 16-head tiles, the
    second masked; a prompt chunk's one 32-head tile, 10 or 11 heads masked) give each head the bits it gets among 32
    heads (whole tiles)."""

    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(R)
    L, D, n, W = 512, 256, 2104, 2051
    lat = torch.randn((n, L), generator=gen).to(torch.bfloat16).cuda()
    if kv == "fp8":
        cache = torch.zeros((n, L + kv8.PAD), dtype=torch.uint8, device="cuda")
    else:
        cache = torch.zeros((n, L), dtype=torch.bfloat16, device="cuda")
    latent.latent_write(lat, cache, torch.zeros((1,), dtype=torch.int32, device="cuda"))
    wk, wv = (torch.randn((32, D, L), generator=gen) * L ** -0.5 for _ in range(2))
    q = torch.randn((R, 32, D), generator=gen).to(torch.bfloat16).cuda()
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = []
    for r in range(R):
        sel = torch.sort(torch.randperm(n - R, generator=gen)[:2048 - r % 4]).values
        tokens[r, :sel.numel()] = sel.to(torch.int32)
        counts.append(sel.numel())
    tokens, counts = tokens.cuda(), torch.tensor(counts, dtype=torch.int32, device="cuda")
    pos = torch.tensor([n - R], dtype=torch.int32, device="cuda")

    def run(H: int, first: int) -> list[torch.Tensor]:
        a = latent.AbsorbW(wk[first:first + H].cuda(), wv[first:first + H].cuda())
        qa = latent.absorb_q(q[:, first:first + H].contiguous(), a,
                             torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda"))
        s = latent.LatentScratch(R, H, latent.chunks_for(n), "cuda")
        dense = latent.attention(qa, cache, pos, s, scale=D ** -0.5, nch=latent.chunks_for(n),
                                 out=torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda")).clone()
        picked = torch.zeros((R, H, L), dtype=torch.bfloat16, device="cuda")
        latent.sparse_attention(qa, cache, tokens, counts, picked, D ** -0.5)
        expanded = [latent.expand_v(o, a, torch.empty((R, H, D), dtype=torch.bfloat16, device="cuda")).clone()
                    for o in (dense, picked)]
        return [qa.clone(), dense, picked, *expanded]

    every = run(32, 0)
    for H, first in SLICES:
        got = run(H, first)
        assert all(torch.equal(g, w[:, first:first + H]) for g, w in zip(got, every)), (H, first)


@pytest.mark.parametrize("R", [5, 72], ids=["window", "prompt"])
def test_kda_chain_gives_22_and_21_heads_the_bits_they_get_among_32(R):
    """KDA on three ranks of GLM-5.3-Flash (a block a head): 22 or 21 heads' read-outs and states, the short chain
    and the wide one, are the bits those heads get among 32."""

    from tensorfold.families.glm5_next.cuda import kda

    gen = torch.Generator().manual_seed(R)
    C = 32 * 128

    def randn(*shape, scale=1.0):
        return torch.randn(shape, generator=gen) * scale

    q, k, v, A, G = (randn(R, C, scale=0.5).to(torch.bfloat16).cuda() for _ in range(5))
    fa, ga = (randn(R, 128).to(torch.bfloat16).cuda() for _ in range(2))
    b = randn(R, 32).to(torch.bfloat16).cuda()
    cw = randn(3 * C, 4, scale=0.3).to(torch.bfloat16).float().cuda()        # q | k | v channels, 4 taps
    cs = randn(3, 3 * C, scale=0.5).to(torch.bfloat16).cuda()                # the conv window: 3 rows of q | k | v
    st = randn(32, 128, 128, scale=0.05).cuda()
    a_log, dt = randn(32).cuda(), randn(C).cuda()
    nw = (1 + 0.1 * randn(128)).to(torch.bfloat16).cuda()

    def run(H: int, first: int) -> tuple[torch.Tensor, torch.Tensor]:
        hs = slice(first * 128, (first + H) * 128)
        taps = torch.cat([cw[j * C:(j + 1) * C][hs] for j in range(3)]).contiguous()
        window = torch.cat([cs[:, j * C:(j + 1) * C][:, hs] for j in range(3)], dim=1).contiguous()
        p = torch.cat([q[:, hs], k[:, hs], v[:, hs], fa, ga, b[:, first:first + H]], dim=1).contiguous()
        state = torch.empty((H, 128, 128), dtype=torch.float32, device="cuda")
        out = kda.chain(p, 3 * H * 128 + 256, A[:, hs].contiguous(), G[:, hs].contiguous(), window, taps,
                        st[first:first + H].contiguous(), a_log[first:first + H].contiguous(), dt[hs].contiguous(),
                        nw, 1e-5, -5.0, R, kda.KDAScratch(R, H, "cuda"), state)
        return out.clone(), state

    out, state = run(32, 0)
    for H, first in SLICES:
        got, got_state = run(H, first)
        assert torch.equal(got, out[:, first * 128:(first + H) * 128]), (H, first)
        assert torch.equal(got_state, state[first:first + H]), (H, first)
