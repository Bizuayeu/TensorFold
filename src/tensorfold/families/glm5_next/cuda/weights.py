"""Load one rank of MLX affine 4-bit weights, or EXL3 or ModelOpt NVFP4 routed experts with BF16 elsewhere, preserving heads and quantization groups at split boundaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tensorfold.cuda import experts as grouped

from tensorfold.cuda.exl3.experts import Exl3RoutedExperts as Exl3Experts
from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4.linear import Fp4Linear
from .. import modelopt_nvfp4
from . import latent
from .qmm import B16, Q4, as_i32, make_b16, make_q4, quantize4, stack_b16, stack_q4
from .split import UNIT

PREFIX = "model.language_model."


def bits_of(quant: dict) -> int:
    """The checkpoint's one bit width; a mixed-bit encode (bits such as "mixed_k34_per_tensor") is refused by name."""

    bits = quant.get("bits", 4)
    if isinstance(bits, int) or (isinstance(bits, str) and bits.isdigit()):
        return int(bits)
    raise ValueError(f"this checkpoint's quantization bits are {bits!r}: GLM-5.3 on CUDA reads one bit width a "
                     "checkpoint, so mixed-bit EXL3 encodes are not supported yet")


@dataclass
class Config:
    hidden: int
    layers: int
    vocab: int
    eps: float
    heads: int
    q_lora: int
    kv_lora: int
    qk_dim: int
    v_dim: int
    lin_heads: int
    lin_dim: int
    conv: int
    lower: float
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    dense_width: int
    routed_scale: float
    norm_topk: bool
    streams: int
    hc_iters: int
    hc_eps: float
    index_heads: int
    index_dim: int
    index_topk: int
    kpool: int
    limit: float
    kinds: list[str]           # per layer: "kda" or "dsa"
    mlp_kinds: list[str]       # per layer: "dense" or "moe"
    eos: tuple[int, ...]
    mtp_layers: int
    group_size: int
    bits: int
    # "mlx" (affine 4-bit everywhere), "exl3" (EXL3 routed experts, BF16 elsewhere) or "nvfp4" (ModelOpt NVFP4 routed
    # experts and dense MLPs, BF16 elsewhere)
    quant: str = "mlx"

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        want = raw.get("tensorfold_activation_dtype")
        if want not in (None, "bfloat16", "float32"):
            raise ValueError(f"tensorfold_activation_dtype {want!r}: bfloat16 or float32")
        if want == "float32":
            raise ValueError("tensorfold_activation_dtype float32 is the Mac engine; the CUDA engine stays bf16")
        t = dict(raw.get("text_config") or raw)
        lin = dict(t.get("linear_attn_config") or {})
        quant = raw.get("quantization") or raw.get("quantization_config") or {}
        eos = t.get("eos_token_id", raw.get("eos_token_id"))
        eos = tuple(int(e) for e in eos) if isinstance(eos, list) else (int(eos),)
        n = int(t["num_hidden_layers"])
        kinds = ["kda" if k == "linear_attention" else "dsa" for k in t["layer_types"]]
        dense = int(t.get("first_k_dense_replace", 3))
        mlp_kinds = list(t.get("mlp_layer_types") or ["dense"] * dense + ["sparse"] * (n - dense))
        mlp_kinds = ["moe" if k == "sparse" else "dense" for k in mlp_kinds]
        method = str(quant.get("quant_method") or "mlx").lower()
        if method == "modelopt":
            algo = str(quant.get("quant_algo") or "").lower()
            method = "nvfp4" if modelopt_nvfp4(quant) else f"modelopt {algo or 'without quant_algo'}"
        return cls(
            hidden=int(t["hidden_size"]), layers=n, vocab=int(t["vocab_size"]), eps=float(t["rms_norm_eps"]),
            heads=int(t["num_attention_heads"]), q_lora=int(t["q_lora_rank"]), kv_lora=int(t["kv_lora_rank"]),
            qk_dim=int(t["qk_nope_head_dim"]) + int(t.get("qk_rope_head_dim", 0)), v_dim=int(t["v_head_dim"]),
            lin_heads=int(lin.get("num_heads", t.get("linear_num_heads", 64))),
            lin_dim=int(lin.get("head_dim", t.get("linear_head_dim", 128))),
            conv=int(lin.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4))),
            lower=float(lin.get("gate_lower_bound", t.get("linear_lower_bound", -5.0))),
            experts=int(t["n_routed_experts"]), top_k=int(t["num_experts_per_tok"]),
            moe_width=int(t["moe_intermediate_size"]),
            shared_width=int(t["moe_intermediate_size"]) * int(t.get("n_shared_experts", 1)),
            dense_width=int(t["intermediate_size"]), routed_scale=float(t["routed_scaling_factor"]),
            norm_topk=bool(t.get("norm_topk_prob", True)), streams=int(t.get("hc_mult", 4)),
            hc_iters=int(t.get("hc_sinkhorn_iters", 20)), hc_eps=float(t.get("hc_eps", 1e-6)),
            index_heads=int(t.get("index_n_heads", 32)), index_dim=int(t.get("index_head_dim", 128)),
            index_topk=int(t.get("index_topk", 2048)), kpool=int(t.get("index_kpool", 4)),
            limit=float(t.get("swiglu_limit", 10.0)), kinds=kinds, mlp_kinds=mlp_kinds, eos=eos,
            mtp_layers=int(t.get("num_nextn_predict_layers", 0)), group_size=int(quant.get("group_size", 64)),
            bits=bits_of(quant), quant=method,
        )

    @property
    def dense_limit(self) -> int:
        """Largest context (tokens) where DSA's top-k selection keeps every visible key (dense attention)."""

        return self.index_topk + self.kpool - 1


@dataclass
class HCW:
    fn: torch.Tensor          # [24, S*D] bf16
    base: torch.Tensor        # [24] fp32
    scale: torch.Tensor       # [3] fp32


@dataclass
class KDAW:
    proj: Q4                  # [q | k | v | f_a | g_a | b]
    fb: Q4
    gb: Q4
    conv: torch.Tensor        # [3 HL 128, taps] bf16
    a_log: torch.Tensor       # [HL] fp32
    dt_bias: torch.Tensor     # [HL 128] fp32
    norm: torch.Tensor        # [128] bf16
    o: Q4
    heads: int

    @property
    def fa_off(self) -> int:
        return 3 * self.heads * 128

    @property
    def ga_off(self) -> int:
        return self.fa_off + 128

    @property
    def b_off(self) -> int:
        return self.ga_off + 128


@dataclass
class IndexW:
    """Replicated DSA indexer weights include key and query projections, head weights, key LayerNorm, pool gates, and pool position bias."""

    kw: Q4
    qb: Q4
    ln_w: torch.Tensor
    ln_b: torch.Tensor
    gate: torch.Tensor
    ape: torch.Tensor


@dataclass
class DSAW:
    proj: Q4                  # [q_a | kv_a]
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    q_b: Q4
    kv_k: Q4 | B16 | None     # key rows of kv_b for the local heads (TF_GLM_LATENT=0 only)
    kv_v: Q4 | B16 | None     # value rows (TF_GLM_LATENT=0 only)
    o: Q4
    heads: int
    index: IndexW | None = None
    absorb: object = None     # latent.AbsorbW: kv_b split per head, for attention on the latent cache


@dataclass
class MLPW:
    gu: Q4 | B16 | tuple[Fp4Linear, Fp4Linear]   # [gate | up]; NVFP4: gate and up, each with its own tensor scale
    down: Q4 | B16 | Fp4Linear
    width: int


@dataclass
class MoEW:
    router: torch.Tensor      # [E, D] bf16
    bias: torch.Tensor        # [E] fp32
    experts: grouped.Experts | Exl3Experts | nvx.Experts4  # 4-bit: E + 1 (shared expert last); else the E routed
    shared: MLPW | None = None            # EXL3 and NVFP4 checkpoints: the shared expert (BF16)


@dataclass
class LayerW:
    index: int
    kind: str
    attn_hc: HCW | None
    ffn_hc: HCW | None
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    kda: KDAW | None = None
    dsa: DSAW | None = None
    mlp: MLPW | None = None
    moe: MoEW | None = None


@dataclass
class MTPW:
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh: Q4                    # [D, 2D]: input [embedding | hidden]
    norm: torch.Tensor        # shared_head.norm
    layer: LayerW             # DSA + MoE, plain residual (no hyper-connections)


@dataclass
class Weights:
    cfg: Config
    embed: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    layers: list[LayerW]
    norm: torch.Tensor
    head: Q4 | B16 | Fp4Linear        # NVFP4 (W4A16) where the checkpoint stores it so; it drafts too
    mtp: MTPW | None
    rank: int
    world: int
    device: torch.device
    plan: Any                         # split.ShardPlan: this rank's heads, widths and vocabulary
    comm: Any = None
    meta: dict = field(default_factory=dict)
    draft_head: Q4 | None = None      # BF16 heads: a 4-bit copy the draft steps read (drafts only propose)

    @property
    def vocab_offset(self) -> int:
        return self.plan.span(self.cfg.vocab, UNIT)[0]

    @property
    def vocab_spans(self) -> list[tuple[int, int]]:
        """Every rank's [first, end) of the head's vocabulary, in rank order (ranks of three: unequal widths)."""

        return self.plan.spans(self.cfg.vocab, UNIT)

    def nbytes(self) -> int:
        total = 0
        seen = set()

        def add(t):
            nonlocal total
            if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()
            elif isinstance(t, (Q4, B16, grouped.Experts, Exl3Experts, nvx.Experts4, Fp4Linear, HCW, KDAW, DSAW, MLPW,
                                MoEW, LayerW, MTPW, IndexW, latent.AbsorbW, latent.AbsorbQ4)):
                for v in vars(t).values():
                    add(v)
            elif isinstance(t, (list, tuple)):
                for v in t:
                    add(v)
            elif isinstance(t, dict):
                for v in t.values():
                    add(v)

        add(self.embed)
        add(self.layers)
        add(self.norm)
        add(self.head)
        add(self.draft_head)
        add(self.mtp)
        return total


def load(model_dir: str | Path, *, rank: int, world: int = 2, device: str = "cuda", mtp: bool = True,
         layers: Sequence[int] | None = None) -> Weights:
    """Rank ``rank`` of ``world`` from a checkpoint or rank folder, MTP included unless ``mtp`` is False, with its
    share of the head's vocabulary; ``layers``: only those layers (tests holding a few real ones of several ranks)."""

    from .split import RankReader, ShardPlan

    cfg = Config.read(model_dir)
    if cfg.quant not in ("mlx", "exl3", "nvfp4"):
        raise ValueError(f"GLM-5.3-Flash's CUDA engine reads MLX 4-bit, EXL3 or ModelOpt NVFP4 checkpoints, not "
                         f"{cfg.quant}")
    exl3, nvfp4 = cfg.quant == "exl3", cfg.quant == "nvfp4"
    bf16 = exl3 or nvfp4                     # every weight but the routed experts (and NVFP4's dense MLPs) is BF16
    if nvfp4:
        from tensorfold.cuda import precision

        # cc-defer: W4A16 only; GLM's own math (W4A4) waits for a grouped NVFP4 expert kernel that takes input scales
        if precision.asked() and precision.mode() == precision.CHECKPOINT:
            raise ValueError("--precision checkpoint: GLM-5.3-Flash's grouped NVFP4 experts take bf16 rows only (no "
                             "path for the checkpoint's input scales); serve it with --precision full")
    dev = torch.device(device)
    plan = ShardPlan(cfg, world, rank)
    rd = RankReader(model_dir, plan)
    HL = plan.count(cfg.heads)
    LL = plan.count(cfg.lin_heads)

    def t(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        x = rd.get(PREFIX + name)
        if dtype is not None:
            x = x.to(dtype)
        return x.to(dev)

    def trip(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (as_i32(t(name + ".weight")), t(name + ".scales"), t(name + ".biases"))

    def q4(name: str) -> Q4 | B16:
        return make_b16(t(name + ".weight")) if bf16 else make_q4(*trip(name))

    def packed(name: str) -> bool:
        """Whether projection ``name`` is stored NVFP4 (W4A16: attention and the head of the checkpoints that re-pack
        them; the routed experts and dense MLPs have their own readers)."""

        return nvfp4 and PREFIX + name + ".weight_scale" in rd.index

    def lin(name: str) -> Q4 | B16 | Fp4Linear:
        """A projection as stored: NVFP4 where the checkpoint has its scales, else ``q4``."""

        return fp4(name) if packed(name) else q4(name)

    def stack(names: list[str]) -> Q4 | B16 | Fp4Linear:
        if any(packed(n) for n in names):
            return stack_fp4(names)
        if bf16:
            return stack_b16([t(n + ".weight") for n in names])
        return stack_q4([trip(n) for n in names])

    def stack_fp4(names: list[str]) -> Fp4Linear:
        """NVFP4 projections of one input as one matmul under their one weight scale, the codes' rows zero-padded to a
        multiple of 8: each row of its output then starts 16-byte aligned, as the matmuls reading columns of it in
        place (KDA's f_a and g_a) load their rows."""

        if not all(packed(n) for n in names):
            raise ValueError(f"{', '.join(names)}: some stored NVFP4 and some not; one matmul takes one format")
        scales = sorted({float(rd.get(PREFIX + n + ".weight_scale_2")) for n in names})
        if len(scales) != 1:
            raise ValueError(f"{', '.join(names)}: weight_scale_2 {scales} differ; these projections run as one "
                             "matmul under one weight scale")
        codes = torch.cat([t(n + ".weight") for n in names])
        blocks = torch.cat([t(n + ".weight_scale").view(torch.uint8) for n in names])
        pad = -codes.shape[0] % 8
        if pad:
            codes = torch.cat([codes, codes.new_zeros((pad, codes.shape[1]))])
            blocks = torch.cat([blocks, blocks.new_zeros((pad, blocks.shape[1]))])
        return Fp4Linear.from_checkpoint(codes, blocks, scales[0])

    def hc(i: int, site: str) -> HCW:
        return HCW(t(f"layers.{i}.hc_{site}_fn").contiguous(), t(f"layers.{i}.hc_{site}_base", torch.float32),
                   t(f"layers.{i}.hc_{site}_scale", torch.float32))

    def kda(i: int) -> KDAW:
        p = f"layers.{i}.self_attn."
        proj = stack([p + "q_proj", p + "k_proj", p + "v_proj", p + "f_a_proj", p + "g_a_proj", p + "b_proj"])
        # the KDA kernels take fp32 taps: NVIDIA's NVFP4 checkpoint's as stored, MLX's and EXL3's bf16 widened exactly
        conv = torch.cat([t(p + f"{x}_conv1d.weight", torch.float32) for x in "qkv"]).reshape(
            3 * LL * 128, cfg.conv).contiguous()
        return KDAW(proj, lin(p + "f_b_proj"), lin(p + "g_b_proj"), conv, t(p + "A_log", torch.float32).contiguous(),
                    t(p + "dt_bias", torch.float32).contiguous(), t(p + "o_norm.weight"), lin(p + "o_proj"), LL)

    def dsa(i: int) -> DSAW:
        p = f"layers.{i}.self_attn."
        proj = stack([p + "q_a_proj", p + "kv_a_proj_with_mqa"])
        rows = torch.arange(HL * 512, device=dev).view(HL, 512)
        krows, vrows = rows[:, :cfg.qk_dim].reshape(-1), rows[:, cfg.qk_dim:].reshape(-1)
        if bf16:
            w = t(p + "kv_b_proj.weight")
        else:
            w, s, b = trip(p + "kv_b_proj")
        kv_k = kv_v = None                    # the latent path reads only its own copy (absorb)
        if not latent.ENABLED:
            absorb = None
            if bf16:
                kv_k, kv_v = make_b16(w[krows]), make_b16(w[vrows])
            else:
                kv_k = make_q4(w[krows], s[krows], b[krows])
                kv_v = make_q4(w[vrows], s[vrows], b[vrows])
        elif bf16:
            absorb = latent.AbsorbW.from_rows(w[krows].float(), w[vrows].float(), HL)
        elif cfg.group_size == 64:        # the checkpoint's own 4-bit rows, read as they are stored
            absorb = latent.AbsorbQ4((w[krows], s[krows], b[krows]), (w[vrows], s[vrows], b[vrows]), HL)
        else:
            absorb = latent.AbsorbW.from_rows(latent.dequant_mlx4(w[krows], s[krows], b[krows], cfg.group_size),
                                              latent.dequant_mlx4(w[vrows], s[vrows], b[vrows], cfg.group_size), HL)
        ix = IndexW(stack([p + "indexer.wk", p + "indexer.weights_proj"]), lin(p + "indexer.wq_b"),
                    t(p + "indexer.k_norm.weight"), t(p + "indexer.k_norm.bias"),
                    t(p + "indexer.index_kpool_compress_gate", torch.bfloat16).contiguous(),
                    t(p + "indexer.index_kpool_compress_ape", torch.bfloat16).contiguous())
        return DSAW(proj, t(p + "q_a_layernorm.weight"), t(p + "kv_a_layernorm.weight"), lin(p + "q_b_proj"),
                    kv_k, kv_v, lin(p + "o_proj"), HL, ix, absorb)

    def fp4(name: str) -> Fp4Linear:
        """A ModelOpt NVFP4 projection as stored: codes, e4m3 scales and the fp32 weight scale (W4A16)."""

        return Fp4Linear.from_checkpoint(t(name + ".weight"), t(name + ".weight_scale"),
                                         float(rd.get(PREFIX + name + ".weight_scale_2")))

    def mlp(p: str) -> MLPW:
        if nvfp4 and PREFIX + p + "gate_proj.weight_scale" in rd.index:     # the dense MLP (the shared expert: BF16)
            gate, up = fp4(p + "gate_proj"), fp4(p + "up_proj")
            return MLPW((gate, up), fp4(p + "down_proj"), gate.n)
        gu = stack([p + "gate_proj", p + "up_proj"])
        return MLPW(gu, q4(p + "down_proj"), gu.n // 2)

    def nvfp4_experts(i: int) -> bool:
        """Whether layer ``i``'s routed experts are stored NVFP4 (the MTP layer's are BF16)."""

        return PREFIX + f"layers.{i}.mlp.experts.0.gate_proj.weight_scale" in rd.index

    def expert_names(i: int) -> list[str]:
        """Layer ``i``'s expert tensors in the order ``moe`` reads them (none for a dense layer; ``cfg.layers``: MTP)."""

        mtp_layer = i == cfg.layers and cfg.mtp_layers and mtp
        if not mtp_layer and (i >= cfg.layers or cfg.mlp_kinds[i] != "moe"):
            return []
        if nvfp4 and not nvfp4_experts(i):
            return []                             # BF16 experts: read a projection at a time as they are packed
        p = PREFIX + f"layers.{i}.mlp."
        parts = (("trellis", "suh", "svh") if exl3 else ("weight", "weight_scale", "weight_scale_2") if nvfp4
                 else ("weight", "scales", "biases"))
        names = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names += [p + f"experts.{e}.{proj}.{x}" for e in range(cfg.experts) for x in parts]
            if not bf16:
                names += [p + f"shared_experts.{proj}.{x}" for x in parts]
        return names

    def on_device(tensors: list[torch.Tensor]) -> torch.Tensor:
        """``torch.stack(tensors).to(dev)`` without a host copy: uploaded tensors stacked on the device, host ones copied into their slots."""

        if all(x.is_cuda for x in tensors):
            return torch.stack(tensors)
        out = torch.empty((len(tensors), *tensors[0].shape), dtype=tensors[0].dtype, device=dev)
        for slot, x in zip(out, tensors):
            slot.copy_(x)
        return out

    def moe_exl3(p: str) -> Exl3Experts:
        from tensorfold.cuda.exl3 import experts as generic
        gate = generic.prepare(
            [(rd.get(PREFIX + p + f"experts.{e}.gate_proj.trellis").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.gate_proj.suh").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.gate_proj.svh").to(dev)) for e in range(cfg.experts)],
            [(rd.get(PREFIX + p + f"experts.{e}.up_proj.trellis").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.up_proj.suh").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.up_proj.svh").to(dev)) for e in range(cfg.experts)],
            [(rd.get(PREFIX + p + f"experts.{e}.down_proj.trellis").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.down_proj.suh").to(dev),
              rd.get(PREFIX + p + f"experts.{e}.down_proj.svh").to(dev)) for e in range(cfg.experts)],
            "mcg", device=dev)
        return gate

    def moe_nvfp4(i: int) -> nvx.Experts4:
        """Layer ``i``'s routed experts as ``Experts4`` blocks: the stored NVFP4 bytes, or BF16 ones (the MTP layer's)
        packed by ModelOpt's recipe a projection at a time; those only draft, so replies never depend on them."""

        p = PREFIX + f"layers.{i}.mlp.experts."

        def stacked(proj: str, x: str) -> torch.Tensor:
            return on_device([rd.get(p + f"{e}.{proj}.{x}") for e in range(cfg.experts)])

        parts = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if nvfp4_experts(i):
                parts.append(tuple(stacked(proj, x) for x in ("weight", "weight_scale", "weight_scale_2")))
            else:
                w = stacked(proj, "weight")
                parts.append(nvx.quantize(w))
                del w
        ex = nvx.make(*parts, limit=cfg.limit)
        del parts
        return ex

    def moe(i: int) -> MoEW:
        p = f"layers.{i}.mlp."
        router = t(p + "gate.weight", torch.bfloat16).contiguous()
        bias = t(p + "gate.e_score_correction_bias", torch.float32).contiguous()
        if exl3:
            return MoEW(router, bias, moe_exl3(p), mlp(p + "shared_experts."))
        if nvfp4:
            return MoEW(router, bias, moe_nvfp4(i), mlp(p + "shared_experts."))
        parts = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ws, ss, bs = [], [], []
            for e in range(cfg.experts):
                ws.append(as_i32(rd.get(PREFIX + p + f"experts.{e}.{proj}.weight")))
                ss.append(rd.get(PREFIX + p + f"experts.{e}.{proj}.scales"))
                bs.append(rd.get(PREFIX + p + f"experts.{e}.{proj}.biases"))
            ws.append(as_i32(rd.get(PREFIX + p + f"shared_experts.{proj}.weight")))
            ss.append(rd.get(PREFIX + p + f"shared_experts.{proj}.scales"))
            bs.append(rd.get(PREFIX + p + f"shared_experts.{proj}.biases"))
            parts[proj] = (on_device(ws), on_device(ss), on_device(bs))
            del ws, ss, bs
        ex = grouped.make([parts["gate_proj"], parts["up_proj"]], parts["down_proj"], 64, limit=cfg.limit)
        del parts
        return MoEW(router, bias, ex)

    layer_events: list = []                              # each layer's event, recorded once its work is queued

    def layer(i: int, plain: bool = False, nxt: int | None = None) -> LayerW:
        kind = "dsa" if plain else cfg.kinds[i]
        mk = "moe" if plain else cfg.mlp_kinds[i]
        if len(layer_events) >= 2:                       # at most two layers queued ahead of the GPU
            layer_events.pop(0).synchronize()
        up = None if exl3 else dev                       # MLX and NVFP4 experts come uploaded (EXL3 unpacks on the host)
        rd.prefetch(expert_names(i), up)                 # already queued, except for the first layer
        rd.prefetch(expert_names(i + 1 if nxt is None else nxt), up)   # two layers in flight: reads overlap copies
        lw = LayerW(i, kind, None if plain else hc(i, "attn"), None if plain else hc(i, "ffn"),
                    t(f"layers.{i}.input_layernorm.weight"), t(f"layers.{i}.post_attention_layernorm.weight"))
        if kind == "kda":
            lw.kda = kda(i)
        else:
            lw.dsa = dsa(i)
        if mk == "dense":
            lw.mlp = mlp(f"layers.{i}.mlp.")
        else:
            lw.moe = moe(i)
        if i % 8 == 7:                            # each release waits for the device; a layer leaves few temporaries
            torch.cuda.empty_cache()
        layer_events.append(torch.cuda.current_stream().record_event())
        return lw

    if bf16:
        embed = rd.get(PREFIX + "embed_tokens.weight").to(torch.bfloat16).contiguous().to(dev)
    else:
        embed = (as_i32(rd.get(PREFIX + "embed_tokens.weight")).to(dev), rd.get(PREFIX + "embed_tokens.scales").to(dev),
                 rd.get(PREFIX + "embed_tokens.biases").to(dev))
    try:                                          # a failed load still cancels the reads queued ahead
        which = list(range(cfg.layers)) if layers is None else sorted(set(layers))
        if any(not 0 <= i < cfg.layers for i in which):
            raise ValueError(f"layers {which}: the model has layers 0 .. {cfg.layers - 1}")
        built = [layer(i, nxt=n) for i, n in zip(which, which[1:] + [cfg.layers])]   # the last one reads MTP's ahead
        lo, hi = plan.span(cfg.vocab, UNIT)
        draft_head = None
        if bf16 and "lm_head.weight_scale" in rd.index:      # an NVFP4 head (W4A16): drafts read it too
            head = Fp4Linear.from_checkpoint(rd.get("lm_head.weight")[lo:hi].to(dev),
                                             rd.get("lm_head.weight_scale")[lo:hi].to(dev),
                                             float(rd.get("lm_head.weight_scale_2")))
        elif bf16:
            head = make_b16(rd.get("lm_head.weight")[lo:hi].to(dev))
            # Draft steps use the quantized head; verification keeps the original head.
            draft_head = quantize4(head.weight)
        else:
            hw, hs, hb = (rd.get("lm_head." + x) for x in ("weight", "scales", "biases"))
            head = make_q4(as_i32(hw[lo:hi]).to(dev), hs[lo:hi].to(dev), hb[lo:hi].to(dev))
        mtpw = None
        if cfg.mtp_layers and mtp:
            i = cfg.layers
            mtpw = MTPW(t(f"layers.{i}.enorm.weight"), t(f"layers.{i}.hnorm.weight"), q4(f"layers.{i}.eh_proj"),
                        t(f"layers.{i}.shared_head.norm.weight"), layer(i, plain=True))
        w = Weights(cfg, embed, built, t("norm.weight"), head, mtpw, rank, world, dev, plan, draft_head=draft_head)
        w.meta.update(layers=which)
    finally:
        rd.close()
    torch.cuda.empty_cache()
    return w
