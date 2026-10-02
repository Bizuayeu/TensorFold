"""TF_GLM_KV=fp8: DSA's latent cache (512 values a token and layer) and the indexer's pooled keys as FP8 e4m3 rows
with a power-of-two scale each, about half the bytes of bf16 (the index keys and gates stay bf16).

Adapted from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, patch 0038-glm-kv-fp8 (Apache-2.0).

Lossy, never inexact: a row's bytes are a function of its bf16 row alone (``store_row``, the one writer of latents
and pooled keys for prompt chunks and decode windows alike), and the readers take the codes to bf16 (exact) and fold
each row's scale into their fp32 products (exact for a power of two), so a row's attention depends on its query, its
keys and the cache only: drafted == serial, resumed == fresh and any prompt chunking hold as with bf16
(tests/cuda/test_glm_kv8.py).

A quantized cache is a uint8 tensor [rows, width + PAD]: the row's e4m3 codes, its fp32 scale, 12 zero bytes (rows
stay 16-byte aligned; tensorfold.cuda.geometry.mla_row_bytes counts them). One tensor a cache keeps the engine's
row handling (slices, clones, kept snapshots' saved rows, ``decode.row_bytes``) as it is; kernels tell the formats
apart by dtype. The scale is 2^(ceil(log2 amax) - SHIFT): the row's largest value quantizes into [128, 256], under
e4m3's 448, so nothing saturates; a floating-point format's relative precision does not depend on the scale, so a
power of two loses nothing to a finer scale but the values it pushes below e4m3's normal range (2^-6), under 2^-14
of the row's largest."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

PAD = 16             # bytes after an FP8 row's codes: its fp32 scale, then zeros (the kernels' literal 16)
SHIFT = 8            # a row's scale: 2^(ceil(log2 amax) - SHIFT), so its codes stay within +-256 (the kernels' 8)


def quantized(cache: torch.Tensor) -> bool:
    return cache.dtype == torch.uint8


def width(cache: torch.Tensor) -> int:
    """Values a row of ``cache`` holds."""
    return cache.shape[-1] - PAD if quantized(cache) else cache.shape[-1]


def view(cache: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """What a kernel takes for ``cache`` and whether it is FP8: a bf16 cache as it is, an FP8 one as its e4m3 codes
    (rows of width + PAD bytes, the scale at byte ``width`` of a row)."""
    if not quantized(cache):
        return cache, False
    if cache.dim() != 2 or not cache.is_contiguous() or cache.shape[1] % PAD:
        raise ValueError(f"an FP8 cache: contiguous rows of a width + {PAD} bytes, a multiple of {PAD}")
    return cache.view(torch.float8_e4m3fn), True


# -- the format in torch (tests, and the definition the kernels must equal) -------------------------------------------
def scales(amax: torch.Tensor) -> torch.Tensor:
    """fp32 amax -> the fp32 power-of-two scale (the kernels' ``scale_of``, bit for bit)."""
    bits = amax.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((e - SHIFT).clamp(1, 254) << 23).view(torch.float32)


def quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """bf16 rows [n, width] -> an FP8 cache's rows [n, width + PAD], as the write kernels store them."""
    x = x.to(torch.bfloat16).float()
    s = scales(x.abs().amax(dim=1))
    inv = ((254 - (s.view(torch.int32) >> 23)) << 23).view(torch.float32)
    out = torch.zeros((x.shape[0], x.shape[1] + PAD), dtype=torch.uint8, device=x.device)
    out[:, :x.shape[1]] = (x * inv[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    out[:, x.shape[1]:x.shape[1] + 4] = s[:, None].contiguous().view(torch.uint8)
    return out


def dequantize(cache: torch.Tensor) -> torch.Tensor:
    """A cache's rows as fp32 [n, width] (exact: e4m3 codes times a power of two; bf16 rows as they are)."""
    if not quantized(cache):
        return cache.float()
    w = width(cache)
    codes = cache[..., :w].contiguous().view(torch.float8_e4m3fn).float()
    return codes * cache[..., w:w + 4].contiguous().view(torch.float32)


# -- the format in kernels ------------------------------------------------------------------------------------------
@triton.jit
def scale_of(amax):
    """fp32 amax -> 2^(ceil(log2 amax) - SHIFT) and its reciprocal, both exact powers of two (biased exponent held
    within 1 .. 254: a zero row gets 2^-126 and zeros)."""
    bits = amax.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    e = tl.minimum(tl.maximum(e - 8, 1), 254)
    return (e << 23).to(tl.float32, bitcast=True), ((254 - e) << 23).to(tl.float32, bitcast=True)


@triton.jit
def store_row(C, row, x, LW: tl.constexpr):
    """Row ``row`` (int64) of an FP8 cache (C: its e4m3 codes) <- x fp32 [LW] as bf16: the bf16 row's codes and
    scale (``quantize_rows``)."""
    k = tl.arange(0, LW)
    xf = x.to(tl.bfloat16).to(tl.float32)
    s, inv = scale_of(tl.max(tl.abs(xf), 0))
    at = C + row * (LW + 16)
    tl.store(at + k, (xf * inv).to(tl.float8e4nv))
    tl.store((at + LW).to(tl.pointer_type(tl.float32), bitcast=True), s)


@triton.jit
def load_rows(C, rows, ok, k, LW: tl.constexpr):
    """Rows ``rows`` (int64 [n]) of an FP8 cache: their codes as a bf16 tile [n, LW] (exact) and their scales [n];
    rows with ok false are not read (zeros, scale 1)."""
    at = C + rows * (LW + 16)
    kv = tl.load(at[:, None] + k[None, :], mask=ok[:, None], other=0.0).to(tl.bfloat16)
    s = tl.load((at + LW).to(tl.pointer_type(tl.float32), bitcast=True), mask=ok, other=1.0)
    return kv, s
