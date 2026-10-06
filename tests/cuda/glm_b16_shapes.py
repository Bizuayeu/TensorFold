"""The BF16 matmuls one decode step of GLM-5.3-Flash runs on a rank, as (site, n, k): the pinned NVFP4 checkpoint's
headers (tests/fixtures/glm53_flash_nvfp4) cut by ``split.ShardPlan`` and stacked as ``weights.load`` stacks them.

Left out: kv_b's per-head halves (kv_k / kv_v), which decode never reads with the latent cache (TF_GLM_LATENT, on by
default), and the dense layers' MLPs, NVFP4 (``Fp4Linear``) in this checkpoint, not BF16."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from tensorfold.families.glm5_next.cuda import split
from tensorfold.families.glm5_next.cuda.weights import PREFIX, Config

HERE = Path(__file__).parents[1] / "fixtures" / "glm53_flash_nvfp4"
WORLDS = (2, 3)
DECODE_ROWS = tuple(range(1, 9))          # a pending token and up to 7 drafts (engine.MAX_ROWS), MTP's windows within


def _headers() -> dict:
    return json.loads(gzip.decompress((HERE / "headers.json.gz").read_bytes()))


def matmuls(world: int, rank: int, cfg: Config | None = None, headers: dict | None = None) -> list[tuple[str, int, int]]:
    """Every BF16 matmul of one decode step on ``rank`` of ``world``, once per call (layers repeat their shapes)."""

    cfg = Config.read(HERE) if cfg is None else cfg
    headers = _headers() if headers is None else headers
    plan = split.ShardPlan(cfg, world, rank)

    def shape(name: str) -> tuple[int, int] | None:
        full = name if name == "lm_head" else PREFIX + name
        dtype, (n, k) = headers[full + ".weight"]
        if dtype != "BF16":
            return None
        if name == "lm_head":
            lo, hi = plan.span(cfg.vocab, split.UNIT)
            return hi - lo, k
        kind = split.rule(full + ".weight")
        if kind == "row":
            lo, hi = plan.cut(full + ".weight", n)
            return hi - lo, k
        if kind == "col":
            lo, hi = plan.cut(full + ".weight", k)
            return n, hi - lo
        return n, k

    def stacked(names: list[str]) -> tuple[int, int]:
        parts = [shape(n) for n in names]
        return sum(p[0] for p in parts), parts[0][1]

    out: list[tuple[str, int, int]] = []

    def add(site: str, nk: tuple[int, int] | None) -> None:
        if nk is not None:
            out.append((site, *nk))

    last = cfg.layers + (1 if cfg.mtp_layers else 0)
    for i in range(last):
        mtp = i == cfg.layers
        a = f"layers.{i}.self_attn."
        if not mtp and cfg.kinds[i] == "kda":
            add("kda.proj", stacked([a + x for x in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj")]))
            add("kda.f_b", shape(a + "f_b_proj"))
            add("kda.g_b", shape(a + "g_b_proj"))
            add("kda.o", shape(a + "o_proj"))
        else:
            add("dsa.proj", stacked([a + "q_a_proj", a + "kv_a_proj_with_mqa"]))
            add("dsa.q_b", shape(a + "q_b_proj"))
            add("dsa.o", shape(a + "o_proj"))
            add("ix.wk", stacked([a + "indexer.wk", a + "indexer.weights_proj"]))
            add("ix.wq_b", shape(a + "indexer.wq_b"))
        m = f"layers.{i}.mlp." + ("shared_experts." if mtp or cfg.mlp_kinds[i] == "moe" else "")
        gate = shape(m + "gate_proj")
        if gate is not None:
            add("mlp.gate_up", stacked([m + "gate_proj", m + "up_proj"]))
            add("mlp.down", shape(m + "down_proj"))
        if mtp:
            add("mtp.eh", shape(f"layers.{i}.eh_proj"))
    add("head", shape("lm_head"))
    return out


def decode_shapes() -> list[tuple[int, int]]:
    """The distinct (n, k) of every rank's decode matmuls at TP=2 and TP=3."""

    cfg, headers = Config.read(HERE), _headers()
    return sorted({(n, k) for w in WORLDS for r in range(w) for _, n, k in matmuls(w, r, cfg, headers)})
