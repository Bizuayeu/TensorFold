"""The headers of nvidia/GLM-5.3-Flash-NVFP4 with its attention projections and lm_head re-packed W4A16 NVFP4
(Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16, ``in_proj_layout`` "split-qkv-bfg"; every safetensors header in
tests/fixtures/glm53_flash_nvfp4_attn_w4a16): the pinned checkpoint's tensors with those projections as e2m1 codes, e4m3
scales per 16 inputs and an fp32 weight scale. Each of its 148,407 tensors takes one split rule, every rank of two and
three cuts the NVFP4 projections between whole 16-input groups with K a multiple of 64, the family reads its
MIXED_PRECISION config, and the startup estimate holds the packed projections as stored."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from tensorfold.cuda import geometry
from tensorfold.families.glm5_next.cuda.split import AXIS, ShardPlan, rule

FIXTURES = Path(__file__).parent / "fixtures"
HERE = FIXTURES / "glm53_flash_nvfp4_attn_w4a16"
PINNED = FIXTURES / "glm53_flash_nvfp4"
GIB = 2 ** 30
KDA = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj", "f_b_proj", "g_b_proj", "o_proj")
DSA = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj", "indexer.wq_b")
LAYER = re.compile(r"layers\.(\d+)\.self_attn\.(.+)\.weight$")


def _headers(where: Path) -> dict:
    return json.loads(gzip.decompress((where / "headers.json.gz").read_bytes()))


@pytest.fixture(scope="module")
def headers() -> dict:
    return _headers(HERE)


@pytest.fixture(scope="module")
def pinned() -> dict:
    return _headers(PINNED)


@pytest.fixture(scope="module")
def text() -> dict:
    return json.loads((HERE / "config.json").read_text())["text_config"]


def test_every_tensor_takes_one_rule(headers):
    kinds = {name: rule(name) for name in headers}                  # an ambiguous or missing rule raises
    assert len(kinds) == 148407
    assert {n for n, k in kinds.items() if k == "drop"} == (
        {n for n in headers if n.startswith("model.visual.")} | {n for n in headers if n.endswith(".input_scale")})
    for name, (_, shape) in headers.items():
        if name.endswith(".weight_scale_2"):
            assert shape == [] and kinds[name] == "rep", name


def test_only_attention_and_the_head_are_repacked(headers, pinned, text):
    """Against the pinned checkpoint: no name gone, the 746 new names the scales of the 373 weights that became NVFP4
    (BF16 [N, K] to U8 [N, K/2], no input scale), and those are every KDA and DSA projection of the decoder layers but
    the indexer's wk and weights_proj, and lm_head; the MTP layer stays BF16."""

    layers = int(text["num_hidden_layers"])
    kinds = ["kda" if k == "linear_attention" else "dsa" for k in text["layer_types"]]
    assert not set(pinned) - set(headers)
    changed = {n for n in pinned if headers[n] != pinned[n]}
    assert len(headers) - len(pinned) == 746 and len(changed) == 373
    for name in changed:
        (dtype, shape), (was, full) = headers[name], pinned[name]
        assert was == "BF16" and dtype == "U8" and shape == [full[0], full[1] // 2], name
        base = name[:-len(".weight")]
        assert headers[base + ".weight_scale"] == ["F8_E4M3", [full[0], full[1] // 16]], name
        assert headers[base + ".weight_scale_2"] == ["F32", []], name
        assert base + ".input_scale" not in headers, name
    assert set(headers) - set(pinned) == {n[:-len(".weight")] + s for n in changed for s in (".weight_scale",
                                                                                              ".weight_scale_2")}
    want = {"lm_head.weight"} | {f"model.language_model.layers.{i}.self_attn.{p}.weight"
                                 for i in range(layers) for p in (KDA if kinds[i] == "kda" else DSA)}
    assert changed == want
    assert sum(kinds[int(LAYER.search(n).group(1))] == "kda" for n in changed if LAYER.search(n)) == 34 * 9


@pytest.mark.torch                       # the rank's plan reads the engine's Config (weights.py imports torch)
@pytest.mark.parametrize("world", [2, 3])
def test_every_rank_cuts_whole_groups(headers, world):
    """Each rank's part of an NVFP4 projection: its codes and their e4m3 scales cut at the same elements (whole
    16-input groups), and K a multiple of 64 (``Fp4Linear``); three ranks take KDA's 64 heads 22/21/21."""

    from tensorfold.families.glm5_next.cuda.weights import Config

    cfg = Config.read(HERE)
    seen = 0
    for rank in range(world):
        plan = ShardPlan(cfg, world, rank)
        for name, (dtype, shape) in headers.items():
            if dtype != "U8" or ".mlp." in name:
                continue
            seen += 1
            base = name[:-len(".weight")]
            n, k = shape[0], 2 * shape[1]
            kind = rule(name)
            assert rule(base + ".weight_scale") == kind and rule(base + ".weight_scale_2") == "rep"
            if kind == "row" or name.startswith("lm_head."):          # rank folders keep the whole head
                lo, hi = plan.cut(name, n)
                assert plan.cut(base + ".weight_scale", n) == (lo, hi)
                n = hi - lo
            elif kind == "col":
                lo, hi = plan.cut(name, shape[AXIS[kind]])
                slo, shi = plan.cut(base + ".weight_scale", headers[base + ".weight_scale"][1][1])
                assert (2 * lo, 2 * hi) == (16 * slo, 16 * shi), name
                k = 2 * (hi - lo)
            assert n > 0 and k % 64 == 0, (name, rank, n, k)
    assert seen == 373 * world


def _rank_bytes(where: Path, world: int, rank: int, names) -> dict[str, int]:
    from tensorfold.families.glm5_next.cuda.weights import Config

    transform = geometry.split_weights(rule, ShardPlan(Config.read(where), world, rank))
    return {n: transform(n, {"dtype": d, "shape": s})[0] for n, (d, s) in names.items()}


@pytest.mark.torch
def test_rank_bytes(headers, pinned, capsys):
    """split_weights on both checkpoints' headers, rank 0 of two: every tensor the pinned one holds alike but the
    re-packed ones, which hold their codes and e4m3 scales with the rows padded to 128 (``qmm.pack``), the weight
    scale a number; kv_b as the latent path's BF16 copy either way (dequantized at load); the head without its 4-bit
    draft copy (the NVFP4 head drafts itself)."""

    axl, pin = _rank_bytes(HERE, 2, 0, headers), _rank_bytes(PINNED, 2, 0, pinned)
    changed = {n for n in pinned if headers[n] != pinned[n]}
    stored = {n[:-len(".weight")] + s for n in changed for s in (".weight", ".weight_scale", ".weight_scale_2")}
    assert all(axl[n] == pin[n] for n in pinned if n not in stored)

    def packed(name: str) -> int:
        n, k = pinned[name][1]
        kind = rule(name)
        if kind == "row" or name.startswith("lm_head."):
            n //= 2
        elif kind == "col":
            k //= 2
        if ".kv_b_proj." in name:
            return n * k * 2                                         # AbsorbW: the rank's rows in BF16
        rows = -(-n // 128) * 128
        return rows * k // 2 + rows * k // 16

    for name in changed:
        base = name[:-len(".weight")]
        assert axl[name] + axl[base + ".weight_scale"] == packed(name), name
        assert axl[base + ".weight_scale_2"] == 0
    kv_b = [n for n in changed if ".kv_b_proj." in n]
    assert all(axl[n] + axl[n[:-len(".weight")] + ".weight_scale"] == pin[n] for n in kv_b)
    saved = sum(pin.values()) - sum(axl.values())
    with capsys.disabled():
        print(f"\nre-packed attention and head, rank 0 of two: {sum(axl.values()) / GIB:.2f} GiB, "
              f"{saved / GIB:.2f} GiB under the pinned checkpoint's {sum(pin.values()) / GIB:.2f}")


def test_the_family_reads_the_mixed_precision_config(tmp_path, monkeypatch):
    """quant_algo MIXED_PRECISION whose layers are NVFP4 or W4A16_NVFP4 is read on CUDA; another algo among them is
    refused by name."""

    import sys

    from tensorfold import families
    from tensorfold.families import glm5_next

    config = json.loads((HERE / "config.json").read_text())
    families.require_readable(families.detect(HERE), config, "cuda")
    monkeypatch.setattr(sys, "platform", "linux")
    glm5_next.check(HERE)
    q = config["quantization_config"]
    fp8 = {**config, "quantization_config": {**q, "quantized_layers": {**q["quantized_layers"],
                                                                         "lm_head": {"quant_algo": "FP8"}}}}
    (tmp_path / "config.json").write_text(json.dumps(fp8))
    with pytest.raises(ValueError, match="FP8"):
        glm5_next.check(tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="CUDA engine only"):
        glm5_next.check(HERE)


@pytest.mark.torch
def test_the_engine_config_reads_it_as_nvfp4(tmp_path):
    from tensorfold.families.glm5_next.cuda.weights import Config

    assert Config.read(HERE).quant == "nvfp4"
    config = json.loads((HERE / "config.json").read_text())
    q = config["quantization_config"]
    odd = {**config, "quantization_config": {**q, "quantized_layers": {"lm_head": {"quant_algo": "FP8"}}}}
    (tmp_path / "config.json").write_text(json.dumps(odd))
    assert Config.read(tmp_path).quant != "nvfp4"


def test_the_checkpoint_counts_its_nvfp4_attention(headers, pinned):
    """The startup agreement's checkpoint kind: how many attention and head projections are NVFP4 (none pinned)."""

    from tensorfold.families.glm5_next import nvfp4_attention

    assert nvfp4_attention(headers) == 373
    assert nvfp4_attention(pinned) == 0
