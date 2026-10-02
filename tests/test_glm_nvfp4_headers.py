"""The pinned ModelOpt NVFP4 checkpoint's headers (nvidia/GLM-5.3-Flash-NVFP4 at 423acf37, every safetensors header
read over HTTP ranges into tests/fixtures/glm53_flash_nvfp4): each of its 147,661 tensors takes one split rule, the
NVFP4 splits keep whole 16-input groups and the grouped kernel's 32-wide tiles on both ranks, and the startup estimate
gives each rank's bytes."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from tensorfold.cuda import geometry
from tensorfold.families.glm5_next.cuda.split import ShardPlan, rule

HERE = Path(__file__).parent / "fixtures" / "glm53_flash_nvfp4"
GIB = 2 ** 30
EXPERT = re.compile(r"layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.")
DENSE = re.compile(r"layers\.(\d+)\.mlp\.(gate|up|down)_proj\.")


@pytest.fixture(scope="module")
def headers() -> dict:
    return json.loads(gzip.decompress((HERE / "headers.json.gz").read_bytes()))


@pytest.fixture(scope="module")
def text() -> dict:
    return json.loads((HERE / "config.json").read_text())["text_config"]


def test_every_tensor_takes_one_rule(headers):
    kinds = {name: rule(name) for name in headers}                  # an ambiguous or missing rule raises
    assert len(kinds) == 147661
    assert {n for n, k in kinds.items() if k == "drop"} == (
        {n for n in headers if n.startswith("model.visual.")} | {n for n in headers if n.endswith(".input_scale")})
    for name, (_, shape) in headers.items():
        if name.endswith(".weight_scale_2"):
            assert shape == [] and kinds[name] == "rep", name


def test_nvfp4_splits_keep_whole_groups_and_tiles(headers, text):
    """Each rank's part of an NVFP4 weight: whole 16-input groups (scales split with their codes), and the shapes the
    kernels take: N and K multiples of 32 for the grouped experts, K a multiple of 64 for the dense projections."""

    width, dense, hidden = int(text["moe_intermediate_size"]), int(text["intermediate_size"]), int(text["hidden_size"])
    layers = int(text["num_hidden_layers"])
    seen = {"expert": 0, "dense": 0}
    for name, (dtype, shape) in headers.items():
        if dtype != "U8" or not name.endswith(".weight"):
            continue
        base = name[:-len(".weight")]
        sdtype, sshape = headers[base + ".weight_scale"]
        assert sdtype == "F8_E4M3" and headers[base + ".weight_scale_2"] == ["F32", []], base
        n, k = shape[0], 2 * shape[1]
        assert sshape == [n, k // 16], base
        kind = rule(name)
        assert rule(base + ".weight_scale") == kind, base
        m = EXPERT.search(name) or DENSE.search(name)
        assert m and int(m.group(1)) < layers, name                     # MTP layer 45 stays BF16
        expert = m.re is EXPERT
        seen["expert" if expert else "dense"] += 1
        if kind == "row":                                               # gate/up: output rows
            n //= 2
        else:                                                           # down: input columns, groups whole
            assert kind == "col" and sshape[1] % 2 == 0, base
            k //= 2
        assert k % 16 == 0
        if expert:
            assert n % 32 == 0 and k % 32 == 0, base
            assert (n, k) in ((width // 2, hidden), (hidden, width // 2))
        else:
            assert k % 64 == 0 and (n, k) in ((dense // 2, hidden), (hidden, dense // 2))
    first = int(text["first_k_dense_replace"])
    assert seen == {"expert": (layers - first) * int(text["n_routed_experts"]) * 3, "dense": first * 3}


def _component(name: str, layers: int) -> str:
    if name.startswith("model.visual."):
        return "visual"
    m = EXPERT.search(name)
    if m:
        return "MTP routed experts (NVFP4 drafts)" if int(m.group(1)) == layers else "routed experts NVFP4"
    if DENSE.search(name):
        return "dense MLP NVFP4"
    if ".shared_experts." in name:
        return "shared experts BF16"
    if name.startswith("lm_head."):
        return "lm_head BF16 + 4-bit draft copy"
    if "embed_tokens" in name:
        return "embed BF16"
    if ".self_attn." in name:
        return "attention BF16"
    return "other"


@pytest.mark.torch                       # the rank's plan reads the engine's Config (weights.py imports torch)
def test_rank_bytes(headers, text, capsys):
    """split_weights on the real headers, each rank: the routed experts as stored (blocks hold the codes and their
    e4m3 scales byte for byte), the MTP layer's BF16 experts at the NVFP4 size they are packed to, nothing for input
    scales or the visual tower."""

    layers = int(text["num_hidden_layers"])
    first = int(text["first_k_dense_replace"])
    experts, width = int(text["n_routed_experts"]), int(text["moe_intermediate_size"])
    hidden, dense = int(text["hidden_size"]), int(text["intermediate_size"])
    from tensorfold.families.glm5_next.cuda.weights import Config

    transform = geometry.split_weights(rule, ShardPlan(Config.read(HERE), 2, 0))
    parts: dict[str, int] = {}
    for name, (dtype, shape) in headers.items():
        size, host = transform(name, {"dtype": dtype, "shape": shape})
        assert host == 0
        key = _component(name, layers)
        parts[key] = parts.get(key, 0) + size
    matrix = hidden * width // 2 * 9 // 16 + 4          # one expert matrix a rank: codes, e4m3 scales, its fp32 scale
    assert parts["routed experts NVFP4"] == (layers - first) * experts * 3 * matrix
    assert parts["MTP routed experts (NVFP4 drafts)"] == experts * 3 * matrix
    assert parts["dense MLP NVFP4"] == first * 3 * (hidden * dense // 2 * 9 // 16)
    assert parts["visual"] == 0
    vocab = int(text["vocab_size"])
    assert parts["lm_head BF16 + 4-bit draft copy"] == vocab // 2 * hidden * 2 + vocab // 2 * hidden * 9 // 16
    total = sum(parts.values())
    with capsys.disabled():
        print("\nGLM-5.3-Flash NVFP4, one rank of two (startup estimate, GiB):")
        for key, size in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<36} {size / GIB:8.2f}")
        print(f"  {'total':<36} {total / GIB:8.2f}")


def test_the_family_reads_modelopt_nvfp4_on_cuda(tmp_path, monkeypatch):
    import sys

    from tensorfold import families
    from tensorfold.families import glm5_next

    config = json.loads((HERE / "config.json").read_text())
    families.require_readable(families.detect(HERE), config, "cuda")
    with pytest.raises(ValueError):
        families.require_readable(families.detect(HERE), config, "mlx")
    monkeypatch.setattr(sys, "platform", "linux")
    glm5_next.check(HERE)
    fp8 = {**config, "quantization_config": {**config["quantization_config"], "quant_algo": "FP8"}}
    (tmp_path / "config.json").write_text(json.dumps(fp8))
    with pytest.raises(ValueError, match="ModelOpt FP8"):
        glm5_next.check(tmp_path)
