"""Each rank's share of GLM-5.3-Flash (``split.ShardPlan``) on the pinned NVFP4 checkpoint's headers (the 147,661
tensors of tests/fixtures/glm53_flash_nvfp4): two ranks take exactly the halves the two-rank split took before the plan
(byte for byte, every tensor), on bytes that number their own positions."""

from __future__ import annotations

import gzip
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")                     # the plan reads the engine's Config (weights.py imports torch)

from tensorfold.families.glm5_next.cuda import split  # noqa: E402
from tensorfold.families.glm5_next.cuda.weights import Config  # noqa: E402

HERE = Path(__file__).parent / "fixtures" / "glm53_flash_nvfp4"


def _half_split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int):
    """``split.split_bytes`` as it was at 1b81c37 (two ranks, halves), kept to pin the two-rank shares."""

    if kind == "rep":
        return raw, list(shape)
    if kind == "row":
        rows = shape[0]
        if rows % 2:
            raise ValueError(f"row split of odd leading dim {shape}")
        per = raw.size // rows
        half = rows // 2
        return raw[rank * half * per:(rank + 1) * half * per], [half] + list(shape[1:])
    if kind == "col":
        if len(shape) != 2 or shape[1] % 2:
            raise ValueError(f"column split needs an even 2-D shape, got {shape}")
        view = raw.reshape(shape[0], shape[1] * itemsize)
        half = shape[1] // 2
        part = np.ascontiguousarray(view[:, rank * half * itemsize:(rank + 1) * half * itemsize])
        return part.reshape(-1), [shape[0], half]
    if kind == "dim1":
        if len(shape) < 2 or shape[1] % 2:
            raise ValueError(f"split of the second axis needs an even second dim, got {shape}")
        inner = int(np.prod(shape[2:])) * itemsize
        view = raw.reshape(shape[0], shape[1] * inner)
        half = shape[1] // 2
        part = np.ascontiguousarray(view[:, rank * half * inner:(rank + 1) * half * inner])
        return part.reshape(-1), [shape[0], half] + list(shape[2:])
    raise ValueError(kind)


def _numbered(nbytes: int) -> np.ndarray:
    """``nbytes`` bytes that number their own 8-byte positions (any shifted cut reads different bytes)."""

    return np.arange(-(-nbytes // 8), dtype=np.uint64).view(np.uint8)[:nbytes]


AXIS = {"row": 0, "col": 1, "dim1": 1}


@pytest.fixture(scope="module")
def headers() -> dict:
    return json.loads(gzip.decompress((HERE / "headers.json.gz").read_bytes()))


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.read(HERE)


def _cases(headers: dict, plan: split.ShardPlan):
    """Every kept tensor as (name, its case): the bytes a rank takes depend on the case alone."""

    for name, (dtype, shape) in headers.items():
        kind = split.rule(name)
        if kind == "drop":
            continue
        if name.startswith("lm_head."):          # kept whole by the split, cut to the rank's vocabulary by the loader
            yield name, (dtype, tuple(shape), "row", plan.axis(name))
        else:
            yield name, (dtype, tuple(shape), kind, plan.axis(name) if kind != "rep" else 0)


def test_two_ranks_take_the_halves_of_every_tensor(headers, cfg):
    """world=2: each rank's bytes of every tensor are the halves the split took before the plan."""

    plans = [split.ShardPlan(cfg, 2, rank) for rank in (0, 1)]
    checked: dict[tuple, bool] = {}
    count = 0
    for (name, case), (_, case1) in zip(_cases(headers, plans[0]), _cases(headers, plans[1])):
        assert case == case1, name
        count += 1
        if case in checked:
            continue
        dtype, shape, kind, _ = case
        itemsize = split.DTYPE_BYTES[dtype]
        raw = _numbered(int(np.prod(shape)) * itemsize)
        for rank, plan in enumerate(plans):
            want, want_shape = _half_split_bytes(raw, list(shape), itemsize, kind, rank)
            cut = None if kind == "rep" else plan.cut(name, shape[AXIS[kind]])
            got, got_shape = split.split_bytes(raw, list(shape), itemsize, kind, cut)
            assert got_shape == want_shape and np.array_equal(got, want), (name, rank)
        checked[case] = True
    assert count == 147661 - sum(split.rule(n) == "drop" for n in headers)
    assert len(checked) > 20                     # the distinct cases the 147,661 tensors fall into


def test_three_ranks_put_every_tensor_back_together(headers, cfg):
    """world=3: the three ranks' parts of every tensor, joined along its split axis, are the tensor; none is empty."""

    plans = [split.ShardPlan(cfg, 3, rank) for rank in range(3)]
    checked: dict[tuple, bool] = {}
    count = 0
    for name, case in _cases(headers, plans[0]):
        count += 1
        if case in checked:
            continue
        dtype, shape, kind, _ = case
        itemsize = split.DTYPE_BYTES[dtype]
        raw = _numbered(int(np.prod(shape)) * itemsize)
        parts = []
        for plan in plans:
            cut = None if kind == "rep" else plan.cut(name, shape[AXIS[kind]])
            data, part_shape = split.split_bytes(raw, list(shape), itemsize, kind, cut)
            assert data.size and int(np.prod(part_shape)) * itemsize == data.size, (name, plan.rank)
            parts.append(data.reshape(part_shape[0], -1) if kind == "row" else
                         data.reshape(shape[0], -1) if kind != "rep" else data)
        if kind == "rep":
            assert all(np.array_equal(p, raw) for p in parts), name
        else:
            whole = np.concatenate(parts, axis=0 if kind == "row" else 1)
            assert np.array_equal(whole.reshape(-1), raw), name
        checked[case] = True
    assert count == 147661 - sum(split.rule(n) == "drop" for n in headers)


def test_three_rank_shares_of_glm_flash(headers, cfg):
    """The shares the TP=3 design fixed: heads 22/21/21 (the remainder to the first ranks), expert widths 704/704/640, vocabulary 51,648/51,648/51,584,
    dense MLP 4,096 each; and every NVFP4 projection's part keeps whole 16-input groups and the 32-wide tiles."""

    from tensorfold.families.glm5_next.cuda.split import UNIT

    plans = [split.ShardPlan(cfg, 3, rank) for rank in range(3)]
    assert [p.count(cfg.heads) for p in plans] == [22, 21, 21]
    assert [p.count(cfg.lin_heads) for p in plans] == [22, 21, 21]
    assert [p.count(cfg.moe_width, UNIT) for p in plans] == [704, 704, 640]
    assert [p.count(cfg.shared_width, UNIT) for p in plans] == [704, 704, 640]
    assert [p.count(cfg.vocab, UNIT) for p in plans] == [51648, 51648, 51584]
    assert [p.count(cfg.dense_width, UNIT) for p in plans] == [4096, 4096, 4096]
    assert [p.span(cfg.vocab, UNIT) for p in plans] == [(0, 51648), (51648, 103296), (103296, 154880)]
    # the stored tensors' parts: rows, bytes (two codes a byte) and e4m3 scales (one per 16 inputs)
    L = "model.language_model.layers."
    want = {L + "3.self_attn.q_b_proj.weight": (0, [22 * 256, 21 * 256, 21 * 256]),
            L + "3.self_attn.kv_b_proj.weight": (0, [22 * 512, 21 * 512, 21 * 512]),
            L + "3.self_attn.o_proj.weight": (1, [22 * 256, 21 * 256, 21 * 256]),
            L + "0.self_attn.o_proj.weight": (1, [22 * 128, 21 * 128, 21 * 128]),
            L + "0.self_attn.q_proj.weight": (0, [22 * 128, 21 * 128, 21 * 128]),
            L + "0.self_attn.q_conv1d.weight": (0, [22 * 128, 21 * 128, 21 * 128]),
            L + "0.self_attn.b_proj.weight": (0, [22, 21, 21]),
            L + "0.self_attn.A_log": (0, [22, 21, 21]),
            L + "0.self_attn.dt_bias": (0, [22 * 128, 21 * 128, 21 * 128]),
            L + "3.mlp.experts.7.gate_proj.weight": (0, [704, 704, 640]),
            L + "3.mlp.experts.7.down_proj.weight": (1, [352, 352, 320]),
            L + "3.mlp.experts.7.down_proj.weight_scale": (1, [44, 44, 40]),
            L + "45.mlp.experts.7.down_proj.weight": (1, [704, 704, 640]),      # the MTP layer's BF16 experts
            L + "3.mlp.shared_experts.gate_proj.weight": (0, [704, 704, 640]),
            L + "3.mlp.shared_experts.down_proj.weight": (1, [704, 704, 640]),
            L + "0.mlp.down_proj.weight": (1, [2048, 2048, 2048]),
            L + "0.mlp.down_proj.weight_scale": (1, [256, 256, 256]),
            L + "0.mlp.gate_proj.weight": (0, [4096, 4096, 4096]),
            "lm_head.weight": (0, [51648, 51648, 51584])}
    for name, (axis, sizes) in want.items():
        size = headers[name][1][axis]
        cuts = [p.cut(name, size) for p in plans]
        assert [hi - lo for lo, hi in cuts] == sizes, name
        assert cuts[0][0] == 0 and cuts[-1][1] == size and all(a[1] == b[0] for a, b in pairwise(cuts)), name
    for name, (dtype, shape) in headers.items():
        if dtype != "U8" or not name.endswith(".weight") or split.rule(name) == "drop":
            continue
        kind = split.rule(name)
        scales = headers[name[:-len("weight")] + "weight_scale"][1]
        for plan in plans:
            n0, n1 = plan.cut(name, shape[0]) if kind == "row" else (0, shape[0])
            k0, k1 = (0, shape[1]) if kind == "row" else plan.cut(name, shape[1])
            s0, s1 = (0, scales[1]) if kind == "row" else plan.cut(name, scales[1])
            assert (k1 - k0) * 2 == (s1 - s0) * 16, name                      # whole 16-input groups
            if ".mlp.experts." in name:
                assert (n1 - n0) % 32 == 0 and (k1 - k0) * 2 % 32 == 0, name  # Experts4's tiles
            else:
                assert (k1 - k0) * 2 % 64 == 0, name                          # the dense projection's K


def test_shares_cut_between_whole_units_and_leave_no_rank_empty():
    from tensorfold.cuda.geometry import split_units

    assert [split_units(64, 3, r) for r in range(3)] == [(0, 22), (22, 43), (43, 64)]
    assert [split_units(32, 3, r) for r in range(3)] == [(0, 11), (11, 22), (22, 32)]
    assert [split_units(4, 3, r) for r in range(3)] == [(0, 2), (2, 3), (3, 4)]
    assert [split_units(64, 2, r) for r in range(2)] == [(0, 32), (32, 64)]
    assert [split_units(5, 2, r) for r in range(2)] == [(0, 3), (3, 5)]
    assert split_units(7, 1, 0) == (0, 7)
    for units, world in ((2, 3), (1, 2)):                          # fewer units than ranks
        with pytest.raises(ValueError, match="empty"):
            split_units(units, world, 0)
    with pytest.raises(ValueError):
        split.ShardPlan(None, 3, 3)


def test_three_rank_startup_estimates(headers, cfg, capsys):
    """split_weights on the real headers for each rank of three: the routed experts and the head at the rank's width
    and vocabulary, each rank smaller than the one before."""

    import re

    from tensorfold.cuda import geometry
    from tensorfold.families.glm5_next.cuda.split import UNIT

    expert = re.compile(r"layers\.(\d+)\.mlp\.experts\.")
    first = cfg.mlp_kinds.count("dense")
    totals = []
    for rank in range(3):
        plan = split.ShardPlan(cfg, 3, rank)
        transform = geometry.split_weights(split.rule, plan)
        routed = head = total = 0
        for name, (dtype, shape) in headers.items():
            size = transform(name, {"dtype": dtype, "shape": shape})[0]
            total += size
            m = expert.search(name)
            routed += size if m and int(m.group(1)) < cfg.layers else 0
            head += size if name.startswith("lm_head.") else 0
        width, vocab = plan.count(cfg.moe_width, UNIT), plan.count(cfg.vocab, UNIT)
        assert routed == (cfg.layers - first) * cfg.experts * 3 * (cfg.hidden * width * 9 // 16 + 4)
        # the BF16 head and its 4-bit draft copy, whose rows ``qmm.pack`` pads to 128 (51,648 hold 51,712)
        assert head == vocab * cfg.hidden * 2 + -(-vocab // 128) * 128 * cfg.hidden * 9 // 16
        totals.append(total)
    assert totals[0] > totals[1] > totals[2]
    # the caches and buffers are estimated at the largest rank's share: per-head caches hold 22 heads, not 32
    text = json.loads((HERE / "config.json").read_text())["text_config"]
    two, three = (geometry.mla_cache_bytes(text, world, 64, latent=False) for world in (2, 3))
    assert two - three == (11 + 1) * 64 * (32 - 22) * (256 + 256) * 2          # 11 DSA layers and the MTP layer
    with capsys.disabled():
        print("\nGLM-5.3-Flash NVFP4, ranks of three (startup estimate of the weights, GiB): " +
              " / ".join(f"{t / 2 ** 30:.2f}" for t in totals))


def test_each_split_tensor_names_its_axis():
    """Heads by the layer's kind (the MTP layer's attention is DSA), widths and the vocabulary in 64-element units;
    a model whose counts all differ, so a tensor read against the wrong one shows."""

    from types import SimpleNamespace

    cfg = SimpleNamespace(heads=4, lin_heads=6, layers=2, kinds=["kda", "dsa"], moe_width=640, shared_width=768,
                          dense_width=896, vocab=1024)
    plan = split.ShardPlan(cfg, 2, 0)
    L = "model.language_model.layers."
    want = {L + "1.self_attn.q_b_proj.weight": (4, 1), L + "1.self_attn.kv_b_proj.scales": (4, 1),
            L + "1.self_attn.o_proj.weight": (4, 1), L + "2.self_attn.o_proj.weight": (4, 1),
            L + "0.self_attn.o_proj.weight": (6, 1), L + "0.self_attn.q_proj.weight": (6, 1),
            L + "0.self_attn.v_conv1d.weight": (6, 1), L + "0.self_attn.g_b_proj.weight": (6, 1),
            L + "0.self_attn.b_proj.biases": (6, 1), L + "0.self_attn.A_log": (6, 1), L + "0.self_attn.dt_bias": (6, 1),
            L + "1.mlp.experts.5.down_proj.trellis": (640, 64), L + "2.mlp.experts.0.up_proj.weight": (640, 64),
            L + "1.mlp.shared_experts.gate_proj.weight": (768, 64), L + "0.mlp.down_proj.weight_scale": (896, 64),
            "lm_head.scales": (1024, 64)}
    for name, axis in want.items():
        assert split.rule(name) in ("row", "col", "dim1") or name.startswith("lm_head."), name
        assert plan.axis(name) == axis, name
    for name in (L + "1.self_attn.q_a_proj.weight", L + "0.self_attn.f_a_proj.weight", "lm_head"):
        with pytest.raises(ValueError):
            plan.axis(name)
