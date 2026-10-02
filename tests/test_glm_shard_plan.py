"""Each rank's share of GLM-5.3-Flash (``split.ShardPlan``) on the pinned NVFP4 checkpoint's headers (the 147,661
tensors of tests/fixtures/glm53_flash_nvfp4): two ranks take exactly the halves the two-rank split took before the plan
(byte for byte, every tensor), on bytes that number their own positions."""

from __future__ import annotations

import gzip
import json
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
