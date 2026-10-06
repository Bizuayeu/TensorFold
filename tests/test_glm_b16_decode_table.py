"""TF_GLM_B16_DECODE_TABLE: decode windows' BF16 matmuls take the per-shape tiles unless it is 0; other values are
refused."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.torch

from tensorfold.families.glm5_next.cuda.qmm import decode_table


def test_the_switch_is_on_unless_zero():
    assert decode_table({}) and decode_table({"TF_GLM_B16_DECODE_TABLE": ""})
    assert decode_table({"TF_GLM_B16_DECODE_TABLE": " 1 "})
    assert not decode_table({"TF_GLM_B16_DECODE_TABLE": "0"})


@pytest.mark.parametrize("value", ["2", "on", "true", "off"])
def test_other_values_are_refused(value):
    with pytest.raises(ValueError, match="TF_GLM_B16_DECODE_TABLE is 0 or 1"):
        decode_table({"TF_GLM_B16_DECODE_TABLE": value})
