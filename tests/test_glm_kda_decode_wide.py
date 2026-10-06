"""TF_GLM_KDA_DECODE_WIDE: decode windows run KDA's three-kernel chain unless it is 0; other values are refused."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.glm5_next.cuda.kda import decode_wide


def test_the_switch_is_on_unless_zero():
    assert decode_wide({}) and decode_wide({"TF_GLM_KDA_DECODE_WIDE": ""})
    assert decode_wide({"TF_GLM_KDA_DECODE_WIDE": " 1 "})
    assert not decode_wide({"TF_GLM_KDA_DECODE_WIDE": "0"})


@pytest.mark.parametrize("value", ["2", "on", "true", "off"])
def test_other_values_are_refused(value):
    with pytest.raises(ValueError, match="TF_GLM_KDA_DECODE_WIDE is 0 or 1"):
        decode_wide({"TF_GLM_KDA_DECODE_WIDE": value})
