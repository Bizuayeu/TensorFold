"""GLM's CUDA settings on CPU: TF_GLM_KV (``kv_kind``), the prompt overlap's TF_GLM_PREFILL_OVERLAP and
TF_GLM_OVERLAP_PIECES (``overlap.settings``), and the row pieces it cuts a prompt chunk into (``overlap.ranges``)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from tensorfold.families.glm5_next.cuda import kv_kind


def test_kv_kind():
    for value in ("", " ", "bf16", " bf16 "):
        assert kv_kind({"TF_GLM_KV": value}) == "bf16", value
    assert kv_kind({}) == "bf16" and kv_kind({"TF_GLM_KV": " fp8 "}) == "fp8"
    for bad in ("FP8", "fp16", "int8", "0"):
        with pytest.raises(ValueError, match="TF_GLM_KV"):
            kv_kind({"TF_GLM_KV": bad})


def test_kv_kind_reads_the_environment(monkeypatch):
    monkeypatch.setenv("TF_GLM_KV", "fp8")
    assert kv_kind() == "fp8"
    monkeypatch.delenv("TF_GLM_KV")
    assert kv_kind() == "bf16"


def _overlap():
    pytest.importorskip("triton")                 # overlap imports the glue's triton kernels
    from tensorfold.families.glm5_next.cuda import overlap

    return overlap


def test_overlap_settings():
    overlap = _overlap()
    most = overlap.PIECES_MAX
    assert overlap.settings({}) == (True, 4)
    assert overlap.settings({"TF_GLM_PREFILL_OVERLAP": "", "TF_GLM_OVERLAP_PIECES": ""}) == (True, 4)
    assert overlap.settings({"TF_GLM_PREFILL_OVERLAP": " 0 ", "TF_GLM_OVERLAP_PIECES": " 1 "}) == (False, 1)
    assert overlap.settings({"TF_GLM_OVERLAP_PIECES": str(most)}) == (True, most)
    for bad in ("2", "on", "true", "-1"):
        with pytest.raises(ValueError, match="TF_GLM_PREFILL_OVERLAP"):
            overlap.settings({"TF_GLM_PREFILL_OVERLAP": bad})
    for bad in ("0", str(most + 1), "-1", "4.0", "four"):
        with pytest.raises(ValueError, match=f"TF_GLM_OVERLAP_PIECES: a whole number from 1 to {most}"):
            overlap.settings({"TF_GLM_OVERLAP_PIECES": bad})


def test_overlap_ranges():
    overlap = _overlap()
    assert overlap.ROW_STEP == 128
    assert overlap.ranges(2048, 4) == [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
    assert overlap.ranges(2000, 3) == [(0, 768), (768, 1536), (1536, 2000)]
    assert overlap.ranges(300, 4) == [(0, 128), (128, 256), (256, 300)]           # fewer pieces than asked
    assert overlap.ranges(100, 4) == [(0, 100)] and overlap.ranges(2048, 1) == [(0, 2048)]
    for R in range(1, 2049, 37):
        for pieces in range(1, overlap.PIECES_MAX + 1):
            cut = overlap.ranges(R, pieces)
            assert 1 <= len(cut) <= pieces and cut[0][0] == 0 and cut[-1][1] == R, (R, pieces)
            assert all(a[1] == b[0] and a[1] % overlap.ROW_STEP == 0 for a, b in pairwise(cut)), (R, pieces)
