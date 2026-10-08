"""The ranks' startup comparison of their settings: the refusal names each setting that differs, with rank 0's value
and every other rank's that differs, then the rows each rank gathered."""

from __future__ import annotations

from tensorfold.families.glm5_next.cuda.engine import different_settings

NAMES = ["draft model", "TF_GLM_KV", "TF_GLM_HEAT_CEILING", "the checkpoint's NVFP4 attention"]
HINT = "; pull the draft model on every machine (or pass --drafter none to all) and give all the same flags and checkpoint"


def test_settings_alike_on_every_rank_are_not_refused():
    assert different_settings(NAMES, [[1, 0, 930, 16]] * 3) is None


def test_only_the_settings_that_differ_are_named():
    rows = [[1, 0, 930, 16], [1, 0, 930, 0], [1, 1, 930, 16]]
    assert different_settings(NAMES, rows) == (
        "the ranks were started with different settings (TF_GLM_KV 0 on rank 0, 1 on rank 2; the checkpoint's NVFP4 "
        "attention 16 on rank 0, 0 on rank 1): rank 0 [1, 0, 930, 16], rank 1 [1, 0, 930, 0], rank 2 [1, 1, 930, 16]"
        + HINT)


def test_a_setting_two_ranks_hold_otherwise_names_both():
    rows = [[1, 0, 0, 16], [1, 0, 930, 16], [1, 0, 940, 16]]
    assert different_settings(NAMES, rows) == (
        "the ranks were started with different settings (TF_GLM_HEAT_CEILING 0 on rank 0, 930 on rank 1, 940 on rank "
        "2): rank 0 [1, 0, 0, 16], rank 1 [1, 0, 930, 16], rank 2 [1, 0, 940, 16]" + HINT)
