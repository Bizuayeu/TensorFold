"""GLM-5.3-Flash on two or three ranks, before any GPU work (any machine): the CLI takes ``--tp`` from the family's
CUDA_TP, ranks past the first follow, the engine refuses what only two ranks run (EXL3 experts, the DFlash2 drafter),
and the startup estimate refuses a rank folder split for another rank count."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tensorfold import cli

pytest.importorskip("torch")                     # the engine's Config lives in weights.py, which imports torch

TEXT = {"hidden_size": 512, "num_hidden_layers": 2, "vocab_size": 1280, "rms_norm_eps": 1e-5,
        "num_attention_heads": 4, "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0, "v_head_dim": 256,
        "linear_attn_config": {"num_heads": 4, "head_dim": 128, "short_conv_kernel_size": 4},
        "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 512, "n_shared_experts": 1,
        "intermediate_size": 384, "routed_scaling_factor": 2.5, "layer_types": ["linear_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse"], "eos_token_id": [1000], "num_nextn_predict_layers": 1}


def _family(**members):
    return SimpleNamespace(title="Test family", model_type="test", package=SimpleNamespace(**members))


def test_serve_parses_three_ranks():
    args = cli.build_parser().parse_args(["serve", "owner/model", "--tp", "3", "--rank", "2", "--master", "10.0.0.1"])
    assert (args.tp, args.rank, args.master) == (3, 2, "10.0.0.1")


def test_ranks_past_the_first_follow(tmp_path, monkeypatch, capsys):
    from tensorfold.cuda import precision, prompt_precision

    monkeypatch.setattr(precision, "set_mode", lambda *a, **k: None)
    monkeypatch.setattr(prompt_precision, "set_fp8", lambda *a: None)
    made, followed = [], []
    engine = SimpleNamespace(follow=lambda: followed.append(True))
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine, CUDA_TP=(2, 3))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--tp", "3",
                                          "--rank", "2", "--master", "10.0.0.1"])
    assert cli._serve_cuda(args, family, tmp_path, 4096) == 0
    assert (made[0]["tp"], made[0]["rank"]) == (3, 2) and followed == [True]
    out = capsys.readouterr().out
    assert "rank 2 of 3" in out and "rank 2 ready" in out


@pytest.mark.parametrize("tp, rank, master, match", [
    (3, 0, "10.0.0.1", "--tp 1 or 2, not 3"),            # a family without CUDA_TP: one or two ranks
    (3, 0, "", "--tp 3 needs --master"),
    (2, 2, "10.0.0.1", "--rank 2 needs --tp 3 or more"),
])
def test_rank_counts_the_family_does_not_run_are_refused_before_loading(tmp_path, tp, rank, master, match):
    made = []
    family = _family(cuda_engine=lambda *a, **k: made.append(k))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--tp", str(tp),
                                          "--rank", str(rank)] + (["--master", master] if master else []))
    with pytest.raises(ValueError, match=match):
        cli._serve_cuda(args, family, tmp_path)
    assert not made


def test_glm_runs_on_two_or_three_ranks(tmp_path, monkeypatch):
    from tensorfold.families import glm5_next
    from tensorfold.families.glm5_next.cuda import engine as glm_engine

    assert glm5_next.CUDA_TP == (2, 3)
    made = []
    monkeypatch.setattr(glm_engine, "GlmEngine", lambda *a, **k: made.append(k) or SimpleNamespace(**k))
    assert glm5_next.cuda_engine(tmp_path, tp=3, rank=2, master="10.0.0.1").world == 3
    assert glm5_next.cuda_engine(tmp_path, tp=2, rank=1, master="10.0.0.1").world == 2
    for tp in (1, 4):
        with pytest.raises(ValueError, match="--tp 2 or 3"):
            glm5_next.cuda_engine(tmp_path, tp=tp, master="10.0.0.1")
    assert len(made) == 2


def _config(path, **quant):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": TEXT, **quant}))
    return path


@pytest.mark.parametrize("world", [3, 4])
def test_exl3_and_dflash2_run_on_two_ranks_only(tmp_path, world):
    """Refused before any device or rank exchange, with the reason: EXL3 experts split in 128-wide Hadamard blocks, and
    the DFlash2 drafter halves its heads and MLP."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    exl3 = _config(tmp_path / "exl3", quantization_config={"quant_method": "exl3", "bits": 4, "codebook": "mcg"})
    with pytest.raises(ValueError, match="EXL3.*two ranks.*128-wide Hadamard"):
        GlmEngine(exl3, rank=0, master="", port=0, world=world, comm=object())
    mlx = _config(tmp_path / "mlx", quantization={"bits": 4, "group_size": 64, "mode": "affine"})
    with pytest.raises(ValueError, match="DFlash2.*two ranks.*--drafter none"):
        GlmEngine(mlx, rank=0, master="", port=0, world=world, drafter=tmp_path, comm=object())


def _rank_file(path, metadata):
    from tensorfold.families.glm5_next.cuda import split

    path.mkdir(parents=True, exist_ok=True)
    import numpy as np

    split.write(str(path / "model-00001-of-00001.rank0.safetensors"),
                [("lm_head.weight", "BF16", [4, 8], np.zeros(64, dtype=np.uint8))], metadata)
    return path


@pytest.mark.parametrize("metadata, world, found", [(None, 3, 2), ({"tensorfold_world": "2"}, 3, 2),
                                                    ({"tensorfold_world": "3"}, 2, 3)])
def test_the_estimate_refuses_a_rank_folder_split_for_another_rank_count(tmp_path, metadata, world, found):
    from tensorfold.cuda import capacity

    folder = _rank_file(tmp_path, metadata)
    with pytest.raises(ValueError, match=f"split for {found} ranks, not {world}"):
        capacity.headers(folder, rank=0, world=world)
    with pytest.raises(ValueError, match=f"split for {found} ranks, not {world}"):
        capacity.estimate_weights(folder, lambda name, info: (0, 0), rank=0, world=world)
    assert "lm_head.weight" in capacity.headers(folder, rank=0, world=found)
    assert "lm_head.weight" in capacity.headers(folder, rank=0)                  # no rank count asked: not checked
