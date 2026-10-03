"""GLM-5.3-Flash's ranks on the communicator interface (tensorfold.cuda.comm, #219): every rank count opens through
``open_comm``, and ``send_recv`` (a prompt chunk's exchanges between any ranks) is an optional capability like
``exchange``, so a transport over NCCL keeps it and the split default."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import comm  # noqa: E402

TEXT = {"hidden_size": 512, "num_hidden_layers": 2, "vocab_size": 1280, "rms_norm_eps": 1e-5,
        "num_attention_heads": 4, "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0, "v_head_dim": 256,
        "linear_attn_config": {"num_heads": 4, "head_dim": 128, "short_conv_kernel_size": 4},
        "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 512, "n_shared_experts": 1,
        "intermediate_size": 384, "routed_scaling_factor": 2.5, "layer_types": ["linear_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse"], "eos_token_id": [1000], "num_nextn_predict_layers": 1}


class Opened(Exception):
    pass


@pytest.mark.parametrize("world, rank", [(2, 1), (3, 2)])
def test_every_rank_count_opens_through_open_comm(tmp_path, monkeypatch, world, rank):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    opened = []

    def open_comm(*a, **k):
        opened.append((a, k))
        raise Opened

    monkeypatch.setattr(comm, "open_comm", open_comm)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.decode", SimpleNamespace(Engine=None))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": TEXT,
                                                      "quantization": {"bits": 4, "group_size": 64}}))
    with pytest.raises(Opened):
        GlmEngine(tmp_path, rank=rank, master="10.0.0.1", port=29551, world=world)
    assert opened == [((rank, world, "10.0.0.1", 29551), {})]


class Gathers:
    rank, world = 0, 3

    def all_gather(self, send, recv):
        pass

    def barrier(self):
        pass


def test_a_transport_keeps_nccls_send_recv_and_the_split_default():
    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import reduce

    seen = []
    nccl = Gathers()
    nccl.send_recv = lambda sends, recvs: seen.append((sends, recvs))
    fast = comm.Transport(nccl)
    fast.send_recv([("a", 1)], [("b", 2)])
    assert seen == [([("a", 1)], [("b", 2)])]
    assert reduce.settings({}, comm=fast) == "split"
    assert reduce.settings({}, comm=comm.Transport(Gathers())) == "gather"     # no point-to-point: gathered


def test_three_ranks_name_the_same_backend(monkeypatch):
    keys = {"tf_comm_backend/1": b"nccl", "tf_comm_backend/2": b"fast"}
    store = SimpleNamespace(set=lambda k, v: keys.__setitem__(k, v.encode()), get=lambda k: keys[k])
    monkeypatch.setattr(comm, "NCCL", lambda *a, **k: SimpleNamespace(rank=0, world=3, store=store))
    with pytest.raises(RuntimeError, match="different TF_COMM_BACKEND: nccl, nccl, fast"):
        comm.open_comm(0, 3, "192.0.2.1", 29551)
