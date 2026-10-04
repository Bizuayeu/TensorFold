"""GLM-5.3-Flash image prompts on the CUDA engine, on the tiny synthetic checkpoint (one GPU as rank 0 of two).

Image rows whose features are the replaced tokens' own embeddings must leave the text prompt's state, MTP cache
and reply bit for bit: the rows reach the main model and the MTP head, chunk boundaries included, and nothing else
changes. An image prompt never resumes or keeps a prompt state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import D, _checkpoint, _forget, _generate, _state, _TwoCopies  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.vision.qwen_cuda import EncodedVision  # noqa: E402

IMAGE = 999                                   # a placeholder id no prompt below otherwise holds
ROWS = tuple(range(20, 45)) + tuple(range(60, 63))      # two images across 16-row prefill chunks


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_vision")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies())


def _prompts():
    text = [int(t) for t in np.random.default_rng(9).integers(0, 900, size=100)]
    image = [IMAGE if i in ROWS else t for i, t in enumerate(text)]
    return text, image


def _features(engine, tokens) -> torch.Tensor:
    """The embedding rows of ``tokens``: what the tower would have to return to stand in for them."""

    from tensorfold.families.glm5_next.cuda import glue

    ids = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    out = torch.empty((len(tokens), D), dtype=torch.bfloat16, device="cuda")
    return glue.embed(ids, engine.w.embed, D, 1, out)


def test_image_rows_reach_the_model_and_the_mtp_head(engine):
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    text, image = _prompts()
    payload = EncodedVision(ROWS, _features(engine, [text[i] for i in ROWS]), None, 0)
    ref = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=16)
    first = prefill(ref, text, None)
    want = [t.clone() for t in _state(ref)]
    e = Engine(engine.w, capacity=2560, max_rows=8, prefill_rows=16)
    assert prefill(e, image, None, vision=payload) == first
    assert all(torch.equal(a, b) for a, b in zip(_state(e), want))
    prefill(e, image, None)                                       # the placeholders' own embeddings differ
    assert not all(torch.equal(a, b) for a, b in zip(_state(e), want))
    with pytest.raises(ValueError):
        prefill(e, image, None, vision=payload, keep_at=50, keep=lambda snap: None)


@pytest.mark.parametrize("sampling", [Sampling(77, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_an_image_request_replies_as_its_text_twin_and_keeps_no_state(engine, sampling):
    text, image = _prompts()
    features = _features(engine, [text[i] for i in ROWS])
    encoded = []
    engine.vision = SimpleNamespace(encode=lambda prepared, prompt: encoded.append(prompt) or
                                    EncodedVision(ROWS, features, None, 0))
    engine.image_token = IMAGE
    try:
        _forget(engine)
        want, _ = _generate(engine, text, sampling)
        kept = [list(c.ids) for c in engine.cache]
        assert kept                                               # a text prompt is kept for its resend
        out: list[int] = []
        engine.request.policy, engine.request.stop_eos = None, False
        stats = engine.generate(list(image), 24, sampling, out.extend, vision=object())
        assert out == want and stats["cached"] == 0 and encoded == [image]
        assert [list(c.ids) for c in engine.cache] == kept        # no image prompt state is kept
        again, stats = _generate(engine, text, sampling)           # the text prompt still resumes, same reply
        assert again == want and stats["cached"] > 0
    finally:
        engine.vision = engine.image_token = None
        _forget(engine)


LONG = 2_400                                  # past the 2,051-token dense limit
LONG_ROWS = tuple(range(1_990, 2_110)) + tuple(range(2_300, 2_320))   # across a 256-row chunk and the dense limit


@pytest.fixture(scope="module")
def engine_long(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_vision_long")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=2_600, prefill_rows=256)


@pytest.mark.parametrize("sampling", [Sampling(31, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_a_long_image_prompt_replies_as_its_text_twin(engine_long, sampling):
    text = [int(t) for t in np.random.default_rng(13).integers(0, 900, size=LONG)]
    image = [IMAGE if i in LONG_ROWS else t for i, t in enumerate(text)]
    features = _features(engine_long, [text[i] for i in LONG_ROWS])
    engine_long.vision = SimpleNamespace(encode=lambda prepared, prompt: EncodedVision(LONG_ROWS, features, None, 0))
    engine_long.image_token = IMAGE
    try:
        _forget(engine_long)
        want, _ = _generate(engine_long, text, sampling, draft=False)
        _forget(engine_long)
        out: list[int] = []
        engine_long.request.policy, engine_long.request.stop_eos = None, False
        stats = engine_long.generate(list(image), 24, sampling, out.extend, vision=object())
        assert out == want and stats["cached"] == 0
    finally:
        engine_long.vision = engine_long.image_token = None
        _forget(engine_long)


def test_a_server_without_the_tower_refuses_images(engine):
    with pytest.raises(ValueError, match="--vision"):
        engine.generate([1, 2, 3], 4, None, lambda new: None, vision=object())


# -- three ranks (test_glm_tp3's threads on one GPU) -----------------------------------------------------------------
from test_glm_tp3 import LONG as TP3_LONG  # noqa: E402
from test_glm_tp3 import SHAPE, Ranks, _caches, _equal  # noqa: E402
from threadcomm import run_ranks  # noqa: E402

TP3_CASES = [("nvfp4", None, "bf16", "split"), ("nvfp4", TP3_LONG, "fp8", "split"), ("mlx", None, "fp8", "scatter"),
             ("mlx", None, "bf16", "gather")]
# GLM-5.3-Flash's vision settings as far as the language ranks read them (no tower: rank 0's is stood in for)
TINY_VISION = {"hidden_size": 8, "out_hidden_size": D, "depth": 1, "patch_size": 2, "temporal_patch_size": 2,
               "spatial_merge_size": 2, "in_channels": 3, "intermediate_size": 12, "num_heads": 2,
               "projection_intermediate_size": 20}


@pytest.fixture(scope="module")
def tp3_checkpoints(tmp_path_factory):
    import json

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    out = {}
    for quant in ("mlx", "nvfp4"):
        path = tmp_path_factory.mktemp(f"glm_vision_tp3_{quant}")
        _checkpoint(path, nvfp4=quant == "nvfp4", **SHAPE)
        config = json.loads((path / "config.json").read_text())
        (path / "config.json").write_text(json.dumps({**config, "image_token_id": IMAGE, "vision_config": TINY_VISION}))
        (path / "processor_config.json").write_text(json.dumps({"image_processor": {"max_image_tokens": 8000}}))
        out[quant] = path
        # one thread loads and compiles the kernels first, not three at once
        e = GlmEngine(path, rank=0, master="", port=0, world=3, comm=_TwoCopies(3), context=TP3_LONG, graphs=False)
        _generate(e, list(range(2100)), None, tokens=4)
        del e
    torch.cuda.empty_cache()
    return out


@pytest.fixture(scope="module", params=TP3_CASES,
                ids=[f"{q}-{'long' if c else 'dense'}-{kv}-{red}" for q, c, kv, red in TP3_CASES])
def three(request, tp3_checkpoints):
    quant, context, kv, red = request.param
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TF_GLM_KV", kv)
        mp.setenv("TF_GLM_PREFILL_REDUCE", red)
        ranks = Ranks(tp3_checkpoints[quant], 3, **({"context": context} if context else {}))
    assert [(e.e.pbuf.scatter, e.e.pbuf.split) for e in ranks.engines] == [(red != "gather", red == "split")] * 3
    ranks.context, ranks.reduce = context, red
    yield ranks
    del ranks
    torch.cuda.empty_cache()


def _twins(three):
    """A text prompt, its image twin (placeholders over some rows) and the features that stand in for the replaced
    tokens' embeddings: images across 64- and 300-row chunks, or the 2,048-row chunk and the dense limit."""

    if three.context is None:
        size, rows = 320, tuple(range(20, 45)) + tuple(range(58, 70)) + tuple(range(295, 305))
    else:
        size, rows = 2100, tuple(range(1990, 2060)) + tuple(range(2080, 2090))
    text = [int(t) for t in np.random.default_rng(29).integers(0, 900, size=size)]
    image = [IMAGE if i in rows else t for i, t in enumerate(text)]
    return text, image, rows, _features(three.engines[0], [text[i] for i in rows])


def _ask(e, prompt, sampling, draft, vision=None):
    out: list[int] = []
    e.request.policy, e.request.stop_eos = None, False
    stats = e.generate(list(prompt), 24, sampling, out.extend, draft=draft, vision=vision)
    return out, stats


@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)], ids=["greedy", "top_k"])
def test_an_image_prompt_on_three_ranks_replies_as_its_text_twin(three, sampling):
    """Serial and drafted, an image prompt leaves its text twin's reply and caches on every rank, keeps no prompt
    state, and a kept text prompt still resumes after it, as a fresh prefill does."""

    text, image, rows, features = _twins(three)
    three.forget()
    want, _ = three.serve(lambda e: _ask(e, text, sampling, False))
    caches = three.each(lambda r, e: _caches(e.e))
    for e in three.engines:
        e.image_token = IMAGE
    three.engines[0].vision = SimpleNamespace(encode=lambda prepared, prompt: EncodedVision(rows, features, None, 0))
    try:
        for draft in (False, True):
            three.forget()
            got, stats = three.serve(lambda e, draft=draft: _ask(e, image, sampling, draft, object()))
            assert got == want and stats["cached"] == 0, draft
            assert all(_equal(a, b) for a, b in zip(caches, three.each(lambda r, e: _caches(e.e)))), draft
            assert all(not e.cache for e in three.engines), draft            # no image prompt state is kept
        three.forget()
        first, _ = three.serve(lambda e: _ask(e, text, sampling, True))
        three.serve(lambda e: _ask(e, image, sampling, True, object()))
        again, stats = three.serve(lambda e: _ask(e, text, sampling, True))
        assert first == again == want and stats["cached"] == len(text) - 1
    finally:
        for e in three.engines:
            e.vision = e.image_token = None
        three.forget()


def test_image_rows_leave_the_same_state_in_any_chunking_on_three_ranks(three, monkeypatch):
    """Every rank's first token and caches (the MTP head's too) of an image prompt, in 300-, 64- and 7-row chunks
    (past the dense limit 2,048 and 64), equal its text twin's: the features reach every rank's rows whichever rank
    glues them."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill
    from tensorfold.vision.glm_cuda import share_encoded

    text, image, rows, features = _twins(three)
    monkeypatch.setenv("TF_GLM_PREFILL_REDUCE", three.reduce)

    def run(prompt, chunk, vision):
        def fn(r, e):
            d = Engine(e.w, capacity=e.capacity_plan["cache_slots"], max_rows=8, prefill_rows=chunk,
                       long_context=e.w.meta["long_context"], kv=e.kv)
            shared = None
            if vision:
                shared = share_encoded(EncodedVision(rows, features, None, 0) if r == 0 else None, r, e.comm, prompt,
                                       IMAGE, D, "cuda", 3)
            first = prefill(d, prompt, None, vision=shared)
            st = d.st
            out = first, _caches(d) + [st.mtp_kc[:st.mtp_len].clone()]
            del d
            return out
        return three.each(fn)

    chunks = (300, 64, 7) if three.context is None else (2048, 64)
    want = run(text, chunks[0], False)
    for chunk in chunks:
        got = run(image, chunk, True)
        for rank, ((a, x), (b, y)) in enumerate(zip(want, got)):
            assert a == b and _equal(x, y), (chunk, rank)


def test_a_rank_started_without_vision_is_named(tp3_checkpoints):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    def start(r, comm):
        GlmEngine(tp3_checkpoints["mlx"], rank=r, master="", port=0, world=3, comm=comm, graphs=False,
                  vision=r == 1)

    with pytest.raises(RuntimeError, match=r"different settings.*--vision.*rank 0 \[.*\], rank 1 \["):
        run_ranks(start, 3)
