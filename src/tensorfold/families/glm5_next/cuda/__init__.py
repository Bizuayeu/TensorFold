"""CUDA verification preserves serial bits with row-invariant kernels and fp32 partials gathered and summed in rank order."""

import os

LATENT = os.environ.get("TF_GLM_LATENT", "1") != "0"   # DSA caches its 512-wide latent unless TF_GLM_LATENT=0
KV_KINDS = ("bf16", "fp8")


def kv_kind(env=None) -> str:
    """TF_GLM_KV: ``bf16`` (the default) or ``fp8``, the format of the DSA latent cache and the indexer's pooled keys
    (``kv8``). fp8 is lossy (replies differ from bf16's); drafted replies still equal serial ones."""

    kind = (os.environ if env is None else env).get("TF_GLM_KV", "bf16").strip() or "bf16"
    if kind not in KV_KINDS:
        raise ValueError(f"TF_GLM_KV is bf16 or fp8, not {kind!r}")
    return kind
