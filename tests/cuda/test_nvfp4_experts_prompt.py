"""Grouped NVFP4 experts, prompt form: each pair's output is its expert's product (exact bf16 weights, one fp32 chain),
within the decode kernel's distance of an fp64 reference, and the same bits whatever the chunk or the other pairs."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4 import format as fmt

E, D, NI, TOP = 6, 256, 96, 3     # NI = 96: a part-filled column tile (3 of 4 blocks) and an odd K for down (3 blocks)


def _proj(n: int, k: int, g: torch.Generator):
    words = torch.randint(0, 256, (E, n, k // 2), generator=g, dtype=torch.uint8)
    scales = torch.randint(0x28, 0x40, (E, n, k // 16), generator=g, dtype=torch.uint8)
    glob = torch.rand(E, generator=g) * 0.02 + 0.005
    return words, scales, glob


def _dense(p, e: int) -> torch.Tensor:
    w, s, g = p
    return torch.from_numpy(fmt.dequant("nvfp4", w[e].numpy(), s[e].numpy(), float(g[e]))).double()


@pytest.fixture(scope="module")
def layer():
    g = torch.Generator().manual_seed(21)
    gate, up, down = _proj(NI, D, g), _proj(NI, D, g), _proj(D, NI, g)
    ex = nvx.make(*[tuple(t.cuda() for t in p) for p in (gate, up, down)])
    return ex, gate, up, down


def _picks(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    routed = torch.stack([torch.randperm(E, generator=g)[:TOP] for _ in range(rows)])
    return torch.cat([routed, torch.full((rows, 1), E)], dim=1).to(torch.int32).cuda()


def _plan(picks: torch.Tensor, prefill: bool = True) -> grouped.Plan:
    plan = grouped.Plan(picks.shape[0], picks.shape[1], E + 1, "cuda", prefill=prefill)
    grouped.route(picks.contiguous(), plan, nvx.STAGED_TILE if prefill else nvx.PREFILL_TILE)
    return plan


def _gate_up(ex, x, picks, prompt=True):
    out = torch.zeros((x.shape[0] * (TOP + 1), NI), dtype=torch.bfloat16, device="cuda")
    (nvx.prompt_gate_up if prompt else nvx.gate_up)(x, ex, _plan(picks, prompt), out, x.shape[0], skip=E)
    return out.view(x.shape[0], TOP + 1, NI)


def _down(ex, act, picks, prompt=True, dtype=torch.float32):
    out = torch.zeros((act.shape[0], D), dtype=dtype, device="cuda")
    (nvx.prompt_down if prompt else nvx.down)(act, ex, _plan(picks, prompt), out, picks.shape[0], skip=E)
    return out.view(picks.shape[0], TOP + 1, D)


def _rows(n: int, k: int, seed: int) -> torch.Tensor:
    return (torch.randn(n, k, generator=torch.Generator().manual_seed(seed)) * 0.5).to(torch.bfloat16).cuda()


R = 300                           # ~150 pairs an expert: several items each, the last part-filled


def test_gate_up_is_each_pairs_swiglu_within_the_decode_kernels_distance(layer):
    ex, gate, up, _ = layer
    x, picks = _rows(R, D, 1), _picks(R, 2)
    act, dec = _gate_up(ex, x, picks), _gate_up(ex, x, picks, prompt=False)
    assert torch.equal(act[:, TOP], torch.zeros_like(act[:, TOP]))                   # the shared slot untouched
    for r in range(0, R, 23):
        for s in range(TOP):
            e = int(picks[r, s])
            gv = (x[r].double().cpu() @ _dense(gate, e).t()).float().to(torch.bfloat16).float()
            uv = (x[r].double().cpu() @ _dense(up, e).t()).float().to(torch.bfloat16).float()
            want = (gv / (1 + torch.exp(-gv))).to(torch.bfloat16).float() * uv
            err = float((act[r, s].float().cpu() - want).abs().max() / want.abs().max())
            assert err < 2e-2, (r, s, err)
    # bf16 roundings of g and u decide most of the gap: the two kernels' rows differ in few places, by bf16 steps
    near = (act[:, :TOP].float() - dec[:, :TOP].float()).abs() <= 2 ** -6 * dec[:, :TOP].float().abs().amax()
    assert bool(near.all())


def test_down_is_each_pairs_product_no_further_than_the_decode_kernels(layer):
    ex, _, _, down = layer
    picks = _picks(R, 3)
    act = _rows(R * (TOP + 1), NI, 4)
    y, dec = _down(ex, act, picks), _down(ex, act, picks, prompt=False)
    rows = act.view(R, TOP + 1, NI)
    err, old = [], []
    for r in range(0, R, 17):
        for s in range(TOP):
            want = rows[r, s].double().cpu() @ _dense(down, int(picks[r, s])).t()
            scale = float(want.abs().max())
            err.append(float((y[r, s].double().cpu() - want).abs().max()) / scale)
            old.append(float((dec[r, s].double().cpu() - want).abs().max()) / scale)
    assert max(err) < 1e-5 and max(err) <= 2 * max(old), (max(err), max(old))
    assert torch.equal(y[:, TOP], torch.zeros_like(y[:, TOP]))
    yb = _down(ex, act, picks, dtype=torch.bfloat16)
    assert torch.equal(yb[:, :TOP], y[:, :TOP].to(torch.bfloat16))


@pytest.mark.parametrize("cuts", [[R], [100, 200], [37] * 8 + [4], [1] * 5 + [295]],
                         ids=["whole", "100+200", "37s", "ones"])
def test_a_pairs_bits_never_depend_on_the_chunk(layer, cuts):
    ex = layer[0]
    x, picks = _rows(R, D, 5), _picks(R, 6)
    act = _gate_up(ex, x, picks)
    y = _down(ex, act.view(-1, NI), picks)
    r0 = 0
    for n in cuts:
        part = _gate_up(ex, x[r0:r0 + n].contiguous(), picks[r0:r0 + n])
        assert torch.equal(part[:, :TOP], act[r0:r0 + n, :TOP]), (cuts, r0)
        yp = _down(ex, part.reshape(-1, NI).contiguous(), picks[r0:r0 + n])
        assert torch.equal(yp[:, :TOP], y[r0:r0 + n, :TOP]), (cuts, r0)
        r0 += n


def test_a_pairs_bits_never_depend_on_the_other_pairs(layer):
    ex = layer[0]
    x, picks = _rows(R, D, 7), _picks(R, 8)
    act = _gate_up(ex, x, picks)
    order = torch.randperm(R, generator=torch.Generator().manual_seed(9)).cuda()
    other = _rows(R, D, 10)
    other[order[:50]] = x[order[:50]]                                          # 50 rows kept, the rest new, reordered
    perm = torch.cat([order[:50], order[50:]])
    got = _gate_up(ex, other[perm].contiguous(), picks[perm].contiguous())
    assert torch.equal(got[:50, :TOP], act[order[:50], :TOP])


def test_the_prompt_kernel_takes_only_a_prompt_plan_in_its_items(layer):
    ex = layer[0]
    x, picks = _rows(4, D, 11), _picks(4, 12)
    out = torch.zeros((4 * (TOP + 1), NI), dtype=torch.bfloat16, device="cuda")
    for plan in (_plan(picks, prefill=False), grouped.Plan(4, TOP + 1, E + 1, "cuda", prefill=True)):
        if plan.prefill:
            grouped.route(picks, plan, nvx.PREFILL_TILE)
        with pytest.raises(ValueError):
            nvx.prompt_gate_up(x, ex, plan, out, 4, skip=E)
