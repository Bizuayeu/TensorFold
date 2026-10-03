"""TF_GLM_PREFILL_REDUCE (default gather): how a prompt chunk's ranks sum their fp32 partials at each exchange site.

gather: every rank all-gathers every rank's partial b.part[:R] ([world, R, D] fp32); the consumer (hc_post, or the
MTP block's residual add) sums them rank 0 first and rounds the branch to bf16.
scatter: an exact reduce-scatter. Rank j owns rows ``shares(R, world)[j]``; each rank sends every owner its rows of
the partial, the owner sums them rank 0 first and rounds to bf16 in one kernel (``glue.rank_sum``, the consumer's own
expression), then sends its summed rows to every rank. Each rank widens the summed rows back to fp32 into slot 0 of
gather's [world, R, D] layout and fills the other slots with -0.0, so the consumer runs gather's very kernel (same
WORLD, same slot stride) and adds x + (-0.0) == x: every bit is gather's. (hc_post compiled for one slot instead
rounds a few outputs in a million differently on GB10.) A rank of ``world`` owning ``n`` rows
receives (world - 1) x n x D x 4 + (R - n) x D x 2 bytes instead of (world - 1) x R x D x 4: half at three ranks,
three quarters at two. A chunk (or a piece of one) with fewer rows than ranks gathers; decode windows and the MTP
head's draft windows never use a prompt buffer.

The exact reduce-scatter (each rank adds its share in rank order, then the sums are gathered) follows ashhart/
TensorFold PR #159's all-to-all and rank-order sum (drowzeys, Apache-2.0, commit 028698c); written for this tree,
by rows over point-to-point sends (D = 4,096 does not split in three) and with the sums sent as bf16."""

from __future__ import annotations

import os

import torch

from . import glue

MODES = ("gather", "scatter")


def settings(env=None) -> str:
    """The mode from TF_GLM_PREFILL_REDUCE (gather or scatter)."""

    env = os.environ if env is None else env
    mode = str(env.get("TF_GLM_PREFILL_REDUCE", "") or "gather").strip()
    if mode not in MODES:
        raise ValueError(f"TF_GLM_PREFILL_REDUCE: gather or scatter, not {mode!r}")
    return mode


def shares(R: int, world: int) -> list[tuple[int, int]]:
    """[lo, hi) rows each rank owns, in rank order: about equal, the first R % world one row more."""

    base, extra = divmod(R, world)
    out, lo = [], 0
    for j in range(world):
        hi = lo + base + (j < extra)
        out.append((lo, hi))
        lo = hi
    return out


def sums(comm, part: torch.Tensor, room: torch.Tensor, scatter: bool) -> torch.Tensor:
    """Every rank's partials of ``part`` [R, D] (fp32, contiguous) for the consumer, [world, R, D] fp32 in ``room``
    (world x R x D): gathered, or with ``scatter`` (and R >= world) the sums in slot 0 and -0.0 in the others.

    Scatter's scratch lives in ``room`` ahead of the slots that overwrite it: the peers' partials of this rank's rows
    ([world - 1, ceil(R / world), D], at most R x D) in slot 0, the summed bf16 rows at the start of slot 1."""

    R, d = part.shape
    world = comm.world
    if not scatter or R < world:
        out = room[:world * R * d]
        comm.all_gather(part.reshape(-1), out)
        return out.view(world, R, d)
    cut = shares(R, world)
    me = comm.rank
    lo, hi = cut[me]
    n = hi - lo
    slot = -(-R // world)                   # recv's rows per peer: the largest share
    recv = room[:(world - 1) * slot * d].view(world - 1, slot, d)
    half = room[R * d:R * d + R * d // 2].view(torch.bfloat16).view(R, d)
    peers = [j for j in range(world) if j != me]
    comm.send_recv([(part[a:b], j) for j, (a, b) in enumerate(cut) if j != me],
                   [(recv[i, :n], j) for i, j in enumerate(peers)])
    glue.rank_sum(part[lo:hi], recv[:, :n], me, half[lo:hi])
    comm.send_recv([(half[lo:hi], j) for j in peers], [(half[a:b], j) for j, (a, b) in enumerate(cut) if j != me])
    out = room[:world * R * d].view(world, R, d)
    out[0].copy_(half)
    # cc-defer: -0.0 written per exchange (R x D x 4 x (world - 1) bytes); if the three machines' profile shows it,
    # keep the slots -0.0 for good and give the scratch its own buffer (counted in the startup estimate)
    room[R * d:world * R * d].fill_(-0.0)
    return out
