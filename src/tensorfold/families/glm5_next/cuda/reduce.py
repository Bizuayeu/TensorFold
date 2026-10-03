"""TF_GLM_PREFILL_REDUCE (default gather): how a prompt chunk's ranks sum their fp32 partials at each exchange site,
and with split, which rows each rank glues.

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
split: scatter's first half, then each rank glues only the rows it owns and the glued rows are gathered (``overlap``
drives it at the hyper-connection exchanges; the MTP block's residual adds take scatter). ``owned`` hands the owner
every rank's partials of its rows as [world, n, D] in rank order (its own copied into its slot), which hc_post sums
with gather's very kernel (WORLD = world, rows independent), so the owner's rows carry gather's bits; hc_post and the
next hc_pre (or the stream means) then run over n rows, not R, and ``share`` sends the owner's rows of their outputs
(the bf16 normed rows the next block reads, D x 2 bytes a row) to every rank. Between exchanges a rank's streams,
post and comb are current on its own rows only; each piece of a chunk keeps the same owners at every exchange. The
64-group sums hc_pre writes beside the normed rows are not shared: no prompt matmul reads them. A rank receives
(world - 1) x n x D x 4 bytes and then (R - n) x D x 2, scatter's count, and skips scatter's widening and -0.0 fill.

The exact reduce-scatter (each rank adds its share in rank order, then the sums are gathered) follows ashhart/
TensorFold PR #159's all-to-all and rank-order sum (drowzeys, Apache-2.0, commit 028698c); written for this tree,
by rows over point-to-point sends (D = 4,096 does not split in three) and with the sums sent as bf16. Gluing each
rank's own rows of the reduce-scatter follows MiaAI-Lab's patch 0010-glm-hc-split for TensorFold v0.6.0 (Apache-2.0),
with shares of uneven size: their 0067/0068 found a 2,048-row chunk at three ranks (3 x 683 = 2,049) running unsplit
without a word."""

from __future__ import annotations

import os

import torch

from . import glue

MODES = ("gather", "scatter", "split")


def settings(env=None) -> str:
    """The mode from TF_GLM_PREFILL_REDUCE (gather, scatter or split)."""

    env = os.environ if env is None else env
    mode = str(env.get("TF_GLM_PREFILL_REDUCE", "") or "gather").strip()
    if mode not in MODES:
        raise ValueError(f"TF_GLM_PREFILL_REDUCE: gather, scatter or split, not {mode!r}")
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


def owned(comm, part: torch.Tensor, room: torch.Tensor) -> torch.Tensor:
    """split's first half: every rank's partials of this rank's rows ``shares(R, world)[rank]`` of ``part`` [R, D]
    (fp32, contiguous, R >= world), [world, n, D] in rank order in ``room`` (world x R x D), for hc_post."""

    R, d = part.shape
    world, me = comm.world, comm.rank
    cut = shares(R, world)
    lo, hi = cut[me]
    got = room[:world * (hi - lo) * d].view(world, hi - lo, d)
    comm.send_recv([(part[a:b], j) for j, (a, b) in enumerate(cut) if j != me],
                   [(got[j], j) for j in range(world) if j != me])
    got[me].copy_(part[lo:hi])
    return got


def share(comm, outs: list[torch.Tensor]) -> None:
    """split's second half: each rank sends its rows ``shares(R, world)[rank]`` of every [R, ...] tensor of ``outs``
    (rows contiguous) to every other rank and takes theirs into their rows."""

    world, me = comm.world, comm.rank
    cut = shares(outs[0].shape[0], world)
    lo, hi = cut[me]
    comm.send_recv([(t[lo:hi], j) for j in range(world) if j != me for t in outs],
                   [(t[a:b], j) for j, (a, b) in enumerate(cut) if j != me for t in outs])
