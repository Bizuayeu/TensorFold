"""TF_GLM_PREFILL_OVERLAP (default 1): a prompt chunk's rank exchanges in row pieces on a second CUDA stream.

Unpieced, each exchange site (after the attention block, after the MLP / MoE block) writes this rank's fp32 partial
b.part[:R], all-gathers it ([world, R, D]), and hc_post and the next hc_pre run over all R rows. Pieced, the site
writes its partial a piece of rows at a time and each piece's all-gather starts on the second stream as soon as the
piece is written (into its own [world, n, D] slice of b.gath); the glue (hc_post, the DFlash2 taps, the next hc_pre)
then runs piece by piece, each piece waiting for its own gather only, while the later pieces are still on the wire.
Every kernel involved is row-independent and each row's partials are still summed rank 0 first by hc_post, so every
bit is the unpieced path's; only the order in time changes. Decode windows and the MTP head never take this path.
A reduce-scattering buffer (TF_GLM_PREFILL_REDUCE=scatter) reduce-scatters each piece in its slot (``reduce.sums``).
A splitting one (TF_GLM_PREFILL_REDUCE=split) gives each piece of at least ``world`` rows to its owners
(``reduce.owned``), glues only this rank's share of it and sends that share's rows of the glue's outputs to every rank
on the second stream (``reduce.share``); a shorter piece is gathered and glued whole on every rank. The owners are a
function of the chunk's pieces, the same at every exchange. Split also takes an unpieced chunk this way, as one piece
(TF_GLM_PREFILL_OVERLAP=0 then still makes an Overlap, of one piece); ``glued`` counts the rows glued.

The idea of overlapping a prompt chunk's exchanges with the next rows' work follows MiaAI-Lab's patches 0010 and 0033
for TensorFold v0.6.0 (Apache-2.0); this module is written for this tree and keeps the all-gather (or the exact
reduce-scatter)."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from . import prof, reduce

ROW_STEP = 128           # piece boundaries on multiples of this (the BF16 prompt matmul's 128-row blocks)
PIECES_MAX = 16          # TF_GLM_OVERLAP_PIECES at most: an Overlap's slots and events are made for this many


def settings(env=None) -> tuple[bool, int]:
    """(on, pieces) from TF_GLM_PREFILL_OVERLAP (0 or 1) and TF_GLM_OVERLAP_PIECES (1 to PIECES_MAX)."""

    env = os.environ if env is None else env
    on = str(env.get("TF_GLM_PREFILL_OVERLAP", "") or "1").strip()
    if on not in ("0", "1"):
        raise ValueError(f"TF_GLM_PREFILL_OVERLAP: 0 or 1, not {on!r}")
    pieces = str(env.get("TF_GLM_OVERLAP_PIECES", "") or "4").strip()
    if not pieces.isdecimal() or not 1 <= int(pieces) <= PIECES_MAX:
        raise ValueError(f"TF_GLM_OVERLAP_PIECES: a whole number from 1 to {PIECES_MAX}, not {pieces!r}")
    return on == "1", int(pieces)


def ranges(R: int, pieces: int) -> list[tuple[int, int]]:
    """[lo, hi) row pieces of a chunk: ``pieces`` about equal ones on ROW_STEP boundaries (fewer when R is short)."""

    per = -(-R // pieces)
    size = -(-per // ROW_STEP) * ROW_STEP
    return [(lo, min(R, lo + size)) for lo in range(0, R, size)]


class Overlap:
    """A prompt buffer's pieced (or split) exchanges: the second stream, its events, the current chunk's pieces."""

    def __init__(self, w, b, pieces: int) -> None:
        self.w, self.b, self.pieces = w, b, pieces
        self.stream = None                      # made with the events at the first pieced chunk
        self.active = False
        self.cut: list[tuple[int, int]] = []
        self.got: list[torch.Tensor | None] = [None] * PIECES_MAX      # each piece's partials (or sums) for its glue
        self.glued = 0                          # rows this rank glued at the current chunk's exchanges

    def begin(self, R: int) -> bool:
        """Pieces for a chunk of R rows; False (and inactive) when it has fewer than two, unless split glues it."""

        self.cut = ranges(R, self.pieces)
        self.active = len(self.cut) > 1 or (self.b.split and R >= self.b.world)
        self.glued = 0
        if self.active:
            if self.stream is None:
                self.stream = torch.cuda.Stream()
                self.filled = [torch.cuda.Event() for _ in range(PIECES_MAX)]
                self.gathered = [torch.cuda.Event() for _ in range(PIECES_MAX)]
                self.glues = [torch.cuda.Event() for _ in range(PIECES_MAX)]
                self.shared = [torch.cuda.Event() for _ in range(PIECES_MAX)]
            self.stream.wait_stream(torch.cuda.current_stream())
        return self.active

    def finish(self) -> None:
        if self.active:
            torch.cuda.current_stream().wait_stream(self.stream)
        self.active = False

    def _slot(self, lo: int, hi: int) -> torch.Tensor:
        """Piece [lo, hi)'s room in b.gath (pieces never share bytes): world x n x D fp32."""

        b, world = self.b, self.b.world
        d = b.part.shape[1]
        return b.gath[world * lo * d:world * hi * d]

    def _split(self, lo: int, hi: int) -> bool:
        return self.b.split and hi - lo >= self.b.world

    def partials(self, fill: Callable[[int, int], None]) -> None:
        """``fill(lo, hi)`` writes b.part[lo:hi]; each piece's exchange starts on the second stream once written."""

        main = torch.cuda.current_stream()
        comm, b = self.w.comm, self.b
        for j, (lo, hi) in enumerate(self.cut):
            fill(lo, hi)
            self.filled[j].record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.filled[j])
                part, room = b.part[lo:hi], self._slot(lo, hi)
                self.got[j] = (reduce.owned(comm, part, room) if self._split(lo, hi)
                               else reduce.sums(comm, part, room, b.scatter))
                self.gathered[j].record(self.stream)

    def glue(self, then: Callable[[int, int, torch.Tensor], None], outs: tuple[torch.Tensor, ...] = ()) -> None:
        """``then(lo, hi, partials)`` for each piece in order, once that piece's exchange is done: over the piece's
        rows, or (split) over this rank's share of them, whose rows of each [rows, ...] tensor of ``outs`` then go to
        every rank on the second stream. Every rank's rows of ``outs`` are in when this returns."""

        main = torch.cuda.current_stream()
        world, rank = self.b.world, self.w.comm.rank
        sent = []
        for j, (lo, hi) in enumerate(self.cut):
            with prof.timed("hc: exchange wait"):
                main.wait_event(self.gathered[j])
            if not self._split(lo, hi):
                then(lo, hi, self.got[j])
                self.glued += hi - lo
                continue
            a, b = reduce.shares(hi - lo, world)[rank]
            then(lo + a, lo + b, self.got[j])
            self.glued += b - a
            self.glues[j].record(main)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.glues[j])
                reduce.share(self.w.comm, [t[lo:hi] for t in outs])
                self.shared[j].record(self.stream)
            sent.append(j)
        with prof.timed("hc: share wait"):
            for j in sent:
                main.wait_event(self.shared[j])
