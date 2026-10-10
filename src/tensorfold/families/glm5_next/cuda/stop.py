"""A serial run's stop on every rank: rank 0's wish rides as one float32 word on the next verify sample's all-gather.
A prompt's stop between its chunks (``PromptStop``): every rank's wish as one int32 of an all-gather after each."""

from __future__ import annotations

import os

import torch


class StopVote:
    """Wraps ``on_tokens``: a true result is this rank's wish to stop; ``stop`` is the decision every rank shares."""

    def __init__(self, on_tokens=None) -> None:
        self.fn = on_tokens
        self.mine = False
        self.agreed = False
        self.flags: torch.Tensor | None = None

    def __call__(self, tokens) -> bool:
        if self.fn is not None and self.fn(tokens):
            self.mine = True
        return self.mine

    def gather(self, gather):
        """``gather`` with this rank's wish as one word more on its first call, given back without that word."""

        first = [True]

        def voting(words: torch.Tensor) -> torch.Tensor:
            if not first[0]:
                return gather(words)
            first[0] = False
            flag = torch.full((1,), 1.0 if self.mine else 0.0, dtype=words.dtype, device=words.device)
            got = gather(torch.cat([words.reshape(-1), flag]))
            self.flags = got[:, -1]
            return got[:, :-1].contiguous()

        return voting

    @property
    def stop(self) -> bool:
        if self.flags is not None:
            self.agreed = self.agreed or bool((self.flags > 0).any())
            self.flags = None
        return self.agreed


def stopped(e) -> bool:
    """Whether every rank agreed to end this run; False outside a voting run."""

    vote = getattr(e, "vote", None)
    return vote is not None and vote.stop


class PromptStopped(Exception):
    """A prompt's prefill ended after a chunk on a stop every rank agreed to; ``at``: the rows committed (a prefix's
    state, as a shorter prompt's)."""

    def __init__(self, at: int) -> None:
        super().__init__(f"the prompt stopped after {at} rows: the client left")
        self.at = at


def prompt_stop_on(env=None) -> bool:
    """TF_GLM_PROMPT_STOP: 1 (default) a prompt votes after each chunk but the last whether to stop (``PromptStop``);
    0 every prompt fills to its end. Every rank must agree: the vote is a collective."""

    value = (os.environ if env is None else env).get("TF_GLM_PROMPT_STOP", "").strip() or "1"
    if value not in ("0", "1"):
        raise ValueError(f"TF_GLM_PROMPT_STOP is 0 or 1, not {value!r}")
    return value == "1"


class PromptStop:
    """``poll`` (rank 0: the client left; other ranks None) is this rank's wish; each call all-gathers every rank's
    wish as one int32 and stops when any rank wishes, so every rank stops after the same chunk. A poll that raises
    never wishes; once agreed, the stop holds."""

    def __init__(self, w, poll=None) -> None:
        self.w, self.poll = w, poll
        self.stop = False

    def mine(self) -> bool:
        try:
            return bool(self.poll is not None and self.poll())
        except Exception:  # noqa: BLE001 - a broken poll never stops the ranks apart
            return False

    def __call__(self) -> bool:
        wish = self.mine()
        if self.w.comm is not None and self.w.world > 1:
            got = torch.empty((self.w.world,), dtype=torch.int32, device=self.w.device)
            self.w.comm.all_gather(torch.full((1,), int(wish), dtype=torch.int32, device=self.w.device), got)
            wish = bool(got.max())
        self.stop = self.stop or wish
        return self.stop
