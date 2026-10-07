"""Prompt-lookup ("copy") drafts: when the context's last tokens occurred before, propose what followed them.

A reply that quotes or edits its prompt (or repeats itself) is cheap to draft: its last ``MATCH`` tokens (the pending
token included) are searched in the prompt and the reply so far, and the tokens after the latest earlier occurrence
become the drafts. They only propose; verification keeps the serial sample, so replies stay exact. An occurrence that
starts inside the reply must match the last ``REPLY_MATCH`` tokens: code repeats short boilerplate whose
continuations differ (MiaAI-Lab measured reply-sourced copies keeping 23% of their drafts, prompt-sourced 74%). After a
copied round that missed, the next copies propose at most ``MISS_MOST`` until one keeps all of its drafts.

Every rank holds the same prompt and samples the same tokens, so each computes the same proposals with no exchange.
Pure numpy (no torch). Ported from MiaAI-Lab's patches 0007-glm-copy-drafts and 0032-glm-code-copy-drafts (its
reply_match and miss_most; not its padding to 16-row windows), Apache License 2.0."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

MATCH = 8          # the context's last this many tokens must have occurred before (in the prompt)
REPLY_MATCH = 16   # ... or this many, for an occurrence that starts inside the reply
MOST = 5           # drafts a round: the window (6 rows) stays within the captured graphs (engine.GRAPH_ROWS)
# 5 is MOST: a miss does not cut the next copies. Measured at TP=2 against 3 (the MTP depth, a 4-row window) on edits
# and on two loads whose copies miss often, a rename of the edit passage (10% of the copied drafts missed) and unit
# tests for it (41%): 5 was 1.0% faster on edits and renames and the same on the tests, the same tokens; 3 helped
# none of them. Kept as a constant for a load where cutting pays.
MISS_MOST = 5      # drafts a copied round after one that missed


def enabled(env=None) -> bool:
    """TF_GLM_COPY_DRAFTS: copy drafts ahead of the MTP head (``1``, the default) or none (``0``); every rank the
    same, since a copied round verifies a wider window."""

    value = (os.environ if env is None else env).get("TF_GLM_COPY_DRAFTS", "1").strip() or "1"
    if value not in ("0", "1"):
        raise ValueError(f"TF_GLM_COPY_DRAFTS is 0 or 1, not {value!r}")
    return value == "1"


class CopyDrafts:
    """One sequence's context (prompt, then committed reply tokens and the pending one) and its copy proposals;
    ``prompt``: how many leading tokens are the prompt (occurrences starting at or past it are the reply's)."""

    def __init__(self, context: Sequence[int], prompt: int) -> None:
        self.prompt = int(prompt)
        self.proposed = 0              # drafts of the last proposal not yet settled by ``extend``
        self.missed = False            # the last settled copied round kept fewer than all of its drafts
        n = len(context)
        self.buf = np.empty((max(1024, 2 * n),), dtype=np.int32)
        self.buf[:n] = np.asarray(context, dtype=np.int32) if n else 0
        self.length = n

    def extend(self, tokens: Sequence[int]) -> None:
        """Committed tokens (the last one is the next round's pending token) join the context; after a proposal,
        they settle whether its round kept all of its drafts (they then number at least those plus one)."""

        k = len(tokens)
        if not k:
            return
        if self.proposed:
            self.missed = k <= self.proposed
            self.proposed = 0
        if self.length + k > self.buf.shape[0]:
            grown = np.empty((2 * (self.length + k),), dtype=np.int32)
            grown[:self.length] = self.buf[:self.length]
            self.buf = grown
        self.buf[self.length:self.length + k] = np.asarray(tokens, dtype=np.int32)
        self.length += k

    def starts(self) -> np.ndarray:
        """Ascending starts of the earlier occurrences of the context's last ``MATCH`` tokens (not the suffix), those
        inside the reply matching ``REPLY_MATCH``."""

        L, n = self.length, MATCH
        if L <= n:
            return np.empty((0,), dtype=np.int64)
        ctx = self.buf
        q = ctx[L - n:L]
        # starts 0 .. L - n - 1 (each leaves a token after its match): by the last token, then the others
        hits = np.flatnonzero(ctx[n - 1:L - 1] == q[n - 1])
        for k in range(n - 1):
            if not hits.size:
                break
            hits = hits[ctx[hits + k] == q[k]]
        extra = REPLY_MATCH - n
        if hits.size:
            # an occurrence starting inside the reply: the extra tokens before it must match too
            inside = hits >= self.prompt
            ok = ~inside | ((hits >= extra) & (L - n - extra >= 0))
            for k in range(1, extra + 1):
                sel = inside & ok
                if not sel.any():
                    break
                ok[sel] = ctx[hits[sel] - k] == ctx[L - n - k]
            hits = hits[ok]
        return hits

    def propose(self, room: int | None = None) -> list[int]:
        """Up to ``min(MOST, room)`` drafts (``MISS_MOST`` after a missed copied round): what followed the latest
        earlier occurrence that has that many tokens after it, else the most after any occurrence (the earliest);
        [] when the suffix never occurred before."""

        most = MISS_MOST if self.missed else MOST
        k = most if room is None else min(most, int(room))
        self.proposed = 0
        if k < 1:
            return []
        hits = self.starts()
        if not hits.size:
            return []
        L, n = self.length, MATCH
        full = hits[hits <= L - n - k]
        s = int(full[-1]) if full.size else int(hits[0])
        out = self.buf[s + n:min(s + n + k, L)].tolist()
        self.proposed = len(out)
        return out
