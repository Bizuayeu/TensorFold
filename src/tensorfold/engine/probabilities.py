"""Request-owned target probabilities, keyed by absolute position across replay."""

import math


class Probabilities:
    def __init__(self, top: int, start: int, count: int):
        self.top, self.start, self.count = top, start, count
        self.rows: dict[int, dict] = {}

    def add(self, positions, tokens, values, top_ids, top_values):
        for pos, token, value, ids, scores in zip(positions, tokens, values, top_ids, top_values, strict=True):
            if not self.start <= pos < self.start + self.count:
                continue
            if not all(math.isfinite(x) for x in (value, *scores)):
                raise RuntimeError("target log probabilities are not finite")
            row = {"id": int(token), "logprob": value, "top": list(zip(ids, scores, strict=True))}
            old = self.rows.get(pos)
            if old is not None and old != row:
                raise RuntimeError("a replay changed the target log probabilities")
            self.rows[pos] = row

    def emitted(self, tokens):
        rows = [self.rows[self.start + i] for i in range(len(tokens))]
        if [row["id"] for row in rows] != list(tokens):
            raise RuntimeError("target probabilities do not match emitted tokens")
        return rows


class LabelProbabilities(Probabilities):
    """Collect label logits and full-vocabulary logsumexp only at the prompt's final position."""

    def __init__(self, labels, start: int):
        super().__init__(0, start, 1)
        self.labels = [int(token) for token in labels]
        self.label_logits: list[float] | None = None
        self.logsumexp: float | None = None

    def add_labels(self, position: int, logits: list[float], logsumexp: float) -> None:
        if position == self.start:
            self.label_logits, self.logsumexp = logits, logsumexp


class PromptProbabilities:
    """A prompt's teacher-forced rows: position p (1 .. len - 1) holds prompt[p]'s log probability given prompt[:p],
    its rank in the vocabulary and the ``top`` most likely (id, log probability)."""

    def __init__(self, top: int, prompt):
        self.top, self.prompt = top, [int(t) for t in prompt]
        self.rows: dict[int, dict] = {}

    def add(self, position: int, logprob: float, rank: int, top: list) -> None:
        if not 1 <= position < len(self.prompt):
            raise ValueError(f"prompt position {position} is outside the prompt")
        if not all(math.isfinite(x) for x in (logprob, *(value for _, value in top))):
            raise RuntimeError("prompt log probabilities are not finite")
        row = {"id": self.prompt[position], "logprob": logprob, "rank": rank, "top": list(top)}
        old = self.rows.get(position)
        if old is not None and old != row:
            raise RuntimeError("a replay changed the prompt log probabilities")
        self.rows[position] = row

    def emitted(self) -> list[dict]:
        """Rows for positions 1 .. len - 1, in order; every one must be there."""

        missing = [p for p in range(1, len(self.prompt)) if p not in self.rows]
        if missing:
            raise RuntimeError(f"prompt log probabilities are missing for {len(missing)} positions")
        return [self.rows[p] for p in range(1, len(self.prompt))]
