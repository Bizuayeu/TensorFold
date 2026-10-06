"""``PromptProbabilities``, a prompt's teacher-forced rows: what it refuses, and that it hands out every row or none."""

from __future__ import annotations

import math

import pytest

from tensorfold.engine.probabilities import PromptProbabilities

PROMPT = [5, 6, 7]


def test_positions_outside_the_prompt_are_refused():
    rows = PromptProbabilities(2, PROMPT)
    for position in (0, 3, -1):                     # position 0 has no row: nothing precedes it
        with pytest.raises(ValueError, match="outside the prompt"):
            rows.add(position, -1.0, 1, [])
    assert rows.rows == {}


def test_values_that_are_not_finite_are_refused():
    rows = PromptProbabilities(2, PROMPT)
    for logprob, top in ((math.nan, []), (-math.inf, []), (-1.0, [(6, -0.5), (7, math.nan)])):
        with pytest.raises(RuntimeError, match="not finite"):
            rows.add(1, logprob, 1, top)
    assert rows.rows == {}


def test_a_replay_must_give_the_same_row():
    rows = PromptProbabilities(2, PROMPT)
    rows.add(1, -1.0, 2, [(5, -0.5), (6, -1.0)])
    rows.add(1, -1.0, 2, [(5, -0.5), (6, -1.0)])          # the same row again: kept
    for changed in ((-1.5, 2, [(5, -0.5), (6, -1.0)]), (-1.0, 3, [(5, -0.5), (6, -1.0)]), (-1.0, 2, [(5, -0.5)])):
        with pytest.raises(RuntimeError, match="a replay changed"):
            rows.add(1, *changed)
    assert rows.rows[1] == {"id": 6, "logprob": -1.0, "rank": 2, "top": [(5, -0.5), (6, -1.0)]}


def test_every_position_must_be_there():
    rows = PromptProbabilities(1, PROMPT)
    rows.add(2, -2.0, 1, [(7, -2.0)])
    with pytest.raises(RuntimeError, match="missing for 1 positions"):
        rows.emitted()
    rows.add(1, -1.0, 1, [(6, -1.0)])
    assert [(row["id"], row["logprob"]) for row in rows.emitted()] == [(6, -1.0), (7, -2.0)]
    assert PromptProbabilities(1, [5]).emitted() == []
