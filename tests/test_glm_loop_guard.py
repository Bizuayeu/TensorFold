"""TF_GLM_LOOP_GUARD's guard through the real generate_gated, on replayed token streams (after MiaAI-Lab's
tools/test_loop_guard.py for patch 0091): a clean block, an exact cycle, a mostly-"the" block, a loop after </think>,
near cycles and natural frequencies, two independent guards, and the guard beside the thinking budget."""

import random

from tensorfold.engine.call_gate import ThinkBudget, generate_gated
from tensorfold.engine.loop_guard import LoopGuard

THINK_END, CLOSE = 7, [10, 7, 11]
ANSWER = [50, 51, 52, 2]
rnd = random.Random(1)
PROSE = [rnd.randrange(100, 5000) for _ in range(900)]
CLEAN = PROSE[:600] + [THINK_END, 50, 51, 2]
CYCLE = PROSE[:300] + [9, 8, 9, 6] * 400


def run(gates, stream, rounds=4, cap=3000):
    """Feed ``stream`` (the model's tokens) in rounds of ``rounds`` through generate_gated; after the close the model
    answers ANSWER. Returns the reply."""
    out = []

    def generate(ids, count, take):
        tail = list(stream)[len(out):]           # the model goes on where the reply stopped
        if CLOSE[-1] in ids:                      # after the close it answers
            tail = list(ANSWER)
        while tail and count > 0:
            chunk, tail = tail[:rounds][:count], tail[rounds:]
            if take(chunk):
                return {}
            count -= len(chunk)
        return {}

    def on_tokens(new):
        out.extend(new)
        return False

    generate_gated(generate, [0], cap, gates, on_tokens)
    return out


def guard():
    return LoopGuard(CLOSE, THINK_END)


def test_a_clean_block_is_untouched():
    g = guard()
    assert run([g], CLEAN) == CLEAN and g.fires == 0


def test_an_exact_cycle_is_closed_within_a_window():
    g = guard()
    out = run([g], CYCLE)
    assert g.fires == 1 and out[-len(CLOSE) - 4:-4] == CLOSE and out[-4:] == ANSWER
    assert len(out) < 300 + 256 + 4 + 8 + len(CLOSE) + 8


def test_a_block_that_is_half_one_token_is_closed():
    mixed = []
    for i in range(1200):
        mixed += [5, PROSE[i % len(PROSE)]] if i % 3 else [5, 5, PROSE[i % len(PROSE)]]
    g = guard()
    out = run([g], mixed)
    assert g.fires == 1 and len(out) < 700


def test_a_loop_after_the_think_block_is_not_watched():
    after = PROSE[:50] + [THINK_END] + [3] * 800
    g = guard()
    assert run([g], after) == after and g.fires == 0


def test_near_cycles_and_natural_frequencies_never_fire():
    r = random.Random(2)
    near = [PROSE[i % 11] if i % 12 else r.randrange(6000, 9000) for i in range(3000)]
    g = guard()
    run([g], near)
    assert g.fires == 0
    zipf = r.choices(range(100, 5100), weights=[1 / k for k in range(1, 5001)], k=5000)    # the top token ~11%
    g = guard()
    run([g], zipf, cap=5000)
    assert g.fires == 0


def test_two_guards_are_independent():
    a, b = guard(), guard()
    run([a], CYCLE)
    run([b], CLEAN)
    assert a.fires == 1 and b.fires == 0


def test_with_the_thinking_budget_the_block_closes_once():
    out = run([ThinkBudget(350, CLOSE, THINK_END), guard()], CYCLE)
    assert out.count(THINK_END) == 1
