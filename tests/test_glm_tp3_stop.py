"""#301's stop on three ranks of unequal vocabulary shares (GLM-5.3-Flash's 448/448/384 of 1,280, here 24/24/16 of
64): rank 0's wish rides the verify sample's gather of padded top-k candidates and the nucleus rule's first gather
alike, so every rank ends after the same round, a reply equals one rank's, and the next request runs at once."""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tests import test_glm_serial_stop as two  # noqa: E402
from tests.test_glm_serial_stop import KINDS, SAMPLINGS, WAIT, allocations, cpu_loops  # noqa: E402,F401  (fixtures)

SPANS = [(0, 24), (24, 48), (48, 64)]          # the last shard narrower: top_k 20 + MARGIN pads it with -inf


class Uneven(two.FakeEngine):
    def __init__(self, rank: int, world: int, comm) -> None:
        super().__init__(rank, world, comm)
        if world == 3:
            lo, hi = SPANS[rank]
            self.w.vocab_offset, self.w.vocab_spans, self.share = lo, SPANS, hi - lo


@pytest.fixture(autouse=True)
def uneven(monkeypatch):
    monkeypatch.setattr(two, "FakeEngine", Uneven)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", sorted(SAMPLINGS))
def test_a_stop_on_rank0_ends_three_ranks_after_the_next_round(kind, mode):
    sampling = SAMPLINGS[mode]
    one = two.run_ranks(kind, world=1, sampling=sampling, vote=False)[0][0][0]
    ref = two.run_ranks(kind, world=3, sampling=sampling, vote=False)[0][0][0]
    assert ref.tokens == one.tokens and len(ref.tokens) == 60     # the shares do not move a reply
    for stop_at in (1, 3):
        ranks, comm = two.run_ranks(kind, world=3, sampling=sampling, stop_at=stop_at)
        assert [res.rounds for res, _, _ in ranks] == [stop_at + 1] * 3
        assert all(res.tokens == ranks[0][0].tokens and e.st.pos == ranks[0][2].st.pos for res, _, e in ranks)
        assert all(e.vote.stop for _, _, e in ranks) and not any(e.vote.mine for _, _, e in ranks[1:])
        sent = [t for call in ranks[0][1][:stop_at] for t in call]
        assert [7] + sent == ref.tokens[:1 + len(sent)] and ranks[0][0].tokens == ref.tokens[:len(ranks[0][0].tokens)]
        assert comm.sizes[0] == comm.sizes[1] == comm.sizes[2]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", sorted(SAMPLINGS))
def test_the_vote_never_changes_a_three_rank_reply(kind, mode):
    sampling = SAMPLINGS[mode]
    plain, unvoted = two.run_ranks(kind, world=3, sampling=sampling, vote=False)
    ranks, comm = two.run_ranks(kind, world=3, sampling=sampling)
    for res, _, e in ranks:
        assert res.tokens == plain[0][0].tokens and res.rounds == plain[0][0].rounds and not e.vote.stop
    assert sum(comm.sizes[0]) == sum(unvoted.sizes[0]) + plain[0][0].rounds     # one word a verify sample
    assert comm.sizes[0] == comm.sizes[1] == comm.sizes[2]


# -- the engine: GlmEngine._run on three ranks, stopped the ways the server stops a reply ----------------------------
def _run(runs: list) -> list:
    """Each rank runs ``runs`` in order (prompt, max_tokens, rank 0's on_tokens): per rank, the stats of each."""

    comm = two.Comm(3)
    out: list = [None] * 3
    errors: list = []

    def rank(r: int) -> None:
        try:
            g = two.glm_rank(r, comm)
            g.e, g.world = Uneven(r, 3, comm), 3
            mine = []
            for prompt, count, server in runs:
                fn = server if r == 0 else (lambda new: None)
                mine.append((g._run(prompt, count, None, True, fn, two.engine_mod.encode_policy("3"), None, True),
                             g.e.st.pos, g.e.vote))
            out[r] = mine
        except BaseException as exc:            # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=rank, args=(r,), name=f"rank-{r}") for r in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert not any(t.is_alive() for t in threads), "a rank is still decoding: the ranks disagree"
    if errors:
        raise errors[0]
    return out


class Server:
    """Rank 0's callback as the server's: ``stops(calls, heard)`` says a reply is cut (a client gone, a stop sequence
    seen) and every later call says stop too; ``at`` is the call that first did (1: the first token's)."""

    def __init__(self, stops) -> None:
        self.stops, self.heard, self.calls, self.at = stops, [], 0, None

    def __call__(self, new) -> bool:
        self.calls += 1
        self.heard += list(new)
        if self.at is None and self.stops(self.calls, self.heard):
            self.at = self.calls
        return self.at is not None


def seen(out: list[int], stop: list[int]) -> bool:
    return any(out[i:i + len(stop)] == stop for i in range(len(out) - len(stop) + 1))


def test_a_client_that_leaves_stops_every_rank_and_the_next_request_runs(fake_prefill):
    prompt = list(range(20, 20 + two.PROMPT))
    (full, _, _), = _run([(prompt, 500, lambda new: False)])[0]
    gone = Server(lambda calls, heard: calls >= 4)          # emit fails from the fourth call on
    runs = _run([(prompt, 500, gone), (prompt, 500, lambda new: False)])
    assert gone.at == 4 and [r[0][0]["rounds"] for r in runs] == [4] * 3 and all(r[0][0]["stopped"] for r in runs)
    assert len({r[0][1] for r in runs}) == 1 and all(r[0][2] is None and r[1][2] is None for r in runs)
    assert all("stopped" not in r[1][0] and r[1][0]["sha256"] == full["sha256"] for r in runs)


def test_a_stop_string_stops_every_rank_and_the_next_request_runs(fake_prefill):
    prompt = list(range(20, 20 + two.PROMPT))
    tokens: list[int] = []
    (full, _, _), = _run([(prompt, 500, lambda new: tokens.extend(new))])[0]
    stop = tokens[6:8]                              # a sequence the reply first reaches some rounds in
    assert not seen(tokens[:7], stop)
    cut = Server(lambda calls, heard: seen(heard, stop))
    runs = _run([(prompt, 500, cut), (prompt, 500, lambda new: False)])
    assert 1 < cut.at < full["rounds"] and [r[0][0]["rounds"] for r in runs] == [cut.at] * 3
    assert all(r[0][0]["stopped"] for r in runs) and len({r[0][1] for r in runs}) == 1
    assert all("stopped" not in r[1][0] and r[1][0]["sha256"] == full["sha256"] for r in runs)


@pytest.fixture
def fake_prefill(monkeypatch):
    def prefill(e, prompt, sampling, *, mtp=True, drafter=None, resume=None, keep_at=None, keep=None,
                prompt_logprobs=None, vision=None):
        e.st.pos = e.st.mtp_len = len(prompt)
        return 7

    monkeypatch.setattr(two.decode, "prefill", prefill)
