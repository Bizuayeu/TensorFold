"""A prompt whose client left stops after its next chunk on every rank (TF_GLM_PROMPT_STOP): each rank's wish is one
int32 of an all-gather after every chunk but the last. The real ``decode.prefill`` loop, chunks that only commit."""

from __future__ import annotations

import importlib
import threading
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tests.test_cuda_geometry import allocations  # noqa: E402,F401  (fixture: fake triton, so decode imports)
from tests.test_glm_serial_stop import WAIT, Comm  # noqa: E402

decode = engine_mod = stop = None               # the CUDA modules, imported per test under the fake triton

ROWS, FIRST = 4, 7                              # a chunk's rows; the token every prefill samples
PROMPT = list(range(20, 38))                    # 18 tokens: chunks of 4, 4, 4, 4, 2


@pytest.fixture(autouse=True)
def cpu_prefill(allocations, monkeypatch):  # noqa: F811
    global decode, engine_mod, stop
    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    engine_mod = importlib.import_module("tensorfold.families.glm5_next.cuda.engine")
    stop = importlib.import_module("tensorfold.families.glm5_next.cuda.stop")
    monkeypatch.setattr(decode, "stage", lambda w, st, b, chunk: len(chunk))
    monkeypatch.setattr(decode, "chunks_for", lambda st, R: 1)
    monkeypatch.setattr(decode, "compute", lambda w, st, b, R, **kw: torch.zeros((R, 8)))

    def commit(w, st, b, R, keep):
        st.commits.append(keep)
        st.pos += keep

    monkeypatch.setattr(decode, "commit", commit)


class Prefilling:
    """The surface ``decode.prefill`` reads, without a model."""

    def __init__(self, rank: int, world: int, comm) -> None:
        self.w = NS(comm=comm, world=world, device=torch.device("cpu"), mtp=None)
        self.st = NS(pos=0, commits=[], rec=[torch.zeros(1)], cur=[], conv=torch.zeros(1))
        self.pbuf = NS(fnormed=torch.zeros((ROWS, 2)))
        self.prefill_rows = ROWS
        self.constraint = self.window = self.heat = self.prompt_stop = None

    def reset(self) -> None:
        self.st.pos = 0
        self.st.commits.clear()

    def sample(self, logits, positions, sampling):
        return [FIRST]

    def follow(self, tokens) -> None:
        pass


def threads(world: int, fn) -> list:
    """``fn(rank)`` on a thread a rank: per rank its result."""

    out: list = [None] * world
    errors: list = []

    def rank(r: int) -> None:
        try:
            out[r] = fn(r)
        except BaseException as exc:            # noqa: BLE001  reported by the test
            errors.append(exc)

    ts = [threading.Thread(target=rank, args=(r,), name=f"rank-{r}") for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(WAIT)
    assert not any(t.is_alive() for t in ts), "a rank is still filling: the ranks disagree"
    if errors:
        raise errors[0]
    return out


def fill(world: int, prompt=PROMPT, *, wisher: int | None = 0, after: int = 2, vote: bool = True, poll=None,
         keep_at: int | None = None):
    """Each rank prefills ``prompt``; ``wisher`` wishes once ``after`` chunks are committed: per rank (the first
    token, or the PromptStopped, the engine, its kept states) and the Comm."""

    comm = Comm(world) if world > 1 else None

    def rank(r: int):
        e = Prefilling(r, world, comm)
        mine = poll if poll is not None else (lambda: len(e.st.commits) >= after) if r == wisher else None
        e.prompt_stop = stop.PromptStop(e.w, mine) if vote else None
        kept: list = []
        try:
            got = decode.prefill(e, prompt, None, mtp=False, keep_at=keep_at, keep=kept.append if keep_at else None)
        except stop.PromptStopped as cut:
            got = cut
        return got, e, kept

    return threads(world, rank), comm


def test_the_setting_is_on_unless_0():
    assert stop.prompt_stop_on({}) and stop.prompt_stop_on({"TF_GLM_PROMPT_STOP": "1"})
    assert stop.prompt_stop_on({"TF_GLM_PROMPT_STOP": " "})
    assert not stop.prompt_stop_on({"TF_GLM_PROMPT_STOP": "0"})
    with pytest.raises(ValueError, match="TF_GLM_PROMPT_STOP"):
        stop.prompt_stop_on({"TF_GLM_PROMPT_STOP": "yes"})


@pytest.mark.parametrize("world,wisher", [(2, 0), (2, 1), (3, 0), (3, 2)])
def test_any_ranks_wish_stops_every_rank_after_the_same_chunk(world, wisher):
    ranks, comm = fill(world, wisher=wisher, after=2)
    for got, e, _ in ranks:
        assert isinstance(got, stop.PromptStopped) and got.at == 8
        assert e.st.pos == 8 and e.st.commits == [4, 4] and e.prompt_stop.stop
    assert all(sizes == [1, 1] for sizes in comm.sizes)                   # one int32 a chunk, the same on every rank


@pytest.mark.parametrize("world", [1, 2, 3])
def test_without_a_wish_a_prompt_fills_as_without_the_vote(world):
    plain, _ = fill(world, vote=False)
    voted, comm = fill(world, wisher=None)
    for (a, ea, _), (b, eb, _) in zip(plain, voted):
        assert a == b == FIRST and ea.st.commits == eb.st.commits == [4, 4, 4, 4, 2]
        assert not eb.prompt_stop.stop
    if comm is not None:
        assert all(sizes == [1] * 4 for sizes in comm.sizes)              # none after the last chunk


def test_a_wish_on_the_last_chunk_or_a_one_chunk_prompt_stops_nothing():
    ranks, _ = fill(2, after=5)
    assert all(got == FIRST and e.st.pos == len(PROMPT) for got, e, _ in ranks)
    ranks, comm = fill(2, PROMPT[:ROWS], after=0)
    assert all(got == FIRST for got, _, _ in ranks) and comm.sizes == [[], []]


def test_a_broken_poll_never_wishes():
    def broken():
        raise OSError("the socket is gone")

    ranks, _ = fill(2, poll=broken)
    assert all(got == FIRST and e.st.pos == len(PROMPT) for got, e, _ in ranks)


def test_a_stop_holds_once_agreed():
    calls = iter([True, False])
    vote = stop.PromptStop(NS(comm=None, world=1, device=torch.device("cpu")), lambda: next(calls))
    assert vote() and vote() and vote.stop


def test_a_state_kept_in_a_committed_chunk_is_kept():
    """keep_at at a chunk's end: the chunk that reaches it commits, then the stop; the state is a prefix's."""

    ranks, _ = fill(2, PROMPT[:9], after=2, keep_at=8)
    for got, e, kept in ranks:
        assert isinstance(got, stop.PromptStopped) and got.at == 8
        assert [snap.ids for snap in kept] == [PROMPT[:8]]


# -- the engine: GlmEngine._run on two ranks, the prompt's chunks through the real prefill ----------------------------
def glm_rank(r: int, comm):
    g = object.__new__(engine_mod.GlmEngine)
    g.e = Prefilling(r, 2, comm)
    g.w, g.rank, g.world, g.eos = g.e.w, r, 2, (5,)
    g.drafter = None
    g.costs, g.cache, g.live = {}, [], []
    g._remember = g.cache.append
    return g


def runs(cancelled, *, setting: str = "1", monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("TF_GLM_PROMPT_STOP", setting)
    comm = Comm(2)

    def rank(r: int):
        g = glm_rank(r, comm)
        stats = g._run(PROMPT, 1, None, True, lambda new: None, engine_mod.encode_policy("3"), None, True,
                       **({"cancelled": cancelled} if r == 0 else {}))
        return stats, g

    return threads(2, rank), comm


def test_engine_run_reports_the_stop_on_both_ranks_and_cuts_the_live_rows(monkeypatch):
    commits = {}

    def cancelled():
        commits.setdefault("seen", 0)
        commits["seen"] += 1
        return commits["seen"] >= 3                 # rank 0 polls after each chunk: the third chunk's wish

    ranks, comm = runs(cancelled, monkeypatch=monkeypatch)
    for stats, g in ranks:
        assert stats["prompt_stopped"] == 12 and stats["cached"] == 0
        assert g.live == PROMPT[:12] and g.e.st.pos == 12
        assert g.e.prompt_stop is None and g.e.vote is None
    assert comm.sizes[0] == comm.sizes[1] == [1, 1, 1]


def test_engine_run_without_a_stop_or_with_the_setting_off_fills_to_the_end(monkeypatch):
    ranks, comm = runs(lambda: False, monkeypatch=monkeypatch)
    assert all("prompt_stopped" not in stats and g.e.st.pos == len(PROMPT) for stats, g in ranks)
    assert comm.sizes[0] == [1] * 4
    ranks, comm = runs(lambda: True, setting="0", monkeypatch=monkeypatch)
    assert all("prompt_stopped" not in stats and g.e.st.pos == len(PROMPT) for stats, g in ranks)
    assert comm.sizes == [[], []]


def test_generate_raises_request_cancelled_after_a_stopped_prompt(monkeypatch):
    from tensorfold.server.cancellation import RequestCancelled

    g = object.__new__(engine_mod.GlmEngine)
    g.vision, g.limit, g.serial_only, g.policy, g.request = None, 1000, True, "0", NS()
    g.drafter, g.w = None, NS(mtp=None)
    g._resume = lambda prompt, code: None
    g._ring = lambda: None
    g._share = lambda values: values
    seen = {}

    def run(*args, cancelled=None):
        seen["cancelled"] = cancelled
        return {"prefill_s": 0.1, "cached": 0, "prompt_stopped": 8}

    g._run = run

    def poll():
        return True

    with pytest.raises(RequestCancelled, match="after 8"):
        g.generate(PROMPT, 4, None, lambda new: None, cancelled=poll)
    assert seen["cancelled"] is poll


def test_the_server_hands_the_engine_its_cancellation(tmp_path):
    from tests.test_cuda_server_disconnect import app_for

    class Engine:
        eos = (0,)
        concurrent = False

        def __init__(self):
            self.request = threading.local()
            self.polls: list = []

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, cancelled=None):
            self.polls.append(cancelled)
            on_tokens([ord("a")])
            return {}

    engine = Engine()
    app = app_for(tmp_path, engine)

    def poll():
        return False

    app.run({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}, True, lambda delta: True,
            cancelled=poll)
    assert engine.polls == [poll]
