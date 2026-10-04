"""TF_GLM_HEAT_HIGH / TF_GLM_HEAT_LOW: a prefill waits between chunks while any rank's hottest zone is above HIGH, every
rank together, until the hottest of all is at or below LOW; three rank threads with their own temperatures."""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.glm5_next.cuda import heat  # noqa: E402
from tensorfold.families.glm5_next.cuda.decode import _hottest  # noqa: E402
from tests.test_glm_serial_stop import WAIT, Comm  # noqa: E402


@pytest.fixture
def zones(tmp_path):
    for i, milli in enumerate((57800, 61200, 55500)):
        (tmp_path / f"zone{i}").write_text(f"{milli}\n")
    (tmp_path / "zone9").write_text("n/a\n")            # unreadable: skipped
    return str(tmp_path / "zone*")


def test_the_bands_come_from_the_environment_together(zones):
    assert heat.Heat.from_env({}) is None
    h = heat.Heat.from_env({"TF_GLM_HEAT_HIGH": "92", "TF_GLM_HEAT_LOW": " 88 ", "TF_GLM_HEAT_ZONES": zones})
    assert (h.high, h.low, h.zones) == (92.0, 88.0, zones)
    with pytest.raises(ValueError, match="set together"):
        heat.Heat.from_env({"TF_GLM_HEAT_HIGH": "92"})
    with pytest.raises(ValueError, match="must be below"):
        heat.Heat.from_env({"TF_GLM_HEAT_HIGH": "88", "TF_GLM_HEAT_LOW": "88", "TF_GLM_HEAT_ZONES": zones})
    with pytest.raises(ValueError, match="no file matches"):
        heat.Heat(92, 88, zones + "-none")


def test_a_reading_is_the_hottest_zone(zones):
    assert heat.Heat(92, 88, zones).read() == 61.2


class Clock:
    def __init__(self) -> None:
        self.now, self.sleeps = 0.0, 0

    def sleep(self, s: float) -> None:
        self.now += s
        self.sleeps += 1


def three_ranks(temps: list[list[float]], zones: str, high=92.0, low=88.0):
    """Each rank reads its own sequence (its last value once spent) and waits through one all-gather a reading."""

    comm, out = Comm(3), [None] * 3

    class W:
        world = 3

    W.comm = comm

    def rank(r: int) -> None:
        clock, seq = Clock(), iter(temps[r])
        h = heat.Heat(high, low, zones, sleep=clock.sleep, clock=lambda: clock.now)
        last = [temps[r][-1]]
        h.read = lambda: last.__setitem__(0, next(seq, last[0])) or last[0]
        out[r] = (h.wait(_hottest(W, "cpu")), clock.sleeps)

    threads = [threading.Thread(target=rank, args=(r,), name=f"rank-{r}") for r in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    return out, comm


def test_below_high_one_gather_and_no_wait(zones):
    out, comm = three_ranks([[80.0], [91.9], [92.0]], zones)
    assert out == [(0.0, 0)] * 3
    assert comm.sizes == [[1]] * 3


def test_one_hot_rank_holds_every_rank_until_the_hottest_is_at_low(zones):
    # rank 1 is hot; rank 0 is cold all along yet waits; rank 2 cools to LOW before rank 1 does
    out, comm = three_ranks([[70.0], [93.5, 91.0, 89.0, 88.5, 88.0, 95.0], [90.0, 88.0, 86.0]], zones)
    assert out == [(8.0, 4)] * 3                       # four readings of 2 s, left on rank 1's 88.0
    assert comm.sizes == [[1] * 5] * 3                 # the same gathers on every rank


def test_the_wait_ends_on_the_gathered_maximum_not_a_local_reading(zones):
    # rank 0 reaches LOW on its first reading in the wait; rank 2 stays above it for two more
    out, comm = three_ranks([[93.0, 80.0], [85.0], [90.0, 89.0, 88.1, 87.0]], zones)
    assert out == [(6.0, 3)] * 3
    assert comm.sizes == [[1] * 4] * 3


def test_a_wait_is_printed_where_the_engine_prints(zones, capsys):
    """Start, a line a minute while it lasts, and end, as [tensorfold] lines on stdout (no logging setup needed)."""

    clock, seq = Clock(), iter([93.0] + [90.0] * 31 + [87.5])
    h = heat.Heat(92, 88, zones, sleep=clock.sleep, clock=lambda: clock.now)
    h.read = lambda: next(seq)
    assert h.wait(lambda t: t) == 64.0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["[tensorfold] heat: hottest zone 93.0 C above 92 C; the prefill waits for 88 C",
                     "[tensorfold] heat: waiting 60 s, hottest zone 90.0 C",
                     "[tensorfold] heat: 87.5 C after 64 s; the prefill goes on"]


def test_one_rank_reads_its_own_zones():
    class W:
        comm, world = None, 1

    assert _hottest(W, "cpu")(93.25) == 93.25
