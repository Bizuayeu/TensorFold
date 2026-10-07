"""TF_GLM_HEAT_HIGH / TF_GLM_HEAT_LOW: a prefill waits between chunks while any rank's hottest zone is above HIGH, every
rank together, until the hottest of all is at or below LOW; TF_GLM_HEAT_CEILING: also while the hottest plus the last
chunk's rise is above CEILING. Three rank threads with their own temperatures."""

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


def test_the_ceiling_comes_from_the_environment_with_the_bands(zones):
    env = {"TF_GLM_HEAT_HIGH": "92", "TF_GLM_HEAT_LOW": "88", "TF_GLM_HEAT_ZONES": zones}
    assert heat.Heat.from_env(env).ceiling is None
    assert heat.Heat.from_env(env | {"TF_GLM_HEAT_CEILING": " "}).ceiling is None
    assert heat.Heat.from_env(env | {"TF_GLM_HEAT_CEILING": " 93 "}).ceiling == 93.0
    with pytest.raises(ValueError, match="TF_GLM_HEAT_CEILING needs TF_GLM_HEAT_HIGH and TF_GLM_HEAT_LOW"):
        heat.Heat.from_env({"TF_GLM_HEAT_CEILING": "93", "TF_GLM_HEAT_ZONES": zones})
    for ceiling in ("92", "90", "nan"):
        with pytest.raises(ValueError, match=r"TF_GLM_HEAT_HIGH \(92.0\) must be below TF_GLM_HEAT_CEILING"):
            heat.Heat.from_env(env | {"TF_GLM_HEAT_CEILING": ceiling})
    with pytest.raises(ValueError, match="TF_GLM_HEAT_CEILING: '93C' is not a temperature"):
        heat.Heat.from_env(env | {"TF_GLM_HEAT_CEILING": "93C"})


def test_a_reading_is_the_hottest_zone(zones):
    assert heat.Heat(92, 88, zones).read() == 61.2


class Clock:
    def __init__(self) -> None:
        self.now, self.sleeps = 0.0, 0

    def sleep(self, s: float) -> None:
        self.now += s
        self.sleeps += 1


def three_ranks(temps: list[list[float]], zones: str, high=92.0, low=88.0, ceiling=None, chunks=1):
    """Each rank reads its own sequence (its last value once spent) and waits through one all-gather a reading, before
    each of ``chunks`` chunks of one prompt; the seconds waited in all."""

    comm, out = Comm(3), [None] * 3

    class W:
        world = 3

    W.comm = comm

    def rank(r: int) -> None:
        clock, seq = Clock(), iter(temps[r])
        h = heat.Heat(high, low, zones, ceiling=ceiling, sleep=clock.sleep, clock=lambda: clock.now)
        last = [temps[r][-1]]
        h.read = lambda: last.__setitem__(0, next(seq, last[0])) or last[0]
        h.start()
        out[r] = (sum(h.wait(_hottest(W, "cpu")) for _ in range(chunks)), clock.sleeps)

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


def one_rank(readings: list[float], zones: str, ceiling=93.0):
    """A Heat at 92 / 88 C that reads ``readings`` in turn (and fails past their end) on a clock that sleeps."""

    clock, seq = Clock(), iter(readings)
    h = heat.Heat(92, 88, zones, ceiling=ceiling, sleep=clock.sleep, clock=lambda: clock.now)
    h.read = lambda: next(seq)
    return h


def test_without_a_ceiling_a_steep_rise_below_high_goes_on(zones):
    h = one_rank([70.0, 80.6, 87.6, 91.9], zones, ceiling=None)
    h.start()
    assert [h.wait(lambda t: t) for _ in range(4)] == [0.0] * 4


def test_a_chunk_waits_when_the_last_rise_would_carry_it_past_the_ceiling(zones):
    """1M tokens at 92 / 88 C: a check at 87.6 C let a chunk through that rose 7 C, to 94.3 C. With CEILING 93 the
    small rises of the early chunks go on, the 7 C one waits, and it waits past LOW until 7 C more stays at 93."""

    h = one_rank([70.0, 71.5, 70.5, 80.6, 87.6, 87.0, 86.4, 85.8], zones)
    h.start()
    # 70 first (no rise), +1.5, -1 (no rise), +10.1 to 80.6 (90.7 C: goes), +7 to 87.6 (94.6 C: waits); 87.0 is at
    # LOW yet 94.0 C with the rise, 86.4 is 93.4, 85.8 is 92.8
    assert [h.wait(lambda t: t) for _ in range(5)] == [0.0] * 4 + [6.0]


def test_the_wait_ends_at_the_ceiling_less_the_rise_and_the_next_rise_counts_from_there(zones, capsys):
    h = one_rank([80.5, 87.5, 88.0, 86.5, 86.0, 90.0, 87.5, 93.0, 87.4], zones)
    h.start()
    # +7 to 87.5 waits for 93 - 7 = 86, not for LOW (88.0 holds); the next chunk starts at 86.0, so 90.0 is a 4 C rise
    # (not 2.5 from the check at 87.5): it waits for LOW, 89 being above; 93.0 is above HIGH and 5.5 C up: 87.5
    assert [h.wait(lambda t: t) for _ in range(4)] == [0.0, 6.0, 2.0, 2.0]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        ("[tensorfold] heat: hottest zone 87.5 C plus the last chunk's 7.0 C rise above 93 C; the prefill waits for "
         "86 C"),
        "[tensorfold] heat: 86.0 C after 6 s; the prefill goes on",
        ("[tensorfold] heat: hottest zone 90.0 C plus the last chunk's 4.0 C rise above 93 C; the prefill waits for "
         "88 C"),
        "[tensorfold] heat: 87.5 C after 2 s; the prefill goes on",
        "[tensorfold] heat: hottest zone 93.0 C above 92 C; the prefill waits for 87.5 C",
        "[tensorfold] heat: 87.4 C after 2 s; the prefill goes on"]


def test_a_prompt_starts_without_the_last_prompts_rise(zones):
    for start, waited in ((True, 0.0), (False, 2.0)):
        h = one_rank([80.0, 86.0, 90.0, 85.0], zones)
        h.start()
        assert [h.wait(lambda t: t) for _ in range(2)] == [0.0, 0.0]          # 86 + 6 = 92: under 93
        if start:
            h.start()
        assert h.wait(lambda t: t) == waited               # a first chunk at 90 C; with the old 4 C rise, 94


def test_every_rank_takes_the_rise_of_the_gathered_maximum(zones):
    # the hottest rank changes between the checks: 80.6 on rank 1, then 87.6 on rank 2; the gathered rise is 7 C (each
    # rank's own is -10.6 to 8.6), so every rank waits for 86 C and leaves at 85.5 together
    out, comm = three_ranks([[70.0, 75.0], [80.6, 70.0], [79.0, 87.6, 87.0, 85.5]], zones, ceiling=93.0, chunks=2)
    assert out == [(4.0, 2)] * 3
    assert comm.sizes == [[1] * 4] * 3


def test_one_rank_reads_its_own_zones():
    class W:
        comm, world = None, 1

    assert _hottest(W, "cpu")(93.25) == 93.25
