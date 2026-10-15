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


def test_the_wait_ends_at_the_ceiling_less_the_larger_rise(zones, capsys):
    """A chunk that begins straight after another is judged by such a chunk's rise; a wait ends where the larger of
    that rise and the one counted from the wait's end stays within CEILING."""

    h = one_rank([79.0, 80.5, 87.5, 88.0, 86.5, 86.0, 90.0, 87.0, 85.9, 93.0, 85.5], zones)
    h.start()
    # 79 first; 80.5 follows the prompt's start (its 1.5 C is not a rise between chunks); +7 to 87.5 waits for
    # 93 - 7 = 86, not for LOW (88.0 holds). The next chunk begins at 86.0 after the wait, so its 4 C to 90.0 is not
    # one between chunks: 90.0 plus the 7 C still waits, for 93 - 7 = 86 (the larger rise). 93.0 is above HIGH, 7.1 C
    # up from 85.9: it waits for 85.9
    assert [h.wait(lambda t: t) for _ in range(5)] == [0.0, 0.0, 6.0, 4.0, 2.0]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        ("[tensorfold] heat: hottest zone 87.5 C plus a chunk's 7.0 C rise, one run straight after another, above "
         "93 C; the prefill waits for 86 C"),
        "[tensorfold] heat: 86.0 C after 6 s; the prefill goes on",
        ("[tensorfold] heat: hottest zone 90.0 C plus a chunk's 7.0 C rise, one run straight after another, above "
         "93 C; the prefill waits for 86 C"),
        "[tensorfold] heat: 85.9 C after 4 s; the prefill goes on",
        "[tensorfold] heat: hottest zone 93.0 C above 92 C; the prefill waits for 85.9 C",
        "[tensorfold] heat: 85.5 C after 2 s; the prefill goes on"]


def test_a_rise_across_the_gap_before_a_prompt_does_not_count(zones):
    """2026-10-15, two 38,962-token prompts back to back at 92 / 88 / 93 C: the second began at 81.9 C after the gap
    between them, its first chunk took the zone back to 87.8 C, and that 5.9 C, counted as a rise, made every later
    chunk wait (19 waits, 67.0 s against 28.9 s). Between chunks the zone rose under 1 C."""

    h = one_rank([56.0, 70.0, 80.0, 84.0, 86.0, 87.0, 87.4, 81.9, 87.8, 88.1, 88.3], zones)
    h.start()
    assert [h.wait(lambda t: t) for _ in range(7)] == [0.0] * 7      # 80.0 + 10, 84.0 + 4, ..., 87.4 + 0.4
    h.start()
    assert [h.wait(lambda t: t) for _ in range(4)] == [0.0] * 4      # 87.8 + 0.4 (the last prompt's), 88.1 + 0.3


def test_after_a_wait_the_next_chunk_is_judged_by_a_rise_between_chunks(zones):
    # a 5.9 C rise between chunks waits at 87.8 C for 87.1; the zone reads 69.3 after 2 s. The chunk after the wait
    # takes it to 85.4 (16.1 C from the wait's end), but the next chunk begins straight after it: 85.4 + 5.9 = 91.3 goes
    h = one_rank([75.0, 81.9, 87.8, 69.3, 85.4, 86.0], zones)
    h.start()
    assert [h.wait(lambda t: t) for _ in range(5)] == [0.0, 0.0, 2.0, 0.0, 0.0]


def test_where_each_chunk_must_wait_the_waits_are_as_long_as_before(zones, capsys):
    """Late in a 1M-token prompt (7 C between chunks): a wait before every chunk, each to CEILING less the rise from
    the last wait's end, as when every rise was counted from where its chunk began."""

    h = one_rank([74.0, 81.0, 88.0, 75.0, 89.0, 74.0, 88.5, 73.0], zones)
    h.start()
    assert [h.wait(lambda t: t) for _ in range(5)] == [0.0, 0.0, 2.0, 2.0, 2.0]
    floors = [line.rsplit("waits for ", 1)[1] for line in capsys.readouterr().out.splitlines() if "waits for" in line]
    assert floors == ["86 C", "79 C", "78.5 C"]


def test_a_prompt_starts_without_the_last_prompts_rise(zones):
    for start, waited in ((True, 0.0), (False, 2.0)):
        h = one_rank([80.0, 86.0, 90.0, 85.0], zones)
        h.start()
        assert [h.wait(lambda t: t) for _ in range(2)] == [0.0, 0.0]          # 86 + 6 = 92: under 93
        if start:
            h.start()
        assert h.wait(lambda t: t) == waited               # a first chunk at 90 C; with the old 4 C rise, 94


def test_a_chunk_begun_with_no_zone_read_has_no_rise(zones):
    # a reading with no readable zone on any rank is -inf: the next chunk's rise is not infinite (a wait without end)
    h = one_rank([float("-inf"), 90.0], zones)
    h.start()
    assert [h.wait(lambda t: t) for _ in range(2)] == [0.0, 0.0]


def test_every_rank_takes_the_rise_of_the_gathered_maximum(zones):
    # after the prompt's first chunk the hottest rank changes between the checks: 80.6 on rank 1, then 87.6 on rank 2;
    # the gathered rise is 7 C (each rank's own is -10.6 to 8.6), so every rank waits for 86 C and leaves at 85.5
    out, comm = three_ranks([[60.0, 70.0, 75.0], [60.0, 80.6, 70.0], [60.0, 79.0, 87.6, 87.0, 85.5]], zones,
                            ceiling=93.0, chunks=3)
    assert out == [(4.0, 2)] * 3
    assert comm.sizes == [[1] * 5] * 3


def test_one_rank_reads_its_own_zones():
    class W:
        comm, world = None, 1

    assert _hottest(W, "cpu")(93.25) == 93.25
