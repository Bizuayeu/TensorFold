"""TF_GLM_HEAT_HIGH and TF_GLM_HEAT_LOW (degrees C, both or neither; unset, off): a prompt's prefill waits before a
chunk while the hottest thermal zone of any rank is above HIGH, every rank together, until the hottest of all is at or
below LOW. The wait changes when a chunk runs, never what it computes, so the bits are those of a run without it.

Before every chunk each rank reads its hottest zone (TF_GLM_HEAT_ZONES, a glob of millidegree files, default the
host's ACPI zones under /sys/class/thermal) and the ranks all-gather that one number; every rank decides from the
gathered maximum, so all of them gather the same count of times and enter and leave the wait together (one rank
waiting alone would leave the others spinning their GPUs in the exchange). While waiting they read and gather again
every ``EVERY`` seconds. There is no cap on a wait: a room that stays hot holds the request."""

from __future__ import annotations

import glob
import logging
import os
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

ZONES = "/sys/class/thermal/thermal_zone*/temp"
EVERY = 2.0              # s between readings while waiting (the period of the hosts' thermal watch)
REPORT = 60.0            # s between log lines while a wait goes on


class Heat:
    def __init__(self, high: float, low: float, zones: str = ZONES, *, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not low < high:
            raise ValueError(f"TF_GLM_HEAT_LOW ({low}) must be below TF_GLM_HEAT_HIGH ({high})")
        self.high, self.low, self.zones = high, low, zones
        self.sleep, self.clock = sleep, clock
        if not glob.glob(zones):
            raise ValueError(f"TF_GLM_HEAT_ZONES: no file matches {zones!r}")

    @classmethod
    def from_env(cls, env=None) -> Heat | None:
        env = os.environ if env is None else env
        high, low = env.get("TF_GLM_HEAT_HIGH", "").strip(), env.get("TF_GLM_HEAT_LOW", "").strip()
        if not high and not low:
            return None
        if not high or not low:
            raise ValueError("TF_GLM_HEAT_HIGH and TF_GLM_HEAT_LOW are set together")
        return cls(float(high), float(low), env.get("TF_GLM_HEAT_ZONES", "").strip() or ZONES)

    def read(self) -> float:
        """This host's hottest zone in degrees C."""

        temps = []
        for path in glob.glob(self.zones):
            try:
                with open(path) as f:
                    temps.append(int(f.read().strip()) / 1000)
            except (OSError, ValueError):
                continue
        return max(temps, default=float("-inf"))

    def wait(self, hottest: Callable[[float], float]) -> float:
        """Read, take every rank's maximum through ``hottest`` and wait while it is above HIGH, until it is at or
        below LOW; the seconds waited."""

        top = hottest(self.read())
        if top <= self.high:
            return 0.0
        t0 = last = self.clock()
        log.info("heat: hottest zone %.1f C above %.1f C; the prefill waits for %.1f C", top, self.high, self.low)
        while top > self.low:
            self.sleep(EVERY)
            top = hottest(self.read())
            if self.clock() - last >= REPORT:
                last = self.clock()
                log.info("heat: waiting %.0f s, hottest zone %.1f C", last - t0, top)
        waited = self.clock() - t0
        log.info("heat: %.1f C after %.0f s; the prefill goes on", top, waited)
        return waited
