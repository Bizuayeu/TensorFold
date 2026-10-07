"""TF_GLM_HEAT_HIGH and TF_GLM_HEAT_LOW (degrees C, both or neither; unset, off): a prompt's prefill waits before a
chunk while the hottest thermal zone of any rank is above HIGH, every rank together, until the hottest of all is at or
below LOW. The wait changes when a chunk runs, never what it computes, so the bits are those of a run without it.

Before every chunk each rank reads its hottest zone (TF_GLM_HEAT_ZONES, a glob of millidegree files, default the
host's ACPI zones under /sys/class/thermal) and the ranks all-gather that one number; every rank decides from the
gathered maximum, so all of them gather the same count of times and enter and leave the wait together (one rank
waiting alone would leave the others spinning their GPUs in the exchange). While waiting they read and gather again
every ``EVERY`` seconds. There is no cap on a wait: a room that stays hot holds the request.

TF_GLM_HEAT_CEILING (degrees C, above HIGH, with the bands; unset, off) looks one chunk ahead: a chunk heats about as
much as the one before it, so the prefill also waits while the hottest plus the last chunk's rise is above CEILING,
until that and LOW both hold. The rise is the gathered maximum now less the one the last chunk began at (the end of
its wait, if it waited), at least 0; a prompt's first chunk has none. It is the same on every rank, from the same
gathered numbers."""

from __future__ import annotations

import glob
import os
import time
from collections.abc import Callable

ZONES = "/sys/class/thermal/thermal_zone*/temp"
EVERY = 2.0              # s between readings while waiting (the period of the hosts' thermal watch)
REPORT = 60.0            # s between lines while a wait goes on (printed like the engine's other lines, on each rank)


def _degrees(env, name: str) -> float | None:
    raw = env.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}: {raw!r} is not a temperature in degrees C") from None


class Heat:
    def __init__(self, high: float, low: float, zones: str = ZONES, *, ceiling: float | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        if not low < high:
            raise ValueError(f"TF_GLM_HEAT_LOW ({low}) must be below TF_GLM_HEAT_HIGH ({high})")
        if ceiling is not None and not high < ceiling:
            raise ValueError(f"TF_GLM_HEAT_HIGH ({high}) must be below TF_GLM_HEAT_CEILING ({ceiling})")
        self.high, self.low, self.ceiling, self.zones = high, low, ceiling, zones
        self.sleep, self.clock = sleep, clock
        self.began: float | None = None          # the gathered maximum the last chunk began at; None: a prompt's first
        if not glob.glob(zones):
            raise ValueError(f"TF_GLM_HEAT_ZONES: no file matches {zones!r}")

    @classmethod
    def from_env(cls, env=None) -> Heat | None:
        env = os.environ if env is None else env
        high, low, ceiling = (_degrees(env, f"TF_GLM_HEAT_{k}") for k in ("HIGH", "LOW", "CEILING"))
        if high is None and low is None:
            if ceiling is not None:
                raise ValueError("TF_GLM_HEAT_CEILING needs TF_GLM_HEAT_HIGH and TF_GLM_HEAT_LOW")
            return None
        if high is None or low is None:
            raise ValueError("TF_GLM_HEAT_HIGH and TF_GLM_HEAT_LOW are set together")
        return cls(high, low, env.get("TF_GLM_HEAT_ZONES", "").strip() or ZONES, ceiling=ceiling)

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

    def start(self) -> None:
        """A prompt begins: its first chunk has no rise."""

        self.began = None

    def wait(self, hottest: Callable[[float], float]) -> float:
        """Before a chunk: read, take every rank's maximum through ``hottest`` and wait while it is above HIGH or, with
        the last chunk's rise, above CEILING, until it is at or below LOW and the rise keeps it at or below CEILING;
        the seconds waited."""

        top = hottest(self.read())
        rise = 0.0 if self.began in (None, float("-inf")) else max(0.0, top - self.began)    # -inf: no zone read
        ceiling = float("inf") if self.ceiling is None else self.ceiling
        if top <= self.high and top + rise <= ceiling:
            self.began = top
            return 0.0
        floor = min(self.low, ceiling - rise)
        t0 = last = self.clock()
        why = (f"above {self.high:g} C" if top > self.high else
               f"plus the last chunk's {rise:.1f} C rise above {ceiling:g} C")
        print(f"[tensorfold] heat: hottest zone {top:.1f} C {why}; the prefill waits for {floor:g} C", flush=True)
        while top > floor:
            self.sleep(EVERY)
            top = hottest(self.read())
            if self.clock() - last >= REPORT:
                last = self.clock()
                print(f"[tensorfold] heat: waiting {last - t0:.0f} s, hottest zone {top:.1f} C", flush=True)
        waited = self.clock() - t0
        print(f"[tensorfold] heat: {top:.1f} C after {waited:.0f} s; the prefill goes on", flush=True)
        self.began = top
        return waited
