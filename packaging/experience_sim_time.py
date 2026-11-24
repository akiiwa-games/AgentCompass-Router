"""The experience run's simulated clock as the time a process reads (packaging/experience_run.py).

Installed into the run's sandbox virtualenv as `xp_sim_time.py` with a `xp_sim_time.pth` line that imports it at start-up,
so every interpreter of the installed product reads it before any product module is imported. It does nothing unless the
environment names a clock file (`XP_CLOCK_FILE`, one per scenario): then `time.time`, `time.time_ns`,
`datetime.datetime.now` / `utcnow` / `today` and `datetime.date.today` answer the real time plus the offset in that file
(seconds, a float the run rewrites when its clock moves). Durations (`time.monotonic`, `time.perf_counter`, sleeps,
timeouts) stay real. The run's fixtures that are not this interpreter (the fake Paseo host, the fixture CLI) read the same
file with `offset()`.

Nothing in the router is replaced: the product reads its clock where it always does; only the clock answers simulated
time, the way the provider usage fixture answers simulated windows.
"""
from __future__ import annotations

import datetime as _dt
import os
import time as _time

FILE_ENV = "XP_CLOCK_FILE"
_real_time = _time.time
_real_time_ns = _time.time_ns
_real_datetime = _dt.datetime
_real_date = _dt.date
_cache = {"at": -1.0, "offset": 0.0}
RECHECK_SECONDS = 0.2


def offset(path: str | None = None) -> float:
    """Seconds the simulated clock is ahead of the real one: the clock file's number, 0.0 without a file or a number."""
    path = path or os.environ.get(FILE_ENV)
    if not path:
        return 0.0
    try:
        with open(path, encoding="utf-8") as handle:
            return float(handle.read().strip() or 0.0)
    except (OSError, ValueError):
        return 0.0


def _offset() -> float:
    now = _time.monotonic()
    if now - _cache["at"] >= RECHECK_SECONDS:
        _cache["offset"], _cache["at"] = offset(), now
    return _cache["offset"]


def sim_time() -> float:
    return _real_time() + _offset()


def sim_time_ns() -> int:
    return _real_time_ns() + int(_offset() * 1e9)


class _Kind(type(_real_datetime)):
    """isinstance / issubclass against the replacement classes answer as against the real ones."""

    def __instancecheck__(cls, obj):
        return isinstance(obj, cls.__real__)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, cls.__real__)


class SimDatetime(_real_datetime, metaclass=_Kind):
    __real__ = _real_datetime

    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(sim_time(), tz)

    @classmethod
    def utcnow(cls):
        return cls.fromtimestamp(sim_time(), _dt.timezone.utc).replace(tzinfo=None)

    @classmethod
    def today(cls):
        return cls.fromtimestamp(sim_time())


class SimDate(_real_date, metaclass=_Kind):
    __real__ = _real_date

    @classmethod
    def today(cls):
        return cls.fromtimestamp(sim_time())


def install() -> bool:
    """Answer simulated time from now on in this interpreter, when a clock file is named; True when installed."""
    if not os.environ.get(FILE_ENV) or _time.time is sim_time:
        return False
    _time.time, _time.time_ns = sim_time, sim_time_ns
    _dt.datetime, _dt.date = SimDatetime, SimDate
    return True


def write(path, seconds: float) -> None:
    """Set the offset (the run's side): written whole and replaced, so a reader never sees half a number."""
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(repr(float(seconds)))
    os.replace(temporary, path)


install()
