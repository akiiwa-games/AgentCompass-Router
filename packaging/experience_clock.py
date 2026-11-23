"""The simulated clock and usage arithmetic of the experience run (packaging/experience_run.py).

Pure functions and small value types, no I/O: how a scenario's quota windows move when its clock advances and when a
launched worker is charged to them. The experience run serves these windows to the router as the provider usage fixture's
readings; the router never sees this module.

Time is in simulated hours from the start of the scenario. A window has a length (`hours`), a used share in percent and
the simulated hour at which it next resets. When the clock passes that hour the window resets: its used share returns to
zero and the next reset is one window length later. A worker is charged when it launches, all at once, by
`cost_pct` of each window it draws (the router's own reservation would hold that much while the worker runs).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Window:
    """One quota window of one account, on the scenario's simulated clock."""
    id: str
    hours: float
    used: float
    reset_at: float
    models: list | None = None          # a window scoped to models (Claude's weekly_fable), as the provider reports it
    models_except: list | None = None

    def remaining_hours(self, now: float) -> float:
        return max(0.0, self.reset_at - now)


@dataclass
class ResetEvent:
    account: str
    window: str
    at: float
    used_before: float

    @property
    def unused(self) -> float:
        return max(0.0, 100.0 - self.used_before)


@dataclass
class Charge:
    account: str
    window: str
    at: float
    pct: float
    applied: float                      # what fitted under 100 %; pct - applied is work beyond the window's capacity
    used_after: float


@dataclass
class AccountUsage:
    name: str
    windows: list[Window] = field(default_factory=list)

    def window(self, wid: str) -> Window | None:
        return next((w for w in self.windows if w.id == wid), None)


def start_window(wid: str, hours: float, used: float, elapsed: float) -> Window:
    """A window `elapsed` (0..1) through its period with `used` percent spent at simulated hour 0."""
    elapsed = min(max(float(elapsed), 0.0), 0.999)
    return Window(wid, float(hours), min(max(float(used), 0.0), 100.0), round(float(hours) * (1.0 - elapsed), 6))


def advance(accounts: list[AccountUsage], start: float, end: float) -> list[ResetEvent]:
    """Move every window from `start` to `end` (simulated hours); returns the resets that happened, in time order."""
    if end < start:
        raise ValueError(f"the clock does not run backwards ({start} -> {end})")
    events = []
    for account in accounts:
        for window in account.windows:
            while window.reset_at <= end:
                events.append(ResetEvent(account.name, window.id, window.reset_at, window.used))
                window.used = 0.0
                window.reset_at = round(window.reset_at + window.hours, 6)
    return sorted(events, key=lambda e: (e.at, e.account, e.window))


def cost_pct(hours: float, usd_per_hour: float | None, capacity_usd: float | None) -> float | None:
    """Percent of a window `hours` active worker-hours of a model spend: the hour's list-price dollars over the window's
    dollars, the same quotient as quota_economy.unit_rate. None when either number is unknown (no cited data)."""
    if not usd_per_hour or not capacity_usd:
        return None
    return 100.0 * float(hours) * float(usd_per_hour) / float(capacity_usd)


def charge(account: AccountUsage, wid: str, pct: float, now: float) -> Charge | None:
    """Add `pct` to one window, capped at 100 %; the part over the cap is returned in the Charge, never dropped."""
    window = account.window(wid)
    if window is None or pct is None:
        return None
    room = max(0.0, 100.0 - window.used)
    applied = min(room, max(0.0, float(pct)))
    window.used = round(window.used + applied, 6)
    return Charge(account.name, wid, now, float(pct), applied, window.used)


def snapshot(accounts: list[AccountUsage], now: float) -> dict:
    """{account: {window: {used, resets_in_hours}}} at `now`, for the timeline."""
    return {a.name: {w.id: {"used": round(w.used, 3), "resets_in": round(w.remaining_hours(now), 3)} for w in a.windows}
            for a in accounts}
