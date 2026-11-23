"""The experience run's simulated clock and usage arithmetic (packaging/experience_clock.py)."""
import pytest

import experience_clock as clock


def account(*windows):
    return clock.AccountUsage("acct", list(windows))


def test_start_window_places_the_reset_by_the_elapsed_share():
    window = clock.start_window("five_hour", 5, 40, 0.4)
    assert (window.used, window.reset_at) == (40, 3.0)
    assert clock.start_window("usage", 168, 0, 0.0).reset_at == 168
    assert clock.start_window("x", 10, 150, 0.5).used == 100      # a reading never exceeds 100 %


def test_advance_resets_a_window_once_per_period_passed_and_reports_what_was_left():
    short = clock.Window("five_hour", 5, 60, 2.0)
    long = clock.Window("usage", 168, 30, 100.0)
    events = clock.advance([account(short, long)], 0.0, 13.0)
    assert [(e.window, e.at, e.used_before, e.unused) for e in events] == [("five_hour", 2.0, 60, 40), ("five_hour", 7.0, 0.0, 100.0),
                                                                             ("five_hour", 12.0, 0.0, 100.0)]
    assert (short.used, short.reset_at) == (0.0, 17.0)
    assert (long.used, long.reset_at) == (30, 100.0)


def test_advance_refuses_to_run_backwards():
    with pytest.raises(ValueError):
        clock.advance([account(clock.Window("w", 5, 0, 5))], 3.0, 2.0)


def test_cost_is_dollars_over_capacity_and_unknown_without_either():
    assert clock.cost_pct(2.0, 3.0, 60.0) == pytest.approx(10.0)
    assert clock.cost_pct(1.0, None, 60.0) is None
    assert clock.cost_pct(1.0, 3.0, None) is None


def test_charge_caps_at_full_and_keeps_the_part_that_did_not_fit():
    usage = account(clock.Window("five_hour", 5, 90, 3.0))
    done = clock.charge(usage, "five_hour", 25.0, 1.0)
    assert (done.pct, done.applied, done.used_after) == (25.0, pytest.approx(10.0), 100.0)
    again = clock.charge(usage, "five_hour", 5.0, 1.5)
    assert (again.applied, again.used_after) == (0.0, 100.0)
    assert clock.charge(usage, "missing", 5.0, 1.0) is None


def test_snapshot_reads_used_and_hours_to_reset():
    usage = account(clock.Window("five_hour", 5, 12.5, 4.0))
    assert clock.snapshot([usage], 1.5) == {"acct": {"five_hour": {"used": 12.5, "resets_in": 2.5}}}
