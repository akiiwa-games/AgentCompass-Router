"""Durable confirmed quota failures, independent of optimistic usage caches.

Contract: an explicit quota-exhausted response latches the quota windows the failing
model draws (the whole account when they are unknown); other windows stay usable.
Network errors and generic 429s do not qualify. A latched window is released by a
successful reading newer than the latch that shows it under 100% used, by that
reading's reset time passing, or by an explicit clear; an account-wide latch narrows
to the windows such a reading shows exhausted. No latch waits for a person (owner,
2026-11-24): every latch is also due at a time (`due_at`), after which the next roll
may pick its candidates again and so re-probes the quota: the reset the error itself
states (`stated_reset`, the one parse of a vendor's sentence; an allowance with no
reader is held exactly until then), else the latches' shared re-probe period
(account_state.LATCH_RETRY_HOURS) counted from the observation, which is not a guess at
recovery. No remote error text is stored.

A used-up quota with a known reset (a retry-after the vendor gave as data, else the sentence's own) is also an
observation of the allowance's size (`observation`), with the period when the sentence states one ("today's free
messages"): kept apart from the latch, so a release or a clear keeps it, and replaced by the next one;
axis_scores.free_allowance_period reads the period's spend from it.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import account_state
import override_store


def store_path():
    return Path(os.environ.get('ROUTER_STATE_DIR') or Path.home() / '.agent-compass') / 'quota-exhaustion.json'


def _read(path):
    try:
        return json.loads(override_store.retry_denied(lambda: path.read_text(encoding='utf-8')))
    except FileNotFoundError:
        return {'accounts': {}}


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result if result.tzinfo else result.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


_CONFIRMED = re.compile(
    r'individual quota reached|quota[_ ](?:exceeded|exhausted)|insufficient_quota|'
    r'(?:usage|credit|spending|weekly|monthly) limit (?:reached|exceeded)|'
    r'usage_limit_reached|insufficient credits|额度(?:已)?(?:耗尽|用尽)|'
    # A free allowance used up (OpenCode's hosted free tier, anomalyco/opencode#59387, 2026-10-26: "Free usage exceeded,
    # subscribe to Go", "You've used today's free messages. They reset at ...").
    r"free usage exceeded|used today['’]s free messages", re.I)


def confirmed(error):
    """True when a provider's error text explicitly says a quota is used up (network errors and generic 429s are not)."""
    return isinstance(error, str) and bool(_CONFIRMED.search(error))


# The reset a limit error states, in the forms the vendor's own sentences use (anomalyco/opencode#59387 and the issues
# it collects, read 2026-11-24): a wall-clock time ("They reset at 21:00", "They reset at 11:59 PM"), a date and time
# ("They reset at Oct 28, 2026, 12:59AM"), both shown in the user's local time, and a countdown in seconds
# ("reintentando en 74000s", the client's "retrying in" line).
_RESET_DATE = re.compile(r'resets? at\s+([A-Za-z]{3,9}\.? \d{1,2},? \d{4},? \d{1,2}:\d{2}\s?[AaPp][Mm])')
_RESET_CLOCK = re.compile(r'resets? at\s+(\d{1,2}):(\d{2})(?:\s?([AaPp][Mm]))?(?![\d:])')
_RESET_COUNTDOWN = re.compile(r'(?:retrying in|reintentando en)\s+(\d+)\s*s\b', re.I)
# A zone written right after the time: an IANA name ("(America/New_York)") or UTC/GMT with an optional offset.
_ZONE = re.compile(r'\s*\(?\s*(?:(?P<iana>[A-Za-z]+/[A-Za-z_]+(?:/[A-Za-z_]+)?)|'
                   r'(?P<utc>UTC|GMT)(?![A-Za-z])\s*(?P<off>[+-]\d{1,2}(?::?\d{2})?)?)')
_UNKNOWN_ZONE = object()


def _stated_zone(error, end):
    """The zone the sentence states right after a time ending at `end`: a tzinfo, None when it states none, or
    _UNKNOWN_ZONE for a zone name no zone database knows (the sentence is then unreadable)."""
    m = _ZONE.match(error, end)
    if not m:
        return None
    if m.group('iana'):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            return ZoneInfo(m.group('iana'))
        except (ZoneInfoNotFoundError, ValueError):
            return _UNKNOWN_ZONE
    offset = m.group('off')
    if not offset:
        return timezone.utc
    sign = -1 if offset[0] == '-' else 1
    digits = offset[1:].replace(':', '')
    hours, minutes = (digits[:-2], digits[-2:]) if len(digits) > 2 else (digits, '0')
    if int(hours) > 14 or int(minutes) > 59:
        return _UNKNOWN_ZONE
    return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))


# The period a limit error names, in the vendor's own words: "You've used today's free messages" is a day's allowance
# (anomalyco/opencode#59387, read 2026-11-24). A sentence that names none gives no period, never a guess.
_PERIODS = ((re.compile(r"\btoday['’]s\b", re.I), 24.0),)


def stated_period(error):
    """The length in hours of the allowance period a limit error names (`_PERIODS`), else None."""
    if not isinstance(error, str):
        return None
    return next((hours for pattern, hours in _PERIODS if pattern.search(error)), None)


def longest_reset_hours():
    """The longest reset period any window of the shipped plan templates declares (hours), the furthest a quota can
    reset from now; None when the templates cannot be read."""
    import provider_inventory
    periods = []
    for plans in provider_inventory.shipped_plan_templates().values():
        for template in (plans.values() if isinstance(plans, dict) else ()):
            if not isinstance(template, dict):
                continue
            for window in (template.get('windows') or {}).values():
                hours = window.get('hours') if isinstance(window, dict) else None
                if isinstance(hours, (int, float)) and not isinstance(hours, bool) and hours > 0:
                    periods.append(hours)
            seconds = (template.get('cadence') or {}).get('window_seconds') if isinstance(template.get('cadence'), dict) else None
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds > 0:
                periods.append(seconds / 3600)
    return max(periods) if periods else None


def _parsed_reset(error, at):
    """The time the sentence states, before the plausibility bounds of `stated_reset`."""
    if (m := _RESET_DATE.search(error)):
        zone = _stated_zone(error, m.end())
        if zone is _UNKNOWN_ZONE:
            return None
        words = re.findall(r'[A-Za-z]+|\d+', m.group(1))     # month, day, year, hour, minute, AM/PM
        text = f'{words[0][:3].title()} {words[1]} {words[2]} {words[3]}:{words[4]} {words[5].upper()}'
        try:
            moment = datetime.strptime(text, '%b %d %Y %I:%M %p')
        except ValueError:
            return None
        return moment.replace(tzinfo=zone) if zone else moment.astimezone()
    if (m := _RESET_CLOCK.search(error)):
        zone = _stated_zone(error, m.end())
        if zone is _UNKNOWN_ZONE:
            return None
        hour, minute, half = int(m.group(1)), int(m.group(2)), (m.group(3) or '').upper()
        if half:
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if half == 'PM' else 0)
        if hour > 23 or minute > 59:
            return None
        # The next occurrence of that wall time in the stated zone, else in the machine's local time.
        wall = at.astimezone(zone) if zone else at.astimezone()
        day = wall.replace(tzinfo=None)
        for days in (0, 1, 2):
            naive = (day + timedelta(days=days)).replace(hour=hour, minute=minute, second=0, microsecond=0)
            moment = naive.replace(tzinfo=zone) if zone else naive.astimezone()
            if moment > at:
                return moment
        return None
    if (m := _RESET_COUNTDOWN.search(error)):
        try:
            return at + timedelta(seconds=int(m.group(1)))
        except OverflowError:
            return None
    return None


def stated_reset(error, at):
    """The reset time a limit error states, as an aware datetime, else None. The boundary parse of the vendor's sentence,
    in one place: a sentence in none of the forms above gives None (the latch's re-probe period applies), never a guess.
    `at` is when the error was observed: a countdown counts from it, and a bare clock time is its next occurrence. A
    zone the sentence states is used; a time with none is the machine's local time (the vendor's client prints wall
    time for its user). A reset not after `at`, or further from it than the longest reset period any shipped plan
    window has (`longest_reset_hours`), is unreadable too."""
    if not isinstance(error, str):
        return None
    at = _time(at) if isinstance(at, str) else at
    if at is None:
        return None
    return _bounded(_parsed_reset(error, at), at)


def _bounded(moment, at):
    """`moment` when it lies after `at` and no further than the longest reset period of the shipped plans, else None."""
    longest = longest_reset_hours()
    if moment is None or longest is None or not at < moment <= at + timedelta(hours=longest):
        return None
    return moment


def record(account, error, *, windows=None, at=None, source=None, path=None, resets_at=None):
    """Latch the quota windows `windows` (the whole account when None) for an error that `confirmed` reads as a used-up
    quota, with its reset: `resets_at`, a time the vendor gave as data (a retry-after field, read by the caller), beats
    the reset the sentence states (`stated_reset`); both are held to the bounds of `stated_reset`. The error text is not
    stored."""
    if not confirmed(error):
        return False
    path = path or store_path()
    at = at or datetime.now(timezone.utc).isoformat()
    given = _time(resets_at) if isinstance(resets_at, str) else resets_at
    resets = _bounded(given, _time(at)) or stated_reset(error, _time(at))
    with override_store.locked(path):
        doc = _read(path)
        previous = doc.setdefault('accounts', {}).get(account) or {}
        # Replaying an old observation must not resurrect a manually cleared latch.
        if (_time(previous.get('observed_at')) or datetime.min.replace(tzinfo=timezone.utc)) >= _time(at):
            return False
        entry = {'observed_at': at, 'source': source, 'blocked': True,
                 **({'resets_at': resets.isoformat()} if resets else {})}
        # Windows accumulate while latched; unknown windows latch the whole account.
        if windows and not (previous.get('blocked') and not previous.get('windows')):
            entry['windows'] = sorted(set(windows) | set(previous.get('windows') or [] if previous.get('blocked') else []))
        doc['accounts'][account] = entry
        period = stated_period(error)
        if resets and (period is None or resets - timedelta(hours=period) <= _time(at)):
            # The pool was at 0 at `at` with a known reset: what it spent since the period began (the reset less the
            # period, stated here or declared with the plan's measurement) is this account's observed capacity, read by
            # `observation`. Kept apart from the latch, so a release or a clear keeps it; the next one replaces it.
            doc.setdefault('observations', {})[account] = {
                'observed_at': at, 'resets_at': resets.isoformat(), **({'period_hours': period} if period else {}),
                **({'windows': sorted(set(windows))} if windows else {})}
        override_store.write(path, doc)
    return True


def observation(account, *, path=None):
    """The account's latest observation of a used-up allowance with a known reset: {'observed_at', 'resets_at' (aware
    datetimes), 'period_hours' (when the error stated one), 'windows'?}, else None. Local state of this machine, like a
    usage reading: never public data and never a setting."""
    row = (_read(path or store_path()).get('observations') or {}).get(account)
    if not isinstance(row, dict):
        return None
    observed, resets, period = _time(row.get('observed_at')), _time(row.get('resets_at')), row.get('period_hours')
    if period is not None and (isinstance(period, bool) or not isinstance(period, (int, float)) or period <= 0):
        return None
    if observed is None or resets is None or not observed < resets or (
            period is not None and not resets - timedelta(hours=period) <= observed):
        return None
    return {'observed_at': observed, 'resets_at': resets, **({'period_hours': float(period)} if period else {}),
            **({'windows': list(row['windows'])} if isinstance(row.get('windows'), list) else {})}


ALL = '*'


def due_at(row):
    """When a latch ends by itself: the reset its error stated, else its observation plus the latches' re-probe period.
    None for a row with no time."""
    stated = _time((row or {}).get('resets_at'))
    if stated is not None:
        return stated
    observed = _time((row or {}).get('observed_at'))
    return observed + timedelta(hours=account_state.LATCH_RETRY_HOURS) if observed else None


def stated_reset_of(row):
    """The reset the latch's error stated, else None."""
    return _time((row or {}).get('resets_at'))


def moment_text(moment):
    """A latch's end as a person reads it: local time to the minute with its offset."""
    return moment.astimezone().isoformat(timespec='minutes') if moment else 'the next roll'


def ends(account, *, path=None):
    """(when the account's latch ends by itself, whether that is the reset its error stated), or None when it has no
    latch. The one answer every text about a latch gives (`due_at`)."""
    row = (_read(path or store_path()).get('accounts') or {}).get(account) or {}
    if not row.get('blocked'):
        return None
    return due_at(row), stated_reset_of(row) is not None


def end_text(account, *, path=None):
    """The sentence that says when the account's latch ends by itself: at the reset its error stated, else when it is
    re-probed. None when it has no latch."""
    found = ends(account, path=path)
    if found is None:
        return None
    due, stated = found
    return (f'0 left until {moment_text(due)} (the reset the provider stated)' if stated
            else f'0 left until {moment_text(due)}, when it is re-probed')


def exhausted_windows(account, usage=None, *, path=None, now=None, read_only=False):
    """Window ids still latched for this account; {ALL} for an account-wide latch, empty when free. A fresh reading that
    shows recovery clears or narrows the latch on disk; `read_only` returns the same answer and writes nothing."""
    path = path or store_path()
    row = (_read(path).get('accounts') or {}).get(account) or {}
    if not row.get('blocked'):
        return set()
    latched = set(row.get('windows') or []) or {ALL}
    usage = usage or {}
    now = now or datetime.now(timezone.utc)
    due = due_at(row)
    if due is not None and due <= now:
        # Due: the next roll re-probes the quota (a launch that meets the limit again latches it again).
        if read_only:
            return set()
        return set() if clear(account, path=path, expected_observed_at=row['observed_at']) else latched
    fetched = _time(usage.get('fetched_at'))
    observed = _time(row.get('observed_at'))
    readings = {w['id']: w for w in usage.get('windows') or [] if isinstance(w, dict) and isinstance(w.get('id'), str)}
    remaining = latched
    if usage.get('available') is True and not usage.get('stale') and fetched and observed and fetched > observed and readings:
        exhausted = set()
        for wid, window in readings.items():
            used = window.get('used_pct')
            reset = _time(window.get('resets_at'))
            if isinstance(used, (int, float)) and 0 <= used < 100 or reset is not None and reset <= now:
                continue
            exhausted.add(wid)
        remaining = exhausted if ALL in latched else (latched & exhausted) | (latched - set(readings))
    if remaining == latched:
        return latched
    if read_only:
        return remaining
    if not remaining:
        return set() if clear(account, path=path, expected_observed_at=row['observed_at']) else latched
    with override_store.locked(path):
        doc = _read(path)
        current = doc.setdefault('accounts', {}).get(account) or {}
        if current.get('observed_at') != row['observed_at']:
            return latched  # A concurrent, newer failure outranks this recovery observation.
        current['windows'] = sorted(remaining)
        override_store.write(path, doc)
    return remaining


def stopping(account, usage=None, *, path=None, windows=None, now=None, read_only=False):
    """The latched windows that stop a launch drawing `windows`: {ALL} when the latch covers the whole account, else
    the latched ones among `windows` (none when `windows` is unknown); empty when the launch is free."""
    latched = exhausted_windows(account, usage, path=path, now=now, read_only=read_only)
    return {ALL} if ALL in latched else latched & set(windows or ())


def clear(account, *, path=None, expected_observed_at=None):
    path = path or store_path()
    with override_store.locked(path):
        doc = _read(path)
        row = doc.setdefault('accounts', {}).setdefault(account, {})
        if expected_observed_at is not None and row.get('observed_at') != expected_observed_at:
            return False  # A concurrent, newer failure outranks this recovery observation.
        row.update(blocked=False, observed_at=datetime.now(timezone.utc).isoformat())
        override_store.write(path, doc)
        return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clear', metavar='ACCOUNT', help='Explicitly clear one confirmed quota failure')
    args = parser.parse_args()
    if args.clear:
        clear(args.clear)
    print(json.dumps(_read(store_path()), ensure_ascii=False, indent=2))
