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


def stated_reset(error, at):
    """The reset time a limit error states, as an aware datetime, else None. The boundary parse of the vendor's sentence,
    in one place: a sentence in none of the forms above gives None (the latch's re-probe period applies), never a guess.
    `at` is when the error was observed: a countdown counts from it, and a bare clock time is its next occurrence."""
    if not isinstance(error, str):
        return None
    at = _time(at) if isinstance(at, str) else at
    if at is None:
        return None
    if (m := _RESET_DATE.search(error)):
        words = re.findall(r'[A-Za-z]+|\d+', m.group(1))     # month, day, year, hour, minute, AM/PM
        text = f'{words[0][:3].title()} {words[1]} {words[2]} {words[3]}:{words[4]} {words[5].upper()}'
        try:
            return datetime.strptime(text, '%b %d %Y %I:%M %p').astimezone()
        except ValueError:
            return None
    if (m := _RESET_CLOCK.search(error)):
        hour, minute, half = int(m.group(1)), int(m.group(2)), (m.group(3) or '').upper()
        if half:
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if half == 'PM' else 0)
        if hour > 23 or minute > 59:
            return None
        local = at.astimezone()
        moment = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if moment <= local:
            moment = (local.replace(tzinfo=None) + timedelta(days=1)).replace(
                hour=hour, minute=minute, second=0, microsecond=0).astimezone()
        return moment
    if (m := _RESET_COUNTDOWN.search(error)):
        return at + timedelta(seconds=int(m.group(1)))
    return None


def record(account, error, *, windows=None, at=None, source=None, path=None):
    """Latch the quota windows `windows` (the whole account when None) for an error that `confirmed` reads as a used-up
    quota, with the reset the error states (`stated_reset`) when it states one. The error text is not stored."""
    if not confirmed(error):
        return False
    path = path or store_path()
    at = at or datetime.now(timezone.utc).isoformat()
    resets = stated_reset(error, _time(at))
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
        override_store.write(path, doc)
    return True


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
