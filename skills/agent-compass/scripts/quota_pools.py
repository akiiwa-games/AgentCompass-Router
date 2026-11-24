"""Quota windows and pools: which windows a bundle draws, their state, and the launch blockers they raise."""

from datetime import datetime, timedelta, timezone
import math
import re

import window_scope
import provider_inventory
import usage_reading
import account_fleet
import axis_scores
import quota_economy


WINDOW_ALIASES = {'seven_day': 'usage', 'usage-10080m': 'usage', 'usage-5h': 'five_hour', 'usage-300m': 'five_hour'}


def canonical_window_id(spec, wid):
    windows = (spec or {}).get('windows') or {}
    if wid in windows:
        return wid
    mapped = WINDOW_ALIASES.get(wid)
    return mapped if mapped in windows else wid


def live_dispatch_holds(records):
    """One live hold per worker. The newest row for an agent_id counts; a row with no agent_id still counts."""
    bare = []
    newest = {}
    for rec in records or ():
        if not isinstance(rec, dict):
            continue
        agent = rec.get('agent_id')
        if not isinstance(agent, str) or not agent:
            bare.append(rec)
            continue
        prev = newest.get(agent)
        if prev is None or _hold_is_newer(rec, prev):
            newest[agent] = rec
    return bare + list(newest.values())


def _hold_is_newer(rec, prev):
    rec_at = parse_time(rec.get('at') or rec.get('created_at') or rec.get('updated_at'))
    prev_at = parse_time(prev.get('at') or prev.get('created_at') or prev.get('updated_at'))
    if rec_at is not None and (prev_at is None or rec_at >= prev_at):
        return True
    if rec_at is None and prev_at is None:
        return str(rec.get('dispatch_id') or '') >= str(prev.get('dispatch_id') or '')
    return False


def inflight_reserved_usd(account, spec, wid, wspec, state, win_cap_usd, records, accounts, policy, now):
    """Hold API-equivalent budget only for live dispatch leases, once per dispatch ID.

    Active commitments remain fully reserved until terminal/release/lease expiry.
    One worker's older open round is not added to its newer round. This is conservative
    while usage catches up; provider percentages cannot reliably attribute consumption
    to one concurrent worker. Plan/select/record never reserve.
    """
    if not win_cap_usd or win_cap_usd <= 0:
        return 0.0
    active = {}
    for rec in live_dispatch_holds(records):
        if not isinstance(rec, dict) or rec.get('stage') != 'dispatch' or not rec.get('dispatch_id'):
            continue
        expiry = parse_time(rec.get('lease_until'))
        if rec.get('state') not in ('prepared', 'launching', 'launched', 'started') or expiry is None or expiry <= now:
            continue
        bundle = rec.get('bundle') or {}
        if provider_inventory.quota_source(bundle.get('account'), accounts) != provider_inventory.quota_source(account, accounts):
            continue
        bound = dict(bundle)
        bound.setdefault('account', account)
        if wid not in bundle_pools(bound, accounts):
            continue
        amount = rec.get('expected_usd')
        if isinstance(amount, (int, float)) and not isinstance(amount, bool) and amount > 0:
            active[rec['dispatch_id']] = float(amount)
    return sum(active.values())


def parse_time(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def clamp(value, low, high):
    return max(low, min(high, value))


def _explicit_draws(bundle):
    """The bundle's draws list when the key is present, else None. An empty list is explicit."""
    if not isinstance(bundle, dict) or 'draws' not in bundle:
        return None
    raw = bundle.get('draws')
    if not isinstance(raw, (list, tuple)):
        return []
    found = []
    for wid in raw:
        if isinstance(wid, str) and wid and wid not in found:
            found.append(wid)
    return found


def bundle_pools(bundle, accounts, observed_windows=None):
    """Window ids this bundle draws, in declaration order (configured windows, then observed-only ones).

    An explicit `draws` on the passed bundle wins, including an empty list. Otherwise the catalog
    bundle's `draws` wins. Otherwise each window's data-declared scope decides (window_scope.binds on the
    canonical model): the configured spec's `models`/`models_except` and the reading's own `models`.
    A window no source scopes binds every model, observed-only windows included.
    """
    bundle = bundle if isinstance(bundle, dict) else {}
    accounts = accounts if isinstance(accounts, dict) else {}
    specs = accounts.get('accounts') if isinstance(accounts.get('accounts'), dict) else {}
    account = bundle.get('account') if isinstance(bundle.get('account'), str) else None
    spec = specs.get(account) if account else None
    bid = bundle.get('id') or bundle.get('bundle_id')
    if not isinstance(spec, dict):
        spec = None
        for name, candidate in specs.items():
            if isinstance(candidate, dict) and bid and bid in (candidate.get('bundles') or {}):
                account, spec = name, candidate
                break
    configured = spec.get('windows') if isinstance(spec, dict) and isinstance(spec.get('windows'), dict) else {}
    observed_windows = observed_windows or {}
    windows = {**configured, **{wid: {} for wid in observed_windows if wid not in configured}}
    declared = _explicit_draws(bundle)
    if declared is None and isinstance(spec, dict) and bid:
        catalog = (spec.get('bundles') or {}).get(bid)
        if isinstance(catalog, dict):
            declared = _explicit_draws(catalog)
    if declared is not None:
        return [wid for wid in declared if wid in windows]
    names = (account_fleet.canonical(bundle.get('model') or '', accounts), bundle.get('model') or '')
    return [wid for wid, wspec in windows.items() if window_scope.binds_any(names, wspec, observed_windows.get(wid))]


def applicable_windows(account_spec, model, usage_row, bundle=None, accounts=None):
    """(window id, spec, observed window) for windows that bind this model or bundle.

    With a bundle, explicit draws (including []) remain authoritative. Otherwise
    historical model tags and known observed scopes determine applicability.
    Observation-only hard limits have no invented capacity, burn rate or hours.
    `accounts` is the catalogue the model name resolves against (its models, aliases and families).
    """
    observed = {}
    for w in (usage_row or {}).get('windows', []):
        wid = w.get('id')
        if wid:
            observed.setdefault(canonical_window_id(account_spec, wid), w)
    configured = (account_spec or {}).get('windows') or {}
    windows = {**configured, **{wid: {} for wid in observed if wid not in configured}}
    if bundle is None:
        if isinstance(model, str) and model and model in ((accounts or {}).get('models') or {}):
            ids = [wid for wid, wspec in windows.items() if window_scope.binds_any((model,), wspec, observed.get(wid))]
        else:
            ids = bundle_pools({'account': '_billing', 'model': model or ''},
                               {**(accounts or {}), 'accounts': {'_billing': account_spec}}, observed)
    else:
        if accounts is None:
            acct = bundle.get('account') if isinstance(bundle, dict) and isinstance(bundle.get('account'), str) else '_billing'
            accounts = {'accounts': {acct: account_spec}}
            if not isinstance(bundle, dict) or 'account' not in bundle:
                bundle = {**(bundle or {}), 'account': acct}
        ids = bundle_pools(bundle, accounts, observed)
    return [(wid, windows[wid], observed.get(wid)) for wid in ids if wid in windows]


def observed_reset(observed, now):
    """Whether the reading shows a reset still ahead (a future `resets_at`, else a countdown whose reset has not passed)."""
    reset = parse_time((observed or {}).get('resets_at'))
    seconds = (observed or {}).get('seconds_until_reset')
    return (reset is not None and reset > now) or (
        reset is None and isinstance(seconds, (int, float)) and seconds > 0 and (observed or {}).get('reset_time_passed') is not True)


def below_launch_floor(window, observed, usage_row, now):
    """Existing launch protection, shared by ranking and dispatch."""
    floor = window.get('launch_min_remaining_pct')
    reset = parse_time((observed or {}).get('resets_at'))
    active_reset = observed_reset(observed, now)
    if floor is not None and (observed or {}).get('rate_limited') is True and active_reset:
        return True
    fetched = parse_time(usage_row.get('fetched_at'))
    if (usage_row.get('status') != 'fresh' or usage_row.get('stale') is True or usage_row.get('available') is False
            or ('fetched_at' in usage_row and (fetched is None or not -60 <= (now - fetched).total_seconds() <= 900))):
        return False
    remaining = (observed or {}).get('remaining_pct_before_reset')
    if isinstance(remaining, (int, float)) and remaining <= 0 and (reset is None or active_reset):
        return True
    return bool(floor is not None and active_reset
                and isinstance(remaining, (int, float)) and remaining <= floor)


def latched_windows(accounts, account):
    """Windows a confirmed quota failure latched for this account (provider_inventory reads quota_exhaustion)."""
    return set((((accounts.get('_discovery') or {}).get('accounts') or {}).get(account) or {}).get('exhausted_windows') or ())


def latched_reset(accounts, account):
    """When this account's latched windows end by themselves (quota_exhaustion.due_at: the reset the latching error
    stated, else the latch's re-probe time), else None."""
    row = (((accounts.get('_discovery') or {}).get('accounts') or {}).get(account) or {})
    return parse_time(row.get('exhausted_resets_at'))


def latched_end_kind(accounts, account):
    """Which end this account's latch has (quota_exhaustion.end_kind: the reset the provider stated, else the time it
    is re-probed)."""
    import quota_exhaustion
    return quota_exhaustion.end_kind((((accounts.get('_discovery') or {}).get('accounts') or {}).get(account) or {})
                                     .get('exhausted_reset_stated'))


def latched_wait_text(accounts, account, wid):
    """Why a latched window waits: 0 left until its end, and which end that is; None when it has no known end."""
    import quota_exhaustion
    reset = latched_reset(accounts, account)
    if reset is None:
        return None
    return (f'{wid} has 0 left until {quota_exhaustion.moment_text(reset)} '
            f'({latched_end_kind(accounts, account)}); start after it')


def cycle_observation(window, observed, usage_row, now, read_failed_ok=False):
    """The window's reading, or None when a reading that is not current cannot describe the current cycle.

    Usage only grows inside a cycle, so an old reading still bounds what is left until its reset. No reading
    describes the cycle after its own reset has passed. A reading that is not current cannot either when the
    account was unavailable, the reading has no time or is from the future, or it was taken more than the
    window's `hours` before its reset (an earlier cycle). A provider rate-limit observation is always kept.
    `read_failed_ok` (the five-axis path, owner decision of round 13b): an account whose latest read failed still has its
    last successful reading of this cycle, which is current-period data and is kept, labelled by the caller with its read
    time. A failing usage endpoint (even a 429 from it) is a read failure, not a quota signal; quota failures arrive as
    provider holds (`provider_retry`, `model_latch`), which evaluate_bundle applies before any reading is consulted.
    """
    if observed is None or observed.get('rate_limited') is True:
        return observed
    if isinstance(observed.get('members'), list):
        # A pool: each member's reading is kept or dropped as one account's would be, and the pool is what is kept.
        kept = [m for m in (cycle_observation(window, member, usage_row, now, read_failed_ok)
                            for member in observed['members']) if m is not None]
        pooled = usage_reading.pooled_reading(observed.get('id'), kept, observed.get('models'))
        passed = [m.get('position') for m in observed['members'] if m.get('reset_time_passed') is True]
        if pooled is not None and passed:
            # Said where it is read (review N2): which members wait for a new reading, so the pool is the others.
            pooled['passed_members'] = {'positions': passed, 'of': len(observed['members'])}
        return pooled
    if observed.get('reset_time_passed') is True:
        return None
    if usage_row.get('status') in (None, 'fresh'):
        return observed     # current, or a caller that does not classify freshness (usage_reading.usage_snapshot does)
    fetched = parse_time(usage_row.get('fetched_at'))
    if ((usage_row.get('available') is False and not read_failed_ok) or fetched is None
            or (fetched - now).total_seconds() > 60):
        return None
    reset, hours = parse_time(observed.get('resets_at')), window.get('hours')
    if reset is not None and isinstance(hours, (int, float)) and fetched < reset - timedelta(hours=hours):
        return None
    return observed


def used_up_in_cycle(observed, now):
    """A reading from this cycle (cycle_observation) at 0% left holds until its reset, however old it is."""
    if not observed or observed.get('reset_time_passed') is True:
        return False
    left = observed.get('observed_remaining_pct')
    if left is None:
        left = observed.get('remaining_pct_before_reset')
    reset = parse_time(observed.get('resets_at'))
    seconds = observed.get('seconds_until_reset')
    active_reset = (reset is not None and reset > now) or (
        reset is None and isinstance(seconds, (int, float)) and seconds > 0)
    return active_reset and isinstance(left, (int, float)) and left <= 0


def window_refusal(wid, window, observed, usage_row, now, latched=()):
    """Why a drawn window refuses any launch now, whatever the task's size; None when it does not."""
    if wid in latched:
        return f'{wid}: quota confirmed exhausted'
    if below_launch_floor(window, observed, usage_row, now):
        # With no reset ahead in the reading there is no time to start at: the refusal says so.
        return f'{wid}: confirmed exhausted or below launch floor' + (
            '' if observed_reset(observed, now) else '; no reset of it is observed')
    return None


def short_window_limits(spec, wid, policy, now):
    """(small, floor %, safety margin %) of a short window; a small window keeps the larger reserve."""
    cap = quota_economy.capacity_usd(spec, wid)
    small = cap is not None and cap < policy.get('small_short_capacity_usd', 50)
    return (small,
            policy.get('small_short_window_floor_pct', 25) if small else policy.get('short_window_floor_pct', 10),
            policy.get('small_short_safety_margin_pct', 12) if small else policy.get('safety_margin_pct', 5))


def reserve_inflight(account, spec, wid, wspec, state, records, accounts, policy, now):
    """Apply active in-flight reservation deduction to a known window state; returns (state, reserved_usd, reserved_pct)."""
    if not state or not state.get('known') or not records:
        return state, 0.0, 0.0
    win_cap_usd = quota_economy.capacity_usd(spec, wid)
    reserved_usd = inflight_reserved_usd(account, spec, wid, wspec, state, win_cap_usd, records, accounts, policy, now)
    if reserved_usd > 0 and win_cap_usd:
        reserved_pct = min(state['remaining'], round(100.0 * reserved_usd / win_cap_usd, 2))
        if state.get('members'):
            # The in-flight work draws on the members by the one split (shares): never above a member's room.
            members = _draw(state, reserved_pct, win_cap_usd)
            state = {**pool_row({k: state[k] for k in ('id', 'hours', 'short', 'stale') if k in state}, members),
                     **{k: state[k] for k in ('deferred_hours', 'start_offset_hours') if k in state}}
            state.update(observed_remaining=state['remaining'] + reserved_pct, reserved_usd=round(reserved_usd, 2))
            return state, reserved_usd, reserved_pct
        state = {**state, 'observed_remaining': state['remaining'],
                 'remaining': max(0.0, state['remaining'] - reserved_pct),
                 'reserved_usd': round(reserved_usd, 2)}
        return state, reserved_usd, reserved_pct
    return state, 0.0, 0.0


def _draw(state, pct, cap_usd):
    """The pool's members after `pct` (percent of one account, at most what the pool has left) is drawn by `shares`."""
    taken = {m['position']: share * pct for m, share in shares(state, pct)}
    members = state['members']
    return [{**m, 'remaining': m['remaining'] - taken.get(m['position'], 0.0),
             **({'observed_remaining': m['remaining'], 'reserved_usd': round(taken[m['position']] * cap_usd / 100.0, 2)}
                if taken.get(m['position'], 0.0) > 0 else {})} for m in members]


def launch_blockers(bundle, accounts, usage, now, latched=None, records=None):
    """[(window id, reason)] for windows this bundle draws that stop a launch now, whatever the task's size.

    These are the task-independent outcomes of evaluate_bundle for "start now": its exclusions (window_refusal:
    the confirmed-exhaustion latch and the launch floor; a short window with an active reset but no reading) and
    the windows it excludes or defers to their reset, which no task size escapes (a short window at or under its
    floor; a priced window at or under the safety margin; a window this cycle's reading shows used up). Active
    in-flight reservations (`records`) are deducted from remaining capacity just as evaluate_bundle deducts them.
    `usage` is usage_reading.usage_snapshot after provider_inventory.merge_usage; windows the model does not draw
    (window_scope) never appear.

    The capacities a window is priced on are the public data's, as in `plan`: `accounts` should be `five_axis_overlay`
    output, which a caller probing many bundles builds once and passes down (raw accounts are overlaid here, per call).
    """
    accounts = accounts if isinstance(accounts, dict) else {}
    if '_five_axis' not in accounts:
        accounts = axis_scores.five_axis_overlay(accounts)
    spec = (accounts.get('accounts') or {}).get(bundle.get('account'))
    if not isinstance(spec, dict):
        # An account the configuration does not describe still has its readings: every window binds.
        spec = {}
        accounts = {**accounts, 'accounts': {**(accounts.get('accounts') or {}), bundle['account']: spec}}
    if spec.get('unmetered'):
        return []       # a purpose-limited channel never blocks a launch
    if spec.get('five_axis_free_pool'):
        # A free allowance's only reading is its local spend, a lower bound (axis_scores.free_allowance_period): only its
        # confirmed exhaustion stops a launch, until it is due.
        latched = latched_windows(accounts, bundle['account']) if latched is None else set(latched)
        return [(wid, latched_wait_text(accounts, bundle['account'], wid) or f'{wid} has 0 left until it is re-probed')
                for wid in bundle_pools(bundle, accounts) if wid in latched]
    policy = accounts.get('scoring') or {}
    row = ((usage or {}).get('accounts') or {}).get(bundle['account']) or {}
    stale = row.get('status') == 'stale' or row.get('stale') is True
    latched = latched_windows(accounts, bundle['account']) if latched is None else set(latched)
    model = account_fleet.canonical(bundle.get('model') or '', accounts)
    found = []
    for wid, wspec, observed in applicable_windows(spec, model, row, bundle, accounts):
        observed = cycle_observation(wspec, observed, row, now)
        reset_label = (observed or {}).get('resets_at') or f"in {((observed or {}).get('seconds_until_reset') or 0) / 3600:.1f}h"
        reason = (wid in latched and latched_wait_text(accounts, bundle['account'], wid)) or window_refusal(
            wid, wspec, observed, row, now, latched) or (
            f"{wid}: used up until its reset at {reset_label}; start after its reset"
            if used_up_in_cycle(observed, now) else None)
        state = None if reason else window_state(wid, wspec, observed, spec, now, policy.get('short_window_hours', 24),
                                                 account_stale=stale)
        if state is not None and state['known']:
            state, _, _ = reserve_inflight(bundle['account'], spec, wid, wspec, state, records, accounts, policy, now)
        if state is None:
            pass
        elif state['known'] and isinstance(wspec.get('launch_min_remaining_pct'), (int, float)) and state['remaining'] <= float(wspec['launch_min_remaining_pct']):
            floor = float(wspec['launch_min_remaining_pct'])
            reason = f"{wid}: confirmed exhausted or below launch floor ({state['remaining']:.0f}% left, floor {floor:g}%)"
        elif not state['short']:
            # Priced work needs remaining - need >= safety margin; no task size passes at or under the margin. A window
            # of unknown capacity cannot price the need at all, so it is held at the margin too (as candidate_eval does).
            margin = policy.get('safety_margin_pct', 5)
            if state['known'] and state['remaining'] <= margin and (
                    quota_economy.capacity_usd(spec, wid) is None
                    or quota_economy.unit_rate(spec, wid, model, account_fleet.row_spec(accounts, model))):
                if state['reset_hours'] == float('inf'):
                    reason = (f"{wid} has {state['remaining']:.1f}% left, within the {margin:g}% safety margin "
                              "(this balance does not reset)")
                else:
                    reason = (f"{wid} has {state['remaining']:.1f}% left, within the {margin:g}% safety margin, "
                              f"for {state['reset_hours']:.1f}h")
        elif not state['known']:
            reset = parse_time((observed or {}).get('resets_at'))
            seconds = (observed or {}).get('seconds_until_reset')
            if ((reset is not None and reset > now)
                    or (reset is None and ((isinstance(seconds, (int, float)) and seconds > 0)
                                           or (observed or {}).get('reset_time_passed') is False))) and (observed or {}).get('reset_time_passed') is not True:
                wait = (reset - now).total_seconds() / 3600 if reset is not None else (seconds / 3600 if isinstance(seconds, (int, float)) else 0.0)
                reason = f'{wid} has an active reset in {wait:.1f}h with unknown remaining percentage'
        elif state['remaining'] <= (floor := short_window_limits(spec, wid, policy, now)[1]):
            reason = (f"{wid} is empty ({state['remaining']:.0f}% left, floor {floor:g}%) "
                      f"for {state['reset_hours']:.1f}h; start after its reset")
        if reason:
            found.append((wid, reason))
    return found


def metered_usd_uncapped(spec):
    """A dollar account with no budget has no window and is not excluded for exhaustion.

    It stays a metered last resort. A missing reading is the existing data-gap path;
    this predicate is only the account whose configuration states no budget.
    """
    if not isinstance(spec, dict) or spec.get('backend_kind') != 'metered_usd':
        return False
    cap = spec.get('capacity_usd')
    balance = cap.get('balance') if isinstance(cap, dict) else None
    return not isinstance(balance, (int, float)) or isinstance(balance, bool) or balance <= 0


def window_is_short(wspec, policy):
    """True for a 5-hour-class window (a reset period shorter than `short_window_hours`): it decides whether work can start
    now and never enters the five-axis economy."""
    hours = (wspec or {}).get('hours')
    return hours is not None and hours != float('inf') and hours < policy['short_window_hours']


def pool_row(base, members):
    """A pool's window state from its members' own states (each `window_state` of one account's window; owner
    2026-11-24: each member is priced and gated on its own window), by these rules: a member whose state is not known
    is out (no capacity, no share), as one account's unknown window is; the pool's full capacity and what is left are
    the known members' sums (in percent of one account); its reset is the next partial refill, the soonest reset of a
    known member that has spent some (else the soonest); it is stale when a known member is, unstarted when every known
    member is. Members in step give the merged window's figures."""
    known = [m for m in members if m.get('known')]
    row = {**base, 'members': members, 'known': bool(known)}
    if not known:
        return row
    spent = [m for m in known if m['remaining'] < m.get('full', 100.0)]
    first = min(spent or known, key=lambda m: m['reset_hours'])
    expiry = first.get('expiry_hours')
    row.update({'full': sum(m.get('full', 100.0) for m in known), 'remaining': sum(m['remaining'] for m in known),
                'target_remaining': sum(m['target_remaining'] for m in known), 'reset_hours': first['reset_hours'],
                'end_hours': first['reset_hours'] if expiry is None else min(first['reset_hours'], expiry),
                'expiry_hours': expiry, 'reset_at': first.get('reset_at'),
                'stale': bool(base.get('stale')) or any(m.get('stale') for m in known)})
    row.pop('unstarted', None)
    if all(m.get('unstarted') for m in known):
        row['unstarted'] = True
    for k in ('natural_reset_hours', 'expected_reset_at'):
        row.pop(k, None)
        if k in first:
            row[k] = first[k]
    return row


def shares(state, amount=None):
    """[(member, share of the work)]: the one split of a pool's work over its members, read by the price, the need, the
    substitute slack, the gates and the in-flight reservations alike. The gateway's choice among members is taken as
    balanced (owner 2026-11-24, an assumption and not a measurement), by room (review round 2): equal shares among the
    known members, none given more than its room (what it has left, in percent of one account); what a member cannot
    hold goes to the others in equal parts. Work beyond every member's room is split evenly over all known members, as
    one window prices the work past its remaining. `amount` is the work in percent of one account; None (or 0) is the
    limit of a small work: equal shares among the members with quota left, or among all known ones when none has. A
    member whose state is not known takes no share. Members in step get equal shares."""
    known = [m for m in state['members'] if m.get('known')]
    if not known:
        return []
    if amount is None or amount <= 1e-12:
        serving = [m for m in known if m['remaining'] > 1e-9] or known
        return [(m, 1.0 / len(serving)) for m in serving]
    taken = [0.0] * len(known)
    left = float(amount)
    while left > 1e-12:
        open_ = [i for i, m in enumerate(known) if m['remaining'] - taken[i] > 1e-12]
        if not open_:
            break
        part = left / len(open_)
        for i in open_:
            take = min(part, known[i]['remaining'] - taken[i])
            taken[i] += take
            left -= take
    if left > 1e-12:
        taken = [t + left / len(known) for t in taken]
    return [(m, t / amount) for m, t in zip(known, taken) if t > 0]


def hours_before(state, horizon, amount=None):
    """Hours of the work that fall before a window's reset, `horizon(st)` for one account's window. For a pool each
    member's share of the work (`shares`, of the work `amount`) runs to its own reset, so the pool's hours before reset
    are the members' horizons weighted by their shares (the merged window's when the members are in step)."""
    if not state.get('members'):
        return horizon(state)
    split = shares(state, amount)
    return sum(share * horizon(m) for m, share in split) if split else horizon(state)


def window_state(wid, spec, observed, account_spec, now, short_hours, account_stale=False, expected_reset_at=None):
    """Remaining capacity, time to the end of the allowance and how sure we are."""
    if isinstance((observed or {}).get('members'), list):
        # A pool: each member on its own window, then pooled (pool_row).
        members = [{**window_state(wid, spec, m, account_spec, now, short_hours, account_stale, expected_reset_at),
                    'label': m.get('label'), 'position': m.get('position')} for m in observed['members']]
        hours = spec.get('hours')
        return pool_row({'id': wid, 'hours': hours, 'stale': bool(account_stale),
                         'short': hours is not None and hours != float('inf') and hours < short_hours}, members)
    hours = spec.get('hours')
    starter = hours is None or hours == float('inf')
    expires = parse_time(account_spec.get('expires_at'))
    row = {'id': wid, 'hours': hours, 'short': (hours is not None and hours != float('inf') and hours < short_hours),
           'known': False, 'stale': bool(account_stale)}
    if starter:
        cap = quota_economy.capacity_usd(account_spec, wid)
        if cap is None or cap <= 0:
            return row
        full = 100.0
        row['full'] = full
        remaining = (observed or {}).get('remaining_pct_before_reset')
        if remaining is None:
            remaining = (observed or {}).get('observed_remaining_pct')
        if remaining is None:
            used_usd = (observed or {}).get('used_usd')
            if used_usd is None:
                used_usd = account_spec.get('local_spend_usd', 0.0)
            remaining = max(0.0, full - 100.0 * float(used_usd or 0.0) / cap)
        expiry_hours = (expires - now).total_seconds() / 3600 if expires is not None and expires > now else None
        end_hours = expiry_hours if expiry_hours is not None else float('inf')
        row.update({'known': True, 'remaining': remaining, 'reset_hours': float('inf'), 'end_hours': end_hours,
                    'expiry_hours': expiry_hours, 'target_remaining': 0.0, 'reset_at': None})
        return row
    if observed is None or hours is None:
        return row
    # Percentages are of one account's allowance; a pool is its members' windows, each one account's (pool_row).
    row['full'] = full = 100.0
    remaining = observed.get('remaining_pct_before_reset')
    if remaining is None:
        remaining = observed.get('observed_remaining_pct')
        row['stale'] = row['stale'] or remaining is not None
    seconds = observed.get('seconds_until_reset')
    if seconds is None and (natural_reset := parse_time(observed.get('resets_at'))) is not None:
        seconds = (natural_reset - now).total_seconds()
    if row['short'] and remaining == 100 and seconds is None and observed.get('resets_at') is None:
        # An unstarted rolling short window (used_pct == 0, resets_at == null) has full 5h capacity ready.
        seconds = hours * 3600
        row['stale'] = bool(account_stale)
        row['unstarted'] = True
    elif (row['short'] or account_fleet.account_family(account_spec) == 'antigravity' or spec.get('starts_on_first_use')) and remaining == 100:
        row['unstarted'] = True
    expected_hours = ((expected_reset_at - now).total_seconds() / 3600
                      if expected_reset_at is not None and expected_reset_at > now else None)
    if expected_hours is not None:
        row.update(natural_reset_hours=seconds / 3600 if seconds is not None else hours,
                   expected_reset_at=expected_reset_at.isoformat())
        seconds = min(seconds, expected_hours * 3600) if seconds is not None else expected_hours * 3600
    if remaining is None or seconds is None:
        return row
    end_hours = seconds / 3600
    expiry_hours = (expires - now).total_seconds() / 3600 if expires is not None and expires > now else None
    if expiry_hours is not None:
        end_hours = min(end_hours, expiry_hours)
    row.update({'known': True, 'remaining': remaining, 'reset_hours': seconds / 3600, 'end_hours': end_hours,
                'expiry_hours': expiry_hours, 'target_remaining': full * min(1.0, end_hours / hours),
                'reset_at': observed.get('resets_at')})
    return row


def after_reset(state, hours, *, start_hours=None, holds=None):
    """The same allowance once it renews: in full, or what the state says comes back (`renews_to`). `start_hours`
    anchors the candidate's start when another window defers past this window's reset. None when no reset was observed,
    the reading is stale or predicted, or the plan ends first.

    A pool renews member by member: at each member's own reset that member comes back as one account's window does and
    the others stay as they are (their start deferred to then). The answer is the next partial refill that gives some
    quota back and, with `holds` (the caller's test of a renewed state: does it hold the work), the first that holds it;
    when none does, the last. Members in step renew together, as the merged window did."""
    if state.get('members'):
        return _pool_after_reset(state, hours, start_hours, holds)
    wait = state['reset_hours']
    if state.get('unstarted'):
        return None  # An unstarted rolling window has no observed reset timestamp.
    if state.get('expected_reset_at') and wait < state.get('natural_reset_hours', wait):
        return None  # A prediction must never authorize work against an imagined refill.
    if wait <= 0 or wait == float('inf') or state.get('stale'):
        return None
    at = wait if start_hours is None else max(wait, float(start_hours))
    expiry = state['expiry_hours']
    left = None if expiry is None else expiry - at
    if left is not None and left <= 0:
        return None
    end = state['hours'] if left is None else min(state['hours'], left)
    full = state.get('full', 100.0)
    # A rolling limit gives back only what has left it by then (`renews_to`, cadence_window_state).
    return dict(state, remaining=state.get('renews_to', full), reset_hours=at + state['hours'], end_hours=end,
                target_remaining=full * min(1.0, end / state['hours']), deferred_hours=at,
                expiry_hours=left)


def _pool_after_reset(state, hours, start_hours, holds):
    renewals = {m['position']: after_reset(m, hours, start_hours=start_hours) for m in state['members'] if m.get('known')}
    base = {k: state[k] for k in ('id', 'hours', 'short') if k in state}
    last = None
    for at in sorted({r['deferred_hours'] for r in renewals.values() if r is not None}):
        members = []
        for m in state['members']:
            r = renewals.get(m['position']) if m.get('known') else None
            if r is not None and r['deferred_hours'] <= at:
                members.append({**r, 'label': m.get('label'), 'position': m['position']})
            elif m.get('known'):
                members.append({**m, 'start_offset_hours': at})
            else:
                members.append(m)
        last = dict(pool_row({**base, 'stale': False}, members), deferred_hours=at)
        if last['remaining'] > state['remaining'] + 1e-9 and (holds is None or holds(last)):
            return last
    return last


def cadence_limits(spec):
    """(window_seconds, limit) of an account-declared dispatch cadence (e.g. ChatGPT Web 5 per 12h), or None when the
    account declares none or the declaration is not two finite positive numbers."""
    cad = (spec or {}).get('cadence')
    if not isinstance(cad, dict):
        return None
    win_sec, cap = cad.get('window_seconds'), cad.get('limit')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (win_sec, cap)):
        return None
    return float(win_sec), float(cap)


def cadence_horizon(accounts):
    """Seconds back that any declared cadence window reaches, or None when no account declares one."""
    windows = [found[0] for spec in ((accounts or {}).get('accounts') or {}).values() if (found := cadence_limits(spec))]
    return max(windows, default=None)


def cadence_dispatches(account, records, now, win_sec):
    """When each dispatch on `account` inside the cadence window was launched, each dispatch once, oldest first.

    A dispatch counts from its launch until the window has passed over that moment, whoever chose the route and whether
    or not it has ended. The registry's rows (`stage: launch`, dispatch_registry.launch_records) say which dispatches
    were launched and when; `at` is None for one that never was. A `record` row (a route the user named) stands for its
    launch only when the registry holds no dispatch of its decision: a launch outside roll leaves no other row. A plan
    row, a live reservation row (`stage: dispatch`) or anything else is not a dispatch of its own."""
    def inside(value):
        t = parse_time(value)
        return None if t is None or (now - t).total_seconds() >= win_sec else min(t, now)

    rows = [r for r in records or () if isinstance(r, dict)]
    registered = {r.get('decision_id') for r in rows if r.get('stage') == 'launch' and r.get('decision_id')}
    found = {}
    for i, r in enumerate(rows):
        if (r.get('bundle') or {}).get('account') != account:
            continue
        if r.get('stage') == 'launch':
            key = ('launch', r.get('dispatch_id') or i)
        elif (r.get('stage') == 'record' and r.get('choice') not in (None, 'none')
              and r.get('decision_id') not in registered):
            key = ('record', r.get('decision_id') or i)
        else:
            continue
        t = inside(r.get('at'))
        if t is not None:
            found[key] = t if key not in found else min(found[key], t)
    return sorted(found.values())


def cadence_window_state(account, spec, records, now):
    """(wspec, state, limit_count) for an account-declared dispatch cadence window, or None when the account declares no
    `cadence`. A cadence is a rolling limit: the seat a dispatch takes comes back one window after its launch, whatever
    the others did. While a seat is free nothing of the allowance expires (`reset_hours` = `end_hours` = the period,
    `target_remaining` 100). With no seat free, `reset_hours` is the wait until enough dispatches have left the window
    for one seat, and `renews_to` what is free then: the limit less the dispatches still inside (after_reset)."""
    limits = cadence_limits(spec)
    if limits is None:
        return None
    win_sec, cap = limits
    period_h = win_sec / 3600.0
    events = cadence_dispatches(account, records, now, win_sec)
    used = len(events)
    state = {
        'id': 'cadence', 'hours': period_h, 'short': False, 'known': True, 'stale': False,
        'full': 100.0, 'remaining': clamp(100.0 * (cap - used) / cap, 0.0, 100.0),
        'reset_hours': period_h, 'end_hours': period_h, 'expiry_hours': None, 'target_remaining': 100.0,
        'reset_at': None, 'unstarted': not events,
    }
    if cap - used < 1:
        leaves = [max(1e-6, win_sec - (now - t).total_seconds()) for t in events]     # oldest first
        wait = leaves[math.ceil(used - cap + 1) - 1]      # the last of the dispatches that must leave for one seat
        inside = sum(1 for left in leaves if left > wait)
        state.update(reset_hours=wait / 3600.0, end_hours=wait / 3600.0,
                     renews_to=clamp(100.0 * (cap - inside) / cap, 0.0, 100.0))
    return {'hours': period_h}, state, cap
