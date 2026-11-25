"""Task-bound decisions and durable dispatch reservations (SQLite BEGIN IMMEDIATE).

Plan records are never reservations. Only prepared/launching/launched/started rows
with a live lease hold budget. No daemon or agent is started by this module.
"""
from contextlib import contextmanager
from copy import deepcopy
from worker_context import own_session_ids, require_parent_id
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import NamedTuple
import uuid
from host_placement import canonical_dir
import provider_inventory
import public_data


def _overlaid(accounts=None):
    import account_fleet
    import axis_scores
    fleet = accounts if accounts is not None else account_fleet.load_accounts()
    return fleet if '_five_axis' in fleet else axis_scores.five_axis_overlay(fleet)


def bundle_windows(bundle, accounts=None):
    """Quota windows a dispatched bundle draws, as routing sees them (the five-axis overlay gives a free pool its
    allowance window); None (whole account) when they cannot be resolved."""
    try:
        import quota_pools
        return quota_pools.bundle_pools(bundle, _overlaid(accounts)) or None
    except Exception:
        return None


def quota_latch(bundle, accounts=None):
    """(latch?, windows) for a used-up quota the bundle's provider reported: the windows the bundle draws; the whole
    account (None) when its account declares no quota window at all, so no narrower window exists to latch, or when
    the accounts cannot be read; nothing (False) when the account declares windows and the bundle draws none of them
    (a model launched under another prefix of the account, such as OpenCode Go beside the free allowance: its quota
    is none of those windows, and the classifier's hold of its own pool is the smallest true scope)."""
    try:
        import quota_pools
        fleet = _overlaid(accounts)
        drawn = quota_pools.bundle_pools(bundle, fleet)
        if drawn:
            return True, list(drawn)
        spec = (fleet.get('accounts') or {}).get((bundle or {}).get('account')) or {}
        return (False, None) if spec.get('windows') else (True, None)
    except Exception:
        return True, None

ACTIVE = ('prepared', 'launching', 'launched', 'started')
TERMINAL = ('completed', 'failed', 'create_failed', 'canceled', 'expired')
DECISION_TTL = 900
PREPARE_LEASE = 300
RUN_LEASE = 600
# A bridge sends a create only within this window after the claim, so a request the daemon has not admitted by the
# (later) create deadline can never arrive. The deadline leaves room for the daemon's own provider timeouts (120 s
# refresh, 60 s run start, then the prompt); Antigravity's cold start needs longer. Both stay inside RUN_LEASE.
CREATE_SEND_WINDOW = 60
CREATE_DEADLINE = 240
CREATE_DEADLINES = {'antigravity-resilient': 480}
MAX_LIFETIME = 86400
# A parent continuation is this event kind. The reason text is an audit note and is not read back.
HOST_FOLLOWUP_KIND = 'host_followup'
# A worker whose last dispatch is older than this is not matched to a new brief: its prompt cache is gone, and a turn sent
# to it by message since then is not on its dispatch. A worker the caller names (`--resume`) is resumed whatever its age.
REUSE_MAX_IDLE_HOURS = 24
LABELS = ('router.decision_id', 'router.dispatch_id', 'router.parent_id',
          'router.task_id', 'router.account', 'router.bundle_id', 'router.request_hash')
BOUND = ('provider', 'settings')


class Refused(ValueError):
    """A rule of the dispatch registry or its flow refused the request on purpose; its text is the rule's own. A plain
    ValueError is not one, since Python raises those too (an unpacking, a conversion): a reader that tells a refusal from a
    defect of AgentCompass (dispatch_fallback.Chain.cause_of) reads this type."""


class NotBound(Refused):
    """Paseo answered, and this worker does not carry this dispatch's labels."""


class CapacityChanged(Refused):
    """The decision's bundle can no longer start (an exclusion, a launch floor, a disabled account or model, a live
    model list that dropped it): the attempt failed before anything was sent, so the brief may go to the next candidate."""


class FullCapacityRefused(CapacityChanged):
    """A recorded constraint the user did not name (a project rule) whose work the router estimates beyond a quota
    window's full capacity plus its margin (candidate_eval.renewal_capacity_refusal): no renewal holds it, so there is
    no time to wait for. Its text is the rule's numbers per window (`shortfalls`, the candidate's `full_capacity_short`)
    and its `next_action` the two ways out."""

    error = 'The route was refused by the full-capacity rule'

    def __init__(self, account, model, shortfalls):
        parts = '; '.join(f"{wid}: the router estimates the work at about {s['need_pct']:.0f}% of this window, whose "
                          f"full capacity is {s['full']:.0f}% and keeps a {s['margin_pct']:g}% margin"
                          for wid, s in shortfalls.items())
        super().__init__(f'{account}/{model} cannot hold the work: {parts} (an estimate of the router, not a reading)')
        self.shortfalls = shortfalls

    def next_action(self):
        return ('Nothing was launched. Roll again without the constraint so the router picks a route whose quota holds '
                "the work; or, if the user names this route (roll --provider/--account/--model --basis \"<the user's "
                'words>"), it launches past the estimate with a warning.')


class WorkspaceBusy(Refused):
    """The workspace already has a writer; `holders` names each one."""

    def __init__(self, workspace, holders):
        super().__init__(f'Workspace {workspace} already has a writer')
        self.workspace, self.holders = workspace, holders


class ResumedWorkerBusy(Refused):
    """The explicitly resumed worker already has an active dispatch."""

    def __init__(self, agent_id):
        super().__init__(f'Worker {agent_id} already has an active dispatch')
        self.agent_id = agent_id


class LaunchFloorRefused(Refused):
    """One or more applicable quota windows are below their existing launch protection floor."""

    def __init__(self, floors):
        super().__init__(floors[0][1])
        self.floors = list(floors)


class RouteFactRefused(Refused):
    """A named or forced route cannot start because of a fact of the account, model or host (quota window, launch
    protection, cooldown, latch, missing credentials / sign-in, `cwd_regex`, or unreadable live model list), rather than
    an invalid flag or input. Carries `start_after_hours`, `retry_at` and `waiting_on` when every refusing quota window
    has an observed reset or, latched, a known end (`allocator.refusal_schedule`); `waiting_end` names a latch's end
    (`quota_pools.latched_end_kind`), which a re-probe time never calls a reset. `recorded`: refused at the re-check
    before launch (`prepare`), after the route's record was written. `windows`: the quota windows a launch fact refuses
    on (`launch_fact_refusal`), whose time the route's own plan row gives when there is one (`timed_by`)."""

    def __init__(self, cause, *, user_named, start_after_hours=None, retry_at=None, waiting_on=None, waiting_end=None,
                 recorded=False, windows=()):
        super().__init__(cause)
        self.windows = tuple(windows)
        self.recorded = bool(recorded)
        self.cause = str(cause)
        self.user_named = bool(user_named)
        self.start_after_hours = start_after_hours
        self.retry_at = retry_at
        self.waiting_on = waiting_on
        self.waiting_end = waiting_end

    def next_action(self):
        who = 'the route the user named' if self.user_named else 'the forced route'
        if self.start_after_hours is not None and self.retry_at:
            when_reset = (f', {self.waiting_on}: {self.waiting_end}' if self.waiting_on and self.waiting_end else
                          f', when {self.waiting_on} resets' if self.waiting_on else '')
            return (f'{self.nothing()}: {who} cannot start ({self.cause}); it can start in '
                    f'{self.start_after_hours}h (at {self.retry_at}{when_reset}). Tell the user why and when it can '
                    'start; the user may wait until then, name another route, or let the router choose (roll without '
                    '--provider, --account, --model, --effort, or --basis).')
        return (f'{self.nothing()}: {who} cannot start ({self.cause}). Tell the user why; the user '
                'may name another route, or let the router choose (roll without --provider, --account, --model, '
                '--effort, or --basis).')

    def nothing(self):
        return 'Nothing was launched' if self.recorded else 'Nothing was recorded or launched'

    def timed_by(self, candidate, accounts, now):
        """This launch-fact refusal with its time taken from the route's own plan row (`candidate`): the row's binding
        wait over everything that holds it (`plan_row_holds`), the time an automatic plan of the same reading defers to,
        when the row waits on every refusing window and nothing it excludes on has no time; else no time. A binding
        window the fact does not name is added to the cause in the row's words. The one source of a refusal's time
        wherever a plan row is read (the record of a classified task, `prepare`); `allocator.refusal_schedule` gives it
        only where there is none (a record with no task)."""
        import allocator
        import quota_pools
        if not self.windows or candidate is None:
            return self
        exclusions, waits = plan_row_holds(candidate, True)
        hours_by = {w: candidate['waits'][w] for w in waits}
        account = candidate['bundle']['account']
        self.start_after_hours = self.retry_at = self.waiting_on = self.waiting_end = None
        if exclusions or not set(self.windows) <= set(hours_by):
            return self
        self.waiting_on = max(hours_by, key=hours_by.get)
        hours = hours_by[self.waiting_on]
        self.start_after_hours = allocator.advised_hours(hours)
        self.retry_at = (now + timedelta(hours=hours)).isoformat()
        if self.waiting_on in quota_pools.latched_windows(accounts, account):
            self.waiting_end = quota_pools.latched_end_kind(accounts, account)
        if self.waiting_on not in self.windows:
            self.cause = f"{self.cause}; {(candidate.get('wait_reasons') or {}).get(self.waiting_on, self.waiting_on)}"
            self.args = (self.cause,)
        return self

    def answer(self, base=None, *, for_roll=False):
        out = dict(base or {}, status='blocked', caller_action='tell_user', cause=self.cause,
                   next_action=self.next_action())
        if for_roll:
            out.update(launched=False, dispatch_mode='none', error=self.cause, next=out['next_action'])
        if self.start_after_hours is not None:
            out['start_after_hours'] = self.start_after_hours
        if self.retry_at:
            out['retry_at'] = self.retry_at
        return out


def launch_fact_refusal(bundle, accounts, usage, now, *, user_named, records, recorded=False):
    """The fact refusal of a recorded route that a window it draws stops whatever the task's size (an account
    protection of `check_launch_quota`: its launch floors, an unavailable account; `quota_pools.launch_blockers`: a
    latch, a window used up in this cycle), with `allocator.refusal_schedule`'s time when every such window has an
    observed reset or a known end; None when nothing stops it. `records` reads the in-flight reservations, only when a
    reading needs them. One source for the record (route_judge._check_record_launch) and, for a route the user named,
    the re-check before launch (`prepare`); where either reads the route's plan row, the time is the row's
    (`RouteFactRefused.timed_by`)."""
    import allocator
    import quota_pools
    account = bundle['account']
    floors = []
    try:
        check_launch_quota(bundle, accounts, usage, now)
    except LaunchFloorRefused as exc:
        floors = list(exc.floors)
    except Refused as exc:
        return RouteFactRefused(str(exc), user_named=user_named, recorded=recorded)
    acc_usage = (usage.get('accounts') or {}).get(account) or {}
    blockers = []
    read = None
    if (floors or acc_usage.get('status') == 'fresh' or acc_usage.get('rate_limited_until')
            or quota_pools.latched_windows(accounts, account)):
        read = records()
        blockers = quota_pools.launch_blockers(bundle, accounts, usage, now, records=read)
    if not floors and not blockers:
        return None
    spec_windows = ((accounts.get('accounts') or {}).get(account) or {}).get('windows') or {}
    floor_by_wid = dict(floors)
    blocker_by_wid = {wid: f"Account {account!r} window {wid!r} is exhausted or below safety floor ({reason})"
                      for wid, reason in blockers}
    ordered_wids = list(dict.fromkeys([*(wid for wid, _ in floors), *(wid for wid, _ in blockers)]))
    reasons = {
        wid: (floor_by_wid[wid]
              if wid in floor_by_wid and ((spec_windows.get(wid) or {}).get('launch_min_remaining_pct')
                                          or wid not in blocker_by_wid)
              else blocker_by_wid[wid])
        for wid in ordered_wids
    }
    start_after_hours, retry_at, waiting_on = allocator.refusal_schedule(
        bundle, accounts, usage, now, blocked_wids=ordered_wids, records=read,
    )
    if waiting_on in reasons:
        ordered_wids = [waiting_on, *(wid for wid in ordered_wids if wid != waiting_on)]
    latched = waiting_on is not None and waiting_on in quota_pools.latched_windows(accounts, account)
    return RouteFactRefused('; '.join(reasons[wid] for wid in ordered_wids), user_named=user_named,
                            start_after_hours=start_after_hours, retry_at=retry_at, waiting_on=waiting_on,
                            waiting_end=quota_pools.latched_end_kind(accounts, account) if latched else None,
                            recorded=recorded, windows=ordered_wids)


def plan_row_holds(candidate, named):
    """What holds a decision's bundle in its own plan row at launch: `(exclusions, waits)`, each wait a window with a
    known time. A recorded route (`named`: a route the user named, an explicit resume) is not the router's choice, so
    what only automatic routing is held to does not stop it: an account out of automatic routing, a model unchecked (an
    explicit resume continues a worker whose model was unchecked after its launch), the account's dispatch cadence, the
    ability fit and the constraint filters, and a provider hold (a named model on a held account or model is the probe
    that can lift it, provider_retry)."""
    import candidate_eval
    import provider_retry
    exclusions = list(candidate['excluded'])
    waits = [w for w, hours in (candidate.get('waits') or {}).items() if hours]
    if named:
        exclusions = [e for e in exclusions if e not in candidate_eval.AUTOMATIC_ROUTING_EXCLUSIONS
                      and not e.startswith(('ability ', 'explicit model constraint', 'explicit effort constraint',
                                            provider_retry.EXCLUSION_PREFIX))]
        waits = [w for w in waits if w not in candidate_eval.AUTOMATIC_ROUTING_WAITS]
        if waits:
            # A named --fast on a row that must wait is held by the wait, which carries its time; that fast mode is
            # offered only to a row that can start now is no second refusal beside it.
            exclusions = [e for e in exclusions if e != candidate_eval.FAST_WAIT_REFUSAL]
    return exclusions, waits


def hold_cause(candidate, exclusions, waits):
    """The words of what holds a plan row: its exclusions, then each wait as the row states it (the window and its
    state, the binding one first; candidate_eval `wait_reasons`)."""
    reasons, hours = candidate.get('wait_reasons') or {}, candidate.get('waits') or {}
    return '; '.join([*exclusions, *(reasons[w] for w in sorted(waits, key=lambda w: -hours[w]))])


def named_route_refusal(candidate, exclusions, waits, accounts, now, *, user_named, recorded=False):
    """The fact refusal of a route the user named whose own plan row cannot start now (`plan_row_holds`): the row's
    words for each window and its state and, when only waits hold it, the row's own time (its binding wait, the time an
    automatic plan of the same reading defers to). An exclusion has no time, so a refusal with one carries none. One
    source for the check at record (route_judge.record_constraint) and the re-check before launch (`prepare`)."""
    import allocator
    import quota_pools
    cause = hold_cause(candidate, exclusions, waits)
    if exclusions:
        return RouteFactRefused(cause, user_named=user_named, recorded=recorded)
    waiting_on = max(waits, key=lambda w: candidate['waits'][w])
    hours = candidate['waits'][waiting_on]
    account = candidate['bundle']['account']
    latched = waiting_on in quota_pools.latched_windows(accounts, account)
    return RouteFactRefused(cause, user_named=user_named, start_after_hours=allocator.advised_hours(hours),
                            retry_at=(now + timedelta(hours=hours)).isoformat(), waiting_on=waiting_on,
                            waiting_end=quota_pools.latched_end_kind(accounts, account) if latched else None,
                            recorded=recorded)


def plan_row(alloc_task, bundle, usage, now, accounts, records, route, named):
    """The decision's bundle as `allocator.plan` ranks it for `alloc_task` under `route` (task_intake.ROUTE_FIELDS),
    or None when the ranking has no row of it."""
    import allocator
    import candidate_ranking
    plan = allocator.plan(alloc_task, usage, now, records=records, accounts=accounts,
                          named_accounts={bundle['account']} if named else (), route=route)
    # A fast route with no fast variant in the ranking is answered by its plain row: its exclusions (the fast-mode
    # refusal among them) say why, not "no longer matches".
    return (candidate_ranking.candidate_for_bundle(plan['ranking'], bundle['bundle_id'], bundle.get('fast'))
            or (bundle.get('fast') and candidate_ranking.candidate_for_bundle(plan['ranking'], bundle['bundle_id']))
            or None)


def utcnow():
    return datetime.now(timezone.utc)


def stamp(now):
    return now.isoformat()


def parse(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def db_path():
    return Path(os.environ.get('ROUTER_DISPATCH_DB') or Path.home() / '.agent-compass/dispatch.sqlite3')


@contextmanager
def transaction(path=None, upkeep=False, *, timeout=10):
    """One immediate transaction. A committed change is an effect of this call (public_data.effect: a reservation, a
    decision or dispatch row, a launch record), after which a fall-back never re-runs; ``upkeep`` marks a write that only
    brings recorded state in step with what happened on its own (a lease that ran out, a worker that ended, a side effect,
    ground truth or end notice an earlier dispatch owes), which is not."""
    path = Path(path or db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    try:
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        connection.execute('CREATE TABLE IF NOT EXISTS dispatches (id TEXT PRIMARY KEY, decision_id TEXT UNIQUE NOT NULL, data TEXT NOT NULL)')
        yield connection
        changed = connection.total_changes
        connection.commit()
        if changed and not upkeep:
            public_data.effect('dispatch registry write')   # a reservation or dispatch row: never re-run
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def rows(connection, table):
    return [json.loads(r[0]) for r in connection.execute('SELECT data FROM ' + table)]


def get(connection, table, key):
    row = connection.execute('SELECT data FROM ' + table + ' WHERE id=?', (key,)).fetchone()
    if row is None:
        raise Refused('Unknown ' + {'dispatches': 'dispatch'}.get(table, table.rstrip('s')) + ' id')
    return json.loads(row[0])


def save(connection, row):
    connection.execute('INSERT INTO dispatches(id,decision_id,data) VALUES (?,?,?) '
                       'ON CONFLICT(id) DO UPDATE SET data=excluded.data',
                       (row['dispatch_id'], row['decision_id'], json.dumps(row, ensure_ascii=False)))


def event(row, state, now, reason=None, *, kind=None):
    row.update(state=state, updated_at=stamp(now))
    record = {'state': state, 'at': stamp(now)}
    if reason:
        record['reason'] = reason
    if kind:
        record['kind'] = kind
    row.setdefault('events', []).append(record)
    if state in TERMINAL:
        row.update(released_at=stamp(now), lease_until=stamp(now))


def expire(connection, now):
    # An ended worker is settled before a lease lapse, so a dead watcher does not wait out the lease.
    # A question that already went idle is reopened here when the host has started the parent's reply.
    settle_ended_workers(connection, now)
    refresh_live_rounds(connection, now)
    for row in rows(connection, 'dispatches'):
        if row['state'] in ACTIVE and parse(row['lease_until']) <= now:
            event(row, 'expired', now, 'lease_elapsed_without_fresh_host_observation')
            save(connection, row)


def resolve_bundle(bundle, accounts, explicit_override=False):
    """Catalog defaults are not the complete supported explicit-effort contract."""
    import account_fleet
    import fast_modes
    import auto_bundles
    model = account_fleet.canonical(bundle.get('model'), accounts)
    if bundle.get('fast'):
        fast_modes.validate(account_fleet.bundle_harness(bundle, bundle.get('account')), model,
                                      account_fleet.row_spec(accounts, model))
    pool = account_fleet.pin_pool(accounts, account=bundle.get('account'), provider=bundle.get('provider'))
    siblings = account_fleet.pin_matching_bundles(bundle.get('launch_model') or bundle.get('model') or '', pool, accounts)
    bid = bundle.get('bundle_id') or bundle.get('id')
    by_id = next((b for b in siblings if bid and b.get('id') == bid), None)
    carrier = next((b for b in siblings if bundle.get('pin_carrier') and b.get('id') == bundle['pin_carrier']), None)
    if carrier and bundle.get('effort') in (account_fleet.dispatch_allowed_efforts(carrier, accounts, model) or ()):
        # The plan already matched the effort the user named to the launch contract; the row exists only for this task.
        return dict(carrier, id=bid, bundle_id=bid, effort=bundle['effort'], pin_carrier=bundle['pin_carrier'],
                    **({'fast': True} if bundle.get('fast') else {}))
    if by_id and by_id.get('effort') == bundle.get('effort'):
        return dict(by_id, bundle_id=by_id['id'], **({'fast': True} if bundle.get('fast') else {}))
    if not by_id:
        exact = (next((b for b in siblings if b.get('effort') == bundle.get('effort') and not b.get('disabled')), None)
                 or next((b for b in siblings if b.get('effort') == bundle.get('effort')), None))
        if exact:
            return dict(exact, bundle_id=exact['id'], **({'fast': True} if bundle.get('fast') else {}))
    # A named level with no row of its own arrives without the row's list, so every row of the model speaks for what it offers.
    contract = {eff for row in (bundle, *siblings) for eff in account_fleet.dispatch_allowed_efforts(row, accounts, model) or ()}
    if not explicit_override:
        return None
    exact = (next((b for b in siblings if b.get('effort') == bundle.get('effort') and not b.get('disabled')), None)
             or next((b for b in siblings if b.get('effort') == bundle.get('effort')), None))
    if exact:
        return dict(exact, bundle_id=exact['id'], **({'fast': True} if bundle.get('fast') else {}))
    if not siblings:
        raise Refused(f"named_model_unavailable: {bundle.get('model')} has no row on "
                         f"{bundle.get('account')}/{bundle.get('provider')}. Tell the user; do not substitute another model.")
    if bundle.get('effort') not in contract:
        raise Refused(account_fleet.unaccepted_effort(
            account_fleet.bundle_harness(siblings[0], bundle.get('account')), bundle.get('model'), bundle.get('effort'),
            contract | {s.get('effort') for s in siblings}))
    base = next((b for b in siblings if not b.get('disabled')), siblings[0])
    identity = {k: bundle.get(k) for k in ('provider', 'account', 'effort')}
    identity['model'] = account_fleet.row_key(base.get('model') or '', accounts)
    key = 'user_override_' + digest(identity)[:24]
    mode = base.get('modeId') or auto_bundles.HARNESS_MODE.get(base.get('harness'))
    return dict(base, id=key, bundle_id=key, effort=bundle['effort'], **({'modeId': mode} if mode else {}),
                **({'launch_model': bundle['launch_model']} if bundle.get('launch_model') else {}),
                **({'fast': True} if bundle.get('fast') else {}))


def check_launch_quota(bundle, accounts, usage, now):
    """Same existing account protection for prepared MCP and direct SDK launches.

    Local disablement also applies after preparation. For quota, only fresh,
    active-window observations hard-block; unknown/stale is not exhausted.
    """
    import account_fleet
    import fast_modes
    import quota_pools
    reason = provider_inventory.unavailable_reason(accounts.get('_discovery') or {}, bundle['account'], bundle['provider'])
    if reason:
        raise Refused(reason)
    if bundle['account'] not in accounts['accounts']:
        raise Refused('Account is no longer configured in the current provider inventory')
    if bundle.get('fast'):
        model = account_fleet.canonical(bundle['model'], accounts)
        fast_modes.validate(account_fleet.bundle_harness(bundle, bundle['account']), model,
                                      account_fleet.row_spec(accounts, model))
    usage = provider_inventory.merge_usage(usage, accounts, now)
    spec = accounts['accounts'][bundle['account']]
    row = (usage.get('accounts') or {}).get(bundle['account']) or {}
    floors = []
    for wid, window, observed in quota_pools.applicable_windows(spec, bundle.get('model'), row, bundle, accounts):
        if quota_pools.below_launch_floor(window, observed, row, now):
            floor = window.get('launch_min_remaining_pct', 0)
            floors.append((
                wid,
                f"Account {bundle['account']} {wid} is below its existing launch protection ({floor}% remaining floor)"
                + ('' if quota_pools.observed_reset(observed, now) else '; no reset of it is observed'),
            ))
    if floors:
        raise LaunchFloorRefused(floors)


EARLIER_VERSION = ('decision {decision_id} was recorded by an earlier version of AgentCompass ({detail}); roll or '
                   'record it again')


def earlier_version(task):
    """Why a stored task is in a shape an earlier version recorded and this one does not read, or None. A decision is
    derived state, so it is not upgraded: a task profile this version does not validate, or fields only AgentCompass
    sets for itself (task_intake.router_fields; v0.7.0's `record --fast` wrote `force_fast` into the stored task).
    The one owner of that refusal: dispatch_registry.prepare raises EARLIER_VERSION, and dispatch_fallback ends a
    relaunch on it."""
    import task_intake
    if not isinstance(task, dict):
        return None
    if task.get('profile') is not None:
        try:
            task_intake.validate_profile(task['profile'])
        except ValueError:
            return 'its task profile is not one this version reads'
    found = task_intake.router_fields(task)
    if found:
        return f"its task carries {', '.join(found)}, which this version sets only on the route it plans"
    return None


def issue(result, task, now=None, path=None, allocation_task=None, listed_rows=None, route=None, *, from_roll=False):
    """Register a decision, without committing capacity. Legacy unbound decisions cannot dispatch. `listed_rows`
    (provider_inventory.listed_rows) are the rows of a model the user named that the shipped data has no row for and the
    account's provider lists: the decision keeps them, and `prepare` plans with them. `route`: the route fields a record
    decided (task_intake.ROUTE_FIELDS: `unclassified`, `force_fast`), kept beside the task and passed to `prepare`'s
    plan; only a user_override or explicit resume record keeps them."""
    now = now or utcnow()
    bundle = dict(result.get('selected_bundle') or {})
    if not bundle or result.get('choice') in (None, 'none'):
        return {}
    bundle.setdefault('bundle_id', result.get('choice'))
    import account_fleet
    explicit = result.get('stage') == 'record' and result.get('constraint') == 'user_override' and bool(result.get('basis'))
    listed_rows = listed_rows if explicit else None
    named = explicit or (result.get('stage') == 'record' and result.get('constraint') == 'explicit_resume')
    route = {k: v for k, v in (route or {}).items() if v} if named else {}
    # The catalog row as a plan reads it: the windows a row draws are settled by the overlay (a free pool's allowance is
    # declared there, after the auto bundles were derived), and the stored bundle is what the latch, the in-flight
    # reservation and the launch facts read.
    accounts = _overlaid(provider_inventory.with_listed_rows(account_fleet.load_accounts(), listed_rows))
    resolved = resolve_bundle(bundle, accounts, explicit)
    if resolved:
        bundle = {**bundle, **resolved}
    parent = task.get('parent_agent_id')
    task_id = task.get('task_id')
    row = {'decision_id': str(uuid.uuid4()), 'task_id': task_id, 'parent_id': parent,
           'task_hash': digest(task), 'task': task, 'allocation_task': allocation_task or task, 'bundle': bundle,
           'bound': bool(isinstance(parent, str) and parent and isinstance(task_id, str) and task_id),
           'created_at': stamp(now), 'valid_until': stamp(now + timedelta(seconds=DECISION_TTL)),
           'constraint': result.get('constraint'),
           # Why a route was forced (the user's words, the project rule): a relaunch that may not move it says so.
           **({'basis': str(result['basis'])[:400]} if result.get('constraint') and result.get('basis') else {}),
           **({'from_roll': True} if from_roll else {}),
           **({'listed_rows': listed_rows} if listed_rows else {}),
           **({'route': route} if route else {}),
           'effort_override': bool(explicit and resolved and resolved['id'].startswith('user_override_')),
           'dispatchable': not result.get('host_policy_violation')
                           and bool(resolved)
                           and (result.get('decision') or {}).get('action') != 'defer'}
    with transaction(path) as connection:
        connection.execute('INSERT INTO decisions VALUES (?,?)', (row['decision_id'], json.dumps(row, ensure_ascii=False)))
    return {k: row[k] for k in ('decision_id', 'task_id', 'parent_id', 'task_hash', 'bound', 'valid_until')}


def canonical_request(request):
    allowed = {'title', 'provider', 'initialPrompt', 'workspaceId', 'cwd', 'settings', 'labels', 'notifyOnFinish'}
    if set(request) - allowed:
        raise Refused('Unsupported create_agent fields; put dispatch metadata in labels')
    value = {k: v for k, v in request.items() if k != 'labels' and v is not None}
    value.setdefault('settings', {})
    value.setdefault('notifyOnFinish', True)
    value['labels'] = {k: v for k, v in (request.get('labels') or {}).items() if not k.startswith('router.')}
    return value


def bound_hash(request):
    """Bind only what the decision chose; title/prompt/workspace/cwd stay the parent's to edit until launch."""
    value = canonical_request(request)
    return digest({k: value[k] for k in BOUND})


# The one table of launcher families this Paseo adapter launches, with each family's permission mode.
# Accounts are admitted by their discovered family, never by name.
PASEO_LAUNCHER_MODES = {'codex': 'full-access', 'claude': 'bypassPermissions', 'pi': None, 'omp': 'full',
                        'antigravity-resilient': 'full-access', 'opencode': 'build', 'grok': None, 'acp': None}


def launch_request(bundle, launch):
    if not isinstance(launch, dict) or not isinstance(launch.get('initialPrompt'), str) or not launch['initialPrompt'].strip():
        raise Refused('launch.initialPrompt must contain the reviewed assignment')
    if not isinstance(launch.get('title'), str) or not 0 < len(launch['title']) <= 60:
        raise Refused('launch.title must contain 1..60 characters')
    if set(launch) - {'title', 'initialPrompt', 'cwd', 'labels', 'notifyOnFinish'}:
        raise Refused('Launch configuration comes from the decision; only title/prompt/cwd/labels/notify are inputs')
    if 'cwd' in launch and not (isinstance(launch['cwd'], str) and os.path.isabs(launch['cwd'])):
        raise Refused('launch.cwd must be an absolute directory')
    if any(k.startswith('router.') for k in launch.get('labels', {})):
        raise Refused('router.* labels are generated, not caller supplied')
    provider = bundle['provider']
    family = provider_inventory.launcher_family(provider)
    if family not in PASEO_LAUNCHER_MODES:
        raise Refused('This dispatch adapter supports verified Paseo providers only; use the provider-specific runner otherwise')
    model = bundle.get('launch_model') or bundle['model']
    settings = {}
    # Pi has fixed tool permissions and advertises no selectable modes. Ignore
    # the obsolete full-access value in saved/discovered Pi bundles as well.
    mode = None if family == 'pi' else bundle.get('modeId') or PASEO_LAUNCHER_MODES[family]
    if mode:
        settings['modeId'] = mode
    if bundle.get('effort') and family != 'antigravity-resilient':
        settings['thinkingOptionId'] = bundle['effort']
    if bundle.get('fast'):
        import account_fleet
        from fast_modes import validate
        accounts = account_fleet.load_accounts()
        c_model = account_fleet.canonical(bundle['model'], accounts)
        validate(family, c_model, account_fleet.row_spec(accounts, c_model))
    if family == 'codex':
        settings['features'] = {'service_tier': 'priority' if bundle.get('fast') else 'default'}
    elif family == 'claude':
        settings['features'] = {'fast_mode': False}     # no Claude fast contract (fast_modes.HARNESSES): always off
    return canonical_request(dict(launch, provider=provider + '/' + model, settings=settings))


def held_records(stored, now):
    """The live reservations among stored dispatch rows, one per worker."""
    import quota_pools
    live = [{'stage': 'dispatch', 'dispatch_id': r['dispatch_id'], 'state': r['state'],
             'at': r['created_at'], 'lease_until': r['lease_until'], 'choice': r['bundle']['bundle_id'],
             'bundle': r['bundle'], 'expected_usd': r['expected_usd'], 'usage': r['usage'],
             **({'agent_id': r['agent_id']} if r.get('agent_id') else {})}
            for r in stored
            if r.get('dispatch_mode', 'paseo') == 'paseo' and r['state'] in ACTIVE and parse(r['lease_until']) > now]
    return quota_pools.live_dispatch_holds(live)


def active_records(connection, now):
    return held_records(rows(connection, 'dispatches'), now)


def _read_only(path, timeout=5.0):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=timeout)
    return connection


def read_dispatches(path=None, timeout=0.5):
    """Every stored dispatch row, read without writing, creating the file, or waiting on a writer longer than `timeout`
    seconds (sqlite3.OperationalError then). [] when no registry exists yet."""
    target = Path(path or db_path())
    if not target.exists():
        return []
    connection = _read_only(target, timeout)
    try:
        return rows(connection, 'dispatches')
    except sqlite3.OperationalError as exc:
        if 'no such table' in str(exc):
            return []
        raise
    finally:
        connection.close()


def reservations(now=None, path=None, read_only=False, *, timeout=5.0):
    target = Path(path or db_path())
    if not target.exists():
        return []
    now = now or utcnow()
    if read_only:
        ended, pending = _reservation_maintenance(target, now, timeout=timeout)
        if not ended:
            # A dry read stays byte-identical unless a local agent record shows a worker already ended.
            if pending:
                _finish_pending_side_effects(path, now)
                _record_pending_ground_truth(path, now)
            connection = _read_only(target, timeout)
            try:
                return active_records(connection, now)
            finally:
                connection.close()
    with transaction(path, upkeep=True, timeout=timeout) as connection:
        if read_only:
            settle_ended_workers(connection, now)
            refresh_live_rounds(connection, now)
        else:
            expire(connection, now)
        active = active_records(connection, now)
    _finish_pending_side_effects(path, now)
    _record_pending_ground_truth(path, now)
    return active


def _stored_time(row, key):
    """`row[key]` read as a time, or None when the row has no such text or holds something that is not a time."""
    value = row.get(key)
    if not isinstance(value, str):
        return None
    try:
        return parse(value)
    except ValueError:
        return None


def launch_record(row, now, since=None):
    """One stored dispatch as the cadence count reads it (quota_pools.cadence_dispatches), whatever its surface or state:
    `at` is when it was launched, or None when it never was. A dispatch is launched from `launched_at` (the Paseo claim,
    a started host CLI, an external runner's launch) unless the host refused its create (`create_failed`: the prompt
    never left); a reservation still holding its lease before launch counts from its creation, as it holds the account's
    other windows. The registry is derived state: a row the count cannot place (no object, no `dispatch_id`, a `bundle`
    that is not an object, a launch time that is not a time, or neither time at all) is no launch in any window, read
    without a default time and without an exception. None when `since` is given and the row was launched, or else
    created, before it."""
    row = row if isinstance(row, dict) else {}
    launched, created = _stored_time(row, 'launched_at'), _stored_time(row, 'created_at')
    seen = launched or created
    if since is not None and seen is not None and seen < since:
        return None
    bundle = row.get('bundle') if isinstance(row.get('bundle'), dict) else None
    lease = _stored_time(row, 'lease_until')
    if not row.get('dispatch_id') or bundle is None or (row.get('launched_at') and launched is None):
        at = None
    elif launched and row.get('state') != 'create_failed':
        at = row['launched_at']
    elif row.get('state') in ACTIVE and not row.get('native_handover') and created and lease is not None and lease > now:
        at = row['created_at']
    else:
        at = None
    return {'stage': 'launch', 'dispatch_id': row.get('dispatch_id'), 'decision_id': row.get('decision_id'),
            'bundle': bundle or {}, 'at': at}


def launches(stored, accounts, now):
    """The stored dispatches a declared cadence window can still hold (launch_record each): those launched or created
    since the longest cadence window of `accounts` began, and those carrying no readable time (read as no launch, so
    their decision is still known to the count). [] when no account declares one."""
    import quota_pools
    horizon = quota_pools.cadence_horizon(accounts)
    if horizon is None:
        return []
    since = now - timedelta(seconds=horizon)
    return [record for r in stored if (record := launch_record(r, now, since)) is not None]


def launch_records(accounts, now=None, path=None):
    """launches() of the registry, read without writing or creating the file."""
    import quota_pools
    target = Path(path or db_path())
    if quota_pools.cadence_horizon(accounts) is None or not target.exists():
        return []
    connection = _read_only(target)
    try:
        stored = rows(connection, 'dispatches')
    except sqlite3.OperationalError as exc:
        if 'no such table' in str(exc):
            return []
        raise
    finally:
        connection.close()
    return launches(stored, accounts, now or utcnow())


def paseo_agent_record(agent_id, paseo_home=None):
    """Read local Paseo agent JSON for agent_id if present on disk; None when absent."""
    if not isinstance(agent_id, str) or not agent_id.strip():
        return None
    agents_dir = os.environ.get('ROUTER_PASEO_AGENTS_DIR')
    base = Path(agents_dir) if agents_dir else Path(paseo_home or os.environ.get('PASEO_HOME') or Path.home() / '.paseo') / 'agents'
    if not base.is_dir():
        return None
    for candidate in base.glob(f'*/{agent_id}.json'):
        try:
            doc = json.loads(candidate.read_text(encoding='utf-8', errors='replace'))
            if isinstance(doc, dict):
                return doc
        except (OSError, ValueError):
            continue
    return None


# Receipt reasons whose transition is recorded. Ordinary finished/error transitions stay unmarked: observers
# already assert those events have no reason.
WATCHER_TERMINAL_REASONS = frozenset({
    'superseded by a later dispatch',
    'worker unavailable',
    'worker removed before it was observed',
})


def host_running_evidence(snapshot):
    """The host snapshot shows this worker still running or waiting on a permission, not a stale idle read."""
    if not isinstance(snapshot, dict) or snapshot.get('archivedAt') or snapshot.get('providerUnavailable'):
        return False
    status = snapshot.get('status')
    if status in ('running', 'initializing'):
        return True
    attention = snapshot.get('attentionReason')
    pending = snapshot.get('pendingPermissions') or ()
    if attention == 'permission' or pending or (snapshot.get('pendingPermissionCount') or 0) > 0:
        return status in ('idle', 'initializing', None) or attention == 'permission'
    return False


def running_evidence_reason(snapshot):
    status = snapshot.get('status') or ''
    updated = snapshot.get('updatedAt') or ''
    attention = snapshot.get('attentionReason') or ''
    pending = bool(snapshot.get('pendingPermissions') or (snapshot.get('pendingPermissionCount') or 0))
    if status not in ('running', 'initializing') and (attention == 'permission' or pending):
        return f'host_still_running status={status} attention=permission updated_at={updated}'
    return f'host_still_running status={status} updated_at={updated}'


def _turn_landed(row, doc):
    last = doc.get('lastUserMessageAt') if isinstance(doc, dict) else None
    launched = row.get('launched_at')
    try:
        return isinstance(last, str) and isinstance(launched, str) and parse(last) >= parse(launched)
    except ValueError:
        return False


def _undelivered_resume(row, doc):
    status = doc.get('status') if isinstance(doc, dict) else None
    return (row.get('operation') == 'resume' and not row.get('started_at') and not _turn_landed(row, doc)
            and status != 'running' and not (isinstance(doc, dict) and doc.get('archivedAt')))


def _updated_before_launch(row, doc):
    """True when the document carries a timestamp older than this launch. A missing timestamp says nothing."""
    updated = doc.get('updatedAt') if isinstance(doc, dict) else None
    launched = row.get('launched_at')
    if not isinstance(updated, str) or not isinstance(launched, str):
        return False
    try:
        return parse(updated) < parse(launched)
    except ValueError:
        return False


def _bound_dispatch(doc):
    labels = doc.get('labels') if isinstance(doc, dict) and isinstance(doc.get('labels'), dict) else {}
    bound = labels.get('router.dispatch_id')
    return bound if isinstance(bound, str) and bound else None


def host_generation(row, doc, later_ids=()):
    """Whether this host document is evidence for this dispatch.

    `current` matches this round (or carries no dispatch label). `superseded` is a later dispatch that has taken
    a round this dispatch already bound. None is not evidence: the timestamp is older than the launch, the row is
    a resume the host has not bound yet (it still wears the previous round's label), or the other label is not a
    later dispatch."""
    if not isinstance(doc, dict) or _updated_before_launch(row, doc):
        return None
    bound = _bound_dispatch(doc)
    if bound == row.get('dispatch_id'):
        return 'current'
    if bound is None:
        if row.get('operation') == 'resume' and not _at_or_after(
                doc.get('lastUserMessageAt'), row.get('created_at') or row.get('launched_at')):
            return None
        return 'current'
    if ('bound_unlabeled_from' in row and (bound or '') == row.get('bound_unlabeled_from')
            and bound not in set(later_ids or ())
            and _at_or_after(doc.get('lastUserMessageAt'), row.get('created_at') or row.get('launched_at'))):
        return 'current'
    if row.get('operation') == 'resume' and not row.get('started_at'):
        return None
    if bound in set(later_ids or ()):
        return 'superseded'
    return None


def _open_for_disk_settle(row):
    if row.get('dispatch_mode', 'paseo') != 'paseo':
        return False
    if row['state'] in ACTIVE:
        return True
    return row['state'] == 'expired' and bool(row.get('launched_at'))


def terminal_from_agent_doc(row, doc, later_ids=()):
    """The terminal state a local Paseo agent JSON proves, or (None, None) when it says nothing.

    The document has to be this round's. Permission and a still-running status are not endings. A resume whose
    turn has not landed is not ended by a stale idle or closed document. A missing document says nothing."""
    if not isinstance(doc, dict):
        return None, None
    generation = host_generation(row, doc, later_ids)
    if generation == 'superseded':
        return 'canceled', f'paseo_agent_record: superseded by {_bound_dispatch(doc)}'
    if generation != 'current':
        return None, None
    status = doc.get('status') if isinstance(doc.get('status'), str) else ''
    attention = doc.get('attentionReason') if isinstance(doc.get('attentionReason'), str) else ''
    pending = doc.get('pendingPermissions') or ()
    if attention == 'permission' or pending or (doc.get('pendingPermissionCount') or 0) > 0:
        return None, None
    if status in ('running', 'initializing'):
        return None, None
    undelivered = _undelivered_resume(row, doc)
    reason = f'paseo_agent_record: status={status} attention={attention}'
    if doc.get('archivedAt') or status == 'closed':
        if undelivered and not doc.get('archivedAt'):
            return None, None
        return ('completed' if attention == 'finished' else 'canceled'), reason
    if undelivered:
        return None, None
    if status == 'error' or attention == 'error' or doc.get('providerUnavailable'):
        return 'failed', reason
    if status == 'idle' and (attention == 'finished' or row.get('started_at')):
        return 'completed', reason
    return None, None


def _agent_doc_for(row):
    agent_id = row.get('agent_id') or row.get('dispatch_id')
    doc = paseo_agent_record(agent_id)
    if doc is None and row.get('agent_id') and row['agent_id'] != row.get('dispatch_id'):
        doc = paseo_agent_record(row['dispatch_id'])
    return doc


def _later_dispatch_ids(connection, row):
    agent_id = row.get('agent_id')
    if not agent_id:
        return []
    ref = row.get('launched_at') or row.get('created_at')
    found = []
    for other in rows(connection, 'dispatches'):
        if other['dispatch_id'] == row['dispatch_id'] or other.get('agent_id') != agent_id:
            continue
        other_stamp = other.get('launched_at') or other.get('created_at')
        if ref and later(other_stamp, ref):
            found.append(other['dispatch_id'])
    return found


def _safe_dispatch_id(value):
    return (isinstance(value, str) and 0 < len(value) <= 128 and value[0].isalnum()
            and all(character.isalnum() or character in '-_' for character in value))


def watcher_receipt(dispatch_id):
    """The watcher's on-disk receipt, or None. A bad id never becomes a path."""
    if not _safe_dispatch_id(dispatch_id):
        return None
    root = Path(os.environ.get('ROUTER_STATE_DIR') or Path.home() / '.agent-compass')
    try:
        document = json.loads((root / 'paseo-notifications' / f'{dispatch_id}.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def state_from_receipt(row, receipt):
    """The terminal state the watcher's receipt proves for this row, or (None, None).

    The reason is the ending. The receipt's raw host status is not read: a superseded row's receipt carries the
    next round's `running`, and that must not renew the old row. A timestamp older than this launch is ignored."""
    terminal = receipt.get('terminal') if isinstance(receipt, dict) else None
    if not isinstance(terminal, dict) or not isinstance(terminal.get('reason'), str):
        return None, None
    reason = terminal['reason']
    at = terminal.get('at')
    launched = row.get('launched_at')
    if isinstance(at, str) and isinstance(launched, str):
        try:
            if parse(at) < parse(launched):
                return None, None
        except ValueError:
            return None, None
    if reason == 'finished':
        return 'completed', reason
    if reason in ('error', 'provider retry stuck'):
        return 'failed', reason
    if reason in WATCHER_TERMINAL_REASONS:
        return 'canceled', reason
    if reason == 'closed':
        return ('completed' if terminal.get('attentionReason') == 'finished' else 'canceled'), reason
    return None, None


def _queue_side_effects(row, receipt, target):
    """Remember idempotent effects still owed. State and these flags are stored apart, so a row that is already
    terminal can still clear a provider hold or record a failure the first writer skipped."""
    due = list(row.get('side_effects_due') or [])
    done = set(row.get('terminal_side_effects') or [])
    terminal = receipt.get('terminal') if isinstance(receipt, dict) else None
    launched = row.get('launched_at')

    def add(name, include):
        if include and name not in done and name not in due:
            due.append(name)

    add('provider_hold_clear', target == 'completed' and provider_replied(terminal, launched))
    add('provider_failure_record', target == 'failed' and provider_failed(terminal, launched))
    add('launch_latches', target == 'completed')
    if due != list(row.get('side_effects_due') or []):
        row['side_effects_due'] = due
        return True
    return False


def _apply_terminal(row, target, now, reason):
    if not target or row['state'] == target:
        return False
    if target == 'completed':
        row['ground_truth_pending'] = True
    event(row, target, now, reason)
    return True


def apply_ending(connection, row, now, receipt_only=False):
    """Write this row's ending. A watcher receipt outranks the agent JSON. An existing real terminal is kept.

    Returns True when the row was saved."""
    if row.get('dispatch_mode', 'paseo') != 'paseo':
        return False
    open_row = _open_for_disk_settle(row)
    already = row['state'] in TERMINAL and row['state'] != 'expired'
    if not open_row and not already:
        return False
    receipt = watcher_receipt(row['dispatch_id'])
    receipt_state, receipt_reason = state_from_receipt(row, receipt)
    changed = False
    bound = receipt.get('bound_agent') if isinstance(receipt, dict) and isinstance(receipt.get('bound_agent'), dict) else None
    if not row.get('agent_id') and isinstance(receipt, dict):
        found_id = (bound or {}).get('id') or (
            receipt.get('agent_id') if receipt.get('created_agent') or receipt.get('agent_id') == row['dispatch_id'] else None
        )
        if isinstance(found_id, str) and found_id:
            row['agent_id'] = found_id
            changed = True
    if open_row and not row.get('observed_at') and bound and isinstance(bound.get('id'), str) and bound['id']:
        row['agent_id'] = bound['id']
        row['observed_at'] = bound.get('at') or stamp(now)
        if receipt.get('started') is True and not row.get('started_at'):
            row['started_at'] = bound.get('started_at') or bound.get('at') or stamp(now)
        if not already and not receipt_state and row['state'] in ('launching', 'expired'):
            event(row, 'started' if receipt.get('started') else 'launched', now, 'watcher_observed_agent')
            row['lease_until'] = stamp(min(now + timedelta(seconds=RUN_LEASE), parse(row['hard_until'])))
        changed = True
    if already:
        target = row['state']
    elif receipt_state:
        target = receipt_state
        if target == 'failed' and isinstance(receipt, dict):
            terminal = receipt.get('terminal') or {}
            code = terminal.get('failure_code')
            if not isinstance(code, str) or not code:
                code = 'provider_retry_stuck' if receipt_reason == 'provider retry stuck' else None
            if code:
                row['failure_code'] = code
                row['failure_detail'] = {key: terminal[key] for key in ('partial_work', 'cancelled') if key in terminal}
        reason = ('watcher_terminal: ' + receipt_reason) if receipt_reason in WATCHER_TERMINAL_REASONS else None
        if target == 'completed':
            row['ground_truth_other_ids'] = _later_dispatch_ids(connection, row)
        changed = _apply_terminal(row, target, now, reason) or changed
    elif receipt_only:
        target = row['state']
    else:
        later_ids = _later_dispatch_ids(connection, row)
        target, reason = terminal_from_agent_doc(row, _agent_doc_for(row), later_ids)
        if target and target != row['state']:
            if target == 'completed':
                row['ground_truth_other_ids'] = later_ids
            changed = _apply_terminal(row, target, now, reason) or changed
        else:
            target = row['state']
    changed = _queue_side_effects(row, receipt, target) or changed
    changed = _note_end_notice(row, receipt) or changed
    if changed:
        save(connection, row)
    return changed


_END_NOTICE_STATES = ('completed', 'failed', 'canceled', 'create_failed')
# The watcher already decided not to send, or the parent's own notifyOnFinish is the message.
_SKIP_CATCHUP_STATUS = frozenset({
    'resumed', 'canceled', 'relaunched', 'parent_unavailable', 'creation_failed', 'continued',
})


# How a send whose earlier attempt Paseo answered `agent_request_outcome_unknown` ended: the notice was in the parent's
# timeline, or it was sent once under the next derived id (paseo_dispatch_bridge.mjs deliverNotice).
_UNKNOWN_OUTCOME_ENDINGS = ('found', 'resent')


def end_notice_message_id(dispatch_id):
    """The same id the watcher uses for its terminal message, so a race is one message."""
    return hashlib.sha256(f'router-finish:{dispatch_id}:terminal'.encode()).hexdigest()


def followed_up(row):
    """A parent continuation is the event kind. The reason text is not evidence."""
    return any(isinstance(item, dict) and item.get('kind') == HOST_FOLLOWUP_KIND
               for item in row.get('events') or [])


def _note_end_notice(row, receipt):
    """Record the one end notice for a real terminal. A row that is still open, or that has no watcher receipt, is left alone.

    `sent` is never sent again. A follow-up turn belongs to Paseo's notifyOnFinish; the event kind records that,
    and the reason text is not read. A receipt the watcher already delivered, or one it intentionally did not send,
    is recorded as sent, with the id the notice ended on. Anything else is due for one catch-up: so is a receipt that
    claims a later dispatch (`continued` with `continued_by`, decided by `due_end_notices`) and one the watcher gave up
    as `undelivered` (the catch-up records it, and the parent's next router answer names it)."""
    if row.get('state') not in _END_NOTICE_STATES:
        return False
    notice = row.get('end_notice')
    if isinstance(notice, dict) and notice.get('state') == 'sent':
        return False
    message_id = end_notice_message_id(row['dispatch_id'])
    if followed_up(row):
        row['end_notice'] = {'state': 'sent', 'message_id': message_id, 'source': 'notify_on_finish'}
        return True
    if not isinstance(receipt, dict):
        return False
    delivered = receipt.get('delivered')
    status = receipt.get('status')
    if status == 'delivered' or (isinstance(delivered, list) and 'terminal' in delivered):
        # After Paseo answered that a send's outcome is unknown, the watcher found the notice in the parent's
        # timeline or sent it once under a new id (paseo_dispatch_bridge.mjs deliverNotice).
        outcome = receipt.get('delivery_outcome')
        source = 'watcher_' + outcome if outcome in _UNKNOWN_OUTCOME_ENDINGS else 'watcher'
        used = receipt.get('delivery_message_id')
        row['end_notice'] = {'state': 'sent', 'message_id': used if _is_message_id(used) else message_id, 'source': source}
        return True
    claim = status == 'continued' and isinstance(receipt.get('continued_by'), str)
    if status in _SKIP_CATCHUP_STATUS and not claim:
        row['end_notice'] = {'state': 'sent', 'message_id': message_id, 'source': status}
        return True
    if isinstance(notice, dict) and notice.get('state') == 'due':
        return False
    row['end_notice'] = {'state': 'due', 'message_id': message_id}
    return True


def _is_message_id(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _at_or_after(value, ref):
    try:
        return isinstance(value, str) and isinstance(ref, str) and parse(value) >= parse(ref)
    except ValueError:
        return False


def _own_turn_landed(row):
    """Whether this dispatch's own turn reached its worker: a seen run (`started_at`), or its prompt on the worker after
    its launch, as its watcher recorded it (`started`, `user_message_at`) or as the worker's agent JSON shows it while
    it carries this dispatch's labels. `observed_at` is not this: `observe` stamps it on a resume bound before its turn
    lands too."""
    if row.get('started_at'):
        return True
    launched = row.get('launched_at')
    receipt = watcher_receipt(row['dispatch_id'])
    if isinstance(receipt, dict) and (receipt.get('started') is True or _at_or_after(receipt.get('user_message_at'), launched)):
        return True
    doc = _agent_doc_for(row)
    return _bound_dispatch(doc) == row['dispatch_id'] and _turn_landed(row, doc)


def _later_turn(all_rows, row):
    """What later turns this row's parent gave this row's worker say about its end notice: (`landed`, id) when one
    landed, (`pending`, id) when one is still on its way, else (None, None): none, or each ended without landing.

    A later turn is a `resume` row on the same worker from the same parent, launched after this one (the bridge refuses
    a resume from any other parent)."""
    agent_id, parent = row.get('agent_id'), row.get('parent_id')
    if not agent_id or not parent:
        return None, None
    ref = row.get('launched_at') or row.get('created_at')
    waiting = None
    for other in all_rows:
        if (other['dispatch_id'] == row['dispatch_id'] or other.get('operation') != 'resume'
                or other.get('agent_id') != agent_id or other.get('parent_id') != parent
                or not (ref and later(other.get('launched_at') or other.get('created_at'), ref))):
            continue
        if _own_turn_landed(other):
            return 'landed', other['dispatch_id']
        if other['state'] not in TERMINAL:
            waiting = waiting or other['dispatch_id']
    return ('pending', waiting) if waiting else (None, None)


NOT_HANDLED_NEXT = ('No observer took this failed attempt to AgentCompass\'s automatic retry or relaunch within the '
                    'time a fallback step holds its claim, so nothing was relaunched and the parent is told of the failure '
                    'as it is.')


# The bridge's intervals before a notice's next id after an unknown outcome (paseo_dispatch_bridge.mjs RESEND_AFTER_MS),
# in seconds: a notice waiting one out is not handed to a reader before it ends. A test reads both.
RESEND_AFTER_SECONDS = (15 * 60, 60 * 60)


def _plus(value, seconds):
    try:
        return stamp(parse(value) + timedelta(seconds=seconds))
    except (TypeError, ValueError):
        return None


def _unhandled_covered(by_id, row, now):
    """A failed attempt whose failure the fallback covers and that no fallback step holds or decided: no `handled`
    entry in its chain, or a claim no longer held (`claim_held`: its holder ended, or a claim naming no holder outlived
    its lease). The fallback is on."""
    import dispatch_fallback
    if row.get('state') not in ('failed', 'create_failed') or not dispatch_fallback.enabled():
        return False
    root = by_id.get(chain_root(row), row)
    handled = ((root.get('fallback') or {}).get('handled') or {}).get(row['dispatch_id'])
    if isinstance(handled, dict) and handled.get('action') != 'pending':
        return False
    if claim_held(handled, now):
        return False
    return bool(dispatch_fallback.failure_of(row)['covered'])


def unclosable(row):
    """Why a fallback step may not close this attempt as not relaunched (dispatch_fallback.abandon), or None when it
    may: only an attempt that ended in a failure (`failed`, `create_failed`) the fallback covers is one a fallback step
    owes a relaunch, so only its step's watcher closes it when that step ended twice without an answer. Read in the
    transaction that takes the claim (`update_fallback` `attempt`), so a dispatch still running, ended otherwise, or a
    failure the fallback does not relaunch is never closed by it."""
    import dispatch_fallback
    if row.get('state') not in ('failed', 'create_failed'):
        return f"it has not ended in a failure (state {row.get('state')})"
    failure = dispatch_fallback.failure_of(row)
    if not failure['covered']:
        return f"its failure is not one the fallback relaunches ({failure['label']})"
    return None


def _hold_end(row):
    """When the hold of an unhandled covered failure ends: RUN_LEASE after the registry recorded the attempt's end
    (`released_at`, stamped by the reader that settled it, not when the worker failed)."""
    return _plus(row.get('released_at') or row.get('updated_at'), RUN_LEASE)


def _awaits_fallback(by_id, row, now):
    """A failed attempt `reconcile` observes so its watcher is restarted and its fallback runs, whichever reader
    settled it: a worker this dispatch acknowledged (an observation bound it: `observed_at`; a resume the host refused
    never was), its covered failure unhandled, inside its hold and the root dispatch's lifetime."""
    root = by_id.get(chain_root(row), row)
    return (row.get('dispatch_mode', 'paseo') == 'paseo' and row.get('state') == 'failed' and bool(row.get('launched_at'))
            and bool(row.get('agent_id')) and bool(row.get('observed_at')) and takeover_allowed(root, now)
            and _unhandled_covered(by_id, row, now) and later(_hold_end(row), now))


def awaits_fallback(row, now=None, path=None):
    """`_awaits_fallback` for one row, its chain's first dispatch read from the registry."""
    root_id = chain_root(row)
    root = row if root_id == row['dispatch_id'] else inspect(root_id, path)
    return _awaits_fallback({root_id: root}, row, now or utcnow())


def awaiting_fallback(path=None, now=None):
    """The failed attempts `reconcile` observes beside the open dispatches (`_awaits_fallback`)."""
    if not Path(path or db_path()).exists():
        return []
    now = now or utcnow()
    with transaction(path, upkeep=True) as connection:
        all_rows = rows(connection, 'dispatches')
        by_id = {row['dispatch_id']: row for row in all_rows}
        return [row for row in all_rows if _awaits_fallback(by_id, row, now)]


def _chain_hold(by_id, row, now):
    """What this attempt's brief says about its end notice now, from the chain's record on its first dispatch
    (dispatch_fallback), as (hold, not_before): `relaunched` (another attempt carries the brief, and its own notice is
    the brief's: this one is not owed), `open` (a retry or relaunch may still start: nothing is sent and nothing final is
    recorded; `not_before` is when that can change), `close` (no observer took a covered failure within its hold: the
    caller closes the attempt in the chain, so no later observer relaunches it, and sends the notice), or None (the chain
    is final for this attempt, or no fallback applies: the notice is sent).

    What ends `open`: a fallback step's outcome (relaunched, final, scheduled); a scheduled retry's guard lapsing (its
    claim then retires the retry, `dispatch_fallback.Chain.claim`, so it never starts); a claim no longer held
    (`claim_held`: its holder ended, or a claim naming no holder outlived its own `until`);
    the fallback switched off (`AGENTCOMPASS_ROLL_FALLBACK=off`: nothing more is relaunched); and, for a failure no
    observer took, the end of its hold (`_hold_end`). `reconcile` observes such a failure meanwhile (`_awaits_fallback`)
    so its watcher is restarted and the fallback runs. A live watcher's open fallback is one of these: its step holds a
    claim (`pending`, naming the step's process) or a scheduled retry while it runs."""
    import dispatch_fallback
    root = by_id.get(chain_root(row), row)
    fallback = root.get('fallback') or {}
    handled = (fallback.get('handled') or {}).get(row['dispatch_id'])
    action = handled.get('action') if isinstance(handled, dict) else None
    if action == 'relaunched':
        return 'relaunched', None
    if action == 'final' or not dispatch_fallback.enabled():
        return None, None
    if action == 'scheduled':
        return ('open', fallback.get('until')) if takeover_allowed(root, now) and fallback_guard(root, now) else (None, None)
    if claim_held(handled, now):
        # A live step past its lease still holds it (step_binds): its own end writes the chain's record, or its watcher
        # runs the step again, so no time moves it.
        return 'open', handled['until'] if later(handled.get('until'), now) else None
    if not _unhandled_covered(by_id, row, now):
        return None, None
    if not takeover_allowed(root, now):
        return 'close', None
    end = _hold_end(row)
    return ('open', end) if later(end, now) else ('close', None)


def _resend_wait(row, now):
    """When this row's notice may take its next id, while its receipt waits one out (paseo_dispatch_bridge.mjs
    deliverNotice `waiting`: `pending.unknown`), else None. A read before then would send and read nothing."""
    receipt = watcher_receipt(row['dispatch_id'])
    pending = receipt.get('pending') if isinstance(receipt, dict) else None
    if not isinstance(pending, dict) or pending.get('unknown') is not True:
        return None
    attempt = pending.get('attempt') if isinstance(pending.get('attempt'), int) and pending['attempt'] > 0 else 0
    if attempt >= len(RESEND_AFTER_SECONDS):
        return None
    due = _plus(pending.get('sent_at'), RESEND_AFTER_SECONDS[attempt])
    return due if later(due, now) else None


def _close_unhandled(connection, by_id, row, now):
    """Record a covered failure no observer took as final in its chain, in the caller's transaction: a fallback step that
    claims it later finds it handled (`dispatch_fallback.Chain.claim`) and relaunches nothing."""
    root = by_id.get(chain_root(row), row)
    fallback = dict(root.get('fallback') or {})
    fallback['handled'] = {**(fallback.get('handled') or {}), row['dispatch_id']: {
        'action': 'final', 'result': 'not_handled', 'root_dispatch_id': root['dispatch_id'],
        'lines': fallback.get('lines') or [], 'text': NOT_HANDLED_NEXT, 'at': stamp(now)}}
    fallback['updated_at'] = stamp(now)
    root['fallback'] = fallback
    save(connection, root)


def _notice_step(all_rows, by_id, row, now):
    """What a reader does now with one due end notice, decided without writing or a host call, as (action, detail,
    close): `relaunched` / `continued` (detail: the later dispatch) record it sent; `wait` (detail: when it can first
    move, or None when another row's change moves it) leaves it due and untouched; `send` hands it to the bridge.
    `close` says to close the attempt in its chain first (`_close_unhandled`)."""
    hold, not_before = _chain_hold(by_id, row, now)
    if hold == 'relaunched':
        return 'relaunched', None, False
    if hold == 'open':
        return 'wait', not_before, False
    close = hold == 'close'
    verdict, later_id = _later_turn(all_rows, row)
    if verdict == 'landed':
        return 'continued', later_id, close
    if verdict == 'pending':
        return 'wait', None, False
    waiting = _resend_wait(row, now)
    if waiting:
        return 'wait', waiting, False
    return 'send', None, close


def _due(row):
    notice = row.get('end_notice')
    return row.get('state') in _END_NOTICE_STATES and isinstance(notice, dict) and notice.get('state') == 'due'


def due_end_notices(path=None, now=None, *, timeout=10):
    """Terminal rows whose one end notice can be sent now. An open row is not due, even if an older flag remains.

    One brief gets one end notice, when its chain is final (`_chain_hold`): while an attempt's retry or relaunch may still
    start, its notice is not returned; an attempt another attempt replaced is recorded `sent` with source `relaunched`
    (the replacing attempt's notice reports the brief); an attempt no observer took in time is closed in its chain first.
    A notice waiting out its next id's interval (`_resend_wait`) is not returned until then either, so a read spends no
    bridge process on a notice that cannot move.

    The parent owns the end message of a worker it gave a later turn that landed (`_later_turn`): such a row is recorded
    here as `sent` with source `continued` and `continued_by`, and the later dispatch's own notice reports the worker;
    this turn's reply is not sent separately. A row whose later turn is still on its way is not returned and stays due:
    a later read decides. Once every later turn ended without landing, the row is returned and its own notice is sent."""
    target = Path(path or db_path())
    if not target.exists():
        return []
    probe = _read_only(target, min(timeout, 0.5))
    try:
        if not any(_due(row) for row in rows(probe, 'dispatches')):
            return []
    except sqlite3.OperationalError as exc:
        if 'no such table' in str(exc):
            return []
        raise
    finally:
        probe.close()
    now = now or utcnow()
    with transaction(path, upkeep=True, timeout=timeout) as connection:
        found = []
        all_rows = rows(connection, 'dispatches')
        by_id = {row['dispatch_id']: row for row in all_rows}
        for row in all_rows:
            if not _due(row):
                continue
            notice = row['end_notice']
            action, detail, close = _notice_step(all_rows, by_id, row, now)
            if close:
                _close_unhandled(connection, by_id, row, now)
            if action == 'relaunched':
                row['end_notice'] = dict(notice, state='sent', source='relaunched')
                save(connection, row)
            elif action == 'continued':
                row['end_notice'] = dict(notice, state='sent', source='continued', continued_by=detail)
                save(connection, row)
            elif action == 'send':
                found.append({'dispatch_id': row['dispatch_id'], 'parent_id': row.get('parent_id'),
                              'agent_id': row.get('agent_id'), 'message_id': notice.get('message_id'),
                              'ended_as': row['state']})
        return found


def needs_reader(path=None, timeout=0.5, now=None):
    """Whether any end notice or fallback chain needs a reader now, read without writing, without creating the file and
    without a host call (a writer holding the database longer than `timeout` seconds raises sqlite3.OperationalError).

    Returns {'needed': bool, 'due': [dispatch ids], 'observe': [dispatch ids], 'next_at': ISO stamp or None}: `due`, the
    notices a reader would act on now (`due_end_notices` would send, close, or record them sent); `observe`, the failed
    attempts `reconcile` would observe so their fallback runs (`_awaits_fallback`); `needed`, whether either is not
    empty; `next_at`, the earliest time a notice that waits now can move (a resend interval's end, a guard's `until`,
    a claim's `until`, a hold's end), None when no waiting notice has a known time."""
    now = now or utcnow()
    all_rows = read_dispatches(path, timeout)
    by_id = {row['dispatch_id']: row for row in all_rows}
    due, observe, times = [], [], []
    for row in all_rows:
        if _awaits_fallback(by_id, row, now):
            observe.append(row['dispatch_id'])
        if not _due(row):
            continue
        action, detail, _close = _notice_step(all_rows, by_id, row, now)
        if action != 'wait':
            due.append(row['dispatch_id'])
        elif detail:
            times.append(detail)
    return {'needed': bool(due or observe), 'due': due, 'observe': observe,
            'next_at': min(times, key=parse) if times else None}


def mark_end_notice_sent(dispatch_id, source, path=None, message_id=None):
    """Record that this dispatch's one end notice was sent, with the id it ended on when the bridge named one. A second
    call, or a row that is no longer due, changes nothing."""
    if not _safe_dispatch_id(dispatch_id) or not Path(path or db_path()).exists():
        return False
    with transaction(path, upkeep=True) as connection:
        try:
            row = get(connection, 'dispatches', dispatch_id)
        except ValueError:
            return False
        notice = row.get('end_notice')
        if not isinstance(notice, dict) or notice.get('state') != 'due':
            return False
        row['end_notice'] = dict(notice, state='sent', source=source,
                                 **({'message_id': message_id} if _is_message_id(message_id) else {}))
        save(connection, row)
        return True


_UNDELIVERED_SOURCES = ('watcher_undelivered', 'catchup_undelivered')


def name_undelivered(parent_id, now=None, path=None):
    """The end notices of this parent's dispatches that were given up as undelivered and not named to it yet, each
    recorded named (`named_at`) in the same transaction: the parent's own router answer names it once. Another parent's
    notices are never returned."""
    if not parent_id or not Path(path or db_path()).exists():
        return []
    now = now or utcnow()
    with transaction(path, upkeep=True) as connection:
        named = []
        for row in rows(connection, 'dispatches'):
            notice = row.get('end_notice')
            if (row.get('parent_id') != parent_id or not isinstance(notice, dict)
                    or notice.get('source') not in _UNDELIVERED_SOURCES or notice.get('named_at')):
                continue
            row['end_notice'] = dict(notice, named_at=stamp(now))
            save(connection, row)
            named.append({'dispatch_id': row['dispatch_id'], 'agent_id': row.get('agent_id'), 'ended_as': row['state'],
                          'message_id': notice.get('message_id')})
        return named


def _message_after(doc, stamp_value):
    last = doc.get('lastUserMessageAt') if isinstance(doc, dict) else None
    if not isinstance(last, str) or not isinstance(stamp_value, str):
        return False
    try:
        return parse(last) > parse(stamp_value)
    except ValueError:
        return False


def authorized_followup(row, doc, later_ids):
    """The parent answered an idle question: the same round is running again, with a user message after the release."""
    if row.get('state') != 'completed' or host_generation(row, doc, later_ids) != 'current':
        return False
    if not isinstance(doc, dict) or doc.get('archivedAt') or doc.get('providerUnavailable'):
        return False
    if doc.get('status') not in ('running', 'initializing'):
        return False
    return _message_after(doc, row.get('released_at'))


def refresh_live_rounds(connection, now):
    """Re-count a round the host has been allowed to continue. Does not invent a new dispatch."""
    for row in rows(connection, 'dispatches'):
        if row.get('dispatch_mode', 'paseo') != 'paseo' or row['state'] != 'completed':
            continue
        try:
            if now >= parse(row['hard_until']):
                continue
        except (TypeError, ValueError):
            continue
        doc = _agent_doc_for(row)
        later_ids = _later_dispatch_ids(connection, row)
        if not authorized_followup(row, doc, later_ids):
            continue
        row.pop('released_at', None)
        row['lease_until'] = stamp(min(now + timedelta(seconds=RUN_LEASE), parse(row['hard_until'])))
        row.setdefault('started_at', stamp(now))
        event(row, 'started', now, f"host_followup status={doc.get('status') or ''}", kind=HOST_FOLLOWUP_KIND)
        save(connection, row)


def _round_needs_write(row, doc, later_ids, now):
    try:
        if now >= parse(row['hard_until']):
            return False
    except (TypeError, ValueError):
        return False
    return authorized_followup(row, doc, later_ids)


def settle_ended_workers(connection, now):
    """Finish open paseo rows from the watcher receipt, or else from a current agent JSON. Does not resurrect."""
    for row in rows(connection, 'dispatches'):
        if _open_for_disk_settle(row) or row.get('side_effects_due'):
            apply_ending(connection, row, now)


def _reservation_maintenance(path, now, *, timeout=5.0):
    """Whether a read must write: an ending to record, a follow-up to count again, or effects still owed."""
    connection = _read_only(path, timeout)
    try:
        ended = pending = False
        stored = rows(connection, 'dispatches')
        for row in stored:
            if row.get('ground_truth_pending') or row.get('side_effects_due'):
                pending = True
            if ended:
                continue
            later_ids = _later_dispatch_ids(connection, row)
            doc = _agent_doc_for(row)
            if _open_for_disk_settle(row):
                receipt_state, _reason = state_from_receipt(row, watcher_receipt(row['dispatch_id']))
                if receipt_state and receipt_state != row['state']:
                    ended = True
                    continue
                target, _reason = terminal_from_agent_doc(row, doc, later_ids)
                if target and target != row['state']:
                    ended = True
                    continue
            if _round_needs_write(row, doc, later_ids, now):
                ended = True
        return ended, pending
    finally:
        connection.close()


def _finish_pending_side_effects(path, now):
    """Run provider-hold and launch-latch effects flagged on a row, then clear the flag. A failure leaves it."""
    target = Path(path or db_path())
    if not target.exists():
        return
    connection = _read_only(target)
    try:
        pending = [row for row in rows(connection, 'dispatches') if row.get('side_effects_due')]
    finally:
        connection.close()
    if not pending:
        return
    import provider_retry
    for row in pending:
        due = list(row.get('side_effects_due') or [])
        receipt = watcher_receipt(row['dispatch_id']) or {}
        terminal = receipt.get('terminal') if isinstance(receipt.get('terminal'), dict) else None
        done = []
        if 'provider_failure_record' in due:
            try:
                provider_retry.record(row.get('bundle'), terminal or {}, dispatch_id=row['dispatch_id'])
                done.append('provider_failure_record')
            except Exception:
                pass
        if 'provider_hold_clear' in due:
            try:
                provider_retry.clear(row['bundle'].get('account'), provider_retry.launch_model(row['bundle']),
                                     include_account=True,
                                     launched_at=row.get('launched_at') or row.get('created_at'),
                                     bundle=row['bundle'])
                done.append('provider_hold_clear')
            except Exception:
                pass
        if 'launch_latches' in due:
            try:
                import account_state
                import launch_failures
                import model_latch
                launch_failures.clear(row['bundle'].get('account'))
                account_state.clear_latch(row['bundle'].get('account'),
                                          launched_at=row.get('launched_at') or row.get('created_at'))
                model_latch.clear(row['bundle'].get('account'), model_latch.launch_model(row['bundle']),
                                  launched_at=row.get('launched_at') or row.get('created_at'))
                done.append('launch_latches')
            except Exception:
                pass
        if not done:
            continue
        try:
            with transaction(path, upkeep=True) as writing:
                latest = get(writing, 'dispatches', row['dispatch_id'])
                still = [item for item in (latest.get('side_effects_due') or []) if item not in done]
                if still:
                    latest['side_effects_due'] = still
                else:
                    latest.pop('side_effects_due', None)
                latest['terminal_side_effects'] = list(dict.fromkeys([*(latest.get('terminal_side_effects') or []), *done]))
                save(writing, latest)
        except (OSError, sqlite3.Error, ValueError):
            continue


# A ground-truth collection that raised an OSError is tried once more, at a read at least this long after it failed:
# one within the same command (a reconcile reads the registry twice within milliseconds) would not outlast a lock.
GROUND_TRUTH_RETRY_SECONDS = 60


def measure_completion(row, now, *, agent_doc, other_dispatch_ids=()):
    """The fields a completed row saves: its ground truth, or what stands for it when the collection raised. Measuring
    a completion is not the completion, so every caller completes the row and runs what follows either way.

    Of what can escape the collection (its own reads already turn a missing or unreadable transcript, config or git
    answer into a reason), only an OSError can be transient: a stat of a file or directory refused for a moment. The
    first one leaves the row pending with `ground_truth_retry` (`error`, the class, and `at`, when it failed), and
    `_record_pending_ground_truth` tries once more `GROUND_TRUTH_RETRY_SECONDS` later. Any other class, and an OSError
    on that retry, is a defect that fails the same way again: it is recorded at once, every field `null` with
    `collection_failed:<class>` (dispatch_ground_truth.failed_ground_truth). The exception's text is never kept."""
    import dispatch_ground_truth
    try:
        return dispatch_ground_truth.collect_ground_truth(
            row, idle_threshold=RUN_LEASE, recorded_at=stamp(now), agent_doc=agent_doc,
            other_dispatch_ids=set(other_dispatch_ids or ()))
    except Exception as exc:
        if isinstance(exc, OSError) and not row.get('ground_truth_retry'):
            return {'ground_truth_pending': True, 'ground_truth_other_ids': sorted(other_dispatch_ids or ()),
                    'ground_truth_retry': {'error': type(exc).__name__, 'at': stamp(now)}}
        return dispatch_ground_truth.failed_ground_truth(recorded_at=stamp(now), exc=exc)


def _save_measurement(row, fields):
    """Apply `measure_completion`'s fields: a record ends the pending state, a retry keeps it."""
    row.update(fields)
    if 'ground_truth_recorded_at' in fields:
        for key in ('ground_truth_pending', 'ground_truth_other_ids', 'ground_truth_retry'):
            row.pop(key, None)


def _retry_due(row, now):
    retry = row.get('ground_truth_retry')
    if not isinstance(retry, dict):
        return True
    try:
        return now >= parse(retry.get('at')) + timedelta(seconds=GROUND_TRUTH_RETRY_SECONDS)
    except (TypeError, ValueError):
        return True


def _record_pending_ground_truth(path, now):
    """Record ground truth only for rows this process just flagged. Never scan historical completions. A row waiting
    for its one retry (`measure_completion`) is left until the retry is due."""
    target = Path(path or db_path())
    if not target.exists():
        return
    connection = _read_only(target)
    try:
        pending = [row for row in rows(connection, 'dispatches')
                   if row.get('ground_truth_pending') and row.get('state') == 'completed' and _retry_due(row, now)]
    finally:
        connection.close()
    if not pending:
        return
    for row in pending:
        fields = measure_completion(row, now, agent_doc=paseo_agent_record(row.get('agent_id')),
                                    other_dispatch_ids=row.get('ground_truth_other_ids') or ())
        try:
            with transaction(path, upkeep=True) as writing:
                latest = get(writing, 'dispatches', row['dispatch_id'])
                if not latest.get('ground_truth_pending'):
                    continue
                _save_measurement(latest, fields)
                save(writing, latest)
        except (OSError, sqlite3.Error, ValueError):
            continue


def unbound_ended_resume(row):
    """A resume dispatch that ended without Paseo ever showing the worker on this turn: the worker's own state is unchanged."""
    return (isinstance(row, dict) and row.get('operation') == 'resume' and not row.get('observed_at')
            and row.get('state') in TERMINAL and row.get('state') != 'expired')


def _parent_histories(parent_id, now, path):
    """Every worker a dispatch of `parent_id` names, with its dispatches oldest first."""
    with transaction(path, upkeep=True) as connection:
        expire(connection, now)
        all_rows = rows(connection, 'dispatches')
    by_agent = {}
    for row in all_rows:
        if row.get('parent_id') != parent_id or unbound_ended_resume(row):
            continue
        agent_id = row.get('agent_id')
        if isinstance(agent_id, str) and agent_id.strip():
            by_agent.setdefault(agent_id.strip(), []).append(row)
    for history in by_agent.values():
        history.sort(key=lambda r: (r.get('created_at') or '', r.get('updated_at') or ''))
    return by_agent


class Refusal(NamedTuple):
    """Why a worker takes no new turn, and what the caller does about that."""
    reason: str
    remedy: str


class WorkerState(NamedTuple):
    """A state of a worker that takes no new turn: why, what the caller does about it, and whether a fresh worker (which
    has none of this worker's context) is also an answer."""
    reason: str
    fix: str
    fresh: bool


# The one place that says what to do about a worker in a state that takes no turn. The registry reads it from the Paseo
# record; the host's refusal of a resume (`worker_not_idle`, `worker_not_found`) names the same states by these keys
# (its `condition`), and dispatch_flow reads them from here.
WORKER_STATES = {
    'running': WorkerState('Paseo shows it running', 'wait until its turn ends (a new turn now would collide with it)', False),
    'initializing': WorkerState('Paseo shows it initializing', 'wait until it is idle', False),
    'closed': WorkerState('Paseo shows it closed', 'open it in Paseo so it is idle again', True),
    'error': WorkerState('it ended in an error', 'read its error (get_agent_activity on the worker) and deal with it', True),
    'provider_unavailable': WorkerState('its provider is unavailable',
                                        'wait until its provider answers again (the tray account list shows its state)', False),
    'permission_pending': WorkerState('it waits for a permission',
                                      'answer its pending permission (list_pending_permissions and respond_to_permission, '
                                      'or the Paseo app)', False),
    'archived': WorkerState('it is archived', 'unarchive it in Paseo', True),
    'absent': WorkerState('Paseo has no worker with this id', 'check the id with `paseo ls`', False),
}


def resume_way_out(state):
    """The way out for a worker in `state` (a key of WORKER_STATES): the fix, then the same `--resume` again."""
    known = WORKER_STATES[state]
    return (f'{known.fix[0].upper()}{known.fix[1:]}, then roll --resume again'
            + ('; or roll without --resume to start a fresh worker if its context is not needed' if known.fresh else '') + '.')


def worker_state_refusal(state):
    return Refusal(WORKER_STATES[state].reason, resume_way_out(state))


NOT_NAMED_BY_PARENT = Refusal('no dispatch of this parent names it',
                              'Check the id (get_agent_status, `paseo ls`): only a worker this session started with roll can '
                              'be resumed; roll without --resume to start a fresh worker.')


def _reuse_refusal(agent_id, history, parent_id, now, paseo_home, named):
    """Why the worker of `history` takes no new turn of `parent_id` now, and its Paseo record: `(Refusal, doc)`, the
    refusal None when it can. `named`: the caller names this worker, so the age of its last dispatch refuses nothing.
    The remedy follows the reason (WORKER_STATES): a fresh worker is offered only where it helps (it answers no
    permission, ends no running turn, and has none of this worker's context)."""
    if any(r['state'] in ACTIVE or (r['state'] == 'expired' and r.get('launched_at') and parse(r['hard_until']) > now)
           for r in history):
        return Refusal('a dispatch to it is still open',
                       'Wait for that dispatch\'s completion notification (its next_argv or `agentcompass dispatch '
                       '--dispatch-action reconcile` is optional to inspect it), then roll --resume again.'), None
    latest = history[-1]
    if latest.get('state') != 'completed':
        return Refusal(f"its last dispatch ended as {latest.get('state')}, not completed",
                       'Read how it ended (get_agent_activity on the worker) before resuming it; roll without --resume '
                       'to start a fresh worker if its context is not needed.'), None
    last_seen = latest.get('updated_at') or latest.get('created_at')
    if not named and (not last_seen or now - parse(last_seen) > timedelta(hours=REUSE_MAX_IDLE_HOURS)):
        return Refusal(f'its last dispatch is more than {REUSE_MAX_IDLE_HOURS} hours old',
                       'Roll without --resume to start a fresh worker, or name the worker with --resume.'), None
    doc = paseo_agent_record(agent_id, paseo_home=paseo_home)
    if isinstance(doc, dict):
        labels = doc.get('labels') if isinstance(doc.get('labels'), dict) else {}
        owner = labels.get('router.parent_id') or labels.get('paseo.parent-agent-id')
        if owner and owner != parent_id:
            return Refusal('Paseo records another parent as its owner',
                           'Only its owner session can resume it: roll from that session, or roll without --resume to '
                           'start a fresh worker.'), doc
        status = doc.get('status') or doc.get('lastStatus')
        # The order the bridge checks the same states in (its `worker_not_idle`).
        state = ('archived' if doc.get('archivedAt')
                 else status if status in ('running', 'initializing', 'closed')
                 else 'error' if status == 'error' or doc.get('attentionReason') == 'error'
                 else 'provider_unavailable' if doc.get('providerUnavailable')
                 else 'permission_pending' if doc.get('attentionReason') == 'permission' or doc.get('pendingPermissions')
                 else None)
        # `closed`: Paseo collected the idle runtime. A resume must cold-restore it from persistence first,
        # which outlasts the launch call (~25 s on Antigravity) and can land the turn on a new provider session.
        if state:
            return worker_state_refusal(state), doc
    return None, doc


def reuse_refusal(parent_id, agent_id, now=None, path=None, paseo_home=None):
    """Why `reusable_workers(parent_id, named=agent_id)` does not hold `agent_id`, and the way out; None when it does."""
    if not isinstance(parent_id, str) or not parent_id.strip() or not Path(path or db_path()).exists():
        return NOT_NAMED_BY_PARENT
    now = now or utcnow()
    history = _parent_histories(parent_id, now, path).get(agent_id)
    if not history:
        return NOT_NAMED_BY_PARENT
    return _reuse_refusal(agent_id, history, parent_id, now, paseo_home, True)[0]


def reusable_workers(parent_id, now=None, path=None, paseo_home=None, named=None):
    """Completed, idle Paseo workers owned by parent_id, newest first, with compaction/context state. `named`: the worker
    the caller names, held whatever the age of its last dispatch."""
    if not isinstance(parent_id, str) or not parent_id.strip() or not Path(path or db_path()).exists():
        return []
    import account_fleet
    import worker_sessions
    now = now or utcnow()
    by_agent = _parent_histories(parent_id, now, path)
    if not by_agent:
        return []
    p_home = Path(paseo_home or os.environ.get('PASEO_HOME') or Path.home() / '.paseo')
    home_dir = p_home.parent if p_home.name == '.paseo' else Path(os.environ.get('ROUTER_PROVIDER_HOME') or Path.home())
    max_tokens = worker_sessions.rotation_threshold(account_fleet.load_accounts())
    candidates = []
    for agent_id, history in by_agent.items():
        reason, doc = _reuse_refusal(agent_id, history, parent_id, now, paseo_home, agent_id == named)
        if reason:
            continue
        latest = history[-1]
        state = worker_sessions.session_state(agent_id, home=home_dir)
        carried = worker_sessions.worker_historical_tokens(state)
        worn = bool(isinstance(carried, (int, float)) and carried >= max_tokens)
        bundle = dict(latest.get('bundle') or {})
        bundle_id = bundle.get('bundle_id') or bundle.get('id')
        chain_id = next((r.get('chain_id') for r in reversed(history) if r.get('chain_id')), None) or (
            history[0].get('task_id') or f'worker-{agent_id[:8]}')
        task_ids = list(dict.fromkeys(r.get('task_id') for r in reversed(history) if r.get('task_id')))
        brief_paths = list(dict.fromkeys(r.get('brief_path') for r in reversed(history) if r.get('brief_path')))
        titles = list(dict.fromkeys((r.get('request') or {}).get('title') or '' for r in reversed(history)
                                    if (r.get('request') or {}).get('title')))
        profiles = [r['profile'] for r in reversed(history) if isinstance(r.get('profile'), dict)]
        read_only_review = bool(history and all(bool(r.get('read_only_review')) for r in history))
        doc_cwd = doc.get('cwd') if isinstance(doc, dict) else None
        worker_cwd = doc_cwd or latest.get('cwd') or (latest.get('request') or {}).get('cwd')
        candidates.append({
            'agent_id': agent_id,
            'dispatch_id': latest.get('dispatch_id'),
            'bundle': bundle,
            'bundle_id': bundle_id,
            'chain_id': chain_id,
            'task_id': latest.get('task_id'),
            'task_ids': task_ids,
            'brief_path': latest.get('brief_path'),
            'brief_paths': brief_paths,
            'workspace': latest.get('workspace'),
            'cwd': worker_cwd,
            'title': titles[0] if titles else '',
            'titles': titles,
            'profiles': profiles,
            'read_only_review': read_only_review,
            'updated_at': latest.get('updated_at') or latest.get('created_at') or '',
            'session_state': state,
            'worn': worn,
        })
    candidates.sort(key=lambda c: c['updated_at'], reverse=True)
    return candidates


def chain_root(row):
    """The dispatch the parent was given for this brief: an automatic relaunch is an attempt of its first dispatch."""
    return row.get('attempt_of') or row['dispatch_id']


RETRY_CANCELED_NEXT = ('The parent released this brief while AgentCompass waited to retry it, so the scheduled retry is '
                       'canceled and nothing runs this brief.')
RESUMED_NEXT = ('The parent sent a follow-up turn to the failed worker and it is running again, so AgentCompass '
                'retired the pending fallback and continues on that worker.')


def retire_scheduled(fallback, now, result, next_action, root_id=None, *, current=None, extra_ids=()):
    """End a pending fallback in the caller's transaction: a scheduled same-bundle retry (`current=None`, for release or
    expiry) or any pending retry/switch when the parent resumed `current`. The first dispatch's attempt record becomes
    final and pending claims become terminal, so a due-retry or watcher call that runs later finds it handled and starts
    nothing. Returns the retired attempt's id, or None when nothing was retired."""
    retry_of = fallback.get('retry_of')
    handled = dict(fallback.get('handled') or {})
    if current is None:
        if fallback.get('state') != 'retrying' or not retry_of:
            return None
        targets = [retry_of]
    else:
        pending = [k for k, v in handled.items() if isinstance(v, dict) and v.get('action') in ('scheduled', 'pending')]
        if retry_of and retry_of not in pending:
            pending.insert(0, retry_of)
        resumed = fallback.get('state') == 'final' and (fallback.get('outcome') or {}).get('result') == result
        if fallback and not resumed and fallback.get('state') not in ('retrying', 'relaunching') and not pending:
            return None
        targets = list(dict.fromkeys([*pending, *(k for k in extra_ids if k), current]))
    outcome = {'action': 'final', 'result': result, 'root_dispatch_id': root_id,
               'lines': fallback.get('lines') or [], 'text': next_action, 'at': stamp(now)}
    fallback['handled'] = {**handled, **{target: outcome for target in targets}}
    fallback.update(state='final', outcome={'result': result, 'next_action': next_action}, updated_at=stamp(now),
                    **({'current': current} if current is not None else {}))
    for key in ('until', 'retry_at', 'retry_of'):
        fallback.pop(key, None)
    return targets[0]


def fallback_guard(row, now):
    """AgentCompass is between two attempts of this brief (waiting out a transient backoff or relaunching): no worker
    runs, but one is about to, so the brief counts as dispatched until the guard lapses. A relaunch or claim whose step
    stopped (`claim_stopped`) is no guard: nothing is about to run until a step takes it over."""
    guard = row.get('fallback') or {}
    within = 'hard_until' not in row or takeover_allowed(row, now)
    if (guard.get('state') == 'retrying' and within and later(guard.get('until'), now)) or relaunch_held(guard, now):
        return True
    return any(claim_held(item, now) for item in (guard.get('handled') or {}).values())


def unresolved(row, now, exempt_chain=None):
    """A launch whose worker may be alive or may still appear; an expired Paseo lease stays open until a later
    observation. For non-Paseo host claims (`orca`, `t3`, `native`, `herdr`, `cli`) where roll has no daemon
    reconcile channel, the claim holds while its lease is live. A brief AgentCompass is relaunching is open too,
    except to that relaunch itself (`exempt_chain`, its first dispatch)."""
    if row.get('dispatch_mode', 'paseo') != 'paseo':
        return row['state'] in ACTIVE and parse(row['lease_until']) > now
    return row['state'] in ACTIVE or (row['state'] == 'expired' and bool(row.get('launched_at')) and parse(row['hard_until']) > now) \
        or (fallback_guard(row, now) and chain_root(row) != exempt_chain)


def holder(row):
    return {'dispatch_id': row['dispatch_id'], 'state': row['state'], 'agent_id': row.get('agent_id'),
            'task_id': row.get('task_id'), 'title': (row.get('request') or {}).get('title')}


def is_writer(row, now, caller, exempt_chain=None):
    # A row from before dispatches recorded their access counts as a writer. The caller is not its own conflict.
    return row.get('access') != 'read_only' and not (caller and row.get('agent_id') == caller) \
        and unresolved(row, now, exempt_chain)


def row_directories(row):
    request = row.get('request') if isinstance(row.get('request'), dict) else {}
    return {d for d in (canonical_dir(row.get('workspace')),
                        canonical_dir(row.get('cwd')),
                        canonical_dir(request.get('cwd'))) if d}


def writers_in(connection, directory, now, caller=None, exempt_chain=None):
    place = canonical_dir(directory)
    if not place:
        return []
    return [holder(r) for r in rows(connection, 'dispatches')
            if place in row_directories(r) and is_writer(r, now, caller, exempt_chain)]


def writers_at(directory, caller=None, now=None, path=None):
    """Unsettled writers recorded in canonical `directory` across all hosts."""
    now = now or utcnow()
    if not directory or not Path(path or db_path()).exists():
        return []
    with transaction(path, upkeep=True) as connection:
        expire(connection, now)
        return writers_in(connection, directory, now, caller)


def prepare(decision_id, task, launch, usage, accounts=None, now=None, path=None, resume_agent_id=None, fallback_prompt=None,
            access='write', new_worker=False, workspace=None, cwd=None, attempt=None, resume_bundle=None,
            explicit_resume=False):
    """`attempt` ({attempt_of, attempt, previous_attempt}) prepares an automatic relaunch of a brief whose earlier
    attempt failed (dispatch_fallback): that chain's own rows do not block it, its own live rows still do."""
    require_parent_id(task.get('parent_agent_id'))
    if isinstance(launch, dict) and 'workspaceId' in launch:
        raise Refused('launch cannot carry a host workspaceId; pass canonical directory workspace instead')
    exempt = (attempt or {}).get('attempt_of')
    canon_ws = canonical_dir(workspace)
    cwd = canon_ws or cwd
    if canon_ws and isinstance(launch, dict):
        launch = dict(launch, cwd=canon_ws)
    import account_fleet
    import dispatch_ground_truth
    import fast_modes
    import quota_pools
    import task_intake
    now = now or utcnow()
    accounts = accounts or account_fleet.load_accounts()
    launch_git = dispatch_ground_truth.capture_launch_git(cwd or (launch.get('cwd') if isinstance(launch, dict) else None))
    with transaction(path) as connection:
        expire(connection, now)
        decision = get(connection, 'decisions', decision_id)
        if not decision['bound'] or decision['task_hash'] != digest(task):
            raise Refused('Decision is unbound or belongs to another task/parent/input')
        detail = earlier_version(decision.get('allocation_task'))
        if detail:
            # Neither a new launch nor a resume of its prepared row proceeds under it; other decisions are not affected.
            raise Refused(EARLIER_VERSION.format(decision_id=decision_id, detail=detail))
        if isinstance(resume_bundle, dict):
            bundle = dict(resume_bundle)
            resolved_resume = resolve_bundle(bundle, accounts, False)
            if resolved_resume:
                bundle = {**bundle, **resolved_resume}
            bundle.setdefault('bundle_id', bundle.get('id'))
        else:
            bundle = decision['bundle']
        fast_note = None
        if resume_bundle is None and bundle.get('fast'):
            # A stored fast route whose harness has no fast contract now launches a new worker in its plain mode, as a
            # route named with --fast that cannot run fast does (route_judge.record_constraint), with the same note.
            model = account_fleet.canonical(bundle.get('model'), accounts)
            if lacks := fast_modes.refusal(account_fleet.bundle_harness(bundle, bundle.get('account')), model,
                                           account_fleet.row_spec(accounts, model)):
                bundle = {k: v for k, v in bundle.items() if k != 'fast'}
                fast_note = fast_modes.not_applied(lacks)
        request = launch_request(bundle, launch)
        if cwd is None and request.get('cwd'):
            cwd = request['cwd']
        request_hash = bound_hash(request)
        previous = connection.execute('SELECT data FROM dispatches WHERE decision_id=?', (decision_id,)).fetchone()
        if previous:
            row = json.loads(previous[0])
            if row['state'] == 'prepared':
                # Re-preparing revises the assignment text in place; the bound launch fields cannot differ.
                request['labels'].update({k: row['request']['labels'][k] for k in LABELS})
                row['request'] = request
                if canon_ws:
                    row['workspace'] = canon_ws
                if cwd and row.get('launch_commit') is None:
                    row.update(cwd=cwd, **launch_git)
                save(connection, row)
            return row
        if parse(decision['valid_until']) <= now or not decision['dispatchable']:
            raise Refused('Decision expired, deferred or requires a new route; run plan or record again')
        if canon_ws and access != 'read_only':
            # Same transaction as the insert: of two concurrent rolls only the first becomes the writer.
            holders = writers_in(connection, canon_ws, now, task['parent_agent_id'], exempt)
            if holders:
                raise WorkspaceBusy(canon_ws, holders)
        busy_agents = set()
        for old in rows(connection, 'dispatches'):
            if not unresolved(old, now, exempt):
                continue
            if not new_worker and (old['parent_id'], old['task_id']) == (decision['parent_id'], decision['task_id']):
                raise Refused('This parent/task already has an active dispatch; reconcile it first')
            if old.get('agent_id'):
                busy_agents.add(old['agent_id'])
        user_named = decision.get('constraint') == 'user_override' and resume_bundle is None
        fact = None
        if user_named:
            # A route the user named is held by the facts its record checked, with the same words; a quota window's
            # time comes from the route's plan row below (RouteFactRefused.timed_by), as at its record.
            fact = launch_fact_refusal(bundle, accounts, usage, now, user_named=True,
                                       records=lambda: active_records(connection, now), recorded=True)
            if fact and not fact.windows:
                raise fact
        else:
            try:
                check_launch_quota(bundle, accounts, usage, now)
            except ValueError as exc:
                raise CapacityChanged(str(exc)) from exc
        if resume_bundle is None and decision.get('listed_rows'):
            accounts = provider_inventory.with_listed_rows(accounts, decision['listed_rows'])
        if resume_bundle is None and decision.get('effort_override'):
            resolved = resolve_bundle(bundle, accounts, decision.get('constraint') == 'user_override')
            if resolved is None or any(resolved.get(k) != bundle.get(k)
                    for k in ('id', 'provider', 'account', 'model', 'effort', 'modeId', 'launch_model')):
                raise Refused('Explicit override contract changed; record a new decision')
            # Add only to this calculation, never to defaults or persistent account settings.
            accounts = deepcopy(accounts)
            accounts['accounts'][bundle['account']]['bundles'][bundle['bundle_id']] = resolved
        # Capacity check and insertion share the same transaction across concurrent parents.
        alloc_task = dict(decision['allocation_task'])
        profile = (
            alloc_task.get('profile')
            if isinstance(alloc_task.get('profile'), dict)
            else (task.get('profile') if isinstance(task.get('profile'), dict) else None)
        )
        read_only_review = (
            bool(profile.get('read_only_review'))
            if isinstance(profile, dict) and 'read_only_review' in profile
            else bool(isinstance(profile, dict) and profile.get('work_kind') == 'code_audit')
        )
        if read_only_review and isinstance(alloc_task.get('chain'), dict):
            cw = alloc_task['chain'].get('worker')
            if not resume_agent_id or (isinstance(cw, dict) and cw.get('read_only_review') is False):
                alloc_task.pop('chain', None)
        # The route fields come from this decision only (task_intake.ROUTE_FIELDS): its record's own (`unclassified`),
        # and what its constraint makes of the bundle it names. A caller's task never carries them.
        named = decision.get('constraint') == 'user_override' or resume_bundle is not None
        route = dict(decision.get('route') or {}) if named else {}
        if bundle.get('fast') and named:
            route['force_fast'] = True
        elif not bundle.get('fast'):
            alloc_task['disable_fast'] = True
        if named:
            route['user_named_bundle'] = bundle['bundle_id']
        if decision.get('constraint') == 'user_override' and resume_bundle is None:
            route['pinned_bundle'] = bundle['bundle_id']
        # The records the plan read (route_judge.plan_stage): the ledger when a cadence or a chain needs it, without this
        # decision's own row, which is the launch being prepared and not one before it.
        records = active_records(connection, now) + launches(rows(connection, 'dispatches'), accounts, now)
        if quota_pools.cadence_horizon(accounts) is not None or (alloc_task.get('chain') or {}).get('id'):
            import decision_ledger
            records += [r for r in decision_ledger.read_ledger() if r.get('decision_id') != decision_id]
        candidate = plan_row(alloc_task, bundle, usage, now, accounts, records, route, named)
        if fact:
            raise fact.timed_by(candidate, accounts, now)
        if candidate is None or any(candidate['bundle'].get(k) != bundle.get(k) for k in ('provider', 'account', 'model', 'effort')):
            raise CapacityChanged('Decision bundle no longer matches installed account configuration')
        exclusions, waits = plan_row_holds(candidate, named)
        if bundle.get('fast') and not candidate['bundle'].get('fast') and not exclusions:
            exclusions.append('fast mode quota is no longer sufficient')
        short = candidate.get('full_capacity_short') or {}
        if (decision.get('constraint') and not named and exclusions and not waits
                and set(exclusions) <= {s.get('text') for s in short.values()}):
            # A constraint held only by the full-capacity rule: the rule's numbers and the ways out, not "changed".
            raise FullCapacityRefused(bundle['account'], bundle['model'],
                                      {wid: s for wid, s in short.items() if s.get('text') in exclusions})
        if user_named and (exclusions or waits):
            # A route the user named held by a fact: the refusal its record gives for the same reading.
            raise named_route_refusal(candidate, exclusions, waits, accounts, now, user_named=True, recorded=True)
        if exclusions or (resume_bundle is None and waits):
            raise CapacityChanged('Capacity/availability changed: '
                                  + hold_cause(candidate, exclusions, waits if resume_bundle is None else []))
        if resume_agent_id is None and candidate.get('action') == 'resume':
            chain_worker = ((alloc_task.get('chain') or {}).get('worker')
                            if isinstance(alloc_task.get('chain'), dict) else None) or {}
            if isinstance(chain_worker.get('agent_id'), str) and chain_worker['agent_id'].strip():
                resume_agent_id = chain_worker['agent_id'].strip()
        if resume_agent_id in busy_agents:
            if explicit_resume:
                raise ResumedWorkerBusy(resume_agent_id)
            resume_agent_id = None
        dispatch_id = str(uuid.uuid4())
        labels = {'router.decision_id': decision_id, 'router.dispatch_id': dispatch_id,
                  'router.parent_id': decision['parent_id'], 'router.task_id': decision['task_id'],
                  'router.account': bundle['account'], 'router.bundle_id': bundle['bundle_id'],
                  'router.request_hash': request_hash}
        request['labels'].update(labels)
        chain_source = alloc_task if isinstance(alloc_task.get('chain'), dict) else (
            task if not read_only_review and isinstance(task.get('chain'), dict) else {}
        )
        chain_id = (chain_source.get('chain') or {}).get('id') if (resume_agent_id or not read_only_review) else None
        row = {'dispatch_id': dispatch_id, 'decision_id': decision_id,
               'parent_id': decision['parent_id'], 'task_id': decision['task_id'], 'task_hash': decision['task_hash'],
               'request_hash': request_hash, 'request': request, 'bundle': bundle,
               'expected_usd': candidate['expected_usd'], 'usage': usage.get('accounts', {}),
               'created_at': stamp(now), 'lease_until': stamp(now + timedelta(seconds=PREPARE_LEASE)),
               'hard_until': stamp(now + timedelta(seconds=MAX_LIFETIME)),
               'operation': 'resume' if resume_agent_id else 'create',
               'agent_id': resume_agent_id, 'access': access,
               **({'workspace': canon_ws} if canon_ws else {}),
               **({'cwd': cwd} if cwd else {}),
               **launch_git,
               'fallback_prompt': fallback_prompt if resume_agent_id else None,
               'read_only_review': read_only_review,
               **({'profile': profile} if isinstance(profile, dict) else {}),
               **({'brief_path': task['brief_path']} if task.get('brief_path') else {}),
               **({'chain_id': chain_id} if chain_id else {}),
               **({k: attempt[k] for k in ('attempt_of', 'attempt', 'previous_attempt') if attempt.get(k) is not None}
                  if attempt else {}),
               # A route the user named launches past the full-capacity rule, past a free allowance's local
               # estimate, and with --fast past the fast burn estimate, with its warning (candidate_eval).
               **({'warnings': warned} if (warned := [*([fast_note] if fast_note else []),
                                                       *(candidate[k] for k in ('capacity_warning', 'estimate_warning',
                                                                                'fast_estimate_warning')
                                                         if candidate.get(k))]) else {})}
        event(row, 'prepared', now)
        save(connection, row)
        return row


def dispatch_constraint(dispatch_id, path=None):
    """The constraint of the decision a dispatch launches (`user_override`, `explicit_resume`, ...), else None. The spawn
    gate reads it read-only without taking a write lock to admit a bound launch on an unchecked model only when the user
    named that route."""
    if not isinstance(dispatch_id, str) or not dispatch_id:
        return None
    target = Path(path or db_path())
    if not target.exists():
        return None
    connection = _read_only(target, 0.5)
    try:
        try:
            row = get(connection, 'dispatches', dispatch_id)
        except sqlite3.OperationalError as exc:
            if 'no such table' in str(exc):
                return None
            raise
        except ValueError:
            return None
        if row.get('constraint'):
            return row['constraint']
        did = str(row.get('decision_id') or '')
        if not did:
            return None
        parts = did.split(':')
        base_id = parts[1] if len(parts) == 3 else did
        try:
            dec = get(connection, 'decisions', base_id)
        except sqlite3.OperationalError as exc:
            if 'no such table' in str(exc):
                return None
            raise
        except ValueError:
            return None
        return dec.get('constraint')
    finally:
        connection.close()


def named_resume(row, path=None):
    """The dispatch continues a worker the caller named (`roll --resume`, decision constraint `explicit_resume`): it
    never becomes a create, and a failure of it is never relaunched on another worker."""
    return (decision_record(row['decision_id'], path) or {}).get('constraint') == 'explicit_resume'


def fallback_to_create(dispatch_id, claim_id, now=None, path=None):
    """Switch a claimed resume dispatch to a fresh create when the previous worker cannot be resumed (an automatic
    resume only: `named_resume` dispatches are released instead). Only the claim that is still `claim_id`'s, launching
    a resume, is switched; otherwise None, and the caller creates nothing: no worker is created for a row that cannot
    bind it."""
    now = now or utcnow()
    with transaction(path) as connection:
        row = get(connection, 'dispatches', dispatch_id)
        if row['state'] != 'launching' or row.get('operation') != 'resume' or row.get('claim_id') != claim_id:
            return None
        if isinstance(row.get('fallback_prompt'), str) and row['fallback_prompt'].strip():
            row['request']['initialPrompt'] = row['fallback_prompt']
        row['operation'] = 'create'
        row['agent_id'] = None
        row['updated_at'] = stamp(now)
        save(connection, row)
        return row


def inspect(dispatch_id, path=None):
    with transaction(path, upkeep=True) as connection:
        return get(connection, 'dispatches', dispatch_id)


def decision_record(decision_id, path=None):
    with transaction(path, upkeep=True) as connection:
        return get(connection, 'decisions', decision_id)


def current_attempt(dispatch_id, path=None):
    """The attempt now carrying this brief: the dispatch the parent was given, or the relaunch that replaced it."""
    row = inspect(dispatch_id, path)
    root = row if chain_root(row) == dispatch_id else inspect(chain_root(row), path)
    return (root.get('fallback') or {}).get('current') or root['dispatch_id']


def update_fallback(root_id, change, now=None, path=None, attempt=None):
    """Apply change(fallback dict) to the first dispatch's attempt record in one transaction; returns the row. With
    `attempt` (a dispatch id of the chain), that row as this transaction reads it is change's second argument, so a
    decision about it is taken on the state it has when the change is written."""
    now = now or utcnow()
    with transaction(path) as connection:
        row = get(connection, 'dispatches', root_id)
        fallback = dict(row.get('fallback') or {})
        if attempt is None:
            change(fallback)
        else:
            change(fallback, row if attempt == root_id else get(connection, 'dispatches', attempt))
        fallback['updated_at'] = stamp(now)
        row['fallback'] = fallback
        save(connection, row)
        return row


def create_deadline(row):
    """When a create the daemon never admitted is settled as failed; a claim is what starts the clock."""
    seconds = CREATE_DEADLINES.get(provider_inventory.launcher_family(row['bundle']['provider']), CREATE_DEADLINE)
    return parse(row['launched_at']) + timedelta(seconds=seconds)


def resume_deadline(row):
    """When an unconfirmed resume whose relabel never landed is settled as failed; past `send_by`, no relabel can arrive."""
    return parse(row['launched_at']) + timedelta(seconds=CREATE_DEADLINE)


def resolve_unconfirmed_resume(dispatch_id, now=None, path=None, from_receipt=False):
    """Settle a resume whose target worker still does not carry this dispatch's labels once its deadline has passed."""
    now = now or utcnow()
    with transaction(path) as connection:
        row = get(connection, 'dispatches', dispatch_id)
        if row['state'] not in ('launching', 'expired') or row.get('operation') != 'resume' or row.get('observed_at'):
            return row, False
        if not from_receipt and now < resume_deadline(row):
            return row, False
        event(row, 'create_failed', now, 'host_never_bound_resume')
        row['creation'] = {'phase': 'not_admitted'}
        _note_end_notice(row, watcher_receipt(row['dispatch_id']))
        save(connection, row)
        return row, True


def resolve_creation(dispatch_id, creation, now=None, path=None):
    """Settle a create no agent has appeared for, from the daemon's durable record of it (None: never admitted).

    `failed` means the daemon's run is over and it never reruns a key by itself, so nothing can appear later. Neither
    can a request never admitted once the bridge's send window has passed. An admitted create still in flight has no
    cancel on the host and can still produce a worker, so it stays open however late it is.

    Returns (row, whether this call ended it), so that of several processes settling one row only one acts on it. An
    unknown outcome says nothing against the provider, so it carries no failure_class and cools nothing down."""
    now = now or utcnow()
    with transaction(path) as connection:
        row = get(connection, 'dispatches', dispatch_id)
        if row['state'] not in ('launching', 'expired') or row.get('agent_id') or row.get('operation') != 'create':
            return row, False
        phase = creation.get('phase') if creation else None
        transitioned = True
        if phase == 'completed' or (phase == 'failed' and creation.get('agent_created')):
            # A worker existed, so it may have run: not the same as a create that never got that far.
            event(row, 'canceled', now, 'host_confirmed_agent_removed')
            row['creation'] = {'phase': phase, 'removed': True}
        elif phase == 'failed':
            cause = 'outcome_unknown' if creation.get('outcome_unknown') else creation.get('failure_class') or 'unknown'
            event(row, 'create_failed', now, 'host_creation_failed:' + cause)
            if not creation.get('outcome_unknown'):
                row['failure_class'] = creation.get('failure_class') or 'unknown'
                if isinstance(creation.get('failure_code'), str):
                    row['failure_code'] = creation['failure_code']
            row['creation'] = {'phase': phase}
        elif creation is None and now >= create_deadline(row):
            event(row, 'create_failed', now, 'host_never_admitted_create')
            row['creation'] = {'phase': 'not_admitted'}
        else:
            row['creation'] = {'phase': phase or 'not_admitted', 'in_flight': bool(phase)}
            transitioned = False
        if transitioned:
            _note_end_notice(row, watcher_receipt(row['dispatch_id']))
        save(connection, row)
        return row, transitioned


def _in_flight_claim(row, now):
    """A claim whose roll may still be sending it: launching, never bound to its worker by an observation, and inside its
    create window (`create_deadline`, from the claim). A resume carries its worker's id from the claim on, but until an
    observation binds it this dispatch has not acknowledged that worker."""
    return (row['state'] == 'launching' and not row.get('observed_at') and bool(row.get('launched_at'))
            and now < create_deadline(row))


def missing_agent(dispatch_id, agent_id, now=None, path=None):
    """A successful host lookup found a previously acknowledged agent removed. A claim still in flight is the roll's
    (`_in_flight_claim`): its resume answers for the worker, and it is left as it is."""
    now = now or utcnow()
    with transaction(path, upkeep=True) as connection:
        row = get(connection, 'dispatches', dispatch_id)
        if not agent_id or row.get('agent_id') != agent_id:
            raise Refused('Missing-agent observation does not match an acknowledged agent')
        if _in_flight_claim(row, now):
            return row
        if row['state'] in ACTIVE or row['state'] == 'expired':
            event(row, 'canceled', now, 'host_confirmed_agent_removed')
            _note_end_notice(row, watcher_receipt(dispatch_id))
            save(connection, row)
        return row


def claim(request, parent_id=None, call_id=None, now=None, path=None):
    """One-use pre-launch claim. Retries of the same host tool-call ID are idempotent."""
    now = now or utcnow()
    labels = request.get('labels') or {}
    with transaction(path) as connection:
        row = get(connection, 'dispatches', labels.get('router.dispatch_id'))
        require_parent_id(row['parent_id'])
        if any(labels.get(k) != row['request']['labels'][k] for k in LABELS):
            raise Refused('Dispatch labels do not match decision/account/task/parent')
        if parent_id is not None and parent_id != row['parent_id']:
            raise Refused('Dispatch belongs to another parent')
        if bound_hash(request) != row['request_hash']:
            raise Refused('Dispatch launch configuration changed (provider/model/mode/effort); '
                             'title/initialPrompt/workspaceId may be edited, these may not')
        if row['state'] == 'launching' and call_id and row.get('claim_id') == call_id and parse(row['lease_until']) > now:
            return row
        if row['state'] != 'prepared' or parse(row['lease_until']) <= now:
            raise Refused('Dispatch already claimed, terminal or expired; reconcile instead of replaying create_agent')
        import account_fleet
        import usage_reading
        try:
            check_launch_quota(row['bundle'], account_fleet.load_accounts(), usage_reading.usage_snapshot(now=now), now)
        except ValueError as exc:
            raise CapacityChanged(str(exc)) from exc
        row.update(claim_id=call_id or str(uuid.uuid4()), launched_at=stamp(now),
                   lease_until=stamp(min(now + timedelta(seconds=RUN_LEASE), parse(row['hard_until']))))
        event(row, 'launching', now)
        save(connection, row)
        return row


def finished_before(terminal, launched_at):
    """The bridge's notifier saw this launch finish; later host lifecycle (idle collection, plugin reload,
    archiving) must not rewrite that outcome."""
    if not isinstance(terminal, dict) or terminal.get('reason') != 'finished':
        return False
    at = terminal.get('at')
    try:
        return isinstance(at, str) and parse(at) >= parse(launched_at)
    except ValueError:
        return False


def provider_failed(terminal, launched_at):
    """The bridge's notifier saw this launch fail at its provider: stopped while the provider kept retrying without
    progress, or its turn ended with the provider's error."""
    if not isinstance(terminal, dict) or terminal.get('failure_code') not in ('provider_retry_stuck', 'provider_turn_failed'):
        return False
    at = terminal.get('at')
    try:
        return isinstance(at, str) and parse(at) >= parse(launched_at)
    except ValueError:
        return False


def provider_replied(terminal, launched_at):
    """The bridge's notifier saw this launch's turn end on the model's own reply. Paseo's idle/finished state reads the
    same after a cancel, so only this lifts a provider hold; a terminal without the flag says nothing."""
    if not isinstance(terminal, dict) or terminal.get('reason') != 'finished' or terminal.get('replied') is not True:
        return False
    at = terminal.get('at')
    try:
        return isinstance(at, str) and isinstance(launched_at, str) and parse(at) >= parse(launched_at)
    except ValueError:
        return False



def snapshot_turn_failure(dispatch_id, snapshot, terminal, path=None):
    """`terminal` as the notifier would have written it when this observation, not the notifier, is the first to see the
    launch's turn end in error: the host snapshot's `lastError` read as a turn failure (the bridge's turnFailure), so the
    classifier's scope attributes it to the provider or to the work like every other turn failure. Anything else, or a
    terminal that already names a provider failure, is returned unchanged."""
    import provider_retry
    if not isinstance(snapshot, dict) or (isinstance(terminal, dict)
                                          and terminal.get('failure_code') in provider_retry.FAILURE_CODES):
        return terminal
    message = provider_retry.turn_error_text(snapshot.get('lastError'))
    updated = snapshot.get('updatedAt')
    if not message or not isinstance(updated, str) or not (
            snapshot.get('status') == 'error' or snapshot.get('attentionReason') == 'error'):
        return terminal
    try:
        row = inspect(dispatch_id, path)
    except (ValueError, OSError):
        return terminal
    launched_at = row.get('launched_at')
    labels = snapshot.get('labels') or {}
    if (row.get('state') in TERMINAL or not isinstance(launched_at, str) or snapshot.get('id') != row.get('agent_id', snapshot.get('id'))
            or any(labels.get(k) != row['request']['labels'][k] for k in LABELS)):
        return terminal
    try:
        if parse(updated) < parse(launched_at):
            return terminal
        # A resumed worker's error is this turn's only once the turn was seen or its message sent after the launch.
        last_message = snapshot.get('lastUserMessageAt')
        if row.get('operation') == 'resume' and not row.get('started_at') and not (
                isinstance(last_message, str) and parse(last_message) >= parse(launched_at)):
            return terminal
    except ValueError:
        return terminal
    return {**(terminal if isinstance(terminal, dict) else {}), 'reason': 'error', 'at': updated,
            'failure_code': provider_retry.TURN_FAILED_CODE, 'provider_message': message}


def classify_provider_failure(dispatch_id, terminal, path=None):
    """Record a provider failure the notifier saw (provider_retry asks the classifier, so outside the registry's
    transaction) and return the scope it was recorded with, or None."""
    import provider_retry
    if not isinstance(terminal, dict) or terminal.get('failure_code') not in provider_retry.FAILURE_CODES:
        return None
    try:
        row = inspect(dispatch_id, path)
    except (ValueError, OSError):
        return None
    if row.get('state') in TERMINAL or not isinstance(row.get('launched_at'), str) \
            or not provider_failed(terminal, row['launched_at']):
        return None
    try:
        provider_retry.record(row['bundle'], terminal, dispatch_id=dispatch_id)
    except OSError:
        pass  # Best effort: provider_retry.ingest records it from the watcher's receipt on the next roll.
    try:
        return provider_retry.dispatch_scope(dispatch_id)
    except OSError:
        return None


def later(first, second):
    """True when ISO stamp `first` is after `second` (a stamp or a datetime); False when either is missing or malformed."""
    try:
        if isinstance(second, str):
            second = parse(second)
        return isinstance(first, str) and isinstance(second, datetime) and parse(first) > second
    except ValueError:
        return False


def process_started(pid):
    """When process `pid` started, as the OS counts it (Windows: its creation FILETIME; Linux: its start tick), so a pid
    the OS reused for another process does not read as the same one. None when no process of this user runs under it:
    none at all, or one this user may not query (Windows refuses PROCESS_QUERY_LIMITED_INFORMATION, which the owner of a
    process is granted across integrity levels, only for another user's or a protected process: a system process given
    a dead step's pid after a restart; elsewhere `kill(pid, 0)` refused). 0 where the OS gives no start time for a
    process of this user (the pid alone then names it)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # No such process, or access denied: another user's or a protected process, never a step of this user.
            return None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
                return None   # ended: a handle someone still holds keeps its pid open, not the process running
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            kernel.CloseHandle(handle)
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        pass
    try:
        os.kill(pid, 0)
    except OSError:     # PermissionError included: another user's process, never a step of this user
        return None
    return 0


def process_identity():
    """This process, as a fallback step's claim names its holder (`step_binds`)."""
    return {'pid': os.getpid(), 'started': process_started(os.getpid())}


def step_binds(record, now):
    """A fallback step's claim or relaunch in progress still binds, decided by the fact where the record names one. A
    record naming its step's process (`holder`) binds while that process runs (a process of this user with that pid
    and that start time: process_started) and has not released it (`released_at`, dispatch_fallback.Chain.release), however long the step takes: its lease (`until`) does
    not end it. A live step is working, not hung, because every call it waits on outside its own process is bounded by
    that call's own timeout (dispatch_flow.host_call, scorer_client.run_scorer); meanwhile the parent reads
    FALLBACK_RUNNING_NEXT (dispatch_flow.with_fallback). A record naming no holder (written by older code) binds until
    its lease ends."""
    if not isinstance(record, dict) or record.get('released_at'):
        return False
    holder = record.get('holder')
    if not isinstance(holder, dict):
        return later(record.get('until'), now)
    started = process_started(holder.get('pid'))
    return started is not None and holder.get('started') in (None, started)


def claim_held(item, now):
    """A fallback step's claim on a failed attempt (`pending`, dispatch_fallback.Chain.claim) that still binds
    (`step_binds`). The step's process ending (it exited or was killed, the machine restarted) releases it at once for
    whoever reads it next; so does the step itself when it stops by an exception (dispatch_fallback.handle). Only a held
    claim keeps others from taking the attempt over or keeps the brief counted as being relaunched."""
    return isinstance(item, dict) and item.get('action') == 'pending' and step_binds(item, now)


def relaunch_held(fallback, now):
    """A step is launching the brief's next attempt (`relaunching`) and still binds (`step_binds`)."""
    return fallback.get('state') == 'relaunching' and step_binds(fallback, now)


def takeover_allowed(root, now):
    """A stopped fallback step or a due retry of `root` is taken over only while the root dispatch's `hard_until`
    (`MAX_LIFETIME`) is in the future."""
    return isinstance(root, dict) and later(root.get('hard_until'), now)


def _stopped_step(fallback, now):
    if not isinstance(fallback, dict) or fallback.get('state') == 'final':
        return False
    if fallback.get('state') == 'relaunching' and not step_binds(fallback, now):
        return True
    return any(isinstance(item, dict) and item.get('action') == 'pending'
               and not step_binds(item, now) for item in (fallback.get('handled') or {}).values())


def claim_stopped(root, now):
    """A step took this brief's failed attempt and stopped before it finished, while the root dispatch is still within
    its lifetime (`takeover_allowed`). Nothing runs the brief until the next step takes the claim over."""
    return takeover_allowed(root, now) and _stopped_step((root or {}).get('fallback') or {}, now)


def fallback_expired(root, now):
    """An unclosed stopped fallback step or scheduled retry whose root dispatch is past `hard_until`
    (`not takeover_allowed`): nothing may launch for it, and the chain must be closed once as final."""
    if not isinstance(root, dict) or takeover_allowed(root, now):
        return False
    fallback = root.get('fallback') or {}
    if not isinstance(fallback, dict) or fallback.get('state') == 'final':
        return False
    if _stopped_step(fallback, now) or fallback.get('state') == 'retrying':
        return True
    return any(isinstance(item, dict) and item.get('action') == 'scheduled'
               for item in (fallback.get('handled') or {}).values())


def agent_resumed(snapshot, row, terminal=None):
    """True when `row` is a failed attempt whose bound host agent is running a new turn (or finished one sent after it
    failed). A stuck provider retry whose cancel failed never stopped, so its running status is still the failed turn."""
    if row.get('state') != 'failed' or not row.get('agent_id') or not isinstance(snapshot, dict):
        return False
    if snapshot.get('id') != row['agent_id']:
        return False
    if (row.get('failure_detail') or {}).get('cancelled') is False or (terminal or {}).get('cancelled') is False:
        return False
    if snapshot.get('archivedAt') or snapshot.get('providerUnavailable'):
        return False
    status = snapshot.get('status')
    if status in ('running', 'initializing'):     # Paseo keeps the failed turn's attention flag until it is viewed
        return True
    if snapshot.get('attentionReason') == 'error' and not later(snapshot.get('lastUserMessageAt'),
                                                                  snapshot.get('attentionTimestamp')):
        return False                                # the follow-up turn, if any, also errored
    if status == 'idle' and (snapshot.get('attentionReason') == 'permission'
                             or (snapshot.get('pendingPermissionCount') or 0) > 0):
        return True
    last_message = snapshot.get('lastUserMessageAt')
    released_at = row.get('released_at')
    if status == 'idle' and isinstance(last_message, str) and isinstance(released_at, str):
        try:
            ended = parse(released_at)
            if isinstance((terminal or {}).get('at'), str):
                ended = max(ended, parse(terminal['at']))
            return parse(last_message) > ended
        except ValueError:
            return False
    return False


def observe(dispatch_id, snapshot, now=None, path=None, terminal=None):
    """Accept only the bound host agent; start/terminal receipts are idempotent.

    `terminal` is the notifier's first terminal sighting of this dispatch, when it has one."""
    now = now or utcnow()
    import provider_retry
    terminal = snapshot_turn_failure(dispatch_id, snapshot, terminal, path)
    scope = classify_provider_failure(dispatch_id, terminal, path)
    with transaction(path, upkeep=True) as connection:
        row = get(connection, 'dispatches', dispatch_id)
        agent_id = snapshot.get('id')
        if row['state'] in TERMINAL and row['state'] != 'expired' and agent_resumed(snapshot, row, terminal):
            root = row if chain_root(row) == dispatch_id else get(connection, 'dispatches', chain_root(row))
            fallback = dict(root.get('fallback') or {})
            was_resumed = (fallback.get('state') == 'final'
                           and (fallback.get('outcome') or {}).get('result') == 'resumed'
                           and fallback.get('current') == row['dispatch_id']
                           and not any(isinstance(v, dict) and v.get('action') in ('scheduled', 'pending')
                                       for v in (fallback.get('handled') or {}).values()))
            if not was_resumed and retire_scheduled(fallback, now, 'resumed', RESUMED_NEXT,
                                                    root['dispatch_id'], current=row['dispatch_id']):
                root['fallback'] = fallback
                root.setdefault('events', []).append({
                    'state': root['state'], 'at': stamp(now),
                    'reason': f'pending fallback retired: parent resumed {row["dispatch_id"]}'})
                save(connection, root)
                if root['dispatch_id'] == dispatch_id:
                    row = root
            return row
        labels = snapshot.get('labels') or {}
        later_ids = _later_dispatch_ids(connection, row)
        if any(labels.get(k) != row['request']['labels'][k] for k in LABELS):
            prev_label = labels.get('router.dispatch_id') or ''
            owner = labels.get('router.parent_id') or labels.get('paseo.parent-agent-id')
            ref_stamp = row.get('created_at') or row.get('launched_at')
            last_msg = snapshot.get('lastUserMessageAt')
            try:
                msg_after_send = isinstance(last_msg, str) and isinstance(ref_stamp, str) and parse(last_msg) >= parse(ref_stamp)
            except ValueError:
                msg_after_send = False
            delivered_unlabeled = (
                row.get('operation') == 'resume'
                and isinstance(agent_id, str) and agent_id == row.get('agent_id')
                and (not owner or owner == row.get('parent_id'))
                and prev_label not in set(later_ids)
                and (
                    ('bound_unlabeled_from' in row and prev_label == row.get('bound_unlabeled_from'))
                    or (
                        not row.get('observed_at')
                        and not snapshot.get('archivedAt')
                        and snapshot.get('status') == 'running'
                        and msg_after_send
                    )
                )
            )
            if not delivered_unlabeled:
                raise NotBound('Host snapshot belongs to another dispatch/parent/task')
            row.setdefault('bound_unlabeled_from', prev_label)
        if not isinstance(agent_id, str) or not agent_id or row.get('agent_id') not in (None, agent_id):
            raise Refused('Host agent ID mismatch')
        provider, _, model = row['request']['provider'].partition('/')
        if snapshot.get('provider') != provider or snapshot.get('model') != model:
            raise Refused('Host provider/model differs from dispatched bundle')
        if snapshot.get('status') == 'running':
            settings = row['request']['settings']
            if settings.get('modeId') and snapshot.get('currentModeId') != settings['modeId']:
                raise Refused('Host permission mode differs from dispatched settings')
            effort = snapshot.get('effectiveThinkingOptionId') or snapshot.get('thinkingOptionId')
            if settings.get('thinkingOptionId') and effort != settings['thinkingOptionId']:
                raise Refused('Host thinking option differs from dispatched settings')
        # A completed worker can be resumed; its later quota failure still matters.
        if snapshot.get('status') == 'error':
            updated_error = snapshot.get('updatedAt')
            if isinstance(updated_error, str) and parse(updated_error) >= parse(row['launched_at']):
                import account_state
                import quota_exhaustion
                # The classifier decides when it read this failure (provider_retry.record); with no answer at all the
                # quota pattern may record the windows the bundle draws.
                latch, windows = quota_latch(row['bundle']) if scope is None else (False, None)
                try:
                    if latch:
                        quota_exhaustion.record(row['bundle']['account'], snapshot.get('lastError'),
                                                windows=windows, at=updated_error, source=agent_id)
                except OSError:
                    pass  # Best effort: a busy store must not stop this observation or the reconcile that made it.
                try:
                    account_state.record_launch_error(row['bundle']['account'], snapshot.get('lastError'),
                                                      code=snapshot.get('lastErrorClass'), at=updated_error, source=agent_id)
                except OSError:
                    pass
        try:
            within_life = now < parse(row['hard_until'])
        except (TypeError, ValueError):
            within_life = False
        followup = within_life and authorized_followup(row, snapshot, later_ids)
        # A real terminal stays, except an idle question whose parent has started another turn of this same round.
        if row['state'] in TERMINAL and row['state'] != 'expired' and not followup:
            return row
        if row['state'] == 'prepared':
            raise Refused('Agent observed without a launch claim')
        updated = snapshot.get('updatedAt')
        if not isinstance(updated, str) or parse(updated) < parse(row['launched_at']):
            if 'bound_unlabeled_from' in row and isinstance(snapshot.get('lastUserMessageAt'), str):
                ref = row.get('created_at') or row['launched_at']
                cand = updated if isinstance(updated, str) else snapshot['lastUserMessageAt']
                if parse(cand) < parse(ref):
                    raise Refused('Host snapshot predates this launch')
            else:
                raise Refused('Host snapshot predates this launch')
        row['agent_id'] = agent_id
        row['observed_at'] = stamp(now)
        if isinstance(snapshot.get('cwd'), str):
            row['cwd'] = snapshot['cwd']
        status = snapshot.get('status')
        reason = snapshot.get('attentionReason')
        # A resumed worker carries this dispatch's labels before its turn starts, so its idle/finished state may
        # still be the previous turn's. Only a seen run or a user message after this launch shows the new turn.
        last_message = snapshot.get('lastUserMessageAt')
        try:
            delivered = isinstance(last_message, str) and parse(last_message) >= parse(row['launched_at'])
        except ValueError:
            delivered = False
        undelivered = row.get('operation') == 'resume' and not row.get('started_at') and not delivered and status != 'running'
        if undelivered:
            reason = None
        # Superseded, unavailable, and removed are this row's ending. The snapshot status can be the next round.
        watcher_ending = isinstance(terminal, dict) and terminal.get('reason') in WATCHER_TERMINAL_REASONS
        stuck = provider_failed(terminal, row['launched_at'])
        target = None
        if watcher_ending:
            target = 'canceled'
        elif stuck:
            # The provider failed this turn (or the bridge stopped it retrying): whatever the host says after, it failed.
            target = 'failed'
            # A failed turn the classifier read as the worker's own error (or this task's input) is not the provider's.
            row['failure_code'] = provider_retry.outcome_code(terminal['failure_code'], scope)
            row['failure_detail'] = {k: terminal[k] for k in ('partial_work', 'cancelled') if k in terminal}
            if row['failure_code'] in provider_retry.WORKER_CODES and isinstance(terminal.get('provider_message'), str):
                # The worker's own error ends the chain: its answer carries the text, as the watcher's notice does.
                row['failure_detail']['worker_error'] = provider_retry.turn_error_text(terminal['provider_message'])
        elif finished_before(terminal, row['launched_at']) and status != 'running':
            target = 'completed'
        elif status == 'error' or reason == 'error' or snapshot.get('providerUnavailable'):
            target = 'failed'
        elif status == 'closed' or snapshot.get('archivedAt'):
            # Paseo closes a finished worker once it collects the idle runtime; only an unfinished one was canceled.
            target = 'completed' if reason == 'finished' else 'canceled'
        elif reason == 'permission' or snapshot.get('pendingPermissionCount', 0) > 0:
            target = 'started' if row.get('started_at') else 'launched'
        elif status == 'idle' and (reason == 'finished' or row.get('started_at')):
            target = 'completed'
        elif status == 'running':
            row.setdefault('started_at', stamp(now))
            target = 'started'
        elif status in ('initializing', 'idle'):
            target = 'launched'
        if target is None:
            raise Refused('Unrecognized host status; reservation retained until lease expiry')
        # A resume's prompt after its launch is its turn landing on a worker an earlier round used: kept so the end
        # notice of that earlier round knows it (`_own_turn_landed`). A create's states are as before: its own turn is
        # the worker's first, and `started_at` there means a seen run.
        if delivered and row.get('operation') == 'resume':
            row.setdefault('started_at', stamp(now))
        # A fresh host observation can recover an expired lease, but never outlive the hard bound.
        if target in ACTIVE and now >= parse(row['hard_until']):
            target = 'expired'
        # The same state read again is no sign of life, so only a turn that landed renews or recovers a resume's lease.
        # Without this every observation restored it, and release could never end one whose turn never came.
        if target in ACTIVE and undelivered and (row['state'] == 'expired' or now >= parse(row['lease_until'])):
            target = 'expired'
        # An expired row returns to an active state only with evidence the worker is still running. A stale idle
        # snapshot is not that evidence.
        if target in ACTIVE and row['state'] == 'expired' and not host_running_evidence(snapshot):
            target = 'expired'
        if target in ACTIVE:
            row.pop('released_at', None)
            if not undelivered:
                row['lease_until'] = stamp(min(now + timedelta(seconds=RUN_LEASE), parse(row['hard_until'])))
        prev_state = row['state']
        later_dispatch_ids = set()
        if target == 'completed' and agent_id:
            ref_stamp = row.get('launched_at') or row.get('created_at')
            for other in rows(connection, 'dispatches'):
                if other['dispatch_id'] != dispatch_id and other.get('agent_id') == agent_id:
                    other_stamp = other.get('launched_at') or other.get('created_at')
                    if ref_stamp and later(other_stamp, ref_stamp):
                        later_dispatch_ids.add(other['dispatch_id'])
        if row['state'] != target:
            reason_text = None
            kind = None
            if followup and target in ACTIVE:
                reason_text = f"host_followup status={snapshot.get('status') or ''}"
                kind = HOST_FOLLOWUP_KIND
            elif prev_state == 'expired' and target in ACTIVE:
                reason_text = running_evidence_reason(snapshot)
            elif target in TERMINAL and isinstance(terminal, dict) and terminal.get('reason') in WATCHER_TERMINAL_REASONS:
                reason_text = 'watcher_terminal: ' + terminal['reason']
            event(row, target, now, reason_text, kind=kind)
        if row['state'] in _END_NOTICE_STATES:
            _note_end_notice(row, watcher_receipt(dispatch_id))
        save(connection, row)
    if target == 'launched' and not row.get('started_at') and row.get('launch_commit') is None \
            and isinstance(row.get('cwd'), str) and row['cwd'].strip():
        import dispatch_ground_truth
        launch_git = dispatch_ground_truth.capture_launch_git(row['cwd'])
        with transaction(path, upkeep=True) as connection:
            latest = get(connection, 'dispatches', dispatch_id)
            if latest.get('launch_commit') is None and not latest.get('started_at'):
                latest.update(launch_git)
                save(connection, latest)
                row = latest
    if target == 'completed' and (prev_state != 'completed' or 'ground_truth_recorded_at' not in row):
        fields = measure_completion(row, now, agent_doc=paseo_agent_record(agent_id),
                                    other_dispatch_ids=later_dispatch_ids)
        with transaction(path, upkeep=True) as connection:
            latest = get(connection, 'dispatches', dispatch_id)
            _save_measurement(latest, fields)
            save(connection, latest)
            row = latest
    # Only a turn positively seen ending on the model's reply lifts a hold: a cancelled probe reads as finished too.
    if target == 'completed' and provider_replied(terminal, row.get('launched_at')):
        try:
            provider_retry.clear(row['bundle'].get('account'), provider_retry.launch_model(row['bundle']),
                                 include_account=True, launched_at=row.get('launched_at') or row.get('created_at'),
                                 bundle=row['bundle'])
        except OSError:
            pass
    if target in ('started', 'completed'):
        import account_state
        import launch_failures
        try:
            launch_failures.clear(row['bundle'].get('account'))
        except OSError:
            pass  # Best effort, and this observation is already saved.
        try:
            account_state.clear_latch(row['bundle'].get('account'), launched_at=row.get('launched_at') or row.get('created_at'))
        except OSError:
            pass
        import model_latch
        try:
            model_latch.clear(row['bundle'].get('account'), model_latch.launch_model(row['bundle']),
                              launched_at=row.get('launched_at') or row.get('created_at'))
        except OSError:
            pass
    return row


def release(dispatch_id, parent_id, reason, now=None, path=None, failed=False):
    """Explicit cancellation/known pre-send failure, not an invented terminal host receipt. The parent's cancellation
    (not `failed`) of a brief AgentCompass is waiting to retry also cancels that retry, in the same transaction
    (retire_scheduled); the returned row then carries `canceled_retry`, the failed attempt whose watcher to stop."""
    now = now or utcnow()
    with transaction(path) as connection:
        row = get(connection, 'dispatches', dispatch_id)
        if parent_id != row['parent_id']:
            raise Refused('Dispatch belongs to another parent')
        if row['state'] in ('launching', 'launched', 'started') and not failed:
            raise Refused('Observe terminal status before releasing a launched dispatch; release does not cancel an agent')
        if row['state'] == 'expired' and (row.get('creation') or {}).get('in_flight') and not failed:
            raise Refused('Paseo is still creating this worker and cannot cancel a create; release does not stop it')
        canceled = None
        if not failed:
            root = row if chain_root(row) == dispatch_id else get(connection, 'dispatches', chain_root(row))
            fallback = dict(root.get('fallback') or {})
            canceled = retire_scheduled(fallback, now, 'canceled', RETRY_CANCELED_NEXT, root['dispatch_id'])
            if canceled:
                # The failed row stays as it ended; only its attempt record and history change.
                root['fallback'] = fallback
                root.setdefault('events', []).append({'state': root['state'], 'at': stamp(now),
                                                      'reason': 'scheduled retry canceled: ' + reason[:200]})
                save(connection, root)
                if root['dispatch_id'] == dispatch_id:
                    row = root
        if row['state'] not in TERMINAL or row['state'] == 'expired':
            event(row, 'failed' if failed else 'canceled', now, reason)
            save(connection, row)
        return dict(row, canceled_retry=canceled) if canceled else row


def finalize_dispatch(dispatch_id, now=None, path=None, receipt_only=False):
    """Apply this dispatch's watcher receipt, and the side effects still owed, without a host call.

    `receipt_only` ignores the agent JSON. An ending the receipt does not carry leaves the row unchanged.
    Safe to call again: a stored terminal is kept, and effects that already ran are not repeated."""
    if not _safe_dispatch_id(dispatch_id) or not Path(path or db_path()).exists():
        raise Refused('invalid dispatch id')
    now = now or utcnow()
    with transaction(path) as connection:
        apply_ending(connection, get(connection, 'dispatches', dispatch_id), now, receipt_only=receipt_only)
    _finish_pending_side_effects(path, now)
    _record_pending_ground_truth(path, now)
    return inspect(dispatch_id, path)


def pending(path=None):
    if not Path(path or db_path()).exists():
        return []
    with transaction(path, upkeep=True) as connection:
        # Include expired started/uncertain Paseo launches so a later observation can repair them.
        return [r for r in rows(connection, 'dispatches')
                if r.get('dispatch_mode', 'paseo') == 'paseo' and unresolved(r, utcnow())]


def open_host_claims(now=None, path=None):
    """Open claims of hosts other than Paseo (`unresolved`), which `pending` leaves out and reconcile does not observe."""
    if not Path(path or db_path()).exists():
        return []
    now = now or utcnow()
    with transaction(path, upkeep=True) as connection:
        return [r for r in rows(connection, 'dispatches')
                if r.get('dispatch_mode', 'paseo') != 'paseo' and unresolved(r, now)]


def _supersede_native_handovers(connection, parent_id, task_id, now):
    """Cancel earlier unconsumed native hand-overs of `(parent_id, task_id)` when the same task is rolled again."""
    if not parent_id or not task_id:
        return
    for old in rows(connection, 'dispatches'):
        if (old.get('native_handover') is True
                and not old.get('launched_at')
                and (old.get('state') not in TERMINAL or old.get('state') == 'expired')
                and (old.get('parent_id'), old.get('task_id')) == (parent_id, task_id)):
            event(old, 'canceled', now, 'superseded_by_reroll')
            save(connection, old)


def record_host_claim(surface, decision, task, bundle, *, label=None, usage=None, access='write',
                      workspace=None, cwd=None, native_handover=False, native_tool=None, launch_fields=None,
                      hook_model=None, new_worker=False, now=None, path=None):
    """Record a non-Paseo host dispatch (`orca`, `t3`, `native`, `herdr`, `cli`) in `dispatches` under the same
    canonical-directory writer rule and SQLite BEGIN IMMEDIATE transaction as Paseo. The claim is `prepared`: it holds the
    workspace before the worker starts, and carries a launch time only once host_claim_launched (or claim_native_handovers
    when `native_handover` is True) says it started."""
    parent_id = task.get('parent_agent_id')
    require_parent_id(parent_id)
    if native_handover and (not isinstance(native_tool, str) or not native_tool.strip()
                            or not isinstance(launch_fields, dict) or not launch_fields):
        raise ValueError('A native hand-over requires native_tool and non-empty launch_fields.')
    now = now or utcnow()
    canon_ws = canonical_dir(workspace)
    landing = canon_ws or cwd
    decision_id = decision.get('decision_id') or f'{surface}-{uuid.uuid4()}'
    bundle_copy = dict(bundle or {})
    bundle_copy.setdefault('bundle_id', bundle_copy.get('id') or decision.get('choice') or f'{surface}:unknown')
    raw_usd = (decision.get('expected_usd') if isinstance(decision.get('expected_usd'), (int, float))
               else bundle_copy.get('expected_usd'))
    expected_usd = float(raw_usd) if isinstance(raw_usd, (int, float)) and not isinstance(raw_usd, bool) else None
    sids = own_session_ids()
    with transaction(path) as connection:
        expire(connection, now)
        by_dec = connection.execute('SELECT data FROM dispatches WHERE decision_id=?', (decision_id,)).fetchone()
        if by_dec:
            existing_row = json.loads(by_dec[0])
            if existing_row['state'] in ACTIVE:
                return existing_row
            decision_id = f'{surface}:{decision_id}:{uuid.uuid4().hex[:8]}'
        if native_handover and not new_worker:
            _supersede_native_handovers(connection, parent_id, task.get('task_id'), now)
        if canon_ws and access != 'read_only':
            holders = writers_in(connection, canon_ws, now, parent_id)
            if holders:
                raise WorkspaceBusy(canon_ws, holders)
        dispatch_id = str(uuid.uuid4())
        row = {
            'dispatch_id': dispatch_id,
            'decision_id': decision_id,
            'dispatch_mode': surface,
            'parent_id': parent_id,
            'task_id': task.get('task_id'),
            'request': {**({'title': label} if label else {}), **({'cwd': landing} if landing else {})},
            'bundle': bundle_copy,
            'expected_usd': expected_usd,
            'usage': (usage or {}).get('accounts', {}),
            'created_at': stamp(now),
            'lease_until': stamp(now + timedelta(seconds=PREPARE_LEASE)),
            'hard_until': stamp(now + timedelta(seconds=MAX_LIFETIME)),
            'operation': 'create',
            'agent_id': None,
            'access': access,
            **({'native_handover': True, 'native_tool': native_tool.strip(),
                'launch_fields': deepcopy(launch_fields)} if native_handover else {}),
            **({'hook_model': hook_model} if native_handover and hook_model else {}),
            **({'constraint': decision['constraint']} if decision.get('constraint') else {}),
            **({'host_session_id': task['host_session_id']} if task.get('host_session_id') else {}),
            **({'session_ids': sids} if sids else {}),
            **({'workspace': canon_ws} if canon_ws else {}),
            **({'cwd': landing} if landing else {}),
            **({'profile': task['profile']} if isinstance(task.get('profile'), dict) else {}),
            **({'brief_path': task['brief_path']} if task.get('brief_path') else {}),
        }
        event(row, 'prepared', now)
        save(connection, row)
        return row


def hand_over_native(dispatch_id, *, native_tool, launch_fields, hook_model=None, new_worker=False, now=None, path=None):
    """Record a one-use `native` hand-over from an unlaunched CLI host claim when `roll.cli` hands launch to the caller."""
    if not isinstance(native_tool, str) or not native_tool.strip() or not isinstance(launch_fields, dict) or not launch_fields:
        raise ValueError('A native hand-over requires native_tool and non-empty launch_fields.')
    now = now or utcnow()
    with transaction(path) as connection:
        source = get(connection, 'dispatches', dispatch_id)
        if source.get('launched_at'):
            return source
        if not new_worker:
            _supersede_native_handovers(connection, source.get('parent_id'), source.get('task_id'), now)
        raw_dec = str(source.get('decision_id') or f'native-{uuid.uuid4()}')
        parts = raw_dec.split(':')
        base_dec = parts[1] if len(parts) == 3 else raw_dec
        row = {
            'dispatch_id': str(uuid.uuid4()),
            'decision_id': f'native:{base_dec}:{uuid.uuid4().hex[:8]}',
            'dispatch_mode': 'native',
            'native_handover': True,
            'native_tool': native_tool.strip(),
            'launch_fields': deepcopy(launch_fields),
            **({'hook_model': hook_model} if hook_model else {}),
            'parent_id': source.get('parent_id'),
            'task_id': source.get('task_id'),
            'request': deepcopy(source.get('request') or {}),
            'bundle': deepcopy(source.get('bundle') or {}),
            'expected_usd': source.get('expected_usd'),
            'usage': deepcopy(source.get('usage') or {}),
            'created_at': stamp(now),
            'lease_until': stamp(now + timedelta(seconds=PREPARE_LEASE)),
            'hard_until': stamp(now + timedelta(seconds=MAX_LIFETIME)),
            'operation': 'create',
            'agent_id': None,
            'access': source.get('access', 'write'),
            **({k: deepcopy(source[k]) for k in ('constraint', 'host_session_id', 'session_ids',
                                                 'workspace', 'cwd', 'profile', 'brief_path') if source.get(k)}),
        }
        event(row, 'prepared', now, 'cli_fallback_to_native')
        save(connection, row)
        return row


def _handover_open(row, now):
    """True when `row` is an unconsumed native hand-over within its 24h hard lifetime (`MAX_LIFETIME` degrade bound)."""
    hard = _stored_time(row, 'hard_until')
    return (isinstance(row, dict)
            and row.get('native_handover') is True
            and not row.get('launched_at')
            and row.get('state') in ('prepared', 'expired')
            and hard is not None
            and now < hard)


def native_handovers(now=None, *, call_id=None, path=None, timeout=0.5):
    """Open native hand-overs (and any already claimed by `call_id` for idempotent retry), newest `created_at` first.
    Read-only: returns `[]` when no registry exists yet, and raises `(OSError, ValueError, sqlite3.Error)` when the
    registry is unreadable or locked so the gate denies."""
    now = now or utcnow()
    stored = read_dispatches(path, timeout=timeout)
    matched = [
        r for r in stored
        if isinstance(r, dict) and r.get('native_handover') is True and (
            _handover_open(r, now)
            or (bool(call_id) and r.get('claim_id') == call_id
                and r.get('state') not in ('canceled', 'failed', 'create_failed'))
        )
    ]
    matched.sort(key=lambda r: str(r.get('created_at') or ''), reverse=True)
    return matched


def claim_native_handovers(requests, *, now=None, path=None, timeout=0.5):
    """Consume all requested `(dispatch_ids, call_id)` native hand-overs atomically in one transaction.
    If any item has no open (or `call_id`-matching) hand-over, none is consumed and `None` is returned."""
    items = [(list(dict.fromkeys(str(d) for d in (ids or ()) if d)), cid) for ids, cid in (requests or ())]
    if not items:
        return []
    if any(not ids for ids, _ in items):
        return None
    now = now or utcnow()
    with transaction(path, timeout=timeout) as connection:
        claimed_ids = set()
        to_save = []
        claimed_rows = []
        for ids, call_id in items:
            matched_row = None
            if call_id:
                for did in ids:
                    if did in claimed_ids:
                        continue
                    found = connection.execute('SELECT data FROM dispatches WHERE id=?', (did,)).fetchone()
                    if found:
                        existing = json.loads(found[0])
                        if (existing.get('native_handover') is True and existing.get('claim_id') == call_id
                                and existing.get('state') not in ('canceled', 'failed', 'create_failed')):
                            matched_row = existing
                            claimed_ids.add(did)
                            break
            if matched_row is None:
                for did in ids:
                    if did in claimed_ids:
                        continue
                    found = connection.execute('SELECT data FROM dispatches WHERE id=?', (did,)).fetchone()
                    if not found:
                        continue
                    row = json.loads(found[0])
                    if not _handover_open(row, now):
                        continue
                    hard = parse(row['hard_until'])
                    row.update(claim_id=call_id or str(uuid.uuid4()), launched_at=stamp(now),
                               lease_until=stamp(min(now + timedelta(seconds=RUN_LEASE), hard)))
                    event(row, 'launched', now, 'native_handover_claimed')
                    matched_row = row
                    claimed_ids.add(did)
                    to_save.append(row)
                    break
            if matched_row is None:
                return None
            claimed_rows.append(matched_row)
        for row in to_save:
            save(connection, row)
        return claimed_rows


def roll_decision_ids(path=None, *, timeout=0.5):
    """Decision IDs owned by `roll` (`from_roll` decisions and `dispatches` rows), read without creating the DB.
    A ledger `record` row whose `decision_id` is in this set belongs to `roll`, never to a standalone `record` launch.
    Raises `(OSError, ValueError, sqlite3.Error)` when an existing database cannot be read."""
    target = Path(path or db_path())
    if not target.exists():
        return set()
    owned = set()
    connection = _read_only(target, timeout)
    try:
        for table in ('decisions', 'dispatches'):
            try:
                table_rows = rows(connection, table)
            except sqlite3.OperationalError as exc:
                if 'no such table' in str(exc):
                    continue
                raise
            for row in table_rows:
                if not isinstance(row, dict):
                    continue
                did = row.get('decision_id')
                if not isinstance(did, str) or not did:
                    continue
                if table == 'dispatches' or row.get('from_roll') is True:
                    owned.add(did)
                    parts = did.split(':')
                    if len(parts) == 3:
                        owned.add(parts[1])
    finally:
        connection.close()
    return owned


def host_claim_launched(dispatch_id, *, now=None, path=None):
    """Stamp a host claim (record_host_claim) launched once its worker started: its launch time, and the run lease."""
    now = now or utcnow()
    with transaction(path) as connection:
        row = get(connection, 'dispatches', dispatch_id)
        if row['state'] in ('prepared', 'expired'):
            row.update(launched_at=stamp(now),
                       lease_until=stamp(min(now + timedelta(seconds=RUN_LEASE), parse(row['hard_until']))))
            event(row, 'launched', now)
            save(connection, row)
        return row


def settle_host_claim(dispatch_id, target_state='canceled', note=None, *, now=None, path=None):
    """Settle a non-Paseo host claim (`completed`, `failed`, or `canceled`) when its adapter observes completion or failure."""
    if not dispatch_id or target_state not in TERMINAL or not Path(path or db_path()).exists():
        return None
    now = now or utcnow()
    with transaction(path) as connection:
        found = connection.execute('SELECT data FROM dispatches WHERE id=?', (dispatch_id,)).fetchone()
        if not found:
            return None
        row = json.loads(found[0])
        if row['state'] not in TERMINAL:
            event(row, target_state, now, note)
            save(connection, row)
        return row


def record_external_launch(dispatch_mode, external_id, decision_id, bundle, *, parent_id=None, task_id=None,
                           profile=None, brief_path=None, cwd=None, completed=False, now=None, path=None):
    """Persist an external non-Paseo runner dispatch (`grok-bot`, `dot`) in `dispatches` without holding Paseo budget."""
    if not external_id or not decision_id:
        return None
    import dispatch_ground_truth
    now = now or utcnow()
    dispatch_id = f'{dispatch_mode}:{external_id}'
    launch_git = dispatch_ground_truth.capture_launch_git(cwd, external_runner=True)
    with transaction(path) as connection:
        existing = connection.execute('SELECT data FROM dispatches WHERE id=?', (dispatch_id,)).fetchone()
        if existing:
            row = json.loads(existing[0])
        else:
            by_dec = connection.execute('SELECT data FROM dispatches WHERE decision_id=?', (decision_id,)).fetchone()
            if by_dec:
                return json.loads(by_dec[0])
            row = {
                'dispatch_id': dispatch_id,
                'decision_id': decision_id,
                'dispatch_mode': dispatch_mode,
                'external_task_id': external_id,
                'parent_id': parent_id,
                'task_id': task_id,
                'bundle': dict(bundle or {}),
                'created_at': stamp(now),
                'launched_at': stamp(now),
                'lease_until': stamp(now),
                'hard_until': stamp(now + timedelta(seconds=MAX_LIFETIME)),
                'operation': 'create',
                'agent_id': None,
                'access': 'write' if cwd else 'read_only',
                **({'cwd': cwd} if cwd else {}),
                **launch_git,
                **({'profile': profile} if isinstance(profile, dict) else {}),
                **({'brief_path': brief_path} if brief_path else {}),
            }
            event(row, 'external_launched', now)
            save(connection, row)
    if completed:
        return complete_external(dispatch_mode, external_id, now=now, path=path)
    return row


def complete_external(dispatch_mode, external_id, *, now=None, path=None):
    """Mark a previously recorded external runner dispatch `completed` and persist its ground-truth fields."""
    target_db = Path(path or db_path())
    if not external_id or not target_db.exists():
        return None
    dispatch_id = f'{dispatch_mode}:{external_id}'
    with transaction(target_db) as connection:
        found = connection.execute('SELECT data FROM dispatches WHERE id=?', (dispatch_id,)).fetchone()
        if not found:
            return None
        row = json.loads(found[0])
    if row.get('state') == 'completed' and ('ground_truth_recorded_at' in row or row.get('ground_truth_retry')):
        return row
    now = now or utcnow()
    fields = measure_completion(row, now, agent_doc=None)
    with transaction(target_db) as connection:
        latest = get(connection, 'dispatches', dispatch_id)
        if latest.get('state') != 'completed':
            event(latest, 'completed', now)
        _save_measurement(latest, fields)
        save(connection, latest)
        return latest
