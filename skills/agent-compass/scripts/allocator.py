"""Deterministic worker allocation: capability gates, task fit across the four capability categories
(`creativity`, `coding`, `aesthetics`, `tool_operation`), continuity with a coupled chain's existing worker,
and the opportunity cost of the quota the work would use.

Scores are panel standard deviations (`z = (x - x_anchor) / sd_panel`, the accounts.json `calibration_panel` anchor at High = 0) from
`references/five-axis-scores.json`. In `five_axis_fit`, each demanded category's zero point is its level-1 target `t1`
(the panel's weakest model), level 2 targets the panel median `t2`, and level 3 targets the panel's best `t3`:
`fit = sum_a w_a * (min(z_a, c_a) - t1_a) / sum_a w_a`, the cap `c_a` being the target at levels 1 and 2 and the
category's ceiling (the best measured model) at level 3 (scores below `t1_a` may go below zero, and economy outside
alternatives require positive fit above zero). The formula is defined in `docs/technical-reference.md` section 1. Every number that drives a choice
lives in `references/accounts.json` or `references/five-axis-scores.json` and is returned in the per-candidate breakdown.
"""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Also support the path-loaded allocator and python -I entrypoints.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import account_state
import host_policy
import live_models
import model_latch
import provider_retry
import provider_inventory
import accounts_override
import account_fleet
import axis_scores
import candidate_eval
import candidate_ranking
import fast_modes
import quota_economy
import quota_overrides
import quota_pools
import task_intake
import user_scores
import worker_sessions


def advised_hours(hours):
    """A wait the caller is told to wait or to allow (`--can-wait-hours`): `hours` rounded up to a tenth, so that waiting that
    long finds the window reset and allowing it passes the `available_in_hours > can_wait` check. Rounding to nearest could
    advise 4.0 for a 4.04h wait, which asks again."""
    advised = math.ceil(hours * 10) / 10
    while advised < hours:
        advised += 0.1
    return round(advised, 1)


def refusal_schedule(bundle, accounts, usage, now, *, blocked_wids=None, latched=None, records=None):
    """`(start_after_hours, retry_at, waiting_on)` when every refusing quota window (`blocked_wids`, else
    `quota_pools.launch_blockers`) has an observed reset (`quota_pools.after_reset`; for a latched window, the latch's
    own end, `quota_pools.latched_reset`: the reset the latching error stated, else its re-probe time), computed by the
    same owners the automatic deferral path uses (`quota_pools.window_state`, `quota_pools.after_reset`,
    `candidate_eval.binding_wait`, `advised_hours`); `(None, None, None)` otherwise."""
    accounts = accounts if isinstance(accounts, dict) else {}
    if '_five_axis' not in accounts:
        accounts = axis_scores.five_axis_overlay(accounts)
    usage = provider_inventory.merge_usage(usage, accounts, now)
    latched_set = quota_pools.latched_windows(accounts, bundle['account']) if latched is None else set(latched)
    if blocked_wids is None:
        blocked_wids = {
            wid for wid, _ in quota_pools.launch_blockers(
                bundle, accounts, usage, now, latched=latched_set, records=records,
            )
        }
    else:
        blocked_wids = set(blocked_wids)
    if not blocked_wids:
        return None, None, None
    spec = (accounts.get('accounts') or {}).get(bundle.get('account')) or {}
    policy = accounts.get('scoring') or {}
    row = ((usage or {}).get('accounts') or {}).get(bundle['account']) or {}
    stale = row.get('status') == 'stale' or row.get('stale') is True
    model = account_fleet.canonical(bundle.get('model') or '', accounts)
    available_at, waiting_on, waits, reset_times = None, None, {}, {}
    seen = set()
    for wid, wspec, observed in quota_pools.applicable_windows(spec, model, row, bundle, accounts):
        if wid not in blocked_wids:
            continue
        seen.add(wid)
        if wid in latched_set:
            # 0 left until the latch's known end, as the automatic path waits for it (candidate_eval); none known: no
            # schedule.
            reset_dt = quota_pools.latched_reset(accounts, bundle['account'])
            if reset_dt is None or reset_dt <= now:
                return None, None, None
            available_at, waiting_on = candidate_eval.binding_wait(
                available_at, waiting_on, (reset_dt - now).total_seconds() / 3600, wid, waits,
            )
            reset_times[wid] = reset_dt.isoformat()
            continue
        observed = quota_pools.cycle_observation(wspec, observed, row, now)
        state = quota_pools.window_state(
            wid, wspec, observed, spec, now, policy.get('short_window_hours', 24), account_stale=stale,
        )
        if not state['known'] or wspec.get('hours') is None:
            return None, None, None
        state, _, _ = quota_pools.reserve_inflight(
            bundle['account'], spec, wid, wspec, state, records, accounts, policy, now,
        )
        launch_floor = (float(wspec['launch_min_remaining_pct'])
                        if isinstance(wspec.get('launch_min_remaining_pct'), (int, float)) else None)
        if launch_floor is not None and state.get('full', 100.0) <= launch_floor:
            return None, None, None
        renewed = quota_pools.after_reset(state, wspec['hours'])
        if renewed is None:
            return None, None, None
        available_at, waiting_on = candidate_eval.binding_wait(
            available_at, waiting_on, renewed['deferred_hours'], wid, waits,
        )
        reset_dt = quota_pools.parse_time(state.get('reset_at')) or (now + timedelta(hours=renewed['deferred_hours']))
        reset_times[wid] = reset_dt.isoformat()
    if seen != blocked_wids or not available_at or not waiting_on:
        return None, None, None
    return advised_hours(available_at), reset_times[waiting_on], waiting_on


def named_pin_block(task, candidates):
    """Why what the user named cannot run, one line per row; [] when the task names nothing."""
    if not (task.get('constraint_model') or task.get('constraint_bundle') or task.get('constraint_effort') or task.get('pinned_bundle')):
        return []
    off_pin = ('explicit model constraint', 'explicit effort constraint')
    pinned = task.get('pinned_bundle')
    rows = [c for c in candidates if (pinned and candidate_ranking.candidate_for_bundle([c], pinned, task.get('force_fast')) is c)
            or (not pinned and not any(r in off_pin for r in c['excluded']))]
    raw = task.get('constraint_raw_model')
    spelled, colon_eff = task_intake.split_model_effort(raw) if raw else (task.get('constraint_model'), None)
    eff = task.get('constraint_effort')
    eff_part = None if (eff and not colon_eff and spelled and spelled.lower().endswith('-' + eff)) else eff
    named = (task.get('pinned_bundle') or (task.get('constraint_bundle') if not raw else None)
             or ':'.join(x for x in (spelled, eff_part) if x))
    if not rows:
        return [f'{named}: no configured bundle offers it']
    return [f"{named} via {c['id']}: {'; '.join(c['excluded'])}" for c in rows if c['excluded']]


def named_model_keys(task, candidates, accounts):
    """The `row_key`s of the model `task` names (`constraint_model`, `constraint_bundle`, or `pinned_bundle`), or None
    when the task names no model."""
    keys = None
    for bundle_id in (task.get('pinned_bundle'), task.get('constraint_bundle')):
        if bundle_id:
            base_id = bundle_id[:-len(fast_modes.ID_SUFFIX)] if bundle_id.endswith(fast_modes.ID_SUFFIX) else bundle_id
            cand = (candidate_ranking.candidate_for_bundle(candidates, bundle_id, task.get('force_fast'))
                    or candidate_ranking.candidate_for_bundle(candidates, base_id))
            found = {account_fleet.row_key(cand['bundle']['model'], accounts)} if cand is not None else set()
            keys = found if keys is None else (keys & found)
    pin = task.get('constraint_raw_model') or task.get('constraint_model')
    if pin:
        pool = account_fleet.pin_pool(accounts, extra=[c['bundle'] for c in candidates])
        found = {account_fleet.row_key(b['model'], accounts) for b in account_fleet.pin_matching_bundles(pin, pool, accounts)}
        keys = found if keys is None else (keys & found)
    return keys


def plan(raw_task, usage, now=None, unavailable=(), records=None, accounts=None, capabilities=None, overrides=None,
         host_policy_config=None, *, read_local_state=True, named_accounts=(), route=None):
    """`named_accounts`: accounts a named route or an explicit resume names (dispatch_registry.prepare). Their facts are
    read whether or not they take part in automatic routing (candidate_ranking.rank). `route`: the route fields of a
    decision AgentCompass recorded (task_intake.ROUTE_FIELDS); a caller's task never carries them."""
    accounts = accounts_override.normalize(accounts or account_fleet.load_accounts())
    capabilities = capabilities or account_fleet.load_capabilities(accounts=accounts)
    now = now or datetime.now(timezone.utc)
    usage = provider_inventory.merge_usage(usage, accounts, now)
    active_overrides = (quota_overrides.load_overrides(now) if read_local_state else []) if overrides is None else overrides
    usage, accounts, applied = quota_overrides.apply_overrides(active_overrides, usage, accounts, now)
    accounts = axis_scores.five_axis_overlay(accounts, read_local_state=read_local_state, now=now)
    usage = axis_scores.with_local_readings(usage, accounts, now)      # a free allowance's period, read from local spend
    policy = accounts['scoring']
    task = task_intake.normalize_task(raw_task, policy, capabilities, accounts, route=route)
    if task['worker'] is None and task['chain_id'] and records:
        for record in reversed(records):
            if record.get('chain') == task['chain_id'] and (record.get('bundle') or {}).get('bundle_id'):
                w = {'bundle': record['bundle']['bundle_id'], 'from_ledger': True}
                wid = record.get('worker_id') or (record.get('worker') or {}).get('agent_id')
                if wid and read_local_state:
                    w['agent_id'] = wid
                    w.update(worker_sessions.session_state(wid))
                task['worker'] = w
                break
    candidates, eligible, system = candidate_ranking.rank(task, usage, policy, accounts, capabilities, now, unavailable, applied, records,
                                        host_policy_config, read_local_state=read_local_state,
                                        named_accounts=named_accounts)
    # Metered last resort: only when no subscription bundle is eligible, re-rank with checked metered bundles admitted.
    if not eligible and any(c['excluded'] == [candidate_eval.METERED_LAST_RESORT] for c in candidates):
        task['metered_fallback'] = True
        candidates, eligible, system = candidate_ranking.rank(task, usage, policy, accounts, capabilities, now, unavailable, applied,
                                            records, host_policy_config, read_local_state=read_local_state,
                                            named_accounts=named_accounts)
    worker_id = (task['worker'] or {}).get('bundle')
    resume = next((c for c in candidates if c['id'] == worker_id), None)
    decision = {'task': {k: task[k] for k in ('kind', 'difficulty', 'hours', 'chain_hours', 'coupling')}, 'system': system,
                'assumptions': assumptions(candidates, accounts)}
    for err in candidate_eval.QUOTA_DATA_ERRORS:
        unread = sorted({c['bundle']['account'] for c in candidates if any(err in str(r) for r in c['excluded'])})
        if unread:
            decision.setdefault('warnings', []).append(
                f"{', '.join(unread)}: {err}; out of automatic routing and no substitute for other accounts")
    aged = {}
    for c in candidates:
        at = c['components'].get('reading_at')
        if at and not c['excluded']:
            aged.setdefault(c['bundle']['account'], at)
    for name, at in sorted(aged.items()):
        decision.setdefault('warnings', []).append(
            f"{name}: priced from its last good usage reading, taken at {at} in the current reset period "
            f"(the latest read failed); a provider hold on a model or pool still wins over this reading")
    if (sync := accounts.get('_host_policy_sync') or {}).get('mode') == 'error':
        warning = (host_policy.upgrade_failed(sync.get('error')) if sync.get('stage') == 'upgrade'
                   else host_policy.sync_failed(sync.get('error')))
        decision.setdefault('warnings', []).append(warning)
    state_policy = (host_policy.load_policy() if read_local_state else {'allowed_accounts': list((accounts.get('accounts') or {}).keys())}) if host_policy_config is None else host_policy_config
    state_warnings = account_state.warnings(accounts.get('_discovery'), accounts['accounts'],
                                            lambda name: host_policy.account_allowed(state_policy, name))
    if state_warnings:
        decision.setdefault('warnings', []).extend(state_warnings)
    live_doc = system.pop('_live_models', None) or (live_models.load_or_refresh(
        accounts=accounts,
        allowed=lambda name: host_policy.account_allowed(state_policy, name),
        now=now,
    ) if read_local_state else {'accounts': {}})
    latches = model_latch.active(now=now) if read_local_state else {}
    latch_warnings = live_models.warnings(
        live_doc,
        accounts,
        lambda name: host_policy.account_allowed(state_policy, name),
        covered=set(latches),
    )
    latch_warnings += model_latch.warnings(latches, lambda name: host_policy.account_allowed(state_policy, name)) if read_local_state else []
    latch_warnings += provider_retry.warnings(provider_retry.active(now=now),
                                              lambda name: host_policy.account_allowed(state_policy, name)) if read_local_state else []
    if latch_warnings:
        # First: the roll answer shows only the first few warnings, and a model that cannot start must be seen.
        decision['warnings'] = [*latch_warnings, *decision.get('warnings', [])]
    decision['task'].update(difficulty_source=task['difficulty_source'], axis_demand=axis_scores.five_axis_demand(task),
                            max_recommended_effort=None)
    if task.get('wants_fast'):
        decision['task']['wants_fast'] = True
    if task.get('read_only_review'):
        decision['task']['read_only_review'] = True
    if task.get('metered_fallback'):
        decision['task']['metered_fallback'] = True
    if task.get('constraint_model') or task.get('constraint_bundle') or task.get('pinned_bundle'):
        decision['task']['named_model'] = (task.get('pinned_bundle') or task.get('constraint_bundle')
                                           or task.get('constraint_raw_model') or task.get('constraint_model'))
    if task['profile']:
        decision['task'].update(profile=task['profile'], profile_difficulty=task['profile_difficulty'])
    if applied:
        decision['overrides'] = applied
    ranked_candidates = [*eligible, *[c for c in candidates if c['excluded']]]
    result = {'stage': 'plan', 'source': 'deterministic_allocator', 'observed_at': now.isoformat(), 'decision': decision,
              'ranking': ranked_candidates, 'usage': usage}
    if not eligible:
        closest = sorted(
            (c for c in candidates if c['excluded']),
            key=lambda c: (len(c['excluded']), -(c['components']['fit'] if c['components'].get('fit') is not None else float('-inf'))),
        )[:2]
        named_block = named_pin_block(task, candidates)
        decision.update({'action': 'none',
                         'reason': 'No eligible bundle. Closest: ' + ' | '.join(
                             f"{c['id']}: {'; '.join(candidate_eval.said_exclusions(c))}" for c in closest)})
        kinds = [candidate_eval.local_settings(c) for c in closest]
        if closest and all(kinds):
            # Nothing but the user's own settings on this machine keeps the closest candidates out (roll.none_next).
            decision['local_settings'] = {
                'exclusions': [f"{c['id']}: {said}" for c in closest for said in candidate_eval.said_exclusions(c)],
                'named_route_reaches': all(set(k) <= set(candidate_eval.LIFTED_BY_NAMING) for k in kinds)}
        if named_block:
            # What the user named cannot run. Say why and stop; nothing else stands in for it.
            decision['blocked'] = 'named_model_unavailable'
            decision.setdefault('warnings', []).extend(named_block)
            decision['reason'] = ' | '.join(named_block)
            result['next'] = ('The model the user named cannot run now: ' + '; '.join(named_block)[:300]
                              + '. Tell the user; do not substitute another model.')
        result['choice'] = 'none'
        return result
    can_wait = task['can_wait_hours']
    ready = [c for c in eligible if (c['available_in_hours'] or 0) <= can_wait]
    # Rule: `can_wait_hours` is the one limit on waiting, named or not. Within it, candidates compete by `five_axis_order`
    # (net score first, then wait). When nothing eligible can start within it, defer to the candidate that can start soonest
    # (equal waits broken by `five_axis_order`), never to a higher score days away and never to another model when one was named;
    # the caller rolls again once that window resets.
    pool = ready or sorted(
        eligible,
        key=lambda c: (round(float(c['available_in_hours'] or 0.0), 4), *candidate_ranking.five_axis_order(c)),
    )
    pinned_cand = (
        candidate_ranking.candidate_for_bundle(pool, task['pinned_bundle'], task.get('force_fast'))
        if task.get('pinned_bundle') else None
    )
    top = pinned_cand or pool[0]
    # Bounded by `can_wait_hours`: never name candidates beyond the caller's allowed wait as an improvement. On an unpinned
    # plan `ready[0]` already chose the highest-scoring candidate within `can_wait_hours`, so this is set only when
    # `pinned_bundle` picked a lower-scoring bundle while a higher-scoring candidate can start within `can_wait_hours`.
    better = ready[0] if ready else None
    if better is not None and better is not top and better['available_in_hours'] and better['score'] > top['score']:
        decision['better_if_waiting'] = {'bundle': better['id'], 'hours': round(better['available_in_hours'], 1),
                                         'score_gain': round(better['score'] - top['score'], 4)}
    forced = pinned_cand is not None
    decision.update({'action': top['action'],
                     'continuity': (f"{task['coupling']}-coupling chain stays with its owner"
                                    if (top is resume and top['action'] == 'resume' and not forced) else None)})
    if top['available_in_hours']:
        decision['wait_for'] = {'bundle': top['id'], 'account': top['bundle']['account'], 'window': top.get('waiting_on'),
                                'hours': advised_hours(top['available_in_hours'])}
    if (top['available_in_hours'] or 0) > can_wait:
        # can_wait_hours is a limit: the plan defers and says the wait is longer than allowed, it never starts later quietly
        after = f" ({top['bundle']['account']}'s {top['waiting_on']} window resets then)" if top.get('waiting_on') else ''
        within = f'within {can_wait:g}h' if can_wait else 'now'
        decision['reason'] = (f"The chain owner {top['id']} can resume in {advised_hours(top['available_in_hours'])}h{after}."
                              if top is resume and top['action'] == 'resume'
                              else f"Nothing eligible can start {within}; the earliest, {top['id']}, can start in {advised_hours(top['available_in_hours'])}h{after}.")
    # The margin is to the best other model, not to another effort, harness or account of the same model.
    rival = next((c for c in pool if c is not top
                  and account_fleet.row_key(c['bundle']['model'], accounts) != account_fleet.row_key(top['bundle']['model'], accounts)), None)
    margin = round(top['score'] - rival['score'], 4) if rival is not None and not forced else None
    decision['margin'] = margin
    if top.get('data_errors'):
        for note in top.get('notes') or ():
            if str(note).startswith('data error:'):
                w_note = f"{top['id']}: {note}"
                if w_note not in decision.get('warnings', ()):
                    decision.setdefault('warnings', []).append(w_note)
    if top['bundle'].get('fast'):
        decision['fast'] = True
        decision['fast_reason'] = top['components'].get('fast_reason')
    decision.setdefault('reason', explain(top, system))
    # No required ability: the fit and the economy chose this effort.
    chosen = top['bundle'].get('effort')
    decision['task']['max_recommended_effort'] = chosen if chosen in user_scores.EFFORTS else None
    result['choice'] = top['id']
    result['selected_bundle'] = dict(top['bundle'], bundle_id=top.get('base_id') or top['id'], expected_usd=top.get('expected_usd'),
                                     expected_hours=top.get('expected_hours'))
    result['expected_usd'] = top.get('expected_usd')
    result['expected_hours'] = top.get('expected_hours')
    result['decision']['alternatives'] = list(dict.fromkeys(c['id'] for c in pool if c['id'] != top['id']))[:3]
    if top['available_in_hours']:
        decision.update({'action': 'defer', 'start_after_hours': advised_hours(top['available_in_hours'])})
        # What a caller that will not wait can have instead: the best candidate that starts now, when there is one;
        # for a named model (`constraint_model`, `constraint_bundle`, `pinned_bundle`), only a bundle of that model.
        allowed_models = named_model_keys(task, candidates, accounts)
        ready_now = next((c for c in eligible if not c['available_in_hours']
                          and (allowed_models is None
                               or account_fleet.row_key(c['bundle']['model'], accounts) in allowed_models)),
                         None)
        if ready_now is not None:
            decision['starts_now'] = ready_now['id']
    if task.get('wants_fast') and not top['bundle'].get('fast'):
        decision['fast_not_applied'] = (top.get('fast_not_offered')
                                        or 'fast mode was priced as a candidate of its own and the plain mode ranked higher')
    return result


def assumptions(candidates, accounts):
    notes = []
    for c in candidates:
        if c['excluded']:
            continue
        spec = accounts['accounts'][c['bundle']['account']]
        if quota_pools.parse_time(spec.get('expires_at')) and spec.get('expires_precision') == 'date':
            notes.append(f"{c['bundle']['account']}: expiry is a date only; end of day assumed")
    return sorted(set(notes))


def explain(top, system):
    parts = [f"{top['id']} ({top['action']}) scored {top['score']:.4f}"]
    cap = top.get('capacity')
    if cap:
        rem_usd_str = f" (${cap['remaining_usd']:.2f})" if cap.get('remaining_usd') is not None else ""
        exp_usd_str = f" (~${top['expected_usd']:.2f})" if top.get('expected_usd') is not None else ""
        priced = (f"need probability {cap['need_start']:.2f} -> {cap['need_end']:.2f}, quota price "
                  f"{cap['quota_price']:.5f} fit/USD" if cap.get('priced') else 'a short window: feasibility only, not priced')
        members = round(cap['full'] / 100.0)
        left = (f"{cap['remaining']:.0f}% left" if members == 1
                else f"{cap['remaining']:.0f}% of one account left in a pool of {members} (full = {cap['full']:.0f}%)")
        parts.append(f"{cap['window']}: {left}{rem_usd_str}, work ~{top['expected_pct']}%{exp_usd_str} "
                     f"(chain ~{top['chain_pct']}%), {priced}, "
                     + (f"before its {'expiry' if cap['expires_first'] else 'reset'} in {cap['end_hours']:.0f}h, "
                        if cap.get('end_hours') is not None else 'with no reset, ')
                     + f"opportunity cost {top['components']['opportunity_cost']:.4f}")
    if top['bundle'].get('fast'):
        parts.append(f"fast mode enabled ({top['components'].get('fast_reason') or 'priced as its own candidate'})")
    if top['components'].get('boost'):
        parts.append(f"includes a temporary user-override boost of {top['components']['boost']:+.2f}")
    if top['components'].get('continuity'):
        parts.append(f"continuing its related worker adds {top['components']['continuity']:.2f} (one demand level)")
    if top.get('unscored_named_dispatch'):
        parts.append('unscored named dispatch (fit unmeasured)')
    return '; '.join(parts)
