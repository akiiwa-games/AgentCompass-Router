"""roll retries and relaunches a failed dispatch by itself (owner requirement 2026-11-10): isolated registry, fake host
that keeps each worker's state, fake judge that allocates with the real allocator. No live host or classifier."""
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import unittest
from unittest.mock import patch

import allocator
import account_fleet
import decision_ledger
import dispatch_fallback as fallback
import dispatch_flow as flow
import dispatch_registry as registry
import provider_retry
import route_judge
import usage_reading

from router_test_support import isolated_router_state, profile
import test_dispatch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'audit'))
import router_audit  # noqa: E402


def setUpModule():
    isolated_router_state(unittest.addModuleCleanup, policy=False)


class Host:
    """A Paseo daemon that keeps every worker it created. `refuse` maps a provider/model to the create failure it
    answers; `fail` maps one to what its worker does after it starts."""

    def __init__(self, test):
        self.test = test
        self.agents = {}          # dispatch id -> snapshot
        self.creates = []
        self.refuse = {}
        self.terminal = {}        # dispatch id -> the watcher's terminal receipt
        self.alive_at_create = []

    def alive(self):
        return [a for a in self.agents.values() if a['status'] in ('running', 'initializing')]

    def __call__(self, request):
        op = request['operation']
        if op == 'create':
            r = request['request']
            self.creates.append(r['provider'])
            # The point of the whole feature: never a second live worker on the brief.
            self.alive_at_create.append(len(self.alive()))
            if r['provider'] in self.refuse:
                return {'ok': False, 'sent': False, 'error': 'host_creation_failed', **self.refuse[r['provider']]}
            provider, _, model = r['provider'].partition('/')
            settings = r.get('settings') or {}
            self.agents[request['dispatch_id']] = {
                'id': request['dispatch_id'], 'provider': provider, 'model': model, 'status': 'running',
                'currentModeId': settings.get('modeId'), 'thinkingOptionId': settings.get('thinkingOptionId'),
                'updatedAt': registry.stamp(registry.utcnow() + timedelta(seconds=1)), 'labels': r['labels'],
                'initialPrompt': r['initialPrompt']}
            return {'ok': True, 'sent': True, 'agent': dict(self.agents[request['dispatch_id']]),
                    'completion_notification': {'status': 'watching'}}
        if op == 'observe':
            agent = self.agents.get(request['dispatch_id'])
            note = {'terminal': self.terminal[request['dispatch_id']]} if request['dispatch_id'] in self.terminal else None
            return {'ok': True, 'agent': dict(agent) if agent else None, 'creation': None,
                    'completion_notification': note}
        if op == 'workspace_agents':
            return {'ok': True, 'agents': []}
        raise AssertionError(op)

    def end(self, dispatch_id, status='error', terminal=None, **changes):
        """The worker's turn ends; `terminal` is what the bridge's watcher recorded."""
        self.agents[dispatch_id].update(status=status, updatedAt=registry.stamp(registry.utcnow() + timedelta(seconds=2)),
                                        **changes)
        if terminal is not None:
            self.terminal[dispatch_id] = {'at': registry.stamp(registry.utcnow() + timedelta(seconds=2)), **terminal}


class FallbackTests(unittest.TestCase):
    setUp_dispatch, snapshot = test_dispatch.DispatchTests.setUp, test_dispatch.DispatchTests.snapshot
    issue = test_dispatch.DispatchTests.issue

    def setUp(self):
        self.setUp_dispatch()
        # A second account that can take the work when the first attempt's model is refused or held.
        self.usage['accounts']['codex'] = {'status': 'fresh', 'fetched_at': self.now.isoformat(), 'windows': [
            {'id': 'usage', 'used_pct': 0, 'remaining_pct_before_reset': 100,
             'resets_at': (self.now + timedelta(hours=100)).isoformat(), 'seconds_until_reset': 100 * 3600}]}
        self.retry_store = self.root / provider_retry.FILE
        env = patch.dict(os.environ, {'ROUTER_PROVIDER_RETRY': str(self.retry_store), 'ROUTER_STATE_DIR': str(self.root),
                                      'ROUTER_MODEL_LATCH': str(self.root / 'latch.json')})
        env.start()
        self.addCleanup(env.stop)
        accounts = patch.object(account_fleet, 'load_accounts', return_value=self.accounts)
        accounts.start()
        self.addCleanup(accounts.stop)
        self.scopes = []

        def classify(account, provider, model, message, failure):
            return (self.scopes.pop(0) if self.scopes else 'unknown'), {'source': 'system_one', 'scorer_type': 'fake'}
        classifier = patch.object(provider_retry, 'classify_scope', side_effect=classify)
        classifier.start()
        self.addCleanup(classifier.stop)
        self.host = Host(self)
        self.ledger, self.plans = [], []
        test = self

        class Judge:
            @staticmethod
            def plan_task(task, args, unavailable=None):
                test.plans.append(sorted(unavailable or ()))
                planned = allocator.plan(dict(task), test.usage, registry.utcnow(),
                                         unavailable=unavailable or (), accounts=test.accounts)
                return dict(planned, **registry.issue(planned, task)), 0
        self.judge = Judge
        self.enterContext(patch.object(usage_reading, 'usage_snapshot', side_effect=lambda *_, **__: test.usage))
        self.enterContext(patch.object(decision_ledger, 'append_ledger', side_effect=test.ledger.append))
        self.launch = {'title': 'Fixture task', 'initialPrompt': 'Only fixture work'}

    # Helpers.

    def first(self, task=None, decision=None):
        """The parent's dispatch: planned, prepared and launched as roll does."""
        task = task or self.task
        if decision is None:
            planned = allocator.plan(task, self.usage, self.now, accounts=self.accounts,
                                     route={'pinned_bundle': 'lab_sonnet55_high'})
            decision = dict(planned, **registry.issue(planned, task, self.now))
        row = registry.prepare(decision['decision_id'], task, self.launch, self.usage, accounts=self.accounts)
        flow.launch(row['dispatch_id'], 'parent-a', host=self.host, watch_seconds=0)
        return registry.inspect(row['dispatch_id'])

    def handle(self, dispatch_id):
        """What the bridge's watcher runs when it sees the attempt fail."""
        return fallback.handle(dispatch_id, judge=self.judge, host=self.host)

    def at(self, **delta):
        """Run the block as if this much time had passed."""
        real = registry.utcnow
        return patch.object(registry, 'utcnow', side_effect=lambda: real() + timedelta(**delta))

    def transient_root(self, task=None, decision=None, minutes=None):
        """A first dispatch whose turn the provider failed transiently; `minutes`: the provider's own retry time."""
        self.scopes = ['transient']
        root = self.first(task=task, decision=decision)
        terminal = {'reason': 'error', 'failure_code': 'provider_turn_failed', 'provider_message': '429 retry later'}
        if minutes:
            terminal['next'] = registry.stamp(registry.utcnow() + timedelta(minutes=minutes))
        self.host.end(root['dispatch_id'], 'error', terminal=terminal)
        return root

    def fallback_lines(self):
        return [e for e in self.ledger if e.get('stage') == 'fallback']

    def assert_one_worker(self):
        self.assertTrue(all(n == 0 for n in self.host.alive_at_create), self.host.alive_at_create)
        self.assertLessEqual(len(self.host.alive()), 1)

    # Covered failures: the next candidate starts in the same dispatch.

    def test_create_failed_with_invalid_model_relaunches_on_the_next_candidate(self):
        import model_latch
        self.host.refuse['claude-lab/claude-sonnet-5-5'] = {'failure_class': 'unknown', 'failure_code': 'INVALID_MODEL'}
        root = self.first()
        self.assertEqual(root['state'], 'create_failed')
        out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        new = registry.inspect(out['dispatch_id'])
        self.assertEqual((new['attempt_of'], new['attempt'], new['previous_attempt']), (root['dispatch_id'], 2, root['dispatch_id']))
        self.assertNotEqual(new['bundle']['model'], 'claude-sonnet-5-5')
        self.assertIn('claude-lab/claude-sonnet-5-5', model_latch.active())
        self.assertIn('INVALID_MODEL', out['lines'][0])
        self.assertIn(f"switched to {new['bundle']['model']}", out['lines'][0])
        self.assert_one_worker()
        # The new worker is told what failed and to look at the workspace first.
        prompt = self.host.agents[new['dispatch_id']]['initialPrompt']
        self.assertIn('claude-sonnet-5-5', prompt.split('---')[0])
        self.assertIn('uncommitted changes', prompt)
        self.assertTrue(prompt.endswith('Only fixture work'))

    def test_a_relaunched_worker_that_already_completed_is_the_relaunch_not_a_failure(self):
        """A fast relaunched worker can end before the chain reads its row; completed is success, not a failure."""
        import dispatch_flow
        self.host.refuse['claude-lab/claude-sonnet-5-5'] = {'failure_class': 'unknown', 'failure_code': 'INVALID_MODEL'}
        root = self.first()
        real_launch = dispatch_flow.launch

        def launch_then_finish(dispatch_id, *args, **kwargs):
            receipt = real_launch(dispatch_id, *args, **kwargs)
            self.host.end(dispatch_id, 'idle', terminal={'reason': 'finished', 'status': 'idle'})
            registry.observe(dispatch_id, self.host.agents[dispatch_id], terminal=self.host.terminal[dispatch_id])
            return receipt
        with patch.object(dispatch_flow, 'launch', side_effect=launch_then_finish):
            out = self.handle(root['dispatch_id'])
        self.assertEqual(registry.inspect(out['dispatch_id'])['state'], 'completed')
        self.assertEqual(out['action'], 'relaunched')
        self.assertEqual(len(out['attempts']), 2)
        self.assertNotIn('failed', out['lines'][-1])

    def test_a_bundle_refused_at_prepare_goes_to_the_next_candidate(self):
        """A live model list that dropped the model after the plan (any exclusion) refuses the bundle at prepare."""
        planned = allocator.plan(dict(self.task), self.usage, self.now, accounts=self.accounts)
        decision = dict(planned, **registry.issue(planned, self.task, self.now))
        dropped = planned['selected_bundle']
        real_plan = allocator.plan

        def plan(task, usage, now=None, unavailable=(), **kwargs):
            # From now on the provider no longer offers the planned model.
            return real_plan(task, usage, now, unavailable=[*unavailable, dropped['bundle_id']], **kwargs)
        with patch.object(allocator, 'plan', side_effect=plan):
            with self.assertRaises(registry.CapacityChanged):
                registry.prepare(decision['decision_id'], self.task, self.launch, self.usage, accounts=self.accounts)
            out = fallback.after_refusal(decision['decision_id'], self.launch, 'Capacity/availability changed',
                                         judge=self.judge, host=self.host)
        self.assertEqual(out['action'], 'relaunched')
        row = registry.inspect(out['dispatch_id'])
        self.assertNotEqual(row['bundle']['bundle_id'], dropped['bundle_id'])
        # No row existed for the refused bundle, so the relaunch is the dispatch the parent holds.
        self.assertEqual(row['fallback']['current'], row['dispatch_id'])
        self.assertIn('refused_before_launch', row['fallback']['lines'][0])
        self.assertEqual(self.plans[-1], [dropped['bundle_id']])

    def test_a_worker_that_ended_in_error_is_relaunched_once_it_is_over(self):
        root = self.first()
        self.assertEqual(root['state'], 'started')
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error', 'status': 'error'})
        out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertEqual(registry.inspect(root['dispatch_id'])['state'], 'failed')
        self.assertIn('worker_error', out['lines'][0])
        self.assertEqual(self.plans, [[root['bundle']['bundle_id']]])
        self.assert_one_worker()

    def test_a_stuck_provider_retry_is_relaunched_only_when_the_stop_took(self):
        root = self.first()
        stuck = {'reason': 'provider retry stuck', 'failure_code': 'provider_retry_stuck', 'provider_message': 'overloaded',
                 'attempts': 5, 'partial_work': True, 'cancelled': False}
        self.host.end(root['dispatch_id'], 'running', terminal=stuck)
        refused = self.handle(root['dispatch_id'])
        self.assertEqual((refused['action'], refused['result']), ('final', 'declined'))   # it may still be running
        self.assertEqual(self.host.creates, ['claude-lab/claude-sonnet-5-5'])
        self.assertIn('stop_failed', refused['lines'][0])
        other_task = dict(self.task, task_id='stuck-2')
        sol = dict(account_fleet.bundles(self.accounts)['sol_high'], bundle_id='sol_high')
        other = self.first(task=other_task, decision=self.issue(task=other_task, choice='sol_high', selected_bundle=sol))
        self.host.end(other['dispatch_id'], 'idle', terminal=dict(stuck, cancelled=True))
        out = self.handle(other['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertIn('provider_retry_stuck (classifier scope unknown)', out['lines'][0])
        self.assertIn('hold', out['attempts'][0])   # the hold provider_retry recorded, as the tray shows it

    def test_a_turn_failed_by_its_provider_moves_on_and_holds_as_the_classifier_said(self):
        self.scopes = ['account']
        root = self.first()
        self.host.end(root['dispatch_id'], 'error', terminal={
            'reason': 'error', 'failure_code': 'provider_turn_failed', 'provider_message': '429 quota exhausted'})
        out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertIn('claude-lab', provider_retry.active())    # the account, as the classifier scoped it
        new = registry.inspect(out['dispatch_id'])
        self.assertNotEqual(new['bundle']['account'], 'claude-lab')
        self.assertIn('classifier scope account', out['lines'][0])

    def test_the_lease_moves_to_the_new_bundle(self):
        root = self.first()
        self.assertEqual([r['dispatch_id'] for r in registry.reservations()], [root['dispatch_id']])
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        out = self.handle(root['dispatch_id'])
        held = registry.reservations()
        self.assertEqual([r['dispatch_id'] for r in held], [out['dispatch_id']])
        self.assertEqual(held[0]['choice'], registry.inspect(out['dispatch_id'])['bundle']['bundle_id'])
        self.assertNotEqual(held[0]['choice'], root['bundle']['bundle_id'])

    # Transient, task input, the owner's model, nothing left.

    def test_transient_retries_the_same_bundle_after_its_backoff_when_the_task_can_wait(self):
        root = self.transient_root(task=dict(self.task, can_wait_hours=1))
        out = self.handle(root['dispatch_id'])
        # Nothing waits in the step: it records the retry time and returns (provider_retry: 2 minutes for a first hold).
        self.assertEqual(out['action'], 'scheduled')
        wait = (registry.parse(out['retry_at']) - registry.utcnow()).total_seconds()
        self.assertTrue(100 <= wait <= 125, wait)
        self.assertEqual(len(self.host.creates), 1)
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['retry_of'], fb['retry_at']), ('retrying', root['dispatch_id'], out['retry_at']))
        with self.at(minutes=3):
            out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        new = registry.inspect(out['dispatch_id'])
        self.assertEqual(new['bundle']['bundle_id'], root['bundle']['bundle_id'])
        self.assertIn('retried the same model after its backoff', out['lines'][0])
        self.assertEqual(self.plans, [])   # no re-allocation for a retry
        self.assertEqual([e['action'] for e in self.fallback_lines()], ['scheduled', 'retry_same'])
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['current']), ('relaunched', new['dispatch_id']))
        self.assertNotIn('retry_at', fb)
        self.assert_one_worker()

    def test_transient_without_a_wait_starts_the_next_candidate_now(self):
        root = self.transient_root()
        out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertNotEqual(registry.inspect(out['dispatch_id'])['bundle']['bundle_id'], root['bundle']['bundle_id'])

    def test_a_backoff_longer_than_the_run_lease_keeps_its_claim_until_the_retry(self):
        """Review F1: a second observer while the backoff runs past RUN_LEASE must not relaunch the brief, and the
        first one must not then rewrite the chain to final/declined."""
        root = self.transient_root(task=dict(self.task, can_wait_hours=1), minutes=16)
        first = self.handle(root['dispatch_id'])
        self.assertEqual(first['action'], 'scheduled')
        self.assertGreater((registry.parse(first['retry_at']) - registry.utcnow()).total_seconds(), registry.RUN_LEASE)
        # Past RUN_LEASE, before the retry time: the watcher looks again, the parent observes, the parent rolls again.
        with self.at(seconds=registry.RUN_LEASE + 60):
            again = self.handle(root['dispatch_id'])
            seen = flow.launch(root['dispatch_id'], 'parent-a', host=self.host, watch_seconds=0)
            self.assertEqual([r['dispatch_id'] for r in registry.pending()], [root['dispatch_id']])
        self.assertEqual((again['action'], again['retry_at']), ('scheduled', first['retry_at']))
        self.assertEqual(seen['retry_at'], first['retry_at'])
        self.assertIn('retries the same model by itself', seen['next_action'])
        self.assertEqual(len(self.host.creates), 1)
        # Due: one observer starts the retry; the other reads what it did.
        with self.at(minutes=17):
            started = self.handle(root['dispatch_id'])
            late = self.handle(root['dispatch_id'])
        self.assertEqual(started['action'], 'relaunched')
        self.assertEqual((late['action'], late['dispatch_id']), ('relaunched', started['dispatch_id']))
        self.assertEqual(len(self.host.creates), 2)
        self.assert_one_worker()
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['current']), ('relaunched', started['dispatch_id']))
        self.assertEqual(registry.current_attempt(root['dispatch_id']), started['dispatch_id'])

    def test_an_observation_starts_a_due_retry_without_the_watcher(self):
        root = self.transient_root(task=dict(self.task, can_wait_hours=1))
        self.assertEqual(self.handle(root['dispatch_id'])['action'], 'scheduled')
        real_handle = fallback.handle
        with self.at(minutes=3), patch.object(fallback, 'handle', side_effect=lambda dispatch_id, **kw: real_handle(
                dispatch_id, **dict(kw, judge=self.judge))):
            seen = flow.launch(root['dispatch_id'], 'parent-a', host=self.host, watch_seconds=0)
        self.assertEqual(len(self.host.creates), 2)
        self.assertEqual((seen['root_dispatch_id'], seen['attempt'], seen['state']), (root['dispatch_id'], 2, 'started'))
        self.assertEqual(seen['dispatch_id'], registry.current_attempt(root['dispatch_id']))

    # Review round 2 (round 055): G1 the parent cancels a scheduled retry; G2 the guard lives as long as the retry may run.

    def scheduled_root(self, minutes, can_wait_hours=1):
        task = dict(self.task, can_wait_hours=can_wait_hours)
        root = self.transient_root(task=task, minutes=minutes)
        out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'scheduled')
        return root, task

    def observe_old(self, dispatch_id):
        """The parent observes its dispatch (dispatch launch / next_argv), which starts a due retry (resume_due)."""
        real_handle = fallback.handle
        with patch.object(fallback, 'handle', side_effect=lambda d, **kw: real_handle(d, **dict(kw, judge=self.judge))):
            return flow.launch(dispatch_id, 'parent-a', host=self.host, watch_seconds=0)

    def roll_again(self, task):
        planned = allocator.plan(dict(task), self.usage, registry.utcnow(), accounts=self.accounts)
        decision = dict(planned, **registry.issue(planned, task))
        return registry.prepare(decision['decision_id'], task, self.launch, self.usage, accounts=self.accounts)

    def test_R3_the_parents_release_cancels_a_scheduled_retry(self):
        """Review G1 (probe R3): release on the failed first dispatch cancels the retry in one transaction; the
        watcher, an observation and resume_due that run at the retry time start nothing."""
        root, _ = self.scheduled_root(minutes=16)
        released = registry.release(root['dispatch_id'], 'parent-a', 'parent cancels this brief')
        self.assertEqual(released['canceled_retry'], root['dispatch_id'])
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['outcome']['result']), ('final', 'canceled'))
        self.assertEqual(fb['handled'][root['dispatch_id']]['result'], 'canceled')
        self.assertFalse({'until', 'retry_at', 'retry_of'} & set(fb))
        self.assertEqual(registry.pending(), [])
        self.assertEqual(flow.public(registry.inspect(root['dispatch_id']))['next_action'], registry.RETRY_CANCELED_NEXT)
        with self.at(minutes=17):
            watcher = self.handle(root['dispatch_id'])
            seen = self.observe_old(root['dispatch_id'])
            self.assertIsNone(fallback.resume_due(root['dispatch_id'], host=self.host))
        self.assertEqual((watcher['action'], watcher['result']), ('final', 'canceled'))
        self.assertEqual(seen['dispatch_id'], root['dispatch_id'])
        self.assertEqual(self.host.creates, ['claude-lab/claude-sonnet-5-5'])

    def test_the_release_command_stops_the_watcher_of_the_canceled_retry(self):
        import argparse
        root, _ = self.scheduled_root(minutes=16)
        notes = self.root / 'paseo-notifications'
        notes.mkdir(exist_ok=True)
        (notes / f"{root['dispatch_id']}.json").write_bytes(b'{}')
        args = argparse.Namespace(stage='dispatch', dispatch_action='release', dispatch_id=root['dispatch_id'],
                                  parent_id='parent-a', basis='the owner dropped this brief')
        with patch.object(flow, 'require_actual_parent'), patch.object(flow, 'reconcile', return_value=[]):
            out = flow.cli(args)
        self.assertEqual(out['canceled_retry'], root['dispatch_id'])
        self.assertTrue((notes / f"{root['dispatch_id']}.json.creation-failed").exists())

    def test_R4_the_brief_cannot_be_dispatched_again_while_its_retry_may_run(self):
        """Review G2 (probe R4): past retry_at + RUN_LEASE with no watcher and no observer, the guard still holds, so
        rolling the brief again is refused; the next observation then runs the one retry."""
        root, task = self.scheduled_root(minutes=3)
        with self.at(minutes=3 + 11):
            self.assertEqual([r['dispatch_id'] for r in registry.pending()], [root['dispatch_id']])
            self.assertIn('retries the same model by itself', flow.public(registry.inspect(root['dispatch_id']))['next_action'])
            with self.assertRaisesRegex(ValueError, 'already has an active dispatch'):
                self.roll_again(task)
        with self.at(minutes=3 + 40):
            seen = self.observe_old(root['dispatch_id'])
        self.assertEqual((seen['attempt'], seen['state']), (2, 'started'))
        self.assertEqual(len(self.host.creates), 2)
        self.assert_one_worker()

    def test_R4_a_retry_nobody_started_within_the_wait_is_retired_not_run_beside_a_new_dispatch(self):
        """Review G2: once the wait the task allows is over the guard lapses and the brief may be rolled again; the old
        retry is then retired by whatever touches it, so the brief never runs twice."""
        root, task = self.scheduled_root(minutes=3)
        with self.at(hours=1, minutes=1):
            self.assertEqual(registry.pending(), [])
            row = self.roll_again(task)
            flow.launch(row['dispatch_id'], 'parent-a', host=self.host, watch_seconds=0)
            self.host.end(row['dispatch_id'], 'idle', terminal={'reason': 'finished'})
            registry.observe(row['dispatch_id'], self.host.agents[row['dispatch_id']])
        self.assertEqual(len(self.host.creates), 2)
        with self.at(hours=1, minutes=30):
            seen = self.observe_old(root['dispatch_id'])
            late_watcher = self.handle(root['dispatch_id'])
        self.assertEqual(len(self.host.creates), 2)
        self.assertEqual(seen['dispatch_id'], root['dispatch_id'])
        self.assertEqual(seen['next_action'], fallback.RETRY_EXPIRED_NEXT)
        self.assertEqual((late_watcher['action'], late_watcher['result']), ('final', 'retry_expired'))
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['outcome']['result']), ('final', 'retry_expired'))

    def test_the_guard_of_a_scheduled_retry_lasts_the_wait_the_task_allows(self):
        root, _ = self.scheduled_root(minutes=3, can_wait_hours=2)
        fb = registry.inspect(root['dispatch_id'])['fallback']
        hours = (registry.parse(fb['until']) - registry.parse(fb['handled'][root['dispatch_id']]['retry_at'])).total_seconds()
        self.assertGreater(hours, 3600 * 1.9)   # can_wait_hours from the failure, well past retry_at + RUN_LEASE

    def test_a_stale_step_on_a_live_chain_never_rewrites_it_to_final(self):
        """A step whose prepare is refused because another attempt of the brief is live leaves the chain as it is."""
        root = self.first()
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        out = self.handle(root['dispatch_id'])
        chain = fallback.Chain.of(registry.inspect(root['dispatch_id']), judge=self.judge, host=self.host)
        failed = registry.inspect(root['dispatch_id'])
        stale = chain.run(fallback.failure_of(failed), failed)
        self.assertEqual((stale['action'], stale['dispatch_id']), ('relaunched', out['dispatch_id']))
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['current']), ('relaunched', out['dispatch_id']))
        self.assert_one_worker()

    def test_the_same_bundle_retry_is_a_routed_start_for_the_audit(self):
        """Review F3: the retry writes a plan line with its choice, so router_audit.find_unrouted_spawns accepts it."""
        root = self.transient_root(task=dict(self.task, can_wait_hours=1), minutes=16)
        self.handle(root['dispatch_id'])
        with self.at(minutes=17):
            out = self.handle(root['dispatch_id'])
            spawned = registry.utcnow()
        self.assertEqual(out['action'], 'relaunched')
        routed = [e for e in self.ledger if e.get('stage') in ('plan', 'record')]
        self.assertEqual(len(routed), 1)
        self.assertEqual((routed[0]['choice'], routed[0]['decision_id'], routed[0]['root_dispatch_id']),
                         (root['bundle']['bundle_id'], registry.inspect(out['dispatch_id'])['decision_id'],
                          root['dispatch_id']))
        # Through the audit's own ledger reader.
        ledger = self.root / 'audit-ledger.jsonl'
        ledger.write_bytes(''.join(json.dumps(e) + '\n' for e in self.ledger).encode('utf-8'))
        rows = router_audit.from_ledger(ledger)
        spawns = [{'at': spawned, 'label': 'same-bundle retry'}]
        self.assertEqual(router_audit.find_unrouted_spawns(spawns, rows), [])
        self.assertEqual(len(router_audit.find_unrouted_spawns(spawns, [r for r in rows if r['stage'] == 'fallback'])), 1)

    def test_a_task_input_failure_is_not_relaunched(self):
        self.scopes = ['task_input']
        root = self.first()
        self.host.end(root['dispatch_id'], 'error', terminal={
            'reason': 'error', 'failure_code': 'provider_turn_failed', 'provider_message': 'prompt is too long'})
        out = self.handle(root['dispatch_id'])
        self.assertEqual((out['action'], out['result'], out['covered']), ('final', 'declined', False))
        self.assertEqual(len(self.host.creates), 1)
        public = flow.public(registry.inspect(root['dispatch_id']))
        self.assertEqual(public['next_action'], flow.TASK_INPUT_NEXT + flow.WORKER_ERROR_TEXT.format(error='prompt is too long'))
        self.assertEqual(public['worker_error'], 'prompt is too long')

    # A worker error an observation sees first is read by the classifier like the watcher's (snapshot_turn_failure).

    def observed_error(self, message='Error: the test suite failed in tests/test_parser.py'):
        root = self.first()
        self.host.end(root['dispatch_id'], 'error', lastError=message)     # no watcher terminal: observe reads it
        return root, self.handle(root['dispatch_id'])

    def test_a_worker_error_the_classifier_reads_as_the_works_own_ends_the_chain_and_carries_it(self):
        """Rule (provider_retry.WORKER_CODES, failure_of): a turn failure the classifier scopes `none` is the worker's
        own, so no other candidate is tried and the one answer carries the worker's error."""
        self.scopes = ['none']
        root, out = self.observed_error()
        self.assertEqual((out['action'], out['result'], out['covered']), ('final', 'declined', False))
        self.assertEqual(len(self.host.creates), 1)
        row = registry.inspect(root['dispatch_id'])
        self.assertEqual((row['state'], row['failure_code']), ('failed', 'worker_turn_failed'))
        public = flow.public_chain(row)
        self.assertTrue(public['next_action'].startswith(flow.WORKER_TURN_FAILED_NEXT))
        self.assertIn('the test suite failed in tests/test_parser.py', public['next_action'])
        self.assertEqual(public['worker_error'], 'Error: the test suite failed in tests/test_parser.py')

    def test_the_workers_error_is_kept_by_the_watchers_rule_on_the_snapshot_path(self):
        """Round 2 review A: the worker's own error text an observation reads is kept and forwarded by the one rule the
        watcher applies (provider_retry.turn_error_text: stripped, cut at MESSAGE_LIMIT), not a second limit."""
        self.scopes = ['none']
        message = '  Error: ' + 'x' * 900 + '  '
        root, out = self.observed_error(message)
        public = flow.public_chain(registry.inspect(root['dispatch_id']))
        self.assertEqual(public['worker_error'], provider_retry.turn_error_text(message))
        self.assertEqual(len(public['worker_error']), len(message.strip()))

    def test_a_worker_error_the_classifier_reads_as_the_providers_moves_to_the_next_candidate(self):
        self.scopes = ['model']
        root, out = self.observed_error('429 model overloaded, try again later')
        self.assertEqual(out['action'], 'relaunched', out.get('text'))
        self.assertEqual(len(self.host.creates), 2)
        self.assertEqual(registry.inspect(root['dispatch_id'])['failure_code'], 'provider_turn_failed')

    def test_a_worker_error_the_classifier_cannot_read_is_relaunched_as_before(self):
        """Rule: with no classifier answer (scope `unknown`), or no error text at all, the next candidate is tried."""
        root, out = self.observed_error()                                    # no scope queued: unknown
        self.assertEqual(out['action'], 'relaunched', out.get('text'))
        self.assertEqual(len(self.host.creates), 2)
        self.assertNotIn('worker_error', flow.public(registry.inspect(root['dispatch_id'])))

    def test_a_worker_error_with_no_text_is_relaunched_as_before(self):
        self.scopes = ['none']
        root, out = self.observed_error(message='')
        self.assertEqual(out['action'], 'relaunched', out.get('text'))
        self.assertIsNone(registry.inspect(root['dispatch_id']).get('failure_code'))
        self.assertEqual(self.scopes, ['none'])                              # the classifier was not asked

    def ends_with_a_reason_when_recorded_earlier(self, edit):
        import json
        import sqlite3
        from contextlib import closing
        root = self.first()
        with closing(sqlite3.connect(registry.db_path(), isolation_level=None)) as connection:
            doc = json.loads(connection.execute('SELECT data FROM decisions WHERE id=?', (root['decision_id'],)).fetchone()[0])
            edit(doc['allocation_task'])
            connection.execute('UPDATE decisions SET data=? WHERE id=?', (json.dumps(doc), root['decision_id']))
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error', 'status': 'error'})
        out = fallback.handle(root['dispatch_id'], judge=route_judge, host=self.host)
        self.assertEqual((out['action'], out['result']), ('final', 'older_profile'))
        self.assertIn('earlier version', out['text'])
        self.assertIn('roll the brief again', out['text'])
        self.assertEqual(len(self.host.creates), 1)                       # nothing was relaunched
        self.assertEqual(registry.inspect(root['dispatch_id'])['fallback']['state'], 'final')
        return registry.decision_record(root['decision_id'])['allocation_task']

    def test_a_dispatch_routed_before_an_update_ends_with_a_reason_when_it_fails_after_it(self):
        # Rule (dispatch_registry.earlier_version): a decision recorded under an older task profile is not
        # reinterpreted, so its failure is not replanned (the real judge refuses that profile) and not retried: the
        # chain ends with a reason and nothing is relaunched.
        old = {**profile(), 'realtime_loop': False,
               'axis_demand': {'divergence': 1, 'global_rigor': 2, 'module_precision': 2, 'vision_tools': 0}}
        stored = self.ends_with_a_reason_when_recorded_earlier(lambda task: task.update(profile=old))
        self.assertEqual(old, stored['profile'])

    def test_a_dispatch_whose_stored_task_carries_force_fast_ends_with_a_reason_when_it_fails(self):
        # Rule (dispatch_registry.earlier_version): v0.7.0's `record --fast` wrote force_fast into the stored task;
        # that shape is an earlier version's, so the chain ends with the same reason, not invalid_task_input.
        stored = self.ends_with_a_reason_when_recorded_earlier(lambda task: task.update(force_fast=True))
        self.assertTrue(stored['force_fast'])

    def owner_named(self, task=None):
        decision_bundle = dict(self.bundle)
        result = {'stage': 'record', 'choice': 'constraint', 'constraint': 'user_override', 'basis': 'use sonnet',
                  'selected_bundle': decision_bundle, 'decision': {'action': 'start'}}
        return dict(result, **registry.issue(result, task or self.task, self.now))

    def test_an_owner_named_model_is_never_switched(self):
        self.scopes = ['model']
        root = self.first(decision=self.owner_named())
        self.host.end(root['dispatch_id'], 'error', terminal={
            'reason': 'error', 'failure_code': 'provider_turn_failed', 'provider_message': 'model overloaded'})
        out = self.handle(root['dispatch_id'])
        self.assertEqual((out['action'], out['result']), ('final', 'owner_named'))
        self.assertEqual(len(self.host.creates), 1)
        self.assertIn('roll it without --model', out['text'])
        self.assertEqual(flow.public(registry.inspect(root['dispatch_id']))['next_action'], out['text'])

    def test_an_owner_named_model_is_retried_after_a_transient_failure_when_the_task_can_wait(self):
        task = dict(self.task, can_wait_hours=1)
        root = self.transient_root(task=task, decision=self.owner_named(task))
        self.assertEqual(self.handle(root['dispatch_id'])['action'], 'scheduled')
        with self.at(minutes=3):
            out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertEqual(self.host.creates, ['claude-lab/claude-sonnet-5-5'] * 2)
        routed = [e for e in self.ledger if e.get('stage') in ('plan', 'record')]
        self.assertEqual([(e['stage'], e['constraint']) for e in routed], [('record', 'user_override')])

    def test_an_owner_named_retry_keeps_what_its_record_decided(self):
        # Rule (task_intake.ROUTE_FIELDS): a named record's route fields (here: launched unclassified, priced from its
        # hours) stay with its transient retry, which is the same named route.
        task = dict(self.task, can_wait_hours=1)
        result = {'stage': 'record', 'choice': 'constraint', 'constraint': 'user_override', 'basis': 'use sonnet',
                  'selected_bundle': dict(self.bundle), 'decision': {'action': 'start'}}
        decision = dict(result, **registry.issue(result, task, self.now, route={'unclassified': 'fixture'}))
        root = self.transient_root(task=task, decision=decision)
        self.assertEqual(self.handle(root['dispatch_id'])['action'], 'scheduled')
        with self.at(minutes=3):
            self.assertEqual(self.handle(root['dispatch_id'])['action'], 'relaunched')
        retried = [e['decision_id'] for e in self.ledger if e.get('source') == 'fallback_retry']
        self.assertEqual(len(retried), 1)
        self.assertEqual(registry.decision_record(retried[0]).get('route'), {'unclassified': 'fixture'})

    def test_an_owner_named_model_waits_only_what_the_task_allows(self):
        """Review F2: the owner's model follows can_wait_hours too; a backoff past it reports to the parent."""
        for name, task, minutes in (('no wait given', dict(self.task, task_id='owner-a'), None),
                                    ('backoff past the wait', dict(self.task, task_id='owner-b', can_wait_hours=1), 120)):
            with self.subTest(name):
                creates = len(self.host.creates)
                root = self.transient_root(task=task, decision=self.owner_named(task), minutes=minutes)
                out = self.handle(root['dispatch_id'])
                self.assertEqual((out['action'], out['result']), ('final', 'owner_named'))
                self.assertEqual(len(self.host.creates), creates + 1)
                fb = registry.inspect(root['dispatch_id'])['fallback']
                self.assertEqual(fb['state'], 'final')
                self.assertNotIn('until', fb)
                self.assertIn('roll it without --model', out['text'])

    def test_when_every_candidate_is_held_the_dispatch_ends_blocked_with_each_attempt(self):
        refusal = {'failure_class': 'unknown', 'failure_code': 'INVALID_MODEL'}
        self.host.refuse = type('Every', (dict,), {'__contains__': lambda s, k: True, '__getitem__': lambda s, k: refusal})()
        root = self.first()
        out = self.handle(root['dispatch_id'])
        self.assertEqual((out['action'], out['result']), ('final', 'blocked'))
        self.assertEqual(len(out['attempts']), len(self.host.creates))
        self.assertEqual(len(set(self.host.creates)), len(self.host.creates))   # each bundle at most once
        self.assertGreater(len(self.host.creates), 1)
        self.assertIn('every startable candidate', out['text'])
        for line in out['lines']:
            self.assertIn('INVALID_MODEL', line)
        # The parent's dispatch says so too, and nothing holds a lease.
        self.assertEqual(flow.public(registry.inspect(root['dispatch_id']))['next_action'], out['text'])
        self.assertEqual(registry.reservations(), [])
        self.assertEqual(self.fallback_lines()[-1]['action'], 'blocked')

    # A next attempt that cannot be made ends in one answer that says why (Chain.next_attempt).

    def worker_error_root(self, task=None, decision=None):
        root = self.first(task=task, decision=decision)
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        return root

    def test_a_defect_in_the_replan_is_named_as_one_with_its_stack_frames_and_keeps_the_attempts_cause(self):
        """Rule: the parent's text is the one route_judge.failure_cause allows, a defect of AgentCompass (an exception
        type outside its refusals, the local state and the host) is said to be one, the ledger line keeps its kind, its
        class and its stack frames (never its message: dispatch_fallback.frames), and the failed attempt keeps its own
        cause."""
        root = self.worker_error_root()

        class Broken:
            @staticmethod
            def plan_task(task, args, unavailable=None):
                return None.missing
        out = fallback.handle(root['dispatch_id'], judge=Broken, host=self.host)
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        self.assertIn('internal error of its own', out['text'])
        self.assertIn('attempt 1: claude-sonnet-5-5 on claude-lab failed: worker_error', out['text'])
        self.assertNotIn("'NoneType' object", out['text'])          # failure_cause names the type only
        line = self.fallback_lines()[-1]
        self.assertEqual((line['action'], line['error_kind']), ('not_relaunched', 'internal'))
        self.assertEqual(line['error_type'], 'AttributeError')
        self.assertIn('plan_task', line['traceback'])                  # the frame that raised
        self.assertNotIn("'NoneType' object", json.dumps(line))
        self.assertEqual(line['failed']['failure'], 'worker_error')
        self.assertEqual(line['next_step']['failure'], 'replan_failed')

    def test_an_exceptions_message_reaches_neither_the_parent_nor_the_ledger(self):
        """Round 2 review A: one rule for what is written down about a failure, wherever it goes: the stack frames, the
        exception class and only the text route_judge.failure_cause allows out. A message carrying a token or a user's
        words (here a re-plan whose exception quotes them) is in no answer and no ledger line."""
        root = self.worker_error_root()

        class Leaky:
            @staticmethod
            def plan_task(task, args, unavailable=None):
                raise RuntimeError('host replied: key FIXTURE-TOKEN-VALUE, user said: private words')
        out = fallback.handle(root['dispatch_id'], judge=Leaky, host=self.host)
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        written = json.dumps([out, flow.public_chain(registry.inspect(root['dispatch_id'])), self.fallback_lines()])
        for secret in ('FIXTURE-TOKEN-VALUE', 'private words'):
            self.assertNotIn(secret, written)
        line = self.fallback_lines()[-1]
        # A RuntimeError carries a host's answer: the environment's, its text withheld by failure_cause.
        self.assertEqual((line['error_kind'], line['error_type']), ('environment', 'RuntimeError'))
        self.assertIn('in plan_task', line['traceback'])
        self.assertEqual(line['error'], route_judge.withheld_cause(RuntimeError()))

    def test_a_value_error_no_rule_raised_is_a_defect_not_a_refusal(self):
        """Round 2 review C: a refusal is raised on purpose (dispatch_registry.Refused and the owners' refusal types);
        a ValueError Python raises (an unpacking) is a defect of AgentCompass (`internal`), its text withheld."""
        root = self.worker_error_root()

        class Unpacking:
            @staticmethod
            def plan_task(task, args, unavailable=None):
                planned, = (1, 2)
                return planned
        out = fallback.handle(root['dispatch_id'], judge=Unpacking, host=self.host)
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        self.assertIn('internal error of its own', out['text'])
        self.assertNotIn('too many values to unpack', out['text'])
        line = self.fallback_lines()[-1]
        self.assertEqual((line['error_kind'], line['error_type']), ('internal', 'ValueError'))
        # A registry refusal is still one, in the rule's own words.
        self.assertIsInstance(registry.CapacityChanged('x'), registry.Refused)
        cause = fallback.Chain.cause_of(registry.Refused('the workspace has a writer'))
        self.assertEqual(cause, {'kind': 'refused', 'text': route_judge.failure_cause(registry.Refused('the workspace has a writer'))})

    def test_a_refusal_of_the_local_state_reaches_the_parent_with_its_fix(self):
        """Round 3 review R3-2: a local-state file the re-plan reads and this version refuses on purpose
        (state_upgrade.StateIssue, here the real accounts_override.load_override on a schema it has no upgrade step
        for) reaches the parent as a refusal with the file and its fix, not as a defect with its text withheld."""
        root = self.worker_error_root()
        override = self.root / 'accounts-override.json'
        override.write_text(json.dumps({'schema_version': 999}), encoding='utf-8')
        import accounts_override

        class Override:
            @staticmethod
            def plan_task(task, args, unavailable=None):
                accounts_override.load_override(override)
        out = fallback.handle(root['dispatch_id'], judge=Override, host=self.host)
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        self.assertIn('schema_version 999 has no upgrade step', out['text'])
        self.assertIn(f'Fix: delete "{override}"', out['text'])
        self.assertNotIn('internal error of its own', out['text'])
        self.assertEqual(self.fallback_lines()[-1]['error_kind'], 'refused')
        # The other owners the re-plan reaches refuse with the same type.
        import provider_inventory
        import state_upgrade
        broken = self.root / 'provider-inventory.json'
        broken.write_text('{not json', encoding='utf-8')
        with self.assertRaises(state_upgrade.StateIssue):
            provider_inventory.previous_inventory(broken)

    def test_a_refusal_of_the_local_state_reads_as_one_in_the_owners_words(self):
        root = self.worker_error_root()
        real_prepare = registry.prepare
        error = OSError(13, 'Permission denied', 'D:/work/state/dispatch.sqlite3')

        def locked(*a, **kw):
            if kw.get('attempt'):
                raise error
            return real_prepare(*a, **kw)
        with patch.object(registry, 'prepare', side_effect=locked):
            out = self.handle(root['dispatch_id'])
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        self.assertIn(route_judge.failure_cause(error), out['text'])
        self.assertNotIn('internal error', out['text'])
        self.assertEqual(self.fallback_lines()[-1]['error_kind'], 'environment')

    def test_a_stored_task_with_no_profile_is_not_replanned_and_says_so(self):
        """Rule: a re-plan that answers an error is not "no candidate left": the dispatch ends not_relaunched with the
        re-plan's own reason, after one attempt."""
        task = {k: v for k, v in self.task.items() if k != 'profile'}
        root = self.worker_error_root(task=task, decision=self.issue(task=task))
        out = fallback.handle(root['dispatch_id'], judge=route_judge, host=self.host)
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'), out.get('text'))
        self.assertNotIn('every startable candidate', out['text'])
        self.assertIn('not_replanned', out['text'])
        self.assertIn('scorer_unavailable', out['text'])
        self.assertEqual(len(self.host.creates), 1)

    def test_a_due_retry_that_cannot_be_issued_ends_in_one_answer_not_a_past_promise(self):
        import sqlite3
        task = dict(self.task, can_wait_hours=1)
        root = self.transient_root(task=task)
        self.assertEqual(self.handle(root['dispatch_id'])['action'], 'scheduled')
        with self.at(minutes=3), patch.object(registry, 'issue', side_effect=sqlite3.OperationalError('database is locked')):
            out = self.handle(root['dispatch_id'])
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        self.assertIn('retry_failed', out['text'])
        public = flow.public_chain(registry.inspect(root['dispatch_id']))
        self.assertNotIn('by itself', public['next_action'])
        self.assertEqual(len(self.host.creates), 1)

    def rule_forced(self, task=None):
        result = {'stage': 'record', 'choice': 'constraint', 'constraint': 'hard_constraint',
                  'basis': 'project rule: this repository is edited by sonnet', 'selected_bundle': dict(self.bundle),
                  'decision': {'action': 'start'}}
        return dict(result, **registry.issue(result, task or self.task, self.now))

    def test_a_route_a_project_rule_forces_is_never_moved_to_another_model(self):
        """Rule (FORCED_ROUTES): a hard_constraint route, like the owner's model, is retried only as the rule allows and
        otherwise ends saying the forced route could not start."""
        self.scopes = ['model']
        root = self.first(decision=self.rule_forced())
        self.host.end(root['dispatch_id'], 'error', terminal={
            'reason': 'error', 'failure_code': 'provider_turn_failed', 'provider_message': 'model overloaded'})
        out = self.handle(root['dispatch_id'])
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        self.assertEqual(len(self.host.creates), 1)
        self.assertIn('forced (hard_constraint: project rule', out['text'])
        self.assertIn('could not start', out['text'])

    def test_a_route_a_project_rule_forces_is_retried_on_itself_after_a_transient_failure(self):
        task = dict(self.task, can_wait_hours=1)
        root = self.transient_root(task=task, decision=self.rule_forced(task))
        self.assertEqual(self.handle(root['dispatch_id'])['action'], 'scheduled')
        with self.at(minutes=3):
            out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertEqual(self.host.creates, ['claude-lab/claude-sonnet-5-5'] * 2)
        routed = [e for e in self.ledger if e.get('stage') in ('plan', 'record')]
        self.assertEqual([(e['stage'], e['constraint']) for e in routed], [('record', 'hard_constraint')])

    def test_every_error_roll_answers_for_a_final_outcome_is_in_the_roll_reference(self):
        import roll
        text = (Path(__file__).resolve().parents[1] / 'references' / 'roll-reference.md').read_text(encoding='utf-8')
        self.assertEqual(len(roll.FALLBACK_ERRORS), len(fallback.NOTHING_RUNS))
        for code in roll.FALLBACK_ERRORS:
            self.assertIn(f'`{code}`', text)

    # One worker per brief, observation, bookkeeping.

    def test_a_second_observer_reads_the_first_ones_outcome_instead_of_relaunching(self):
        root = self.first()
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        out = self.handle(root['dispatch_id'])
        again = self.handle(root['dispatch_id'])
        self.assertEqual(again['dispatch_id'], out['dispatch_id'])
        self.assertEqual(len(self.host.creates), 2)

    def test_while_waiting_out_a_backoff_the_brief_counts_as_dispatched(self):
        task = dict(self.task, can_wait_hours=1)
        root = self.transient_root(task=task)
        self.assertEqual(self.handle(root['dispatch_id'])['action'], 'scheduled')
        # Mid-wait, the parent rolls the same brief again: it gets this dispatch back and cannot prepare another.
        self.assertEqual([r['dispatch_id'] for r in registry.pending()], [root['dispatch_id']])
        public = flow.public(registry.inspect(root['dispatch_id']))
        self.assertIn('retries the same model by itself', public['next_action'])
        planned = allocator.plan(dict(task), self.usage, registry.utcnow(), accounts=self.accounts)
        decision = registry.issue(planned, task)
        with self.assertRaisesRegex(ValueError, 'already has an active dispatch'):
            registry.prepare(decision['decision_id'], task, self.launch, self.usage, accounts=self.accounts)
        with self.at(minutes=3):
            self.assertEqual(self.handle(root['dispatch_id'])['action'], 'relaunched')

    def test_observing_the_parents_dispatch_follows_the_current_attempt(self):
        root = self.first()
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        out = self.handle(root['dispatch_id'])
        seen = flow.launch(root['dispatch_id'], 'parent-a', host=self.host, watch_seconds=0)
        self.assertEqual(seen['dispatch_id'], out['dispatch_id'])
        self.assertEqual((seen['root_dispatch_id'], seen['attempt']), (root['dispatch_id'], 2))
        self.assertEqual(seen['state'], 'started')
        self.assertEqual(len(seen['attempts']), 2)
        self.assertEqual(registry.current_attempt(root['dispatch_id']), out['dispatch_id'])
        # The failed row itself points at the relaunch.
        self.assertIn(out['dispatch_id'], flow.public(registry.inspect(root['dispatch_id']))['next_action'])

    def test_each_attempt_is_written_to_the_ledger_and_registry_of_the_same_dispatch(self):
        self.host.refuse['claude-lab/claude-sonnet-5-5'] = {'failure_class': 'model_unavailable'}
        root = self.first()
        out = self.handle(root['dispatch_id'])
        lines = self.fallback_lines()
        self.assertEqual(len(lines), 1)
        self.assertEqual((lines[0]['action'], lines[0]['root_dispatch_id'], lines[0]['attempt']),
                         ('switch', root['dispatch_id'], 1))
        self.assertEqual(lines[0]['failed']['failure'], 'create_failed:model_unavailable')
        self.assertEqual(lines[0]['next']['decision_id'], registry.inspect(out['dispatch_id'])['decision_id'])
        attempts = registry.inspect(root['dispatch_id'])['fallback']['attempts']
        self.assertEqual([a['dispatch_id'] for a in attempts], [root['dispatch_id'], out['dispatch_id']])
        self.assertEqual(attempts[0]['state'], 'create_failed')
        # The new attempt's watcher names the parent's dispatch and the attempts.
        note = json.loads((self.root / 'paseo-notifications' / f"{out['dispatch_id']}.json.attempts").read_bytes())
        self.assertEqual(note['root_dispatch_id'], root['dispatch_id'])
        self.assertEqual(len(note['lines']), 2)

    def test_turned_off_it_relaunches_nothing_and_the_texts_say_roll_again(self):
        root = self.first()
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        with patch.dict(os.environ, {'AGENTCOMPASS_ROLL_FALLBACK': 'off'}):
            self.assertEqual(self.handle(root['dispatch_id'])['action'], 'disabled')
            flow.reconcile(root['dispatch_id'], host=self.host)
            self.assertIn('before rolling again', flow.public(registry.inspect(root['dispatch_id']))['next_action'])
            self.assertNotIn('AGENTCOMPASS_FALLBACK_ARGV', flow.bridge_env())
        self.assertIn('fallback', json.loads(flow.bridge_env()['AGENTCOMPASS_FALLBACK_ARGV']))

    def test_parent_followup_retires_scheduled_retry_and_due_calls_start_nothing(self):
        root, _ = self.scheduled_root(minutes=16)
        self.assertEqual([r['dispatch_id'] for r in registry.pending()], [root['dispatch_id']])
        # The parent sends a follow-up prompt to the failed worker; it runs again before the backoff is over.
        self.host.end(root['dispatch_id'], 'running', attentionReason='error')   # Paseo keeps the stale flag
        flow.reconcile(host=self.host)
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['outcome']['result'], fb['current']),
                         ('final', 'resumed', root['dispatch_id']))
        self.assertEqual(fb['handled'][root['dispatch_id']]['result'], 'resumed')
        self.assertFalse({'until', 'retry_at', 'retry_of'} & set(fb))
        self.assertEqual(registry.pending(), [])
        self.assertEqual(registry.current_attempt(root['dispatch_id']), root['dispatch_id'])
        self.assertEqual(flow.public(registry.inspect(root['dispatch_id']))['next_action'], registry.RESUMED_NEXT)
        # Even once the resumed worker finishes and the scheduled retry time passes, nothing starts.
        self.host.end(root['dispatch_id'], 'idle', attentionReason='finished')
        with self.at(minutes=17):
            self.assertIsNone(fallback.resume_due(root['dispatch_id'], host=self.host))
            watcher = self.handle(root['dispatch_id'])
            seen = self.observe_old(root['dispatch_id'])
        self.assertEqual((watcher['action'], watcher['result']), ('final', 'resumed'))
        self.assertEqual(seen['dispatch_id'], root['dispatch_id'])
        self.assertEqual(len(self.host.creates), 1)
        self.assertFalse(any('switched to' in line for line in watcher.get('lines') or ()))

    def test_parent_followup_retires_pending_switch_and_restores_current_attempt(self):
        task = dict(self.task, can_wait_hours=1)
        root = self.first(task=task)
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        flow.reconcile(root['dispatch_id'], host=self.host)
        second = self.handle(root['dispatch_id'])
        self.assertEqual(second['action'], 'relaunched')
        self.assertEqual(registry.current_attempt(root['dispatch_id']), second['dispatch_id'])
        # Attempt 2 fails and its switch is claimed/pending while the parent resumes attempt 1.
        self.host.end(second['dispatch_id'], 'error', terminal={'reason': 'error'})
        flow.reconcile(second['dispatch_id'], host=self.host)
        chain = fallback.Chain.of(registry.inspect(root['dispatch_id']), judge=self.judge, host=self.host)
        chain.claim(second['dispatch_id'])
        self.assertEqual([r['dispatch_id'] for r in registry.pending()], [root['dispatch_id']])
        self.host.end(root['dispatch_id'], 'running', attentionReason='error')   # Paseo keeps the stale flag
        flow.reconcile(host=self.host)
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertEqual((fb['state'], fb['outcome']['result'], fb['current']),
                         ('final', 'resumed', root['dispatch_id']))
        self.assertEqual(fb['handled'][second['dispatch_id']]['result'], 'resumed')
        self.assertEqual(registry.current_attempt(root['dispatch_id']), root['dispatch_id'])
        self.assertEqual(self.handle(second['dispatch_id'])['result'], 'resumed')
        self.assertEqual(len(self.host.creates), 2)

    def test_failed_attempt_running_again_when_handle_runs_does_not_relaunch(self):
        # 2026-11-11 incident: worker ends in error (failure_code=None -> worker_error/switch), parent resumes it,
        # then fallback handle runs. Neither before replan nor right before launch may a second worker start.
        for mode in ('already_running', 'resumed_during_replan'):
            with self.subTest(mode=mode):
                creates_before = len(self.host.creates)
                root = self.first(task=dict(self.task, task_id=f'resumed-{mode}'))
                self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
                flow.reconcile(root['dispatch_id'], host=self.host)
                self.assertEqual(registry.inspect(root['dispatch_id'])['state'], 'failed')
                if mode == 'already_running':
                    self.host.end(root['dispatch_id'], 'running', attentionReason='error')   # Paseo keeps the stale flag
                    out = self.handle(root['dispatch_id'])
                else:
                    real_plan = self.judge.plan_task

                    def resume_then_plan(task, args, unavailable=None):
                        self.host.end(root['dispatch_id'], 'running', attentionReason='error')   # Paseo keeps the stale flag
                        return real_plan(task, args, unavailable)
                    with patch.object(self.judge, 'plan_task', side_effect=resume_then_plan):
                        out = self.handle(root['dispatch_id'])
                self.assertEqual((out['action'], out['result']), ('final', 'resumed'))
                self.assertEqual(len(self.host.creates), creates_before + 1)
                self.assertEqual(registry.current_attempt(root['dispatch_id']), root['dispatch_id'])
                self.assertFalse(any('switched to' in line for line in out['lines']), out['lines'])
                notes = self.root / 'paseo-notifications'
                self.assertFalse(list(notes.glob('*.attempts')) if notes.exists() else [])
                fb = registry.inspect(root['dispatch_id'])['fallback']
                self.assertEqual((fb['state'], fb['outcome']['result'], fb['current']),
                                 ('final', 'resumed', root['dispatch_id']))
                self.assertFalse(any('switched to' in line for line in fb['lines']), fb['lines'])
                self.assertEqual(self.fallback_lines()[-1]['action'], 'resumed')


    # A step that stopped: its claim ends with it, not with RUN_LEASE (experience run, round 3).

    @staticmethod
    def ended_process():
        """The identity of a process that ran and exited, as a claim it took names its holder."""
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        child.wait()
        return {'pid': child.pid, 'started': 1}

    def killed_step(self, root, relaunching=False, holder=None):
        """The watcher's fallback step took the attempt's claim and its process was killed outright: nothing of it ran
        past the claim (or, `relaunching`, past preparing the next attempt), and no handler of its own ran. `holder`:
        the identity its claim names (default: a process that ran and exited)."""
        class Killed(BaseException):
            pass

        def die(*_, **__):
            raise Killed()
        with patch.object(registry, 'process_identity', return_value=holder or self.ended_process()), \
                patch.object(fallback.Chain, 'release', lambda *_: None), \
                patch.object(*((flow, 'launch') if relaunching else (fallback.Chain, 'run')), die), \
                self.assertRaises(Killed):
            self.handle(root['dispatch_id'])

    def failed_root(self, task=None):
        root = self.first(task=task)
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        return root

    def test_a_killed_steps_claim_is_taken_over_at_once_and_nothing_promises_a_relaunch_meanwhile(self):
        root = self.failed_root()
        self.killed_step(root)
        row = registry.inspect(root['dispatch_id'])
        self.assertFalse(registry.fallback_guard(row, registry.utcnow()))
        self.assertEqual(flow.public_chain(row)['next_action'], flow.FALLBACK_STOPPED_NEXT)
        # The watcher saw its step end without an answer and runs it once more: it takes the claim over, no lease wait.
        with self.at(minutes=1):
            out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertEqual(len(self.host.creates), 2)
        self.assert_one_worker()

    def test_observing_the_dispatch_takes_over_a_killed_step(self):
        """Nobody's watcher left (the machine restarted): the parent's next_argv observation carries the step."""
        root = self.failed_root()
        self.killed_step(root)
        real_handle = fallback.handle
        with self.at(minutes=1), patch.object(fallback, 'handle', side_effect=lambda dispatch_id, **kw: real_handle(
                dispatch_id, **dict(kw, judge=self.judge))):
            seen = flow.launch(root['dispatch_id'], 'parent-a', host=self.host, watch_seconds=0)
        self.assertEqual(len(self.host.creates), 2)
        self.assertNotEqual(seen['dispatch_id'], root['dispatch_id'])
        self.assertEqual(seen['dispatch_id'], registry.current_attempt(root['dispatch_id']))
        self.assert_one_worker()

    def test_a_step_that_stops_by_an_exception_releases_its_claim(self):
        """roll's own synchronous step or a CLI step interrupted (an error of its own, Ctrl+C): released on the way out."""
        root = self.failed_root()
        with patch.object(self.judge, 'plan_task', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.handle(root['dispatch_id'])
        fb = registry.inspect(root['dispatch_id'])['fallback']
        self.assertFalse(registry.claim_held(fb['handled'][root['dispatch_id']], registry.utcnow()))
        self.assertEqual(flow.public_chain(registry.inspect(root['dispatch_id']))['next_action'], flow.FALLBACK_STOPPED_NEXT)
        out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertEqual(len(self.host.creates), 2)

    def test_taking_over_a_step_killed_while_relaunching_releases_the_attempt_it_only_prepared(self):
        root = self.failed_root()
        self.killed_step(root, relaunching=True)
        fb = registry.inspect(root['dispatch_id'])['fallback']
        prepared = fb['current']
        self.assertEqual((fb['state'], registry.inspect(prepared)['state']), ('relaunching', 'prepared'))
        self.assertFalse(registry.fallback_guard(registry.inspect(root['dispatch_id']), registry.utcnow()))
        self.assertTrue(registry.claim_stopped(fb, registry.utcnow()))
        out = self.handle(root['dispatch_id'])
        self.assertEqual(out['action'], 'relaunched')
        self.assertNotEqual(out['dispatch_id'], prepared)
        self.assertNotEqual(registry.inspect(prepared)['state'], 'prepared')
        self.assertNotIn(prepared, [a.get('dispatch_id') for a in out['attempts']])
        self.assertEqual(len(self.host.creates), 2)
        self.assert_one_worker()

    def test_a_step_that_ends_twice_without_an_answer_closes_the_brief_as_not_relaunched(self):
        root = self.failed_root()
        self.killed_step(root)
        self.killed_step(root)    # the watcher's one retry took the claim over and ended the same way
        out = fallback.abandon(root['dispatch_id'], judge=self.judge, host=self.host)
        self.assertEqual((out['action'], out['result']), ('final', 'not_relaunched'))
        self.assertIn(fallback.STEP_STOPPED_DETAIL, out['text'])
        self.assertIn('nothing is running it', out['text'])
        # Every later reader says the same and starts nothing: a step, an observation, the dispatch's own answer.
        self.assertEqual(self.handle(root['dispatch_id'])['text'], out['text'])
        self.assertIsNone(fallback.resume_due(root['dispatch_id'], host=self.host))
        row = registry.inspect(root['dispatch_id'])
        self.assertEqual(flow.public_chain(row)['next_action'], out['text'])
        self.assertFalse(registry.fallback_guard(row, registry.utcnow()))
        self.assertEqual(len(self.host.creates), 1)

    def test_abandon_leaves_a_claim_a_running_step_holds(self):
        root = self.failed_root()
        fallback.Chain.of(root, judge=self.judge, host=self.host).claim(root['dispatch_id'])   # held by this process
        self.assertEqual(fallback.abandon(root['dispatch_id'], judge=self.judge, host=self.host)['action'], 'pending')
        self.assertEqual(registry.inspect(root['dispatch_id'])['fallback']['handled'][root['dispatch_id']]['action'],
                         'pending')

    def test_abandon_leaves_a_dispatch_that_has_not_failed_unchanged(self):
        """Round 2 review B: abandon closes only an attempt that ended in a failure the fallback covers, decided in the
        transaction that takes the claim (dispatch_registry.unclosable). A dispatch running fine is returned unchanged
        with why, and keeps running: nothing is closed, nothing tells the parent to roll it again."""
        root = self.first()
        out = fallback.abandon(root['dispatch_id'], judge=self.judge, host=self.host)
        self.assertEqual((out['action'], out['state']), ('unchanged', root['state']))
        self.assertIn('has not ended in a failure', out['meaning'])
        row = registry.inspect(root['dispatch_id'])
        self.assertEqual(row['state'], root['state'])
        self.assertNotIn(root['dispatch_id'], (row.get('fallback') or {}).get('handled') or {})
        self.assertNotIn(fallback.STEP_STOPPED_DETAIL, json.dumps(flow.public_chain(row)))
        self.assertEqual(len(self.host.creates), 1)
        # Its own failure later is relaunched as any other: the fallback was not closed by the stray call.
        self.host.end(root['dispatch_id'], 'error', terminal={'reason': 'error'})
        self.assertEqual(self.handle(root['dispatch_id'])['action'], 'relaunched')

    def test_a_live_step_holds_its_claim_past_its_lease(self):
        """Round 2 review E: a claim naming a live holder (pid and start time match) is held however long its step runs;
        no observer takes it over when RUN_LEASE passes. Only a claim naming no holder (older code) ends by time."""
        root = self.failed_root()
        fallback.Chain.of(root, judge=self.judge, host=self.host).claim(root['dispatch_id'])   # held by this process
        with self.at(seconds=registry.RUN_LEASE + 60):
            now = registry.utcnow()
            self.assertEqual(self.handle(root['dispatch_id'])['action'], 'pending')
            self.assertIsNone(fallback.resume_due(root['dispatch_id'], host=self.host))
            row = registry.inspect(root['dispatch_id'])
            self.assertTrue(registry.fallback_guard(row, now))
            self.assertEqual(flow.public_chain(row)['next_action'],
                             flow.FALLBACK_RUNNING_NEXT.format(where=(row['fallback'].get('lines') or ['attempt 1'])[-1]))
        self.assertEqual(len(self.host.creates), 1)
        item = registry.inspect(root['dispatch_id'])['fallback']['handled'][root['dispatch_id']]
        legacy = {k: v for k, v in item.items() if k != 'holder'}
        self.assertTrue(registry.claim_held(legacy, registry.utcnow()))
        self.assertFalse(registry.claim_held(legacy, registry.parse(item['until']) + timedelta(seconds=1)))

    @staticmethod
    def unopenable_pid():
        """A live process this user may not query (Windows: OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) refused with
        access denied), such as a system process; None when none is found."""
        import ctypes
        import psutil
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        for proc in psutil.process_iter(['name']):
            if (proc.info['name'] or '').lower() not in ('lsass.exe', 'csrss.exe', 'smss.exe', 'services.exe', 'wininit.exe'):
                continue
            if not kernel.OpenProcess(0x1000, False, proc.pid) and ctypes.get_last_error() == 5:
                return proc.pid
        return None

    @unittest.skipUnless(os.name == 'nt', 'Windows OpenProcess')
    def test_a_dead_holders_pid_given_to_a_process_this_user_cannot_open_does_not_hold_the_claim(self):
        """Round 3 review R3-1: the holder is a process this same user started, so a claim is held only when a process
        of that pid can be opened by this user and its start time is the recorded one. A dead step whose pid a system
        process now carries (after a restart) holds nothing: its attempt is taken over, not promised forever."""
        pid = self.unopenable_pid()
        if pid is None:
            self.skipTest('no process refuses this user here')
        self.assertIsNone(registry.process_started(pid))
        for delta in ({'hours': 1}, {'hours': 24}, {'days': 30}):
            with self.subTest(**delta):
                creates = len(self.host.creates)
                root = self.failed_root(task=dict(self.task, task_id=f"r3-1-{'-'.join(map(str, delta.values()))}"))
                self.killed_step(root, holder={'pid': pid, 'started': 1})
                with self.at(**delta):
                    item = registry.inspect(root['dispatch_id'])['fallback']['handled'][root['dispatch_id']]
                    self.assertFalse(registry.claim_held(item, registry.utcnow()))
                    self.assertFalse(registry.fallback_guard(registry.inspect(root['dispatch_id']), registry.utcnow()))
                    self.assertEqual(self.handle(root['dispatch_id'])['action'], 'relaunched')
                self.assertEqual(len(self.host.creates), creates + 2)

    def test_a_live_holder_of_this_user_still_holds_and_another_users_pid_does_not(self):
        """E's rule stays: this process, as a holder, reads alive with its own start time. Without /proc, a pid whose
        `kill(pid, 0)` is refused (another user's process) is not a step of this user."""
        me = registry.process_identity()
        self.assertIsNotNone(me['started'])
        self.assertEqual(registry.process_started(me['pid']), me['started'])
        claim = {'action': 'pending', 'at': registry.stamp(registry.utcnow()), 'holder': me,
                 'until': registry.stamp(registry.utcnow() - timedelta(days=30))}
        self.assertTrue(registry.claim_held(claim, registry.utcnow()))

        def refused(*_):
            raise PermissionError(1, 'Operation not permitted')
        no_proc = type('Os', (), {'name': 'posix', 'kill': staticmethod(refused), 'getpid': staticmethod(os.getpid)})
        with patch.object(registry, 'os', no_proc):
            self.assertIsNone(registry.process_started(987654))

    def test_of_processes_racing_for_a_released_claim_exactly_one_takes_it(self):
        """A step runs only once its claim is taken (`handle`): of four processes taking over the same released claim
        on the real SQLite file at once, exactly one takes it and the others read it as being relaunched. Each stays
        alive until all have answered, so the winner's claim is still held while the others read it."""
        root = self.failed_root()
        self.killed_step(root)
        sync = self.root / 'race'
        sync.mkdir()
        script = textwrap.dedent('''
            import json, os, sys, time
            from pathlib import Path
            sys.path.insert(0, sys.argv[1])
            import dispatch_fallback, dispatch_registry
            root_id, sync, me = sys.argv[2], Path(sys.argv[3]), sys.argv[4]
            chain = dispatch_fallback.Chain.of(dispatch_registry.inspect(root_id))
            (sync / f'ready-{me}').write_text('')
            while not (sync / 'go').exists():
                time.sleep(0.002)
            try:
                chain.claim(root_id)
                answer = {'took': True, 'pid': os.getpid()}
            except dispatch_fallback.AlreadyHandled as done:
                answer = {'took': False, 'action': done.outcome.get('action')}
            (sync / f'answer-{me}').write_text(json.dumps(answer))
            while not (sync / 'done').exists():
                time.sleep(0.01)
        ''')
        scripts = str(Path(fallback.__file__).resolve().parent)
        children = [subprocess.Popen([sys.executable, '-X', 'utf8', '-c', script, scripts, root['dispatch_id'], str(sync),
                                      str(n)]) for n in range(4)]
        try:
            deadline = time.monotonic() + 120
            while len(list(sync.glob('ready-*'))) < 4 and time.monotonic() < deadline:
                time.sleep(0.02)
            (sync / 'go').write_text('')
            while len(list(sync.glob('answer-*'))) < 4 and time.monotonic() < deadline:
                time.sleep(0.02)
            answers = [json.loads(p.read_text()) for p in sync.glob('answer-*')]
        finally:
            (sync / 'done').write_text('')
            for child in children:
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        self.assertEqual(len(answers), 4, answers)
        winners = [a for a in answers if a['took']]
        self.assertEqual(len(winners), 1, answers)
        self.assertEqual([a['action'] for a in answers if not a['took']], ['pending'] * 3)
        item = registry.inspect(root['dispatch_id'])['fallback']['handled'][root['dispatch_id']]
        self.assertEqual(item['holder']['pid'], winners[0]['pid'])
        self.assertEqual(len(self.host.creates), 1)


class RollFallbackTests(unittest.TestCase):
    """The roll process itself relaunches when the create fails while it waits for it."""
    setUp = FallbackTests.setUp
    setUp_dispatch = test_dispatch.DispatchTests.setUp
    snapshot = test_dispatch.DispatchTests.snapshot
    issue = test_dispatch.DispatchTests.issue
    first = FallbackTests.first
    handle = FallbackTests.handle

    def test_roll_answers_with_the_relaunched_worker_and_its_attempts(self):
        import argparse
        import roll
        self.host.refuse['claude-lab/claude-sonnet-5-5'] = {'failure_class': 'unknown', 'failure_code': 'INVALID_MODEL'}
        brief = self.root / 'task.md'
        brief.write_bytes(b'# Fixture brief\n\nKeep only the current rules.\n')
        args = argparse.Namespace(brief=brief, file=[], apply_to=None, local_only=None, hours=0.5, title=None,
                                  workspace=None, task_id='roll-fallback', wait=0, parent_id=None, parent_host=None,
                                  parent_address=None, provider=None, model=None, effort=None, basis=None,
                                  usage_cache=None, dry_run=False, verbose=False, task=None, new_worker=True,
                                  resume=None, fast=False, keep_whole=None, access=None, allow_concurrent=None,
                                  account=None, can_wait_hours=None)
        test = self

        def plan_stage(document, _):
            first = not document.get('unavailable')
            planned = allocator.plan(document['task'], test.usage, registry.utcnow(),
                                     unavailable=document.get('unavailable') or (), accounts=test.accounts,
                                     route={'pinned_bundle': 'lab_sonnet55_high'} if first else None)
            return dict(planned, **registry.issue(planned, document['task'])), 0

        def plan_task(task, args, unavailable=None):
            return plan_stage({'task': task, 'unavailable': list(unavailable or ())}, args)
        judge = type('Judge', (), {'plan_stage': staticmethod(plan_stage), 'plan_task': staticmethod(plan_task)})
        self.enterContext(test_dispatch.fake_owners(self))
        with patch.dict(os.environ, {'PASEO_AGENT_ID': 'parent-a', 'ROUTER_PARENT_AGENT_ID': ''}), \
                patch.object(roll, 'WORKER_PROMPTS_DIR', self.root / 'worker-prompts'), \
                patch.object(roll, 'RESULTS_DIR', self.root / 'roll-results'), \
                patch.object(flow, 'paseo_reachable', return_value=True), \
                patch.object(flow, 'host_call', self.host):
            result, code = roll.run(args, judge)
        self.assertEqual((result['status'], code), ('launched', 0))
        self.assertNotEqual(result['worker']['model'], 'claude-sonnet-5-5')
        self.assertEqual(len(result['attempts']), 2)
        self.assertIn('INVALID_MODEL', result['attempts'][0])
        self.assertEqual(self.host.creates[0], 'claude-lab/claude-sonnet-5-5')
        self.assertEqual(len(self.host.alive()), 1)
        detail = json.loads(Path(result['details']).read_text(encoding='utf-8'))
        self.assertEqual(detail['worker']['root_dispatch_id'], registry.inspect(detail['worker']['dispatch_id'])['attempt_of'])

    def test_roll_answers_a_scheduled_retry_as_pending_with_its_time(self):
        import roll
        retry_at = registry.stamp(registry.utcnow() + timedelta(minutes=2))
        out, code = roll.fallback_result({}, {'action': 'scheduled', 'retry_at': retry_at, 'root_dispatch_id': 'root-1',
                                              'lines': ['attempt 1: m on a failed: provider_turn_failed']},
                                         {}, 'parent-a', None)
        self.assertEqual((roll.status_of(out, code), code), ('pending', 0))
        answer = roll.compact(out, 'pending', None)
        self.assertEqual(answer['retry_at'], retry_at)
        self.assertIn(retry_at, answer['next_action'])
        self.assertIn('root-1', answer['next_argv'])

    def test_unclassified_or_request_invalid_create_refusal_stops_with_no_cooldown_and_no_fallback(self):
        import argparse
        import launch_failures
        import roll
        cooldown_store = self.root / 'launch-failures.json'
        for idx, refusal in enumerate((
            {'failure_class': 'unknown'},
            {'failure_class': 'request_invalid'},
        )):
            with self.subTest(refusal=refusal):
                self.host.creates.clear()
                self.host.refuse = {'claude-lab/claude-sonnet-5-5': refusal}
                brief = self.root / f'task-{idx}.md'
                brief.write_bytes(b'# Fixture brief\n\nKeep only the current rules.\n')
                args = argparse.Namespace(brief=brief, file=[], apply_to=None, local_only=None, hours=0.5, title=None,
                                          workspace=None, task_id=f'no-walk-{idx}', wait=0, parent_id=None, parent_host=None,
                                          parent_address=None, provider=None, model=None, effort=None, basis=None,
                                          usage_cache=None, dry_run=False, verbose=False, task=None, new_worker=True,
                                          resume=None, fast=False, keep_whole=None, access=None, allow_concurrent=None,
                                          account=None, can_wait_hours=None)
                test = self

                def plan_stage(document, _):
                    planned = allocator.plan(document['task'], test.usage, registry.utcnow(),
                                             unavailable=document.get('unavailable') or (), accounts=test.accounts,
                                             route={'pinned_bundle': 'lab_sonnet55_high'})
                    return dict(planned, **registry.issue(planned, document['task'])), 0
                judge = type('Judge', (), {'plan_stage': staticmethod(plan_stage)})
                self.enterContext(test_dispatch.fake_owners(self))
                with patch.dict(os.environ, {'PASEO_AGENT_ID': 'parent-a', 'ROUTER_PARENT_AGENT_ID': '',
                                             'ROUTER_LAUNCH_FAILURES': str(cooldown_store)}), \
                        patch.object(roll, 'WORKER_PROMPTS_DIR', self.root / 'worker-prompts'), \
                        patch.object(roll, 'RESULTS_DIR', self.root / 'roll-results'), \
                        patch.object(flow, 'paseo_reachable', return_value=True), \
                        patch.object(flow, 'host_call', self.host):
                    result, code = roll.run(args, judge)
                self.assertEqual((result['status'], code), ('error', 1))
                self.assertEqual(len(self.host.creates), 1)
                self.assertEqual(launch_failures.active(path=cooldown_store), {})
                self.assertFalse(cooldown_store.exists())

        # Contrast property: a candidate-scoped refusal DOES record cooldown and fall back to the next candidate.
        self.host.creates.clear()
        self.host.refuse = {'claude-lab/claude-sonnet-5-5': {'failure_class': 'model_unavailable'}}
        with patch.dict(os.environ, {'ROUTER_LAUNCH_FAILURES': str(cooldown_store)}):
            root = self.first(task=dict(self.task, task_id='cand-walk'))
            out = self.handle(root['dispatch_id'])
            self.assertEqual(out['action'], 'relaunched')
            self.assertEqual(len(self.host.creates), 2)
            self.assertTrue(launch_failures.active(path=cooldown_store))



class StoredTaskRelaunchTests(unittest.TestCase):
    """The relaunch plans the task AgentCompass stored for the brief, never a caller's document: the stored task is built
    by the real path (roll -> route_judge.plan_stage with the classifier -> dispatch_registry), and the relaunch is
    planned by the real route_judge. The host refuses the first create (a covered failure), so roll promises a relaunch."""
    setUp = FallbackTests.setUp
    setUp_dispatch = test_dispatch.DispatchTests.setUp
    snapshot = test_dispatch.DispatchTests.snapshot
    issue = test_dispatch.DispatchTests.issue
    at = FallbackTests.at

    def roll_refused_first(self, task_id, *extra, refuse=True, can_wait_hours=None):
        """roll with the real judge; the classifier marks the brief local-only; the first create is refused. `extra`:
        more patches for this roll. Returns (roll's details, exit code, the providers the host was asked to create)."""
        import argparse
        import contextlib
        import roll
        import scorer_client
        creates = []

        def host(request):
            if request['operation'] == 'create':
                creates.append(request['request']['provider'])
                if refuse and len(creates) == 1:
                    return {'ok': False, 'sent': False, 'error': 'host_creation_failed', 'failure_class': 'model_unavailable'}
            return self.host(request)
        brief = self.root / f'{task_id}.md'
        brief.write_bytes(b'# Fixture brief\n\nKeep only the current rules.\n')
        args = argparse.Namespace(brief=brief, file=[], apply_to=None, local_only=None, hours=0.5, title=None,
                                  workspace=None, task_id=task_id, wait=0, parent_id=None, parent_host=None,
                                  parent_address=None, provider=None, model=None, effort=None, basis=None,
                                  usage_cache=None, dry_run=False, verbose=False, task=None, new_worker=True,
                                  resume=None, fast=False, keep_whole=None, access=None, allow_concurrent=None,
                                  account=None, can_wait_hours=can_wait_hours, no_score=False, score=False, demo=False,
                                  score_timeout=5)
        classified = profile(local_only=True, reason='the fixture classifier keeps this work on this machine')
        meta = {'source': 'system_one', 'scorer_type': 'fixture'}
        with contextlib.ExitStack() as stack:
            stack.enter_context(test_dispatch.fake_owners(self))
            stack.enter_context(patch.dict(os.environ, {'PASEO_AGENT_ID': 'parent-a', 'ROUTER_PARENT_AGENT_ID': ''}))
            stack.enter_context(patch.object(scorer_client, 'score_task', return_value=(classified, meta)))
            stack.enter_context(patch.object(roll, 'WORKER_PROMPTS_DIR', self.root / 'worker-prompts'))
            stack.enter_context(patch.object(roll, 'RESULTS_DIR', self.root / 'roll-results'))
            stack.enter_context(patch.object(flow, 'paseo_reachable', return_value=True))
            stack.enter_context(patch.object(flow, 'host_call', host))
            for one in extra:
                stack.enter_context(one)
            result, code = roll.run(args, route_judge)
        return json.loads(Path(result['details']).read_text(encoding='utf-8')), code, creates

    def test_the_other_ways_a_stored_task_relaunches_plan_it_as_stored(self):
        """The watcher's step after a worker error (it claims the failure first: `dispatch --dispatch-action fallback`)
        re-plans the stored task; a transient failure's scheduled retry issues its decision on the stored task itself."""
        for task_id in ('watcher-worker-error', 'transient-retry'):
            with self.subTest(task_id=task_id):
                transient = task_id == 'transient-retry'
                result, code, creates = self.roll_refused_first(task_id, refuse=False,
                                                                can_wait_hours=1 if transient else None)
                self.assertEqual((result['caller_action'], len(creates)), ('already_launched_do_not_respawn', 1))
                root = self.root_of(result)
                self.assertEqual(registry.decision_record(root['decision_id'])['allocation_task']['needs']
                                 .get('local_only_by'), 'classifier')
                terminal = {'reason': 'error'}
                if transient:
                    self.scopes = ['transient']
                    terminal.update(failure_code='provider_turn_failed', provider_message='429 retry later')
                self.host.end(root['dispatch_id'], 'error', terminal=terminal)
                out = fallback.handle(root['dispatch_id'], judge=route_judge, host=self.host)
                if transient:
                    self.assertEqual(out['action'], 'scheduled', out)
                    with self.at(minutes=3):
                        out = fallback.handle(root['dispatch_id'], judge=route_judge, host=self.host)
                self.assertEqual(out['action'], 'relaunched', out)
                new = registry.inspect(out['dispatch_id'])
                self.assertEqual(new['bundle']['bundle_id'] == root['bundle']['bundle_id'], transient)

    def root_of(self, result):
        worker = result['worker']
        return registry.inspect(worker.get('root_dispatch_id') or worker['dispatch_id'])

    def test_a_brief_the_classifier_marked_local_only_is_relaunched_on_the_next_candidate(self):
        result, code, creates = self.roll_refused_first('stored-local-only')
        root = self.root_of(result)
        stored = registry.decision_record(root['decision_id'])['allocation_task']
        self.assertEqual(stored['needs'].get('local_only_by'), 'classifier')
        self.assertEqual((result['caller_action'], code), ('already_launched_do_not_respawn', 0), result.get('next'))
        self.assertEqual(len(creates), 2)
        self.assertEqual(len(result['attempts']), 2)
        self.assertNotIn('not_prepared', ' '.join(result['attempts']))
        self.assertEqual(root['fallback']['state'], 'relaunched')

    def test_a_relaunch_that_cannot_be_made_ends_in_one_answer_that_nothing_runs(self):
        """Whatever stops the next attempt before anything is sent, the promise is answered: roll tells the session the
        brief did not start, and every later observation of the dispatch says the same, never 'relaunches by itself'."""
        real_prepare = registry.prepare

        def failing_prepare(*a, **kw):
            if kw.get('attempt'):   # the relaunch's prepare
                raise RuntimeError('fixture: the relaunch cannot be prepared')
            return real_prepare(*a, **kw)

        def failing_plan(task, args, unavailable=None):
            if unavailable:   # the relaunch's plan: the failed bundle is out of it
                raise RuntimeError('fixture: the relaunch cannot be planned')
            return real_plan(task, args, unavailable)
        for task_id in ('prepare-raises', 'replan-raises'):
            with self.subTest(task_id=task_id):
                if task_id == 'prepare-raises':
                    stop = patch.object(registry, 'prepare', side_effect=failing_prepare)
                else:
                    real_plan = route_judge.plan_task
                    stop = patch.object(route_judge, 'plan_task', side_effect=failing_plan)
                result, code, creates = self.roll_refused_first(task_id, stop)
                self.assertEqual(len(creates), 1)
                self.assertEqual((result['caller_action'], result['error']), ('tell_user', 'relaunch_not_started'),
                                 result.get('next'))
                self.assertIn('did not start', result['next'])
                self.assertNotIn('by itself', result['next'])
                root = self.root_of(result)
                self.assertEqual((root['fallback']['state'], root['fallback']['outcome']['result']),
                                 ('final', 'not_relaunched'))
                observed = flow.public_chain(root)
                self.assertIn('did not start', observed['next_action'])
                self.assertNotIn('by itself', observed['next_action'])
                # The watcher, arriving second, reads the same final outcome, which its notice carries (NOTHING_RUNS).
                again = fallback.handle(root['dispatch_id'], judge=route_judge, host=self.host)
                self.assertEqual((again['action'], again['result']), ('final', 'not_relaunched'))
                self.assertIn(again['result'], fallback.NOTHING_RUNS)

    def test_a_launch_that_fails_after_prepare_releases_it_and_leaves_one_worker_at_most(self):
        """Review finding 1: the relaunch's claim raises after its prepare. One answer (the brief did not start), the
        prepared row released, and following that answer (roll again, then observe the old dispatch) starts one worker."""
        real_claim = registry.claim
        calls = []

        def claim(*a, **kw):
            calls.append(1)
            if len(calls) == 2:      # the relaunch's claim
                raise registry.Refused('fixture: account_disabled_in_magpie at claim')
            return real_claim(*a, **kw)
        result, code, creates = self.roll_refused_first('claim-raises', patch.object(registry, 'claim', side_effect=claim))
        self.assertEqual((result['caller_action'], result['error']), ('tell_user', 'relaunch_not_started'), result.get('next'))
        self.assertIn('did not start', result['next'])
        root = self.root_of(result)
        attempts = [registry.inspect(a['dispatch_id']) for a in root['fallback']['attempts'] if a.get('dispatch_id')]
        self.assertFalse([r for r in attempts if r['state'] == 'prepared'], [r['state'] for r in attempts])
        again, _, more = self.roll_refused_first('claim-raises', refuse=False)
        self.assertEqual(again['caller_action'], 'already_launched_do_not_respawn', again.get('next'))
        before = len(self.host.creates)
        flow.launch(root['dispatch_id'], 'parent-a', host=self.host, watch_seconds=0)
        self.assertEqual(len(self.host.creates), before)
        self.assertLessEqual(len(self.host.alive()), 1)

    def test_the_bridge_sends_the_text_of_every_final_result_after_which_nothing_runs(self):
        """paseo_dispatch_bridge.mjs composeNotice keeps its own copy of NOTHING_RUNS: the two lists are one."""
        import re
        bridge = (Path(__file__).resolve().parents[3] / 'setup' / 'paseo_dispatch_bridge.mjs').read_text(encoding='utf-8')
        found = re.search(r"const NOTHING_RUNS = new Set\(\[([^\]]*)\]\)", bridge)
        self.assertIsNotNone(found)
        self.assertEqual(sorted(re.findall(r"'([a-z_]+)'", found.group(1))), sorted(fallback.NOTHING_RUNS))


if __name__ == '__main__':
    unittest.main()
