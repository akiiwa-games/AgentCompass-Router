"""What a caller of `agentcompass roll` reads when it does not get what it asked for: each answer names the fact that decided it
and a way out that works for that fact (`roll-reference.md`, "Every answer names the fact").

The tests assert rules: the text carries the facts the owning function computed (a window, the hours, a code, the caller's own
path), and offers no way out that cannot work for them. They never assert which route wins.
"""
import argparse
import contextlib
import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import dispatch_flow as flow
import dispatch_registry as registry
import route_judge
import test_dispatch as td
from router_test_support import isolated_router_state


def setUpModule():
    isolated_router_state(unittest.addModuleCleanup, policy=False)


def with_profile(task):
    task['profile'] = td.profile_for(task)
    return task


# What the bridge really answers (setup/paseo_dispatch_bridge.mjs: `main`, `hostFailure`, `execute`; dispatch_flow.host_call for
# the two it makes itself): its `sent`, and the `condition` tokens it merges under a code. A create or a resume that failed
# after the daemon was connected is `sent: true`; one that never reached the daemon is `sent: false`.
CONNECT_CONDITIONS = ('refused', 'timeout', 'authentication', 'credential_unreadable', 'rejected', 'unreachable')
REQUEST_CONDITIONS = ('timeout', 'connection_lost', 'daemon_error', 'invalid_response', 'authentication', 'bridge_error')
SENT_CODES = ('host_request_failed', 'host_timeout', 'invalid_host_response', 'create_outcome_uncertain_reconcile_by_label',
              'resume_outcome_uncertain_reconcile_by_agent')


def bridge_answer(error, condition=None, **fields):
    """The bridge's answer for `error`, with the `sent` its source gives that code."""
    return {'ok': False, 'sent': error in SENT_CODES, 'error': error, **({'condition': condition} if condition else {}), **fields}


def starts_the_daemon(text):
    return 'start the daemon' in text.lower()


def full(res):
    return json.loads(Path(res['details']).read_text(encoding='utf-8')) if res.get('details') else res


class Plain(unittest.TestCase):
    """A roll with a fake judge (the real allocator and the real roll), a fake Paseo host, and isolated state."""
    setUp = td.DispatchTests.setUp
    issue = td.DispatchTests.issue
    prepare = td.DispatchTests.prepare
    snapshot = td.DispatchTests.snapshot
    _setup_paseo_env = td.ContinuationRollTests._setup_paseo_env

    def args(self, brief, **over):
        base = dict(task=None, file=[], apply_to=None, local_only=None, keep_whole=None, hours=0.5, title=None, task_id=None,
                    wait=0, parent_id=None, parent_host=None, parent_address=None, provider=None, account=None, model=None,
                    effort=None, basis=None, usage_cache=None, dry_run=False, verbose=False, fast=False, resume=None,
                    new_worker=False, workspace=None, can_wait_hours=None, access=None)
        base.update(over)
        return argparse.Namespace(brief=brief, **base)

    def brief_file(self, name='b.md', text='# Task\n'):
        path = self.root / name
        path.write_text(text, encoding='utf-8')
        return path

    def run_roll(self, brief=None, host=None, judge=None, env=None, **over):
        import roll
        original = roll.build_task
        _, standard_host, standard_judge, _ = self._setup_paseo_env()
        standard_judge.record_constraint = staticmethod(route_judge.record_constraint)
        self.enterContext(td.fake_owners(self))
        with patch.dict(os.environ, {'PASEO_AGENT_ID': 'parent-a', **(env or {})}), \
                patch.object(roll, 'WORKER_PROMPTS_DIR', self.root / 'wp'), patch.object(roll, 'RESULTS_DIR', self.root / 'rr'), \
                patch.object(flow, 'host_call', host or standard_host), \
                patch.object(roll, 'build_task', lambda *a, **k: with_profile(original(*a, **k))):
            return roll.run(self.args(brief or self.brief_file(), **over), judge or standard_judge)

    def set_remaining(self, five, seven, five_hours=3.0, seven_hours=100.0, account='claude-lab'):
        for window in self.usage['accounts'][account]['windows']:
            remaining, hours = (five, five_hours) if window['id'] == 'five_hour' else (seven, seven_hours)
            window.update(used_pct=100 - remaining, remaining_pct_before_reset=remaining, seconds_until_reset=hours * 3600,
                          resets_at=(self.now + timedelta(hours=hours)).isoformat())

    def make_worker(self, name, doc=None, finish=True, final_status='idle'):
        agents = self.root / '.paseo' / 'agents' / 'workspace'
        agents.mkdir(parents=True, exist_ok=True)
        row = self.prepare(self.issue())
        registry.claim(row['request'], now=self.now)
        registry.observe(row['dispatch_id'], self.snapshot(row, id=name), now=self.now)
        if finish:
            registry.observe(row['dispatch_id'], self.snapshot(row, 'idle', attentionReason='finished', id=name), now=self.now)
        document = {'id': name, 'status': final_status, 'labels': {'router.parent_id': 'parent-a'}}
        document.update(doc or {})
        (agents / f'{name}.json').write_text(json.dumps(document), encoding='utf-8')
        return row


class HandBuilt(unittest.TestCase):
    """A roll whose plan is written out by the test: what roll says about the facts a plan carries."""
    setUp, snapshot = td.DispatchTests.setUp, td.DispatchTests.snapshot
    roll = td.RollTests.roll
    brief = td.RollTests.brief


class BlockedAnswers(Plain):
    """Finding 1: `none` and `defer` say the fact that decided them, first, and a way out that works for it."""

    def test_quota_held_roll_defers_and_names_the_account_the_window_and_the_hours(self):
        # Rule (allocator.plan, roll.defer_next): an account held back only by resetting quota windows defers until the
        # last exhausted window resets; the answer (`caller_action: wait_for_reset`) names that account, that window and
        # that wait in both `reason` and `next_action`, and tells the caller to roll again then.
        for label, five, seven in (('five-hour-empty', 1, 100), ('both-empty', 1, 1)):
            with self.subTest(label):
                five_hours, seven_hours = 3.0, 100.0
                self.set_remaining(five, seven, five_hours=five_hours, seven_hours=seven_hours)
                res, code = self.run_roll(task_id='t-' + label)
                self.assertEqual((code, res['status'], res.get('caller_action')), (2, 'blocked', 'wait_for_reset'))
                exhausted = {'five_hour': five_hours} | ({'seven_day': seven_hours} if seven < 5 else {})
                window = max(exhausted, key=exhausted.get)
                for text in (res['reason'][:160], res['next_action']):
                    for fact in ('claude-lab', window, f'{exhausted[window]:.1f}h'):
                        self.assertIn(fact, text)
                self.assertIn('roll again then', res['next_action'].lower())
                self.setUp()


class BlockedHandBuilt(HandBuilt):
    """Finding 1, from plans the test writes out: what roll says about the facts a plan carries."""

    def test_none_without_a_quota_hold_points_at_the_reason(self):
        decision = dict(stage='plan', choice='none', selected_bundle={}, decision={'action': 'none', 'reason': 'No eligible bundle.'})
        (res, code), _ = HandBuilt.roll(self, decision, task='x')
        self.assertEqual(code, 2)
        self.assertIn('route.reason', res['next_action'])

    def test_defer_names_the_window_that_sets_the_wait_and_only_the_ways_out_that_work(self):
        """Rule (`roll.defer_next`): a deferral launches nothing and names the window and the wait; rolling again without
        `--can-wait-hours` is offered only when the plan found a candidate that can start now (`decision.starts_now`); when a
        model or route was already named, `next_action` omits telling the caller to name a route with `--provider --account
        --model --basis` and tells it not to substitute another model."""
        waiting = {'action': 'defer', 'start_after_hours': 2.5, 'wait_for': {'bundle': 'lab_sonnet55_high', 'account': 'claude-lab',
                                                                              'window': 'five_hour', 'hours': 2.5}}
        for label, plan in (('something starts now', dict(waiting, starts_now='lab_other')), ('nothing starts now', waiting)):
            with self.subTest(label):
                decision = dict(stage='plan', choice='lab_sonnet55_high', selected_bundle=self.bundle, decision=plan)
                (res, code), _ = HandBuilt.roll(self, decision, task='x')
                self.assertEqual((code, res['status']), (2, 'blocked'))
                for fact in ("claude-lab's five_hour window", '2.5h', '--basis', 'yourself'):
                    self.assertIn(fact, res['next_action'])
                self.assertEqual('without --can-wait-hours' in res['next_action'], 'starts_now' in plan)
                self.assertEqual('lab_other' in res['next_action'], 'starts_now' in plan)
                self.setUp()
            with self.subTest('named ' + label):
                named_plan = dict(plan, task={'named_model': 'claude-sonnet-5-5'})
                decision = dict(stage='plan', choice='lab_sonnet55_high', selected_bundle=self.bundle, decision=named_plan)
                (res, code), _ = HandBuilt.roll(self, decision, task='x')
                self.assertEqual((code, res['status'], res.get('caller_action')), (2, 'blocked', 'wait_for_reset'))
                for fact in ("claude-lab's five_hour window", '2.5h', 'yourself', 'do not substitute'):
                    self.assertIn(fact, res['next_action'])
                self.assertNotIn('--basis', res['next_action'])
                self.assertNotIn('name a route', res['next_action'])
                self.assertEqual('without --can-wait-hours' in res['next_action'], 'starts_now' in plan)
                self.assertEqual('lab_other' in res['next_action'], 'starts_now' in plan)
                self.setUp()


class FlagsAreNeverDropped(Plain):
    """Finding 3."""

    def test_basis_alone_is_said_to_name_nothing(self):
        res, _ = self.run_roll(basis='because', task_id='t-basis')
        self.assertTrue(any('--basis' in w and 'ignored' in w for w in res.get('warnings', [])), res)

    def test_fast_that_was_not_applied_says_so_and_why(self):
        res, _ = self.run_roll(fast=True, task_id='t-fast')
        bundle = full(res)['route']['bundle']
        if not bundle.get('fast'):
            self.assertTrue(any(w.startswith('--fast was not applied') for w in res.get('warnings', [])), res)
        else:
            self.assertFalse(any('--fast was not applied' in w for w in res.get('warnings', [])), res)

    def test_files_given_to_a_route_that_is_not_dot_are_said_not_to_be_sent(self):
        # Rule (roll.files_not_sent): only Dot receives --file; any other route's answer says the files were not sent and
        # names the route that did not get them.
        (self.root / 'x.txt').write_text('x', encoding='utf-8')
        res, _ = self.run_roll(file=[self.root / 'x.txt'], task_id='t-file')
        provider = full(res)['route']['bundle']['provider']
        self.assertNotEqual(provider, 'dot')
        self.assertTrue(any('not sent' in w and provider in w for w in res.get('warnings', [])), res)

    def test_a_named_route_that_is_not_dot_is_not_told_it_runs_on_this_machine(self):
        import roll
        out = roll.files_not_sent({}, argparse.Namespace(file=['x'], apply_to=None), 'chatgpt-web')
        self.assertIn('chatgpt-web', out['warnings'][0])
        self.assertNotIn('this machine', out['warnings'][0])


class ExitCodes(Plain):
    """Finding 8: what `roll-reference.md` says about the exit code of each kind of answer is what the code does."""

    def reference_bullet(self):
        text = (Path(__file__).resolve().parents[1] / 'references' / 'roll-reference.md').read_text(encoding='utf-8')
        start = text.index('- A Dot runner that launched nothing')
        return text[start:text.index('\n', start)]

    def test_each_kind_of_answer_exits_as_the_reference_says(self):
        bullet = self.reference_bullet()
        host = LaunchRefusedBeforeAnyWorker.host_answering
        res, code = self.run_roll(host=host(self, bridge_answer('host_connection_failed', 'refused')), task_id='t-x1')
        self.assertEqual((code, res['status']), (1, 'error'))            # refused before any worker existed
        self.assertIn('exits 1', bullet)
        self.setUp()
        text = 'Account claude-lab five_hour is below its existing launch protection (5% remaining floor); refresh quota or replan'
        with patch.object(registry, 'check_launch_quota', side_effect=ValueError(text)):
            res, code = self.run_roll(task_id='t-x2')
        self.assertEqual((code, res['status']), (2, 'blocked'))          # refused before any launch
        self.setUp()
        self.make_worker('agent-x')
        res, code = self.run_roll(self.brief_file('x.md'), resume='agent-x', task_id='t-x3',
                                  host=host(self, bridge_answer('worker_not_idle', 'running')))
        self.assertEqual((code, res['status']), (2, 'blocked'))          # a resume the host refused
        self.assertIn('`blocked` exits 2', bullet)
        self.setUp()
        res, code = self.run_roll(host=host(self, bridge_answer('host_timeout')), task_id='t-x4')
        self.assertEqual((code, res['status']), (0, 'pending'))
        self.assertIn('`pending` result exits 0', bullet)

    def test_the_documents_say_the_plan_fields_are_rounded_up(self):
        base = Path(__file__).resolve().parents[1]
        for name in ('roll-reference.md', 'router-internals.md'):
            text = (base / 'references' / name).read_text(encoding='utf-8')
            at = text.index('decision.start_after_hours')
            sentence = text[at:at + 300]
            self.assertIn('rounded up', sentence, name)


class NamedRouteRefusals(Plain):
    """Finding 4: flags that name a route without the rest are refused in one answer, in `roll`'s words."""
    REQUIRED = ('provider', 'account', 'model')

    def test_each_refusal_names_what_was_given_what_is_missing_and_the_other_way_out(self):
        values = dict(provider='claude-lab', account='claude-lab', model='sonnet', effort='high')
        for given in (('account',), ('provider',), ('model',), ('effort',), ('provider', 'account'), ('account', 'model'),
                      ('provider', 'model'), ('provider', 'account', 'effort')):
            for basis in (None, 'the user said so'):
                with self.subTest(given=given, basis=bool(basis)):
                    with self.assertRaises(ValueError) as caught:
                        self.run_roll(basis=basis, task_id='t-named', **{flag: values[flag] for flag in given})
                    text = str(caught.exception)
                    missing = [flag for flag in self.REQUIRED if flag not in given] + ([] if basis else ['basis'])
                    self.assertTrue(missing)
                    gave_part, missing_part = text.split('This request gave ', 1)[1].split(' and is missing ', 1)
                    for flag in given:
                        self.assertIn(f'--{flag}', gave_part)
                    self.assertEqual('--basis' in gave_part, bool(basis))
                    for flag in missing:
                        self.assertIn(f'--{flag}', missing_part.split('.', 1)[0])
                    self.assertNotIn('--effort', missing_part.split('.', 1)[0])        # optional: never asked for
                    self.assertIn('drop', text)                                        # the other way out: no named route
                    self.assertNotIn('record', text)                                   # `record` is a stage the caller never ran
                    self.setUp()

    def test_record_reads_the_rule_roll_owns(self):
        import roll
        for given in ({}, {'account': 'a'}, {'provider': 'p', 'account': 'a', 'model': 'm'}, {'effort': 'high', 'basis': 'u'}):
            with self.subTest(given=given):
                args = argparse.Namespace(**{'provider': None, 'account': None, 'model': None, 'effort': None, 'basis': None, **given})
                with self.assertRaises(ValueError) as caught:
                    route_judge.record_constraint(args)
                self.assertEqual(str(caught.exception), roll.named_route_refusal(args, require=True))
        self.assertIsNone(roll.named_route_refusal(argparse.Namespace(provider=None, account=None, model=None, effort=None, basis=None)))

    def test_a_complete_named_route_is_not_refused_for_its_flags(self):
        import roll
        args = argparse.Namespace(provider='p', account='a', model='m', effort=None, basis='the user said so')
        self.assertIsNone(roll.named_route_refusal(args))
        self.assertIsNone(roll.named_route_refusal(argparse.Namespace(provider=None, account=None, model=None, effort=None, basis=None)))

    def test_an_input_error_is_not_sent_to_files_or_the_installation(self):
        text = route_judge.failure_answer(ValueError('A named route needs --provider'), 'roll')['next_action']
        system = route_judge.failure_answer(OSError(13, 'Permission denied', 'D:/x'), 'roll')['next_action']
        self.assertNotEqual(text, system)
        self.assertNotIn('installation', text)
        self.assertNotIn('unreadable', text)
        self.assertIn('installation', system)


class PaseoReplacedByCli(Plain):
    """Finding 4 (CLI): the condition that ended Paseo is said."""

    def test_the_condition_that_ended_paseo_is_named(self):
        import roll
        popen = MagicMock(return_value=MagicMock(pid=4242))
        with patch.object(roll.subprocess, 'Popen', popen), patch.object(flow, 'paseo_reachable', return_value=False):
            res, _ = self.run_roll(env={'PASEO_AGENT_ID': ''}, parent_id='parent-a', task_id='t-cli')
        self.assertEqual(res['status'], 'launched')
        self.assertTrue(any('Paseo was not used' in w and 'not reachable' in w for w in res.get('warnings', [])), res)

    def test_paseo_choice_names_the_first_condition_that_says_no(self):
        import roll
        args = argparse.Namespace(parent_id=None)
        with patch.dict(os.environ, {'PASEO_AGENT_ID': ''}):
            self.assertEqual(roll.paseo_choice('dot', 'cli', args), (False, None))      # not a launcher question
            self.assertIn('herdr', roll.paseo_choice('claude-lab', 'herdr', args)[1])
            with patch.object(roll, 'can_launch_via_paseo', return_value=False):
                self.assertIn("'claude-lab'", roll.paseo_choice('claude-lab', 'cli', args)[1])
            with patch.object(roll, 'can_launch_via_paseo', return_value=True):
                self.assertIn('no Paseo parent', roll.paseo_choice('claude-lab', 'cli', args)[1])
                with patch.object(flow, 'paseo_reachable', return_value=False):
                    self.assertIn('not reachable', roll.paseo_choice('claude-lab', 'cli', argparse.Namespace(parent_id='p'))[1])
                with patch.object(flow, 'paseo_reachable', return_value=True):
                    self.assertEqual(roll.paseo_choice('claude-lab', 'cli', argparse.Namespace(parent_id='p')), (True, None))


class LaunchRefusedBeforeAnyWorker(Plain):
    """Finding 2: no worker exists, so nothing is collected; each failure says what failed and what to do."""
    CODES = ('sdk_setup_failed', 'host_connection_failed', 'bridge_unavailable', 'create_send_window_elapsed',
             'parent_agent_not_found')
    STEPS = {   # a step the caller can take for that failure (commands and variables that exist)
        'sdk_setup_failed': ('agentcompass setup',), 'host_connection_failed': ('paseo ls',),
        'bridge_unavailable': ('agentcompass setup', 'Node.js'), 'create_send_window_elapsed': ('Roll the brief again',),
        'parent_agent_not_found': ('PASEO_AGENT_ID', '--parent-id')}

    def host_answering(self, response):
        def host(request):
            if request['operation'] == 'workspace_agents':
                return {'ok': True, 'placeable': True, 'agents': []}
            if request['operation'] in ('create', 'resume'):
                return dict(response)
            return {'ok': False, 'error': 'agent_not_observed'}
        return host

    def test_each_code_says_what_failed_and_never_sends_the_caller_to_collect(self):
        texts = {}
        for fallback in ('on', 'off'):
            for code in self.CODES:
                with self.subTest(code=code, fallback=fallback):
                    answer = bridge_answer(code, 'refused' if code == 'host_connection_failed' else None)
                    res, _ = self.run_roll(host=self.host_answering(answer), env={'AGENTCOMPASS_ROLL_FALLBACK': fallback},
                                           task_id=f't-{code}-{fallback}')
                    self.assertEqual((res['status'], res['cause']), ('error', code))
                    self.assertNotIn('continuation_argv', res['next_action'])
                    self.assertNotIn('collect the result', res['next_action'])
                    self.assertTrue(any(step in res['next_action'] for step in self.STEPS[code]), res['next_action'])
                    texts.setdefault(fallback, {})[code] = res['next_action']
                    self.setUp()
        for by_code in texts.values():       # a code-specific fact, not one text for every failure
            self.assertEqual(len(set(by_code.values())), len(by_code))

    def test_the_setup_hint_of_the_bridge_reaches_the_caller_as_a_sentence(self):
        for hint in ('Paseo bridge dependencies are missing. Run: agentcompass setup, or set ROUTER_PASEO_SDK_ROOT.',
                     'Paseo bridge dependencies are missing. Run: agentcompass setup'):
            with self.subTest(hint):
                res, _ = self.run_roll(host=self.host_answering(bridge_answer('sdk_setup_failed', setupHint=hint)), task_id='t-hint')
                self.assertIn(hint.rstrip('.') + '.', res['next_action'])
                self.setUp()


class ConditionAdvice(Plain):
    """Finding 2 of round 2: the advice follows the bridge's `condition`, from one table that create and resume both read."""

    def resume_refused_by(self, answer):
        self.make_worker('agent-n')

        def host(request):
            if request['operation'] == 'workspace_agents':
                return {'ok': True, 'placeable': True, 'agents': []}
            if request['operation'] == 'resume':
                return dict(answer)
            return {'ok': False, 'error': 'agent_not_observed'}
        return self.run_roll(self.brief_file('n.md'), host=host, resume='agent-n', task_id='t-n')[0]

    def test_every_token_of_the_bridges_vocabulary_has_its_facts(self):
        tokens = {*CONNECT_CONDITIONS, *REQUEST_CONDITIONS, *registry.WORKER_STATES}
        facts = {token: flow.condition_facts(token) for token in tokens}
        for token, (what, fix) in facts.items():
            with self.subTest(token):
                self.assertTrue(what.strip() and fix.strip())
        self.assertEqual(len({what for what, _ in facts.values()}), len(tokens))       # each token its own cause

    def test_a_create_that_never_connected_follows_its_condition(self):
        # Rule: starting the daemon is advised where nothing listens (`refused`) and nowhere else; a daemon that is up and
        # refuses the credential, or does not answer, is not helped by being started again.
        for token in CONNECT_CONDITIONS:
            with self.subTest(token):
                res, _ = self.run_roll(host=LaunchRefusedBeforeAnyWorker.host_answering(
                    self, bridge_answer('host_connection_failed', token)), task_id='t-' + token)
                self.assertEqual((res['status'], res['cause']), ('error', 'host_connection_failed'))
                what, fix = flow.HOST_CONDITIONS[token]
                self.assertIn(what.lower(), res['next_action'].lower())
                self.assertIn(fix.lower(), res['next_action'].lower())
                self.assertEqual(starts_the_daemon(res['next_action']), token == 'refused')
                self.setUp()

    def test_a_daemon_that_closed_the_handshake_is_running_and_is_not_started_again(self):
        # Rule: `rejected` is a daemon that accepted the connection, so it runs; the way out is its log and its version.
        what, fix = flow.HOST_CONDITIONS['rejected']
        self.assertIn('closed the handshake', what)
        self.assertFalse(starts_the_daemon(fix))
        self.assertIn('log', fix)
        self.assertIn('version', fix)

    def test_what_did_not_reach_the_daemon_is_not_said_to_have_reached_it(self):
        # Rule (the bridge's `bridge_error`): its own code threw, nothing reached the daemon; every other request class is a
        # daemon or connection failure after the bridge had connected. A text says each one as it is.
        for token in REQUEST_CONDITIONS:
            with self.subTest(token):
                res, _ = self.run_roll(host=LaunchRefusedBeforeAnyWorker.host_answering(
                    self, bridge_answer('host_request_failed', token)), task_id='t-reach-' + token)
                says_reached = 'reached the daemon' in res['next_action'] and 'nothing of it reached' not in res['next_action']
                self.assertFalse(says_reached)
                self.assertEqual('nothing of it reached the daemon' in res['next_action'], token == 'bridge_error')
                self.setUp()

    def test_the_composed_texts_join_their_sentences(self):
        # Rule: a fix is one clause with no semicolon, and the step that follows it starts a new sentence.
        for token, (what, fix) in flow.HOST_CONDITIONS.items():
            self.assertNotIn(';', fix, token)
        for token in (*CONNECT_CONDITIONS, 'some_new_class'):
            text = flow.not_sent_next('host_connection_failed', token)
            self.assertNotRegex(text, r'[,;] then\b', token)
            self.assertRegex(text, r'\. Then roll the brief again\.$', token)
        for token in (*REQUEST_CONDITIONS, 'some_new_class'):
            text = flow.resume_refusal_next('host_resume_failed', token)
            self.assertNotRegex(text, r'[,;] then\b', token)
            self.assertRegex(text, r'\. Then roll --resume again with the same id\.$', token)

    def test_a_named_resume_that_never_connected_reads_the_same_table(self):
        for token in CONNECT_CONDITIONS:
            with self.subTest(token):
                res = self.resume_refused_by(bridge_answer('host_connection_failed', token))
                self.assertEqual(res['cause'], 'resume_worker_unavailable')
                what, fix = flow.HOST_CONDITIONS[token]
                self.assertIn(what.lower(), res['next_action'].lower())
                self.assertIn(fix.lower(), res['next_action'].lower())
                self.assertEqual(starts_the_daemon(res['next_action']), token == 'refused')
                self.setUp()

    def test_a_named_resume_that_failed_before_it_was_sent_reads_the_request_classes(self):
        for token in REQUEST_CONDITIONS:
            with self.subTest(token):
                res = self.resume_refused_by(bridge_answer('host_resume_failed', token))
                what, fix = flow.HOST_CONDITIONS[token]
                self.assertIn(what.lower(), res['next_action'].lower())
                self.assertIn(fix.lower(), res['next_action'].lower())
                self.assertIn('host_resume_failed', res['next_action'])
                self.setUp()

    def test_a_token_the_table_does_not_hold_is_printed_and_presumes_no_cause(self):
        for answer in (bridge_answer('host_connection_failed', 'some_new_class'),):
            res, _ = self.run_roll(host=LaunchRefusedBeforeAnyWorker.host_answering(self, answer), task_id='t-new')
            self.assertIn('some_new_class', res['next_action'])
            self.assertFalse(starts_the_daemon(res['next_action']))
            self.setUp()
            res = self.resume_refused_by(answer)
            self.assertIn('some_new_class', res['next_action'])
            self.assertFalse(starts_the_daemon(res['next_action']))
            self.setUp()

    def test_a_condition_that_is_free_text_is_never_printed(self):
        res, _ = self.run_roll(host=LaunchRefusedBeforeAnyWorker.host_answering(
            self, bridge_answer('host_connection_failed', 'Free text, with a secret=abc')), task_id='t-free')
        self.assertNotIn('secret=abc', json.dumps(res))
        self.assertFalse(starts_the_daemon(res['next_action']))


class PendingNamesWhatIsKnown(Plain):
    """Finding 8: `pending` stays the status while a create may exist; the text and the cause follow the error."""

    def host_answering(self, response):
        return LaunchRefusedBeforeAnyWorker.host_answering(self, response)

    def test_unconfirmed_creates_name_the_error_and_do_not_say_still_creating(self):
        for code in ('host_timeout', 'invalid_host_response', 'create_outcome_uncertain_reconcile_by_label'):
            with self.subTest(code):
                res, code_ = self.run_roll(host=self.host_answering(bridge_answer(code)), task_id='t-' + code)
                self.assertEqual((res['status'], res['cause'], code_), ('pending', code, 0))
                self.assertNotIn('still creating', res['next_action'])
                self.assertIn('Do not roll again', res['next_action'])
                self.assertIn('next_argv', res['next_action'])
                self.assertIn('next_argv', res)
                self.setUp()

    def test_a_request_that_failed_after_it_reached_the_daemon_is_pending_with_its_condition(self):
        # The bridge answers host_request_failed with `sent: true` on a create: a worker may exist.
        for token in REQUEST_CONDITIONS:
            with self.subTest(token):
                answer = bridge_answer('host_request_failed', token)
                self.assertTrue(answer['sent'])
                res, code = self.run_roll(host=self.host_answering(answer), task_id='t-req-' + token)
                self.assertEqual((res['status'], res['cause'], code), ('pending', f'host_request_failed:{token}', 0))
                self.assertIn(flow.HOST_CONDITIONS[token][0].lower(), res['next_action'].lower())
                self.assertNotIn('still creating', res['next_action'])
                self.assertIn('Do not roll again', res['next_action'])
                self.assertIn('next_argv', res)
                self.setUp()

    def test_a_resume_the_daemon_may_have_received_is_pending_with_a_cause_and_its_own_text(self):
        generic = 'Run next_argv to observe the same operation; do not roll again.'
        for answer in (bridge_answer('host_request_failed', 'connection_lost'),
                       bridge_answer('resume_outcome_uncertain_reconcile_by_agent')):
            with self.subTest(answer['error']):
                self.make_worker('agent-n')

                def host(request, answer=answer):
                    if request['operation'] == 'workspace_agents':
                        return {'ok': True, 'placeable': True, 'agents': []}
                    if request['operation'] == 'resume':
                        return dict(answer)
                    return {'ok': False, 'error': 'agent_not_observed'}
                res, _ = self.run_roll(self.brief_file('n.md'), host=host, resume='agent-n', task_id='t-n')
                self.assertEqual(res['status'], 'pending')
                self.assertTrue(res['cause'].startswith(answer['error']), res)
                self.assertNotEqual(res['next_action'], generic)
                self.assertIn('Do not roll again', res['next_action'])
                self.assertIn('next_argv', res)
                if answer.get('condition'):
                    self.assertIn(flow.HOST_CONDITIONS[answer['condition']][0].lower(), res['next_action'].lower())
                self.setUp()

    def test_a_create_the_host_is_carrying_out_is_still_creating(self):
        res, _ = self.run_roll(host=self.host_answering({'ok': True, 'sent': True}), task_id='t-creating')
        self.assertEqual(res['status'], 'pending')
        self.assertIn('still creating', res['next_action'])
        self.assertNotIn('cause', res)


class FallbackKeepsClass(Plain):
    """Finding 12."""

    def test_a_request_scoped_refusal_keeps_its_class_with_the_fallback_on_and_off(self):
        host = LaunchRefusedBeforeAnyWorker.host_answering(self, bridge_answer('host_creation_failed', failure_class='request_invalid'))
        causes = {}
        for fallback in ('on', 'off'):
            res, _ = self.run_roll(host=host, env={'AGENTCOMPASS_ROLL_FALLBACK': fallback}, task_id='t-' + fallback)
            causes[fallback] = res['cause']
            self.setUp()
        self.assertEqual(causes['on'], causes['off'])
        self.assertIn('request_invalid', causes['on'])


class RefusedBeforeLaunchDetail(Plain):
    """Finding 6."""

    def test_the_refusal_text_of_a_first_launch_is_in_the_attempt_line(self):
        text = 'Account claude-lab five_hour is below its existing launch protection (5% remaining floor); refresh quota or replan'
        with patch.object(registry, 'check_launch_quota', side_effect=ValueError(text)):
            res, _ = self.run_roll(task_id='t-cap')
        self.assertEqual(res['status'], 'blocked')
        self.assertIn(text, res['next_action'])

    def test_line_prints_the_detail_of_any_attempt_that_has_one(self):
        import dispatch_fallback
        line = dispatch_fallback.line({'attempt': 2, 'model': 'm', 'account': 'a', 'failure': 'not_prepared', 'detail': 'the reason'})
        self.assertIn('the reason', line)


class ResumeRefusals(Plain):
    """Finding 7."""

    def resume(self, name, **over):
        return self.run_roll(self.brief_file(f'{name}.md'), resume='agent-' + name, task_id='t-' + name, **over)

    def test_the_way_out_follows_the_reason(self):
        # Rule (dispatch_registry.WORKER_STATES): the text carries the state's own fix, and offers a fresh worker only for
        # a state where one helps.
        cases = {
            'permission': (dict(doc={'attentionReason': 'permission'}), 'permission_pending'),
            'running': (dict(final_status='running'), 'running'),
            'status-error': (dict(final_status='error'), 'error'),
            'provider-unavailable': (dict(doc={'providerUnavailable': True}), 'provider_unavailable'),
            'provider-and-permission': (dict(doc={'providerUnavailable': True, 'attentionReason': 'permission'}), 'provider_unavailable'),
        }
        for label, (options, state) in cases.items():
            with self.subTest(label):
                self.make_worker('agent-' + label, **options)
                res, _ = self.resume(label)
                known = registry.WORKER_STATES[state]
                self.assertEqual(res.get('cause'), 'resume_worker_unavailable', res)
                self.assertIn(known.fix.lower(), res['next_action'].lower())
                self.assertEqual('fresh worker' in res['next_action'], known.fresh)
                self.setUp()

    def test_a_fresh_worker_is_offered_only_where_it_helps(self):
        # Rule (the worker's context is not carried): a fresh worker answers no permission, ends no running turn and does
        # not make a provider answer, so none of those offers one; a worker that is gone, closed or in error can be replaced.
        for state in ('permission_pending', 'running', 'initializing', 'provider_unavailable'):
            self.assertFalse(registry.WORKER_STATES[state].fresh, state)
            self.assertNotIn('fresh worker', registry.resume_way_out(state))
        for state in ('error', 'closed', 'archived'):
            self.assertTrue(registry.WORKER_STATES[state].fresh, state)
            self.assertIn('fresh worker', registry.resume_way_out(state))

    def test_the_registry_gives_the_remedy_with_the_reason(self):
        refusal = registry.reuse_refusal('parent-a', 'agent-nobody')
        self.assertTrue(refusal.reason and refusal.remedy)
        self.make_worker('agent-open', finish=False, final_status='running')       # a dispatch to it is still open: its way out is to wait, not a fresh worker
        refusal = registry.reuse_refusal('parent-a', 'agent-open')
        self.assertIn('still open', refusal.reason)
        self.assertIn('next_argv', refusal.remedy)
        self.assertNotIn('fresh worker', refusal.remedy)

    def test_the_registry_and_the_host_say_the_same_thing_about_a_state(self):
        # One owner (dispatch_registry.WORKER_STATES): the host's condition and the registry's reading give one remedy.
        for state in registry.WORKER_STATES:
            with self.subTest(state):
                self.assertEqual(flow.resume_refusal_next('worker_not_idle', state), registry.resume_way_out(state))
                self.assertEqual(registry.worker_state_refusal(state).remedy, registry.resume_way_out(state))

    def test_a_named_worker_the_host_refuses_says_the_condition(self):
        cases = (bridge_answer('worker_not_idle', 'permission_pending'), bridge_answer('worker_not_idle', 'running'),
                 bridge_answer('worker_not_idle', 'provider_unavailable'), bridge_answer('worker_not_found', 'archived'),
                 bridge_answer('worker_not_found', 'absent'), bridge_answer('host_connection_failed', 'refused'),
                 bridge_answer('worker_not_idle'))
        for answer in cases:
            with self.subTest(answer['error'], condition=answer.get('condition')):
                self.make_worker('agent-n')

                def host(request, answer=answer):
                    if request['operation'] == 'workspace_agents':
                        return {'ok': True, 'placeable': True, 'agents': []}
                    if request['operation'] == 'resume':
                        return dict(answer)
                    return {'ok': False, 'error': 'agent_not_observed'}
                res, _ = self.run_roll(self.brief_file('n.md'), host=host, resume='agent-n', task_id='t-n')
                self.assertEqual(res['cause'], 'resume_worker_unavailable')
                self.assertIn(answer['error'], res['next_action'])
                condition = answer.get('condition')
                if condition:
                    self.assertIn(condition, res['next_action'])
                if condition in registry.WORKER_STATES:
                    self.assertIn(registry.resume_way_out(condition), res['next_action'])
                self.assertNotIn('without --resume to let the router choose', res['next_action'])
                self.setUp()

    def test_a_fault_of_the_input_or_the_data_is_not_said_to_be_the_workers(self):
        import roll
        self.make_worker('agent-v')
        with patch.object(registry, 'prepare', side_effect=ValueError('task.profile is required: the classifier always produces it')):
            res, _ = self.resume('v')
        self.assertIn('task.profile is required', res['next_action'])
        self.assertNotIn('cannot take a new turn', res['next_action'])
        self.assertIn('not a state of the worker', res['next_action'])
        import turn_refusal
        with patch.object(turn_refusal, 'gate_refusal', side_effect=ValueError('data row for X is malformed')):
            reason = roll._worker_gate_refusal({'agent_id': 'a', 'bundle': {'account': 'claude-lab', 'bundle_id': 'lab_sonnet55_high'}},
                                               accounts=self.accounts)
        text = roll.gate_refusal_text('a', reason)
        self.assertIn('data row for X is malformed', text)
        self.assertNotIn('cannot take a new turn', text)
        self.assertIn('cannot take a new turn', roll.gate_refusal_text('a', 'account claude-lab is disabled'))

    def test_a_flag_naming_another_account_is_an_error_not_a_silent_launch(self):
        # Documented rule (roll-reference.md, `--resume`): flags naming another route are an error; the worker's own route is
        # its own. `--account` is one of them (named_route_conflict).
        self.make_worker('agent-r')
        with self.assertRaises(ValueError) as caught:
            self.run_roll(self.brief_file('r.md'), resume='agent-r', account='another-account', task_id='t-r')
        self.assertIn('--account', str(caught.exception))
        self.assertIn('--resume', str(caught.exception))
        text = (Path(__file__).resolve().parents[1] / 'references' / 'roll-reference.md').read_text(encoding='utf-8')
        sentence = text[text.index('`--resume` names the worker'):].split('\n', 1)[0]
        self.assertIn('`--account`', sentence.split('are an error', 1)[0])


class FailureOutsideTheResultPath(unittest.TestCase):
    """Finding 5."""

    def run_main(self, *argv, raising=None):
        import route_judge
        out = io.StringIO()
        stage = (patch.object(route_judge, '_run_stage', side_effect=raising) if raising else contextlib.nullcontext())
        with patch.object(sys, 'argv', ['agentcompass', *argv]), contextlib.redirect_stdout(out), stage, \
                patch.object(route_judge.public_data, 'refresh_in_background'), \
                patch.object(route_judge.antigravity_plan, 'refresh_in_background'):
            code = route_judge.main()
        return code, json.loads(out.getvalue())

    def test_a_missing_brief_is_named_with_status_cause_and_a_next_action_that_points_at_what_is_there(self):
        missing = str(Path('D:/work/nowhere/brief.md'))
        code, answer = self.run_main('roll', '--brief', missing)
        self.assertEqual(code, 1)
        self.assertEqual(answer['status'], 'error')
        self.assertIn('brief.md', answer['cause'])
        self.assertIn('FileNotFoundError', answer['cause'])
        for field in ('cause', 'next_action'):
            self.assertIn(field, answer)
        self.assertNotIn('detail', answer['next_action'])      # it points at no field that is not there

    def test_every_error_class_gets_a_cause_and_no_free_text_of_a_host(self):
        for error, must in ((OSError(13, 'Permission denied', 'D:/x/y'), 'D:/x/y'), (KeyError('model'), 'model'),
                            (ValueError('fixed text'), 'fixed text'), (RuntimeError('{"reply": "secret-body"}'), 'RuntimeError')):
            with self.subTest(type(error).__name__):
                code, answer = self.run_main('roll', '--task', 'x', raising=error)
                self.assertEqual((code, answer['status']), (1, 'error'))
                self.assertIn(must, answer['cause'])
                self.assertNotIn('secret-body', json.dumps(answer))
                self.assertIn('next_action', answer)


class DotFailures(HandBuilt):
    """Finding 9: the runner's exit code and its documented meaning are in the answer; its output is not."""
    OUTPUT = ('Invalid API key: sk-live-0123456789abcdef\n'
              'POST https://alice:hunter2pass@dot.example.net/v1/tasks failed\n'
              'Cookie: sid=Fe26.2**abc123def456**zz\n'
              'signed in as alice.smith@example.com\n'
              '{"error":{"message":"Your organization org-AbCdEf0123 must be verified"}}\n')
    PLANTED = ('sk-live-0123456789abcdef', 'hunter2pass', 'Fe26.2', 'alice.smith@example.com', 'org-AbCdEf0123', 'Invalid API key')

    def roll_dot(self, returncode, stderr):
        import roll

        def fake_run(command, **_):
            return subprocess.CompletedProcess(command, returncode, '', stderr)
        decision = dict(stage='plan', choice='dot_text', selected_bundle={'provider': 'dot', 'account': 'dot', 'model': 'dot'},
                        decision={'action': 'start'})
        with patch.object(roll.subprocess, 'run', fake_run):
            (result, code), _ = HandBuilt.roll(self, decision, task='x')
        return result, code

    def test_a_runner_that_launched_nothing_says_its_exit_code_and_what_it_means_and_never_exits_zero(self):
        import roll
        for returncode in (1, 2, 3, 9):
            with self.subTest(returncode):
                result, code = self.roll_dot(returncode, 'dot2api is unreachable\n')
                self.assertEqual(code, 1)
                self.assertEqual(result['status'], 'error')
                self.assertEqual(result['cause'], f'dot_runner_exit_{returncode}')
                self.assertIn(f'exited {returncode}', result['next_action'])
                self.assertIn(roll.DOT_EXIT_MEANING.get(returncode, 'ended abnormally'), result['next_action'])
                self.assertIn('details', result['next_action'])

    def test_the_meanings_are_the_runners_own(self):
        # Rule: roll states what the runner documents (SKILL.md, Outcomes), by the runner's EXIT_* constants.
        import roll
        runner = (Path(__file__).resolve().parents[2] / 'dot-subagent' / 'scripts' / 'run_dot_task.py').read_text(encoding='utf-8')
        for name, value in (('EXIT_FAILED', 1), ('EXIT_USAGE', 2), ('EXIT_PENDING', 3)):
            self.assertIn(f'{name} = {value}', runner)
            self.assertIn(value, roll.DOT_EXIT_MEANING)

    def test_nothing_of_the_runners_output_is_in_the_answer_but_it_is_kept_in_the_details_file(self):
        for returncode in (1, 2, 3):
            with self.subTest(returncode):
                result, _ = self.roll_dot(returncode, self.OUTPUT)
                for planted in self.PLANTED:
                    self.assertNotIn(planted, json.dumps(result))
                    self.assertNotIn(planted, result['next_action'])
                details = Path(result['details'])
                self.assertTrue(details.exists())
                self.assertIn('hunter2pass', details.read_text(encoding='utf-8'))      # kept there, as the answer says

    def test_a_launched_task_keeps_its_exit_code(self):
        result, code = self.roll_dot(3, 'task_id=dot-task-9\n')
        self.assertEqual((code, result['status']), (0, 'launched'))


class NamedRouteFactRefusalTests(unittest.TestCase):
    """Contract (roll-reference.md, router-internals.md): a named route refused by an account/model/host fact answers
    `status: blocked`, `caller_action: tell_user` (exit 2), carrying `start_after_hours` and `retry_at` from the same
    owner as the automatic deferral path when every refusing quota window has an observed reset, and naming no `plan`,
    `replan`, or `substitute`; request-input refusals stay `status: error` (exit 1); standalone `record` answers
    identically; forced-route relaunch (`dispatch_fallback`) keeps its endings without telling the parent to replan."""

    def setUp(self):
        from datetime import datetime, timezone
        import provider_inventory
        from router_test_support import profile
        self.home = isolated_router_state(self.addCleanup, discovery=False, policy=False)
        self.state = Path(os.environ['ROUTER_STATE_DIR'])
        self.state.mkdir(parents=True, exist_ok=True)
        self.cache_path = Path(os.environ['ROUTER_USAGE_CACHE'])
        self.ledger_path = Path(os.environ['ROUTER_LEDGER'])
        self.db_path = Path(os.environ['ROUTER_DISPATCH_DB'])
        self.policy_path = self.state / 'host-policy.json'
        self.policy_path.write_text(
            json.dumps({'allowed_accounts': ['claude'], 'known_accounts': ['claude']}), encoding='utf-8',
        )
        (self.home / '.claude').mkdir(parents=True, exist_ok=True)
        (self.home / '.claude' / '.credentials.json').write_text(
            json.dumps({'claudeAiOauth': {'accessToken': 'fixture-claude-token', 'subscriptionType': 'pro'}}),
            encoding='utf-8',
        )
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.enterContext(patch.object(provider_inventory, 'current', side_effect=provider_inventory.discover))
        self.enterContext(patch.object(
            provider_inventory, '_which', side_effect=lambda product, *a, **k: 'fixture-claude' if product == 'claude' else None,
        ))
        self.enterContext(patch.dict(os.environ, {
            'ROUTER_HOST_POLICY': str(self.policy_path),
            'PASEO_AGENT_ID': 'parent-named',
            'ROUTER_LAUNCH_COOLDOWN': 'on',
        }))
        import scorer_client
        self.enterContext(patch.object(
            scorer_client, 'score_task', return_value=(profile(), {'source': 'system_one', 'model': 'fixture'}),
        ))

    def write_usage_cache(self, windows, *, account='claude', plan='pro'):
        import provider_inventory
        cred_key = ((provider_inventory.discover().get('accounts') or {}).get(account) or {}).get('credential_key')
        doc = {
            'providers': {
                account: {
                    'available': True,
                    'stale': False,
                    'plan': plan,
                    'plan_label': plan,
                    'plan_source': 'profile_api',
                    **({'credential_key': cred_key} if cred_key else {}),
                    'fetched_at': self.now.isoformat(),
                    'windows': windows,
                },
            },
        }
        self.cache_path.write_text(json.dumps(doc), encoding='utf-8')

    def run_cli(self, *argv):
        buf = io.StringIO()
        with patch.object(sys, 'argv', ['agentcompass', *argv]), contextlib.redirect_stdout(buf):
            code = route_judge.main()
        return code, json.loads(buf.getvalue())

    NAMED_FLAGS = ('--provider', 'claude', '--account', 'claude', '--model', 'claude-opus-5-5', '--effort', 'high',
                   '--basis', 'the user said: have Opus look at this one')

    def test_a_and_d_floor_refusal_with_observed_reset_matches_automatic_start_time_for_roll_and_record(self):
        # Claude Pro declares `five_hour` `launch_min_remaining_pct: 25`; 85% used leaves 15% with an observed reset in 2h15m.
        reset_dt = self.now + timedelta(hours=2, minutes=15)
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 85.0, 'resets_at': reset_dt.isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        auto_code, auto_ans = self.run_cli('roll', '--task', 'Review architecture')
        self.assertEqual((auto_code, auto_ans['status'], auto_ans.get('caller_action')), (2, 'blocked', 'wait_for_reset'), auto_ans)
        expected_hours = auto_ans['start_after_hours']
        self.ledger_path.unlink(missing_ok=True)
        self.db_path.unlink(missing_ok=True)

        # (a) Named roll
        roll_code, roll_ans = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
        self.assertEqual((roll_code, roll_ans['status'], roll_ans.get('caller_action')), (2, 'blocked', 'tell_user'), roll_ans)
        self.assertAlmostEqual(roll_ans['start_after_hours'], expected_hours, places=1)
        self.assertEqual(roll_ans['retry_at'], reset_dt.isoformat())
        for text in (roll_ans['cause'], roll_ans['next_action']):
            lowered = text.lower()
            for forbidden in ('plan', 'replan', 'substitute', 'pick an account', 'change a flag', 'change what cause names'):
                self.assertNotIn(forbidden, lowered, text)
        self.assertFalse(self.ledger_path.exists())
        self.assertFalse(self.db_path.exists())

        # (d) Standalone record answers identically from the same owner
        rec_code, rec_ans = self.run_cli('record', '--constraint', 'user_override', *self.NAMED_FLAGS)
        self.assertEqual((rec_code, rec_ans['status'], rec_ans.get('caller_action')), (2, 'blocked', 'tell_user'), rec_ans)
        self.assertEqual(rec_ans['cause'], roll_ans['cause'])
        self.assertEqual(rec_ans['next_action'], roll_ans['next_action'])
        self.assertAlmostEqual(rec_ans['start_after_hours'], roll_ans['start_after_hours'], places=1)
        self.assertEqual(rec_ans['retry_at'], roll_ans['retry_at'])
        self.assertFalse(self.ledger_path.exists())
        self.assertFalse(self.db_path.exists())

    def test_b_fact_refusals_without_observed_reset_carry_no_time(self):
        import launch_failures
        import model_latch
        import quota_exhaustion
        # 1. A window latched in quota-exhaustion.json, with no reset its error stated and none observed for five_hour.
        # Its latch still ends by itself (quota_exhaustion.due_at: the re-probe time), and that end is the schedule, as
        # the automatic path waits for it (candidate_eval, allocator.refusal_schedule; free-pool branch, addendum 3
        # point 2); the answer names it as a re-probe, never a reset.
        self.write_usage_cache([
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        self.assertTrue(quota_exhaustion.record(
            'claude', 'Individual quota reached', windows=['five_hour'], at=(self.now + timedelta(seconds=1)).isoformat(),
        ))
        try:
            due, stated = quota_exhaustion.ends('claude')
            self.assertFalse(stated)
            code, ans = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
            self.assertEqual((code, ans['status'], ans.get('caller_action')), (2, 'blocked', 'tell_user'), ans)
            self.assertIn('five_hour', ans['cause'])
            self.assertEqual(datetime.fromisoformat(ans['retry_at']), due)
            self.assertGreaterEqual(ans['start_after_hours'], (due - self.now).total_seconds() / 3600 - 0.1)
            self.assertIn(quota_exhaustion.end_kind(False), ans['next_action'])
            self.assertNotIn('five_hour resets', ans['next_action'])
            self.assertNotIn('cannot start now', ans['next_action'])
        finally:
            quota_exhaustion.clear('claude')

        # 2. Cooldown, model latch, and missing credentials all answer `blocked` / `tell_user` with no time, no `cannot start now`, and no `replan`.
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        bundle = {'account': 'claude', 'provider': 'claude', 'model': 'claude-opus-5-5'}
        launch_failures.record(bundle, 'provider_unavailable', now=self.now)
        try:
            code, cool_ans = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
            self.assertEqual((code, cool_ans['status'], cool_ans.get('caller_action')), (2, 'blocked', 'tell_user'), cool_ans)
            self.assertIn('cooling down', cool_ans['cause'])
            self.assertNotIn('start_after_hours', cool_ans)
            self.assertNotIn('cannot start now', cool_ans['next_action'])
            self.assertNotIn('replan', cool_ans['cause'].lower() + cool_ans['next_action'].lower())
        finally:
            launch_failures.clear('claude')

        model_latch.record(bundle, 'INVALID_MODEL', at=self.now)
        try:
            code, latch_ans = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
            self.assertEqual((code, latch_ans['status'], latch_ans.get('caller_action')), (2, 'blocked', 'tell_user'), latch_ans)
            self.assertIn('INVALID_MODEL', latch_ans['cause'])
            self.assertNotIn('start_after_hours', latch_ans)
            self.assertNotIn('cannot start now', latch_ans['next_action'])
        finally:
            model_latch.clear('claude')

        (self.home / '.claude' / '.credentials.json').unlink()
        self.cache_path.unlink()
        code, cred_ans = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
        self.assertEqual((code, cred_ans['status'], cred_ans.get('caller_action')), (2, 'blocked', 'tell_user'), cred_ans)
        self.assertNotIn('start_after_hours', cred_ans)
        self.assertNotIn('cannot start now', cred_ans['next_action'])
        self.assertNotIn('reroute', cred_ans['cause'].lower())

        # 3. Unreadable live model list for a model with no row answers `blocked` / `tell_user` (host fact), while a
        # readable list omitting that model answers `error` (request input, tested in test_c).
        (self.home / '.claude' / '.credentials.json').write_text(
            json.dumps({'claudeAiOauth': {'accessToken': 'fixture-claude-token', 'subscriptionType': 'pro'}}),
            encoding='utf-8',
        )
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        import live_models
        live_models.invalidate()
        with patch.object(live_models, 'fetch_from_paseo', return_value={'ok': False, 'error': 'host_connection_failed'}):
            code, list_ans = self.run_cli(
                'roll', '--task', 'Review architecture',
                '--provider', 'claude', '--account', 'claude', '--model', 'claude-zeta-1', '--effort', 'high',
                '--basis', 'the user said: try claude-zeta-1',
            )
        self.assertEqual((code, list_ans['status'], list_ans.get('caller_action')), (2, 'blocked', 'tell_user'), list_ans)
        self.assertIn('model list of provider claude could not be read', list_ans['cause'])
        self.assertNotIn('start_after_hours', list_ans)
        self.assertNotIn('cannot start now', list_ans['next_action'])
        live_models.invalidate()

    def named_refusal_at_record_and_prepare(self, decision, task, args):
        """(record's refusal, prepare's refusal) of the same named route for the current usage cache."""
        import account_fleet
        import usage_reading
        with self.assertRaises(registry.RouteFactRefused) as at_record:
            route_judge.record_constraint(args, task=dict(task, task_id=task['task_id'] + '-again'),
                                          classified={'task': task, 'scorer': None})
        with self.assertRaises(registry.RouteFactRefused) as at_prepare:
            registry.prepare(decision['decision_id'], task, {'title': 'Review', 'initialPrompt': 'Review'},
                             usage_reading.usage_snapshot(self.cache_path, now=self.now),
                             accounts=account_fleet.load_accounts(), now=self.now)
        return at_record.exception, at_prepare.exception

    def test_e_a_named_route_a_quota_fact_holds_is_one_fact_refusal_at_the_plan_and_before_launch(self):
        """Rule (brief D2; dispatch_registry.launch_fact_refusal, named_route_refusal, RouteFactRefused): a route the
        user named that a quota window with an observed reset holds answers `blocked` / `tell_user`, exit 2, with a cause
        naming the window and its state, and the schedule of that reset; the check at record and the re-check before
        launch read the same facts (the task-independent launch facts, then the named route's own plan row) and give the
        same refusal for the same reading; no text says "reset not yet observed"."""
        from router_test_support import profile
        healthy = [{'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
                   {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()}]
        reset = self.now + timedelta(hours=20, minutes=12)
        short_reset = self.now + timedelta(hours=2, minutes=12)
        facts = (
            # (label, windows, held window, work hours, held only by the size of the work)
            ('used up with an observed reset', [healthy[0], {'id': 'seven_day', 'used_pct': 100.0,
                                                             'resets_at': reset.isoformat()}], 'seven_day', 0.5, False),
            ('short of the work with an observed reset', [healthy[0], {'id': 'seven_day', 'used_pct': 88.0,
                                                                       'resets_at': reset.isoformat()}],
             'seven_day', 4.0, True),
            # Work beyond the window's full capacity: a named route launches past that estimate with its warning
            # (full_capacity_warning), so the window at 0 holds it until its reset, not without a time.
            ('used up with an observed reset, the work beyond full capacity',
             [{'id': 'five_hour', 'used_pct': 100.0, 'resets_at': short_reset.isoformat()}, healthy[1]], 'five_hour',
             400.0, False),
        )   # used up with no observed reset: test_free_pool_unpublished (the cache reader pairs a used-up reading with its reset)
        args = route_judge.build_parser().parse_args(['record', '--constraint', 'user_override', *self.NAMED_FLAGS])
        for n, (label, windows, wid, hours, by_size) in enumerate(facts):
            with self.subTest(fact=label):
                self.write_usage_cache(healthy)
                task = {'parent_agent_id': 'parent-named', 'task_id': f't-fact-{n}', 'objective': 'Review',
                        'expected_hours': hours, 'profile': profile()}
                decision = route_judge.record_constraint(args, task=task, classified={'task': task, 'scorer': None})
                self.assertTrue(decision.get('decision_id'), decision)
                self.write_usage_cache(windows)
                code, ans = self.run_cli('roll', '--task', 'Review architecture', '--hours', str(hours), *self.NAMED_FLAGS)
                self.assertEqual((code, ans['status'], ans.get('caller_action')), (2, 'blocked', 'tell_user'), ans)
                self.assertIn(wid, ans['cause'])
                self.assertNotIn('reset not yet observed', ans['cause'] + ans['next_action'])
                held_reset = short_reset if wid == 'five_hour' else reset
                self.assertGreaterEqual(datetime.fromisoformat(ans['retry_at']), held_reset - timedelta(seconds=1))
                self.assertLessEqual(datetime.fromisoformat(ans['retry_at']), held_reset + timedelta(seconds=1))
                self.assertGreaterEqual(ans['start_after_hours'], (held_reset - self.now).total_seconds() / 3600 - 0.01)
                # What holds it: a fact of the window whatever the task's size, or only this work's size (the named
                # route's own plan row, which the record now reads as the re-check before launch does).
                import account_fleet
                import provider_inventory
                import usage_reading
                accounts = account_fleet.load_accounts()
                usage = provider_inventory.merge_usage(usage_reading.usage_snapshot(self.cache_path, now=self.now),
                                                       accounts, self.now)
                task_free = registry.launch_fact_refusal(dict(decision['selected_bundle'], account='claude'), accounts,
                                                         usage, self.now, user_named=True, records=lambda: [])
                self.assertEqual(task_free is None, by_size, task_free and task_free.cause)
                if by_size:
                    # The reading changed between the record and the launch: roll answers the re-check's refusal the same
                    # way (prepare raises it; nothing is launched).
                    with patch.object(route_judge, '_check_named_plan_row'):
                        code_p, ans_p = self.run_cli('roll', '--task', 'Review architecture', '--hours', str(hours),
                                                     *self.NAMED_FLAGS)
                    self.assertEqual((code_p, ans_p['status'], ans_p.get('caller_action')), (2, 'blocked', 'tell_user'),
                                     ans_p)
                    self.assertEqual((ans_p['cause'], ans_p['start_after_hours']), (ans['cause'], ans['start_after_hours']))
                    self.assertTrue(ans_p['next_action'].startswith('Nothing was launched:'), ans_p['next_action'])
                    self.assertTrue(ans['next_action'].startswith('Nothing was recorded or launched:'), ans['next_action'])
                at_record, at_prepare = self.named_refusal_at_record_and_prepare(decision, task, args)
                self.assertEqual(at_record.cause, at_prepare.cause)
                self.assertEqual((at_record.start_after_hours, at_record.waiting_on),
                                 (at_prepare.start_after_hours, at_prepare.waiting_on))
                self.assertLess(abs(datetime.fromisoformat(at_record.retry_at)
                                    - datetime.fromisoformat(at_prepare.retry_at)), timedelta(seconds=1))
                self.assertIn(wid, at_prepare.cause)

    def test_f_a_named_route_two_windows_hold_is_refused_with_the_later_wait_at_the_plan_and_before_launch(self):
        """Rule (D1 a wait the router offers is a promise; D2 one source of a refusal's time: the named route's own plan
        row, RouteFactRefused.timed_by and named_route_refusal): when two windows hold the route, the refusal's time is
        the row's binding wait, the later of the two, with that window named in `cause` and as `waiting_on`, at the
        record and at the re-check before launch alike, also when the earlier window is a launch fact whose own schedule
        (allocator.refusal_schedule) is the earlier reset."""
        import account_fleet
        import allocator
        import provider_inventory
        import usage_reading
        from router_test_support import profile
        healthy = [{'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
                   {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()}]
        short_reset = self.now + timedelta(hours=1, minutes=12)
        long_reset = self.now + timedelta(hours=20, minutes=12)
        seven_short = {'id': 'seven_day', 'used_pct': 88.0, 'resets_at': long_reset.isoformat()}
        cases = (
            # (label, windows, the earlier window is a launch fact)
            ('a launch fact and a later wait of the row', [{'id': 'five_hour', 'used_pct': 100.0,
                                                            'resets_at': short_reset.isoformat()}, seven_short], True),
        )
        args = route_judge.build_parser().parse_args(['record', '--constraint', 'user_override', *self.NAMED_FLAGS])
        for n, (label, windows, fact_first) in enumerate(cases):
            with self.subTest(case=label):
                self.write_usage_cache(healthy)
                task = {'parent_agent_id': 'parent-named', 'task_id': f't-two-{n}', 'objective': 'Review',
                        'expected_hours': 4.0, 'profile': profile()}
                decision = route_judge.record_constraint(args, task=task, classified={'task': task, 'scorer': None})
                self.assertTrue(decision.get('decision_id'), decision)
                self.write_usage_cache(windows)
                accounts = account_fleet.load_accounts()
                usage = provider_inventory.merge_usage(usage_reading.usage_snapshot(self.cache_path, now=self.now),
                                                       accounts, self.now)
                bundle = dict(decision['selected_bundle'], account='claude')
                fact = registry.launch_fact_refusal(bundle, accounts, usage, self.now, user_named=True,
                                                    records=lambda: [])
                self.assertEqual(fact is not None, fact_first, fact and fact.cause)
                if fact_first:
                    # The fact's own schedule is the earlier reset: the refusal must not promise it.
                    _, own_retry, own_window = allocator.refusal_schedule(bundle, accounts, usage, self.now,
                                                                          blocked_wids=fact.windows, records=[])
                    self.assertEqual(own_window, 'five_hour')
                    self.assertLess(abs(datetime.fromisoformat(own_retry) - short_reset), timedelta(seconds=1))
                at_record, at_prepare = self.named_refusal_at_record_and_prepare(decision, task, args)
                for refused in (at_record, at_prepare):
                    self.assertEqual(refused.waiting_on, 'seven_day', refused.cause)
                    self.assertLess(abs(datetime.fromisoformat(refused.retry_at) - long_reset), timedelta(seconds=1))
                    self.assertGreaterEqual(refused.start_after_hours,
                                            (long_reset - self.now).total_seconds() / 3600 - 0.01)
                    self.assertIn('five_hour', refused.cause)
                    self.assertIn('seven_day', refused.cause)
                self.assertEqual(at_record.cause, at_prepare.cause)
                self.assertEqual(at_record.start_after_hours, at_prepare.start_after_hours)

    def test_g_a_plan_row_with_two_waits_refuses_with_the_later_ones_window_and_hours(self):
        """Rule (D1 a wait the router offers is a promise: at its end every gate passes; named_route_refusal): a row two
        windows hold refuses the named route with the binding wait, the later one: its window as `waiting_on`, its hours
        (advised) and the time they end, whichever order the row lists them in."""
        bundle = {'account': 'claude', 'bundle_id': 'opus', 'id': 'opus'}
        for waits in ({'five_hour': 1.2, 'seven_day': 20.2}, {'seven_day': 20.2, 'five_hour': 1.2}):
            with self.subTest(order=list(waits)):
                row = {'bundle': bundle, 'excluded': [], 'waits': waits,
                       'wait_reasons': {w: f'{w} holds it for {h}h' for w, h in waits.items()}}
                exclusions, held = registry.plan_row_holds(row, True)
                refused = registry.named_route_refusal(row, exclusions, held, {'accounts': {}}, self.now,
                                                       user_named=True)
                self.assertEqual(refused.waiting_on, 'seven_day')
                self.assertEqual(refused.start_after_hours, __import__('allocator').advised_hours(20.2))
                self.assertEqual(datetime.fromisoformat(refused.retry_at), self.now + timedelta(hours=20.2))
                self.assertTrue(refused.cause.startswith('seven_day holds it'), refused.cause)

    def test_c_request_input_refusals_stay_error_with_present_next_action(self):
        import live_models
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        live_models.invalidate()
        with patch.object(live_models, 'fetch_from_paseo', return_value={
            'ok': True, 'providers': {'claude': {'ok': True, 'models': [{'id': 'claude-opus-5-5'}]}},
        }):
            for label, extra in (
                ('unaccepted effort', ('--provider', 'claude', '--account', 'claude', '--model', 'claude-opus-5-5',
                                       '--effort', 'turbo', '--basis', 'quote')),
                ('model not listed by provider', ('--provider', 'claude', '--account', 'claude',
                                                  '--model', 'claude-zeta-2', '--effort', 'high', '--basis', 'quote')),
                ('missing route flags', ('--provider', 'claude')),
                ('basis over limit', ('--provider', 'claude', '--account', 'claude', '--model', 'claude-opus-5-5',
                                      '--basis', 'x' * 401)),
            ):
                with self.subTest(label=label):
                    code, ans = self.run_cli('roll', '--task', 'Review architecture', *extra)
                    self.assertEqual((code, ans['status']), (1, 'error'), ans)
                    self.assertNotIn('caller_action', ans)
                    self.assertTrue(
                        ans['next_action'].startswith('The request was refused as given. Change what cause names'),
                        ans['next_action'],
                    )
        live_models.invalidate()

    def test_forced_route_prepare_refusal_in_fallback_never_tells_parent_to_replan(self):
        import account_fleet
        import dispatch_fallback
        import model_latch
        import roll
        import usage_reading
        from router_test_support import profile
        # Issue a valid named decision while healthy, then check both prepare-time refusal paths:
        # (1) check_launch_quota floor protection, and (2) candidate_eval exclusion via model_latch at prepare time.
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        args = route_judge.build_parser().parse_args([
            'record', '--constraint', 'user_override', *self.NAMED_FLAGS,
        ])
        task = {'parent_agent_id': 'parent-named', 'task_id': 't-forced', 'objective': 'Review',
                'expected_hours': 0.5, 'profile': profile()}
        decision = route_judge.record_constraint(args, task=task, classified={'task': task, 'scorer': None})
        task2 = {**task, 'task_id': 't-forced-latch'}
        decision2 = route_judge.record_constraint(args, task=task2, classified={'task': task2, 'scorer': None})

        # Path 1: check_launch_quota floor refusal inside prepare
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 90.0, 'resets_at': (self.now + timedelta(hours=2)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        launch = {'title': 'Review', 'initialPrompt': 'Review'}

        def named_fact(decision_item, task_item):
            """A route the user named, held by a fact at prepare: the fact refusal (brief D2), never a failed attempt."""
            with self.assertRaises(registry.RouteFactRefused) as caught:
                registry.prepare(
                    decision_item['decision_id'], task_item, launch,
                    usage_reading.usage_snapshot(self.cache_path, now=self.now),
                    accounts=account_fleet.load_accounts(), now=self.now,
                )
            answer = caught.exception.answer({'stage': 'roll'}, for_roll=True)
            self.assertEqual((answer['status'], answer['caller_action']), ('blocked', 'tell_user'))
            combined = (answer['cause'] + '\n' + answer['next_action']).lower()
            for forbidden in ('replan', 'reroute', 'refresh quota', 'transient'):
                self.assertNotIn(forbidden, combined)
            return caught.exception

        floor = named_fact(decision, task)
        self.assertIn('five_hour', floor.cause)
        self.assertIsNotNone(floor.retry_at)       # the floor's window has an observed reset

        # Path 2: candidate exclusion inside prepare — model_latch and latched quota window
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        model_latch.record({'account': 'claude', 'provider': 'claude', 'model': 'claude-opus-5-5'}, 'INVALID_MODEL', at=self.now)
        try:
            latched_model = named_fact(decision2, task2)
            self.assertIn('INVALID_MODEL', latched_model.cause)
            self.assertIsNone(latched_model.retry_at)
        finally:
            model_latch.clear('claude')

        # Path 3 (F2 + F9): latched quota window at prepare time for both `user_override` and `hard_constraint`
        # calls `quota_pools.window_refusal` (one owner, no "refresh quota or replan") and compact carries `tell_user`.
        import quota_exhaustion
        task_uo = {**task, 'task_id': 't-forced-exhausted-uo'}
        dec_uo = route_judge.record_constraint(args, task=task_uo, classified={'task': task_uo, 'scorer': None})
        args_hc = route_judge.build_parser().parse_args([
            'record', '--constraint', 'hard_constraint', '--provider', 'claude', '--account', 'claude',
            '--model', 'claude-opus-5-5', '--effort', 'high', '--basis', 'project rule',
        ])
        task_hc = {**task, 'task_id': 't-forced-exhausted-hc'}
        dec_hc = route_judge.record_constraint(args_hc, task=task_hc, classified={'task': task_hc, 'scorer': None})
        self.assertTrue(quota_exhaustion.record(
            'claude', 'Individual quota reached', windows=['five_hour'], at=(self.now + timedelta(seconds=1)).isoformat(),
        ))
        try:
            latched = named_fact(dec_uo, task_uo)
            self.assertIn('five_hour', latched.cause)
            for label, dec_item, task_item, args_item, expected_err in (
                ('hard_constraint', dec_hc, task_hc, args_hc, 'relaunch_not_started'),
            ):
                with self.subTest(label=label):
                    with self.assertRaises(registry.CapacityChanged) as caught_ex:
                        registry.prepare(
                            dec_item['decision_id'], task_item, launch,
                            usage_reading.usage_snapshot(self.cache_path, now=self.now),
                            accounts=account_fleet.load_accounts(), now=self.now,
                        )
                    self.assertNotIn('replan', str(caught_ex.exception).lower())
                    self.assertNotIn('refresh quota', str(caught_ex.exception).lower())
                    out_ex = dispatch_fallback.after_refusal(
                        dec_item['decision_id'], launch, caught_ex.exception, judge=route_judge,
                    )
                    res_ex, code_ex = roll.fallback_result({'stage': 'roll'}, out_ex, dec_item, 'parent-named', args_item)
                    self.assertEqual((code_ex, res_ex['caller_action'], res_ex['error']), (2, 'tell_user', expected_err))
                    compact_ex = roll.compact(res_ex, 'blocked', None)
                    self.assertEqual((compact_ex['status'], compact_ex.get('caller_action')), ('blocked', 'tell_user'))
                    combined_ex = (res_ex['next'] + '\n' + '\n'.join(res_ex.get('attempts') or ())).lower()
                    self.assertNotIn('replan', combined_ex)
                    self.assertNotIn('refresh quota', combined_ex)
                    self.assertNotIn('reroute', combined_ex)
        finally:
            quota_exhaustion.clear('claude')

    def test_f1_prepare_and_claim_floor_refusal_with_unsettled_ended_worker_return_capacity_changed_at_once(self):
        import account_fleet
        import allocator
        import time
        import usage_reading
        from router_test_support import profile
        # Healthy initial state: prepare a first worker and write its finished watcher receipt on disk (unsettled),
        # plus prepare one dispatch to test `claim` as well, and record both an automatic and a forced decision.
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        accounts = account_fleet.load_accounts()
        usage_ok = usage_reading.usage_snapshot(self.cache_path, now=self.now)
        args_forced = route_judge.build_parser().parse_args([
            'record', '--constraint', 'user_override', *self.NAMED_FLAGS,
        ])
        t_ended = {'parent_agent_id': 'parent-named', 'task_id': 't-ended', 'objective': 'First',
                   'expected_hours': 0.05, 'hours': 0.05, 'profile': profile()}
        dec_ended = route_judge.record_constraint(args_forced, task=t_ended, classified={'task': t_ended, 'scorer': None})
        t_claim = {**t_ended, 'task_id': 't-claim'}
        dec_claim = route_judge.record_constraint(args_forced, task=t_claim, classified={'task': t_claim, 'scorer': None})
        t_auto = {**t_ended, 'task_id': 't-auto'}
        planned = allocator.plan(t_auto, usage_ok, self.now, accounts=accounts)
        self.assertNotIn(planned['choice'], (None, 'none'), planned)
        dec_auto = dict(planned, **registry.issue(planned, t_auto, self.now))
        t_forced = {**t_ended, 'task_id': 't-forced-f1'}
        dec_forced = route_judge.record_constraint(args_forced, task=t_forced, classified={'task': t_forced, 'scorer': None})

        row_ended = registry.prepare(dec_ended['decision_id'], t_ended, {'title': 'First', 'initialPrompt': 'First'},
                                     usage_ok, accounts=accounts, now=self.now)
        row_claim = registry.prepare(dec_claim['decision_id'], t_claim, {'title': 'Claim', 'initialPrompt': 'Claim'},
                                     usage_ok, accounts=accounts, now=self.now)
        notif_dir = self.state / 'paseo-notifications'
        notif_dir.mkdir(parents=True, exist_ok=True)
        (notif_dir / f"{row_ended['dispatch_id']}.json").write_text(
            json.dumps({'terminal': {'reason': 'finished', 'at': (self.now + timedelta(seconds=5)).isoformat()}}),
            encoding='utf-8',
        )

        # Now `five_hour` drops below the 25% launch protection floor while `row_ended`'s receipt is still unsettled on disk.
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 90.0, 'resets_at': (self.now + timedelta(hours=2)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        usage_low = usage_reading.usage_snapshot(self.cache_path, now=self.now)
        accounts_low = account_fleet.load_accounts()

        for label, dec_item, task_item, refusal in (
            ('automatic', dec_auto, t_auto, registry.CapacityChanged),
            ('forced', dec_forced, t_forced, registry.RouteFactRefused),     # a route the user named: a fact (brief D2)
        ):
            with self.subTest(kind=label):
                t0 = time.monotonic()
                with self.assertRaises(refusal):
                    registry.prepare(
                        dec_item['decision_id'], task_item, {'title': label, 'initialPrompt': label},
                        usage_low, accounts=accounts_low, now=self.now,
                    )
                elapsed = time.monotonic() - t0
                self.assertLess(elapsed, 1.0, f'{label} prepare took {elapsed:.2f}s')

        t0 = time.monotonic()
        with self.assertRaises(registry.CapacityChanged):
            registry.claim(row_claim['request'], parent_id='parent-named', now=self.now)
        elapsed_claim = time.monotonic() - t0
        self.assertLess(elapsed_claim, 1.0, f'claim took {elapsed_claim:.2f}s')

    def test_n1_f3_f4_unknown_name_signed_out_isolated_account_and_wrong_provider_on_signed_in_account(self):
        # `claude` is signed in with fresh usage; `codex` has no credentials on this fixture machine.
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        # Configure an isolated `claude-lab` provider in Paseo's config.json, first signed in, then signed out.
        paseo_home = Path(os.environ.get('PASEO_HOME') or (self.home / '.paseo'))
        paseo_home.mkdir(parents=True, exist_ok=True)
        lab_home = self.home / '.claude-lab'
        lab_home.mkdir(parents=True, exist_ok=True)
        (paseo_home / 'config.json').write_text(json.dumps({
            'agents': {
                'providers': {
                    'claude-lab': {
                        'extends': 'claude',
                        'env': {'CLAUDE_CONFIG_DIR': str(lab_home)},
                    },
                },
            },
        }), encoding='utf-8')
        lab_creds = lab_home / '.credentials.json'
        lab_creds.write_text(
            json.dumps({'claudeAiOauth': {'accessToken': 'fixture-lab-token', 'subscriptionType': 'pro'}}),
            encoding='utf-8',
        )
        lab_creds.unlink()

        # 1. Unknown name (`claud` — known to neither `accounts['accounts']` nor discovery): input error (exit 1).
        code_unk, ans_unk = self.run_cli(
            'roll', '--task', 'Review architecture',
            '--provider', 'claude', '--account', 'claud', '--model', 'claude-opus-5-5', '--basis', 'quote',
        )
        self.assertEqual((code_unk, ans_unk['status']), (1, 'error'), ans_unk)
        self.assertNotIn('caller_action', ans_unk)
        self.assertIn('account claud is not configured on this machine', ans_unk['cause'])
        self.assertIn('Ask the user which one they meant; do not substitute another account.', ans_unk['cause'])
        self.assertNotIn('cannot start now', ans_unk['next_action'])

        # 2. Known signed-out isolated account (`claude-lab`) named with its own provider: account fact `blocked / tell_user` (exit 2).
        code_iso, ans_iso = self.run_cli(
            'roll', '--task', 'Review architecture',
            '--provider', 'claude-lab', '--account', 'claude-lab', '--model', 'claude-opus-5-5', '--basis', 'quote',
        )
        self.assertEqual((code_iso, ans_iso['status'], ans_iso.get('caller_action')), (2, 'blocked', 'tell_user'), ans_iso)
        self.assertIn('missing_credentials', ans_iso['cause'])
        self.assertIn('sign in again', ans_iso['cause'])
        self.assertNotIn('not configured on this machine', ans_iso['cause'])
        self.assertNotIn('Ask the user which one they meant', ans_iso['cause'])
        rec_code_iso, rec_iso = self.run_cli(
            'record', '--constraint', 'user_override',
            '--provider', 'claude-lab', '--account', 'claude-lab', '--model', 'claude-opus-5-5', '--basis', 'quote',
        )
        self.assertEqual((rec_code_iso, rec_iso['status'], rec_iso.get('caller_action')), (2, 'blocked', 'tell_user'), rec_iso)
        self.assertEqual(rec_iso['cause'], ans_iso['cause'])

        # 3. Signed-in account (`claude`) named with wrong provider (`codex`): input pairing error (exit 1), never "sign in again".
        code_wrong, ans_wrong = self.run_cli(
            'roll', '--task', 'Review architecture',
            '--provider', 'codex', '--account', 'claude', '--model', 'claude-opus-5-5', '--basis', 'quote',
        )
        self.assertEqual((code_wrong, ans_wrong['status']), (1, 'error'), ans_wrong)
        self.assertNotIn('caller_action', ans_wrong)
        self.assertIn('account claude does not launch through provider codex', ans_wrong['cause'])
        self.assertIn('Name the provider it launches through; do not substitute another account.', ans_wrong['cause'])
        self.assertNotIn('missing_credentials', ans_wrong['cause'])
        self.assertNotIn('sign in', ans_wrong['cause'].lower())

    def test_n2_compact_cause_ends_after_last_whole_sentence_and_points_to_rest(self):
        import roll

        def assert_compact_cause_rule(raw_cause, compacted, *, detail_path, in_next, has_boundary=True):
            self.assertLessEqual(len(compacted), roll.CAUSE_LIMIT, (len(raw_cause), len(compacted), compacted))
            if len(raw_cause) <= roll.CAUSE_LIMIT:
                self.assertEqual(compacted, raw_cause)
                return
            if detail_path:
                self.assertNotIn(detail_path, compacted)
                self.assertIn('the details file', compacted)
            else:
                self.assertNotIn('the details file', compacted)
            if in_next:
                self.assertIn('next_action', compacted)
            else:
                self.assertNotIn('next_action', compacted)
            if '; the rest (' in compacted:
                kept = compacted.split('; the rest (', 1)[0]
                self.assertTrue(raw_cause.startswith(kept + '; '), (kept, raw_cause))
            elif ' The rest (' in compacted:
                kept = compacted.split(' The rest (', 1)[0]
                self.assertTrue(kept.endswith(('.', '!', '?')) and raw_cause.startswith(kept + ' '), (kept, raw_cause))
            else:
                self.assertIn('... (the rest is in ', compacted)
                kept = compacted.split('... (the rest is in ', 1)[0]
                self.assertTrue(bool(kept) and raw_cause.startswith(kept), (kept, raw_cause))
                if has_boundary:
                    self.assertTrue(not kept[-1].isspace() and raw_cause[len(kept)] in ' ,;:',
                                    (kept, raw_cause[len(kept):len(kept) + 10]))

        # Real 3-window refusal (346 chars): `five_hour` below 25% launch protection + `seven_day` & `weekly_fable` exhausted.
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 85.0, 'resets_at': (self.now + timedelta(days=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 100.0, 'resets_at': (self.now + timedelta(days=1)).isoformat()},
            {'id': 'weekly_fable', 'used_pct': 100.0, 'resets_at': (self.now + timedelta(days=2)).isoformat()},
        ], plan='pro')
        code, ans = self.run_cli(
            'roll', '--task', 'Review architecture',
            '--provider', 'claude', '--account', 'claude', '--model', 'claude-fable-5-2', '--effort', 'high',
            '--basis', 'the user said: use claude-fable-5-2',
        )
        self.assertEqual((code, ans['status'], ans.get('caller_action')), (2, 'blocked', 'tell_user'), ans)
        detail_doc = full(ans)
        self.assertEqual(len(detail_doc['cause']), 346, detail_doc['cause'])
        self.assertIn('weekly_fable: confirmed exhausted or below launch floor', detail_doc['cause'])
        self.assertIn(detail_doc['cause'], ans['next_action'])
        assert_compact_cause_rule(
            detail_doc['cause'], ans['cause'], detail_path=ans['details'], in_next=True, has_boundary=True,
        )
        self.assertIn('1 more', ans['cause'])

        def sentence_cause(target_len, sep='; '):
            clauses = []
            idx = 1
            while True:
                body = f'Clause {idx} states why window w{idx} refused the launch'
                if sep == '. ':
                    body += '.'
                joined = (sep if sep == '; ' else ' ').join([*clauses, body])
                rem = target_len - len(joined)
                if rem <= 90:
                    tail = ('detail ' * 20)[:rem - (len(sep) if sep == '; ' else 1)]
                    if sep == '. ':
                        tail = tail[:-1].rstrip() + '.'
                        pad = target_len - len(' '.join([*clauses, body, tail]))
                        tail = tail[:-1] + ('x' * pad) + '.'
                        out = ' '.join([*clauses, body, tail])
                    else:
                        tail = tail.rstrip()
                        pad = target_len - len('; '.join([*clauses, body, tail]))
                        tail = tail + ('x' * pad)
                        out = '; '.join([*clauses, body, tail])
                    self.assertEqual(len(out), target_len)
                    return out
                clauses.append(body)
                idx += 1

        def word_cause(target_len):
            stream = ' '.join(f'word{i:03d}' for i in range(200))
            head = stream[:target_len - 6].rstrip()
            pad = target_len - len(head) - 1
            out = f"{head} {'w' * pad}"
            self.assertEqual(len(out), target_len)
            return out

        long_detail_path = 'D:/work/isolated-state/.agent-compass/roll-results/1795493207-5492aad9-long-detail-record-path.json'
        cases = [
            ('299_sentences', sentence_cause(299), True),
            ('299_words', word_cause(299), True),
            ('300_sentences', sentence_cause(300), True),
            ('300_words', word_cause(300), True),
            ('301_semicolons', sentence_cause(301, '; '), True),
            ('301_dots', sentence_cause(301, '. '), True),
            ('301_long_first_sentence', word_cause(250) + '; ' + ('z' * 49), True),
            ('301_words', word_cause(301), True),
            ('346_three_windows', detail_doc['cause'], True),
            ('900_semicolons', sentence_cause(900, '; '), True),
            ('900_dots', sentence_cause(900, '. '), True),
            ('900_words', word_cause(900), True),
            ('900_no_boundary', 'x' * 900, False),
        ]
        for label, raw_cause, has_boundary in cases:
            for in_next, detail_path in ((True, long_detail_path), (False, long_detail_path), (True, None)):
                with self.subTest(case=label, in_next=in_next, has_file=bool(detail_path)):
                    nxt = f'Tell the user ({raw_cause}).' if in_next else 'Fix the issue and retry.'
                    comp = roll.compact({'error': raw_cause, 'next': nxt}, 'error', detail_path)
                    assert_compact_cause_rule(
                        raw_cause, comp['cause'], detail_path=detail_path, in_next=in_next, has_boundary=has_boundary,
                    )

        # With no details file and full cause not in `next_action`, nothing holds the rest, so nothing is cut.
        multi_dot = sentence_cause(346, '. ')
        uncut = roll.compact({'error': multi_dot, 'next': 'Fix it.'}, 'error', None)
        self.assertEqual(uncut['cause'], multi_dot)

    def test_f5_cwd_regex_is_blocked_tell_user_without_cannot_start_now(self):
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        override_path = Path(os.environ['AGENTCOMPASS_ACCOUNTS_OVERRIDE'])
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(
            json.dumps({'schema_version': 2, 'accounts': {'claude': {'cwd_regex': '^D:/work/only(/|$)'}}}),
            encoding='utf-8',
        )
        try:
            code, ans = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
            self.assertEqual((code, ans['status'], ans.get('caller_action')), (2, 'blocked', 'tell_user'), ans)
            self.assertIn('cwd_regex', ans['cause'])
            # The whole cause is in the details file: the compact one is cut at 300 characters, which a long worker
            # directory in the sentence reaches (roll._compact_cause).
            self.assertIn('a named route does not reach past it', full(ans)['cause'])
            self.assertNotIn('substitute', ans['cause'].lower())
            self.assertNotIn('cannot start now', ans['next_action'])
            self.assertNotIn('Change what cause names', ans['next_action'])
            self.assertNotIn('start_after_hours', ans)

            rec_code, rec_ans = self.run_cli('record', '--constraint', 'user_override', *self.NAMED_FLAGS)
            self.assertEqual((rec_code, rec_ans['status'], rec_ans.get('caller_action')), (2, 'blocked', 'tell_user'), rec_ans)
            self.assertIn("this process's working directory", rec_ans['cause'])
            self.assertNotIn('cannot start now', rec_ans['next_action'])
        finally:
            override_path.unlink(missing_ok=True)

    def test_f6_multiple_refusing_windows_names_time_deciding_window_first_and_others_after(self):
        # Case 1: `five_hour` resets in 1h (90% used, below 25% floor) and `seven_day` resets in 30h (100% used).
        # `seven_day` decides the 30.0h wait and must appear first in `cause`, with `five_hour` after it.
        r_5h = self.now + timedelta(hours=1)
        r_7d = self.now + timedelta(hours=30)
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 90.0, 'resets_at': r_5h.isoformat()},
            {'id': 'seven_day', 'used_pct': 100.0, 'resets_at': r_7d.isoformat()},
        ])
        code1, ans1 = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
        self.assertEqual((code1, ans1['status'], ans1.get('caller_action')), (2, 'blocked', 'tell_user'), ans1)
        self.assertAlmostEqual(ans1['start_after_hours'], 30.0, places=1)
        self.assertEqual(ans1['retry_at'], r_7d.isoformat())
        self.assertIn('seven_day', ans1['cause'])
        self.assertIn('five_hour', ans1['cause'])
        self.assertLess(ans1['cause'].index('seven_day'), ans1['cause'].index('five_hour'), ans1['cause'])

        # Case 2 (reverse): `five_hour` resets in 4h and `seven_day` resets in 2h -> `five_hour` decides 4.0h and leads.
        r_5h_late = self.now + timedelta(hours=4)
        r_7d_early = self.now + timedelta(hours=2)
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 90.0, 'resets_at': r_5h_late.isoformat()},
            {'id': 'seven_day', 'used_pct': 100.0, 'resets_at': r_7d_early.isoformat()},
        ])
        code2, ans2 = self.run_cli('roll', '--task', 'Review architecture', *self.NAMED_FLAGS)
        self.assertEqual((code2, ans2['status'], ans2.get('caller_action')), (2, 'blocked', 'tell_user'), ans2)
        self.assertAlmostEqual(ans2['start_after_hours'], 4.0, places=1)
        self.assertEqual(ans2['retry_at'], r_5h_late.isoformat())
        self.assertIn('five_hour', ans2['cause'])
        self.assertIn('seven_day', ans2['cause'])
        self.assertLess(ans2['cause'].index('five_hour'), ans2['cause'].index('seven_day'), ans2['cause'])

    def test_f8_prepare_floor_refusal_on_forced_route_never_claims_user_named(self):
        import account_fleet
        import usage_reading
        from router_test_support import profile
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(hours=3)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        args_hc = route_judge.build_parser().parse_args([
            'record', '--constraint', 'hard_constraint', '--provider', 'claude', '--account', 'claude',
            '--model', 'claude-opus-5-5', '--effort', 'high', '--basis', 'project rule',
        ])
        task_hc = {'parent_agent_id': 'parent-named', 'task_id': 't-f8-hc', 'objective': 'Review',
                   'expected_hours': 0.05, 'hours': 0.05, 'profile': profile()}
        dec_hc = route_judge.record_constraint(args_hc, task=task_hc, classified={'task': task_hc, 'scorer': None})
        self.write_usage_cache([
            {'id': 'five_hour', 'used_pct': 90.0, 'resets_at': (self.now + timedelta(hours=2)).isoformat()},
            {'id': 'seven_day', 'used_pct': 10.0, 'resets_at': (self.now + timedelta(days=4)).isoformat()},
        ])
        with self.assertRaises(registry.CapacityChanged) as caught:
            registry.prepare(
                dec_hc['decision_id'], task_hc, {'title': 'Review', 'initialPrompt': 'Review'},
                usage_reading.usage_snapshot(self.cache_path, now=self.now),
                accounts=account_fleet.load_accounts(), now=self.now,
            )
        self.assertNotIsInstance(caught.exception.__cause__, registry.RouteFactRefused)

    def test_every_final_error_code_in_roll_appears_in_roll_reference(self):
        import roll
        ref_path = Path(route_judge.__file__).resolve().parents[1] / 'references' / 'roll-reference.md'
        text = ref_path.read_text(encoding='utf-8')
        for code in roll.FINAL_ERRORS.values():
            with self.subTest(code=code):
                self.assertIn(f'`{code}`', text)


if __name__ == '__main__':
    unittest.main()
