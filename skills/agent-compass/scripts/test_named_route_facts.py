"""A route the user names launches as long as the provider supports it; only facts refuse it, and the refusal states the
fact (router-internals.md, record). Rules, never outcomes: which fact refuses, the wording every stage shares, and that a
named route with only a data gap launches carrying that data error."""
import argparse
import copy
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import account_fleet
import allocator
import auto_bundles
import axis_scores
import decision_ledger
import dispatch_registry
import candidate_eval
import candidate_ranking
import fast_modes
import launch_failures
import live_models
import route_judge
import scorer_client
import task_intake
from test_routing_parity import NOW
from test_routing_review import RecordCase

LISTS = {'claude': ['claude-opus-5-5', 'claude-zeta-1'], 'codex': ['gpt-6-astra', 'gpt-6.2-nova']}


def fixture_fetch(providers, cwd=None, timeout=10):
    return {'ok': True, 'providers': {p: {'ok': True, 'models': [{'id': m} for m in LISTS.get(p, [])]} for p in providers}}


class NamedRouteCase(RecordCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(live_models.invalidate)

    def launched(self, accounts=None, task=None, **fields):
        task = task or self.task(parent_agent_id='parent-1', task_id='t-' + os.urandom(4).hex(), disable_fast=False)
        fleet = accounts or self.accounts
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, \
                patch.dict(os.environ, {'ROUTER_DISPATCH_DB': os.path.join(directory, 'db.sqlite3')}):
            result = self.record(task, fleet, **fields)
            with patch.object(account_fleet, 'load_accounts', return_value=copy.deepcopy(fleet)):
                row = dispatch_registry.prepare(result['decision_id'], task, {'title': 'x', 'initialPrompt': 'go'},
                                                copy.deepcopy(self.usage), accounts=copy.deepcopy(fleet), now=NOW)
        return result, row


class FastModeFactTests(NamedRouteCase):
    """Item 9: whether a harness launches a model fast is one fact with one owner (fast_modes.refusal); a missing price
    is a data error, not a refusal."""

    def contracted(self):
        """A model whose fast contract holds on its harness, with its fast-mode price taken out of the data."""
        for bundle in account_fleet.bundles(self.accounts).values():
            harness = account_fleet.bundle_harness(bundle, bundle['account'])
            info = account_fleet.model_spec(self.accounts, bundle['model'])
            if bundle.get('effort') and fast_modes.supported(harness, bundle['model'], info):
                model = account_fleet.canonical(bundle['model'], self.accounts)
                real = axis_scores.five_axis_overlay

                def unpriced(*args, **kwargs):     # the shipped scores overlay carries the price: take it out there
                    out = real(*args, **kwargs)
                    out['models'][model].setdefault('five_axis', {})['fast_mode'] = {'status': 'missing'}
                    return out
                patcher = patch.object(axis_scores, 'five_axis_overlay', side_effect=unpriced)
                patcher.start()
                self.addCleanup(patcher.stop)
                return self.accounts, bundle
        self.skipTest('the reference fleet has no row with a verified fast contract')

    def uncontracted(self):
        """A route without a fast contract, a Claude one first (Claude's fast contract was withdrawn, owner 2026-11-24)."""
        for bundle in sorted(account_fleet.bundles(self.accounts).values(),
                             key=lambda b: account_fleet.bundle_harness(b, b['account']) != 'claude'):
            harness = account_fleet.bundle_harness(bundle, bundle['account'])
            if bundle.get('effort') and not fast_modes.supported(
                    harness, bundle['model'], account_fleet.model_spec(self.accounts, bundle['model'])):
                return bundle, harness
        self.skipTest('the reference fleet has no row without a fast contract')

    def test_no_contract_launches_plain_as_if_fast_were_not_given_with_one_note(self):
        """Owner, 2026-11-24: a route the user names with --fast whose harness cannot launch the model fast runs in its
        plain mode as if --fast had not been given: no refusal, one note saying why, the plain mode's reservation and
        expected spend. A stored fast route of such a model (an older decision) is still refused with the reason."""
        bundle, harness = self.uncontracted()
        model = account_fleet.canonical(bundle['model'], self.accounts)
        reason = fast_modes.refusal(harness, model, account_fleet.model_spec(self.accounts, model))
        self.assertEqual(reason, f'{model} has no verified fast mode on {harness}')
        named = dict(account=bundle['account'], provider=bundle['provider'], model=bundle['model'],
                     effort=bundle['effort'])
        fast_result, fast_row = self.launched(fast=True, **named)
        plain_result, plain_row = self.launched(**named)
        self.assertNotIn('fast', fast_result['selected_bundle'])
        self.assertNotIn('force_fast', fast_result.get('_route') or {})
        self.assertEqual(fast_result['fast_not_applied'], reason)
        note = fast_modes.not_applied(reason)
        self.assertEqual([w for w in fast_result.get('warnings') or () if '--fast' in w], [note])
        self.assertEqual([w for w in fast_result.get('warnings') or () if w != note],
                         list(plain_result.get('warnings') or ()))
        self.assertFalse(fast_row['bundle'].get('fast'))
        self.assertEqual(fast_row['request']['settings'].get('features'),
                         plain_row['request']['settings'].get('features'))
        self.assertEqual(fast_row['expected_usd'], plain_row['expected_usd'])
        self.assertEqual(fast_result['selected_bundle'].get('expected_usd'),
                         plain_result['selected_bundle'].get('expected_usd'))
        with self.assertRaises(ValueError) as stored:
            dispatch_registry.resolve_bundle(dict(bundle, fast=True), self.accounts, explicit_override=True)
        self.assertEqual(str(stored.exception), reason)

    def test_a_named_fast_route_only_the_fast_burn_estimate_holds_launches_fast_with_one_warning(self):
        """Owner, 2026-11-24: a route the user named with --fast that only the fast mode's consumption estimate holds
        launches fast, with one warning in the estimate's numbers; record and prepare agree; a fact (a window at 0) still
        refuses it; automatic routing offers no fast twin on that reading. No shipped row cites its fast multipliers, so
        a synthetic cited row stands in (`cited_fast`), on a Codex plan whose capacity the shipped data cites."""
        from router_test_support import CITED_FAST, cited_fast, usage_windows
        fleet = copy.deepcopy(self.accounts)
        fleet['accounts']['codex']['plan'] = 'pro'
        named = dict(account='codex', provider='codex', model='gpt-6.1-sol', effort='high')

        def run(remaining, **fields):
            self.usage = usage_windows(NOW, {'codex': [('usage', remaining, 100)]})
            task = self.task(parent_agent_id='parent-1', task_id='t-' + os.urandom(4).hex(), disable_fast=False,
                             expected_hours=8)
            args = argparse.Namespace(**{'basis': 'User asked', 'input': None, 'usage_cache': None, 'accounts': None,
                                         'constraint': 'user_override', 'specialty': None, 'reason_category': None,
                                         'fast': False, 'parent_id': None, **named, **fields})
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, \
                    patch.dict(os.environ, {'ROUTER_DISPATCH_DB': os.path.join(directory, 'db.sqlite3')}), \
                    patch.object(route_judge.usage_reading, 'usage_snapshot', return_value=copy.deepcopy(self.usage)), \
                    patch.object(account_fleet, 'load_accounts', return_value=copy.deepcopy(fleet)), \
                    patch.object(route_judge, 'datetime', wraps=datetime) as clock, \
                    patch.object(decision_ledger, 'datetime', wraps=datetime) as ledger_clock:
                clock.now.return_value = ledger_clock.now.return_value = NOW
                result = route_judge.record_constraint(args, task, {'task': task})     # planned: the plan row is read
                row = dispatch_registry.prepare(result['decision_id'], task, {'title': 'x', 'initialPrompt': 'go'},
                                                copy.deepcopy(self.usage), accounts=copy.deepcopy(fleet), now=NOW)
            return task, row

        with cited_fast('gpt-6.1-sol'):
            task, plain = run(15)
            _, fast = run(15, fast=True)
            self.assertTrue(fast['bundle'].get('fast'))
            self.assertEqual(fast['request']['settings']['features'], {'service_tier': 'priority'})
            warned = [w for w in fast.get('warnings') or () if 'fast mode' in w]
            self.assertEqual(len(warned), 1, fast.get('warnings'))
            self.assertIn("this is the router's estimate, so the run may stop at the limit", warned[0])
            self.assertIn('launched fast because the user named it with --fast', warned[0])
            self.assertAlmostEqual(fast['expected_usd'], plain['expected_usd'] * CITED_FAST['price_multiplier'], places=1)
            automatic = self.plan(task, fleet)
            self.assertIsNone(candidate_ranking.candidate_for_bundle(automatic['ranking'], 'sol_high', True))
            with self.assertRaises(dispatch_registry.RouteFactRefused):
                run(0, fast=True)

    def test_roll_says_the_note_once_and_a_resume_reads_fast_as_absent_on_a_route_without_a_contract(self):
        import roll
        bundle, harness = self.uncontracted()
        model = account_fleet.canonical(bundle['model'], self.accounts)
        reason = fast_modes.refusal(harness, model, account_fleet.model_spec(self.accounts, model))
        from_roll = self.record(account=bundle['account'], provider=bundle['provider'], model=bundle['model'],
                                effort=bundle['effort'], fast=True, _from_roll=True)
        self.assertFalse([w for w in from_roll.get('warnings') or () if '--fast' in w])     # roll's own note says it
        args = argparse.Namespace(fast=True, basis=None, provider=None, account=None, model=None, effort=None)
        self.assertEqual(roll.ignored_flag_notes(args, from_roll, ''), [fast_modes.not_applied(reason)])
        with patch.object(account_fleet, 'load_accounts', return_value=copy.deepcopy(self.accounts)):
            resume = argparse.Namespace(provider=None, account=None, model=None, effort=None, fast=True)
            self.assertIsNone(roll.named_route_conflict(resume, dict(bundle, bundle_id=bundle['id']), self.accounts))
            self.assertEqual(roll.ignored_flag_notes(resume, {'selected_bundle': dict(bundle)}, 'agent-1'),
                             [fast_modes.not_applied(reason)])

    def test_a_contract_without_a_price_launches_fast_carrying_the_data_error(self):
        fleet, bundle = self.contracted()
        result, row = self.launched(fleet, account=bundle['account'], provider=bundle['provider'],
                                    model=bundle['model'], effort=bundle['effort'], fast=True)
        self.assertTrue(row['bundle'].get('fast'))
        self.assertTrue(any('fast_mode has no cited public pricing' in e for e in result.get('data_errors') or ()),
                        result.get('data_errors'))


    def test_an_unchecked_model_named_fast_launches_fast(self):
        """The routing preferences govern automatic routing only (router-internals.md, Pins): the bundle a user_override
        launches keeps its fast variant, so record and launch agree."""
        bundle = next((b for b in account_fleet.bundles(self.accounts).values() if b.get('effort') and fast_modes.supported(
            account_fleet.bundle_harness(b, b['account']), b['model'], account_fleet.model_spec(self.accounts, b['model']))), None)
        if bundle is None:
            self.skipTest('the reference fleet has no row with a verified fast contract')
        fleet = copy.deepcopy(self.accounts)
        row = fleet['accounts'][bundle['account']]['bundles'][bundle['id']]
        row.update(disabled=True, enabled=False)
        row.pop('disabled_by_default', None)
        result, launched = self.launched(fleet, account=bundle['account'], provider=bundle['provider'],
                                         model=bundle['model'], effort=bundle['effort'], fast=True)
        self.assertTrue(launched['bundle'].get('fast'))
        self.assertTrue(any('unchecked in model management' in w for w in result.get('warnings') or ()), result.get('warnings'))

    def test_the_registry_writes_a_fast_route_as_the_plain_bundle_id_with_its_fast_field(self):
        """Round 6, finding 4: fast mode is the bundle's `fast` field; `<id>+fast` (fast_modes.ID_SUFFIX) is only the
        ranking's name for the variant. The registry stores the catalog row's plain id with `fast: True`, whether the
        route was a named record or a plan's fast variant (allocator selected_bundle: base_id), so a resume reads
        the worker's fast mode from the field (roll.named_route_conflict)."""
        import argparse
        import roll
        bundle = next((b for b in account_fleet.bundles(self.accounts).values() if b.get('effort') and fast_modes.supported(
            account_fleet.bundle_harness(b, b['account']), b['model'], account_fleet.model_spec(self.accounts, b['model']))), None)
        if bundle is None:
            self.skipTest('the reference fleet has no row with a verified fast contract')
        _, launched = self.launched(account=bundle['account'], provider=bundle['provider'],
                                    model=bundle['model'], effort=bundle['effort'], fast=True)
        variant = fast_modes.variant_id(bundle['id'])
        task = self.task(parent_agent_id='parent-1', task_id='t-' + os.urandom(4).hex())
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, \
                patch.dict(os.environ, {'ROUTER_DISPATCH_DB': os.path.join(directory, 'db.sqlite3')}), \
                patch.object(account_fleet, 'load_accounts', return_value=copy.deepcopy(self.accounts)):
            issued = dispatch_registry.issue({'stage': 'plan', 'choice': variant, 'decision': {'action': 'start'},
                                              'selected_bundle': dict(bundle, bundle_id=bundle['id'], fast=True)}, task)
            planned = dispatch_registry.decision_record(issued['decision_id'])['bundle']
        stored = {'named record, launched': launched['bundle'], 'plan fast variant': planned}
        for writer, written in stored.items():
            with self.subTest(writer=writer):
                self.assertEqual(written['bundle_id'], bundle['id'])
                self.assertFalse(written['bundle_id'].endswith(fast_modes.ID_SUFFIX))
                self.assertIs(written.get('fast'), True)
                fast_resume = argparse.Namespace(provider=None, account=None, model=None, effort=None, fast=True)
                self.assertIsNone(roll.named_route_conflict(fast_resume, written, self.accounts))

    def priced_contract(self):
        for bundle in account_fleet.bundles(self.accounts).values():
            if bundle.get('effort') and fast_modes.supported(account_fleet.bundle_harness(bundle, bundle['account']),
                                                             bundle['model'], account_fleet.model_spec(self.accounts, bundle['model'])):
                return bundle
        self.skipTest('the reference fleet has no row with a verified fast contract')

    def test_the_economy_refuses_a_forced_fast_route_with_the_same_wording(self):
        """The third stage of the one answer: candidate_eval, for the bundle a named fast route launches."""
        bundle, harness = self.uncontracted()
        model = account_fleet.canonical(bundle['model'], self.accounts)
        fact = fast_modes.refusal(harness, model, account_fleet.model_spec(self.accounts, model))
        plan = self.plan(self.task(disable_fast=False), route={
            'force_fast': True, 'pinned_bundle': bundle['id'], 'user_named_bundle': bundle['id']})
        self.assertIn(fact, self.row(plan, bundle['id'])['excluded'])

    def test_an_unpriced_fast_launch_reserves_the_plain_modes_spend(self):
        """fast-modes.md: a named fast route whose multipliers are not cited is reserved at the plain mode's spend."""
        fleet, bundle = self.contracted()
        named = dict(account=bundle['account'], provider=bundle['provider'], model=bundle['model'], effort=bundle['effort'])
        _, plain = self.launched(fleet, **named)
        _, fast = self.launched(fleet, fast=True, **named)
        self.assertTrue(fast['bundle'].get('fast'))
        self.assertFalse(plain['bundle'].get('fast'))
        self.assertEqual(fast['expected_usd'], plain['expected_usd'])

    def test_a_named_fast_route_on_an_account_without_quota_windows_launches_fast(self):
        """No window prices it (a metered account without a budget: quota_pools.metered_usd_uncapped): the route the
        user named is still that route, and automatic routing offers no fast twin there."""
        bundle = self.priced_contract()
        fleet = copy.deepcopy(self.accounts)
        fleet['accounts'][bundle['account']].update(backend_kind='metered_usd', billing='metered')
        fleet['accounts'][bundle['account']]['bundles'][bundle['id']].update(disabled=False, enabled=True)
        self.usage = copy.deepcopy(self.usage)
        self.usage.get('accounts', {}).pop(bundle['account'], None)     # no window reading either
        probe = self.plan(self.task(disable_fast=False), fleet, route={
            'force_fast': True, 'pinned_bundle': bundle['id'], 'user_named_bundle': bundle['id']})
        self.assertEqual(self.row(probe, bundle['id'] + '+fast').get('economy', {}).get('windows'), [])
        _, row = self.launched(fleet, account=bundle['account'], provider=bundle['provider'], model=bundle['model'],
                               effort=bundle['effort'], fast=True)
        self.assertTrue(row['bundle'].get('fast'))
        plan = self.plan(self.task(disable_fast=False, metered_ok=True), fleet)
        self.assertFalse([c for c in plan['ranking'] if c['bundle']['account'] == bundle['account']
                          and c['bundle'].get('fast')])

class EffortFactTests(NamedRouteCase):
    """Item 4: an effort outside the launch contract is refused, naming the efforts the harness does accept."""

    def test_record_and_launch_share_the_unaccepted_effort_wording(self):
        bundle = next(b for b in account_fleet.bundles(self.accounts).values()
                      if b.get('effort') and account_fleet.dispatch_allowed_efforts(b, self.accounts))
        harness = account_fleet.bundle_harness(bundle, bundle['account'])
        model = account_fleet.canonical(bundle['model'], self.accounts)
        accepted = set(account_fleet.dispatch_allowed_efforts(bundle, self.accounts))
        with self.assertRaises(ValueError) as at_record:
            self.record(account=bundle['account'], provider=bundle['provider'], model=bundle['model'], effort='turbo')
        with self.assertRaises(ValueError) as at_launch:
            dispatch_registry.resolve_bundle(dict(bundle, effort='turbo'), self.accounts, explicit_override=True)
        for raised in (at_record, at_launch):
            text = str(raised.exception)
            self.assertTrue(text.startswith(f'turbo is not an effort {harness}/'), text)
            self.assertIn(model, text)
            listed = text[text.index('(') + 1:text.rindex(')')].split(', ')
            self.assertTrue(accepted <= set(listed), (accepted, listed))
            self.assertEqual(listed, [e for e in auto_bundles.EFFORTS if e in listed] + [e for e in listed if e not in auto_bundles.EFFORTS])


class ProviderListFactTests(NamedRouteCase):
    """Items 5 and 5c: a named model with no row is decided by the named account's live model list."""

    def test_a_model_the_provider_does_not_list_is_unavailable_and_says_so(self):
        with patch.object(live_models, 'fetch_from_paseo', side_effect=fixture_fetch):
            with self.assertRaisesRegex(ValueError, r'^named_model_unavailable: .*provider claude does not list claude-zeta-2'):
                self.record(account='claude', provider='claude', model='claude-zeta-2', effort='high')

    def test_an_unreadable_list_refuses_with_why_it_could_not_be_read(self):
        with patch.object(live_models, 'fetch_from_paseo', return_value={'ok': False, 'error': 'host_connection_failed'}):
            with self.assertRaisesRegex(ValueError, r'^named_model_unavailable: .*model list of provider claude could not be read'):
                self.record(account='claude', provider='claude', model='claude-zeta-1', effort='high')

    def test_a_listed_model_launches_on_its_derived_row_unscored_with_a_warning(self):
        for provider, model in (('codex', 'gpt-6.2-nova'), ('claude', 'claude-zeta-1')):
            with self.subTest(model=model), patch.object(live_models, 'fetch_from_paseo', side_effect=fixture_fetch):
                self.assertFalse([b for b in account_fleet.bundles(self.accounts).values() if b['model'] == model])
                result, row = self.launched(account=provider, provider=provider, model=model, effort='high')
                self.assertEqual((row['bundle']['account'], row['bundle']['model'], row['bundle'].get('effort')),
                                 (provider, model, 'high'))
                self.assertTrue(any('has no row in the shipped data' in w and f'provider {provider} lists it' in w
                                    for w in result.get('warnings') or ()), result.get('warnings'))
                self.assertTrue(any(e.startswith(f'{provider}/{model} has no row in the shipped data: unscored and unpriced')
                                    for e in result.get('data_errors') or ()), result.get('data_errors'))
                # One calculation only: the shipped fleet never gains the row.
                self.assertFalse([b for b in account_fleet.bundles(self.accounts).values() if b['model'] == model])
            live_models.invalidate()

    def test_a_listed_route_with_a_value_that_cannot_be_passed_safely_is_refused_where_it_launches(self):
        """Rule (router-internals §5, Launch values): a model a provider lists and an effort the user names are values
        from the user's machine: never dropped by a pattern, made safe where they are used, and a value that cannot be
        passed as its own argument refuses that launch there (launch_values.LaunchValueError), before anything starts."""
        import launch_values
        import public_data
        import roll
        with patch.object(live_models, 'fetch_from_paseo', side_effect=fixture_fetch):
            result, row = self.launched(account='codex', provider='codex', model='gpt-6.2-nova', effort='-c')
        self.assertEqual(row['bundle'].get('effort'), '-c')
        started, before = [], public_data._EFFECTS[0]
        out = {'parent_id': 'p', 'parent_host': 'paseo', 'parent_address': 'p'}
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(roll.host_placement, 'place', lambda *a, **k: {'cwd': folder}), \
                patch.object(roll, 'CLI_RUNS_DIR', Path(folder) / 'runs'), \
                patch.object(roll.subprocess, 'Popen', lambda *a, **k: started.append(a)):
            with self.assertRaises(launch_values.LaunchValueError):
                roll.cli(argparse.Namespace(workspace=None), out, row['bundle'], 'task')
        self.assertEqual((started, public_data._EFFECTS[0]), ([], before))

    def test_a_named_record_and_an_explicit_resume_mark_their_decision_as_an_effect(self):
        """Rule (router-internals §5, no effect is repeated) on main's named-route paths: a named record (a listed model
        included) and `roll --resume`'s record each write their decision through the one ledger writer, which marks it."""
        import public_data
        marked = []
        real = public_data.effect
        with patch.object(public_data, 'effect', lambda what: marked.append(what) or real(what)):
            with patch.object(live_models, 'fetch_from_paseo', side_effect=fixture_fetch):
                self.record(account='codex', provider='codex', model='gpt-6.2-nova', effort='high')
            self.assertEqual(marked, ['dispatch registry write', 'ledger row'])     # the decision row, then its record
            del marked[:]
            bundle = next(b for b in account_fleet.bundles(self.accounts).values() if b['account'] == 'codex')
            task = self.task(parent_agent_id='parent-1')
            args = argparse.Namespace(usage_cache=None, accounts=None)
            with patch.object(account_fleet, 'load_accounts', return_value=copy.deepcopy(self.accounts)), \
                    patch.object(route_judge.usage_reading, 'usage_snapshot', return_value=copy.deepcopy(self.usage)):
                route_judge.record_resume(args, task, {'task': task, 'scorer': None}, dict(bundle), 'agent-1')
        self.assertIn('ledger row', marked)

    def test_the_providers_list_outranks_a_guessed_family_only_when_named(self):
        catalog = {'providers': {'claude': {'ok': True, 'checked_at': 0, 'models': ['claude-zeta-1']}}}
        families = {n: r.get('family') for n, r in ((self.accounts.get('_discovery') or {}).get('accounts') or {}).items()}

        def rows(named):
            out = auto_bundles.derive(copy.deepcopy(self.accounts), catalog, 0, account_families=families, named=named)
            return [b for b in (out['bundles'].get('claude') or {}).values() if b['model'] == 'claude-zeta-1']
        self.assertEqual(rows(()), [])
        self.assertTrue(rows({('claude', 'claude-zeta-1')}))


class UnclassifiedFactTests(NamedRouteCase):
    """Item 1: no classifier answer is a data gap a named route crosses with --hours, never an invented estimate."""

    def setUp(self):
        super().setUp()
        # The real planning of a task without a profile, not conftest's fixture profile for bare tasks.
        unwrapped = patch.object(task_intake, 'normalize_task', task_intake.normalize_task.__wrapped__)
        unwrapped.start()
        self.addCleanup(unwrapped.stop)

    def pinned(self, task):
        args = argparse.Namespace(basis='User asked', provider='codex', account='codex', model='gpt-6-astra',
                                  effort='high', input=None, usage_cache=None, accounts=None, constraint='user_override',
                                  specialty=None, no_score=True, dry_run=False, keep_whole=None, fast=False,
                                  reason_category=None, parent_id=None)
        return args, route_judge.classify_pinned(task, args)

    def test_with_hours_it_launches_unclassified_and_the_ledger_says_so(self):
        task = {'objective': 'x', 'task_id': 't-' + os.urandom(4).hex(), 'expected_hours': 1.0,
                'parent_agent_id': 'parent-1'}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, \
                patch.dict(os.environ, {'ROUTER_DISPATCH_DB': os.path.join(directory, 'db.sqlite3'),
                                        'ROUTER_LEDGER': os.path.join(directory, 'ledger.jsonl')}), \
                patch.object(scorer_client, 'score_task', side_effect=AssertionError('no classifier call')):
            args, classified = self.pinned(task)
            self.assertNotIn('result', classified)
            result = route_judge.record_constraint(args, task, classified)
            ledger = [r for r in decision_ledger.read_ledger() if r.get('decision_id') == result['decision_id']]
            row = dispatch_registry.prepare(result['decision_id'], task, {'title': 'x', 'initialPrompt': 'go'},
                                            copy.deepcopy(self.usage), accounts=copy.deepcopy(self.accounts), now=NOW)
        self.assertTrue(any(e.startswith('unclassified (') and '--hours 1 ' in e for e in result.get('data_errors') or ()))
        self.assertTrue(ledger and any(e.startswith('unclassified (') for e in ledger[-1].get('data_errors') or ()), ledger)
        self.assertEqual(row['bundle']['model'], 'gpt-6-astra')

    def test_without_hours_it_is_refused_and_the_refusal_names_hours(self):
        _, classified = self.pinned({'objective': 'x', 'task_id': 't-1', 'parent_agent_id': 'parent-1'})
        self.assertIn('result', classified)
        self.assertIn('--hours', classified['result'].get('next') or '')
        self.assertIn('--hours was not given', classified['result'].get('reason') or '')



class CallerTaskBoundaryTests(NamedRouteCase):
    """Finding 1 (round 5): a field AgentCompass sets for itself is never settable from a caller's task. It is refused by
    name where a caller's task enters (plan --input, record), and router code passes it only as a route field
    (allocator.plan(route=...)); the planner never reads one from a task (round 6: a stored task of an earlier version
    is dispatch_registry.earlier_version's, test_derived_state)."""

    KEYS = (*task_intake.NOT_CALLER_INPUT, 'needs.force_fast')

    @staticmethod
    def carrying(key, task):
        if key.startswith('needs.'):
            return dict(task, needs={**(task.get('needs') or {}), key.split('.', 1)[1]: True})
        return dict(task, **{key: 'opus55_high' if key.endswith('bundle') else True})

    def test_every_router_field_in_a_callers_task_is_refused_by_name_where_it_enters(self):
        self.assertTrue(set(task_intake.ROUTE_FIELDS) <= set(task_intake.NOT_CALLER_INPUT))
        path = os.path.join(self.root, 'in.json')
        seen = lambda plan: (plan['decision'], [(r['bundle'].get('bundle_id') or r['bundle'].get('id'), r.get('excluded')) for r in plan['ranking']])
        plain = seen(self.plan(self.task()))
        for key in self.KEYS:
            task = self.carrying(key, self.task())
            text = f'^invalid_task_input: {key} is set by AgentCompass itself'
            with self.subTest(key=key, reader='allocator.plan reads no route field from a task'):
                self.assertEqual(seen(self.plan(task)), plain)
            with self.subTest(key=key, reader='plan --input'), self.assertRaisesRegex(ValueError, text):
                route_judge._plan_input_task({'task': task}, argparse.Namespace(hours=None, can_wait_hours=None))
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(__import__('json').dumps({'task': task}))
            with self.subTest(key=key, reader='record --input'), self.assertRaisesRegex(ValueError, text):
                route_judge._record_task(argparse.Namespace(input=__import__('pathlib').Path(path), parent_id='p-1'), None)

    def test_the_two_preferences_lift_only_for_the_bundle_a_named_decision_passes(self):
        """The lift follows from a user_override or explicit-resume decision (prepare passes user_named_bundle), never
        from pinning alone and never from a task field."""
        bundle = next(b for b in account_fleet.bundles(self.accounts).values() if b['account'] == 'claude')
        off = {'allowed_accounts': [a for a in self.accounts['accounts'] if a != 'claude']}
        for route, lifted in (({'pinned_bundle': bundle['id']}, False), (None, False),
                              ({'pinned_bundle': bundle['id'], 'user_named_bundle': bundle['id']}, True)):
            with self.subTest(route=route):
                plan = self.plan(self.task(), host_policy_config=off, route=route)
                self.assertEqual(candidate_eval.OUT_OF_AUTOMATIC_ROUTING not in self.row(plan, bundle['id'])['excluded'],
                                 lifted)

    def test_only_a_named_decision_keeps_route_fields(self):
        route = {'unclassified': 'fixture', 'force_fast': True}
        task = self.task(parent_agent_id='parent-1', task_id='t-' + os.urandom(4).hex())
        bundle = dict(next(iter(account_fleet.bundles(self.accounts).values())))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, \
                patch.dict(os.environ, {'ROUTER_DISPATCH_DB': os.path.join(directory, 'db.sqlite3')}):
            for result, kept in (
                ({'stage': 'plan', 'choice': bundle['id']}, False),
                ({'stage': 'record', 'choice': 'constraint', 'constraint': 'hard_constraint', 'basis': 'rule'}, False),
                ({'stage': 'record', 'choice': 'constraint', 'constraint': 'user_override', 'basis': 'User asked'}, True),
            ):
                with self.subTest(stage=result['stage'], constraint=result.get('constraint')), \
                        patch.object(account_fleet, 'load_accounts', return_value=copy.deepcopy(self.accounts)):
                    issued = dispatch_registry.issue(dict(result, selected_bundle=dict(bundle, bundle_id=bundle['id'])),
                                                     task, route=route)
                    self.assertEqual('route' in dispatch_registry.decision_record(issued['decision_id']), kept)

    def test_planning_without_a_profile_is_never_reached_by_an_automatic_plan(self):
        """unclassified is a route field only a named record passes: an automatic plan of a task with no profile is
        refused, and a caller cannot ask for it."""
        task = {k: v for k, v in self.task().items() if k != 'profile'}
        unwrapped = patch.object(task_intake, 'normalize_task', task_intake.normalize_task.__wrapped__)
        with unwrapped, self.assertRaisesRegex(ValueError, 'task.profile is required'):
            self.plan(task)
        with unwrapped, self.assertRaisesRegex(ValueError, 'task.profile is required'):
            self.plan(dict(task, unclassified='no classifier'))      # a task field is never read as the route's
        with self.assertRaisesRegex(ValueError, '^invalid_task_input: unclassified'):
            route_judge._plan_input_task({'task': dict(task, unclassified='no classifier')},
                                         argparse.Namespace(hours=None, can_wait_hours=None))


class ListedModelRefusalTests(NamedRouteCase):
    """A provider's refusal of a listed-only model (an effort it does not take, for one) is about that model: it never
    takes the account out of automatic routing (launch_failures: account-wide only when the provider never came up)."""

    def test_a_refusal_about_a_listed_model_cools_only_that_model(self):
        listed = {'account': 'codex', 'provider': 'codex', 'model': 'gpt-6.2-nova', 'effort': 'turbo'}
        self.assertIsNone(launch_failures.candidate_failure_class('request_invalid', 'invalid_argument'))
        self.assertIsNone(launch_failures.record(listed, 'request_invalid', failure_code='invalid_argument'))
        launch_failures.record(listed, 'model_unavailable', failure_code='unsupported_model')
        self.assertEqual(set(launch_failures.active()), {launch_failures.key('codex', 'gpt-6.2-nova')})
        plan = self.plan(self.task())
        cooled = [c['id'] for c in plan['ranking'] if c['bundle']['account'] == 'codex'
                  and any('host refused to create a worker' in e for e in c['excluded'])]
        self.assertEqual(cooled, [])
