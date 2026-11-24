"""Fast contracts across model, harness, billing and launch surfaces; no live calls."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime, timezone

import allocator
import account_fleet
import axis_scores
import candidate_eval
import candidate_ranking
import dispatch_registry as registry
import fast_modes
import roll
import route_judge
import usage_reading
from router_test_support import CITED_FAST, cited_fast, isolated_router_state, usage_windows


def setUpModule():
    isolated_router_state(unittest.addModuleCleanup)


def codex_priced(accounts):
    """The reference Codex account on a plan whose window capacity the shipped data cites (`pro`, one weekly `usage`
    window): the reference account's own capacity is unknown, which keeps it out of automatic routing."""
    accounts['accounts']['codex']['plan'] = 'pro'
    return accounts


class FastModesTests(unittest.TestCase):
    def test_only_an_effective_fast_capable_true_enables_fast(self):
        now = datetime.now(timezone.utc)
        accounts = codex_priced(account_fleet.load_accounts(account_fleet.reference_fleet_path()))
        usage = usage_windows(now, {'codex': [('usage', 95, 100)],
                                    'claude': [('five_hour', 95, 3), ('seven_day', 95, 100)]})
        task = dict(work_kind='code_audit', difficulty='routine', expected_hours=0.01,
                    needs={'local_only': True, 'fast': True})
        for provider, model, bid in [('codex', 'gpt-6.1-sol', 'sol_high'),
                                     ('claude', 'claude-opus-5-5', 'opus55_high')]:
            for policy in (False, True, None):
                with self.subTest(provider=provider, fast_capable=policy), cited_fast(model):
                    info = accounts['models'][model]
                    if policy is None:
                        info.pop('fast_capable', None)
                    else:
                        info['fast_capable'] = policy
                    for forced in (False, True):
                        ranking = allocator.plan(task, usage, now, accounts=accounts, route={'force_fast': forced})['ranking']
                        plain = candidate_ranking.candidate_for_bundle(ranking, bid, False)
                        self.assertFalse(plain['bundle'].get('fast') if plain else False)
                        if forced:
                            # Rule (fast-modes.md): a fast candidate needs a verified contract on its harness and cited
                            # multipliers. Claude has no fast contract, so even a row declaring fast_capable is never fast.
                            fast = candidate_ranking.candidate_for_bundle(ranking, bid, True)
                            self.assertEqual(bool(fast and fast['bundle'].get('fast')), provider == 'codex' and policy is True)
                    bundle = dict(account_fleet.bundles(accounts)[bid], fast=True)
                    args = SimpleNamespace(basis='Diagnostic fixture', provider=provider, account=provider,
                        model=model, effort='high', input=None, usage_cache=None, accounts=None,
                        constraint='user_override', specialty=None, fast=True)
                    with patch.object(account_fleet, 'load_accounts', return_value=accounts), \
                            patch.object(usage_reading, 'usage_snapshot', return_value=usage):
                        if policy is not True or provider == 'claude':
                            # A named --fast without a contract launches plain (owner, 2026-11-24); a stored fast bundle
                            # of such a model is refused with the reason before anything starts.
                            recorded = route_judge.record_constraint(args)
                            self.assertNotIn('fast', recorded['selected_bundle'])
                            self.assertRegex(recorded['fast_not_applied'], f'^{model} has no verified fast mode on {provider}$')
                            with self.assertRaisesRegex(ValueError, f'^{model} has no verified fast mode on {provider}$'):
                                registry.resolve_bundle(bundle, accounts)
                            with patch.object(roll, 'provider_definition', return_value=(provider, {})), \
                                    patch.object(roll.subprocess, 'Popen') as spawn:
                                with self.assertRaisesRegex(ValueError, f'^{model} has no verified fast mode on {provider}$'):
                                    roll.cli(SimpleNamespace(wait=0), {}, bundle, 'fixture')
                                spawn.assert_not_called()
                        else:
                            self.assertTrue(route_judge.record_constraint(args)['selected_bundle']['fast'])
                            self.assertTrue(registry.resolve_bundle(bundle, accounts)['fast'])
                    self.assertIsNotNone(registry.resolve_bundle(dict(bundle, fast=False), accounts))

    def test_fast_needs_a_declared_row_on_the_harness_of_its_api_family(self):
        # Rule (fast-modes.md): fast runs only where the row declares fast_capable: true, on its api_family's harness.
        for harness in fast_modes.HARNESSES:
            fast_modes.validate(harness, 'model-a', {'fast_capable': True, 'api_family': harness})
        other = {'codex': 'claude'}
        for harness, info in [('pi', {'fast_capable': True, 'api_family': 'codex'}),
                              ('antigravity-resilient', {'fast_capable': True, 'api_family': 'claude'}),
                              *((h, {'fast_capable': True, 'api_family': other[h]}) for h in fast_modes.HARNESSES),
                              *((h, {'api_family': h}) for h in fast_modes.HARNESSES),
                              *((h, {'fast_capable': False, 'api_family': h}) for h in fast_modes.HARNESSES),
                              ('claude', {}), ('claude', {'fast_capable': True, 'api_family': 'claude'})]:
            with self.subTest(harness=harness, info=info), self.assertRaises(ValueError):
                fast_modes.validate(harness, 'model-a', info)
            self.assertFalse(fast_modes.supported(harness, 'model-a', info))

    def test_a_missing_catalog_row_is_a_named_error(self):
        for check in (fast_modes.supported, fast_modes.validate):
            with self.subTest(check=check.__name__), self.assertRaisesRegex(ValueError, 'needs its catalog row'):
                check('codex', 'model-a', None)

    def test_fast_and_normal_requests_preserve_model_effort_and_no_pi_feature(self):
        for provider, model in [('codex', 'gpt-6-astra'), ('claude', 'claude-opus-5-5')]:
            for fast in (False, True):
                b = dict(provider=provider, model=model, effort='medium', fast=fast)
                before = copy.deepcopy(b)
                if provider == 'claude' and fast:
                    # Claude has no fast contract: a fast Claude bundle (an old decision) never reaches a launch request.
                    with self.assertRaisesRegex(ValueError, 'has no verified fast mode on claude'):
                        registry.launch_request(b, dict(title='fixture', initialPrompt='fixture'))
                    continue
                request = registry.launch_request(b, dict(title='fixture', initialPrompt='fixture'))
                self.assertEqual(b, before)
                self.assertEqual(request['provider'], provider + '/' + model)
                self.assertEqual(request['settings']['thinkingOptionId'], 'medium')
                expected = {'service_tier': 'priority' if fast else 'default'} if provider == 'codex' else {'fast_mode': False}
                self.assertEqual(request['settings']['features'], expected)
        pi = registry.launch_request(dict(provider='pi', model='gpt-6-astra', effort='high'),
                                     dict(title='pi', initialPrompt='fixture'))
        self.assertNotIn('features', pi['settings'])
        self.assertNotIn('modeId', pi['settings'])
        with self.assertRaises(ValueError):
            registry.launch_request(dict(provider='pi', model='gpt-6-astra', fast=True),
                                    dict(title='pi', initialPrompt='fixture'))

    def test_cli_session_settings_set_fast_without_model_switch_and_never_turn_claude_fast_on(self):
        for fast in (False, True):
            args = roll.CLI_RUNS['claude'][1]('claude-opus-5-5', 'medium', 'C:/fixture', fast)
            self.assertEqual(args[args.index('--model') + 1], 'claude-opus-5-5')
            self.assertEqual(args[args.index('--effort') + 1], 'medium')
            self.assertEqual(json.loads(args[args.index('--settings') + 1]), {'fastMode': False})     # never on
            args = roll.CLI_RUNS['codex'][1]('gpt-6-astra', 'medium', 'C:/fixture', fast)
            self.assertIn('service_tier="priority"' if fast else 'service_tier="default"', args)
            self.assertIn('features.fast_mode=' + str(fast).lower(), args)

    def test_no_bundle_of_a_claude_account_is_fast(self):
        # Rule (fast-modes.md): Claude has no fast contract. No derived bundle, plan row or forced route of a Claude
        # account is fast, even on a row declaring fast_capable with cited multipliers.
        now = datetime.now(timezone.utc)
        accounts = account_fleet.load_accounts(account_fleet.reference_fleet_path())
        claude_models = sorted({b['model'] for b in account_fleet.bundles(accounts).values() if b['harness'] == 'claude'})
        self.assertTrue(claude_models)
        for model in claude_models:
            accounts['models'][model]['fast_capable'] = True
        usage = usage_windows(now, {'claude': [('five_hour', 10, 3), ('seven_day', 10, 100)]})
        task = dict(work_kind='code_audit', difficulty='routine', expected_hours=0.01,
                    needs={'local_only': True, 'fast': True})
        with cited_fast(*claude_models):
            for forced in (False, True):
                ranking = allocator.plan(dict(task), usage, now, accounts=accounts, route={'force_fast': forced})['ranking']
                claude_rows = [c for c in ranking if c['bundle']['harness'] == 'claude']
                self.assertTrue(claude_rows)
                self.assertEqual([c['id'] for c in claude_rows if c['bundle'].get('fast')], [])
        for bid, bundle in account_fleet.bundles(accounts).items():
            if bundle['harness'] == 'claude':
                self.assertFalse(bundle.get('fast'), bid)
                self.assertFalse(bid.endswith(fast_modes.ID_SUFFIX), bid)

    def test_a_cited_fast_row_is_its_own_candidate_priced_at_the_cited_multiples(self):
        now = datetime.now(timezone.utc)
        accounts = codex_priced(account_fleet.load_accounts(account_fleet.reference_fleet_path()))
        usage = usage_windows(now, {'codex': [('usage', 95, 1)]})
        task = dict(work_kind='code_audit', difficulty='routine', expected_hours=0.01,
                    needs={'local_only': True, 'fast': True})
        def rows(t, load=accounts):
            route = {k: t.pop(k) for k in ('force_fast',) if k in t}      # a route field, never a task field
            return allocator.plan(t, usage, now, accounts=load, route=route)['ranking']
        with cited_fast('gpt-6.1-sol', 'gpt-6-astra'):
            plain = candidate_ranking.candidate_for_bundle(rows(task), 'sol_high', False)
            self.assertFalse(plain['bundle'].get('fast'))
            paid = candidate_ranking.candidate_for_bundle(rows(dict(task, force_fast=True)), 'sol_high', True)
            self.assertEqual((paid['id'], paid['bundle']['fast']), ('sol_high+fast', True))
            self.assertEqual(paid['base_id'], 'sol_high')
            self.assertGreater(paid['expected_usd'], 0)
            self.assertTrue(any(f"{CITED_FAST['price_multiplier']:g}x burn" in n for n in paid['notes']))
            self.assertIsNone(candidate_ranking.candidate_for_bundle(rows(dict(task, force_fast=True, disable_fast=True)), 'sol_high', True))
            accounts['accounts']['codex']['bundles']['sol_high']['disabled'] = True
            self.assertTrue(candidate_ranking.candidate_for_bundle(rows(dict(task, force_fast=True)), 'sol_high', False)['excluded'])
            accounts['accounts']['codex']['bundles']['sol_high']['disabled'] = False
            usage = usage_windows(now, {'codex': [('usage', 0, 100)]})
            forced_empty = allocator.plan(dict(task), usage, now, accounts=accounts, route={'force_fast': True})
            self.assertEqual([c for c in forced_empty['ranking'] if not c['excluded'] and c['bundle']['account'] == 'codex'], [])
            self.assertIsNone(candidate_ranking.candidate_for_bundle(forced_empty['ranking'], 'sol_high', True))
            self.assertIn(candidate_eval.FAST_WAIT_REFUSAL,
                          candidate_ranking.candidate_for_bundle(forced_empty['ranking'], 'sol_high', False)['excluded'])
            # Without `force_fast`, `--fast` is a preference: the plain candidate waits on its window reset with `fast_not_offered`.
            waited_plain = candidate_ranking.candidate_for_bundle(rows(dict(task)), 'sol_high', False)
            self.assertEqual(waited_plain['excluded'], [])
            self.assertEqual(waited_plain['waiting_on'], 'usage')
            self.assertAlmostEqual(waited_plain['available_in_hours'], 100.0, places=2)
            self.assertEqual(waited_plain['fast_not_offered'], candidate_eval.FAST_WAIT_REFUSAL)


if __name__ == '__main__':
    unittest.main()
