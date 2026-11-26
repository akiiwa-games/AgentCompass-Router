"""Contract tests for allocator.py, override_store.py, and antigravity_telemetry.py."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

import allocator
import account_fleet
import accounts_override
import axis_scores
import ledger_snapshot
import quota_overrides
import quota_pools
import task_intake
import worker_sessions
import antigravity_telemetry
import override_store
import route_judge as router
import decision_ledger
import subscription_cli
import usage_reading
from router_test_support import isolated_router_state, profile, profile_for, temporary_root, usage_windows


EXPIRING_SUBSCRIPTION = {
    'plan': 'Grok subscription', 'billing': 'subscription', 'verified': True,
    'expires_at': '2026-10-10T23:59:00+08:00', 'expires_precision': 'date',
    'windows': {'credits': {'hours': 168}},
    'bundles': {'grok47_high': {'model': 'grok-4.7', 'effort': 'high'}},
}
_real_load_accounts = account_fleet.load_accounts


def reference_capabilities():
    return account_fleet.load_capabilities(accounts=account_fleet.load_accounts(account_fleet.reference_fleet_path()))


def _load_accounts_with_expiring_subscription(path=None, document=None, *, read_only=False):
    data = _real_load_accounts(path, document=document, read_only=read_only)
    if path is None and document is None:
        data['accounts']['grok'] = json.loads(json.dumps(EXPIRING_SUBSCRIPTION))
    return data


def setUpModule():
    isolated_router_state(unittest.addModuleCleanup)
    fixture = patch.object(account_fleet, 'load_accounts', _load_accounts_with_expiring_subscription)
    fixture.start()
    unittest.addModuleCleanup(fixture.stop)


class AllocatorTests(unittest.TestCase):
    DOT_ROUTES = ('web_research', 'text_processing', 'creative_ideation', 'architecture_reasoning')
    BULK = {'work_kind': 'text_processing', 'difficulty': 'hard', 'expected_hours': 6}
    SCARCE = {'claude': [('five_hour', 90, 3), ('seven_day', 12, 100), ('weekly_fable', 100, 100)], 'codex': [('usage', 12, 100)]}
    PLENTY = {'claude': [('five_hour', 95, 3), ('seven_day', 95, 100), ('weekly_fable', 100, 100)], 'codex': [('usage', 95, 100)]}

    def setUp(self):
        self.now = datetime(2026, 10, 9, 3, 0, tzinfo=timezone.utc)

    def usage(self, **overrides):
        windows = {
            'claude': [('five_hour', 90, 3), ('seven_day', 60, 100), ('weekly_fable', 100, 100)],
            'codex': [('usage', 60, 100)],
            'grok': [('credits', 0, 2.2)],
            'antigravity': [('gemini-5h', 90, 3), ('gemini-weekly', 60, 100), ('3p-5h', 90, 3), ('3p-weekly', 60, 100)],
        }
        windows.update(overrides)
        windows.setdefault('claude-lab', list(windows['claude']))
        return usage_windows(self.now, windows)

    def no_flash(self, **accounts):
        flash = [('gemini-5h', 3, 3), ('gemini-weekly', 5, 100), ('3p-5h', 3, 3), ('3p-weekly', 5, 100)]
        return self.usage(antigravity=flash, **accounts)

    def plan(self, task, usage=None, now=None, **kwargs):
        needs = task.get('needs') if isinstance(task.get('needs'), dict) else {}
        if task.get('work_kind') not in self.DOT_ROUTES and 'local_only' not in needs:
            task = dict(task, needs=dict(needs, local_only=True))
        return allocator.plan(task, usage or self.usage(), now or self.now, **kwargs)

    def rows(self, result):
        return {c['id']: c for c in result['ranking']}

    def chain(self, bundle='sol_xhigh', coupling='high', total=6, tokens=120000, compacted=None, compactions=None, **extra):
        worker = {'bundle': bundle, 'context_tokens': tokens,
                  **({'compacted_tokens': compacted} if compacted is not None else {}),
                  **({'compactions': compactions} if compactions is not None else {})}
        return {'id': 'refactor-1', 'coupling': coupling, 'total_hours': total, 'worker': worker, **extra}

    def profile(self, **fields):
        access = {'access': 'read-only'} if fields.get('read_only_review') else {}
        return profile(**{'local_only': True, **access, **fields})

    def unsubscribed(self, account):
        accounts = account_fleet.load_accounts()
        accounts['accounts'].pop(account)
        return accounts

    def codex_bundles(self):
        return [b for b, v in account_fleet.bundles(account_fleet.load_accounts()).items() if v['account'] == 'codex']

    def test_a_high_coupling_chain_asks_its_window_for_the_whole_chain_and_a_medium_one_for_the_step(self):
        """Rule (`candidate_eval.evaluate_bundle`, quota-window check): the hours a window must cover are the chain's `total_hours`
        (scaled by the bundle's duration) when `coupling` is `high`, else the step's `expected_hours`; when what is left minus that
        spend is under the safety margin, the window holds the bundle back until its reset. The spend of one step is read from the
        plan's own row (expected USD over the window's capacity), the chain's is that times chain hours / step hours, so the
        fixture follows the data: the window is left with the margin plus the midpoint of the two, which covers the step now
        but holds the high-coupling chain until reset."""
        step_hours, chain_hours, margin = 1, 3, 5
        task = lambda coupling: {'work_kind': 'code_implementation', 'difficulty': 'coupled', 'expected_hours': step_hours,
                                 'chain': self.chain(coupling=coupling, total=chain_hours)}
        full = self.plan(task('high'), self.usage(codex=[('usage', 100, 100)]))
        owner = next(c for c in full['ranking'] if c['id'] == 'sol_xhigh')
        capacity = axis_scores.five_axis_overlay(account_fleet.load_accounts(), read_local_state=False)['accounts']['codex']['capacity_usd']['usage']
        step_pct = 100 * owner['expected_usd'] / capacity
        chain_pct = step_pct * chain_hours / step_hours
        self.assertGreater(chain_pct, step_pct + 10, 'the fixture needs a chain clearly larger than a step')
        left = margin + (step_pct + chain_pct) / 2
        self.assertLess(left, 100)
        usage = self.usage(codex=[('usage', left, 100)])
        owner_row = lambda result: [r for r in result['ranking'] if r['id'] == 'sol_xhigh'][0]
        high = self.plan(task('high'), usage)
        self.assertEqual(owner_row(high)['excluded'], [])
        self.assertEqual((owner_row(high)['available_in_hours'], owner_row(high)['waiting_on']), (100.0, 'usage'),
                         'a window that covers the step but not the whole chain holds the high-coupling chain owner until reset')
        self.assertNotEqual(high['choice'], 'sol_xhigh', 'a quota-held owner does not start now: the ranking picks a ready route')
        medium = self.plan(task('medium'), usage)
        self.assertIsNone(owner_row(medium)['available_in_hours'], 'the same window covers the step of a medium-coupling chain now')
        self.assertEqual((medium['choice'], medium['decision']['action']), ('sol_xhigh', 'resume'))

    def test_chain_continuity_and_handoff_rotation(self):
        # 1. Active high-coupling chain resumes owner; a blocked owner gives way to the next route at any coupling
        # Rule: the owner resumes while the whole chain fits its window with the margin left. The share the chain needs is
        # computed from the plan's own row (chain hours / step hours x the step's spend, over the window's capacity), so the
        # fixture follows the data: the window is left with that share plus 15 points (the margin is 5).
        chain_task = {'work_kind': 'code_implementation', 'difficulty': 'coupled', 'expected_hours': 2, 'chain': self.chain()}
        full = self.plan(chain_task, self.usage(codex=[('usage', 100, 100)]))
        owner = next(c for c in full['ranking'] if c['id'] == 'sol_xhigh')
        capacity = axis_scores.five_axis_overlay(account_fleet.load_accounts(), read_local_state=False)['accounts']['codex']['capacity_usd']['usage']
        chain_pct = 100 * owner['expected_usd'] * (6 / 2) / capacity
        left = min(100, int(chain_pct) + 15)
        self.assertGreater(left, chain_pct + 10, 'the whole chain of the owner must fit the full window with room for the margin')
        roomy = self.usage(codex=[('usage', left, 100)])
        resumed = self.plan(chain_task, roomy)
        self.assertEqual((resumed['choice'], resumed['decision']['action']), ('sol_xhigh', 'resume'))
        blocked_usage = self.usage(codex=[('usage', 1, 121)])
        for coupling in ('high', 'medium', 'low'):
            with self.subTest(coupling=coupling):
                blocked = self.plan({'work_kind': 'code_implementation', 'difficulty': 'coupled', 'expected_hours': 2,
                                     'chain': self.chain(coupling=coupling)}, blocked_usage)
                self.assertNotEqual(blocked['selected_bundle']['account'], 'codex')

        # 2. Whole chain must fit for high coupling, only current step for medium coupling
        tight_usage = self.usage(codex=[('usage', 8, 100)])
        task = {'work_kind': 'code_implementation', 'difficulty': 'coupled', 'expected_hours': 1, 'chain': self.chain(total=20)}
        tight = self.plan(task, tight_usage)
        self.assertTrue(next(c for c in tight['ranking'] if c['id'] == 'sol_xhigh')['excluded'])
        self.assertNotEqual(tight['choice'], 'sol_xhigh')

        # 4. Carried context (>=2M historical tokens) rotates unconditionally and unlocks re-allocation; compactions alone (<2M) still resume
        base_task = {'work_kind': 'code_implementation', 'difficulty': 'coupled', 'expected_hours': 1}
        for chain_kw, expected_action in (
            ({'tokens': 120000, 'compacted': 360000, 'compactions': 2}, 'resume'),
            ({'tokens': 60000, 'compacted': 8 * 244000, 'handoff_ready': True}, 'rotate'),
            ({'tokens': 60000, 'compacted': 3000000}, 'rotate'),
        ):
            with self.subTest(chain_kw=chain_kw):
                res = self.plan(dict(base_task, chain=self.chain(**chain_kw)), roomy)
                self.assertEqual(res['decision']['action'], expected_action)
                if expected_action == 'rotate':
                    # Rule (candidate_eval: continuity only on `resume`): a rotated owner is re-allocated as a fresh start,
                    # with no continuity value; which bundle then wins is the ranking's, not pinned here.
                    owner = next(c for c in res['ranking'] if c['id'] == 'sol_xhigh')
                    self.assertEqual((owner['action'], owner['components']['continuity']), ('rotate', 0.0))


    def test_session_log_compaction_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            def agent(agent_id, provider, session):
                record = home / '.paseo' / 'agents' / 'E-PROJECTS-x' / f'{agent_id}.json'
                record.parent.mkdir(parents=True, exist_ok=True)
                record.write_text(json.dumps({'persistence': {'provider': provider, 'sessionId': session}}), encoding='utf-8')
            def log(relative, lines):
                path = home / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(l, separators=(',', ':')) + '\n' for l in lines), encoding='utf-8')
            reading = lambda n: {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {'last_token_usage': {'input_tokens': n}}}}
            agent('a1', 'codex', 's-codex')
            log('.codex/sessions/2026/10/12/rollout-2026-10-12T08-00-00-s-codex.jsonl', [
                reading(240000), {'type': 'compacted', 'payload': {}}, reading(250000), {'type': 'compacted', 'payload': {}}, reading(120000)])
            agent('a2', 'claude', 's-claude')
            log('.claude/projects/E--PROJECTS-x/s-claude.jsonl', [
                {'type': 'system', 'subtype': 'compact_boundary', 'compactMetadata': {'preTokens': 970000}},
                {'type': 'assistant', 'message': {'usage': {'input_tokens': 10, 'cache_read_input_tokens': 500000, 'cache_creation_input_tokens': 90}}}])
            agent('a3', 'pi', 's-pi')
            log('.pi/agent/sessions/x/2026-10-12_s-pi.jsonl', [{'type': 'compaction', 'tokensBefore': 260000}] * 3 + [
                {'type': 'message', 'message': {'role': 'assistant', 'usage': {'input': 1000, 'cacheRead': 40000, 'cacheWrite': 0}}}])
            agent('a4', 'claude-lab', 's-c2')
            log('.claude-lab/projects/E--PROJECTS-x/s-c2.jsonl', [
                {'type': 'system', 'subtype': 'compact_boundary', 'compactMetadata': {'preTokens': 510000}},
                {'type': 'assistant', 'message': {'usage': {'input_tokens': 50, 'cache_read_input_tokens': 200000, 'cache_creation_input_tokens': 50}}}])
            self.assertEqual(worker_sessions.session_state('a1', home), {'compactions': 2, 'compacted_tokens': 490000, 'context_tokens': 120000, 'historical_context_tokens': 610000})
            self.assertEqual(worker_sessions.session_state('a2', home), {'compactions': 1, 'compacted_tokens': 970000, 'context_tokens': 500100, 'historical_context_tokens': 1470100})
            self.assertEqual(worker_sessions.session_state('a3', home), {'compactions': 3, 'compacted_tokens': 780000, 'context_tokens': 41000, 'historical_context_tokens': 821000})
            self.assertEqual(worker_sessions.session_state('a4', home), {'compactions': 1, 'compacted_tokens': 510000, 'context_tokens': 200100, 'historical_context_tokens': 710100})
            agent('a1-arch', 'codex', 's-codex-arch')
            log('.codex/archived_sessions/rollout-2026-10-20T21-21-50-s-codex-arch.jsonl', [reading(249566)])
            self.assertEqual(worker_sessions.session_state('a1-arch', home), {'compactions': 0, 'compacted_tokens': 0, 'context_tokens': 249566, 'historical_context_tokens': 249566})
            self.assertEqual(worker_sessions.session_state('missing', home)['telemetry_status'], 'unknown')

            # Historical 2M context cannot be masked by a small post-compaction compressed context_tokens
            agent('a5', 'claude', 's-claude-long-context')
            log('.claude/projects/E--PROJECTS-x/s-claude-long-context.jsonl', [
                {'type': 'system', 'subtype': 'compact_boundary', 'compactMetadata': {'preTokens': 1950000, 'postTokens': 12000, 'cumulativeDroppedTokens': 1938000}},
                {'type': 'assistant', 'message': {'usage': {'input_tokens': 100, 'cache_read_input_tokens': 80000, 'cache_creation_input_tokens': 0}}}])
            st5 = worker_sessions.session_state('a5', home)
            self.assertEqual((st5['compactions'], st5['context_tokens'], st5['historical_context_tokens']), (1, 80100, 2030100))
            orig_log = worker_sessions.session_log
            with patch.object(worker_sessions, 'session_log', side_effect=lambda aid, h=None: orig_log(aid, home)):
                rot = self.plan({'work_kind': 'code_implementation', 'difficulty': 'coupled',
                                 'chain': {'id': 'c5', 'coupling': 'high', 'worker': {'agent_id': 'a5', 'bundle': 'sol_xhigh', 'context_tokens': 80100}}})
                self.assertEqual(rot['decision']['action'], 'rotate')

    def test_harness_selection_and_invariance(self):
        bundles = account_fleet.bundles(account_fleet.load_accounts())
        for bundle_id in ('pi_astra_high', 'pi_astra_xhigh', 'pi_sol_high', 'pi_sol_xhigh', 'pi_sol_max'):
            b = bundles[bundle_id]
            self.assertEqual((b['provider'], b['account'], b['launch_model']), ('pi', 'codex', 'openai-codex/' + b['model']))

        # antigravity is kept scarce so the harness order, not Argon's price, decides among the GPT siblings
        claude_low = self.no_flash(claude=[('five_hour', 90, 3), ('seven_day', 5, 100), ('weekly_fable', 100, 100)])
        explore = self.plan({'work_kind': 'exploration_forensics'}, claude_low)
        gpt_exp = [c for c in explore['ranking'] if c['score'] is not None and c['bundle']['account'] == 'codex']
        self.assertEqual(gpt_exp[0]['bundle']['provider'], 'pi')
        scores = {c['id']: c['score'] for c in explore['ranking']}
        # The harness order never touches the score: the Pi bundle and its Codex twin score alike and differ in place only.
        self.assertAlmostEqual(scores[gpt_exp[0]['id']], scores[gpt_exp[0]['id'][3:]], places=6)
        build = self.plan({'work_kind': 'code_implementation', 'difficulty': 'hard'}, claude_low)
        gpt = [c for c in build['ranking'] if c['score'] is not None and c['bundle']['account'] == 'codex']
        self.assertEqual(gpt[0]['bundle']['provider'], 'codex')
        research = self.plan({'work_kind': 'web_research', 'needs': {'local_only': True}}, claude_low)
        gpt_res = [c for c in research['ranking'] if c['score'] is not None and c['bundle']['account'] == 'codex']
        self.assertEqual(gpt_res[0]['bundle']['provider'], 'pi')

        for kind in ('exploration_forensics', 'code_implementation', 'code_audit'):
            with_pi = self.plan({'work_kind': kind})
            without = self.plan({'work_kind': kind}, unavailable=[b for b in bundles if b.startswith('pi_')])
            self.assertEqual((with_pi['selected_bundle']['model'], with_pi['selected_bundle']['effort']),
                             (without['selected_bundle']['model'], without['selected_bundle']['effort']), kind)
            self.assertEqual(with_pi['decision']['margin'], without['decision']['margin'], kind)

    def test_dot_eligibility_respects_route_modality_and_local_needs(self):
        capabilities = reference_capabilities()
        # Caller false stands. Route implies fill local_only only when the caller and the profile left it unset.
        self.assertTrue(task_intake.dot_may_take('exploration_forensics', {'local_only': False}, capabilities))
        self.assertTrue(task_intake.dot_may_take('exploration_forensics', {}, capabilities, {'local_only': False}))
        self.assertTrue(task_intake.dot_may_take('web_research', {}, capabilities, {'local_only': False}))
        self.assertFalse(task_intake.dot_may_take('exploration_forensics', {}, capabilities, {'local_only': True}))
        self.assertTrue(task_intake.dot_may_take('exploration_forensics', {'local_only': False}, capabilities, {'local_only': True}))
        self.assertFalse(task_intake.dot_may_take('exploration_forensics', {}, capabilities))
        self.assertFalse(task_intake.dot_may_take('visual_analysis', {}, capabilities))
        self.assertFalse(task_intake.dot_may_take('code_implementation', {'visual': True}, capabilities))
        for kind in ('web_research', 'exploration_forensics'):
            opened = self.plan({'work_kind': kind, 'difficulty': 'hard', 'needs': {'local_only': False},
                                'profile': self.profile(work_kind=kind, local_only=False)})
            admitted = {c['bundle']['model'] for c in opened['ranking'] if c['score'] is not None and not c['excluded']}
            self.assertIn('dot', admitted, kind)

        # Eligibility only: Dot is admitted for non-local non-visual kinds; which candidate wins is not a rule.
        for kind in ('web_research', 'text_processing', 'creative_ideation', 'code_implementation'):
            res = self.plan({'work_kind': kind, 'difficulty': 'routine', 'needs': {'local_only': False}},
                            self.usage(grok=[('credits', 100, 165)]))
            self.assertFalse(self.rows(res)['dot_text']['excluded'], kind)
            local = self.plan({'work_kind': kind, 'difficulty': 'routine', 'needs': {'local_only': True}})
            self.assertNotEqual(local['choice'], 'dot_text', kind)
        for difficulty in ('hard', 'coupled'):
            self.assertFalse(self.rows(self.plan({'work_kind': 'web_research', 'difficulty': difficulty}))['dot_text']['excluded'], difficulty)

    def test_short_window_deferral_and_wait_limits(self):
        usage = self.usage(claude=[('five_hour', 3, 1.0), ('seven_day', 60, 100), ('weekly_fable', 100, 100)], codex=[('usage', 1, 121)])
        task = {'work_kind': 'visual_analysis', 'difficulty': 'routine', 'expected_hours': 1, 'needs': {'visual': True}}
        no_argon = ['argon_high'] + [b for b in account_fleet.bundles(account_fleet.load_accounts()) if b.startswith(('agy_opus', 'agy_sonnet'))]
        # The cheapest sufficient candidate sits behind Claude's short five-hour window: a task that can wait waits for the reset.
        deferred = self.plan(dict(task, can_wait_hours=2), usage, unavailable=no_argon)
        self.assertEqual(deferred['decision']['action'], 'defer')

        long_wait = self.usage(claude=[('five_hour', 3, 4), ('seven_day', 60, 100), ('weekly_fable', 100, 100)], codex=[('usage', 1, 121)])
        # Rule (`allocator.plan`): a chosen candidate that must wait is a deferral whether or not the caller allowed the wait;
        # one longer than allowed says so in the reason, and `starts_now` names a candidate only when one starts at once.
        for can_wait, over_limit in ((1, True), (5, False)):
            with self.subTest(can_wait=can_wait):
                waiting = self.plan(dict(task, can_wait_hours=can_wait), long_wait, unavailable=['flash_high'] + no_argon)
                decision = waiting['decision']
                self.assertEqual(decision['action'], 'defer')
                self.assertEqual(decision['start_after_hours'], decision['wait_for']['hours'])
                self.assertEqual('Nothing eligible can start within' in decision['reason'], over_limit)
                ready = [c['id'] for c in waiting['ranking'] if not c['excluded'] and not c['available_in_hours']]
                self.assertEqual(decision.get('starts_now'), ready[0] if ready else None)

    def test_claude_2_backup_safety_floor_and_inflight_reservation(self):
        normal_claude2 = self.usage(
            claude=[('five_hour', 90, 3), ('seven_day', 45, 100), ('weekly_fable', 100, 100)],
            **{'claude-lab': [('five_hour', 90, 3), ('seven_day', 85, 100), ('weekly_fable', 100, 100)]},
        )
        # With every big pool empty claude-lab is what takes the work, at the effort that fits its remaining window (in ordinary
        # times Argon's shorter Artificial Analysis time takes this task instead, so no claude-lab choice is pinned there)
        all_big_empty = self.usage(
            claude=[('five_hour', 90, 3), ('seven_day', 0, 100), ('weekly_fable', 100, 100)],
            codex=[('usage', 1, 100)],
            antigravity=[('gemini-5h', 90, 3), ('gemini-weekly', 2, 100), ('3p-5h', 90, 3), ('3p-weekly', 2, 100)],
            **{'claude-lab': [('five_hour', 95, 3), ('seven_day', 83, 100), ('weekly_fable', 100, 100)]},
        )
        # Rule (`candidate_eval.evaluate_bundle`, `allocator.plan`): exhausted windows with a known reset hold their
        # candidates in `available_in_hours` rather than `excluded`, while with `can_wait_hours == 0` only ready-now
        # candidates (`claude-lab`) compete to start immediately.
        empty_plan = self.plan({'work_kind': 'exploration_forensics'}, all_big_empty)
        ready_rows = [c for c in empty_plan['ranking'] if not c['excluded'] and not c['available_in_hours']]
        self.assertTrue(ready_rows)
        self.assertEqual({c['bundle']['account'] for c in ready_rows}, {'claude-lab'}, [c['id'] for c in ready_rows])
        self.assertEqual(empty_plan['decision']['action'], 'start')
        self.assertEqual(empty_plan['selected_bundle']['account'], 'claude-lab')

        task = {'work_kind': 'exploration_forensics', 'difficulty': 'hard', 'expected_hours': 0.7, 'needs': {'local_only': True}}
        usage_low = self.usage(**{'claude-lab': [('five_hour', 23, 2.5), ('seven_day', 61, 130)], 'claude': [('five_hour', 0, 2.0), ('seven_day', 0, 100)]})
        low_row = self.rows(self.plan(task, usage_low))['lab_sonnet55_high']
        self.assertFalse(low_row['excluded'])
        self.assertEqual((low_row['available_in_hours'], low_row['waiting_on']), (2.5, 'five_hour'))

        usage_40 = self.usage(**{'claude-lab': [('five_hour', 40, 2.5), ('seven_day', 61, 130)], 'claude': [('five_hour', 0, 2.0), ('seven_day', 0, 100)]})
        usage_40['accounts']['claude-lab']['fetched_at'] = (self.now + timedelta(seconds=12)).isoformat()
        inflight = [{
            'at': (self.now + timedelta(seconds=10)).isoformat(), 'stage': 'dispatch', 'dispatch_id': 'fixture-dispatch',
            'state': 'started', 'lease_until': (self.now + timedelta(minutes=10)).isoformat(),
            'choice': 'lab_sonnet55_high', 'hours': 0.7, 'expected_usd': 3.08, 'kind': 'exploration_forensics',
            'bundle': {'provider': 'claude-lab', 'account': 'claude-lab', 'model': 'claude-sonnet-5-5', 'bundle_id': 'lab_sonnet55_high', 'expected_usd': 3.08},
            'usage': {'claude-lab': {'windows': [{'id': 'five_hour', 'used_pct': 60.0}, {'id': 'seven_day', 'used_pct': 39.0}]}},
        }]
        s2 = self.rows(allocator.plan(task, usage_40, self.now + timedelta(minutes=2), records=inflight))['lab_sonnet55_high']
        self.assertFalse(s2['excluded'])
        self.assertEqual(s2['waiting_on'], 'five_hour')
        self.assertGreater(s2['available_in_hours'], 0.0)
        self.assertTrue(any('$3.08' in n and 'reserved by in-flight' in n for n in s2['notes']), s2['notes'])

        # Unstarted 5h window (used_pct=0, resets_at=None) still enforces 5h wall against oversized or burst tasks
        unstarted_5h = copy.deepcopy(all_big_empty)
        unstarted_5h['accounts']['claude-lab']['status'] = 'fresh'
        unstarted_5h['accounts']['claude-lab']['windows'][0] = {
            'id': 'five_hour', 'used_pct': 0.0, 'resets_at': None, 'seconds_until_reset': None,
            'observed_remaining_pct': 100.0, 'remaining_pct_before_reset': None}
        self.assertFalse(self.rows(self.plan(task, unstarted_5h))['lab_sonnet55_high']['excluded'])
        burst = [dict(inflight[0], dispatch_id=f'd-{i}', expected_usd=6.6) for i in (1, 2)]
        self.assertTrue(self.rows(allocator.plan(
            {'work_kind': 'exploration_forensics', 'difficulty': 'hard', 'expected_hours': 1.5, 'needs': {'local_only': True}},
            unstarted_5h, self.now + timedelta(minutes=2), records=burst))['lab_sonnet55_high']['excluded'])

        # Stale claude-lab with active 5h reset and unknown/clamped used_pct or oversized 3h task is excluded
        stale_c2 = self.usage(
            claude=[('five_hour', 75, 4.0), ('seven_day', 20, 35.0), ('weekly_fable', 100, 35.0)],
        )
        stale_c2['accounts']['claude-lab'] = {
            'status': 'stale',
            'stale': True,
            'fetched_at': (self.now - timedelta(minutes=13)).isoformat(),
            'windows': [
                {'id': 'five_hour', 'used_pct': None, 'resets_at': (self.now + timedelta(hours=4)).isoformat(),
                 'seconds_until_reset': 4 * 3600, 'reset_time_passed': False,
                 'observed_remaining_pct': None, 'remaining_pct_before_reset': None},
                {'id': 'seven_day', 'used_pct': 74.0, 'resets_at': (self.now + timedelta(hours=5)).isoformat(),
                 'seconds_until_reset': 5 * 3600, 'reset_time_passed': False,
                 'observed_remaining_pct': 26.0, 'remaining_pct_before_reset': None},
            ],
        }
        res_1h = self.plan({'work_kind': 'exploration_forensics', 'difficulty': 'hard', 'expected_hours': 1.0,
                            'needs': {'local_only': True}}, stale_c2)
        self.assertTrue(self.rows(res_1h)['lab_sonnet55_high']['excluded'])
        self.assertNotEqual(res_1h['selected_bundle']['account'], 'claude-lab')

        # Even if five_hour telemetry is completely missing, an oversized task exceeding full 5h Pro capacity is excluded
        missing_5h = copy.deepcopy(stale_c2)
        missing_5h['accounts']['claude-lab']['windows'] = [missing_5h['accounts']['claude-lab']['windows'][1]]
        custom_acc = account_fleet.load_accounts()
        custom_acc['accounts']['claude-lab']['bundles']['lab_opus55_high'].pop('disabled', None)
        res_opus_3h = self.plan({'work_kind': 'frontend_ui', 'difficulty': 'hard', 'expected_hours': 3.0,
                                 'needs': {'local_only': True}}, missing_5h, accounts=custom_acc)
        self.assertTrue(self.rows(res_opus_3h)['lab_opus55_high']['excluded'])
        res_sonnet_4h = self.plan({'work_kind': 'exploration_forensics', 'difficulty': 'hard', 'expected_hours': 3.8,
                                   'needs': {'local_only': True}}, missing_5h)
        self.assertTrue(self.rows(res_sonnet_4h)['lab_sonnet55_high']['excluded'])

    def test_constraints_and_validation(self):
        self.assertTrue(self.plan({'work_kind': 'code_implementation', 'difficulty': 'hard',
                                   'constraints': {'model': 'claude-opus-5'}})['selected_bundle']['model'].startswith('claude-opus-5-5'))
        self.assertIn('account not verified as launchable',
                      self.rows(self.plan({'work_kind': 'code_implementation', 'difficulty': 'coupled'}))['work_sol_xhigh']['excluded'])
        with self.assertRaises(ValueError):
            self.plan({'difficulty': 'impossible'})
        with self.assertRaises(ValueError):
            self.plan({'chain': {'coupling': 'tight'}})
        accounts = account_fleet.load_accounts()
        models = account_fleet.models_table(accounts)
        for bundle in account_fleet.bundles(accounts).values():
            self.assertIn(account_fleet.canonical(bundle['model'], accounts), models, bundle['id'])

    def test_heuristic_profile_and_schema_contract(self):
        capabilities = reference_capabilities()
        policy = account_fleet.load_accounts()['scoring']
        accounts = account_fleet.load_accounts()
        for banned in (
            'infer_task_profile', 'infer_kind_from_objective', 'infer_fast_preference',
            'is_readonly_review_task', 'infer_creativity', 'multi_task_marks',
            'DIVERGENT_WORDS', 'FAST_REQUEST_WORDS', 'READONLY_REVIEW_PHRASES',
            'GUI_WORDS', 'COMPUTER_USE_PHRASES', 'CODE_FILE_RE',
            'OBJECTIVE_LOCAL_RUNTIME_RE', 'LOCAL_MACHINE_RE', 'MULTI_TASK_MARK_RE',
        ):
            self.assertFalse(hasattr(allocator, banned), f'{banned} must be removed')
        for kind in capabilities['routes']:
            for difficulty in (None, 'routine', 'hard', 'coupled', 'frontier'):
                task = {'work_kind': kind, **({'difficulty': difficulty} if difficulty else {})}
                normalized = task_intake.normalize_task(task, policy, capabilities, accounts)
                # No default table: the caller's difficulty, else the one the profile derives.
                self.assertEqual(normalized['difficulty'], difficulty or normalized['profile_difficulty'], task)
        good = self.profile()
        for bad in (dict(good, primary_level=5), dict(good, secondary_level=True),
                    {k: v for k, v in good.items() if k != 'wants_fast'},
                    {k: v for k, v in good.items() if k != 'secondary_kind'},
                    dict(good, secondary_kind='none', secondary_level=2),
                    dict(good, secondary_kind='live_loop', secondary_level=0),
                    dict(good, secondary_kind=good['work_kind'], secondary_level=1),
                    dict(good, axis_demand={'creativity': 1, 'global_rigor': 1, 'module_precision': 1, 'vision_tools': 0}),
                    dict(good, needs_gui='yes'), dict(good, wants_fast='yes'), dict(good, read_only_review=1),
                    dict(good, access='sudo'), 'deep'):
            with self.assertRaises(ValueError):
                task_intake.validate_profile(bad)
        schema = task_intake.TASK_PROFILE_SCHEMA
        self.assertEqual(set(schema['required']), set(schema['properties']))
        self.assertEqual(set(schema['required']), {
            'work_kind', 'primary_level', 'secondary_kind', 'secondary_level',
            'access', 'local_only', 'needs_visual', 'wants_fast', 'expected_hours', 'task_count',
        })
        for name, prop in schema['properties'].items():
            if prop['type'] == 'integer':
                self.assertNotIn('enum', prop, name)
                self.assertEqual((prop['minimum'], prop['maximum']), task_intake.PROFILE_INTEGERS[name])
        # local_only is resolved strictly from caller needs -> profile.local_only -> route implies -> False, never from objective/cwd text.
        prof_local = task_intake.normalize_task(
            {'objective': 'Pure text', 'profile': self.profile(local_only=True)},
            policy, capabilities, accounts,
        )
        self.assertTrue(prof_local['local_only'])
        caller_false = task_intake.normalize_task(
            {'objective': 'Run pytest', 'cwd': r'D:\work\AgentCompass', 'needs': {'local_only': False},
             'profile': self.profile(local_only=True)},
            policy, capabilities, accounts,
        )
        self.assertFalse(caller_false['local_only'])

    def healthy_quota(self, depleted=(), remaining=90):
        usage = self.usage()
        for name, account in usage['accounts'].items():
            if name == 'grok':
                continue  # The legacy expiry fixture is not an active subscription in these quota scenarios.
            for window in account['windows']:
                rem = remaining if name in depleted and window['seconds_until_reset'] > 24 * 3600 else 90
                window.update(remaining_pct_before_reset=rem, observed_remaining_pct=rem)
        return usage

    def test_cross_domain_offline_routes(self):
        # Route selection follows the classified profile work_kind regardless of incidental keywords in objective.
        capabilities = reference_capabilities()
        policy = account_fleet.load_accounts()['scoring']
        for objective, kind in (
            ('Click the Save button in the desktop settings', 'code_audit'),
            ('Compare screenshots with the UI mockup', 'code_implementation'),
            ('检查截图并列出布局差异', 'visual_analysis'),
            ('Implement the desktop UI component in React', 'frontend_ui'),
            ('Batch rename i18n keys uniformly', 'text_processing'),
            ('Search the web for public documentation', 'web_research'),
            ('Translate the deployment runbook without executing it', 'text_processing'),
            ('Write a multi-phase system migration spec', 'architecture_reasoning'),
            ('Invent a novel game mechanic combining rhythm and deduction', 'creative_ideation'),
            ('Follow the CI release runbook and health check', 'live_loop'),
            ('Diagnose the memory leak in frontend rendering', 'exploration_forensics'),
            ('Implement a standalone algorithm', 'code_implementation'),
            ('用鼠标点击设置里的保存', 'gui_control'),
        ):
            with self.subTest(objective=objective, kind=kind):
                prof = self.profile(work_kind=kind)
                self.assertEqual(task_intake.task_kind({'objective': objective, 'profile': prof}, capabilities), kind)

    def test_classified_read_only_review_profile_and_its_own_hours(self):
        ro_prof = self.profile(work_kind='code_audit', read_only_review=True, access='read-only')
        ro_norm = task_intake.normalize_task(
            {'objective': 'Independent review (read-only) of the authentication module', 'profile': ro_prof},
            account_fleet.load_accounts()['scoring'],
            reference_capabilities(),
            account_fleet.load_accounts(),
        )
        self.assertEqual(ro_norm['kind'], 'code_audit')
        self.assertEqual(ro_norm['profile']['access'], 'read-only')
        self.assertTrue(ro_norm['read_only_review'])
        self.assertEqual(ro_norm['hours'], ro_prof['expected_hours'])      # the classifier's own estimate; no per-difficulty default

    def test_existing_bundles_keep_tag_pools_and_draws_override(self):
        accounts = account_fleet.load_accounts()
        flat = account_fleet.bundles(accounts)
        # Expected pools are contract examples, not a copy of the production tag predicate.
        for bid, expected in (
            ('opus55_high', ['five_hour', 'seven_day']),
            ('sonnet55_high', ['five_hour', 'seven_day']),
            ('fable51_high', ['five_hour', 'seven_day', 'weekly_fable']),
            ('fable52_high', ['five_hour', 'seven_day', 'weekly_fable']),
            ('sol_high', ['usage']),
            ('pi_sol_high', ['usage']),
            ('flash_high', ['gemini-5h', 'gemini-weekly']),
            ('argon_high', ['gemini-5h', 'gemini-weekly']),
            ('agy_opus55_high', ['3p-5h', '3p-weekly']),
            ('dot_text', []),
        ):
            with self.subTest(bundle=bid):
                bundle = flat[bid]
                spec = accounts['accounts'][bundle['account']]
                self.assertEqual(quota_pools.bundle_pools(bundle, accounts), expected)
                self.assertEqual([wid for wid, _, _ in quota_pools.applicable_windows(
                    spec, bundle['model'], {}, bundle, accounts)], expected)
                if 'draws' not in bundle:
                    self.assertEqual([wid for wid, _, _ in quota_pools.applicable_windows(
                        spec, account_fleet.canonical(bundle['model'], accounts), {}, accounts=accounts)], expected)
        for bid, draws in (('flash_high', ['3p-weekly']), ('agy_opus55_high', ['3p-weekly']),
                           ('agy_opus55_high', [])):
            with self.subTest(bundle=bid, draws=draws):
                self.assertEqual(quota_pools.bundle_pools({**flat[bid], 'draws': draws}, accounts), draws)
        for family, canonical_name in (('opus', 'claude-opus-5-5'), ('sonnet', 'claude-sonnet-5-5')):
            for effort in ('low', 'medium', 'high'):
                bundle = flat[f'agy_{family}55_{effort}']
                self.assertEqual(quota_pools.bundle_pools(bundle, accounts), ['3p-5h', '3p-weekly'])
                self.assertEqual(account_fleet.canonical(bundle['model'], accounts), canonical_name)
                # The bundle holds the bare model; Paseo takes the effort as `thinkingOptionId` (launch_request).
                self.assertEqual(bundle['model'], canonical_name)
                self.assertNotIn('launch_model', bundle)
                self.assertEqual((bundle['provider'], bundle['modeId']), ('antigravity-resilient', 'full-access'))
                self.assertNotIn('disabled', bundle)
        self.assertNotIn('when', flat['agy_opus55_low'])

    def test_agy_claude_3p_reservation_launch_and_short_window(self):
        import dispatch_registry
        accounts = account_fleet.load_accounts()
        spec = accounts['accounts']['antigravity']
        lease = (self.now + timedelta(hours=2)).isoformat()
        record = {
            'stage': 'dispatch', 'dispatch_id': 'agy-opus-1', 'state': 'started', 'lease_until': lease,
            'bundle': {'account': 'antigravity', 'bundle_id': 'agy_opus55_high', 'model': 'claude-opus-5-5-high'},
            'expected_usd': 4.0,
        }
        for wid, cap in (('3p-5h', 17.6), ('3p-weekly', 88.0)):
            self.assertEqual(quota_pools.inflight_reserved_usd(
                'antigravity', spec, wid, spec['windows'][wid], {}, cap, [record], accounts, {}, self.now), 4.0)
        for wid in ('gemini-5h', 'gemini-weekly'):
            self.assertEqual(quota_pools.inflight_reserved_usd(
                'antigravity', spec, wid, spec['windows'][wid], {}, 100.0, [record], accounts, {}, self.now), 0.0)
        # Catalog draws win even when the record's model string would tag-match the Gemini pool.
        mismatched = {'stage': 'dispatch', 'dispatch_id': 'agy-opus-2', 'state': 'started', 'lease_until': lease,
                      'expected_usd': 4.0,
                      'bundle': {'account': 'antigravity', 'bundle_id': 'agy_opus55_high', 'model': 'gemini-3.8-flash-high'}}
        self.assertEqual(quota_pools.inflight_reserved_usd(
            'antigravity', spec, '3p-5h', spec['windows']['3p-5h'], {}, 17.6, [mismatched], accounts, {}, self.now), 4.0)
        self.assertEqual(quota_pools.inflight_reserved_usd(
            'antigravity', spec, 'gemini-5h', spec['windows']['gemini-5h'], {}, 182.0, [mismatched], accounts, {}, self.now), 0.0)

        resolved = dispatch_registry.resolve_bundle(
            {'provider': 'antigravity-resilient', 'account': 'antigravity', 'model': 'claude-opus-5-5', 'effort': 'high'},
            accounts)
        self.assertEqual(resolved['id'], 'agy_opus55_high')
        self.assertEqual(resolved['model'], 'claude-opus-5-5')
        with self.assertRaises(ValueError):
            dispatch_registry.resolve_bundle(
                {'provider': 'antigravity-resilient', 'account': 'antigravity', 'model': 'claude-opus-5-5', 'effort': 'xhigh'},
                accounts, explicit_override=True)
        request = dispatch_registry.launch_request(
            account_fleet.bundles(accounts)['agy_opus55_high'], {'title': 'agy claude', 'initialPrompt': 'fixture'})
        # The bare model and a separate thinking option (the plugin lists `claude-opus-5-5-high` only as an alias).
        self.assertEqual(request['provider'], 'antigravity-resilient/claude-opus-5-5')
        self.assertEqual(request['settings'].get('modeId'), 'full-access')
        self.assertEqual(request['settings'].get('thinkingOptionId'), 'high')

        sparse = account_fleet.load_accounts()
        accounts_override.apply(sparse, {'accounts': {'antigravity': {'bundles': {
            'flash_high': {'disabled': True}}}}})
        kept = sparse['accounts']['antigravity']['bundles']
        self.assertIs(kept['flash_high'].get('disabled'), True)
        self.assertNotIn('disabled', kept['agy_opus55_high'])
        self.assertNotIn('disabled', kept['agy_sonnet55_high'])

    def test_configured_rotation_threshold_and_missing_key_errors(self):
        import dispatch_registry as registry

        base = account_fleet.load_accounts(account_fleet.reference_fleet_path())
        caps = account_fleet.load_capabilities(accounts=base)
        roomy = self.usage(codex=[('usage', 80, 100)])
        step_task = {'work_kind': 'code_implementation', 'difficulty': 'coupled', 'expected_hours': 1}

        # 1. Thresholds below 250k (permitted by TUNING_LIMITS 100k..10M) scale context burn and rotate without a 250k floor.
        accounts_150k = copy.deepcopy(base)
        accounts_150k['scoring']['rotate_after_carried_tokens'] = 150_000
        fresh = self.plan(step_task, roomy, accounts=accounts_150k, capabilities=caps)
        fresh_usd = self.rows(fresh)['sol_xhigh']['expected_usd']
        at_120k = self.plan(
            dict(step_task, chain=self.chain(total=2, tokens=120_000, compacted=0, compactions=0)),
            roomy,
            accounts=accounts_150k,
            capabilities=caps,
        )
        self.assertEqual(at_120k['decision']['action'], 'resume')
        extra = accounts_150k['scoring']['context_burn_extra_at_250k'] * (120_000 / 150_000)
        self.assertAlmostEqual(self.rows(at_120k)['sol_xhigh']['expected_usd'], fresh_usd * (1 + extra), delta=0.02)
        accounts_250k = copy.deepcopy(base)
        accounts_250k['scoring']['rotate_after_carried_tokens'] = 250_000
        at_120k_250k = self.plan(
            dict(step_task, chain=self.chain(total=2, tokens=120_000, compacted=0, compactions=0)),
            roomy,
            accounts=accounts_250k,
            capabilities=caps,
        )
        self.assertGreater(self.rows(at_120k)['sol_xhigh']['expected_usd'],
                           self.rows(at_120k_250k)['sol_xhigh']['expected_usd'] + 0.5)
        at_150k = self.plan(
            dict(step_task, chain=self.chain(total=2, tokens=90_000, compacted=60_000, compactions=1)),
            roomy,
            accounts=accounts_150k,
            capabilities=caps,
        )
        self.assertEqual(at_150k['decision']['action'], 'rotate')

        # 2. Non-default threshold flows through dispatch_registry.reusable_workers; missing/invalid values raise surfaced errors.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / 'dispatch.sqlite3'
            paseo_home = root / '.paseo'
            agents = paseo_home / 'agents' / 'ws'
            agents.mkdir(parents=True)
            (agents / 'w-reusable.json').write_text(
                json.dumps({'id': 'w-reusable', 'status': 'idle', 'labels': {'router.parent_id': 'parent-1'}}),
                encoding='utf-8',
            )
            with registry.transaction(db) as conn:
                registry.save(conn, {
                    'dispatch_id': 'disp-1', 'decision_id': 'dec-1', 'state': 'completed',
                    'agent_id': 'w-reusable', 'parent_id': 'parent-1',
                    'created_at': self.now.isoformat(), 'updated_at': self.now.isoformat(),
                    # prepare() writes this bound. reusable_workers expires first, and that read
                    # requires it on every completed Paseo row.
                    'hard_until': (self.now + timedelta(seconds=registry.MAX_LIFETIME)).isoformat(),
                    'bundle': {'bundle_id': 'sol_xhigh', 'account': 'codex', 'provider': 'codex', 'model': 'gpt-6.1-sol'},
                    'task': {'task_id': 't-1'},
                    'request': {'title': 'worker task'},
                })

            accounts_750k = copy.deepcopy(base)
            accounts_750k['scoring']['rotate_after_carried_tokens'] = 750_000
            with patch.object(account_fleet, 'load_accounts', return_value=accounts_750k):
                with patch.object(worker_sessions, 'session_state', return_value={'historical_context_tokens': 700_000, 'context_tokens': 100_000, 'compacted_tokens': 600_000}):
                    under = registry.reusable_workers('parent-1', now=self.now + timedelta(minutes=1), path=db, paseo_home=paseo_home)
                    self.assertEqual([w['worn'] for w in under], [False])
                with patch.object(worker_sessions, 'session_state', return_value={'historical_context_tokens': 800_000, 'context_tokens': 100_000, 'compacted_tokens': 700_000}):
                    over = registry.reusable_workers('parent-1', now=self.now + timedelta(minutes=1), path=db, paseo_home=paseo_home)
                    self.assertEqual([w['worn'] for w in over], [True])

            missing = copy.deepcopy(base)
            missing['scoring'].pop('rotate_after_carried_tokens', None)
            with self.assertRaises(KeyError):
                worker_sessions.rotation_threshold(missing)
            with self.assertRaises(KeyError):
                self.plan(step_task, roomy, accounts=missing, capabilities=caps)
            with patch.object(account_fleet, 'load_accounts', return_value=missing), \
                    patch.object(worker_sessions, 'session_state', return_value={'historical_context_tokens': 100_000}):
                with self.assertRaises(KeyError):
                    registry.reusable_workers('parent-1', now=self.now + timedelta(minutes=1), path=db, paseo_home=paseo_home)
            for bad_val in (0, -100, True, '750000'):
                with self.assertRaises(ValueError):
                    worker_sessions.rotation_threshold({'rotate_after_carried_tokens': bad_val})

    def test_a_wait_names_the_window_that_sets_it(self):
        """Rule (`candidate_eval.evaluate_bundle`, `allocator.plan`): `available_in_hours` is the reset of the window that
        holds the candidate back, and the decision names that window (`wait_for`)."""
        usage = self.usage(claude=[('five_hour', 3, 1.0), ('seven_day', 60, 100), ('weekly_fable', 100, 100)], codex=[('usage', 1, 121)])
        task = {'work_kind': 'visual_analysis', 'difficulty': 'routine', 'expected_hours': 1, 'needs': {'visual': True}}
        no_argon = ['argon_high'] + [b for b in account_fleet.bundles(account_fleet.load_accounts()) if b.startswith(('agy_opus', 'agy_sonnet'))]
        deferred = self.plan(dict(task, can_wait_hours=2), usage, unavailable=no_argon)
        wait_for = deferred['decision']['wait_for']
        window = next(w for w in usage['accounts'][wait_for['account']]['windows'] if w['id'] == wait_for['window'])
        self.assertAlmostEqual(window['seconds_until_reset'] / 3600, deferred['decision']['start_after_hours'], delta=0.1)
        long_wait = self.usage(claude=[('five_hour', 3, 4), ('seven_day', 60, 100), ('weekly_fable', 100, 100)], codex=[('usage', 1, 121)])
        impatient = self.plan(dict(task, can_wait_hours=1), long_wait, unavailable=['flash_high'] + no_argon)
        self.assertEqual(impatient['decision']['action'], 'defer')
        self.assertIn(impatient['decision']['wait_for']['window'], impatient['decision']['reason'])
        self.assertIn(impatient['decision']['wait_for']['account'], impatient['decision']['reason'])

    def waiting_fixture(self, five_hours, seven_hours, can_wait, seven_left=3):
        usage = self.usage(claude=[('five_hour', 3, five_hours), ('seven_day', seven_left, seven_hours), ('weekly_fable', 100, 100)],
                           codex=[('usage', 1, 121)])
        task = {'work_kind': 'visual_analysis', 'difficulty': 'routine', 'expected_hours': 1, 'needs': {'visual': True},
                'can_wait_hours': can_wait}
        no_argon = ['argon_high'] + [b for b in account_fleet.bundles(account_fleet.load_accounts()) if b.startswith(('agy_opus', 'agy_sonnet'))]
        return self.plan(task, usage, unavailable=['flash_high'] + no_argon)

    def test_two_windows_that_both_defer_name_the_one_that_sets_the_wait(self):
        """Rule (`candidate_eval.evaluate_bundle`): a candidate held by two windows can start when the last of them returns,
        and `waiting_on` / `wait_for` name that window, whichever of the two is listed first."""
        for label, five_hours, seven_hours, later in (('five-later', 2.0, 1.0, 'five_hour'), ('seven-later', 1.0, 2.0, 'seven_day')):
            with self.subTest(label):
                decision = self.waiting_fixture(five_hours, seven_hours, can_wait=5)['decision']
                self.assertEqual(decision['action'], 'defer')
                self.assertEqual(decision['wait_for']['window'], later)
                self.assertEqual(decision['start_after_hours'], max(five_hours, seven_hours))

    def test_a_candidate_held_by_several_windows_waits_for_the_last_to_return(self):
        """Rule (`candidate_eval.binding_wait`): each window that holds a candidate back moves its wait to the later reset,
        and the window named is the one that sets it, in whatever order the windows are met; every window that holds it
        back is kept with its own wait."""
        import candidate_eval
        for windows in ((('five_hour', 1.0), ('gemini-5h', 2.0), ('3p-5h', 1.5)), (('a', 3.0), ('b', 2.0), ('c', 1.0)),
                        (('x', 2.0), ('y', 2.0))):
            with self.subTest(windows=windows):
                waiting, waits = (None, None), {}
                for window, hours in windows:
                    waiting = candidate_eval.binding_wait(*waiting, hours, window, waits)
                latest = max(hours for _, hours in windows)
                self.assertEqual(waiting[0], latest)
                self.assertEqual(dict(windows)[waiting[1]], latest)
                self.assertEqual(waits, dict(windows))

    def test_the_wait_a_deferral_advises_covers_the_reset_and_is_within_the_limit_when_allowed(self):
        """Rule (`allocator.advised_hours`, `plan` check `available_in_hours > can_wait`): the wait a deferral advises is
        rounded up, so waiting that long finds the window reset, and allowing exactly that wait is no longer over the limit,
        whatever the second decimal of the wait is."""
        for reset in (4.04, 4.06, 0.31, 2.0, 4.99, 0.05):
            with self.subTest(reset=reset):
                first = self.waiting_fixture(reset, 100, can_wait=0.01, seven_left=60)['decision']
                self.assertEqual(first['action'], 'defer')
                advised = first['start_after_hours']
                self.assertGreaterEqual(advised, reset)                  # waiting that long finds the window reset
                self.assertIn(f'{advised}h', first['reason'])            # the text carries the advised value, not another rounding
                self.assertIn('Nothing eligible can start within', first['reason'])
                again = self.waiting_fixture(reset, 100, can_wait=advised, seven_left=60)['decision']
                self.assertNotIn('Nothing eligible can start within', again['reason'])

    def test_quota_held_candidates_wait_at_any_horizon_while_reading_facts_stay_exclusions(self):
        """Rule (`candidate_eval.evaluate_bundle`, `quota_pools.after_reset`): a candidate held back only by quota windows
        (short, small-short, 0%-remaining fresh, multi-day weekly, or `cadence`) with a known reset keeps `available_in_hours`
        and `waiting_on`, whatever the length of the wait, and `--can-wait-hours` reaches past 6h. Reading and capacity facts
        (stale readings, unknown remaining share, insufficient full capacity, non-resetting allowances, subscription ending
        before `wait + hours`, confirmed-exhaustion latches) remain exclusions."""
        with_prof = lambda d: {**d, 'profile': profile_for(d)}
        accounts = copy.deepcopy(account_fleet.load_accounts())
        accounts['accounts']['claude']['bundles']['fable52_high']['disabled'] = False
        for wait_h in (2.5, 8.0, 100.0, 168.0):
            usage = self.usage(
                claude=[('five_hour', 0, wait_h), ('seven_day', 80, 140), ('weekly_fable', 100, 140)],
                codex=[('usage', 0, wait_h)],
                **{'claude-lab': [('five_hour', 0, wait_h), ('seven_day', 80, 140), ('weekly_fable', 100, 140)]},
            )
            for name in ('claude', 'codex', 'claude-lab'):
                usage['accounts'][name]['status'] = 'fresh'
                usage['accounts'][name]['fetched_at'] = self.now.isoformat()
            planned = allocator.plan(
                with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 0.5,
                           'needs': {'local_only': True}, 'can_wait_hours': 10.0}),
                usage, self.now, accounts=copy.deepcopy(accounts), overrides=[],
                host_policy_config={'allowed_accounts': ['claude', 'codex', 'claude-lab']},
            )
            for bid, exp_win in (('opus55_high', 'five_hour'), ('astra_high', 'usage'), ('lab_sonnet55_high', 'five_hour')):
                cand = self.rows(planned)[bid]
                self.assertEqual(cand['excluded'], [], (wait_h, bid, cand['excluded']))
                self.assertAlmostEqual(cand['available_in_hours'], wait_h, places=6)
                self.assertEqual(cand['waiting_on'], exp_win)
            self.assertNotIn('next_reset', planned['decision'])
            self.assertEqual(planned['decision']['action'], 'defer')
            if wait_h == 8.0:
                self.assertNotIn('Nothing eligible can start within', planned['decision']['reason'])

        # Reading and capacity facts stay exclusions:
        # 1. Stale reading on an empty window
        stale_u = self.usage(codex=[('usage', 0, 8.0)])
        stale_u['accounts']['codex']['status'] = 'stale'
        stale_plan = allocator.plan(
            with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 0.5,
                       'needs': {'local_only': True}}),
            stale_u, self.now, accounts=copy.deepcopy(accounts), overrides=[],
            host_policy_config={'allowed_accounts': ['codex']},
        )
        self.assertTrue(self.rows(stale_plan)['astra_high']['excluded'])
        self.assertEqual(stale_plan['choice'], 'none')

        # 2. Subscription ending before `wait + hours` (both single-window and multi-window)
        exp_accounts = copy.deepcopy(accounts)
        exp_accounts['accounts']['claude']['expires_at'] = (self.now + timedelta(hours=50)).isoformat()
        exp_u = self.usage(claude=[('five_hour', 90, 3.0), ('seven_day', 0, 49.5), ('weekly_fable', 100, 100.0)])
        exp_plan = allocator.plan(
            with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 1.0,
                       'needs': {'local_only': True}}),
            exp_u, self.now, accounts=exp_accounts, overrides=[],
            host_policy_config={'allowed_accounts': ['claude']},
        )
        opus_exp = self.rows(exp_plan)['opus55_high']
        self.assertIn('subscription ends before the work would finish', opus_exp['excluded'])

        # 3. Confirmed-exhaustion latch (`quota_exhaustion`) stays an exclusion even when a reset timestamp is in the future
        latched_accounts = copy.deepcopy(accounts)
        latched_accounts['_discovery'] = {'accounts': {'codex': {'status': 'ready', 'exhausted_windows': ['usage']}}}
        latched_u = self.usage(codex=[('usage', 0, 4.0)])
        latched_plan = allocator.plan(
            with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 0.5,
                       'needs': {'local_only': True}}),
            latched_u, self.now, accounts=latched_accounts, overrides=[],
            host_policy_config={'allowed_accounts': ['codex']},
        )
        self.assertTrue(any('confirmed exhausted' in r for r in self.rows(latched_plan)['astra_high']['excluded']))

    def test_can_wait_hours_orders_by_ranking_within_limit_and_earliest_start_beyond_limit(self):
        """Rules 2 & 3 (`allocator.plan`, `candidate_ranking.five_axis_order`):
        - Within `can_wait_hours`, candidates compete by `five_axis_order`, and nothing starting after `can_wait_hours` is
          chosen while any candidate starts within `can_wait_hours`.
        - When nothing eligible can start within `can_wait_hours`, `plan` defers to the least `available_in_hours` (ties broken
          by `five_axis_order`), never to a higher score days away; `better_if_waiting` never names a candidate beyond
          `can_wait_hours`."""
        import candidate_ranking
        with_prof = lambda d: {**d, 'profile': profile_for(d)}
        accounts = copy.deepcopy(account_fleet.load_accounts())
        accounts['accounts']['claude']['bundles']['fable52_high']['disabled'] = False
        # Codex returns in 8h; Claude returns in 120h (5 days); Claude-lab returns in 3h on five_hour.
        usage = self.usage(
            claude=[('five_hour', 90, 3.0), ('seven_day', 0, 120.0), ('weekly_fable', 0, 120.0)],
            codex=[('usage', 0, 8.0)],
            **{'claude-lab': [('five_hour', 0, 3.0), ('seven_day', 85, 100.0), ('weekly_fable', 100, 100.0)]},
        )
        task_base = {'work_kind': 'architecture_reasoning', 'difficulty': 'coupled', 'expected_hours': 0.5,
                     'needs': {'local_only': True}}
        policy_cfg = {'allowed_accounts': ['claude', 'codex', 'claude-lab']}

        # 1. Beyond limit (`can_wait_hours=0` and `can_wait_hours=2`): nothing starts within limit, so earliest start (3.0h) wins,
        #    never a higher-scoring candidate 8h or 120h away.
        for limit in (0, 2.0):
            beyond = allocator.plan(with_prof({**task_base, 'can_wait_hours': limit}),
                                    copy.deepcopy(usage), self.now, accounts=copy.deepcopy(accounts),
                                    overrides=[], host_policy_config=policy_cfg)
            elig = [c for c in beyond['ranking'] if not c['excluded']]
            least_wait = min(c['available_in_hours'] for c in elig)
            chosen = self.rows(beyond)[beyond['choice']]
            self.assertEqual(beyond['decision']['action'], 'defer')
            self.assertAlmostEqual(chosen['available_in_hours'], least_wait, places=6)
            # Among equal least waits, `five_axis_order` breaks ties:
            tied = sorted((c for c in elig if round(c['available_in_hours'], 4) == round(least_wait, 4)),
                          key=candidate_ranking.five_axis_order)
            self.assertEqual(beyond['choice'], tied[0]['id'])
            self.assertNotIn('better_if_waiting', beyond['decision'])

        # 2. Within limit (`can_wait_hours=10`): 3h and 8h candidates are in `ready`; 120h candidates are excluded from winning,
        #    and `choice` equals the first `five_axis_order` candidate with `(available_in_hours or 0) <= 10`.
        within_10 = allocator.plan(with_prof({**task_base, 'can_wait_hours': 10.0}),
                                   copy.deepcopy(usage), self.now, accounts=copy.deepcopy(accounts),
                                   overrides=[], host_policy_config=policy_cfg)
        elig_10 = [c for c in within_10['ranking'] if not c['excluded']]
        ready_10 = sorted((c for c in elig_10 if (c['available_in_hours'] or 0) <= 10.0),
                          key=candidate_ranking.five_axis_order)
        self.assertEqual(within_10['choice'], ready_10[0]['id'])
        self.assertLessEqual(self.rows(within_10)[within_10['choice']]['available_in_hours'] or 0.0, 10.0)
        self.assertNotIn('better_if_waiting', within_10['decision'])

    def test_deferral_and_long_wait_pricing_are_invariant_to_input_order_and_respect_crossed_resets(self):
        """Rules (`AGENTS.md` order invariance; `candidate_eval.evaluate_bundle` & `quota_economy.five_axis_spend_cost`
        post-wait window alignment):
        - Reversing account/bundle/window order changes neither `choice`, `start_after_hours`, nor per-candidate scores/waits.
        - A second drawn window whose reset falls at or before `available_in_hours` is priced on its renewed cycle (`used_usd == 0`),
          not its expired pre-wait cycle; one resetting after `available_in_hours` keeps its observed `used_usd` and splits
          crossing spend at `reset_hours - available_in_hours`."""
        with_prof = lambda d: {**d, 'profile': profile_for(d)}
        accounts = copy.deepcopy(account_fleet.load_accounts())
        accounts['accounts']['claude']['bundles']['fable52_high']['disabled'] = False
        # `fable52_high` on `claude` draws `five_hour`, `seven_day` (resets in 20h, 80% used), and `weekly_fable` (empty, resets in 100h).
        usage = self.usage(
            claude=[('five_hour', 90, 3.0), ('seven_day', 20, 20.0), ('weekly_fable', 0, 100.0)],
            codex=[('usage', 0, 50.0)],
        )
        rev_accounts = copy.deepcopy(accounts)
        rev_accounts['accounts'] = dict(reversed(list(rev_accounts['accounts'].items())))
        for spec in rev_accounts['accounts'].values():
            if isinstance(spec, dict) and isinstance(spec.get('bundles'), dict):
                spec['bundles'] = dict(reversed(list(spec['bundles'].items())))
            if isinstance(spec, dict) and isinstance(spec.get('windows'), dict):
                spec['windows'] = dict(reversed(list(spec['windows'].items())))
        rev_usage = copy.deepcopy(usage)
        for row in rev_usage['accounts'].values():
            row['windows'] = list(reversed(row['windows']))

        for can_wait in (0.0, 60.0, 120.0):
            task = with_prof({'work_kind': 'creative_ideation', 'difficulty': 'coupled',
                              'expected_hours': 0.5, 'needs': {'local_only': True}, 'can_wait_hours': can_wait})
            a = allocator.plan(copy.deepcopy(task), copy.deepcopy(usage), self.now, accounts=copy.deepcopy(accounts),
                               overrides=[], host_policy_config={'allowed_accounts': ['claude', 'codex']})
            b = allocator.plan(copy.deepcopy(task), copy.deepcopy(rev_usage), self.now, accounts=copy.deepcopy(rev_accounts),
                               overrides=[], host_policy_config={'allowed_accounts': ['codex', 'claude']})
            self.assertEqual((a['choice'], a['decision']['action'], a['decision'].get('start_after_hours'), a['decision'].get('wait_for')),
                             (b['choice'], b['decision']['action'], b['decision'].get('start_after_hours'), b['decision'].get('wait_for')))
            self.assertEqual([c['id'] for c in a['ranking'] if not c['excluded']],
                             [c['id'] for c in b['ranking'] if not c['excluded']])
            by_id_a = {c['id']: (c['score'], c['available_in_hours'], c['waiting_on'], tuple(sorted(c['excluded']))) for c in a['ranking']}
            by_id_b = {c['id']: (c['score'], c['available_in_hours'], c['waiting_on'], tuple(sorted(c['excluded']))) for c in b['ranking']}
            self.assertEqual(by_id_a, by_id_b)

        # Crossed reset check on `fable52_high`: `seven_day` (reset 20h <= 100h wait on `weekly_fable`) is renewed (`used_usd == 0`).
        fable = self.rows(a)['fable52_high']
        seven_econ = next(w for w in fable['economy']['windows'] if w['window'] == 'seven_day')
        self.assertEqual(seven_econ['used_usd'], 0.0)
        self.assertTrue(any('seven_day also renews before the 100.0h wait ends' in n for n in fable['notes']), fable['notes'])

        # Uncrossed reset check: `five_hour` defers by 4h while `seven_day` resets in 4.5h (0.5h after start) on a multi-hour task.
        cross_u = self.usage(claude=[('five_hour', 0, 4.0), ('seven_day', 50, 4.5), ('weekly_fable', 100, 100.0)])
        cross_plan = allocator.plan(
            with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 3.0,
                       'needs': {'local_only': True}, 'can_wait_hours': 5.0}),
            cross_u, self.now, accounts=copy.deepcopy(accounts), overrides=[],
            host_policy_config={'allowed_accounts': ['claude']},
        )
        opus = self.rows(cross_plan)['opus55_high']
        seven_cross = next(w for w in opus['economy']['windows'] if w['window'] == 'seven_day')
        self.assertGreater(seven_cross['used_usd'], 0.0)
        self.assertGreater(seven_cross['spend_after_reset'], 0.0)
        self.assertAlmostEqual(
            seven_cross['spend_before_reset'] / (seven_cross['spend_before_reset'] + seven_cross['spend_after_reset']),
            0.5 / opus['duration_hours'], places=2,
        )

    def test_fast_that_the_plan_did_not_apply_says_why(self):
        """Rule (`allocator.plan`): a task that wants fast mode and is routed to a plain bundle carries the reason the plan
        holds for it (`fast_not_applied`); one routed to a fast bundle carries none."""
        for plan in (self.plan({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 1, 'wants_fast': True}),
                     self.plan({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 1})):
            decision = plan['decision']
            if plan['decision']['task'].get('wants_fast') and not plan['selected_bundle'].get('fast'):
                self.assertTrue(str(decision.get('fast_not_applied') or '').strip())
            else:
                self.assertNotIn('fast_not_applied', decision)

    def test_named_model_held_only_by_resetting_windows_defers_to_its_earliest_bundle_at_any_wait_limit(self):
        """Rule (`allocator.plan`, `allocator.named_pin_block`): a named model whose matching bundles are held back only by
        quota windows (or `cadence`) with an observed reset is a deferral (`decision.action == 'defer'`, `start_after_hours`),
        never `choice: 'none'` / `named_model_unavailable` and never a substitute model, whether the wait is within or beyond
        `can_wait_hours`. Beyond `can_wait_hours` it defers to the matching bundle with the least `available_in_hours`; only
        when every matching bundle is excluded by a fact does `plan` return `choice: 'none'` / `named_model_unavailable`
        (`named_pin_block` reports `c['excluded']` only, never `'cadence quota exhausted'`)."""
        with_prof = lambda d: {**d, 'profile': profile_for(d)}
        accounts = copy.deepcopy(account_fleet.load_accounts())
        # Give a second Claude account (`claude-lab`) `claude-opus-5-5` too so bare naming spans two accounts with different waits.
        accounts['accounts']['claude-lab']['bundles']['lab_opus55_high'] = dict(
            accounts['accounts']['claude']['bundles']['opus55_high'],
            id='lab_opus55_high', account='claude-lab', provider='claude-lab',
        )
        usage = self.usage(
            claude=[('five_hour', 2, 2.0), ('seven_day', 90, 100.0), ('weekly_fable', 100, 100.0)],
            codex=[('usage', 90, 100.0)],
            **{'claude-lab': [('five_hour', 0, 8.0), ('seven_day', 90, 100.0), ('weekly_fable', 100, 100.0)]},
        )
        policy_cfg = {'allowed_accounts': ['claude', 'codex', 'claude-lab']}
        for can_wait in (0.0, 1.0, 3.0, 10.0):
            planned = allocator.plan(
                with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 0.3,
                           'needs': {'local_only': True}, 'can_wait_hours': can_wait,
                           'constraints': {'model': 'claude-opus-5-5'}}),
                copy.deepcopy(usage), self.now, accounts=copy.deepcopy(accounts), overrides=[],
                host_policy_config=policy_cfg,
            )
            elig_waits = {c['bundle']['account']: c['available_in_hours'] for c in planned['ranking'] if not c['excluded']}
            self.assertEqual(elig_waits, {'claude': 2.0, 'claude-lab': 8.0})
            self.assertNotEqual(planned['choice'], 'none', (can_wait, planned['decision']))
            self.assertNotIn('blocked', planned['decision'])
            self.assertEqual(planned['decision']['action'], 'defer')
            self.assertEqual(planned['selected_bundle']['model'], 'claude-opus-5-5')
            if can_wait < 2.0:
                # Beyond limit: earliest-starting matching bundle (2.0h on `claude`, never 8.0h on `claude-lab` or ready `codex`).
                self.assertEqual(planned['decision']['start_after_hours'], 2.0)

        # Cadence hold on a named model also defers rather than claiming "cadence quota exhausted":
        cad_accounts = copy.deepcopy(accounts)
        cad_accounts['accounts']['codex']['cadence'] = {'window_seconds': 12 * 3600, 'limit': 1}
        cad_records = [{'stage': 'launch', 'dispatch_id': 'd-1', 'decision_id': 'dec-1',
                        'at': (self.now - timedelta(hours=10)).isoformat(),
                        'bundle': {'account': 'codex', 'provider': 'codex', 'model': 'gpt-6.1-sol', 'bundle_id': 'sol_high'}}]
        cad_plan = allocator.plan(
            with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 0.5,
                       'needs': {'local_only': True}, 'can_wait_hours': 0.0,
                       'constraints': {'model': 'gpt-6.1-sol'}}),
            self.usage(codex=[('usage', 90, 100.0)], claude=[('five_hour', 90, 3.0), ('seven_day', 90, 100.0)]),
            self.now, records=cad_records, accounts=cad_accounts, overrides=[],
            host_policy_config={'allowed_accounts': ['claude', 'codex']},
        )
        self.assertEqual(cad_plan['decision']['action'], 'defer')
        self.assertEqual(cad_plan['selected_bundle']['model'], 'gpt-6.1-sol')
        self.assertEqual(cad_plan['decision']['wait_for']['window'], 'cadence')
        self.assertNotIn('cadence quota exhausted', json.dumps(cad_plan))

        # Every matching bundle excluded by a fact still returns `choice: 'none'` / `named_model_unavailable`:
        stale_opus = copy.deepcopy(usage)
        stale_opus['accounts']['claude'].update(status='stale', fetched_at=self.now.isoformat())
        stale_opus['accounts']['claude']['windows'][0]['remaining_pct_before_reset'] = 0
        stale_opus['accounts']['claude-lab'].update(status='stale', fetched_at=self.now.isoformat())
        blocked = allocator.plan(
            with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 1.0,
                       'needs': {'local_only': True}, 'can_wait_hours': 10.0,
                       'constraints': {'model': 'claude-opus-5-5'}}),
            stale_opus, self.now, accounts=copy.deepcopy(accounts), overrides=[],
            host_policy_config=policy_cfg,
        )
        self.assertEqual((blocked['choice'], blocked['decision'].get('blocked')), ('none', 'named_model_unavailable'))

    def test_post_wait_window_renewal_enforces_full_capacity_on_short_and_long_windows(self):
        """Rule (`candidate_eval.renewal_capacity_refusal`, `candidate_eval.evaluate_bundle`): every place a window is read
        as renewed—including a second short window whose reset falls at or before a longer window's `available_in_hours`—checks
        full renewed capacity against the period's spend plus margin, so a candidate excluded after the wait is never deferred
        before the wait (invariant to window order)."""
        with_prof = lambda d: {**d, 'profile': profile_for(d)}
        accounts = copy.deepcopy(account_fleet.load_accounts())
        task = with_prof({'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 3.8,
                          'needs': {'local_only': True}})
        # Step 1 (t=0, seven_day ample, five_hour at 90% resetting in 1.0h): 3.8h splits as 1.0h before reset + 2.8h after,
        # so candidates whose 3.8h exceeds one 5h period's `100% - 5%` margin but whose 1.0h and 2.8h pieces each fit are eligible now.
        u_now = self.usage(claude=[('five_hour', 90, 1.0), ('seven_day', 90, 100.0), ('weekly_fable', 100, 100.0)])
        p_now = allocator.plan(copy.deepcopy(task), u_now, self.now, accounts=copy.deepcopy(accounts), overrides=[],
                               host_policy_config={'allowed_accounts': ['claude']})
        # Step 3 (t=10h after waiting: five_hour unstarted at 100%, seven_day renewed at 100%): all 3.8h falls inside one 5h period.
        u_after = self.usage(claude=[('five_hour', 100, 5.0), ('seven_day', 100, 168.0), ('weekly_fable', 100, 90.0)])
        u_after['accounts']['claude']['windows'][0].update(used_pct=0, resets_at=None, seconds_until_reset=None)
        p_after = allocator.plan(copy.deepcopy(task), u_after, self.now + timedelta(hours=10),
                                 accounts=copy.deepcopy(accounts), overrides=[],
                                 host_policy_config={'allowed_accounts': ['claude']})
        split_only = [cid for cid, row in self.rows(p_now).items()
                      if not row['excluded'] and any('five_hour full capacity' in e for e in self.rows(p_after)[cid]['excluded'])]
        self.assertTrue(split_only, 'expected at least one bundle that fits split across 1h+2.8h but exceeds one 5h period')

        # Step 2 (t=0, seven_day at 0% resetting in 10.0h, in both window orders): deferring to 10.0h crosses five_hour's 1.0h
        # reset, so every `split_only` bundle must already carry the same `five_hour full capacity` exclusion as Step 3.
        for rev in (False, True):
            wins = [('five_hour', 90, 1.0), ('seven_day', 0, 10.0), ('weekly_fable', 100, 100.0)]
            if rev:
                wins.reverse()
            u_defer = self.usage(claude=wins)
            acc_defer = copy.deepcopy(accounts)
            if rev:
                acc_defer['accounts']['claude']['windows'] = dict(reversed(list(acc_defer['accounts']['claude']['windows'].items())))
            p_defer = allocator.plan(copy.deepcopy(task), u_defer, self.now, accounts=acc_defer, overrides=[],
                                     host_policy_config={'allowed_accounts': ['claude']})
            for cid in split_only:
                self.assertEqual(
                    [e for e in self.rows(p_defer)[cid]['excluded'] if 'five_hour full capacity' in e],
                    [e for e in self.rows(p_after)[cid]['excluded'] if 'five_hour full capacity' in e],
                    (rev, cid),
                )
            self.assertEqual(p_defer['choice'], p_after['choice'])

    def test_named_model_starts_now_only_ever_names_a_bundle_of_that_model(self):
        """Rule (`allocator.plan`, `allocator.named_model_keys`, `roll.defer_next`): `decision.starts_now` is the offer
        'this can start at once instead'; for every way a model is named (`constraints.model`, `constraints.bundle`,
        `route={'pinned_bundle': ...}`), `starts_now` equals the best eligible ready-now bundle of that named model's
        `row_key` when one exists and is absent otherwise—never another model's bundle."""
        import roll
        with_prof = lambda d: {**d, 'profile': profile_for(d)}
        base_accounts = copy.deepcopy(account_fleet.load_accounts())
        base_task = {'work_kind': 'code_implementation', 'difficulty': 'hard', 'expected_hours': 0.3,
                     'needs': {'local_only': True}, 'can_wait_hours': 3.0}
        ways = (
            ('constraint_model', {'constraints': {'model': 'claude-opus-5-5'}}, None),
            ('constraint_bundle', {'constraints': {'bundle': 'opus55_high'}}, None),
            ('pinned_bundle', {}, {'pinned_bundle': 'opus55_high'}),
            ('user_named_pinned_bundle', {}, {'pinned_bundle': 'opus55_high', 'user_named_bundle': 'opus55_high'}),
        )

        # 1. Only other models (`codex`) can start now while `claude-opus-5-5` waits 2.0h: `starts_now` is absent everywhere.
        usage_none_same = self.usage(
            claude=[('five_hour', 2, 2.0), ('seven_day', 90, 100.0), ('weekly_fable', 100, 100.0)],
            codex=[('usage', 90, 100.0)],
        )
        for label, extra, route in ways:
            with self.subTest(case='no_same_model_ready', way=label):
                planned = allocator.plan(
                    with_prof({**base_task, **extra}), copy.deepcopy(usage_none_same), self.now,
                    accounts=copy.deepcopy(base_accounts), overrides=[],
                    host_policy_config={'allowed_accounts': ['claude', 'codex']}, route=route,
                )
                decision = planned['decision']
                self.assertEqual(decision['action'], 'defer')
                self.assertTrue(decision['task'].get('named_model'))
                self.assertNotIn('starts_now', decision)
                nxt = roll.defer_next(planned, decision['start_after_hours'])
                self.assertIn('do not substitute another model', nxt)
                self.assertNotIn('without --can-wait-hours', nxt)

        # 2. Both a higher-ranked other model (`codex`, boosted) and a ready-now bundle of `claude-opus-5-5` on `claude-lab`
        #    can start now while `claude`'s `opus55_high` waits 2.0h: `starts_now` picks only the ready-now `claude-opus-5-5`
        #    bundle when eligible (`constraint_model`, `pinned_bundle`) and stays absent on `constraint_bundle`.
        two_acct = copy.deepcopy(base_accounts)
        two_acct['accounts']['claude-lab']['bundles']['lab_opus55_low'] = dict(
            two_acct['accounts']['claude']['bundles']['opus55_high'],
            id='lab_opus55_low', effort='low', account='claude-lab', provider='claude-lab', disabled=False,
        )
        usage_both_ready = self.usage(
            claude=[('five_hour', 2, 2.0), ('seven_day', 90, 100.0), ('weekly_fable', 100, 100.0)],
            codex=[('usage', 95, 100.0)],
            **{'claude-lab': [('five_hour', 95, 3.0), ('seven_day', 95, 100.0), ('weekly_fable', 100, 100.0)]},
        )
        codex_boost = [{'id': 'b-codex', 'account': 'codex', 'intent': 'priority', 'boost': 2.0,
                        'until': (self.now + timedelta(hours=5)).isoformat()}]
        for label, extra, route in ways:
            with self.subTest(case='same_and_other_ready', way=label):
                planned = allocator.plan(
                    with_prof({**base_task, **extra}), copy.deepcopy(usage_both_ready), self.now,
                    accounts=copy.deepcopy(two_acct), overrides=copy.deepcopy(codex_boost),
                    host_policy_config={'allowed_accounts': ['claude', 'codex', 'claude-lab']}, route=route,
                )
                decision = planned['decision']
                self.assertEqual(decision['action'], 'defer')
                same_ready = [c['id'] for c in planned['ranking']
                              if not c['excluded'] and not c['available_in_hours']
                              and account_fleet.row_key(c['bundle']['model'], two_acct) == 'claude-opus-5-5']
                if route:
                    other_ready_ahead = [c['id'] for c in planned['ranking']
                                         if not c['excluded'] and not c['available_in_hours']
                                         and account_fleet.row_key(c['bundle']['model'], two_acct) != 'claude-opus-5-5'
                                         and c['score'] > self.rows(planned)[same_ready[0]]['score']]
                    self.assertTrue(other_ready_ahead)
                expected_starts_now = same_ready[0] if same_ready else None
                self.assertEqual(decision.get('starts_now'), expected_starts_now)
                if expected_starts_now is not None:
                    self.assertEqual(
                        account_fleet.row_key(self.rows(planned)[decision['starts_now']]['bundle']['model'], two_acct),
                        'claude-opus-5-5',
                    )


class OverrideContractTests(unittest.TestCase):
    """Contract tests for temporary priority overrides, reset confirmation, receipts, and atomic store."""

    def setUp(self):
        self.root = temporary_root(self.addCleanup)
        self.path = self.root / 'overrides.json'
        self.now = datetime(2026, 10, 16, 9, tzinfo=timezone.utc)
        self.baseline = self.now - timedelta(hours=1)
        self.reset = self.now + timedelta(days=6)
        self.accounts = account_fleet.load_accounts(account_fleet.reference_fleet_path())
        self.usage = {'accounts': {'codex': {
            'status': 'fresh', 'fetched_at': self.now.isoformat(), 'observation_id': 'new',
            'windows': [{'id': 'usage', 'used_pct': 94, 'remaining_pct_before_reset': 6,
                         'seconds_until_reset': 6 * 86400, 'resets_at': self.reset.isoformat()}]}}}
        self.override = {'id': 'priority', 'account': 'codex', 'model': 'gpt-6-astra',
                         'intent': 'user-priority', 'boost': 0.35, 'stop_on_reset': True,
                         'created_at': self.baseline.isoformat(),
                         'expected_reset_at': (self.now - timedelta(minutes=40)).isoformat(),
                         'until': (self.now + timedelta(hours=23)).isoformat(),
                         'baseline_windows': {'usage': {'account': 'codex', 'used_pct': 94,
                             'resets_at': self.reset.isoformat(), 'fetched_at': self.baseline.isoformat(),
                             'observation_id': 'old'}}}
        env = patch.dict(os.environ, {'ROUTER_OVERRIDES': str(self.path),
                                      'ROUTER_CAPACITY': str(self.root / 'capacity.json')})
        env.start()
        self.addCleanup(env.stop)

    def args(self, **changes):
        fields = dict(account='codex', model='gpt-6-astra', window='usage', intent='priority',
                      basis='user priority', boost=0.35, resets_at=self.override['expected_reset_at'],
                      resets_in=None, expires_at=None, expires_in=None, until=self.override['until'],
                      stop_on_reset=True, list=False, clear=None, usage_cache=None)
        return argparse.Namespace(**dict(fields, **changes))

    def replenished(self):
        usage = copy.deepcopy(self.usage)
        usage['accounts']['codex']['windows'][0].update(
            used_pct=0, remaining_pct_before_reset=100, resets_at=(self.now + timedelta(days=7)).isoformat())
        return usage

    def same_cycle(self, used=0):
        usage = self.replenished()
        usage['accounts']['codex']['windows'][0].update(
            used_pct=used, remaining_pct_before_reset=100 - used, resets_at=self.reset.isoformat())
        return usage

    def depleted(self, remaining=5):
        usage = copy.deepcopy(self.usage)
        usage['accounts']['codex']['windows'][0].update(used_pct=100 - remaining, remaining_pct_before_reset=remaining)
        return usage

    def test_overdue_forecast_cannot_grant_capacity_and_boost_targets_named_model(self):
        original = copy.deepcopy(self.usage)
        plan = allocator.plan({'work_kind': 'code_implementation', 'difficulty': 'hard',
                               'expected_hours': 1, 'needs': {'local_only': True}},
                              self.usage, self.now, accounts=self.accounts, overrides=[self.override])
        astra = next(r for r in plan['ranking'] if r['id'] == 'astra_high')
        sol = next(r for r in plan['ranking'] if r['id'] == 'sol_high')
        # Rule: an overdue `expected_reset_at` never grants immediate capacity; the candidate still waits on the observed 6-day reset.
        self.assertFalse(astra['excluded'])
        self.assertEqual((astra['available_in_hours'], astra['waiting_on']), (144.0, 'usage'))
        self.assertNotEqual(plan['choice'], 'astra_high')
        self.assertEqual(astra['components']['boost'], 0.35)
        self.assertNotIn('boost', sol['components'])
        self.assertTrue(any('overdue_uncertain' in n for n in astra['notes']))
        self.assertEqual(self.usage, original)

    def test_reset_confirmation_scheduled_and_one_off(self):
        one_off = dict(self.override, reset_confirmation='replenishment')
        self.assertTrue(quota_overrides.override_reset_landed(self.override, self.replenished(), self.accounts, self.now))
        self.assertFalse(quota_overrides.override_reset_landed(self.override, self.same_cycle(), self.accounts, self.now))
        self.assertTrue(quota_overrides.override_reset_landed(one_off, self.same_cycle(), self.accounts, self.now))
        for invalid in ({'status': 'stale'}, {'stale': True}, {'available': False}, {'account': 'codex-work'},
                        {'fetched_at': self.baseline.isoformat()}, {'observation_id': 'old'},
                        {'fetched_at': (self.now - timedelta(minutes=16)).isoformat()},
                        {'fetched_at': (self.now + timedelta(seconds=1)).isoformat()}):
            with self.subTest(invalid=invalid):
                usage = self.replenished()
                usage['accounts']['codex'].update(invalid)
                self.assertFalse(quota_overrides.override_reset_landed(self.override, usage, self.accounts, self.now))
                sc = self.same_cycle()
                sc['accounts']['codex'].update(invalid)
                self.assertFalse(quota_overrides.override_reset_landed(one_off, sc, self.accounts, self.now))

    def test_reset_receipts_and_pruning(self):
        self.path.write_text(json.dumps({'overrides': [self.override], 'unrelated': 7}))
        quota_overrides.observe_landed_overrides({'priority': self.override}, self.path, self.now)
        first = json.loads(self.path.read_text())
        marked = first['overrides'][0]
        self.assertEqual(marked['reset_confirmed_at'], self.now.isoformat())
        later = self.now + timedelta(hours=2)
        quota_overrides.observe_landed_overrides({'priority': marked}, self.path, later)
        self.assertEqual(json.loads(self.path.read_text()), first)
        quota_overrides.prune_landed_overrides({'priority': marked}, self.path, later)
        final = json.loads(self.path.read_text())
        self.assertEqual((final['overrides'], final['last_reset']['at'], final['unrelated']), ([], self.now.isoformat(), 7))

        # Confirmed marker survives stale usage; future/pre-creation marker is rejected; replaced record is untouched
        stale_usage = copy.deepcopy(self.usage)
        stale_usage['accounts']['codex']['status'] = 'stale'
        self.assertTrue(quota_overrides.override_reset_landed(marked, stale_usage, self.accounts, later))
        for bad_when in (self.now + timedelta(seconds=1), self.baseline - timedelta(seconds=1)):
            self.assertFalse(quota_overrides.override_reset_landed(dict(self.override, reset_confirmed_at=bad_when.isoformat()),
                                                             self.usage, self.accounts, self.now))
        newer = dict(self.override, boost=0.4)
        self.path.write_text(json.dumps({'overrides': [newer]}))
        quota_overrides.prune_landed_overrides({'priority': self.override}, self.path, self.now)
        self.assertEqual(json.loads(self.path.read_text())['overrides'], [newer])

    def test_the_override_write_is_an_effect_before_its_ledger_row(self):
        """Rule (round 4, review round 3 E3): an override the call was asked to write is an effect the moment it is
        written, so a data error between the write and the ledger row is not re-run on the shipped set."""
        import public_data
        marked, at = [], []
        real = public_data.effect
        with patch.object(usage_reading, 'usage_snapshot', return_value=self.usage), \
                patch.object(public_data, 'effect', lambda what: marked.append(what) or real(what)), \
                patch.object(decision_ledger, 'append_ledger', lambda row: at.append(list(marked))):
            quota_overrides.manage_overrides(self.args(), self.now)
        self.assertEqual(at, [['quota override']])

    @pytest.mark.slow
    def test_upsert_rebaseline_and_atomic_concurrency(self):
        with patch.object(usage_reading, 'usage_snapshot', return_value=self.usage):
            first = quota_overrides.manage_overrides(self.args(), self.now)['active'][0]
            override_rows = [row for row in decision_ledger.read_ledger() if row.get('stage') == 'override']
            self.assertEqual((override_rows[-1]['action'], override_rows[-1]['account'], override_rows[-1]['removed']),
                             ('upsert', 'codex', 0))
            second = quota_overrides.manage_overrides(self.args(boost=0.4), self.now + timedelta(seconds=1))['active'][0]
            self.assertEqual((second['id'], second['created_at'], second['boost']), (first['id'], first['created_at'], 0.4))

        # Atomic write failure retains original file
        self.path.write_text('{"overrides": []}')
        with override_store.locked(self.path), patch.object(override_store.os, 'replace', side_effect=OSError('fixture failure')):
            with self.assertRaises(OSError):
                override_store.write(self.path, {'overrides': [self.override]})
        self.assertEqual(json.loads(self.path.read_text()), {'overrides': []})

        # Concurrent process upserts do not lose intents
        cache = self.root / 'usage.json'
        cache.write_text(json.dumps({'providers': {'codex': {}}}))
        env = dict(os.environ, ROUTER_USAGE_CACHE=str(cache), ROUTER_LEDGER=str(self.root / 'ledger.jsonl'),
                   ROUTER_HOST_POLICY=str(self.root / 'unused-policy.json'), PYTHONDONTWRITEBYTECODE='1')
        script = str(Path(router.__file__))
        processes = [subprocess.Popen([sys.executable, '-B', script, 'override', '--account', 'codex',
                                       '--intent', intent, '--basis', 'fixture', '--boost', '0.1'],
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for intent in ('a', 'b', 'a', 'c', 'd')]
        try:
            for p in processes:
                out, err = p.communicate(timeout=30)
                self.assertEqual(p.returncode, 0, err or out)
        finally:
            for p in processes:
                if p.poll() is None:
                    p.kill()
                    p.communicate()
        rows = json.loads(self.path.read_text())['overrides']
        self.assertEqual(({r['intent'] for r in rows}, len(rows)), ({'a', 'b', 'c', 'd'}, 4))

    def test_depletion_retirement(self):
        self.assertTrue(quota_overrides.override_depleted(self.override, self.depleted(5), self.accounts))
        self.assertFalse(quota_overrides.override_depleted(self.override, self.depleted(5.5), self.accounts))
        other = dict(self.override, id='other', account='claude', model='claude-opus-5-5')
        self.path.write_text(json.dumps({'v': 1, 'overrides': [self.override, other]}))
        _, _, applied = quota_overrides.apply_overrides([self.override], self.depleted(4), self.accounts, self.now)
        self.assertEqual(applied, [])
        self.assertEqual(json.loads(self.path.read_text()), {'v': 1, 'overrides': [other]})


class AntigravityTelemetryTests(unittest.TestCase):
    """Contract tests for Antigravity resilient transcript telemetry and handoff rotation."""

    def setUp(self):
        self.home = temporary_root(self.addCleanup)
        self.agent = self.home / '.paseo/agents/workspace/agent.json'
        self.agent.parent.mkdir(parents=True)
        self.log = self.home / '.gemini/antigravity-cli/brain/fixture/.system_generated/logs/transcript.jsonl'
        self.log.parent.mkdir(parents=True)

    def persistence(self, data=None, session=None):
        self.agent.write_text(json.dumps({'persistence': {'provider': 'antigravity-resilient',
            'sessionId': session or 'plugin:' + json.dumps({'version': 1, 'data': data or {'conversationId': 'fixture'}})}}))

    def test_resilient_bundles_and_session_key_binding(self):
        bundles = account_fleet.bundles(account_fleet.load_accounts(account_fleet.reference_fleet_path()))
        for name in ('flash_high', 'argon_high'):
            self.assertEqual((bundles[name]['provider'], bundles[name]['account'], bundles[name]['modeId']),
                             ('antigravity-resilient', 'antigravity', 'full-access'))
        session_key = 'workspace/session:fixture'
        binding = self.home / '.paseo/antigravity-resilient/conversations/session-keys' / (antigravity_telemetry.safe_key(session_key) + '.json')
        binding.parent.mkdir(parents=True)
        binding.write_text(json.dumps({'conversationId': 'fixture', 'secret': 'secret-canary'}))
        self.log.write_text(json.dumps({'type': 'PLANNER_RESPONSE', 'step_index': 1, 'usage': {'input_tokens': 123}}))
        for structured in (False, True):
            self.persistence({'sessionKey': session_key})
            if structured:
                self.agent.write_text(json.dumps({'persistence': {'provider': 'antigravity-resilient',
                    'version': 1, 'data': {'sessionKey': session_key}}}))
            self.assertEqual(worker_sessions.session_state('agent', self.home)['context_tokens'], 123)

    def test_native_transcript_checkpoints_and_partial_evidence(self):
        self.persistence()
        steps = [
            {'type': 'PLANNER_RESPONSE', 'step_index': 1, 'content': 'secret-canary',
             'input_tokens': 20000, 'cache_read_tokens': 180000, 'output_tokens': 999999},
            {'type': 'CHECKPOINT', 'step_index': 2, 'created_at': 'old'},
            {'type': 'PLANNER_RESPONSE', 'step_index': 3, 'usage': {'input_tokens': 10000, 'cache_read_tokens': 20000}},
            {'type': 'CHECKPOINT', 'step_index': 2, 'created_at': 'replayed-later'},
            {'type': 'CHECKPOINT', 'step_index': 4},
            {'type': 'PLANNER_RESPONSE', 'step_index': 5, 'input_tokens': 12000, 'cache_read_tokens': 10000,
             'usage': {'input_tokens': 1, 'cache_read_tokens': 2}},
        ]
        self.log.write_text('\n'.join(map(json.dumps, steps)))
        result = worker_sessions.session_state('agent', self.home)
        self.assertEqual((result['telemetry_status'], result['compactions'], result['compacted_tokens'], result['context_tokens']),
                         ('observed', 2, 230000, 22000))
        self.assertNotIn('secret-canary', json.dumps(result))

        # Partial or missing evidence
        self.log.write_text(json.dumps({'type': 'CHECKPOINT', 'step_index': 1}) + '\n{partial')
        self.assertEqual(worker_sessions.session_state('agent', self.home)['telemetry_status'], 'partial')
        self.persistence({'conversationId': '../outside'})
        self.assertEqual(worker_sessions.session_state('agent', self.home)['telemetry_status'], 'unknown')

    def test_carried_context_triggers_handoff_rotation(self):
        self.persistence()
        self.log.write_text('\n'.join(map(json.dumps, [
            {'type': 'PLANNER_RESPONSE', 'step_index': 1, 'usage': {'input_tokens': 2100000}},
            {'type': 'CHECKPOINT', 'step_index': 2},
            {'type': 'PLANNER_RESPONSE', 'step_index': 3, 'usage': {'input_tokens': 22000}},
        ])))
        accounts = account_fleet.load_accounts(account_fleet.reference_fleet_path())
        now = datetime(2026, 10, 16, 9, tzinfo=timezone.utc)
        original_session_log = worker_sessions.session_log
        with patch.object(worker_sessions, 'session_log', side_effect=lambda agent_id, home=None: original_session_log(agent_id, self.home)):
            plan = allocator.plan({'work_kind': 'code_implementation', 'difficulty': 'hard',
                'needs': {'local_only': True}, 'chain': {'coupling': 'high', 'handoff_ready': True,
                    'worker': {'bundle': 'argon_high', 'agent_id': 'agent'}}},
                {'accounts': {}}, now, accounts=accounts, overrides=[])
        argon = next(c for c in plan['ranking'] if c['id'] == 'argon_high')
        self.assertEqual(argon['action'], 'rotate')
        self.assertTrue(any('2.1M' in note for note in argon['notes']))


class RouterIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        scorer_env = patch.dict(os.environ, {'ROUTER_DISABLE_LIVE_SCORER': '1'})
        scorer_env.start()
        self.addCleanup(scorer_env.stop)

    def cache(self, directory):
        def window(window_id, used, hours):
            return {'id': window_id, 'used_pct': used, 'resets_at': (self.now + timedelta(hours=hours)).isoformat()}
        row = lambda *windows: {'fetched_at': self.now.isoformat(), 'available': True, 'stale': False, 'windows': list(windows)}
        path = Path(directory) / 'usage-cache.json'
        path.write_text(json.dumps({'providers': {
            'claude': row(window('five_hour', 10, 3), window('seven_day', 40, 100), window('weekly_fable', 0, 100)),
            'codex': row(window('usage', 40, 100)),
            'antigravity': row(window('gemini-5h', 10, 3), window('gemini-weekly', 20, 100))}}), encoding='utf-8')
        return path

    @staticmethod
    def dumps_task(document):
        """The CLI loads its own allocator module, so a test task carries the profile a classifier would give it."""
        task = document['task']
        return json.dumps({**document, 'task': {**task, 'profile': task.get('profile') or profile_for(task)}})

    def run_main(self, argv):
        with patch('sys.argv', ['route_judge.py'] + argv), patch('builtins.print') as shown:
            code = router.main()
        return code, json.loads(shown.call_args.args[0])

    def test_plan_dry_run_no_eligible_and_record_agreement(self):
        self.addCleanup(setattr, decision_ledger, 'LEDGER_WRITES', True)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(decision_ledger, 'ledger_path', return_value=Path(directory) / 'ledger.jsonl'), \
                patch.object(subscription_cli, 'cli_prefix', side_effect=AssertionError('Unexpected model call')):
            task = Path(directory) / 'task.json'
            cache = str(self.cache(directory))

            # 1. Chain ledgering and auto-resume
            task.write_text(self.dumps_task({'task': {
                'objective': 'continue refactor', 'work_kind': 'code_implementation', 'difficulty': 'coupled',
                'expected_hours': 1, 'chain': {'id': 'c1', 'coupling': 'high', 'worker': {'bundle': 'sol_xhigh'}}}}), encoding='utf-8')
            code, res = self.run_main(['plan', '--input', str(task), '--usage-cache', cache])
            self.assertEqual((code, res['choice'], res['decision']['action']), (0, 'sol_xhigh', 'resume'))
            task.write_text(self.dumps_task({'task': {'work_kind': 'code_implementation', 'difficulty': 'coupled',
                                                 'expected_hours': 1, 'chain': {'id': 'c1', 'coupling': 'high'}}}), encoding='utf-8')
            _, again = self.run_main(['plan', '--input', str(task), '--usage-cache', cache])
            self.assertEqual((again['choice'], again['decision']['action']), ('sol_xhigh', 'resume'))

            # 2. Dry-run leaves no extra ledger row; non-eligible exits 2
            before_count = len(decision_ledger.read_ledger())
            task.write_text(self.dumps_task({'task': {'work_kind': 'frontend_ui', 'difficulty': 'routine',
                                                 'expected_hours': 0.5, 'needs': {'local_only': True}}}), encoding='utf-8')
            code, dry = self.run_main(['plan', '--input', str(task), '--usage-cache', cache, '--dry-run'])
            self.assertEqual((code, len(decision_ledger.read_ledger())), (0, before_count))
            planned = dry['choice']

            # 3. Record agreement check
            task.write_text(json.dumps({'task': {'work_kind': 'frontend_ui', 'difficulty': 'routine',
                                                 'expected_hours': 0.5, 'needs': {'local_only': True}}}), encoding='utf-8')
            def record(basis, provider, account, model, effort, *extra):
                return self.run_main(['record', '--input', str(task), '--usage-cache', cache,
                                      '--constraint', 'user_override', '--basis', basis, '--provider', provider,
                                      '--account', account, '--model', model, '--effort', effort, *extra])[1]
            override = record('User: use Sol max', 'codex', 'codex', 'gpt-6.1-sol', 'max', '--reason-category', 'quality_insufficient')
            agreed = record('User: Flash', 'antigravity-resilient', 'antigravity', 'gemini-3.8-flash-high', 'high')
            # record skips the classifier. Without a caller profile, agrees is a marker, not a comparison.
            self.assertEqual((override['agrees'], override['classifier_skipped'], override['router_choice']),
                             (None, True, None))
            self.assertEqual((agreed['agrees'], agreed['classifier_skipped'], agreed['router_choice']),
                             (None, True, None))
            skipped_row = decision_ledger.read_ledger()[-1]
            self.assertTrue(skipped_row['classifier_skipped'])
            self.assertIsNone(skipped_row['agrees'])
            task.write_text(json.dumps({'task': {
                'work_kind': 'frontend_ui', 'expected_hours': 0.5, 'needs': {'local_only': True},
                'profile': profile(work_kind='frontend_ui', primary_level=1, local_only=True,
                                   reason='caller profile')}}), encoding='utf-8')
            profiled_flash = record('User: Flash', 'antigravity-resilient', 'antigravity', 'gemini-3.8-flash-high', 'high')
            profiled_sol = record('User: Sol max', 'codex', 'codex', 'gpt-6.1-sol', 'max')
            self.assertEqual((profiled_flash['classifier_skipped'], profiled_flash['agrees'], profiled_flash['router_choice']),
                             (False, planned == 'flash_high', planned))
            self.assertEqual((profiled_sol['classifier_skipped'], profiled_sol['agrees'], profiled_sol['router_choice']),
                             (False, planned == 'sol_max', planned))
            self.assertIn(False, (profiled_flash['agrees'], profiled_sol['agrees']))

            task.write_text(self.dumps_task({'task': {'work_kind': 'visual_analysis', 'expected_hours': 0.5, 'needs': {'visual': True},
                                                 'constraints': {'model': 'grok-4.7'}}}), encoding='utf-8')
            code, blocked = self.run_main(['plan', '--input', str(task), '--usage-cache', cache])
            self.assertEqual((code, blocked['choice']), (2, 'none'))


class ToolOperationAndTaxonomyR2Tests(unittest.TestCase):
    """Contract tests for Taxonomy R2 `tool_operation` levels, clean deletion of the legacy responsiveness /
    realtime_loop path, continuity switching threshold, family vs alias resolution, and ledger read-side mapping."""

    def setUp(self):
        self.now = datetime(2026, 10, 9, 3, 0, tzinfo=timezone.utc)

    def healthy_usage(self):
        windows = {
            'claude': [('five_hour', 90, 3), ('seven_day', 80, 100), ('weekly_fable', 100, 100)],
            'claude-lab': [('five_hour', 90, 3), ('seven_day', 80, 100), ('weekly_fable', 100, 100)],
            'codex': [('usage', 80, 100)],
            'grok': [('credits', 80, 100)],
            'antigravity': [('gemini-5h', 90, 3), ('gemini-weekly', 80, 100), ('3p-5h', 90, 3), ('3p-weekly', 80, 100)],
        }
        return usage_windows(self.now, windows)

    def test_legacy_realtime_and_responsiveness_defaults_are_deleted(self):
        for banned_attr in (
            'RESPONSIVENESS_DEFAULTS',
            'bundle_responsiveness',
            'resolve_realtime_effort',
            'clearing_levels',
            'can_step_down',
            'five_axis_effort_ratio',
            'REALTIME_LOOP_QUESTION',
            'infer_realtime_loop',
        ):
            self.assertFalse(hasattr(allocator, banned_attr), f'{banned_attr} must be deleted')
        self.assertNotIn('realtime_loop', task_intake.REQUIRED_BOOL_FLAGS)
        self.assertNotIn('realtime_loop', task_intake.TASK_PROFILE_SCHEMA['properties'])
        self.assertNotIn('realtime_loop', task_intake.TASK_PROFILE_SCHEMA['required'])
        self.assertFalse((Path(__file__).resolve().parent / 'refresh_speed_data.py').exists())

    def test_all_calibrated_classifier_cases_validate_and_plan(self):
        from collections import Counter
        base_accounts = copy.deepcopy(account_fleet.load_accounts())
        capabilities = account_fleet.load_capabilities(accounts=base_accounts)
        usage = self.healthy_usage()
        cases_path = Path(__file__).resolve().parents[1] / 'references' / 'classifier-cases.json'
        doc = json.loads(cases_path.read_text(encoding='utf-8-sig'))
        rubric = doc.get('dimension_rubric') or {}
        for key in ('anti_inflation_rule', 'creativity_vs_architecture', 'composite_secondary_kind', 'access_and_local_only'):
            self.assertIn(key, rubric)
        cases = doc['calibration_cases']
        ids = [c['id'] for c in cases]
        self.assertEqual(len(ids), len(set(ids)), 'duplicate calibration case ids')
        required_keys = set(task_intake.TASK_PROFILE_SCHEMA['required'])
        validated_profiles = []
        for case in cases:
            self.assertEqual(set(case['expected_profile']), required_keys, case['id'])
            prof = task_intake.validate_profile(dict(case['expected_profile']))
            validated_profiles.append(prof)
            self.assertEqual(set(prof['axis_demand']), set(axis_scores.FIVE_AXES), case['id'])
            self.assertNotIn('realtime_loop', prof, case['id'])
            task_obj = case.get('task') if isinstance(case.get('task'), dict) else case
            res = allocator.plan(
                {
                    'objective': task_obj.get('objective', ''),
                    'context': task_obj.get('context', ''),
                    'expected_hours': prof['expected_hours'],
                    'profile': prof,
                },
                usage, now=self.now, accounts=base_accounts, capabilities=capabilities, overrides=[],
            )
            self.assertIn('choice', res, case['id'])
        # MECE (不重不漏) invariants: every defined route and every demand/profile level covered without per-route pileup.
        route_counts = Counter(p['work_kind'] for p in validated_profiles)
        self.assertEqual(set(route_counts), set(doc['route_definitions']))
        self.assertLessEqual(max(route_counts.values()), 5, route_counts)
        for cat in axis_scores.FIVE_AXES:
            self.assertEqual({p['axis_demand'][cat] for p in validated_profiles}, {0, 1, 2, 3}, cat)
        self.assertEqual({p['primary_level'] for p in validated_profiles}, {1, 2, 3})
        self.assertEqual({p['secondary_level'] for p in validated_profiles}, {0, 1, 2, 3})
        self.assertEqual({p['access'] for p in validated_profiles}, set(task_intake.ACCESS_LEVELS))
        for flag in (*task_intake.REQUIRED_BOOL_FLAGS, 'needs_gui', 'read_only_review', 'multi_task'):
            self.assertEqual({p[flag] for p in validated_profiles}, {False, True}, flag)

    def test_tool_operation_levels_0_and_1_ask_for_no_ability_target(self):
        base_accounts = axis_scores.five_axis_overlay(copy.deepcopy(account_fleet.load_accounts()))
        targets = base_accounts['_five_axis']['targets']
        self.assertEqual(axis_scores.five_axis_targets({'creativity': 0, 'coding': 2, 'aesthetics': 0, 'tool_operation': 0}, targets).get('tool_operation'), None)
        self.assertEqual(axis_scores.five_axis_targets({'creativity': 0, 'coding': 2, 'aesthetics': 0, 'tool_operation': 1}, targets).get('tool_operation'), None)

        # Stripping tool_operation facet scores completely does not change choice, fit, or net score at tool_operation <= 1
        stripped = copy.deepcopy(base_accounts)
        for spec in stripped['models'].values():
            if isinstance(spec, dict) and isinstance(spec.get('five_axis'), dict):
                cats = spec['five_axis'].get('categories') or {}
                if 'tool_operation' in cats:
                    cats['tool_operation'] = {'value': None, 'by_effort': {}, 'by_level': {'2': {}, '3': {}},
                                              'status': 'missing', 'facets': {}, 'missing_facets': ['responsiveness', 'image_recognition']}
        capabilities = account_fleet.load_capabilities(accounts=base_accounts)
        usage = self.healthy_usage()
        for tool_level in (0, 1):
            task = {
                'objective': 'Implement the retry backoff module and unit tests',
                'expected_hours': 0.5,
                'profile': profile(work_kind='code_implementation', local_only=True, expected_hours=0.5,
                                   axis_demand={'creativity': 1, 'coding': 2, 'aesthetics': 0, 'tool_operation': tool_level}),
            }
            before = allocator.plan(task, usage, now=self.now, accounts=stripped, capabilities=capabilities, overrides=[])
            after = allocator.plan(task, usage, now=self.now, accounts=base_accounts, capabilities=capabilities, overrides=[])
            self.assertEqual(before['choice'], after['choice'])
            before_scores = {c['id']: (c['score'], c['excluded'], c['components'].get('fit')) for c in before['ranking']}
            after_scores = {c['id']: (c['score'], c['excluded'], c['components'].get('fit')) for c in after['ranking']}
            self.assertEqual(before_scores, after_scores)

        # Outbound models whose tool_operation is missing (dot_text, grok_bot) are NOT excluded by tool_operation at level 0 or 1
        for tool_level in (0, 1):
            outbound_task = {
                'objective': 'Gather release dates from three public SDK changelogs',
                'expected_hours': 0.3,
                'needs': {'local_only': False, 'outbound_ok': True},
                'profile': profile(work_kind='web_research', primary_level=1,
                                   local_only=False, expected_hours=0.3,
                                   axis_demand={'creativity': 0, 'coding': 0, 'aesthetics': 0, 'tool_operation': tool_level}),
            }
            planned_out = allocator.plan(outbound_task, usage, now=self.now, accounts=base_accounts,
                                         capabilities=capabilities, overrides=[])
            by_id = {c['id']: c for c in planned_out['ranking']}
            self.assertEqual(by_id['dot_text']['excluded'], [])
            self.assertIsNotNone(by_id['dot_text']['score'])
            if 'grok_bot' in by_id:
                self.assertFalse(any('tool_operation' in r for r in by_id['grok_bot']['excluded']), by_id['grok_bot']['excluded'])

    def test_tool_operation_level_2_scores_per_effort_responsiveness_and_excludes_unmeasured(self):
        base_accounts = copy.deepcopy(account_fleet.load_accounts())
        capabilities = account_fleet.load_capabilities(accounts=base_accounts)
        usage = self.healthy_usage()

        live_cli_task = {
            'objective': '持续轮询命令行行情接口，价差触及阈值时立刻调用下单命令',
            'expected_hours': 0.5,
            'needs': {'local_only': False, 'outbound_ok': True},
            'profile': profile(
                work_kind='live_loop', primary_level=1, expected_hours=0.5,
                needs_visual=False, needs_gui=False, local_only=False,
                axis_demand={'creativity': 0, 'coding': 0, 'aesthetics': 0, 'tool_operation': 2},
            ),
        }
        res = allocator.plan(live_cli_task, usage, now=self.now, accounts=base_accounts, capabilities=capabilities, overrides=[])
        by_id = {c['id']: c for c in res['ranking']}

        # Lower/medium effort rows on multi-effort models have higher responsiveness (and level-2 fit) than high
        self.assertGreater(
            by_id['sol_medium']['five_axis_fit']['tool_operation']['facets']['responsiveness']['score'],
            by_id['sol_high']['five_axis_fit']['tool_operation']['facets']['responsiveness']['score'],
        )
        self.assertGreater(
            by_id['opus55_medium']['five_axis_fit']['tool_operation']['facets']['responsiveness']['score'],
            by_id['opus55_high']['five_axis_fit']['tool_operation']['facets']['responsiveness']['score'],
        )
        # Slow-but-measured models (e.g. argon_high) are scored continuously, never hard-floor excluded
        self.assertEqual(by_id['argon_high']['excluded'], [])
        self.assertIsNotNone(by_id['argon_high']['score'])
        # With shipped maintainer scores in references/user-scores.json, dot_text carries maintainer responsiveness;
        # when public manual scores are empty, channels with no responsiveness measurement leave automatic routing via Mode B.
        self.assertEqual(by_id['dot_text']['excluded'], [])
        with patch.object(axis_scores.user_scores, 'load_public', return_value={'schema_version': 2, 'scores': {}}):
            res_unscored = allocator.plan(live_cli_task, usage, now=self.now, accounts=copy.deepcopy(base_accounts),
                                          capabilities=capabilities, overrides=[])
        by_id_unscored = {c['id']: c for c in res_unscored['ranking']}
        for unmeasured_id in ('dot_text', 'grok_bot'):
            if unmeasured_id in by_id_unscored:
                self.assertTrue(
                    any('no trusted data for the demanded category tool_operation' in r
                        for r in by_id_unscored[unmeasured_id]['excluded']),
                    by_id_unscored[unmeasured_id]['excluded'],
                )

    def test_tool_operation_level_3_requires_image_input_and_scores_min_of_responsiveness_and_vision(self):
        base_accounts = copy.deepcopy(account_fleet.load_accounts())
        capabilities = account_fleet.load_capabilities(accounts=base_accounts)
        usage = self.healthy_usage()

        gui_task = {
            'objective': '持续在模拟器窗口里根据画面点击完成日常任务',
            'expected_hours': 0.5,
            'profile': profile(
                work_kind='gui_control', primary_level=2, expected_hours=0.5,
                needs_visual=True, needs_gui=True, local_only=False,
                axis_demand={'creativity': 0, 'coding': 1, 'aesthetics': 1, 'tool_operation': 3},
            ),
        }
        base_accounts['accounts']['codex']['bundles']['deepseek_test'] = {
            'model': 'deepseek-v4-pro', 'effort': 'high', 'provider': 'codex',
        }
        base_accounts['accounts']['codex']['bundles']['unknown_image_test'] = {
            'model': 'grok-4.7-build-fast', 'effort': 'high', 'provider': 'codex',
        }
        res = allocator.plan(gui_task, usage, now=self.now, accounts=base_accounts, capabilities=capabilities, overrides=[])
        by_id = {c['id']: c for c in res['ranking']}
        # Models with image_input == 'no' (deepseek-v4-pro) and 'unknown' (grok-4.7-build-fast) are excluded when
        # tool_operation >= 3; dot carries a dated maintainer image-input declaration (2026-11-12).
        self.assertTrue(any('does not accept image input' in r for r in by_id['deepseek_test']['excluded']),
                        by_id['deepseek_test']['excluded'])
        self.assertTrue(any('image_input modality has no cited source' in r for r in by_id['unknown_image_test']['excluded']),
                        by_id['unknown_image_test']['excluded'])
        self.assertFalse(any('image input' in r or 'image_input' in r for r in by_id['dot_text']['excluded']),
                         by_id['dot_text']['excluded'])
        # Named constraint overrides the exclusion with a surfaced note/decision alert
        pinned = allocator.plan(
            {**gui_task, 'constraints': {'model': 'claude-opus-5-5'}},
            usage, now=self.now, accounts=base_accounts, capabilities=capabilities, overrides=[],
        )
        self.assertTrue(pinned['selected_bundle']['model'].startswith('claude-opus-5-5'))

    def test_family_guesses_have_no_scores_while_date_aliases_resolve_to_calibrated_model(self):
        import auto_bundles
        accounts = copy.deepcopy(account_fleet.load_accounts())
        for variant_id in ('claude-opus-5-6', 'gpt-6-experimental', 'gemini-2.5-pro'):
            with self.subTest(variant=variant_id):
                prof = auto_bundles.default_profile(variant_id, accounts)
                self.assertEqual(prof['source'], 'family')
                self.assertEqual(account_fleet.model_spec(accounts, variant_id), {})
        for variant_id, exact_id in (('claude-opus-5-5-20260901', 'claude-opus-5-5'), ('gpt-6.1-sol-20260901', 'gpt-6.1-sol'),
                                     ('gpt-6-astra-preview', 'gpt-6-astra'), ('grok-4.7-latest', 'grok-4.7')):
            with self.subTest(variant=variant_id):
                prof = auto_bundles.default_profile(variant_id, accounts)
                self.assertEqual(prof['source'], 'calibrated')
                self.assertEqual(account_fleet.model_spec(accounts, variant_id), account_fleet.model_spec(accounts, exact_id))

    def test_high_coupling_continuity_keeps_close_worker_and_switches_past_one_demand_level(self):
        accounts = copy.deepcopy(account_fleet.load_accounts())
        accounts['accounts']['claude']['bundles']['fable52_high']['disabled'] = False
        capabilities = account_fleet.load_capabilities(accounts=accounts)
        usage = self.healthy_usage()
        # High-creativity task (creativity=3, continuity = t3 - t2 = 1.19 SD):
        # 1. `fable52_high` (creativity -0.93, within 1.19 SD of Argon +0.25 / Opus 0.0) stays when continuing (`coupling='high'`).
        close_chain = {'id': 'c-close', 'coupling': 'high', 'total_hours': 2,
                       'worker': {'bundle': 'fable52_high', 'context_tokens': 100000}}
        div_task = {
            'objective': 'Continue open-ended cross-domain concept invention',
            'expected_hours': 0.5,
            'profile': profile(work_kind='creative_ideation', primary_level=3,
                               access='read-only', local_only=True, expected_hours=0.5,
                               axis_demand={'creativity': 3, 'coding': 0, 'aesthetics': 0, 'tool_operation': 0}),
        }
        kept = allocator.plan({**div_task, 'chain': close_chain}, usage, now=self.now,
                              accounts=accounts, capabilities=capabilities, overrides=[])
        self.assertEqual((kept['choice'], kept['decision']['action']), ('fable52_high', 'resume'))
        self.assertEqual(kept['decision']['continuity'], 'high-coupling chain stays with its owner')

        medium_chain = {**close_chain, 'coupling': 'medium'}
        kept_med = allocator.plan({**div_task, 'chain': medium_chain}, usage, now=self.now,
                                  accounts=accounts, capabilities=capabilities, overrides=[])
        if kept_med['decision']['action'] == 'resume':
            self.assertEqual(kept_med['decision']['continuity'], 'medium-coupling chain stays with its owner')

        # 2. A worker whose deficit exceeds continuity + rival cost (e.g. `sol_medium` with creativity ~ -2.1 vs `argon_high` +0.25)
        # switches to the stronger candidate instead of forcing a bad continuation.
        far_chain = {'id': 'c-far', 'coupling': 'high', 'total_hours': 2,
                     'worker': {'bundle': 'sol_medium', 'context_tokens': 100000}}
        switched = allocator.plan({**div_task, 'chain': far_chain}, usage, now=self.now,
                                  accounts=accounts, capabilities=capabilities, overrides=[])
        self.assertNotEqual(switched['choice'], 'sol_medium')
        self.assertGreater(
            next(c for c in switched['ranking'] if c['id'] == switched['choice'])['score'],
            next(c for c in switched['ranking'] if c['id'] == 'sol_medium')['score'],
        )

    def test_normalize_task_local_only_matches_origin_main_without_gui_expansion(self):
        caps = account_fleet.load_capabilities()
        accounts = account_fleet.load_accounts()
        policy = accounts['scoring']
        for task in (
            {'objective': 'Click the Save button in the dialog'},
            {'work_kind': 'gui_control', 'objective': 'Operate the application'},
            {'work_kind': 'other_or_mixed', 'objective': 'Take a screenshot of the result'},
            {'objective': 'Design a browser dashboard UI layout in React'},
        ):
            with self.subTest(task=task):
                norm = task_intake.normalize_task(task, policy, caps, accounts)
                self.assertFalse(norm['local_only'])


class ExplainPoolWordingTests(unittest.TestCase):
    """A pooled window's percentage is of one account, so a pool of N holds N x 100 when full."""

    def top(self, remaining, full):
        return {'id': 'claude-2__opus55_medium', 'action': 'start', 'score': 0.15, 'expected_pct': 10.3, 'chain_pct': 10.3,
                'bundle': {}, 'components': {'opportunity_cost': 0.01},
                'capacity': {'window': 'seven_day', 'remaining': remaining, 'full': full, 'priced': False,
                             'end_hours': 50.0, 'expires_first': False}}

    def test_pool_names_its_size(self):
        text = allocator.explain(self.top(152.0, 200.0), {})
        self.assertIn('seven_day: 152% of one account left in a pool of 2 (full = 200%)', text)

    def test_single_account_stays_plain(self):
        text = allocator.explain(self.top(27.0, 100.0), {})
        self.assertIn('seven_day: 27% left,', text)
        self.assertNotIn('pool', text)

    def test_refusal_schedule_pool_member_alignment_and_named_pin_block_fast_wait_fallback(self):
        """Rules (allocator.refusal_schedule F5, allocator.named_pin_block F7): (1) when a pool's soonest member is
        stale, `refusal_schedule` returns both `start_after_hours` and `retry_at` from the non-stale member `after_reset`
        actually chose; (2) `named_pin_block` with `pinned_bundle` + `force_fast` falls back to the plain row when no
        fast twin exists, reporting its exclusion instead of 'no configured bundle offers it'."""
        now = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
        accounts = copy.deepcopy(account_fleet.load_accounts())
        accounts['accounts']['codex']['windows'] = {'usage': {'hours': 168, 'launch_min_remaining_pct': 5}}
        stale_reset = (now + timedelta(hours=2)).isoformat()
        fresh_reset = (now + timedelta(hours=6)).isoformat()
        m1 = {**usage_reading.window_state({'id': 'usage', 'used_pct': 100.0, 'resets_at': stale_reset}, now),
              'position': 1, 'remaining_pct_before_reset': None}
        m2 = {**usage_reading.window_state({'id': 'usage', 'used_pct': 100.0, 'resets_at': fresh_reset}, now),
              'position': 2}
        pooled = usage_reading.pooled_reading('usage', [m1, m2])
        usage = {'read_ok': True, 'accounts': {'codex': {
            'status': 'fresh', 'fetched_at': now.isoformat(), 'windows': [pooled],
        }}}
        bundle = {'id': 'astra_high', 'account': 'codex', 'provider': 'codex', 'model': 'gpt-6-astra', 'draws': ['usage']}
        hours, retry_at, wid = allocator.refusal_schedule(bundle, accounts, usage, now, blocked_wids={'usage'})
        self.assertEqual((hours, retry_at, wid), (6.0, fresh_reset, 'usage'))

        import candidate_eval
        plain = {'id': 'astra_high', 'base_id': 'astra_high', 'bundle': {'fast': False},
                 'excluded': [candidate_eval.FAST_WAIT_REFUSAL]}
        lines = allocator.named_pin_block({'pinned_bundle': 'astra_high', 'force_fast': True}, [plain])
        self.assertEqual(lines, [f"astra_high via astra_high: {candidate_eval.FAST_WAIT_REFUSAL}"])


if __name__ == '__main__':
    unittest.main()
