"""Orca and T3 Code as parent hosts: session detection, addressing and the native hand-off text."""
import argparse
import os
import unittest
from unittest.mock import patch

import roll
import worker_context as wc


def parent(env, **flags):
    args = argparse.Namespace(parent_id=None, parent_host=None, parent_address=None)
    vars(args).update(flags)
    with patch.dict(os.environ, env, clear=True):
        return wc.resolve_parent(args)


class OrcaParentTests(unittest.TestCase):
    def test_plain_orca_agent_is_addressed_by_its_session_prefix(self):
        env = {'ORCA_AGENT_SESSION_ID': 'sess-1', 'ORCA_TERMINAL_HANDLE': 'term_abc', 'ORCA_PANE_KEY': 'p'}
        self.assertEqual(parent(env), ('sess-1', 'orca', 'orca_session_id:sess-1'))

    def test_dispatched_orca_worker_keeps_its_minted_handle(self):
        env = {'ORCA_AGENT_SESSION_ID': 'sess-2', 'ORCA_TERMINAL_HANDLE': 'structworker_9'}
        self.assertEqual(parent(env), ('sess-2', 'orca', 'structworker_9'))

    def test_terminal_handle_alone_is_enough(self):
        self.assertEqual(parent({'ORCA_TERMINAL_HANDLE': 'term_abc'}), ('term_abc', 'orca', 'term_abc'))

    def test_pane_key_without_an_address_is_not_orca(self):
        with self.assertRaisesRegex(ValueError, 'addressable parent'):
            parent({'ORCA_PANE_KEY': 'p'})

    def test_orca_beats_the_provider_session_inside_it(self):
        env = {'ORCA_AGENT_SESSION_ID': 'sess-1', 'CLAUDE_CODE_SESSION_ID': 'claude-9'}
        self.assertEqual(parent(env)[:2], ('sess-1', 'orca'))

    def test_the_claude_session_inside_orca_is_recorded_beside_the_orca_parent(self):
        # Issue #2: Claude's hooks report its own session, so the route must carry it as well as the Orca parent.
        env = {'ORCA_AGENT_SESSION_ID': 'sess-1', 'CLAUDE_CODE_SESSION_ID': 'claude-9'}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(wc.harness_session('sess-1'), 'claude-9')
            self.assertEqual(wc.harness_session('claude-9'), '')
        with patch.dict(os.environ, {'ORCA_AGENT_SESSION_ID': 'sess-1'}, clear=True):
            self.assertEqual(wc.harness_session('sess-1'), '')

    def test_explicit_parent_in_orca_derives_the_address(self):
        self.assertEqual(parent({}, parent_host='orca', parent_id='sess-3'), ('sess-3', 'orca', 'orca_session_id:sess-3'))
        self.assertEqual(parent({'ORCA_TERMINAL_HANDLE': 'structworker_1'}, parent_host='orca', parent_id='sess-3'),
                         ('sess-3', 'orca', 'structworker_1'))
        self.assertEqual(parent({}, parent_host='orca', parent_id='sess-3', parent_address='term_x')[2], 'term_x')

    def test_paseo_still_wins_and_never_turns_into_orca(self):
        env = {'PASEO_AGENT_ID': 'agent-1', 'ORCA_AGENT_SESSION_ID': 'sess-1'}
        self.assertEqual(parent(env), ('agent-1', 'paseo', 'agent-1'))
        with self.assertRaisesRegex(ValueError, '--parent-host orca conflicts with PASEO_AGENT_ID agent-1'):
            parent(env, parent_host='orca')

    def test_orca_variables_do_not_leak_into_an_explicit_other_host(self):
        env = {'ORCA_AGENT_SESSION_ID': 'sess-1'}
        self.assertEqual(parent(env, parent_host='native', parent_id='c', parent_address='run:r'), ('c', 'native', 'run:r'))


class T3ParentTests(unittest.TestCase):
    def test_pi_provider_env_marks_t3_and_the_session_is_the_parent_thread(self):
        env = {'T3_MCP_URL': 'http://127.0.0.1:1/mcp', 'T3_MCP_BEARER_TOKEN': 't', 'PI_SESSION_ID': 'thread-1'}
        self.assertEqual(parent(env), ('thread-1', 't3', 'thread-1'))

    def test_claude_and_codex_threads_need_the_explicit_flag_or_env(self):
        env = {'CODEX_THREAD_ID': 'thread-2'}
        self.assertEqual(parent(env), ('thread-2', 'paseo', 'thread-2'))
        self.assertEqual(parent(env, parent_host='t3'), ('thread-2', 't3', 'thread-2'))
        self.assertEqual(parent(dict(env, ROUTER_PARENT_HOST='t3')), ('thread-2', 't3', 'thread-2'))

    def test_half_a_t3_credential_is_not_t3(self):
        self.assertEqual(parent({'T3_MCP_URL': 'u', 'CODEX_THREAD_ID': 'x'})[1], 'paseo')

    def test_unknown_host_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'paseo[|]native[|]orca[|]t3'):
            parent({'CODEX_THREAD_ID': 'x'}, parent_host='cursor')


class NativeHandOffTests(unittest.TestCase):
    bundle = {'provider': 'codex', 'model': 'gpt-6-astra', 'effort': 'high'}

    def test_orca_gets_a_worker_start_command_for_the_routed_bundle(self):
        text = roll.native_next('orca', self.bundle, 'orca_session_id:s')
        self.assertIn('orca orchestration worker-start', text)
        self.assertIn('--agent codex', text)
        self.assertIn('--model gpt-6-astra --effort high', text)
        self.assertIn('orca_session_id:s', text)

    def test_t3_gets_delegate_task_and_no_polling(self):
        text = roll.native_next('t3', self.bundle, 'thread-1')
        self.assertIn('delegate_task', text)
        self.assertIn('do not poll', text)

    def test_generic_native_text_is_unchanged(self):
        self.assertIn('--spec', roll.native_next('native', self.bundle, 'run:r'))
        self.assertEqual(roll.NATIVE_HOSTS, wc.NATIVE_HOSTS)

    def test_host_placement_expresses_workspace_directory_per_host_and_refuses_remote_surfaces(self):
        import tempfile
        import host_placement
        with tempfile.TemporaryDirectory() as tmp:
            canon = host_placement.validate_workspace(tmp)
            for host in ('orca', 't3', 'native'):
                placed = host_placement.place(host, canon, bundle=self.bundle, address='addr-1')
                self.assertTrue(placed['ok'])
                self.assertEqual(placed['cwd'], canon)
                self.assertEqual(placed['workspace'], canon)
                self.assertIn(canon, placed['next'])
            for remote in ('dot', 'chatgpt-web', 'grok-bot'):
                refused = host_placement.place(remote, canon)
                self.assertEqual((refused['ok'], refused['error']), (False, 'host_cannot_place_directory'))

    def test_orca_native_next_uses_wire_launch_model_not_bare_model(self):
        pi_bundle = {'provider': 'pi', 'model': 'gpt-6-astra', 'launch_model': 'openai-codex/gpt-6-astra', 'effort': 'high'}
        spec = {'host': 'orca', 'tool': 'shell:orca worker launch',
                'launch_fields': {'--agent': 'pi', '--model': 'openai-codex/gpt-6-astra', '--effort': 'high'}}
        text = roll.native_next('orca', pi_bundle, 'orca_session_id:s', native_spec=spec)
        self.assertIn('--agent pi', text)
        self.assertIn('--model openai-codex/gpt-6-astra', text)


class NativeExpressibilityContractTests(unittest.TestCase):
    """Contract tests for `status: native` expressibility (asserting rules, never pinning routing winners)."""

    def setUp(self):
        import importlib.util
        import tempfile
        import account_fleet
        import dispatch_registry as registry
        import router_test_support as rts

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.realpath(self._tmp.name)
        from pathlib import Path
        self.state = Path(self.root) / 'state'
        self.state.mkdir(parents=True, exist_ok=True)
        self.home = Path(self.root) / 'home'
        for sub in ('.claude', '.claude-lab', '.codex', '.codex-2', '.gemini'):
            (self.home / sub).mkdir(parents=True, exist_ok=True)

        gate_path = Path(__file__).resolve().parents[3] / 'setup' / 'router_gate_hook.py'
        spec = importlib.util.spec_from_file_location('router_gate_hook_contract', gate_path)
        self.gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gate)

        self.account_fleet = account_fleet
        self.registry = registry
        self.rts = rts
        base_doc = account_fleet.reference_fleet_document()
        self.accounts = account_fleet.load_accounts(document=base_doc)
        self.accounts['accounts']['claude']['plan'] = 'max20'
        for fable_bid in ('fable51_high', 'fable52_high'):
            self.accounts['accounts']['claude']['bundles'][fable_bid].pop('disabled', None)
        self.accounts['accounts']['claude-lab']['plan'] = 'max20'
        self.accounts['accounts']['codex']['plan'] = 'pro'
        self.accounts['accounts']['antigravity']['plan'] = 'ultra20'
        self.discovery = {
            'accounts': {
                'claude': {'family': 'claude', 'status': 'ready', 'home': str(self.home / '.claude')},
                'claude-lab': {'family': 'claude', 'status': 'ready', 'home': str(self.home / '.claude-lab')},
                'codex': {'family': 'codex', 'status': 'ready', 'home': str(self.home / '.codex')},
                'antigravity': {'family': 'antigravity', 'status': 'ready', 'home': str(self.home / '.gemini')},
                'pi': {'family': 'pi', 'status': 'ready'},
                'opencode': {'family': 'opencode', 'status': 'ready'},
                'dot': {'family': 'dot', 'status': 'ready'},
            },
            'launchers': {
                'claude': {'family': 'claude', 'status': 'ready'},
                'claude-lab': {'family': 'claude', 'status': 'ready'},
                'codex': {'family': 'codex', 'status': 'ready'},
                'antigravity-resilient': {'family': 'antigravity-resilient', 'status': 'ready'},
                'pi': {'family': 'pi', 'status': 'ready'},
                'opencode': {'family': 'opencode', 'status': 'ready'},
            },
        }
        self.accounts['_discovery'] = self.discovery
        (self.state / 'host-policy.json').write_text(
            '{"allowed_accounts": ["claude", "claude-lab", "codex", "antigravity", "pi", "opencode", "dot"]}\n',
            encoding='utf-8',
        )
        self.brief = Path(self.root) / 'brief.md'
        self.brief.write_text('# Implement contract check\n', encoding='utf-8')
        self.ledger = self.state / 'router-ledger.jsonl'
        self.db_path = self.state / 'dispatch.sqlite3'

    def _run_roll(self, argv, *, env, selected_bundle=None, cli_ok=True, paseo_ok=False, host_error='host_connection_failed'):
        import datetime as dt
        from pathlib import Path
        import dispatch_flow as flow
        import route_judge
        import subscription_cli
        import usage_reading

        clean_env = {
            'HOME': str(self.home),
            'USERPROFILE': str(self.home),
            'ROUTER_PROVIDER_HOME': str(self.home),
            'ROUTER_STATE_DIR': str(self.state),
            'ROUTER_DISPATCH_DB': str(self.db_path),
            'ROUTER_LEDGER': str(self.ledger),
            'PASEO_AGENT_ID': '',
            'HC_TERMINAL_ID': '',
            'ORCA_AGENT_SESSION_ID': '',
            'ORCA_TERMINAL_HANDLE': '',
            'T3_MCP_URL': '',
            'T3_MCP_BEARER_TOKEN': '',
            'CLAUDE_SESSION_ID': '',
            'CLAUDE_CODE_SESSION_ID': '',
            'CLAUDECODE': '',
            'CLAUDE_CONFIG_DIR': '',
            'CODEX_THREAD_ID': '',
            'CODEX_SESSION_ID': '',
            'CODEX_SANDBOX': '',
            'CODEX_CI': '',
            'CODEX_HOME': '',
            'ANTIGRAVITY_CONVERSATION_ID': '',
            'GEMINI_CLI_HOME': '',
            'PI_SESSION_ID': '',
            'OPENCODE_SESSION_ID': '',
            'ROUTER_PARENT_HOST': '',
            'ROUTER_PARENT_AGENT_ID': '',
            **env,
        }
        now = dt.datetime(2026, 11, 24, 0, 0, tzinfo=dt.timezone.utc)
        usage = {
            'read_ok': True,
            'accounts': {
                acct: {
                    'status': 'fresh', 'fetched_at': now.isoformat(),
                    'windows': [
                        {'id': wid, 'used_pct': 10, 'remaining_pct_before_reset': 90,
                         'observed_remaining_pct': 90, 'seconds_until_reset': 9000,
                         'resets_at': (now + dt.timedelta(hours=3)).isoformat()}
                        for wid in (spec.get('windows') or {})
                    ],
                }
                for acct, spec in self.accounts['accounts'].items() if spec.get('windows')
            },
        }

        class FakeProc:
            pid = 9001
            returncode = 0
            stderr = 'task_id=dot-1\n'

            def __init__(self, *_, stdout=None, **__):
                if stdout is not None:
                    stdout.write(b'ok')

        def fake_host(req):
            if paseo_ok and host_error == 'host_connection_failed' and req.get('operation') == 'create':
                prov, _, mod = req['request']['provider'].partition('/')
                return {
                    'ok': True, 'sent': True,
                    'agent': {'id': 'paseo-w1', 'provider': prov, 'model': mod,
                              'currentModeId': req['request']['settings'].get('modeId'),
                              'thinkingOptionId': req['request']['settings'].get('thinkingOptionId'),
                              'status': 'running', 'updatedAt': dt.datetime.now(dt.timezone.utc).isoformat(),
                              'labels': req['request']['labels']},
                    'completion_notification': {'status': 'watching'},
                }
            return {'ok': False, 'sent': False, 'error': host_error, 'condition': 'refused'}

        orig_build_task = roll.build_task
        orig_plan_stage = route_judge.plan_stage

        def patched_plan(document, p_args):
            if selected_bundle is not None:
                dec = {'stage': 'plan', 'choice': selected_bundle.get('bundle_id') or selected_bundle['model'],
                       'selected_bundle': selected_bundle, 'decision': {'action': 'start'}}
                return dict(dec, **self.registry.issue(dec, document['task'])), 0
            return orig_plan_stage(document, p_args)

        popen_side = FakeProc if cli_ok else OSError('cli binary missing')
        parsed = route_judge.build_parser().parse_args(['roll', '--brief', str(self.brief), *argv])
        parsed.brief = Path(parsed.brief[0])
        with patch.dict(os.environ, clean_env, clear=False), \
                patch.object(self.account_fleet, 'load_accounts', return_value=self.accounts), \
                patch.object(self.registry.provider_inventory, 'current', return_value=self.discovery), \
                patch.object(usage_reading, 'usage_snapshot', side_effect=lambda *_, **__: usage), \
                patch.object(subscription_cli, 'subscription_env', side_effect=lambda account=None: dict(os.environ)), \
                patch.object(subscription_cli, 'cli_prefix', side_effect=lambda name: [name]), \
                patch.object(roll, 'WORKER_PROMPTS_DIR', self.state / 'wp'), \
                patch.object(roll, 'RESULTS_DIR', self.state / 'rr'), \
                patch.object(roll, 'CLI_RUNS_DIR', self.state / 'cli'), \
                patch.object(flow, 'paseo_reachable', return_value=paseo_ok), \
                patch.object(flow, 'host_call', fake_host), \
                patch.object(roll.subprocess, 'Popen', side_effect=popen_side), \
                patch.object(roll.subprocess, 'run', return_value=FakeProc()), \
                patch.object(route_judge, 'plan_stage', side_effect=patched_plan), \
                patch.object(roll, 'build_task',
                             lambda *a, **k: dict(orig_build_task(*a, **k),
                                                  profile=self.rts.profile_for({'work_kind': 'code_implementation', 'difficulty': 'medium'}))):
            return roll.run(parsed, route_judge)

    def test_caller_native_host_identifies_all_eight_hosts_from_env_without_guessing(self):
        self.assertEqual(wc.caller_native_host('orca', {}), 'orca')
        self.assertEqual(wc.caller_native_host('t3', {}), 't3')
        self.assertIsNone(wc.caller_native_host('paseo', {'CLAUDE_CODE_SESSION_ID': 'c1'}))
        self.assertIsNone(wc.caller_native_host('herdr', {'HC_TERMINAL_ID': 'h1'}))
        self.assertIsNone(wc.caller_native_host('native', {}))
        self.assertEqual(wc.caller_native_host('native', {'CLAUDE_CODE_SESSION_ID': 'c1'}), 'claude')
        self.assertEqual(wc.caller_native_host('native', {'CLAUDECODE': '1'}), 'claude')
        self.assertEqual(wc.caller_native_host('native', {'CODEX_THREAD_ID': 'x1'}), 'codex')
        self.assertEqual(wc.caller_native_host('native', {'CODEX_SESSION_ID': 'x2'}), 'codex')
        self.assertEqual(wc.caller_native_host('native', {'ANTIGRAVITY_CONVERSATION_ID': 'a1'}), 'antigravity-resilient')
        self.assertEqual(wc.caller_native_host('native', {'PI_SESSION_ID': 'p1'}), 'pi')
        self.assertEqual(wc.caller_native_host('native', {'OPENCODE_SESSION_ID': 'o1'}), 'opencode')
        # Unsourced markers are excluded and never identify a native host:
        for unsourced in ({'CLAUDE_SESSION_ID': 'c-legacy'}, {'CODEX_CI': '1'}, {'CODEX_SANDBOX': 'seatbelt'}):
            self.assertIsNone(wc.caller_native_host('native', unsourced), unsourced)
        # Conflicting session markers refuse to guess:
        self.assertIsNone(wc.caller_native_host('native', {'CLAUDE_CODE_SESSION_ID': 'c1', 'CODEX_THREAD_ID': 'x1'}))

    def test_declared_native_tools_cite_vendor_sources_and_link_to_harness_capabilities(self):
        tools = self.account_fleet.NATIVE_HOST_TOOLS
        self.assertEqual(set(tools), {'claude', 'codex', 'antigravity-resilient', 'orca', 't3', 'pi', 'opencode', 'herdr'})
        for host, decl in tools.items():
            self.assertTrue(decl.get('as_of'), host)
            self.assertTrue(decl.get('sources') and all(s.startswith('https://') for s in decl['sources']), host)
        for harness in ('claude', 'codex', 'antigravity-resilient', 'pi', 'opencode'):
            self.assertIs(self.account_fleet.HARNESS_CAPABILITIES[harness]['native_tool'], tools[harness])
        for no_subagent in ('pi', 'opencode', 'herdr'):
            self.assertIsNone(tools[no_subagent]['tool'])

    def test_native_answer_carries_declared_launch_fields_and_passes_gate_follow_through(self):
        from pathlib import Path
        bundles = self.account_fleet.bundles(self.accounts)
        sonnet = dict(bundles['sonnet55_high'], bundle_id='sonnet55_high', account='claude')
        fable52 = {'provider': 'claude', 'account': 'claude', 'model': 'claude-fable-5-2', 'effort': 'high', 'bundle_id': 'fable52_high'}
        fable51 = {'provider': 'claude', 'account': 'claude', 'model': 'claude-fable-5-1', 'effort': 'high', 'bundle_id': 'fable51_high'}
        sol = dict(bundles['sol_high'], bundle_id='sol_high', account='codex')
        flash = dict(bundles['flash_high'], bundle_id='flash_high', account='antigravity')
        pi_astra = dict(bundles['pi_astra_high'], bundle_id='pi_astra_high', account='pi')

        cases = (
            ('claude-same-family-same-row', fable51,
             ['--parent-host', 'native', '--parent-id', 'c-1', '--parent-address', 'inbox:c1',
              '--parent-model', 'claude-fable-5-1', '--new-worker', '--task-id', 'tc-1'],
             {'CLAUDE_CODE_SESSION_ID': 'c-1'},
             lambda lf: {'session_id': 'c-1', 'tool_use_id': 'u-c1', 'tool_name': 'Agent',
                         'tool_input': {'subagent_type': 'general-purpose', 'prompt': 'go', **lf}}),
            ('claude-other-family-target-row', fable52,
             ['--parent-host', 'native', '--parent-id', 'c-2', '--parent-address', 'inbox:c2',
              '--parent-model', 'claude-sonnet-5-5', '--new-worker', '--task-id', 'tc-2'],
             {'CLAUDE_CODE_SESSION_ID': 'c-2'},
             lambda lf: {'session_id': 'c-2', 'tool_use_id': 'u-c2', 'tool_name': 'Agent',
                         'tool_input': {'subagent_type': 'general-purpose', 'prompt': 'go', **lf}}),
            ('codex', sol,
             ['--parent-host', 'native', '--parent-id', 'x-1', '--parent-address', 'inbox:x1', '--new-worker', '--task-id', 'tx-1'],
             {'CODEX_THREAD_ID': 'x-1'},
             lambda lf: {'session_id': 'x-1', 'tool_use_id': 'u-x1', 'tool_name': 'spawn_agent',
                         'tool_input': {'message': 'go', **lf}}),
            ('antigravity', flash,
             ['--parent-host', 'native', '--parent-id', 'a-1', '--parent-address', 'inbox:a1',
              '--parent-model', flash['model'], '--new-worker', '--task-id', 'ta-1'],
             {'ANTIGRAVITY_CONVERSATION_ID': 'a-1'},
             lambda lf: {'conversationId': 'a-1', 'modelName': flash['model'],
                         'toolCall': {'id': 'u-a1', 'name': 'invoke_subagent', 'args': {'TypeName': 'self', **lf}}}),
            ('orca-pi', pi_astra,
             ['--new-worker', '--task-id', 'to-1'],
             {'ORCA_AGENT_SESSION_ID': 'o-1'},
             lambda lf: {'session_id': 'o-1', 'tool_use_id': 'u-o1', 'tool_name': 'Bash',
                         'tool_input': {'command': f'orca orchestration worker-start --spec "go" --agent {lf["--agent"]} --model {lf["--model"]} --effort {lf["--effort"]} --json'}}),
            ('t3-codex', sol,
             ['--parent-host', 't3', '--new-worker', '--task-id', 'tt-1'],
             {'CODEX_THREAD_ID': 't3-1'},
             lambda lf: {'session_id': 't3-1', 'tool_use_id': 'u-t1', 'tool_name': 'delegate_task',
                         'tool_input': {'task': 'go', **lf}}),
        )
        for label, bundle, flags, env, build_payload in cases:
            with self.subTest(case=label):
                compact_res, code = self._run_roll(flags, env=env, selected_bundle=bundle)
                self.assertEqual((code, compact_res['status']), (0, 'native'), compact_res)
                self.assertTrue(compact_res.get('native_tool'))
                self.assertIsInstance(compact_res.get('launch_fields'), dict)
                self.assertTrue(compact_res['launch_fields'])
                full = __import__('json').loads(Path(compact_res['details']).read_text(encoding='utf-8'))
                self.assertEqual(compact_res['next_action'], full['next'])
                self.assertEqual(compact_res['launch_fields'], full['launch_fields'])
                payload = build_payload(compact_res['launch_fields'])
                with patch.dict(os.environ, {'ROUTER_DISPATCH_DB': str(self.db_path), 'ROUTER_STATE_DIR': str(self.state),
                                             'ROUTER_PROVIDER_HOME': str(self.home), **env}), \
                        patch.object(self.gate, 'load_dispatch_accounts', return_value=self.accounts):
                    allowed = self.gate.evaluate_payload(payload, ledger_path=self.ledger)
                    expected_allow = {'decision': 'allow'} if 'toolCall' in payload else {}
                    self.assertEqual(allowed, expected_allow, (label, allowed))
                    # Second launch with a different call ID is refused (hand-over consumed):
                    second = (dict(payload, toolCall=dict(payload['toolCall'], id='u-2'))
                              if 'toolCall' in payload else dict(payload, tool_use_id='u-2'))
                    denied = self.gate.evaluate_payload(second, ledger_path=self.ledger)
                    self.assertIn(denied.get('decision'), ('block', 'deny'), (label, denied))

    def test_inexpressible_route_never_answers_native_and_launches_or_blocks(self):
        bundles = self.account_fleet.bundles(self.accounts)
        sonnet = dict(bundles['sonnet55_high'], bundle_id='sonnet55_high', account='claude')
        sonnet_lab = dict(bundles['lab_sonnet55_high'], bundle_id='lab_sonnet55_high', account='claude-lab')
        fable_target = {'provider': 'claude', 'account': 'claude', 'model': 'claude-fable-5-2', 'effort': 'high', 'bundle_id': 'fable52_high'}
        fable_older = {'provider': 'claude', 'account': 'claude', 'model': 'claude-fable-5-1', 'effort': 'high', 'bundle_id': 'fable51_high'}
        sol = dict(bundles['sol_high'], bundle_id='sol_high', account='codex')
        sol_fast = dict(sol, fast=True)
        pi_astra = dict(bundles['pi_astra_high'], bundle_id='pi_astra_high', account='pi')
        argon_high = dict(bundles['argon_high'], bundle_id='argon_high', account='antigravity')
        custom_unlaunchable = {'provider': 'acp', 'account': 'claude', 'model': 'claude-fable-5-1', 'effort': 'high'}

        inexpressible_cases = (
            # B1 rule cases for Claude line alias:
            ('claude-unknown-parent-model', sonnet,
             ['--parent-host', 'native', '--parent-id', 'c0', '--parent-address', 'in:c0'],
             {'CLAUDE_CODE_SESSION_ID': 'c0'}),
            ('claude-same-family-older-parent-newer-route', fable_target,
             ['--parent-host', 'native', '--parent-id', 'c-sf1', '--parent-address', 'in:csf1', '--parent-model', 'claude-fable-5-1'],
             {'CLAUDE_CODE_SESSION_ID': 'c-sf1'}),
            ('claude-same-family-newer-parent-older-route', fable_older,
             ['--parent-host', 'native', '--parent-id', 'c-sf2', '--parent-address', 'in:csf2', '--parent-model', 'claude-fable-5-2'],
             {'CLAUDE_CODE_SESSION_ID': 'c-sf2'}),
            ('claude-other-family-non-target-route', fable_older,
             ['--parent-host', 'native', '--parent-id', 'c-of', '--parent-address', 'in:cof', '--parent-model', 'claude-sonnet-5-5'],
             {'CLAUDE_CODE_SESSION_ID': 'c-of'}),
            ('claude-unknown-parent-family', fable_target,
             ['--parent-host', 'native', '--parent-id', 'c-uf', '--parent-address', 'in:cuf', '--parent-model', 'custom-bedrock-model-v1'],
             {'CLAUDE_CODE_SESSION_ID': 'c-uf'}),
            ('claude-env-default-line-model', fable_target,
             ['--parent-host', 'native', '--parent-id', 'c-env1', '--parent-address', 'in:ce1', '--parent-model', 'claude-sonnet-5-5'],
             {'CLAUDE_CODE_SESSION_ID': 'c-env1', 'ANTHROPIC_DEFAULT_FABLE_MODEL': 'claude-fable-5-1'}),
            ('claude-env-subagent-model', sonnet,
             ['--parent-host', 'native', '--parent-id', 'c-env2', '--parent-address', 'in:ce2', '--parent-model', 'claude-sonnet-5-5'],
             {'CLAUDE_CODE_SESSION_ID': 'c-env2', 'CLAUDE_CODE_SUBAGENT_MODEL': 'haiku'}),
            ('claude-env-effort-override', sonnet,
             ['--parent-host', 'native', '--parent-id', 'c-env3', '--parent-address', 'in:ce3', '--parent-model', 'claude-sonnet-5-5'],
             {'CLAUDE_CODE_SESSION_ID': 'c-env3', 'CLAUDE_CODE_EFFORT_LEVEL': 'low'}),
            ('claude-cross-provider', sol,
             ['--parent-host', 'native', '--parent-id', 'c1', '--parent-address', 'in:c1', '--parent-model', 'claude-sonnet-5-5'],
             {'CLAUDE_CODE_SESSION_ID': 'c1'}),
            ('claude-other-account', sonnet_lab,
             ['--parent-host', 'native', '--parent-id', 'c2', '--parent-address', 'in:c2', '--parent-model', 'claude-sonnet-5-5'],
             {'CLAUDE_CODE_SESSION_ID': 'c2'}),
            # Codex, Antigravity, Orca, T3, bare Pi, OpenCode, unknown:
            ('codex-fast-bundle', sol_fast,
             ['--parent-host', 'native', '--parent-id', 'x-fast', '--parent-address', 'in:xf'],
             {'CODEX_THREAD_ID': 'x-fast'}),
            ('codex-pi-harness', pi_astra, ['--parent-host', 'native', '--parent-id', 'x1', '--parent-address', 'in:x1'], {'CODEX_THREAD_ID': 'x1'}),
            ('agy-no-parent-model', argon_high, ['--parent-host', 'native', '--parent-id', 'a1', '--parent-address', 'in:a1'], {'ANTIGRAVITY_CONVERSATION_ID': 'a1'}),
            ('agy-mismatched-parent-model', argon_high,
             ['--parent-host', 'native', '--parent-id', 'a2', '--parent-address', 'in:a2', '--parent-model', 'gemini-3.8-flash-high'],
             {'ANTIGRAVITY_CONVERSATION_ID': 'a2'}),
            # Multi-account family (`claude` + `claude-lab` both ready) or unsupported effort on Orca / T3:
            ('orca-ambiguous-claude-accounts', sonnet, [], {'ORCA_AGENT_SESSION_ID': 'o0'}),
            ('orca-other-claude-account', sonnet_lab, [], {'ORCA_AGENT_SESSION_ID': 'o1'}),
            ('orca-opencode-with-effort', {'provider': 'opencode', 'account': 'opencode', 'model': 'gpt-6.1-sol', 'effort': 'high', 'bundle_id': 'oc_sol_high'},
             [], {'ORCA_AGENT_SESSION_ID': 'o-oc'}),
            ('t3-ambiguous-claude-accounts', sonnet, ['--parent-host', 't3'], {'CODEX_THREAD_ID': 't0'}),
            ('t3-other-claude-account', sonnet_lab, ['--parent-host', 't3'], {'CODEX_THREAD_ID': 't1'}),
            ('t3-opencode-with-effort', {'provider': 'opencode', 'account': 'opencode', 'model': 'gpt-6.1-sol', 'effort': 'high', 'bundle_id': 'oc_sol_high'},
             ['--parent-host', 't3'], {'CODEX_THREAD_ID': 't-oc'}),
            ('bare-pi', sol, ['--parent-host', 'native', '--parent-id', 'p1', '--parent-address', 'in:p1'], {'PI_SESSION_ID': 'p1'}),
            ('opencode', sol, ['--parent-host', 'native', '--parent-id', 'oc1', '--parent-address', 'in:oc1'], {'OPENCODE_SESSION_ID': 'oc1'}),
            ('unknown-native-env', sol, ['--parent-host', 'native', '--parent-id', 'u1', '--parent-address', 'in:u1'], {}),
        )
        for idx, (label, bundle, flags, env) in enumerate(inexpressible_cases):
            with self.subTest(mode='cli_available', case=label):
                res, code = self._run_roll([*flags, '--new-worker', '--task-id', f'inexp-ok-{idx}'],
                                           env=env, selected_bundle=bundle, cli_ok=True)
                self.assertEqual((code, res['status']), (0, 'launched'), (label, res))
                self.assertNotEqual(res['status'], 'native')
            with self.subTest(mode='no_launcher', case=label):
                res_b, code_b = self._run_roll([*flags, '--new-worker', '--task-id', f'inexp-blk-{idx}'],
                                               env=env, selected_bundle=bundle, cli_ok=False, paseo_ok=False)
                self.assertEqual((code_b, res_b['status']), (2, 'blocked'), (label, res_b))
                self.assertNotEqual(res_b['status'], 'native')
        # Codex `[agents.default]` `config_file` in `CODEX_HOME/config.toml` layers after requested overrides -> not expressible:
        from pathlib import Path
        codex_cfg = self.home / '.codex' / 'config.toml'
        codex_cfg.write_text('[agents.default]\nconfig_file = "custom-role.toml"\n', encoding='utf-8')
        try:
            res_cfg, code_cfg = self._run_roll(
                ['--parent-host', 'native', '--parent-id', 'x-cfg', '--parent-address', 'in:xcfg', '--new-worker', '--task-id', 'inexp-cfg'],
                env={'CODEX_THREAD_ID': 'x-cfg'}, selected_bundle=sol, cli_ok=True,
            )
            self.assertEqual((code_cfg, res_cfg['status']), (0, 'launched'), res_cfg)
        finally:
            codex_cfg.unlink(missing_ok=True)
        # Unlaunchable non-CLI family also blocks instead of returning unrecorded `native`:
        res_u, code_u = self._run_roll(['--parent-host', 'native', '--parent-id', 'c-u', '--parent-address', 'in:cu',
                                        '--parent-model', 'claude-sonnet-5-5', '--new-worker', '--task-id', 'inexp-acp'],
                                       env={'CLAUDE_CODE_SESSION_ID': 'c-u'}, selected_bundle=custom_unlaunchable, cli_ok=False)
        self.assertEqual((code_u, res_u['status']), (2, 'blocked'))
        self.assertEqual(self.registry.native_handovers(path=self.db_path), [])

    def test_claude_older_version_without_effort_is_refused_with_non_native_roll_way_out(self):
        """B3: `Agent` without `effort` is never accepted silently; both `next_action` and the gate denial name v2.1.292+
        and tell an older Claude Code session to re-run `roll` without `--parent-host native`."""
        sonnet = dict(self.account_fleet.bundles(self.accounts)['sonnet55_high'], bundle_id='sonnet55_high', account='claude')
        res_n, code_n = self._run_roll(
            ['--parent-host', 'native', '--parent-id', 'c-old', '--parent-address', 'in:cold',
             '--parent-model', 'claude-sonnet-5-5', '--new-worker', '--task-id', 'b3-task'],
            env={'CLAUDE_CODE_SESSION_ID': 'c-old'}, selected_bundle=sonnet,
        )
        self.assertEqual((code_n, res_n['status']), (0, 'native'))
        self.assertIn('v2.1.292', res_n['next_action'])
        self.assertIn('without `--parent-host native`', res_n['next_action'])
        with patch.dict(os.environ, {'ROUTER_DISPATCH_DB': str(self.db_path), 'ROUTER_STATE_DIR': str(self.state),
                                     'ROUTER_PROVIDER_HOME': str(self.home), 'CLAUDE_CODE_SESSION_ID': 'c-old'}), \
                patch.object(self.gate, 'load_dispatch_accounts', return_value=self.accounts):
            denied = self.gate.evaluate_payload(
                {'session_id': 'c-old', 'tool_use_id': 'u-old', 'tool_name': 'Agent',
                 'tool_input': {'subagent_type': 'general-purpose', 'prompt': 'go', 'model': 'sonnet'}},
                ledger_path=self.ledger,
            )
        self.assertEqual(denied['decision'], 'block')
        self.assertIn('v2.1.292', denied['reason'])
        self.assertIn('--parent-host native', denied['reason'])
        # Hand-over stays unconsumed, and following the instruction (`roll` without `--parent-host native`) launches via CLI:
        self.assertEqual(len([h for h in self.registry.native_handovers(path=self.db_path) if not h.get('launched_at')]), 1)
        res_cli, code_cli = self._run_roll(
            ['--new-worker', '--task-id', 'b3-task-cli'],
            env={'CLAUDE_CODE_SESSION_ID': 'c-old'}, selected_bundle=sonnet, cli_ok=True,
        )
        self.assertEqual((code_cli, res_cli['status']), (0, 'launched'))

    def test_cli_launch_for_native_caller_writes_cli_footer_without_unreachable_inbox_on_both_paths(self):
        """B4: both CLI launch paths (no Paseo, and Paseo `parent_agent_not_found` fallback) share one owner and write
        `Parent host: cli` with `no worker-to-parent mailbox`, matching `next_action` ('nothing calls back')."""
        import json
        from pathlib import Path
        import dispatch_flow as flow
        sol = dict(self.account_fleet.bundles(self.accounts)['sol_high'], bundle_id='sol_high', account='codex')
        res_direct, code_direct = self._run_roll(
            ['--parent-host', 'native', '--parent-id', 'c-cli1', '--parent-address', 'in:unreachable-1',
             '--new-worker', '--task-id', 'b4-direct'],
            env={'CLAUDE_CODE_SESSION_ID': 'c-cli1'}, selected_bundle=sol, cli_ok=True, paseo_ok=False,
        )
        self.assertEqual((code_direct, res_direct['status']), (0, 'launched'))
        self.assertIn('nothing calls back', res_direct['next_action'])
        full_direct = json.loads(Path(res_direct['details']).read_text(encoding='utf-8'))
        prompt_direct = Path(full_direct['worker_prompt_file']).read_text(encoding='utf-8')
        self.assertIn('Parent host: `cli`. Parent message address: `none (no worker-to-parent mailbox)`.', prompt_direct)
        self.assertNotIn('in:unreachable-1', prompt_direct)

        res_fb, code_fb = self._run_roll(
            ['--parent-host', 'native', '--parent-id', 'c-cli2', '--parent-address', 'in:unreachable-2',
             '--new-worker', '--task-id', 'b4-fb'],
            env={'CLAUDE_CODE_SESSION_ID': 'c-cli2'}, selected_bundle=sol, cli_ok=True, paseo_ok=True,
            host_error='parent_agent_not_found',
        )
        self.assertEqual((code_fb, res_fb['status']), (0, 'launched'))
        self.assertIn('nothing calls back', res_fb['next_action'])
        full_fb = json.loads(Path(res_fb['details']).read_text(encoding='utf-8'))
        prompt_fb = Path(full_fb['worker_prompt_file']).read_text(encoding='utf-8')
        self.assertIn('Parent host: `cli`. Parent message address: `none (no worker-to-parent mailbox)`.', prompt_fb)
        self.assertNotIn('in:unreachable-2', prompt_fb)

    def test_named_inexpressible_route_is_obeyed_never_substituted_and_never_native(self):
        for idx, (flags, env, expected_model, expected_account) in enumerate((
            (['--parent-host', 'native', '--parent-id', 'nc-1', '--parent-address', 'in:nc1',
              '--provider', 'codex', '--account', 'codex', '--model', 'gpt-6.1-sol', '--effort', 'high', '--basis', 'user named sol'],
             {'CLAUDE_CODE_SESSION_ID': 'nc-1'}, 'gpt-6.1-sol', 'codex'),
            (['--parent-host', 'native', '--parent-id', 'na-1', '--parent-address', 'in:na1',
              '--parent-model', 'gemini-3.8-flash-high',
              '--provider', 'antigravity-resilient', '--account', 'antigravity', '--model', 'gemini-4-argon-high', '--basis', 'user named argon'],
             {'ANTIGRAVITY_CONVERSATION_ID': 'na-1'}, 'gemini-4-argon-high', 'antigravity'),
            (['--parent-host', 'native', '--parent-id', 'np-1', '--parent-address', 'in:np1',
              '--provider', 'claude', '--account', 'claude', '--model', 'claude-sonnet-5-5', '--effort', 'high', '--basis', 'user named sonnet'],
             {'PI_SESSION_ID': 'np-1'}, 'claude-sonnet-5-5', 'claude'),
        )):
            with self.subTest(idx=idx, launcher='cli'):
                res, code = self._run_roll([*flags, '--new-worker', '--task-id', f'named-ok-{idx}', '--hours', '0.5'],
                                           env=env, cli_ok=True)
                self.assertEqual((code, res['status']), (0, 'launched'), res)
                self.assertEqual((res['worker']['model'], res['worker']['account']), (expected_model, expected_account))
            with self.subTest(idx=idx, launcher='blocked'):
                res_b, code_b = self._run_roll([*flags, '--new-worker', '--task-id', f'named-blk-{idx}', '--hours', '0.5'],
                                               env=env, cli_ok=False, paseo_ok=False)
                self.assertEqual((code_b, res_b['status']), (2, 'blocked'), res_b)
                self.assertEqual((res_b['worker']['model'], res_b['worker']['account']), (expected_model, expected_account))


if __name__ == '__main__':
    unittest.main()
