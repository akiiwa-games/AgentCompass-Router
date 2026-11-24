import datetime as dt
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_gate_hook as gate

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 10, 14, 6, 0, tzinfo=UTC)

SKILL_ROOT = Path(__file__).resolve().parents[1] / "skills/agent-compass"


def setUpModule():
    """Run against this checkout with isolated state, even after another copy was imported."""
    global _state, _environment, _saved_modules, _saved_path, _gate_paths
    _state = tempfile.TemporaryDirectory()
    _saved_path = sys.path[:]
    names = {path.stem for path in (SKILL_ROOT / "scripts").glob("*.py")}
    _saved_modules = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    _environment = mock.patch.dict(os.environ, {
        "ROUTER_SKILL_ROOT": str(SKILL_ROOT), "ROUTER_STATE_DIR": _state.name,
        "HOME": _state.name, "USERPROFILE": _state.name,
    })
    _environment.start()
    for name in ("CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR", "CODEX_HOME",
                 "PASEO_AGENT_ID", *gate.router_module("worker_context").OWN_SESSION_ID_VARS):
        os.environ.pop(name, None)   # restored by _environment.stop(); discovery and session IDs stay isolated
    home = Path(_state.name)
    _gate_paths = mock.patch.multiple(gate, HOME=home,
        DEFAULT_LEDGER=home / '.agent-compass/router-ledger.jsonl',
        DEFAULT_PASEO_AGENTS=home / '.paseo/agents')
    _gate_paths.start()
    sys.path.insert(0, str(SKILL_ROOT / "scripts"))


def tearDownModule():
    for path in (SKILL_ROOT / "scripts").glob("*.py"):
        sys.modules.pop(path.stem, None)
    sys.modules.update(_saved_modules)
    sys.path[:] = _saved_path
    _gate_paths.stop()
    _environment.stop()
    _state.cleanup()


_UNSET = object()


def _default_account(provider: str) -> str:
    if provider == "pi":
        return "codex"
    if provider == "antigravity-resilient":
        return "antigravity"
    return provider


def writer_ledger_row(
    provider: str = "codex",
    model: object = _UNSET,
    effort: object = _UNSET,
    *,
    account: str | None = None,
    bundle_id: str | None = None,
    stage: str = "plan",
    choice: str | None = None,
    source: str | None = None,
    constraint: str | None = None,
    trigger: str | None = None,
    basis: str | None = None,
    at: dt.datetime | str | None = None,
    session_ids: object = _UNSET,
    parent_id: str | None = None,
    parent_agent_id: str | None = None,
    task_id: str | None = None,
    host_session_id: str | None = None,
    host_policy_violation: dict | None = None,
    bundle_extra: dict | None = None,
    **extra,
) -> dict:
    """Build a plan/record ledger row through `decision_ledger.finalize` so writer shape changes reach gate tests."""
    decision_ledger = gate.router_module("decision_ledger")
    if provider == "dot" and model is _UNSET:
        resolved_model = "dot"
        resolved_effort = None if effort is _UNSET else effort
        bundle_id = bundle_id or "dot_text"
        choice = choice or "dot_text"
    else:
        resolved_model = "gpt-6-astra" if model is _UNSET else model
        resolved_effort = "high" if effort is _UNSET else effort
    resolved_account = account if account is not None else _default_account(provider)
    resolved_choice = choice if choice is not None else ("constraint" if stage == "record" else f"{provider}_choice")
    resolved_source = source or ("constraint_record" if stage == "record" else "deterministic_allocator")
    when = (gate.parse_ts(at) if isinstance(at, str) else at) or (T0 - dt.timedelta(minutes=1))
    selected_bundle = {
        "provider": provider,
        "account": resolved_account,
        "model": resolved_model,
        "effort": resolved_effort,
        **({"bundle_id": bundle_id} if bundle_id is not None else {}),
        **(bundle_extra or {}),
    }
    parent = parent_id or parent_agent_id
    result = {
        "stage": stage,
        "source": resolved_source,
        "choice": resolved_choice,
        "constraint": constraint,
        "basis": basis,
        "selected_bundle": selected_bundle,
        "usage": {},
        **({"trigger": trigger} if trigger else {}),
        **({"parent_id": parent} if parent else {}),
        **({"host_policy_violation": host_policy_violation} if host_policy_violation is not None else {}),
    }
    task_doc = {
        **({"task_id": task_id} if task_id else {}),
        **({"host_session_id": host_session_id} if host_session_id else {}),
    }
    written: list[dict] = []
    sids = [] if session_ids is _UNSET else (list(session_ids) if session_ids else [])
    with mock.patch.object(decision_ledger.dispatch_registry, "issue", return_value={}), \
            mock.patch.object(decision_ledger, "own_session_ids", return_value=sids), \
            mock.patch.object(decision_ledger, "append_ledger", side_effect=lambda row: written.append(dict(row)) or True):
        decision_ledger.finalize(result, 0.0, task=task_doc, records=(), now=when)
    row = written[-1]
    if isinstance(at, str):
        row["at"] = at
    if extra:
        row.update(extra)
    return row


class RouterGateHookTests(unittest.TestCase):
    def test_confirmed_exhaustion_blocks_without_cache_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {'ROUTER_STATE_DIR': directory}):
            exhaustion = gate.router_module('quota_exhaustion')
            exhaustion.record('worker', 'Individual quota reached')
            missing = Path(directory) / 'missing.json'
            self.assertIsNotNone(gate.check_account_quota_exhausted('worker', missing, T0))
            exhaustion.clear('worker')
            self.assertIsNone(gate.check_account_quota_exhausted('worker', missing, T0))

    def test_the_latched_quota_deny_names_the_account_the_windows_and_the_way_to_clear_the_record(self):
        """Contract (parent-address-gate.md): a recorded confirmed failure refuses a launch drawing a latched window,
        and the deny says which account and windows are recorded and the command that clears the record."""
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {'ROUTER_STATE_DIR': directory}):
            exhaustion = gate.router_module('quota_exhaustion')
            missing = Path(directory) / 'missing.json'
            exhaustion.record('lab-acct', 'Individual quota reached', windows=['five_hour', 'weekly'])
            text = gate.check_account_quota_exhausted('lab-acct', missing, T0, windows=['weekly', 'monthly'])
            script = Path(exhaustion.__file__).resolve().as_posix()
            clear = f'python -X utf8 "{script}" --clear lab-acct'
            self.assertIn(clear, text)
            said = text.replace(clear, '')    # the account and windows are named by the refusal itself, not only by the command
            self.assertIn('lab-acct', said)
            self.assertIn('weekly', said)
            self.assertNotIn('monthly', said)    # recorded windows only, and only those this launch draws
            self.assertNotIn('five_hour', said)
            # No hold waits for a person (quota_exhaustion.due_at): the deny says when the record ends by itself, and the
            # clear is only a way to end it early.
            row = exhaustion._read(exhaustion.store_path())['accounts']['lab-acct']
            self.assertEqual(exhaustion.ends('lab-acct'), (exhaustion.due_at(row), False))
            self.assertIn(exhaustion.end_text('lab-acct'), said)       # the one sentence of when the latch ends
            self.assertIn('newer successful usage reading', said)
            exhaustion.clear('lab-acct')
            exhaustion.record('lab-acct', 'Usage limit reached. It resets at 15:00 UTC', windows=['weekly'])
            row = exhaustion._read(exhaustion.store_path())['accounts']['lab-acct']
            text = gate.check_account_quota_exhausted('lab-acct', missing, T0, windows=['weekly'])
            self.assertEqual(exhaustion.ends('lab-acct'), (exhaustion.stated_reset_of(row), True))
            self.assertIn(exhaustion.end_text('lab-acct'), text)
            exhaustion.clear('lab-acct')
            exhaustion.record('lab-acct', 'Individual quota reached')
            text = gate.check_account_quota_exhausted('lab-acct', missing, T0, windows=['weekly'])
            self.assertIn('every window of lab-acct', text)
            with mock.patch.object(gate.sys, 'frozen', True, create=True):
                self.assertIn(f'agentcompass -X utf8 "{script}" --clear lab-acct',
                              gate.check_account_quota_exhausted('lab-acct', missing, T0, windows=['weekly']))

    def run_main(self, raw):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, 'stdin', io.StringIO(raw)), mock.patch.object(sys, 'stdout', out), \
                mock.patch.object(sys, 'stderr', err):
            try:
                code = gate.main()
            except Exception as exc:    # the caller of a hook sees a traceback and exit 1
                code = exc
        return code, out.getvalue().strip(), err.getvalue().strip()

    UNPARSED = 'echo "secret-token-xyz'    # an unterminated quote parses in neither dialect
    SHELL_PAYLOADS = {
        'claude': {'tool_name': 'Bash', 'tool_input': {'command': UNPARSED}},
        'antigravity': {'toolCall': {'name': 'run_command', 'args': {'command': UNPARSED}}, 'workspaceRoots': []},
        'pi': {'toolName': 'bash', 'input': {'command': UNPARSED}},
    }

    def assert_allow_with_word(self, host, raw, word, secrets=()):
        """Each host reads its own allow shape; the word reaches it without becoming a deny: a `systemMessage` for Claude
        Code and Codex, the optional `reason` for Antigravity, stderr for Pi (whose extension shows it)."""
        code, out, err = self.run_main(raw)
        self.assertEqual(code, 0)
        doc = json.loads(out)
        if host == 'claude':
            self.assertEqual(list(doc), ['systemMessage'])
            said, other = doc['systemMessage'], err
        elif host == 'antigravity':    # the documented optional `reason` of a PreToolUse answer
            self.assertEqual(list(doc), ['decision', 'reason'])
            self.assertEqual(doc['decision'], 'allow')
            said, other = doc['reason'], err
        else:    # Pi: the generated extension reads stderr beside a `{block: false}`
            self.assertEqual(doc, {'block': False})
            said, other = err, ''
        self.assertTrue(said.startswith('[Router Gate]'), said)
        self.assertIn(word, said)
        self.assertEqual(other, '')
        for secret in secrets:
            self.assertNotIn(secret, said)

    def test_a_shell_command_read_in_neither_dialect_is_allowed_and_the_answer_says_it_was_not_read(self):
        for host, payload in self.SHELL_PAYLOADS.items():
            with self.subTest(host=host):
                self.assert_allow_with_word(host, json.dumps(payload), 'ShellParseError', secrets=('secret-token-xyz',))

    def test_an_exception_while_deciding_is_allowed_and_the_answer_says_which_type_not_its_text(self):
        payload = {'tool_name': 'Agent', 'tool_input': {'subagent_type': 'general-purpose', 'prompt': 'task text'},
                   'cwd': 'E:/x'}
        with mock.patch.object(gate, 'recent_routes', side_effect=OSError('C:/secret/ledger path')):
            self.assert_allow_with_word('claude', json.dumps(payload), 'OSError', secrets=('secret', 'task text'))

    def test_a_check_that_throws_is_allowed_and_the_answer_names_the_check_and_the_error_type(self):
        turn = {'tool_name': 'send_agent_prompt', 'tool_input': {'agentId': 'someone-else', 'prompt': 'hello'}}
        with mock.patch.object(gate.router_module('account_fleet'), 'load_accounts',
                               side_effect=RuntimeError('token=abc')):
            self.assert_allow_with_word('claude', json.dumps(turn), 'account fleet (RuntimeError)', secrets=('abc',))
        gate.UNEVALUATED.clear()
        cache = self.quota_cache({'claude': {'windows': [self.window('five_hour', 10)]}})
        self.assertIsNone(gate.check_account_quota_exhausted('claude', cache, T0, records=[], model='claude-opus-5-5'))
        self.assertEqual(gate.UNEVALUATED, [])    # a check that ran says nothing
        with mock.patch.object(gate.router_module('usage_reading'), 'usage_snapshot', side_effect=KeyError('k')):
            self.assertIsNone(gate.check_account_quota_exhausted(
                'claude', cache, T0, records=[], model='claude-opus-5-5'))
        self.assertEqual(gate.UNEVALUATED, ['quota window verdict (KeyError)'])

    def test_a_clean_call_after_an_unevaluated_one_in_the_same_process_carries_no_word(self):
        """evaluate_payload starts each decision with nothing unevaluated: a caller that evaluates payloads in one process."""
        self.assertIn('systemMessage', gate.evaluate_payload(self.SHELL_PAYLOADS['claude']))
        self.assertEqual(gate.evaluate_payload({'tool_name': 'Bash', 'tool_input': {'command': 'ls'}}), {})
        self.assertEqual(gate.UNEVALUATED, [])

    def test_only_a_shell_command_line_gets_the_notice_that_it_was_not_read(self):
        """Contract: the notice is for text the tool's contract calls a shell command line (`command`, `cmd`,
        `CommandLine`, the literal `cmd` of an exec_command call). The code of a code-mode tool is a program: read for
        launches as before, but never said to be an unread shell command."""
        programs = ("// don't forget\nconst x = 1; text(String(x));", 'const m = /it\'s/.test(s); text("it\'s");',
                    "print('don\\'t')")
        for program in programs:    # each is unreadable as a shell command line in both dialects
            self.assertRaises(gate.ShellParseError, gate.simple_commands, program, gate.POSIX)
            self.assertRaises(gate.ShellParseError, gate.simple_commands, program, gate.POWERSHELL)
        for tool in ('exec', 'codemode', 'execute_code'):
            for code in programs:
                with self.subTest(tool=tool, code=code):
                    gate.evaluate_payload({'tool_name': tool, 'tool_input': {'code': code}})
                    self.assertEqual(gate.UNEVALUATED, [])
        gate.evaluate_payload({'tool_name': 'exec', 'tool_input': {'code': 'await tools.exec_command({cmd: "echo \\"x"});'}})
        self.assertEqual(gate.UNEVALUATED, ['reading a shell command for launches (ShellParseError)'])

    DRIVER = """const handlers = {};
const written = [];
process.stderr.write = (text) => { written.push(String(text)); return true; };
const out = {};
for (const [name, extension, tool, input, hasUI, worker, notifyThrows] of JSON.parse(process.argv[2])) {
  if (!handlers[extension]) {
    const loaded = await import(`./${extension}.mjs`);
    loaded.default({ on: (event, h) => { if (event === "tool_call") handlers[extension] = h; } });
  }
  const shown = [];
  written.length = 0;
  if (worker) process.env.AGENTCOMPASS_WORKER = "1";
  const ctx = {
    hasUI,
    sessionManager: { getSessionId: () => "pi-sess-42" },
    ui: { notify: (message, level) => {
      if (notifyThrows) throw new Error("notify failed");
      shown.push([level, message]);
    } },
  };
  const returned = await handlers[extension]({ toolName: tool, toolCallId: "pi-tc-7", input }, ctx);
  out[name] = { returned: returned ?? null, shown: shown, stderr: [...written] };
}
process.stdout.write(JSON.stringify(out));
"""

    @unittest.skipUnless(shutil.which("node"), "node runs the generated Pi extension")
    def test_the_generated_pi_extension_shows_the_word_the_gate_leaves_beside_an_allow(self):
        """Contract (pi docs, extensions.md): a `tool_call` handler gets `(event, ctx)` and its return only controls
        blocking; `ctx.hasUI` is true in the TUI and RPC modes, where `ctx.ui.notify(message, "warning")` shows a message,
        and false in print and JSON mode, where UI methods are no-ops. So the stderr word is notified with a UI and
        written to the process's stderr without one; a block still blocks and a clean allow says nothing. The verdict is
        the gate's alone: showing the word, or text a failed gate left on stderr, never changes it."""
        import re
        import sync_host_globals as sg
        argv = json.dumps(sg.ROUTER_HOOK_ARGV)
        self.assertIn(argv, sg.PI_ROUTER_GATE_TS)
        fakes = {   # gates that answer as told: stdout is the verdict, stderr is what it also wrote
            "deny": "import sys; sys.stderr.write('[Router Gate] note\\\n'); print('{\"block\": true, \"reason\": \"deny\"}')",
            "crash": "import sys; sys.stderr.write('Traceback: boom\\\n'); sys.exit(1)",
            "echo": "import json, sys; d = json.load(sys.stdin); print(json.dumps({'block': True, 'reason': f\"{d.get('sessionId')}|{d.get('tool_call_id')}\"}))",
        }
        argvs = {"real": [sys.executable, "-X", "utf8", str(Path(gate.__file__).resolve())],
                 **{name: [sys.executable, "-c", code] for name, code in fakes.items()}}
        scenarios = [["ui", "real", "bash", {"command": self.UNPARSED}, True, False, False],
                     ["headless", "real", "bash", {"command": self.UNPARSED}, False, False, False],
                     ["clean", "real", "bash", {"command": "ls"}, True, False, False],
                     ["block", "real", "request_user_input", {}, True, True, False],
                     ["allow-notify-throws", "real", "bash", {"command": self.UNPARSED}, True, False, True],
                     ["deny-with-stderr", "deny", "bash", {"command": "ls"}, True, False, False],
                     ["deny-with-stderr-notify-throws", "deny", "bash", {"command": "ls"}, True, False, True],
                     ["crash", "crash", "bash", {"command": "ls"}, True, False, False],
                     ["echo-ids", "echo", "bash", {"command": "ls"}, True, False, False]]
        with tempfile.TemporaryDirectory() as directory:
            for name, hook in argvs.items():
                source = sg.PI_ROUTER_GATE_TS.replace(argv, json.dumps(hook))
                (Path(directory) / f"{name}.mjs").write_text(re.sub(r"(\w+): any\b", r"\1", source), encoding="utf-8", newline="\n")
            (Path(directory) / "driver.mjs").write_text(self.DRIVER, encoding="utf-8", newline="\n")
            done = subprocess.run(["node", "driver.mjs", json.dumps(scenarios)], cwd=directory, capture_output=True,
                                  text=True, encoding="utf-8", timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        out = json.loads(done.stdout)
        ui, headless = out["ui"], out["headless"]
        self.assertIsNone(ui["returned"])
        self.assertEqual([level for level, _ in ui["shown"]], ["warning"])
        notice = ui["shown"][0][1]
        self.assertTrue(notice.startswith("[Router Gate]") and "ShellParseError" in notice, notice)
        self.assertEqual(ui["stderr"], [])
        self.assertIsNone(headless["returned"])
        self.assertEqual((headless["shown"], headless["stderr"]), ([], [notice + "\n"]))
        self.assertEqual(out["clean"], {"returned": None, "shown": [], "stderr": []})
        self.assertTrue(out["block"]["returned"]["block"])
        self.assertEqual((out["block"]["shown"], out["block"]["stderr"]), ([], []))
        self.assertIsNone(out["allow-notify-throws"]["returned"])
        # A deny that also wrote stderr stays a deny whether or not showing the word works.
        for name in ("deny-with-stderr", "deny-with-stderr-notify-throws"):
            self.assertEqual(out[name]["returned"], {"block": True, "reason": "deny"}, name)
        # A gate that crashed (exit 1) shows nothing: its stderr is a traceback, not the gate's word.
        self.assertEqual(out["crash"], {"returned": None, "shown": [], "stderr": []})
        # F3: generated Pi extension forwards ctx.sessionManager.getSessionId() and event.toolCallId to the gate.
        self.assertEqual(out["echo-ids"]["returned"], {"block": True, "reason": "pi-sess-42|pi-tc-7"})

    def test_unreadable_stdin_is_allowed_with_a_word_in_the_shape_of_an_unknown_host(self):
        for label, raw in (('empty', ''), ('blank', '  \n'), ('not json', '{oops'), ('a list', '[1]')):
            with self.subTest(stdin=label):
                self.assert_allow_with_word('claude', raw, 'stdin')

    def test_an_answer_with_nothing_unevaluated_carries_no_word(self):
        payloads = {'claude': ({}, {'tool_name': 'Bash', 'tool_input': {'command': 'ls'}}),
                    'antigravity': ({'decision': 'allow'},
                                    {'toolCall': {'name': 'run_command', 'args': {'command': 'ls'}}}),
                    'pi': ({'block': False}, {'toolName': 'bash', 'input': {'command': 'ls'}})}
        for host, (shape, payload) in payloads.items():
            with self.subTest(host=host):
                self.assertEqual(self.run_main(json.dumps(payload)), (0, json.dumps(shape), ''))

    def test_a_bypassed_gate_does_not_report_what_it_did_not_read(self):
        with mock.patch.dict(os.environ, {'ROUTER_GATE_BYPASS': '1'}):
            self.assertEqual(self.run_main(json.dumps(self.SHELL_PAYLOADS['claude'])), (0, '{}', ''))

    def setUp(self):
        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        self.state = Path(state.name)
        env = mock.patch.dict(os.environ, {
            "ROUTER_GATE_BYPASS": "0",
            "ROUTER_LEDGER": str(self.state / "ledger.jsonl"),
            "ROUTER_USAGE_CACHE": str(self.state / "usage.json"),
            "ROUTER_PASEO_AGENTS_DIR": str(self.state / "agents"),
            "ROUTER_GATE_WINDOW_SECONDS": "900",
            "ROUTER_DISPATCH_DB": str(self.state / "dispatch.sqlite3"),
            "ROUTER_HOST_POLICY": str(self.state / "host-policy.json"),
            "AGENTCOMPASS_ACCOUNTS_OVERRIDE": str(self.state / "accounts-override.json"),
            "ROUTER_ACCOUNTS_FILE": str(gate.router_module("account_fleet").reference_fleet_path()),
        })
        env.start()
        self.addCleanup(env.stop)
        # The machine's installed-provider discovery would otherwise rewrite each account's verified flag.
        inventory = mock.patch.object(gate.router_module("provider_inventory"), "current", return_value={"accounts": {}})
        inventory.start()
        self.addCleanup(inventory.stop)

    def route(self, provider="codex", model=_UNSET, effort=_UNSET, **fields):
        return writer_ledger_row(provider, model, effort, **fields)

    def handover(
        self,
        provider: str = "codex",
        model: object = _UNSET,
        effort: object = _UNSET,
        *,
        surface: str = "native",
        account: str | None = None,
        bundle_id: str | None = None,
        parent_id: str = "parent-native",
        task_id: str = "task-native",
        host_session_id: str | None = None,
        session_ids: list[str] | None = None,
        constraint: str | None = None,
        decision_id: str | None = None,
        new_worker: bool = False,
        now: dt.datetime = T0,
        bundle_extra: dict | None = None,
    ) -> dict:
        """Record a one-use native hand-over through `dispatch_registry.record_host_claim` as `roll` writes it."""
        registry = gate.router_module("dispatch_registry")
        resolved_model = ("dot" if provider == "dot" else "gpt-6-astra") if model is _UNSET else model
        resolved_effort = (None if provider == "dot" else "high") if effort is _UNSET else effort
        resolved_account = account if account is not None else _default_account(provider)
        bundle = {
            "provider": provider,
            "account": resolved_account,
            "model": resolved_model,
            **({"effort": resolved_effort} if resolved_effort is not None else {}),
            **({"bundle_id": bundle_id} if bundle_id is not None else {}),
            **(bundle_extra or {}),
        }
        decision = {
            "decision_id": decision_id or f"dec-{task_id}-{provider}",
            "choice": bundle_id or ("dot_text" if provider == "dot" else f"{provider}_choice"),
            **({"constraint": constraint} if constraint else {}),
        }
        task = {
            "parent_agent_id": parent_id,
            "task_id": task_id,
            **({"host_session_id": host_session_id} if host_session_id else {}),
        }
        accounts_doc = gate.load_dispatch_accounts() or gate.router_module("account_fleet").load_accounts(
            gate.router_module("account_fleet").reference_fleet_path()
        )
        wire_model = bundle.get("launch_model") or resolved_model
        if provider == "pi" and not str(wire_model).startswith("openai-codex/"):
            wire_model = f"openai-codex/{wire_model}"
        elif provider == "opencode" and not str(wire_model).startswith("opencode/"):
            wire_model = f"opencode/{wire_model}"
        eff = None if provider.startswith("antigravity") else resolved_effort
        hook_model = None
        if surface == "orca":
            native_tool = "shell:orca worker launch"
            ag = "antigravity" if provider.startswith("antigravity") else ("claude" if provider.startswith("claude") else provider)
            launch_fields = {"--agent": ag, "--model": wire_model, **({"--effort": eff} if eff else {})}
        elif surface == "t3":
            native_tool = "delegate_task"
            drv = ("claudeAgent" if provider.startswith("claude")
                   else "antigravity" if provider.startswith("antigravity") else provider)
            opt_key = "reasoningEffort" if drv == "codex" else ("effort" if drv == "claudeAgent" else None)
            launch_fields = {"target": {"driverKind": drv, "model": wire_model,
                                        **({"options": {opt_key: eff}} if opt_key and eff else {})}}
        elif provider.startswith("claude"):
            native_tool = "Agent"
            fams = gate.router_module("model_resolution").family_patterns(accounts_doc)
            hit = gate.router_module("model_resolution").family_row(wire_model, fams)
            line = hit["line"] if isinstance(hit, dict) and hit.get("line") else wire_model
            launch_fields = {"model": line, **({"effort": eff} if eff else {})}
        elif provider.startswith("antigravity"):
            native_tool = "invoke_subagent"
            launch_fields = {"Model": "inherit"}
            hook_model = wire_model
        else:
            native_tool = "spawn_agent"
            launch_fields = {"model": wire_model, **({"reasoning_effort": eff} if eff else {})}
        with mock.patch.object(registry, "own_session_ids", return_value=list(session_ids or ())):
            return registry.record_host_claim(
                surface, decision, task, bundle, native_handover=True,
                native_tool=native_tool, launch_fields=launch_fields, hook_model=hook_model,
                new_worker=new_worker, now=now,
            )

    def test_worker_questions_are_blocked_but_main_questions_and_messages_pass(self):
        worker_id = self.existing_agent()
        agent_file = self.state / 'agents' / 'project' / (worker_id + '.json')
        for labels in ({'paseo.parent-agent-id': 'actual-parent'}, {'router.parent_id': 'actual-parent'}):
            agent_file.write_text(json.dumps({'labels': labels}), encoding='utf-8')
            payloads = (
                {'tool_name': 'functions.request_user_input_async', 'tool_input': {}},
                {'tool_name': 'AskUserQuestion', 'tool_input': {}},
                {'toolName': 'request_user_input_async', 'input': {}},
                {'toolCall': {'name': 'request_user_input', 'args': {}}},
                {'tool_name': 'exec', 'tool_input': 'await tools.request_user_input_async({questions: []})'},
            )
            for payload in payloads if 'paseo.parent-agent-id' in labels else payloads[:1]:
                with self.subTest(labels=labels, payload=payload):
                    with mock.patch.dict(os.environ, {'PASEO_AGENT_ID': worker_id, 'AGENTCOMPASS_WORKER': ''}):
                        denied = gate.evaluate_payload(payload)
                        self.assertIn('actual parent ID', denied['reason'])
                    with mock.patch.dict(os.environ, {'PASEO_AGENT_ID': '', 'AGENTCOMPASS_WORKER': ''}):
                        allowed = gate.evaluate_payload(payload)
                        self.assertNotIn('reason', allowed)
            with mock.patch.dict(os.environ, {'PASEO_AGENT_ID': worker_id}):
                self.assertEqual(gate.evaluate_payload({'tool_name': 'exec', 'tool_input':
                    'text("request_user_input_async({questions: []})")'}), {})
                self.assertEqual(gate.evaluate_payload({'tool_name': 'send_agent_prompt', 'tool_input':
                    {'agentId': 'actual-parent', 'prompt': 'question', 'background': True, 'notifyOnFinish': False}}), {})
        with mock.patch.dict(os.environ, {'PASEO_AGENT_ID': '', 'AGENTCOMPASS_WORKER': '1'}):
            self.assertEqual(gate.evaluate_payload({'tool_name': 'request_user_input_async'})['decision'], 'block')

    def test_parent_continuation_keeps_completion_notifications(self):
        worker_id = self.existing_agent(provider='')
        agent_file = self.state / 'agents' / 'project' / (worker_id + '.json')
        cases = (
            ('router owner', 'parent', {'router.parent_id': 'parent'}, False, True),
            ('paseo owner', 'parent', {'paseo.parent-agent-id': 'parent'}, False, True),
            ('both owners', 'parent', {'router.parent_id': 'parent', 'paseo.parent-agent-id': 'parent'}, False, True),
            ('enabled', 'parent', {'router.parent_id': 'parent'}, True, False),
            ('default', 'parent', {'router.parent_id': 'parent'}, None, False),
            ('other caller', 'peer', {'router.parent_id': 'parent'}, False, False),
            ('unknown caller', '', {'router.parent_id': 'parent'}, False, False),
            ('unknown owner', 'parent', {}, False, False),
            ('conflicting owners', 'parent', {'router.parent_id': 'parent', 'paseo.parent-agent-id': 'other'}, False, False),
            ('self-conflicting owner', 'parent', {'router.parent_id': 'parent', 'paseo.parent-agent-id': worker_id}, False, False),
            ('self', worker_id, {'router.parent_id': worker_id}, False, False),
            ('missing record', 'parent', None, False, False),
            ('unreadable record', 'parent', 'broken json', False, False),
        )
        for label, caller, labels, notify, blocked in cases:
            if labels is None:
                agent_file.unlink(missing_ok=True)
            else:
                agent_file.write_text(json.dumps({'labels': labels}) if isinstance(labels, dict) else labels,
                                      encoding='utf-8', newline='\n')
            args = {'agentId': worker_id, 'prompt': 'continue', 'background': True}
            if notify is not None:
                args['notifyOnFinish'] = notify
            with self.subTest(case=label), \
                    mock.patch.dict(os.environ, {'PASEO_AGENT_ID': caller}), \
                    mock.patch.object(gate, 'load_dispatch_accounts', return_value={}):
                result = gate.evaluate_payload({'tool_name': 'send_agent_prompt', 'tool_input': args})
                self.assertEqual('reason' in result, blocked)
                if blocked:
                    self.assertIn('notifyOnFinish: true or omit it', result['reason'])

        # Formats carry the same semantics; do not multiply the ownership table
        # by each transport representation.
        agent_file.write_text(json.dumps({'labels': {'router.parent_id': 'parent'}}), encoding='utf-8')
        args = {'agentId': worker_id, 'prompt': 'continue', 'notifyOnFinish': False}
        for payload in (
            {'tool_name': 'mcp__paseo__send_agent_prompt', 'tool_input': args},
            {'toolName': 'send_agent_prompt', 'input': args},
            {'toolCall': {'name': 'send_agent_prompt', 'args': args}},
            {'tool_name': 'exec', 'tool_input': 'await tools.mcp__paseo__send_agent_prompt(' + json.dumps(args) + ')'},
        ):
            with self.subTest(format=payload), mock.patch.dict(os.environ, {'PASEO_AGENT_ID': 'parent'}):
                self.assertIn('notifyOnFinish', gate.evaluate_payload(payload)['reason'])


    def exhausted_cache(self, **changes):
        row = {"available": True, "stale": False, "fetched_at": T0.isoformat(),
               "windows": [{"id": "five_hour", "used_pct": 100,
                            "resets_at": (T0 + dt.timedelta(hours=2)).isoformat()}]}
        row.update(changes)
        path = self.state / "usage.json"
        path.write_text(json.dumps({"providers": {"claude-lab": row, "claude": row}}), encoding="utf-8")
        return path

    def existing_agent(self, provider="claude-lab/claude-sonnet-5-5", owner=None):
        """An existing Paseo agent; `owner` writes the persisted parent label that makes it that caller's worker."""
        directory = self.state / "agents" / "project"
        directory.mkdir(parents=True, exist_ok=True)
        agent_id = "11111111-1111-4111-8111-111111111111"
        doc = {"provider": provider, **({"labels": {"router.parent_id": owner}} if owner else {})}
        (directory / (agent_id + ".json")).write_text(json.dumps(doc), encoding="utf-8")
        return agent_id

    def write_ledger(self, directory: str, rows: list[dict]) -> Path:
        path = Path(directory) / "router-ledger.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    def test_non_subagent_tools_always_pass_without_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_ledger = Path(tmp) / "missing.jsonl"
            for payload in (
                {"tool_name": "Read", "tool_input": {"file_path": "a.py"}},
                {"tool_name": "Bash", "tool_input": {"command": 'git commit -m "fix spawn_agent and create_agent()"'}},
                {"tool_name": "exec_command", "tool_input": {"cmd": "rg -n create_agent src/"}},
                {"toolCall": {"name": "view_file", "args": {"AbsolutePath": "E:/a.py"}}},
            ):
                res = gate.evaluate_payload(payload, ledger_path=missing_ledger, now=T0)
                self.assertNotIn("deny", json.dumps(res))
                self.assertNotIn("block", [ k for k, v in res.items() if v is True or v == "block"])

    def test_a_built_in_tool_stand_in_agent_is_not_a_launch_unless_redefined(self):
        # Rule: a Claude Code built-in agent that is a host tool or command in agent form (HOST_TOOL_AGENTS) launches
        # no worker; a project or user definition of that name replaces the built-in and is a launch like any other.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.jsonl"
            project, home = root / "repo" / "pkg", root / "claude-home"
            project.mkdir(parents=True)

            def decide(agent, tool="Agent", cwd=project):
                payload = {"tool_name": tool, "cwd": str(cwd), "tool_input": {"subagent_type": agent, "prompt": "read https://example.com"}}
                with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(home)}):
                    return gate.evaluate_payload(payload, ledger_path=missing, now=T0).get("decision")

            for agent in gate.HOST_TOOL_AGENTS:
                self.assertEqual([decide(agent), decide(agent, tool="Task")], [None, None], agent)
                # The same name on another host's sub-agent tool is that host's own agent: a launch.
                self.assertEqual(decide(agent, tool="spawn_agent"), "block", agent)
                file = project / ".claude" / "agents" / "own.md"
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(f"---\nname: {agent}\ntools: Bash\n---\n", encoding="utf-8", newline="\n")
                self.assertEqual(decide(agent), "block", agent)
                file.unlink()
            # Every agent type that does the session's own work in another context stays a launch.
            for agent in ("Explore", "Plan", "general-purpose", "claude", "fork", "web-fetch-2", ""):
                self.assertEqual(decide(agent), "block", agent)
            # A definition of that name is found by its frontmatter name, in a subfolder, in a parent project
            # directory or in the user's Claude home; a file without a name is known by its file name.
            cases = (
                (root / "repo" / ".claude" / "agents" / "tools" / "fetcher.md", "---\nname: web-fetch\ntools: Bash, Write\n---\nDo anything.\n"),
                (project / ".claude" / "agents" / "x.md", "---\nname: \"web-fetch\"\n---\n"),
                (home / "agents" / "web-fetch.md", "No frontmatter.\n"),
            )
            for file, body in cases:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(body, encoding="utf-8", newline="\n")
                self.assertEqual(decide("web-fetch"), "block", file)
                file.unlink()
                self.assertIsNone(decide("web-fetch"), file)
            # Another agent's definition does not turn the built-in into a launch.
            other = project / ".claude" / "agents" / "web-fetch.md"
            other.write_text("---\nname: page-reader\n---\n", encoding="utf-8", newline="\n")
            self.assertIsNone(decide("web-fetch"))

    def test_blocks_unrouted_subagent_tools_across_harnesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.jsonl"
            # Claude Agent tool
            claude_res = gate.evaluate_payload(
                {"tool_name": "Agent", "tool_input": {"subagent_type": "Explore", "prompt": "find files"}},
                ledger_path=missing,
                now=T0,
            )
            self.assertEqual(claude_res["decision"], "block")
            self.assertEqual(claude_res["hookSpecificOutput"]["permissionDecision"], "deny")

            # Codex spawn_agent tool
            codex_res = gate.evaluate_payload(
                {"tool_name": "spawn_agent", "tool_input": {"message": "do work"}},
                ledger_path=missing,
                now=T0,
            )
            self.assertEqual(codex_res["decision"], "block")

            # Antigravity invoke_subagent tool
            agy_res = gate.evaluate_payload(
                {"toolCall": {"name": "invoke_subagent", "args": {"Subagents": [{"TypeName": "research"}]}}},
                ledger_path=missing,
                now=T0,
            )
            self.assertEqual(agy_res["decision"], "deny")
            self.assertIn("Router Gate", agy_res["reason"])

            # Paseo codemode create_agent (Claude & Pi 'codemode' tool name)
            paseo_res = gate.evaluate_payload(
                {"tool_name": "mcp__paseo__execute_code", "tool_input": {"code": "await create_agent({provider: 'codex'})"}},
                ledger_path=missing,
                now=T0,
            )
            self.assertEqual(paseo_res["decision"], "block")

            pi_codemode_res = gate.evaluate_payload(
                {"toolName": "codemode", "input": {"code": "await tools.mcp__paseo__create_agent({provider: 'codex'})"}},
                ledger_path=missing,
                now=T0,
            )
            self.assertTrue(pi_codemode_res["block"])

            # Codex unified exec string input
            codex_exec_res = gate.evaluate_payload(
                {"tool_name": "exec", "tool_input": 'text(await tools.exec_command({cmd:"orca agent launch -m gpt-6-astra \'task\'"}))'},
                ledger_path=missing,
                now=T0,
            )
            self.assertEqual(codex_exec_res["decision"], "block")

            # Shell worker spawns (paseo agent create, orca agent launch, run_dot_task.py)
            for cmd in (
                "paseo agent create --provider codex/gpt-6-astra",
                "orca agent launch -m gpt-6-astra 'task'",
                "python ~/.codex/skills/dot-subagent/scripts/run_dot_task.py --prompt 'hi'",
            ):
                shell_res = gate.evaluate_payload(
                    {"tool_name": "Bash", "tool_input": {"command": cmd}},
                    ledger_path=missing,
                    now=T0,
                )
                self.assertEqual(shell_res["decision"], "block", msg=cmd)

    def test_orca_and_t3_worker_launches_are_gated_but_their_other_commands_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-ledger.jsonl"

            def decide(tool, tool_input):
                return gate.evaluate_payload({"tool_name": tool, "tool_input": tool_input}, ledger_path=missing, now=T0).get("decision", "allow")

            for cmd in (
                'orca orchestration worker-start --spec "task" --worktree current --agent codex --json',
                "orca.cmd agent launch --agent claude --prompt hi",
                r"C:\Users\me\.orca\bin\orca.exe orchestration worker-start --task t1 --agent claude",
            ):
                self.assertEqual(decide("Bash", {"command": cmd}), "block", msg=cmd)
            for cmd in (
                "orca status --json",
                "orca worktree list --json",
                "orca orchestration check --wait --json",
                "orca orchestration worker-start --help",
            ):
                self.assertNotEqual(decide("Bash", {"command": cmd}), "block", msg=cmd)
            for tool in ("delegate_task", "mcp__t3-code__delegate_task"):
                self.assertEqual(decide(tool, {"task": "do it"}), "block", msg=tool)
            self.assertNotEqual(decide("mcp__t3-code__task_status", {"taskId": "t"}), "block")

    def test_recent_plan_is_found_behind_snapshot_sized_lines(self):
        """Plan lines carry a decision snapshot (up to ~24 KB); the tail read must still reach the route behind them."""
        with tempfile.TemporaryDirectory() as tmp:
            heavy = {"at": (T0 - dt.timedelta(minutes=1)).isoformat(), "stage": "snapshot-only", "pad": "x" * 24000}
            route = self.route("codex", "gpt-6.1-sol", "high", choice="sol_high", bundle_id="sol_high",
                               at=(T0 - dt.timedelta(minutes=4)).isoformat())
            ledger = self.write_ledger(tmp, [route] + [dict(heavy, n=i) for i in range(30)])
            self.assertGreater(ledger.stat().st_size, 64 * 1024 * 8)
            routes = gate.recent_routes(ledger, now=T0)
            self.assertTrue(any(r.get("choice") == route["choice"] for r in routes))

    def test_native_launch_requires_native_handover_or_standalone_record_never_plan(self):
        """Contract: a `plan` row never authorizes a native launch; only an unconsumed `roll` native hand-over or a
        recent standalone `record` row of this session does."""
        with tempfile.TemporaryDirectory() as tmp:
            plan_row = self.route("codex", "gpt-6.1-sol", "high", choice="sol_high", bundle_id="sol_high",
                                  session_ids=["sess-1"], parent_id="sess-1",
                                  at=(T0 - dt.timedelta(minutes=2)).isoformat())
            plan_ledger = self.write_ledger(tmp, [plan_row])
            # 1. A recent `plan` row (even from the same session and matching model) blocks every native launch tool.
            for payload, blocked_key, blocked_val in (
                ({"session_id": "sess-1", "tool_name": "Agent",
                  "tool_input": {"subagent_type": "Explore", "prompt": "find files"}}, "decision", "block"),
                ({"session_id": "sess-1", "tool_name": "spawn_agent",
                  "tool_input": {"message": "work", "model": "gpt-6.1-sol", "effort": "high"}}, "decision", "block"),
                ({"toolCall": {"name": "invoke_subagent", "args": {"Subagents": []}}}, "decision", "deny"),
            ):
                verdict = gate.evaluate_payload(payload, ledger_path=plan_ledger, now=T0)
                self.assertEqual(verdict.get(blocked_key), blocked_val, payload)
                self.assertNotIn("兼容检查", verdict.get("reason", ""))

            # 2. A standalone `record` row of this session authorizes matching native launches (and blocks another session).
            record_ledger = self.write_ledger(tmp, [
                self.route("claude", "claude-haiku-5-5", "high", stage="record", constraint="user_override",
                           bundle_id=None, session_ids=["sess-1"], parent_id="sess-1",
                           at=(T0 - dt.timedelta(minutes=4)).isoformat()),
                self.route("antigravity", "gemini-3.8-flash-high", "high", stage="record", constraint="user_override",
                           bundle_id="flash_high", session_ids=["sess-1"], parent_id="sess-1",
                           at=(T0 - dt.timedelta(minutes=4)).isoformat()),
            ])
            self.assertEqual(
                gate.evaluate_payload(
                    {"session_id": "sess-1", "tool_name": "Agent",
                     "tool_input": {"subagent_type": "Explore", "model": "claude-haiku-5-5", "prompt": "find files"}},
                    ledger_path=record_ledger, now=T0,
                ),
                {},
            )
            self.assertEqual(
                gate.evaluate_payload(
                    {"session_id": "sess-other", "tool_name": "Agent",
                     "tool_input": {"subagent_type": "Explore", "model": "claude-haiku-5-5", "prompt": "find files"}},
                    ledger_path=record_ledger, now=T0,
                )["decision"],
                "block",
            )
            self.assertEqual(
                gate.evaluate_payload(
                    {"conversationId": "sess-1", "modelName": "gemini-3.8-flash-high",
                     "toolCall": {"id": "agy-1", "name": "invoke_subagent", "args": {"Model": "inherit"}}},
                    ledger_path=record_ledger, now=T0,
                ),
                {"decision": "allow"},
            )

            # 3. An unconsumed `roll` native hand-over authorizes one launch (even at minute 16, past the 15-min ledger
            # window), is idempotent on retries of the same tool_use_id, blocks a second launch, and degrades at 24h.
            missing = Path(tmp) / "missing.jsonl"
            claim = self.handover("codex", "gpt-6.1-sol", "high", bundle_id="sol_high",
                                  parent_id="sess-native", session_ids=["sess-native"], now=T0)
            at_min_16 = T0 + dt.timedelta(minutes=16)
            call_1 = {"session_id": "sess-native", "tool_use_id": "tu-1", "tool_name": "spawn_agent",
                      "tool_input": {"message": "work", "model": "gpt-6.1-sol", "reasoning_effort": "high"}}
            # Another session cannot consume it
            other_sess = dict(call_1, session_id="sess-stranger", tool_use_id="tu-stranger")
            self.assertEqual(gate.evaluate_payload(other_sess, ledger_path=missing, now=at_min_16)["decision"], "block")
            # Wrong model does not consume the hand-over
            wrong_model = dict(call_1, tool_use_id="tu-wrong",
                               tool_input={"message": "work", "model": "gpt-6-astra", "reasoning_effort": "high"})
            wrong_res = gate.evaluate_payload(wrong_model, ledger_path=missing, now=at_min_16)
            self.assertEqual(wrong_res["decision"], "block")
            self.assertIn("model: requested='gpt-6-astra'", wrong_res["reason"])
            self.assertNotIn("兼容检查", wrong_res["reason"])
            # First matching launch at minute 16 passes and consumes the hand-over; same tool_use_id retry stays allowed
            self.assertEqual(gate.evaluate_payload(call_1, ledger_path=missing, now=at_min_16), {})
            self.assertEqual(gate.evaluate_payload(call_1, ledger_path=missing, now=at_min_16), {})
            registry = gate.router_module("dispatch_registry")
            self.assertEqual(registry.inspect(claim["dispatch_id"])["state"], "launched")
            # Second launch with a new tool_use_id (or none) is blocked
            for second_id in ("tu-2", None):
                second_call = {k: v for k, v in dict(call_1, tool_use_id=second_id).items() if v is not None}
                self.assertEqual(gate.evaluate_payload(second_call, ledger_path=missing, now=at_min_16)["decision"], "block")
            # Unconsumed hand-over past MAX_LIFETIME (24h) degrades to blocked
            stale_claim = self.handover("codex", "gpt-6.1-sol", "high", bundle_id="sol_high",
                                        parent_id="sess-stale", session_ids=["sess-stale"], task_id="task-stale", now=T0)
            after_24h = T0 + dt.timedelta(seconds=registry.MAX_LIFETIME + 1)
            stale_call = {"session_id": "sess-stale", "tool_use_id": "tu-stale", "tool_name": "spawn_agent",
                          "tool_input": {"message": "work", "model": "gpt-6.1-sol", "reasoning_effort": "high"}}
            self.assertEqual(gate.evaluate_payload(stale_call, ledger_path=missing, now=after_24h)["decision"], "block")
            self.assertIsNone(registry.inspect(stale_claim["dispatch_id"]).get("launched_at"))

    def test_real_roll_writers_authorize_only_when_roll_hands_over_natively(self):
        """End-to-end with the real `roll`, `route_judge`, `decision_ledger` and `dispatch_registry` writers:
        (a) a Paseo-launched `roll` (automatic or `--provider` pinned) blocks native launches in that session;
        (b) a deferred `roll` (`action: defer`) blocks native launches;
        (c) all three `roll` `dispatch_mode == 'native'` answers (`NATIVE_HOSTS`, `cli` without CLI family, `cli` Popen
            failure) record a one-use native hand-over that authorizes exactly one native launch;
        (d) two wrapped native launches in one `exec` call cannot share a single hand-over."""
        roll_mod = gate.router_module("roll")
        route_judge = gate.router_module("route_judge")
        decision_ledger = gate.router_module("decision_ledger")
        registry = gate.router_module("dispatch_registry")
        flow = gate.router_module("dispatch_flow")
        account_fleet = gate.router_module("account_fleet")
        usage_reading = gate.router_module("usage_reading")
        subscription_cli = gate.router_module("subscription_cli")
        rts = gate.router_module("router_test_support")

        accounts = account_fleet.load_accounts(account_fleet.reference_fleet_path())
        accounts["accounts"]["claude-lab"]["plan"] = "max20"
        self.write_settings(allowed=["claude", "claude-lab", "codex", "antigravity"])
        usage = {
            "read_ok": True,
            "accounts": {
                "claude-lab": {
                    "status": "fresh", "fetched_at": T0.isoformat(),
                    "windows": [
                        {"id": "five_hour", "used_pct": 10, "remaining_pct_before_reset": 90,
                         "observed_remaining_pct": 90, "seconds_until_reset": 9000,
                         "resets_at": (T0 + dt.timedelta(hours=2.5)).isoformat()},
                        {"id": "seven_day", "used_pct": 10, "remaining_pct_before_reset": 90,
                         "observed_remaining_pct": 90, "seconds_until_reset": 400000,
                         "resets_at": (T0 + dt.timedelta(hours=110)).isoformat()},
                    ],
                },
            },
        }
        brief = self.state / "brief.md"
        brief.write_text("# Implement fix\n", encoding="utf-8")
        ledger = self.state / "ledger.jsonl"

        def fake_paseo_host(request):
            op = request.get("operation")
            if op == "workspace_agents":
                return {"ok": True, "placeable": True, "agents": []}
            if op == "create":
                now_iso = dt.datetime.now(UTC).isoformat()
                prov, _, mod = request["request"]["provider"].partition("/")
                snap = {
                    "id": "agent-paseo-1", "provider": prov, "model": mod,
                    "currentModeId": request["request"]["settings"].get("modeId"),
                    "thinkingOptionId": request["request"]["settings"].get("thinkingOptionId"),
                    "status": "running", "updatedAt": now_iso, "labels": request["request"]["labels"],
                }
                return {"ok": True, "sent": True, "agent": snap, "completion_notification": {"status": "watching"}}
            return {"ok": False, "error": "agent_not_observed"}

        original_build_task = roll_mod.build_task

        lab_home = self.state / ".claude-lab"
        lab_home.mkdir(parents=True, exist_ok=True)
        disc_payload = {
            "accounts": {"claude-lab": {"family": "claude", "status": "ready", "home": str(lab_home)}},
            "launchers": {"claude-lab": {"family": "claude", "status": "ready"}},
        }
        accounts["_discovery"] = disc_payload

        def run_real_roll(*argv, env=None, host_fn=fake_paseo_host, current_accounts=None, current_usage=None, current_disc=None):
            args = route_judge.build_parser().parse_args(["roll", "--brief", str(brief), *argv])
            args.brief = args.brief[0]
            active_accounts = current_accounts or accounts
            active_disc = current_disc or disc_payload
            active_usage = current_usage or usage
            active_accounts.setdefault("_discovery", active_disc)
            with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": "parent-a", **(env or {})}), \
                    mock.patch.object(account_fleet, "load_accounts", return_value=active_accounts), \
                    mock.patch.object(registry.provider_inventory, "current", return_value=active_disc), \
                    mock.patch.object(usage_reading, "usage_snapshot", side_effect=lambda *_, **__: active_usage), \
                    mock.patch.object(subscription_cli, "subscription_env", side_effect=lambda account=None: dict(os.environ)), \
                    mock.patch.object(subscription_cli, "cli_prefix", side_effect=lambda name: [name]), \
                    mock.patch.object(roll_mod, "WORKER_PROMPTS_DIR", self.state / "wp"), \
                    mock.patch.object(roll_mod, "RESULTS_DIR", self.state / "rr"), \
                    mock.patch.object(flow, "host_call", host_fn), \
                    mock.patch.object(roll_mod, "build_task",
                                      lambda *a, **k: dict(original_build_task(*a, **k),
                                                           profile=rts.profile_for({"work_kind": "code_implementation", "difficulty": "hard"}))):
                return roll_mod.run(args, route_judge)

        # (a) Paseo-launched roll (both automatic `plan` row and `--provider` pinned `record` row) never authorizes native Agent
        for extra_flags, tid in (((), "t-paseo-auto"),
                                 (("--provider", "claude-lab", "--account", "claude-lab", "--model", "claude-sonnet-5-5",
                                   "--effort", "high", "--basis", "user asked"), "t-paseo-pinned")):
            res, code = run_real_roll("--task-id", tid, *extra_flags)
            self.assertEqual((code, res["status"]), (0, "launched"))
            with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": "parent-a"}):
                denied = gate.evaluate_payload(
                    {"session_id": "parent-a", "tool_name": "Agent",
                     "tool_input": {"subagent_type": "general-purpose", "prompt": "work", "model": "sonnet"}},
                    ledger_path=ledger, now=dt.datetime.now(UTC),
                )
            self.assertEqual(denied["decision"], "block", (tid, denied))

        # (b) Deferred roll (`action: defer`) writes a `plan` row with a bundle and still authorizes no native launch
        defer_accounts = json.loads(json.dumps(accounts))
        defer_accounts["accounts"]["claude-lab"]["cadence"] = {"window_seconds": 4 * 3600, "limit": 1}
        defer_accounts["scoring"]["defer_if_reset_within_hours"] = 4
        now_live = dt.datetime.now(UTC)
        prior = registry.record_host_claim(
            "cli", {"decision_id": "prior-cadence"}, {"parent_agent_id": "parent-a", "task_id": "t-prior"},
            dict(account_fleet.bundles(defer_accounts)["lab_sonnet55_high"], bundle_id="lab_sonnet55_high"),
            now=now_live - dt.timedelta(hours=2),
        )
        registry.host_claim_launched(prior["dispatch_id"], now=now_live - dt.timedelta(hours=2))
        registry.settle_host_claim(prior["dispatch_id"], "completed", now=now_live - dt.timedelta(hours=1))
        res_def, code_def = run_real_roll("--task-id", "t-defer", "--can-wait-hours", "4", "--new-worker",
                                          current_accounts=defer_accounts)
        self.assertEqual((code_def, res_def["status"], res_def.get("caller_action")), (2, "blocked", "wait_for_reset"))
        self.assertEqual(decision_ledger.read_ledger(ledger)[-1].get("action"), "defer")
        with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": "parent-a"}):
            denied_def = gate.evaluate_payload(
                {"session_id": "parent-a", "tool_name": "Agent",
                 "tool_input": {"subagent_type": "general-purpose", "prompt": "work", "model": "sonnet"}},
                ledger_path=ledger, now=dt.datetime.now(UTC),
            )
        self.assertEqual(denied_def["decision"], "block")

        # (c) All `dispatch_mode == 'native'` paths of `roll` authorize one native launch and block the second:
        #   c0: `--parent-host native` inside Claude Code (`Agent` with `launch_fields`)
        res_c0, code_c0 = run_real_roll(
            "--task-id", "t-native-0", "--parent-host", "native", "--parent-id", "claude-s0",
            "--parent-address", "inbox:s0", "--parent-model", "claude-sonnet-5-5", "--new-worker",
            env={"PASEO_AGENT_ID": "", "CLAUDE_CODE_SESSION_ID": "claude-s0", "CLAUDE_CONFIG_DIR": str(lab_home)},
        )
        self.assertEqual((code_c0, res_c0["status"], res_c0["native_tool"], res_c0["launch_fields"]),
                         (0, "native", "Agent", {"model": "sonnet", "effort": "high"}))
        c0_payload = {"session_id": "claude-s0", "tool_use_id": "c0-a", "tool_name": "Agent",
                      "tool_input": {"subagent_type": "general-purpose", "prompt": "work", **res_c0["launch_fields"]}}
        self.assertEqual(gate.evaluate_payload(c0_payload, ledger_path=ledger, now=dt.datetime.now(UTC)), {})
        self.assertEqual(
            gate.evaluate_payload(dict(c0_payload, tool_use_id="c0-b"), ledger_path=ledger, now=dt.datetime.now(UTC))["decision"],
            "block",
        )

        #   c1: `surface in NATIVE_HOSTS` (`orca` terminal with inner Claude session on the single default `claude` home)
        default_claude_home = Path(os.environ["HOME"]) / ".claude"
        default_claude_home.mkdir(parents=True, exist_ok=True)
        orca_accounts = json.loads(json.dumps(accounts))
        orca_accounts["accounts"]["claude"]["plan"] = "max20"
        orca_disc = {
            "accounts": {"claude": {"family": "claude", "status": "ready", "home": str(default_claude_home)}},
            "launchers": {"claude": {"family": "claude", "status": "ready"}},
        }
        orca_accounts["_discovery"] = orca_disc
        orca_usage = {"read_ok": True, "accounts": {"claude": usage["accounts"]["claude-lab"]}}
        res_c1, code_c1 = run_real_roll(
            "--task-id", "t-native-1", "--new-worker",
            current_accounts=orca_accounts, current_usage=orca_usage, current_disc=orca_disc,
            env={"PASEO_AGENT_ID": "", "ORCA_AGENT_SESSION_ID": "orca-s1",
                 "ORCA_TERMINAL_HANDLE": "term_1", "CLAUDE_CODE_SESSION_ID": "claude-s1"},
        )
        self.assertEqual((code_c1, res_c1["status"], res_c1["native_tool"]),
                         (0, "native", "shell:orca worker launch"))
        c1_payload = {
            "session_id": "claude-s1", "tool_use_id": "c1-a", "tool_name": "Bash",
            "tool_input": {"command": 'orca orchestration worker-start --spec "x" --agent claude --model claude-sonnet-5-5 --effort high --json'},
        }
        self.assertEqual(gate.evaluate_payload(c1_payload, ledger_path=ledger, now=dt.datetime.now(UTC)), {})
        self.assertEqual(
            gate.evaluate_payload(dict(c1_payload, tool_use_id="c1-b"), ledger_path=ledger, now=dt.datetime.now(UTC))["decision"],
            "block",
        )

        #   c2: `roll.cli` when `family not in CLI_RUNS` (no Paseo reachable and no local CLI runner for provider)
        with mock.patch.object(flow, "paseo_reachable", return_value=False), \
                mock.patch.object(roll_mod, "provider_definition", return_value=("custom-no-cli", {})):
            res_c2, code_c2 = run_real_roll(
                "--task-id", "t-native-2", "--parent-id", "sess-c2", "--parent-model", "claude-sonnet-5-5", "--new-worker",
                env={"PASEO_AGENT_ID": "", "CLAUDE_CODE_SESSION_ID": "sess-c2", "CLAUDE_CONFIG_DIR": str(lab_home)},
            )
        self.assertEqual((code_c2, res_c2["status"]), (0, "native"))
        c2_payload = {"session_id": "sess-c2", "tool_use_id": "c2-a", "tool_name": "Agent",
                      "tool_input": {"subagent_type": "general-purpose", "prompt": "work", **res_c2["launch_fields"]}}
        self.assertEqual(gate.evaluate_payload(c2_payload, ledger_path=ledger, now=dt.datetime.now(UTC)), {})
        self.assertEqual(
            gate.evaluate_payload(dict(c2_payload, tool_use_id="c2-b"), ledger_path=ledger, now=dt.datetime.now(UTC))["decision"],
            "block",
        )

        #   c3: `roll.cli` when `subprocess.Popen` raises OSError (`CLI could not start; handed over to Agent`)
        with mock.patch.object(flow, "paseo_reachable", return_value=False), \
                mock.patch.object(roll_mod, "provider_definition", return_value=("claude", {})), \
                mock.patch.object(roll_mod.subprocess, "Popen", side_effect=OSError("missing binary")):
            res_c3, code_c3 = run_real_roll(
                "--task-id", "t-native-3", "--parent-id", "sess-c3", "--parent-model", "claude-sonnet-5-5", "--new-worker", "--verbose",
                env={"PASEO_AGENT_ID": "", "CLAUDE_CODE_SESSION_ID": "sess-c3", "CLAUDE_CONFIG_DIR": str(lab_home)},
            )
        self.assertEqual((code_c3, res_c3["dispatch_mode"], res_c3["caller_action"]),
                         (0, "native", "launch_with_native_host_tool"))
        c3_payload = {"session_id": "sess-c3", "tool_use_id": "c3-a", "tool_name": "Agent",
                      "tool_input": {"subagent_type": "general-purpose", "prompt": "work", **res_c3["launch_fields"]}}
        self.assertEqual(gate.evaluate_payload(c3_payload, ledger_path=ledger, now=dt.datetime.now(UTC)), {})
        self.assertEqual(
            gate.evaluate_payload(dict(c3_payload, tool_use_id="c3-b"), ledger_path=ledger, now=dt.datetime.now(UTC))["decision"],
            "block",
        )

        # (d) Two wrapped `spawn_agent` calls inside one `exec` payload cannot share a single hand-over, AND (F2)
        # refusing the call leaves the single hand-over `prepared` (`launched_at is None`) so a single retry succeeds.
        wrap_claim = self.handover("codex", "gpt-6-astra", "high", parent_id="sess-wrap", session_ids=["sess-wrap"],
                                   task_id="t-wrap", now=T0)
        two_spawns = {
            "session_id": "sess-wrap", "tool_use_id": "wrap-1", "tool_name": "exec",
            "tool_input": 'await spawn_agent({message: "one", model: "gpt-6-astra", reasoning_effort: "high"}); '
                          'await spawn_agent({message: "two", model: "gpt-6-astra", reasoning_effort: "high"})',
        }
        refused_two = gate.evaluate_payload(two_spawns, ledger_path=ledger, now=T0)
        self.assertEqual(refused_two["decision"], "block")
        self.assertIn("本次调用启动 2 个子 agent，只有 1 个可用授权", refused_two["reason"])
        after_refused = registry.inspect(wrap_claim["dispatch_id"])
        self.assertEqual((after_refused["state"], after_refused.get("launched_at")), ("prepared", None))
        one_spawn = {
            "session_id": "sess-wrap", "tool_use_id": "wrap-single", "tool_name": "spawn_agent",
            "tool_input": {"message": "one", "model": "gpt-6-astra", "reasoning_effort": "high"},
        }
        self.assertEqual(gate.evaluate_payload(one_spawn, ledger_path=ledger, now=T0), {})
        self.assertEqual(registry.inspect(wrap_claim["dispatch_id"])["state"], "launched")
        spent = gate.evaluate_payload(dict(one_spawn, tool_use_id="wrap-after-host-decline"), ledger_path=ledger, now=T0)
        self.assertEqual(spent["decision"], "block")
        self.assertIn("门禁放行时即已核销", spent["reason"])

    def test_unreadable_or_locked_registry_denies_native_launches_fast_without_blocking_non_launches(self):
        """Contract Degrade (F1): unreadable or write-locked `dispatch.sqlite3` authorizes no native launch, never
        falling into `evaluate_payload`'s allow catch-all or waiting 10s on a write lock; built-in tool agents and
        non-launch tools do not touch the registry and pass immediately."""
        import sqlite3
        import time

        registry = gate.router_module("dispatch_registry")
        db_file = Path(os.environ["ROUTER_DISPATCH_DB"])
        ledger = self.write_ledger(str(self.state), [
            self.route("codex", "gpt-6.1-sol", "high", stage="record", constraint="user_override",
                       bundle_id="sol_high", session_ids=["s-f1"], parent_id="s-f1"),
        ])
        native_call = {
            "session_id": "s-f1", "tool_use_id": "tu-f1", "tool_name": "spawn_agent",
            "tool_input": {"message": "work", "model": "gpt-6.1-sol", "reasoning_effort": "high"},
        }
        # 1. Corrupt `dispatch.sqlite3` denies native launches even when a ledger `record` row exists:
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_file.write_bytes(b"not a sqlite database" * 100)
        corrupt_res = gate.evaluate_payload(native_call, ledger_path=ledger, now=T0)
        self.assertEqual(corrupt_res["decision"], "block", corrupt_res)
        self.assertIn("DatabaseError", corrupt_res["reason"])
        self.assertNotIn("Allowed without checking", str(corrupt_res))

        # 2. Write-locked `dispatch.sqlite3` with an open automatic hand-over denies in < 3s (never 10s), leaves the
        #    hand-over `prepared`, and built-in tool agents + `Read` pass immediately with no unevaluated notice:
        db_file.unlink()
        missing = self.state / "missing-f1.jsonl"
        claim = self.handover("codex", "gpt-6.1-sol", "high", bundle_id="sol_high",
                              parent_id="s-f1", session_ids=["s-f1"], task_id="t-f1-lock", now=T0)
        writer = sqlite3.connect(db_file, timeout=1.0, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            for non_launch in (
                {"tool_name": "Read", "tool_input": {"file_path": "README.md"}},
                {"tool_name": "Agent", "tool_input": {"subagent_type": "web-fetch", "prompt": "read url"}},
                {"tool_name": "Agent", "tool_input": {"subagent_type": "claude-code-guide", "prompt": "hooks"}},
            ):
                t_fast = time.monotonic()
                self.assertEqual(gate.evaluate_payload(non_launch, ledger_path=missing, now=T0), {})
                self.assertLess(time.monotonic() - t_fast, 0.5)
            t_lock = time.monotonic()
            locked_res = gate.evaluate_payload(native_call, ledger_path=missing, now=T0)
            elapsed = time.monotonic() - t_lock
        finally:
            writer.rollback()
            writer.close()
        self.assertEqual(locked_res["decision"], "block", locked_res)
        self.assertIn("OperationalError", locked_res["reason"])
        self.assertLess(elapsed, 3.0)
        self.assertIsNone(registry.inspect(claim["dispatch_id"]).get("launched_at"))
        # Once the writer lock releases, the untouched hand-over authorizes the retry:
        self.assertEqual(gate.evaluate_payload(native_call, ledger_path=missing, now=T0), {})
        self.assertEqual(registry.inspect(claim["dispatch_id"])["state"], "launched")

    def test_later_check_failure_in_same_call_leaves_native_handover_unconsumed(self):
        """Contract Decision (F2): hand-overs are claimed only after every check across the whole call passes."""
        registry = gate.router_module("dispatch_registry")
        missing = self.state / "missing-f2.jsonl"
        claim = self.handover("codex", "gpt-6-astra", "high", parent_id="s-f2", session_ids=["s-f2"],
                              task_id="t-f2", now=T0)
        mixed_call = {
            "session_id": "s-f2", "tool_use_id": "tu-f2-mixed", "tool_name": "exec",
            "tool_input": 'await spawn_agent({message: "ok", model: "gpt-6-astra", reasoning_effort: "high"}); '
                          'await paseo.create_agent({provider: "codex/gpt-6-astra"})',
        }
        res = gate.evaluate_payload(mixed_call, ledger_path=missing, now=T0)
        self.assertEqual(res["decision"], "block")
        self.assertIsNone(registry.inspect(claim["dispatch_id"]).get("launched_at"))

    def test_native_handover_is_one_use_sequentially_and_under_concurrent_claims(self):
        """Contract Decision (N6): a `roll` native hand-over authorizes at most one native launch through real writers,
        both sequentially and when two calls race past `native_handovers` into `claim_native_handovers`, and the unused
        singular `claim_native_handover` wrapper is gone."""
        import concurrent.futures
        import threading

        registry = gate.router_module("dispatch_registry")
        self.assertFalse(hasattr(registry, "claim_native_handover"))
        missing = self.state / "missing-n6.jsonl"

        # 1. Sequential: first launch consumes the hand-over, second launch with a distinct tool_use_id is refused.
        seq_claim = self.handover("codex", "gpt-6.1-sol", "high", bundle_id="sol_high",
                                  parent_id="s-n6-seq", session_ids=["s-n6-seq"], task_id="t-n6-seq", now=T0)
        seq_call = {
            "session_id": "s-n6-seq", "tool_use_id": "tu-n6-1", "tool_name": "spawn_agent",
            "tool_input": {"message": "work", "model": "gpt-6.1-sol", "reasoning_effort": "high"},
        }
        self.assertEqual(gate.evaluate_payload(seq_call, ledger_path=missing, now=T0), {})
        second_seq = gate.evaluate_payload(dict(seq_call, tool_use_id="tu-n6-2"), ledger_path=missing, now=T0)
        self.assertEqual(second_seq["decision"], "block")
        self.assertEqual(registry.inspect(seq_claim["dispatch_id"])["claim_id"], "tu-n6-1")

        # 2. Race into `claim_native_handovers`: two concurrent launches both read `native_handovers` while open, then
        #    race into `claim_native_handovers`; `_handover_open` inside the transaction admits at most one.
        race_claim = self.handover("codex", "gpt-6.1-sol", "high", bundle_id="sol_high",
                                   parent_id="s-n6-race", session_ids=["s-n6-race"], task_id="t-n6-race", now=T0)
        real_native_handovers = registry.native_handovers
        barrier = threading.Barrier(2, timeout=5.0)

        def synced_handovers(*args, **kwargs):
            rows = real_native_handovers(*args, **kwargs)
            barrier.wait()
            return rows

        race_base = {
            "session_id": "s-n6-race", "tool_name": "spawn_agent",
            "tool_input": {"message": "work", "model": "gpt-6.1-sol", "reasoning_effort": "high"},
        }
        with mock.patch.object(registry, "native_handovers", side_effect=synced_handovers):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(gate.evaluate_payload, dict(race_base, tool_use_id=f"tu-race-{i}"),
                                ledger_path=missing, now=T0)
                    for i in range(2)
                ]
                verdicts = [f.result(timeout=10.0) for f in futures]
        allowed = [v for v in verdicts if v == {}]
        blocked = [v for v in verdicts if v.get("decision") == "block"]
        self.assertEqual((len(allowed), len(blocked)), (1, 1), verdicts)
        self.assertIn(registry.inspect(race_claim["dispatch_id"])["claim_id"], ("tu-race-0", "tu-race-1"))

    def test_sessionless_caller_matches_neither_named_handover_nor_named_record_while_bare_record_works(self):
        """Contract Decision & Degrade (N1, N4, F3, F8, F9): one session rule governs both facts — a hand-over or
        `record` row that names a session is matched only by a call naming that same session, and a call carrying no
        session ID matches only what names no session (a bare-shell `record` row)."""
        self.assertFalse(hasattr(gate, "OWN_SESSION_ID_VARS"))
        registry = gate.router_module("dispatch_registry")
        missing = self.state / "missing-f3.jsonl"
        owner_claim = self.handover("antigravity", "gemini-3.8-flash-high", "high", bundle_id="flash_high",
                                    parent_id="agy-owner", session_ids=["agy-owner"], task_id="t-agy-f3", now=T0)
        # 1. Session-less Antigravity launch cannot steal `agy-owner`'s hand-over, and deny text names setup + restart (N9: never roll):
        sessionless_agy = {"modelName": "gemini-3.8-flash-high",
                           "toolCall": {"id": "tc-anon", "name": "invoke_subagent", "args": {"Model": "inherit"}}}
        denied = gate.evaluate_payload(sessionless_agy, ledger_path=missing, now=T0)
        self.assertEqual(denied["decision"], "deny", denied)
        self.assertIn("未携带会话 ID", denied["reason"])
        self.assertIn("setup", denied["reason"])
        self.assertIn("重启", denied["reason"])
        self.assertNotIn("roll --brief", denied["reason"])
        self.assertIsNone(registry.inspect(owner_claim["dispatch_id"]).get("launched_at"))
        # Real Antigravity payload (`conversationId` + concrete `modelName` + `toolCall.id`) matches and retries idempotently:
        real_agy = {"conversationId": "agy-owner", "modelName": "gemini-3.8-flash-high",
                    "toolCall": {"id": "tc-owner-1", "name": "invoke_subagent", "args": {"Model": "inherit"}}}
        self.assertEqual(gate.evaluate_payload(real_agy, ledger_path=missing, now=T0), {"decision": "allow"})
        self.assertEqual(gate.evaluate_payload(real_agy, ledger_path=missing, now=T0), {"decision": "allow"})
        self.assertEqual(
            gate.evaluate_payload(
                {**real_agy, "toolCall": {**real_agy["toolCall"], "id": "tc-owner-2"}},
                ledger_path=missing, now=T0,
            )["decision"],
            "deny",
        )
        # 2. N1: A `record` row of session `s1` authorizes `s1`, and refuses both `s2` and session-less calls
        #    (including a v0.7.6-shaped Pi payload without `sessionId`):
        s1_ledger = self.write_ledger(str(self.state), [
            self.route("codex", "gpt-6.1-sol", "high", stage="record", constraint="hard_constraint",
                       bundle_id="sol_high", session_ids=["s1"], parent_id="s1"),
        ])
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s1", "tool_name": "spawn_agent",
                 "tool_input": {"message": "x", "model": "gpt-6.1-sol", "reasoning_effort": "high"}},
                ledger_path=s1_ledger, now=T0,
            ),
            {},
        )
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s2", "tool_name": "spawn_agent",
                 "tool_input": {"message": "x", "model": "gpt-6.1-sol", "reasoning_effort": "high"}},
                ledger_path=s1_ledger, now=T0,
            )["decision"],
            "block",
        )
        sless_codex = gate.evaluate_payload(
            {"tool_name": "spawn_agent",
             "tool_input": {"message": "x", "model": "gpt-6.1-sol", "reasoning_effort": "high"}},
            ledger_path=s1_ledger, now=T0,
        )
        self.assertEqual(sless_codex.get("decision"), "block", sless_codex)
        self.assertIn("未携带会话 ID", sless_codex["reason"])
        self.assertIn("重启", sless_codex["reason"])
        self.assertNotIn("roll --brief", sless_codex["reason"])
        pi_v076 = gate.evaluate_payload(
            {"toolName": "spawn_agent", "cwd": str(self.state),
             "input": {"message": "x", "model": "gpt-6.1-sol", "reasoning_effort": "high"}},
            ledger_path=s1_ledger, now=T0,
        )
        self.assertEqual(pi_v076.get("block"), True, pi_v076)
        self.assertIn("未携带会话 ID", pi_v076["reason"])
        self.assertIn("重启", pi_v076["reason"])
        self.assertNotIn("roll --brief", pi_v076["reason"])
        # 3. F9: Section 6 standalone `record` written in a bare shell (`session_ids=[]`, `parent_id=None`) names no
        #    session and keeps authorizing matching launches (session-less and sessioned):
        bare_ledger = self.write_ledger(str(self.state), [
            self.route("claude", "claude-haiku-5-5", "high", stage="record", constraint="hard_constraint",
                       bundle_id=None, session_ids=[], parent_id=None),
        ])
        self.assertEqual(
            gate.evaluate_payload(
                {"tool_name": "Agent", "tool_input": {"subagent_type": "Explore", "model": "haiku", "prompt": "find"}},
                ledger_path=bare_ledger, now=T0,
            ),
            {},
        )
        for idx in range(2):
            self.assertEqual(
                gate.evaluate_payload(
                    {"session_id": f"s-bare-{idx}", "tool_name": "Agent",
                     "tool_input": {"subagent_type": "Explore", "model": "haiku", "prompt": "find"}},
                    ledger_path=bare_ledger, now=T0,
                ),
                {},
            )

    def test_unnamed_model_and_provider_resolve_from_host_facts_never_as_wildcard(self):
        """Contract Decision (F4 & N3): each host's route resolves only from what the call carries and declared catalog
        data (zero model-name or substring rules: `Explore` without `model` and Antigravity `flash`/`flash_lite`/`pro`
        tier keywords never guess a model), denying and naming the launch field to pass."""
        registry = gate.router_module("dispatch_registry")
        missing = self.state / "missing-f4.jsonl"

        # 1. Claude `Agent`: `Explore`, `explore`, and `general-purpose` without `model` (or `model="inherit"`) are
        #    denied even when the session holds a `claude-haiku-5-5` or `claude-opus-5-5` hand-over, naming `model`;
        #    custom `.claude/agents/rev.md` frontmatter (`model: opus`) resolves from the file and passes.
        haiku_claim = self.handover("claude", "claude-haiku-5-5", "high", bundle_id=None,
                                    parent_id="s-c1", session_ids=["s-c1"], task_id="t-f4-haiku", now=T0)
        for sub_name in ("Explore", "explore", "general-purpose"):
            denied_haiku = gate.evaluate_payload(
                {"session_id": "s-c1", "tool_name": "Agent", "tool_input": {"subagent_type": sub_name, "prompt": "x"}},
                ledger_path=missing, now=T0,
            )
            self.assertEqual(denied_haiku["decision"], "block", sub_name)
            self.assertIn("`model`", denied_haiku["reason"])
            self.assertIn("`effort`", denied_haiku["reason"])
        self.assertIsNone(registry.inspect(haiku_claim["dispatch_id"]).get("launched_at"))

        opus_claim = self.handover("claude", "claude-opus-5-5", "high", bundle_id="opus55_high",
                                   parent_id="s-c2", session_ids=["s-c2"], task_id="t-f4-opus", now=T0)
        for bad_args in (
            {"subagent_type": "Explore", "prompt": "x"},
            {"subagent_type": "general-purpose", "prompt": "x"},
            {"subagent_type": "general-purpose", "prompt": "x", "model": "inherit"},
        ):
            denied = gate.evaluate_payload(
                {"session_id": "s-c2", "tool_name": "Agent", "tool_input": bad_args},
                ledger_path=missing, now=T0,
            )
            self.assertEqual(denied["decision"], "block", bad_args)
            self.assertIn("`model`", denied["reason"])
        self.assertIsNone(registry.inspect(opus_claim["dispatch_id"]).get("launched_at"))
        project = self.state / "proj-f4"
        (project / ".claude" / "agents").mkdir(parents=True, exist_ok=True)
        (project / ".claude" / "agents" / "rev.md").write_text(
            "---\nname: rev\nmodel: opus\neffort: high\n---\nReview\n", encoding="utf-8",
        )
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s-c2", "cwd": str(project), "tool_use_id": "tu-rev",
                 "tool_name": "Agent", "tool_input": {"subagent_type": "rev", "prompt": "x"}},
                ledger_path=missing, now=T0,
            ),
            {},
        )

        # 2. Codex `spawn_agent`: omitted `tool_input.model` uses top-level `payload.model`; missing both denies naming `model`.
        sol_claim = self.handover("codex", "gpt-6.1-sol", "high", bundle_id="sol_high",
                                  parent_id="s-x1", session_ids=["s-x1"], task_id="t-f4-sol", now=T0)
        denied_codex = gate.evaluate_payload(
            {"session_id": "s-x1", "tool_name": "spawn_agent", "tool_input": {"message": "x"}},
            ledger_path=missing, now=T0,
        )
        self.assertEqual(denied_codex["decision"], "block")
        self.assertIn("`model`", denied_codex["reason"])
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s-x1", "model": "gpt-6-astra", "tool_name": "spawn_agent", "tool_input": {"message": "x"}},
                ledger_path=missing, now=T0,
            )["decision"],
            "block",
        )
        self.assertIsNone(registry.inspect(sol_claim["dispatch_id"]).get("launched_at"))
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s-x1", "model": "gpt-6.1-sol", "tool_use_id": "tu-sol",
                 "tool_name": "spawn_agent", "tool_input": {"message": "x", "reasoning_effort": "high"}},
                ledger_path=missing, now=T0,
            ),
            {},
        )

        # 3. Antigravity `invoke_subagent`: omitted/`inherit` `Model` uses concrete `modelName`; `auto`/absent and
        #    abstract tier keywords (`flash`, `flash_lite`, `pro`) never match by substring and deny naming `Model`/`modelName`.
        flash_claim = self.handover("antigravity", "gemini-3.8-flash-high", "high", bundle_id="flash_high",
                                    parent_id="s-af", session_ids=["s-af"], task_id="t-f4-agy-flash", now=T0)
        for tier_kw in ("flash", "flash_lite", "pro"):
            denied_tier = gate.evaluate_payload(
                {"conversationId": "s-af", "modelName": "gemini-3.8-flash-high",
                 "toolCall": {"id": f"af-{tier_kw}", "name": "invoke_subagent", "args": {"TypeName": "self", "Model": tier_kw}}},
                ledger_path=missing, now=T0,
            )
            self.assertEqual(denied_tier["decision"], "deny", tier_kw)
            self.assertIn("`Model=\"inherit\"`", denied_tier["reason"])
            self.assertNotIn("指定具体模型 ID", denied_tier["reason"])
        self.assertIsNone(registry.inspect(flash_claim["dispatch_id"]).get("launched_at"))

        argon_claim = self.handover("antigravity", "gemini-4-argon-high", "high", bundle_id="argon_high",
                                    parent_id="s-a1", session_ids=["s-a1"], task_id="t-f4-agy", now=T0)
        for missing_model_name in (
            {"conversationId": "s-a1",
             "toolCall": {"id": "a-0", "name": "invoke_subagent", "args": {"TypeName": "self", "Model": "inherit"}}},
            {"conversationId": "s-a1", "modelName": "auto",
             "toolCall": {"id": "a-1", "name": "invoke_subagent", "args": {"TypeName": "self"}}},
        ):
            denied_missing = gate.evaluate_payload(missing_model_name, ledger_path=missing, now=T0)
            self.assertEqual(denied_missing["decision"], "deny", missing_model_name)
            self.assertIn("`modelName`", denied_missing["reason"])
            self.assertNotIn("Model=\"inherit\"", denied_missing["reason"])
        for bad_agy in (
            {"conversationId": "s-a1", "modelName": "gemini-3.8-flash-high",
             "toolCall": {"id": "a-2", "name": "invoke_subagent", "args": {"TypeName": "self"}}},
            {"conversationId": "s-a1",
             "toolCall": {"id": "a-3", "name": "invoke_subagent", "args": {"TypeName": "self", "Model": "pro"}}},
        ):
            denied_agy = gate.evaluate_payload(bad_agy, ledger_path=missing, now=T0)
            self.assertEqual(denied_agy["decision"], "deny", bad_agy)
            self.assertIn("Model=\"inherit\"", denied_agy["reason"])
        self.assertIsNone(registry.inspect(argon_claim["dispatch_id"]).get("launched_at"))
        self.assertEqual(
            gate.evaluate_payload(
                {"conversationId": "s-a1", "modelName": "gemini-4-argon-high",
                 "toolCall": {"id": "a-ok", "name": "invoke_subagent", "args": {"TypeName": "self", "Model": "inherit"}}},
                ledger_path=missing, now=T0,
            ),
            {"decision": "allow"},
        )

        # 4. Multi-provider `delegate_task` (`target.driverKind`/`target.model`) and `orca orchestration worker-start`
        #    require explicit provider and model and name those fields when missing:
        self.handover("codex", "gpt-6.1-sol", "high", surface="t3", bundle_id="sol_high",
                      parent_id="s-t3", session_ids=["s-t3"], task_id="t-f4-t3", now=T0)
        denied_t3 = gate.evaluate_payload(
            {"session_id": "s-t3", "tool_name": "delegate_task", "tool_input": {"task": "work"}},
            ledger_path=missing, now=T0,
        )
        self.assertEqual(denied_t3["decision"], "block")
        self.assertIn("target.driverKind", denied_t3["reason"])
        self.assertIn("target.model", denied_t3["reason"])

        # 5. N8 & N13: Each entry of an `invoke_subagent` `Subagents` list is one launch (its own `Model` checked, one
        #    hand-over per entry, zero claims committed when any entry fails or fewer hand-overs exist than entries,
        #    and N13's refusal names how many sub-agents the call launches vs how many open hand-overs this session holds).
        n8_single = self.handover("antigravity-resilient", "gemini-3.8-flash-high", "high", bundle_id="flash_high",
                                  parent_id="s-n8", session_ids=["s-n8"], task_id="t-n8-1", now=T0)
        for label, subagents in (
            ("two pro entries", [{"TypeName": "research", "Model": "pro"}, {"TypeName": "research", "Model": "pro"}]),
            ("one mismatched concrete model", [{"TypeName": "research", "Model": "gemini-4-argon-high"}]),
            ("two inherit entries on one handover", [{"TypeName": "research"}, {"TypeName": "research"}]),
            ("three inherit entries on one handover", [{"TypeName": "research"}, {"TypeName": "research"}, {"TypeName": "research"}]),
        ):
            with self.subTest(n8=label):
                res_n8 = gate.evaluate_payload(
                    {"conversationId": "s-n8", "modelName": "gemini-3.8-flash-high",
                     "toolCall": {"id": f"tc-{label}", "name": "invoke_subagent", "args": {"Subagents": subagents}}},
                    ledger_path=missing, now=T0,
                )
                self.assertEqual(res_n8["decision"], "deny", (label, res_n8))
                self.assertEqual(registry.inspect(n8_single["dispatch_id"])["state"], "prepared", label)
                if "inherit entries on one handover" in label:
                    self.assertIn(f"本次调用启动 {len(subagents)} 个子 agent，只有 1 个可用授权", res_n8["reason"])
                    self.assertNotIn("本会话没有尚未使用的", res_n8["reason"])
        # Adding a second matching hand-over lets a two-entry `Subagents` call pass, claiming both idempotently:
        n8_second = self.handover("antigravity-resilient", "gemini-3.8-flash-high", "high", bundle_id="flash_high",
                                  parent_id="s-n8", session_ids=["s-n8"], task_id="t-n8-2", now=T0)
        batch_ok = {
            "conversationId": "s-n8", "modelName": "gemini-3.8-flash-high",
            "toolCall": {"id": "tc-n8-two", "name": "invoke_subagent",
                         "args": {"Subagents": [{"TypeName": "research", "Model": "inherit"}, {"TypeName": "research"}]}},
        }
        self.assertEqual(gate.evaluate_payload(batch_ok, ledger_path=missing, now=T0), {"decision": "allow"})
        self.assertEqual(gate.evaluate_payload(batch_ok, ledger_path=missing, now=T0), {"decision": "allow"})
        self.assertEqual(registry.inspect(n8_single["dispatch_id"])["state"], "launched")
        self.assertEqual(registry.inspect(n8_second["dispatch_id"])["state"], "launched")
        self.assertEqual(
            gate.evaluate_payload(
                {"conversationId": "s-n8", "modelName": "gemini-3.8-flash-high",
                 "toolCall": {"id": "tc-n8-third", "name": "invoke_subagent",
                              "args": {"Subagents": [{"TypeName": "research"}]}}},
                ledger_path=missing, now=T0,
            )["decision"],
            "deny",
        )

        # 6. N11: `model_line_matches` reads `families` through `model_resolution.family_patterns` (returning False on
        #    `ReferenceDataError` and applying only to Claude native tools, never matching `pro` on `spawn_agent`), and
        #    `router_gate_hook.py` carries no `"anthropic"` or `"-resilient"` name rules.
        self.assertFalse(hasattr(gate, "anthropic_line_matches"))
        self.assertFalse(gate.model_line_matches("sonnet", "claude-sonnet-5-5", {"accounts": {}}))
        pro_claim = self.handover("codex", "gpt-6-pro", "high", bundle_id=None,
                                  parent_id="s-n11-pro", session_ids=["s-n11-pro"], task_id="t-n11-pro", now=T0)
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s-n11-pro", "tool_name": "spawn_agent",
                 "tool_input": {"message": "x", "model": "pro", "reasoning_effort": "high"}},
                ledger_path=missing, now=T0,
            )["decision"],
            "block",
        )
        self.assertEqual(registry.inspect(pro_claim["dispatch_id"])["state"], "prepared")
        gate_src = Path(gate.__file__).read_text(encoding="utf-8")
        self.assertNotIn("anthropic", gate_src.lower())
        self.assertNotIn("-resilient", gate_src)
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s-t3", "tool_use_id": "tu-t3-ok", "tool_name": "delegate_task",
                 "tool_input": {"task": "work", "target": {"driverKind": "codex", "model": "gpt-6.1-sol",
                                                           "options": {"reasoningEffort": "high"}}}},
                ledger_path=missing, now=T0,
            ),
            {},
        )
        self.handover("codex", "gpt-6.1-sol", "high", surface="orca", bundle_id="sol_high",
                      parent_id="s-orca", session_ids=["s-orca"], task_id="t-f4-orca", now=T0)
        denied_orca = gate.evaluate_payload(
            {"session_id": "s-orca", "tool_name": "Bash",
             "tool_input": {"command": 'orca orchestration worker-start --spec "x" --worktree current --json'}},
            ledger_path=missing, now=T0,
        )
        self.assertEqual(denied_orca["decision"], "block")
        self.assertIn("--agent", denied_orca["reason"])
        self.assertIn("--model", denied_orca["reason"])
        self.assertEqual(
            gate.evaluate_payload(
                {"session_id": "s-orca", "tool_use_id": "tu-orca-ok", "tool_name": "Bash",
                 "tool_input": {"command": 'orca orchestration worker-start --spec "x" --agent codex --model gpt-6.1-sol --effort high --json'}},
                ledger_path=missing, now=T0,
            ),
            {},
        )

    def test_expired_or_self_only_ledger_entry_still_blocks(self):
        """Rule: a plan/record row without a route bundle (or outside the TTL window) authorizes nothing."""
        decision_ledger = gate.router_module("decision_ledger")
        route_judge = gate.router_module("route_judge")
        captured: list[dict] = []
        with mock.patch.object(decision_ledger.dispatch_registry, "issue", return_value={}), \
                mock.patch.object(decision_ledger, "append_ledger", side_effect=lambda r: captured.append(dict(r)) or True):
            route_judge.scorer_unavailable_result(
                {"objective": "x"}, {"source": "scorer_unavailable", "model": None}, 0.0)
            scorer_unavailable_row = dict(captured[-1], at=(T0 - dt.timedelta(minutes=1)).isoformat())
            decision_ledger.finalize(
                {"stage": "plan", "choice": "none", "selected_bundle": None, "decision": {"action": "none"}},
                0.0, task={"objective": "x"}, records=(), now=T0 - dt.timedelta(minutes=1))
            no_eligible_row = captured[-1]

        recent = (T0 - dt.timedelta(minutes=1)).isoformat()
        expired_plan = self.route("codex", "gpt-6.1-sol", "high", choice="sol_high", bundle_id="sol_high",
                                  at=(T0 - dt.timedelta(minutes=30)).isoformat())
        expired_record = self.route("codex", "gpt-6.1-sol", "high", stage="record", constraint="user_override",
                                    bundle_id="sol_high", at=(T0 - dt.timedelta(minutes=30)).isoformat())
        self.assertFalse(hasattr(gate, "has_route_bundle"))
        self.assertFalse(hasattr(gate, "carries_route"))
        self.assertTrue(decision_ledger.carries_route(expired_plan))
        self.assertTrue(decision_ledger.carries_route(expired_record))
        routeless_rows = [
            scorer_unavailable_row,
            no_eligible_row,
            {"at": recent, "stage": "plan", "choice": "self"},
            {"at": recent, "stage": "plan", "choice": "sol_high"},
            {"at": recent, "stage": "plan", "choice": "sol_high", "bundle": None},
            {"at": recent, "stage": "plan", "choice": "sol_high",
             "bundle": {"account": "codex", "model": "gpt-6.1-sol"}},
            {"at": recent, "stage": "plan", "choice": "sol_high",
             "bundle": {"provider": "codex", "model": "gpt-6.1-sol"}},
            {"at": recent, "stage": "plan", "choice": "sol_high",
             "bundle": {"provider": "codex", "account": "codex", "model": "   "}},
            {"at": recent, "stage": "plan", "choice": "sol_high",
             "selected_bundle": {"provider": "codex", "account": "codex", "model": "gpt-6.1-sol", "effort": "high"}},
            {"at": recent, "stage": "plan", "choice": "sol_high",
             "provider": "codex", "account": "codex", "model": "gpt-6.1-sol", "effort": "high"},
        ]
        for row in routeless_rows:
            self.assertFalse(decision_ledger.carries_route(row), row)

        with tempfile.TemporaryDirectory() as tmp:
            ledger = self.write_ledger(tmp, [expired_plan, expired_record, *routeless_rows])
            self.assertEqual(gate.recent_routes(ledger, now=T0), [])
            res = gate.evaluate_payload(
                {"tool_name": "spawn_agent", "tool_input": {"message": "hi"}},
                ledger_path=ledger,
                now=T0,
            )
            self.assertEqual(res["decision"], "block")
            # Direct match_route callers also refuse routeless rows even when requested carries no launch fields.
            matched, _ = gate.match_route(routeless_rows, {}, {}, "tool:spawn_agent")
            self.assertIsNone(matched)

    def test_rows_of_the_removed_select_and_delegate_stages_authorize_nothing(self):
        """Rule: a row of any stage other than plan/record authorizes nothing, even if it carries a bundle."""
        decision_ledger = gate.router_module("decision_ledger")
        recent = (T0 - dt.timedelta(minutes=1)).isoformat()
        full_bundle = {"provider": "codex", "account": "codex", "model": "gpt-6.1-sol", "effort": "high"}
        non_launch_rows = [
            {"at": recent, "stage": "select", "choice": "sol_high", "bundle": full_bundle},
            {"at": recent, "stage": "delegate", "choice": "delegate", "bundle": full_bundle},
            {"at": recent, "stage": "split", "choice": "split", "bundle": full_bundle},
            {"at": recent, "stage": "fallback", "action": "relaunch", "failed": full_bundle, "bundle": full_bundle},
            {"at": recent, "stage": "override", "action": "upsert", "account": "codex", "model": "gpt-6.1-sol",
             "bundle": full_bundle},
        ]
        for row in non_launch_rows:
            self.assertFalse(decision_ledger.carries_route(row), row)
        # Conversely, un-efforted bundles (dot_text) and record rows without bundle_id do carry a route.
        self.assertTrue(decision_ledger.carries_route(self.route("dot")))
        self.assertTrue(decision_ledger.carries_route(self.route("claude", "claude-opus-5-5", "high", stage="record",
                                                                 constraint="user_override", bundle_id=None)))
        with tempfile.TemporaryDirectory() as tmp:
            ledger = self.write_ledger(tmp, non_launch_rows)
            self.assertEqual(gate.recent_routes(ledger, now=T0), [])
            res = gate.evaluate_payload(
                {"tool_name": "spawn_agent", "tool_input": {"message": "hi"}},
                ledger_path=ledger,
                now=T0,
            )
            self.assertEqual(res["decision"], "block")

    def test_dot_status_and_doctor_pass_without_ledger_and_dot_route_blocks_local_subagent(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.jsonl"
            for cmd in (
                "python ~/.codex/skills/dot-subagent/scripts/run_dot_task.py doctor",
                "python ~/.codex/skills/dot-subagent/scripts/run_dot_task.py status --task-id task_123",
                "python ~/.codex/skills/dot-subagent/scripts/run_dot_task.py run --task-id task_123 --wait 60",
            ):
                res = gate.evaluate_payload(
                    {"tool_name": "Bash", "tool_input": {"command": cmd}},
                    ledger_path=missing,
                    now=T0,
                )
                self.assertEqual(res, {}, msg=cmd)

            dot_row = self.route("dot", stage="record", constraint="dot_closed_loop",
                                 at=(T0 - dt.timedelta(minutes=2)).isoformat())
            self.assertNotIn("effort", dot_row["bundle"])
            self.assertTrue(gate.is_dot_route(dot_row))
            self.assertFalse(gate.is_dot_route({"stage": "plan", "choice": "dot_text"}))
            self.assertFalse(gate.is_dot_route(self.route("codex", "gpt-6-astra", "high", choice="dot_text")))
            dot_ledger = self.write_ledger(tmp, [dot_row])
            # Allowing run_dot_task.py when dot_text is selected
            ok_dot = gate.evaluate_payload(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "python ~/.codex/skills/dot-subagent/scripts/run_dot_task.py run --prompt-file task.md"},
                },
                ledger_path=dot_ledger,
                now=T0,
            )
            self.assertEqual(ok_dot, {})

            # Blocking local subagent bypass when dot_text is selected
            bypass_local = gate.evaluate_payload(
                {"tool_name": "Agent", "tool_input": {"subagent_type": "general-purpose", "prompt": "do it locally"}},
                ledger_path=dot_ledger,
                now=T0,
            )
            self.assertEqual(bypass_local["decision"], "block")
            self.assertIn("dot_text", bypass_local["reason"])

    def test_claude_2_quota_exhaustion_and_route_mismatch_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "usage-cache.json"
            paseo_dir = Path(tmp) / "paseo_agents"
            proj_dir = paseo_dir / "E-PROJECTS-demo"
            proj_dir.mkdir(parents=True)
            # claude-lab's five_hour declares a 25% launch floor; claude-magpie is not configured, so only a
            # used-up window refuses it (no account name implies a floor).
            for account_name, choice_name, agent_uuid, used in (
                ("claude-lab", "lab_sonnet55_high", "11111111-1111-4111-8111-111111111111", 85.0),
                ("claude-magpie", "sonnet55_high_claude_magpie", "22222222-2222-4222-8222-222222222222", 100.0),
            ):
                cache.write_text(
                    json.dumps(
                        {
                            "providers": {
                                account_name: {
                                    "available": True,
                                    "stale": False,
                                    "fetched_at": (T0 - dt.timedelta(minutes=1)).isoformat(),
                                    "windows": [
                                        {
                                            "id": "five_hour",
                                            "used_pct": used,
                                            "resets_at": (T0 + dt.timedelta(hours=2)).isoformat(),
                                        }
                                    ],
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                (proj_dir / f"{agent_uuid}.json").write_text(
                    json.dumps({"id": agent_uuid, "provider": account_name, "labels": {"router.parent_id": "parent"}}),
                    encoding="utf-8",
                )
                claude_ledger = self.write_ledger(
                    tmp,
                    [self.route(account_name, "claude-sonnet-5-5", "high", account=account_name, choice=choice_name)],
                )
                # 1. Blocks new secondary Claude create_agent / CLI when five_hour is below its floor even if ledger matches
                res_create = gate.evaluate_payload(
                    {"tool_name": "mcp__paseo__create_agent", "tool_input": {"provider": account_name, "prompt": "work"}},
                    ledger_path=claude_ledger,
                    usage_cache_path=cache,
                    paseo_agents_dir=paseo_dir,
                    now=T0,
                )
                self.assertEqual(res_create["decision"], "block")
                self.assertIn("five_hour", res_create["reason"])

                res_cli = gate.evaluate_payload(
                    {"tool_name": "Bash", "tool_input": {"command": f"CLAUDE_CONFIG_DIR=~/.{account_name} paseo agent create --prompt 'work'"}},
                    ledger_path=claude_ledger,
                    usage_cache_path=cache,
                    paseo_agents_dir=paseo_dir,
                    now=T0,
                )
                self.assertEqual(res_cli["decision"], "block")
                self.assertIn(account_name, res_cli["reason"])

                # 2. Blocks the parent's run_turn / send_message to its own existing secondary Claude worker the same way;
                # the same message from a session that does not own it is communication and passes.
                turn_payload = {
                    "toolName": "codemode",
                    "input": {"code": f'await paseo.run_turn({{agentId: "{agent_uuid}", prompt: "continue"}})'},
                }
                res_turn = gate.evaluate_payload(
                    dict(turn_payload, parent_agent_id="parent"),
                    ledger_path=claude_ledger,
                    usage_cache_path=cache,
                    paseo_agents_dir=paseo_dir,
                    now=T0,
                )
                self.assertTrue(res_turn["block"])
                self.assertIn("five_hour", res_turn["reason"])
                for caller in ("peer", None):
                    res_peer = gate.evaluate_payload(
                        dict(turn_payload, parent_agent_id=caller), ledger_path=claude_ledger, usage_cache_path=cache,
                        paseo_agents_dir=paseo_dir, now=T0)
                    self.assertFalse(res_peer.get("block"), caller)

                # 3. Blocks secondary Claude create_agent when quota is healthy (20% used) but recent ledger picked pi_astra_high
                cache.write_text(
                    json.dumps(
                        {
                            "providers": {
                                account_name: {
                                    "available": True,
                                    "stale": False,
                                    "fetched_at": (T0 - dt.timedelta(minutes=1)).isoformat(),
                                    "windows": [
                                        {
                                            "id": "five_hour",
                                            "used_pct": 20.0,
                                            "resets_at": (T0 + dt.timedelta(hours=2)).isoformat(),
                                        }
                                    ],
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                other_ledger = self.write_ledger(
                    tmp,
                    [self.route("pi", "gpt-6-astra", "high", stage="record", constraint="user_override",
                                bundle_id="pi_astra_high")],
                )
                res_mismatch = gate.evaluate_payload(
                    {
                        "tool_name": "Agent",
                        "tool_input": {"subagent_type": "general-purpose", "prompt": "work", "model": "sonnet"},
                    },
                    ledger_path=other_ledger,
                    usage_cache_path=cache,
                    paseo_agents_dir=paseo_dir,
                    now=T0,
                )
                self.assertEqual(res_mismatch["decision"], "block")
            self.assertIn("pi_astra_high", res_mismatch["reason"])

    def test_current_paseo_create_contract_checks_quota_and_normalizes_pair(self):
        self.exhausted_cache()
        for provider in ("claude-lab", "claude-lab/claude-sonnet-5-5"):
            payload = {"tool_name": "mcp__paseo__create_agent", "tool_input": {
                "provider": provider, "initialPrompt": "work", "settings": {"thinkingOptionId": "high"}}}
            result = gate.evaluate_payload(payload, now=T0)
            self.assertIn("five_hour", result["reason"])
        self.assertEqual(gate.normalize_launch({"provider": "codex/gpt-6-astra",
                                               "settings": {"thinkingOptionId": "high"}}),
                         {"provider": "codex", "model": "gpt-6-astra", "effort": "high"})

    def test_current_send_agent_prompt_direct_and_native_wrappers(self):
        # A turn to the caller's own idle worker is dispatch and meets the quota check through every wrapper; the
        # same turn from a caller that does not own the agent, or with no caller id, is communication and passes.
        self.exhausted_cache()
        agent_id = self.existing_agent(owner="parent")
        code = f'await tools.mcp__paseo__send_agent_prompt({{agentId: "{agent_id}", prompt: "continue"}})'
        payloads = [{"tool_name": name, "tool_input": {"agentId": agent_id, "prompt": "continue"}}
                    for name in ("send_agent_prompt", "mcp__paseo__send_agent_prompt", "run_turn", "send_message")] + [
            {"tool_name": "exec", "tool_input": code},
            {"toolName": "codemode", "input": {"code": code}},
            {"toolCall": {"name": "execute_code", "args": {"code": code}}},
            {"tool_name": "exec_command", "tool_input": {"cmd": f'paseo agent send {agent_id} continue'}},
        ]
        for caller, owned in (("parent", True), ("peer", False), ("", False)):
            for payload in payloads:
                with self.subTest(caller=caller, payload=payload), mock.patch.dict(os.environ, {"PASEO_AGENT_ID": caller}):
                    result = gate.evaluate_payload(payload, now=T0)
                    if owned:
                        self.assertIn("five_hour", result["reason"])
                    else:
                        self.assertNotIn("reason", result)
        self.exhausted_cache(stale=True)
        with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": "parent"}):
            self.assertIn("five_hour", gate.evaluate_payload({"tool_name": "send_agent_prompt", "tool_input": {
                "agentId": agent_id, "prompt": "continue"}}, now=T0)["reason"])

    def test_pi_launch_namespace_matches_canonical_ledger_on_schedule_and_host_launch(self):
        # Actual ledger and handover records retain canonical model, not bundle.launch_model.
        for prov, canonical_model, launch_model in (
            ("pi", "gpt-6-astra", "openai-codex/gpt-6-astra"),
            ("opencode", "deepseek-v4.1-flash-free", "opencode/deepseek-v4.1-flash-free"),
        ):
            plan_ledger = self.write_ledger(str(self.state), [self.route(prov, canonical_model, "high", session_ids=["s1"])])
            for idx, args in enumerate((
                {"provider": f"{prov}/{launch_model}"},
                {"provider": prov, "model": launch_model},
            )):
                args["settings"] = {"thinkingOptionId": "high"}
                host_launch = {"session_id": "s1", "tool_use_id": f"pi-{prov}-{idx}",
                               "tool_name": "delegate_task",
                               "tool_input": {"task": "work", "target": {"driverKind": prov, "model": launch_model}}}
                schedule = {"tool_name": "mcp__paseo__create_schedule", "session_id": "s1",
                            "tool_input": dict(args, prompt="work", cron="7 * * * *")}
                # Plan row authorizes schedule only, not native delegate_task; native handover authorizes delegate_task.
                self.assertEqual(gate.evaluate_payload(schedule, ledger_path=plan_ledger, now=T0), {})
                self.assertEqual(gate.evaluate_payload(host_launch, ledger_path=plan_ledger, now=T0)["decision"], "block")
                self.handover(prov, canonical_model, "high", surface="t3", parent_id="s1", session_ids=["s1"],
                              task_id=f"t-{prov}-{idx}", now=T0)
                self.assertEqual(gate.evaluate_payload(host_launch, ledger_path=plan_ledger, now=T0), {})

    def test_pi_namespace_normalization_preserves_owner_model_and_effort_checks(self):
        ledger = self.write_ledger(str(self.state), [self.route("pi", stage="record", constraint="user_override")])
        for provider, effort, diagnostic in (
            ("pi/openai-codex/gpt-6.1-sol", "high", "model"),
            ("pi/openai-codex/gpt-6-astra", "max", "effort"),
            ("codex/openai-codex/gpt-6-astra", "high", "provider"),
            ("pi/other-backend/gpt-6-astra", "high", "model"),
        ):
            args = {"provider": provider, "settings": {"thinkingOptionId": effort}}
            result = gate.evaluate_payload({"tool_name": "delegate_task", "tool_input": args}, ledger_path=ledger, now=T0)
            self.assertIn(diagnostic, result["reason"])

    def test_antigravity_high_model_accepts_omitted_thinking_option(self):
        for provider in ("antigravity", "antigravity-resilient"):
            for model in ("gemini-4-argon-high", "gemini-3.8-flash-high"):
                plan_ledger = self.write_ledger(str(self.state), [self.route(provider, model, "high", session_ids=["s1"])])
                args = {"provider": provider + "/" + model, "settings": {"modeId": "full-access"}}
                self.assertEqual(gate.evaluate_payload({"tool_name": "mcp__paseo__create_schedule", "session_id": "s1",
                    "tool_input": dict(args, prompt="work", cron="7 * * * *")},
                    ledger_path=plan_ledger, now=T0), {})
                self.handover(provider, model, "high", surface="t3", parent_id="s1", session_ids=["s1"],
                              task_id=f"t-{provider}-{model}", now=T0)
                self.assertEqual(gate.evaluate_payload(
                    {"session_id": "s1", "tool_name": "delegate_task",
                     "tool_input": {"task": "work", "target": {"driverKind": "antigravity", "model": model}}},
                    ledger_path=plan_ledger, now=T0), {})

    def test_quota_never_blocks_inspection_even_with_provider_or_agent_id(self):
        self.exhausted_cache()
        agent_id = self.existing_agent()
        for name in ("list_models", "inspect_provider", "get_agent_status", "mcp__paseo__list_models", "update_agent"):
            self.assertEqual(gate.evaluate_payload({"tool_name": name, "tool_input": {
                "provider": "claude-lab/claude-sonnet-5-5", "agentId": agent_id}}, now=T0), {})
        code = f'await paseo.get_agent_status({{agentId:"{agent_id}"}}); await paseo.list_models({{provider:"claude-lab"}})'
        self.assertEqual(gate.evaluate_payload({"toolName": "codemode", "input": {"code": code}}, now=T0), {"block": False})

    def test_full_window_blocks_until_reset_unless_reading_is_from_a_previous_cycle(self):
        # The fixture's five_hour window resets at T0+2h, so its current cycle began at T0-3h.
        week = {"id": "seven_day", "used_pct": 100, "resets_at": (T0 + dt.timedelta(days=1)).isoformat()}
        for changes, blocked in (
                ({}, True),
                ({"stale": True}, True),
                ({"fetched_at": (T0 - dt.timedelta(hours=1)).isoformat()}, True),
                ({"stale": True, "fetched_at": (T0 - dt.timedelta(hours=1)).isoformat()}, True),
                ({"windows": [week], "fetched_at": (T0 - dt.timedelta(days=5)).isoformat()}, True),
                ({"fetched_at": (T0 - dt.timedelta(hours=4)).isoformat()}, False),
                ({"windows": [week], "fetched_at": (T0 - dt.timedelta(days=7)).isoformat()}, False),
                ({"windows": [{"id": "five_hour", "used_pct": 100, "resets_at": T0.isoformat()}]}, False),
                ({"fetched_at": None}, False), ({"fetched_at": "bad"}, False),
                ({"fetched_at": (T0 + dt.timedelta(hours=1)).isoformat()}, False),
                ({"available": False}, False)):
            with self.subTest(changes=changes):
                cache = self.exhausted_cache(**changes)
                self.assertEqual(gate.check_account_quota_exhausted("claude-lab", cache, T0) is not None, blocked)
        # Dict-shaped window specs with custom launch_min_remaining_pct apply across non-Claude accounts too.
        custom_accounts = {"families": {"rows": []}, "provider_names": {"defaults": {}, "rows": {}},
                           "accounts": {"team-proxy": {"windows": {"spend": {"hours": 168, "launch_min_remaining_pct": 15}}}}}
        cache = self.state / "usage.json"
        for used_pct, blocked in ((86.0, True), (80.0, False)):
            cache.write_text(json.dumps({"providers": {"team-proxy": {
                "available": True, "stale": False, "fetched_at": T0.isoformat(),
                "windows": [{"id": "spend", "used_pct": used_pct, "resets_at": (T0 + dt.timedelta(days=2)).isoformat()}],
            }}}), encoding="utf-8")
            self.assertEqual(gate.check_account_quota_exhausted("team-proxy", cache, T0, accounts=custom_accounts) is not None, blocked)

    def write_settings(self, allowed=None, accounts=None):
        policy = self.state / "host-policy.json"
        if allowed is None:
            policy.unlink(missing_ok=True)
        else:
            policy.write_text(json.dumps({"allowed_accounts": allowed}), encoding="utf-8")
        (self.state / "accounts-override.json").write_text(
            json.dumps({"schema_version": 2, "accounts": accounts or {}}), encoding="utf-8"
        )

    def test_preferences_refuse_launches_only_and_facts_refuse_own_turns_only(self):
        # Gate contract (parent-address-gate.md): an unchecked model refuses a launch, host policy refuses neither a launch
        # nor a turn, a turn to an agent the caller does not own is communication and is not checked, and the caller's own
        # idle worker is refused only for facts (confirmed quota exhaustion here; window readings and compaction elsewhere).
        from contextlib import contextmanager, nullcontext

        names = ("own", "peer", "free")
        ids = {name: f"{i:08d}-0000-4000-8000-000000000000" for i, name in enumerate(names, 1)}
        directory = self.state / "agents" / "project"
        directory.mkdir(parents=True, exist_ok=True)
        sonnet = {"provider": "claude-lab", "config": {"model": "claude-sonnet-5-5", "thinkingOptionId": "high"}}
        for name, doc in zip(names, (
                {**sonnet, "labels": {"router.account": "claude-lab", "router.bundle_id": "lab_sonnet55_high",
                                      "router.parent_id": "actual-parent"}},
                {**sonnet, "labels": {"router.account": "claude-lab", "router.bundle_id": "lab_sonnet55_high",
                                      "router.parent_id": "another-parent"}},
                {"provider": "claude-lab/claude-sonnet-5-5"})):
            (directory / (ids[name] + ".json")).write_text(json.dumps(doc), encoding="utf-8")
        ledger = self.write_ledger(str(self.state), [self.route("claude-lab", "claude-sonnet-5-5", "high"),
                                                     self.route("claude-lab", "claude-opus-5-5", "high")])

        def turns(agent, caller):
            code = f'await tools.mcp__paseo__send_agent_prompt({{agentId: "{ids[agent]}", prompt: "continue"}})'
            return [{"tool_name": tool, "tool_input": {"agentId": ids[agent], "prompt": "continue"},
                     "parent_agent_id": caller} for tool in ("send_agent_prompt", "mcp__paseo__run_turn", "send_message")] + [
                {"toolName": "codemode", "input": {"code": code}, "parent_agent_id": caller}]

        def launch(model):
            return {"tool_name": "create_agent", "tool_input": {
                "provider": f"claude-lab/claude-{model}-5-5", "settings": {"thinkingOptionId": "high"},
                "labels": {"router.account": "claude-lab", "router.bundle_id": f"lab_{model}55_high",
                           "router.dispatch_id": f"d-{model}"}}}

        @contextmanager
        def latched_exhaustion():
            exhaustion = gate.router_module("quota_exhaustion")
            with mock.patch.dict(os.environ, {"ROUTER_STATE_DIR": str(self.state)}):
                self.assertTrue(exhaustion.record("claude-lab", "Individual quota reached", at=T0.isoformat()))
                try:
                    yield
                finally:
                    exhaustion.clear("claude-lab")

        communication = turns("peer", "actual-parent") + turns("free", "actual-parent") + turns("own", "another-parent") + turns("own", None)
        own_turns = turns("own", "actual-parent")
        everyone = ["claude", "codex", "antigravity", "dot"]
        unchecked = {"claude-lab": {"bundles": {"lab_sonnet55_high": {"disabled": True},
                                                "lab_sonnet55_medium": {"disabled": True},
                                                "lab_opus55_high": {"disabled": False}}}}
        with mock.patch.object(gate.router_module("dispatch_registry"), "claim", return_value=None):
            # (case, settings, fact context, cause in the block reason, blocked calls, passed calls)
            for case, settings, fact_ctx, cause, blocked, passed in (
                    ("out of automatic routing", {"allowed": everyone}, nullcontext, None, (),
                     own_turns + communication + [launch("sonnet")]),
                    ("unchecked model", {"accounts": unchecked}, nullcontext, "档位 lab_sonnet55_high",
                     [launch("sonnet")], own_turns + communication + [launch("opus")]),
                    ("confirmed quota exhaustion", {}, latched_exhaustion,
                     "confirmed quota failure is recorded for every window of claude-lab",
                     own_turns + [launch("opus")], communication),
            ):
                self.write_settings(**settings)
                with fact_ctx():
                    for payload in blocked:
                        with self.subTest(case=case, blocked=payload):
                            result = gate.evaluate_payload(payload, ledger_path=ledger, now=T0)
                            self.assertIn(cause, result["reason"])
                            self.assertIn("roll --brief", result["reason"])
                    for payload in passed:
                        with self.subTest(case=case, passed=payload):
                            self.assertNotIn("reason", gate.evaluate_payload(payload, ledger_path=ledger, now=T0))

    def test_a_launch_on_an_unchecked_model_passes_only_on_the_users_naming(self):
        # Gate contract (parent-address-gate.md; amendment 2): the checkbox is a preference, so a launch whose authority
        # is a `user_override` decision (the user named the route) passes on an unchecked model; a launch on any other
        # authority (a plan, a project rule, an explicit resume record) is refused as before.
        def bound(model):
            return {"tool_name": "create_agent", "tool_input": {
                "provider": f"claude-lab/claude-{model}-5-5", "settings": {"thinkingOptionId": "high"},
                "labels": {"router.account": "claude-lab", "router.bundle_id": f"lab_{model}55_high",
                           "router.dispatch_id": f"d-{model}"}}}

        self.write_settings(accounts={"claude-lab": {"bundles": {f"lab_sonnet55_{e}": {"disabled": True}
                                                                 for e in ("high", "medium")}}})
        ledger = self.write_ledger(str(self.state), [self.route("claude-lab", "claude-sonnet-5-5", "high")])
        registry = gate.router_module("dispatch_registry")
        with mock.patch.object(registry, "claim", return_value=None):
            for constraint, refused in (("user_override", False), ("explicit_resume", True),
                                        ("hard_constraint", True), (None, True)):
                with self.subTest(launch="bound", constraint=constraint), \
                        mock.patch.object(registry, "dispatch_constraint", return_value=constraint):
                    result = gate.evaluate_payload(bound("sonnet"), ledger_path=ledger, now=T0)
                    self.assertEqual("reason" in result, refused, result)
                    if refused:
                        self.assertIn("档位 lab_sonnet55_high", result["reason"])
                        self.assertIn("user_override", result["reason"])

        # A native launch names a model only; it resolves to an account when one account offers it.
        native = {"tool_name": "Agent", "tool_input": {"subagent_type": "general-purpose", "prompt": "work",
                                                         "model": "sonnet", "effort": "high"}}
        real_accounts = gate.load_dispatch_accounts

        def claude_only():
            accounts = real_accounts()
            accounts["accounts"] = {k: v for k, v in accounts["accounts"].items()
                                    if k == "claude" or not any("sonnet" in str(b.get("model"))
                                                                for b in (v.get("bundles") or {}).values())}
            return accounts

        self.write_settings(accounts={"claude": {"bundles": {
            f"sonnet55_{e}": {"disabled": True} for e in ("low", "medium", "high", "xhigh", "max")}}})
        missing = self.state / "missing.jsonl"
        with mock.patch.object(gate, "load_dispatch_accounts", side_effect=claude_only):
            # Both standalone `record` rows and `roll` native hand-overs pass on unchecked models only with `user_override`.
            for constraint, refused in (("user_override", False), ("hard_constraint", True)):
                row = self.route("claude", "claude-sonnet-5-5", "high", stage="record", constraint=constraint)
                ledger = self.write_ledger(str(self.state), [row])
                with self.subTest(launch="native-record", constraint=constraint):
                    result = gate.evaluate_payload(native, ledger_path=ledger, now=T0)
                    self.assertEqual("reason" in result, refused, result)
                    if refused:
                        self.assertIn("在模型管理中未勾选", result["reason"])
            for constraint, refused in (("user_override", False), (None, True)):
                claim = self.handover("claude", "claude-sonnet-5-5", "high", parent_id="sess-u", session_ids=["sess-u"],
                                      task_id=f"tu-{constraint}", constraint=constraint, now=T0)
                with self.subTest(launch="native-handover", constraint=constraint):
                    result = gate.evaluate_payload(dict(native, session_id="sess-u"), ledger_path=missing, now=T0)
                    self.assertEqual("reason" in result, refused, result)
                    if refused:
                        self.assertIn("在模型管理中未勾选", result["reason"])
                        self.assertIsNone(registry.inspect(claim["dispatch_id"]).get("launched_at"))

    def test_a_native_launch_resolves_to_the_sessions_own_account_only_when_discovery_records_it(self):
        # Gate contract (parent-address-gate.md): a native sub-agent runs on the calling session's own account. When
        # discovery records which account that is (one account of the family whose home is the session's directory),
        # a launch naming a model resolves to it for the checkbox rule only: only a user_override row passes an
        # unchecked model, and the refusal names the model asked for, the account and the matched record. When it is
        # not recorded, or the launch names no model, nothing is guessed. Quota, window and floor checks resolve the
        # launch as it is written, so the own account's exhausted window does not refuse it.
        native = {"tool_name": "Agent", "tool_input": {"subagent_type": "general-purpose", "prompt": "work",
                                                         "model": "opus", "effort": "high"}}
        proj_agents = self.state / "proj-own-acct"
        (proj_agents / ".claude" / "agents").mkdir(parents=True, exist_ok=True)
        (proj_agents / ".claude" / "agents" / "quick-find.md").write_text(
            "---\nname: quick-find\nmodel: haiku\neffort: high\n---\nFind\n", encoding="utf-8"
        )
        custom_no_raw_model = {"cwd": str(proj_agents), "tool_name": "Agent",
                               "tool_input": {"subagent_type": "quick-find", "prompt": "work"}}
        own_home, lab_home = self.state / ".claude-own", self.state / ".claude-lab"
        real_accounts = gate.load_dispatch_accounts

        def discovered(rows):
            def load():
                accounts = real_accounts()
                accounts["_discovery"] = {"accounts": rows}
                return accounts
            return load
        opus = [b for b, row in real_accounts()["accounts"]["claude"]["bundles"].items()
                if str(row.get("model")).startswith("claude-opus-")]
        self.write_settings(accounts={"claude": {"bundles": {b: {"disabled": True} for b in opus}}})
        recorded = {"claude": {"family": "claude", "home": str(own_home)},
                    "claude-lab": {"family": "claude", "home": str(lab_home)}}
        ambiguous = {"claude": {"family": "claude", "home": str(own_home)},
                     "claude-lab": {"family": "claude", "home": str(own_home)}}
        cases = (   # (discovery rows, row constraint, refused)
            (recorded, "user_override", False), (recorded, "hard_constraint", True), (recorded, "dot_closed_loop", True),
            (recorded, None, True),
            ({}, None, False), (ambiguous, None, False),    # not recorded / not one account: unresolved, as before
        )
        missing = self.state / "missing.jsonl"
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(own_home)}):
            for idx, (rows, constraint, refused) in enumerate(cases):
                if constraint:
                    ledger = self.write_ledger(
                        str(self.state),
                        [self.route("claude", "claude-opus-5-5", "high", bundle_id="opus55_high",
                                    stage="record", constraint=constraint)],
                    )
                    sess = {}
                else:
                    ledger = missing
                    sess = {"session_id": f"own-sess-{idx}"}
                with self.subTest(rows=sorted(rows), ambiguous=rows is ambiguous, constraint=constraint), \
                        mock.patch.object(gate, "load_dispatch_accounts", side_effect=discovered(rows)):
                    if not constraint:
                        self.handover("claude", "claude-opus-5-5", "high", bundle_id="opus55_high",
                                      parent_id=sess["session_id"], session_ids=[sess["session_id"]],
                                      task_id=f"own-m-{idx}", now=T0)
                    result = gate.evaluate_payload(dict(native, **sess), ledger_path=ledger, now=T0)
                    self.assertEqual("reason" in result, refused, result)
                    if refused:
                        self.assertIn("在模型管理中未勾选", result["reason"])
                        for named in ("模型 opus", "账号 claude ", "opus55_high"):
                            self.assertIn(named, result["reason"])
                        self.assertNotIn("?", result["reason"])

            # A call whose raw `tool_input` omits `model` (resolved from `.claude/agents/quick-find.md` frontmatter)
            # is not resolved to `own` for `launch_target` even when every catalogued bundle of `claude` is unchecked.
            every = list(real_accounts()["accounts"]["claude"]["bundles"])
            self.write_settings(accounts={"claude": {"bundles": {b: {"disabled": True} for b in every}}})
            self.handover("claude", "claude-haiku-5-5", "high", bundle_id=None,
                          parent_id="own-all-off", session_ids=["own-all-off"], task_id="t-all-off", now=T0)
            with self.subTest(launch="custom frontmatter haiku, every model unchecked"), \
                    mock.patch.object(gate, "load_dispatch_accounts", side_effect=discovered(recorded)):
                self.assertNotIn("reason", gate.evaluate_payload(
                    dict(custom_no_raw_model, session_id="own-all-off"), ledger_path=missing, now=T0))

            # The own account's window is exhausted and its models are checked: the native launch is not resolved to
            # it for that check, exactly as before the own-account resolution existed.
            self.write_settings(accounts={})
            cache = self.state / "usage-exhausted.json"
            cache.write_text(json.dumps({"providers": {"claude": {
                "available": True, "stale": False, "fetched_at": T0.isoformat(),
                "windows": [{"id": w, "used_pct": 100, "resets_at": (T0 + dt.timedelta(hours=3)).isoformat()}
                            for w in ("five_hour", "seven_day")]}}}), encoding="utf-8")
            with mock.patch.object(gate, "load_dispatch_accounts", side_effect=discovered(recorded)):
                for q_idx, (payload, q_model, q_bid) in enumerate((
                    (native, "claude-opus-5-5", "opus55_high"),
                    (custom_no_raw_model, "claude-haiku-5-5", None),
                )):
                    with self.subTest(quota=payload["tool_input"].get("model")):
                        sid = f"own-quota-{q_idx}"
                        self.handover("claude", q_model, "high", bundle_id=q_bid,
                                      parent_id=sid, session_ids=[sid], task_id=f"tq-{q_idx}", now=T0)
                        self.assertNotIn("reason", gate.evaluate_payload(
                            dict(payload, session_id=sid), ledger_path=missing, now=T0, usage_cache_path=cache))

    def test_a_caller_known_only_by_its_host_session_owns_the_worker_whose_parent_record_runs_that_session(self):
        # Gate contract (parent-address-gate.md): ownership is proven when every parent label names the caller, by its
        # Paseo agent ID, by the host session ID its hook payload carries, or as a Paseo agent whose record runs that
        # session (`persistence.sessionId`). Without such a proof a turn is communication and is not checked.
        directory = self.state / "agents" / "project"
        directory.mkdir(parents=True, exist_ok=True)
        sonnet = {"provider": "claude-lab", "config": {"model": "claude-sonnet-5-5", "thinkingOptionId": "high"}}
        records = {
            "parent-agent": {"provider": "claude", "persistence": {"sessionId": "sess-parent"}},
            "worker-of-parent": {**sonnet, "persistence": {"sessionId": "sess-worker"},
                                 "labels": {"router.account": "claude-lab", "router.parent_id": "parent-agent"}},
            "worker-of-session": {**sonnet, "labels": {"router.account": "claude-lab", "router.parent_id": "sess-direct"}},
            "worker-mixed": {**sonnet, "labels": {"router.account": "claude-lab", "router.parent_id": "parent-agent",
                                                  "paseo.parent-agent-id": "someone-else"}},
            "self-labelled": {**sonnet, "persistence": {"sessionId": "sess-self"},
                              "labels": {"router.account": "claude-lab", "router.parent_id": "self-labelled"}},
        }
        for agent_id, doc in records.items():
            (directory / (agent_id + ".json")).write_text(json.dumps(doc), encoding="utf-8")
        ledger = self.write_ledger(str(self.state), [self.route("claude-lab", "claude-sonnet-5-5", "high")])
        exhaustion = gate.router_module("quota_exhaustion")

        cases = (   # (target, hook session id, owned: the fact refuses the turn)
            ("worker-of-parent", "sess-parent", True),      # the parent's record runs the caller's session
            ("worker-of-session", "sess-direct", True),     # the label names the caller's session itself
            ("worker-of-parent", "sess-other", False),      # another session: communication
            ("worker-of-parent", None, False),              # nothing names the caller
            ("worker-mixed", "sess-parent", False),         # conflicting parent labels prove nothing
            ("worker-of-parent", "sess-worker", False),     # the worker messaging itself is not its own parent
            ("self-labelled", "sess-self", False),          # a record naming itself as parent: the caller is the target
        )
        with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": "", "ROUTER_STATE_DIR": str(self.state)}):
            self.assertTrue(exhaustion.record("claude-lab", "Individual quota reached", at=T0.isoformat()))
            try:
                for target, session, owned in cases:
                    payload = {"tool_name": "mcp__paseo__send_agent_prompt",
                               "tool_input": {"agentId": target, "prompt": "continue"},
                               **({"session_id": session} if session else {})}
                    with self.subTest(target=target, session=session):
                        result = gate.evaluate_payload(payload, ledger_path=ledger, now=T0)
                        self.assertEqual("reason" in result, owned, result)
                        if owned:
                            self.assertIn("confirmed quota failure is recorded for every window of claude-lab", result["reason"])
            finally:
                exhaustion.clear("claude-lab")

    def test_a_native_launch_on_a_refused_record_is_blocked_and_a_named_one_passes(self):
        # Gate contract (parent-address-gate.md): `record` writes `host_policy_violation` only on constraints other
        # than `user_override` when the account is out of automatic routing; `check_route_authority` decides on that
        # `host_policy_violation` fact alone, so a project rule carrying it is blocked and a `user_override` row from
        # the writer (which never carries it) is launch authority.
        launch = {"tool_name": "Agent", "tool_input": {"subagent_type": "general-purpose", "prompt": "work",
                                                         "model": "claude-sonnet-5-5"}}
        self.write_settings(allowed=["codex"])  # claude is out of automatic routing on this host
        route_judge = gate.router_module("route_judge")
        decision_ledger = gate.router_module("decision_ledger")
        env_vars = gate.router_module("env_vars")
        ledger = self.state / "ledger.jsonl"
        for constraint, refused in (("hard_constraint", True), ("dot_closed_loop", True), ("user_override", False)):
            ledger.unlink(missing_ok=True)
            args = route_judge.build_parser().parse_args([
                "record", "--constraint", constraint, "--basis", "user or project rule quote",
                "--provider", "claude", "--account", "claude", "--model", "claude-sonnet-5-5", "--effort", "high",
                "--no-score", "--hours", "0.5",
            ])
            with mock.patch.object(env_vars, "user_env", side_effect=os.environ.get):
                route_judge.record_constraint(args)
            written = decision_ledger.read_ledger(ledger)[-1]
            self.assertEqual("host_policy_violation" in written, refused, written)
            with self.subTest(constraint=constraint):
                result = gate.evaluate_payload(launch, ledger_path=ledger, now=dt.datetime.now(UTC))
                self.assertEqual("reason" in result, refused, result)
                if refused:
                    self.assertIn("out of automatic routing", result["reason"])
                    self.assertIn("--constraint user_override", result["reason"])
        # Fact-only check: `check_route_authority` does not exempt `constraint == "user_override"` if a violation dict is present.
        synthetic = dict(self.route("claude", "claude-sonnet-5-5", "high", stage="record", constraint="user_override"),
                         host_policy_violation={"reason": "account claude is out of automatic routing on this host"})
        self.assertIsNotNone(gate.check_route_authority(synthetic, "tool:Agent"))

    def test_running_owner_control_exempts_disabled_and_context_rotation_gates(self):
        wid = self.existing_agent()
        agent_file = self.state / 'agents' / 'project' / (wid + '.json')
        owner = {'router.parent_id': 'parent'}
        cases = (
            ('router owner', 'parent', owner, {'lastStatus': 'running'}, False),
            ('paseo owner', 'parent', {'paseo.parent-agent-id': 'parent'}, {'lastStatus': 'running'}, False),
            ('both parents', 'parent', {**owner, 'paseo.parent-agent-id': 'parent'}, {'lastStatus': 'running'}, False),
            ('current status', 'parent', owner, {'status': 'running'}, False),
            ('idle', 'parent', owner, {'lastStatus': 'idle'}, True),
            ('permission wait', 'parent', owner, {'lastStatus': 'waiting'}, True),
            ('unknown status', 'parent', owner, {}, True),
            ('current idle wins', 'parent', owner, {'lastStatus': 'running', 'status': 'idle'}, True),
            ('current unknown wins', 'parent', owner, {'lastStatus': 'running', 'status': None}, True),
            ('activeTurn alone', 'parent', owner, {'activeTurn': {'turnId': 't'}}, True),
            ('idle activeTurn', 'parent', owner, {'status': 'idle', 'activeTurn': {'turnId': 't'}}, True),
            ('nonowner', 'peer', owner, {'lastStatus': 'running'}, True),
            ('unknown caller', '', owner, {'lastStatus': 'running'}, True),
            ('unknown owner', 'parent', {}, {'lastStatus': 'running'}, True),
            ('conflicting parents', 'parent', {**owner, 'paseo.parent-agent-id': 'peer'}, {'lastStatus': 'running'}, True),
            ('self', wid, {'router.parent_id': wid}, {'lastStatus': 'running'}, True),
        )
        worker_sessions = gate.router_module('worker_sessions')
        # A preference switch (host policy, an unchecked model) refuses no turn. A fact (the rotation limit, a used-up
        # window) refuses only the caller's own idle or unknown-state worker; a message to an agent the caller does not
        # own, or whose ownership cannot be proven, is communication.
        communication = {'nonowner', 'unknown caller', 'unknown owner', 'conflicting parents', 'self'}
        for settings, tokens, cause in (
                ({'allowed': []}, 0, None),
                ({'accounts': {'claude-lab': {'bundles': {'lab_sonnet55_high': {'disabled': True}}}}}, 0, None),
                ({}, 2_000_000, 'historical_context_tokens=2,000,000/2,000,000'),
                ({}, 0, 'five_hour')):
            self.write_settings(**settings)
            if cause == 'five_hour':
                self.exhausted_cache()
            else:
                (self.state / 'usage.json').unlink(missing_ok=True)
            with mock.patch.object(worker_sessions, 'session_state', return_value={'context_tokens': tokens}):
                for label, caller, labels, status, blocked in cases:
                    doc = {'provider': 'claude-lab/claude-sonnet-5-5',
                           'labels': {**labels, 'router.account': 'claude-lab', 'router.bundle_id': 'lab_sonnet55_high'},
                           **status}
                    agent_file.write_text(json.dumps(doc), encoding='utf-8')
                    # Wrap-up text is deliberately identical for idle and running targets.
                    args = {'agentId': wid, 'prompt': 'commit, write handoff, report', 'notifyOnFinish': True}
                    payloads = [
                        {'tool_name': tool, 'tool_input': args}
                        for tool in ('send_agent_prompt', 'mcp__paseo__run_turn', 'send_message')]
                    payloads.extend((
                        {'toolName': 'send_agent_prompt', 'input': args},
                        {'toolCall': {'name': 'send_agent_prompt', 'args': args}},
                        {'tool_name': 'exec', 'tool_input': 'await tools.mcp__paseo__send_agent_prompt(' + json.dumps(args) + ')'},
                        {'toolName': 'codemode', 'input': {'code':
                            f'const id = "{wid}"; await paseo.send_agent_prompt({{agentId: id, prompt: "wrap up"}})'}},
                    ))
                    script = ("import subprocess; paseo = r'C:\\Program Files\\Paseo\\paseo.cmd'; "
                              f"target = '{wid}'; args = [paseo, 'agent', 'send', target, '--prompt', 'wrap up']; "
                              "subprocess.run(args)")
                    commands = (
                        f'paseo agent send {wid} --prompt "wrap up"',
                        f'paseo send {wid} "wrap up"',
                        f'& "C:\\Program Files\\Paseo\\paseo.cmd" agent send {wid} --prompt "wrap up"',
                        f'python -X utf8 -c "{script}"',
                        script,
                        f"python -c \"import subprocess; subprocess.run(['paseo', 'send', '{wid}', 'wrap up'])\"",
                    )
                    payloads.extend({'toolCall': {'name': 'run_command', 'args': {'CommandLine': command}}}
                                    for command in commands)
                    payloads.append({'tool_name': 'exec', 'tool_input':
                        'await tools.exec_command(' + json.dumps({'cmd': commands[0]}) + ')'})
                    for payload in payloads:
                        with self.subTest(gate=cause, case=label, payload=payload), \
                                mock.patch.dict(os.environ, {'PASEO_AGENT_ID': caller}):
                            result = gate.evaluate_payload(payload, now=T0)
                            expected = bool(cause) and blocked and label not in communication
                            self.assertEqual('reason' in result, expected)
                            if expected:
                                self.assertIn(cause, result['reason'])
                                self.assertIn('cancel_agent', result['reason'])
                                self.assertIn('roll --brief', result['reason'])

    def test_cli_inspection_cancel_and_quoted_examples_are_not_continuations(self):
        wid = self.existing_agent()
        self.write_settings(allowed=[])
        commands = [f'paseo agent {verb} {wid}' for verb in ('status', 'activity', 'list', 'cancel', 'stop')]
        commands += [f'paseo agent send {wid} --help', f'echo "paseo agent send {wid}"',
                     f'paseo agent status {wid} --label "paseo send {wid}"',
                     f"python -c \"print('paseo agent send {wid}')\"",
                     f"python -c \"import subprocess; subprocess.run(['paseo', 'agent', 'status', '{wid}'])\""]
        for command in commands:
            with self.subTest(command=command):
                self.assertNotIn('reason', gate.evaluate_payload({
                    'tool_name': 'exec_command', 'tool_input': {'cmd': command}}, now=T0))

    def test_running_control_keeps_launch_quota_and_notification_checks(self):
        wid = self.existing_agent()
        labels = {'router.parent_id': 'parent', 'router.account': 'claude-lab', 'router.bundle_id': 'lab_sonnet55_high'}
        (self.state / 'agents' / 'project' / (wid + '.json')).write_text(json.dumps({
            'provider': 'claude-lab/claude-sonnet-5-5', 'labels': labels, 'lastStatus': 'running'}), encoding='utf-8')
        self.write_settings(allowed=[])
        with mock.patch.dict(os.environ, {'PASEO_AGENT_ID': 'parent'}):
            # Host policy is not a launch condition: the bound launch reaches its dispatch claim.
            result = gate.evaluate_payload({'tool_name': 'create_agent', 'tool_input': {
                'provider': 'claude-lab/claude-sonnet-5-5', 'labels': labels, 'lastStatus': 'running'}}, now=T0)
            self.assertIn('Bound dispatch rejected', result['reason'])
            self.write_settings()
            # Spoofing running state in a launch request cannot bypass quota.
            self.exhausted_cache()
            result = gate.evaluate_payload({'tool_name': 'create_agent', 'tool_input': {
                'provider': 'claude-lab/claude-sonnet-5-5', 'labels': labels, 'lastStatus': 'running'}}, now=T0)
            self.assertIn('five_hour', result['reason'])
            # notifyOnFinish: false is still blocked even on a running owned worker.
            result = gate.evaluate_payload({'tool_name': 'send_agent_prompt', 'tool_input': {
                'agentId': wid, 'prompt': 'wrap up', 'notifyOnFinish': False}}, now=T0)
            self.assertIn('notifyOnFinish: true or omit it', result['reason'])
            # Control message with completion notifications enabled skips quota on a running owned worker,
            # including when the account's confirmed-exhaustion latch is set.
            exhaustion = gate.router_module('quota_exhaustion')
            exhaustion.record('claude-lab', 'Individual quota reached', windows=['five_hour'])
            self.addCleanup(exhaustion.clear, 'claude-lab')
            result = gate.evaluate_payload({'tool_name': 'send_agent_prompt', 'tool_input': {
                'agentId': wid, 'prompt': 'wrap up', 'notifyOnFinish': True}}, now=T0)
            self.assertNotIn('reason', result)
            # The caller's own idle or unknown-state worker stays blocked by quota on the same message; another
            # caller's message to it is communication and passes.
            for caller, state, blocked in (
                ('parent', {'lastStatus': 'idle'}, True),
                ('parent', {}, True),
                ('other', {'lastStatus': 'running'}, False),
                ('other', {'lastStatus': 'idle'}, False),
            ):
                (self.state / 'agents' / 'project' / (wid + '.json')).write_text(json.dumps({
                    'provider': 'claude-lab/claude-sonnet-5-5', 'labels': labels, **state}), encoding='utf-8')
                with mock.patch.dict(os.environ, {'PASEO_AGENT_ID': caller}):
                    result = gate.evaluate_payload({'tool_name': 'send_agent_prompt', 'tool_input': {
                        'agentId': wid, 'prompt': 'wrap up', 'notifyOnFinish': True}}, now=T0)
                self.assertEqual('reason' in result, blocked, (caller, state))
                if blocked:
                    self.assertIn('cancel_agent', result['reason'])
                    self.assertIn('roll --brief', result['reason'])

    def quota_cache(self, rows):
        path = self.state / "usage.json"
        path.write_text(json.dumps({"providers": {name: {"available": True, "stale": False, "fetched_at": T0.isoformat(),
                                                         **row} for name, row in rows.items()}}), encoding="utf-8")
        return path

    @staticmethod
    def window(wid, used, hours=2.0, **extra):
        return {"id": wid, "used_pct": used, "resets_at": (T0 + dt.timedelta(hours=hours)).isoformat(), **extra}

    def owned_worker(self, account, bundle_id, provider, status="running"):
        directory = self.state / "agents" / "project"
        directory.mkdir(parents=True, exist_ok=True)
        agent_id = "33333333-3333-4333-8333-333333333333"
        doc = {
            "provider": provider,
            "labels": {
                "router.parent_id": "parent", "paseo.parent-agent-id": "parent",
                "router.account": account, "router.bundle_id": bundle_id,
            },
        }
        if status is not None:
            doc["lastStatus"] = status
        (directory / (agent_id + ".json")).write_text(json.dumps(doc), encoding="utf-8")
        return agent_id

    def test_antigravity_third_party_exhaustion_blocks_only_models_that_draw_it(self):
        # The 2026-11-04 reading: 3p-weekly used up, the Gemini pools have room. The fixture account is on Ultra 5x, whose
        # Claude / GPT track has no capacity in the data: a window inside the safety margin blocks whether or not its
        # capacity is known.
        self.quota_cache({"antigravity": {"windows": [
            self.window("gemini-weekly", 28.4, 67), self.window("gemini-5h", 7.3, 4),
            self.window("3p-weekly", 99.26, 7), self.window("3p-5h", 0, 5)]}})
        for bundle_id, blocked in (("argon_high", False), ("flash_high", False),
                                   ("agy_opus55_low", True), ("agy_sonnet55_high", True)):
            agent_id = self.owned_worker("antigravity", bundle_id, "antigravity-resilient")
            for caller, status, is_dispatch in (
                ("parent", "running", False),
                ("parent", "idle", True),
                ("parent", None, True),
                ("other", "running", False),
                ("other", "idle", False),
            ):
                self.owned_worker("antigravity", bundle_id, "antigravity-resilient", status)
                expect_blocked = is_dispatch and blocked
                with self.subTest(bundle=bundle_id, caller=caller, status=status), \
                        mock.patch.dict(os.environ, {"PASEO_AGENT_ID": caller}):
                    result = gate.evaluate_payload({"tool_name": "send_agent_prompt", "tool_input": {
                        "agentId": agent_id, "prompt": "wrap up"}}, now=T0)
                    self.assertEqual("reason" in result, expect_blocked, result)
                    if expect_blocked:
                        self.assertIn("3p-weekly", result["reason"])
            ledger = self.write_ledger(str(self.state), [self.route("antigravity-resilient", "gemini-4-argon-high")])
            launch = {"tool_name": "create_agent", "tool_input": {
                "provider": "antigravity-resilient/gemini-4-argon-high", "settings": {"modeId": "full-access"},
                "labels": {"router.account": "antigravity", "router.bundle_id": bundle_id}}}
            with self.subTest(launch=bundle_id),                     mock.patch.object(gate.router_module("dispatch_registry"), "claim", return_value=None):
                self.assertEqual("reason" in gate.evaluate_payload(launch, ledger_path=ledger, now=T0), blocked)

    def test_used_up_five_hour_window_blocks_every_account_alike(self):
        five = self.window("five_hour", 100)
        week = self.window("seven_day", 20, 100)
        for account, model in (("claude", "claude-opus-5-5"), ("claude-2", "claude-opus-5-5"),
                               ("claude-lab", "claude-sonnet-5-5"), ("codex", "gpt-6.1-sol")):
            cache = self.quota_cache({account: {"windows": [five, week]}})
            with self.subTest(account=account):
                self.assertIn("five_hour", gate.check_account_quota_exhausted(account, cache, T0, model=model))
                for fresh_row in ({"windows": [self.window("five_hour", 30), week]},
                                  {"windows": [{"id": "five_hour", "used_pct": 0, "resets_at": None}, week]}):
                    cache = self.quota_cache({account: fresh_row})   # partly used, or unstarted (no reset yet)
                    self.assertIsNone(gate.check_account_quota_exhausted(account, cache, T0, model=model))

    def test_scoped_week_blocks_only_its_own_model(self):
        cache = self.quota_cache({"claude": {"windows": [
            self.window("weekly_fable", 100, 40), self.window("seven_day", 30, 40), self.window("five_hour", 10)]}})
        self.assertIn("weekly_fable", gate.check_account_quota_exhausted("claude", cache, T0, model="claude-fable-5-1"))
        self.assertIsNone(gate.check_account_quota_exhausted("claude", cache, T0, model="claude-opus-5-5"))
        # A scope the provider reports on the reading needs no configuration or code (a new Sonnet-only week).
        cache = self.quota_cache({"claude": {"windows": [
            self.window("weekly_sonnet", 100, 40, models=["Sonnet"]), self.window("seven_day", 30, 40)]}})
        self.assertIn("weekly_sonnet", gate.check_account_quota_exhausted("claude", cache, T0, model="claude-sonnet-5-5"))
        self.assertIsNone(gate.check_account_quota_exhausted("claude", cache, T0, model="claude-opus-5-5"))
        # Without any declared scope the window binds every model.
        cache = self.quota_cache({"claude": {"windows": [self.window("weekly_mystery", 100, 40)]}})
        self.assertIsNotNone(gate.check_account_quota_exhausted("claude", cache, T0, model="claude-opus-5-5"))

    def test_magpie_pool_counts_every_account_it_holds(self):
        # claude-lab's five_hour keeps a 25% launch floor: 20% of one account is under it, 2 x 20% is not.
        for used, accounts, blocked in ((60, 2, False), (80, 2, False), (80, 1, True)):
            cache = self.quota_cache({"claude-lab": {"windows": [
                self.window("five_hour", used, accounts=accounts), self.window("seven_day", 30, 100, accounts=accounts)]}})
            with self.subTest(used=used, accounts=accounts):
                self.assertEqual(gate.check_account_quota_exhausted(
                    "claude-lab", cache, T0, model="claude-sonnet-5-5") is not None, blocked)

    def test_stale_reading_holds_inside_its_cycle_only(self):
        week = self.window("seven_day", 100, 24)
        for fetched, blocked in ((T0 - dt.timedelta(days=2), True), (T0 - dt.timedelta(days=8), False)):
            cache = self.quota_cache({"codex": {"fetched_at": fetched.isoformat(), "windows": [
                dict(week, id="usage")]}})
            with self.subTest(fetched=fetched):
                self.assertEqual(gate.check_account_quota_exhausted(
                    "codex", cache, T0, model="gpt-6.1-sol") is not None, blocked)

    def test_route_spec_resolves_account_from_ledger_or_bundle_catalog_without_name_inference(self):
        self.assertFalse(hasattr(gate, "secondary_claude_provider"))
        accounts = gate.load_dispatch_accounts()
        # Real writer row resolves launch fields and catalog `launch_model` via `bundle.bundle_id`
        spec = gate.route_spec(
            self.route("claude-lab", "claude-sonnet-5-5", "high", choice="lab_sonnet55_high", bundle_id="lab_sonnet55_high"),
            accounts,
        )
        self.assertEqual(spec["provider"], "claude-lab")
        self.assertEqual(spec["account"], "claude-lab")
        self.assertEqual(spec["model"], "claude-sonnet-5-5")
        self.assertEqual(spec["effort"], "high")
        wire_accounts = json.loads(json.dumps(accounts))
        wire_accounts["accounts"]["codex"]["bundles"]["wire_only"] = {
            "provider": "codex", "model": "canonical-row", "launch_model": "vendor/wire-row", "effort": "high",
        }
        wire_spec = gate.route_spec(
            self.route("codex", "canonical-row", "high", choice="other_choice", bundle_id="wire_only"),
            wire_accounts,
        )
        self.assertEqual(wire_spec["model"], "vendor/wire-row")
        # Arbitrary account resolved from ledger bundle without claude-2 suffix guessing
        custom = gate.route_spec(
            self.route("team-east", "claude-sonnet-5-5", "high", account="team-east", choice="custom_worker"),
            accounts,
        )
        self.assertEqual(custom["provider"], "team-east")
        self.assertEqual(custom["account"], "team-east")
        # Shapes no writer produces (`choice` without `bundle`, `selected_bundle`, top-level fields) resolve nothing
        for unproduced in (
            {"choice": "lab_sonnet55_high"},
            {"choice": "sonnet55_high_2"},
            {"stage": "plan", "selected_bundle": {"provider": "claude-lab", "account": "claude-lab", "model": "claude-sonnet-5-5"}},
            {"stage": "plan", "provider": "claude-lab", "account": "claude-lab", "model": "claude-sonnet-5-5", "effort": "high"},
        ):
            self.assertEqual(gate.route_spec(unproduced, accounts), {}, unproduced)

    def test_quota_gate_counts_other_inflight_reservations_like_routing(self):
        # Antigravity gemini-weekly (the shipped capacity of the fixture account's plan) at 93% used has 7% left (> 5%
        # safety margin). An active reservation of 3% of that capacity from another dispatch brings remaining to 4% (<= 5%).
        cache = self.quota_cache({"antigravity": {"windows": [
            self.window("gemini-weekly", 93.0, 100),
            self.window("gemini-5h", 10, 4),
            self.window("3p-weekly", 10, 100),
            self.window("3p-5h", 10, 4),
        ]}})
        accounts = gate.load_dispatch_accounts()
        _, candidates = gate.resolve_target({"bundle_id": "argon_high"}, accounts)
        axis_scores = gate.router_module("axis_scores")
        capacity = axis_scores.five_axis_overlay(accounts)["accounts"]["antigravity"]["capacity_usd"]["gemini-weekly"]
        other_record = {
            "stage": "dispatch",
            "dispatch_id": "other-dispatch",
            "state": "started",
            "at": T0.isoformat(),
            "lease_until": (T0 + dt.timedelta(minutes=10)).isoformat(),
            "choice": "argon_high",
            "bundle": {"account": "antigravity", "provider": "antigravity-resilient", "model": "gemini-4-argon-high", "effort": "high"},
            "expected_usd": round(0.03 * capacity, 2),
        }
        # Without the in-flight reservation, 7% > 5% safety margin is allowed.
        self.assertIsNone(gate.check_account_quota_exhausted(
            "antigravity", cache, T0, accounts=accounts, bundles=candidates, records=()))
        # With another dispatch's active reservation, remaining drops to 4% <= 5% and is blocked.
        reason = gate.check_account_quota_exhausted(
            "antigravity", cache, T0, accounts=accounts, bundles=candidates, records=[other_record])
        self.assertIsNotNone(reason)
        self.assertIn("gemini-weekly", reason)
        # A dispatch's own reservation is excluded when claiming that dispatch itself.
        self.assertIsNone(gate.check_account_quota_exhausted(
            "antigravity", cache, T0, accounts=accounts, bundles=candidates,
            records=[other_record], exclude_dispatch_id="other-dispatch"))

    def test_a_final_report_marker_does_not_change_the_quota_hold(self):
        registry = gate.router_module("dispatch_registry")
        agent_id = "22222222-2222-4222-8222-222222222222"
        row = {
            "dispatch_id": "dispatch-final",
            "decision_id": "decision-final",
            "dispatch_mode": "paseo",
            "parent_id": "parent-a",
            "state": "started",
            "agent_id": agent_id,
            "created_at": T0.isoformat(),
            "updated_at": T0.isoformat(),
            "launched_at": T0.isoformat(),
            "lease_until": (T0 + dt.timedelta(minutes=10)).isoformat(),
            "hard_until": (T0 + dt.timedelta(hours=1)).isoformat(),
            "bundle": {"bundle_id": "lab_sonnet55_high", "account": "claude-lab",
                       "provider": "claude-lab", "model": "claude-sonnet-5-5"},
            "expected_usd": 6.0,
            "usage": {},
            "request": {"provider": "claude-lab/claude-sonnet-5-5", "labels": {}, "settings": {}},
            "events": [{"state": "started", "at": T0.isoformat()}],
        }
        with registry.transaction() as connection:
            registry.save(connection, row)
        quiet = {"tool_name": "send_agent_prompt",
                 "tool_input": {"agentId": "parent-a", "prompt": "still working", "notifyOnFinish": False}}
        report = {"tool_name": "send_agent_prompt",
                  "tool_input": {"agentId": "parent-a", "prompt": "[Worker final report]\ndone", "notifyOnFinish": False}}
        with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": agent_id, "AGENTCOMPASS_WORKER": "1"}):
            self.assertEqual(gate.evaluate_payload(quiet, now=T0), {})
            self.assertEqual(registry.inspect("dispatch-final")["expected_usd"], 6.0)
            self.assertEqual(gate.evaluate_payload(report, now=T0), {})
        stored = registry.inspect("dispatch-final")
        self.assertEqual(stored["state"], "started")
        self.assertEqual(stored["expected_usd"], 6.0)
        self.assertNotIn("quota_released_reason", stored)
        self.assertTrue(registry.reservations(now=T0))
        other = dict(row, dispatch_id="dispatch-other", decision_id="decision-other", agent_id="worker-b",
                     expected_usd=6.0, quota_released_reason=None)
        other.pop("quota_released_reason")
        with registry.transaction() as connection:
            registry.save(connection, other)
        quoted = {"tool_name": "send_agent_prompt",
                  "tool_input": {"agentId": "worker-b", "prompt": "see [Worker final report] in the contract",
                                 "notifyOnFinish": False}}
        with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": "parent-a", "AGENTCOMPASS_WORKER": ""}):
            gate.evaluate_payload(quoted, now=T0)
        self.assertEqual(registry.inspect("dispatch-other")["expected_usd"], 6.0)
        self.assertNotIn("quota_released_reason", registry.inspect("dispatch-other"))

    def test_without_the_window_modules_only_the_latch_is_checked_and_a_missing_owner_is_not_swallowed(self):
        cache = self.quota_cache({"claude": {"windows": [self.window("five_hour", 100)]}})
        real = gate.router_module

        def missing(name):
            if name == "turn_refusal":
                return real(name)
            raise ImportError(name)
        with mock.patch.object(gate, "router_module", side_effect=missing):
            self.assertIsNone(gate.check_account_quota_exhausted("claude", cache, T0, model="claude-opus-5-5"))

    def test_a_hook_over_a_mismatched_skill_tree_denies_and_names_the_action(self):
        real = gate.router_module
        worker = {"labels": {"router.parent_id": "parent-fixture", "router.account": "claude-lab",
                             "router.bundle_id": "lab_sonnet55_high"}}
        payload = {"tool_name": "send_agent_prompt", "tool_input": {"agentId": "worker-fixture", "prompt": "continue"},
                   "parent_agent_id": "parent-fixture"}

        class WithoutAttribute:
            def __init__(self, module, missing):
                self.module, self.missing = module, missing

            def __getattr__(self, name):
                if name == self.missing:
                    raise AttributeError(name)
                return getattr(self.module, name)

        def missing_module(name):
            if name == "turn_refusal":
                raise ImportError(name)
            return real(name)

        def missing_attribute(name):
            return WithoutAttribute(real(name), "bundle_rows") if name == "account_fleet" else real(name)
        for label, router_module in (("module", missing_module), ("attribute", missing_attribute)):
            settings = {"accounts": {"claude-lab": {"bundles": {"lab_sonnet55_high": {"model": "claude-sonnet-5-5"}}}}}
            with self.subTest(missing=label), mock.patch.object(gate, "router_module", side_effect=router_module),                     mock.patch.object(gate, "load_paseo_agent", return_value=worker),                     mock.patch.object(gate, "load_dispatch_accounts", return_value=settings):
                result = gate.evaluate_payload(payload, now=T0)
                self.assertEqual(result.get("decision"), "block", result)
                self.assertIn("agentcompass setup", result["reason"])
                self.assertIn("不一致", result["reason"])

    def test_interleaved_decisions_match_older_route_in_both_orders(self):
        a = self.route(stage="record", constraint="user_override", parent_agent_id="parent-a", task_id="task-a")
        b = self.route("dot", stage="record", constraint="dot_closed_loop", parent_agent_id="parent-b", task_id="task-b")
        for rows in ([a, b], [b, a]):
            ledger = self.write_ledger(str(self.state), rows)
            for identity in ({"parentAgentId": "parent-a"}, {"parentAgentId": "parent-a", "taskId": "task-a"}):
                result = gate.evaluate_payload({"tool_name": "delegate_task", "tool_input": {
                    "provider": "codex/gpt-6-astra", "settings": {"thinkingOptionId": "high"}, **identity}},
                    ledger_path=ledger, now=T0)
                self.assertEqual(result, {})
            # N1: a session-less call cannot match `parent-a` or `parent-b`'s session-named `record` row.
            anon = gate.evaluate_payload({"tool_name": "delegate_task", "tool_input": {
                "provider": "codex/gpt-6-astra", "settings": {"thinkingOptionId": "high"}}},
                ledger_path=ledger, now=T0)
            self.assertEqual(anon.get("decision"), "block")

    def test_same_task_new_decision_rejects_old_owner_with_diagnostics(self):
        # Both native hand-over re-rolls (superseding the unconsumed row of `(parent_id, task_id)`) and record rows
        # honor the newest decision for an identified task and reject the superseded route with field diagnostics.
        missing = self.state / "missing.jsonl"
        first_claim = self.handover("codex", "gpt-6-astra", "high", surface="t3",
                                    parent_id="parent-a", session_ids=["parent-a"], task_id="task-a", now=T0)
        self.handover("pi", "gpt-6-astra", "high", surface="t3", bundle_id="pi_astra_high",
                      parent_id="parent-a", session_ids=["parent-a"], task_id="task-a",
                      now=T0 + dt.timedelta(seconds=5))
        registry = gate.router_module("dispatch_registry")
        self.assertEqual(registry.inspect(first_claim["dispatch_id"])["state"], "canceled")
        handover_res = gate.evaluate_payload({"session_id": "parent-a", "tool_name": "delegate_task", "tool_input": {
            "task": "work",
            "target": {"driverKind": "codex", "model": "gpt-6-astra", "options": {"reasoningEffort": "high"}}}},
            ledger_path=missing, now=T0 + dt.timedelta(seconds=10))
        self.assertEqual(handover_res["decision"], "block")
        self.assertIn("target.driverKind: requested='codex', routed='pi'", handover_res["reason"])
        self.assertIn("pi_astra_high", handover_res["reason"])
        self.assertNotIn("兼容检查", handover_res["reason"])

        identity = {"parent_agent_id": "parent-r", "task_id": "task-r"}
        ledger = self.write_ledger(str(self.state), [
            self.route(stage="record", constraint="user_override", **identity),
            self.route("pi", stage="record", constraint="user_override", bundle_id="pi_astra_high", **identity),
        ])
        result = gate.evaluate_payload({"tool_name": "delegate_task", "tool_input": {
            "provider": "codex/gpt-6-astra", "settings": {"thinkingOptionId": "high"},
            "parentAgentId": "parent-r", "taskId": "task-r"}},
            ledger_path=ledger, now=T0)
        self.assertEqual(result["decision"], "block")
        self.assertIn("provider: requested='codex', routed='pi'", result["reason"])
        self.assertIn("pi_astra_high", result["reason"])
        self.assertNotIn("兼容检查", result["reason"])

    def test_parent_alone_does_not_collapse_concurrent_tasks(self):
        missing = self.state / "missing.jsonl"
        self.handover("codex", "gpt-6-astra", "high", surface="t3", parent_id="a", session_ids=["a"], task_id="t1", now=T0)
        self.handover("pi", "gpt-6-astra", "high", surface="t3", bundle_id="pi_astra_high",
                      parent_id="a", session_ids=["a"], task_id="t2", now=T0 + dt.timedelta(seconds=1))
        self.assertEqual(gate.evaluate_payload({"session_id": "a", "tool_name": "delegate_task", "tool_input": {
            "task": "work",
            "target": {"driverKind": "codex", "model": "gpt-6-astra", "options": {"reasoningEffort": "high"}}}},
            ledger_path=missing, now=T0 + dt.timedelta(seconds=2)), {})

    def test_a_host_terminal_parent_matches_the_harness_session_it_runs(self):
        # Issue #2: roll inside an Orca terminal records the Orca session as the parent (the worker's reply
        # address) and the Claude Code session it runs in; Claude's hook payload carries only the latter.
        missing = self.state / "missing.jsonl"
        launch = {"tool_name": "Bash", "tool_input": {
            "command": 'orca orchestration worker-start --spec "x" --agent codex --model gpt-6-astra --effort high --json'}}
        self.handover("codex", "gpt-6-astra", "high", surface="orca",
                      parent_id="orca-sess", host_session_id="claude-uuid", task_id="t-orca-1", now=T0)
        self.assertEqual(gate.evaluate_payload(dict(launch, session_id="other-claude"), ledger_path=missing, now=T0)["decision"],
                         "block")
        self.assertEqual(gate.evaluate_payload(dict(launch, session_id="claude-uuid"), ledger_path=missing, now=T0), {})
        self.handover("codex", "gpt-6-astra", "high", surface="orca",
                      parent_id="orca-sess", host_session_id="claude-uuid", task_id="t-orca-2", now=T0)
        self.assertEqual(gate.evaluate_payload(dict(launch, session_id="orca-sess"), ledger_path=missing, now=T0), {})

    def test_nested_task_correlation_and_ttl_are_enforced(self):
        ledger = self.write_ledger(str(self.state), [
            self.route(stage="record", constraint="user_override", task={"task_id": "a"}),
            self.route("pi", stage="record", constraint="user_override", task={"task_id": "a"}),
        ])
        payload = {"tool_name": "delegate_task", "tool_input": {"provider": "codex/gpt-6-astra", "taskId": "a"}}
        self.assertIn("routed='pi'", gate.evaluate_payload(payload, ledger_path=ledger, now=T0)["reason"])
        expired = self.route(stage="record", constraint="user_override")
        expired["at"] = (T0 - dt.timedelta(minutes=16)).isoformat()
        ledger = self.write_ledger(str(self.state), [expired, self.route("dot", stage="record", constraint="dot_closed_loop")])
        self.assertEqual(gate.evaluate_payload(payload, ledger_path=ledger, now=T0)["decision"], "block")

    @mock.patch.object(gate, 'load_dispatch_accounts', return_value={})
    def test_structured_model_effort_and_correlation_mismatches(self, _settings):
        ledger = self.write_ledger(str(self.state), [
            self.route(stage="record", constraint="user_override",
                       parent_agent_id="a", task_id="task-a", agent_id="target-a"),
        ])
        for extra, diagnostic in (({"provider": "pi/gpt-6-astra"}, "provider"),
                                  ({"provider": "codex/gpt-6-sol"}, "model"),
                                  ({"settings": {"thinkingOptionId": "max"}}, "effort"),
                                  ({"parentAgentId": "b"}, "parent/task/target"),
                                  ({"taskId": "task-b"}, "parent/task/target"),
                                  ({"targetAgentId": "target-b"}, "parent/task/target")):
            args = {"provider": "codex/gpt-6-astra", "settings": {"thinkingOptionId": "high"},
                    "parentAgentId": "a", **extra}
            result = gate.evaluate_payload({"tool_name": "delegate_task", "tool_input": args}, ledger_path=ledger, now=T0)
            self.assertIn(diagnostic, result["reason"])

    def test_claude_agent_tool_family_alias_is_the_routed_model(self):
        # Claude's Agent tool accepts only family aliases; each row changes one condition.
        ledger = self.write_ledger(str(self.state), [
            self.route("claude", "claude-sonnet-5-5", "high", stage="record", constraint="user_override"),
        ])
        for model, effort, blocked_on in (("sonnet", "high", None),
                                          ("claude-sonnet-5-5", "high", None),
                                          ("opus", "high", "model"),
                                          ("sonnet", "max", "effort")):
            args = {"subagent_type": "general-purpose", "prompt": "do it", "model": model, "effort": effort}
            result = gate.evaluate_payload({"tool_name": "Agent", "tool_input": args}, ledger_path=ledger, now=T0)
            with self.subTest(model=model, effort=effort):
                if blocked_on:
                    self.assertIn(f"{blocked_on}: requested=", result["reason"])
                else:
                    self.assertEqual(result, {})
        # An alias names a Claude family; it cannot claim a non-Claude route.
        ledger = self.write_ledger(str(self.state), [self.route(stage="record", constraint="user_override")])
        result = gate.evaluate_payload({"tool_name": "Agent", "tool_input": {"prompt": "x", "model": "sonnet"}},
                                       ledger_path=ledger, now=T0)
        self.assertIn("model: requested='sonnet'", result["reason"])

    def test_current_wrapped_launch_uses_literal_settings_and_all_calls(self):
        ledger = self.write_ledger(str(self.state), [self.route()])
        with mock.patch.object(gate.router_module("dispatch_registry"), "claim", return_value=None):
            for code, allowed in (
                ("await paseo.create_agent({provider:'codex/gpt-6-astra', settings:{thinkingOptionId:'high'}, labels:{'router.dispatch_id':'d1'}})", True),
                ('await tools.mcp__paseo__create_agent({provider:"codex/gpt-6-astra", settings:{thinkingOptionId:"high"}})', False),
                ('await paseo.create_agent({provider:"codex/gpt-6-astra", labels:{"router.dispatch_id":"d1"}}); await paseo.create_agent({provider:"pi/gpt-6-astra"})', False),
            ):
                result = gate.evaluate_payload({"tool_name": "exec", "tool_input": code}, ledger_path=ledger, now=T0)
                self.assertEqual(result == {}, allowed, result)
            # Quoted examples and readonly provider arguments are not executing calls.
            self.exhausted_cache()
            code = 'await paseo.list_models({provider:"claude-lab"}); await paseo.create_agent({provider:"codex/gpt-6-astra", labels:{"router.dispatch_id":"d1"}, initialPrompt:"create_agent({provider: \'claude-lab\'})"})'
            self.assertEqual(gate.evaluate_payload({"tool_name": "exec", "tool_input": code}, ledger_path=ledger, now=T0), {})
        quoted = '// create_agent({provider:"claude-lab"})\ntext("create_agent({})")'
        self.assertEqual(gate.evaluate_payload({"toolName": "codemode", "input": {"code": quoted}}, now=T0), {"block": False})

    def test_mixed_wrapper_still_checks_shell_launch(self):
        code = 'await paseo.send_agent_prompt({agentId:"unknown", prompt:"continue"}); await tools.exec_command({cmd:"orca agent launch -m gpt-6-astra \'task\'"})'
        result = gate.evaluate_payload({"tool_name": "exec", "tool_input": code}, now=T0)
        self.assertIn("shell:orca worker launch", result["reason"])

    @pytest.mark.slow
    def test_native_cli_entrypoint_uses_isolated_state_and_response_contract(self):
        for payload, expected in (({"tool_name": "spawn_agent", "tool_input": {}}, "block"),
                                  ({"toolCall": {"name": "invoke_subagent", "args": {}}}, "deny"),
                                  ({"toolName": "create_agent", "input": {}}, True)):
            proc = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(gate.__file__))], input=json.dumps(payload),
                                  text=True, encoding="utf-8", capture_output=True, check=True)
            response = json.loads(proc.stdout)
            self.assertEqual(response.get("decision", response.get("block")), expected)

    def test_worker_compaction_limit_blocks_worn_turn_and_allows_fresh_or_unknown(self):
        proj_dir = self.state / "agents" / "proj"
        log_dir = self.state / ".claude" / "projects" / "proj"
        proj_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        parent_id = "00000000-0000-0000-0000-000000000001"
        wid = "11111111-2222-3333-4444-555555555555"
        fresh_wid = "22222222-3333-4444-5555-666666666666"
        worker_doc = {
            "id": wid,
            "provider": "claude",
            "labels": {"router.parent_id": parent_id, "paseo.parent-agent-id": parent_id},
            "persistence": {"provider": "claude", "sessionId": "s-comp"},
        }
        (proj_dir / f"{wid}.json").write_text(json.dumps(worker_doc), encoding="utf-8")
        (proj_dir / f"{fresh_wid}.json").write_text(
            json.dumps({"id": fresh_wid, "provider": "claude", "labels": {"router.parent_id": parent_id}}),
            encoding="utf-8",
        )
        boundary = json.dumps({"type": "system", "subtype": "compact_boundary", "compactMetadata": {"preTokens": 900000}})
        log_file = log_dir / "s-comp.jsonl"

        # 1. Degrade (missing log) and Allow (even with 2 compactions, 1.8M historical tokens < 2M threshold)
        for contents in (None, boundary + "\n" + boundary + "\n"):
            if contents is not None:
                log_file.write_text(contents, encoding="utf-8")
            res = gate.evaluate_payload({"tool_name": "mcp__paseo__run_turn", "tool_input": {"agentId": wid, "prompt": "next"}}, now=T0)
            self.assertEqual(res, {})

        # 2. Block (>= 2M historical tokens) the parent's turn to its own worker across direct turn tool, literal
        # codemode, and dynamic-var codemode; another caller's message, or one with no caller id, is communication.
        log_file.write_text(boundary + "\n" + boundary + "\n" + boundary + "\n", encoding="utf-8")
        for payload, blocked_key in (
            ({"tool_name": "mcp__paseo__run_turn", "tool_input": {"agentId": wid, "prompt": "next"}}, "decision"),
            ({"toolName": "codemode", "input": {"code": f'await paseo.run_turn({{agentId: "{wid}", prompt: "next"}})'}}, "block"),
            ({"toolName": "codemode", "input": {"code": f'const id = "{wid}"; await paseo.send_agent_prompt({{agentId: id, prompt: "next"}})'}}, "block"),
        ):
            res = gate.evaluate_payload(dict(payload, parent_agent_id=parent_id), now=T0)
            self.assertTrue(res.get(blocked_key), payload)
            self.assertIn("historical_context_tokens=2,700,000/2,000,000", res["reason"])
            self.assertIn("roll --brief", res["reason"])
            for caller in ("peer", None):
                self.assertNotIn("reason", gate.evaluate_payload(dict(payload, parent_agent_id=caller), now=T0))

        # 3. Allow: mentioning worn worker UUID inside prompt text to a fresh worker is not blocked
        handoff_code = f'const prompt = "prev worker {wid}"; await paseo.send_agent_prompt({{agentId: "{fresh_wid}", prompt}})'
        self.assertEqual(gate.evaluate_payload({"toolName": "codemode", "input": {"code": handoff_code}}, now=T0), {"block": False})

        # 4. Allow: child reporting upward to its parent or peer messaging a user main session (labels={}) is never blocked
        child_id = "33333333-4444-5555-6666-777777777777"
        (proj_dir / f"{child_id}.json").write_text(
            json.dumps({"id": child_id, "provider": "claude", "labels": {"paseo.parent-agent-id": wid}}),
            encoding="utf-8",
        )
        self.assertEqual(
            gate.evaluate_payload({"parent_agent_id": child_id, "tool_name": "mcp__paseo__send_agent_prompt",
                                   "tool_input": {"agentId": wid, "prompt": "report"}}, now=T0),
            {},
        )
        (proj_dir / f"{wid}.json").write_text(json.dumps({**worker_doc, "labels": {}}), encoding="utf-8")
        self.assertEqual(
            gate.evaluate_payload({"tool_name": "mcp__paseo__send_agent_prompt",
                                   "tool_input": {"agentId": wid, "prompt": "cross-lane notice"}}, now=T0),
            {},
        )

    # --- MCP-prefixed names (a proxy such as Magpie re-exports tools as mcp__<server>__<tool>) ---

    def test_base_tool_name_strips_only_mcp_server_prefixes(self):
        for name, base in (
                ("Agent", "Agent"),
                ("mcp__magpie__Agent", "Agent"),
                ("mcp__magpie__mcp__paseo__send_agent_prompt", "send_agent_prompt"),
                ("mcp__t3-code__delegate_task", "delegate_task"),
                # A server name may hold `__` or end in `_`; the classifier backtracks exactly like the matcher.
                ("mcp__paseo___create_agent", "create_agent"),
                ("mcp__my__srv__Agent", "Agent"),
                ("mcp__a__b__mcp__c___Workflow", "Workflow"),
                ("mcp__claude_ai_Claude_Docs__batch", "mcp__claude_ai_Claude_Docs__batch"),  # not a hooked tool
                ("mcp__paseo__", "mcp__paseo__"),       # no tool behind the prefix: left as is
                ("xmcp__paseo__create_agent", "xmcp__paseo__create_agent"),
                ("ListAgents", "ListAgents")):
            with self.subTest(name=name):
                self.assertEqual(gate.base_tool_name(name), base)

    def test_every_sub_agent_launch_name_blocks_without_a_route_prefixed_or_not(self):
        """Each launcher, bare and behind one or two MCP prefixes, is blocked for want of a router record."""
        missing = self.state / "missing.jsonl"
        launches = {
            "Agent": {"subagent_type": "Explore", "prompt": "look"},
            "Task": {"prompt": "look"},
            "spawn_agent": {"message": "work"},
            "invoke_subagent": {"Subagents": []},
            "create_agent": {"provider": "codex/gpt-6-astra", "prompt": "work"},
            "delegate_task": {"task": "work"},
            # Workflow: its script's agent() calls start sub-agents inside the runtime, out of any hook's sight.
            "Workflow": {"script": "export const meta = {name: 'x', description: 'y'}\nawait agent('go')"},
            # create_schedule: Paseo "starts a new agent on a cron cadence".
            "create_schedule": {"prompt": "work", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"},
            # update_schedule that re-targets the schedule's new-agent launch.
            "update_schedule": {"id": "s1", "provider": "codex", "model": "gpt-6-astra"},
        }
        for base, args in launches.items():
            for name in (base, f"mcp__magpie__{base}", f"mcp__magpie__mcp__paseo__{base}",
                         f"mcp__paseo___{base}", f"mcp__my__srv__{base}"):   # server names ending in `_` / holding `__`
                with self.subTest(name=name):
                    result = gate.evaluate_payload({"tool_name": name, "tool_input": args}, ledger_path=missing, now=T0)
                    self.assertEqual(result.get("decision"), "block")
                    self.assertIn(f"tool:{base}", result["reason"])
        # Antigravity and Pi payload shapes go through the same names.
        agy = gate.evaluate_payload({"toolCall": {"name": "mcp__magpie__invoke_subagent", "args": {}}},
                                    ledger_path=missing, now=T0)
        self.assertEqual(agy["decision"], "deny")
        pi = gate.evaluate_payload({"toolName": "mcp__magpie__Workflow", "input": {"script": "x"}},
                                   ledger_path=missing, now=T0)
        self.assertTrue(pi["block"])

    def test_prefixed_wrappers_and_shell_tools_are_inspected_like_bare_ones(self):
        missing = self.state / "missing.jsonl"
        for payload in (
                {"tool_name": "mcp__magpie__Bash", "tool_input": {"command": "orca agent launch -m gpt-6-astra 'task'"}},
                {"tool_name": "mcp__magpie__mcp__paseo__execute_code",
                 "tool_input": {"code": "await tools.mcp__magpie__mcp__paseo__create_agent({provider: 'codex'})"}},
                {"tool_name": "mcp__paseo__execute_code", "tool_input": {"code": "await create_agent({provider: 'codex'})"}}):
            with self.subTest(payload=payload):
                self.assertEqual(gate.evaluate_payload(payload, ledger_path=missing, now=T0).get("decision"), "block")
        self.assertEqual(gate.evaluate_payload(
            {"tool_name": "mcp__magpie__Bash", "tool_input": {"command": "git status"}}, ledger_path=missing, now=T0), {})

    def test_tools_that_do_not_start_an_agent_stay_allowed_without_a_route(self):
        """Names that merely contain "agent"/"schedule", and tools that only reach an existing session."""
        missing = self.state / "missing.jsonl"
        for name, args in (
                ("mcp__paseo__get_agent_status", {"agentId": "a"}),
                ("mcp__paseo__list_agents", {}),
                ("mcp__magpie__mcp__paseo__list_agents", {}),
                ("mcp__paseo__get_agent_activity", {"agentId": "a"}),
                ("mcp__paseo__archive_agent", {"agentId": "a"}),
                ("mcp__paseo__cancel_agent", {"agentId": "a"}),
                ("ListAgents", {}),
                ("TaskStop", {"task_id": "t"}),
                ("mcp__magpie__TaskStop", {"task_id": "t"}),
                ("SubagentHandback", {}),
                # SendMessage resumes an existing agent or reaches another session (incl. a worker's parent);
                # it cannot create an agent, so it is not a launch.
                ("SendMessage", {"to": "main", "message": "done"}),
                ("mcp__magpie__SendMessage", {"to": "worker", "message": "status?"}),
                # A heartbeat prompts the caller itself; run_schedule_once fires a schedule gated at creation.
                ("mcp__paseo__create_heartbeat", {"prompt": "check", "cron": "7 * * * *"}),
                ("mcp__paseo__run_schedule_once", {"id": "s1"}),
                ("mcp__paseo__update_schedule", {"id": "s1", "cron": "9 * * * *"}),
                ("mcp__paseo__list_schedules", {}),
                ("CronCreate", {"cron": "7 * * * *", "prompt": "check"}),
                ("mcp__someserver__agent_notes", {}),
                ("mcp__someserver__Agent_list", {})):
            with self.subTest(name=name):
                self.assertEqual(gate.evaluate_payload({"tool_name": name, "tool_input": args},
                                                       ledger_path=missing, now=T0), {})

    def test_prefixed_launch_passes_with_the_same_route_as_a_bare_one(self):
        ledger = self.write_ledger(str(self.state), [
            self.route("claude", "claude-sonnet-5-5", "high", stage="record", constraint="user_override"),
        ])
        args = {"subagent_type": "general-purpose", "prompt": "do it", "model": "sonnet", "effort": "high"}
        for name in ("Agent", "mcp__magpie__Agent"):
            with self.subTest(name=name):
                self.assertEqual(gate.evaluate_payload({"tool_name": name, "tool_input": args},
                                                       ledger_path=ledger, now=T0), {})
        mismatch = gate.evaluate_payload({"tool_name": "mcp__magpie__Agent", "tool_input": dict(args, model="opus")},
                                         ledger_path=ledger, now=T0)
        self.assertIn("model: requested='opus'", mismatch["reason"])
        # A worker route is not a Workflow authorization.
        workflow = gate.evaluate_payload({"tool_name": "mcp__magpie__Workflow", "tool_input": {"script": "x"},
                                          "session_id": "s1"}, ledger_path=ledger, now=T0)
        self.assertIn("--trigger workflow", workflow["reason"])

    def test_recorded_project_review_sub_agent_stays_exempt_under_a_prefix(self):
        """`agentcompass record` for a project review sub-agent authorizes its native Agent launch."""
        record = self.route("claude", "claude-opus-5-5", "high", stage="record", constraint="user_override",
                            at=(T0 - dt.timedelta(minutes=2)).isoformat())
        ledger = self.write_ledger(str(self.state), [record])
        args = {"subagent_type": "code-reviewer", "prompt": "review", "model": "opus"}
        for name in ("Agent", "mcp__magpie__Agent"):
            with self.subTest(name=name):
                self.assertEqual(gate.evaluate_payload({"tool_name": name, "tool_input": args},
                                                       ledger_path=ledger, now=T0), {})

    def test_prefixed_send_agent_prompt_keeps_worker_checks_and_parent_exemptions(self):
        wid = self.existing_agent()
        agent_file = self.state / "agents" / "project" / (wid + ".json")
        self.exhausted_cache()
        prefixed = "mcp__magpie__mcp__paseo__send_agent_prompt"
        cases = (
            # (labels, status, caller, blocked): the parent's own running worker is controlled, its idle one takes a
            # turn (refused on the used-up window), and a worker this caller does not own receives communication.
            ({"router.parent_id": "parent"}, "running", "parent", False),
            ({"router.parent_id": "parent"}, "idle", "parent", True),
            ({"router.parent_id": "parent"}, "running", "peer", False),
            ({"router.parent_id": "parent"}, "idle", "peer", False),
        )
        for labels, status, caller, blocked in cases:
            agent_file.write_text(json.dumps({
                "provider": "claude-lab/claude-sonnet-5-5", "lastStatus": status,
                "labels": {**labels, "router.account": "claude-lab", "router.bundle_id": "lab_sonnet55_high"}}),
                encoding="utf-8")
            for name in ("mcp__paseo__send_agent_prompt", prefixed):
                with self.subTest(name=name, status=status, caller=caller), \
                        mock.patch.dict(os.environ, {"PASEO_AGENT_ID": caller}):
                    result = gate.evaluate_payload({"tool_name": name, "tool_input": {
                        "agentId": wid, "prompt": "wrap up", "notifyOnFinish": True}}, now=T0)
                    self.assertEqual("reason" in result, blocked)
        # A parent turning off its own worker's completion notification is refused under the prefix too.
        agent_file.write_text(json.dumps({"provider": "claude-lab/claude-sonnet-5-5", "lastStatus": "running",
                                          "labels": {"router.parent_id": "parent"}}), encoding="utf-8")
        with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": "parent"}):
            result = gate.evaluate_payload({"tool_name": prefixed, "tool_input": {
                "agentId": wid, "prompt": "x", "notifyOnFinish": False}}, now=T0)
        self.assertIn("notifyOnFinish", result["reason"])

    # --- Workflow and schedules: their own authorization, matched to the call and to the calling session ---

    def workflow_record(self, **fields):
        """A record row as route_judge writes it: parent_id/host_session_id are the parent link, session_ids the writer."""
        defaults = dict(
            provider="claude", model="claude-opus-5-5", effort="high",
            stage="record", constraint="user_override", trigger="workflow",
            at=(T0 - dt.timedelta(minutes=2)).isoformat(),
            parent_id="paseo-parent", host_session_id="parent-claude-session",
            session_ids=["claude-session", "paseo-agent"],
        )
        defaults.update(fields)
        return writer_ledger_row(**defaults)

    def session_plan(self, provider="codex", model="gpt-6-astra", effort="high", **fields):
        defaults = dict(session_ids=["claude-session"])
        defaults.update(fields)
        return writer_ledger_row(provider, model, effort, **defaults)

    def test_workflow_needs_this_sessions_workflow_record(self):
        call = {"tool_input": {"script": "export const meta = {name: 'x', description: 'y'}"}}
        unrelated_review = self.workflow_record(trigger=None)          # e.g. a code-reviewer record
        routeless_workflow = dict(self.workflow_record(), bundle=None)
        cases = (
            ("no ledger", [], {"session_id": "claude-session"}, "--trigger workflow"),
            ("unrelated record", [unrelated_review], {"session_id": "claude-session"}, "--trigger workflow"),
            ("routeless workflow record", [routeless_workflow], {"session_id": "claude-session"}, "--trigger workflow"),
            ("worker plan", [self.session_plan("claude", "claude-opus-5-5")],
             {"session_id": "claude-session"}, "--trigger workflow"),
            ("hard constraint", [self.workflow_record(constraint="hard_constraint")],
             {"session_id": "claude-session"}, "--trigger workflow"),
            ("other session", [self.workflow_record()], {"session_id": "another-session"}, "--trigger workflow"),
            # R3: the parent link names the parent; it never authorizes the parent's Workflow.
            ("parent of the writer", [self.workflow_record()], {"session_id": "parent-claude-session"},
             "--trigger workflow"),
            ("parent agent of the writer", [self.workflow_record()], {"parent_agent_id": "paseo-parent"},
             "--trigger workflow"),
            ("row without session_ids", [self.workflow_record(session_ids=None)], {"session_id": "claude-session"},
             "--trigger workflow"),
            ("no session id", [self.workflow_record()], {}, "会话 ID"),
            ("expired", [self.workflow_record(at=(T0 - dt.timedelta(minutes=20)).isoformat())],
             {"session_id": "claude-session"}, "--trigger workflow"),
            ("same Claude session", [self.workflow_record()], {"session_id": "claude-session"}, None),
            ("same Paseo agent", [self.workflow_record()], {"parent_agent_id": "paseo-agent"}, None),
        )
        for label, rows, identity, blocked_on in cases:
            ledger = self.write_ledger(str(self.state), rows) if rows else self.state / "missing.jsonl"
            for name in ("Workflow", "mcp__magpie__Workflow"):
                with self.subTest(case=label, name=name), mock.patch.dict(os.environ, {"PASEO_AGENT_ID": ""}):
                    result = gate.evaluate_payload({"tool_name": name, **call, **identity}, ledger_path=ledger, now=T0)
                    if blocked_on:
                        self.assertIn(blocked_on, result["reason"])
                        self.assertNotIn("roll --brief", result["reason"].split("不要调用")[0])
                    else:
                        self.assertEqual(result, {})
        # Direct check_workflow_record also rejects a workflow record lacking a route bundle.
        self.assertIsNotNone(gate.check_workflow_record([routeless_workflow], {"claude-session"}))
        # Under Paseo the calling agent's own ID identifies the session too; the parent's ID does not.
        ledger = self.write_ledger(str(self.state), [self.workflow_record()])
        for agent_id, allowed in (("paseo-agent", True), ("paseo-parent", False)):
            with self.subTest(paseo_agent=agent_id), mock.patch.dict(os.environ, {"PASEO_AGENT_ID": agent_id}):
                result = gate.evaluate_payload({"tool_name": "Workflow", **call}, ledger_path=ledger, now=T0)
                self.assertEqual(result == {}, allowed)

    def test_schedule_needs_this_sessions_recent_plan_with_its_provider_and_model(self):
        plan = self.session_plan()
        routeless_plan = dict(plan, bundle=None)
        review_record = self.workflow_record(trigger=None)
        same_as_plan_record = dict(plan, stage="record", choice="constraint", constraint="user_override")
        mcp_cases = (
            ("no provider", [plan], {"prompt": "x", "cron": "7 * * * *"}, "未指定 provider/model"),
            ("provider only", [plan], {"prompt": "x", "cron": "7 * * * *", "provider": "codex"}, "未指定 model"),
            ("no plan", [review_record], {"prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"},
             "没有 plan 记录"),
            ("routeless plan", [routeless_plan], {"prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"},
             "没有 plan 记录"),
            ("record is not a plan", [same_as_plan_record],
             {"prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"}, "没有 plan 记录"),
            # R2: a plan another session made, or one that names no writer, is not this session's.
            ("other session's plan", [dict(plan, session_ids=["session-a"])],
             {"prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"}, "本会话没有 plan 记录"),
            ("plan without session_ids", [self.route("codex", "gpt-6-astra", "high")],
             {"prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"}, "本会话没有 plan 记录"),
            ("other model", [plan], {"prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6.1-sol"},
             "model: requested='gpt-6.1-sol'"),
            ("matching plan", [plan], {"prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"}, None),
        )
        for label, rows, args, blocked_on in mcp_cases:
            ledger = self.write_ledger(str(self.state), rows)
            for name in ("create_schedule", "mcp__paseo__create_schedule", "mcp__magpie__mcp__paseo__create_schedule"):
                with self.subTest(case=label, name=name):
                    result = gate.evaluate_payload({"tool_name": name, "tool_input": args, "session_id": "claude-session"},
                                                   ledger_path=ledger, now=T0)
                    if blocked_on:
                        self.assertIn(blocked_on, result["reason"])
                        self.assertIn("plan --input", result["reason"])
                        self.assertNotIn("roll --brief", result["reason"])
                    else:
                        self.assertEqual(result, {})
        self.assertIn("没有 plan 记录", gate.check_schedule_plan(
            [routeless_plan], {"provider": "codex/gpt-6-astra"}, {}, "tool:create_schedule", None, {"claude-session"}) or "")
        ledger = self.write_ledger(str(self.state), [plan])
        with mock.patch.dict(os.environ, {"PASEO_AGENT_ID": ""}):
            anonymous = gate.evaluate_payload({"tool_name": "create_schedule", "tool_input": {
                "prompt": "x", "cron": "7 * * * *", "provider": "codex/gpt-6-astra"}}, ledger_path=ledger, now=T0)
        self.assertIn("会话 ID", anonymous["reason"])
        for args, blocked in (({"id": "s1", "provider": "codex", "model": "gpt-6-astra"}, False),
                              ({"id": "s1", "provider": "codex", "model": "gpt-6.1-sol"}, True),
                              ({"id": "s1", "model": "gpt-6-astra"}, True),
                              ({"id": "s1", "cron": "9 * * * *"}, False)):
            with self.subTest(update=args):
                result = gate.evaluate_payload({"tool_name": "mcp__paseo__update_schedule", "tool_input": args,
                                                "session_id": "claude-session"}, ledger_path=ledger, now=T0)
                self.assertEqual("reason" in result, blocked)

    def test_paseo_schedule_cli_is_the_same_launch_as_the_mcp_tool(self):
        routed = self.write_ledger(str(self.state), [self.session_plan()])
        missing = self.state / "missing.jsonl"
        script = ("import subprocess; subprocess.run(['paseo', 'schedule', 'create', '--cron', '7 * * * *', "
                  "'--provider', 'codex/gpt-6-astra', 'work'])")
        dynamic = "import subprocess; subprocess.run(['paseo', 'schedule', 'create', '--provider', p, 'work'])"
        cases = (
            # (command, ledger, blocked)
            ('paseo schedule create --cron "*/5 * * * *" --provider codex "do x"', missing, True),
            ('paseo schedule create --every 5m --provider codex/gpt-6-astra --run-now "do x"', missing, True),
            ('paseo schedule create --every 5m --provider=codex/gpt-6-astra "do x"', routed, False),
            ('paseo schedule create --every 5m --provider codex/gpt-6-astra "do x"', routed, False),
            ('paseo schedule create --every 5m --provider codex/gpt-6.1-sol "do x"', routed, True),
            ('paseo schedule create --every 5m "do x"', routed, True),               # caller's default provider
            (r'& "C:\Program Files\Paseo\paseo.cmd" schedule create --every 5m --provider codex "x"', missing, True),
            ("paseo schedule update s1 --provider codex --model gpt-6-astra", routed, False),
            ("paseo schedule update s1 --model gpt-6.1-sol --provider codex", routed, True),
            ("paseo schedule update s1 --model gpt-6-astra", routed, True),
            ("paseo schedule update s1 --cron '9 * * * *' --prompt 'use --provider x'", missing, False),
            (f'python -c "{script}"', missing, True),
            (f'python -c "{script}"', routed, False),
            (f'python -c "{dynamic}"', routed, True),
            ("paseo schedule create --help", missing, False),
            ("paseo schedule run-once s1", missing, False),
            ("paseo schedule ls", missing, False),
            ('echo "paseo schedule create --provider codex x"', missing, False),
        )
        for command, ledger, blocked in cases:
            for payload in ({"tool_name": "Bash", "tool_input": {"command": command}, "session_id": "claude-session"},
                            {"tool_name": "mcp__magpie__PowerShell", "tool_input": {"command": command},
                             "session_id": "claude-session"}):
                with self.subTest(command=command, tool=payload["tool_name"]):
                    result = gate.evaluate_payload(payload, ledger_path=ledger, now=T0)
                    self.assertEqual("reason" in result, blocked)
                    if blocked:
                        self.assertIn("plan --input", result["reason"])

    def test_paseo_cli_global_options_and_wrappers_do_not_hide_a_launch(self):
        """R1: commander takes global options before and between command words; wrappers run the same argv."""
        missing = self.state / "missing.jsonl"
        schedule = "schedule create --provider codex/gpt-6-astra --run-now x"
        launches = [
            f"paseo --json {schedule}", f"paseo -q {schedule}", f"paseo --quiet {schedule}",
            f"paseo --host localhost:6767 {schedule}", f"paseo --host=localhost:6767 {schedule}",
            f"paseo -o json {schedule}", f"paseo --home C:/h --no-color --no-headers {schedule}",
            "paseo schedule --json create --provider codex/gpt-6-astra --run-now x",
            "paseo schedule --host h:1 create --provider codex/gpt-6-astra x",
            "paseo --json schedule update s1 --provider codex/gpt-6-astra",
            f'bash -c "paseo {schedule}"', f"sh -c 'paseo --json {schedule}'", f'/usr/bin/bash -c "paseo {schedule}"',
            f'pwsh -Command "paseo {schedule}"', f'cmd /c paseo {schedule}', f'cmd.exe /c "paseo {schedule}"',
            f"npx paseo {schedule}", f"npx -y paseo --json {schedule}", f"pnpm dlx paseo {schedule}",
            f"paseo.ps1 {schedule}", rf"& C:\tools\paseo.ps1 --json {schedule}",
            'Start-Process paseo -ArgumentList "schedule","create","--provider","codex/gpt-6-astra","x"',
            'Start-Process -FilePath paseo.cmd -ArgumentList "--json schedule create --provider codex/gpt-6-astra x"',
            "Start-Process paseo -ArgumentList 'schedule','--json','create','--provider','codex/gpt-6-astra','x' -NoNewWindow",
            # The same rule for the other Paseo launch commands.
            "paseo --json run --provider codex x", "paseo -q agent run --provider codex x",
            "paseo agent --json run --provider codex x", 'bash -c "paseo --json run --provider codex x"',
            'Start-Process paseo -ArgumentList "run","--provider","codex","x"',
            "python -c \"import subprocess; subprocess.run(['paseo', '--json', 'schedule', 'create', '--provider', 'codex/gpt-6-astra', 'x'])\"",
        ]
        reads = [
            "paseo --json schedule ls", "paseo schedule --json ls", "paseo -q schedule inspect s1",
            "paseo --json schedule logs s1", "paseo --json schedule run-once s1", "paseo --json schedule create --help",
            "paseo schedule create -h", 'bash -c "paseo --json schedule ls"', "npx paseo --json ls",
            "paseo --json ls", "paseo --host h:1 inspect 11111111-1111-4111-8111-111111111111",
            'Start-Process paseo -ArgumentList "schedule","ls"', "paseo --json start", "paseo -q status",
            'echo "paseo --json schedule create --provider codex x"',
        ]
        for command in launches + reads:
            with self.subTest(command=command):
                result = gate.evaluate_payload({"tool_name": "Bash", "tool_input": {"command": command},
                                                "session_id": "claude-session"}, ledger_path=missing, now=T0)
                self.assertEqual("reason" in result, command in launches)
        # A send behind global options is still a turn to that worker (target checks apply).
        wid = self.existing_agent()
        calls = gate.cli_turn_calls("Bash", {"command": f"paseo --json send --no-wait {wid} 'go'"})
        self.assertEqual(calls, [("run_turn", {"agentId": wid})])
        calls = gate.cli_turn_calls("Bash", {"command": f"paseo -q agent --host h:1 send {wid} go"})
        self.assertEqual(calls, [("run_turn", {"agentId": wid})])

    def test_matcher_and_gate_classify_every_name_alike(self):
        """The installed matcher sends a name to the hook exactly when the gate treats it as a hooked tool."""
        import re
        matcher = re.compile(gate.hook_matcher())
        questions = re.compile("|".join(gate.WORKER_QUESTION_TOOLS))
        bases = sorted(gate.HOOKED_TOOLS) + ["TaskStop", "ListAgents", "get_agent_status", "batch", "AgentX"]
        prefixes = ("", "mcp__paseo__", "mcp__magpie__mcp__paseo__", "mcp__paseo___", "mcp__my__srv__",
                    "mcp__a_-b__", "mcp___", "mcp__", "xmcp__paseo__", "mcp__paseo__x", "mcp____", "mcp__-__")
        for base in bases:
            for prefix in prefixes:
                name = prefix + base
                with self.subTest(name=name):
                    hooked = bool(matcher.search(name)) and not questions.search(name)
                    self.assertEqual(hooked, gate.base_tool_name(name) in gate.HOOKED_TOOLS)

    def test_single_prefix_pattern_equals_the_repeated_one_on_random_names(self):
        """R4: one optional `mcp__<server>__` accepts exactly the names (and base) the repeated form did."""
        import random
        import re
        names = "|".join(sorted(gate.HOOKED_TOOLS))
        repeated = re.compile(r"(?:mcp__[A-Za-z0-9_-]+?__)*(" + names + ")")
        alphabet = ["mcp__", "__", "_", "-", "a", "Z", "9", "mcp", "m", "x", "é", " "]
        rng = random.Random(20261110)
        bases = sorted(gate.HOOKED_TOOLS) + ["Agen", "TaskStop", ""]
        for _ in range(20000):
            name = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 7))) + rng.choice(bases)
            old = repeated.fullmatch(name)
            with self.subTest(name=name):
                self.assertEqual(gate.base_tool_name(name), old.group(1) if old else name)

    @pytest.mark.slow
    def test_adversarial_names_classify_in_linear_time(self):
        """R4: the repeated form took over 60 s on this 10k-character name; the gate and the matcher must not."""
        script = (
            "import re, sys, time; sys.path.insert(0, sys.argv[1]); import router_gate_hook as g\n"
            "names = ['mcp__a__' * 1250 + 'zz', 'mcp__' + '_' * 10000 + 'zz', 'mcp__a' * 2000 + '__zz',\n"
            "         'mcp__a__' * 1250 + 'Agent', 'mcp__a__' * 1250 + 'run_turn(']\n"
            "matcher = re.compile(g.hook_matcher())\n"
            "start = time.perf_counter()\n"
            "for name in names:\n"
            "    g.base_tool_name(name); matcher.search(name); g.CODEMODE_TURN_AGENT_RE.search(name)\n"
            "    g.WRAPPED_AGENT_CALL_RE.search(name)\n"
            "print(time.perf_counter() - start)\n")
        out = subprocess.run([sys.executable, "-c", script, str(Path(gate.__file__).parent)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertLess(float(out.stdout), 2.0)

    def test_hook_matcher_routes_exactly_the_gated_names(self):
        """Claude Code tests a non-plain matcher with `new RegExp(matcher).test(name)`; Python's search is the same."""
        import re
        matcher = re.compile(gate.hook_matcher())
        for name in ("Agent", "mcp__magpie__Agent", "Workflow", "mcp__magpie__Workflow", "Task",
                     "mcp__paseo__create_agent", "mcp__magpie__mcp__paseo__send_agent_prompt",
                     "mcp__paseo__create_schedule", "mcp__paseo__update_schedule", "mcp__t3-code__delegate_task",
                     "mcp__magpie__Bash", "Bash", "PowerShell", "exec_command", "mcp__paseo__execute_code",
                     "spawn_agent", "invoke_subagent", "AskUserQuestion", "mcp__x__request_user_input_async",
                     "functions.request_user_input"):
            with self.subTest(name=name):
                self.assertIsNotNone(matcher.search(name))
        for name in ("ListAgents", "TaskStop", "mcp__magpie__TaskStop", "SendMessage", "Read",
                     "mcp__paseo__get_agent_status", "mcp__paseo__list_agents", "mcp__paseo__create_heartbeat",
                     "mcp__paseo__run_schedule_once", "SubagentHandback", "AgentX", "xAgent", "executor"):
            with self.subTest(name=name):
                self.assertIsNone(matcher.search(name))


class BoundDispatchGateTests(unittest.TestCase):
    def setUp(self):
        dispatch_registry = gate.router_module('dispatch_registry')
        account_fleet = gate.router_module('account_fleet')
        host_policy = gate.router_module('host_policy')
        self.registry = dispatch_registry
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        env = mock.patch.dict(os.environ, {
            'ROUTER_DISPATCH_DB': str(root / 'dispatch.sqlite3'),
            'ROUTER_OVERRIDES': str(root / 'overrides.json'),
            'ROUTER_CAPACITY': str(root / 'capacity.json'),
            'ROUTER_LEDGER': str(root / 'ledger.jsonl'),
            'ROUTER_USAGE_CACHE': str(root / 'usage.json'),
            'ROUTER_GATE_BYPASS': '0', 'PASEO_AGENT_ID': '',
            'ROUTER_HOST_POLICY': str(root / 'host-policy.json'),
            'AGENTCOMPASS_ACCOUNTS_OVERRIDE': str(root / 'accounts-override.json'),
            'ROUTER_ACCOUNTS_FILE': str(gate.router_module('account_fleet').reference_fleet_path()),
        })
        env.start()
        self.addCleanup(env.stop)
        policy = mock.patch.object(host_policy, 'load_policy', return_value=None)
        policy.start()
        self.addCleanup(policy.stop)
        # The machine's installed-provider discovery would otherwise decide whether the fixture account can launch.
        inventory = mock.patch.object(gate.router_module('provider_inventory'), 'current', return_value={'accounts': {}})
        inventory.start()
        self.addCleanup(inventory.stop)
        accounts = account_fleet.load_accounts(account_fleet.reference_fleet_path())
        bundle = dict(account_fleet.bundles(accounts)['argon_high'], bundle_id='argon_high')
        self.task = {'parent_agent_id': 'parent-fixture', 'task_id': 'task-fixture',
                     'work_kind': 'code_implementation', 'expected_hours': 0.5, 'needs': {'local_only': True},
                     'profile': {'work_kind': 'code_implementation', 'primary_level': 2, 'secondary_kind': 'none', 'secondary_level': 0,
                                 'needs_visual': False, 'needs_gui': False, 'local_only': True, 'wants_fast': False,
                                 'read_only_review': False, 'access': 'write', 'expected_hours': 0.5,
                                 'multi_task': False, 'task_count': '1',
                                 'axis_demand': {'creativity': 1, 'coding': 2, 'aesthetics': 0, 'tool_operation': 1}}}
        self.now = dt.datetime.now(UTC)
        decision = dispatch_registry.issue({'choice': 'argon_high', 'selected_bundle': bundle}, self.task, self.now)
        self.row = dispatch_registry.prepare(decision['decision_id'], self.task,
            {'title': 'fixture', 'initialPrompt': 'do fixture work', 'cwd': str(root)},
            {'read_ok': True, 'accounts': {'antigravity': {'windows': [
                {'id': wid, 'remaining_pct_before_reset': 80, 'observed_remaining_pct': 80, 'seconds_until_reset': hours * 3600,
                 'resets_at': (self.now + dt.timedelta(hours=hours)).isoformat()}
                for wid, hours in (('gemini-5h', 3), ('gemini-weekly', 100))]}}}, accounts, self.now)

    def payload(self, request=None, **changes):
        return dict({'tool_name': 'mcp__paseo__create_agent',
            'tool_input': request or self.row['request'], 'parent_agent_id': 'parent-fixture',
            'tool_use_id': 'call-a'}, **changes)

    def test_modern_paseo_requires_supported_labels_not_global_ledger(self):
        request = dict(self.row['request'])
        request.pop('labels')
        denied = gate.evaluate_payload(self.payload(request), now=self.now)
        self.assertEqual(denied['decision'], 'block')
        self.assertIn('supports labels', denied['reason'])

    def test_route_judge_mention_does_not_hide_a_later_shell_launch(self):
        for command in ('echo route_judge.py && paseo run --provider codex/gpt-6-astra task',
                        'python route_judge.py plan --input task.json; paseo run --provider codex/gpt-6-astra task'):
            trigger = gate.detect_subagent_launch('exec_command', {'cmd': command})
            self.assertEqual(trigger, 'shell:paseo agent create')
            denied = gate.evaluate_payload({'tool_name': 'exec_command', 'tool_input': {'cmd': command}}, now=self.now)
            self.assertEqual(denied['decision'], 'block')

    def test_quoted_separators_are_argument_text_not_new_commands(self):
        for command, expected in (
                ('grep -n "a\\|b" ~/.codex/skills/dot-subagent/scripts/run_dot_task.py', None),
                ("rg 'x|y' scripts/run_dot_task.py", None),
                ('bash -c "echo hi; paseo run --provider codex/gpt-6-astra task"', 'shell:paseo agent create'),
                ('grep -n "a|b" x.py; paseo run --provider codex/gpt-6-astra task', 'shell:paseo agent create')):
            self.assertEqual(gate.detect_subagent_launch('exec_command', {'cmd': command}), expected, command)

    def test_a_launcher_name_given_to_another_program_is_not_a_launch(self):
        """The gate judges the program each simple command runs, not text anywhere in the line."""
        for command in (
                # Reproduced: blocked as shell:run_dot_task.py because `skills/dot-subagent ` is followed by a space.
                'cd /d/work/agentcompass && cat agentcompass/test_docs.py; ls skills/route-agent-work '
                'skills/dot-subagent skills/chatgpt-pro-consult',
                'ls skills/dot-subagent', 'ls -la ~/.codex/skills/dot-subagent/scripts/run_dot_task.py',
                'cat skills/dot-subagent/scripts/run_dot_task.py',
                "rg -n 'paseo agent create|orca agent launch' setup", 'rg dot-subagent skills',
                "git grep -n 'orca agent launch'", 'git grep run_dot_task.py -- skills',
                r'Get-Content skills\dot-subagent\scripts\run_dot_task.py',
                'Get-Content -Path skills/dot-subagent/SKILL.md | Select-Object -First 5',
                'echo dot-subagent run_dot_task.py', 'echo "paseo run --provider codex x"',
                "Write-Output 'orca agent launch task'",
                "git commit -F - <<'EOF'\nfix the gate; paseo run --provider codex x\nEOF\ngit log -1",
                "git commit -m @'\nfix the gate; orca agent launch task\n'@"):
            with self.subTest(command=command):
                self.assertIsNone(gate.detect_subagent_launch('Bash', {'command': command}))
                self.assertEqual(gate.evaluate_payload({'tool_name': 'Bash', 'tool_input': {'command': command}},
                                                       now=self.now), {})

    def test_a_launch_anywhere_in_a_compound_line_reaches_the_launch_checks(self):
        dot = 'skills/dot-subagent/scripts/run_dot_task.py --prompt-file task.md'
        for tool, command, trigger in (
                ('Bash', 'git status; ls; orca agent launch -m gpt-6-astra task', 'shell:orca worker launch'),
                ('Bash', 'echo a | grep a && orca agent launch --agent claude --prompt hi', 'shell:orca worker launch'),
                ('Bash', 'cd skills && paseo run --provider codex/gpt-6-astra task', 'shell:paseo agent create'),
                ('Bash', f'cd /d/work && python -X utf8 {dot}', 'shell:run_dot_task.py'),
                ('Bash', f'python3 -u -W ignore {dot}', 'shell:run_dot_task.py'),
                ('Bash', 'FOO=1 dot-subagent run --prompt hi', 'shell:run_dot_task.py'),
                ('Bash', "bash -lc 'cd x && orca agent launch task'", 'shell:orca worker launch'),
                ('Bash', 'cmd /c "cd x & paseo agent create --provider codex"', 'shell:paseo agent create'),
                ('Bash', 'out=$(orca agent launch task)', 'shell:orca worker launch'),
                # PowerShell forms: the call operator, an assignment, `;` and `|`.
                ('PowerShell', rf'Set-Location D:\work; & python -X utf8 D:\work\{dot}', 'shell:run_dot_task.py'),
                ('PowerShell', r"& 'C:\Program Files\Paseo\paseo.cmd' run --provider codex x",
                 'shell:paseo agent create'),
                ('PowerShell', '$result = orca.cmd agent launch task', 'shell:orca worker launch'),
                ('PowerShell', 'Get-ChildItem | Out-Null; orca.exe orchestration worker-start --task t1',
                 'shell:orca worker launch')):
            with self.subTest(command=command):
                self.assert_launch(tool, command, trigger)

    def assert_launch(self, tool, command, trigger):
        self.assertEqual(gate.detect_subagent_launch(tool, {'command': command}), trigger)
        denied = gate.evaluate_payload({'tool_name': tool, 'tool_input': {'command': command}}, now=self.now)
        self.assertEqual(denied['decision'], 'block')
        self.assertIn(f'({trigger})', denied['reason'])

    def test_valid_lines_parse_in_the_dialect_of_the_sending_tool(self):
        """Valid shell is never undecidable: what follows it on the line is still judged."""
        commit = "git commit -m \"$(cat <<'EOF'\nIt's the gate; paseo run is quoted here.\nEOF\n)\""
        self.assertIsNone(gate.detect_subagent_launch('Bash', {'command': commit}))
        for tool, command, trigger in (
                # A here-document with an apostrophe inside $(...): the commit-message idiom.
                ('Bash', commit + ' && paseo run --provider codex x', 'shell:paseo agent create'),
                # PowerShell: a double-quoted Windows path may end in a backslash; ` is the escape.
                ('PowerShell', r'cd "D:\work\" ; paseo run --provider codex x', 'shell:paseo agent create'),
                ('PowerShell', r'python D:\work\run_dot_task.py --out "D:\out\" --prompt t', 'shell:run_dot_task.py'),
                ('PowerShell', 'echo "`$(" && orca agent launch task', 'shell:orca worker launch'),
                ('PowerShell', '<# a; b #> orca agent launch task', 'shell:orca worker launch')):
            with self.subTest(command=command):
                self.assert_launch(tool, command, trigger)

    def test_shells_eval_and_wrappers_run_the_command_they_are_given(self):
        for tool, command, trigger in (
                ('Bash', "bash <<'EOF'\ncd /d/work\npaseo run --provider codex x\nEOF", 'shell:paseo agent create'),
                ('Bash', 'sh -s <<EOF\norca agent launch task\nEOF', 'shell:orca worker launch'),
                ('Bash', "eval 'paseo run --provider codex x'", 'shell:paseo agent create'),
                ('PowerShell', 'Invoke-Expression "orca agent launch task"', 'shell:orca worker launch'),
                ('PowerShell', "iex 'paseo run --provider codex x'", 'shell:paseo agent create'),
                ('PowerShell', "Start-Process -FilePath python -ArgumentList @('run_dot_task.py','--prompt','t')",
                 'shell:run_dot_task.py'),
                ('PowerShell', r'cmd /c "C:\Program Files\Paseo\paseo.cmd" run --provider codex x',
                 'shell:paseo agent create'),
                ('Bash', 'timeout 600 paseo run --provider codex x', 'shell:paseo agent create'),
                ('Bash', 'timeout -s KILL 300 orca agent launch task', 'shell:orca worker launch'),
                ('Bash', '/usr/bin/env python run_dot_task.py --prompt t', 'shell:run_dot_task.py'),
                ('Bash', 'uv run python run_dot_task.py --prompt t', 'shell:run_dot_task.py'),
                ('Bash', 'poetry run python run_dot_task.py --prompt t', 'shell:run_dot_task.py'),
                ('Bash', "wsl -d Ubuntu -- bash -lc 'orca agent launch task'", 'shell:orca worker launch'),
                ('Bash', 'nice -n 10 orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'stdbuf -o0 orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'setsid orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'winpty orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'sudo -u worker orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'nohup orca agent launch task &', 'shell:orca worker launch'),
                ('Bash', "env -S 'paseo run --provider codex x'", 'shell:paseo agent create'),
                ('Bash', "env --split-string='orca agent launch' task", 'shell:orca worker launch'),
                ('Bash', 'uvx orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'uv tool run orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'pipx run orca agent launch task', 'shell:orca worker launch'),
                ('Bash', 'cmd //c paseo run --provider codex x', 'shell:paseo agent create'),   # Git Bash's /c
                # A shell with options and no script file still runs its here-document.
                ('Bash', "bash -s arg <<'EOF'\norca agent launch task\nEOF", 'shell:orca worker launch'),
                ('Bash', "bash -o pipefail <<'EOF'\norca agent launch task\nEOF", 'shell:orca worker launch'),
                # Start-Process joins -ArgumentList into one Windows command line: quotes written into an element
                # keep a spaced script path together.
                ('PowerShell', r'''Start-Process python -ArgumentList '"D:\my work\run_dot_task.py"','--prompt','t' ''',
                 'shell:run_dot_task.py'),
                ('PowerShell', r'Start-Process python -ArgumentList "`"D:\my work\run_dot_task.py`" --prompt t"',
                 'shell:run_dot_task.py'),
                ('PowerShell', r'''Start-Process python -ArgumentList @('"D:\my work\run_dot_task.py"','--prompt','t')''',
                 'shell:run_dot_task.py')):
            with self.subTest(command=command):
                self.assert_launch(tool, command, trigger)
        # A heredoc fed to a program that is not a shell, or to a shell given a script file, is that program's stdin;
        # another subcommand of a wrapper is its own program. A spaced path Start-Process passes unquoted is cut at
        # the space by Start-Process itself, so python runs `D:\my`, not the Dot runner.
        for tool, command in (
                ('Bash', "cat <<'EOF'\npaseo run x\nEOF"), ('Bash', 'uv pip install codex'),
                ('Bash', "bash run.sh <<'EOF'\npaseo run x\nEOF"), ('Bash', "sh ./install.sh <<'EOF'\norca agent launch t\nEOF"),
                ('PowerShell', r'Start-Process python -ArgumentList "D:\my work\run_dot_task.py --prompt t"')):
            with self.subTest(command=command):
                self.assertIsNone(gate.detect_subagent_launch(tool, {'command': command}))

    def test_posix_arithmetic_is_not_a_here_document(self):
        for command in ('(( x = 1 << 2 ))\norca agent launch task', 'echo $(( 1 << 2 ))\norca agent launch task'):
            with self.subTest(command=command):
                self.assert_launch('Bash', command, 'shell:orca worker launch')

    def codex_call(self, cmd, **extra):
        """An exec_command call as Codex sends it: a bare command string in `cmd`, `shell` only sometimes."""
        return {'tool_name': 'exec_command',
                'tool_input': dict({'cmd': cmd, 'workdir': 'D:/work', 'yield_time_ms': 10000}, **extra)}

    def test_codex_exec_command_is_read_in_the_host_default_shell(self):
        """A tool that names no shell runs the host's: PowerShell on Windows."""
        here_string = "@'\nIt's the plan:\norca agent launch runs tasks, isn't it\n'@ | Set-Content plan.md"
        with mock.patch.object(gate, 'HOST_SHELL_DIALECT', gate.POWERSHELL):
            for cmd, trigger in (
                    ("@'\nprint(\"it's\")\n'@ | python -; paseo run --provider codex x", 'shell:paseo agent create'),
                    (r'''$p = 'D:\work'; cd "$p\"; paseo run --provider codex x''', 'shell:paseo agent create'),
                    (r'python D:\work\run_dot_task.py --out "D:\out\" --prompt t', 'shell:run_dot_task.py'),
                    ("<# it's #> orca agent launch task", 'shell:orca worker launch'),
                    ("Start-Process -FilePath python -ArgumentList @('run_dot_task.py','--prompt','t')",
                     'shell:run_dot_task.py'),
                    ('Write-Output "a`"b"; orca agent launch task', 'shell:orca worker launch'),
                    ('orca `\nagent launch task', 'shell:orca worker launch')):
                with self.subTest(cmd=cmd):
                    payload = self.codex_call(cmd)
                    self.assertEqual(gate.detect_subagent_launch(payload['tool_name'], payload['tool_input']), trigger)
                    denied = gate.evaluate_payload(payload, now=self.now)
                    self.assertEqual(denied['decision'], 'block')
                    self.assertIn(f'({trigger})', denied['reason'])
            # A here-string is text, also inside the code-mode `exec` wrapper.
            self.assertEqual(gate.evaluate_payload(self.codex_call(here_string), now=self.now), {})
            wrapped = {'tool_name': 'exec',
                       'tool_input': 'text(await tools.exec_command(' + json.dumps({'cmd': here_string}) + '))'}
            self.assertEqual(gate.evaluate_payload(wrapped, now=self.now), {})
        with mock.patch.object(gate, 'HOST_SHELL_DIALECT', gate.POSIX):
            self.assertEqual(gate.evaluate_payload(self.codex_call("cat <<'EOF'\npaseo run x\nEOF"), now=self.now), {})

    def test_the_shell_argument_of_a_call_decides_its_dialect(self):
        heredoc = "cat <<'EOF'\npaseo run x\nEOF"      # a launch only when `<<` is not a here-document
        start = "Start-Process -FilePath python -ArgumentList @('run_dot_task.py','--prompt','t')"
        with mock.patch.object(gate, 'HOST_SHELL_DIALECT', gate.POWERSHELL):
            for shell in ('bash', 'bash.exe', '/usr/bin/bash'):
                with self.subTest(shell=shell):
                    self.assertIsNone(gate.detect_subagent_launch('exec_command', self.codex_call(heredoc, shell=shell)['tool_input']))
        with mock.patch.object(gate, 'HOST_SHELL_DIALECT', gate.POSIX):
            self.assertIsNone(gate.detect_subagent_launch('exec_command', self.codex_call(start)['tool_input']))
            for shell in ('powershell', 'pwsh'):
                with self.subTest(shell=shell):
                    self.assertEqual(gate.detect_subagent_launch(
                        'exec_command', self.codex_call(start, shell=shell)['tool_input']), 'shell:run_dot_task.py')

    def test_a_line_the_chosen_dialect_cannot_parse_is_read_in_the_other(self):
        for tool, command, trigger in (
                ('Bash', "@'\nprint(\"it's\")\n'@ | python -; orca agent launch task", 'shell:orca worker launch'),
                ('PowerShell', 'echo "a\\"b"; orca agent launch task', 'shell:orca worker launch')):
            with self.subTest(tool=tool, command=command):
                self.assert_launch(tool, command, trigger)

    def test_posix_brace_expansion_is_not_a_command_break(self):
        for command in ('echo {a,b} dot-subagent', 'ls skills/{dot-subagent,x}', 'echo x{1..3} orca agent launch'):
            with self.subTest(command=command):
                self.assertIsNone(gate.detect_subagent_launch('Bash', {'command': command}))
        self.assert_launch('Bash', '{ cd /d/work; orca agent launch task; }', 'shell:orca worker launch')

    def test_forms_left_fail_open_on_purpose(self):
        """The accepted gap: the gate prevents accidental launches and is not a security boundary.

        Commands built at run time (xargs, find -exec), bash backticks (` is PowerShell's escape), inline node code,
        a command piped into a shell, a PowerShell expression as the script path, and a non-literal program are not
        read as launches.
        """
        for tool, command in (
                ('Bash', 'ls | xargs paseo run'), ('Bash', r'find . -name task.md -exec orca agent launch {} \;'),
                ('Bash', 'echo `orca agent launch task`'),
                ('Bash', "node -e \"require('child_process').exec('orca agent launch task')\""),
                ('Bash', "echo 'orca agent launch task' | bash"),
                ('PowerShell', "python (Join-Path $d 'run_dot_task.py') --prompt t"),
                ('Bash', '$LAUNCHER exec task')):
            with self.subTest(command=command):
                self.assertIsNone(gate.detect_subagent_launch(tool, {'command': command}))

    def test_launcher_inspection_commands_are_not_launches(self):
        dot = 'python ~/.codex/skills/dot-subagent/scripts/run_dot_task.py'
        for command in (f'{dot} --help', f'{dot} -h', f'{dot} run --check-only --prompt hi', f'{dot} doctor',
                        f'{dot} status', f'{dot} run --task-id t1', f'{dot} run --task-id=t1', 'dot-subagent status',
                        'cd x && orca agent launch --help', 'orca agent launch -h', 'paseo run --help', 'paseo agent create -h',
                        'orca orchestration worker-start --help', 'orca orchestration worker-start -h'):
            with self.subTest(command=command):
                self.assertIsNone(gate.detect_subagent_launch('Bash', {'command': command}))

    def test_a_one_shot_model_cli_call_is_not_a_launch(self):
        """A model CLI asked for one answer from a shell is a quick call; it starts no worker that needs a route."""
        for tool, command in (
                ('Bash', "codex exec -m gpt-6-astra 'task'"), ('PowerShell', '$answer = codex.cmd exec task'),
                ('Bash', 'cat notes.md | claude -p "summarize"'), ('Bash', 'agy -p task'), ('Bash', 'gemini -p task'),
                ('Bash', 'opencode run task'), ('Bash', 'pi -p task'), ('Bash', 'grok -p task')):
            with self.subTest(command=command):
                self.assertIsNone(gate.detect_subagent_launch(tool, {'command': command}))
                self.assertEqual(gate.evaluate_payload({'tool_name': tool, 'tool_input': {'command': command}},
                                                       now=self.now), {})

    def test_a_line_that_does_not_parse_is_not_a_launch(self):
        """No shell runs it; the gate keeps its fail-open degrade for an undecidable line."""
        for command in ('echo "unterminated; paseo run --provider codex x', "orca agent launch 'task",
                        'echo $(orca agent launch task'):
            with self.subTest(command=command):
                self.assertIsNone(gate.detect_subagent_launch('Bash', {'command': command}))

    def test_labelled_launch_works_without_any_legacy_ledger_and_is_single_use(self):
        self.assertEqual(gate.evaluate_payload(self.payload(), now=self.now), {})
        self.assertEqual(gate.evaluate_payload(self.payload(), now=self.now), {})
        denied = gate.evaluate_payload(self.payload(tool_use_id='call-b'), now=self.now)
        self.assertEqual(denied['decision'], 'block')
        self.assertIn('already claimed', denied['reason'])

    def test_labels_cannot_authorize_another_account_task_bundle_or_parent(self):
        for key in ('router.account', 'router.task_id', 'router.parent_id', 'router.decision_id'):
            args = json.loads(json.dumps(self.row['request']))
            args['labels'][key] = 'unrelated'
            self.assertEqual(gate.evaluate_payload(self.payload(args), now=self.now)['decision'], 'block')
        wrong_bundle = dict(self.row['request'], provider='antigravity-resilient/gemini-3.8-flash-high')
        self.assertEqual(gate.evaluate_payload(self.payload(wrong_bundle), now=self.now)['decision'], 'block')
        self.assertEqual(gate.evaluate_payload(self.payload(parent_agent_id='other-parent'), now=self.now)['decision'], 'block')
        self.assertEqual(self.registry.inspect(self.row['dispatch_id'])['state'], 'prepared')

    def test_literal_wrapper_binds_routed_settings_not_assignment_text(self):
        def payload(args):
            return {'toolName': 'codemode', 'input': {'code': 'await create_agent(' + json.dumps(args) + ')'},
                    'parent_agent_id': 'parent-fixture'}
        other_effort = dict(self.row['request'], settings=dict(self.row['request']['settings'], thinkingOptionId='low'))
        self.assertTrue(gate.evaluate_payload(payload(other_effort), now=self.now)['block'])
        revised = dict(self.row['request'], initialPrompt='revised work', title='revised', workspaceId='other')
        self.assertEqual(gate.evaluate_payload(payload(revised), now=self.now), {'block': False})

    def test_dynamic_wrapper_cannot_change_a_partially_parsed_bound_request(self):
        literal = json.dumps(self.row['request'])
        for expression in (literal[:-1] + ', ...changes}',
                           literal[:-1] + ', [key]: value}',
                           literal[:-1] + ', initialPrompt: taskPrompt}',
                           literal + ' || otherRequest'):
            payload = {'toolName': 'codemode', 'input': {'code': 'await create_agent(' + expression + ')'},
                       'parent_agent_id': 'parent-fixture'}
            self.assertTrue(gate.evaluate_payload(payload, now=self.now)['block'])
        self.assertEqual(self.registry.inspect(self.row['dispatch_id'])['state'], 'prepared')


def usage_row(account, named=None):
    return gate.router_module("turn_refusal").usage_row(account, named, gate.HOME)


class UsageRowTests(unittest.TestCase):
    """The router reads quota only from the canonical cache (or an explicit named file): `turn_refusal.usage_row`, which the hook calls."""

    def setUp(self):
        self.state = Path(os.environ["ROUTER_STATE_DIR"])
        self.canonical = self.state / "usage-cache.json"
        self.old_float = gate.HOME / ".usage-float" / "usage-cache.json"
        self.addCleanup(lambda: [path.unlink() for path in (self.canonical, self.old_float) if path.exists()])
        os.environ.pop("ROUTER_USAGE_CACHE", None)
        reg = mock.patch.object(gate.router_module("usage_cache"), "_registry", return_value="")
        reg.start()
        self.addCleanup(reg.stop)

    def write(self, path, used, fetched):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"providers": {"codex": {"fetched_at": fetched, "windows": [{"id": "5h", "used_pct": used}]}}}),
                        encoding="utf-8")

    def test_canonical_cache_or_named_file_is_read_and_old_usage_float_copy_is_ignored(self):
        self.write(self.old_float, 90, "2026-01-02T00:00:00+00:00")
        self.assertIsNone(usage_row("codex"))
        self.write(self.canonical, 10, "2026-01-01T00:00:00+00:00")
        self.assertEqual(usage_row("codex")["windows"][0]["used_pct"], 10)
        named = self.state / "named.json"
        self.write(named, 5, "2026-01-01T00:00:00+00:00")
        self.addCleanup(named.unlink)
        self.assertEqual(usage_row("codex", named)["windows"][0]["used_pct"], 5)
        self.assertIsNone(usage_row("codex", self.state / "missing.json"))


class DenyMessageEntryTests(unittest.TestCase):
    """The hook tells a blocked agent to run the public command, not a script path."""

    def test_deny_texts_name_the_public_command_only(self):
        self.assertTrue(gate.ROUTER_CMD == "agentcompass" or gate.ROUTER_CMD.endswith("/agentcompass"))
        texts = [gate.build_deny_reason("Agent", 900), gate.DISABLED_RECOVERY]
        for text in texts:
            self.assertIn(gate.ROUTER_CMD + " roll --brief", text)
            self.assertNotIn("route_judge", text)
            self.assertNotIn(".py", text)


class ConfiguredRotationThresholdGateTests(unittest.TestCase):
    def setUp(self):
        import shutil

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.agents_dir = self.tmp / ".paseo" / "agents"
        (self.agents_dir / "ws").mkdir(parents=True, exist_ok=True)
        self.wid = "88888888-4444-4444-8888-999999999999"
        (self.agents_dir / "ws" / f"{self.wid}.json").write_text(
            json.dumps({
                "id": self.wid,
                "provider": "claude",
                "lastStatus": "idle",
                "labels": {"router.parent_id": "parent-main"},
            }),
            encoding="utf-8",
        )

    def test_gate_uses_configured_threshold_and_surfaces_missing_key(self):
        account_fleet = gate.router_module("account_fleet")
        worker_sessions = gate.router_module("worker_sessions")
        custom_accounts = {"scoring": {"rotate_after_carried_tokens": 750_000}, "accounts": {}}
        with mock.patch.object(account_fleet, "load_accounts", return_value=custom_accounts):
            with mock.patch.object(
                worker_sessions,
                "session_state",
                return_value={"compactions": 1, "compacted_tokens": 600_000, "context_tokens": 100_000, "historical_context_tokens": 700_000},
            ):
                self.assertIsNone(
                    gate.check_worker_compaction_limit(self.wid, paseo_agents_dir=self.agents_dir, home=self.tmp, identities={"parent-main"})
                )
            with mock.patch.object(
                worker_sessions,
                "session_state",
                return_value={"compactions": 1, "compacted_tokens": 700_000, "context_tokens": 100_000, "historical_context_tokens": 800_000},
            ):
                reason = gate.check_worker_compaction_limit(self.wid, paseo_agents_dir=self.agents_dir, home=self.tmp, identities={"parent-main"})
                self.assertIsNotNone(reason)
                self.assertIn("historical_context_tokens=800,000/750,000", reason)
                self.assertIn("rotate_after_carried_tokens=750,000", reason)

        with mock.patch.object(account_fleet, "load_accounts", return_value={"scoring": {}, "accounts": {}}), \
                mock.patch.object(worker_sessions, "session_state", return_value={"historical_context_tokens": 800_000}):
            with self.assertRaises(KeyError):
                gate.check_worker_compaction_limit(self.wid, paseo_agents_dir=self.agents_dir, home=self.tmp, identities={"parent-main"})


if __name__ == "__main__":
    unittest.main()


def test_legacy_account_flag_does_not_break_rotation_check(tmp_path, monkeypatch):
    """Rule (AGENTS.md, what a released version wrote for the user is kept; parent-address-gate.md): an
    `accounts-override.json` carrying `verified: false` (v0.1.0 to v0.5.0) is upgraded by `account_fleet.load_accounts`
    without raising, so `check_worker_compaction_limit` still evaluates the configured rotation threshold on an owned
    idle worker (allowing below threshold and blocking at threshold)."""
    account_fleet = gate.router_module('account_fleet')
    host_policy = gate.router_module('host_policy')
    provider_inventory = gate.router_module('provider_inventory')
    worker_sessions = gate.router_module('worker_sessions')
    override = tmp_path / 'accounts-override.json'
    policy = tmp_path / 'host-policy.json'
    agents_dir = tmp_path / '.paseo' / 'agents'
    (agents_dir / 'ws').mkdir(parents=True, exist_ok=True)
    (agents_dir / 'ws' / 'worker.json').write_text(
        json.dumps({'id': 'worker', 'provider': 'codex', 'lastStatus': 'idle',
                    'labels': {'router.parent_id': 'parent-main'}}),
        encoding='utf-8',
    )
    monkeypatch.setenv('AGENTCOMPASS_ACCOUNTS_OVERRIDE', str(override))
    monkeypatch.setenv('ROUTER_HOST_POLICY', str(policy))
    monkeypatch.setattr(account_fleet, 'accounts_override_path', lambda: override)
    monkeypatch.setattr(host_policy, 'policy_path', lambda: (policy, False))
    monkeypatch.setattr(provider_inventory, 'current', lambda: {'accounts': {}})
    threshold = worker_sessions.rotation_threshold(worker_sessions.shipped_scoring())

    override.write_text(json.dumps({'schema_version': 2, 'accounts': {'codex': {'verified': False}}}), encoding='utf-8')
    monkeypatch.setattr(worker_sessions, 'session_state',
                        lambda *args, **kwargs: {'historical_context_tokens': threshold - 1})
    assert gate.check_worker_compaction_limit(
        'worker', paseo_agents_dir=agents_dir, home=tmp_path, identities={'parent-main'},
    ) is None

    override.write_text(json.dumps({'schema_version': 2, 'accounts': {'codex': {'verified': False}}}), encoding='utf-8')
    policy.unlink(missing_ok=True)
    monkeypatch.setattr(worker_sessions, 'session_state',
                        lambda *args, **kwargs: {'historical_context_tokens': threshold})
    reason = gate.check_worker_compaction_limit(
        'worker', paseo_agents_dir=agents_dir, home=tmp_path, identities={'parent-main'},
    )
    assert reason is not None and f'rotate_after_carried_tokens={threshold:,}' in reason


def hold_broken_fetched_set(monkeypatch):
    """A fetched set held by this process whose accounts.json breaks account_fleet.load_accounts with a data error."""
    public_data = gate.router_module('public_data')
    public_data.release()
    monkeypatch.setattr(public_data, '_EFFECTS', [0])
    shipped = public_data._shipped_set(public_data.BUNDLED)
    broken = json.loads(shipped['files']['accounts.json'])
    del broken['accounts']
    public_data._HELD[public_data._key(None, None)] = {
        **shipped, 'source': 'fetched', 'version': 'abc123def456', 'id': 'fetched:abc123def456',
        'files': {**shipped['files'], 'accounts.json': json.dumps(broken).encode()}}
    return public_data


def test_the_gate_reads_its_configuration_from_the_shipped_set_when_the_fetched_one_fails(monkeypatch):
    """Rule (router-internals §5, never on the failure line): load_dispatch_accounts swallows its errors into allow-all,
    so the reference read inside it falls back to the shipped set first; the degrade applies only if that fails too."""
    public_data = hold_broken_fetched_set(monkeypatch)
    accounts = gate.load_dispatch_accounts()
    assert accounts and accounts.get('accounts'), 'degraded to allow-all instead of reading the shipped set'
    assert public_data.held()['source'] == 'bundled'
    state = public_data.read_state()
    assert state['set_aside_version'] == 'abc123def456' and 'ReferenceDataError' in state['set_aside_error']
    public_data.release()


def test_the_quota_verdict_is_decided_on_the_shipped_set_when_the_fetched_one_fails(monkeypatch):
    public_data = hold_broken_fetched_set(monkeypatch)
    latched_windows, without_own, window_refusal = gate.router_owner('turn_refusal', 'latched_windows', 'without_own', 'window_refusal')
    decided = []

    def refusal(account, accounts, *args):
        decided.append((public_data.held()['source'], bool(accounts.get('accounts'))))
        return ('model', 'codex:weekly', 'used up')
    real_owner = gate.router_owner
    monkeypatch.setattr(gate, 'router_owner', lambda module, *names: (
        (lambda *a: set()), without_own, refusal) if module == 'turn_refusal' else real_owner(module, *names))
    reason = gate.check_account_quota_exhausted('codex', records=[])
    assert decided == [('bundled', True)]
    assert reason and 'codex:weekly' in reason
    assert public_data.read_state()['set_aside_version'] == 'abc123def456'
    public_data.release()


def test_gate_matches_bundle_wire_launch_model_without_crossing_provider_identities(tmp_path, monkeypatch):
    """Rule (router-internals Pins): a ledger row's bundle launches as its catalog launch_model (else model), for both
    plan and user_override record rows; resolve_target applies account_fleet.pin_matching_bundles over the constrained pool."""
    real_accounts = gate.load_dispatch_accounts()
    accounts = json.loads(json.dumps(real_accounts))
    accounts["accounts"]["codex"]["bundles"].update({
        "auto:codex:codex:deepseek-ai/DeepSeek-V4-Pro": {
            "provider": "codex", "model": "deepseek-ai/DeepSeek-V4-Pro", "launch_model": "deepseek-ai/DeepSeek-V4-Pro", "effort": "high"},
        "auto:codex:codex:deepseek-v4-pro-0813": {
            "provider": "codex", "model": "deepseek-v4-pro-0813", "launch_model": "deepseek-v4-pro", "effort": "high"},
        "auto:codex:codex:opencode/deepseek-v4-flash": {
            "provider": "codex", "model": "opencode/deepseek-v4-flash", "launch_model": "opencode/deepseek-v4-flash", "effort": "high"},
        "auto:codex:codex:deepseek-v4.1-flash-1017": {
            "provider": "codex", "model": "deepseek-v4.1-flash-1017", "launch_model": "deepseek/deepseek-v4-flash", "effort": "high"},
        "auto:codex:codex:vendor-b/novel-unmeasured": {
            "provider": "codex", "model": "vendor-b/novel-unmeasured", "launch_model": "vendor-b/novel-unmeasured", "effort": "high"},
        "auto:codex:codex:vendor-a/novel-unmeasured": {
            "provider": "codex", "model": "vendor-a/novel-unmeasured", "launch_model": "vendor-a/novel-unmeasured", "effort": "high"},
        "auto:codex:codex:codex/novel-gpt-9": {
            "provider": "codex", "model": "codex/novel-gpt-9", "launch_model": "codex/novel-gpt-9", "effort": "high"},
    })
    monkeypatch.setattr(gate, "load_dispatch_accounts", lambda: accounts)

    for wire_model, expected_id in (
        ("deepseek-ai/DeepSeek-V4-Pro", "auto:codex:codex:deepseek-ai/DeepSeek-V4-Pro"),
        ("DeepSeek-AI/DeepSeek-V4-Pro", "auto:codex:codex:deepseek-ai/DeepSeek-V4-Pro"),
        ("deepseek-v4-pro", "auto:codex:codex:deepseek-v4-pro-0813"),
        ("opencode/deepseek-v4-flash", "auto:codex:codex:opencode/deepseek-v4-flash"),
        ("deepseek/deepseek-v4-flash", "auto:codex:codex:deepseek-v4.1-flash-1017"),
        ("vendor-a/novel-unmeasured", "auto:codex:codex:vendor-a/novel-unmeasured"),
        ("openai/novel-gpt-9", "auto:codex:codex:codex/novel-gpt-9"),
    ):
        acct, found = gate.resolve_target({"provider": "codex", "model": wire_model, "effort": "high"}, accounts)
        assert (acct, [b["bundle_id"] for b in found]) == ("codex", [expected_id])

    pairs = (
        ("auto:codex:codex:deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro",
         "deepseek-ai/DeepSeek-V4-Pro", ("deepseek-v4-pro", "deepseek-v4-pro-0813")),
        ("auto:codex:codex:deepseek-v4-pro-0813", "deepseek-v4-pro-0813",
         "deepseek-v4-pro", ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-v4-pro-0813")),
        ("auto:codex:codex:opencode/deepseek-v4-flash", "opencode/deepseek-v4-flash",
         "opencode/deepseek-v4-flash", ("deepseek/deepseek-v4-flash", "deepseek-v4.1-flash-1017")),
        ("auto:codex:codex:deepseek-v4.1-flash-1017", "deepseek-v4.1-flash-1017",
         "deepseek/deepseek-v4-flash", ("opencode/deepseek-v4-flash", "deepseek-v4.1-flash-1017")),
    )
    registry = gate.router_module("dispatch_registry")
    ledger_path = tmp_path / "ledger.jsonl"
    missing_ledger = tmp_path / "missing.jsonl"
    for p_idx, (bid, row_model, own_wire, rival_wires) in enumerate(pairs):
        # 1. A `plan` row alone never authorizes `delegate_task`, while a native hand-over (from `roll`) does once
        #    and rejects rival wire spellings without consuming the hand-over.
        ledger_path.write_text(json.dumps(writer_ledger_row(
            "codex", row_model, "high", stage="plan", choice=bid, bundle_id=bid,
        )) + "\n", encoding="utf-8")
        plan_denied = gate.evaluate_payload({
            "session_id": f"s-wire-{p_idx}",
            "tool_name": "delegate_task",
            "tool_input": {"provider": f"codex/{own_wire}", "settings": {"thinkingOptionId": "high"}},
        }, ledger_path=ledger_path, now=T0)
        assert plan_denied.get("decision") == "block", (bid, plan_denied)

        with mock.patch.object(registry, "own_session_ids", return_value=[f"s-wire-{p_idx}"]):
            registry.record_host_claim(
                "t3", {"decision_id": f"dec-wire-{p_idx}", "choice": bid},
                {"parent_agent_id": f"s-wire-{p_idx}", "task_id": f"t-wire-{p_idx}"},
                {"provider": "codex", "account": "codex", "model": row_model, "effort": "high", "bundle_id": bid},
                native_handover=True, native_tool="delegate_task",
                launch_fields={"target": {"driverKind": "codex", "model": own_wire, "options": {"reasoningEffort": "high"}}},
                now=T0,
            )
        for wrong in rival_wires:
            denied = gate.evaluate_payload({
                "session_id": f"s-wire-{p_idx}",
                "tool_name": "delegate_task",
                "tool_input": {"task": "w", "target": {"driverKind": "codex", "model": wrong,
                                                       "options": {"reasoningEffort": "high"}}},
            }, ledger_path=missing_ledger, now=T0)
            assert "model:" in denied.get("reason", ""), (bid, "handover", wrong, denied)
        ok_handover = gate.evaluate_payload({
            "session_id": f"s-wire-{p_idx}",
            "tool_name": "delegate_task",
            "tool_input": {"task": "w", "target": {"driverKind": "codex", "model": own_wire,
                                                   "options": {"reasoningEffort": "high"}}},
        }, ledger_path=missing_ledger, now=T0)
        assert ok_handover == {}, (bid, "handover", ok_handover)

        # 2. A standalone `record` row also resolves the catalog wire `launch_model` via `bundle_id`.
        ledger_path.write_text(json.dumps(writer_ledger_row(
            "codex", row_model, "high", stage="record", choice="constraint", bundle_id=bid, constraint="user_override",
        )) + "\n", encoding="utf-8")
        ok = gate.evaluate_payload({
            "tool_name": "delegate_task",
            "tool_input": {"provider": f"codex/{own_wire}", "settings": {"thinkingOptionId": "high"}},
        }, ledger_path=ledger_path, now=T0)
        assert ok == {}, (bid, "record", ok)
        for wrong in rival_wires:
            denied = gate.evaluate_payload({
                "tool_name": "delegate_task",
                "tool_input": {"provider": f"codex/{wrong}", "settings": {"thinkingOptionId": "high"}},
            }, ledger_path=ledger_path, now=T0)
            assert "model:" in denied.get("reason", ""), (bid, "record", wrong, denied)

    # M12c & R2-1 (A2) through `gate.evaluate_payload`: `resolve_target` matches over `account_fleet.pin_pool` BEFORE
    # filtering by `effort` or `disabled`, so a missing effort on `vendor-a/novel-unmeasured` never falls back onto
    # `vendor-b/novel-unmeasured@medium`, and an unchecked `vendor-a` is gated on its own checkbox (admitted only on
    # `user_override`, never masked by an enabled `vendor-b`).
    m12c_accounts = json.loads(json.dumps(accounts))
    m12c_accounts["accounts"]["codex"]["bundles"] = {
        "b_vb_med": {
            "provider": "codex", "model": "vendor-b/novel-unmeasured", "launch_model": "vendor-b/novel-unmeasured",
            "effort": "medium", "disabled": True,
        },
        "b_va_high": {
            "provider": "codex", "model": "vendor-a/novel-unmeasured", "launch_model": "vendor-a/novel-unmeasured",
            "effort": "high", "disabled": False,
        },
    }
    monkeypatch.setattr(gate, "load_dispatch_accounts", lambda: m12c_accounts)
    with mock.patch.object(registry, "own_session_ids", return_value=["s-m12c"]):
        registry.record_host_claim(
            "t3", {"decision_id": "dec-m12c", "choice": "b_va_high@medium"},
            {"parent_agent_id": "s-m12c", "task_id": "t-m12c"},
            {"provider": "codex", "account": "codex", "model": "vendor-a/novel-unmeasured", "effort": "medium"},
            native_handover=True, native_tool="delegate_task",
            launch_fields={"target": {"driverKind": "codex", "model": "vendor-a/novel-unmeasured",
                                      "options": {"reasoningEffort": "medium"}}},
            now=T0,
        )
    res_m12c = gate.evaluate_payload({
        "session_id": "s-m12c",
        "tool_name": "delegate_task",
        "tool_input": {"task": "w", "target": {"driverKind": "codex", "model": "vendor-a/novel-unmeasured",
                                               "options": {"reasoningEffort": "medium"}}},
    }, ledger_path=missing_ledger, now=T0)
    assert res_m12c == {}, res_m12c

    a2_accounts = json.loads(json.dumps(accounts))
    a2_accounts["accounts"]["codex"]["bundles"] = {
        "b_vb_high": {
            "provider": "codex", "model": "vendor-b/novel-unmeasured", "launch_model": "vendor-b/novel-unmeasured",
            "effort": "high", "disabled": False,
        },
        "b_va_high": {
            "provider": "codex", "model": "vendor-a/novel-unmeasured", "launch_model": "vendor-a/novel-unmeasured",
            "effort": "high", "disabled": True,
        },
    }
    monkeypatch.setattr(gate, "load_dispatch_accounts", lambda: a2_accounts)
    with mock.patch.object(registry, "own_session_ids", return_value=["s-a2-auto"]):
        registry.record_host_claim(
            "t3", {"decision_id": "dec-a2-auto", "choice": "b_va_high"},
            {"parent_agent_id": "s-a2-auto", "task_id": "t-a2-auto"},
            {"provider": "codex", "account": "codex", "model": "vendor-a/novel-unmeasured", "effort": "high",
             "bundle_id": "b_va_high"},
            native_handover=True, native_tool="delegate_task",
            launch_fields={"target": {"driverKind": "codex", "model": "vendor-a/novel-unmeasured",
                                      "options": {"reasoningEffort": "high"}}},
            now=T0,
        )
    verdict_auto = gate.evaluate_payload({
        "session_id": "s-a2-auto",
        "tool_name": "delegate_task",
        "tool_input": {"task": "w", "target": {"driverKind": "codex", "model": "vendor-a/novel-unmeasured",
                                               "options": {"reasoningEffort": "high"}}},
    }, ledger_path=missing_ledger, now=T0)
    assert verdict_auto.get("decision") == "block", verdict_auto
    for extra, should_allow in (
        ({"constraint": "hard_constraint"}, False),
        ({"constraint": "user_override"}, True),
    ):
        ledger_path.write_text(json.dumps(writer_ledger_row(
            "codex", "vendor-a/novel-unmeasured", "high", stage="record", choice="b_va_high", bundle_id="b_va_high", **extra,
        )) + "\n", encoding="utf-8")
        verdict = gate.evaluate_payload({
            "tool_name": "delegate_task",
            "tool_input": {"provider": "codex/vendor-a/novel-unmeasured", "settings": {"thinkingOptionId": "high"}},
        }, ledger_path=ledger_path, now=T0)
        assert (verdict == {}) is should_allow, (extra, verdict)

    # Round 4: bare model target names every bundle of the row across accounts, while `P/name` obeys `P` (and falls back
    # only when `P` names nothing in the pool):
    r4_accounts = json.loads(json.dumps(real_accounts))
    r4_accounts["accounts"] = {
        "acct_a": {
            "billing": "subscription", "family": "codex", "windows": {"usage": {"hours": 168}},
            "bundles": {"b_a": {"provider": "acct_a", "model": "gpt-5.5", "launch_model": "gpt-5.5", "effort": "high"}},
        },
        "codex": {
            "billing": "subscription", "family": "codex", "windows": {"usage": {"hours": 168}},
            "bundles": {"b_b": {"provider": "codex", "model": "gpt-5.5", "launch_model": "openai/gpt-5.5", "effort": "high"}},
        },
    }
    acct_bare, found_bare = gate.resolve_target({"model": "gpt-5.5", "effort": "high"}, r4_accounts)
    assert acct_bare is None  # spans both accounts (`b_a` and `b_b`)
    assert gate.resolve_target({"model": "codex/gpt-5.5", "effort": "high"}, r4_accounts)[0] == "codex"
    assert gate.resolve_target({"model": "openai/gpt-5.5", "effort": "high"}, r4_accounts)[0] == "codex"
    assert gate.resolve_target({"model": "acct_a/gpt-6-astra", "effort": "high"}, r4_accounts) == (None, [])

    # Clause 3 (owner, 2026-11-20): a prefix naming nothing in the pool is dropped and the rest read from the start:
    c3_gate_accounts = json.loads(json.dumps(real_accounts))
    c3_gate_accounts["accounts"] = {
        "codex": {
            "billing": "subscription", "family": "codex", "windows": {"usage": {"hours": 168}},
            "bundles": {
                "b_flash": {"provider": "codex", "model": "deepseek-v4.1-flash-1017",
                            "launch_model": "deepseek/deepseek-v4-flash", "effort": "high"},
                "b_pro": {"provider": "codex", "model": "deepseek-v4-pro-0813",
                          "launch_model": "deepseek-v4-pro", "effort": "high"},
            },
        },
    }
    assert gate.resolve_target({"model": "opencode/deepseek-v4-flash", "effort": "high"}, c3_gate_accounts)[0] == "codex"
    assert gate.resolve_target({"model": "google-vertex/deepseek-ai/DeepSeek-V4-Pro", "effort": "high"}, c3_gate_accounts)[0] == "codex"
    c3_gate_with_oc = json.loads(json.dumps(c3_gate_accounts))
    c3_gate_with_oc["accounts"]["opencode"] = {
        "billing": "subscription", "family": "opencode", "windows": {"usage": {"hours": 168}},
        "bundles": {"b_oc": {"provider": "opencode", "model": "gpt-5.5", "launch_model": "gpt-5.5", "effort": "high"}},
    }
    assert gate.resolve_target({"model": "opencode/deepseek-v4-flash", "effort": "high"}, c3_gate_with_oc) == (None, [])

    # Malformed `/` or `auto:` pins are refused at the gate boundary (`resolve_target` and `evaluate_payload`), and for
    # every `(0..4 colons) x (0..4 slashes)` string around empty and non-empty segments (with and without `auto:`) the
    # gate answers allow or block, never an exception:
    for bad_model in ("/gpt-5.5", "/", "openai/", "openai//gpt-5.5", ":high", "auto:acct:harness", "auto::harness:gpt-5.5"):
        with pytest.raises(ValueError, match="invalid_model_pin"):
            gate.resolve_target({"model": bad_model, "effort": "high"}, r4_accounts)
        denied_spawn = gate.evaluate_payload({
            "tool_name": "spawn_agent",
            "tool_input": {"model": bad_model},
        }, ledger_path=ledger_path, now=T0)
        assert "invalid_model_pin" in denied_spawn.get("reason", ""), (bad_model, denied_spawn)
    for bad_provider in ("/gpt-5.5", "codex/", "codex//gpt-5.5"):
        denied_prov = gate.evaluate_payload({
            "tool_name": "delegate_task",
            "tool_input": {"provider": bad_provider, "settings": {"thinkingOptionId": "high"}},
        }, ledger_path=ledger_path, now=T0)
        assert "invalid_model_pin" in denied_prov.get("reason", ""), (bad_provider, denied_prov)

    odd_gate_strings = {"auto:acct:harness"}
    for n_colons in range(5):
        for n_slashes in range(5):
            for seps in ((":" * n_colons) + ("/" * n_slashes), ("/" * n_slashes) + (":" * n_colons)):
                slots = len(seps) + 1
                for fill in (
                    [""] * slots,
                    ["seg"] * slots,
                    [""] + ["seg"] * (slots - 1),
                    ["seg"] * (slots - 1) + [""],
                    ["seg"] * (slots - 1) + ["high"],
                ):
                    built = "".join(fill[idx] + (seps[idx] if idx < len(seps) else "") for idx in range(slots))
                    odd_gate_strings.add(built)
                    odd_gate_strings.add("auto:" + built)
    for odd_str in sorted(odd_gate_strings):
        for tname, tinput in (
            ("spawn_agent", {"model": odd_str}),
            ("delegate_task", {"provider": odd_str}),
            ("Read", {"file_path": odd_str}),
        ):
            verdict = gate.evaluate_payload({"tool_name": tname, "tool_input": tinput}, ledger_path=ledger_path, now=T0)
            assert verdict == {} or (verdict.get("decision") == "block" and bool(verdict.get("reason"))), (tname, odd_str, verdict)
