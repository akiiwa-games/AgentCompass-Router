"""Host-neutral workspace directory validation and per-host directory placement.

`--workspace <dir>` is a host-neutral placement claim: an existing directory that
this worker writes in and claims exclusively. Core code validates and canonicalizes
the directory once; each launch surface maps that canonical directory to its own
native placement (`cwd`, `--cwd`, or native orchestrator instructions) through
`place()` and reports structured refusal when it cannot place the directory.
"""
import json
import os
from pathlib import Path

import launch_values

REMOTE_SURFACES = ('dot', 'chatgpt-web', 'grok-bot')


def _wire_model(bundle):
    """The wire model ID `bundle` launches with (`launch_model` when present, else `model`)."""
    raw = (bundle or {}).get('launch_model') or (bundle or {}).get('model')
    return str(raw).strip() if isinstance(raw, str) and raw.strip() else ''


def _codex_wire_model(bundle):
    raw = _wire_model(bundle)
    return raw.removeprefix('openai-codex/') if (bundle or {}).get('provider') == 'pi' else raw


def _claude_alias_env_override(env, effort):
    """Return a caller environment variable that overrides Claude `Agent` alias/effort resolution, or None."""
    for var_name, val in env.items():
        clean = str(val or '').strip()
        if not clean:
            continue
        if ((var_name.startswith('ANTHROPIC_DEFAULT_') and var_name.endswith('_MODEL') and var_name != 'ANTHROPIC_DEFAULT_MODEL')
                or var_name in ('ANTHROPIC_SMALL_FAST_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL_FORCE')):
            return f'{var_name} is set in the caller environment'
        if var_name in ('CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY') and clean.lower() not in ('0', 'false'):
            return f'{var_name} selects a third-party Claude provider with different alias targets'
    base_url = str(env.get('ANTHROPIC_BASE_URL') or '').strip().rstrip('/')
    if base_url and not base_url.startswith('https://api.anthropic.com'):
        return 'ANTHROPIC_BASE_URL points outside first-party Anthropic API'
    env_effort = str(env.get('CLAUDE_CODE_EFFORT_LEVEL') or '').strip()
    if env_effort and env_effort.lower() != str(effort or '').lower():
        return f'CLAUDE_CODE_EFFORT_LEVEL={env_effort!r} overrides per-invocation effort'
    return None


def _orchestrator_account_refusal(caller_host, fam, account, accounts, env):
    """Refusal reason when an orchestrator server (`orca`, `t3`) cannot prove it will run `account` without an account flag."""
    import provider_inventory
    if account != fam:
        return f'{caller_host} has no account selector and runs only the single default {fam!r} account, not {account!r}'
    home = Path(env.get('ROUTER_PROVIDER_HOME') or env.get('HOME') or env.get('USERPROFILE') or Path.home())
    inv_accounts = (((accounts or {}).get('_discovery') or {}).get('accounts') or {})
    active_same = [
        acct_name for acct_name, acct_row in inv_accounts.items()
        if isinstance(acct_row, dict) and acct_row.get('family') == fam
        and acct_row.get('status', 'ready') in ('ready', 'usage_failing', 'plan_inactive')
    ]
    if len(active_same) > 1 or (len(active_same) == 1 and active_same[0] != fam):
        return (f'{caller_host} cannot disambiguate account {account!r} without an account/providerInstanceId selector '
                f'({", ".join(sorted(active_same))})')
    if len(active_same) == 1:
        sole_row = inv_accounts[fam]
        if 'orca' in (sole_row.get('sources') or ()):
            return f'{caller_host} cannot verify selection of Orca-managed account {account!r}'
        if fam in provider_inventory.SESSION_ACCOUNT_DIRS and sole_row.get('home'):
            default_rel = provider_inventory.SESSION_ACCOUNT_DIRS[fam][1]
            if Path(sole_row['home']).expanduser().resolve() != (home / default_rel).resolve():
                return f'{caller_host} runs the default ~/{default_rel} home, not {sole_row["home"]!r}'
    if caller_host == 'orca':
        if fam == 'codex' and provider_inventory.orca_codex_homes(home):
            return 'Orca has managed codex-accounts homes whose active selection in Orca GlobalSettings is unread'
        if fam == 'claude' and any((ud / 'claude-accounts').is_dir() for ud in provider_inventory.orca_user_data_dirs(home)):
            return 'Orca has managed claude-accounts profiles whose active selection in Orca GlobalSettings is unread'
    return ''


def native_handover_spec(parent_host, bundle, accounts=None, *, parent_model=None, environ=None):
    """Return `(spec, '')` when the caller's declared native tool (`account_fleet.NATIVE_HOST_TOOLS`) is known from
    `worker_context.caller_native_host` and can express `bundle` without ambiguity, else `(None, reason)`."""
    import account_fleet
    import model_resolution
    import provider_inventory
    import worker_context

    env = os.environ if environ is None else environ
    caller_host = worker_context.caller_native_host(parent_host, env)
    if caller_host is None:
        return None, (
            f'caller host tool could not be determined from environment for --parent-host {parent_host!r} '
            '(no unique session marker among CLAUDE_CODE_SESSION_ID/CLAUDECODE, CODEX_THREAD_ID/CODEX_SESSION_ID, '
            'ANTIGRAVITY_CONVERSATION_ID, ORCA_AGENT_SESSION_ID, T3_MCP_URL+T3_MCP_BEARER_TOKEN)'
        )
    decl = account_fleet.NATIVE_HOST_TOOLS.get(caller_host) or {}
    tool = decl.get('tool')
    if not tool:
        return None, f'caller host {caller_host!r} has no native sub-agent tool'
    if not isinstance(bundle, dict) or not (bundle.get('model') or bundle.get('launch_model')):
        return None, 'routed bundle has no model'
    if accounts is None:
        accounts = account_fleet.load_accounts()
    account = bundle.get('account') or bundle.get('provider')
    harness = account_fleet.bundle_harness(bundle, account, bundle.get('bundle_id') or bundle.get('id'))
    wire_model = _wire_model(bundle)
    if bundle.get('fast'):
        return None, f'{tool} has no fast-mode launch field for {wire_model!r}'
    effort = str(bundle.get('effort') or '').strip() or None
    if effort and harness in account_fleet.EFFORT_IN_MODEL:
        effort = None

    if decl.get('account_scope') == 'own_session':
        if harness != decl.get('harness'):
            return None, f"{tool} runs only on harness {decl.get('harness')!r}, not {harness!r} ({bundle.get('provider')!r})"
        inventory = (accounts or {}).get('_discovery') or {}
        own_account = provider_inventory.session_account(decl['session_family'], inventory, environ=env)
        if not own_account:
            return None, f"{tool} requires the calling session's {decl['session_family']!r} account to be uniquely discovered"
        if account != own_account:
            return None, f"{tool} runs only on the caller's own account {own_account!r}, not {account!r}"
        mode = decl.get('model_mode')
        if mode == 'line_target':
            if env_override := _claude_alias_env_override(env, effort):
                return None, f'{tool} alias cannot guarantee routed {wire_model!r}: {env_override}'
            given = str(parent_model or '').strip()
            if not given:
                return None, (
                    f"{tool} accepts only model line aliases whose resolved version depends on the caller's active model "
                    '(--parent-model was not given)'
                )
            families = model_resolution.family_patterns(accounts)
            hit = model_resolution.family_row(wire_model, families)
            parent_hit = model_resolution.family_row(given, families)
            known_lines = decl.get('known_lines') or ()
            line = hit.get('line') if isinstance(hit, dict) else None
            if line not in known_lines:
                return None, f'{tool} has no model line alias for {wire_model!r}'
            if not (isinstance(parent_hit, dict) and parent_hit.get('maker') == 'anthropic'
                    and parent_hit.get('line') in known_lines):
                return None, (
                    f'{tool} runs parent model {given!r} when the parent family is unknown to Claude Code, '
                    f'so alias {line!r} cannot select {wire_model!r}'
                )
            bundle_rk = account_fleet.row_key(bundle.get('model') or wire_model, accounts)
            parent_rk = account_fleet.row_key(given, accounts)
            if parent_hit is hit:
                if parent_rk != bundle_rk:
                    return None, (
                        f'{tool} alias {line!r} inherits same-family parent model {given!r} ({parent_rk}), '
                        f'not routed {wire_model!r} ({bundle_rk})'
                    )
            else:
                target = (hit.get('models') or [None])[0]
                target_rk = account_fleet.row_key(target, accounts) if target else None
                if not target_rk or bundle_rk != target_rk:
                    return None, (
                        f'{tool} alias {line!r} from parent {given!r} resolves to {target or "none"!r}, '
                        f'not routed {wire_model!r}'
                    )
            fields = {decl['model_field']: line}
            if decl.get('effort_field') and effort:
                fields[decl['effort_field']] = effort
            return {'host': caller_host, 'tool': tool, 'launch_fields': fields}, ''
        if mode == 'caller_own_model':
            given = str(parent_model or '').strip()
            if not given:
                return None, (
                    f"{tool} accepts only Model='inherit' (the caller's own active model), and the host exposes no "
                    'session model in the environment (--parent-model was not given)'
                )
            if given != wire_model and account_fleet.row_key(given, accounts) != account_fleet.row_key(wire_model, accounts):
                return None, f"{tool} Model='inherit' runs caller model {given!r}, not routed {wire_model!r}"
            return {'host': caller_host, 'tool': tool, 'launch_fields': {decl['model_field']: 'inherit'}, 'hook_model': wire_model}, ''
        if decl.get('effort_field') and not effort:
            return None, f"{tool} requires an explicit {decl['effort_field']} so caller config cannot override it"
        acct_row = ((inventory.get('accounts') or {}).get(own_account)) or {}
        if isinstance(acct_row, dict) and acct_row.get('home'):
            cfg_path = Path(acct_row['home']) / 'config.toml'
            if cfg_path.is_file():
                try:
                    cfg_doc = provider_inventory.read_toml(cfg_path.read_text(encoding='utf-8', errors='replace'))
                except OSError:
                    cfg_doc = {}
                for tbl in ('agents', 'agent_roles'):
                    role_tbl = cfg_doc.get(tbl) if isinstance(cfg_doc, dict) else None
                    default_role = role_tbl.get('default') if isinstance(role_tbl, dict) else None
                    if isinstance(default_role, dict) and default_role.get('config_file'):
                        return None, f'{tool} default role in {cfg_path} sets config_file and can override requested model/effort'
        fields = {decl['model_field']: _codex_wire_model(bundle)}
        if decl.get('effort_field') and effort:
            fields[decl['effort_field']] = effort
        return {'host': caller_host, 'tool': tool, 'launch_fields': fields}, ''

    if caller_host == 'orca':
        if harness not in decl['harnesses']:
            return None, f"Orca worker-start supports harnesses {decl['harnesses']}, not {harness!r}"
        if effort and harness in (decl.get('effort_unsupported_harnesses') or ()):
            return None, f'Orca worker-start does not support --effort for harness {harness!r}'
        fam = account_fleet.harness_family(harness)
        if acct_refusal := _orchestrator_account_refusal('Orca worker-start', fam, account, accounts, env):
            return None, acct_refusal
        agent_name = decl['agent_names'].get(harness, harness)
        fields = {decl['agent_field']: agent_name, decl['model_field']: wire_model}
        if effort:
            fields[decl['effort_field']] = effort
        return {'host': 'orca', 'tool': tool, 'launch_fields': fields}, ''

    if caller_host == 't3':
        driver = (decl.get('drivers') or {}).get(harness)
        if not driver:
            return None, f"T3 delegate_task has no built-in driverKind for harness {harness!r}"
        opt_id = (decl.get('option_effort_ids') or {}).get(harness)
        if effort and not opt_id:
            return None, f'T3 delegate_task has no verified effort option for harness {harness!r}'
        fam = account_fleet.harness_family(harness)
        if acct_refusal := _orchestrator_account_refusal(f'T3 target.driverKind={driver!r}', fam, account, accounts, env):
            return None, acct_refusal
        target_obj = {'driverKind': driver, 'model': wire_model}
        if opt_id and effort:
            target_obj['options'] = {opt_id: effort}
        return {'host': 't3', 'tool': tool, 'launch_fields': {'target': target_obj}}, ''

    return None, f'unsupported native host {caller_host!r}'


def parent_cwd():
    """The parent's directory: a local worker without `--workspace` starts here, whatever surface launches it. Paseo
    gives the agent PASEO_AGENT_ID names its directory in PASEO_AGENT_CWD; without that id the variable is no parent's
    of this roll (a Herdr, native or cli launch run outside the Paseo session inherits it), and the worker starts where
    the command runs."""
    if (os.environ.get('PASEO_AGENT_ID') or '').strip() and os.environ.get('PASEO_AGENT_CWD'):
        return os.environ['PASEO_AGENT_CWD']
    return os.getcwd()


def worker_directory(workspace, worker=None):
    """The one answer to where a local worker runs: the canonical `--workspace` when given (a resumed worker too: the
    host resumes a worker only in its own directory); else, for a worker roll resumes (`worker`, the registry's answer
    for it, `dispatch_registry.reusable_workers`: its Paseo record's cwd, else its latest row's, else the request's,
    then its recorded workspace), that directory, None when the registry knows none; else `parent_cwd()`. roll puts it
    in the task (`task['cwd']`), so the plan, its re-plan at the launch, `record` and a named roll match an account's
    `cwd_regex` against it, and every launch surface places the worker there (`place`)."""
    if ws := canonical_dir(workspace):
        return ws
    if worker is not None:
        own = worker.get('cwd') or worker.get('workspace')
        return own if isinstance(own, str) and own.strip() else None
    return parent_cwd()


class InvalidWorkspace(ValueError):
    """`--workspace` was given a value that is not an existing directory."""

    def __init__(self, raw, message):
        super().__init__(message)
        self.raw = raw


def canonical_dir(raw_dir):
    """Canonical path string (resolved, normpath, normcased) for directory comparisons, or None."""
    if raw_dir is None:
        return None
    text = str(raw_dir).strip()
    if not text:
        return None
    expanded = Path(text).expanduser()
    try:
        resolved = expanded.resolve()
    except (OSError, RuntimeError, ValueError):
        resolved = Path(os.path.abspath(str(expanded)))
    return os.path.normcase(os.path.normpath(str(resolved)))


def validate_workspace(raw_dir):
    """Validate `--workspace` once before classification, allocation or launch; return canonical_dir."""
    if raw_dir is None:
        return None
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise InvalidWorkspace(raw_dir, '--workspace requires an existing directory path.')
    raw_text = raw_dir.strip()
    candidate = Path(raw_text).expanduser()
    try:
        exists = candidate.exists()
        is_dir = candidate.is_dir()
    except OSError as exc:
        raise InvalidWorkspace(raw_dir, f'--workspace path could not be accessed ({raw_text}): {exc}') from exc
    if not is_dir:
        if exists:
            raise InvalidWorkspace(raw_dir, f'--workspace must be an existing directory, not a file: {raw_text}')
        hint = (' (--workspace takes a directory path, never a host workspace ID such as wks_...)'
                if raw_text.startswith('wks_') else '')
        raise InvalidWorkspace(raw_dir, f'--workspace directory does not exist: {raw_text}{hint}')
    return canonical_dir(candidate)


def native_next(host, bundle, address, workspace=None, *, native_spec=None):
    """What the calling session does to start the routed worker with its own host's native tool and `launch_fields`."""
    import account_fleet
    fields = (native_spec or {}).get('launch_fields') or {}
    tool = (native_spec or {}).get('tool')
    wire_model = _wire_model(bundle)
    provider = bundle.get('provider')
    effort = None if bundle.get('harness') == 'antigravity-resilient' else bundle.get('effort')
    if host == 'orca' or tool == 'shell:orca worker launch':
        agent_val = fields.get('--agent') or account_fleet.harness_family(provider)
        model_val = fields.get('--model') or wire_model
        effort_val = fields.get('--effort') if fields else effort
        flags = (f'--agent {launch_values.shell_word(agent_val)} --worktree current'
                 + (f' --model {launch_values.shell_word(model_val)}' if model_val else '')
                 + (f' --effort {launch_values.shell_word(effort_val)}' if effort_val else ''))
        where = (f' Run the command from directory `{workspace}` so `--worktree current` places the worker there.'
                 if workspace else '')
        return ('Start the worker with Orca orchestration, once, using the executable you ran `skills get` with: '
                f'`orca orchestration worker-start --spec "<contents of worker_prompt_file>" {flags} --json` (run `run-create` first if no '
                f'Run is bound).{where} Use no other sub-agent tool. The worker reports to {address} through its injected preamble; collect it with '
                '`orca orchestration check --wait --types worker_done,escalation,question --json`.')
    if host == 't3' or tool == 'delegate_task':
        target_json = json.dumps((fields.get('target') if isinstance(fields.get('target'), dict)
                                  else {'driverKind': provider, 'model': wire_model}), ensure_ascii=False)
        where = f', working directory `{workspace}`' if workspace else ''
        return ('Start the worker with the T3 Code MCP tool `delegate_task` (`orchestrator_capabilities` first, once per thread): '
                f'pass `task` = full text of `worker_prompt_file`, `target` = `{target_json}` (`launch_fields.target`){where}, mode `async`. '
                'T3 delivers the result to this parent thread; do not poll `task_status` or use another sub-agent tool.')
    where = f' in working directory `{workspace}`' if workspace else ''
    mode_note = f' (and permission `modeId` `{bundle["modeId"]}` from `route_bundle` when supported)' if bundle.get('modeId') else ' (including `route_bundle.modeId` when supported)'
    if tool and fields:
        fields_json = json.dumps(fields, ensure_ascii=False)
        version_note = (
            ' Per-invocation `effort` on `Agent` requires Claude Code v2.1.292+; on an older version whose `Agent` schema '
            'lacks `effort`, do not omit `effort`—re-run `roll` without `--parent-host native` so AgentCompass launches '
            'the worker directly, or ask the user.'
            if tool == 'Agent' and 'effort' in fields else ''
        )
        return (f'Launch one native sub-agent with `{tool}`{where}, passing `launch_fields` `{fields_json}`{mode_note} '
                'and the full contents of `worker_prompt_file` as its prompt (not the filename). '
                f'Preserve parent address `{address}` and the injected worker preamble; do not launch a second worker on this hand-over.{version_note}')
    return (f'Launch the native host worker{where} with exactly route.bundle{mode_note} '
            'and the contents of worker_prompt_file as --spec (not the filename). Preserve '
            'the live coordinator inbox, parent session ID and injected worker preamble; do not switch hosts.')


def place(surface, workspace, *, default_cwd=None, bundle=None, address=None, paseo_probe=None, native_spec=None):
    """Map canonical `workspace` (or None) to host-native placement for `surface`: a remote runner
    (REMOTE_SURFACES), `paseo`, `herdr`, `cli` or a native host (worker_context.NATIVE_HOSTS), the surfaces roll builds."""
    if surface in REMOTE_SURFACES:
        if workspace:
            return {
                'ok': False,
                'surface': surface,
                'workspace': workspace,
                'error': 'host_cannot_place_directory',
                'caller_action': 'check_error',
                'next': (f'Nothing was launched: {surface} runs remotely and cannot place a worker in local directory '
                         f'{workspace}. Omit --workspace for remote work, or route to a local provider.'),
            }
        return {'ok': True, 'surface': surface, 'workspace': None, 'cwd': None}
    if surface == 'paseo':
        if workspace and isinstance(paseo_probe, dict) and paseo_probe.get('placeable') is False:
            err = paseo_probe.get('error') or 'paseo_workspace_not_found_for_directory'
            enclosing = [str(d) for d in (paseo_probe.get('enclosing_directories') or ()) if d]
            if err == 'paseo_workspaces_unreadable':
                nxt = (f'Nothing was launched: Paseo workspaces could not be read to verify placement for {workspace}, '
                       'and Paseo cannot place a sub-agent in an unverified directory. Check the local Paseo daemon, '
                       'then roll again.')
            else:
                enc_note = (f' Paseo only has workspace(s) rooted at {", ".join(enclosing)} '
                            f'(never auto-picked, because Paseo would place the worker there instead of {workspace}).'
                            if enclosing else '')
                nxt = (f'Nothing was launched: Paseo has no workspace rooted at {workspace}, and Paseo createAgent '
                       f'with callerAgentId cannot place a sub-agent in an arbitrary directory without a matching '
                       f'workspace.{enc_note} Register {workspace} as a Paseo workspace first, or omit --workspace '
                       'when the worker creates its own worktree.')
            return {
                'ok': False,
                'surface': 'paseo',
                'workspace': workspace,
                'error': err,
                'caller_action': 'check_error',
                'next': nxt,
            }
        landing = workspace or default_cwd
        return {
            'ok': True,
            'surface': 'paseo',
            'workspace': workspace,
            'cwd': landing,
            'launch': {'cwd': workspace} if workspace else {},
        }
    if surface in ('herdr', 'cli'):
        landing = workspace or default_cwd or str(Path.cwd())
        return {'ok': True, 'surface': surface, 'workspace': workspace, 'cwd': landing}
    # A native host (worker_context.NATIVE_HOSTS): roll builds no other surface.
    landing = workspace or default_cwd or str(Path.cwd())
    return {
        'ok': True,
        'surface': surface,
        'workspace': workspace,
        'cwd': landing,
        'next': native_next(surface, bundle or {}, address, workspace=workspace, native_spec=native_spec),
    }
