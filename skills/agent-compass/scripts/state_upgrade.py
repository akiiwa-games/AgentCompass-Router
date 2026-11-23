"""Read-only upgrade checks for local ~/.agent-compass state, hooks and global rule blocks.

Every diagnostic names the offending file and a fix that works when followed:
  - re-run "agentcompass setup"
  - edit the file (what to change is named)
  - delete "<file>", with what the deletion discards when the file holds something the user chose
Only what needs the user is listed: a cache or database the product rebuilds or skips itself is not.

Runtime loaders call the same validators so an unreadable or wrong-schema file
fails fast with the same message `agentcompass doctor` prints.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

FIX_SETUP = 're-run "agentcompass setup"'
LEGACY_SETTINGS_REASON = (
    '旧设置残留。在设置里保存一次即可清掉，或运行 agentcompass setup。'
)

# Overlay words of an earlier release (v0.1.0 to v0.5.0) and the name they carry today: the release's own rewrite of the
# command placeholder, the variable names it renamed (AGENTCOMPASS_* is the one canonical name), and the skill rename.
# A token with no current name (a bare script path placeholder, a script file name, ROUTE_MANAGER_*) is not rewritten.
HOST_RULE_UPGRADES = {
    'python -X utf8 @@ROUTE_JUDGE@@': '@@AGENTCOMPASS@@',
    'ROUTER_WORKER': 'AGENTCOMPASS_WORKER',
    'ROUTER_ACCOUNTS_OVERRIDE': 'AGENTCOMPASS_ACCOUNTS_OVERRIDE',
    'ROUTER_LIVE_MODELS': 'AGENTCOMPASS_LIVE_MODELS',
    'route-agent-work': 'agent-compass',
}
# The files `host_rules.load` reads; the others in `host-rules/` (dispatch.md, personal-routing-policy.md) are the user's own
# documents, which no release or check reads as an overlay.
HOST_RULE_OVERLAYS = ('shared.md', 'codex.md', 'daemon-append.md')
# `judges` was read by the select and delegate stages (removed 2026-11-21); a file a released version wrote may still carry
# it, so it is accepted and kept as written, and nothing reads it.
HOST_POLICY_KEYS = frozenset({'allowed_accounts', 'judges', 'usage_providers', 'known_accounts'})
RETIRED_PLACEHOLDER = '@@ROUTE_JUDGE@@'     # no current meaning on its own; `python -X utf8 @@ROUTE_JUDGE@@` is renamed above
RULE_BEGIN = '<!-- agentcompass:begin -->'
RULE_END = '<!-- agentcompass:end -->'
# What an earlier release's rule block held and a current render never does (an overlay's words are carried through
# `upgrade_host_rule_text`, and the retired placeholder is refused); a name the user wrote in prose stays in the block.
STALE_RULE_TOKENS = (*HOST_RULE_UPGRADES, RETIRED_PLACEHOLDER)
LEGACY_HOOK_TOKENS = (
    'route-agent-work',
    'cli/_internal/agentcompass/data/setup/router_gate_hook.py',
    'ROUTER_WORKER',
    'ROUTER_PASEO_LEGACY_NO_LABELS',
    'ROUTER_ACCOUNTS_OVERRIDE',
)
OBSOLETE_USAGE_CONFIG_KEYS = (
    'llm_proxy_timezone',
    'llm_proxy_budget_duration',
    'llm_proxy_api_key',
    'llm_proxy_endpoint',
    'llm_proxy_provider_id',
    'litellm_proxies',
)


def fix_delete(path: Path | str, discards: str | None = None) -> str:
    """The delete fix; `discards` says what the user loses by it when the file holds something they chose."""
    return f'delete "{path}"' + (f' ({discards})' if discards else '')


class StateIssue(ValueError):
    """The local state under ~/.agent-compass refused on purpose: a file in a shape this version does not read or
    cannot carry over. Its text is `format_issue`'s (the file, what is wrong, the fix that works when followed), so a
    reader that tells a refusal from a defect of AgentCompass (dispatch_fallback.refusals) passes it on whole."""


def format_issue(path: Path | str, detail: str, fix: str) -> str:
    return f'{path}: {detail}. Fix: {fix}.'


def _env(environ: dict | None) -> dict:
    return os.environ if environ is None else environ


def state_dir(home: Path | None = None, environ: dict | None = None) -> Path:
    env = _env(environ)
    configured = str(env.get('ROUTER_STATE_DIR') or '').strip()
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    base = Path(home) if home is not None else Path.home()
    return base / '.agent-compass'


def obsolete_usage_config_keys(doc) -> list[str]:
    if not isinstance(doc, dict):
        return []
    return [key for key in OBSOLETE_USAGE_CONFIG_KEYS if key in doc]


def usage_config_issue(path: Path | str, keys) -> str:
    listed = ', '.join(keys)
    return format_issue(
        path,
        f'usage config still has llm-proxy settings that no longer apply ({listed}). {LEGACY_SETTINGS_REASON}',
        FIX_SETUP,
    )


def clear_obsolete_usage_file(path: Path) -> list[str]:
    """Drop leftover llm-proxy keys from one settings file. Every other key stays."""
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, ValueError):
        return []
    if not isinstance(doc, dict):
        return []
    keys = obsolete_usage_config_keys(doc)
    if not keys:
        return []
    for key in keys:
        doc.pop(key, None)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    return keys


def check_usage_config(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding='utf-8-sig')
    except (OSError, UnicodeError):
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    keys = obsolete_usage_config_keys(doc)
    if not keys:
        return None
    return usage_config_issue(path, keys)


def check_accounts_override(path: Path) -> str | None:
    if not path.exists():
        return None
    import accounts_override
    try:
        accounts_override.load_override(path)
    except ValueError as exc:
        return str(exc)
    return None


def _path_writable(path: Path) -> bool:
    """Read-only writability check without creating or modifying files."""
    if path.exists():
        return os.access(path, os.W_OK)
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return os.access(parent, os.W_OK)


def check_legacy_disabled_upgrade(accounts_path: Path, policy_path: Path) -> str | None:
    """Read-only: flag when a lingering `verified: false` cannot be upgraded because a target file is not writable."""
    if not accounts_path.exists():
        return None
    import accounts_override
    try:
        doc = accounts_override.load_override(accounts_path)
    except (OSError, ValueError):
        return None
    flagged = accounts_override.legacy_disabled_accounts(doc)
    if not flagged:
        return None
    unwritable = [p for p in (policy_path, accounts_path) if not _path_writable(p)]
    if not unwritable:
        return None
    target = unwritable[0]
    extra = f' (and {unwritable[1]})' if len(unwritable) > 1 else ''
    return format_issue(
        target,
        f'legacy verified=false account-off setting ({", ".join(flagged)}) cannot be upgraded because {target}{extra} '
        'is not writable (the exclusion stays enforced in memory)',
        f'make "{target}" writable and run "agentcompass providers" or "agentcompass setup"',
    )


def host_policy_fix(path: Path | str) -> str:
    return ('edit the file to a JSON object with "allowed_accounts" (a list of account names) and, if wanted, '
            '"known_accounts", "usage_providers" (lists of names), or '
            + fix_delete(path, 'this discards which accounts take part in automatic routing on this machine; every '
                               'account does again until you set them in the tray'))


def validate_host_policy_doc(doc, path: Path | str) -> dict:
    if not isinstance(doc, dict) or 'allowed_accounts' not in doc:
        raise ValueError(format_issue(
            path, 'host-policy configuration requires allowed_accounts', host_policy_fix(path),
        ))
    unknown = sorted(set(doc) - HOST_POLICY_KEYS)
    if unknown:
        raise ValueError(format_issue(
            path, f'host-policy configuration has unsupported fields ({", ".join(unknown)})', host_policy_fix(path),
        ))
    for key, value in doc.items():
        if not isinstance(value, list) or any(not isinstance(v, str) or not v or v != v.strip() for v in value):
            raise ValueError(format_issue(
                path, f'host-policy configuration {key} must be a list of nonempty names', host_policy_fix(path),
            ))
    return doc


def check_host_policy(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding='utf-8-sig')
    except UnicodeError:
        return format_issue(path, 'host-policy configuration must be UTF-8', host_policy_fix(path))
    except OSError as exc:
        return format_issue(path, f'host-policy configuration unreadable ({exc})', host_policy_fix(path))
    try:
        doc = json.loads(raw)
    except ValueError:
        return format_issue(path, 'host-policy configuration invalid JSON', host_policy_fix(path))
    try:
        validate_host_policy_doc(doc, path)
    except ValueError as exc:
        return str(exc)
    return None


def check_user_scores(path: Path) -> str | None:
    """The file's shape, and the names in it that resolve to no row of the shipped catalog: those entries are kept and
    ignored for scoring (user_scores.apply_overlay), and named here once."""
    if not path.exists():
        return None
    import public_data
    import user_scores
    try:
        doc = user_scores.load_local(path)
    except ValueError as exc:
        return str(exc)
    try:
        accounts = public_data.document(user_scores.REFERENCES / 'accounts.json')
    except (OSError, ValueError):
        return None
    unknown = user_scores.unresolved_models(doc, accounts)
    if not unknown:
        return None
    return format_issue(path, 'scores kept but ignored: no model of this name in the catalog (' + ', '.join(unknown) + ')',
                        "move each score to the model's current name with `agentcompass score set` and clear the old one "
                        'with `agentcompass score clear`, or leave it until the model is in the catalog')


def upgrade_host_rule_text(text: str) -> str:
    """An overlay an earlier release accepted, in today's words; the file itself is the user's and is never rewritten."""
    for old, new in HOST_RULE_UPGRADES.items():
        text = text.replace(old, new)
    return text


def overlay_fix(path: Path) -> str:
    return (f'edit the overlay: replace {RETIRED_PLACEHOLDER} with `@@AGENTCOMPASS@@` (the command), or '
            + fix_delete(path, 'this discards your overlay: the shipped default rules are used instead'))


def read_host_rule_file(path: Path) -> str:
    """The text of one ``host-rules`` overlay carried into the current names (`upgrade_host_rule_text`). The overlay is
    the user's own text, so a name mentioned in it is left as written. Raises ``ValueError`` for a file that is
    unreadable or still uses the retired placeholder, which cannot be rendered."""
    try:
        text = path.read_bytes().decode('utf-8-sig')    # bytes, not text mode: the overlay keeps its line endings
    except (OSError, UnicodeError) as exc:
        raise ValueError(format_issue(path, f'host-rules file unreadable ({exc})', overlay_fix(path))) from exc
    text = upgrade_host_rule_text(text)
    if RETIRED_PLACEHOLDER in text:
        raise ValueError(format_issue(
            path, f'host-rules overlay uses the placeholder {RETIRED_PLACEHOLDER}, which has no current meaning',
            overlay_fix(path),
        ))
    return text


def check_host_rule_file(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        read_host_rule_file(path)
    except ValueError as exc:
        return str(exc)
    return None


def _validate_profile(profile) -> None:
    import task_intake
    task_intake.validate_profile(profile)


def row_has_old_profile(row: dict) -> bool:
    """True when a ledger row carries a task profile that fails the four-category schema."""
    if not isinstance(row, dict):
        return False
    snap = row.get('decision_snapshot')
    if isinstance(snap, dict):
        snap_task = snap.get('task')
        if isinstance(snap_task, dict) and 'profile' in snap_task and snap_task['profile'] is not None:
            try:
                _validate_profile(snap_task['profile'])
            except ValueError:
                return True
    if 'profile' in row and row['profile'] is not None:
        try:
            _validate_profile(row['profile'])
        except ValueError:
            return True
    return False


def check_dispatch_db(path: Path) -> str | None:
    """Only a database sqlite cannot read. One an older version made is not a fault: missing tables are created on use
    (`dispatch_registry.transaction`) and a decision recorded under an older task profile is refused on its own
    (`dispatch_registry.prepare`) without touching the rest."""
    if not path.exists():
        return None
    try:
        uri = path.resolve().as_uri() + '?mode=ro'
        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
    except (OSError, sqlite3.Error) as exc:
        return format_issue(path, f'dispatch database unreadable ({exc})', _dispatch_db_fix(path))
    try:
        conn.execute("SELECT name FROM sqlite_master").fetchall()
    except sqlite3.Error as exc:
        return format_issue(path, f'dispatch database corrupt ({exc})', _dispatch_db_fix(path))
    finally:
        conn.close()
    return None


def _dispatch_db_fix(path: Path) -> str:
    return fix_delete(path, 'this discards the recorded decisions and the reservations of workers still running; '
                            'a new database is created on the next dispatch')


def _overrides_fix(path: Path) -> str:
    return fix_delete(path, 'this discards the temporary priorities it holds')


def check_provider_inventory(path: Path) -> str | None:
    if not path.exists():
        return None
    import provider_inventory
    try:
        provider_inventory.previous_inventory(path)
    except ValueError as exc:
        return str(exc)
    return None


def check_router_overrides(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, ValueError) as exc:
        return format_issue(path, f'router overrides invalid JSON ({exc})', _overrides_fix(path))
    if not isinstance(doc, dict):
        return format_issue(path, 'router overrides must be a JSON object', _overrides_fix(path))
    rows = doc.get('overrides')
    if rows is not None and not isinstance(rows, list):
        return format_issue(path, 'router overrides.overrides must be a JSON array', _overrides_fix(path))
    for item in rows or ():
        if not isinstance(item, dict):
            return format_issue(path, 'router override entry must be a JSON object', _overrides_fix(path))
    return None


def _check_global_rule_text(path: Path | str, text: str) -> str | None:
    if RULE_BEGIN in text and RULE_END in text:
        start = text.index(RULE_BEGIN)
        end = text.index(RULE_END, start) + len(RULE_END)
        managed = text[start:end]
        found = [tok for tok in STALE_RULE_TOKENS if tok in managed]
        if found:
            return format_issue(
                path,
                f'managed global rule block contains older token(s): {", ".join(found)}',
                FIX_SETUP,
            )
        return None
    stripped = text.strip()
    if stripped.startswith('# Agent Routing (Global)') or RETIRED_PLACEHOLDER in text or 'route-agent-work' in text:
        return format_issue(
            path,
            'older un-markered global routing rule block',
            FIX_SETUP,
        )
    return None


def check_hooks_and_rules(home: Path, environ: dict | None = None) -> list[str]:
    issues: list[str] = []
    hook_files = [
        home / '.claude' / 'settings.json',
        home / '.codex' / 'hooks.json',
        home / '.gemini' / 'config' / 'hooks.json',
        home / '.pi' / 'agent' / 'extensions' / 'router-gate.ts',
        home / '.config' / 'opencode' / 'plugins' / 'router-gate.js',
    ]
    for pattern, filename in (('.claude-*', 'settings.json'), ('.codex-*', 'hooks.json')):
        try:
            dirs = sorted(p for p in home.glob(pattern) if p.is_dir())
        except OSError:
            dirs = []
        for d in dirs:
            hook_files.append(d / filename)

    bundled = None
    try:
        from agentcompass import install
        bundled = install.bundled_cli(environ)
    except Exception:
        bundled = None

    for hpath in hook_files:
        if not hpath.is_file():
            continue
        try:
            text = hpath.read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            issues.append(format_issue(hpath, f'hook configuration unreadable ({exc})', FIX_SETUP))
            continue
        norm = text.replace('\\\\', '/').replace('\\', '/')
        bad_tokens = [tok for tok in LEGACY_HOOK_TOKENS if tok in norm]
        if bad_tokens:
            issues.append(format_issue(
                hpath, f'installed hook references older path/variable ({", ".join(bad_tokens)})', FIX_SETUP,
            ))
            continue
        if ('router_gate_hook.py' in norm or 'gate-hook' in norm) and bundled is not None:
            if str(bundled).replace('\\', '/').lower() not in norm.lower():
                issues.append(format_issue(
                    hpath, 'installed hook does not use the bundled agentcompass.exe', FIX_SETUP,
                ))

    rule_files = [
        home / '.claude' / 'CLAUDE.md',
        home / '.codex' / 'AGENTS.md',
        home / '.gemini' / 'GEMINI.md',
        home / '.gemini' / 'config' / 'GEMINI.md',
        home / '.pi' / 'agent' / 'AGENTS.md',
        home / '.config' / 'opencode' / 'AGENTS.md',
    ]
    for pattern, filename in (('.claude-*', 'CLAUDE.md'), ('.codex-*', 'AGENTS.md')):
        try:
            dirs = sorted(p for p in home.glob(pattern) if p.is_dir())
        except OSError:
            dirs = []
        for d in dirs:
            rule_files.append(d / filename)

    for rpath in rule_files:
        if not rpath.is_file():
            continue
        try:
            text = rpath.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        issue = _check_global_rule_text(rpath, text)
        if issue:
            issues.append(issue)

    env = _env(environ)
    paseo_home = Path(env.get('PASEO_HOME') or (home / '.paseo'))
    paseo_cfg = paseo_home / 'config.json'
    if paseo_cfg.is_file():
        try:
            pdoc = json.loads(paseo_cfg.read_text(encoding='utf-8-sig'))
            prompt = ((pdoc.get('daemon') or {}).get('appendSystemPrompt')) if isinstance(pdoc, dict) else None
            if isinstance(prompt, str) and prompt:
                issue = _check_global_rule_text(paseo_cfg, prompt)
                if issue:
                    issues.append(issue)
        except (OSError, UnicodeError, ValueError):
            pass

    skill_roots = [
        home / '.claude' / 'skills',
        home / '.codex' / 'skills',
        home / '.agents' / 'skills',
        home / '.gemini' / 'skills',
    ]
    for pattern in ('.claude-*', '.codex-*'):
        try:
            dirs = sorted(p for p in home.glob(pattern) if p.is_dir())
        except OSError:
            dirs = []
        for d in dirs:
            skill_roots.append(d / 'skills')
    for sroot in skill_roots:
        old_skill = sroot / 'route-agent-work'
        if old_skill.exists() or old_skill.is_symlink():
            # Setup retires only the links its own manifest records; this one predates the manifest.
            issues.append(format_issue(
                old_skill, 'older skill link route-agent-work (renamed to agent-compass)', fix_delete(old_skill),
            ))
        cur_skill = sroot / 'agent-compass'
        try:
            if cur_skill.exists() or cur_skill.is_symlink():
                resolved = str(cur_skill.resolve()).replace('\\', '/')
                if 'cli/_internal/agentcompass/data/skills' in resolved or not (cur_skill / 'SKILL.md').is_file():
                    issues.append(format_issue(
                        cur_skill, 'skill link points to an older or missing install layout', FIX_SETUP,
                    ))
        except OSError:
            issues.append(format_issue(cur_skill, 'broken skill link', FIX_SETUP))

    return issues


def collect_issues(home: Path | None = None, environ: dict | None = None) -> list[str]:
    """Return per-file upgrade instructions for any old-shape local state. Never writes."""
    env = _env(environ)
    base_home = Path(home) if home is not None else Path.home()
    sdir = state_dir(base_home, env)
    issues: list[str] = []

    override_env = str(env.get('AGENTCOMPASS_ACCOUNTS_OVERRIDE') or '').strip()
    accounts_path = Path(os.path.expandvars(override_env)).expanduser() if override_env else sdir / 'accounts-override.json'
    policy_env = str(env.get('ROUTER_HOST_POLICY') or '').strip()
    policy_path = Path(os.path.expandvars(policy_env)).expanduser() if policy_env else sdir / 'host-policy.json'
    db_env = str(env.get('ROUTER_DISPATCH_DB') or '').strip()
    db_path = Path(os.path.expandvars(db_env)).expanduser() if db_env else sdir / 'dispatch.sqlite3'
    overrides_env = str(env.get('ROUTER_OVERRIDES') or '').strip()
    overrides_path = Path(os.path.expandvars(overrides_env)).expanduser() if overrides_env else sdir / 'router-overrides.json'

    for issue in (
        check_accounts_override(accounts_path),
        check_legacy_disabled_upgrade(accounts_path, policy_path),
        check_usage_config(sdir / 'usage-config.json'),
        check_host_policy(policy_path),
        check_user_scores(sdir / 'user-scores.json'),
        check_dispatch_db(db_path),
        check_router_overrides(overrides_path),
        check_provider_inventory(sdir / 'provider-inventory.json'),
    ):
        if issue:
            issues.append(issue)

    for name in HOST_RULE_OVERLAYS:
        issue = check_host_rule_file(sdir / 'host-rules' / name)
        if issue:
            issues.append(issue)

    issues.extend(check_hooks_and_rules(base_home, env))
    return issues
