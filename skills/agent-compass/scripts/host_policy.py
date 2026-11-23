"""Local dispatch availability. No profile preserves the shared routing defaults."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import account_state
import accounts_override
import override_store
import state_upgrade


class HostPolicyError(ValueError):
    """Safe, local configuration diagnostic (never includes profile contents)."""


def policy_path():
    configured = os.environ.get('ROUTER_HOST_POLICY')
    if not configured and os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
                configured = winreg.QueryValueEx(key, 'ROUTER_HOST_POLICY')[0]
        except OSError:
            pass
    if configured and not isinstance(configured, str):
        raise HostPolicyError('ROUTER_HOST_POLICY must be a path string.')
    return (Path(configured).expanduser(), True) if configured else (Path.home() / '.agent-compass/host-policy.json', False)


# Named upgrade step for the v0.1.0 to v0.5.0 account-off flag (`accounts.<name>.verified == False` in
# accounts-override.json), which v0.5.1 moved to `host-policy.json` (`allowed_accounts` + `known_accounts`).
LEGACY_DISABLED_UPGRADE = 'accounts-override verified=false -> host-policy allowed_accounts (v0.1.0 to v0.5.0)'


def _override_path(override_path=None):
    if override_path is not None:
        return Path(override_path)
    import account_fleet
    return account_fleet.accounts_override_path()


def _legacy_disabled_from_override(override_path=None):
    """Read-only: account names carrying `verified: false` in accounts-override.json, else []."""
    path = _override_path(override_path)
    if not path.exists():
        return []
    try:
        doc = accounts_override.load_override(path)
    except (OSError, ValueError):
        return []
    return accounts_override.legacy_disabled_accounts(doc)


def _read_policy_file(path, explicit=False):
    try:
        raw = path.read_text(encoding='utf-8-sig')
    except FileNotFoundError:
        if not explicit:
            return None
        raise HostPolicyError(state_upgrade.format_issue(
            path, 'host-policy configuration missing',
            'create it as a JSON object with "allowed_accounts" (a list of account names), or unset ROUTER_HOST_POLICY',
        )) from None
    except UnicodeError:
        raise HostPolicyError(state_upgrade.format_issue(
            path, 'host-policy configuration must be UTF-8', state_upgrade.host_policy_fix(path),
        )) from None
    except OSError as exc:
        raise HostPolicyError(state_upgrade.format_issue(
            path, f'host-policy configuration unreadable ({exc})', state_upgrade.host_policy_fix(path),
        )) from None
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise HostPolicyError(state_upgrade.format_issue(
            path, 'host-policy configuration invalid JSON', state_upgrade.host_policy_fix(path),
        )) from None
    try:
        return state_upgrade.validate_host_policy_doc(data, path)
    except ValueError as exc:
        raise HostPolicyError(str(exc)) from None


def load_policy():
    path, explicit = policy_path()
    doc = _read_policy_file(path, explicit=explicit)
    if doc is None:
        return None
    flagged = _legacy_disabled_from_override()
    if not flagged:
        return doc
    result = dict(doc)
    result['allowed_accounts'] = [name for name in result['allowed_accounts'] if name not in flagged]
    if 'known_accounts' in result:
        result['known_accounts'] = list(dict.fromkeys([*result['known_accounts'], *flagged]))
    return result


def upgrade_legacy_disabled(accounts=None, *, override_path=None, policy_file=None, read_only=False):
    """Carry v0.1.0..v0.5.0 `accounts.<name>.verified == False` from accounts-override.json into host-policy.json
    (`allowed_accounts` omitting `<name>`, `known_accounts` keeping `<name>` so `sync_discovered` does not re-add it),
    then remove `verified: false` from accounts-override.json while keeping every other field.

    Process-safe (locks `policy_file` then `override_path`), crash-safe (writes `host-policy.json` first), and
    idempotent. With `read_only=True`, inspects without writing and returns the flagged names.
    """
    ov_path = _override_path(override_path)
    pol_path = Path(policy_file) if policy_file is not None else policy_path()[0]
    flagged = _legacy_disabled_from_override(ov_path)
    if not flagged or read_only:
        return flagged
    with override_store.locked(pol_path), override_store.locked(ov_path):
        override_doc = accounts_override.load_override(ov_path)
        flagged = accounts_override.legacy_disabled_accounts(override_doc)
        if not flagged:
            return []
        existing = _read_policy_file(pol_path, explicit=False) if pol_path.exists() else None
        if existing is not None:
            policy = dict(existing)
            policy['allowed_accounts'] = [name for name in policy['allowed_accounts'] if name not in flagged]
            if 'known_accounts' in policy:
                policy['known_accounts'] = list(dict.fromkeys([*policy['known_accounts'], *flagged]))
        else:
            known = list(dict.fromkeys([
                *((accounts.get('accounts') or {}) if isinstance(accounts, dict) else ()),
                *(((accounts.get('_discovery') or {}).get('accounts') or {}) if isinstance(accounts, dict) else ()),
                *(override_doc.get('accounts') or {}),
            ]))
            allowed = [name for name in known if name not in flagged]
            policy = {'allowed_accounts': allowed, 'known_accounts': list(dict.fromkeys([*allowed, *flagged]))}
        # Write host-policy.json first so a crash before cleaning accounts-override.json never loses the user's choice.
        override_store.write(pol_path, policy)
        for name in flagged:
            entry = (override_doc.get('accounts') or {}).get(name)
            if isinstance(entry, dict) and entry.get('verified') is False:
                entry.pop('verified', None)
        override_store.write(ov_path, override_doc)
        return flagged


def sync_discovered(accounts):
    """Admit only unseen, ready registered accounts; a known omission is an intentional deny."""
    policy = load_policy()
    if policy is None:
        return {'mode': 'default', 'added': []}
    if 'known_accounts' not in policy:
        return {'mode': 'static', 'added': []}
    discovery = accounts.get('_discovery') or {}
    rows = discovery.get('accounts') or {}
    specs = accounts.get('accounts') or {}
    now = datetime.now(timezone.utc)
    ready = []
    for name, row in rows.items():
        spec = specs.get(name) or {}
        # Only subscription accounts enroll automatically; metered/unknown billing needs an explicit allow.
        if not account_state.dispatchable(row) or spec.get('verified') is not True or accounts_override.manual_only_billing(name, accounts):
            continue
        if not any(not b.get('disabled') and not b.get('discovery_unavailable')
                   for b in (spec.get('bundles') or {}).values()):
            continue
        expires = spec.get('expires_at')
        if expires:
            try:
                moment = datetime.fromisoformat(expires.replace('Z', '+00:00'))
                if (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)) <= now:
                    continue
            except (AttributeError, TypeError, ValueError):
                continue
        ready.append(name)
    seen_fast = set(policy['known_accounts']) | set(policy['allowed_accounts'])
    if not any(name not in seen_fast for name in ready):
        return {'mode': 'enabled', 'added': []}
    path = policy_path()[0]
    with override_store.locked(path):
        policy = load_policy()
        if policy is None or 'known_accounts' not in policy:
            return {'mode': 'default' if policy is None else 'static', 'added': []}
        seen = set(policy['known_accounts']) | set(policy['allowed_accounts'])
        added = [name for name in ready if name not in seen]
        if added:
            policy['allowed_accounts'].extend(added)
            policy['known_accounts'] = list(dict.fromkeys([*policy['known_accounts'], *policy['allowed_accounts']]))
            override_store.write(path, policy)
        return {'mode': 'enabled', 'added': added}


def account_allowed(policy, account):
    """Whether the account takes part in automatic routing on this host (no policy file: every account does).

    An account outside `allowed_accounts` is never chosen by the router: not by a plan, a fallback, an automatic resume
    of an idle worker, nor as the scorer. It is not banned: a route the user names in the request still
    dispatches on it (`route_judge.record_constraint`, constraint `user_override`), and a turn to an existing session on
    it is not a routing choice. The model-management checkbox is the same kind of switch for a model
    (`model_unchecked`)."""
    return policy is None or account in policy['allowed_accounts']


def outside_routing(account):
    """The one wording of an account outside `allowed_accounts`; callers add why it is or is not launched."""
    return f'account {account} is out of automatic routing on this host (not in allowed_accounts of host-policy.json)'


def unchecked_routing(account, model):
    """The one wording of a model unchecked in model management; callers add why it is or is not launched."""
    return f'{account}/{model} is out of automatic routing on this host (unchecked in model management)'


# What an automatic plan adds to `outside_routing` / `unchecked_routing`: whose setting it is, and what it does.
NOT_CHOSEN = "the user's own setting on this machine, so the router does not choose it"


def sync_failed(error):
    """The one wording of a `sync_discovered` that could not run (`account_fleet.load_accounts` keeps the error)."""
    return (f'host-policy.json could not be updated with newly discovered accounts ({error}); an account discovered since '
            'it was last written is not in its allowed_accounts, so it stays out of automatic routing until the file can '
            'be written')


def model_unchecked(bundle, accounts):
    """Whether the bundle is unchecked in model management (`disabled`, the user's checkbox or off by default): its own
    row when it has one, else every row of its model on its account (a level with no row of its own)."""
    import account_fleet
    catalog = account_fleet.bundles(accounts)
    bundle_id = bundle.get('bundle_id') or bundle.get('id')
    if bundle_id in catalog:
        return bool(catalog[bundle_id].get('disabled'))
    key = account_fleet.row_key(bundle.get('model') or '', accounts)
    rows = [b for b in catalog.values()
            if b['account'] == bundle.get('account') and account_fleet.row_key(b['model'], accounts) == key]
    return bool(rows) and all(b.get('disabled') for b in rows)


def automatic_routing_exclusions(bundle, accounts, policy):
    """The automatic-routing switches a launch on `bundle` reaches past, in a fixed order: `account` (outside
    `allowed_accounts`), `model` (unchecked in model management). Both are preferences: the router never chooses past
    them, and a route the user names does (`route_judge.record_constraint`)."""
    return [name for name, out in (('account', not account_allowed(policy, bundle.get('account'))),
                                   ('model', model_unchecked(bundle, accounts))) if out]


def exclusion_texts(bundle, accounts, policy):
    """The wording of each switch `automatic_routing_exclusions` names, in its order."""
    texts = {'account': lambda: outside_routing(bundle.get('account')),
             'model': lambda: unchecked_routing(bundle.get('account'), bundle.get('model'))}
    return [texts[name]() for name in automatic_routing_exclusions(bundle, accounts, policy)]


def legacy_cause(account, accounts):
    """('legacy', text) when the account still carries leftover meter text, else None. A fact: it refuses a launch and
    a parent's new turn to its own worker alike."""
    legacy = ((accounts or {}).get('accounts') or {}).get(account, {}).get('legacy_meter')
    return ('legacy', legacy) if isinstance(legacy, str) and legacy.strip() else None


def disabled_cause(account, candidates, accounts):
    """(kind, detail) when a launch of the target needs more than a routing decision, else None. Kinds: `legacy`
    (`legacy_cause`, a fact: refused whatever the route), `bundles` (ids all unchecked in model management;
    `candidates` are the account's matching bundle rows: out of automatic routing, so only a route the user named
    launches it). A new turn to an existing worker reads only `legacy_cause`: continuing a worker is not choosing its
    model. Host policy is not a cause: it governs automatic routing only (`account_allowed`)."""
    found = legacy_cause(account, accounts)
    if found:
        return found
    if candidates and all(c.get('disabled') for c in candidates):
        return 'bundles', [c['bundle_id'] for c in candidates]
    return None


def disabled_text(cause):
    """The spawn gate's wording of a `disabled_cause`: the leftover meter text itself, or the unchecked bundles."""
    kind, detail = cause
    return detail if kind == 'legacy' else "档位 " + ", ".join(detail) + " 在模型管理中未勾选（不参与自动派发）"


def usage_providers(policy, defaults):
    if policy is None or 'usage_providers' not in policy:
        return list(defaults)
    allowed = set(policy['usage_providers'])
    return [name for name in defaults if name in allowed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True)
    args = parser.parse_args()
    try:
        allowed = account_allowed(load_policy(), args.account)
    except HostPolicyError as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps({'account': args.account, 'allowed': allowed}))
    return 0 if allowed else 1


if __name__ == '__main__':
    raise SystemExit(main())
