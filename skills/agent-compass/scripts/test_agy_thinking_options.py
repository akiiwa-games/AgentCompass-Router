"""Antigravity through Paseo's `antigravity-resilient` provider takes a bare model id and the thinking option as its own
field (`thinkingOptionId`), each model with the options it lists itself. The plugin keeps `<model>-<option>` only as an
alias. These tests state the rules that follow: what a launch passes, what a listing derives, and what an earlier
spelling (a stored choice, a hold, a persistence record) still means."""
import copy
from datetime import datetime, timedelta, timezone
import json

import pytest

import account_fleet
import antigravity_telemetry
import auto_bundles
import dispatch_registry
import live_models
import provider_inventory
import provider_retry

PROVIDER = 'antigravity-resilient'
NOW = 2_000_000_000.0
# Paseo's list as the bridge normalizes it (`normalizeProviderModels`): options and aliases per model.
LISTING = [
    {'id': 'gemini-4-hydrogen', 'aliases': [f'gemini-4-hydrogen-{o}' for o in ('low', 'medium', 'high', 'xhigh')],
     'thinkingOptionIds': ['low', 'medium', 'high', 'xhigh'], 'defaultThinkingOptionId': 'high'},
    {'id': 'gemini-pro-orchestrator', 'thinkingOptionIds': []},
    {'id': 'gpt-oss-120b-medium', 'thinkingOptionIds': []},
]


@pytest.fixture
def fleet():
    return account_fleet.load_accounts(account_fleet.reference_fleet_path())


def derived(fleet, models=LISTING, overrides=None):
    catalog = {'providers': {PROVIDER: {'ok': True, 'type': 'paseo_provider', 'checked_at': NOW, 'models': models}}}
    return auto_bundles.derive(fleet, catalog, NOW, overrides=overrides)['bundles']['antigravity']


def launch(bundle):
    return dispatch_registry.launch_request(dict(bundle, provider=PROVIDER), {'title': 't', 'initialPrompt': 'p'})


def test_a_launch_passes_the_bare_model_and_the_option_only_for_a_model_that_has_one(fleet):
    rows = derived(fleet)
    with_option = launch(rows[f'auto:antigravity:{PROVIDER}:gemini-4-hydrogen@low'])
    assert (with_option['provider'], with_option['settings']['thinkingOptionId']) == (f'{PROVIDER}/gemini-4-hydrogen', 'low')
    without = launch(rows[f'auto:antigravity:{PROVIDER}:gemini-pro-orchestrator'])
    assert without['provider'] == f'{PROVIDER}/gemini-pro-orchestrator'
    assert 'thinkingOptionId' not in without['settings']


def test_a_listing_derives_one_bundle_per_option_the_model_lists_and_none_for_a_model_without(fleet):
    rows = derived(fleet)
    hydrogen = {b['effort']: b for b in rows.values() if b['launch_model'] == 'gemini-4-hydrogen'}
    assert set(hydrogen) == {'low', 'medium', 'high', 'xhigh'}
    assert all(b['efforts'] == ['low', 'medium', 'high', 'xhigh'] for b in hydrogen.values())
    assert all(b['launch_model'] == 'gemini-4-hydrogen' for b in hydrogen.values())
    plain = [b for b in rows.values() if b['launch_model'] in ('gemini-pro-orchestrator', 'gpt-oss-120b-medium')]
    assert len(plain) == 2 and all('effort' not in b and 'efforts' not in b for b in plain)


def test_an_option_the_model_does_not_list_is_refused_by_the_launch_contract(fleet):
    row = derived(fleet)[f'auto:antigravity:{PROVIDER}:gemini-4-hydrogen@low']
    assert 'ultra' not in account_fleet.dispatch_allowed_efforts(dict(row, account='antigravity'), fleet)
    assert 'low' in account_fleet.dispatch_allowed_efforts(dict(row, account='antigravity'), fleet)


def test_a_choice_stored_under_the_id_of_an_earlier_listing_still_applies(fleet):
    """An id written for `gpt-oss-120b-medium` read as `gpt-oss-120b` at `medium` (the name's suffix was its effort);
    that model takes no effort now, and the user's unchecked box under the old id still unchecks it."""
    old = f'auto:antigravity:{PROVIDER}:gpt-oss-120b@medium'
    overrides = {'schema_version': 2, 'accounts': {'antigravity': {'bundles': {old: {'disabled': True}}}}}
    row = next(b for b in derived(fleet, overrides=overrides).values() if b['launch_model'] == 'gpt-oss-120b-medium')
    assert row['_v0_7_0_id'] == old and row['disabled'] is True
    assert derived(fleet)[next(k for k, b in derived(fleet).items() if b['launch_model'] == 'gpt-oss-120b-medium')]['disabled'] is False


def test_a_listing_without_options_keeps_the_effort_in_the_name_as_before(fleet):
    """A cached list of an earlier plugin (ids `<model>-<effort>`, no option fields) derives what it derived then."""
    rows = derived(fleet, models=[{'id': 'gemini-4-hydrogen-xhigh'}, {'id': 'gemini-4-hydrogen-medium'}])
    assert {(b['launch_model'], b['effort']) for b in rows.values()} == {
        ('gemini-4-hydrogen-xhigh', 'xhigh'), ('gemini-4-hydrogen-medium', 'medium')}


def test_a_listing_keeps_each_models_options_and_reads_an_alias_as_its_model_at_that_option():
    raw = {'ok': True, 'models': [
        {'id': 'm', 'aliases': ['m-low', 'm-high'], 'thinkingOptionIds': ['low', ' high '], 'defaultThinkingOptionId': 'high'},
        {'id': 'm-high', 'thinkingOptionIds': []},
        {'id': 'old'}]}
    entry = live_models._normalize_entry(raw, '2026-11-26T00:00:00+00:00')
    assert [m.get('thinkingOptionIds') for m in entry['models']] == [['low', 'high'], [], None]
    read = lambda name: live_models.listing(PROVIDER, name, entry)
    assert read('m-low') == {'id': 'm', 'effort': 'low', 'options': ['low', 'high'], 'default': 'high'}
    assert read('m') == {'id': 'm', 'effort': None, 'options': ['low', 'high'], 'default': 'high'}
    # An exact id wins over an alias: `m-high` is a model of its own that takes no option.
    assert read('m-high') == {'id': 'm-high', 'effort': None, 'options': [], 'default': None}
    assert read('old')['options'] is None and read('nothing') is None
    assert read(f'{PROVIDER}/m-low')['effort'] == 'low'


def test_naming_an_alias_derives_the_model_at_that_option(fleet):
    live = {'providers': {PROVIDER: live_models._normalize_entry({'ok': True, 'models': LISTING}, '2026-11-26T00:00:00+00:00')}}
    rows, why = provider_inventory.listed_rows(fleet, 'antigravity', PROVIDER, 'gemini-4-hydrogen-low', None, live)
    assert why is None
    assert {(b['launch_model'], b['effort']) for b in rows.values()} == {('gemini-4-hydrogen', 'low')}


def test_a_hold_written_under_the_effort_suffixed_name_still_keeps_that_option_out(tmp_path, monkeypatch):
    monkeypatch.setenv('ROUTER_PROVIDER_RETRY', str(tmp_path / provider_retry.FILE))
    medium, high = ({'account': 'antigravity', 'provider': PROVIDER, 'harness': PROVIDER, 'model': 'gemini-4-carbon',
                     'effort': effort} for effort in ('medium', 'high'))
    when = datetime(2026, 11, 26, 4, 0, tzinfo=timezone.utc)
    entry = {'account': 'antigravity', 'provider': PROVIDER, 'model': 'gemini-4-carbon-medium', 'scope': 'model',
             'failure_code': 'provider_turn_failed', 'observed_at': when.isoformat(),
             'retry_at': (when + timedelta(hours=1)).isoformat()}
    held = {'antigravity/gemini-4-carbon-medium': entry}
    assert provider_retry.entry_for(medium, held) is entry
    assert provider_retry.entry_for(high, held) is None


def test_the_agy_cli_names_a_model_with_its_effort_and_the_bundle_keeps_them_apart():
    assert account_fleet.agy_cli_model('gemini-4-neon', 'low') == 'gemini-4-neon-low'
    assert account_fleet.agy_cli_model('gemini-4-neon-low', 'low') == 'gemini-4-neon-low'
    assert account_fleet.agy_cli_model('gemini-pro-orchestrator', None) == 'gemini-pro-orchestrator'


def test_a_persistence_record_reads_the_conversation_from_either_field(tmp_path):
    conversation = 'c0ffee-1234'
    log = tmp_path / '.gemini/antigravity-cli/brain' / conversation / '.system_generated/logs/transcript.jsonl'
    log.parent.mkdir(parents=True)
    log.write_text('', encoding='utf-8')
    for data in ({'sessionId': conversation}, {'conversationId': conversation}, {'sessionId': conversation, 'conversationId': conversation}):
        persistence = {'provider': PROVIDER, 'sessionId': 'plugin:' + json.dumps({'version': 1, 'data': data})}
        assert antigravity_telemetry.transcript(copy.deepcopy(persistence), tmp_path) == (PROVIDER, log)
    assert antigravity_telemetry.transcript({'provider': PROVIDER, 'sessionId': conversation}, tmp_path) == (PROVIDER, log)
