from datetime import datetime, timedelta, timezone
import quota_exhaustion as q

# A latch observed an hour ago: inside the latches' re-probe period (quota_exhaustion.due_at).
RECENT = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
BEFORE = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()


def test_only_confirmed_quota_errors_latch_and_persist(tmp_path):
    path = tmp_path / 'quota.json'
    assert not q.record('account', 'HTTP 429: temporary rate limit', path=path)
    assert not q.record('account', 'Network timeout', path=path)
    assert q.record('account', 'Individual quota reached', at=RECENT, path=path)
    assert q.stopping('account', path=path) == {q.ALL}    # no window known: the whole account
    assert 'Individual quota' not in path.read_text()
    assert q.stopping('another', path=path) == set()


def test_old_or_failed_quota_cannot_release_but_fresh_positive_can(tmp_path):
    path = tmp_path / 'quota.json'
    q.record('account', 'quota_exceeded', at=RECENT, path=path)
    usage = {'available': True, 'fetched_at': BEFORE, 'windows': [{'id': 'usage', 'used_pct': 30}]}
    assert q.stopping('account', usage, path=path) == {q.ALL}    # a reading older than the latch releases nothing
    usage['fetched_at'] = datetime.now(timezone.utc).isoformat()
    usage['stale'] = True
    assert q.stopping('account', usage, path=path) == {q.ALL}
    usage['stale'] = False
    usage['windows'][0]['used_pct'] = 100
    assert q.stopping('account', usage, path=path, windows=['usage']) == {'usage'}    # narrowed to the window it shows used up
    usage['windows'][0]['used_pct'] = 10
    assert q.stopping('account', usage, path=path, windows=['usage']) == set()
    assert q.stopping('account', path=path, windows=['usage']) == set()


def test_manual_clear_does_not_replay_an_old_failure(tmp_path):
    path = tmp_path / 'quota.json'
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    q.record('account', 'quota exhausted', at=old, path=path)
    q.clear('account', path=path)
    assert not q.record('account', 'quota exhausted', at=old, path=path)
    assert q.stopping('account', path=path) == set()


def test_only_the_exhausted_window_latches_and_a_fresh_reading_releases_it(tmp_path):
    path = tmp_path / 'quota.json'
    assert q.record('agy', 'Individual quota reached', windows=['3p-5h', '3p-weekly'],
                    at=RECENT, path=path)
    gemini, claude = ['gemini-5h', 'gemini-weekly'], ['3p-5h', '3p-weekly']
    assert q.stopping('agy', path=path) == set() and q.stopping('agy', path=path, windows=gemini) == set()
    assert q.stopping('agy', path=path, windows=claude) == set(claude)    # only the recorded windows a launch draws
    future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    reading = {'available': True, 'fetched_at': datetime.now(timezone.utc).isoformat(), 'windows': [
        {'id': 'gemini-weekly', 'used_pct': 95}, {'id': '3p-weekly', 'used_pct': 100, 'resets_at': future},
        {'id': '3p-5h', 'used_pct': 33}]}
    # The fresh reading narrows the latch to the window it shows exhausted.
    assert q.exhausted_windows('agy', reading, path=path) == {'3p-weekly'}
    assert q.stopping('agy', reading, path=path, windows=['3p-weekly']) == {'3p-weekly'}
    assert q.stopping('agy', reading, path=path, windows=['3p-5h']) == set()
    reading['windows'][1]['used_pct'] = 10
    reading['fetched_at'] = datetime.now(timezone.utc).isoformat()
    assert q.exhausted_windows('agy', reading, path=path) == set()
    assert q.stopping('agy', path=path, windows=claude) == set()


def test_account_wide_latch_narrows_to_exhausted_windows_and_reset_releases(tmp_path):
    path = tmp_path / 'quota.json'
    q.record('agy', 'quota exhausted', at=RECENT, path=path)
    assert q.stopping('agy', path=path, windows=['gemini-5h']) == {q.ALL}
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    reading = {'available': True, 'fetched_at': datetime.now(timezone.utc).isoformat(), 'windows': [
        {'id': 'gemini-5h', 'used_pct': 1}, {'id': '3p-weekly', 'used_pct': 100}, {'id': '3p-5h', 'used_pct': 100,
                                                                                 'resets_at': past}]}
    assert q.exhausted_windows('agy', reading, path=path) == {'3p-weekly'}
    # The reading's own reset time having passed releases that window; a reset date alone, with no reading, never does.
    assert q.stopping('agy', path=path, windows=['gemini-5h', '3p-5h']) == set()


def test_zero_remaining_blocks_even_without_a_configured_floor_or_reset():
    import quota_pools
    now = datetime.now(timezone.utc)
    row = {'status': 'fresh', 'fetched_at': now.isoformat()}
    assert quota_pools.below_launch_floor({}, {'remaining_pct_before_reset': 0}, row, now)
    assert not quota_pools.below_launch_floor({}, {'remaining_pct_before_reset': 20}, row, now)
    assert not quota_pools.below_launch_floor({}, {}, row, now)


def test_routing_excludes_only_bundles_drawing_a_latched_window():
    import allocator
    import account_fleet
    accounts = account_fleet.load_accounts(account_fleet.REFERENCES / 'accounts.json')
    accounts['accounts']['antigravity']['verified'] = True
    accounts['_discovery'] = {'accounts': {'antigravity': {
        'family': 'antigravity', 'status': 'ready', 'exhausted_windows': ['3p-weekly']}}}
    decision = allocator.plan({'objective': 'Reply with exactly OK', 'hours': 0.1}, {}, accounts=accounts,
                              overrides=[], host_policy_config=None)
    rows = {row['id']: row for row in decision['ranking']}
    latched = '3p-weekly: quota confirmed exhausted; refresh quota or replan'
    assert latched in rows['agy_opus55_high']['excluded']
    assert latched not in rows['argon_high']['excluded']
    assert latched not in rows['flash_high']['excluded']


def test_a_latch_with_no_reading_ends_by_itself_at_the_latches_period(tmp_path):
    # No latch waits for a person (owner, 2026-11-24): with no reading and no clear, it is due after the latches'
    # re-probe period (account_state.LATCH_RETRY_HOURS), counted from the observation.
    import account_state
    path = tmp_path / 'quota.json'
    at = datetime(2026, 11, 24, 8, 0, tzinfo=timezone.utc)
    assert q.record('acme', 'quota exhausted', windows=['allowance'], at=at.isoformat(), path=path)
    due = at + timedelta(hours=account_state.LATCH_RETRY_HOURS)
    before, after = due - timedelta(seconds=1), due + timedelta(seconds=1)
    assert q.stopping('acme', path=path, windows=['allowance'], now=before, read_only=True) == {'allowance'}
    assert q.stopping('acme', path=path, windows=['allowance'], now=after, read_only=True) == set()
    assert q.stopping('acme', path=path, windows=['allowance'], now=before) == {'allowance'}   # read_only or not
    assert q.stopping('acme', path=path, windows=['allowance'], now=after) == set()
