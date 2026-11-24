"""A free pool whose vendor publishes no capacity: an ordinary quota pool (owner, 2026-11-24, addendum 2).

Rules under test (skills/agent-compass/references/router-internals.md, section 2, free pools):
  - the vendor's declaration is data (plan-templates `capacity_unpublished`, a dated vendor citation); it ships no C;
  - C comes from this account's own observation at its limit error (quota_exhaustion.observation: the spend of the
    period that error ended), else the maintainer's shipped measurement (`capacity_measured`), else none: mode B
    (`capacity_unknown`), out of automatic routing, as on main; an observed C is that account's alone;
  - with C the pool is priced by the same function as any window from C, U, t0 and T: equal inputs give an equal
    term, the term is non-decreasing in the used share, and a pool that never resets (T = inf) has a recovery line of 0;
  - with no observed boundary the reading is the conservative one (U over the last T hours, the reset T hours away);
  - a reset the vendor gives as data (OpenCode's retry-after, read back from its own store) beats the sentence's own
    and the latches' re-probe period; an observed reset with no stated period reads the measurement's period;
  - the vendor's limit error, recorded by the real writer (provider_retry.record -> quota_exhaustion.record), is the
    exhaustion of the allowance window with the reset the error states; held until that reset, a quota wait like any
    used-up window; eligible again after it; a sentence with no readable reset falls to the latches' re-probe period.
The measured figure here is a fixture value, not shipped data. No winner, share or score is asserted.
"""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import account_state
import allocator
import axis_scores
import ledger_snapshot
import provider_inventory
import provider_retry
import quota_exhaustion
import route_simulation as sim
import test_five_axis as tf
from router_test_support import profile_for
from test_fleet_parity import shipped_template, snapshot

POOL = 'opencode'
# The vendor's own sentences (anomalyco/opencode#59387 and the issues it collects, read 2026-11-24; research report
# §1.4 and §3), used as fixture text: no real turn is run.
LIMIT = 'Free usage limit reached. Connect a provider or subscribe to OpenCode Go to keep going, or switch models.'
CLOCK = "You've used today's free messages. They reset at 11:59 PM"
DATE = "You've used today's free messages. They reset at {date}"
COUNTDOWN = 'Free usage exceeded, subscribe to Go reintentando en 74000s - intento n.º 1'
UNREADABLE = "You've used today's free messages. They reset at tomorrow morning"
# The shape of a shipped measurement (reference_provenance.valid_measured_capacity), with a test value.
MEASURED = {'kind': 'maintainer', 'as_of': '2026-11-24', 'capacity_usd': 50.0, 'period_hours': 24, 'bound': 'lower_bound',
            'model': 'deepseek-v4.1-flash-free', 'requests': 100,
            'statement': 'fixture measurement for tests; not shipped data'}
FREE_MODEL = 'deepseek-v4.1-flash-free'


def spend_of(usd, gap=(), calls=None):
    """A stand-in for opencode.db: `usd` spent on the free model in any time-bounded read (the period reads), nothing
    unbounded; `calls` collects the bounds of the bounded reads."""
    def read(models, discovery=None, *, read_local_db=True, since=None, until=None):
        if calls is not None and since is not None:
            calls.append((since, until))
        return ({FREE_MODEL: usd} if usd else {}, list(gap)) if since is not None else ({}, [])
    return read


def setUpModule():
    tf.setUpModule()


class UnpublishedFreePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shipped = sim.fresh_accounts(shipped_template(), snapshot())
        cls.doc = tf.document(shipped['models'])
        cls.declaration = {k: v for k, v in provider_inventory.shipped_plan_templates()['opencode']['free'].items()
                           if k != 'capacity_measured'}
        # The plan as declared with no measurement shipped beside it (a pool with no C), and with the fixture one.
        cls.bare = copy.deepcopy(shipped)
        cls.bare['plan_templates'] = {'opencode': {'free': copy.deepcopy(cls.declaration)}}
        cls.accounts = cls.with_measured(shipped)      # a pool with a capacity, for the latch tests

    @classmethod
    def with_measured(cls, acc, measured=MEASURED, family='opencode'):
        acc = copy.deepcopy(acc)
        acc['plan_templates'] = {family: {'free': {**copy.deepcopy(cls.declaration), 'capacity_measured': dict(measured)}}}
        return acc

    def setUp(self):
        self.state = Path(tempfile.mkdtemp(prefix='free-pool-'))
        self.addCleanup(shutil.rmtree, self.state, ignore_errors=True)
        env = patch.dict(os.environ, {'ROUTER_STATE_DIR': str(self.state),
                                      'ROUTER_PROVIDER_RETRY': str(self.state / provider_retry.FILE)})
        env.start()
        self.addCleanup(env.stop)
        self.now = datetime.now(timezone.utc)

    # -- helpers -------------------------------------------------------------------------------------------------------

    def only(self, *names, acc=None):
        acc = copy.deepcopy(acc or self.accounts)
        acc['accounts'] = {n: s for n, s in acc['accounts'].items() if n in names}
        return acc

    def plan(self, acc, now=None, usage_extra=None, **task):
        now = now or self.now
        acc = copy.deepcopy(acc)
        with tf._synthetic_doc_ctx(self.doc):
            usage = sim.snapshot(sim.synthetic_document(acc, now, sim.STATES['ample']), acc, now)
            usage.setdefault('accounts', {}).update(usage_extra or {})
            return allocator.plan({'objective': 't', 'profile': dict(tf.PROFILE), **task}, usage, now, accounts=acc,
                                  overrides=[], records=[], host_policy_config={'allowed_accounts': list(acc['accounts'])})

    def overlay(self, acc, now=None):
        with tf._synthetic_doc_ctx(self.doc):
            return axis_scores.five_axis_overlay(copy.deepcopy(acc), now=now or self.now)

    def rows(self, decision, account=POOL):
        return [c for c in decision['ranking'] if c['bundle']['account'] == account]

    def fail_turn(self, acc, message, dispatch_id, at=None):
        """The real writer: a failed turn of the pool's bundle as the registry hands it to provider_retry."""
        bid, bundle = next(iter(acc['accounts'][POOL]['bundles'].items()))
        with tf._synthetic_doc_ctx(self.doc):
            return provider_retry.record({**bundle, 'id': bid, 'account': POOL}, {
                'failure_code': provider_retry.TURN_FAILED_CODE, 'provider_message': message,
                'at': (at or self.now).isoformat()}, dispatch_id=dispatch_id, accounts=acc,
                classify=lambda *args: ('account', {'source': 'system_one'}))

    def discovered(self, acc):
        """Discovery's reading of the latches (provider_inventory, as every plan and the tray read them)."""
        rows = {name: {'status': 'ready'} for name in acc['accounts']}
        provider_inventory._latch_exhausted_quota(rows, self.state, {})
        return {**copy.deepcopy(acc), '_discovery': {'accounts': rows}}

    # -- the declaration ------------------------------------------------------------------------------------------------

    def test_the_shipped_measurement_has_its_provenance_and_is_read_as_the_pools_capacity(self):
        """The shipped `capacity_measured` (owner-confirmed, 2026-11-24) passes its provenance check and, with no
        observation of the account's own, is the pool's C over its period (source 2 of free_allowance_period)."""
        import reference_provenance
        shipped = provider_inventory.shipped_plan_templates()['opencode']['free']['capacity_measured']
        self.assertTrue(reference_provenance.valid_measured_capacity(shipped))
        spec = self.overlay(self.only(POOL, acc=sim.fresh_accounts(shipped_template(), snapshot())))['accounts'][POOL]
        self.assertEqual(spec['capacity_usd'], {'allowance': shipped['capacity_usd']})
        self.assertEqual(spec['windows'], {'allowance': {'hours': shipped['period_hours']}})
        self.assertEqual((spec['capacity_source']['allowance']['status'], spec['capacity_source']['allowance']['as_of']),
                         ('maintainer', shipped['as_of']))

    def test_the_shipped_declaration_is_a_dated_vendor_citation_on_a_zero_price_plan(self):
        declared = self.declaration['capacity_unpublished']
        self.assertEqual(declared['kind'], 'vendor')
        self.assertTrue(declared['url'].startswith('https://') and declared['as_of'] and declared['statement'])
        self.assertEqual(self.declaration['price_usd_per_month'], 0)

    def test_with_no_capacity_the_declared_pool_is_excluded_as_on_main(self):
        """Rule (axis_scores.free_allowance_period, source 3): no observation and no shipped measurement is no C, so the
        pool is mode B (`capacity_unknown`), out of automatic routing, as on main. An observation whose period a model
        with no list price spent in gives no C either: the gap is surfaced on the window's source."""
        rows = self.rows(self.plan(self.bare))
        self.assertTrue(rows)
        self.assertTrue(all(any('capacity_unknown (quota unknown, spend unpriced)' in e for e in c['excluded'])
                            for c in rows), [c['excluded'] for c in rows])
        self.assertTrue(quota_exhaustion.record(POOL, CLOCK, windows=['allowance'], at=self.now.isoformat()))
        with patch.object(axis_scores, 'opencode_local_spend', spend_of(3.0, gap=('big-pickle',))):
            spec = self.overlay(self.bare)['accounts'][POOL]
            rows = self.rows(self.plan(self.bare))
        self.assertTrue(spec.get('capacity_unknown'))
        self.assertIn('big-pickle', spec['capacity_source']['allowance']['reason'])
        self.assertTrue(all(any('capacity_unknown' in e for e in c['excluded']) for c in rows))

    def test_the_rule_is_the_declaration_not_the_family_name(self):
        acc = copy.deepcopy(self.bare)
        acc['accounts']['acme-free'] = {**copy.deepcopy(acc['accounts'][POOL]), 'family': 'acme'}
        del acc['accounts'][POOL]
        declared = self.with_measured(acc, family='acme')
        spec = self.overlay(declared)['accounts']['acme-free']
        self.assertFalse(spec.get('capacity_unknown'), spec.get('capacity_source'))
        self.assertEqual(spec['capacity_source']['allowance']['status'], 'maintainer')
        # The same account with no declaration for its family: no measurement is read, mode B.
        self.assertTrue(all(any('capacity_unknown' in e for e in c['excluded'])
                            for c in self.rows(self.plan(acc), 'acme-free')))

    def test_a_model_launched_under_another_prefix_of_the_account_draws_nothing_of_the_free_allowance(self):
        # Rule: the plan's `allowance_launch_prefix` scopes the allowance; OpenCode Go connected on the same account
        # (`opencode-go/...`, a paid product) is not drawn from the free allowance.
        import auto_bundles
        from router_test_support import fleet_with_derived_bundles
        zen, go = 'opencode/deepseek-v4.1-flash-free', 'opencode-go/glm-5.3'
        acc = self.only(POOL)
        catalog = {'providers': {POOL: {'ok': True, 'type': 'paseo_provider', 'checked_at': 1,
                                        'models': [{'id': zen}, {'id': go}]}}}
        acc = fleet_with_derived_bundles(acc, auto_bundles.derive(acc, catalog, 2))
        spec = self.overlay(acc)['accounts'][POOL]
        draws = {b['launch_model']: b['draws'] for b in spec['bundles'].values() if b.get('launch_model') in (zen, go)}
        self.assertEqual(draws, {zen: ['allowance'], go: []})
        rows = {c['bundle'].get('launch_model'): c for c in self.rows(self.plan(acc))}
        self.assertFalse(rows[zen]['excluded'], rows[zen]['excluded'])
        self.assertTrue(rows[go]['excluded'])

    # -- an ordinary quota pool ------------------------------------------------------------------------------------------

    def twin(self, acc, cap, used_usd, resets):
        """A paid window with the same C, U and reset as the pool: a metered account with its own budget (the one kind
        whose capacity the account states itself), the pool's models under other bundle ids, and its reading."""
        acc = copy.deepcopy(acc)
        pool = acc['accounts'][POOL]
        acc['accounts'] = {'twin': {'billing': 'metered', 'verified': True, 'windows': {'allowance': {'hours': 24}},
                                    'capacity_usd': {'allowance': cap},
                                    'bundles': {f'twin_{k}': copy.deepcopy(v) for k, v in pool['bundles'].items()}}}
        import usage_reading
        reading = {'status': 'fresh', 'available': True, 'fetched_at': self.now.isoformat(),
                   'windows': [usage_reading.window_state({'id': 'allowance', 'used_pct': 100 * used_usd / cap,
                                                           'resets_at': resets.isoformat()}, self.now)]}
        return acc, {'twin': reading}

    def terms(self, decision):
        return {c['bundle']['model']: c['components']['opportunity_cost'] for c in decision['ranking']
                if not c['excluded']}

    def test_equal_capacity_use_and_reset_give_the_paid_windows_quota_term(self):
        """Rule (round 11, addendum 2): a free pool's quota term comes out of the same function as a paid window's: for
        equal C, U, t0 and T the term is equal. Each is planned alone, so neither is the other's alternative."""
        acc = self.only(POOL, acc=self.accounts)
        with patch.object(axis_scores, 'opencode_local_spend', spend_of(20.0)):
            pool = self.plan(acc, metered_ok=True)
            twin, reading = self.twin(acc, MEASURED['capacity_usd'], 20.0,
                                      self.now + timedelta(hours=MEASURED['period_hours']))
            paid = self.plan(twin, usage_extra=reading, metered_ok=True)
        econ = [c['economy']['windows'][0] for c in self.rows(pool) if not c['excluded']]
        self.assertTrue(econ and all(w['used_usd'] == 20.0 and w['capacity_usd'] == MEASURED['capacity_usd'] for w in econ))
        self.assertTrue(self.terms(pool))
        self.assertEqual(self.terms(pool), self.terms(paid))

    def test_the_quota_term_is_non_decreasing_in_the_used_share(self):
        acc = self.only(POOL, acc=self.accounts)
        seen = {}
        for used in (0.0, 10.0, 20.0, 30.0, 40.0):
            with patch.object(axis_scores, 'opencode_local_spend', spend_of(used)):
                for model, term in self.terms(self.plan(acc)).items():
                    seen.setdefault(model, []).append(term)
        self.assertTrue(seen and all(len(terms) >= 3 for terms in seen.values()), seen)
        for model, terms in seen.items():
            self.assertEqual(terms, sorted(terms), model)

    def test_a_pool_that_never_resets_has_a_recovery_line_of_zero(self):
        """Rule (quota_economy.five_axis_window_recovery): T = inf (an allowance whose window has no period) does not run
        its clock: elapsed 0 and recovery line 0 at any use."""
        import quota_economy
        import quota_pools
        acc = self.only(POOL, acc=self.bare)
        acc['accounts'][POOL].update(windows={'allowance': {'hours': None}}, capacity_usd={'allowance': 50.0})
        spec = self.overlay(acc)['accounts'][POOL]
        self.assertEqual(spec['capacity_usd'], {'allowance': 50.0})
        for used in (0.0, 12.5, 49.0):
            state = quota_pools.window_state('allowance', spec['windows']['allowance'], None,
                                             {**spec, 'local_spend_usd': used}, self.now, 24)
            rec = quota_economy.five_axis_window_recovery(state, 50.0)
            self.assertEqual((rec['elapsed_frac'], rec['line_now_usd'], rec['reset_hours']), (0.0, 0.0, float('inf')))
            self.assertAlmostEqual(rec['used_usd'], used)

    def test_an_observed_capacity_replaces_a_missing_one_only_for_that_account(self):
        """Rule (axis_scores.free_allowance_period, sources 1 and 2): the account's own observation gives its C (the
        spend from the stated reset less the stated period to the error); another account of the same plan keeps its
        own source: the shipped measurement when one ships, else none."""
        bare = copy.deepcopy(self.bare)
        bare['accounts']['opencode-b'] = {**copy.deepcopy(bare['accounts'][POOL]), 'family': 'opencode'}
        self.assertTrue(quota_exhaustion.record(POOL, CLOCK, windows=['allowance'], at=self.now.isoformat()))
        seen = quota_exhaustion.observation(POOL)
        calls = []
        with patch.object(axis_scores, 'opencode_local_spend', spend_of(12.5, calls=calls)):
            plain, measured = self.overlay(bare)['accounts'], self.overlay(self.with_measured(bare))['accounts']
        self.assertIn((seen['resets_at'] - timedelta(hours=24), seen['observed_at']), calls)
        for accounts in (plain, measured):
            self.assertEqual(accounts[POOL]['capacity_usd'], {'allowance': 12.5})
            self.assertEqual(accounts[POOL]['capacity_source']['allowance']['status'], 'observed')
            self.assertFalse(accounts[POOL].get('capacity_unknown'))
        self.assertTrue(plain['opencode-b'].get('capacity_unknown'))
        self.assertEqual(measured['opencode-b']['capacity_usd'], {'allowance': MEASURED['capacity_usd']})
        self.assertEqual(measured['opencode-b']['capacity_source']['allowance']['status'], 'maintainer')

    def test_an_observation_with_no_stated_period_reads_the_period_of_the_shipped_measurement(self):
        """Rule (free_allowance_period): the period of an observed reset is the error's own, else the shipped
        measurement's; with neither there is no period to read the spend over, so no C."""
        given = self.now + timedelta(hours=20)
        self.assertTrue(quota_exhaustion.record(POOL, LIMIT, windows=['allowance'], at=self.now.isoformat(),
                                                resets_at=given))
        calls = []
        with patch.object(axis_scores, 'opencode_local_spend', spend_of(0.25, calls=calls)):
            measured = self.overlay(self.with_measured(self.only(POOL, acc=self.bare)))['accounts'][POOL]
            plain = self.overlay(self.only(POOL, acc=self.bare))['accounts'][POOL]
        period = timedelta(hours=MEASURED['period_hours'])
        self.assertIn((given - period, self.now), calls)
        self.assertEqual((measured['capacity_usd'], measured['capacity_source']['allowance']['status']),
                         ({'allowance': 0.25}, 'observed'))
        self.assertEqual(measured['local_period']['boundary'], 'observed')
        self.assertEqual(datetime.fromisoformat(measured['local_period']['resets_at']), given)
        self.assertTrue(plain.get('capacity_unknown'))
        self.assertIn('no period', plain['capacity_source']['allowance']['reason'])

    def test_without_an_observed_boundary_the_reading_is_the_conservative_one(self):
        """Rule (free_allowance_period): with no observed reset, U is the spend of the last T hours and the reset is T
        hours away; with one, the period runs from the observed reset (rolled forward by T) less T."""
        calls = []
        acc = self.only(POOL, acc=self.accounts)
        period = timedelta(hours=MEASURED['period_hours'])
        with patch.object(axis_scores, 'opencode_local_spend', spend_of(5.0, calls=calls)):
            reading = self.overlay(acc)['accounts'][POOL]['local_period']
        self.assertEqual(calls, [(self.now - period, self.now)])
        self.assertEqual((reading['boundary'], datetime.fromisoformat(reading['resets_at'])), ('assumed', self.now + period))
        self.assertTrue(quota_exhaustion.record(POOL, CLOCK, windows=['allowance'], at=self.now.isoformat()))
        reset = quota_exhaustion.observation(POOL)['resets_at']
        later = reset + period + timedelta(hours=1)        # two resets after the observed one: rolled forward by T
        calls.clear()
        with patch.object(axis_scores, 'opencode_local_spend', spend_of(0.0, calls=calls)):
            reading = self.overlay(acc, now=later)['accounts'][POOL]['local_period']
        self.assertEqual(reading['boundary'], 'observed')
        self.assertEqual(datetime.fromisoformat(reading['resets_at']), reset + 2 * period)
        self.assertEqual(calls[-1], (reset + period, later))

    def test_the_period_reading_is_what_the_window_is_priced_from(self):
        """The overlay's period reading enters the plan as the window's reading (axis_scores.with_local_readings): used
        share U / C, its reset; a reading of its own is not replaced."""
        acc = self.only(POOL, acc=self.accounts)
        with patch.object(axis_scores, 'opencode_local_spend', spend_of(10.0)):
            overlaid = self.overlay(acc)
        usage = axis_scores.with_local_readings({'accounts': {}}, overlaid, self.now)
        window = usage['accounts'][POOL]['windows'][0]
        self.assertEqual((window['id'], window['used_pct']), ('allowance', 100 * 10.0 / MEASURED['capacity_usd']))
        self.assertEqual(window['resets_at'], overlaid['accounts'][POOL]['local_period']['resets_at'])
        own = {'accounts': {POOL: {'status': 'fresh', 'windows': [{'id': 'allowance', 'used_pct': 1.0}]}}}
        self.assertEqual(axis_scores.with_local_readings(own, overlaid, self.now)['accounts'], own['accounts'])

    # -- who decides, and what a quota message latches -----------------------------------------------------------------

    ZEN, GO = 'opencode/deepseek-v4.1-flash-free', 'opencode-go/glm-5.3'

    def with_go(self):
        import auto_bundles
        from router_test_support import fleet_with_derived_bundles
        acc = self.only(POOL)
        catalog = {'providers': {POOL: {'ok': True, 'type': 'paseo_provider', 'checked_at': 1,
                                        'models': [{'id': self.ZEN}, {'id': self.GO}]}}}
        return fleet_with_derived_bundles(acc, auto_bundles.derive(acc, catalog, 2))

    def bundle_of(self, acc, launch):
        bid, b = next((i, b) for i, b in acc['accounts'][POOL]['bundles'].items() if b.get('launch_model') == launch)
        return {**b, 'id': bid, 'account': POOL}

    def fail_bundle(self, acc, launch, message, dispatch_id, scope='account'):
        with tf._synthetic_doc_ctx(self.doc):
            return provider_retry.record(self.bundle_of(acc, launch), {
                'failure_code': provider_retry.TURN_FAILED_CODE, 'provider_message': message,
                'at': self.now.isoformat()}, dispatch_id=dispatch_id, accounts=acc,
                classify=lambda *args: (scope, {'source': 'system_one'}))

    def latch(self):
        return (quota_exhaustion._read(quota_exhaustion.store_path()).get('accounts') or {}).get(POOL) or {}

    def test_a_quota_message_latches_only_the_windows_the_failing_bundle_draws(self):
        """Rule (provider_retry.record, dispatch_registry.quota_latch): a used-up quota latches the windows the failing
        bundle draws. A Go bundle draws none of the account's windows, so its limit takes nothing of the free allowance
        out (the classifier's hold of its own prefix pool stands), and the allowance's limit takes nothing of Go out."""
        acc = self.with_go()
        self.fail_bundle(acc, self.GO, 'Go plan: monthly limit reached, resets Dec 1', 'g1')
        self.assertFalse(self.latch().get('blocked'), self.latch())
        self.assertEqual(set(provider_retry.active()), {f'{POOL}#opencode-go'})
        zen = next(c for c in self.rows(self.plan(self.discovered(acc))) if c['bundle'].get('launch_model') == self.ZEN)
        self.assertFalse(zen['excluded'], zen['excluded'])
        self.assertFalse(zen.get('available_in_hours'))
        provider_retry.clear_key(f'{POOL}#opencode-go')
        self.fail_bundle(acc, self.ZEN, LIMIT, 'z1')
        self.assertEqual(self.latch().get('windows'), ['allowance'])
        held = self.discovered(acc)
        import quota_pools
        with tf._synthetic_doc_ctx(self.doc):
            self.assertEqual(quota_pools.launch_blockers(self.bundle_of(acc, self.GO), held, {'accounts': {}}, self.now), [])
            self.assertTrue(quota_pools.launch_blockers(self.bundle_of(acc, self.ZEN), held, {'accounts': {}}, self.now))

    def test_an_account_that_declares_no_window_latches_as_a_whole(self):
        """Where the failing bundle's windows are unknown because its account declares no quota window at all, the
        whole account is the smallest true scope (dispatch_registry.quota_latch); where the account declares windows
        the bundle does not draw, nothing is latched."""
        import dispatch_registry
        acc = self.with_go()
        with tf._synthetic_doc_ctx(self.doc):
            self.assertEqual(dispatch_registry.quota_latch(self.bundle_of(acc, self.ZEN), acc), (True, ['allowance']))
            self.assertEqual(dispatch_registry.quota_latch(self.bundle_of(acc, self.GO), acc), (False, None))
            bare = copy.deepcopy(acc)
            bare['accounts']['meter'] = {'plan': 'API', 'billing': 'metered', 'verified': True, 'windows': {},
                                         'bundles': {'m': {'provider': 'claude', 'harness': 'claude', 'model': 'claude-sonnet-5-5'}}}
            self.assertEqual(dispatch_registry.quota_latch({**bare['accounts']['meter']['bundles']['m'], 'account': 'meter'},
                                                           bare), (True, None))

    def test_the_quota_pattern_decides_only_where_the_classifier_answered_account_or_nothing(self):
        """Classifier authority (provider_retry docstring): the quota pattern turns an `account` answer or no answer
        (`unknown`) into a quota record of the windows the bundle draws; `model`, `transient`, `none` and `task_input`
        stand as answered."""
        acc = self.with_go()
        for n, (scope, latched, held) in enumerate((('account', True, False), ('unknown', True, False),
                                                    ('model', False, True), ('transient', False, True),
                                                    ('none', False, False), ('task_input', False, False))):
            with self.subTest(scope=scope):
                (self.state / 'quota-exhaustion.json').unlink(missing_ok=True)
                (self.state / provider_retry.FILE).unlink(missing_ok=True)
                self.fail_bundle(acc, self.ZEN, LIMIT, f's{n}', scope=scope)
                self.assertEqual(bool(self.latch().get('blocked')), latched, self.latch())
                self.assertEqual(bool(provider_retry.active()), held, provider_retry.active())

    # -- the limit error ------------------------------------------------------------------------------------------------

    def expected_resets(self):
        """(message, reset) for each documented form, computed here from the sentence, not by the parser."""
        local = self.now.astimezone()
        clock = local.replace(hour=23, minute=59, second=0, microsecond=0)
        if clock <= local:
            clock = (local.replace(tzinfo=None) + timedelta(days=1)).replace(hour=23, minute=59, second=0,
                                                                             microsecond=0).astimezone()
        dated = (local.replace(tzinfo=None) + timedelta(days=2)).replace(hour=0, minute=59, second=0, microsecond=0)
        date_text = dated.strftime('%b %d, %Y, 12:59AM').replace(' 0', ' ')
        return [(CLOCK, clock), (DATE.format(date=date_text), dated.astimezone()),
                (COUNTDOWN, self.now + timedelta(seconds=74000))]

    def test_each_documented_limit_sentence_holds_the_pool_until_its_stated_reset_then_frees_it(self):
        for n, (message, reset) in enumerate(self.expected_resets()):
            with self.subTest(message=message):
                (self.state / 'quota-exhaustion.json').unlink(missing_ok=True)   # a fresh latch per sentence
                acc = self.only(POOL, 'claude')
                self.assertIsNone(self.fail_turn(acc, message, f'd{n}'))     # not a provider hold
                self.assertEqual(provider_retry.active(), {})
                held = self.discovered(acc)
                self.assertEqual(held['_discovery']['accounts'][POOL]['exhausted_windows'], ['allowance'])
                self.assertEqual(datetime.fromisoformat(held['_discovery']['accounts'][POOL]['exhausted_resets_at']),
                                 reset)
                self.assertNotIn('exhausted_windows', held['_discovery']['accounts']['claude'])  # other accounts untouched
                decision = self.plan(held)
                rows = self.rows(decision)
                self.assertTrue(rows)
                hours = (reset - self.now).total_seconds() / 3600
                for c in rows:     # every bundle drawing the allowance: a quota wait to the stated reset
                    self.assertFalse(c['excluded'], c['excluded'])
                    self.assertAlmostEqual(c['available_in_hours'], hours, places=3)
                    self.assertEqual(c['waiting_on'], 'allowance')
                # Held until the stated reset, not until the latches' re-probe period, when the two differ.
                before = reset - timedelta(minutes=1)
                with patch.object(quota_exhaustion, 'datetime', wraps=datetime) as clock:
                    clock.now.side_effect = lambda tz=None: before
                    self.assertEqual(self.discovered(acc)['_discovery']['accounts'][POOL]['exhausted_windows'],
                                     ['allowance'])
                after = reset + timedelta(minutes=1)
                with patch.object(quota_exhaustion, 'datetime', wraps=datetime) as clock:
                    clock.now.side_effect = lambda tz=None: after
                    freed = self.discovered(acc)
                self.assertNotIn('exhausted_windows', freed['_discovery']['accounts'][POOL])
                self.assertTrue(all(not c['excluded'] and not c['available_in_hours']
                                    for c in self.rows(self.plan(freed, now=after))))

    def test_a_sentence_with_no_readable_reset_waits_until_the_latches_re_probe_time(self):
        """Rule (quota_exhaustion.due_at, brief point 5): a pool at 0 with no stated reset is still a quota-held
        candidate with a known time, its re-probe time. The plan defers to it, the launch blocker and the named route
        say it, and no text says "refresh quota or replan"."""
        import quota_pools
        for message in (LIMIT, UNREADABLE):
            with self.subTest(message=message):
                (self.state / 'quota-exhaustion.json').unlink(missing_ok=True)   # a fresh latch per sentence
                acc = self.only(POOL)
                self.fail_turn(acc, message, f'd-{len(message)}')
                held = self.discovered(acc)
                due = self.now + timedelta(hours=account_state.LATCH_RETRY_HOURS)
                row = held['_discovery']['accounts'][POOL]
                self.assertEqual(row['exhausted_windows'], ['allowance'])
                self.assertEqual(datetime.fromisoformat(row['exhausted_resets_at']), due)
                decision = self.plan(held)
                hours = account_state.LATCH_RETRY_HOURS
                self.assertEqual(decision['decision']['action'], 'defer')
                self.assertGreaterEqual(decision['decision']['start_after_hours'], hours - 0.01)
                self.assertEqual(decision['decision']['wait_for']['window'], 'allowance')
                for c in self.rows(decision):
                    self.assertFalse(c['excluded'], c['excluded'])
                    self.assertAlmostEqual(c['available_in_hours'], hours, places=3)
                    self.assertFalse([n for n in c['notes'] if 'replan' in n], c['notes'])
                    self.assertTrue([n for n in c['notes'] if 're-probed' in n and quota_exhaustion.moment_text(due) in n],
                                    c['notes'])
                bid, bundle = next(iter(held['accounts'][POOL]['bundles'].items()))
                with tf._synthetic_doc_ctx(self.doc):
                    blockers = quota_pools.launch_blockers({**bundle, 'id': bid, 'account': POOL}, held, {}, self.now)
                self.assertEqual([w for w, _ in blockers], ['allowance'])
                self.assertIn(quota_exhaustion.moment_text(due), blockers[0][1])
                self.assertIn('re-probed', blockers[0][1])
                self.assertNotIn('replan', blockers[0][1])
                named = self.plan(held, constraint_bundle=bid)
                self.assertEqual(named['decision']['action'], 'defer')
                self.assertGreaterEqual(named['decision']['start_after_hours'], hours - 0.01)
                # A forced route checked at record (route_judge._check_record_launch) says the same time.
                import route_judge
                from types import SimpleNamespace
                with tf._synthetic_doc_ctx(self.doc), self.assertRaises(ValueError) as refused:
                    route_judge._check_record_launch(SimpleNamespace(account=POOL), SimpleNamespace(model=bundle['model']),
                                                     {**bundle, 'id': bid, 'account': POOL}, held, {'accounts': {}}, self.now)
                said = str(refused.exception)
                self.assertIn(quota_exhaustion.moment_text(due), said)
                self.assertIn('start_after_hours', said)
                self.assertNotIn('remaining capacity', said)
                for moment, latched in ((due - timedelta(minutes=1), True), (due + timedelta(minutes=1), False)):
                    with patch.object(quota_exhaustion, 'datetime', wraps=datetime) as clock:
                        clock.now.side_effect = lambda tz=None, m=moment: m
                        row = self.discovered(acc)['_discovery']['accounts'][POOL]
                    self.assertEqual('exhausted_windows' in row, latched, moment)

    def test_with_only_the_held_pool_signed_in_the_answer_is_the_quota_wait_to_the_reset(self):
        message, reset = self.expected_resets()[2]
        acc = self.only(POOL)
        self.fail_turn(acc, message, 'd-wait')
        decision = self.plan(self.discovered(acc))
        self.assertEqual(decision['decision']['action'], 'defer')
        hours = (reset - self.now).total_seconds() / 3600
        self.assertGreaterEqual(decision['decision']['start_after_hours'], hours - 0.01)
        self.assertEqual(decision['decision']['wait_for']['window'], 'allowance')

    def test_a_held_pool_blocks_a_launch_until_its_reset(self):
        import quota_pools
        message, reset = self.expected_resets()[2]
        acc = self.only(POOL)
        self.fail_turn(acc, message, 'd-launch')
        held = self.discovered(acc)
        bid, bundle = next(iter(held['accounts'][POOL]['bundles'].items()))
        with tf._synthetic_doc_ctx(self.doc):
            blockers = quota_pools.launch_blockers({**bundle, 'id': bid, 'account': POOL}, held, {}, self.now)
            free = quota_pools.launch_blockers({**bundle, 'id': bid, 'account': POOL}, self.only(POOL), {}, self.now)
        self.assertEqual([wid for wid, _ in blockers], ['allowance'])
        self.assertIn(quota_exhaustion.moment_text(reset), blockers[0][1])
        self.assertIn('the reset the provider stated', blockers[0][1])
        self.assertEqual(free, [])


class PaidLatchReadingTests(unittest.TestCase):
    """Review round 2 N2. Rule (quota_exhaustion.exhausted_windows, candidate_eval.evaluate_bundle): where the account has
    a usage reader, the reading decides. A good reading newer than the latch releases it; the latch's end is a wait only
    while no such reading exists; and a window that has a fresh reading is never "quota reading unavailable"."""

    def setUp(self):
        self.state = Path(tempfile.mkdtemp(prefix='paid-latch-'))
        self.addCleanup(shutil.rmtree, self.state, ignore_errors=True)
        env = patch.dict(os.environ, {'ROUTER_STATE_DIR': str(self.state)})
        env.start()
        self.addCleanup(env.stop)
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.base = sim.fresh_accounts(shipped_template(), snapshot())
        self.windows = sorted(self.base['accounts']['claude']['windows'])
        self.task = {'task_id': 'paid-latch', 'objective': 'fixture', 'work_kind': 'code_implementation',
                     'difficulty': 'hard', 'expected_hours': 1.0, 'needs': {'local_only': True}}
        self.task['profile'] = profile_for(self.task)

    def plan_after(self, latch_at, fetched, used):
        """A latch on claude's windows with no stated reset, published by discovery beside a reading `used`% used."""
        store = self.state / 'quota-exhaustion.json'
        store.unlink(missing_ok=True)
        quota_exhaustion.record('claude', 'quota exceeded', windows=self.windows, at=latch_at.isoformat(), path=store)
        doc = sim.synthetic_document(self.base, self.now, ())
        doc['providers']['claude']['fetched_at'] = fetched.isoformat()
        for window in doc['providers']['claude']['windows']:
            window['used_pct'] = used
        inventory = snapshot()
        provider_inventory._latch_exhausted_quota(inventory['accounts'], self.state, doc['providers'])
        accounts = sim.fresh_accounts(shipped_template(), inventory)
        plan = allocator.plan(copy.deepcopy(self.task), sim.snapshot(doc, accounts, self.now), self.now,
                              accounts=accounts, overrides=[], host_policy_config={'allowed_accounts': ['claude']},
                              read_local_state=False)
        return inventory['accounts']['claude'], [c for c in plan['ranking'] if c['bundle'].get('account') == 'claude']

    def test_a_good_reading_newer_than_the_latch_releases_it_and_an_older_one_leaves_a_wait_not_an_unread_window(self):
        import candidate_eval
        for used in (10, 98):
            with self.subTest(used=used, reading='newer'):
                row, rows = self.plan_after(self.now - timedelta(minutes=10), self.now - timedelta(minutes=1), used)
                self.assertNotIn('exhausted_windows', row)
                if used == 10:      # ample: with the latch released nothing holds a claude bundle back
                    for c in rows:
                        self.assertFalse(c['excluded'] or c.get('available_in_hours'), (c['id'], c['excluded']))
            with self.subTest(used=used, reading='older'):
                latch_at = self.now - timedelta(minutes=1)
                row, rows = self.plan_after(latch_at, self.now - timedelta(minutes=5), used)
                self.assertEqual(row['exhausted_windows'], self.windows)
                due = quota_exhaustion.due_at({'observed_at': latch_at.isoformat()})
                for c in rows:
                    self.assertFalse([e for e in c['excluded'] if candidate_eval.QUOTA_READING_UNAVAILABLE in e],
                                     (c['id'], c['excluded']))
                    self.assertIn(c.get('waiting_on'), self.windows, c['id'])
                    self.assertAlmostEqual(c['available_in_hours'], (due - self.now).total_seconds() / 3600, places=3)


class StatedResetParseTests(unittest.TestCase):
    """The boundary parse of the vendor's sentence (quota_exhaustion.stated_reset): its documented forms and nothing else."""
    AT = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)

    def test_the_documented_forms(self):
        local = self.AT.astimezone()
        at_21 = local.replace(hour=21, minute=0, second=0, microsecond=0)
        if at_21 <= local:
            at_21 = (local.replace(tzinfo=None) + timedelta(days=1)).replace(hour=21, minute=0).astimezone()
        cases = {"They reset at 21:00": at_21,
                 "They reset at Nov 26, 2026, 12:59AM": datetime(2026, 11, 26, 0, 59).astimezone(),
                 'reintentando en 74000s - intento n.º 1': self.AT + timedelta(seconds=74000),
                 'retrying in 120s': self.AT + timedelta(seconds=120)}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(quota_exhaustion.stated_reset(text, self.AT), expected)

    def test_a_sentence_it_cannot_read_gives_no_reset(self):
        for text in (LIMIT, UNREADABLE, 'They reset at 25:00', 'They reset at 13:00 PM', 'retrying in 12 minutes', '',
                     'They reset at Foo 28, 2026, 12:59AM', 'They reset at Oct 32, 2026, 12:59AM'):
            with self.subTest(text=text):
                self.assertIsNone(quota_exhaustion.stated_reset(text, self.AT))

    def test_a_zone_the_sentence_states_is_used(self):
        """A clock or date followed by a zone (UTC, GMT+8, an IANA name) is that zone's wall time; with none it is the
        machine's local time (the vendor's client prints wall-clock time for its user)."""
        from zoneinfo import ZoneInfo
        new_york = ZoneInfo('America/New_York')
        cases = {'Usage limit reached. It resets at 15:00 UTC': datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc),
                 'Usage limit reached. It resets at 09:00 UTC': datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc),
                 'Usage limit reached. It resets at 23:30 GMT+8': datetime(2026, 11, 24, 23, 30,
                                                                         tzinfo=timezone(timedelta(hours=8))),
                 'It resets at 3:00 PM (America/New_York)': datetime(2026, 11, 24, 15, 0, tzinfo=new_york),
                 'They reset at Nov 26, 2026, 12:59AM UTC': datetime(2026, 11, 26, 0, 59, tzinfo=timezone.utc)}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(quota_exhaustion.stated_reset(text, self.AT), expected)
        with self.subTest('a zone name no zone database knows is unreadable'):
            self.assertIsNone(quota_exhaustion.stated_reset('It resets at 3:00 PM (Nowhere/Atlantis)', self.AT))

    def test_a_zone_token_is_read_only_when_it_names_one_offset_and_the_machine_zone_only_when_there_is_none(self):
        """Rule (quota_exhaustion._stated_zone, review round 2 N1): `Z` and a numeric offset (`+05:00`, `-0800`) name one
        offset and are read; an abbreviation (PST, EST, JST) or any other zone-shaped token names no single offset, so the
        sentence is unreadable (no reset: the re-probe period applies). The machine's zone applies only when the time
        is followed by no zone token at all."""
        plus5 = timezone(timedelta(hours=5))
        read = {'It resets at 15:00Z': datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc),
                'It resets at 15:00 Z': datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc),
                'It resets at 20:00 +05:00': datetime(2026, 11, 24, 20, 0, tzinfo=plus5),
                'It resets at 20:00+0500': datetime(2026, 11, 24, 20, 0, tzinfo=plus5),
                'It resets at 7:00 AM -08:00': datetime(2026, 11, 24, 7, 0, tzinfo=timezone(timedelta(hours=-8))),
                'They reset at Nov 26, 2026, 12:59AM +05:00': datetime(2026, 11, 26, 0, 59, tzinfo=plus5)}
        for text, expected in read.items():
            with self.subTest(text=text):
                self.assertEqual(quota_exhaustion.stated_reset(text, self.AT), expected)
        for text in ('It resets at 3:00 PM PST', 'It resets at 15:00 EST.', 'It resets at 23:00 JST',
                     'They reset at Nov 26, 2026, 12:59AM CET', 'It resets at 15:00 (PDT)', 'It resets at 15:00 +25:00'):
            with self.subTest(text=text):
                self.assertIsNone(quota_exhaustion.stated_reset(text, self.AT))
        local = quota_exhaustion.stated_reset('It resets at 21:00, try again later', self.AT)
        self.assertEqual(local.utcoffset(), self.AT.astimezone().utcoffset())

    def test_a_reset_not_in_the_future_or_beyond_every_quota_period_is_unreadable(self):
        """A stated reset must lie after the observation and no further than the longest reset period any window of
        the shipped plan templates has (declared data); otherwise the latch's re-probe period applies."""
        longest = quota_exhaustion.longest_reset_hours()
        self.assertIsNotNone(longest)
        templates = provider_inventory.shipped_plan_templates()
        self.assertEqual(longest, max(w['hours'] for plans in templates.values() for t in plans.values()
                                      if isinstance(t, dict) for w in (t.get('windows') or {}).values()
                                      if isinstance(w.get('hours'), (int, float))))
        beyond = self.AT + timedelta(hours=longest) + timedelta(days=1)
        for text in ('They reset at Oct 28, 2025, 12:59AM', 'Free usage limit reached, retrying in 0s',
                     'Free usage limit reached, retrying in 99999999999s',
                     f"They reset at {beyond.strftime('%b')} {beyond.day}, {beyond.year}, 12:59AM UTC",
                     f'retrying in {int(longest * 3600) + 60}s'):
            with self.subTest(text=text):
                self.assertIsNone(quota_exhaustion.stated_reset(text, self.AT))
        self.assertEqual(quota_exhaustion.stated_reset(f'retrying in {int(longest * 3600) - 60}s', self.AT),
                         self.AT + timedelta(seconds=int(longest * 3600) - 60))

    def test_a_plan_templates_document_of_the_wrong_shape_raises_the_reference_readers_own_error(self):
        """Rule (quota_exhaustion.longest_reset_hours, review round 2 N6): the bound is read from plan-templates.json of
        the held set through public_data.document; a document not in the shape it reads raises
        public_data.ReferenceDataError, the reader's own data error, and the reader's own errors pass through: never an
        AttributeError, never a silent bound."""
        import public_data
        window = {'hours': 5}
        for doc in ([], {'claude': []}, {'claude': {'max': []}}, {'claude': {'max': {'windows': []}}},
                    {'claude': {'max': {'windows': {'five_hour': []}}}}, {'claude': {'max': {'windows': {'w': window},
                                                                                              'cadence': []}}}):
            with self.subTest(doc=doc), patch.object(public_data, 'document', return_value=doc):
                with self.assertRaises(public_data.ReferenceDataError):
                    quota_exhaustion.longest_reset_hours()
                with self.assertRaises(public_data.ReferenceDataError):
                    quota_exhaustion.stated_reset(COUNTDOWN, self.AT)
        with patch.object(public_data, 'document', side_effect=public_data.ReferenceDataError('unreadable')):
            with self.assertRaises(public_data.ReferenceDataError):
                quota_exhaustion.longest_reset_hours()
        with patch.object(public_data, 'document', return_value={'claude': {'max': {'windows': {'w': window}}}}):
            self.assertEqual(quota_exhaustion.longest_reset_hours(), 5)

    def test_the_limit_sentences_are_confirmed_quota_failures(self):
        for text in (LIMIT, CLOCK, COUNTDOWN, UNREADABLE):
            self.assertTrue(quota_exhaustion.confirmed(text), text)
        for text in ('HTTP 429: temporary rate limit', 'Network timeout', 'fixture worker failure'):
            self.assertFalse(quota_exhaustion.confirmed(text), text)


class ObservationStoreTests(unittest.TestCase):
    """The account's observation at its limit error (quota_exhaustion.record / observation): local state, kept apart
    from the latch, replaced by the next observation; written only when the sentence states both its reset and its
    period."""

    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix='observation-')) / 'quota-exhaustion.json'
        self.addCleanup(shutil.rmtree, self.path.parent, ignore_errors=True)
        self.at = datetime.now(timezone.utc).replace(microsecond=0)

    def test_a_sentence_stating_reset_and_period_is_an_observation_and_survives_the_latch(self):
        self.assertEqual(quota_exhaustion.stated_period(CLOCK), 24.0)
        self.assertTrue(quota_exhaustion.record(POOL, CLOCK, windows=['allowance'], at=self.at.isoformat(), path=self.path))
        seen = quota_exhaustion.observation(POOL, path=self.path)
        self.assertEqual((seen['observed_at'], seen['resets_at'], seen['period_hours']),
                         (self.at, quota_exhaustion.stated_reset(CLOCK, self.at), 24.0))
        quota_exhaustion.clear(POOL, path=self.path)
        self.assertEqual(quota_exhaustion.observation(POOL, path=self.path), seen)
        self.assertIsNone(quota_exhaustion.observation('another', path=self.path))

    def test_an_error_with_no_reset_observes_nothing_and_the_next_observation_replaces_the_last(self):
        for n, text in enumerate((LIMIT, UNREADABLE)):
            self.assertTrue(quota_exhaustion.record(POOL, text, at=(self.at + timedelta(minutes=n)).isoformat(),
                                                    path=self.path), text)
            self.assertIsNone(quota_exhaustion.observation(POOL, path=self.path), text)
        # A reset with no stated period is an observation without one (the period then comes from the measurement).
        quota_exhaustion.record(POOL, COUNTDOWN, at=(self.at + timedelta(minutes=5)).isoformat(), path=self.path)
        seen = quota_exhaustion.observation(POOL, path=self.path)
        self.assertEqual(seen['resets_at'], self.at + timedelta(minutes=5, seconds=74000))
        self.assertNotIn('period_hours', seen)
        first, second = self.at + timedelta(hours=1), self.at + timedelta(hours=2)
        quota_exhaustion.record(POOL, CLOCK, at=first.isoformat(), path=self.path)
        quota_exhaustion.record(POOL, CLOCK, at=second.isoformat(), path=self.path)
        self.assertEqual(quota_exhaustion.observation(POOL, path=self.path)['observed_at'], second)

    def test_a_reset_given_as_data_beats_the_sentence_and_the_default_within_the_same_bounds(self):
        """Rule (quota_exhaustion.record): a reset the vendor gave as data (`resets_at`) beats the sentence's own and the
        latches' re-probe period; it is held to the bounds of a stated reset (after the error, within the longest plan
        period), else the sentence or the default applies."""
        given = self.at + timedelta(hours=20, minutes=4)
        for n, text in enumerate((LIMIT, CLOCK, COUNTDOWN)):
            with self.subTest(text=text):
                at = self.at + timedelta(minutes=n)
                quota_exhaustion.record(POOL, text, at=at.isoformat(), path=self.path, resets_at=given)
                row = quota_exhaustion._read(self.path)['accounts'][POOL]
                self.assertEqual(quota_exhaustion.due_at(row), given)
                self.assertEqual(quota_exhaustion.observation(POOL, path=self.path)['resets_at'], given)
        longest = quota_exhaustion.longest_reset_hours()
        for n, bad in enumerate((self.at - timedelta(hours=1), self.at + timedelta(hours=longest + 1))):
            with self.subTest(bad=bad):
                at = self.at + timedelta(minutes=10 + n)
                quota_exhaustion.record(POOL, COUNTDOWN, at=at.isoformat(), path=self.path, resets_at=bad)
                row = quota_exhaustion._read(self.path)['accounts'][POOL]
                self.assertEqual(quota_exhaustion.due_at(row), at + timedelta(seconds=74000))


class OpenCodeFailureResetTests(unittest.TestCase):
    """axis_scores.opencode_failure_reset: OpenCode keeps a failed turn's error object, retry-after included, in its own
    session store; Paseo's turn failure keeps only the text. The reset is read back for the text the router got."""
    # The client's error object, verbatim from a live run on 2026-11-24 (parent's message; fixture data, no turn is run).
    ERROR = {"type": "provider.free-limit", "message": LIMIT, "status": 429, "response": {"body": json.dumps(
        {"type": "error", "error": {"type": "FreeUsageLimitError", "message": "Free usage limit reached. Free models work in "
                                    "OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: "
                                    "https://opencode.ai/go"}})}, "retryAfterMs": 72230000}

    def setUp(self):
        root = Path(tempfile.mkdtemp(prefix='opencode-store-'))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        self.db = root / 'opencode.db'
        self.at = datetime(2026, 11, 24, 4, 2, 40, tzinfo=timezone.utc)
        env = patch.dict(os.environ, {'OPENCODE_DB_PATH': str(self.db)})
        env.start()
        self.addCleanup(env.stop)

    def store(self, rows):
        conn = sqlite3.connect(self.db)
        conn.execute('CREATE TABLE IF NOT EXISTS session_message (id TEXT, session_id TEXT, type TEXT, seq INTEGER, '
                     'time_created INTEGER, time_updated INTEGER, data TEXT)')
        for n, (stamp, error) in enumerate(rows):
            ms = int(stamp.timestamp() * 1000)
            conn.execute('INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)', (
                f'm{n}', 's', 'idle', n, ms, ms, json.dumps({'time': {'created': ms}, 'outcome': 'failed', 'error': error})))
        conn.commit()
        conn.close()

    def test_the_newest_row_with_the_same_text_gives_its_stamp_plus_the_retry_after(self):
        failed = self.at - timedelta(seconds=11)
        self.store([(self.at - timedelta(days=12), self.ERROR),                               # an earlier failure
                    (failed, self.ERROR),
                    (failed + timedelta(seconds=2), {**self.ERROR, 'message': 'another error', 'retryAfterMs': 5}),
                    (self.at + timedelta(minutes=5), {**self.ERROR, 'retryAfterMs': 1})])     # after this failure
        expected = failed + timedelta(milliseconds=self.ERROR['retryAfterMs'])
        self.assertEqual(axis_scores.opencode_failure_reset(LIMIT, self.at), expected)
        self.assertIsNone(axis_scores.opencode_failure_reset('Free usage', self.at, cut_at=2000))   # not a cut text
        self.assertEqual(axis_scores.opencode_failure_reset(LIMIT[:20], self.at, cut_at=20), expected)
        self.assertIsNone(axis_scores.opencode_failure_reset('another error, not stored', self.at))
        self.assertIsNone(axis_scores.opencode_failure_reset(LIMIT, self.at, read_local_db=False))

    def test_a_turn_failure_latches_the_pool_until_the_reset_given_as_data(self):
        """End to end through the real writer (provider_retry.record -> quota_exhaustion.record): the sentence states
        no time, the store holds the retry-after, and the latch ends at it, not at the re-probe period."""
        tf.setUpModule()
        accounts = sim.fresh_accounts(shipped_template(), snapshot())
        acc = copy.deepcopy(accounts)
        acc['accounts'] = {POOL: acc['accounts'][POOL]}
        state = Path(tempfile.mkdtemp(prefix='latch-'))
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        self.store([(now - timedelta(seconds=5), self.ERROR)])
        bid, bundle = next(iter(acc['accounts'][POOL]['bundles'].items()))
        with patch.dict(os.environ, {'ROUTER_STATE_DIR': str(state), 'ROUTER_PROVIDER_RETRY': str(state / provider_retry.FILE)}), \
                tf._synthetic_doc_ctx(tf.document(accounts['models'])):
            self.assertIsNone(provider_retry.record({**bundle, 'id': bid, 'account': POOL}, {
                'failure_code': provider_retry.TURN_FAILED_CODE, 'provider_message': LIMIT, 'at': now.isoformat()},
                dispatch_id='d-data', accounts=acc, classify=lambda *args: ('account', {'source': 'system_one'})))
            found = quota_exhaustion.ends(POOL)
        self.assertEqual(found, (now - timedelta(seconds=5) + timedelta(milliseconds=self.ERROR['retryAfterMs']), True))


class BoundedSpendTests(unittest.TestCase):
    """axis_scores.opencode_local_spend bounded by time: only the messages opencode.db stamps (`time_created`, ms) at or
    after `since` and before `until`, at the model's list price; discovery's totals carry no time and are left out."""

    def test_a_bounded_read_sums_only_the_messages_inside_the_period(self):
        root = Path(tempfile.mkdtemp(prefix='opencode-db-'))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        db = root / 'opencode.db'
        until = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
        since = until - timedelta(hours=24)
        tokens = {'input': 1000, 'output': 400, 'reasoning': 100, 'cache': {'read': 50, 'write': 0}}
        stamps = {'before': since - timedelta(hours=1), 'at_start': since, 'inside': until - timedelta(hours=1),
                  'at_end': until}
        conn = sqlite3.connect(db)
        conn.execute('CREATE TABLE message (id TEXT, time_created INTEGER, data TEXT)')
        for key, stamp in stamps.items():
            conn.execute('INSERT INTO message VALUES (?, ?, ?)', (key, int(stamp.timestamp() * 1000), json.dumps(
                {'role': 'assistant', 'modelID': FREE_MODEL, 'providerID': 'opencode', 'tokens': tokens})))
        conn.commit()
        conn.close()
        accounts = sim.fresh_accounts(shipped_template(), snapshot())
        models = axis_scores.five_axis_overlay(copy.deepcopy(accounts), read_local_state=False)['models']
        name, price = axis_scores.opencode_list_prices(models)[FREE_MODEL]
        one = axis_scores.opencode_token_usd(price, 1000, 500, 50, 0)
        discovery = {'accounts': {'opencode': {'usage': {'models': {FREE_MODEL: {'input': 10 ** 9, 'output': 0}}}}}}
        with patch.dict(os.environ, {'OPENCODE_DB_PATH': str(db)}):
            bounded, _ = axis_scores.opencode_local_spend(models, discovery, since=since, until=until)
            unbounded, _ = axis_scores.opencode_local_spend(models, None)
        self.assertAlmostEqual(bounded[name], round(2 * one, 4))           # at_start and inside; not before, not at_end
        self.assertAlmostEqual(unbounded[name], round(4 * one, 4))


if __name__ == '__main__':
    unittest.main()
