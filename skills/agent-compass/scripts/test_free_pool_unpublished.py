"""A free pool whose vendor publishes no capacity (owner, 2026-11-24).

Rules under test (skills/agent-compass/references/router-internals.md, section 2, free pools):
  - the declaration is data: a plan's `capacity_unpublished` in plan-templates.json, a dated vendor citation; a free pool
    with it is an ordinary candidate of automatic routing, without it (and any subscription whose plan is unknown) it
    stays in mode B, whatever the family is called;
  - its quota term is 0 because nothing is paid (quota_economy.unpaid_allowance), and the plan's snapshot says why;
  - the vendor's limit error, recorded by the real writer (provider_retry.record -> quota_exhaustion.record), is the
    exhaustion of the allowance window with the reset the error states; held until that reset, a quota wait like any
    used-up window; eligible again after it; a sentence with no readable reset falls to the latches' re-probe period.
No winner, share or score is asserted.
"""
import copy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
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
from test_fleet_parity import shipped_template, snapshot

POOL = 'opencode'
# The vendor's own sentences (anomalyco/opencode#59387 and the issues it collects, read 2026-11-24; research report
# §1.4 and §3), used as fixture text: no real turn is run.
LIMIT = 'Free usage limit reached. Connect a provider or subscribe to OpenCode Go to keep going, or switch models.'
CLOCK = "You've used today's free messages. They reset at 11:59 PM"
DATE = "You've used today's free messages. They reset at {date}"
COUNTDOWN = 'Free usage exceeded, subscribe to Go reintentando en 74000s - intento n.º 1'
UNREADABLE = "You've used today's free messages. They reset at tomorrow morning"


def setUpModule():
    tf.setUpModule()


class UnpublishedFreePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts = sim.fresh_accounts(shipped_template(), snapshot())
        cls.doc = tf.document(cls.accounts['models'])
        cls.declaration = copy.deepcopy(provider_inventory.shipped_plan_templates()['opencode']['free'])

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

    def plan(self, acc, now=None, **task):
        now = now or self.now
        acc = copy.deepcopy(acc)
        with tf._synthetic_doc_ctx(self.doc):
            usage = sim.snapshot(sim.synthetic_document(acc, now, sim.STATES['ample']), acc, now)
            return allocator.plan({'objective': 't', 'profile': dict(tf.PROFILE), **task}, usage, now, accounts=acc,
                                  overrides=[], records=[], host_policy_config={'allowed_accounts': list(acc['accounts'])})

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

    def test_the_shipped_declaration_is_a_dated_vendor_citation_on_a_zero_price_plan(self):
        declared = self.declaration['capacity_unpublished']
        self.assertEqual(declared['kind'], 'vendor')
        self.assertTrue(declared['url'].startswith('https://') and declared['as_of'] and declared['statement'])
        self.assertEqual(self.declaration['price_usd_per_month'], 0)

    def test_a_declared_pool_is_eligible_and_its_quota_term_is_zero_with_the_reason_recorded(self):
        decision = self.plan(self.accounts)
        rows = self.rows(decision)
        self.assertTrue(rows)
        for c in rows:
            self.assertFalse([e for e in c['excluded'] if 'capacity_unknown' in e or 'quota reading unavailable' in e],
                             c['excluded'])
        eligible = [c for c in rows if not c['excluded']]
        self.assertTrue(eligible, [c['excluded'] for c in rows])
        for c in eligible:
            self.assertEqual(c['components']['opportunity_cost'], 0.0)
            self.assertTrue(c['economy']['cost_basis'].startswith('0: '))
        # The snapshot keeps the reason on every eligible row it records (the pool alone, so no row depends on its rank).
        alone = self.plan(self.only(POOL))
        snap = ledger_snapshot.decision_snapshot(alone)
        kept = [r for r in snap['ranking'] if not r['excluded']]
        self.assertTrue(kept and all(r['economy'].get('cost_basis', '').startswith('0: ') for r in kept), kept)

    def test_the_same_pool_without_the_declaration_stays_in_mode_b(self):
        acc = copy.deepcopy(self.accounts)
        acc['plan_templates'] = {'opencode': {'free': {k: v for k, v in self.declaration.items()
                                                       if k != 'capacity_unpublished'}}}
        rows = self.rows(self.plan(acc))
        self.assertTrue(rows)
        self.assertTrue(all(any('capacity_unknown (quota unknown, spend unpriced)' in e for e in c['excluded'])
                            for c in rows))

    def test_a_subscription_whose_plan_is_unknown_stays_in_mode_b(self):
        acc = copy.deepcopy(self.accounts)
        acc['accounts']['grok']['plan'] = 'a-plan-no-template-names'
        with tf._synthetic_doc_ctx(self.doc):
            spec = axis_scores.five_axis_overlay(copy.deepcopy(acc))['accounts']['grok']
        self.assertTrue(spec.get('capacity_unknown'))
        self.assertNotIn('capacity_unpublished', spec)

    def test_the_rule_is_the_declaration_not_the_family_name(self):
        acc = copy.deepcopy(self.accounts)
        acc['accounts']['acme-free'] = {**copy.deepcopy(acc['accounts'][POOL]), 'family': 'acme'}
        del acc['accounts'][POOL]
        acc['plan_templates'] = {'acme': {'free': copy.deepcopy(self.declaration)}}
        rows = self.rows(self.plan(acc), 'acme-free')
        eligible = [c for c in rows if not c['excluded']]
        self.assertTrue(eligible, [c['excluded'] for c in rows])
        self.assertTrue(all(c['components']['opportunity_cost'] == 0.0 for c in eligible))
        del acc['plan_templates']       # the same account with no declaration for its family: mode B
        self.assertTrue(all(any('capacity_unknown' in e for e in c['excluded'])
                            for c in self.rows(self.plan(acc), 'acme-free')))

    def test_a_model_launched_under_another_prefix_of_the_account_draws_nothing_of_the_free_allowance(self):
        # Rule: the plan's `allowance_launch_prefix` scopes the allowance; OpenCode Go connected on the same account
        # (`opencode-go/...`, a paid product) is neither free nor held by the free allowance's limit.
        import auto_bundles
        from router_test_support import fleet_with_derived_bundles
        zen, go = 'opencode/deepseek-v4.1-flash-free', 'opencode-go/glm-5.3'
        acc = self.only(POOL)
        catalog = {'providers': {POOL: {'ok': True, 'type': 'paseo_provider', 'checked_at': 1,
                                        'models': [{'id': zen}, {'id': go}]}}}
        acc = fleet_with_derived_bundles(acc, auto_bundles.derive(acc, catalog, 2))
        with tf._synthetic_doc_ctx(self.doc):
            spec = axis_scores.five_axis_overlay(copy.deepcopy(acc))['accounts'][POOL]
        draws = {b['launch_model']: b['draws'] for b in spec['bundles'].values() if b.get('launch_model') in (zen, go)}
        self.assertEqual(draws, {zen: ['allowance'], go: []})
        rows = {c['bundle'].get('launch_model'): c for c in self.rows(self.plan(acc))}
        self.assertFalse(rows[zen]['excluded'], rows[zen]['excluded'])
        self.assertTrue(rows[go]['excluded'])
        self.assertNotIn('cost_basis', rows[go].get('economy') or {})

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

    def test_a_sentence_with_no_readable_reset_holds_until_the_latches_re_probe_period(self):
        for message in (LIMIT, UNREADABLE):
            with self.subTest(message=message):
                (self.state / 'quota-exhaustion.json').unlink(missing_ok=True)   # a fresh latch per sentence
                acc = self.only(POOL)
                self.fail_turn(acc, message, f'd-{len(message)}')
                held = self.discovered(acc)
                self.assertEqual(held['_discovery']['accounts'][POOL]['exhausted_windows'], ['allowance'])
                self.assertNotIn('exhausted_resets_at', held['_discovery']['accounts'][POOL])
                self.assertTrue(all(any('allowance: quota confirmed exhausted' in e for e in c['excluded'])
                                    for c in self.rows(self.plan(held))))
                due = self.now + timedelta(hours=account_state.LATCH_RETRY_HOURS)
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
        self.assertIn(reset.isoformat(), blockers[0][1])
        self.assertEqual(free, [])


class StatedResetParseTests(unittest.TestCase):
    """The boundary parse of the vendor's sentence (quota_exhaustion.stated_reset): its documented forms and nothing else."""
    AT = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)

    def test_the_documented_forms(self):
        local = self.AT.astimezone()
        at_21 = local.replace(hour=21, minute=0, second=0, microsecond=0)
        if at_21 <= local:
            at_21 = (local.replace(tzinfo=None) + timedelta(days=1)).replace(hour=21, minute=0).astimezone()
        cases = {"They reset at 21:00": at_21,
                 "They reset at Oct 28, 2026, 12:59AM": datetime(2026, 10, 28, 0, 59).astimezone(),
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

    def test_the_limit_sentences_are_confirmed_quota_failures(self):
        for text in (LIMIT, CLOCK, COUNTDOWN, UNREADABLE):
            self.assertTrue(quota_exhaustion.confirmed(text), text)
        for text in ('HTTP 429: temporary rate limit', 'Network timeout', 'fixture worker failure'):
            self.assertFalse(quota_exhaustion.confirmed(text), text)


if __name__ == '__main__':
    unittest.main()
