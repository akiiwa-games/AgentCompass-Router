"""A gateway account pool is priced and gated on each member's own quota window (owner 2026-11-24: "可以用自己的窗口计价，
magpie内部有自己的选择规则，当它是均衡吧"), never as one merged window with the mean usage and the latest reset.

The rules these tests assert (router-internals.md, Account pools):
  R1 each counted member is one subscription's window with its own used share and reset;
  R2 balanced: the work is split evenly over the serving members (known, quota left; all known ones when none has),
     an owner assumption and not a measurement;
  R3 no new formula: each member's term is quota_economy's own on its C, U, e, reset and share; the pool's term is their
     sum, its price and need the members' mean;
  R4 gates read the members: the pool is exhausted only when every member is, its wait is the earliest member reset
     that restores capacity for the work, its full capacity the members' sum; members in step give today's figures;
  R5 what is not known is not capacity: an unread member, or one with no reset, adds nothing and takes no share.
No test pins a winner, a share or a price copied from output: each compares two computations a rule says are equal or
ordered."""
import copy
from datetime import timedelta
import itertools
import json
import unittest

import ledger_snapshot
import quota_economy
import quota_pools
import usage_reading
import test_five_axis_economy as fleet_fixture
from router_test_support import isolated_router_state


def setUpModule():
    isolated_router_state(unittest.addModuleCleanup, discovery=False, policy=False)


NOW = fleet_fixture.NOW
SPEC = {'id': 'seven_day', 'hours': 168.0}
CAP = 1000.0
W = ('pool-account', 'seven_day')
LADDER = fleet_fixture.ladder_for()


def same_plan(acc):
    """claude-2 on claude's plan, windows and three bundles (as test_five_axis_economy's symmetry test)."""
    acc['accounts']['claude-2']['plan'] = acc['accounts']['claude']['plan']
    acc['accounts']['claude-2']['windows'] = copy.deepcopy(acc['accounts']['claude']['windows'])
    three = {k: v for k, v in acc['accounts']['claude']['bundles'].items()
             if v['model'] in ('claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5') and v['effort'] == 'high'}
    acc['accounts']['claude']['bundles'] = three
    acc['accounts']['claude-2']['bundles'] = {f'claude-2__{k}': {**v, 'account': 'claude-2'} for k, v in three.items()}


def plan_with(*windows):
    """allocator.plan on the shipped fleet with claude-2's windows read from these collector readings."""
    def hook(usage):
        usage['accounts']['claude-2']['windows'] = [usage_reading.window_state(w, NOW) for w in windows]
    return fleet_fixture.Fleet().plan(0.5, 0.5, allowed=('claude', 'claude-2'), mutate=same_plan, usage_hook=hook)


def at(hours):
    return (NOW + timedelta(hours=hours)).isoformat()


def raw_pool(members, wid='seven_day'):
    """A collector reading of a pool: each member (used %, hours to its reset) under a neutral placeholder label."""
    return {'id': wid, 'used_pct': 0.0, 'resets_at': None, 'accounts': len(members),
            'members': [{'label': f'member-{n}', 'used_pct': used, 'resets_at': None if hours is None else at(hours)}
                        for n, (used, hours) in enumerate(members, 1)]}


def pool_state(members, spec=SPEC):
    return quota_pools.window_state(spec['id'], spec, usage_reading.window_state(raw_pool(members, spec['id']), NOW),
                                    {}, NOW, 24)


def single_state(used, hours, spec=SPEC):
    observed = usage_reading.window_state({'id': spec['id'], 'used_pct': used, 'resets_at': at(hours)}, NOW)
    return quota_pools.window_state(spec['id'], spec, observed, {}, NOW, 24)


def merged_state(used, hours, k):
    """The merged window of k members in step, as routing read it before 2026-11-24: one window holding k accounts'
    worth (full 100k), what is left k times one account's, the same reset."""
    one = single_state(used, hours)
    return {**one, 'full': 100.0 * k, 'remaining': one['remaining'] * k, 'target_remaining': one['target_remaining'] * k}


def economy(spend_max=60.0, lump=30.0):
    """The economy's window pricing with the ladder and the burst size held fixed (they come from the account's own
    candidates, which these tests do not vary)."""
    econ = object.__new__(quota_economy.FiveAxisEconomy)
    econ._tables = {}
    econ.spend_max = {W: spend_max}
    econ.lump = lambda wkey, reset_hours=None: lump
    econ.ladder = lambda wkey: LADDER
    return econ


def term(state, spend=40.0, hours=2.0, econ=None):
    """(cost, price, need) of `spend` on a window state, as apply_five_axis_economy's eval_windows composes them: each
    serving window's own cost at its share, summed; price and need the mean (R3)."""
    econ = econ or economy()
    recs = econ.member_records(W, state, CAP)
    gots = [econ.window_cost(W, rec, spend * share, hours, share=share) for _, rec, share in recs]
    return (sum(g['cost'] for g in gots), sum(g['lambda_start'] for g in gots) / len(gots),
            sum(g['need_start'] for g in gots) / len(gots))


GATE_FIELDS = ('known', 'full', 'remaining', 'target_remaining', 'reset_hours', 'end_hours', 'stale')


class MemberReadingTests(unittest.TestCase):
    def test_a_pooled_window_without_its_members_is_not_read(self):
        # Boundary (brief): an old cached reading without member windows is refetched or skipped, never migrated or
        # priced as one merged window.
        old = usage_reading.window_state({'id': 'seven_day', 'used_pct': 15.5, 'resets_at': at(146), 'accounts': 2}, NOW)
        self.assertIsNone(old['remaining_pct_before_reset'])
        self.assertIsNone(old['observed_remaining_pct'])
        self.assertFalse(quota_pools.window_state('seven_day', SPEC, old, {}, NOW, 24)['known'])

    def test_the_pooled_reading_sums_the_members_and_resets_at_the_next_partial_refill(self):
        # R1/R4: what is left is the members' sum; the reset is the soonest reset of a member that spent some.
        got = usage_reading.window_state(raw_pool([(0.0, 146.0), (31.0, 30.0)]), NOW)
        self.assertEqual(got['remaining_pct_before_reset'], 100 + 69)
        self.assertEqual(got['resets_at'], at(30.0))
        self.assertEqual([m['position'] for m in got['members']], [1, 2])


class PoolOfOneTests(unittest.TestCase):
    def test_a_pool_of_one_member_is_the_single_account(self):
        # R1: a pool of one is that one account's window, in every gate field and in its economic term.
        for used, hours in ((0.0, 160.0), (31.0, 30.0), (88.0, 100.0), (100.0, 12.0)):
            with self.subTest(used=used, hours=hours):
                one, alone = pool_state([(used, hours)]), single_state(used, hours)
                self.assertEqual({k: one[k] for k in GATE_FIELDS}, {k: alone[k] for k in GATE_FIELDS})
                for a, b in zip(term(one), term(alone)):
                    self.assertAlmostEqual(a, b, places=12)
                self.assertEqual(quota_pools.after_reset(one, 168.0)['remaining'],
                                 quota_pools.after_reset(alone, 168.0)['remaining'])

    def test_end_to_end_a_pool_of_one_prices_and_waits_as_the_single_account(self):
        # R1 through allocator.plan: the same account read as a one-member pool or as its own window.
        for used, hours in ((31.0, 30.0), (100.0, 40.0)):
            with self.subTest(used=used):
                alone = rows_of(plan_with({'id': 'seven_day', 'used_pct': used, 'resets_at': at(hours)}), 'claude-2')
                one = rows_of(plan_with(dict(raw_pool([(used, hours)]), accounts=1)), 'claude-2')
                self.assertTrue(any(not r[1] for r in alone))      # the account is in routing: not a vacuous equality
                self.assertEqual(one, alone)


def rows_of(decision, account):
    return [(c['id'], c.get('excluded'), c.get('available_in_hours'), c['components'].get('quota_price'),
             c['components'].get('need_probability'), round(c['components'].get('opportunity_cost') or 0.0, 12))
            for c in decision['ranking'] if c['bundle']['account'] == account]


class InStepTests(unittest.TestCase):
    CASES = ((20.0, 100.0, 2), (31.0, 30.0, 2), (0.0, 150.0, 3), (97.0, 40.0, 2), (100.0, 20.0, 3))

    def test_members_in_step_price_as_the_merged_window(self):
        # R3/R4: members with the same used share and reset give the merged window's term (same total C, same price,
        # same need): capacity, use and burst all scale by the share.
        for used, hours, k in self.CASES:
            with self.subTest(used=used, hours=hours, k=k):
                pooled = term(pool_state([(used, hours)] * k))
                merged = term(merged_state(used, hours, k), econ=economy())
                for a, b in zip(pooled, merged):
                    self.assertAlmostEqual(a, b, places=9)

    def test_members_in_step_gate_as_the_merged_window(self):
        # R4: each gate gives today's figure for members in step: full capacity, what is left, reset, hours before the
        # reset, the renewal, the in-flight draw.
        for used, hours, k in self.CASES:
            with self.subTest(used=used, hours=hours, k=k):
                pool, merged = pool_state([(used, hours)] * k), merged_state(used, hours, k)
                self.assertEqual({f: pool[f] for f in GATE_FIELDS}, {f: merged[f] for f in GATE_FIELDS})
                horizon = lambda s: min(5.0, s['reset_hours'])
                self.assertEqual(quota_pools.hours_before(pool, horizon), quota_pools.hours_before(merged, horizon))
                renewed_pool, renewed_merged = quota_pools.after_reset(pool, 168.0), quota_pools.after_reset(merged, 168.0)
                self.assertEqual({f: renewed_pool[f] for f in ('remaining', 'deferred_hours', 'full')},
                                 {f: renewed_merged[f] for f in ('remaining', 'deferred_hours', 'full')})
                drawn = quota_pools._even_draw(pool['members'], min(30.0, pool['remaining']), CAP)
                self.assertAlmostEqual(sum(m['remaining'] for m in drawn), merged['remaining'] - min(30.0, merged['remaining']))


class OrderTests(unittest.TestCase):
    def test_member_order_changes_nothing(self):
        # Invariance to input order (AGENTS.md routing-test rules): the pool's state and term do not depend on which
        # member the gateway lists first.
        members = [(0.0, 146.0), (31.0, 30.0), (64.0, 90.0)]
        base_state = pool_state(members)
        base = term(base_state)
        for order in itertools.permutations(members):
            with self.subTest(order=order):
                state = pool_state(list(order))
                self.assertEqual({f: state[f] for f in GATE_FIELDS}, {f: base_state[f] for f in GATE_FIELDS})
                for a, b in zip(term(state), base):
                    self.assertAlmostEqual(a, b, places=12)
                self.assertEqual(quota_pools.after_reset(state, 168.0)['deferred_hours'],
                                 quota_pools.after_reset(base_state, 168.0)['deferred_hours'])

    def test_end_to_end_member_order_changes_nothing(self):
        for members in ([(0.0, 146.0), (31.0, 30.0)], [(100.0, 146.0), (100.0, 30.0)]):
            with self.subTest(members=members):
                forward = rows_of(plan_with(raw_pool(members)), 'claude-2')
                self.assertTrue(any(not r[1] for r in forward))
                self.assertEqual(rows_of(plan_with(raw_pool(members[::-1])), 'claude-2'), forward)


class MonotoneTests(unittest.TestCase):
    def test_cost_is_non_decreasing_in_each_members_used_share(self):
        # Quota cost non-decreasing in used share (AGENTS.md), member by member: raising one member's use, the others
        # fixed, never lowers the pool's term.
        other = (40.0, 60.0)
        for hours in (30.0, 146.0):
            with self.subTest(hours=hours):
                costs = [term(pool_state([(used, hours), other]))[0] for used in range(0, 100, 5)]
                for low, high in zip(costs, costs[1:]):
                    self.assertGreaterEqual(high, low - 1e-12)

    def test_unused_quota_nearer_its_reset_makes_the_term_no_larger(self):
        # Derived from quota_economy: at the same use, a later point of the period (larger elapsed fraction e) gives a
        # need tail that is no larger (five_axis_need_tail falls toward the reset), the price is a positive combination
        # of tails and the cost its integral over the same spend; the other member's term is unchanged. So of two pools
        # that differ only in one member's reset (the task ends before it), the nearer one's term is not larger.
        other = (55.0, 80.0)
        for used in (0.0, 20.0, 60.0):
            with self.subTest(used=used):
                costs = [term(pool_state([(used, hours), other]))[0] for hours in (160.0, 120.0, 80.0, 40.0, 10.0, 3.0)]
                for farther, nearer in zip(costs, costs[1:]):
                    self.assertLessEqual(nearer, farther + 1e-12)


class UnknownAndEmptyMemberTests(unittest.TestCase):
    def test_an_unread_member_or_one_with_no_reset_adds_no_capacity_and_takes_no_share(self):
        # R5: a member whose window is not known (no reading, no reset: as one account's such window is unknown) is
        # out: the pool is its known members.
        known = (31.0, 30.0)
        for unknown in ((None, 100.0), (50.0, None)):
            with self.subTest(unknown=unknown):
                pool, alone = pool_state([known, unknown]), single_state(*known)
                self.assertEqual({f: pool[f] for f in GATE_FIELDS}, {f: alone[f] for f in GATE_FIELDS})
                for a, b in zip(term(pool), term(alone)):
                    self.assertAlmostEqual(a, b, places=12)
        self.assertFalse(pool_state([(None, 100.0), (50.0, None)])['known'])

    def test_a_member_at_zero_takes_no_share(self):
        # R2: a member with nothing left serves none of the work; the others carry all of it.
        pool = pool_state([(100.0, 30.0), (40.0, 90.0)])
        self.assertEqual([m['position'] for m in quota_pools.serving_members(pool)], [2])
        _, rec, share = economy().member_records(W, pool, CAP)[0]
        self.assertEqual(share, 1.0)
        self.assertAlmostEqual(term(pool)[0], term(single_state(40.0, 90.0))[0], places=12)
        drawn = quota_pools._even_draw(pool['members'], 20.0, CAP)
        self.assertEqual([m['remaining'] for m in drawn], [0.0, 40.0])

    def test_the_pool_is_exhausted_only_when_every_member_is(self):
        # R4: the reading-level exhaustion and launch-floor gates read the members' sum.
        row = {'status': 'fresh', 'fetched_at': NOW.isoformat()}
        for members, empty in (([(100.0, 30.0), (40.0, 90.0)], False), ([(100.0, 30.0), (100.0, 90.0)], True)):
            with self.subTest(members=members):
                observed = usage_reading.window_state(raw_pool(members), NOW)
                self.assertEqual(quota_pools.used_up_in_cycle(observed, NOW), empty)
                self.assertEqual(quota_pools.below_launch_floor(SPEC, observed, row, NOW), empty)
                self.assertEqual(pool_state(members)['remaining'] <= 0, empty)


class WaitTests(unittest.TestCase):
    def test_the_wait_is_the_earliest_member_reset_that_restores_capacity_for_the_work(self):
        # R4: an exhausted pool renews member by member; its wait is the soonest reset that gives quota back, and with
        # the caller's test of the work, the first that holds it.
        pool = pool_state([(100.0, 146.0), (100.0, 30.0), (100.0, 70.0)])
        self.assertEqual(quota_pools.after_reset(pool, 168.0)['deferred_hours'], 30.0)
        two = quota_pools.after_reset(pool, 168.0, holds=lambda r: r['remaining'] >= 150.0)
        self.assertEqual((two['deferred_hours'], two['remaining']), (70.0, 200.0))
        # A member that spent nothing gives nothing back when it resets: it is no refill.
        spare = pool_state([(100.0, 146.0), (0.0, 30.0)])
        self.assertEqual(quota_pools.after_reset(spare, 168.0, holds=lambda r: r['remaining'] > 100.0)['deferred_hours'],
                         146.0)

    def test_end_to_end_the_exhausted_pool_waits_for_its_earliest_member_reset(self):
        d = plan_with(raw_pool([(100.0, 146.0), (100.0, 30.0)]))
        waits = {c.get('available_in_hours') for c in d['ranking']
                 if c['bundle']['account'] == 'claude-2' and not c.get('excluded')}
        self.assertEqual(waits, {30.0})


class ShownTests(unittest.TestCase):
    def test_the_capacity_rows_name_the_member_windows_by_position_and_carry_no_label(self):
        # Rule 6 (brief): the candidate's economy rows name the member windows; owner 2026-11-24: by position only, so
        # the decision snapshot written to the ledger carries no label (the usage reading keeps the masked label).
        d = plan_with(raw_pool([(0.0, 146.0), (31.0, 30.0)]))
        row = next(c for c in d['ranking'] if c['bundle']['account'] == 'claude-2' and not c.get('excluded'))
        members = row['economy']['windows'][0]['members']
        self.assertEqual([m['position'] for m in members], [1, 2])
        self.assertNotIn('member-', json.dumps(ledger_snapshot.decision_snapshot(d), default=str))
        self.assertAlmostEqual(sum(m['cost'] for m in members), row['economy']['windows'][0]['cost'], places=5)


if __name__ == '__main__':
    unittest.main()
