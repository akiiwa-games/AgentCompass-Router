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


LADDERS = {}


def substitute_ladder(substitutes):
    """The fixture ladder with the other members as outside alternatives at its own best fit: what
    FiveAxisEconomy.ladder adds, since five_axis_outside_value of k alternatives at the best fit is that fit times
    the chance one has room (fleet_fixture.ladder_for's `with_outside`)."""
    none_open = 1.0
    for a in substitutes:
        none_open *= 1.0 - a
    key = round(1.0 - none_open, 12)
    if key not in LADDERS:
        LADDERS[key] = fleet_fixture.ladder_for(with_outside=key) if key else LADDER
    return LADDERS[key]


def economy(spend_max=60.0, lump=30.0, substitutes=True):
    """The economy's window pricing with the ladder and the burst size held fixed (they come from the account's own
    candidates, which these tests do not vary). `substitutes=False`: the by-room split alone, each member priced on the
    account's ladder without the other members as its substitutes (the rule as committed before review round 3)."""
    econ = object.__new__(quota_economy.FiveAxisEconomy)
    econ._tables = {}
    econ.spend_max = {W: spend_max}
    econ.lump = lambda wkey, reset_hours=None: lump
    econ.ladder = ((lambda wkey, subs=(): substitute_ladder(subs)) if substitutes
                   else (lambda wkey, subs=(): LADDER))
    return econ


def term(state, spend=40.0, hours=2.0, econ=None):
    """(cost, price, need) of `spend` on a window state, as apply_five_axis_economy's eval_windows composes them: each
    serving window's own cost at its share on its ladder with the other members as substitutes, summed; price and need
    weighted by the shares (R3)."""
    econ = econ or economy()
    recs = econ.member_records(W, state, CAP, spend)
    gots = [econ.window_cost(W, rec, spend * share, hours, share=share, substitutes=subs)
            for (_, rec, share), subs in zip(recs, econ.substitutes(recs, spend))]
    return (sum(g['cost'] for g in gots), sum(g['lambda_start'] * r[2] for g, r in zip(gots, recs)),
            sum(g['need_start'] * r[2] for g, r in zip(gots, recs)))


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
    NO_SLACK = ((99.0, 40.0, 2), (99.5, 10.0, 3), (100.0, 20.0, 3))     # no member has room beyond its share of $40
    SLACK = ((20.0, 100.0, 2), (31.0, 30.0, 2), (0.0, 150.0, 3), (60.0, 60.0, 2), (97.0, 40.0, 2))

    def slacks(self, state, spend=40.0):
        recs = economy().member_records(W, state, CAP, spend)
        return [a for subs in economy().substitutes(recs, spend) for a in subs]

    def test_members_in_step_with_no_slack_price_as_the_merged_window(self):
        # R3/R4 and review round 3 (2): with no room in any other member (substitute slack 0) a member's ladder is the
        # account's own, and members in step give the merged window's term: capacity, use and burst scale by the share.
        for used, hours, k in self.NO_SLACK:
            with self.subTest(used=used, hours=hours, k=k):
                state = pool_state([(used, hours)] * k)
                self.assertEqual(set(self.slacks(state)), {0.0})
                for a, b in zip(term(state), term(merged_state(used, hours, k), econ=economy())):
                    self.assertAlmostEqual(a, b, places=9)

    def test_members_in_step_with_slack_price_at_most_the_merged_window_and_not_below_zero(self):
        # Review round 3 (3): another member with room is an outside alternative of each member's dollars at the
        # account's own best fit (FiveAxisEconomy.substitutes, ladder), which raises the outside value and lowers the
        # ladder; the merged window had no such alternative. So the pool's cost and price are in [0, merged].
        for used, hours, k in self.SLACK:
            with self.subTest(used=used, hours=hours, k=k):
                state = pool_state([(used, hours)] * k)
                self.assertTrue(any(a > 0 for a in self.slacks(state)))      # not the no-slack case
                cost, price, _ = term(state)
                merged_cost, merged_price, _ = term(merged_state(used, hours, k), econ=economy())
                self.assertGreaterEqual(cost, 0.0)
                self.assertGreaterEqual(price, 0.0)
                self.assertLessEqual(cost, merged_cost + 1e-12)
                self.assertLessEqual(price, merged_price + 1e-12)

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
                drawn = quota_pools._draw(pool, min(30.0, pool['remaining']), CAP)
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
    SPEND = 40.0        # $40 of a $1000 window: an equal share is 2% of one account
    USED = [*range(0, 98, 2), 98, 98.5, 99, 99.5, 99.9, 99.99, 100]
    OTHERS = ((0.0, 146.0), (40.0, 60.0), (90.0, 30.0), (99.0, 100.0), (100.0, 146.0))

    def test_cost_is_non_decreasing_in_each_members_used_share_while_it_has_room_for_its_equal_share(self):
        # Quota cost non-decreasing in used share (AGENTS.md), member by member, the other fixed, for several states of
        # the other member and several resets (review round 3 (4)), for as long as the member has room for its equal
        # share of the work (quota_pools.shares: by room). Past that point it carries only its last dollars, and each
        # member's need is read from its own window, not from the pool's demand the split moves onto the others, so
        # the cost may fall (router-internals.md, Account pools: the limit the owner left open, report Round 3).
        equal_share_pct = 100.0 * self.SPEND / CAP / 2
        used = [u for u in self.USED if u <= 100.0 - equal_share_pct]
        for other in self.OTHERS:
            for hours in (10.0, 30.0, 146.0):
                with self.subTest(other=other, hours=hours):
                    costs = [term(pool_state([(u, hours), other]), self.SPEND)[0] for u in used]
                    for low, high in zip(costs, costs[1:]):
                        self.assertGreaterEqual(high, low - 1e-12)

    def test_substitutes_never_raise_the_cost_of_the_by_room_split(self):
        # Review round 3 (6): another member's room only adds an outside alternative, so a member's scarcity is never
        # higher with it: the pool's cost is at most what the by-room split alone gives for the same reading.
        by_room = economy(substitutes=False)
        for other in self.OTHERS:
            for used in (0.0, 50.0, 90.0, 98.0, 99.5, 100.0):
                with self.subTest(other=other, used=used):
                    state = pool_state([(used, 30.0), other])
                    self.assertLessEqual(term(state, self.SPEND)[0], term(state, self.SPEND, econ=by_room)[0] + 1e-12)

    def test_a_member_at_100_percent_is_the_limit_of_one_near_it_and_takes_no_share(self):
        # (b) no jump at 100%: as a member's room shrinks to 0 it carries only that room, so the pool's term tends to
        # its term at 100% (the gap shrinks with the room: at most the top price times the room); (c) a member at 100%
        # takes no share and the pool is the others (shares: none above a member's room).
        for other in ((40.0, 60.0), (0.0, 146.0)):
            with self.subTest(other=other):
                full = term(pool_state([(100.0, 30.0), other]), self.SPEND)[0]
                gaps = [abs(term(pool_state([(100.0 - room, 30.0), other]), self.SPEND)[0] - full)
                        for room in (1e-3, 1e-5, 1e-7)]
                for wider, narrower in zip(gaps, gaps[1:]):
                    self.assertLess(narrower, wider / 10)
                self.assertLess(gaps[-1], 0.01 * full)
                self.assertEqual([m['position'] for m, _ in quota_pools.shares(pool_state([(100.0, 30.0), other]), 4.0)],
                                 [2])

    def test_one_split_by_room_for_price_and_reservations(self):
        # Review round 2 (owner): one split rule, by room, read by the price and the in-flight draw alike. A member gets
        # an equal share, never above its room; what it cannot hold goes to the others.
        pool = pool_state([(99.0, 30.0), (40.0, 60.0), (70.0, 90.0)])
        split = dict((m['position'], share) for m, share in quota_pools.shares(pool, 12.0))
        self.assertAlmostEqual(split[1] * 12.0, 1.0)                    # its room, 1%
        self.assertAlmostEqual(split[2], split[3])                      # the rest in equal parts
        self.assertAlmostEqual(sum(split.values()), 1.0)
        drawn = {m['position']: m['observed_remaining'] - m['remaining']
                 for m in quota_pools._draw(pool, 12.0, CAP) if 'observed_remaining' in m}
        self.assertEqual({k: round(v, 9) for k, v in drawn.items()}, {k: round(s * 12.0, 9) for k, s in split.items()})
        recs = economy().member_records(W, pool, CAP, 120.0)          # $120 = 12% of one account
        self.assertEqual({m['position']: round(share, 9) for m, _, share in recs}, {k: round(s, 9) for k, s in split.items()})

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
        self.assertEqual([m['position'] for m, _ in quota_pools.shares(pool, 4.0)], [2])
        _, rec, share = economy().member_records(W, pool, CAP, 40.0)[0]
        self.assertEqual(share, 1.0)
        self.assertAlmostEqual(term(pool)[0], term(single_state(40.0, 90.0))[0], places=12)
        drawn = quota_pools._draw(pool, 20.0, CAP)
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


class IdleMemberTests(unittest.TestCase):
    """R5 (brief): a member whose window reports no reset yet (an idle account) is treated exactly as one account's
    window with no reset is today: it counts with what it shows left, in every gate."""
    FIVE = {'id': 'five_hour', 'hours': 5.0, 'launch_min_remaining_pct': 25}

    def test_a_spent_member_beside_an_idle_one_passes_the_launch_gate_as_the_idle_account_does(self):
        row = {'status': 'fresh', 'fetched_at': NOW.isoformat()}
        idle = usage_reading.window_state({'id': 'five_hour', 'used_pct': 0.0, 'resets_at': None}, NOW)
        pool = usage_reading.window_state(raw_pool([(100.0, 2.0), (0.0, None)], 'five_hour'), NOW)
        self.assertEqual(quota_pools.below_launch_floor(self.FIVE, pool, row, NOW),
                         quota_pools.below_launch_floor(self.FIVE, idle, row, NOW))
        self.assertEqual(pool['remaining_pct_before_reset'], idle['observed_remaining_pct'])
        both = usage_reading.window_state(raw_pool([(0.0, None), (0.0, None)], 'five_hour'), NOW)
        self.assertEqual(both['remaining_pct_before_reset'], 2 * idle['observed_remaining_pct'])
        state = pool_state([(100.0, 2.0), (0.0, None)], self.FIVE)
        self.assertEqual(state['remaining'], quota_pools.window_state('five_hour', self.FIVE, idle, {}, NOW, 24)['remaining'])

    def test_end_to_end_a_pool_with_an_exhausted_and_an_idle_member_starts_as_the_idle_account(self):
        week = {'id': 'seven_day', 'used_pct': 20.0, 'resets_at': at(100.0)}
        def waits(five):
            rows = rows_of(plan_with(week, five), 'claude-2')
            self.assertTrue(rows)
            return {(r[2], any('five_hour' in e for e in r[1] or ())) for r in rows}
        idle = waits({'id': 'five_hour', 'used_pct': 0.0, 'resets_at': None})
        self.assertEqual(waits(raw_pool([(100.0, 2.0), (0.0, None)], 'five_hour')), idle)


class OutOfStepTests(unittest.TestCase):
    """Review N1: the rules that members in step cannot tell apart (first = mean = soonest = latest)."""

    def test_the_pools_reset_is_the_soonest_reset_of_a_member_that_spent_some(self):
        # R4: the pool's reset is its next partial refill, not the latest reset nor the first listed member's.
        for members, soonest in (([(0.0, 146.0), (31.0, 30.0), (10.0, 90.0)], 30.0),
                                 ([(50.0, 146.0), (0.0, 30.0), (10.0, 90.0)], 90.0),
                                 ([(0.0, 146.0), (0.0, 30.0)], 30.0)):
            with self.subTest(members=members):
                self.assertEqual(pool_state(members)['reset_hours'], soonest)

    def test_the_hours_before_reset_come_from_every_member_by_its_share(self):
        # R4/shares: each member's share of the work runs to its own reset, so the pool's hours before reset are the
        # members' horizons weighted by their shares (equal shares while every member has room).
        pool = pool_state([(0.0, 146.0), (31.0, 3.0), (10.0, 1.0)])
        horizon = lambda s: min(5.0, s['reset_hours'])
        split = quota_pools.shares(pool, 6.0)
        self.assertEqual(len(split), 3)
        self.assertAlmostEqual(quota_pools.hours_before(pool, horizon, 6.0),
                               sum(share * horizon(m) for m, share in split))
        self.assertAlmostEqual(quota_pools.hours_before(pool, horizon, 6.0), (5.0 + 3.0 + 1.0) / 3)


class PassedMemberTests(unittest.TestCase):
    def test_a_member_whose_reset_passed_since_the_reading_is_left_out_and_said(self):
        # R5 and review N2: a member whose reset has passed since its reading is not read until it is read again (as
        # one account's such window); the pool is the others, and the candidate's notes say which member waits.
        raw = raw_pool([(0.0, 146.0), (31.0, -1.0)])
        row = {'status': 'fresh', 'fetched_at': NOW.isoformat()}
        observed = quota_pools.cycle_observation(SPEC, usage_reading.window_state(raw, NOW), row, NOW)
        self.assertEqual(observed['passed_members'], {'positions': [2], 'of': 2})
        alone = single_state(0.0, 146.0)
        state = quota_pools.window_state('seven_day', SPEC, observed, {}, NOW, 24)
        self.assertEqual({f: state[f] for f in GATE_FIELDS}, {f: alone[f] for f in GATE_FIELDS})
        d = plan_with(raw)
        notes = [n for c in d['ranking'] if c['bundle']['account'] == 'claude-2' for n in c.get('notes') or ()]
        self.assertTrue(any('pool member 2 of 2 is left out' in n for n in notes))
        self.assertFalse(any('left out' in n for c in plan_with(raw_pool([(0.0, 146.0), (31.0, 30.0)]))['ranking']
                             if c['bundle']['account'] == 'claude-2' for n in c.get('notes') or ()))


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
