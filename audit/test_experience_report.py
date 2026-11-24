"""How the experience run's report counts answers and reads chains (audit/experience_report.py). No routing outcome is
asserted: the rows here are made up, and only the report's own arithmetic is checked."""
import json

import experience_report as report


def answer(chain, step=0, kind="roll", status="launched", **extra):
    return {"chain": chain, "step": step, "kind": kind, "status": status, "caller_action": extra.pop("caller_action", None),
            "at": extra.pop("at", 0.0), "brief": extra.pop("brief", "b"), "level": extra.pop("level", 2), **extra}


def test_answer_counts_take_only_what_the_session_read():
    answers = [answer("0"), answer("1", status="blocked", caller_action="wait_for_reset"), answer("1", 1, kind="next_argv", status="pending"),
               answer("1", 2, kind="observe", status="failed", worker_state="failed")]
    counts = report.answer_counts(answers)
    assert counts["status"] == {"launched": 1, "blocked": 1, "pending": 1}
    assert counts["status_action"]["blocked / wait_for_reset"] == 1
    assert counts["observations"] == {"observe: failed": 1}


def test_launches_are_the_answers_that_carried_the_launch_not_a_later_observation():
    answers = [answer("0", status="blocked"), answer("0", 1, carried_launch=True), answer("0", 2, kind="observe", status="failed"),
               answer("1", status="error")]
    chains = [{"id": "0", "answers": [0, 1, 2], "outcome": "launched"}, {"id": "1", "answers": [3], "outcome": "handed_to_session"}]
    assert report.launches(answers, chains) == [answers[1]]


def test_dead_end_and_broken_promise_are_read_from_the_chain():
    scenario = {"answers": [answer("0", status="error", next_action=""),
                            answer("1", status="error", next_action="AgentCompass relaunches this brief by itself: do not roll it again."),
                            answer("2", status="blocked", next_action="wait 2h and roll again")]}
    chains = [{"id": "0", "answers": [0], "outcome": "handed_to_session"}, {"id": "1", "answers": [1], "outcome": "handed_to_session"},
              {"id": "2", "answers": [2], "outcome": "handed_to_session"}, {"id": "3", "answers": [], "outcome": "dead_end"}]
    assert [c["id"] for c in chains if report.is_dead_end(scenario, c)] == ["0", "3"]
    assert [c["id"] for c in chains if report.promise_broken(scenario, c)] == ["1"]
    assert not report.promise_broken(scenario, dict(chains[1], outcome="relaunched"))


def test_a_deferral_the_run_did_not_wait_out_is_not_a_dead_end_when_its_next_action_says_what_to_do():
    """A deferral longer than the run waits (experience_run.WAIT_LIMIT_HOURS) tells the session to wait, name a route or
    do the task itself: the session can act on it, so it is counted apart, with the hours it was told."""
    told = "Defer sol62_medium for 66.0h until codex's usage window resets; do not launch now. Roll again then."
    scenario = {"answers": [answer("0", status="blocked", caller_action="wait_for_reset", start_after_hours=66.0, next_action=told),
                            answer("1", status="blocked", caller_action="wait_for_reset", start_after_hours=40.0, next_action="")]}
    chains = [{"id": "0", "answers": [0], "outcome": "deferred_not_waited"}, {"id": "1", "answers": [1], "outcome": "deferred_not_waited"}]
    assert [c["id"] for c in chains if report.is_dead_end(scenario, c)] == ["1"]
    assert report.deferred_past_wait(scenario, chains[0]) == {"chain": "0", "hours": 66.0, "brief": scenario["answers"][0]["brief"]}
    assert report.deferred_past_wait(scenario, chains[1]) is None


def test_the_fit_question_reads_the_recorded_fit_of_hard_work_only():
    ranking = [{"id": "a", "fit": 1.0}, {"id": "b", "fit": 2.0}]
    hard = answer("0", level=report.limit("hard_level"), bundle_id="a", ranking=ranking)
    assert report.harder_on_weaker(hard)["stronger"]["id"] == "b"
    assert report.harder_on_weaker(dict(hard, bundle_id="b")) is None
    assert report.harder_on_weaker(dict(hard, level=2)) is None


def test_the_cost_question_compares_the_routers_opportunity_cost_never_dollars():
    """Cost is components.opportunity_cost as recorded; a dearer list price on a cheaper quota is not dearer."""
    routine = report.limit("routine_level")
    ranking = [{"id": "a", "account": "claude", "opportunity_cost": 0.004, "usd": 4.0},
               {"id": "b", "account": "codex", "opportunity_cost": 0.02, "usd": 1.0}]
    q = report.cost_check(answer("0", level=routine, bundle_id="a", ranking=ranking))
    assert q["recorded"] and not q["dearer"] and q["cheaper"] is None
    q = report.cost_check(answer("0", level=routine, bundle_id="b", ranking=ranking))
    assert q["dearer"] and q["cheaper"]["id"] == "a" and q["ratio"] == 5.0
    near = [dict(ranking[0], opportunity_cost=0.015), ranking[1]]
    q = report.cost_check(answer("0", level=routine, bundle_id="b", ranking=near))
    assert q["cheaper"]["id"] == "a" and not q["dearer"]                                     # under the declared ratio
    free = [dict(ranking[0], opportunity_cost=0.0), ranking[1]]
    assert report.cost_check(answer("0", level=routine, bundle_id="b", ranking=free))["dearer"]   # above a zero cost
    unrecorded = [{"id": "b", "account": "codex", "fit": 1.0}, ranking[0]]
    assert report.cost_check(answer("0", level=routine, bundle_id="b", ranking=unrecorded)) == {"recorded": False}
    assert report.cost_check(answer("0", level=2, bundle_id="b", ranking=ranking)) is None


def test_models_with_nothing_count_a_model_once_keep_the_users_switches_apart_and_say_only_what_was_recorded():
    scenario = {"spec": {"switches": {"outside_allowed": ["codex-2"]}},
                "bundles": {"sonnet_a": {"account": "claude", "provider": "claude", "model": "sonnet"},
                            "sonnet_b": {"account": "claude-2", "provider": "claude-2", "model": "sonnet"},
                            "astra": {"account": "codex", "provider": "codex", "model": "astra"},
                            "pi_astra": {"account": "codex", "provider": "pi", "model": "astra"},
                            "fable": {"account": "claude", "provider": "claude", "model": "fable"},
                            "sol": {"account": "codex-2", "provider": "codex-2", "model": "sol"},
                            "opus": {"account": "claude", "provider": "claude", "model": "opus", "disabled": True}},
                "exclusions": {"astra": {"plans": 4, "eligible": 4, "won": 0, "gaps": [0.1, 0.3], "excluded": {}},
                               "pi_astra": {"plans": 4, "eligible": 0, "won": 0, "gaps": [], "excluded": {"provider discovery: missing_client": 4}}}}
    nothing = report.models_with_nothing(scenario, [answer("0", model="sonnet")], ["codex", "claude", "opencode"])
    assert [e["model"] for e in nothing["models"]] == ["astra", "fable"]
    assert nothing["outside_allowed"] == ["sol"] and nothing["unchecked"] == ["opus"]
    astra = {r["bundle"]: r for r in nothing["models"][0]["bundles"]}
    assert astra["astra"]["recorded"] == "eligible in 4 plan(s), never the top score"   # pi's missing client is not codex's
    assert astra["pi_astra"]["recorded"] == "provider discovery: missing_client ×4"
    # Said where the reader meets it: the fixture installs no pi executable.
    assert "fixture installs no pi executable" in astra["pi_astra"]["note"] and astra["astra"]["note"] is None
    assert nothing["models"][1]["bundles"][0]["recorded"] == "not in any plan's recorded ranking"


def test_a_bundle_in_no_recorded_ranking_says_whether_the_ledgers_cap_may_have_cut_it():
    """ledger_snapshot keeps at most SNAPSHOT_EXCLUDED excluded candidates per plan: a plan at that cap may have cut one."""
    eligible_cap, excluded_cap = report.snapshot_caps()
    at_cap = {"answers": [answer("0", snapshot_rows={"eligible": 2, "excluded": excluded_cap}),
                          answer("1", snapshot_rows={"eligible": 2, "excluded": 3})]}
    cut = report.ranking_cut(at_cap)
    assert (cut["plans"], cut["excluded_at_cap"], cut["eligible_at_cap"], cut["bound"]) == (2, 1, 0, "exact")
    assert report.not_in_ranking(cut)[0] == "not in any plan's recorded ranking"
    assert f"1 of 2 plan(s) kept the ledger's cap of {excluded_cap}" in report.not_in_ranking(cut)[1]
    under = report.ranking_cut({"answers": [answer("0", snapshot_rows={"eligible": 2, "excluded": 3})]})
    assert report.not_in_ranking(under)[0] == "absent from every plan's recorded ranking"
    # Rows written before snapshot_rows: derived from the per-bundle tallies (excluded rows in all against cap x plans).
    old = {"answers": [answer(str(n), ledger_stage="plan", ranking=[]) for n in range(3)],
           "exclusions": {f"b{n}": {"plans": 3, "eligible": 0} for n in range(excluded_cap)}}
    cut = report.ranking_cut(old)
    assert (cut["excluded_at_cap"], cut["bound"]) == (3, "exact") and "derived from these rows" in cut["source"]
    fewer = dict(old, exclusions={"b0": {"plans": 3, "eligible": 0}})
    assert report.ranking_cut(fewer)["excluded_at_cap"] == 0


def test_fixture_errors_and_estimates_are_said_where_they_appear():
    assert report.fixture_notes("provider failure hold on this account ×2", "codex", ["codex"]).startswith("the hold follows a worker error")
    assert report.fixture_notes("provider discovery: missing_client ×1", "codex", ["codex"]) is None
    assert "unknown ×2" in report.fixture_errors({"scopes_answered": {"unknown": 2}})
    assert report.fixture_error_texts()          # briefs.json declares the texts its classifier answers


def test_one_headline_per_event_and_models_counted_once_per_scenario():
    def scenario(name, share):
        return {"name": name, "broken_promises": ["1"], "dead_ends": [], "error": None, "resets": [], "long_answers": [],
                "roll_seconds": {}, "questions": {"hard_on_weaker": [], "routine_on_dearer": []}, "routine_cost": {},
                "not_launched": [{"scenario": name, "chain": "1", "step": 0, "status": "error", "caller_action": "check_error", "brief": "b"},
                                 {"scenario": name, "chain": "2", "step": 0, "status": "blocked", "caller_action": "tell_user", "brief": "b"}],
                "models_with_nothing": {"models": [{"model": "astra", "bundles": [{"provider": "codex", "recorded": "x"}]},
                                                   {"model": "astra-2", "bundles": [{"provider": "codex", "recorded": "x"}]}],
                                        "outside_allowed": [], "unchecked": []},
                "level_x_model": share}
    folded = {"L1": {"sonnet:medium@claude": 3, "sol:high@codex": 1}, "L2": {"sonnet:high@claude": 3}, "L3": {"sonnet:high@claude-2": 2}}
    first, second = scenario("one", folded), scenario("two", {})
    for one in (first, second):
        one["models_with_nothing"]["outside_allowed"] = ["sol"]     # the same model in two scenarios is one model
    items = report.look_first({}, {"scenarios": [first, second]})
    titles = [i["title"] for i in items]
    assert not any("error / check_error" in t for t in titles)          # the broken promise's answer is counted once
    assert any("blocked / tell_user" in t for t in titles)
    nothing = next(i for i in items if "got no work" in i["title"])
    assert nothing["title"].startswith("2 signed-in model(s)") and {r["scenarios"] for r in nothing["rows"]} == {2}
    assert "not counted: 1 model(s) only on accounts outside allowed_accounts" in nothing["title"]
    concentrated = next(i for i in items if "one model" in i["title"])
    assert concentrated["rows"] == [{"scenario": "one", "model": "sonnet", "share": 0.89, "workers": 9, "models_that_worked": 2}]


def test_write_builds_report_and_summary_from_rows(tmp_path):
    rows = {"run": {"started": "t", "seconds": 1, "processes_at_end": [], "processes_after_end": []},
            "scenarios": [{"name": "s", "spec": {"accounts": [{"name": "codex", "family": "codex", "plan": "plus"}]}, "error": None, "seconds": 1,
                           "processes_ended": [], "answers": [answer("0", model="m", effort="high", account="codex", bundle_id="x", expected_usd=1.5,
                                                                     seconds=2.0, stdout_chars=100, carried_launch=True)],
                           "chains": [{"id": "0", "briefs": ["b"], "answers": [0], "outcome": "launched", "commands": 1},
                                      {"id": "1", "briefs": ["b"], "answers": [], "outcome": "batch", "commands": 1},
                                      {"id": "1.0", "briefs": ["b"], "answers": [], "outcome": "launched", "commands": 0, "batch_of": "1"}],
                           "charges": [], "resets": [{"account": "codex", "window": "usage", "at": 168, "used_before": 10, "unused": 90}],
                           "timeline": [{"at": 0, "windows": {"codex": {"usage": {"used": 0, "resets_in": 168}}}}],
                           "exclusions": {}, "bundles": {"x": {"account": "codex", "provider": "codex", "model": "m"},
                                                         "y": {"account": "codex", "provider": "codex", "model": "n"}}}]}
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    path = report.write(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    scenario = summary["scenarios"][0]
    assert scenario["workers"] == 1 and scenario["unused_at_reset"] == 90
    assert [m["model"] for m in scenario["models_with_nothing"]["models"]] == ["n"]
    assert scenario["usd_by_level"]["L2"]["sum"] == 1.5
    assert scenario["batch_workers"] == {"workers": 1, "batch_rolls": 1, "follow_up_commands": 0}
    text = path.read_text(encoding="utf-8")
    assert "## s" in text
    for name in report.THRESHOLDS:
        assert name in text                                              # every threshold is declared in the report


def test_a_reset_after_the_days_last_act_is_overnight():
    scenario = {"timeline": [{"at": 1.0}, {"at": 9.5}, {"at": 25.0}],
                "resets": [{"at": 4.0, "unused": 10}, {"at": 13.0, "unused": 100}, {"at": 26.0, "unused": 50}]}
    assert [r["overnight"] for r in report.overnight(scenario)] == [False, True, True]
