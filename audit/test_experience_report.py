"""How the experience run's report counts answers and reads chains (audit/experience_report.py). No routing outcome is
asserted: the rows here are made up, and only the report's own arithmetic is checked."""
import json

import experience_report as report


def answer(chain, step=0, kind="roll", status="launched", **extra):
    return {"chain": chain, "step": step, "kind": kind, "status": status, "caller_action": extra.pop("caller_action", None),
            "at": extra.pop("at", 0.0), "brief": extra.pop("brief", "b"), "difficulty": extra.pop("difficulty", "moderate"), **extra}


def test_answer_counts_take_only_what_the_session_read():
    answers = [answer("0"), answer("1", status="blocked", caller_action="wait_for_reset"), answer("1", 1, kind="next_argv", status="pending"),
               answer("1", 2, kind="observe", status="failed", worker_state="failed")]
    counts = report.answer_counts(answers)
    assert counts["status"] == {"launched": 1, "blocked": 1, "pending": 1}
    assert counts["status_action"]["blocked / wait_for_reset"] == 1
    assert counts["observations"] == {"observe: failed": 1}


def test_launches_are_the_last_answer_of_each_launched_chain():
    answers = [answer("0", status="blocked"), answer("0", 1), answer("1", status="error")]
    chains = [{"id": "0", "answers": [0, 1], "outcome": "launched"}, {"id": "1", "answers": [2], "outcome": "handed_to_session"}]
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


def test_questions_compare_the_chosen_candidate_with_the_other_eligible_ones():
    ranking = [{"id": "a", "fit": 1.0, "usd": 4.0}, {"id": "b", "fit": 2.0, "usd": 1.0}]
    hard = answer("0", difficulty="hard", bundle_id="a", ranking=ranking)
    assert report.harder_on_weaker(hard)["stronger"]["id"] == "b"
    assert report.harder_on_weaker(dict(hard, bundle_id="b")) is None
    assert report.harder_on_weaker(dict(hard, difficulty="moderate")) is None
    trivial = answer("0", difficulty="trivial", bundle_id="a", ranking=ranking)
    assert report.trivial_on_dearer(trivial)["ratio"] == 4.0
    assert report.trivial_on_dearer(dict(trivial, bundle_id="b")) is None


def test_run_outs_are_charges_that_did_not_fit():
    scenario = {"charges": [{"chain": "0", "pct": 10.0, "applied": 10.0}, {"chain": "1", "pct": 30.0, "applied": 5.0},
                            {"chain": "2", "pct": None, "applied": None}]}
    assert [c["chain"] for c in report.run_outs(scenario)] == ["1"]


def test_write_builds_report_and_summary_from_rows(tmp_path):
    rows = {"run": {"started": "t", "seconds": 1, "processes_at_end": [], "processes_after_end": []},
            "scenarios": [{"name": "s", "spec": {"accounts": [{"name": "codex", "family": "codex", "plan": "plus"}]}, "error": None, "seconds": 1,
                           "processes_ended": [], "answers": [answer("0", model="m", effort="high", account="codex", bundle_id="x", expected_usd=1.5,
                                                                     seconds=2.0, stdout_chars=100)],
                           "chains": [{"id": "0", "briefs": ["b"], "answers": [0], "outcome": "launched", "commands": 1}],
                           "charges": [], "resets": [{"account": "codex", "window": "usage", "at": 168, "used_before": 10, "unused": 90}],
                           "timeline": [{"at": 0, "windows": {"codex": {"usage": {"used": 0, "resets_in": 168}}}}],
                           "exclusions": {}, "bundles": {"x": {"account": "codex", "model": "m"}, "y": {"account": "codex", "model": "n"}}}]}
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    path = report.write(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    scenario = summary["scenarios"][0]
    assert scenario["workers"] == 1 and scenario["unused_at_reset"] == 90
    assert [m["model"] for m in scenario["models_with_nothing"]] == ["n@codex"]
    assert scenario["usd_by_difficulty"]["moderate"]["sum"] == 1.5
    assert "## s" in path.read_text(encoding="utf-8")


def test_a_reset_after_the_days_last_act_is_overnight():
    scenario = {"timeline": [{"at": 1.0}, {"at": 9.5}, {"at": 25.0}],
                "resets": [{"at": 4.0, "unused": 10}, {"at": 13.0, "unused": 100}, {"at": 26.0, "unused": 50}]}
    assert [r["overnight"] for r in report.overnight(scenario)] == [False, True, True]
