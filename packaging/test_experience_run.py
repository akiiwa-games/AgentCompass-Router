"""The experience run's own plumbing (packaging/experience_run.py): the classify step against a fixture product command, the
fixture classifier's answer to a scope question, and where a run directory may be. Nothing here runs the router."""
import json
import sys
import textwrap
import urllib.request
from pathlib import Path

import experience_run as xp

FAKE_PRODUCT = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    args = sys.argv[1:]
    log = Path(__file__).with_name("calls.jsonl")
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(args) + "\\n")
    if args[:3] == ["scorer", "--action", "show"]:
        print(json.dumps({"stage": "scorer", "scorer_type": "openai_compatible", "label": "OpenAI-compatible",
                          "openai_compatible": {"api_key": "fixture-key-canary", "base_url": "http://127.0.0.1:9/v1", "model": "fixture-model"}}))
        raise SystemExit(0)
    assert args[0] == "score" and args[1] == "--input"
    objective = json.loads(Path(args[2]).read_text(encoding="utf-8"))["task"]["objective"]
    if "UNAVAILABLE" in objective:
        print(json.dumps({"stage": "score", "status": "blocked", "source": "scorer_unavailable", "task_profile": None, "error": "scorer_unavailable"}))
        raise SystemExit(2)
    profile = {"work_kind": "code_implementation", "primary_level": 1, "secondary_kind": None, "secondary_level": 0, "access": "write",
               "local_only": False, "needs_visual": False, "wants_fast": False, "expected_hours": 0.5, "task_count": "1", "multi_task": False}
    if "TWO JOBS" in objective:
        profile.update(multi_task=True, task_count="2")
    print(json.dumps({"stage": "score", "source": "system_one", "task_profile": profile}))
''')


def product(tmp_path):
    script = tmp_path / "fake_agentcompass.py"
    script.write_text(FAKE_PRODUCT, encoding="utf-8")
    return [sys.executable, str(script)], tmp_path / "calls.jsonl"


def asked(calls: Path) -> list:
    return [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines() if json.loads(line)[0] == "score"]


CORPUS = {"one": {"text": "fix a typo"}, "split": {"text": "TWO JOBS: a and b"}, "down": {"text": "UNAVAILABLE now"}}


def test_classify_prints_its_plan_and_asks_nothing_without_the_spend_flag(tmp_path, capsys):
    command, calls = product(tmp_path)
    assert xp.classify(CORPUS, tmp_path / "out", spend=False, command=command) == 0
    shown = capsys.readouterr().out
    assert "3 of 3 briefs to ask" in shown and "model fixture-model at http://127.0.0.1:9/v1" in shown
    assert "fixture-key-canary" not in shown
    assert asked(calls) == [] and not (tmp_path / "out" / "classifications.json").exists()


def test_classify_keeps_a_split_answer_whole_writes_as_it_goes_and_resumes(tmp_path):
    command, calls = product(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert xp.classify(CORPUS, out, spend=True, command=command) == 1
    answers = json.loads((out / "classifications.json").read_text(encoding="utf-8"))
    assert sorted(answers) == ["one", "split"]
    assert (answers["split"]["multi_task"], answers["split"]["task_count"]) == (True, "2")
    failed = json.loads((out / "classifications-failed.json").read_text(encoding="utf-8"))
    assert failed["down"]["source"] == "scorer_unavailable"
    assert len(asked(calls)) == 3
    assert xp.classify(CORPUS, out, spend=True, command=command) == 1
    assert len(asked(calls)) == 4                                     # only the brief with no answer is asked again


def test_the_fixture_classifier_answers_a_scope_question_from_the_declared_table_only():
    scopes = {"fixture worker failure": {"scope": "none", "why": "its own work"}}

    def ask(scorer, user):
        body = json.dumps({"messages": [{"role": "system", "content": "scope"}, {"role": "user", "content": user}]}).encode()
        request = urllib.request.Request(scorer.base + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as error:
            return error.code

    with xp.ReplayScorer({}, None, scopes) as scorer:
        question = json.dumps({"account": "a", "provider": "p", "model": "m", "failure": "provider_turn_failed",
                               "provider_message": "fixture worker failure"})
        assert ask(scorer, question)["scope"] == "none"
        assert ask(scorer, question.replace("fixture worker failure", "something else")) == 500
        assert scorer.scopes_answered == {"none": 1}


def test_a_run_directory_beside_the_repository_is_outside_it(tmp_path):
    root = tmp_path / "repo"
    (root / "inner").mkdir(parents=True)
    assert xp.inside(root / "inner", root)
    assert not xp.inside(tmp_path / "repo-runs", root)
    assert not xp.inside(Path(str(root) + "-sibling"), root)


TREE = textwrap.dedent('''
    import subprocess, sys, time
    from pathlib import Path
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(sys.argv[1]).write_text(str(grandchild.pid))
    time.sleep(120)
''')


def test_a_command_past_its_limit_ends_with_every_process_it_started(tmp_path):
    """A hung command's grandchild carries nothing that names it; it ends with the command, by the handle the run holds."""
    import os
    import psutil
    noted = tmp_path / "grandchild.pid"
    try:
        xp.run_owned([sys.executable, "-c", TREE, str(noted)], cwd=tmp_path, env=dict(os.environ), stdin=None, timeout=5)
    except xp.CommandTimedOut as stopped:
        ended = stopped.ended
    else:
        raise AssertionError("the command answered")
    grandchild = int(noted.read_text())
    assert psutil.wait_procs([p for p in (psutil.Process(grandchild),) if p.is_running()] if psutil.pid_exists(grandchild) else [],
                             timeout=10)[1] == []
    if os.name == "nt":
        assert str(grandchild) in ended and len(ended) == 2


def test_a_command_inside_its_limit_answers_as_before(tmp_path):
    import os
    done = xp.run_owned([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], cwd=tmp_path, env=dict(os.environ),
                        stdin="answer", timeout=60)
    assert (done.returncode, done.stdout.strip()) == (0, "ANSWER")


def test_each_answer_keeps_the_blocked_candidates_of_its_plan_with_their_reasons():
    """A plan's snapshot holds eligible rows (scored) and blocked rows (reasons, no components): the eligible go to
    `ranking`, the blocked to `excluded` with model and reasons as recorded, so a reader can say why a stronger or cheaper
    candidate did not take the work."""
    snapshot = [
        {"id": "a", "bundle": {"model": "m1", "account": "x", "effort": "high"}, "score": 0.5, "excluded": [],
         "components": {"fit": 0.2, "opportunity_cost": 0.1}},
        {"id": "b", "bundle": {"model": "m2", "account": "y", "effort": "max"}, "score": None,
         "excluded": ["five_hour is empty for 3.0h", "weekly short"], "components": {}},
    ]
    assert [c["id"] for c in xp.compact_ranking(snapshot)] == ["a"]
    waiting = dict(snapshot[0], id="w", available_in_hours=2.6)
    assert [c["available_in_hours"] for c in xp.compact_ranking([snapshot[0], waiting])] == [None, 2.6]
    assert xp.compact_exclusions(snapshot) == [
        {"id": "b", "model": "m2", "account": "y", "effort": "max", "excluded": ["five_hour is empty for 3.0h", "weekly short"]}]


def test_a_waiting_candidate_keeps_the_routers_own_sentence_of_why_it_waits():
    """A candidate that waits keeps, copied as recorded, the notes in which the router says which window holds it and
    when it can start; its other notes are not taken for that sentence. A ready candidate has none (None); a waiting one
    whose snapshot kept no such note has [] ("not recorded"). Rows past the snapshot's best ones are not `detail`."""
    said = "seven_day renews in 50.4h; start then (assumes a full renewal)"
    row = {"id": "w", "bundle": {"model": "m", "account": "x", "effort": "high"}, "score": 0.5, "excluded": [],
           "components": {"fit": 0.2}, "available_in_hours": 50.4, "notes": ["time at high: 0.15h from the estimate", said]}
    ready = dict(row, id="r", available_in_hours=None)
    silent = dict(row, id="s", notes=["time at high: 0.15h from the estimate"])
    kept = xp.compact_ranking([row, ready, silent] + [dict(ready, id=f"c{i}") for i in range(xp.snapshot_full())])
    assert [(c["wait_reason"], c["detail"]) for c in kept[:3]] == [([said], True), (None, True), ([], True)]
    assert kept[-1]["detail"] is False


def test_the_router_still_writes_the_wait_sentence_the_run_looks_for():
    """WAIT_SENTENCE is candidate_eval's wording for a candidate that starts after a reset; if that wording moves, the run
    would say "not recorded" for every wait, so this reads the router's source for it."""
    source = (xp.ROOT / "skills" / "agent-compass" / "scripts" / "candidate_eval.py").read_text(encoding="utf-8")
    notes = [line for line in source.splitlines() if "notes'].append" in line]
    assert sum(xp.WAIT_SENTENCE in line for line in notes) >= 2


SIM_PROBE = textwrap.dedent('''
    import datetime, json, sys, time
    sys.path.insert(0, sys.argv[1])
    import experience_sim_time
    print(json.dumps({"time": time.time(), "now": datetime.datetime.now(datetime.timezone.utc).timestamp(),
                      "today_is_date": isinstance(datetime.date.today(), datetime.date),
                      "plain_is_datetime": isinstance(datetime.datetime(2026, 1, 1), datetime.datetime)}))
''')


def test_the_simulated_clock_moves_time_and_now_in_a_process_that_names_the_clock_file(tmp_path):
    import os
    import subprocess
    import time
    clock_file = tmp_path / "sim-clock.txt"
    xp.simtime.write(clock_file, 7200.0)
    probe = tmp_path / "probe.py"
    probe.write_text(SIM_PROBE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != xp.simtime.FILE_ENV}
    real = time.time()
    shifted = json.loads(subprocess.run([sys.executable, str(probe), str(xp.HERE)], env={**env, xp.simtime.FILE_ENV: str(clock_file)},
                                        capture_output=True, text=True, check=True, timeout=60).stdout)
    plain = json.loads(subprocess.run([sys.executable, str(probe), str(xp.HERE)], env=env, capture_output=True, text=True, check=True,
                                      timeout=60).stdout)
    assert 7100 < shifted["time"] - real < 7300 and 7100 < shifted["now"] - real < 7300
    assert shifted["today_is_date"] and shifted["plain_is_datetime"]
    assert abs(plain["time"] - real) < 100            # without the file nothing moves


def test_the_scenario_clock_never_runs_the_product_clock_backwards(tmp_path):
    import time
    sim = xp.ScenarioClock(tmp_path / "sim-clock.txt")
    sim.set_hour(5.0)
    ahead = xp.simtime.offset(str(sim.file))
    assert abs(ahead - (sim.t0 + 5 * 3600 - time.time())) < 60
    sim.set_hour(1.0)
    assert xp.simtime.offset(str(sim.file)) == ahead


def fixture_cli(tmp_path, free_limit: dict | None):
    """zero_fake_cli.py copied as an opencode CLI with its own state folder (the script finds state beside its folder)."""
    import shutil
    folder = tmp_path / "bin"
    folder.mkdir()
    shutil.copyfile(xp.HERE / "zero_fake_cli.py", folder / "opencode.py")
    (tmp_path / "state").mkdir()
    if free_limit is not None:
        (tmp_path / "state" / "free-limit.json").write_text(json.dumps({"accounts": {"opencode": free_limit}}), encoding="utf-8")
    return folder / "opencode.py"


def run_fixture(tmp_path, script, model):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in (xp.simtime.FILE_ENV, "FIXTURE_AGENT_ID")}
    env["OPENCODE_DB_PATH"] = str(tmp_path / "opencode.db")
    return subprocess.run([sys.executable, str(script), "exec", "-m", model, "-"], input="do the fixture task\n", env=env,
                          capture_output=True, text=True, timeout=60)


def test_a_free_model_at_its_limit_answers_the_vendors_error_where_the_router_reads_its_reset(tmp_path):
    import datetime as dt
    sys.path.insert(0, str(xp.ROOT / "skills" / "agent-compass" / "scripts"))
    import axis_scores
    script = fixture_cli(tmp_path, {"limited": True, "models": ["deepseek-v4.1-flash-free"], "retryAfterMs": 3_600_000})
    answered = run_fixture(tmp_path, script, "opencode/deepseek-v4.1-flash-free")
    error = json.loads(answered.stderr.strip().splitlines()[-1])
    assert answered.returncode == 1
    assert error == {"type": "provider.free-limit", "status": 429, "retryAfterMs": 3_600_000,
                     "message": "Free usage limit reached. Connect a provider or subscribe to OpenCode Go to keep going, or switch models."}
    import os
    os.environ["OPENCODE_DB_PATH"] = str(tmp_path / "opencode.db")
    try:
        reset = axis_scores.opencode_failure_reset(error["message"], dt.datetime.now(dt.timezone.utc))
    finally:
        del os.environ["OPENCODE_DB_PATH"]
    assert reset is not None
    assert 3500 < (reset - dt.datetime.now(dt.timezone.utc)).total_seconds() <= 3600


def test_a_model_outside_the_limited_set_works_as_before(tmp_path):
    script = fixture_cli(tmp_path, {"limited": True, "models": ["deepseek-v4.1-flash-free"], "retryAfterMs": 3_600_000})
    answered = run_fixture(tmp_path, script, "opencode/some-paid-model")
    assert answered.returncode == 0 and "WORKER-DONE" in answered.stdout
    assert not (tmp_path / "opencode.db").exists()


def test_a_free_spend_is_written_where_the_router_reads_a_free_pools_use(tmp_path, monkeypatch):
    import types
    sys.path.insert(0, str(xp.ROOT / "skills" / "agent-compass" / "scripts"))
    import axis_scores
    db = tmp_path / "share" / "opencode.db"
    db.parent.mkdir()
    sim = xp.ScenarioClock(tmp_path / "sim-clock.txt")
    fake = types.SimpleNamespace(clock=sim, vendor={"opencode": {"spent": []}}, opencode_db=lambda: db)
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE message (id INTEGER PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT)")
    conn.close()
    price = {"input": 0.1, "output": 0.4}
    xp.Scenario.spend_free(fake, "opencode", "free-model", 0.04, price)
    monkeypatch.setenv("OPENCODE_DB_PATH", str(db))
    table = {"free-model": {"five_axis": {"list_price": {"value": price}}}}
    spend, unpriced, unreadable = axis_scores.opencode_local_spend(table)
    assert spend == {"free-model": 0.04} and not unpriced and unreadable is None
    assert fake.vendor["opencode"]["spent"] == [(0.0, 0.04)]


def test_settle_keeps_non_paseo_delivery_continuations_and_observe_pending_follows_fallback_attempts(tmp_path):
    """Rule 6: Herdr and Grok Bot `pending` answers (`check_delivery_do_not_resend`) run `next_argv`; Paseo fallback
    `pending` chains advance to `retry_at`, pass `--carry-fallback`, and follow `current_attempt`."""
    import datetime as dt
    import types

    sim = xp.ScenarioClock(tmp_path / "sim-clock.txt")
    cmd_calls, reconciles, statuses = [], [], {}

    class FakeScenario:
        def __init__(self):
            self.clock = sim
            self.state = tmp_path / "state"
            self.vendor = {}
            self.vendor_limited_at_roll = {}

        def command(self, _op, argv, **_kw):
            cmd_calls.append(list(argv))
            out = {"status": "launched", "launched": True, "state": "completed"}
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(out), stderr="")

        def call(self, _op, *args, **env):
            if args[:3] == ("dispatch", "--dispatch-action", "reconcile"):
                reconciles.append((tuple(args), dict(env)))
                return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")
            did = args[-1]
            doc = statuses[did]()
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(doc), stderr="")

        def advance_to(self, hour: float):
            self.clock.now = hour
            self.clock.set_hour(hour)
            return []

        def usage_of(self, _account):
            return None

    play = object.__new__(xp.Play)
    play.s = FakeScenario()
    play.answers = []
    play.charges = []
    play.free_limit_answers = []
    play.labels = {}
    play.sheet = {"bundles": {}, "capacity": {}, "prices": {}}
    play.corpus = {"b1": {"text": "task"}}
    play.note_resets = lambda _ev: None

    def add_answer(chain, doc, argv, _res, _err, brief, kind="roll"):
        idx = len(play.answers)
        rec = {
            "index": idx, "kind": kind, "brief": brief, "argv": argv,
            "status": doc.get("status"), "caller_action": doc.get("caller_action"),
            "dispatch_mode": doc.get("dispatch_mode"), "launched": doc.get("launched"),
            "dispatch_id": doc.get("dispatch_id"), "agent_id": doc.get("agent_id"),
            "account": doc.get("account"), "model": doc.get("model"), "effort": doc.get("effort"),
            "bundle_id": doc.get("bundle_id"), "expected_usd": doc.get("expected_usd"),
            "decision_id": doc.get("decision_id"), "next_action": doc.get("next_action"),
        }
        play.answers.append(rec)
        chain["answers"].append(idx)
        return rec

    play.answer = add_answer

    # 1. Herdr / Grok Bot `check_delivery_do_not_resend` runs `next_argv` continuation and reaches `launched`.
    herdr_doc = {
        "status": "pending", "launched": None, "dispatch_mode": "herdr",
        "caller_action": "check_delivery_do_not_resend", "next_argv": ["herdr", "status", "job-1"],
    }
    chain1 = {"id": "c1", "day": 1, "index": 0, "commands": 1, "answers": [], "waits": [], "outcome": None}
    add_answer(chain1, herdr_doc, ["roll"], None, None, "b1")
    play.settle(None, chain1, herdr_doc, {}, "b1", ["roll"], 0)
    assert cmd_calls == [["herdr", "status", "job-1"]]
    assert chain1["outcome"] == "launched"

    # 2. Paseo fallback `pending` on terminal `root` advances to `retry_at` and follows `current_attempt` to `d-attempt-2`.
    retry_iso = dt.datetime.fromtimestamp(sim.t0 + 600, tz=dt.timezone.utc).isoformat()
    statuses["d-root"] = lambda: (
        {"dispatch_id": "d-root", "state": "failed", "retry_at": retry_iso, "current_attempt": "d-root"}
        if sim.now < 0.16
        else {"dispatch_id": "d-root", "state": "failed", "current_attempt": "d-attempt-2", "attempts": ["1", "2"]}
    )
    statuses["d-attempt-2"] = lambda: {"dispatch_id": "d-attempt-2", "state": "started", "agent_id": "w-2"}
    paseo_doc = {
        "status": "pending", "launched": None, "dispatch_mode": "paseo",
        "caller_action": "wait_for_notification", "dispatch_id": "d-root",
        "next_action": "The last attempt hit a transient provider failure; AgentCompass retries the same model by itself; do not roll it again.",
    }
    chain2 = {"id": "c2", "day": 1, "index": 1, "commands": 1, "answers": [], "waits": [], "outcome": None}
    add_answer(chain2, paseo_doc, ["roll"], None, None, "b1")
    play.settle(None, chain2, paseo_doc, {}, "b1", ["roll"], 0)
    assert chain2["outcome"] == "relaunched"
    assert chain2["waits"] and chain2["waits"][0] > 0.16
    assert reconciles and all("--carry-fallback" in argv and "AGENTCOMPASS_CARRY_FALLBACK" not in env for argv, env in reconciles)
