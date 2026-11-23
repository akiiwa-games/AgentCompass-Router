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
