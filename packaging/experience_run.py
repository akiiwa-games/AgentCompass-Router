"""Play a main session's working day through the real router under many account situations; write a report a person reads.

    python packaging/experience_run.py [--scenario NAME ...] [--out DIR] [--parallel N] [--days N] [--keep]
    python packaging/experience_run.py --list
    python packaging/experience_run.py classify --out DIR [--spend-classifier-quota]   # without the flag: the plan only

There is no pass or fail. Every scenario (packaging/experience/scenarios.json) is an account situation: the accounts a user
has, how much of each quota window is left when the day starts, and the user's own switches (an account outside
`allowed_accounts`, a model unchecked, a subscription about to end). In each scenario the run plays the same stream of work
(packaging/experience/briefs.json) the way a main session does: `agentcompass roll` with the flags a session really uses, and
then what the answer says to do. It records every answer and writes `report.md` and `rows.json` into the run directory
(audit/experience_report.py builds the report from the rows).

How a scenario runs (the same isolated world as packaging/zero_dispatch_check.py; nothing in the router is replaced):
  - a wheel of this working tree in a fresh virtualenv; one sandbox home per scenario with dummy sign-ins, the sandbox's
    refusal of names and sockets (packaging/sandbox_check.py), the published-data channel off;
  - the external parties are fixtures: the provider usage endpoints (one local server answering every scenario's accounts
    from that scenario's simulated windows), the model behind the classifier (a local server answering each brief with the
    profile declared for it, or with the answers of the real classifier step below), the provider CLIs (the PyInstaller
    fixture CLI, built once and reused), and the Paseo host (packaging/zero_fake_paseo_bridge.mjs);
  - the main session is a Paseo agent (`PASEO_AGENT_ID`), so every launch goes through the host bridge as on this machine.

The simulated clock (packaging/experience_clock.py). Each scenario has its own clock in hours. The stream's entries carry
the hour of the working day they happen at; between days the clock skips the night. A deferral the router answers
(`start_after_hours`) is waited by moving the clock. When the clock moves, windows whose reset hour has passed reset to zero.
When a worker launches, every window its bundle draws is charged at once with `hours x usd_per_hour / capacity_usd` (the
model's list price per active hour and the window's dollars, the same public data and quotient as quota_economy.unit_rate;
the hours are the brief's declared expected hours). The router sees the clock only through the usage readings: before
each roll the run collects usage (`agentcompass usage --force`, what the tray's collector does every five minutes) and the
fixture answers each window's used share and its reset time as `now + (reset hour - clock)`.

What the simulated clock does not cover: everything the router keeps on real time. Ledger timestamps, reservations and
leases, launch-failure cooldowns, durable quota-exhaustion latches, an override's expiry, the dispatch cadence and the
classification cache all run on the wall clock, which moves a few minutes per simulated day. So a cooldown after a failed
launch still holds after the clock skipped a night, and a lease ends when its fixture worker ends (seconds), not after the
worker's simulated hours (the launch charge stands in for it). Fixture workers do no work: they finish, fail, crash or run
slowly as the stream declares.

Classifier. The default run spends no model tokens. `classify` asks the real configured classifier once per brief of the
corpus (`agentcompass score --input` of the installed product, on this machine's real home) and writes each answer whole
to `<out>/classifications.json` as it arrives (a stopped run resumes); `--classifications FILE` replays them instead of
the declared profiles. It spends the owner's classifier quota: without `--spend-classifier-quota` it prints the briefs
and the scorer it would ask and stops; pass the flag only with their consent, each time.
The same fixture answers the router's reading of a worker's error (provider_retry's scope question) from the scopes
declared for the fixture workers' error texts (briefs.json `error_scopes`); any other error text gets no answer, which
the router reads as `unknown`.

Processes. Every process the run starts is ended by the run: workers and watchers of a scenario when the scenario ends,
and at the end (also after Ctrl+C) every process whose command line names the run's sandbox directory. The report says
what was found and ended. At most 4 scenarios run at once (this machine's CPU is unreliable under all-core load).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experience_clock as clock  # noqa: E402
import sandbox_check as sc  # noqa: E402

ROOT = sc.ROOT
DATA = HERE / "experience"
PARENT = "main-session"
BRIDGE = HERE / "zero_fake_paseo_bridge.mjs"
MAX_PARALLEL = 4
FOLLOW_LIMIT = 6                     # answers one roll may lead to before the chain counts as not settling
WAIT_LIMIT_HOURS = 30.0              # a deferral longer than this is not waited for (recorded, the session moves on)
NIGHT_HOURS = 14.0
# How long one command of the session may take before the run stops waiting for it. A roll is synchronous while
# AgentCompass relaunches a brief whose worker fails at once on candidate after candidate (dispatch_fallback goes on until
# no startable candidate is left): on the large fleet one such roll took 822 s (2026-11-24), so a few minutes would cut the
# scenario in the middle of the product's own behaviour. The limit only stops a command that hangs; a command that reaches
# it is recorded as an answer with status `timed_out` (its process tree ended) and the scenario goes on.
COMMAND_TIMEOUT_SECONDS = 3600


# ---------------------------------------------------------------------------------------------------------------------
# declared data


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_cases() -> dict:
    doc = load_json(ROOT / "skills" / "agent-compass" / "references" / "classifier-cases.json")
    return {case["id"]: case for case in doc["calibration_cases"]}


def brief_corpus(briefs_doc: dict) -> dict:
    """{brief id: {"text", "profile", "split_into"?}}: the stream's own briefs plus the calibration cases they name."""
    cases = calibration_cases()
    corpus = {}
    for key, entry in briefs_doc["briefs"].items():
        if "case" in entry:
            case = cases[entry["case"]]
            corpus[key] = {"text": case["task"], "profile": dict(case["expected_profile"]), "source": f"classifier-cases.json {entry['case']}"}
        else:
            corpus[key] = {"text": entry["text"], "profile": dict(entry["profile"]), "source": "briefs.json"}
        for extra in ("split_into", "note"):
            if extra in entry:
                corpus[key][extra] = entry[extra]
    for key, item in corpus.items():
        item["profile"].setdefault("reason", "experience-run fixture: the profile declared for this brief")
        for part in item.get("split_into", []):
            if part not in corpus:
                raise ValueError(f"brief {key} splits into {part}, which is not a brief")
    return corpus


# ---------------------------------------------------------------------------------------------------------------------
# fixtures: provider usage endpoints and the classifier model


class ScenarioClock:
    """One scenario's simulated hour; read by the usage fixture to turn simulated reset hours into real timestamps."""

    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()


class UsageServer:
    """The provider usage endpoints of every scenario at once. Each signed-in account has its own access token; a request
    is answered from that account's simulated windows (experience_clock) on its scenario's clock."""

    def __init__(self):
        self.accounts: dict[str, dict] = {}          # token -> {"family", "plan", "usage": AccountUsage, "clock": ScenarioClock}
        self.requests = 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer.requests += 1
                host, _, rest = self.path.lstrip("/").partition("/")
                path = "/" + rest.split("?", 1)[0]
                token = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                entry = outer.accounts.get(token)
                if entry is None:
                    return self._reply(401, {"error": "unknown fixture token"})
                answer = outer.answer(entry, host, path)
                return self._reply(200, answer) if answer is not None else self._reply(404, {"error": "unknown fixture path"})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._reply(404, {"error": "unknown fixture path"})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @staticmethod
    def answer(entry, host, path):
        real = dt.datetime.now(dt.timezone.utc)
        sim = entry["clock"]
        with sim.lock:
            windows = [copy.copy(w) for w in entry["usage"].windows]
            now = sim.now

        def at(window):
            return real + dt.timedelta(hours=window.remaining_hours(now))

        plan = entry["plan"]
        if host == "chatgpt.com" and path == "/backend-api/wham/usage":
            slots = {}
            for window in windows:
                slot = "primary_window" if window.hours <= 5 else "secondary_window"
                slots[slot] = {"used_percent": round(window.used, 3), "limit_window_seconds": int(window.hours * 3600),
                               "reset_at": int(at(window).timestamp())}
            return {"plan_type": plan, "rate_limit": slots}
        if host == "api.anthropic.com" and path == "/api/oauth/profile":
            kind = "claude_max" if plan.startswith("max") else f"claude_{plan}"
            tier = {"max5": "default_claude_max_5x", "max20": "default_claude_max_20x"}.get(plan, "default_claude_ai")
            return {"organization": {"organization_type": kind, "rate_limit_tier": tier}}
        if host == "api.anthropic.com" and path == "/api/oauth/usage":
            body, limits = {}, []
            for window in windows:
                if window.models:
                    limits.append({"kind": "weekly_scoped", "scope": {"model": {"display_name": window.models[0].capitalize()}},
                                   "percent": round(window.used, 3), "resets_at": at(window).isoformat()})
                elif window.id in ("five_hour", "seven_day"):
                    body[window.id] = {"utilization": round(window.used, 3), "resets_at": at(window).isoformat()}
            if limits:
                body["limits"] = limits
            return body
        return None


class ReplayScorer:
    """The model behind the classifier: answers the OpenAI-compatible request the router sends with the profile declared for
    the brief (found by its `EXPERIENCE-BRIEF: <id>` line), or with the real classifier's stored answer for it. A request
    about a worker's error (provider_retry's scope question, whose input carries `provider_message`) is answered with the
    scope declared for that exact error text (`error_scopes`); any other gets no answer."""

    MARK = re.compile(r"EXPERIENCE-BRIEF:\s*([\w.-]+)")

    def __init__(self, corpus: dict, replay: dict | None = None, error_scopes: dict | None = None):
        self.corpus, self.replay, self.error_scopes = corpus, replay or {}, error_scopes or {}
        self.requests = 0
        self.unknown: list[str] = []
        self.scopes_answered: dict[str, int] = {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                outer.requests += 1
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                user = next((m["content"] for m in body.get("messages", []) if m.get("role") == "user"), "")
                found = outer.MARK.search(user or "")
                key = found.group(1) if found else None
                scope = None if found else outer.scope_of(user)
                if scope is not None:
                    outer.scopes_answered[scope] = outer.scopes_answered.get(scope, 0) + 1
                    return self._send(200, {"choices": [{"message": {"content": json.dumps(
                        {"scope": scope, "reason": "experience-run fixture: the scope declared for this error text"})}}]})
                profile = outer.replay.get(key) or (outer.corpus.get(key) or {}).get("profile")
                if profile is None:
                    outer.unknown.append((user or "")[:200])
                    return self._send(500, {"error": "no declared profile for this brief"})
                self._send(200, {"choices": [{"message": {"content": json.dumps(profile)}}]})

            def _send(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def scope_of(self, text: str) -> str | None:
        """The declared scope of the error a scope question carries, or None (not a scope question, or an undeclared text)."""
        try:
            message = json.loads(text).get("provider_message")
        except (ValueError, AttributeError):
            return None
        entry = self.error_scopes.get(message.strip()) if isinstance(message, str) else None
        return entry.get("scope") if isinstance(entry, dict) else None

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------------------------------------------------------
# the world: one wheel, one virtualenv, one fixture CLI build, shared by every scenario


def fake_cli_build(cache: Path) -> Path:
    """The fixture CLI (packaging/zero_fake_cli.py) as a one-folder PyInstaller build, cached by the script's hash."""
    source = HERE / "zero_fake_cli.py"
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    built = cache / f"fakecli-{key}" / "dist" / "fake-cli"
    if (built / "fake-cli.exe").is_file():
        return built
    work = cache / f"fakecli-{key}"
    shutil.rmtree(work, ignore_errors=True)
    done = subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--name", "fake-cli", "--distpath", str(work / "dist"),
                           "--workpath", str(work / "work"), "--specpath", str(work / "spec"), str(source)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    if done.returncode != 0 or not (built / "fake-cli.exe").is_file():
        raise RuntimeError("could not build the fixture CLI (PyInstaller is required): " + (done.stderr or done.stdout)[-400:])
    return built


def install_world(base: Path, wheel: Path | None) -> tuple[Path, Path]:
    """(venv, wheel): the product installed from a wheel of this working tree (or the given one)."""
    if wheel is None:
        repo = sc.build_fixture_repo(base)
        wheel = sc.build_wheel(repo, base / "dist")
    venv = base / "ve nv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--quiet", str(wheel)], check=True, capture_output=True)
    return venv, wheel


def parse(text: str):
    try:
        return sc.parse_json(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------------------------
# one scenario: a sandbox home, its accounts and its clock


def plan_windows(family: str, plan: str) -> dict:
    templates = load_json(ROOT / "skills" / "agent-compass" / "references" / "plan-templates.json")
    return ((templates.get(family) or {}).get(plan) or {}).get("windows") or {}


class World:
    def __init__(self, base: Path, venv: Path, fakecli: Path, usage: UsageServer, scorer: ReplayScorer, out: Path):
        self.base, self.venv, self.fakecli, self.usage, self.scorer, self.out = base, venv, fakecli, usage, scorer, out
        self.node = shutil.which("node")


class Scenario:
    """One user: a sandbox home with the scenario's sign-ins, switches and simulated windows."""

    def __init__(self, world: World, spec: dict, index: int, states: dict):
        self.world, self.spec, self.index = world, spec, index
        self.name = spec["name"]
        self.report = world.out / "scenarios" / self.name
        self.run = sc.Run(self.report)
        self.box = sc.Sandbox(world.base, f"s{index:02d} {self.name}", world.venv, world.usage, self.run)
        self.box.ports.add(world.scorer.port)
        self.clock = ScenarioClock()
        self.bin = self.box.root / "fixture" / "bin"
        self.state = self.box.root / "fixture" / "state"
        self.work = self.box.root / "work dir"
        shutil.copytree(world.fakecli, self.bin)
        for name in ("codex", "claude", "opencode"):
            shutil.copy2(self.bin / "fake-cli.exe", self.bin / f"{name}.exe")
        (self.bin / "fake-cli.exe").unlink()
        self.state.mkdir(parents=True)
        (self.state / "parents.json").write_text(json.dumps([PARENT]), encoding="utf-8")
        self.work.mkdir()
        self.accounts: list[clock.AccountUsage] = []
        self.account_specs: dict[str, dict] = {}
        self.sign_in(states)

    # ---- environment
    def path(self) -> str:
        parts = [str(self.bin), str(self.box.python.parent), os.environ.get("SYSTEMROOT", r"C:\Windows") + r"\System32"]
        if self.world.node:
            parts.insert(2, str(Path(self.world.node).parent))
        return os.pathsep.join(parts)

    def env(self, **extra) -> dict:
        return {"PATH": self.path(), "FIXTURE_STATE": str(self.state), "FIXTURE_BIN": str(self.bin), "ROUTER_SCORE_CACHE": "off",
                "PASEO_AGENT_ID": PARENT, "ROUTER_PASEO_BRIDGE": str(BRIDGE), **extra}

    def call(self, op, *args, stdin=None, timeout=COMMAND_TIMEOUT_SECONDS, **env):
        argv = [self.box.agentcompass, *map(str, args)]
        return self.command(op, argv, stdin=stdin, timeout=timeout, **env)

    def command(self, op, argv, stdin=None, timeout=COMMAND_TIMEOUT_SECONDS, **env):
        """Run one command of the session; one that does not answer within `timeout` is ended with every process it started
        (run_owned) and comes back as a `TimedOut` result instead of an exception (COMMAND_TIMEOUT_SECONDS)."""
        environment = self.box.env(**self.env(**env))
        try:
            result = run_owned(argv, cwd=self.work, env=environment, stdin=stdin, timeout=timeout)
        except CommandTimedOut as stopped:
            return TimedOut(timeout, stopped.ended)
        op.log(argv, ", ".join(f"{k}={environment[k]}" for k in ("HOME", "ROUTER_STATE_DIR") if k in environment), result)
        return result

    # ---- accounts
    def sign_in(self, states: dict):
        home = self.box.home
        definitions = {}
        usage_names = self.spec.get("usage_by_account") or {}
        for account in self.spec["accounts"]:
            name, family, plan = account["name"], account["family"], account.get("plan")
            token = f"xp-{self.index:02d}-{name}-access"
            self.account_specs[name] = account
            if family == "claude":
                folder = home / ("." + name)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {"accessToken": token, "subscriptionType": plan}}), encoding="utf-8")
                if name != "claude":
                    definitions[name] = {"extends": "claude", "env": {"CLAUDE_CONFIG_DIR": str(folder)}}
            elif family == "codex":
                folder = home / ("." + name)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "auth.json").write_text(json.dumps({"tokens": {"access_token": token, "account_id": f"xp-account-{self.index:02d}-{name}"}}),
                                                  encoding="utf-8")
                if name != "codex":
                    definitions[name] = {"extends": "codex", "env": {"CODEX_HOME": str(folder)}}
            elif family != "opencode":
                raise ValueError(f"scenario {self.name}: the fixtures carry claude, codex and opencode accounts, not {family}")
            windows = []
            state = states[usage_names.get(name) or self.spec.get("usage", "fresh")]
            for wid, wspec in plan_windows(family, plan).items():
                hours = float(wspec.get("hours") or 168)
                used, elapsed = state["short" if hours < 24 else "long"]
                window = clock.start_window(wid, hours, used, elapsed)
                window.models, window.models_except = wspec.get("models"), wspec.get("models_except")
                windows.append(window)
            usage = clock.AccountUsage(name, windows)
            self.accounts.append(usage)
            if family in ("claude", "codex"):
                self.world.usage.accounts[token] = {"family": family, "plan": plan, "usage": usage, "clock": self.clock}
        if definitions:
            paseo = home / ".paseo"
            paseo.mkdir(parents=True, exist_ok=True)
            (paseo / "config.json").write_text(json.dumps({"agents": {"providers": definitions}}, indent=1), encoding="utf-8")

    # ---- the router's own reading of this home (bundles, the windows each draws, window capacities)
    SHEET = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import account_fleet, axis_scores, quota_pools\n"
        "acc = axis_scores.five_axis_overlay(account_fleet.load_accounts())\n"
        "out = {'bundles': {}, 'capacity': {}}\n"
        "for bid, b in account_fleet.bundles(acc).items():\n"
        "    info = account_fleet.model_spec(acc, b['model'], account_fleet.bundle_providers(b, acc))\n"
        "    out['bundles'][bid] = {'account': b['account'], 'provider': b['provider'], 'model': b['model'], 'effort': b.get('effort'),\n"
        "                           'disabled': bool(b.get('disabled')), 'draws': quota_pools.bundle_pools(b, acc),\n"
        "                           'tier': (info.get('five_axis') or {}).get('tier'), 'usd_per_hour': info.get('usd_per_hour')}\n"
        "for name, spec in (acc.get('accounts') or {}).items():\n"
        "    out['capacity'][name] = dict(spec.get('capacity_usd') or {})\n"
        "print(json.dumps(out))\n")

    def read_sheet(self, op) -> dict:
        result = self.box.run_command(op, [self.box.python, "-X", "utf8", "-c", self.SHEET, self.box.scripts], cwd=self.work, **self.env())
        sheet = parse(result.stdout) or {"bundles": {}, "capacity": {}}
        names = set(self.account_specs)
        sheet["bundles"] = {bid: b for bid, b in sheet["bundles"].items() if b["account"] in names}
        sheet["capacity"] = {name: caps for name, caps in sheet["capacity"].items() if name in names}
        return sheet

    # ---- ledger and fixture state
    def ledger(self) -> list[dict]:
        path = self.box.state / "router-ledger.jsonl"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows

    def agent(self, agent_id: str) -> dict:
        path = self.state / "agents" / (re.sub(r"[^\w.-]", "_", agent_id) + ".json")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def set_fault(self, **faults):
        (self.state / "faults.json").write_text(json.dumps(faults), encoding="utf-8")

    def workers(self) -> list[dict]:
        folder = self.state / "workers"
        records = []
        for path in folder.glob("*.json") if folder.is_dir() else []:
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return records

    # ---- clock
    def advance_to(self, hour: float) -> list[clock.ResetEvent]:
        with self.clock.lock:
            if hour <= self.clock.now:
                return []
            events = clock.advance(self.accounts, self.clock.now, hour)
            self.clock.now = hour
        return events

    def windows_now(self) -> dict:
        with self.clock.lock:
            return clock.snapshot(self.accounts, self.clock.now)

    def usage_of(self, name: str) -> clock.AccountUsage | None:
        return next((a for a in self.accounts if a.name == name), None)

    # ---- processes
    def end_processes(self) -> list[str]:
        """End every process whose command line names this scenario's sandbox (workers, watchers) at the scenario's end;
        returns what was found."""
        found = sc.sandbox_processes(self.box.root)
        for entry in found:
            subprocess.run(["taskkill", "/F", "/T", "/PID", entry.split(":")[0]], capture_output=True)
        return found


class TimedOut:
    """The result of a command that did not answer within its limit (Scenario.command)."""

    def __init__(self, seconds: int, ended: list[str]):
        self.returncode, self.stdout, self.timed_out, self.ended = None, "", True, ended
        self.stderr = f"the command did not answer within {seconds} s; ended: {', '.join(ended) or 'nothing left'}"


class CommandTimedOut(Exception):
    """`run_owned`: the command did not answer within its limit; `ended` names the processes its tree ended with."""

    def __init__(self, ended: list[str]):
        super().__init__("timed out")
        self.ended = ended


class Job:
    """A Windows job object that holds one command and every process it starts, so a command that hangs ends with its
    whole tree by the handle this run holds, never by a search of command lines. A process the product starts detached
    (CREATE_BREAKAWAY_FROM_JOB: a watcher) leaves the job as it would on any machine (JOB_OBJECT_LIMIT_BREAKAWAY_OK): it is
    the scenario's, and the scenario's end ends it (Scenario.end_processes)."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.wintypes = ctypes, wintypes
        kernel = self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x800   # JOB_OBJECT_LIMIT_BREAKAWAY_OK
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def add(self, pid: int):
        process = self.kernel.OpenProcess(0x0100 | 0x0001, False, pid)   # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not process:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

    def pids(self) -> list[int]:
        ctypes, wintypes = self.ctypes, self.wintypes

        class Listing(ctypes.Structure):
            _fields_ = [("Assigned", wintypes.DWORD), ("Listed", wintypes.DWORD), ("Ids", ctypes.c_size_t * 1024)]
        listing = Listing()
        if not self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(listing), ctypes.sizeof(listing), None):
            return []
        return [int(listing.Ids[i]) for i in range(listing.Listed)]

    def end(self) -> list[str]:
        ended = [str(pid) for pid in self.pids()]
        self.kernel.TerminateJobObject(self.handle, 1)
        return ended

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_owned(argv, *, cwd, env: dict, stdin: str | None, timeout: float) -> subprocess.CompletedProcess:
    """Run one command and capture its output; past `timeout`, end the command and every process it started that is still
    in its tree (a Windows job object, a POSIX session), by the handles this run holds, and raise CommandTimedOut."""
    job = Job() if os.name == "nt" else None
    try:
        process = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), env=env, text=True, encoding="utf-8", errors="replace",
                                   stdin=subprocess.PIPE if stdin is not None else None, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=os.name != "nt")
        if job:
            try:
                job.add(process.pid)
            except OSError:
                process.kill()
                process.communicate()
                raise
        try:
            out, err = process.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            if job:
                ended = job.end()
            else:
                ended = [str(process.pid)]
                try:
                    os.killpg(process.pid, 9)
                except OSError:
                    pass
            process.kill()
            try:
                process.communicate(timeout=30)
            except subprocess.TimeoutExpired:   # a process that broke away still holds the pipes: stop reading them
                for stream in (process.stdout, process.stderr):
                    if stream:
                        stream.close()
                process.wait()
            raise CommandTimedOut(ended) from None
        return subprocess.CompletedProcess(argv, process.returncode, out, err)
    finally:
        if job:
            job.close()


def find_key(document, key):
    """The first value of `key` anywhere in a nested document, or None."""
    if isinstance(document, dict):
        if document.get(key) is not None:
            return document[key]
        for value in document.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(document, list):
        for value in document:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def compact_ranking(ranking: list) -> list:
    """The eligible candidates of a plan's recorded snapshot (score not None), best first, with the router's own numbers
    copied as recorded: score N, fit, opportunity_cost (the quota cost N subtracts), quota_price, decided_by, the
    candidate's expected_usd and expected_pct. A field the snapshot did not record stays None (ledger_snapshot keeps every
    component for the best 6 eligible candidates and fewer for the next 8)."""
    out = []
    for candidate in ranking:
        if candidate.get("score") is None:
            continue
        bundle = candidate.get("bundle") or {}
        components = candidate.get("components") or {}
        out.append({"id": candidate.get("id"), "model": bundle.get("model"), "account": bundle.get("account"), "provider": bundle.get("provider"),
                    "effort": bundle.get("effort"), "score": candidate.get("score"), "fit": components.get("fit"),
                    "opportunity_cost": components.get("opportunity_cost"), "quota_price": components.get("quota_price"),
                    "decided_by": components.get("decided_by"), "level": components.get("five_axis_level"),
                    "usd": candidate.get("expected_usd"), "pct": candidate.get("expected_pct")})
    return out


TERMINAL = ("completed", "failed", "canceled", "create_failed", "released", "retry_expired")


class Play:
    """Plays the stream in one scenario and keeps every answer, chain, charge, reset and window reading."""

    def __init__(self, scenario: Scenario, corpus: dict, day: list, days: int, stop: threading.Event):
        self.s, self.corpus, self.day, self.days, self.stop = scenario, corpus, day, days, stop
        self.answers: list[dict] = []
        self.chains: list[dict] = []
        self.charges: list[dict] = []
        self.resets: list[dict] = []
        self.timeline: list[dict] = []
        self.labels: dict[str, str] = {}
        self.sheet: dict = {}
        self.setup: list[dict] = []
        self.exclusions: dict[str, dict] = {}     # bundle id -> {"plans", "eligible", "won", "gaps", "excluded": {reason: n}}
        self.counter = 0

    # ---- preparing the home as a user does
    def prepare(self):
        s = self.s
        op = s.run.operation("setup", "sign-ins, classifier, setup, the user's switches")
        steps = [["scorer", "--action", "set", "--type", "openai_compatible", "--base-url", s.world.scorer.base, "--model", "fixture-classifier",
                  "--api-key", "fixture-key"], ["setup", "--yes", "--no-hooks"]]
        for argv in steps:
            result = s.call(op, *argv)
            self.setup.append({"argv": argv[:2], "exit": result.returncode, "tail": (result.stdout + result.stderr)[-600:]})
        switches = s.spec.get("switches") or {}
        if switches.get("outside_allowed"):
            names = list(s.account_specs)
            policy = {"allowed_accounts": [n for n in names if n not in switches["outside_allowed"]], "known_accounts": names}
            (s.box.state / "host-policy.json").write_text(json.dumps(policy, indent=1), encoding="utf-8")
            self.setup.append({"argv": ["host-policy.json"], "exit": 0, "tail": json.dumps(policy)})
        self.sheet = s.read_sheet(op)
        if switches.get("unchecked"):
            path = s.box.state / "accounts-override.json"
            document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            for entry in switches["unchecked"]:
                for bid, bundle in self.sheet["bundles"].items():
                    if bundle["account"] == entry["account"] and bundle["model"] == entry["model"]:
                        document.setdefault("accounts", {}).setdefault(entry["account"], {}).setdefault("bundles", {}).setdefault(bid, {})["disabled"] = True
            path.write_text(json.dumps(document, indent=1), encoding="utf-8")
            self.setup.append({"argv": ["accounts-override.json"], "exit": 0, "tail": json.dumps(switches["unchecked"])})
            self.sheet = s.read_sheet(op)
        for entry in switches.get("plan_ends") or []:
            result = s.call(op, "override", "--account", entry["account"], "--basis", entry["basis"], "--expires-in", entry["in_hours"])
            self.setup.append({"argv": ["override", "--expires-in"], "exit": result.returncode, "tail": (result.stdout + result.stderr)[-600:]})
        self.collect(op)
        self.timeline.append({"at": 0.0, "event": "start", "windows": s.windows_now()})

    def collect(self, op):
        """What the tray's collector does every five minutes: one usage reading of every account."""
        return self.s.call(op, "usage", "--force")

    # ---- the stream
    def text(self, key: str, day: int, index: int, worker: str | None) -> str:
        nonce = hashlib.sha256(f"{self.s.name}{day}{index}{key}{self.counter}".encode()).hexdigest()[:8]
        self.counter += 1
        lines = [self.corpus[key]["text"], "", f"EXPERIENCE-BRIEF: {key}", f"[experience day {day + 1} #{index} {nonce}]"]
        if worker and worker != "ok":
            lines.append(f"FIXTURE-SCENARIO: {worker}")
        return "\n".join(lines) + "\n"

    def brief_file(self, key: str, day: int, index: int, worker: str | None) -> Path:
        folder = self.s.work / "briefs"
        folder.mkdir(exist_ok=True)
        path = folder / f"d{day + 1}-{index:02d}-{self.counter:03d}-{key}.md"
        path.write_text(self.text(key, day, index, worker), encoding="utf-8")
        return path

    def play(self):
        for day in range(self.days):
            start = day * 24.0
            for index, act in enumerate(self.day):
                if self.stop.is_set():
                    return
                self.note_resets(self.s.advance_to(start + float(act["at"])))
                self.act(act, day, index)
                self.timeline.append({"at": round(self.s.clock.now, 3), "event": f"d{day + 1}#{index}", "windows": self.s.windows_now()})
            if day + 1 < self.days:
                self.note_resets(self.s.advance_to((day + 1) * 24.0))

    def note_resets(self, events):
        for event in events:
            self.resets.append({"account": event.account, "window": event.window, "at": round(event.at, 3), "used_before": round(event.used_before, 3),
                                "unused": round(event.unused, 3)})

    def account_for(self, family: str) -> str:
        return next((a["name"] for a in self.s.spec["accounts"] if a["family"] == family), family)

    def act(self, act: dict, day: int, index: int):
        s = self.s
        op = s.run.operation(f"d{day + 1}#{index}", json.dumps(act)[:200])
        keys = act.get("briefs") or [act["brief"]]
        chain = {"id": str(len([c for c in self.chains if "batch_of" not in c])), "day": day + 1, "index": index, "at": round(s.clock.now, 3),
                 "briefs": keys, "flags": sorted(k for k in act if k not in ("at", "brief", "briefs", "label")), "answers": [], "outcome": None,
                 "commands": 0, "waits": []}
        self.chains.append(chain)
        if act.get("resume") and not self.labels.get(act["resume"]):
            chain["outcome"] = "not_played"
            chain["note"] = f"no worker was launched for label {act['resume']}, so there was nothing to resume"
            return
        self.roll(op, chain, self.argv(act, keys, day, index), act, keys)

    def argv(self, act, keys, day, index) -> list:
        s = self.s
        argv = ["roll"]
        if len(keys) > 1:
            for key in keys:
                argv += ["--brief", self.brief_file(key, day, index, act.get("worker"))]
        else:
            argv += ["--task", self.text(keys[0], day, index, act.get("worker"))]
        if act.get("resume"):
            argv += ["--resume", self.labels[act["resume"]]]
        if act.get("can_wait_hours"):
            argv += ["--can-wait-hours", act["can_wait_hours"]]
        if act.get("named"):
            named = act["named"]
            account = self.account_for(named["family"])
            provider = next((b["provider"] for b in self.sheet["bundles"].values() if b["account"] == account and b["model"] == named["model"]), account)
            argv += ["--provider", provider, "--account", account, "--model", named["model"], "--effort", named["effort"], "--basis", named["basis"]]
        if act.get("workspace"):
            folder = s.work / act["workspace"]
            folder.mkdir(exist_ok=True)
            argv += ["--workspace", folder]
        return argv

    def roll(self, op, chain, argv, act, keys, depth=0):
        """One roll and everything its answer leads to."""
        s = self.s
        if act.get("fault") and depth == 0:
            s.set_fault(create=act["fault"])
        self.collect(op)
        before = len(s.ledger())
        started = time.time()
        result = s.call(op, *argv)
        seconds = round(time.time() - started, 2)
        chain["commands"] += 1
        doc = parse(result.stdout) or {}
        # The decisions this roll wrote, in order (one per brief: briefs are routed one at a time).
        written = [r for r in s.ledger()[before:] if r.get("stage") in ("plan", "record", "split")]
        if doc.get("status") == "batch":
            self.answer(chain, doc, argv, result, seconds, keys[0], batch=True)
            if chain["outcome"] is None:
                chain["outcome"] = "batch"
            for position, item in enumerate(doc.get("results") or []):
                key = keys[position] if position < len(keys) else keys[-1]
                sub = {**chain, "id": f"{chain['id']}.{position}", "briefs": [key], "answers": [], "outcome": None, "commands": 0, "waits": [],
                       "batch_of": chain["id"]}
                self.chains.append(sub)
                # A batch routes its briefs in order, and each writes one plan, record or split row.
                mine = written[position] if position < len(written) else {}
                self.answer(sub, item, argv, None, None, key, part_of_batch=True, row=mine)
                self.settle(op, sub, item, {**act, "fault": None}, key, argv, depth, in_batch=True)
            return
        self.answer(chain, doc, argv, result, seconds, keys[0], row=written[-1] if written else {})
        self.settle(op, chain, doc, act, keys[0], argv, depth)

    def details(self, doc: dict) -> dict:
        path = doc.get("details")
        try:
            return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).is_file() else {}
        except (OSError, ValueError):
            return {}

    def plan_row(self, decision_id: str | None) -> dict:
        if not decision_id:
            return {}
        return next((r for r in reversed(self.s.ledger()) if r.get("decision_id") == decision_id), {})

    def answer(self, chain, doc, argv, result, seconds, key, batch=False, part_of_batch=False, kind="roll", row=None):
        full = {} if batch else (self.details(doc) or (doc if kind == "roll" and "route" in doc else {}))
        worker = next((w for w in (full.get("worker"), doc.get("worker")) if isinstance(w, dict)), {})
        decision_id = worker.get("decision_id") or doc.get("decision_id") or find_key(full, "decision_id")
        row = (self.plan_row(decision_id) if decision_id else {}) or (row or {}) if not batch else {}
        decision_id = decision_id or row.get("decision_id")
        snap = row.get("decision_snapshot") or {}
        decision = snap.get("decision") or {}
        bundle = snap.get("selected_bundle") or {}
        route_bundle = (full.get("route") or {}).get("bundle") or {}
        # The profile the router planned with, as its snapshot recorded it; else the one the fixture classifier answered.
        served = self.s.world.scorer.replay.get(key) or self.corpus[key]["profile"]
        profile = (snap.get("task") or {}).get("profile") or (decision.get("task") or {}).get("profile") or served
        record = {
            "scenario": self.s.name, "chain": chain["id"], "step": len(chain["answers"]), "kind": kind, "day": chain["day"], "index": chain["index"],
            "at": round(self.s.clock.now, 3), "brief": key, "work_kind": profile.get("work_kind"), "level": profile.get("primary_level"),
            "declared_hours": profile.get("expected_hours"), "flags": chain["flags"],
            "argv": [shown_arg(a) for a in argv], "exit": result.returncode if result is not None else None, "seconds": seconds,
            "stdout_chars": len(result.stdout) if result is not None else None,
            "status": "timed_out" if getattr(result, "timed_out", False) else doc.get("status") or doc.get("state"), "caller_action": full.get("caller_action") or doc.get("caller_action"),
            "dispatch_mode": full.get("dispatch_mode") or doc.get("dispatch_mode"), "launched": full.get("launched", doc.get("launched")),
            "reason": doc.get("reason") or (full.get("route") or {}).get("reason"), "cause": doc.get("cause") or full.get("cause"),
            "next_action": doc.get("next_action"), "next_argv": doc.get("next_argv"), "warnings": doc.get("warnings"),
            "assumptions": doc.get("assumptions"), "blocked_by": doc.get("blocked_by") or full.get("blocked_by"),
            "start_after_hours": find_key(full, "start_after_hours") or find_key(doc, "start_after_hours") or decision.get("start_after_hours"),
            "wait_hours": decision.get("start_after_hours") or (decision.get("wait_for") or {}).get("hours") or (decision.get("next_reset") or {}).get("hours")
                          or said_hours(doc.get("next_action")),
            "attempts": len(doc.get("attempts") or full.get("attempts") or []) or None,
            "decision_id": decision_id, "dispatch_id": worker.get("dispatch_id"), "agent_id": worker.get("agent_id"),
            "worker_state": worker.get("state"),
            "account": bundle.get("account") or route_bundle.get("account"), "model": bundle.get("model") or route_bundle.get("model"),
            "effort": bundle.get("effort") or route_bundle.get("effort"), "bundle_id": bundle.get("bundle_id") or row.get("choice"),
            "expected_usd": bundle.get("expected_usd") or row.get("expected_usd") or worker.get("expected_usd"),
            "expected_hours": bundle.get("expected_hours"), "router_difficulty": (decision.get("task") or {}).get("difficulty"),
            "decision_reason": decision.get("reason"), "decision_action": decision.get("action"), "ledger_stage": row.get("stage"),
            "ledger_constraint": row.get("constraint"), "ranking": compact_ranking(snap.get("ranking") or []),
            "part_of_batch": part_of_batch, "batch": batch, "carried_launch": False,
        }
        if snap.get("ranking"):
            self.note_ranking(snap["ranking"], snap.get("choice"))
        chain["answers"].append(len(self.answers))
        self.answers.append(record)
        return record

    def note_ranking(self, ranking: list, choice: str | None):
        top = next((c.get("score") for c in ranking if c.get("score") is not None), None)
        for candidate in ranking:
            entry = self.exclusions.setdefault(candidate.get("id"), {"plans": 0, "eligible": 0, "won": 0, "gaps": [], "excluded": {}})
            entry["plans"] += 1
            if candidate.get("id") == choice:
                entry["won"] += 1
            if candidate.get("excluded"):
                for reason in candidate["excluded"]:
                    short = re.sub(r"[\d.]+", "#", str(reason))[:140]
                    entry["excluded"][short] = entry["excluded"].get(short, 0) + 1
            elif candidate.get("score") is not None:
                entry["eligible"] += 1
                if top is not None:
                    entry["gaps"].append(round(top - candidate["score"], 4))

    # ---- following the answer the way a session does
    def settle(self, op, chain, doc, act, key, argv, depth, in_batch=False):
        s = self.s
        record = self.answers[chain["answers"][-1]]
        status, action, mode = record["status"], record["caller_action"], record["dispatch_mode"]
        if status == "launched" or record["launched"] is True:
            self.launched(op, chain, record, act)
            return
        if status == "pending" and isinstance(doc.get("next_argv"), list):
            if chain["commands"] >= FOLLOW_LIMIT:
                chain["outcome"], chain["note"] = "dead_end", f"still pending after {chain['commands']} commands"
                return
            result = s.command(op, [str(a) for a in doc["next_argv"]])
            chain["commands"] += 1
            got = parse(result.stdout) or {}
            follow = self.answer(chain, got, doc["next_argv"], result, None, key, kind="next_argv")
            follow.update({k: record[k] for k in ("account", "model", "effort", "bundle_id", "expected_usd", "decision_id") if follow.get(k) is None})
            if got.get("state") in ("completed", "launched", "started") or got.get("status") == "launched":
                self.launched(op, chain, follow, act)
            elif got.get("status") == "pending" and isinstance(got.get("next_argv"), list):
                self.settle(op, chain, got, act, key, argv, depth)
            else:
                chain["outcome"] = "handed_to_session"
            return
        if mode == "split" or action == "roll_each_line":
            parts = self.corpus[key].get("split_into")
            if not parts or depth > 0:
                chain["outcome"], chain["note"] = "handed_to_session", "split answered and no split of this brief is declared"
                return
            chain["outcome"] = "split"
            day, index = chain["day"] - 1, chain["index"]
            again = ["roll"]
            for part in parts:
                again += ["--brief", self.brief_file(part, day, index, act.get("worker"))]
            self.roll(op, chain, again, {**act, "fault": None}, parts, depth + 1)
            return
        if status == "blocked" and action == "wait_for_reset":
            hours = record["start_after_hours"]
            if not isinstance(hours, (int, float)) or hours <= 0 or hours > WAIT_LIMIT_HOURS or depth >= 3:
                chain["outcome"] = "deferred_not_waited"
                chain["note"] = f"start_after_hours={hours}"
                return
            chain["waits"].append(hours)
            self.note_resets(s.advance_to(s.clock.now + float(hours) + 0.01))
            self.reroll(op, chain, argv, act, key, depth, in_batch)
            return
        hours = record.get("wait_hours")
        if (status == "blocked" and action != "wait_for_reset" and isinstance(hours, (int, float)) and 0 < hours <= float(act.get("can_wait_hours") or 0)
                and depth < 3):
            # The answer names the hour its window returns and the brief can wait that long (the session said so with
            # --can-wait-hours): a session that can wait waits and rolls again.
            chain["waits"].append(hours)
            self.note_resets(s.advance_to(s.clock.now + float(hours) + 0.01))
            self.reroll(op, chain, argv, act, key, depth, in_batch)
            return
        if status == "blocked" and record["blocked_by"]:
            holder = record["blocked_by"]
            holder_id = holder.get("dispatch_id") if isinstance(holder, dict) else holder
            deadline = time.time() + 60
            while time.time() < deadline and s.agent(str(holder_id)).get("status") in ("running", "initializing"):
                time.sleep(1)
            s.call(op, "dispatch", "--dispatch-action", "reconcile")
            chain["commands"] += 1
            chain["waits"].append("writer")
            if depth >= 3:
                chain["outcome"] = "handed_to_session"
                return
            self.reroll(op, chain, argv, act, key, depth, in_batch)
            return
        if record.get("dispatch_id") and status == "error":
            self.observe_after_error(op, chain, record, act)
            return
        chain["outcome"] = "handed_to_session"

    def inbox(self) -> list[dict]:
        folder = self.s.state / "inbox" / PARENT
        messages = []
        for path in sorted(folder.glob("*.json"), key=lambda p: int(p.stem)) if folder.is_dir() else []:
            try:
                messages.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return messages

    def observe_after_error(self, op, chain, record, act):
        """An error answer that names a dispatch and says the router goes on by itself (a relaunch, a retry): the session runs
        nothing and waits for the notification, so the run observes the dispatch the way the product does (reconcile) and
        keeps its final state and whatever reached the parent's mailbox."""
        s = self.s
        before = len(self.inbox())
        deadline = time.time() + 40
        got = {}
        while time.time() < deadline:
            s.call(op, "dispatch", "--dispatch-action", "reconcile")
            result = s.call(op, "dispatch", "--dispatch-action", "status", "--dispatch-id", record["dispatch_id"])
            got = parse(result.stdout) or {}
            if got.get("state") in TERMINAL or got.get("state") in ("launched", "started"):
                break
            time.sleep(2)
        mail = self.inbox()[before:]
        follow = self.answer(chain, got, ["dispatch", "--dispatch-action", "status"], None, None, record["brief"], kind="observe")
        follow["worker_state"] = got.get("state")
        follow["observed"] = {k: got.get(k) for k in ("state", "cause", "attempt_of", "attempts", "retry_at", "current_attempt") if got.get(k) is not None}
        follow["mail"] = [m.get("text", "")[:600] for m in mail]
        if got.get("state") in ("launched", "started", "completed") and got.get("dispatch_id") != record["dispatch_id"]:
            self.launched(op, chain, follow, act)
            chain["outcome"] = "relaunched"
        else:
            chain["outcome"] = "handed_to_session"
            chain["note"] = f"after the error answer the dispatch is {got.get('state')}; {len(mail)} message(s) reached the parent"

    def reroll(self, op, chain, argv, act, key, depth, in_batch):
        """Roll the same brief again: a batch member alone (per-brief flags are refused in a batch), else the same command."""
        if in_batch:
            day, index = chain["day"] - 1, chain["index"]
            argv = ["roll", "--task", self.text(key, day, index, act.get("worker"))]
        self.roll(op, chain, argv, {**act, "fault": None}, [key], depth + 1)

    def launched(self, op, chain, record, act):
        chain["outcome"] = "launched"
        record["carried_launch"] = True        # the answer that carried this worker (the report's launch tables read it)
        if act.get("label") and record.get("agent_id") and not chain.get("batch_of"):
            self.labels[act["label"]] = record["agent_id"]
        self.charge(chain, record)
        if act.get("worker") in ("error", "crash", "quota") and record.get("dispatch_id"):
            self.observe_failure(op, chain, record)

    def charge(self, chain, record):
        account, usd = record.get("account"), record.get("expected_usd")
        usage = self.s.usage_of(account) if account else None
        if usage is None or not usd:
            self.charges.append({"chain": chain["id"], "brief": record["brief"], "account": account, "model": record.get("model"), "at": self.s.clock.now,
                                 "window": None, "pct": None, "note": "not charged: no windows or no expected_usd"})
            return
        bundle = self.sheet["bundles"].get(record.get("bundle_id") or "") or {}
        draws = bundle.get("draws") or [w.id for w in usage.windows]
        for wid in draws:
            capacity = (self.sheet["capacity"].get(account) or {}).get(wid)
            pct = clock.cost_pct(1.0, usd, capacity)
            with self.s.clock.lock:
                done = clock.charge(usage, wid, pct, self.s.clock.now) if pct is not None else None
            self.charges.append({"chain": chain["id"], "brief": record["brief"], "level": record["level"], "account": account,
                                 "model": record.get("model"), "effort": record.get("effort"), "at": round(self.s.clock.now, 3), "window": wid,
                                 "usd": usd, "capacity_usd": capacity, "pct": None if pct is None else round(pct, 4),
                                 "applied": None if done is None else round(done.applied, 4), "used_after": None if done is None else round(done.used_after, 3),
                                 "note": None if pct is not None else "window has no capacity_usd"})

    def observe_failure(self, op, chain, record):
        """A worker that fails after launch: wait for the fixture worker to end, let the router observe it, keep what it says."""
        s = self.s
        deadline = time.time() + 30
        while time.time() < deadline and s.agent(record["dispatch_id"]).get("status") in ("running", "initializing", None):
            time.sleep(0.5)
        s.call(op, "dispatch", "--dispatch-action", "reconcile")
        result = s.call(op, "dispatch", "--dispatch-action", "status", "--dispatch-id", record["dispatch_id"])
        got = parse(result.stdout) or {}
        follow = self.answer(chain, got, ["dispatch", "--dispatch-action", "status"], result, None, record["brief"], kind="observe")
        follow["worker_state"] = got.get("state")
        follow["cause"] = got.get("cause") or got.get("failure_class") or follow.get("cause")
        follow["observed"] = {k: got.get(k) for k in ("state", "cause", "attempt_of", "attempts", "retry_at", "bundle", "account", "model") if got.get(k) is not None}


def said_hours(text) -> float | None:
    """The wait an answer's text names ("returns in 2.5h", "Wait 1.2h"), as a session reads it; None when it names none."""
    found = re.search(r"(?:returns in|[Ww]ait)\s+(\d+(?:\.\d+)?)\s*h\b", str(text or ""))
    return float(found.group(1)) if found else None


def shown_arg(value) -> str:
    text = str(value)
    if "\n" in text:
        found = re.search(r"EXPERIENCE-BRIEF:\s*([\w.-]+)", text)
        return f"<brief {found.group(1) if found else '?'}>"
    if text.endswith(".md"):
        return Path(text).name
    return text


# ---------------------------------------------------------------------------------------------------------------------
# running scenarios


def run_scenario(world: World, spec: dict, index: int, states: dict, corpus: dict, day: list, days: int, stop: threading.Event) -> dict:
    started = time.time()
    result = {"name": spec["name"], "spec": spec, "error": None}
    scenario = play = None
    try:
        scenario = Scenario(world, spec, index, states)
        play = Play(scenario, corpus, day, days, stop)
        play.prepare()
        play.play()
    except Exception:                       # a scenario that cannot finish is reported with what it reached
        result["error"] = traceback.format_exc()[-3000:]
    finally:
        if scenario is not None:
            result["processes_ended"] = scenario.end_processes()
            for token in [t for t, e in world.usage.accounts.items() if e["clock"] is scenario.clock]:
                world.usage.accounts.pop(token, None)
    if play is not None:
        result.update({"setup": play.setup, "answers": play.answers, "chains": play.chains, "charges": play.charges, "resets": play.resets,
                       "timeline": play.timeline, "exclusions": play.exclusions, "labels": play.labels,
                       "bundles": play.sheet.get("bundles"), "capacity": play.sheet.get("capacity"),
                       "end_hour": scenario.clock.now})
    result["seconds"] = round(time.time() - started, 1)
    print(f"  {spec['name']}: {len(result.get('answers') or [])} answers, {result['seconds']} s" + (" (error)" if result["error"] else ""), flush=True)
    return result


def default_out() -> Path:
    return Path(tempfile.gettempdir()) / "agentcompass-experience" / time.strftime("run-%Y%m%d-%H%M%S")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("run", "classify"), default="run")
    parser.add_argument("--scenario", action="append", help="Run only this scenario (repeatable; names from --list).")
    parser.add_argument("--out", type=Path, help="Run directory (outside the repository). Default: <system temp>/agentcompass-experience/run-<time>.")
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL, help=f"Scenarios at once (at most {MAX_PARALLEL}).")
    parser.add_argument("--days", type=int, help="Play this many working days in every scenario instead of each scenario's own.")
    parser.add_argument("--wheel", type=Path, help="Install this wheel instead of building one from the working tree.")
    parser.add_argument("--classifications", type=Path, help="Replay these real classifier answers (from `classify`) instead of the declared profiles.")
    parser.add_argument("--keep", action="store_true", help="Keep the sandbox directory for inspection.")
    parser.add_argument("--list", action="store_true", help="List the scenarios and the day's stream, then exit.")
    parser.add_argument("--spend-classifier-quota", action="store_true",
                        help="classify: ask the real classifier (it spends the owner's quota). Without it, classify prints its plan only.")
    args = parser.parse_args(argv)
    scenarios_doc, briefs_doc = load_json(DATA / "scenarios.json"), load_json(DATA / "briefs.json")
    corpus = brief_corpus(briefs_doc)
    if args.list:
        for spec in scenarios_doc["scenarios"]:
            accounts = ", ".join(f"{a['name']}({a.get('plan') or a['family']})" for a in spec["accounts"])
            print(f"{spec['name']:36s} {accounts}; usage {spec.get('usage', 'fresh')}; switches {spec.get('switches') or {}}; days {spec.get('days', 1)}")
        print(f"\nday: {len(briefs_doc['day'])} entries, corpus {len(corpus)} briefs")
        return 0
    out = (args.out or default_out()).resolve()
    if inside(out, ROOT):
        parser.error("the run directory must be outside the repository")
    if args.command == "classify":
        return classify(corpus, out, spend=args.spend_classifier_quota)
    names = sc.registry_clean()
    if names:
        print("refusing to run: Windows user variables the router reads exist in HKCU\\Environment: " + ", ".join(names), file=sys.stderr)
        return 2
    selected = [s for s in scenarios_doc["scenarios"] if not args.scenario or s["name"] in args.scenario]
    unknown = set(args.scenario or []) - {s["name"] for s in scenarios_doc["scenarios"]}
    if unknown:
        parser.error("unknown scenario(s): " + ", ".join(sorted(unknown)))
    replay = load_json(args.classifications) if args.classifications else None
    out.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="agentcompass-experience-")).resolve()
    started = time.time()
    stop = threading.Event()
    results: list[dict] = []
    meta = {"started": dt.datetime.now().isoformat(timespec="seconds"), "sandbox": str(base), "out": str(out),
            "classifier": "replayed real answers " + str(args.classifications) if replay else "declared profiles (no model tokens)",
            "scenarios": [s["name"] for s in selected], "day_entries": len(briefs_doc["day"]), "days_override": args.days}
    print(f"experience run -> {out}", flush=True)
    interrupt_on_break()
    try:
        sc.NETGUARD = base / "netguard"
        sc.NETGUARD.mkdir()
        cache = Path(tempfile.gettempdir()) / "agentcompass-experience-cache"
        fakecli = fake_cli_build(cache)
        venv, wheel = install_world(base, args.wheel.resolve() if args.wheel else None)
        meta["wheel"] = wheel.name
        sites = venv / ("Lib/site-packages" if os.name == "nt" else next((venv / "lib").glob("python*/site-packages")))
        with UsageServer() as usage, ReplayScorer(corpus, replay, briefs_doc.get("error_scopes")) as scorer:
            sc.write_netguard([sites, sc.NETGUARD], usage.url, [usage.server.server_address[1], scorer.port], base / "net-default.jsonl")
            world = World(base, venv, fakecli, usage, scorer, out)
            workers = max(1, min(MAX_PARALLEL, args.parallel))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_scenario, world, spec, i + 1, scenarios_doc["usage_states"], corpus, briefs_doc["day"],
                                       args.days or int(spec.get("days", 1)), stop) for i, spec in enumerate(selected)]
                try:
                    # Wait in short slices: a signal handler runs only between them (a blocked wait on Windows never runs it).
                    pending = set(futures)
                    while pending:
                        done, pending = concurrent.futures.wait(pending, timeout=1.0)
                        results.extend(future.result() for future in done)
                except KeyboardInterrupt:
                    stop.set()
                    meta["interrupted"] = True
                    for future in futures:
                        future.cancel()
                    # A scenario thread waits on its command; ending the sandbox's processes lets the pool close now.
                    for entry in sc.sandbox_processes(base):
                        subprocess.run(["taskkill", "/F", "/T", "/PID", entry.split(":")[0]], capture_output=True)
                    raise
            meta["scorer_requests"], meta["scorer_unknown"] = scorer.requests, scorer.unknown[:20]
            meta["scopes_answered"] = scorer.scopes_answered
            meta["usage_requests"] = usage.requests
    except KeyboardInterrupt:
        meta["interrupted"] = True
        stop.set()
        print("interrupted: ending every process of the sandbox", flush=True)
    finally:
        # Ending, removing and recording happen before rows.json is written, so the file says what was left behind;
        # a second interrupt meanwhile is ignored (signal handlers off), or the sandbox would stay without a word.
        signals_off()
        left = sc.sandbox_processes(base)
        for entry in left:
            subprocess.run(["taskkill", "/F", "/T", "/PID", entry.split(":")[0]], capture_output=True)
        meta["processes_at_end"] = left
        meta["processes_after_end"] = sc.sandbox_processes(base)
        if not args.keep:
            remove_tree(base)
        meta["sandbox_left"] = str(base) if base.exists() else None
        meta["seconds"] = round(time.time() - started, 1)
        order = {s["name"]: i for i, s in enumerate(selected)}
        results.sort(key=lambda r: order.get(r["name"], 0))
        rows = {"run": meta, "corpus": {k: {"profile": v["profile"], "source": v["source"]} for k, v in corpus.items()},
                "scenarios": results}
        (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
        if meta["sandbox_left"]:
            print(f"the sandbox is left at {base}" + (" (--keep)" if args.keep else ": delete it by hand"), flush=True)
    sys.path.insert(0, str(ROOT / "audit"))
    import experience_report
    report = experience_report.write(out)
    print(f"report: {report}")
    return 130 if meta.get("interrupted") else 0


def inside(path: Path, root: Path) -> bool:
    """`path` is `root` or under it, both resolved (a prefix test would take E:/repo-runs for E:/repo)."""
    return path.resolve().is_relative_to(root.resolve())


def interrupt_on_break() -> None:
    """Ctrl+Break and a termination request end the run like Ctrl+C: its `finally` ends the sandbox's processes, removes the
    sandbox and writes rows.json saying so."""
    import signal

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)


def signals_off() -> None:
    import signal
    for name in ("SIGINT", "SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)


def remove_tree(base: Path) -> None:
    """Delete the run's sandbox; git keeps its object files read-only, so clear that bit and retry a locked file."""
    def writable(func, path, _exc):
        os.chmod(path, 0o700)
        func(path)
    for _ in range(5):
        try:
            shutil.rmtree(base, **({"onexc": writable} if sys.version_info >= (3, 12) else {"onerror": writable}))
        except OSError:
            pass
        if not base.exists():
            return
        time.sleep(2)


def product_command() -> list[str]:
    """The installed `agentcompass` on PATH, else this checkout's entry."""
    entry = shutil.which("agentcompass")
    return [entry] if entry else [sys.executable, "-X", "utf8", str(ROOT / "agentcompass")]


def classify(corpus: dict, out: Path, spend: bool = False, command: list[str] | None = None) -> int:
    """Ask the real configured classifier once per brief (`agentcompass score --input`, the installed product on this
    machine's real home) and keep each answer whole, a split's `multi_task`/`task_count` included, for `--classifications`.

    It spends the owner's classifier quota, so it first prints what it would ask (how many briefs, which scorer, never a
    key) and asks nothing without `--spend-classifier-quota`. Every answer is written as it arrives; a run that stopped
    resumes with the briefs that have no answer yet."""
    command = command or product_command()
    target, failed_path = out / "classifications.json", out / "classifications-failed.json"
    answers = load_json(target) if target.is_file() else {}
    todo = [key for key in corpus if key not in answers]
    shown = subprocess.run([*command, "scorer", "--action", "show"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120)
    view = parse(shown.stdout) or {}
    chosen = view.get(view.get("scorer_type")) if isinstance(view.get(view.get("scorer_type")), dict) else {}
    scorer = (f"{view.get('scorer_type')} ({view.get('label')}): model {chosen.get('model')} at {chosen.get('base_url')}"
              if view.get("scorer_type") else f"unreadable (scorer --action show exited {shown.returncode})")
    print(f"classify: {len(todo)} of {len(corpus)} briefs to ask ({len(answers)} answered already in {target})\n"
          f"scorer: {scorer}")
    if not spend:
        print("nothing asked: this spends the owner's classifier quota; with their consent, run again with --spend-classifier-quota")
        return 0
    folder = out / "classify-input"
    folder.mkdir(parents=True, exist_ok=True)
    failures = {}
    for key in todo:
        path = folder / f"{key}.json"
        path.write_text(json.dumps({"task": {"objective": corpus[key]["text"]}}, ensure_ascii=False), encoding="utf-8", newline="\n")
        done = subprocess.run([*command, "score", "--input", str(path)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
        result = parse(done.stdout) or {}
        profile = result.get("task_profile")
        if isinstance(profile, dict) and profile.get("work_kind") and result.get("source") == "system_one":
            answers[key] = {**profile, "reason": "real classifier answer, replayed by the experience run"}
            write_json(target, answers)
            print(f"  {key}: {profile['work_kind']} L{profile.get('primary_level')}"
                  + (f" multi_task x{profile.get('task_count')}" if profile.get("multi_task") else ""), flush=True)
        else:
            failures[key] = {"exit": done.returncode, "source": result.get("source"), "error": result.get("error"),
                             "tail": (done.stdout + done.stderr)[-400:]}
            write_json(failed_path, failures)
            print(f"  {key}: no answer of the configured classifier (exit {done.returncode}, source {result.get('source')})", flush=True)
    if not failures and failed_path.is_file():
        failed_path.unlink()
    print(f"{len(answers)} answers -> {target}; {len(failures)} without one")
    return 0 if not failures else 1


def write_json(path: Path, document) -> None:
    """Replace `path` with `document` in one step, so an interrupted run keeps the previous complete file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
