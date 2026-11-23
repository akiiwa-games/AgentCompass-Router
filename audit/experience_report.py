"""The report of an experience run (packaging/experience_run.py), for a person to read. Read-only over the run's rows.json.

    python audit/experience_report.py <run dir>

Writes `report.md` (for reading) and `summary.json` (every table of the report as data) next to `rows.json`. It gives no
verdicts: it counts what the main session met, where the work went, what it cost on the simulated clock and how the answers
felt, and lists rows a person should look at. Every number is read from the router's own fields as the run recorded them
(the plan's ranking snapshot, its reason, its profile, its expected_usd); where the router recorded nothing, the report says
"not recorded" instead of computing a stand-in. The "questions" it lists (hard work on a weaker candidate, routine work on
a dearer one, a model that got nothing) are read from the plan's own ranking and reason; whether they are wrong is the
reader's call. Every threshold is in THRESHOLDS and printed at the top of the report.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Every threshold the report uses, with what it means; render prints them all at the top.
THRESHOLDS = {
    "hard_level": (3, "the primary_level of the router's profile that counts as hard work in the higher-fit question"),
    "hard_fit_margin": (0.25, "a candidate's fit (the router's components.fit) above the chosen one's by more than this is clearly higher"),
    "routine_level": (1, "the primary_level that counts as routine work in the dearer-candidate question"),
    "dearer_cost_ratio": (2.0, "the chosen candidate's opportunity_cost at least this many times another eligible candidate's "
                               "(or any cost above a zero one) is dearer"),
    "concentration_min_workers": (8, "a scenario needs at least this many workers before one model's share is listed"),
    "concentration_share": (0.6, "one model's share of a scenario's workers (accounts and efforts folded) from which it is listed"),
    "unused_at_reset_pp": (80, "a long window that resets during the day with at least this many percentage points unused is listed"),
    "long_answer_chars": (2500, "a short result longer than this is more than a session reads at a glance"),
    "slow_roll_seconds": (10.0, "a roll that took longer than this is listed"),
    "not_launched_kinds": (3, "how many kinds of answer that did not launch (status / caller_action) the first look lists"),
    "rows_per_item": (8, "rows shown under each item of the first look"),
}


def limit(name: str):
    return THRESHOLDS[name][0]


LEVELS = ("L1", "L2", "L3")
NOT_A_WAY_OUT = ("", None)
ROOT = Path(__file__).resolve().parents[1]
ESTIMATED = ("expected_usd charged at launch over each drawn window's capacity, on the simulated clock; not measured "
             "provider use")


def fixture_error_texts() -> list[str]:
    """The worker error texts the fixture writes and its classifier answers a scope for (briefs.json error_scopes)."""
    doc = json.loads((ROOT / "packaging" / "experience" / "briefs.json").read_text(encoding="utf-8"))
    return list(doc.get("error_scopes") or {})


def snapshot_caps() -> tuple[int, int]:
    """(eligible, excluded): the most candidates of each kind a plan's ledger row keeps, from the product's ledger_snapshot."""
    scripts = str(ROOT / "skills" / "agent-compass" / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import ledger_snapshot
    return ledger_snapshot.SNAPSHOT_FULL + ledger_snapshot.SNAPSHOT_COMPACT, ledger_snapshot.SNAPSHOT_EXCLUDED


def fixture_clis(run: dict) -> tuple[list[str], str]:
    """The provider CLIs the run's fixture installed, and where that was read: the run's rows, or (rows written before they
    recorded it) the experience run's own FIXTURE_CLIS."""
    if run.get("fixture_clis") is not None:
        return list(run["fixture_clis"]), "the run's rows"
    packaging = str(ROOT / "packaging")
    if packaging not in sys.path:
        sys.path.insert(0, packaging)
    import experience_run
    return list(experience_run.FIXTURE_CLIS), "packaging/experience_run.py FIXTURE_CLIS (these rows predate the record)"


def fixture_errors(run: dict) -> str:
    """What is the fixture's in a worker error the router acted on: the error, and the scope answer that decided how the
    chain ended (relaunch, or end at once) and whether a hold followed."""
    scopes = run.get("scopes_answered")
    told = ", ".join(f"{k} ×{v}" for k, v in sorted(scopes.items())) if scopes else "not recorded"
    return ("the worker errors here are the fixture's, and so is the scope answer that decided how each chain ended "
            f"(briefs.json error_scopes; answered this run: {told})")


def ranking_cut(scenario: dict) -> dict:
    """How many of a scenario's plans kept as many candidates as their ledger row holds (ledger_snapshot caps each row), so
    a bundle in no recorded ranking may have been cut there rather than absent. Each plan's own count when the rows carry
    it (`snapshot_rows`); for rows written before that, the excluded count is derived from the per-bundle tallies (excluded
    rows recorded in all, against the cap times the plans) and the eligible count from each plan's ranking."""
    eligible_cap, excluded_cap = snapshot_caps()
    answers = scenario.get("answers") or []
    counted = [a["snapshot_rows"] for a in answers if a.get("snapshot_rows")]
    if counted:
        return {"plans": len(counted), "excluded_at_cap": sum(1 for c in counted if c["excluded"] >= excluded_cap),
                "eligible_at_cap": sum(1 for c in counted if c["eligible"] >= eligible_cap), "bound": "exact",
                "excluded_cap": excluded_cap, "eligible_cap": eligible_cap, "source": "each plan's own count (snapshot_rows)"}
    plans = [a for a in answers if a.get("ledger_stage") == "plan" and not a.get("batch")]
    excluded = sum(e.get("plans", 0) - e.get("eligible", 0) for e in (scenario.get("exclusions") or {}).values())
    n = len(plans)
    if n and excluded == excluded_cap * n:
        at_cap, bound = n, "exact"
    elif n and excluded <= excluded_cap * n:
        at_cap, bound = max(0, excluded - (excluded_cap - 1) * n), "at least"
    else:
        at_cap, bound = None, "unknown"
    return {"plans": n, "excluded_at_cap": at_cap, "eligible_at_cap": sum(1 for a in plans if len(a.get("ranking") or []) >= eligible_cap),
            "bound": bound, "excluded_cap": excluded_cap, "eligible_cap": eligible_cap,
            "source": f"derived from these rows: {excluded} excluded rows recorded over {n} plan(s), each row holding at most {excluded_cap}"}


def not_in_ranking(cut: dict) -> tuple[str, str, str]:
    """(recorded, note, short note): a bundle in no plan's recorded ranking, and what that means in this scenario against
    the ledger's caps: possibly cut there, or absent. The short note has no per-scenario figures, so the first look can
    group the same fact across scenarios."""
    if cut["excluded_at_cap"] is None:
        return ("not in any plan's recorded ranking", f"whether a plan's ledger row was cut at its cap of {cut['excluded_cap']} "
                f"excluded candidates cannot be told from these rows ({cut['source']})",
                "cut at the ledger's cap or absent: cannot be told from these rows")
    if cut["excluded_at_cap"] or cut["eligible_at_cap"]:
        bound = "" if cut["bound"] == "exact" else f"{cut['bound']} "
        return ("not in any plan's recorded ranking", f"may have been cut there: {bound}{cut['excluded_at_cap']} of {cut['plans']} "
                f"plan(s) kept the ledger's cap of {cut['excluded_cap']} excluded candidates and {cut['eligible_at_cap']} its cap of "
                f"{cut['eligible_cap']} eligible ones (ledger_snapshot; {cut['source']})",
                f"may have been cut: the scenario's plans reached the ledger's cap of {cut['excluded_cap']} excluded or "
                f"{cut['eligible_cap']} eligible candidates")
    return ("absent from every plan's recorded ranking", f"no plan reached the ledger's cap of {cut['excluded_cap']} excluded or "
            f"{cut['eligible_cap']} eligible candidates ({cut['source']})", "absent: no plan reached the ledger's caps")


def fixture_notes(said: str, provider: str | None, clis: list[str]) -> str | None:
    """What in a recorded reason is the fixture's, or None."""
    notes = []
    if "missing_client" in said and provider and provider not in clis:
        notes.append(f"the fixture installs no {provider} executable (only {', '.join(clis)}): this is the fixture's gap, "
                     "not a finding of discovery")
    if "provider failure hold" in said:
        notes.append("the hold follows a worker error the fixture made, and its scope was the fixture's answer (briefs.json error_scopes)")
    return "; ".join(notes) or None


# ---------------------------------------------------------------------------------------------------------------------
# counting


def answer_counts(answers: list[dict]) -> dict:
    """{"status": {status: n}, "status_action": {"status / caller_action": n}} over the answers a session read (roll and
    next_argv answers; the run's own observations are counted apart)."""
    read = [a for a in answers if a.get("kind") in ("roll", "next_argv")]
    return {"status": dict(Counter(str(a.get("status")) for a in read).most_common()),
            "status_action": dict(Counter(f"{a.get('status')} / {a.get('caller_action')}" for a in read).most_common()),
            "observations": dict(Counter(f"{a.get('kind')}: {a.get('worker_state') or a.get('status')}" for a in answers
                                         if a.get("kind") not in ("roll", "next_argv")).most_common())}


def launches(answers: list[dict], chains: list[dict]) -> list[dict]:
    """The answer that carried each launched worker: the one the run marked `carried_launch` in a chain whose outcome is a
    launch (an observation the run made after it is not that answer)."""
    out = []
    for chain in chains:
        if chain.get("outcome") not in ("launched", "relaunched"):
            continue
        carried = [answers[i] for i in chain.get("answers") or [] if answers[i].get("carried_launch")]
        if carried:
            out.append(carried[-1])
    return out


def level_label(answer: dict) -> str:
    """The router's primary_level of the brief's profile, as L1..L3."""
    return f"L{answer['level']}" if answer.get("level") else "(none)"


def share_table(rows: list[dict], row_key, col_key) -> dict:
    """{row: {col: n}} plus totals; keys are functions of a row."""
    table = defaultdict(Counter)
    for row in rows:
        table[row_key(row)][col_key(row)] += 1
    return {r: dict(c.most_common()) for r, c in table.items()}


def model_label(answer: dict) -> str:
    if not answer.get("model"):
        return "(none)"
    return f"{answer['model']}:{answer.get('effort') or '-'}@{answer.get('account')}"


def percentile(values: list[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def chain_answers(scenario: dict, chain: dict) -> list[dict]:
    return [scenario["answers"][i] for i in chain.get("answers") or []]


def is_dead_end(scenario: dict, chain: dict) -> bool:
    """A chain that reached no worker and ended on an answer the session cannot act on: the run had to stop following
    (dead_end, a deferral it could not wait out), or the last answer is an error or a block whose next_action is empty."""
    if chain.get("outcome") in ("dead_end", "deferred_not_waited"):
        return True
    if chain.get("outcome") != "handed_to_session":
        return False
    last = chain_answers(scenario, chain)[-1:] or [{}]
    return last[0].get("next_action") in NOT_A_WAY_OUT


def promise_broken(scenario: dict, chain: dict) -> bool:
    """The session was told the router goes on by itself ("relaunches ... by itself", "do not roll it again") and the
    observation found no worker running the brief afterwards."""
    answers = chain_answers(scenario, chain)
    told = any("by itself" in str(a.get("next_action") or "") or "do not roll it again" in str(a.get("next_action") or "") for a in answers
               if a.get("kind") in ("roll", "next_argv"))
    return told and chain.get("outcome") not in ("launched", "relaunched", "split", "batch")


# ---------------------------------------------------------------------------------------------------------------------
# questions for the reader (from the plan's own ranking)


def harder_on_weaker(answer: dict) -> dict | None:
    """Hard work (THRESHOLDS hard_level) launched on a candidate whose recorded fit is below another eligible candidate's by
    more than hard_fit_margin."""
    ranking = answer.get("ranking") or []
    chosen = next((c for c in ranking if c.get("id") == answer.get("bundle_id")), None)
    if answer.get("level") != limit("hard_level") or chosen is None or chosen.get("fit") is None:
        return None
    better = [c for c in ranking if c.get("fit") is not None and c["fit"] > chosen["fit"] + limit("hard_fit_margin")]
    if not better:
        return None
    best = max(better, key=lambda c: c["fit"])
    return {"chosen": chosen, "stronger": best, "gap": round(best["fit"] - chosen["fit"], 3)}


def cost_check(answer: dict) -> dict | None:
    """For routine work (THRESHOLDS routine_level): the chosen candidate's cost against the other eligible candidates', where
    cost is the router's own ranking cost, `components.opportunity_cost` (what the score N subtracts for the quota the work
    would use on that account's windows at that moment; not dollars). None for other work; `{"recorded": False}` when the
    snapshot did not record the chosen candidate's cost; else the cheapest other candidate with a recorded cost when it is
    below the chosen one's, and whether the chosen one is dearer by dearer_cost_ratio (any cost above a zero one is)."""
    if answer.get("level") != limit("routine_level"):
        return None
    ranking = answer.get("ranking") or []
    chosen = next((c for c in ranking if c.get("id") == answer.get("bundle_id")), None)
    if chosen is None or chosen.get("opportunity_cost") is None:
        return {"recorded": False}
    others = [c for c in ranking if c is not chosen and c.get("opportunity_cost") is not None]
    cheapest = min(others, key=lambda c: c["opportunity_cost"]) if others else None
    if cheapest is None or chosen["opportunity_cost"] <= cheapest["opportunity_cost"]:
        return {"recorded": True, "chosen": chosen, "cheaper": None, "dearer": False, "ratio": None}
    mine, low = chosen["opportunity_cost"], cheapest["opportunity_cost"]
    ratio = round(mine / low, 2) if low > 0 else None
    return {"recorded": True, "chosen": chosen, "cheaper": cheapest, "ratio": ratio,
            "dearer": low <= 0 or ratio >= limit("dearer_cost_ratio")}


# ---------------------------------------------------------------------------------------------------------------------
# cost on the simulated clock


def window_timelines(scenario: dict) -> dict:
    """{"account/window": [(hour, used)]} from the timeline readings, one point per reading."""
    series = defaultdict(list)
    for point in scenario.get("timeline") or []:
        for account, windows in (point.get("windows") or {}).items():
            for wid, state in windows.items():
                series[f"{account}/{wid}"].append((point["at"], state["used"]))
    return dict(series)


def overnight(scenario: dict) -> list[dict]:
    """Each reset with `overnight` set when it fell after the day's last act (the session was off, and the experience run
    skips the night), so a window that refilled unused while nobody worked is told apart from one the day left unused."""
    ends: dict[int, float] = {}
    for point in scenario.get("timeline") or ():
        day = int(point["at"] // 24)
        ends[day] = max(ends.get(day, point["at"]), point["at"])
    return [dict(r, overnight=r["at"] > ends.get(int(r["at"] // 24), float("inf"))) for r in scenario.get("resets") or ()]


# A window running out is not counted. The run charges a launch's expected_usd in one lump at launch, and the router keeps
# no draw of its own to compare it with: a reservation holds the full expected_usd, and the feasibility check
# (candidate_eval) prorates a short window's share before its reset inline. A count from the run's own charge would be a
# formula of the tool's, so the report shows the window readings only.


def usd_by_level(launched: list[dict]) -> dict:
    out = {}
    for level in LEVELS:
        values = [a["expected_usd"] for a in launched if level_label(a) == level and isinstance(a.get("expected_usd"), (int, float))]
        if values:
            out[level] = {"n": len(values), "median": round(statistics.median(values), 2), "mean": round(statistics.mean(values), 2),
                          "max": round(max(values), 2), "sum": round(sum(values), 2)}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the summary (every table as data)


def summarize(rows: dict) -> dict:
    scenarios = []
    clis = fixture_clis(rows.get("run") or {})[0]
    texts = fixture_error_texts()
    for scenario in rows.get("scenarios") or []:
        answers, chains = scenario.get("answers") or [], scenario.get("chains") or []
        launched = launches(answers, chains)
        read = [a for a in answers if a.get("kind") in ("roll", "next_argv")]
        top_level = [c for c in chains if "batch_of" not in c]
        worker_chains = [c for c in chains if c.get("outcome") in ("launched", "relaunched")]
        got = Counter(a.get("bundle_id") for a in launched)
        questions = {"hard_on_weaker": [], "routine_on_dearer": []}
        costs = Counter()
        for a in launched:
            q = harder_on_weaker(a)
            if q:
                questions["hard_on_weaker"].append({"answer": ref(scenario, a), "brief": a["brief"], "chosen": q["chosen"], "stronger": q["stronger"],
                                                    "gap": q["gap"], "reason": a.get("decision_reason")})
            q = cost_check(a)
            if q is None:
                continue
            costs["routine launches"] += 1
            costs["chosen cost not recorded" if not q["recorded"] else "dearer" if q["dearer"]
                  else "lowest recorded cost" if q["cheaper"] is None else "dearer, under the ratio"] += 1
            if q.get("dearer"):
                questions["routine_on_dearer"].append({"answer": ref(scenario, a), "brief": a["brief"], "chosen": q["chosen"], "cheaper": q["cheaper"],
                                                       "ratio": q["ratio"], "reason": a.get("decision_reason")})
        roll_seconds = [a["seconds"] for a in read if isinstance(a.get("seconds"), (int, float))]
        resets = overnight(scenario)
        scenarios.append({
            "name": scenario["name"], "error": scenario.get("error"), "seconds": scenario.get("seconds"),
            "processes_ended": scenario.get("processes_ended") or [],
            "counts": answer_counts(answers), "chains": len(top_level), "outcomes": dict(Counter(c.get("outcome") for c in chains).most_common()),
            "workers": len(launched),
            "not_launched": [ref(scenario, a) | {"status": a.get("status"), "caller_action": a.get("caller_action"), "brief": a["brief"],
                                                 "reason": a.get("reason"), "cause": a.get("cause"),
                                                 "fixture_error": any(t in json.dumps(a, ensure_ascii=False) for t in texts)}
                             for a in read if a.get("status") not in ("launched", "batch")],
            "dead_ends": [c["id"] for c in chains if is_dead_end(scenario, c)],
            "broken_promises": [c["id"] for c in chains if promise_broken(scenario, c)],
            "level_x_model": share_table(launched, level_label, model_label),
            "kind_x_model": share_table(launched, lambda a: a.get("work_kind"), model_label),
            "model_share": dict(Counter(model_label(a) for a in launched).most_common()),
            "bundles_won": dict(got.most_common()),
            "models_with_nothing": models_with_nothing(scenario, launched, clis), "ranking_cut": ranking_cut(scenario),
            "questions": questions, "routine_cost": dict(costs),
            "usd_by_level": usd_by_level(launched),
            "windows": {k: {"start": v[0][1] if v else None, "peak": max((u for _, u in v), default=None), "end": v[-1][1] if v else None}
                        for k, v in window_timelines(scenario).items()},
            "resets": resets, "unused_at_reset": round(sum(r["unused"] for r in resets if not r["overnight"]), 1),
            "unused_overnight": round(sum(r["unused"] for r in resets if r["overnight"]), 1),
            "deferrals": [{"chain": c["id"], "brief": c["briefs"], "waits": c.get("waits")} for c in chains if c.get("waits")],
            "roll_seconds": {"n": len(roll_seconds), "median": percentile(roll_seconds, 0.5), "p90": percentile(roll_seconds, 0.9),
                             "max": max(roll_seconds, default=None)},
            "commands_per_worker": dict(Counter(c.get("commands") for c in worker_chains if "batch_of" not in c).most_common()),
            "batch_workers": {"workers": len([c for c in worker_chains if "batch_of" in c]),
                              "batch_rolls": len({c["batch_of"] for c in worker_chains if "batch_of" in c}),
                              "follow_up_commands": sum(c.get("commands") or 0 for c in worker_chains if "batch_of" in c)},
            "long_answers": [ref(scenario, a) | {"chars": a["stdout_chars"]} for a in read if (a.get("stdout_chars") or 0) > limit("long_answer_chars")],
            "no_way_out": [ref(scenario, a) | {"status": a.get("status")} for a in read
                           if a.get("status") not in ("launched", "batch") and a.get("next_action") in NOT_A_WAY_OUT],
        })
    return {"run": rows.get("run") or {}, "scenarios": scenarios}


def models_with_nothing(scenario: dict, launched: list[dict], clis: list[str]) -> dict:
    """The signed-in models (accounts, efforts and harnesses folded) no worker of this scenario ran, once each. Models only on
    accounts the user left outside allowed_accounts, or only in bundles the user unchecked, are kept apart. For the others
    each bundle (its own harness) carries what its plans recorded about it: its exclusions in the plans' rankings, or how
    often it was eligible and never the top score; a bundle in no recorded ranking says whether the ledger's caps may have
    cut it (ranking_cut). What in a reason is the fixture's (a CLI it does not install, a hold from its error) is said beside
    it (fixture_notes)."""
    absent = None
    bundles = scenario.get("bundles") or {}
    outside = set(((scenario.get("spec") or {}).get("switches") or {}).get("outside_allowed") or ())
    got = {a.get("model") for a in launched}
    by_model = defaultdict(list)
    for bid, bundle in bundles.items():
        by_model[bundle["model"]].append((bid, bundle))
    out = {"models": [], "outside_allowed": [], "unchecked": []}
    for model, members in sorted(by_model.items()):
        if model in got:
            continue
        open_ = [(bid, b) for bid, b in members if b["account"] not in outside and not b.get("disabled")]
        if not open_:
            kind = "outside_allowed" if all(b["account"] in outside for _, b in members) else "unchecked"
            out[kind].append(model)
            continue
        rows = []
        for bid, bundle in open_:
            entry = (scenario.get("exclusions") or {}).get(bid) or {}
            gaps = entry.get("gaps") or []
            reasons = Counter(entry.get("excluded") or {}).most_common(2)
            if reasons:
                said = "; ".join(f"{r} ×{n}" for r, n in reasons)
                note = short = fixture_notes(said, bundle.get("provider"), clis)
            elif entry.get("eligible"):
                said, note, short = f"eligible in {entry['eligible']} plan(s), never the top score", None, None
            else:
                absent = absent or not_in_ranking(ranking_cut(scenario))
                said, note, short = absent
            rows.append({"bundle": bid, "provider": bundle.get("provider"), "account": bundle["account"], "plans": entry.get("plans", 0),
                         "eligible": entry.get("eligible", 0), "median_gap_to_top": round(statistics.median(gaps), 4) if gaps else None,
                         "recorded": said, "note": note, "note_short": short})
        out["models"].append({"model": model, "bundles": rows})
    return out


def masked(text, n=110) -> str:
    """A reason with its numbers masked, so the same fact with different figures counts as one."""
    return re.sub(r"\d+(?:\.\d+)?", "#", " ".join(str(text or "").split()))[:n]


def ref(scenario: dict, answer: dict) -> dict:
    return {"scenario": scenario["name"], "chain": answer.get("chain"), "step": answer.get("step")}


# ---------------------------------------------------------------------------------------------------------------------
# the things most worth a look


def look_first(rows: dict, summary: dict, most: int = 10) -> list[dict]:
    """Candidates for a person's first look, each with the rows that show it, most widespread first. Counting only:
    which are problems is the reader's call."""
    items = []
    shown = limit("rows_per_item")
    broken = [(s["name"], c) for s in summary["scenarios"] for c in s["broken_promises"]]
    if broken:
        items.append({"weight": 100 + len(broken), "title": f"{len(broken)} answer(s) told the session the router would go on by itself "
                      "(relaunch / retry, \"do not roll it again\") and the observed dispatch then ran no worker for the brief",
                      "rows": [{"scenario": n, "chain": c} for n, c in broken]})
    dead = [(s["name"], c) for s in summary["scenarios"] for c in s["dead_ends"]]
    if dead:
        items.append({"weight": 90 + len(dead), "title": f"{len(dead)} chain(s) ended without a worker on an answer the session cannot act on",
                      "rows": [{"scenario": n, "chain": c} for n, c in dead]})
    errors = [s["name"] for s in summary["scenarios"] if s.get("error")]
    if errors:
        items.append({"weight": 95, "title": f"the run could not finish {len(errors)} scenario(s): " + ", ".join(errors), "rows": [{"scenario": n} for n in errors]})
    handed = Counter()
    handed_rows = defaultdict(list)
    told = {(n, c) for n, c in broken}
    for s in summary["scenarios"]:
        for row in s["not_launched"]:
            # One headline per kind of event: an answer of a broken promise is counted there, not again here.
            if row["status"] == "split" or (s["name"], row["chain"]) in told:
                continue
            key = f"{row['status']} / {row['caller_action']}"
            handed[key] += 1
            handed_rows[key].append(row)
    for key, n in handed.most_common(limit("not_launched_kinds")):
        why = Counter(masked(r.get("reason") or r.get("cause")) for r in handed_rows[key]).most_common(2)
        fixture = ("; " + fixture_errors(rows.get("run") or {}) if any(r.get("fixture_error") for r in handed_rows[key]) else "")
        items.append({"weight": 40 + n, "title": f"{n} answer(s) did not launch: {key}; most common reason: "
                      + " | ".join(f"{w} ×{c}" for w, c in why) + fixture,
                      "rows": [{k: v for k, v in r.items() if k not in ("reason", "cause", "fixture_error")}
                               for r in handed_rows[key][:shown]]})
    nothing = defaultdict(set)
    nothing_said = defaultdict(Counter)
    apart = defaultdict(set)
    for s in summary["scenarios"]:
        for entry in s["models_with_nothing"]["models"]:
            nothing[entry["model"]].add(s["name"])
            for row in entry["bundles"]:
                note = f" [{row['note_short']}]" if row.get("note_short") else ""
                nothing_said[entry["model"]][f"{row['provider']}: {masked(row['recorded'], 140)}{note}"] += 1
        for kind in ("outside_allowed", "unchecked"):
            apart[kind].update(s["models_with_nothing"][kind])
    if nothing:
        items.append({"weight": 30 + len(nothing), "title": f"{len(nothing)} signed-in model(s) got no work in at least one scenario "
                      f"(not counted: {len(apart['outside_allowed'])} model(s) only on accounts outside allowed_accounts, "
                      f"{len(apart['unchecked'])} unchecked by the user)",
                      "rows": [{"model": m, "scenarios": len(names), "recorded, per harness": nothing_said[m].most_common(2)}
                               for m, names in sorted(nothing.items(), key=lambda kv: -len(kv[1]))[:shown]]})
    concentrated = []
    for s in summary["scenarios"]:
        by_model = Counter()
        levels = defaultdict(set)
        for level, cols in s["level_x_model"].items():
            for label, n in cols.items():
                model = label.split(":")[0]
                by_model[model] += n
                levels[model].add(level)
        total = sum(by_model.values())
        if total >= limit("concentration_min_workers") and by_model:
            model, n = by_model.most_common(1)[0]
            if n / total >= limit("concentration_share") and set(LEVELS) <= levels[model]:
                concentrated.append({"scenario": s["name"], "model": model, "share": round(n / total, 2), "workers": total,
                                     "models_that_worked": len(by_model)})
    if concentrated:
        items.append({"weight": 35 + len(concentrated), "title": f"in {len(concentrated)} scenario(s) one model (accounts and efforts folded) took "
                      f"{limit('concentration_share'):.0%} or more of the workers, on L1, L2 and L3 work alike", "rows": concentrated})
    weaker = [q | {"scenario": s["name"]} for s in summary["scenarios"] for q in s["questions"]["hard_on_weaker"]]
    if weaker:
        items.append({"weight": 25 + len(weaker), "title": f"{len(weaker)} hard brief(s) launched while an eligible candidate had a clearly higher fit",
                      "rows": [{"scenario": q["scenario"], **q["answer"], "chosen": q["chosen"]["id"], "stronger": q["stronger"]["id"], "gap": q["gap"]}
                               for q in weaker[:shown]]})
    dearer = [q | {"scenario": s["name"]} for s in summary["scenarios"] for q in s["questions"]["routine_on_dearer"]]
    costs = Counter()
    for s in summary["scenarios"]:
        costs.update(s["routine_cost"])
    if dearer:
        items.append({"weight": 20 + len(dearer), "title": f"{len(dearer)} of {costs['routine launches']} routine (L1) launch(es) went to a candidate whose "
                      f"opportunity_cost (the router's ranking cost) was {limit('dearer_cost_ratio'):g}x or more another eligible candidate's; "
                      f"{costs['lowest recorded cost']} went to the lowest recorded cost, {costs['chosen cost not recorded']} not recorded",
                      "rows": [{"scenario": q["scenario"], **q["answer"], "chosen": f"{q['chosen']['id']}@{q['chosen']['account']} ({q['chosen']['opportunity_cost']})",
                                "cheaper": f"{q['cheaper']['id']}@{q['cheaper']['account']} ({q['cheaper']['opportunity_cost']})", "ratio": q["ratio"]}
                               for q in dearer[:shown]]})
    unused = [(s["name"], r) for s in summary["scenarios"] for r in s["resets"]
              if r["unused"] >= limit("unused_at_reset_pp") and r["window"] not in ("five_hour",) and not r["overnight"]]
    if unused:
        items.append({"weight": 15 + len(unused), "title": f"{len(unused)} long window(s) reset during the day with {limit('unused_at_reset_pp')} pp or more unused "
                      f"({ESTIMATED})",
                      "rows": [{"scenario": n, **r} for n, r in unused[:shown]]})
    slow = [(s["name"], s["roll_seconds"]) for s in summary["scenarios"] if (s["roll_seconds"].get("max") or 0) > limit("slow_roll_seconds")]
    if slow:
        items.append({"weight": 10 + len(slow), "title": f"a roll took over {limit('slow_roll_seconds'):.0f} s in {len(slow)} scenario(s)",
                      "rows": [{"scenario": n, **v} for n, v in slow]})
    long_answers = [row for s in summary["scenarios"] for row in s["long_answers"]]
    if long_answers:
        items.append({"weight": 10 + len(long_answers) / 10, "title": f"{len(long_answers)} short result(s) longer than {limit('long_answer_chars')} characters",
                      "rows": long_answers[:shown]})
    items.sort(key=lambda i: -i["weight"])
    return items[:most]


# ---------------------------------------------------------------------------------------------------------------------
# Markdown


def md_table(header: list[str], rows: list[list]) -> list[str]:
    def cell(value):
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]


def clip(text, n=300) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def matrix(table: dict, row_order=None) -> list[str]:
    cols = sorted({c for row in table.values() for c in row})
    rows_ = [r for r in (row_order or sorted(table)) if r in table]
    lines = md_table(["", *cols, "total"], [[r, *[table[r].get(c, "") for c in cols], sum(table[r].values())] for r in rows_])
    totals = Counter()
    for r in rows_:
        totals.update(table[r])
    grand = sum(totals.values()) or 1
    lines.append("| share | " + " | ".join(f"{100 * totals[c] / grand:.0f}%" for c in cols) + f" | {grand} |")
    return lines


def sparkline(points: list, width: int = 14) -> str:
    if not points:
        return ""
    step = max(1, len(points) // width)
    picked = points[::step]
    if picked[-1] != points[-1]:
        picked.append(points[-1])
    return " → ".join(f"h{h:g} {u:.0f}%" for h, u in picked)


def render(rows: dict, summary: dict, items: list[dict]) -> str:
    run = rows.get("run") or {}
    by_name = {s["name"]: s for s in rows.get("scenarios") or []}
    out = ["# Experience run report", "",
           f"Run {run.get('started')} · {len(summary['scenarios'])} scenario(s) · {run.get('seconds')} s · wheel {run.get('wheel')} · classifier: {run.get('classifier')}.",
           "No verdicts: counts, tables and the rows behind them, read from the router's own recorded fields (\"not recorded\" where it recorded"
           " nothing). The clock is simulated (packaging/experience_run.py says what it covers); usage is charged at launch, in one lump, with the"
           " router's own expected_usd over each drawn window's capacity. The router prorates a short window's share before its reset and keeps no"
           " draw of its own, so a window running out is not counted: the window readings are shown as the run's charge left them.", "",
           "Thresholds: " + "; ".join(f"{name} = {value} ({meaning})" for name, (value, meaning) in THRESHOLDS.items()) + ".", "",
           f"Processes: {len(run.get('processes_at_end') or [])} found by command line at the end and ended, {len(run.get('processes_after_end') or [])} left after that; "
           f"per scenario {sum(len(s['processes_ended']) for s in summary['scenarios'])} ended when the scenario finished.", ""]
    out += ["## Ten things most worth a look", ""]
    for n, item in enumerate(items, 1):
        out.append(f"{n}. **{item['title']}**")
        for row in item["rows"][:limit("rows_per_item")]:
            out.append(f"   - `{json.dumps(row, ensure_ascii=False, default=str)[:400]}`")
    out += ["", "## Across scenarios", ""]
    head = ["scenario", "chains", "workers", "not launched", "dead ends", "promises not kept", "unused at resets in the day / overnight (pp, estimated)", "roll s median/p90/max", "error"]
    out += md_table(head, [[s["name"], s["chains"], s["workers"], len(s["not_launched"]), len(s["dead_ends"]), len(s["broken_promises"]), f'{s["unused_at_reset"]} / {s["unused_overnight"]}',
                            f"{s['roll_seconds']['median']}/{s['roll_seconds']['p90']}/{s['roll_seconds']['max']}", "yes" if s["error"] else ""]
                           for s in summary["scenarios"]])
    out += ["", "Answers a session read, by status / caller_action, all scenarios:", ""]
    total = Counter()
    for s in summary["scenarios"]:
        total.update(s["counts"]["status_action"])
    out += md_table(["status / caller_action", "n"], [[k, v] for k, v in total.most_common()])
    shares = defaultdict(Counter)
    for s in summary["scenarios"]:
        for level, cols in s["level_x_model"].items():
            for label, n in cols.items():
                shares[level][label.split("@")[0]] += n
    out += ["", "The router's primary_level against model:effort (all scenarios, accounts folded):", ""]
    out += matrix({k: dict(v) for k, v in shares.items()}, LEVELS)
    for s in summary["scenarios"]:
        out += render_scenario(by_name[s["name"]], s, run)
    return "\n".join(out) + "\n"


def render_scenario(scenario: dict, s: dict, run: dict) -> list[str]:
    spec = scenario.get("spec") or {}
    accounts = ", ".join(f"{a['name']} ({a.get('plan') or a['family']})" for a in spec.get("accounts") or [])
    out = ["", f"## {s['name']}", "",
           f"Accounts: {accounts}. Usage at start: {spec.get('usage', 'fresh')} {spec.get('usage_by_account') or ''}. "
           f"Switches: {spec.get('switches') or 'none'}. Days: {spec.get('days', 1)}. {s['chains']} chains, {s['workers']} workers, {s['seconds']} s.", ""]
    if s.get("error"):
        out += ["The run could not finish this scenario:", "", "```", s["error"][-1500:], "```", ""]
    out += ["### What the session met", ""]
    out += md_table(["status / caller_action", "n"], list(s["counts"]["status_action"].items()))
    if s["counts"]["observations"]:
        out += ["", "The run's own observations after an answer (not commands a session runs): "
                + ", ".join(f"{k} ×{v}" for k, v in s["counts"]["observations"].items())]
    answers = scenario.get("answers") or []
    chains = {c["id"]: c for c in scenario.get("chains") or []}
    if s["not_launched"]:
        out += ["", "Every answer that did not launch, and what following it led to:", ""]
        if any(row.get("fixture_error") for row in s["not_launched"]):
            out += [f"Rows marked [fixture error] end on a worker error the fixture made: {fixture_errors(run)}.", ""]
        rows_ = []
        for row in s["not_launched"]:
            chain = chains.get(row["chain"]) or {}
            a = answers[chain["answers"][row["step"]]] if chain.get("answers") and row["step"] < len(chain["answers"]) else {}
            rows_.append([row["chain"], row["step"], a.get("brief"), f"{a.get('status')} / {a.get('caller_action')}", clip(a.get("cause"), 80),
                          clip(a.get("next_action"), 260), clip(a.get("reason"), 160),
                          f"{chain.get('outcome')} {clip(chain.get('note'), 120)}" + (" [fixture error]" if row.get("fixture_error") else "")])
        out += md_table(["chain", "step", "brief", "answer", "cause", "next_action", "reason", "led to"], rows_)
    dead = s["dead_ends"] + [c for c in s["broken_promises"] if c not in s["dead_ends"]]
    if dead:
        out += ["", "Dead ends and promises not kept, in full:", ""]
        for cid in dead:
            chain = chains[cid]
            out.append(f"- chain {cid} ({', '.join(chain['briefs'])}; flags {chain.get('flags')}) → {chain.get('outcome')} {chain.get('note') or ''}")
            for i in chain.get("answers") or []:
                a = answers[i]
                out.append(f"  - [{a['kind']}] {a.get('status')} / {a.get('caller_action')} · cause {a.get('cause')} · next_action: {clip(a.get('next_action'), 500)}"
                           + (f" · observed {json.dumps(a.get('observed'), ensure_ascii=False)[:500]}" if a.get("observed") else "")
                           + (f" · mail {len(a.get('mail') or [])}" if "mail" in a else ""))
    out += ["", "### Where work went", ""]
    if s["level_x_model"]:
        out += ["The router's primary_level against model:effort@account:", ""]
        out += matrix(s["level_x_model"], LEVELS)
        out += ["", "Work kind against model:", ""]
        out += matrix(s["kind_x_model"])
    else:
        out.append("No worker was launched.")
    nothing = s["models_with_nothing"]
    if nothing["models"]:
        out += ["", "Signed-in models that got nothing, each bundle with what the plans' recorded rankings say about it:", ""]
        out += md_table(["model", "bundle", "harness", "account", "plans", "eligible", "median score gap to the top", "recorded"],
                        [[e["model"], r["bundle"], r["provider"], r["account"], r["plans"], r["eligible"], r["median_gap_to_top"],
                          clip(r["recorded"], 240) + (f" [{r['note']}]" if r.get("note") else "")]
                         for e in nothing["models"] for r in e["bundles"]])
    for kind, what in (("outside_allowed", "only on accounts outside allowed_accounts"), ("unchecked", "unchecked by the user")):
        if nothing[kind]:
            out.append(f"Models {what} (not counted above): {', '.join(nothing[kind])}.")
    if s["questions"]["hard_on_weaker"]:
        out += ["", "Questions: hard work launched while an eligible candidate had a clearly higher fit (the plan's reason given):", ""]
        out += md_table(["chain", "brief", "chosen (fit)", "higher fit", "plan's reason"],
                        [[q["answer"]["chain"], q["brief"], f"{q['chosen']['id']} ({q['chosen']['fit']})", f"{q['stronger']['id']} ({q['stronger']['fit']})", clip(q["reason"], 220)]
                         for q in s["questions"]["hard_on_weaker"]])
    if s["routine_cost"]:
        out += ["", "Routine (L1) launches by the router's ranking cost (components.opportunity_cost) of the chosen candidate against the other "
                "eligible ones: " + ", ".join(f"{k} {v}" for k, v in s["routine_cost"].items()) + "."]
    if s["questions"]["routine_on_dearer"]:
        out += ["", f"Questions: routine work launched on a candidate whose opportunity_cost was {limit('dearer_cost_ratio'):g}x or more another eligible one's:", ""]
        out += md_table(["chain", "brief", "chosen (account, opportunity_cost, score)", "cheaper (account, opportunity_cost, score)", "ratio", "plan's reason"],
                        [[q["answer"]["chain"], q["brief"], f"{q['chosen']['id']} ({q['chosen']['account']}, {q['chosen']['opportunity_cost']}, {q['chosen']['score']})",
                          f"{q['cheaper']['id']} ({q['cheaper']['account']}, {q['cheaper']['opportunity_cost']}, {q['cheaper']['score']})",
                          q["ratio"] if q["ratio"] is not None else "above zero", clip(q["reason"], 220)]
                         for q in s["questions"]["routine_on_dearer"]])
    out += ["", "### What it cost", ""]
    timelines = window_timelines(scenario)
    if timelines:
        out += [f"Window use is the router's own estimate ({ESTIMATED}).", ""]
        out += md_table(["window", "start", "peak", "end", "timeline (simulated hour, estimated used %)"],
                        [[k, v["start"], v["peak"], v["end"], sparkline(timelines.get(k, []))] for k, v in s["windows"].items()])
    if s["resets"]:
        out += ["", f"Resets (quota left unused at each, {ESTIMATED}):", ""]
        out += md_table(["hour", "window", "used before", "unused", "overnight"],
                        [[r["at"], f"{r['account']}/{r['window']}", r["used_before"], r["unused"], "yes" if r["overnight"] else ""] for r in s["resets"]])
    if s["deferrals"]:
        out += ["", "Deferrals and waits:", ""]
        out += md_table(["chain", "brief", "waits (hours, or writer)"], [[d["chain"], ", ".join(d["brief"]), d["waits"]] for d in s["deferrals"]])
    if s["usd_by_level"]:
        out += ["", "Expected USD per dispatch by the router's primary_level (the router's expected_usd: list price over the expected hours):", ""]
        out += md_table(["level", "n", "median", "mean", "max", "sum"], [[k, v["n"], v["median"], v["mean"], v["max"], v["sum"]] for k, v in s["usd_by_level"].items()])
    out += ["", "### How it felt", "",
            f"Roll time (s): median {s['roll_seconds']['median']}, p90 {s['roll_seconds']['p90']}, max {s['roll_seconds']['max']} over {s['roll_seconds']['n']} answers. "
            f"Commands a session ran per worker launched alone: {s['commands_per_worker']}; "
            f"{s['batch_workers']['workers']} worker(s) came from {s['batch_workers']['batch_rolls']} batch roll(s) (one command each), "
            f"with {s['batch_workers']['follow_up_commands']} follow-up command(s)."]
    if s["long_answers"]:
        out.append(f"Short results over {limit('long_answer_chars')} characters: " + ", ".join(f"chain {r['chain']} ({r['chars']})" for r in s["long_answers"][:12]))
    if s["no_way_out"]:
        out.append("Answers that did not launch and give no next_action: " + ", ".join(f"chain {r['chain']} step {r['step']} ({r['status']})" for r in s["no_way_out"]))
    return out


def write(out) -> Path:
    out = Path(out)
    rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    summary = summarize(rows)
    items = look_first(rows, summary)
    summary["look_first"] = items
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    (out / "report.md").write_text(render(rows, summary, items), encoding="utf-8", newline="\n")
    return out / "report.md"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python audit/experience_report.py <run dir>")
    print(write(sys.argv[1]))
