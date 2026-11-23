"""The report of an experience run (packaging/experience_run.py), for a person to read. Read-only over the run's rows.json.

    python audit/experience_report.py <run dir>

Writes `report.md` (for reading) and `summary.json` (every table of the report as data) next to `rows.json`. It gives no
verdicts: it counts what the main session met, where the work went, what it cost on the simulated clock and how the answers
felt, and lists rows a person should look at. The "questions" it lists (hard work on a weaker candidate, trivial work on a
dearer one, a model that got nothing) are read from the plan's own ranking and reason; whether they are wrong is the
reader's call.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

LONG_ANSWER_CHARS = 2500          # a short result longer than this is more than a session reads at a glance
SLOW_ROLL_SECONDS = 10.0
DIFFICULTIES = ("trivial", "simple", "moderate", "hard")
NOT_A_WAY_OUT = ("", None)


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
    """The answer that carried each launched worker: the last answer of every chain whose outcome is a launch."""
    by_index = {i: a for i, a in enumerate(answers)}
    out = []
    for chain in chains:
        if chain.get("outcome") in ("launched", "relaunched") and chain.get("answers"):
            out.append(by_index[chain["answers"][-1]])
    return out


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


def harder_on_weaker(answer: dict, margin: float = 0.25) -> dict | None:
    """Hard work launched on a candidate whose fit is below another eligible candidate's by more than `margin`."""
    ranking = answer.get("ranking") or []
    chosen = next((c for c in ranking if c.get("id") == answer.get("bundle_id")), None)
    if answer.get("difficulty") != "hard" or chosen is None or chosen.get("fit") is None:
        return None
    better = [c for c in ranking if c.get("fit") is not None and c["fit"] > chosen["fit"] + margin]
    if not better:
        return None
    best = max(better, key=lambda c: c["fit"])
    return {"chosen": chosen, "stronger": best, "gap": round(best["fit"] - chosen["fit"], 3)}


def trivial_on_dearer(answer: dict, ratio: float = 2.0) -> dict | None:
    """Trivial or simple work launched on a candidate at least `ratio` times the cost of another eligible candidate."""
    ranking = answer.get("ranking") or []
    chosen = next((c for c in ranking if c.get("id") == answer.get("bundle_id")), None)
    if answer.get("difficulty") not in ("trivial", "simple") or chosen is None or not chosen.get("usd"):
        return None
    cheaper = [c for c in ranking if c.get("usd") and c["usd"] * ratio <= chosen["usd"]]
    if not cheaper:
        return None
    cheapest = min(cheaper, key=lambda c: c["usd"])
    return {"chosen": chosen, "cheaper": cheapest, "ratio": round(chosen["usd"] / cheapest["usd"], 2)}


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


def run_outs(scenario: dict) -> list[dict]:
    """Charges that did not fit under 100 % (the window ran out with this launch), in time order."""
    return [c for c in scenario.get("charges") or [] if c.get("pct") is not None and c.get("applied") is not None and c["applied"] + 1e-9 < c["pct"]]


def after_run_out(scenario: dict, charge: dict, limit: int = 4) -> list[dict]:
    """The next answers after a window ran out: what happened to the work then."""
    later = [a for a in scenario["answers"] if a["at"] >= charge["at"] and a.get("kind") in ("roll", "next_argv")]
    started = next((i for i, a in enumerate(later) if a["chain"] == charge["chain"]), 0)
    return later[started + 1:started + 1 + limit]


def usd_by_difficulty(launched: list[dict]) -> dict:
    out = {}
    for level in DIFFICULTIES:
        values = [a["expected_usd"] for a in launched if a.get("difficulty") == level and isinstance(a.get("expected_usd"), (int, float))]
        if values:
            out[level] = {"n": len(values), "median": round(statistics.median(values), 2), "mean": round(statistics.mean(values), 2),
                          "max": round(max(values), 2), "sum": round(sum(values), 2)}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the summary (every table as data)


def summarize(rows: dict) -> dict:
    scenarios = []
    for scenario in rows.get("scenarios") or []:
        answers, chains = scenario.get("answers") or [], scenario.get("chains") or []
        launched = launches(answers, chains)
        read = [a for a in answers if a.get("kind") in ("roll", "next_argv")]
        top_level = [c for c in chains if "batch_of" not in c]
        worker_chains = [c for c in chains if c.get("outcome") in ("launched", "relaunched")]
        bundles = scenario.get("bundles") or {}
        got = Counter(a.get("bundle_id") for a in launched)
        available = sorted({f"{b['model']}@{b['account']}" for b in bundles.values() if not b.get("disabled")})
        models_got = Counter(f"{a.get('model')}@{a.get('account')}" for a in launched)
        nothing = []
        for label in available:
            if models_got.get(label):
                continue
            ids = [bid for bid, b in bundles.items() if f"{b['model']}@{b['account']}" == label]
            reasons, eligible, gaps = Counter(), 0, []
            for bid in ids:
                entry = (scenario.get("exclusions") or {}).get(bid) or {}
                reasons.update(entry.get("excluded") or {})
                eligible += entry.get("eligible", 0)
                gaps += entry.get("gaps") or []
            nothing.append({"model": label, "bundles": ids, "eligible_in_plans": eligible,
                            "median_gap_to_top": round(statistics.median(gaps), 4) if gaps else None,
                            "top_exclusions": reasons.most_common(3)})
        questions = {"hard_on_weaker": [], "trivial_on_dearer": []}
        for a in launched:
            q = harder_on_weaker(a)
            if q:
                questions["hard_on_weaker"].append({"answer": ref(scenario, a), "brief": a["brief"], "chosen": q["chosen"], "stronger": q["stronger"],
                                                    "gap": q["gap"], "reason": a.get("decision_reason")})
            q = trivial_on_dearer(a)
            if q:
                questions["trivial_on_dearer"].append({"answer": ref(scenario, a), "brief": a["brief"], "chosen": q["chosen"], "cheaper": q["cheaper"],
                                                       "ratio": q["ratio"], "reason": a.get("decision_reason")})
        roll_seconds = [a["seconds"] for a in read if isinstance(a.get("seconds"), (int, float))]
        resets = overnight(scenario)
        scenarios.append({
            "name": scenario["name"], "error": scenario.get("error"), "seconds": scenario.get("seconds"),
            "processes_ended": scenario.get("processes_ended") or [],
            "counts": answer_counts(answers), "chains": len(top_level), "outcomes": dict(Counter(c.get("outcome") for c in chains).most_common()),
            "workers": len(launched),
            "not_launched": [ref(scenario, a) | {"status": a.get("status"), "caller_action": a.get("caller_action"), "brief": a["brief"],
                                                 "reason": a.get("reason"), "cause": a.get("cause")}
                             for a in read if a.get("status") not in ("launched", "batch")],
            "dead_ends": [c["id"] for c in chains if is_dead_end(scenario, c)],
            "broken_promises": [c["id"] for c in chains if promise_broken(scenario, c)],
            "difficulty_x_model": share_table(launched, lambda a: a.get("difficulty"), model_label),
            "kind_x_model": share_table(launched, lambda a: a.get("work_kind"), model_label),
            "model_share": dict(Counter(model_label(a) for a in launched).most_common()),
            "bundles_won": dict(got.most_common()),
            "models_with_nothing": nothing, "questions": questions,
            "usd_by_difficulty": usd_by_difficulty(launched),
            "windows": {k: {"start": v[0][1] if v else None, "peak": max((u for _, u in v), default=None), "end": v[-1][1] if v else None}
                        for k, v in window_timelines(scenario).items()},
            "resets": resets, "unused_at_reset": round(sum(r["unused"] for r in resets if not r["overnight"]), 1),
            "unused_overnight": round(sum(r["unused"] for r in resets if r["overnight"]), 1),
            "run_outs": [{"charge": c, "next": [ref(scenario, a) | {"status": a.get("status"), "model": a.get("model")} for a in after_run_out(scenario, c)]}
                         for c in run_outs(scenario)],
            "deferrals": [{"chain": c["id"], "brief": c["briefs"], "waits": c.get("waits")} for c in chains if c.get("waits")],
            "roll_seconds": {"n": len(roll_seconds), "median": percentile(roll_seconds, 0.5), "p90": percentile(roll_seconds, 0.9),
                             "max": max(roll_seconds, default=None)},
            "commands_per_worker": dict(Counter(c.get("commands") for c in worker_chains if "batch_of" not in c).most_common()),
            "long_answers": [ref(scenario, a) | {"chars": a["stdout_chars"]} for a in read if (a.get("stdout_chars") or 0) > LONG_ANSWER_CHARS],
            "no_way_out": [ref(scenario, a) | {"status": a.get("status")} for a in read
                           if a.get("status") not in ("launched", "batch") and a.get("next_action") in NOT_A_WAY_OUT],
        })
    return {"run": rows.get("run") or {}, "scenarios": scenarios}


def masked(text, n=110) -> str:
    """A reason with its numbers masked, so the same fact with different figures counts as one."""
    return re.sub(r"\d+(?:\.\d+)?", "#", " ".join(str(text or "").split()))[:n]


def ref(scenario: dict, answer: dict) -> dict:
    return {"scenario": scenario["name"], "chain": answer.get("chain"), "step": answer.get("step")}


# ---------------------------------------------------------------------------------------------------------------------
# the things most worth a look


def look_first(rows: dict, summary: dict, limit: int = 10) -> list[dict]:
    """Candidates for a person's first look, each with the rows that show it, most widespread first. Counting only:
    which are problems is the reader's call."""
    items = []
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
    for s in summary["scenarios"]:
        for row in s["not_launched"]:
            if row["status"] == "split":
                continue
            key = f"{row['status']} / {row['caller_action']}"
            handed[key] += 1
            handed_rows[key].append(row)
    for key, n in handed.most_common(3):
        why = Counter(masked(r.get("reason") or r.get("cause")) for r in handed_rows[key]).most_common(2)
        items.append({"weight": 40 + n, "title": f"{n} answer(s) did not launch: {key}; most common reason: " + " | ".join(f"{w} ×{c}" for w, c in why),
                      "rows": [{k: v for k, v in r.items() if k not in ("reason", "cause")} for r in handed_rows[key][:8]]})
    nothing = Counter()
    nothing_rows = defaultdict(list)
    for s in summary["scenarios"]:
        for entry in s["models_with_nothing"]:
            model = entry["model"].split("@")[0]
            nothing[model] += 1
            nothing_rows[model].append(entry["top_exclusions"][0][0] if entry["top_exclusions"] else f"outranked (eligible in {entry['eligible_in_plans']} plans)")
    if nothing:
        items.append({"weight": 30 + len(nothing), "title": f"{len(nothing)} signed-in model(s) got no work in at least one scenario",
                      "rows": [{"model": m, "scenarios": n, "most common reason": Counter(nothing_rows[m]).most_common(1)[0][0]} for m, n in nothing.most_common(8)]})
    concentrated = []
    for s in summary["scenarios"]:
        total = sum(s["model_share"].values())
        if total >= 8:
            label, n = next(iter(s["model_share"].items()))
            levels = [d for d, cols in s["difficulty_x_model"].items() if cols.get(label)]
            if n / total >= 0.6 and len(levels) >= 3:
                concentrated.append({"scenario": s["name"], "model": label, "share": round(n / total, 2), "difficulties": sorted(levels)})
    if concentrated:
        items.append({"weight": 35 + len(concentrated), "title": f"in {len(concentrated)} scenario(s) one model took 60 % or more of the workers "
                      "across trivial to hard work", "rows": concentrated})
    weaker = [q | {"scenario": s["name"]} for s in summary["scenarios"] for q in s["questions"]["hard_on_weaker"]]
    if weaker:
        items.append({"weight": 25 + len(weaker), "title": f"{len(weaker)} hard brief(s) launched while an eligible candidate had a clearly higher fit",
                      "rows": [{"scenario": q["scenario"], **q["answer"], "chosen": q["chosen"]["id"], "stronger": q["stronger"]["id"], "gap": q["gap"]} for q in weaker[:8]]})
    dearer = [q | {"scenario": s["name"]} for s in summary["scenarios"] for q in s["questions"]["trivial_on_dearer"]]
    if dearer:
        items.append({"weight": 20 + len(dearer), "title": f"{len(dearer)} trivial/simple brief(s) launched at twice or more the cost of an eligible candidate",
                      "rows": [{"scenario": q["scenario"], **q["answer"], "chosen": q["chosen"]["id"], "cheaper": q["cheaper"]["id"], "ratio": q["ratio"]} for q in dearer[:8]]})
    outs = [(s["name"], r) for s in summary["scenarios"] for r in s["run_outs"]]
    if outs:
        items.append({"weight": 30 + len(outs), "title": f"{len(outs)} launch(es) charged more than the window had left (the window ran out)",
                      "rows": [{"scenario": n, "chain": r["charge"]["chain"], "window": f"{r['charge']['account']}/{r['charge']['window']}",
                                "pct": r["charge"]["pct"], "applied": r["charge"]["applied"]} for n, r in outs[:8]]})
    unused = [(s["name"], r) for s in summary["scenarios"] for r in s["resets"] if r["unused"] >= 80 and r["window"] not in ("five_hour",) and not r["overnight"]]
    if unused:
        items.append({"weight": 15 + len(unused), "title": f"{len(unused)} long window(s) reset with 80 % or more unused", "rows": [{"scenario": n, **r} for n, r in unused[:8]]})
    slow = [(s["name"], s["roll_seconds"]) for s in summary["scenarios"] if (s["roll_seconds"].get("max") or 0) > SLOW_ROLL_SECONDS]
    if slow:
        items.append({"weight": 10 + len(slow), "title": f"a roll took over {SLOW_ROLL_SECONDS:.0f} s in {len(slow)} scenario(s)",
                      "rows": [{"scenario": n, **v} for n, v in slow]})
    long_answers = [row for s in summary["scenarios"] for row in s["long_answers"]]
    if long_answers:
        items.append({"weight": 10 + len(long_answers) / 10, "title": f"{len(long_answers)} short result(s) longer than {LONG_ANSWER_CHARS} characters",
                      "rows": long_answers[:6]})
    items.sort(key=lambda i: -i["weight"])
    return items[:limit]


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
           "No verdicts: counts, tables and the rows behind them. The clock is simulated (packaging/experience_run.py says what it covers);"
           " usage is charged at launch with the router's own expected_usd over each drawn window's capacity.", "",
           f"Processes: {len(run.get('processes_at_end') or [])} found by command line at the end and ended, {len(run.get('processes_after_end') or [])} left after that; "
           f"per scenario {sum(len(s['processes_ended']) for s in summary['scenarios'])} ended when the scenario finished.", ""]
    out += ["## Ten things most worth a look", ""]
    for n, item in enumerate(items, 1):
        out.append(f"{n}. **{item['title']}**")
        for row in item["rows"][:8]:
            out.append(f"   - `{json.dumps(row, ensure_ascii=False, default=str)[:400]}`")
    out += ["", "## Across scenarios", ""]
    head = ["scenario", "chains", "workers", "not launched", "dead ends", "promises not kept", "unused at resets in the day / overnight (pp)", "roll s median/p90/max", "error"]
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
        for level, cols in s["difficulty_x_model"].items():
            for label, n in cols.items():
                shares[level][label.split("@")[0]] += n
    out += ["", "Difficulty against model:effort (all scenarios, accounts folded):", ""]
    out += matrix({k: dict(v) for k, v in shares.items()}, DIFFICULTIES)
    for s in summary["scenarios"]:
        out += render_scenario(by_name[s["name"]], s)
    return "\n".join(out) + "\n"


def render_scenario(scenario: dict, s: dict) -> list[str]:
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
        rows_ = []
        for row in s["not_launched"]:
            chain = chains.get(row["chain"]) or {}
            a = answers[chain["answers"][row["step"]]] if chain.get("answers") and row["step"] < len(chain["answers"]) else {}
            rows_.append([row["chain"], row["step"], a.get("brief"), f"{a.get('status')} / {a.get('caller_action')}", clip(a.get("cause"), 80),
                          clip(a.get("next_action"), 260), clip(a.get("reason"), 160), f"{chain.get('outcome')} {clip(chain.get('note'), 120)}"])
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
    if s["difficulty_x_model"]:
        out += matrix(s["difficulty_x_model"], DIFFICULTIES)
        out += ["", "Work kind against model:", ""]
        out += matrix(s["kind_x_model"])
    else:
        out.append("No worker was launched.")
    if s["models_with_nothing"]:
        out += ["", "Signed-in models that got nothing, with what excluded or outranked them (from the plans' rankings):", ""]
        out += md_table(["model@account", "eligible in plans", "median score gap to the top", "top exclusions"],
                        [[e["model"], e["eligible_in_plans"], e["median_gap_to_top"], "; ".join(f"{clip(r, 140)} ×{n}" for r, n in e["top_exclusions"])]
                         for e in s["models_with_nothing"]])
    if s["questions"]["hard_on_weaker"]:
        out += ["", "Questions: hard work launched while an eligible candidate had a clearly higher fit (the plan's reason given):", ""]
        out += md_table(["chain", "brief", "chosen (fit)", "higher fit", "plan's reason"],
                        [[q["answer"]["chain"], q["brief"], f"{q['chosen']['id']} ({q['chosen']['fit']})", f"{q['stronger']['id']} ({q['stronger']['fit']})", clip(q["reason"], 220)]
                         for q in s["questions"]["hard_on_weaker"]])
    if s["questions"]["trivial_on_dearer"]:
        out += ["", "Questions: trivial or simple work launched at twice or more the cost of an eligible candidate:", ""]
        out += md_table(["chain", "brief", "chosen ($)", "cheaper ($)", "plan's reason"],
                        [[q["answer"]["chain"], q["brief"], f"{q['chosen']['id']} ({q['chosen']['usd']})", f"{q['cheaper']['id']} ({q['cheaper']['usd']})", clip(q["reason"], 220)]
                         for q in s["questions"]["trivial_on_dearer"]])
    out += ["", "### What it cost", ""]
    timelines = window_timelines(scenario)
    if timelines:
        out += md_table(["window", "start", "peak", "end", "timeline (simulated hour, used %)"],
                        [[k, v["start"], v["peak"], v["end"], sparkline(timelines.get(k, []))] for k, v in s["windows"].items()])
    if s["resets"]:
        out += ["", "Resets (quota left unused at each):", ""]
        out += md_table(["hour", "window", "used before", "unused", "overnight"],
                        [[r["at"], f"{r['account']}/{r['window']}", r["used_before"], r["unused"], "yes" if r["overnight"] else ""] for r in s["resets"]])
    if s["run_outs"]:
        out += ["", "Moments a window ran out and the answers right after:", ""]
        for item in s["run_outs"]:
            c = item["charge"]
            nxt = "; ".join(f"chain {n['chain']} {n['status']} {n.get('model') or ''}" for n in item["next"]) or "nothing after"
            out.append(f"- h{c['at']} {c['account']}/{c['window']}: chain {c['chain']} ({c['brief']}, {c['model']}) needed {c['pct']}% with {c['applied']}% left → {nxt}")
    if s["deferrals"]:
        out += ["", "Deferrals and waits:", ""]
        out += md_table(["chain", "brief", "waits (hours, or writer)"], [[d["chain"], ", ".join(d["brief"]), d["waits"]] for d in s["deferrals"]])
    if s["usd_by_difficulty"]:
        out += ["", "Expected USD per dispatch by declared difficulty (the router's expected_usd):", ""]
        out += md_table(["difficulty", "n", "median", "mean", "max", "sum"], [[k, v["n"], v["median"], v["mean"], v["max"], v["sum"]] for k, v in s["usd_by_difficulty"].items()])
    out += ["", "### How it felt", "",
            f"Roll time (s): median {s['roll_seconds']['median']}, p90 {s['roll_seconds']['p90']}, max {s['roll_seconds']['max']} over {s['roll_seconds']['n']} answers. "
            f"Commands a session ran per worker: {s['commands_per_worker']}."]
    if s["long_answers"]:
        out.append(f"Short results over {LONG_ANSWER_CHARS} characters: " + ", ".join(f"chain {r['chain']} ({r['chars']})" for r in s["long_answers"][:12]))
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
