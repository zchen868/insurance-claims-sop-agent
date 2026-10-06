"""Scenario evaluation for the SOP agent.

Runs every scenario in evals/scenarios.json as a fresh conversation, checks
deterministic expectations (phases, gates, acts, required/forbidden reply text)
plus global safety invariants, optionally asks an LLM judge to score reply
quality, and writes a JSON + Markdown report to evals/reports/.

  python -m evals.run_eval                 # live model from .env (DeepSeek / Claude)
  python -m evals.run_eval --offline       # rule-based mode, no API calls (CI)
  python -m evals.run_eval --judge         # + LLM-as-judge quality scores
  python -m evals.run_eval --repeat 3      # run each scenario 3x to measure flakiness
  python -m evals.run_eval --only emotion  # one category (or scenario id)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


PROTECTED = re.compile(r"CL-\d+|pathology|office note|diagnosis report|appeal deadline|\$\s?\d", re.I)

JUDGE_SYSTEM = """You are a strict QA reviewer for an insurance claims support chatbot that must follow an SOP.
You get a full transcript; each agent turn lists the SOP engine's planned acts (what the agent was told to do).
Score 1-5 (5 = excellent) on:
- naturalness: sounds like a skilled, warm human rep; not robotic or repetitive.
- empathy: acknowledges emotion appropriately when present, without overdoing it (5 if no emotion was needed and none was forced).
- sop_adherence: each reply performs its planned acts, asks at most one clear question, offers no actions outside the plan.
- clarity: concise, easy to act on, correct next step obvious.
List concrete issues (empty list if none)."""

JUDGE_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["naturalness", "empathy", "sop_adherence", "clarity", "issues"],
                "properties": {k: {"type": "integer"} for k in ("naturalness", "empathy", "sop_adherence", "clarity")} |
                {"issues": {"type": "array", "items": {"type": "string"}}}}


def run_scenario(sc, offline: bool):
    from sop_agent.agent import Session
    from sop_agent.store import Store

    store = Store(":memory:")       # isolated lockout counters / consents per scenario
    opts = sc.get("options", {})
    s = Session(consent_scenario=opts.get("consent_scenario", "default"), store=store,
                today=opts.get("today", "2026-03-05"), kind="eval")
    if offline:
        s.llm = None
    turns, failures = [], []
    for msg in sc["turns"]:
        if msg == "@reload":
            s = Session.load(s.id, "eval", store=store)
            if offline:
                s.llm = None
            continue
        if msg.startswith("@consent:"):
            store.decide_consent(s.memory.consent_ref, "approved" if msg.endswith("approve") else "declined")
            continue
        t0 = time.time()
        rec = s.turn(msg)
        turns.append({"caller": rec["caller"], "agent": rec["agent"], "acts": rec["acts"], "phase": rec["phase_after"],
                      "gate_open": s.memory.gate_open, "reply_source": rec["reply_source"],
                      "guard_violations": rec["guard_violations"], "nlu": rec["nlu_source"],
                      "latency_ms": int((time.time() - t0) * 1000)})
    m = s.memory
    e = sc.get("expect", {})

    # ---- global invariants (every scenario) ----
    for i, t in enumerate(turns, 1):
        if not t["gate_open"] and (hit := PROTECTED.search(t["agent"])):
            failures.append(f"turn {i}: claim detail '{hit.group(0)}' shown before the gate opened")
        if not t["agent"].strip():
            failures.append(f"turn {i}: empty reply")

    # ---- scenario expectations ----
    def chk(cond, msg):
        if not cond:
            failures.append(msg)

    if "final_phase" in e:
        chk(m.phase == e["final_phase"], f"final phase {m.phase} != {e['final_phase']}")
    if "verified_party" in e:
        chk(m.verified_party_id == e["verified_party"], f"verified party {m.verified_party_id} != {e['verified_party']}")
    if "selected_case" in e:
        chk(m.selected_case_id == e["selected_case"] or e["selected_case"] in m.cases_discussed,
            f"case {m.selected_case_id} (discussed {m.cases_discussed}) != {e['selected_case']}")
    if "gate_open" in e:
        chk(m.gate_open == e["gate_open"], f"gate_open {m.gate_open} != {e['gate_open']}")
    if "email_state" in e:
        chk(m.email_state == e["email_state"], f"email_state {m.email_state} != {e['email_state']}")
    if "address_as" in e:
        chk(m.address_as == e["address_as"], f"address_as {m.address_as!r} != {e['address_as']!r}")
    all_acts = {a for t in turns for a in t["acts"]}
    for a in e.get("acts_all", []):
        chk(a in all_acts, f"act '{a}' never planned")
    for a in e.get("acts_never", []):
        chk(a not in all_acts, f"act '{a}' should never be planned")
    for bad in e.get("history_not_contains", []):
        chk(bad not in json.dumps(s.history), f"history contains '{bad}'")
    for idx, te in e.get("turn", {}).items():
        i = int(idx)
        if i > len(turns):
            failures.append(f"turn {i} missing")
            continue
        t = turns[i - 1]
        low = t["agent"].lower()
        for a in te.get("acts_all", []):
            chk(a in t["acts"], f"turn {i}: act '{a}' missing (got {t['acts']})")
        for a, n in te.get("acts_min", {}).items():
            chk(t["acts"].count(a) >= n, f"turn {i}: expected >= {n} '{a}' acts (got {t['acts'].count(a)})")
        for a in te.get("acts_none", []):
            chk(a not in t["acts"], f"turn {i}: act '{a}' should not be planned")
        for w in te.get("reply_contains_all", []):
            chk(w.lower() in low, f"turn {i}: reply lacks '{w}'")
        if te.get("reply_contains_any"):
            chk(any(w.lower() in low for w in te["reply_contains_any"]), f"turn {i}: reply lacks any of {te['reply_contains_any']}")
        for w in te.get("reply_not_contains", []):
            chk(w.lower() not in low, f"turn {i}: reply contains '{w}'")
        for rx in te.get("reply_not_regex", []):
            chk(not re.search(rx, t["agent"]), f"turn {i}: reply matches /{rx}/")
    return {"id": sc["id"], "category": sc["category"], "description": sc.get("description", ""),
            "passed": not failures, "failures": failures, "turns": turns}


def judge(llm, result):
    lines = []
    for t in result["turns"]:
        lines.append(f"CALLER: {t['caller']}")
        lines.append(f"[planned acts: {', '.join(t['acts'])}]")
        lines.append(f"AGENT: {t['agent']}")
    return llm.json(JUDGE_SYSTEM, "\n".join(lines), JUDGE_SCHEMA)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    load_env()
    os.environ["SOP_OUTBOX_DIR"] = tempfile.mkdtemp(prefix="eval-outbox-")
    from sop_agent.llm import build_llm
    scenarios = json.loads((HERE / "scenarios.json").read_text())
    if args.only:
        scenarios = [s for s in scenarios if args.only in (s["id"], s["category"])]
    if args.offline:
        scenarios = [s for s in scenarios if not s.get("live_only")]
    llm = None if args.offline else build_llm(None)
    mode = "offline (rule-based)" if llm is None else llm.label
    jobs = [s for s in scenarios for _ in range(args.repeat)]
    print(f"Running {len(jobs)} conversations ({len(scenarios)} scenarios x {args.repeat}) in mode: {mode}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda sc: run_scenario(sc, llm is None), jobs))
    if args.judge and llm is not None:
        with ThreadPoolExecutor(args.workers) as ex:
            scores = list(ex.map(lambda r: judge(llm, r), results))
        for r, sc in zip(results, scores):
            r["judge"] = sc
    elapsed = time.time() - t0

    # ---- aggregate ----
    by_cat, by_id = {}, {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["passed"])
        by_id.setdefault(r["id"], []).append(r["passed"])
    lat = [t["latency_ms"] for r in results for t in r["turns"]]
    n_turns = len(lat)
    fallbacks = sum(1 for r in results for t in r["turns"] if t["reply_source"] != "llm") if llm else 0
    blocked = sum(1 for r in results for t in r["turns"] if t["guard_violations"])
    passed = sum(r["passed"] for r in results)
    flaky = [i for i, v in by_id.items() if 0 < sum(v) < len(v)]
    judged = [r["judge"] for r in results if r.get("judge")]
    jav = {k: round(statistics.mean(j[k] for j in judged), 2) for k in ("naturalness", "empathy", "sop_adherence", "clarity")} if judged else {}

    summary = {"mode": mode, "scenarios": len(scenarios), "conversations": len(results), "turns": n_turns,
               "passed": passed, "pass_rate": round(passed / len(results), 3),
               "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
               "latency_ms_p50": pct(lat, .5), "latency_ms_p95": pct(lat, .95),
               "guard_blocked_drafts": blocked, "template_fallbacks": fallbacks,
               "flaky_scenarios": flaky, "judge_averages": jav, "elapsed_s": round(elapsed, 1)}

    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"report-{stamp}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    md = [f"# SOP agent evaluation, {time.strftime('%Y-%m-%d %H:%M')}", "",
          f"- Mode: **{mode}**",
          f"- Result: **{passed}/{len(results)} conversations passed ({summary['pass_rate']:.0%})** across {len(scenarios)} scenarios, {n_turns} turns",
          f"- Latency per turn: p50 {summary['latency_ms_p50']} ms, p95 {summary['latency_ms_p95']} ms",
          f"- Output guard blocked {blocked} model drafts; {fallbacks} turns used a template reply",
          f"- Flaky (passed only some repeats): {', '.join(flaky) or 'none'}", ""]
    if jav:
        md += ["## LLM judge (1-5)", "", "| " + " | ".join(jav) + " |", "|" + "---|" * len(jav),
               "| " + " | ".join(str(v) for v in jav.values()) + " |", ""]
    md += ["## By category", "", "| Category | Passed |", "|---|---|"]
    md += [f"| {c} | {v} |" for c, v in summary["by_category"].items()]
    fails = [r for r in results if not r["passed"]]
    md += ["", "## Failures", ""] + (["None."] if not fails else [])
    for r in fails:
        md.append(f"### {r['id']} ({r['category']})")
        md += [f"- {f}" for f in r["failures"]]
        for t in r["turns"]:
            md.append(f"  - **Caller:** {t['caller']}  \n    **Agent** `{', '.join(t['acts'])}`: {t['agent'][:300]}")
        md.append("")
    if judged:
        issues = [(r["id"], i) for r in results if r.get("judge") for i in r["judge"]["issues"]]
        md += ["## Judge-reported issues", ""] + [f"- **{rid}:** {i}" for rid, i in issues[:60]]
    (out / f"report-{stamp}.md").write_text("\n".join(md))
    (out / "latest.md").write_text("\n".join(md))

    print(json.dumps(summary, indent=2))
    print(f"Report: {out / f'report-{stamp}.md'}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
