"""Evidence-based executive rollup for model qualification."""
from __future__ import annotations
from typing import Any
import pandas as pd
from .qualification import scorecard

def executive_summary(state:dict[str,Any])->dict[str,Any]:
    gates,verdict=scorecard(state)
    status_score={"PASS":100,"READY_TO_RUN":100,"FUNCTIONAL":100,"ADOPT":100,"WARN":60,"REVIEW":50,"FAIL":0,"BLOCKED":0,"NOT_CHECKED":0}
    gates=gates.copy();gates["Readiness %"]=gates.Status.map(status_score).fillna(25).astype(int)
    p=state.get("preflight") or {}; f=(state.get("result") or {}).get("functional_test",{})
    q=state.get("quality_summary") or {}; b=state.get("benchmark") or {}; o=state.get("optimization") or {}
    completed=int((gates.Status!="NOT_CHECKED").sum())
    blocking_open=int(((gates.Blocking)&(gates.Status.isin(["FAIL","BLOCKED","NOT_CHECKED"]))).sum())
    highlights=[]
    if p.get("decision")=="READY_TO_RUN":highlights.append("Compatibility preflight passed with a resolved model revision.")
    if f.get("status")=="PASS":highlights.append("Task-aware functional inference produced a valid non-empty output.")
    if q.get("decision")=="PASS":highlights.append(f'Golden-set quality passed at {q.get("pass_rate_percent")}%.')
    if b:highlights.append(f'Performance evidence captured: p95 {b.get("latency_seconds",{}).get("p95","—")}s and {b.get("throughput",{}).get("requests_per_second","—")} req/s.')
    lowlights=[]
    if p and p.get("decision")!="READY_TO_RUN":lowlights.append("Compatibility has blocking or unresolved findings.")
    if q and q.get("decision")!="PASS":lowlights.append("Quality is below the configured acceptance threshold.")
    if o and o.get("optimization_verdict") not in {"ADOPT","PASS"}:lowlights.append("The tested optimization has not earned adoption.")
    gaps=[f'{r.Gate}: evidence is missing.' for _,r in gates[gates.Status=="NOT_CHECKED"].iterrows()]
    risks=[]
    if p.get("remote_code_detected"):risks.append(["Unreviewed model code","High","Medium","Review, pin, scan and approve repository code before execution."])
    if not q:risks.append(["Unknown task quality","High","High","Run a representative golden set with an explicit threshold."])
    if not b:risks.append(["Unknown production performance","Medium","High","Benchmark representative shapes, concurrency and percentile latency."])
    if (state.get("security_score") or 0)<80:risks.append(["Incomplete supply-chain review","High","Medium","Complete license, revision, artifact and remote-code controls."])
    risks.append(["Smoke-test overconfidence","High","Medium","Treat functional PASS as basic viability, not production qualification."])
    next_steps=[]
    if blocking_open:next_steps.append(["P0","Close every blocking compatibility, functional, quality and security gate.","Model owner"])
    if not q:next_steps.append(["P0","Define and run a representative golden-set quality gate.","ML/QA lead"])
    if not b:next_steps.append(["P1","Measure p50/p95/p99, throughput, memory and concurrency on target hardware.","Platform lead"])
    if not state.get("portability"):next_steps.append(["P1","Prove the pinned artifact and acceptance test on each target runtime.","ML platform"])
    next_steps.append(["P2","Add CI regression gates, reproducible evidence and rollback criteria.","Engineering lead"])
    return {
        "verdict":verdict,"gates":gates,"completed":completed,"total":len(gates),"blocking_open":blocking_open,
        "model":p.get("model_id","Not selected"),"revision":p.get("resolved_revision","Not resolved"),
        "quality":q.get("pass_rate_percent"),"p95":b.get("latency_seconds",{}).get("p95"),
        "rps":b.get("throughput",{}).get("requests_per_second"),"security":state.get("security_score"),
        "highlights":highlights or ["No validated highlights yet—run the lifecycle tests to create evidence."],
        "lowlights":lowlights or ["No measured lowlights; this does not mean risk is absent."],"gaps":gaps,
        "risks":pd.DataFrame(risks,columns=["Risk","Impact","Likelihood","Mitigation"]),
        "next_steps":pd.DataFrame(next_steps,columns=["Priority","Recommended next step","Owner"]),
    }
