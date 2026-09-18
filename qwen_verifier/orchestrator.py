"""Safe plan and evidence helpers for the guided verification workflow."""
from __future__ import annotations
from typing import Any

PROFILES={
    "Quick Check":{"description":"Metadata, compatibility and functional smoke test.","steps":["Compatibility","Functional"]},
    "Standard Qualification":{"description":"Quick checks plus portability and repeatable performance.","steps":["Compatibility","Functional","Portability","Performance"]},
    "Production Readiness":{"description":"Standard checks plus explicit quality, security and operational gates.","steps":["Compatibility","Functional","Portability","Performance","Quality","Security","Production gates"]},
}

def workflow_status(state:dict[str,Any],profile:str)->list[dict[str,str]]:
    p=state.get("preflight") or {};f=(state.get("result") or {}).get("functional_test",{});q=state.get("quality_summary") or {}
    compatibility="PASSED" if p.get("decision")=="READY_TO_RUN" else ("BLOCKED" if p else "NOT_STARTED")
    functional="PASSED" if f.get("status")=="PASS" else ("FAILED" if f else "NOT_STARTED")
    quality="PASSED" if q.get("decision")=="PASS" else ("FAILED" if q else "NOT_STARTED")
    security_score=state.get("security_score");security="NOT_STARTED" if security_score is None else ("PASSED" if security_score>=80 else "REVIEW_REQUIRED")
    values={
        "Compatibility":compatibility,"Functional":functional,
        "Portability":"PASSED" if state.get("portability") else "NOT_STARTED","Performance":"PASSED" if state.get("benchmark") else "NOT_STARTED",
        "Quality":quality,"Security":security,
        "Production gates":"REVIEW_REQUIRED",
    }
    return [{"step":step,"status":values[step]} for step in PROFILES[profile]["steps"]]

def next_action(state:dict[str,Any],profile:str)->str:
    for row in workflow_status(state,profile):
        if row["status"] in {"NOT_STARTED","BLOCKED","FAILED","REVIEW_REQUIRED"}:return row["step"]
    return "Complete"
