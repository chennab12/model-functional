"""Qualification utilities: quality gates, scorecards, diagnosis, security, TCO and reproducibility."""
from __future__ import annotations
import hashlib,io,json,os,platform,re,sys,time,zipfile
from pathlib import Path
from typing import Any
import pandas as pd

def quality_evaluate(df:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    required={"test_id","category","expected","observed","metric"}
    missing=required-set(df.columns)
    if missing:raise ValueError("Missing columns: "+", ".join(sorted(missing)))
    rows=[]
    for _,r in df.fillna("").iterrows():
        expected,observed,metric=str(r.expected),str(r.observed),str(r.metric).lower()
        if metric=="exact":passed=observed.strip()==expected.strip()
        elif metric=="contains":passed=expected.lower() in observed.lower()
        elif metric=="regex":
            try:passed=bool(re.search(expected,observed))
            except re.error:passed=False
        elif metric=="json_valid":
            try:json.loads(observed);passed=True
            except Exception:passed=False
        else:passed=False
        rows.append({**r.to_dict(),"passed":passed,"result":"✅ PASS" if passed else "❌ FAIL"})
    out=pd.DataFrame(rows); total=len(out); passed=int(out.passed.sum()) if total else 0
    summary={"tests":total,"passed":passed,"failed":total-passed,"pass_rate_percent":round(100*passed/total,2) if total else 0,"decision":"PASS" if total and passed==total else "FAIL"}
    return out,summary

def scorecard(state:dict[str,Any],quality_threshold:float=95.0)->tuple[pd.DataFrame,str]:
    p=state.get("preflight") or {};f=(state.get("result") or {}).get("functional_test",{});q=state.get("quality_summary") or {};b=state.get("benchmark") or {};o=state.get("optimization") or {}
    rows=[
        ["Compatibility",p.get("decision","NOT_CHECKED"),p.get("resolved_revision","—"),True],
        ["Functional",f.get("status","NOT_CHECKED"),f.get("output",{}).get("type","—"),True],
        ["Quality","PASS" if q.get("pass_rate_percent",0)>=quality_threshold else ("NOT_CHECKED" if not q else "FAIL"),f'{q.get("pass_rate_percent","—")}%',True],
        ["Portability","PASS" if state.get("portability") else "NOT_CHECKED",f'{(state.get("portability") or {}).get("ready_targets","—")} ready targets',False],
        ["Performance","PASS" if b else "NOT_CHECKED",f'{b.get("latency_seconds",{}).get("p95","—")}s p95',False],
        ["Optimization",o.get("optimization_verdict","NOT_CHECKED"),f'{o.get("comparison",{}).get("throughput_improvement_percent","—")}% throughput',False],
        ["Security","PASS" if state.get("security_score",0)>=80 else "NOT_CHECKED",f'{state.get("security_score","—")}/100',True],
    ]
    df=pd.DataFrame(rows,columns=["Gate","Status","Evidence","Blocking"])
    failed=df[(df.Blocking)&(df.Status.isin(["FAIL","BLOCKED","NOT_CHECKED"]))]
    verdict="QUALIFIED_FOR_POC" if failed.empty else "HOLD"
    return df,verdict

PATTERNS=[
("OutOfMemory|out of memory","Capacity","Reduce batch/context, use lower validated precision or a smaller model.","On B70 inspect 32 GB VRAM and XPU allocations; do not confuse system RAM with VRAM."),
("GatedRepo|401|403","Access","Accept model terms and provide a read token.","Same on B70; resolve access before copying artifacts to the system."),
("KeyError.*model|unsupported architecture","Compatibility","Upgrade Transformers or add a reviewed model adapter.","Verify the XPU PyTorch/Transformers combination supports every operator."),
("trust_remote_code","Security","Review and pin repository code before explicit opt-in.","Review custom code before it can execute on the B70 host."),
("xpu|XPU","Intel GPU","Check Intel driver, XPU-enabled PyTorch and torch.xpu availability.","Confirm B70 device visibility, driver and upstream PyTorch XPU build."),
("CUDA|cuda","Wrong backend","Install CUDA only for NVIDIA or choose the correct device.","B70 uses XPU, not CUDA; select xpu:0."),
("timeout|timed out","Reliability","Check download/server health, then increase a bounded timeout.","Separate B70 compile/warm-up time from steady-state inference."),
]
def diagnose(text:str)->list[dict[str,str]]:
    hits=[]
    for pattern,category,action,b70 in PATTERNS:
        if re.search(pattern,text,re.I):hits.append({"category":category,"recommended_action":action,"intel_b70_difference":b70})
    return hits or [{"category":"Unknown","recommended_action":"Capture full stack trace, versions, revision and minimal reproduction.","intel_b70_difference":"Also capture B70 driver, torch.xpu status and device properties."}]

def security_review(preflight:dict[str,Any])->tuple[pd.DataFrame,int,str]:
    pinned=bool(preflight.get("resolved_revision"));remote=bool(preflight.get("remote_code_detected"));license_ok=bool(preflight.get("license"));gated=bool(preflight.get("gated"))
    rows=[
        ["Pinned immutable revision",pinned,20,"Always pin the resolved commit before approval."],
        ["Recognized license metadata",license_ok,15,"Manually review commercial and redistribution terms."],
        ["No unreviewed remote code",not remote,25,"Review repository Python and require explicit approval."],
        ["Repository access resolved",not gated,10,"Record gated terms and approval if applicable."],
        ["Safetensors/standard weights",preflight.get("weight_size_gb",0)>0,15,"Prefer safetensors; inspect nonstandard artifacts."],
        ["Secrets excluded from reports",True,15,"Tokens remain memory-only and must never appear in exports."],
    ]
    score=sum(weight for _,passed,weight,_ in rows if passed)
    df=pd.DataFrame([[name,"✅ PASS" if passed else "⚠️ REVIEW",weight,action] for name,passed,weight,action in rows],columns=["Control","Result","Weight","Action"])
    return df,score,"PASS" if score>=80 else "REVIEW"

def capacity(users:int,rpm:float,latency:float,tokens_per_second:float,hourly_cost:float,power_watts:float)->dict[str,float|str]:
    required_rps=users*rpm/60;concurrency=required_rps*latency;capacity_rps=max(tokens_per_second/100,0.001);devices=max(1,int(-(-required_rps//capacity_rps)))
    tokens_hour=tokens_per_second*3600
    return {"required_requests_per_second":round(required_rps,3),"estimated_concurrency":round(concurrency,2),"assumed_device_requests_per_second":round(capacity_rps,3),"estimated_devices":devices,"cost_per_million_tokens":round(hourly_cost/max(tokens_hour,1)*1_000_000,2),"watt_hours_per_million_tokens":round(power_watts/max(tokens_hour,1)*1_000_000,2),"note":"Planning estimate only; replace the 100-token/request assumption with workload measurements."}

def reproducibility_manifest(state:dict[str,Any])->dict[str,Any]:
    p=state.get("preflight") or {};b=state.get("benchmark") or {}
    return {"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"model":p.get("model_id"),"revision":p.get("resolved_revision"),"task":p.get("detected_task"),"config":p.get("config_summary"),"system":p.get("system"),"benchmark_settings":b.get("settings"),"python":sys.version,"platform":platform.platform(),"reports_present":[k for k,v in state.items() if v is not None]}

def reproduction_zip(state:dict[str,Any])->bytes:
    manifest=reproducibility_manifest(state);buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(manifest,indent=2,default=str))
        for name,value in state.items():
            if value is not None:z.writestr(f"reports/{name}.json",json.dumps(value,indent=2,default=str))
        z.writestr("reproduce.sh","python run_verification.py OWNER/MODEL --revision COMMIT_SHA\n")
    return buf.getvalue()

def workflow_yaml()->str:
    return """name: model-qualification
on:
  workflow_dispatch:
  pull_request:
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip
      - run: pip install -r requirements.txt
      - run: python -m compileall -q app.py qwen_verifier tests
      - run: pytest -q
      - run: python run_verification.py Qwen/Qwen2.5-0.5B-Instruct --preflight-only
"""

