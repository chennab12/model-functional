"""Durable, privacy-conscious verification run storage and comparison."""
from __future__ import annotations
import json,os,sqlite3,time,uuid
from pathlib import Path
from typing import Any
import pandas as pd

STATE_KEYS=("preflight","result","portability","benchmark","optimization","quality_summary","security_score","vllm_report")

def _path(path:str|None=None)->Path:
    return Path(path or os.getenv("RUN_STORE_PATH","/tmp/model_verifier_runs.db"))

def _connect(path:str|None=None):
    db=sqlite3.connect(_path(path),timeout=5)
    db.execute("CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,label TEXT,created REAL,model TEXT,revision TEXT,verdict TEXT,payload TEXT)")
    return db

def snapshot(state:dict[str,Any])->dict[str,Any]:
    return {key:state.get(key) for key in STATE_KEYS}

def save_run(state:dict[str,Any],label:str="",path:str|None=None)->str:
    data=snapshot(state);pre=data.get("preflight") or {};functional=(data.get("result") or {}).get("functional_test",{})
    run_id=f'run-{time.strftime("%Y%m%d-%H%M%S",time.gmtime())}-{uuid.uuid4().hex[:6]}'
    verdict=(data.get("result") or {}).get("verdict") or pre.get("decision") or "NOT_CHECKED"
    with _connect(path) as db:
        db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?)",(run_id,label.strip() or run_id,time.time(),pre.get("model_id"),pre.get("resolved_revision"),verdict,json.dumps(data,default=str)))
    return run_id

def list_runs(path:str|None=None,limit:int=100)->pd.DataFrame:
    with _connect(path) as db:
        rows=pd.read_sql_query("SELECT run_id,label,created,model,revision,verdict,payload FROM runs ORDER BY created DESC LIMIT ?",db,params=(limit,))
    if rows.empty:return rows
    rows["created_utc"]=pd.to_datetime(rows.created,unit="s",utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    def metric(payload,key):
        data=json.loads(payload);b=data.get("benchmark") or {};q=data.get("quality_summary") or {}
        values={"quality_percent":q.get("pass_rate_percent"),"p95_seconds":b.get("latency_seconds",{}).get("p95"),"requests_per_second":b.get("throughput",{}).get("requests_per_second"),"device":b.get("device"),"security_score":data.get("security_score")}
        return values[key]
    for key in ("quality_percent","p95_seconds","requests_per_second","device","security_score"):rows[key]=rows.payload.map(lambda value,k=key:metric(value,k))
    return rows

def compare_runs(rows:pd.DataFrame,run_ids:list[str])->pd.DataFrame:
    chosen=rows[rows.run_id.isin(run_ids)].copy()
    columns=["label","model","revision","verdict","device","quality_percent","p95_seconds","requests_per_second","security_score","created_utc"]
    return chosen[[c for c in columns if c in chosen]].set_index("label").T
