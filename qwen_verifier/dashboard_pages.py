"""Grouped pages for the model-verification product experience."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from . import GenericModelVerifier
from .advanced_tabs import quality_tab,scorecard_tab,render_qualification_workbench
from .insights_tabs import render_analytics,render_executive_summary,render_news
from .lifecycle import BenchmarkAgent,PortabilityAgent
from .orchestrator import PROFILES,next_action,workflow_status
from .run_store import compare_runs,list_runs,save_run,snapshot
from .ui_tabs import benchmark_tab,optimization_tab,portability_tab

STATUS_ICON={"PASSED":"✅","RUNNING":"🔵","REVIEW_REQUIRED":"⚠️","BLOCKED":"⛔","FAILED":"❌","NOT_STARTED":"⚪","READY_TO_RUN":"✅"}

def _show_preflight(p):
    decision=p.get("decision","BLOCKED");(st.success if decision=="READY_TO_RUN" else st.error)(f'{STATUS_ICON.get(decision,"⚠️")} Preflight: **{decision}**')
    c1,c2,c3,c4=st.columns(4);c1.metric("Task",p.get("detected_task") or "unknown");c2.metric("Weights",f'{p.get("weight_size_gb",0)} GB');c3.metric("Estimated RAM",f'{p.get("estimated_cpu_ram_needed_gb",0)} GB');c4.metric("Available RAM",f'{p.get("system",{}).get("ram_available_gb","?")} GB')
    rows=pd.DataFrame(p.get("checks",[]))
    if not rows.empty:
        rows["Result"]=rows.status.map({"PASS":"✅ PASS","WARN":"⚠️ WARN","BLOCK":"⛔ BLOCK"}).fillna(rows.status)
        st.dataframe(rows[["Result","name","detail","remediation"]],hide_index=True,use_container_width=True,column_config={"remediation":st.column_config.TextColumn("Recommended action",width="large")})

def _run_preflight(model_input,revision,token,trust_remote,current_key):
    agent=GenericModelVerifier(model_input,token or None,revision or None,trust_remote)
    st.session_state.preflight=agent.preflight();st.session_state.result=None;st.session_state.key=current_key

def _run_functional(model_input,revision,token,trust_remote):
    p=st.session_state.preflight;agent=GenericModelVerifier(model_input,token or None,revision or None,trust_remote)
    st.session_state.result=agent.run(p)

def render_overview():
    render_executive_summary()
    if not st.session_state.get("preflight"):
        st.info("👋 No verification run yet. Open **Verify model** in the navigation and start with the safe metadata preflight.")

def render_verify(model_input,revision,token,trust_remote,current_key,max_model_gb:float=10.0):
    st.header("🧭 Verify model")
    guided,manual=st.tabs(["Guided workflow","Manual controls"])
    with guided:
        profile=st.selectbox("Verification profile",list(PROFILES),index=1,help="Quick is a smoke test; Standard adds portability and performance; Production adds human-owned quality, security and operational gates.")
        st.caption(PROFILES[profile]["description"])
        status=pd.DataFrame(workflow_status(st.session_state,profile));status["State"]=status.status.map(lambda value:f'{STATUS_ICON.get(value,"⚠️")} {value.replace("_"," ").title()}')
        st.dataframe(status[["step","State"]].rename(columns={"step":"Step"}),hide_index=True,use_container_width=True)
        action=next_action(st.session_state,profile);st.info(f'**Recommended next action:** {action}')
        if not st.session_state.get("preflight"):
            st.write("Phase 1 reads only small Hub metadata and configuration. It does not download model weights.")
            if st.button("🔍 Start safe compatibility preflight",type="primary",use_container_width=True):
                try:
                    with st.spinner("Inspecting metadata, compatibility and resources…"):_run_preflight(model_input,revision,token,trust_remote,current_key)
                    st.rerun()
                except Exception as exc:st.error(f'{type(exc).__name__}: {exc}')
        p=st.session_state.get("preflight")
        if p:
            _show_preflight(p)
            oversize=float(p.get("weight_size_gb") or 0)>float(max_model_gb)
            if oversize:st.error(f'⛔ Blocked: reported weights are {p.get("weight_size_gb")} GB, above your {max_model_gb} GB execution guardrail.')
            elif not p.get("compatible"):st.error("Execution is blocked. Resolve the findings above and rerun preflight.")
            elif p.get("remote_code_detected") and not trust_remote:st.error("Execution remains blocked until repository code is reviewed and remote-code execution is explicitly allowed.")
            elif not st.session_state.get("result"):
                st.warning("Phase 2 downloads and executes model artifacts. Functional PASS proves basic viability—not production readiness.")
                approve=st.checkbox("I approve the model download and bounded code execution",key="guided_execution_approval")
                run_perf=st.checkbox("Also run capped CPU/device benchmark after the smoke test",value=profile!="Quick Check",disabled=profile=="Quick Check")
                if st.button("▶️ Continue with approved execution",type="primary",disabled=not approve or oversize,use_container_width=True):
                    with st.status("Running the approved workflow…",expanded=True) as box:
                        try:
                            _run_functional(model_input,revision,token,trust_remote)
                            if profile!="Quick Check":st.session_state.portability=PortabilityAgent(p).run()
                            if run_perf:
                                bench=BenchmarkAgent(p["model_id"],p["detected_task"],p.get("resolved_revision"),token or None,trust_remote)
                                st.session_state.benchmark=bench.run("cpu",1,3,32,1)
                            box.update(label="Approved workflow completed",state="complete")
                        except Exception as exc:box.update(label="Workflow stopped safely",state="error");st.exception(exc)
                    st.rerun()
        result=st.session_state.get("result")
        if result:
            ft=result.get("functional_test",{});(st.success if ft.get("status")=="PASS" else st.error)(f'Functional result: **{ft.get("status","UNKNOWN")}**')
            c1,c2,c3=st.columns(3);c1.metric("Task",ft.get("task","—"));c2.metric("Load",f'{ft.get("model_load_seconds","—")} s');c3.metric("Inference",f'{ft.get("inference_seconds","—")} s')
            if ft.get("output"):st.code(ft["output"].get("preview",""))
            if profile=="Production Readiness":st.warning("Automation stops short of declaring production readiness. Complete representative Quality, Security, reliability, cost and ownership gates under Production readiness.")
    with manual:
        st.subheader("Compatibility evidence")
        if st.button("Rerun compatibility preflight"):
            try:_run_preflight(model_input,revision,token,trust_remote,current_key);st.rerun()
            except Exception as exc:st.exception(exc)
        if st.session_state.get("preflight"):_show_preflight(st.session_state.preflight)
        else:st.info("No preflight evidence yet.")
        if st.session_state.get("preflight"):st.download_button("⬇️ Preflight JSON",json.dumps(st.session_state.preflight,indent=2,default=str),"hf_preflight.json","application/json")
        if st.session_state.get("result"):st.download_button("⬇️ Functional evidence",json.dumps(st.session_state.result,indent=2,default=str),"hf_functional_report.json","application/json")

def render_evaluate():
    st.header("🎯 Evaluate quality")
    tabs=st.tabs(["Golden set","Decision scorecard"])
    with tabs[0]:quality_tab()
    with tabs[1]:scorecard_tab()

def render_performance(token,trust_remote):
    st.header("⚡ Performance & portability")
    p,b,o=st.tabs(["Portability","Benchmark","Optimize"])
    with p:portability_tab(token,trust_remote)
    with b:benchmark_tab(token,trust_remote)
    with o:optimization_tab(token,trust_remote)

def render_production():
    st.header("🏭 Production readiness")
    st.info("Use this workbench after basic functional verification. Every missing blocking gate keeps the model on HOLD.")
    render_qualification_workbench()

def render_reports(root:Path):
    st.header("📁 Runs, comparisons & evidence")
    save,history,baseline=st.tabs(["Save current run","Compare runs","Included baseline"])
    with save:
        label=st.text_input("Run label",placeholder="Qwen CPU baseline")
        if st.button("💾 Save current evidence",type="primary"):
            try:st.success(f'Saved {save_run(st.session_state,label)}')
            except Exception as exc:st.error(f'Could not save run: {exc}')
        st.download_button("⬇️ Current evidence JSON",json.dumps(snapshot(st.session_state),indent=2,default=str),"verification_evidence.json","application/json")
        st.caption("Set RUN_STORE_PATH to a persistent mounted volume or managed database path in production. The default /tmp database can be ephemeral on Community Cloud.")
    with history:
        try:runs=list_runs()
        except Exception as exc:st.error(f'Could not read saved runs: {exc}');runs=pd.DataFrame()
        if runs.empty:st.info("No saved runs yet.")
        else:
            visible=["label","model","revision","verdict","device","quality_percent","p95_seconds","requests_per_second","security_score","created_utc"]
            st.dataframe(runs[visible],hide_index=True,use_container_width=True)
            selected=st.multiselect("Select two or more runs to compare",runs.run_id,format_func=lambda rid:runs.loc[runs.run_id==rid,"label"].iloc[0],max_selections=4)
            if len(selected)>=2:st.dataframe(compare_runs(runs,selected),use_container_width=True)
    with baseline:
        evidence=json.loads((root/"artifacts/latest_verification.json").read_text());st.success("Included real Qwen2.5-0.5B-Instruct baseline evidence.");st.json(evidence,expanded=False)

def render_insights():
    st.header("💡 Insights & help")
    analytics,news,guide=st.tabs(["Usage analytics","AI ML news","Verification guide"])
    with analytics:render_analytics()
    with news:render_news()
    with guide:
        st.subheader("Two-phase safety workflow")
        st.dataframe(pd.DataFrame([
            ["1","Normalize and pin","Accept a Hugging Face URL or owner/model and resolve an immutable revision."],["2","Inspect safely","Read metadata/config, detect access, task, remote code and resource blockers."],["3","Gate execution","Never load weights while a blocking finding remains."],["4","Run bounded inference","Use a task-aware sample and capped output."],["5","Evaluate","Measure representative quality, safety and regression behavior."],["6","Benchmark","Warm up; report percentiles, throughput, memory, errors and configuration."],["7","Qualify","Record security, portability, capacity, ownership and rollback evidence."],["8","Export","Save a reproducible, reviewable evidence bundle."],
        ],columns=["Stage","Control","What it does"]),hide_index=True,use_container_width=True)
        st.markdown("[Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api) · [Transformers pipelines](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial) · [PyTorch XPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)")
