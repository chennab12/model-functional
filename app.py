from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from qwen_verifier import GenericModelVerifier

ROOT=Path(__file__).parent
DEFAULT="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
st.set_page_config(page_title="Universal HF Model Verifier",page_icon="🧪",layout="wide")
st.markdown("""<style>
.block-container{max-width:1500px;padding-top:1rem}.hero{padding:1.5rem 1.8rem;border:1px solid #27506a;border-radius:20px;background:linear-gradient(120deg,#0b1d2c,#123c47);margin-bottom:1rem}.hero h1{margin:0}.hero p{color:#b7cedd;margin-bottom:0}.card{background:#0d2030;border:1px solid #234257;border-radius:14px;padding:1rem;height:100%}.io{background:#081722;border-left:4px solid #35d0ba;border-radius:8px;padding:.8rem;white-space:pre-wrap}
</style>""",unsafe_allow_html=True)
st.markdown("""<section class="hero"><h1>🧪 Universal Hugging Face Model Verifier</h1><p>Paste a model link → inspect compatibility without loading weights → run a safe task-aware smoke test → export evidence.</p></section>""",unsafe_allow_html=True)

with st.sidebar:
    st.header("Model input")
    model_input=st.text_input("HF model URL or owner/model",DEFAULT)
    revision=st.text_input("Revision (optional)",placeholder="main or commit SHA")
    token=st.text_input("HF token (only for private/gated models)",type="password",help="Used in memory for this run; never written to the report.")
    trust_remote=st.checkbox("Allow reviewed remote model code",help="Only enable after reviewing repository Python code.")
    st.divider()
    st.caption("The preflight reads Hub metadata and config.json only. Large weights are loaded only after all blocking checks pass.")

if "preflight" not in st.session_state:st.session_state.preflight=None
if "result" not in st.session_state:st.session_state.result=None
if "key" not in st.session_state:st.session_state.key=None
current_key=(model_input,revision,bool(token),trust_remote)
if st.session_state.key and st.session_state.key!=current_key:
    st.session_state.preflight=None;st.session_state.result=None

t1,t2,t3,t4=st.tabs(["🛡️ 1 · Compatibility preflight","▶️ 2 · Functional test","🧭 3 · How it works","🔬 4 · Included Qwen evidence"])
with t1:
    st.subheader("Check setup before downloading model weights")
    st.write("Checks repository access, Transformers structure, task support, remote-code risk, disk, RAM, CUDA visibility and versioned identity.")
    if st.button("🔍 Run compatibility preflight",type="primary"):
        try:
            with st.spinner("Reading model metadata and small configuration…"):
                agent=GenericModelVerifier(model_input,token or None,revision or None,trust_remote)
                st.session_state.preflight=agent.preflight();st.session_state.result=None;st.session_state.key=current_key
        except Exception as exc:st.error(f"{type(exc).__name__}: {exc}")
    p=st.session_state.preflight
    if p:
        decision=p.get("decision","BLOCKED")
        (st.success if decision=="READY_TO_RUN" else st.error)(f"{'✅' if decision=='READY_TO_RUN' else '⛔'} Preflight decision: **{decision}**")
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Detected task",p.get("detected_task") or "unknown")
        m2.metric("Weight size",f'{p.get("weight_size_gb",0)} GB')
        m3.metric("Est. CPU RAM",f'{p.get("estimated_cpu_ram_needed_gb",0)} GB')
        m4.metric("Available RAM",f'{p.get("system",{}).get("ram_available_gb","?")} GB')
        rows=pd.DataFrame(p.get("checks",[]))
        if not rows.empty:
            rows["Result"]=rows["status"].map({"PASS":"✅ PASS","WARN":"⚠️ WARN","BLOCK":"⛔ BLOCK"}).fillna(rows["status"])
            st.dataframe(rows[["Result","name","detail","remediation"]],hide_index=True,use_container_width=True,height=430,column_config={"Result":st.column_config.TextColumn("Status",width="small"),"name":st.column_config.TextColumn("Check",width="medium"),"detail":st.column_config.TextColumn("Evidence",width="large"),"remediation":st.column_config.TextColumn("Recommended action",width="large")})
        st.download_button("⬇️ Download preflight JSON",json.dumps(p,indent=2,default=str),"hf_preflight.json","application/json")
        if p.get("remote_code_detected") and not trust_remote:st.warning("Remote code was detected. Review the repository first; functional execution remains blocked until explicit opt-in.")

with t2:
    st.subheader("Task-aware functional smoke test")
    p=st.session_state.preflight
    if not p:st.info("Run Compatibility Preflight in Tab 1 first.")
    elif not p.get("compatible"):st.error("Functional test is disabled because preflight has blocking findings.")
    elif p.get("remote_code_detected") and not trust_remote:st.error("Functional test is disabled until remote code is reviewed and explicitly allowed.")
    else:
        st.success(f'Ready: {p["model_id"]} · task={p.get("detected_task")} · revision={p.get("resolved_revision")}')
        st.warning("This step downloads weights and loads the model. A PASS proves basic load/inference/output—not production quality.")
        confirm=st.checkbox("I understand the download, memory and code-execution implications",key="run_confirm")
        if st.button("🚀 Download, load and run smoke test",disabled=not confirm,type="primary"):
            agent=GenericModelVerifier(model_input,token or None,revision or None,trust_remote)
            with st.status("Downloading weights and running the task adapter…",expanded=True) as status:
                result=agent.run(p);st.session_state.result=result
                state="complete" if result.get("verdict")=="FUNCTIONAL" else "error"
                status.update(label=f'Verdict: {result.get("verdict")}',state=state)
        r=st.session_state.result
        if r:
            ft=r["functional_test"]
            (st.success if ft.get("status")=="PASS" else st.error)(f'Functional test: **{ft.get("status")}** · Overall: **{r.get("verdict")}**')
            c1,c2,c3=st.columns(3)
            c1.metric("Task",ft.get("task","—"));c2.metric("Load time",f'{ft.get("model_load_seconds","—")} s');c3.metric("Inference",f'{ft.get("inference_seconds","—")} s')
            if "sample_input" in ft:st.markdown("**Sample input**");st.code(ft["sample_input"])
            if "output" in ft:st.markdown("**Output preview**");st.code(ft["output"]["preview"])
            if "error" in ft:st.exception(RuntimeError(ft["error"]))
            st.download_button("⬇️ Download complete evidence",json.dumps(r,indent=2,default=str),"hf_functional_report.json","application/json")

with t3:
    st.subheader("Two-phase safety and compatibility workflow")
    guide=pd.DataFrame([
        ["1","🔗 Normalize input","Accept only huggingface.co URL or owner/model; preserve revision."],
        ["2","🔐 Check access","Detect public, private, missing and gated repositories."],
        ["3","📄 Inspect structure","Require config.json and standard Transformers weight artifacts."],
        ["4","🧩 Detect task","Use pipeline_tag, then architecture-based fallback."],
        ["5","🛡️ Flag remote code","Never execute custom repository code without explicit opt-in."],
        ["6","💾 Estimate resources","Compare weight-derived disk/RAM estimates with the current host."],
        ["7","⛔ Gate execution","Do not load weights if any blocking preflight check fails."],
        ["8","🧪 Run adapter","Create a small input matching text, image or audio task."],
        ["9","✅ Validate output","Require successful load, inference and a non-empty output."],
        ["10","📦 Export evidence","Record revision, task, input/output, environment, timing and errors."],
    ],columns=["Stage","Control","What it does"])
    st.dataframe(guide,hide_index=True,use_container_width=True)
    st.markdown("### Automatically supported task families")
    st.write("Text generation, text-to-text, summarization, translation, fill-mask, classification, token classification, question answering, feature extraction, image classification, audio classification and speech recognition.")
    st.info("Diffusion, GGUF-only, adapter-only, multimodal chat, custom research architectures and models needing special processors may require a project-specific adapter. The app reports that clearly instead of labeling the model defective.")
    st.markdown("[Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api) · [Transformers pipelines](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial) · [Custom model security](https://huggingface.co/docs/transformers/main/en/custom_models)")

with t4:
    st.subheader("Previously executed Qwen2.5-0.5B-Instruct evidence")
    evidence=json.loads((ROOT/"artifacts/latest_verification.json").read_text())
    st.success("The included baseline proves the original package was run against the real Qwen model.")
    st.json(evidence,expanded=False)
    st.caption("Use this as an example report. New models receive their own preflight and functional report.")

st.caption("A functional smoke test is necessary but not sufficient for production: add golden-set quality, safety, reliability, performance, concurrency and cost evaluations.")

