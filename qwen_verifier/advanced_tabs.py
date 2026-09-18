"""Ten high-ROI qualification tabs for the Streamlit workbench."""
from __future__ import annotations
import ipaddress,json,socket,time
from urllib.parse import urlparse
import pandas as pd
import requests
import streamlit as st
from .lifecycle import available_devices
from .qualification import capacity,diagnose,quality_evaluate,reproduction_zip,reproducibility_manifest,scorecard,security_review,workflow_yaml

REFS={
"evaluate":"https://huggingface.co/docs/evaluate/index",
"vllm":"https://docs.vllm.ai/en/latest/benchmarking/cli/",
"mlflow":"https://mlflow.org/docs/latest/ml/tracking/",
"github":"https://docs.github.com/en/actions",
"slsa":"https://slsa.dev/spec/v1.0/levels",
"xpu":"https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html",
}

def snapshot():
    return {k:st.session_state.get(k) for k in ("preflight","result","portability","benchmark","optimization","quality_summary","security_score")}

def _safe_endpoint(url):
    u=urlparse(url)
    if u.scheme not in {"http","https"} or not u.hostname:raise ValueError("Use an http(s) endpoint.")
    if u.hostname not in {"localhost","127.0.0.1"} and u.scheme!="https":raise ValueError("Remote endpoints must use HTTPS.")
    try:
        ip=ipaddress.ip_address(socket.gethostbyname(u.hostname))
        if (ip.is_private or ip.is_link_local or ip.is_loopback) and u.hostname not in {"localhost","127.0.0.1"}:raise ValueError("Private/link-local remote addresses are blocked.")
    except socket.gaierror:raise ValueError("Endpoint hostname could not be resolved.")
    return url.rstrip("/")

def quality_tab():
    st.subheader("🎯 Quality & regression evaluation")
    st.write("Evaluate observed outputs against a small golden set. A functional model is useful only when task quality meets an explicit threshold.")
    sample=pd.DataFrame([
        ["math-1","reasoning","95%","95%","exact"],
        ["json-1","format","","{\"status\":\"functional\"}","json_valid"],
        ["fact-1","knowledge","Paris","The answer is Paris.","contains"],
    ],columns=["test_id","category","expected","observed","metric"])
    st.download_button("⬇️ Golden-set CSV template",sample.to_csv(index=False),"golden_set_template.csv","text/csv")
    upload=st.file_uploader("Upload completed golden-set CSV",type="csv",key="quality_csv")
    df=pd.read_csv(upload) if upload else st.data_editor(sample,num_rows="dynamic",use_container_width=True)
    threshold=st.slider("Required pass rate",50,100,95)
    if st.button("🎯 Evaluate quality"):
        try:
            results,summary=quality_evaluate(df);summary["required_percent"]=threshold
            summary["decision"]="PASS" if summary["pass_rate_percent"]>=threshold else "FAIL"
            st.session_state.quality_results=results;st.session_state.quality_summary=summary
        except Exception as exc:st.exception(exc)
    if st.session_state.get("quality_summary"):
        s=st.session_state.quality_summary;c1,c2,c3,c4=st.columns(4)
        c1.metric("Tests",s["tests"]);c2.metric("Passed",s["passed"]);c3.metric("Pass rate",f'{s["pass_rate_percent"]}%');c4.metric("Gate",s["decision"])
        st.dataframe(st.session_state.quality_results,hide_index=True,use_container_width=True)
        st.download_button("⬇️ Quality results",st.session_state.quality_results.to_csv(index=False),"quality_results.csv","text/csv")
    st.markdown(f'[Hugging Face evaluation guidance ↗]({REFS["evaluate"]})')

def scorecard_tab():
    st.subheader("📊 Executive qualification scorecard")
    threshold=st.number_input("Quality gate (%)",0.0,100.0,95.0)
    df,verdict=scorecard(snapshot(),threshold)
    st.metric("Overall decision",verdict)
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.info("A HOLD means at least one blocking gate failed or has not been checked. Production approval requires organization-specific safety and reliability evidence.")
    st.download_button("⬇️ Scorecard",df.to_csv(index=False),"qualification_scorecard.csv","text/csv")

def history_tab():
    st.subheader("🔄 Run history & comparison")
    if "run_history" not in st.session_state:st.session_state.run_history=[]
    label=st.text_input("Run label",value=f'run-{len(st.session_state.run_history)+1}')
    if st.button("➕ Capture current session"):
        p=st.session_state.get("preflight") or {};b=st.session_state.get("benchmark") or {};q=st.session_state.get("quality_summary") or {};o=st.session_state.get("optimization") or {}
        st.session_state.run_history.append({"label":label,"model":p.get("model_id"),"revision":p.get("resolved_revision"),"device":b.get("device"),"quality_percent":q.get("pass_rate_percent"),"p95_seconds":b.get("latency_seconds",{}).get("p95"),"requests_per_second":b.get("throughput",{}).get("requests_per_second"),"optimization":o.get("optimization_verdict"),"captured_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    if st.session_state.run_history:
        df=pd.DataFrame(st.session_state.run_history);st.dataframe(df,hide_index=True,use_container_width=True)
        st.download_button("⬇️ Run history",df.to_csv(index=False),"run_history.csv","text/csv")
        st.caption("For multi-user persistence, replace session history with SQLite, object storage or MLflow.")
    st.markdown(f'[MLflow experiment tracking ↗]({REFS["mlflow"]})')

def vllm_tab():
    st.subheader("🚀 vLLM endpoint qualification")
    st.caption("Tests an existing OpenAI-compatible vLLM endpoint. It does not install or start vLLM inside Streamlit.")
    endpoint=st.text_input("Endpoint",value="http://localhost:8000")
    model=st.text_input("Served model name",value=(st.session_state.get("preflight") or {}).get("model_id",""))
    api_key=st.text_input("API key (optional)",type="password",key="vllm_key")
    prompt=st.text_input("Prompt",value="Return exactly: VLLM_OK")
    if st.button("🚀 Probe vLLM server"):
        try:
            base=_safe_endpoint(endpoint);headers={"Authorization":f"Bearer {api_key}"} if api_key else {}
            t=time.perf_counter();models=requests.get(base+"/v1/models",headers=headers,timeout=15);health_s=time.perf_counter()-t;models.raise_for_status()
            payload={"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":16,"temperature":0}
            t=time.perf_counter();resp=requests.post(base+"/v1/chat/completions",headers={**headers,"Content-Type":"application/json"},json=payload,timeout=60);latency=time.perf_counter()-t;resp.raise_for_status()
            report={"status":"PASS","health_seconds":round(health_s,3),"request_seconds":round(latency,3),"models":models.json(),"response":resp.json()}
            st.session_state.vllm_report=report
        except Exception as exc:st.session_state.vllm_report={"status":"FAIL","error":f"{type(exc).__name__}: {exc}"}
    if st.session_state.get("vllm_report"):
        st.json(st.session_state.vllm_report)
        st.download_button("⬇️ vLLM report",json.dumps(st.session_state.vllm_report,indent=2),"vllm_report.json","application/json")
    st.info("B70: use an Intel-XPU-compatible vLLM build/container. Do not use a CUDA wheel. Run vLLM bench externally for concurrency, TTFT and inter-token latency.")
    st.markdown(f'[vLLM benchmark CLI ↗]({REFS["vllm"]})')

def reproducibility_tab():
    st.subheader("🧾 Reproducibility bundle")
    manifest=reproducibility_manifest(snapshot());st.json(manifest,expanded=False)
    st.download_button("⬇️ Manifest JSON",json.dumps(manifest,indent=2,default=str),"manifest.json","application/json")
    st.download_button("📦 Reproduction ZIP",reproduction_zip(snapshot()),"model_reproduction_bundle.zip","application/zip")
    st.info("B70 reports should also capture driver, torch.xpu availability, device name, precision, VRAM and synchronization policy.")

def diagnosis_tab():
    st.subheader("🧰 Automated failure diagnosis")
    text=st.text_area("Paste an exception or log excerpt",height=170,placeholder="OutOfMemoryError...")
    if st.button("🔎 Diagnose"):
        st.session_state.diagnosis=diagnose(text)
    if st.session_state.get("diagnosis"):st.dataframe(pd.DataFrame(st.session_state.diagnosis),hide_index=True,use_container_width=True)

def security_tab():
    st.subheader("🛡️ Security, license & supply chain")
    p=st.session_state.get("preflight")
    if not p:st.info("Run Compatibility first.");return
    df,score,decision=security_review(p);st.session_state.security_score=score
    c1,c2=st.columns(2);c1.metric("Security score",f"{score}/100");c2.metric("Decision",decision)
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.warning("This is a lightweight readiness screen, not a legal opinion, malware analysis or full vulnerability assessment.")
    st.markdown(f'[SLSA supply-chain levels ↗]({REFS["slsa"]})')

def capacity_tab():
    st.subheader("💵 Capacity, cost & power estimator")
    c1,c2,c3=st.columns(3)
    users=c1.number_input("Peak users",1,1_000_000,100)
    rpm=c2.number_input("Requests/user/min",0.01,100.0,0.5)
    latency=c3.number_input("Average latency (s)",0.01,300.0,1.0)
    b=st.session_state.get("benchmark") or {};default_tps=float(b.get("throughput",{}).get("approx_output_tokens_per_second") or 50)
    c4,c5,c6=st.columns(3)
    tps=c4.number_input("Measured tokens/s",0.1,1_000_000.0,default_tps)
    hourly=c5.number_input("Hardware cost/hour ($)",0.0,1000.0,1.0)
    watts=c6.number_input("Average power (W)",0.0,5000.0,190.0)
    report=capacity(int(users),float(rpm),float(latency),float(tps),float(hourly),float(watts))
    st.json(report)
    st.download_button("⬇️ Capacity estimate",json.dumps(report,indent=2),"capacity_estimate.json","application/json")
    st.caption("B70: replace assumed power and tokens/s with measured platform-level values; include host power if comparing total cost.")

def b70_tab():
    st.subheader("🟦 Intel Arc Pro B70 qualification profile")
    d=available_devices();p=st.session_state.get("preflight") or {}
    rows=[
        ["Driver installed","Manual evidence required","Install the current validated Intel GPU driver."],
        ["PyTorch XPU visible","PASS" if d["xpu"] else "NOT_PRESENT",f'torch.xpu availability={d["xpu"]}'],
        ["Model fits 32 GB VRAM","LIKELY" if p.get("estimated_cpu_ram_needed_gb",999)<32 else "REVIEW",f'Conservative estimate={p.get("estimated_cpu_ram_needed_gb","—")} GB; measure actual XPU allocation.'],
        ["Pinned model revision","PASS" if p.get("resolved_revision") else "NOT_CHECKED",p.get("resolved_revision","Run Compatibility.")],
        ["Functional XPU run","NOT_CHECKED","Execute the Functional phase with device=xpu in a B70 environment."],
        ["Quality parity","NOT_CHECKED","Run the same golden set on reference and B70."],
        ["Performance evidence","NOT_CHECKED","Capture TTFT, inter-token latency, p95, tokens/s and VRAM."],
        ["vLLM XPU endpoint","NOT_CHECKED","Serve with the supported Intel-XPU vLLM path and probe the API."],
        ["Stability soak","NOT_CHECKED","Run sustained load and record errors, memory growth and thermals."],
    ]
    df=pd.DataFrame(rows,columns=["Qualification gate","Status","Evidence / next action"]);st.dataframe(df,hide_index=True,use_container_width=True)
    st.download_button("⬇️ B70 checklist",df.to_csv(index=False),"intel_b70_qualification.csv","text/csv")
    st.markdown(f'[PyTorch Intel XPU guide ↗]({REFS["xpu"]})')

def cicd_tab():
    st.subheader("⚙️ GitHub Actions qualification gate")
    yaml=workflow_yaml();st.code(yaml,language="yaml")
    st.download_button("⬇️ model-qualification.yml",yaml,"model-qualification.yml","text/yaml")
    st.info("Place the downloaded file at .github/workflows/model-qualification.yml. Keep heavyweight GPU/B70 testing on a labeled self-hosted runner.")
    st.markdown(f'[GitHub Actions documentation ↗]({REFS["github"]})')

def render_qualification_workbench():
    st.subheader("🏭 Model qualification workbench")
    st.caption("Ten decision-oriented capabilities layered on top of the live compatibility, functional, portability, benchmark and optimization tests.")
    tabs=st.tabs(["🎯 Quality","📊 Scorecard","🔄 History","🚀 vLLM","🧾 Reproduce","🧰 Diagnose","🛡️ Security","💵 Capacity","🟦 B70","⚙️ CI/CD"])
    funcs=[quality_tab,scorecard_tab,history_tab,vllm_tab,reproducibility_tab,diagnosis_tab,security_tab,capacity_tab,b70_tab,cicd_tab]
    for tab,func in zip(tabs,funcs):
        with tab:func()

