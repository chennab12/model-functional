from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from qwen_verifier import QwenFunctionalVerifier

ROOT = Path(__file__).parent
st.set_page_config(page_title="Qwen Functional Verifier", page_icon="🧪", layout="wide")
st.markdown("""<style>
.block-container{max-width:1500px;padding-top:1rem}.hero{padding:1.5rem 1.8rem;border:1px solid #27506a;border-radius:20px;background:linear-gradient(120deg,#0b1d2c,#123c47);margin-bottom:1rem}.hero h1{margin:0}.hero p{color:#b7cedd;margin-bottom:0}.card{background:#0d2030;border:1px solid #234257;border-radius:14px;padding:1rem;height:100%}.io{background:#081722;border-left:4px solid #35d0ba;border-radius:8px;padding:.8rem;white-space:pre-wrap}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_evidence():
    return json.loads((ROOT / "artifacts/latest_verification.json").read_text())

@st.cache_data
def load_steps():
    return pd.read_csv(ROOT / "data/verification_steps.csv")

report = load_evidence()
assessment = report["assessment"]
st.markdown("""<section class="hero"><h1>🧪 Qwen2.5 Functional Verifier</h1><p>Evidence-first verification of <b>Qwen/Qwen2.5-0.5B-Instruct</b>: annotated workflow, actual inputs/outputs, live rerun and TPM decision summary.</p></section>""", unsafe_allow_html=True)
a,b,c,d = st.columns(4)
a.metric("Runtime smoke test", assessment["functional_smoke_test"])
b.metric("Strict quality gates", assessment["strict_quality_gates"])
c.metric("Parameters", f'{report["environment"]["parameter_count"]/1e6:.1f}M')
d.metric("Peak process RAM", f'{report["environment"]["peak_rss_mb"]/1024:.2f} GB')
st.warning("🟡 Verdict: **FUNCTIONAL WITH QUALITY CAVEATS** — load and generation passed; this single smoke run is not a production qualification.")

tabs = st.tabs(["🧭 1 · Annotated steps","🔬 2 · Actual evidence","▶️ 3 · Run the model","📌 4 · TPM gist"])
with tabs[0]:
    st.subheader("Core functional-verification workflow")
    st.caption("Read top to bottom. Every row explains the action, concrete evidence, decision value and authoritative reference.")
    st.dataframe(load_steps(), hide_index=True, use_container_width=True, height=650, column_config={
        "Step": st.column_config.NumberColumn(width="small"),
        "Annotated step": st.column_config.TextColumn(width="medium"),
        "What to do": st.column_config.TextColumn(width="large"),
        "Actual example / evidence": st.column_config.TextColumn(width="large"),
        "Why it matters": st.column_config.TextColumn(width="large"),
        "Source": st.column_config.LinkColumn("Source ↗", display_text="Open ↗"),
    })
    st.download_button("⬇️ Download verification steps", load_steps().to_csv(index=False), "qwen_verification_steps.csv", "text/csv")

with tabs[1]:
    st.subheader("Actual CPU execution evidence")
    st.caption(f'Executed {report["timestamp_utc"]} · revision {report["revision"]} · deterministic decoding')
    for test in report["tests"]:
        icon = "✅" if test["case"] == "Instruction following" else "⚠️"
        with st.expander(f'{icon} {test["case"]}', expanded=True):
            left,right = st.columns(2)
            left.markdown("**Input prompt**")
            left.markdown(f'<div class="io">{test["prompt"]}</div>', unsafe_allow_html=True)
            right.markdown("**Observed output**")
            right.markdown(f'<div class="io">{test["output"]}</div>', unsafe_allow_html=True)
            st.caption(f'Input {test["input_tokens"]} tokens · Output {test["output_tokens"]} tokens · {test["seconds"]} s · {test["tokens_per_second"]} tokens/s')
    st.markdown("#### Automated assertions")
    checks = pd.DataFrame([{"Gate":k.replace("_"," ").title(),"Result":"✅ PASS" if v else "❌ FAIL"} for k,v in report["checks"].items()])
    st.dataframe(checks, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download JSON evidence", json.dumps(report,indent=2), "latest_verification.json", "application/json")

with tabs[2]:
    st.subheader("Run the actual model again")
    st.info("First run downloads roughly 1 GB of model weights and can require about 2 GB RAM. Local execution is more reliable than a small free cloud instance.")
    st.code("python run_verification.py\nstreamlit run app.py", language="bash")
    confirm = st.checkbox("I understand this downloads and loads the model")
    if st.button("🚀 Run verification now", disabled=not confirm, type="primary"):
        with st.status("Loading model and running deterministic tests…", expanded=True) as status:
            try:
                fresh = QwenFunctionalVerifier().run()
                status.update(label="Verification completed", state="complete")
                st.json(fresh)
                st.download_button("Download this run", json.dumps(fresh,indent=2), "qwen_run.json", "application/json")
            except Exception as exc:
                status.update(label="Verification failed", state="error")
                st.exception(exc)
                st.error("Check network access, available RAM, disk space and dependency versions.")
    st.caption("No paid API is used. Transformers downloads the public model and inference runs in this Python process.")

with tabs[3]:
    st.subheader("Gist for a technical TPM")
    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="card"><h3>✅ What passed</h3><p>Repository resolution, tokenizer, model load, 494M parameter tensors, chat template, forward pass, generation and decoding.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h3>⚠️ What failed</h3><p>The tiny model produced 98% instead of 95% and wrapped JSON in a Markdown fence. Add parsing, constrained decoding or a stronger model where contracts matter.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h3>🚫 Not proven</h3><p>Accuracy at scale, safety, fairness, long-context quality, concurrency, accelerator performance, reliability and workload economics.</p></div>', unsafe_allow_html=True)
    st.markdown("### Recommended launch-gate sequence")
    st.markdown("""
1. **Functional:** model loads, produces non-empty output and honors the chat interface.
2. **Quality:** representative golden set and thresholds by task and customer segment.
3. **Performance:** p50/p95/p99 latency, TTFT, inter-token latency and throughput.
4. **Reliability:** repeated runs, concurrency, timeout, OOM and recovery tests.
5. **Safety/security:** abuse, prompt injection, privacy, license and supply-chain review.
6. **Release:** pinned revision, reproducible environment, canary, monitoring, owner and rollback.
""")
    st.success("Use for learning, prototyping and low-risk experiments after task-specific evaluation.")
    st.error("Do not call it production-ready from this smoke test alone.")
    st.markdown("[🤗 Model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [📚 Qwen2 docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2) · [▶️ YouTube tutorials](https://www.youtube.com/results?search_query=Qwen2.5+0.5B+Transformers+tutorial)")

st.caption("Evidence is specific to the recorded revision, prompts, versions and CPU runtime. Rerun after any model, dependency, prompt-template or hardware change.")

