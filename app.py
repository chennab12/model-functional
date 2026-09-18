from __future__ import annotations
import os,uuid
from pathlib import Path
import streamlit as st
from qwen_verifier.analytics import record_visit
from qwen_verifier.dashboard_pages import render_evaluate,render_insights,render_overview,render_performance,render_production,render_reports,render_verify

ROOT=Path(__file__).parent
DEFAULT="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
APP_VERSION="2.0.0"
st.set_page_config(page_title="Universal HF Model Verifier",page_icon="🧪",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{max-width:1500px;padding-top:1rem}.hero{padding:1.25rem 1.5rem;border:1px solid #27506a;border-radius:18px;background:linear-gradient(120deg,#0b1d2c,#123c47);margin-bottom:1rem}.hero h1{margin:0}.hero p{color:#c4d7e4;margin:.35rem 0 0}.stButton button{border-radius:10px}.status-legend{font-size:.85rem;color:#7f8c99}
</style>""",unsafe_allow_html=True)
st.markdown("""<section class="hero"><h1>🧪 Universal Hugging Face Model Verifier</h1><p>Safely inspect → execute → evaluate → benchmark → qualify → export evidence.</p></section>""",unsafe_allow_html=True)

for key in ("preflight","result","portability","benchmark","optimization","quality_summary","security_score","vllm_report","key"):
    if key not in st.session_state:st.session_state[key]=None
if "analytics_session_id" not in st.session_state:st.session_state.analytics_session_id=str(uuid.uuid4())
try:record_visit(st.session_state.analytics_session_id)
except Exception:pass

with st.sidebar:
    st.header("Model")
    model_input=st.text_input("HF model URL or owner/model",DEFAULT)
    revision=st.text_input("Revision",placeholder="Optional: main or commit SHA")
    with st.expander("Advanced & security settings"):
        token=st.text_input("HF token",type="password",help="For private/gated repositories. Held in memory and excluded from reports.")
        trust_remote=st.checkbox("Allow reviewed remote model code",help="Enable only after reviewing and pinning repository code.")
        max_model_gb=st.number_input("Maximum model download (GB)",0.1,1000.0,10.0,0.5,help="Execution is blocked when reported weight size exceeds this guardrail.")
    st.caption("Preflight reads small metadata/configuration first. Weight download always requires a separate approval.")
    st.markdown('<div class="status-legend">⚪ Not started · 🔵 Running · ✅ Passed · ⚠️ Review · ❌ Failed · ⛔ Blocked</div>',unsafe_allow_html=True)
    st.divider();st.caption(f"App v{APP_VERSION} · Python {os.sys.version_info.major}.{os.sys.version_info.minor}")

current_key=(model_input,revision,bool(token),trust_remote)
if st.session_state.key and st.session_state.key!=current_key:
    for key in ("preflight","result","portability","benchmark","optimization","quality_summary","security_score","vllm_report"):st.session_state[key]=None

pages={
    "Decision":[st.Page(render_overview,title="Executive overview",icon="📊",default=True)],
    "Model lifecycle":[
        st.Page(lambda:render_verify(model_input,revision,token,trust_remote,current_key,max_model_gb),title="Verify model",icon="🧭",url_path="verify"),
        st.Page(render_evaluate,title="Evaluate quality",icon="🎯"),
        st.Page(lambda:render_performance(token,trust_remote),title="Performance & portability",icon="⚡",url_path="performance"),
        st.Page(render_production,title="Production readiness",icon="🏭"),
    ],
    "Evidence & learning":[
        st.Page(lambda:render_reports(ROOT),title="Runs & reports",icon="📁",url_path="reports"),
        st.Page(render_insights,title="Insights & help",icon="💡"),
    ],
}
navigation=st.navigation(pages,position="sidebar",expanded=True)
navigation.run()
st.caption("A functional smoke test is necessary but not sufficient for production approval. Use representative quality, safety, reliability, performance, cost and operational gates.")
