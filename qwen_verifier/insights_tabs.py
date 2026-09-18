"""Streamlit views for leadership, site analytics and current AI/ML signals."""
from __future__ import annotations
import html
import pandas as pd
import streamlit as st
from .analytics import analytics_summary
from .executive import executive_summary
from .news import fetch_news

def _state():
    return {k:st.session_state.get(k) for k in ("preflight","result","portability","benchmark","optimization","quality_summary","security_score","vllm_report")}

def _heatmap_html(frame:pd.DataFrame)->str:
    """Render a small accessible heat map without pandas' Matplotlib dependency."""
    maximum=max(float(frame.to_numpy().max()),1.0)
    header="".join(f'<th scope="col">{html.escape(str(column))}</th>' for column in frame.columns)
    rows=[]
    for index,row in frame.iterrows():
        cells=[]
        for value in row:
            number=float(value);alpha=0.08+0.82*(number/maximum)
            cells.append(f'<td style="background:rgba(31,119,180,{alpha:.3f});text-align:center" title="{number:g} events">{number:g}</td>')
        rows.append(f'<tr><th scope="row">{html.escape(str(index))}</th>{"".join(cells)}</tr>')
    return '<div style="overflow-x:auto"><table style="width:100%;border-collapse:separate;border-spacing:2px"><caption style="text-align:left">Darker cells indicate more activity.</caption><thead><tr><th scope="col">Day / hour</th>'+header+'</tr></thead><tbody>'+"".join(rows)+'</tbody></table></div>'

def render_executive_summary():
    st.subheader("📊 Executive model-readiness summary")
    st.caption("Decision view built only from evidence generated in the other tabs. Missing evidence remains an open gate.")
    x=executive_summary(_state())
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Decision",x["verdict"]);c2.metric("Gates evidenced",f'{x["completed"]}/{x["total"]}')
    c3.metric("Blocking gates open",x["blocking_open"]);c4.metric("Quality",f'{x["quality"]}%' if x["quality"] is not None else "Not checked")
    c5.metric("Security",f'{x["security"]}/100' if x["security"] is not None else "Not checked")
    st.info(f'**Model:** {x["model"]}  ·  **Pinned revision:** {x["revision"]}')
    left,right=st.columns([1.25,1])
    with left:
        st.markdown("#### Qualification gates")
        chart=x["gates"].set_index("Gate")[["Readiness %"]];st.bar_chart(chart,horizontal=True,height=310)
        st.dataframe(x["gates"],hide_index=True,use_container_width=True)
    with right:
        st.markdown("#### Performance indicators")
        p1,p2=st.columns(2);p1.metric("p95 latency",f'{x["p95"]} s' if x["p95"] is not None else "Not checked");p2.metric("Throughput",f'{x["rps"]} req/s' if x["rps"] is not None else "Not checked")
        st.markdown("#### Highlights");st.markdown("\n".join(f"- ✅ {v}" for v in x["highlights"]))
        st.markdown("#### Lowlights and gaps");st.markdown("\n".join(f"- ⚠️ {v}" for v in x["lowlights"]+x["gaps"]) or "- None recorded")
    st.markdown("#### Risk register")
    st.dataframe(x["risks"],hide_index=True,use_container_width=True)
    st.markdown("#### Recommended next steps")
    st.dataframe(x["next_steps"],hide_index=True,use_container_width=True)
    st.warning("Executive interpretation: QUALIFIED_FOR_POC is not production approval. Add workload-specific safety, reliability, privacy, cost, operational ownership and rollback gates.")

def render_analytics():
    st.subheader("📈 Usage analytics")
    st.caption("Privacy-first operational analytics for this deployment; no IP address, cookie or browser fingerprint is collected by this package.")
    a=analytics_summary()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Anonymous sessions",a["sessions"],help="Approximate visits; each new Streamlit session counts once.")
    c2.metric("Page reruns",a["page_views"],help="Streamlit reruns caused by visits or interactions.")
    c3.metric("Avg. engaged duration",f'{a["avg_duration_seconds"]} s',help="Time between first and most recent interaction; not passive dwell time.")
    c4.metric("Recorded events",a["events"])
    left,right=st.columns(2)
    with left:
        st.markdown("#### Traffic over time")
        if a["daily"].empty:st.info("No traffic recorded yet.")
        else:st.line_chart(a["daily"].set_index("day")[["sessions","page_views"]])
    with right:
        st.markdown("#### UTC activity heat map")
        if a["heatmap"].empty:st.info("No activity available yet.")
        else:st.markdown(_heatmap_html(a["heatmap"]),unsafe_allow_html=True)
    st.markdown("#### Measurement boundaries")
    st.info("Anonymous sessions approximate both visitors and visits because this privacy-first mode does not set a persistent identity cookie. Duration updates only when a user interacts and Streamlit reruns. The default SQLite file is ephemeral on many cloud hosts; set `ANALYTICS_DB_PATH` to a persistent mounted volume, or replace this module with an approved analytics service for durable multi-instance reporting.")

@st.cache_data(ttl=1800,show_spinner=False)
def _cached_news():return fetch_news()

def render_news():
    st.subheader("📰 AI ML news · high-signal briefing")
    st.caption("Maximum 50 deduplicated items from an allowlist of established research, technology and industry publishers. Refreshes every 30 minutes.")
    if st.button("🔄 Refresh curated feed"):st.cache_data.clear()
    with st.spinner("Reading curated publisher feeds…"):report=_cached_news()
    items=report["items"]
    c1,c2,c3=st.columns(3);c1.metric("Selected items",len(items));c2.metric("Sources responding",sum(x["status"]=="OK" for x in report["sources"]));c3.metric("Last refresh (UTC)",report["fetched_utc"][:16].replace("T"," "))
    themes=pd.DataFrame([{"Theme":k,"Items":v} for k,v in report["themes"].items() if v]).sort_values("Items",ascending=False) if items else pd.DataFrame()
    if not themes.empty:st.bar_chart(themes.set_index("Theme"),horizontal=True)
    st.markdown("#### Trends and projections")
    st.warning("These are signal-based editorial inferences from the current selected feed—not predictions, investment advice, or verified future facts.")
    st.markdown("\n".join(f"- 🔭 {x}" for x in report["projections"]) or "- Insufficient live items to infer themes.")
    grouped={}
    for item in items:grouped.setdefault(item["category"],[]).append(item)
    for group,rows in grouped.items():
        with st.expander(f"{group} · {len(rows)}",expanded=True):
            for row in rows:
                st.markdown(f'**[{row["title"]}]({row["url"]})**  ·  {row["source"]}  ·  {row["published"]}')
                if row["summary"]:st.caption(row["summary"])
    with st.expander("Source health & selection methodology"):
        st.dataframe(pd.DataFrame(report["sources"]),hide_index=True,use_container_width=True)
        st.write(report["method"])
        st.caption("Publisher feed inclusion is a curated quality proxy, not an objective global ‘top 1%’ ranking. Always open the primary source before making a consequential decision.")
