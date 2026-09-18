"""Streamlit renderers for portability, benchmarking and optimization phases."""
import json
import pandas as pd
import streamlit as st
from .lifecycle import BenchmarkAgent,OptimizationAgent,PortabilityAgent,available_devices

def _steps(rows):
    st.dataframe(pd.DataFrame(rows,columns=["Step","Simple annotated action","Intel Arc Pro B70 difference"]),hide_index=True,use_container_width=True)

def portability_tab(token,trust_remote):
    st.subheader("Model portability across runtimes and hardware")
    st.caption("Portability is proven only when the same pinned model and acceptance test run on another target. Detection means readiness, not proof.")
    _steps([
        ["1 · Pin","Reuse revision, input and acceptance rule.","Pin the same commit; never compare B70 against a moving model."],
        ["2 · Inventory","Capture OS, Python, PyTorch, Transformers and custom ops.","Use Intel GPU driver plus upstream XPU-enabled PyTorch; Intel Extension for PyTorch is retired."],
        ["3 · Detect","Check CPU, CUDA, XPU, MPS and serving runtimes.","B70 must appear through torch.xpu; CUDA is not the B70 backend."],
        ["4 · Precision","Verify dtype and operator support.","Start with BF16/FP16 only after validation; retain a known-good reference."],
        ["5 · Execute","Run the identical smoke input.","Move work to xpu and synchronize torch.xpu around timing."],
        ["6 · Quality","Apply the same output/golden-set gate.","A faster B70 result is invalid if quality misses tolerance."],
        ["7 · Evidence","Save revision, device, runtime, output and time.","Add B70 identity, 32 GB VRAM, driver and XPU memory evidence."],
    ])
    p=st.session_state.preflight
    if not p:st.info("Run Compatibility first.")
    else:
        if st.button("🔁 Build portability matrix"):st.session_state.portability=PortabilityAgent(p).run()
        port=st.session_state.portability
        if port:
            st.dataframe(pd.DataFrame(port["device_matrix"]),hide_index=True,use_container_width=True,column_config={"target":st.column_config.TextColumn("Target",width="medium"),"status":st.column_config.TextColumn("Status",width="small"),"evidence":st.column_config.TextColumn("Evidence",width="large"),"next_step":st.column_config.TextColumn("Next action",width="large"),"b70_difference":st.column_config.TextColumn("Intel Arc Pro B70 difference",width="large")})
            st.info(port["interpretation"])
            st.download_button("⬇️ Portability report",json.dumps(port,indent=2,default=str),"hf_portability_report.json","application/json")

def benchmark_tab(token,trust_remote):
    st.subheader("Repeatable model performance benchmark")
    st.caption("Warm up first; measure identical work; keep model load separate from steady-state inference.")
    _steps([
        ["1 · Freeze shape","Fix prompt, output cap, batch and revision.","Identical shape on B70; context and batch drive VRAM."],
        ["2 · Record","Capture software, device and dtype.","Record B70 driver, XPU PyTorch build and device identity."],
        ["3 · Warm up","Exclude initialization from measurements.","Synchronize torch.xpu after B70 warm-up."],
        ["4 · Repeat","Run multiple timed iterations.","B70 is asynchronous: XPU-sync before and after timing."],
        ["5 · Percentiles","Report mean, p50, p95 and p99.","Keep CPU and B70 as separate rows; never average devices."],
        ["6 · Throughput","Report requests/s and approximate tokens/s.","Also capture B70 utilization, VRAM and power when available."],
        ["7 · Quality","Retain output evidence.","Performance wins still need the same quality tolerance."],
    ])
    p=st.session_state.preflight
    if not p or not p.get("compatible"):st.info("Successful Compatibility is required.");return
    devices=[k for k,v in available_devices().items() if v]
    c1,c2,c3,c4=st.columns(4)
    device=c1.selectbox("Device",devices,key="bench_device")
    warmups=c2.number_input("Warm-ups",1,5,1)
    iterations=c3.number_input("Iterations",3,20,5)
    max_tokens=c4.number_input("Max new tokens",8,128,32,8)
    if st.button("⏱️ Run benchmark"):
        agent=BenchmarkAgent(p["model_id"],p["detected_task"],p.get("resolved_revision"),token or None,trust_remote)
        with st.status("Loading, warming up and measuring…",expanded=True) as status:
            try:
                st.session_state.benchmark=agent.run(device,int(warmups),int(iterations),int(max_tokens),1)
                status.update(label="Benchmark complete",state="complete")
            except Exception as exc:status.update(label="Benchmark failed",state="error");st.exception(exc)
    b=st.session_state.benchmark
    if b:
        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("p50 latency",f'{b["latency_seconds"]["p50"]} s');m2.metric("p95 latency",f'{b["latency_seconds"]["p95"]} s')
        m3.metric("Requests/s",b["throughput"]["requests_per_second"]);m4.metric("Approx tokens/s",b["throughput"]["approx_output_tokens_per_second"] or "n/a");m5.metric("Error rate",f'{b.get("reliability",{}).get("error_rate_percent",0)}%')
        x1,x2,x3=st.columns(3);x1.metric("p99",f'{b["latency_seconds"]["p99"]} s');x2.metric("Latency variation",f'{b["latency_seconds"].get("coefficient_of_variation_percent","—")}%');x3.metric("Peak accelerator memory",f'{b.get("resources",{}).get("peak_accelerator_memory_mb")} MB' if b.get("resources",{}).get("peak_accelerator_memory_mb") is not None else "Not available")
        st.line_chart(pd.DataFrame({"iteration":range(1,len(b["samples"])+1),"latency_seconds":b["samples"]}).set_index("iteration"))
        st.caption(b.get("streaming",{}).get("explanation",""))
        st.download_button("⬇️ Benchmark report",json.dumps(b,indent=2,default=str),"hf_benchmark_report.json","application/json")
    st.caption("This is a Transformers pipeline microbenchmark. Use vLLM serve/bench for production serving concurrency.")

def optimization_tab(token,trust_remote):
    st.subheader("Controlled model optimization")
    st.caption("Compare a baseline with one changed variable. The included safe experiment tests identical individual requests versus static batching.")
    _steps([
        ["1 · Baseline","Measure before changing anything.","Fix B70 driver, XPU build, dtype, input and output length."],
        ["2 · One variable","Change batching first.","Increase B70 batch gradually while watching 32 GB VRAM."],
        ["3 · Measure","Reuse warm-up and iteration policy.","Synchronize torch.xpu around every timed region."],
        ["4 · Compare","Calculate throughput and latency change.","B70 parallelism can help throughput but can hurt request latency."],
        ["5 · Quality","Fingerprint first; then use a golden set.","BF16/FP16/quantization needs declared quality tolerance."],
        ["6 · Expand","Try dtype, compile, quantization, cache and serving separately.","B70: native XPU, BF16/FP16, torch.compile if supported, then Intel-XPU vLLM batching."],
        ["7 · Decide","Adopt only when quality and SLOs pass.","Record B70 stack and a rollback configuration."],
    ])
    p=st.session_state.preflight
    if not p or not p.get("compatible"):st.info("Successful Compatibility is required.");return
    devices=[k for k,v in available_devices().items() if v]
    c1,c2,c3,c4=st.columns(4)
    device=c1.selectbox("Device",devices,key="opt_device")
    iterations=c2.number_input("Iterations/trial",2,10,3)
    tokens=c3.number_input("Output-token cap",8,128,32,8)
    batch=c4.number_input("Optimized batch",2,8,4)
    st.warning("This loads and measures the model twice. Start with a small model.")
    if st.button("⚙️ Run baseline vs batching"):
        agent=OptimizationAgent(p["model_id"],p["detected_task"],p.get("resolved_revision"),token or None,trust_remote)
        with st.status("Running before/after trials…",expanded=True) as status:
            try:
                st.session_state.optimization=agent.run(device,int(iterations),int(tokens),int(batch))
                status.update(label="Optimization complete",state="complete")
            except Exception as exc:status.update(label="Optimization failed",state="error");st.exception(exc)
    o=st.session_state.optimization
    if o:
        comp=o["comparison"];c1,c2,c3=st.columns(3)
        c1.metric("Baseline req/s",o["baseline"]["throughput"]["requests_per_second"])
        c2.metric("Optimized req/s",o["optimized"]["throughput"]["requests_per_second"],f'{comp["throughput_improvement_percent"]}%')
        c3.metric("Decision",o["optimization_verdict"])
        st.info(o["decision"]);st.caption(o["b70_difference"])
        st.download_button("⬇️ Optimization report",json.dumps(o,indent=2,default=str),"hf_optimization_report.json","application/json")
