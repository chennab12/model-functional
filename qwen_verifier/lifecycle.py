"""Portability, repeatable benchmarking, and controlled optimization helpers."""
from __future__ import annotations
import importlib.util, platform, statistics, time
from dataclasses import dataclass, asdict
from typing import Any

def percentile(values:list[float],p:float)->float:
    if not values:return 0.0
    ordered=sorted(values); idx=(len(ordered)-1)*p; low=int(idx); high=min(low+1,len(ordered)-1); frac=idx-low
    return ordered[low]*(1-frac)+ordered[high]*frac

def improvement_percent(baseline:float,optimized:float,lower_is_better:bool=True)->float:
    if not baseline:return 0.0
    return round(((baseline-optimized)/baseline if lower_is_better else (optimized-baseline)/baseline)*100,2)

def available_devices()->dict[str,Any]:
    import torch
    cuda=bool(torch.cuda.is_available())
    xpu=bool(hasattr(torch,"xpu") and torch.xpu.is_available())
    mps=bool(hasattr(torch.backends,"mps") and torch.backends.mps.is_available())
    return {"cpu":True,"cuda":cuda,"xpu":xpu,"mps":mps}

def sync_device(device:str)->None:
    import torch
    if device=="cuda" and torch.cuda.is_available():torch.cuda.synchronize()
    elif device=="xpu" and hasattr(torch,"xpu") and torch.xpu.is_available():torch.xpu.synchronize()
    elif device=="mps" and hasattr(torch,"mps"):torch.mps.synchronize()

def pipeline_device(device:str):
    return -1 if device=="cpu" else f"{device}:0"

def reset_peak_memory(device:str)->None:
    import torch
    backend=getattr(torch,device,None)
    if device in {"cuda","xpu"} and backend and hasattr(backend,"reset_peak_memory_stats"):backend.reset_peak_memory_stats()

def peak_memory_mb(device:str)->float|None:
    import torch
    backend=getattr(torch,device,None)
    if device in {"cuda","xpu"} and backend and hasattr(backend,"max_memory_allocated"):
        try:return round(float(backend.max_memory_allocated())/1024**2,2)
        except Exception:return None
    return None

def canonical_output(value:Any)->str:
    """Extract the first comparable semantic output from common pipelines."""
    if isinstance(value,list) and value:return canonical_output(value[0])
    if isinstance(value,dict):
        for key in ("generated_text","summary_text","translation_text","answer","label"):
            if key in value:return str(value[key])
    return repr(value)

@dataclass(frozen=True)
class PortabilityRow:
    target:str; status:str; evidence:str; next_step:str; b70_difference:str

class PortabilityAgent:
    def __init__(self,preflight:dict[str,Any]):self.preflight=preflight
    def run(self)->dict[str,Any]:
        import torch
        d=available_devices(); model=self.preflight.get("model_id","unknown")
        rows=[
            PortabilityRow("Transformers / CPU","READY" if d["cpu"] else "BLOCKED",f"Python {platform.python_version()}, PyTorch {torch.__version__}.","Run the same pinned revision and smoke input.","B70 uses device=xpu instead of cpu; synchronize XPU before timing."),
            PortabilityRow("NVIDIA / CUDA","READY" if d["cuda"] else "NOT_PRESENT","torch.cuda.is_available()="+str(d["cuda"]),"Install matching NVIDIA driver/PyTorch CUDA build.","Not applicable to B70: never use CUDA-only assumptions or nvidia-smi."),
            PortabilityRow("Intel GPU / XPU","READY" if d["xpu"] else "NOT_PRESENT","torch.xpu.is_available()="+str(d["xpu"]), "Install Intel GPU driver and an XPU-enabled upstream PyTorch build.","B70 target: verify driver, PCIe/Resizable BAR, torch.xpu device, 32 GB VRAM and XPU synchronization."),
            PortabilityRow("Apple / MPS","READY" if d["mps"] else "NOT_PRESENT","MPS availability="+str(d["mps"]), "Use an MPS-enabled macOS PyTorch build.","B70 is discrete XPU hardware; MPS behavior and kernels are not transferable."),
            PortabilityRow("vLLM runtime","INSTALLED" if importlib.util.find_spec("vllm") else "NOT_INSTALLED","Python package detection only.","Install the backend-specific vLLM build and serve the pinned revision.","For B70, use the supported Intel XPU vLLM path/container; do not install a CUDA wheel."),
            PortabilityRow("OpenVINO","INSTALLED" if importlib.util.find_spec("openvino") else "NOT_INSTALLED","Python package detection only.","Export/convert separately, then repeat quality checks.","On B70, OpenVINO may offer an Intel GPU path; compare with native PyTorch XPU using identical inputs."),
        ]
        portable=sum(r.status in {"READY","INSTALLED"} for r in rows)
        return {"model":model,"revision":self.preflight.get("resolved_revision"),"device_matrix":[asdict(r) for r in rows],"ready_targets":portable,"interpretation":"Portability is proven only on targets actually executed; NOT_PRESENT is not a model failure."}

class BenchmarkAgent:
    def __init__(self,model_id:str,task:str,revision:str|None=None,token:str|None=None,trust_remote_code:bool=False):
        self.model_id,self.task,self.revision,self.token,self.trust=model_id,task,revision,token,trust_remote_code

    def _load(self,device:str):
        from transformers import pipeline
        return pipeline(task=self.task,model=self.model_id,revision=self.revision,token=self.token,trust_remote_code=self.trust,device=pipeline_device(device))

    def run(self,device:str="cpu",warmups:int=1,iterations:int=5,max_new_tokens:int=32,batch_size:int=1)->dict[str,Any]:
        from .verifier import sample_for
        if not available_devices().get(device):raise RuntimeError(f"{device} is not available in this environment.")
        start=time.perf_counter(); pipe=self._load(device); load_s=time.perf_counter()-start;reset_peak_memory(device)
        sample,kwargs=sample_for(self.task,getattr(pipe,"tokenizer",None))
        if self.task in {"text-generation","text2text-generation","summarization","translation"}:kwargs["max_new_tokens"]=max_new_tokens
        inputs=[sample]*batch_size if batch_size>1 else sample
        for _ in range(warmups):pipe(inputs,**kwargs);sync_device(device)
        lat=[]; outputs=[]; token_counts=[];errors=[];last_output=None
        for _ in range(iterations):
            try:
                sync_device(device);t=time.perf_counter();out=pipe(inputs,**kwargs);sync_device(device);elapsed=time.perf_counter()-t
                last_output=out;lat.append(elapsed);outputs.append(repr(out)[:500])
                if getattr(pipe,"tokenizer",None):
                    try:token_counts.append(len(pipe.tokenizer.encode(canonical_output(out))))
                    except Exception:pass
            except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
        if not lat:raise RuntimeError("All benchmark iterations failed: "+(errors[0] if errors else "unknown error"))
        total=sum(lat);requests=iterations*batch_size
        mean=statistics.mean(lat);std=statistics.stdev(lat) if len(lat)>1 else 0.0;successful_requests=len(lat)*batch_size
        return {"model":self.model_id,"revision":self.revision,"task":self.task,"device":device,"settings":{"warmups":warmups,"iterations":iterations,"batch_size":batch_size,"max_new_tokens":max_new_tokens},"model_load_seconds":round(load_s,3),"latency_seconds":{"mean":round(mean,4),"min":round(min(lat),4),"p50":round(percentile(lat,.5),4),"p95":round(percentile(lat,.95),4),"p99":round(percentile(lat,.99),4),"max":round(max(lat),4),"stddev":round(std,4),"coefficient_of_variation_percent":round(100*std/mean,2) if mean else 0},"throughput":{"requests_per_second":round(successful_requests/total,3),"approx_output_tokens_per_second":round(sum(token_counts)/total,2) if token_counts else None},"reliability":{"successful_iterations":len(lat),"failed_iterations":len(errors),"error_rate_percent":round(100*len(errors)/iterations,2),"errors":errors[:3]},"resources":{"peak_accelerator_memory_mb":peak_memory_mb(device),"power_watts":None,"tokens_per_joule":None},"streaming":{"ttft_seconds":None,"inter_token_latency_seconds":None,"explanation":"Transformers pipeline is non-streaming; use the vLLM endpoint benchmark for TTFT and inter-token latency."},"output_fingerprint":canonical_output(last_output),"samples":lat,"note":"Pipeline microbenchmark with warm-up and synchronized timing. Use representative concurrency plus vLLM serve/bench for production capacity."}

class OptimizationAgent:
    """Compare identical work at batch=1 and a larger batch; quality fingerprint retained."""
    def __init__(self,model_id:str,task:str,revision:str|None=None,token:str|None=None,trust_remote_code:bool=False):
        self.bench=BenchmarkAgent(model_id,task,revision,token,trust_remote_code)
    def run(self,device:str="cpu",iterations:int=3,max_new_tokens:int=32,optimized_batch_size:int=4)->dict[str,Any]:
        base=self.bench.run(device,1,iterations,max_new_tokens,1)
        opt=self.bench.run(device,1,iterations,max_new_tokens,optimized_batch_size)
        bm=base["latency_seconds"]["mean"];om=opt["latency_seconds"]["mean"]
        bt=base["throughput"]["requests_per_second"];ot=opt["throughput"]["requests_per_second"]
        throughput_gain=improvement_percent(bt,ot,False)
        fingerprint_ok=base["output_fingerprint"]==opt["output_fingerprint"]
        verdict="CANDIDATE" if throughput_gain>0 and fingerprint_ok else "REJECT_OR_TUNE"
        return {"strategy":"static batching","baseline":base,"optimized":opt,"comparison":{"throughput_improvement_percent":throughput_gain,"batch_wall_time_change_percent":improvement_percent(bm,om,True),"quality_fingerprint_equal":fingerprint_ok},"optimization_verdict":verdict,"decision":"Adopt only if throughput improves, the real golden set passes, per-request latency meets the SLO, and repeated runs are stable.","b70_difference":"On Arc Pro B70 use XPU, BF16/FP16 where validated, increase batch gradually within 32 GB VRAM, synchronize torch.xpu for timing, and compare against vLLM continuous batching."}
