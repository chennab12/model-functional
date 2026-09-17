"""Generic Hugging Face compatibility preflight and functional verifier."""
from __future__ import annotations
import json, platform, resource, shutil, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

WEIGHTS=(".safetensors",".bin",".pt",".pth",".ckpt")
SUPPORTED={"text-generation","text2text-generation","summarization","translation","fill-mask","text-classification","token-classification","question-answering","feature-extraction","image-classification","audio-classification","automatic-speech-recognition"}

@dataclass(frozen=True)
class Check:
    name:str; status:str; detail:str; blocking:bool=False; remediation:str=""

def normalize_model_id(value:str)->tuple[str,str|None]:
    value=value.strip().rstrip("/")
    if not value: raise ValueError("Enter a Hugging Face URL or owner/model.")
    rev=None
    if "://" in value:
        u=urlparse(value)
        if u.netloc not in {"huggingface.co","www.huggingface.co"}: raise ValueError("Only huggingface.co URLs are accepted.")
        p=[x for x in u.path.split("/") if x]
        if p and p[0]=="models": p=p[1:]
        if len(p)<2: raise ValueError("Expected https://huggingface.co/owner/model.")
        model="/".join(p[:2])
        if len(p)>=4 and p[2] in {"tree","blob","resolve"}: rev=p[3]
    else:
        p=[x for x in value.split("/") if x]
        if len(p)!=2: raise ValueError("Expected owner/model.")
        model="/".join(p)
    return model,rev

def _gb(n:float|int|None)->float: return round(float(n or 0)/(1024**3),2)

def system_snapshot()->dict[str,Any]:
    import psutil
    m=psutil.virtual_memory(); d=shutil.disk_usage(Path.home())
    try:
        import torch
        cuda=bool(torch.cuda.is_available())
        gpu=torch.cuda.get_device_name(0) if cuda else None
        gpu_gb=_gb(torch.cuda.get_device_properties(0).total_memory) if cuda else 0
        tv=torch.__version__
    except Exception:
        cuda,gpu,gpu_gb,tv=False,None,0,None
    return {"python":platform.python_version(),"platform":platform.platform(),"ram_available_gb":_gb(m.available),"ram_total_gb":_gb(m.total),"disk_free_gb":_gb(d.free),"cuda_available":cuda,"gpu":gpu,"gpu_memory_gb":gpu_gb,"torch":tv}

def infer_task(tag:str|None,config:dict)->str|None:
    if tag=="sentiment-analysis": return "text-classification"
    if tag: return tag
    a=" ".join(config.get("architectures") or []).lower()
    rules=[("causallm","text-generation"),("conditionalgeneration","text2text-generation"),("maskedlm","fill-mask"),("sequenceclassification","text-classification"),("tokenclassification","token-classification"),("questionanswering","question-answering"),("imageclassification","image-classification"),("audioclassification","audio-classification")]
    for key,task in rules:
        if key in a:return task
    return "text2text-generation" if config.get("is_encoder_decoder") else None

class CompatibilityAgent:
    """Inspect metadata/config/security/resources without loading model weights."""
    def __init__(self,model_input:str,token:str|None=None,revision:str|None=None):
        self.model_input,self.token,self.revision=model_input,token or None,revision or None

    def _stop(self,model,rev,checks,exc):
        return {"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"model_input":self.model_input,"model_id":model,"requested_revision":rev,"checks":[asdict(x) for x in checks],"compatible":False,"decision":"BLOCKED","error":f"{type(exc).__name__}: {exc}","system":system_snapshot()}

    def run(self)->dict[str,Any]:
        from huggingface_hub import HfApi,hf_hub_download
        from huggingface_hub.utils import GatedRepoError,RepositoryNotFoundError
        start=time.perf_counter(); model,url_rev=normalize_model_id(self.model_input); rev=self.revision or url_rev
        checks=[Check("Input","PASS",f"Normalized to {model}")]
        try:
            info=HfApi(token=self.token).model_info(model,revision=rev,files_metadata=True)
            checks.append(Check("Repository access","PASS","Metadata is accessible."))
        except GatedRepoError as e:
            checks.append(Check("Repository access","BLOCK","Gated access is not authorized.",True,"Accept terms and provide an HF token.")); return self._stop(model,rev,checks,e)
        except RepositoryNotFoundError as e:
            checks.append(Check("Repository access","BLOCK","Not found, private, or unauthorized.",True,"Check link and token.")); return self._stop(model,rev,checks,e)
        except Exception as e:
            checks.append(Check("Repository access","BLOCK",f"{type(e).__name__}: {e}",True,"Check network and URL.")); return self._stop(model,rev,checks,e)
        files=[{"name":x.rfilename,"size":int(x.size or 0)} for x in (info.siblings or [])]
        weights=[x for x in files if x["name"].endswith(WEIGHTS)]; size=sum(x["size"] for x in weights)
        has_config=any(x["name"]=="config.json" for x in files)
        checks.append(Check("Model configuration","PASS" if has_config else "BLOCK","config.json found." if has_config else "config.json missing.",not has_config,"Requires a Transformers config."))
        checks.append(Check("Weight artifacts","PASS" if weights else "BLOCK",f"{len(weights)} file(s), {_gb(size)} GB." if weights else "No standard weights found.",not weights,"Adapter-only/GGUF repositories need a custom adapter."))
        config={}
        if has_config:
            try:
                p=hf_hub_download(model,"config.json",revision=rev,token=self.token); config=json.loads(Path(p).read_text())
                checks.append(Check("Config parse","PASS",f"model_type={config.get('model_type','unknown')}"))
            except Exception as e: checks.append(Check("Config parse","BLOCK",str(e),True,"Verify config and authentication."))
        remote=bool(config.get("auto_map")) or any(x["name"].endswith(".py") for x in files)
        checks.append(Check("Remote code","WARN" if remote else "PASS","Custom Python may be required." if remote else "No custom AutoClass code detected.",False,"Review code and opt in explicitly." if remote else ""))
        task=infer_task(getattr(info,"pipeline_tag",None),config); ok=task in SUPPORTED
        checks.append(Check("Task adapter","PASS" if ok else "BLOCK",f"Detected: {task or 'unknown'}; "+("smoke input available." if ok else "no generic safe input."),not ok,"Add a task adapter and representative input."))
        sys=system_snapshot(); disk_need=_gb(size*1.2); ram_need=_gb(size*1.5)
        disk_ok=not size or sys["disk_free_gb"]>=disk_need; ram_ok=not size or sys["ram_available_gb"]>=ram_need
        checks.append(Check("Disk capacity","PASS" if disk_ok else "BLOCK",f'Available {sys["disk_free_gb"]} GB; estimate {disk_need} GB.',not disk_ok,"Free disk or use a smaller model."))
        checks.append(Check("Memory capacity","PASS" if ram_ok else "BLOCK",f'Available RAM {sys["ram_available_gb"]} GB; CPU estimate {ram_need} GB.',not ram_ok,"Use quantization or more RAM/VRAM."))
        blocked=any(x.blocking for x in checks)
        card=getattr(info,"card_data",None); license_value=getattr(card,"license",None)
        return {"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"model_input":self.model_input,"model_id":model,"requested_revision":rev,"resolved_revision":getattr(info,"sha",None),"pipeline_tag":getattr(info,"pipeline_tag",None),"detected_task":task,"library_name":getattr(info,"library_name",None),"license":license_value,"gated":bool(getattr(info,"gated",False)),"private":bool(getattr(info,"private",False)),"remote_code_detected":remote,"weight_size_gb":_gb(size),"estimated_disk_needed_gb":disk_need,"estimated_cpu_ram_needed_gb":ram_need,"system":sys,"config_summary":{"model_type":config.get("model_type"),"architectures":config.get("architectures"),"torch_dtype":config.get("torch_dtype"),"is_encoder_decoder":config.get("is_encoder_decoder")},"checks":[asdict(x) for x in checks],"compatible":not blocked,"decision":"READY_TO_RUN" if not blocked else "BLOCKED","seconds":round(time.perf_counter()-start,3)}

def sample_for(task:str,tokenizer=None):
    if task=="text-generation":return "Explain in one sentence why model verification matters.",{"max_new_tokens":32,"do_sample":False}
    if task in {"text2text-generation","summarization"}:return "Summarize: Functional verification confirms that a model loads and produces valid output.",{"max_new_tokens":32}
    if task=="translation":return "Translate to French: Model verification is important.",{"max_new_tokens":32}
    if task=="fill-mask":return f"Model verification is {getattr(tokenizer,'mask_token',None) or '[MASK]'}.",{}
    if task=="text-classification":return "This model verification workflow is useful.",{}
    if task=="token-classification":return "Hugging Face is based in New York.",{}
    if task=="question-answering":return {"question":"What does verification confirm?","context":"Verification confirms that a model loads and generates output."},{}
    if task=="feature-extraction":return "Functional model verification.",{}
    if task=="image-classification":
        from PIL import Image
        return Image.new("RGB",(224,224),color=(128,128,128)),{}
    if task in {"audio-classification","automatic-speech-recognition"}:
        import numpy as np
        return {"array":np.zeros(16000,dtype="float32"),"sampling_rate":16000},{}
    raise ValueError(f"No adapter for {task}")

class GenericModelVerifier:
    """Preflight first; model weights and inference only after all blockers pass."""
    def __init__(self,model_input:str,token:str|None=None,revision:str|None=None,trust_remote_code:bool=False):
        self.model_input,self.token,self.revision,self.trust_remote_code=model_input,token or None,revision or None,trust_remote_code
    def preflight(self):return CompatibilityAgent(self.model_input,self.token,self.revision).run()
    def run(self,preflight=None):
        p=preflight or self.preflight()
        if not p.get("compatible"):return {"preflight":p,"functional_test":{"status":"NOT_RUN","reason":"Preflight has blockers."},"verdict":"BLOCKED","production_ready":False}
        if p.get("remote_code_detected") and not self.trust_remote_code:return {"preflight":p,"functional_test":{"status":"NOT_RUN","reason":"Explicit remote-code opt-in required."},"verdict":"BLOCKED","production_ready":False}
        import torch,transformers
        from transformers import pipeline
        task=p["detected_task"]; model=p["model_id"]; rev=p.get("requested_revision") or p.get("resolved_revision"); start=time.perf_counter()
        try:
            pipe=pipeline(task=task,model=model,revision=rev,token=self.token,trust_remote_code=self.trust_remote_code,device=-1)
            loaded=time.perf_counter(); sample,kwargs=sample_for(task,getattr(pipe,"tokenizer",None)); t=time.perf_counter()
            with torch.inference_mode():out=pipe(sample,**kwargs)
            elapsed=time.perf_counter()-t; preview=repr(out); nonempty=bool(out)
            test={"status":"PASS" if nonempty else "FAIL","task":task,"sample_input":repr(sample)[:1000],"output":{"type":type(out).__name__,"nonempty":nonempty,"preview":preview[:1500],"truncated":len(preview)>1500},"model_load_seconds":round(loaded-start,3),"inference_seconds":round(elapsed,3),"environment":{"python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__,"device":"cpu","peak_rss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)}}
        except Exception as e:test={"status":"FAIL","task":task,"error_type":type(e).__name__,"error":str(e),"remediation":"Review model card, adapter, dependencies, token and resources."}
        return {"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"model":model,"revision":rev,"preflight":p,"functional_test":test,"verdict":"FUNCTIONAL" if test["status"]=="PASS" else "NOT_VERIFIED","production_ready":False}

QwenFunctionalVerifier=GenericModelVerifier

