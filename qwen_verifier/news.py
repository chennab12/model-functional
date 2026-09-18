"""Allow-listed, deduplicated AI/ML RSS aggregator."""
from __future__ import annotations
import calendar,html,re,time
from datetime import datetime,timezone
import feedparser,requests

SOURCES=[
    ("OpenAI","https://openai.com/news/rss.xml",5),
    ("Google DeepMind","https://deepmind.google/blog/rss.xml",5),
    ("NVIDIA AI","https://blogs.nvidia.com/blog/category/deep-learning/feed/",4),
    ("Berkeley AI Research","https://bair.berkeley.edu/blog/feed.xml",5),
    ("Google AI","https://blog.google/technology/ai/rss/",4),
    ("Hugging Face","https://huggingface.co/blog/feed.xml",4),
    ("MIT Technology Review","https://www.technologyreview.com/topic/artificial-intelligence/feed",4),
]
CATEGORIES=[
    ("Agents & Applications",("agent","assistant","robot","application")),
    ("Models & Research",("model","research","reasoning","multimodal","benchmark")),
    ("Hardware & Infrastructure",("chip","gpu","inference","data center","compute")),
    ("Safety & Policy",("safety","policy","regulation","risk","responsible")),
    ("Data & MLOps",("data","open source","developer","deployment","platform")),
]
def category(title:str,summary:str="")->str:
    text=(title+" "+summary).lower()
    for name,words in CATEGORIES:
        if any(w in text for w in words):return name
    return "Business & Ecosystem"
def _clean(value:str)->str:return re.sub(r"\s+"," ",html.unescape(re.sub(r"<[^>]+>"," ",value or ""))).strip()
def _key(title:str)->str:return re.sub(r"[^a-z0-9]+","",title.lower())
def deduplicate(items:list[dict])->list[dict]:
    seen=set();out=[]
    for item in items:
        key=_key(item.get("title",""))
        if not key or key in seen:continue
        seen.add(key);out.append(item)
    return out
def fetch_news(limit:int=50,timeout:int=10)->dict:
    items=[];status=[];now=time.time()
    headers={"User-Agent":"ModelVerifierNews/1.0 (+Streamlit; RSS reader)"}
    for source,url,authority in SOURCES:
        try:
            response=requests.get(url,headers=headers,timeout=timeout);response.raise_for_status();feed=feedparser.parse(response.content)
            for entry in feed.entries[:20]:
                stamp=getattr(entry,"published_parsed",None) or getattr(entry,"updated_parsed",None)
                ts=calendar.timegm(stamp) if stamp else 0
                title=_clean(getattr(entry,"title",""));summary=_clean(getattr(entry,"summary",""))[:320]
                items.append({"category":category(title,summary),"title":title,"summary":summary,"source":source,"published":datetime.fromtimestamp(ts,timezone.utc).date().isoformat() if ts else "Unknown","url":getattr(entry,"link",url),"_score":authority*10+max(0,30-(now-ts)/86400) if ts else authority*10})
            status.append({"source":source,"status":"OK","items":len(feed.entries)})
        except Exception as exc:status.append({"source":source,"status":"UNAVAILABLE","items":0,"detail":f"{type(exc).__name__}: {exc}"})
    ranked=sorted(deduplicate(items),key=lambda x:x["_score"],reverse=True)[:min(limit,50)]
    for item in ranked:item.pop("_score",None)
    themes={label:sum(1 for x in ranked if x["category"]==label) for label,_ in CATEGORIES};themes["Business & Ecosystem"]=sum(1 for x in ranked if x["category"]=="Business & Ecosystem")
    top=sorted(themes.items(),key=lambda x:x[1],reverse=True)[:3]
    projections=[f'{name} is a leading current signal ({count} of {len(ranked)} selected items); monitor it for sustained momentum.' for name,count in top if count]
    return {"items":ranked,"sources":status,"themes":themes,"projections":projections,"fetched_utc":datetime.now(timezone.utc).isoformat(),"method":"Recency plus editorial-source authority; projections are signal-based inferences, not forecasts or facts."}
