from qwen_verifier.verifier import normalize_model_id, infer_task, sample_for
from qwen_verifier.lifecycle import canonical_output,improvement_percent,percentile
from qwen_verifier.qualification import capacity,diagnose,quality_evaluate,security_review
from qwen_verifier.analytics import analytics_summary,record_visit
from qwen_verifier.executive import executive_summary
from qwen_verifier.news import category,deduplicate
import pandas as pd

def test_repo_id():
    assert normalize_model_id("Qwen/Qwen2.5-0.5B-Instruct") == ("Qwen/Qwen2.5-0.5B-Instruct", None)

def test_hf_url_and_revision():
    assert normalize_model_id("https://huggingface.co/google/flan-t5-small/tree/main") == ("google/flan-t5-small", "main")

def test_rejects_non_hf_url():
    try:
        normalize_model_id("https://example.com/owner/model")
    except ValueError as exc:
        assert "huggingface.co" in str(exc)
    else:
        raise AssertionError("Non-HF URL must be rejected")

def test_task_inference():
    assert infer_task(None, {"architectures":["Qwen2ForCausalLM"]}) == "text-generation"
    assert infer_task("sentiment-analysis", {}) == "text-classification"

def test_text_samples():
    prompt, kwargs = sample_for("text-generation")
    assert prompt and kwargs["do_sample"] is False

def test_benchmark_math():
    assert percentile([1,2,3,4],.5)==2.5
    assert improvement_percent(10,8,True)==20
    assert improvement_percent(10,12,False)==20

def test_canonical_output_ignores_batch_container():
    one=[{"generated_text":"same"}]
    batch=[[{"generated_text":"same"}],[{"generated_text":"same"}]]
    assert canonical_output(one)==canonical_output(batch)=="same"

def test_quality_metrics():
    df=pd.DataFrame([
        {"test_id":"1","category":"x","expected":"OK","observed":"OK","metric":"exact"},
        {"test_id":"2","category":"x","expected":"Paris","observed":"Answer: Paris","metric":"contains"},
        {"test_id":"3","category":"x","expected":"","observed":"{\"ok\":true}","metric":"json_valid"},
    ])
    _,summary=quality_evaluate(df)
    assert summary["pass_rate_percent"]==100

def test_diagnosis_and_capacity():
    assert diagnose("CUDA out of memory")[0]["category"]=="Capacity"
    result=capacity(100,0.6,1.0,100,1.0,190)
    assert result["required_requests_per_second"]==1.0
    assert result["estimated_devices"]>=1

def test_security_review():
    _,score,decision=security_review({"resolved_revision":"abc","license":"apache-2.0","remote_code_detected":False,"gated":False,"weight_size_gb":1})
    assert score==100 and decision=="PASS"

def test_executive_summary_keeps_missing_evidence_open():
    result=executive_summary({})
    assert result["verdict"]=="HOLD"
    assert result["blocking_open"]>0

def test_executive_summary_accepts_streamlit_none_state():
    result=executive_summary({"security_score":None,"preflight":None,"result":None})
    security=result["gates"].loc[result["gates"].Gate=="Security"].iloc[0]
    assert security.Status=="NOT_CHECKED" and security.Evidence=="—/100"

def test_private_analytics_counts_anonymous_session(tmp_path):
    db=str(tmp_path/"analytics.db")
    record_visit("anon-1",db,100);record_visit("anon-1",db,110)
    result=analytics_summary(db)
    assert result["sessions"]==1 and result["page_views"]==2
    assert result["avg_duration_seconds"]==10

def test_news_grouping_and_deduplication():
    assert category("New agent model released")=="Agents & Applications"
    items=[{"title":"Same story"},{"title":"Same story!"},{"title":"Other"}]
    assert len(deduplicate(items))==2
