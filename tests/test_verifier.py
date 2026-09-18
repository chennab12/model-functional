from qwen_verifier.verifier import normalize_model_id, infer_task, sample_for
from qwen_verifier.lifecycle import canonical_output,improvement_percent,percentile

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
