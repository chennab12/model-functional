from qwen_verifier.verifier import evaluate_outputs, classify_report

def sample(output2="95%", output3='{"model":"Qwen2.5-0.5B-Instruct","status":"functional"}'):
    return [
        {"case":"Instruction following","output":"FUNCTIONAL_OK"},
        {"case":"Arithmetic quality","output":output2},
        {"case":"Structured output","output":output3},
    ]

def test_all_contracts_pass():
    checks=evaluate_outputs(sample())
    assert all(checks.values())

def test_markdown_json_is_semantically_valid_but_not_strict():
    checks=evaluate_outputs(sample(output3='```json\n{"model":"Qwen2.5-0.5B-Instruct","status":"functional"}\n```'))
    assert checks["json_semantics"] is True
    assert checks["strict_json_no_markdown"] is False

def test_functional_is_not_same_as_production_ready():
    report={"model_loaded":True,"generated_nonempty":True,"checks":{"a":True,"b":False}}
    result=classify_report(report)
    assert result["functional_smoke_test"]=="PASS"
    assert result["production_ready"] is False
