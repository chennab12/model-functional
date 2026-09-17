"""Evidence-producing functional verifier for Qwen2.5-0.5B-Instruct."""
from __future__ import annotations
import json
import platform
import resource
import time
from dataclasses import dataclass
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

@dataclass(frozen=True)
class TestCase:
    name: str
    prompt: str
    max_new_tokens: int = 48

CASES = (
    TestCase("Instruction following", "Return exactly this text and nothing else: FUNCTIONAL_OK"),
    TestCase("Arithmetic quality", "A deployment has 120 requests and 114 succeed. What is the success rate? Answer with only the percentage."),
    TestCase("Structured output", "Return valid JSON with keys model and status, using values Qwen2.5-0.5B-Instruct and functional."),
)

def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return None

def evaluate_outputs(tests: list[dict[str, Any]]) -> dict[str, bool]:
    by_name = {t["case"]: t["output"].strip() for t in tests}
    structured = _extract_json(by_name.get("Structured output", ""))
    return {
        "exact_instruction": by_name.get("Instruction following") == "FUNCTIONAL_OK",
        "arithmetic_95_percent": "95%" in by_name.get("Arithmetic quality", ""),
        "json_semantics": bool(structured and structured.get("model") == "Qwen2.5-0.5B-Instruct" and structured.get("status") == "functional"),
        "strict_json_no_markdown": by_name.get("Structured output", "").lstrip().startswith("{"),
    }

def classify_report(report: dict[str, Any]) -> dict[str, Any]:
    smoke = report.get("model_loaded", False) and report.get("generated_nonempty", False)
    checks = report.get("checks", {})
    passed = sum(bool(v) for v in checks.values())
    return {
        "functional_smoke_test": "PASS" if smoke else "FAIL",
        "strict_quality_gates": f"{passed}/{len(checks)} passed" if checks else "not run",
        "production_ready": False,
        "verdict": "FUNCTIONAL WITH QUALITY CAVEATS" if smoke else "NOT FUNCTIONAL",
    }

class QwenFunctionalVerifier:
    """Load, infer, validate, classify and return portable JSON evidence."""
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        self.model_id, self.device = model_id, device

    def run(self) -> dict[str, Any]:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer_seconds = time.perf_counter() - started
        model_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="auto")
        model.eval()
        model_seconds = time.perf_counter() - model_started
        results = []
        for case in CASES:
            messages = [
                {"role": "system", "content": "You are a precise test assistant. Follow the requested output format."},
                {"role": "user", "content": case.prompt},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([rendered], return_tensors="pt")
            t0 = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=case.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            seconds = time.perf_counter() - t0
            new_tokens = generated[0, inputs.input_ids.shape[1]:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            count = int(new_tokens.shape[0])
            results.append({
                "case": case.name, "prompt": case.prompt, "output": output,
                "input_tokens": int(inputs.input_ids.shape[1]), "output_tokens": count,
                "seconds": round(seconds, 3),
                "tokens_per_second": round(count / seconds, 2) if seconds else None,
            })
        report = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": self.model_id,
            "revision": getattr(model.config, "_commit_hash", None),
            "model_loaded": True,
            "generated_nonempty": all(bool(t["output"]) for t in results),
            "checks": evaluate_outputs(results),
            "environment": {
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "device": self.device,
                "platform": platform.platform(),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            },
            "timing_seconds": {
                "tokenizer_load": round(tokenizer_seconds, 3),
                "model_load": round(model_seconds, 3),
                "total_generation": round(sum(t["seconds"] for t in results), 3),
            },
            "tests": results,
        }
        report["assessment"] = classify_report(report)
        return report
