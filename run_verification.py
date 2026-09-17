"""CLI: run the actual model and write machine-readable evidence."""
import argparse
import json
from pathlib import Path
from qwen_verifier import QwenFunctionalVerifier

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/latest_verification.json")
    args = parser.parse_args()
    report = QwenFunctionalVerifier().run()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["assessment"]["functional_smoke_test"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())

