"""CLI: preflight and optionally run any supported Hugging Face model."""
import argparse
import json
from pathlib import Path
from qwen_verifier import GenericModelVerifier

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--token")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", default="artifacts/latest_verification.json")
    args = parser.parse_args()
    agent = GenericModelVerifier(
        args.model, token=args.token, revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    report = agent.preflight()
    if not args.preflight_only:
        report = agent.run(report)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    passed = report.get("decision") == "READY_TO_RUN" if args.preflight_only else report.get("verdict") == "FUNCTIONAL"
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
