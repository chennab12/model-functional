# Qwen2.5-0.5B-Instruct Functional Verifier

A GitHub-ready, evidence-producing Python agent and Streamlit dashboard for verifying whether Qwen/Qwen2.5-0.5B-Instruct is functional.

## Actual result included

The bundled evidence was produced by loading and running the real model on CPU—not by mocking an answer.

- **Functional smoke test:** PASS
- **Model revision:** 7ae557604adf67be50417f59c2c2f167def9a775
- **Parameters loaded:** 494,032,768
- **Exact instruction:** PASS (FUNCTIONAL_OK)
- **Arithmetic contract:** FAIL (model said 98%; expected 95%)
- **JSON semantics:** PASS
- **Strict JSON without Markdown:** FAIL
- **Verdict:** functional with quality caveats; not production-qualified

Results are specific to the recorded environment, model revision and prompts.

## Dashboard tabs

1. Annotated verification workflow with sources
2. Actual prompts, outputs, timing and assertions
3. One-click live rerun
4. TPM gist, risks and production launch gates

## Run locally

Python 3.10–3.12 is recommended.

~~~bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_verification.py
streamlit run app.py
~~~

The first execution downloads roughly 1 GB of public model files and may use around 2 GB process memory. No API key is required.

## Deploy to Streamlit Community Cloud

Push the extracted files to GitHub, create an app at https://share.streamlit.io/, and select app.py. Community Cloud resource limits may be too small for reliable live inference. The bundled evidence dashboard still works; local or larger hosted compute is recommended for reruns.

## Test without downloading the model

~~~bash
pytest -q
python -m compileall -q app.py qwen_verifier tests
~~~

## Agent workflow

The verifier resolves the model identity, loads tokenizer and weights, applies the official chat template, performs deterministic inference, validates strict contracts, captures environment/timing/memory evidence, separates functional status from production readiness, and emits portable JSON.

## Key references

- [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Hugging Face Qwen2 documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2)
- [Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [PyTorch inference mode](https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)

## License

MIT. The Qwen model has its own Apache-2.0 license shown on its model card.

