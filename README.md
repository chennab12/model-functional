# Universal Hugging Face Model Compatibility & Functional Verifier

A GitHub-ready Python agent and Streamlit dashboard that accepts a Hugging Face model URL or owner/model identifier, checks compatibility first, and only then performs a task-aware functional smoke test.

The dashboard now covers the full verification lifecycle in separate tabs:

1. Compatibility preflight
2. Functional verification
3. Portability
4. Performance benchmarking
5. Controlled optimization
6. Annotated guide
7. Included evidence

The **Qualification workbench** contains ten additional nested tabs without
overcrowding the primary lifecycle navigation:

1. **Quality** — golden-set exact, contains, regex and JSON-validity gates
2. **Scorecard** — executive blocking/non-blocking qualification decision
3. **History** — capture and compare session runs
4. **vLLM** — securely probe an existing OpenAI-compatible endpoint
5. **Reproduce** — manifest and downloadable evidence ZIP
6. **Diagnose** — error classification and actionable remediation
7. **Security** — license, revision, remote-code and artifact readiness screen
8. **Capacity** — workload, cost and power planning estimates
9. **B70** — Intel Arc Pro B70 qualification checklist
10. **CI/CD** — downloadable GitHub Actions qualification workflow

The Functional tab also annotates every key step with its Intel Arc Pro B70
difference: artifact handling, XPU runtime, device selection, precision,
input parity, synchronization, quality comparison and evidence capture.

## Why two phases?

A model should not be downloaded or executed blindly. The compatibility agent first inspects small Hub metadata and config.json, detects blockers, and records a decision. Large weights are loaded only when the model and host appear compatible.

## Preflight checks

- URL/repository identifier validation
- Public, private, missing or gated access
- Resolved immutable commit revision
- Transformers config.json availability and parsing
- Standard weight file availability and estimated size
- Pipeline task and safe test-adapter support
- Remote custom-code detection
- Available disk, CPU RAM, CUDA and GPU memory
- Estimated disk and CPU-memory needs
- Actionable blocker remediation

HF tokens are held in memory and are never written to reports.

## Automatically tested tasks

- Text generation and text-to-text generation
- Summarization and translation
- Fill-mask
- Text and token classification
- Question answering
- Feature extraction
- Image classification
- Audio classification and automatic speech recognition

Models such as diffusion pipelines, GGUF-only repositories, adapters, multimodal chat systems and custom research architectures can need a model-specific adapter. They are reported as unsupported—not incorrectly classified as broken.

## Run locally

~~~bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
~~~

Paste a model link into the sidebar, run preflight, review every check, and then run the functional test if enabled.

## Command line

Preflight without downloading weights:

~~~bash
python run_verification.py https://huggingface.co/google/flan-t5-small --preflight-only
~~~

Preflight plus real functional inference:

~~~bash
python run_verification.py Qwen/Qwen2.5-0.5B-Instruct
~~~

Pinned revision and private/gated model:

~~~bash
python run_verification.py owner/model --revision COMMIT_SHA --token YOUR_READ_TOKEN
~~~

Only use --trust-remote-code after reviewing repository code:

~~~bash
python run_verification.py owner/custom-model --trust-remote-code
~~~

## Test the package

~~~bash
pytest -q
python -m compileall -q app.py qwen_verifier tests run_verification.py
~~~

## Important interpretation

- **READY_TO_RUN** means the preflight found no known blocker.
- **FUNCTIONAL** means the model loaded and returned a non-empty result for a small task-aware input.
- Neither result proves production readiness.
- Production qualification also needs representative accuracy, safety, reliability, concurrency, latency, throughput and cost testing.

## Portability, benchmarking and optimization

The portability tab builds a backend matrix for CPU, NVIDIA CUDA, Intel XPU,
Apple MPS, vLLM and OpenVINO. A detected backend is only readiness; portability
is proven only after the same pinned revision and acceptance input run there.

The benchmark tab separates model-load time from warmed steady-state inference
and reports mean, p50, p95, p99, requests/second, approximate tokens/second and
per-iteration evidence.

The optimization tab runs a controlled baseline-versus-static-batching
experiment with identical model, revision, input and output-token cap. It
reports throughput change and preserves an output fingerprint. The fingerprint
is an initial guard, not a replacement for a task-specific golden set.

Every phase includes a per-step Intel Arc Pro B70 column. On B70:

- use the Intel GPU driver and an XPU-enabled upstream PyTorch build;
- require `torch.xpu.is_available()`;
- use `xpu`, not CUDA, as the device;
- synchronize `torch.xpu` before and after timed regions;
- validate BF16/FP16 before adopting lower precision;
- increase batching within the card's 32 GB VRAM;
- record driver, runtime, device, dtype, VRAM and quality evidence;
- use the Intel-XPU vLLM build or container for vLLM serving tests.

Intel Extension for PyTorch is not required by this package; current Intel
functionality is expected through upstream PyTorch.

## References

- [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
- [Transformers pipeline guide](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial)
- [Custom models and remote code](https://huggingface.co/docs/transformers/main/en/custom_models)

The package includes the real Qwen2.5-0.5B-Instruct evidence from the original verifier as an example baseline.
