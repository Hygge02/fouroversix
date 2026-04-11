# SGLang Integration

This repository includes an experimental path for serving `AWQ + fouroversix`
weight-only Qwen3 checkpoints with SGLang's Transformers backend.

Important: this path does **not** reuse the `transformers` built-in
`fouroversix` quantizer, because that implementation is weight-activation
quantization rather than the weight-only format used here.

## Export a local model directory

```bash
source .venv/bin/activate

python -m scripts.sglang.export_qwen3_fouroversix_awq_weight_only \
  --model-name Qwen/Qwen3-8B \
  --output-dir ./artifacts/qwen3-8b-awq-fouroversix-weight-only \
  --trust-remote-code
```

This export path reuses the existing AWQ cache if available at:

`ptq/awq_weight_only/Qwen/Qwen3-8B/b77f559d718cd0a44b6de35a71bc93f67f1d704a1b6a3365103650f17a3ae484`

## Verified launch command

The integration was validated against a local SGLang source tree at
`/root/sglang-src` using the Transformers backend.

```bash
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=''
export SGLANG_USE_CPU_ENGINE=1
export PYTHONPATH=/root/sglang-src/python

python -m sglang.launch_server \
  --model-path ./artifacts/qwen3-8b-awq-fouroversix-weight-only \
  --model-impl transformers \
  --trust-remote-code \
  --device cpu \
  --attention-backend torch_native \
  --sampling-backend pytorch \
  --grammar-backend none \
  --skip-server-warmup \
  --host 127.0.0.1 \
  --port 30011
```

The CPU launch above was used because the current machine reports stale
driver-level GPU memory usage on all four `L20A` GPUs.

## Verified smoke test

After the server reports readiness, the following requests were verified:

```bash
curl http://127.0.0.1:30011/get_model_info
curl http://127.0.0.1:30011/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/root/fouroversix/artifacts/qwen3-8b-awq-fouroversix-weight-only",
    "prompt": "2+2=",
    "max_tokens": 4,
    "temperature": 0
  }'
```

The server loaded the exported architecture
`Qwen3ForCausalLMFourOverSixAWQWeightOnly` and returned a completion through
the OpenAI-compatible endpoint.

## Local SGLang source tweaks used for validation

The validation above used a local SGLang checkout at `/root/sglang-src` with
two small compatibility changes:

- `python/sglang/srt/mem_cache/memory_pool_host.py`
  makes `sgl_kernel.kvcacheio` an optional import so CPU Transformers fallback
  startup is not blocked when hierarchical cache is disabled.
- `python/sglang/srt/model_executor/model_runner.py`
  skips optional `sgl_kernel` CPU thread/allreduce setup when those custom ops
  are unavailable.

## Current limitations

- This is a weight-only compatibility path that performs dequantize-plus-matmul
  in the custom linear module. It is not a native fused SGLang kernel path.
- The custom quantized linear intentionally does not subclass `nn.Linear`, so
  SGLang's Transformers fallback will preserve it instead of rewriting it into
  SGLang linear layers.
- Validation is currently on CPU. Once the GPU driver state is healthy again,
  the next step is to validate the same exported model on the intended multi-GPU
  serving path.
