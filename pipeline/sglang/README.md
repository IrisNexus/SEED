# SGLang Runtime

This runtime uses `sglang.Engine` and the bundled SGLang Python source under
`sglang-src/python`.

## Run

From the repository root:

```bash
export MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sglang.sh qwen7b/hcot.json false 50 true
```

Equivalent command from the repository root:

```bash
PYTHONPATH=.:pipeline/sglang/sglang-src/python \
  python -m pipeline.sglang.main qwen7b/hcot.json false 50 true
```

## Configuration

Configuration lives in `config.py`.

- `GENERATION_BACKEND=sglang` by default.
- `SGLANG_USE_LOCAL_SOURCE=true` by default, so the bundled SGLang source is used.

The runner saves attack responses, branch selections, and attention traces
without an external judge.
