# Transformers Runtime

This runtime uses HuggingFace Transformers.

## Run

From the repository root:

```bash
export MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
bash scripts/run_transformers.sh qwen7b/hcot.json false 50 true
```

Equivalent command from the repository root:

```bash
PYTHONPATH=. python -m pipeline.transformers.main qwen7b/hcot.json false 50 true
```
