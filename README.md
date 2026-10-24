# SEED

Generation pipelines for attack prompts using Transformers or SGLang.

## Layout

- `pipeline/transformers/`: Transformers runtime.
- `pipeline/sglang/`: SGLang runtime and bundled source.
- `dataset/`: attack prompts grouped by model.
- `scripts/`: runtime launchers.

## Run

Use a Python environment with compatible CUDA, PyTorch, Transformers, and SGLang
dependencies. Package versions are listed in `requirements.txt`.

```bash
export MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
bash scripts/run_transformers.sh qwen7b/hcot.json false 50 false
bash scripts/run_sglang.sh qwen7b/hcot.json false 50 false
```

The SGLang launcher uses the `seed` conda environment by default. Set
`SEED_CONDA_ENV` to select a different environment.

Arguments: `dataset_name verbose max_samples include_original_response`.
Set the last argument to `true` to also save a baseline response.
`max_samples` is an upper limit, capped by the number of available samples.

Attack datasets: AIM, DRattack, GCG, H-CoT, PAIR, and ReNeLLM. Model folders:
`qwen7b`, `qwen38`, and `llama`.

Results default to `qwen7b/run_<timestamp>/`. Set `RESULTS_DIR_NAME` to change
the output directory. Runtime parameters are in each pipeline's `config.py`.
