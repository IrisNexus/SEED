# Pipeline

This folder contains the runtime implementations used by SEED.

## `transformers/`

Reference implementation using HuggingFace Transformers.

Entrypoint:

```bash
python -m pipeline.transformers.main <dataset> <verbose> <max_samples> <include_original_response>
```

## `sglang/`

SGLang implementation using an in-process `sglang.Engine`. It includes a local
SGLang Python source tree at:

```text
pipeline/sglang/sglang-src/python
```

The bundled source is used by default.

Entrypoint:

```bash
python -m pipeline.sglang.main <dataset> <verbose> <max_samples> <include_original_response>
```

Both entrypoints expect to be run with the repository root on `PYTHONPATH`. The scripts in
`scripts/` set this up automatically.
