#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGLANG_PYTHON="${PROJECT_ROOT}/pipeline/sglang/sglang-src/python"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${SGLANG_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

SEED_CONDA_ENV="${SEED_CONDA_ENV:-seed}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${SEED_CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
  exec conda run -n "${SEED_CONDA_ENV}" python -m pipeline.sglang.main "$@"
fi

python -m pipeline.sglang.main "$@"
