import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


def _prefer_local_sglang_source():
    """Prepend this pipeline's bundled sglang-src/python to sys.path when enabled."""
    if not bool(getattr(config, "SGLANG_USE_LOCAL_SOURCE", True)):
        return
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_sglang_python = os.path.join(current_dir, "sglang-src", "python")
    local_sglang_python = os.path.abspath(local_sglang_python)
    if os.path.isdir(local_sglang_python) and local_sglang_python not in sys.path:
        sys.path.insert(0, local_sglang_python)

    stale_module_names = []
    for name, module in list(sys.modules.items()):
        if name != "sglang" and not name.startswith("sglang."):
            continue
        module_file = os.path.abspath(getattr(module, "__file__", "") or "")
        if not module_file.startswith(local_sglang_python + os.sep):
            stale_module_names.append(name)
    for name in stale_module_names:
        sys.modules.pop(name, None)


def _assert_local_sglang_module(sgl):
    if not bool(getattr(config, "SGLANG_USE_LOCAL_SOURCE", True)):
        return
    current_dir = os.path.dirname(os.path.abspath(__file__))
    expected_root = os.path.abspath(os.path.join(current_dir, "sglang-src", "python"))
    actual_file = os.path.abspath(getattr(sgl, "__file__", "") or "")
    if not actual_file.startswith(expected_root + os.sep):
        raise RuntimeError(
            "SGLang import is not using the bundled local source: "
            f"expected under {expected_root}, got {actual_file}"
        )


def _resolve_model_dtype():
    requested = str(getattr(config, "MODEL_DTYPE", "bfloat16")).lower().strip()
    if requested == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    if requested == "bfloat16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print("[modeling] CUDA device does not support bf16, fallback to float16.")
            return torch.float16, "float16"
        return torch.bfloat16, "bfloat16"
    if requested == "float16":
        return torch.float16, "float16"
    if requested == "float32":
        return torch.float32, "float32"
    print(f"[modeling] Unknown MODEL_DTYPE='{requested}', fallback to bfloat16/float16 auto.")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    return torch.float16, "float16"


def _load_sglang_engine():
    _prefer_local_sglang_source()
    import sglang as sgl

    _assert_local_sglang_module(sgl)

    engine_kwargs = {
        "model_path": config.MODEL_NAME,
        "tp_size": int(getattr(config, "SGLANG_TP_SIZE", 1)),
        "log_level": getattr(config, "SGLANG_LOG_LEVEL", "error"),
        "disable_overlap_schedule": bool(
            getattr(config, "SGLANG_DISABLE_OVERLAP_SCHEDULE", True)
        ),
        "enable_custom_logit_processor": bool(
            getattr(config, "SGLANG_ENABLE_CUSTOM_LOGIT_PROCESSOR", True)
        ),
        "enable_return_hidden_states": bool(
            getattr(config, "SGLANG_ENABLE_RETURN_HIDDEN_STATES", True)
        ),
        "enable_return_attention_signal": bool(
            getattr(config, "SGLANG_ENABLE_RETURN_ATTENTION_SIGNAL", False)
        ),
        "trust_remote_code": bool(getattr(config, "SGLANG_TRUST_REMOTE_CODE", False)),
    }
    attention_backend = str(getattr(config, "SGLANG_ATTENTION_BACKEND", "")).strip()
    if attention_backend:
        engine_kwargs["attention_backend"] = attention_backend
    prefill_attention_backend = str(
        getattr(config, "SGLANG_PREFILL_ATTENTION_BACKEND", "")
    ).strip()
    if prefill_attention_backend:
        engine_kwargs["prefill_attention_backend"] = prefill_attention_backend
    decode_attention_backend = str(
        getattr(config, "SGLANG_DECODE_ATTENTION_BACKEND", "")
    ).strip()
    if decode_attention_backend:
        engine_kwargs["decode_attention_backend"] = decode_attention_backend
    mem_fraction_static = float(getattr(config, "SGLANG_MEM_FRACTION_STATIC", 0.8))
    if mem_fraction_static > 0:
        engine_kwargs["mem_fraction_static"] = mem_fraction_static
    random_seed = getattr(config, "SGLANG_RANDOM_SEED", None)
    if random_seed is not None:
        engine_kwargs["random_seed"] = int(random_seed)

    print(
        "[modeling] loading sglang engine "
        f"(tp_size={engine_kwargs['tp_size']}, "
        f"disable_overlap_schedule={engine_kwargs['disable_overlap_schedule']})"
    )
    return sgl.Engine(**engine_kwargs)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=bool(getattr(config, "SGLANG_TRUST_REMOTE_CODE", False)),
    )
    backend = str(getattr(config, "GENERATION_BACKEND", "hf")).lower().strip()

    if backend == "sglang":
        engine = _load_sglang_engine()
        return tokenizer, engine

    if backend != "hf":
        raise ValueError(
            f"Unsupported GENERATION_BACKEND={backend!r}. Expected 'hf' or 'sglang'."
        )

    model_dtype, model_dtype_name = _resolve_model_dtype()
    print(f"[modeling] loading model with dtype={model_dtype_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=model_dtype,
        device_map="auto",
        # Force eager attention so output_attentions is consistently available.
        attn_implementation="eager",
    )
    return tokenizer, model
