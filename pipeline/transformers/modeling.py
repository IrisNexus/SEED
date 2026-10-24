import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


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


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
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
