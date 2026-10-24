import math
import re
import json

import torch

from . import config
from .logger import log_print


def _debug_enabled():
    return bool(getattr(config, "DEBUG_VERBOSE", False))


def _debug_log(category, message):
    if _debug_enabled():
        log_print(f"[DEBUG][{category}] {message}")


def _debug_block(category, title, fields=None, body=None):
    if not _debug_enabled():
        return
    width = 88
    log_print(f"[DEBUG][{category}] {'=' * 8} {title} {'=' * 8}")
    for key, value in fields or []:
        log_print(f"[DEBUG][{category}]   {key:<28}: {value}")
    if body:
        log_print(f"[DEBUG][{category}]   {'-' * (width - 18)}")
        for line in str(body).splitlines() or [""]:
            log_print(f"[DEBUG][{category}]   {line}")
    log_print(f"[DEBUG][{category}] {'=' * width}")


def _debug_snippet(text, max_chars=None):
    if text is None:
        return ""
    if max_chars is None:
        max_chars = int(getattr(config, "DEBUG_SNIPPET_CHARS", 600))
    text = str(text).replace("\r", "\\r")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def _debug_token_details(tokenizer, tokens, positions):
    seq = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
    details = []
    for pos in positions:
        try:
            pos_int = int(pos)
        except (TypeError, ValueError):
            continue
        if pos_int < 0 or pos_int >= len(seq):
            details.append(
                {
                    "position": pos_int,
                    "token_id": None,
                    "token_text": "<out_of_range>",
                }
            )
            continue
        token_id = int(seq[pos_int])
        details.append(
            {
                "position": pos_int,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
            }
        )
    return details


def softmax(logits):
    return torch.nn.functional.softmax(logits, dim=-1)


def entropy_nats(probs):
    probs = probs.float()
    mask = probs > 1e-10
    return -(probs[mask] * torch.log(probs[mask])).sum().item()


def _build_option_token_candidates(tokenizer):
    token_ids = set()
    for letter in ("A", "B", "C", "D"):
        for variant in (letter, f" {letter}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(int(ids[0]))
    return sorted(token_ids)


def _constrain_logits_to_allowed(logits, allowed_token_ids):
    if not allowed_token_ids:
        return logits
    constrained = torch.full_like(logits, -float("inf"))
    constrained[:, allowed_token_ids] = logits[:, allowed_token_ids]
    return constrained


def _sample_next_token_from_logits(logits_1d, force_greedy=False):
    """
    Sample next token from a 1D logits tensor using configurable decoding params.
    Falls back to greedy decoding when sampling is disabled.
    """
    if force_greedy or (not getattr(config, "DECODE_USE_SAMPLING", False)):
        return int(torch.argmax(logits_1d).item())

    temperature = float(getattr(config, "DECODE_TEMPERATURE", 1.0))
    top_p = float(getattr(config, "DECODE_TOP_P", 1.0))
    top_k = int(getattr(config, "DECODE_TOP_K", 0))

    if temperature <= 0:
        return int(torch.argmax(logits_1d).item())

    work_logits = logits_1d.float() / temperature
    probs = torch.nn.functional.softmax(work_logits, dim=-1)

    if top_k > 0 and top_k < probs.shape[-1]:
        topk_vals, topk_idx = torch.topk(probs, top_k)
        filtered = torch.zeros_like(probs)
        filtered[topk_idx] = topk_vals
        probs = filtered

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative <= top_p
        if keep_mask.numel() > 0:
            keep_mask[0] = True
        filtered_sorted = sorted_probs * keep_mask
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx] = filtered_sorted
        probs = filtered

    denom = probs.sum()
    if float(denom.item()) <= 0.0:
        return int(torch.argmax(logits_1d).item())

    probs = probs / denom
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


def _extract_direct_option_letter(text):
    if not text:
        return ""
    cleaned = text.strip().upper()
    match = re.match(r"^[\(\[]?\s*([ABCD])\s*[\)\].,:;!\?]*\s*$", cleaned)
    if match:
        return match.group(1)
    return ""


def is_sentence_boundary(tokens, tokenizer):
    if len(tokens) == 0:
        return True

    if isinstance(tokens, torch.Tensor):
        tokens_list = tokens.tolist()
    else:
        tokens_list = list(tokens)

    recent_tokens = tokens_list[-10:] if len(tokens_list) >= 10 else tokens_list
    recent_text = tokenizer.decode(recent_tokens, skip_special_tokens=False)
    recent_text_stripped = recent_text.rstrip()

    for ending in config.SENTENCE_ENDINGS:
        if recent_text_stripped.endswith(ending):
            return True

    if recent_text_stripped == "":
        return True

    return False


def find_last_sentence_boundary_in_generated(tokens_main, initial_token_count, tokenizer):
    if isinstance(tokens_main, torch.Tensor):
        tokens_list = tokens_main.tolist()
    else:
        tokens_list = list(tokens_main)

    generated_count = len(tokens_list) - initial_token_count
    if generated_count <= 0:
        return initial_token_count, False

    for k in range(generated_count, 0, -1):
        span = tokens_list[initial_token_count : initial_token_count + k]
        text = tokenizer.decode(span, skip_special_tokens=True)
        text_stripped = text.rstrip()
        for ending in config.SENTENCE_ENDINGS:
            if text_stripped.endswith(ending):
                return initial_token_count + k, True
    return initial_token_count, False


def should_split(probs, split_count=0):
    h_val = entropy_nats(probs)
    trigger_reason = None
    entropy_threshold = config.get_entropy_threshold(split_count)
    if h_val > entropy_threshold:
        trigger_reason = f"[redacted] (H={h_val:.4f} > {entropy_threshold})"

    if not trigger_reason:
        return False, None, entropy_threshold
    return True, trigger_reason, entropy_threshold


def log_trigger_details(
    probs,
    trigger_reason,
    current_tokens,
    initial_token_count,
    tokenizer,
    token_sets,
    entropy_threshold=None,
):
    h_val = entropy_nats(probs)
    topk_vals, topk_idx = torch.topk(probs, config.TOP_K)
    top_probs = topk_vals.tolist()

    log_print("\n" + "=" * 80)
    log_print(f"🔀 [redacted][redacted][redacted]: {trigger_reason}")
    log_print("-" * 80)

    if current_tokens is not None:
        if isinstance(current_tokens, torch.Tensor):
            tokens_list = current_tokens.tolist()
        else:
            tokens_list = list(current_tokens)
        generated_token_count = len(tokens_list) - initial_token_count
        generated_text = tokenizer.decode(tokens_list[initial_token_count:], skip_special_tokens=True)
        log_print("📍 [redacted][redacted]")
        log_print(f"   - [redacted] token [redacted]: {generated_token_count}")
        log_print(
            f"   - Token [redacted]: {len(tokens_list)} ([redacted]tokens) / {generated_token_count} ([redacted])"
        )
        log_print("   - [redacted]:")
        context_text = "..." + generated_text[-80:] if len(generated_text) > 80 else generated_text
        log_print(f"     '{context_text}'")
        log_print("-" * 80)

    log_print("📊 Top-10 [redacted][redacted]")
    for i in range(min(10, len(top_probs))):
        token_id = topk_idx[i].item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_text = repr(token_text)
        prob = top_probs[i]
        is_harmful = "⚠️" if token_id in token_sets["HARMFUL_TOKEN_IDS"] else "  "
        is_reflection = "⭐" if token_id in token_sets["REFLECTION_TOKEN_IDS"] else "  "
        marker = is_harmful if is_harmful.strip() else is_reflection
        log_print(f"{marker} Top-{i+1:2d}: prob={prob:.6f} | token_id={token_id:6d} | text={token_text}")
    threshold_text = (
        f"{float(entropy_threshold):.4f}"
        if isinstance(entropy_threshold, (int, float))
        else "N/A"
    )
    log_print(f"   [redacted] H = {h_val:.6f} ([redacted]: >{threshold_text})")
    log_print("=" * 80 + "\n")


def _match_inserted_prefix_len(tokens_safety, parallel_start_len, inserted_token_ids):
    if not inserted_token_ids:
        return 0
    safety_segment = tokens_safety[parallel_start_len:]
    safety_segment_list = (
        safety_segment.tolist() if isinstance(safety_segment, torch.Tensor) else list(safety_segment)
    )
    max_len = min(len(safety_segment_list), len(inserted_token_ids))
    matched = 0
    for i in range(max_len):
        if int(safety_segment_list[i]) != int(inserted_token_ids[i]):
            break
        matched += 1
    return matched


def _candidate_attn_to_inserted_span(
    model,
    tokens_safety,
    candidate_id,
    inserted_span_start,
    inserted_span_end,
    device,
    return_reason=False,
):
    if inserted_span_end <= inserted_span_start:
        if return_reason:
            return 0.0, "invalid_span"
        return 0.0

    candidate_tensor = torch.tensor([candidate_id], device=device, dtype=tokens_safety.dtype)
    candidate_tokens = torch.cat([tokens_safety, candidate_tensor])
    out = model(candidate_tokens.unsqueeze(0), output_attentions=True)
    attentions = out.attentions
    if not attentions:
        if return_reason:
            return 0.0, "no_attentions"
        return 0.0

    # last layer attention: [batch, heads, query_len, key_len]
    last_layer_attn = attentions[-1][0]
    query_idx = last_layer_attn.shape[1] - 1
    span_attn = last_layer_attn[:, query_idx, inserted_span_start:inserted_span_end]
    if span_attn.numel() == 0:
        if return_reason:
            return 0.0, "empty_span"
        return 0.0
    score = float(span_attn.mean().item())
    if return_reason:
        return score, "ok"
    return score


def _candidate_attn_per_inserted_token(
    model,
    tokens_safety,
    candidate_id,
    inserted_span_start,
    inserted_span_end,
    device,
):
    if inserted_span_end <= inserted_span_start:
        return [], "invalid_span"

    candidate_tensor = torch.tensor([candidate_id], device=device, dtype=tokens_safety.dtype)
    candidate_tokens = torch.cat([tokens_safety, candidate_tensor])
    out = model(candidate_tokens.unsqueeze(0), output_attentions=True)
    attentions = out.attentions
    if not attentions:
        return [], "no_attentions"

    # last layer attention: [batch, heads, query_len, key_len]
    last_layer_attn = attentions[-1][0]
    query_idx = last_layer_attn.shape[1] - 1
    span_attn = last_layer_attn[:, query_idx, inserted_span_start:inserted_span_end]
    if span_attn.numel() == 0:
        return [], "empty_span"

    # Mean across heads to keep one score per inserted token.
    per_token_scores = span_attn.mean(dim=0).detach().cpu().tolist()
    return [float(x) for x in per_token_scores], "ok"


def _select_attention_driven_next_token(
    model,
    tokenizer,
    tokens_safety,
    probs_b,
    parallel_start_len,
    inserted_token_ids,
    device,
):
    inserted_prefix_len = _match_inserted_prefix_len(
        tokens_safety, parallel_start_len, inserted_token_ids
    )
    if inserted_prefix_len <= 0:
        return None

    inserted_span_start = parallel_start_len
    inserted_span_end = parallel_start_len + inserted_prefix_len
    top_k = min(config.ATTN_TRACE_TOP_K, probs_b.shape[0])
    topk_vals_b, topk_idx_b = torch.topk(probs_b, top_k)

    best_record = None
    for rank in range(top_k):
        candidate_id = int(topk_idx_b[rank].item())
        attn_score = _candidate_attn_to_inserted_span(
            model,
            tokens_safety,
            candidate_id,
            inserted_span_start,
            inserted_span_end,
            device,
        )
        candidate_prob = float(topk_vals_b[rank].item())
        record = {
            "rank": rank + 1,
            "token_id": candidate_id,
            "token_text": tokenizer.decode([candidate_id], skip_special_tokens=False),
            "prob": candidate_prob,
            "attn_to_inserted_prefix": attn_score,
            "inserted_prefix_len": inserted_prefix_len,
        }
        if (
            best_record is None
            or record["attn_to_inserted_prefix"] > best_record["attn_to_inserted_prefix"]
        ):
            best_record = record

    return best_record


def _build_single_token_id_set(tokenizer, token_texts):
    token_ids = set()
    for text in token_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(int(ids[0]))
    return token_ids


def _is_punct_like_token(tokenizer, token_id, punct_token_ids):
    if int(token_id) in punct_token_ids:
        return True
    token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False).strip()
    if token_text == "":
        return False
    return all(ch in ".,?!;:，。！？；：…-—" for ch in token_text)


def _promptlen_attn_scale(prompt_token_len):
    beta = float(getattr(config, "ATTN_PROMPTLEN_SCALE_BETA", 0.0))
    p0 = float(getattr(config, "ATTN_PROMPTLEN_SCALE_P0", 1.0))
    p = max(0.0, float(prompt_token_len))
    if beta <= 0.0 or p0 <= 0.0:
        return 1.0
    return 1.0 + beta * math.log(1.0 + p / p0)


def _is_sglang_backend(runtime):
    return str(runtime.get("backend", "hf")).lower() == "sglang"


def _to_sglang_top_k(top_k):
    k = int(top_k)
    return -1 if k <= 0 else k


def _build_sglang_sampling_params(answer_only_mode=False):
    params = {
        "max_new_tokens": int(config.MAX_TOKENS),
        "temperature": float(config.DECODE_TEMPERATURE)
        if bool(getattr(config, "DECODE_USE_SAMPLING", False))
        else 0.0,
        "top_p": float(getattr(config, "DECODE_TOP_P", 1.0)),
        "top_k": _to_sglang_top_k(getattr(config, "DECODE_TOP_K", 0)),
        "ignore_eos": False,
    }
    if answer_only_mode:
        params["max_new_tokens"] = min(int(config.MAX_TOKENS), 8)
        params["temperature"] = 0.0
        params["regex"] = r"[\s\(\[]?[ABCD][\)\].,:;!\?]?"
    return params


def _to_float_or_none(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_list_value(values, idx, default=None):
    if isinstance(values, list) and 0 <= idx < len(values):
        return values[idx]
    return default


def _normalize_inserted_safety_phrase_text(text):
    if not isinstance(text, str) or not text:
        return ""
    phrase = str(getattr(config, "SORRY_ASSIST_PHRASE", "") or "").strip()
    if not phrase:
        return text

    normalized = text
    compact_phrase = re.sub(r"\s+", "", phrase)
    if compact_phrase:
        normalized = re.sub(re.escape(compact_phrase), phrase, normalized, flags=re.IGNORECASE)

    escaped_phrase = re.escape(phrase)
    normalized = re.sub(
        rf"(?:{escaped_phrase}\s*){{2,}}",
        phrase + " ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(rf"([A-Za-z])({escaped_phrase})", r"\1 \2", normalized)
    return normalized


def _postprocess_sglang_response_text(text, selected_path, answer_only_mode=False):
    out = str(text or "")

    if bool(getattr(config, "SGLANG_NORMALIZE_SAFETY_TEXT", False)):
        out = _normalize_inserted_safety_phrase_text(out)

    if (
        (not answer_only_mode)
        and str(selected_path) == "safety"
        and bool(getattr(config, "SGLANG_FORCE_CLEAN_REFUSAL_ON_SAFETY_PATH", False))
    ):
        clean_refusal = str(getattr(config, "SGLANG_CLEAN_REFUSAL_TEXT", "") or "").strip()
        if clean_refusal:
            out = clean_refusal

    return out.strip() if answer_only_mode else out


def _parse_history_json_from_meta(meta_info, key):
    raw = meta_info.get(key)
    if isinstance(raw, list) and raw:
        payload = raw[-1]
    elif isinstance(raw, str):
        payload = raw
    else:
        return []

    if not isinstance(payload, str):
        return []
    try:
        data = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _parse_int_csv(raw):
    if not raw:
        return None
    out = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except (TypeError, ValueError):
            continue
    if not out:
        return None
    return sorted(set(out))


def _build_sglang_attention_signal_request(prompt_token_len):
    if not bool(getattr(config, "SGLANG_ENABLE_RETURN_ATTENTION_SIGNAL", False)):
        return {}

    source_max = max(
        0, int(getattr(config, "SGLANG_ATTENTION_SIGNAL_SOURCE_MAX_TOKENS", 0))
    )
    if source_max <= 0 or prompt_token_len <= 0:
        return {}

    source_count = min(int(prompt_token_len), source_max)
    mode = str(getattr(config, "SGLANG_ATTENTION_SIGNAL_SOURCE_MODE", "head")).lower()
    if mode == "tail":
        start = int(prompt_token_len) - source_count
        source_positions = list(range(start, start + source_count))
    else:
        source_positions = list(range(source_count))

    step_max = max(
        0, int(getattr(config, "SGLANG_ATTENTION_SIGNAL_DECODE_MAX_STEPS", 0))
    )
    step_stride = max(
        1, int(getattr(config, "SGLANG_ATTENTION_SIGNAL_DECODE_STEP_STRIDE", 1))
    )
    if step_max <= 0:
        return {}
    decode_steps = list(range(0, step_max, step_stride))
    if not decode_steps:
        decode_steps = [0]

    payload = {
        "attention_signal_source_positions": source_positions,
        "attention_signal_decode_steps": decode_steps,
    }
    layers = _parse_int_csv(getattr(config, "SGLANG_ATTENTION_SIGNAL_LAYERS", ""))
    if layers is not None:
        payload["attention_signal_layers"] = layers
    return payload


def _build_sglang_prompt_ids(prompt, runtime):
    tokenizer = runtime["tokenizer"]
    system_prompt = config.build_system_prompt()
    if system_prompt == "":
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(formatted_prompt, add_special_tokens=False), formatted_prompt


def _sglang_meta_last(meta_info, key, default=None):
    value = meta_info.get(key, default)
    if isinstance(value, list) and value:
        return value[-1]
    return value


def _sglang_custom_processors():
    from sglang.srt.sampling.custom_logit_processor import (
        BeamEntropyTelemetryLogitProcessor,
        BeamForceTokenIdsLogitProcessor,
        BeamSafetyBranchLogitProcessor,
    )

    return {
        "entropy": BeamEntropyTelemetryLogitProcessor,
        "force": BeamForceTokenIdsLogitProcessor,
        "safety_branch": BeamSafetyBranchLogitProcessor,
    }


def _sglang_generation_params(max_new_tokens, custom_params=None):
    params = _build_sglang_sampling_params(answer_only_mode=False)
    params["max_new_tokens"] = max(0, int(max_new_tokens))
    if custom_params is not None:
        params["custom_params"] = custom_params
    return params


def _sglang_last_attention_layers(runtime):
    configured = _parse_int_csv(getattr(config, "SGLANG_ATTENTION_SIGNAL_LAYERS", ""))
    if configured is not None:
        return configured

    engine = runtime["model"]
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    model_config = getattr(tokenizer_manager, "model_config", None)
    layer_count = getattr(model_config, "num_attention_layers", None)
    if layer_count is None:
        layer_count = getattr(model_config, "num_hidden_layers", None)
    try:
        layer_count = int(layer_count)
    except (TypeError, ValueError):
        return None
    if layer_count <= 0:
        return None
    return [layer_count - 1]


def _sglang_attention_kwargs(runtime, source_positions, max_new_tokens):
    if not bool(getattr(config, "SGLANG_ENABLE_RETURN_ATTENTION_SIGNAL", False)):
        return {}
    try:
        positions = [int(x) for x in source_positions if int(x) >= 0]
    except (TypeError, ValueError):
        positions = []
    if not positions or int(max_new_tokens) <= 0:
        return {}

    step_cap = max(
        0, int(getattr(config, "SGLANG_ATTENTION_SIGNAL_DECODE_MAX_STEPS", 0))
    )
    if step_cap <= 0:
        return {}
    stride = max(
        1, int(getattr(config, "SGLANG_ATTENTION_SIGNAL_DECODE_STEP_STRIDE", 1))
    )
    steps = list(range(0, min(int(max_new_tokens), step_cap), stride))
    if not steps:
        return {}

    kwargs = {
        "attention_signal_source_positions": positions,
        "attention_signal_decode_steps": steps,
    }
    layers = _sglang_last_attention_layers(runtime)
    if layers is not None:
        kwargs["attention_signal_layers"] = layers
    return kwargs


def _sglang_generate_from_ids(
    input_ids,
    runtime,
    max_new_tokens,
    *,
    custom_processor=None,
    custom_params=None,
    return_logprob=False,
    top_logprobs_num=0,
    attention_source_positions=None,
):
    kwargs = {
        "input_ids": [int(x) for x in input_ids],
        "sampling_params": _sglang_generation_params(
            max_new_tokens=max_new_tokens,
            custom_params=custom_params,
        ),
    }
    if custom_processor is not None:
        kwargs["custom_logit_processor"] = custom_processor.to_str()
    if return_logprob:
        kwargs["return_logprob"] = True
        kwargs["logprob_start_len"] = -1
        kwargs["top_logprobs_num"] = int(top_logprobs_num)
    if attention_source_positions:
        kwargs.update(
            _sglang_attention_kwargs(
                runtime,
                attention_source_positions,
                max_new_tokens=max_new_tokens,
            )
        )

    out = runtime["model"].generate(**kwargs)
    if not isinstance(out, dict):
        return {"text": str(out), "output_ids": [], "meta_info": {}}
    return {
        "text": str(out.get("text", "")),
        "output_ids": [int(x) for x in (out.get("output_ids") or [])],
        "meta_info": out.get("meta_info", {}) or {},
    }


def _sglang_output_logprobs(meta_info):
    output = []
    for item in meta_info.get("output_token_logprobs", []) or []:
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            continue
    return output


def _sglang_top_candidate_ids(meta_info, fallback_ids):
    positions = meta_info.get("output_top_logprobs", []) or []
    first = positions[0] if positions else None
    candidate_ids = []
    if isinstance(first, list):
        for item in first:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                candidate_ids.append(int(item[1]))
            except (TypeError, ValueError):
                continue
    if not candidate_ids:
        candidate_ids = [int(x) for x in fallback_ids[:1]]
    return candidate_ids


def _sglang_attention_values(meta_info):
    values = []
    for item in meta_info.get("attention_signal", []) or []:
        if not isinstance(item, dict):
            values.append(None)
            continue
        try:
            values.append(float(item.get("value")))
        except (TypeError, ValueError):
            values.append(None)
    return values


def _sglang_matching_insert_prefix(output_ids, insertion_ids):
    matched = 0
    for actual_id, expected_id in zip(output_ids, insertion_ids):
        if int(actual_id) != int(expected_id):
            break
        matched += 1
    return matched


def _sglang_insertion_signal_scores(output_ids, attention_values, insertion_ids):
    scores = []
    for idx, value in enumerate(attention_values[: len(output_ids)]):
        if value is None:
            continue
        if _sglang_matching_insert_prefix(output_ids[:idx], insertion_ids) > 0:
            scores.append(float(value))
    return scores


def _sglang_score_mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _build_sglang_custom_params(runtime, prompt, prompt_token_len=None):
    tokenizer = runtime["tokenizer"]
    token_sets = runtime["token_sets"]
    if prompt_token_len is None:
        prompt_token_len = len(tokenizer.encode(prompt or "", add_special_tokens=False))
    use_hf_oracle = bool(getattr(config, "SGLANG_USE_HF_ATTN_ORACLE", False))
    strict_hf_oracle_parity = bool(
        getattr(config, "SGLANG_STRICT_HF_ORACLE_PARITY", True)
        and use_hf_oracle
        and getattr(config, "ENABLE_POST_INSERT_ATTN_SELECTION", False)
        and getattr(config, "ATTN_SELECTION_ONLY", False)
    )
    force_insertion_boundary = bool(
        getattr(config, "SGLANG_FORCE_INSERTION_BOUNDARY", True)
    )
    if strict_hf_oracle_parity:
        force_insertion_boundary = False
    boundary_text = str(
        getattr(config, "SGLANG_INSERTION_BOUNDARY_TEXT", "\n") or ""
    )
    boundary_token_ids = (
        tokenizer.encode(boundary_text, add_special_tokens=False) if boundary_text else []
    )
    if force_insertion_boundary and (not boundary_token_ids):
        fallback_boundary_text = str(
            getattr(config, "SGLANG_INSERTION_BOUNDARY_FALLBACK_TEXT", ".") or "."
        )
        boundary_token_ids = (
            tokenizer.encode(fallback_boundary_text, add_special_tokens=False)
            if fallback_boundary_text
            else []
        )
    punct_token_ids = sorted(
        _build_single_token_id_set(
            tokenizer, getattr(config, "ANTI_DEGEN_PUNCT_TEXTS", [])
        )
    )
    attn_only = bool(
        getattr(config, "ENABLE_POST_INSERT_ATTN_SELECTION", False)
        and getattr(config, "ATTN_SELECTION_ONLY", False)
    )
    hf_oracle_model_path = str(
        getattr(config, "SGLANG_HF_ATTN_ORACLE_MODEL_PATH", "") or ""
    ).strip()
    if not hf_oracle_model_path:
        hf_oracle_model_path = str(getattr(config, "MODEL_NAME", "") or "").strip()
    force_exact_insertion_ids = bool(
        getattr(config, "SGLANG_FORCE_EXACT_INSERTION_IDS", True)
    )
    if strict_hf_oracle_parity:
        force_exact_insertion_ids = False
    return {
        "beam_safety_controller_enabled": False,
        "max_tokens": int(config.MAX_TOKENS),
        "max_insertions": int(config.MAX_INSERTIONS),
        "entropy_scan_chunk_size": int(
            getattr(config, "SGLANG_ENTROPY_SCAN_CHUNK_SIZE", 1)
        ),
        "controller_trace_enabled": bool(
            getattr(config, "SGLANG_CONTROLLER_TRACE_LOG", False)
        ),
        "controller_trace_log_path": str(
            runtime.get("controller_trace_log_path", "") or ""
        ),
        "controller_trace_log_every_scan": bool(
            getattr(config, "SGLANG_CONTROLLER_TRACE_LOG_EVERY_SCAN", False)
        ),
        "debug_verbose": bool(getattr(config, "DEBUG_VERBOSE", False)),
        "debug_log_path": str(
            runtime.get("controller_debug_log_path", "")
            or runtime.get("debug_log_path", "")
            or ""
        ),
        "debug_snippet_chars": int(getattr(config, "DEBUG_SNIPPET_CHARS", 600)),
        "sentence_endings": list(getattr(config, "SENTENCE_ENDINGS", [])),
        "enable_think_prefix": bool(getattr(config, "ENABLE_THINK_PREFIX", False)),
        "think_only_token_ids": [
            int(x) for x in token_sets.get("THINK_ONLY_IDS", [])
        ],
        "top_k": int(getattr(config, "TOP_K", 10)),
        "safety_phrase_token_ids": [int(x) for x in token_sets.get("SORRY_ASSIST_IDS", [])],
        "force_insertion_boundary": force_insertion_boundary,
        "insertion_boundary_token_ids": [int(x) for x in boundary_token_ids],
        "force_exact_insertion_ids": force_exact_insertion_ids,
        "max_trigger_count": int(getattr(config, "MAX_TRIGGER_COUNT", 2)),
        "entropy_threshold_low": float(getattr(config, "ENTROPY_THRESHOLD_LOW", 1.2)),
        "entropy_threshold_high": float(getattr(config, "ENTROPY_THRESHOLD_HIGH", 2.0)),
        "entropy_threshold_decay_rate": float(
            getattr(config, "ENTROPY_THRESHOLD_DECAY_RATE", 3.0)
        ),
        "sorry_bonus_scale": float(getattr(config, "SORRY_BONUS_SCALE", 1.5)),
        "sorry_bonus_min": float(getattr(config, "SORRY_BONUS_MIN", 3.5)),
        "sorry_bonus_max": float(getattr(config, "SORRY_BONUS_MAX", 50.0)),
        "sorry_decay_lambda": float(getattr(config, "SORRY_DECAY_LAMBDA", 0.01)),
        "force_first_token_win": bool(getattr(config, "FORCE_FIRST_TOKEN_WIN", False)),
        "first_token_win_margin": float(getattr(config, "FIRST_TOKEN_WIN_MARGIN", 0.2)),
        "beam_safety_dual_path_mode": True,
        "safety_logprob_bias": float(getattr(config, "SAFETY_LOGPROB_BIAS", 0.0)),
        # Strict 1:1 parity path should not force first split to safety.
        "force_first_split_safety": False,
        "beam_safety_selection_mode_label": "attn_only" if attn_only else "fluency",
        "enable_post_insert_attn_selection": bool(
            getattr(config, "ENABLE_POST_INSERT_ATTN_SELECTION", False)
        ),
        "attn_selection_only": bool(getattr(config, "ATTN_SELECTION_ONLY", False)),
        "attn_only_fallback_use_dual_path": bool(
            getattr(config, "SGLANG_ATTN_ONLY_FALLBACK_USE_DUAL_PATH", True)
        ),
        "attn_selection_metric": str(
            getattr(config, "ATTN_SELECTION_METRIC", "post_insert")
        ),
        "post_insert_attn_threshold": float(
            getattr(config, "POST_INSERT_ATTN_THRESHOLD", 0.0)
        ),
        "safety_overall_attn_threshold": float(
            getattr(config, "SAFETY_OVERALL_ATTN_THRESHOLD", 0.0)
        ),
        "post_insert_attn_min_tokens": int(
            getattr(config, "POST_INSERT_ATTN_MIN_TOKENS", 1)
        ),
        "attn_promptlen_scale_beta": float(
            getattr(config, "ATTN_PROMPTLEN_SCALE_BETA", 0.0)
        ),
        "attn_promptlen_scale_p0": float(
            getattr(config, "ATTN_PROMPTLEN_SCALE_P0", 1.0)
        ),
        "prompt_token_len": int(prompt_token_len),
        "eos_token_id": int(tokenizer.eos_token_id)
        if tokenizer.eos_token_id is not None
        else -1,
        "safety_branch_max_tokens": int(
            getattr(config, "SAFETY_BRANCH_MAX_TOKENS", 100)
        ),
        "decode_use_sampling": bool(getattr(config, "DECODE_USE_SAMPLING", False)),
        "decode_temperature": float(getattr(config, "DECODE_TEMPERATURE", 1.0)),
        "decode_top_p": float(getattr(config, "DECODE_TOP_P", 1.0)),
        "decode_top_k": int(getattr(config, "DECODE_TOP_K", 0)),
        "enable_attn_driven_first_post_insert_token": bool(
            getattr(config, "ENABLE_ATTN_DRIVEN_FIRST_POST_INSERT_TOKEN", False)
        ),
        "attn_trace_top_k": int(getattr(config, "ATTN_TRACE_TOP_K", 10)),
        "anti_degen_min_check_tokens": int(
            getattr(config, "ANTI_DEGEN_MIN_CHECK_TOKENS", 30)
        ),
        "anti_degen_max_punct_ratio": float(
            getattr(config, "ANTI_DEGEN_MAX_PUNCT_RATIO", 0.6)
        ),
        "anti_degen_enabled": bool(getattr(config, "ANTI_DEGEN_ENABLED", False)),
        "anti_degen_repeat_window": int(getattr(config, "ANTI_DEGEN_REPEAT_WINDOW", 6)),
        "anti_degen_punct_token_ids": punct_token_ids,
        "hf_attn_oracle_enabled": bool(
            use_hf_oracle
        ),
        "hf_attn_oracle_model_path": hf_oracle_model_path,
        "hf_attn_oracle_device": str(
            getattr(config, "SGLANG_HF_ATTN_ORACLE_DEVICE", "cpu")
        ),
        "hf_attn_oracle_dtype": str(
            getattr(config, "SGLANG_HF_ATTN_ORACLE_DTYPE", "float32")
        ),
        "strict_hf_oracle_parity": strict_hf_oracle_parity,
        "sglang_enable_return_attention_signal": bool(
            getattr(config, "SGLANG_ENABLE_RETURN_ATTENTION_SIGNAL", False)
        ),
        "sglang_attention_signal_decode_max_steps": int(
            getattr(config, "SGLANG_ATTENTION_SIGNAL_DECODE_MAX_STEPS", 0)
        ),
        "sglang_attention_signal_decode_step_stride": int(
            getattr(config, "SGLANG_ATTENTION_SIGNAL_DECODE_STEP_STRIDE", 1)
        ),
        "sglang_attention_signal_layers": _parse_int_csv(
            getattr(config, "SGLANG_ATTENTION_SIGNAL_LAYERS", "")
        ),
    }


def _sglang_generate(
    prompt,
    runtime,
    use_safety_processor=False,
    use_safety_controller=False,
    answer_only_mode=False,
    return_logprob=False,
):
    tokenizer = runtime["tokenizer"]
    engine = runtime["model"]

    system_prompt = config.build_system_prompt()
    if system_prompt == "":
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    formatted_prompt_token_len = len(
        tokenizer.encode(formatted_prompt, add_special_tokens=False)
    )

    sampling_params = _build_sglang_sampling_params(answer_only_mode=answer_only_mode)
    custom_logit_processor = None
    attn_selection_enabled = bool(
        getattr(config, "ENABLE_POST_INSERT_ATTN_SELECTION", False)
    )
    if use_safety_processor and (not answer_only_mode):
        try:
            from sglang.srt.sampling.custom_logit_processor import (
                BeamSafetyLogitProcessor,
            )

            sampling_params["custom_params"] = _build_sglang_custom_params(
                runtime, prompt, prompt_token_len=formatted_prompt_token_len
            )
            custom_logit_processor = BeamSafetyLogitProcessor.to_str()
        except Exception as exc:
            log_print(f"[sglang] custom safety processor[redacted][redacted][redacted]: {exc}")

    if use_safety_controller and (not answer_only_mode):
        controller_prompt_token_len = len(
            tokenizer.encode(prompt or "", add_special_tokens=False)
        )
        sampling_params["custom_params"] = _build_sglang_custom_params(
            runtime, prompt, prompt_token_len=controller_prompt_token_len
        )
        sampling_params["custom_params"]["beam_safety_controller_enabled"] = True
        custom_logit_processor = None

    kwargs = {
        "prompt": formatted_prompt,
        "sampling_params": sampling_params,
        "custom_logit_processor": custom_logit_processor,
    }
    if return_logprob:
        kwargs["return_logprob"] = True
        kwargs["logprob_start_len"] = -1

    attention_signal_payload = {}
    if use_safety_processor and attn_selection_enabled and (not answer_only_mode):
        use_hf_oracle = bool(getattr(config, "SGLANG_USE_HF_ATTN_ORACLE", False))
        attn_only = bool(getattr(config, "ATTN_SELECTION_ONLY", False))
        strict_hf = bool(
            getattr(config, "SGLANG_STRICT_HF_ORACLE_PARITY", True) and use_hf_oracle
        )
        if not (use_hf_oracle and attn_only and strict_hf):
            attention_signal_payload = _build_sglang_attention_signal_request(
                formatted_prompt_token_len
            )
            if attention_signal_payload:
                kwargs.update(attention_signal_payload)

        if (not attention_signal_payload) and (not (use_hf_oracle and attn_only)):
            kwargs["return_hidden_states"] = True

    out = engine.generate(**kwargs)
    if isinstance(out, dict):
        return str(out.get("text", "")), out.get("meta_info", {}) or {}
    return str(out), {}


def _extract_split_trace_from_meta(meta_info):
    split_paths_raw = _parse_history_json_from_meta(
        meta_info, "beam_safety_split_selected_path_history_json"
    )
    split_modes_raw = _parse_history_json_from_meta(
        meta_info, "beam_safety_split_mode_history_json"
    )
    split_paths = []
    split_modes = []

    if split_paths_raw:
        split_paths = [str(x) for x in split_paths_raw]
        if split_modes_raw:
            for i in range(len(split_paths)):
                split_modes.append(str(_extract_list_value(split_modes_raw, i, "fluency")))
        else:
            split_modes = ["fluency"] * len(split_paths)
        return split_paths, split_modes

    # Backward-compatible fallback from per-token telemetry.
    token_split_raw = meta_info.get("beam_safety_split_count")
    token_paths_raw = meta_info.get("beam_safety_selected_path")
    token_modes_raw = meta_info.get("beam_safety_selection_mode")
    prev = 0
    if not isinstance(token_split_raw, list):
        return split_paths, split_modes

    for i, raw in enumerate(token_split_raw):
        try:
            cur = int(raw)
        except (TypeError, ValueError):
            continue
        if cur > prev:
            split_paths.append(str(_extract_list_value(token_paths_raw, i, "main")))
            split_modes.append(str(_extract_list_value(token_modes_raw, i, "fluency")))
            prev = cur
    return split_paths, split_modes


def _sglang_safety_branch_custom_params(runtime):
    tokenizer = runtime["tokenizer"]
    punct_token_ids = sorted(
        _build_single_token_id_set(
            tokenizer, getattr(config, "ANTI_DEGEN_PUNCT_TEXTS", [])
        )
    )
    return {
        "safety_phrase_token_ids": [
            int(x) for x in runtime["token_sets"].get("SORRY_ASSIST_IDS", [])
        ],
        "force_exact_insertion_ids": bool(
            getattr(config, "SGLANG_FORCE_EXACT_INSERTION_IDS", False)
        ),
        "sorry_bonus_scale": float(getattr(config, "SORRY_BONUS_SCALE", 1.5)),
        "sorry_bonus_min": float(getattr(config, "SORRY_BONUS_MIN", 3.5)),
        "sorry_bonus_max": float(getattr(config, "SORRY_BONUS_MAX", 50.0)),
        "sorry_decay_lambda": float(getattr(config, "SORRY_DECAY_LAMBDA", 0.01)),
        "force_first_token_win": bool(getattr(config, "FORCE_FIRST_TOKEN_WIN", False)),
        "first_token_win_margin": float(getattr(config, "FIRST_TOKEN_WIN_MARGIN", 0.2)),
        "anti_degen_enabled": bool(getattr(config, "ANTI_DEGEN_ENABLED", False)),
        "anti_degen_repeat_window": int(
            getattr(config, "ANTI_DEGEN_REPEAT_WINDOW", 6)
        ),
        "anti_degen_punct_token_ids": punct_token_ids,
    }


def _sglang_generate_entropy_step(tokens, runtime):
    processors = _sglang_custom_processors()
    out = _sglang_generate_from_ids(
        tokens,
        runtime,
        max_new_tokens=1,
        custom_processor=processors["entropy"],
        custom_params={"telemetry_top_k": int(getattr(config, "TOP_K", 10))},
    )
    entropy = _to_float_or_none(_sglang_meta_last(out["meta_info"], "beam_parity_entropy"))
    return out, entropy


def _sglang_mean_lp_after_insert(logprobs, insert_len):
    if not logprobs:
        return -1e9
    after = logprobs[max(0, int(insert_len)) :]
    values = after if after else logprobs
    return float(sum(values) / len(values))


def _sglang_force_candidate(
    safety_tokens,
    candidate_id,
    source_positions,
    runtime,
):
    processors = _sglang_custom_processors()
    out = _sglang_generate_from_ids(
        safety_tokens,
        runtime,
        # The first token is sampled from extend/prefill logits. Request one
        # following decode step so SGLang can emit the candidate query signal.
        max_new_tokens=2,
        custom_processor=processors["force"],
        custom_params={"forced_token_ids": [int(candidate_id)]},
        return_logprob=True,
        attention_source_positions=source_positions,
    )
    signal_values = _sglang_attention_values(out["meta_info"])
    score = next((value for value in signal_values if value is not None), 0.0)
    out["output_ids"] = out["output_ids"][:1]
    output_logprobs = out["meta_info"].get("output_token_logprobs")
    if isinstance(output_logprobs, list):
        out["meta_info"]["output_token_logprobs"] = output_logprobs[:1]
    return out, float(score)


def _sglang_safety_branch(tokens, runtime, max_new_tokens):
    tokenizer = runtime["tokenizer"]
    processors = _sglang_custom_processors()
    insertion_ids = [
        int(x) for x in runtime["token_sets"].get("SORRY_ASSIST_IDS", [])
    ]
    if max_new_tokens <= 0:
        return {
            "tokens": list(tokens),
            "segment_ids": [],
            "logprobs": [],
            "safety_scores": [],
            "post_insert_scores": [],
            "matched_inserted_len": 0,
            "ended_early": False,
            "first_post_insert_attention_driven": False,
        }

    insert_budget = min(int(max_new_tokens), len(insertion_ids))
    insert_out = _sglang_generate_from_ids(
        tokens,
        runtime,
        max_new_tokens=insert_budget,
        custom_processor=processors["safety_branch"],
        custom_params=_sglang_safety_branch_custom_params(runtime),
        return_logprob=True,
        attention_source_positions=[max(0, len(tokens) - 1)],
    )
    segment_ids = list(insert_out["output_ids"])
    logprobs = _sglang_output_logprobs(insert_out["meta_info"])
    insert_attention = _sglang_attention_values(insert_out["meta_info"])
    safety_scores = _sglang_insertion_signal_scores(
        segment_ids, insert_attention, insertion_ids
    )
    matched_inserted_len = _sglang_matching_insert_prefix(segment_ids, insertion_ids)
    source_positions = [
        int(len(tokens) + idx) for idx in range(max(0, matched_inserted_len))
    ]
    current_tokens = [*tokens, *segment_ids]
    eos_id = tokenizer.eos_token_id
    ended_early = bool(
        segment_ids and eos_id is not None and int(segment_ids[-1]) == int(eos_id)
    )
    post_insert_scores = []
    first_post_insert_attention_driven = False
    remaining = int(max_new_tokens) - len(segment_ids)

    if remaining > 0 and not ended_early:
        first_out = None
        first_score = 0.0
        can_attention_select = bool(
            getattr(config, "ENABLE_ATTN_DRIVEN_FIRST_POST_INSERT_TOKEN", False)
            and source_positions
        )
        if can_attention_select:
            probe = _sglang_generate_from_ids(
                current_tokens,
                runtime,
                max_new_tokens=1,
                return_logprob=True,
                top_logprobs_num=max(1, int(getattr(config, "ATTN_TRACE_TOP_K", 10))),
                attention_source_positions=source_positions,
            )
            candidate_ids = _sglang_top_candidate_ids(
                probe["meta_info"], probe["output_ids"]
            )
            best = None
            for candidate_id in candidate_ids:
                candidate_out, candidate_score = _sglang_force_candidate(
                    current_tokens,
                    candidate_id=candidate_id,
                    source_positions=source_positions,
                    runtime=runtime,
                )
                if best is None or candidate_score > best[1]:
                    best = (candidate_out, candidate_score)
            if best is not None:
                first_out, first_score = best
                first_post_insert_attention_driven = True

        if first_out is None:
            first_out = _sglang_generate_from_ids(
                current_tokens,
                runtime,
                max_new_tokens=1,
                return_logprob=True,
                attention_source_positions=source_positions,
            )
            signal_values = _sglang_attention_values(first_out["meta_info"])
            if signal_values and signal_values[0] is not None:
                first_score = float(signal_values[0])

        first_ids = list(first_out["output_ids"])
        segment_ids.extend(first_ids)
        logprobs.extend(_sglang_output_logprobs(first_out["meta_info"]))
        if first_ids and source_positions:
            safety_scores.append(float(first_score))
        current_tokens.extend(first_ids)
        remaining -= len(first_ids)
        ended_early = bool(
            first_ids and eos_id is not None and int(first_ids[-1]) == int(eos_id)
        )

    while remaining > 0 and not ended_early:
        # Long non-streaming SGLang generations can return attention_signal for
        # only part of the decoded segment. Chunking keeps token ids and
        # attention telemetry aligned across the full safety branch budget.
        chunk_new_tokens = min(remaining, 32)
        rest_out = _sglang_generate_from_ids(
            current_tokens,
            runtime,
            max_new_tokens=chunk_new_tokens,
            return_logprob=True,
            attention_source_positions=source_positions,
        )
        rest_ids = list(rest_out["output_ids"])
        if not rest_ids:
            break
        segment_ids.extend(rest_ids)
        logprobs.extend(_sglang_output_logprobs(rest_out["meta_info"]))
        for value in _sglang_attention_values(rest_out["meta_info"])[: len(rest_ids)]:
            if value is None or not source_positions:
                continue
            safety_scores.append(float(value))
            post_insert_scores.append(float(value))
        current_tokens.extend(rest_ids)
        remaining -= len(rest_ids)
        ended_early = bool(
            rest_ids and eos_id is not None and int(rest_ids[-1]) == int(eos_id)
        )

    return {
        "tokens": current_tokens,
        "segment_ids": segment_ids,
        "logprobs": logprobs,
        "safety_scores": safety_scores,
        "post_insert_scores": post_insert_scores,
        "matched_inserted_len": int(matched_inserted_len),
        "ended_early": bool(ended_early),
        "first_post_insert_attention_driven": bool(first_post_insert_attention_driven),
    }


def _sglang_main_branch(tokens, runtime, max_new_tokens):
    out = _sglang_generate_from_ids(
        tokens,
        runtime,
        max_new_tokens=max_new_tokens,
        return_logprob=True,
    )
    tokenizer = runtime["tokenizer"]
    segment_ids = list(out["output_ids"])
    eos_id = tokenizer.eos_token_id
    return {
        "tokens": [*tokens, *segment_ids],
        "segment_ids": segment_ids,
        "logprobs": _sglang_output_logprobs(out["meta_info"]),
        "ended_early": bool(
            segment_ids and eos_id is not None and int(segment_ids[-1]) == int(eos_id)
        ),
    }


def _sglang_route_split(
    main_branch,
    safety_branch,
    runtime,
    prompt_token_len,
    branch_budget,
):
    tokenizer = runtime["tokenizer"]
    insertion_len = len(runtime["token_sets"].get("SORRY_ASSIST_IDS", []))
    mean_lp_orig = _sglang_mean_lp_after_insert(main_branch["logprobs"], insertion_len)
    mean_lp_safety = _sglang_mean_lp_after_insert(
        safety_branch["logprobs"], insertion_len
    )
    safety_scores = safety_branch["safety_scores"]
    post_insert_scores = safety_branch["post_insert_scores"]
    metric = str(getattr(config, "ATTN_SELECTION_METRIC", "post_insert"))
    if metric not in {"post_insert", "safety_overall"}:
        metric = "post_insert"
    if metric == "safety_overall":
        metric_values = safety_scores
        threshold = float(getattr(config, "SAFETY_OVERALL_ATTN_THRESHOLD", 0.0))
    else:
        metric_values = post_insert_scores
        threshold = float(getattr(config, "POST_INSERT_ATTN_THRESHOLD", 0.0))
    metric_raw = _sglang_score_mean(metric_values)
    metric_scaled = metric_raw * _promptlen_attn_scale(prompt_token_len)
    metric_count = len(metric_values)
    attn_enabled = bool(getattr(config, "ENABLE_POST_INSERT_ATTN_SELECTION", False))
    attn_only = bool(getattr(config, "ATTN_SELECTION_ONLY", False))
    attn_ready = attn_enabled and metric_count >= int(
        getattr(config, "POST_INSERT_ATTN_MIN_TOKENS", 1)
    )

    if main_branch["ended_early"] and not safety_branch["ended_early"]:
        choose_safety = False
        mode = "main_eos"
    elif safety_branch["ended_early"] and not main_branch["ended_early"]:
        choose_safety = True
        mode = "safety_eos"
    elif attn_only:
        choose_safety = bool(attn_ready and metric_scaled >= threshold)
        mode = "attn_only" if attn_ready else "attn_only_insufficient_signal"
    else:
        choose_safety = mean_lp_safety > mean_lp_orig
        mode = "fluency"
        safety_budget_exhausted = (
            (not safety_branch["ended_early"])
            and len(safety_branch["segment_ids"]) >= int(branch_budget)
        )
        if attn_enabled and safety_budget_exhausted and attn_ready:
            choose_safety = metric_scaled >= threshold
            mode = "attn_budget_gate"

    punct_ids = _build_single_token_id_set(
        tokenizer, getattr(config, "ANTI_DEGEN_PUNCT_TEXTS", [])
    )
    safety_punct_count = sum(
        1
        for token_id in safety_branch["segment_ids"]
        if _is_punct_like_token(tokenizer, token_id, punct_ids)
    )
    punct_ratio = (
        float(safety_punct_count) / float(len(safety_branch["segment_ids"]))
        if safety_branch["segment_ids"]
        else 0.0
    )
    degenerate = bool(
        getattr(config, "ANTI_DEGEN_ENABLED", False)
        and len(safety_branch["segment_ids"])
        >= int(getattr(config, "ANTI_DEGEN_MIN_CHECK_TOKENS", 30))
        and punct_ratio >= float(getattr(config, "ANTI_DEGEN_MAX_PUNCT_RATIO", 0.6))
    )
    if degenerate and not main_branch["ended_early"]:
        choose_safety = False
        mode = "anti_degen_gate"

    return {
        "choose_safety": bool(choose_safety),
        "selection_mode": mode,
        "metric": metric,
        "metric_raw": float(metric_raw),
        "metric_scaled": float(metric_scaled),
        "metric_threshold": float(threshold),
        "metric_count": int(metric_count),
        "attn_ready": bool(attn_ready),
        "mean_lp_orig": float(mean_lp_orig),
        "mean_lp_safety": float(mean_lp_safety),
        "punct_like_count": int(safety_punct_count),
        "punct_ratio": float(punct_ratio),
        "degenerate_safety": bool(degenerate),
    }


def _run_sglang_beam_search(prompt, runtime, answer_only_mode=False):
    if answer_only_mode:
        modified_response, meta_info = _sglang_generate(
            prompt=prompt,
            runtime=runtime,
            use_safety_processor=False,
            answer_only_mode=True,
            return_logprob=False,
        )
        del meta_info
        return (
            _postprocess_sglang_response_text(
                modified_response,
                selected_path="main",
                answer_only_mode=True,
            ),
            "main",
            0,
            [],
            {"enabled": False, "backend": "sglang_parity_beam_safety", "splits": []},
        )

    modified_response, meta_info = _sglang_generate(
        prompt=prompt,
        runtime=runtime,
        use_safety_controller=True,
        answer_only_mode=False,
        return_logprob=False,
    )
    selected_path_history, _ = _extract_split_trace_from_meta(meta_info)
    selected_path = str(meta_info.get("beam_safety_selected_path") or "main")
    if isinstance(selected_path, list):
        selected_path = str(selected_path[-1] if selected_path else "main")
    raw_trace = meta_info.get("beam_safety_insertion_attention_trace_json")
    if isinstance(raw_trace, list) and raw_trace:
        raw_trace = raw_trace[-1]
    if isinstance(raw_trace, str):
        try:
            insertion_attention_trace = json.loads(raw_trace)
        except (TypeError, ValueError, json.JSONDecodeError):
            insertion_attention_trace = {}
    elif isinstance(raw_trace, dict):
        insertion_attention_trace = raw_trace
    else:
        insertion_attention_trace = {}
    if not isinstance(insertion_attention_trace, dict) or "splits" not in insertion_attention_trace:
        insertion_attention_trace = {
            "enabled": bool(getattr(config, "ENABLE_POST_INSERT_ATTN_SELECTION", False)),
            "backend": "sglang_internal_decoding_controller",
            "splits": [],
        }
    modified_response = _postprocess_sglang_response_text(
        modified_response,
        selected_path=selected_path,
        answer_only_mode=False,
    )
    return (
        modified_response,
        selected_path,
        len(selected_path_history),
        selected_path_history,
        insertion_attention_trace,
    )


def _run_sglang_original_response(prompt, runtime):
    original_response, _ = _sglang_generate(
        prompt=prompt,
        runtime=runtime,
        use_safety_processor=False,
        answer_only_mode=False,
        return_logprob=False,
    )
    return original_response


def run_beam_search(prompt, runtime, answer_only_mode=False):
    if _is_sglang_backend(runtime):
        return _run_sglang_beam_search(
            prompt=prompt,
            runtime=runtime,
            answer_only_mode=answer_only_mode,
        )

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    token_sets = runtime["token_sets"]

    system_prompt = config.build_system_prompt()
    if system_prompt == "":
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    initial_token_count = len(input_ids[0])
    prompt_token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    attn_length_scale = _promptlen_attn_scale(prompt_token_len)

    tokens_main = input_ids[0]
    split_count = 0
    selected_path = "main"
    selected_path_history = []
    insertion_attention_trace = {
        "enabled": bool(getattr(config, "ENABLE_INSERTION_ATTN_TRACE", False)),
        "splits": [],
    }
    allowed_option_token_ids = (
        _build_option_token_candidates(tokenizer) if answer_only_mode else []
    )
    punct_token_ids = _build_single_token_id_set(
        tokenizer, getattr(config, "ANTI_DEGEN_PUNCT_TEXTS", [])
    )

    with torch.no_grad():
        while True:
            if len(tokens_main) - initial_token_count >= config.MAX_TOKENS:
                break

            out = model(tokens_main.unsqueeze(0))
            logits = out.logits[:, -1, :].clone()
            if answer_only_mode and len(tokens_main) == initial_token_count:
                logits = _constrain_logits_to_allowed(logits, allowed_option_token_ids)
            probs = softmax(logits)[0]
            should_trigger, trigger_reason, entropy_threshold = should_split(
                probs, split_count=split_count
            )
            if answer_only_mode:
                should_trigger = False
            if _debug_enabled():
                generated_token_count = len(tokens_main) - initial_token_count
                h_val = entropy_nats(probs)
                _debug_block(
                    "Entropy",
                    "Entropy Trigger Check",
                    [
                        ("decode_step", generated_token_count),
                        ("token_index", len(tokens_main)),
                        ("entropy", f"{h_val:.6f}"),
                        ("threshold", f"{float(entropy_threshold):.6f}"),
                        (
                            "comparison",
                            f"{h_val:.6f} > {float(entropy_threshold):.6f} => "
                            f"{bool(split_count < config.MAX_INSERTIONS and should_trigger)}",
                        ),
                        (
                            "triggered",
                            bool(split_count < config.MAX_INSERTIONS and should_trigger),
                        ),
                        (
                            "trigger_count",
                            f"{int(split_count)}/{int(config.MAX_INSERTIONS)}",
                        ),
                    ],
                )

            if split_count < config.MAX_INSERTIONS and should_trigger:
                log_trigger_details(
                    probs,
                    trigger_reason,
                    tokens_main,
                    initial_token_count,
                    tokenizer,
                    token_sets,
                    entropy_threshold=entropy_threshold,
                )
                at_boundary = is_sentence_boundary(tokens_main, tokenizer)
                skip_think_prefix = not config.ENABLE_THINK_PREFIX
                if not at_boundary:
                    keep_length, found_boundary = find_last_sentence_boundary_in_generated(
                        tokens_main, initial_token_count, tokenizer
                    )
                    if found_boundary:
                        tokens_main = tokens_main[:keep_length].clone()
                        log_print(
                            f"   [redacted][redacted][redacted] {keep_length - initial_token_count} [redacted] token"
                        )
                    else:
                        discarded = len(tokens_main) - initial_token_count
                        tokens_main = tokens_main[:initial_token_count].clone()
                        skip_think_prefix = True
                        log_print(
                            f"   [redacted][redacted][redacted][redacted][redacted] {discarded} token[redacted][redacted] <think>"
                        )

                if split_count == 0:
                    skip_think_prefix = True

                tokens_orig = tokens_main.clone()
                parallel_start_len = len(tokens_main)
                if skip_think_prefix:
                    tokens_safety = tokens_main.clone()
                else:
                    prefix_tensor = torch.tensor(
                        token_sets["THINK_ONLY_IDS"], device=device, dtype=tokens_main.dtype
                    )
                    tokens_safety = torch.cat([tokens_main, prefix_tensor])

                _debug_block(
                    "Branch",
                    "Enter Branch Exploration",
                    [
                        ("split_index", split_count + 1),
                        ("branch_start_token_index", parallel_start_len),
                        ("branch_start_decode_step", parallel_start_len - initial_token_count),
                        ("branch_budget", int(config.SAFETY_BRANCH_MAX_TOKENS)),
                        ("skip_think_prefix", bool(skip_think_prefix)),
                    ],
                )

                log_probs_orig = []
                log_probs_safety = []
                n_steps = config.SAFETY_BRANCH_MAX_TOKENS
                first_bonus = None
                orig_ended_early = False
                safety_ended_early = False
                safety_attn_best_token_ids = []
                safety_attn_records = []
                first_post_insert_token_decoded_by_attn = False
                post_insert_selected_attn_scores = []
                safety_selected_attn_scores = []
                safety_punct_like_count = 0
                safety_same_token_run = 0
                safety_prev_token_id = None
                attn_diag_stats = {
                    "total_calls": 0,
                    "ok": 0,
                    "invalid_span": 0,
                    "no_attentions": 0,
                    "empty_span": 0,
                    "zero_with_ok": 0,
                }
                safety_selected_per_intervention_token = []

                for step in range(n_steps):
                    out_a = model(tokens_orig.unsqueeze(0))
                    logits_a = out_a.logits[:, -1, :].clone()
                    probs_a = softmax(logits_a)[0]
                    next_id_a = _sample_next_token_from_logits(logits_a[0])
                    lp_a = math.log(probs_a[next_id_a].item() + 1e-10)
                    log_probs_orig.append(lp_a)
                    tokens_orig = torch.cat([tokens_orig, torch.tensor([next_id_a], device=device)])
                    if next_id_a == tokenizer.eos_token_id:
                        orig_ended_early = True
                        break

                    out_b = model(tokens_safety.unsqueeze(0))
                    logits_b = out_b.logits[:, -1, :].clone().float()
                    if step < len(token_sets["SORRY_ASSIST_IDS"]):
                        target_id = int(token_sets["SORRY_ASSIST_IDS"][step])
                        probs_b_raw = softmax(logits_b)[0]
                        top1_prob = probs_b_raw.max().item()
                        target_prob = probs_b_raw[target_id].item()
                        if step == 0:
                            gap_log = math.log(top1_prob + 1e-10) - math.log(target_prob + 1e-10)
                            first_bonus = max(
                                config.SORRY_BONUS_MIN,
                                min(
                                    config.SORRY_BONUS_MAX,
                                    config.SORRY_BONUS_SCALE * max(0.0, gap_log),
                                ),
                            )
                            if getattr(config, "FORCE_FIRST_TOKEN_WIN", False):
                                target_logit = logits_b[0, target_id].item()
                                other_logits = logits_b[0].clone()
                                other_logits[target_id] = -float("inf")
                                max_other_logit = torch.max(other_logits).item()
                                required_bonus = (
                                    max_other_logit
                                    - target_logit
                                    + config.FIRST_TOKEN_WIN_MARGIN
                                )
                                first_bonus = max(first_bonus, required_bonus)
                            bonus = first_bonus
                        else:
                            bonus = first_bonus * math.exp(-config.SORRY_DECAY_LAMBDA * step)
                        logits_b[0, target_id] = logits_b[0, target_id] + bonus

                    probs_b = softmax(logits_b)[0]
                    if insertion_attention_trace["enabled"] and step < config.ATTN_TRACE_MAX_STEPS:
                        inserted_prefix_len = _match_inserted_prefix_len(
                            tokens_safety, parallel_start_len, token_sets["SORRY_ASSIST_IDS"]
                        )
                        if inserted_prefix_len > 0:
                            trace_k = min(config.ATTN_TRACE_TOP_K, probs_b.shape[0])
                            topk_vals_b, topk_idx_b = torch.topk(probs_b, trace_k)
                            inserted_span_start = parallel_start_len
                            inserted_span_end = parallel_start_len + inserted_prefix_len
                            best_record = None
                            for rank in range(trace_k):
                                candidate_id = int(topk_idx_b[rank].item())
                                candidate_prob = float(topk_vals_b[rank].item())
                                attn_score, attn_reason = _candidate_attn_to_inserted_span(
                                    model,
                                    tokens_safety,
                                    candidate_id,
                                    inserted_span_start,
                                    inserted_span_end,
                                    device,
                                    return_reason=True,
                                )
                                attn_diag_stats["total_calls"] += 1
                                if attn_reason in attn_diag_stats:
                                    attn_diag_stats[attn_reason] += 1
                                if attn_reason == "ok" and abs(attn_score) < 1e-12:
                                    attn_diag_stats["zero_with_ok"] += 1
                                record = {
                                    "rank": rank + 1,
                                    "token_id": candidate_id,
                                    "token_text": tokenizer.decode(
                                        [candidate_id], skip_special_tokens=False
                                    ),
                                    "prob": candidate_prob,
                                    "attn_to_inserted_prefix": attn_score,
                                    "attn_reason": attn_reason,
                                }
                                if (
                                    best_record is None
                                    or record["attn_to_inserted_prefix"]
                                    > best_record["attn_to_inserted_prefix"]
                                ):
                                    best_record = record

                            if best_record is not None:
                                safety_attn_best_token_ids.append(best_record["token_id"])
                                safety_attn_records.append(
                                    {
                                        "step": step,
                                        "inserted_prefix_len": inserted_prefix_len,
                                        "best_token": best_record,
                                    }
                                )
                                log_print(
                                    "   [attn] safety step "
                                    f"{step}: top{trace_k}[redacted] token="
                                    f"{repr(best_record['token_text'])} "
                                    f"(id={best_record['token_id']}, rank={best_record['rank']}, "
                                    f"p={best_record['prob']:.6f}, "
                                    f"attn={best_record['attn_to_inserted_prefix']:.6f}, "
                                    f"reason={best_record.get('attn_reason', 'na')})"
                                )
                    first_post_insert_step = step == len(token_sets["SORRY_ASSIST_IDS"])
                    use_attn_for_first_post_insert = bool(
                        getattr(
                            config,
                            "ENABLE_ATTN_DRIVEN_FIRST_POST_INSERT_TOKEN",
                            False,
                        )
                    )
                    if (
                        first_post_insert_step
                        and use_attn_for_first_post_insert
                        and not first_post_insert_token_decoded_by_attn
                    ):
                        attn_choice = _select_attention_driven_next_token(
                            model=model,
                            tokenizer=tokenizer,
                            tokens_safety=tokens_safety,
                            probs_b=probs_b,
                            parallel_start_len=parallel_start_len,
                            inserted_token_ids=token_sets["SORRY_ASSIST_IDS"],
                            device=device,
                        )
                        if attn_choice is not None:
                            next_id_b = int(attn_choice["token_id"])
                            first_post_insert_token_decoded_by_attn = True
                            safety_attn_records.append(
                                {
                                    "step": step,
                                    "mode": "attention_driven_first_post_insert_token",
                                    "selected_token": attn_choice,
                                }
                            )
                            log_print(
                                "   [attn-decode] [redacted]token[redacted]: "
                                f"token={repr(attn_choice['token_text'])} "
                                f"(id={attn_choice['token_id']}, rank={attn_choice['rank']}, "
                                f"p={attn_choice['prob']:.6f}, "
                                f"attn={attn_choice['attn_to_inserted_prefix']:.6f})"
                            )
                        else:
                            next_id_b = _sample_next_token_from_logits(logits_b[0])
                    else:
                        next_id_b = _sample_next_token_from_logits(logits_b[0])
                        if (
                            getattr(config, "ANTI_DEGEN_ENABLED", False)
                            and safety_prev_token_id is not None
                            and safety_same_token_run >= int(getattr(config, "ANTI_DEGEN_REPEAT_WINDOW", 6))
                            and _is_punct_like_token(tokenizer, safety_prev_token_id, punct_token_ids)
                        ):
                            adjusted_logits = logits_b[0].clone()
                            adjusted_logits[int(safety_prev_token_id)] = -float("inf")
                            next_id_b = _sample_next_token_from_logits(adjusted_logits)
                            log_print(
                                "   [anti-degen] [redacted][redacted][redacted]token "
                                f"id={int(safety_prev_token_id)}[redacted][redacted] id={int(next_id_b)}"
                            )

                    inserted_prefix_len_for_selected = _match_inserted_prefix_len(
                        tokens_safety, parallel_start_len, token_sets["SORRY_ASSIST_IDS"]
                    )
                    selected_attn_score = 0.0
                    if inserted_prefix_len_for_selected > 0:
                        selected_attn_score, selected_attn_reason = _candidate_attn_to_inserted_span(
                            model=model,
                            tokens_safety=tokens_safety,
                            candidate_id=next_id_b,
                            inserted_span_start=parallel_start_len,
                            inserted_span_end=parallel_start_len + inserted_prefix_len_for_selected,
                            device=device,
                            return_reason=True,
                        )
                        attn_diag_stats["total_calls"] += 1
                        if selected_attn_reason in attn_diag_stats:
                            attn_diag_stats[selected_attn_reason] += 1
                        if selected_attn_reason == "ok" and abs(selected_attn_score) < 1e-12:
                            attn_diag_stats["zero_with_ok"] += 1
                        safety_selected_attn_scores.append(float(selected_attn_score))
                    else:
                        selected_attn_reason = "no_inserted_prefix"

                    selected_token_text = tokenizer.decode([next_id_b], skip_special_tokens=False)
                    eos_marker = " [EOS]" if next_id_b == tokenizer.eos_token_id else ""
                    log_print(
                        "   [attn-selected] safety step "
                        f"{step}: token={repr(selected_token_text)} (id={next_id_b}), "
                        f"attn={selected_attn_score:.6f}, "
                        f"inserted_prefix_len={inserted_prefix_len_for_selected}, "
                        f"reason={selected_attn_reason}{eos_marker}"
                    )

                    if (
                        getattr(config, "SAVE_PER_STEP_INTERVENTION_TOKEN_ATTN", False)
                        and inserted_prefix_len_for_selected > 0
                    ):
                        inserted_span_start = parallel_start_len
                        inserted_span_end = parallel_start_len + inserted_prefix_len_for_selected
                        per_token_attn, per_token_reason = _candidate_attn_per_inserted_token(
                            model=model,
                            tokens_safety=tokens_safety,
                            candidate_id=next_id_b,
                            inserted_span_start=inserted_span_start,
                            inserted_span_end=inserted_span_end,
                            device=device,
                        )
                        intervention_token_ids = [
                            int(x)
                            for x in token_sets["SORRY_ASSIST_IDS"][:inserted_prefix_len_for_selected]
                        ]
                        intervention_token_texts = [
                            tokenizer.decode([tok_id], skip_special_tokens=False)
                            for tok_id in intervention_token_ids
                        ]
                        safety_selected_per_intervention_token.append(
                            {
                                "step": step,
                                "selected_token_id": int(next_id_b),
                                "selected_token_text": selected_token_text,
                                "inserted_prefix_len": int(inserted_prefix_len_for_selected),
                                "intervention_token_ids": intervention_token_ids,
                                "intervention_token_texts": intervention_token_texts,
                                "attn_to_each_intervention_token": per_token_attn,
                                "per_token_reason": per_token_reason,
                                "mean_attn_to_inserted_prefix": float(selected_attn_score),
                            }
                        )

                    is_post_insert_after_first = step > len(token_sets["SORRY_ASSIST_IDS"])
                    if inserted_prefix_len_for_selected > 0 and is_post_insert_after_first:
                        post_insert_selected_attn_scores.append(float(selected_attn_score))

                    lp_b = math.log(probs_b[next_id_b].item() + 1e-10)
                    log_probs_safety.append(lp_b)
                    if _is_punct_like_token(tokenizer, next_id_b, punct_token_ids):
                        safety_punct_like_count += 1
                    if safety_prev_token_id is None or int(next_id_b) != int(safety_prev_token_id):
                        safety_same_token_run = 1
                    else:
                        safety_same_token_run += 1
                    safety_prev_token_id = int(next_id_b)
                    tokens_safety = torch.cat([tokens_safety, torch.tensor([next_id_b], device=device)])
                    if next_id_b == tokenizer.eos_token_id:
                        safety_ended_early = True
                        break

                skip = len(token_sets["SORRY_ASSIST_IDS"])

                def mean_lp_after_boost(lp_list):
                    if not lp_list:
                        return -1e9
                    after = lp_list[skip:]
                    if not after:
                        return sum(lp_list) / len(lp_list)
                    return sum(after) / len(after)

                mean_lp_orig = mean_lp_after_boost(log_probs_orig)
                mean_lp_safety = mean_lp_after_boost(log_probs_safety)
                avg_post_insert_attn = (
                    sum(post_insert_selected_attn_scores) / len(post_insert_selected_attn_scores)
                    if post_insert_selected_attn_scores
                    else 0.0
                )
                avg_safety_selected_attn = (
                    sum(safety_selected_attn_scores) / len(safety_selected_attn_scores)
                    if safety_selected_attn_scores
                    else 0.0
                )
                first10_safety_selected_attn_scores = safety_selected_attn_scores[:10]
                avg_safety_selected_attn_first10 = (
                    sum(first10_safety_selected_attn_scores)
                    / len(first10_safety_selected_attn_scores)
                    if first10_safety_selected_attn_scores
                    else 0.0
                )
                first10_post_insert_selected_attn_scores = post_insert_selected_attn_scores[:10]
                avg_post_insert_attn_first10 = (
                    sum(first10_post_insert_selected_attn_scores)
                    / len(first10_post_insert_selected_attn_scores)
                    if first10_post_insert_selected_attn_scores
                    else 0.0
                )
                budget_exhausted_without_safety_eos = (
                    (not safety_ended_early) and (len(log_probs_safety) >= n_steps)
                )
                selection_mode = "fluency"
                attn_selection_metric = getattr(config, "ATTN_SELECTION_METRIC", "post_insert")
                if attn_selection_metric not in {"post_insert", "safety_overall"}:
                    attn_selection_metric = "post_insert"
                if attn_selection_metric == "safety_overall":
                    attn_metric_value_raw = avg_safety_selected_attn
                    attn_metric_threshold = float(
                        getattr(config, "SAFETY_OVERALL_ATTN_THRESHOLD", 0.0)
                    )
                    attn_metric_count = len(safety_selected_attn_scores)
                else:
                    attn_metric_value_raw = avg_post_insert_attn
                    attn_metric_threshold = float(getattr(config, "POST_INSERT_ATTN_THRESHOLD", 0.0))
                    attn_metric_count = len(post_insert_selected_attn_scores)
                attn_metric_value = attn_metric_value_raw * attn_length_scale
                log_print(
                    "   [attn-summary] safety[redacted]: "
                    f"avg_selected_attn={avg_safety_selected_attn:.6f}, "
                    f"avg_selected_attn_first10={avg_safety_selected_attn_first10:.6f}, "
                    f"valid_steps={len(safety_selected_attn_scores)}, "
                    f"post_insert_avg={avg_post_insert_attn:.6f}, "
                    f"post_insert_avg_first10={avg_post_insert_attn_first10:.6f}, "
                    f"post_insert_steps={len(post_insert_selected_attn_scores)}, "
                    f"safety_ended_early={safety_ended_early}"
                )
                log_print(
                    "   [attn-length-scale] "
                    f"P={prompt_token_len}, "
                    f"P0={float(getattr(config, 'ATTN_PROMPTLEN_SCALE_P0', 0.0)):.3f}, "
                    f"beta={float(getattr(config, 'ATTN_PROMPTLEN_SCALE_BETA', 0.0)):.3f}, "
                    f"scale={attn_length_scale:.6f}, "
                    f"metric_raw={attn_metric_value_raw:.6f}, "
                    f"metric_scaled={attn_metric_value:.6f}"
                )
                log_print(
                    "   [attn-zero-reasons] "
                    f"calls={attn_diag_stats['total_calls']}, "
                    f"ok={attn_diag_stats['ok']}, "
                    f"zero_with_ok={attn_diag_stats['zero_with_ok']}, "
                    f"no_attentions={attn_diag_stats['no_attentions']}, "
                    f"empty_span={attn_diag_stats['empty_span']}, "
                    f"invalid_span={attn_diag_stats['invalid_span']}"
                )

                if orig_ended_early and not safety_ended_early:
                    choose_safety = False
                elif safety_ended_early and not orig_ended_early:
                    choose_safety = True
                else:
                    attn_enabled = bool(getattr(config, "ENABLE_POST_INSERT_ATTN_SELECTION", False))
                    attn_only = bool(getattr(config, "ATTN_SELECTION_ONLY", False))
                    attn_min_tokens = int(getattr(config, "POST_INSERT_ATTN_MIN_TOKENS", 1))
                    attn_ready = attn_enabled and (attn_metric_count >= attn_min_tokens)

                    if attn_only:
                        # Attention-only path decision: never fall back to fluency.
                        if attn_ready:
                            selection_mode = "attn_only"
                            choose_safety = attn_metric_value >= attn_metric_threshold
                        else:
                            selection_mode = "attn_only_insufficient_signal"
                            choose_safety = False
                    else:
                        choose_safety = mean_lp_safety > mean_lp_orig
                        if (
                            attn_enabled
                            and budget_exhausted_without_safety_eos
                            and attn_ready
                        ):
                            selection_mode = "attn_budget_gate"
                            choose_safety = attn_metric_value >= attn_metric_threshold

                safety_token_count = len(log_probs_safety)
                punct_ratio = (
                    float(safety_punct_like_count) / float(safety_token_count)
                    if safety_token_count > 0
                    else 0.0
                )
                degenerate_safety = (
                    getattr(config, "ANTI_DEGEN_ENABLED", False)
                    and safety_token_count >= int(getattr(config, "ANTI_DEGEN_MIN_CHECK_TOKENS", 30))
                    and punct_ratio >= float(getattr(config, "ANTI_DEGEN_MAX_PUNCT_RATIO", 0.6))
                )
                if degenerate_safety and not orig_ended_early:
                    choose_safety = False
                    selection_mode = "anti_degen_gate"
                    log_print(
                        "   [anti-degen] safety[redacted][redacted][redacted][redacted][redacted]safety: "
                        f"punct_ratio={punct_ratio:.3f}, punct_like={safety_punct_like_count}, "
                        f"tokens={safety_token_count}"
                    )

                if budget_exhausted_without_safety_eos:
                    log_print(
                        "   [attn-summary] safety[redacted]token[redacted]EOS: "
                        f"metric={attn_selection_metric}, "
                        f"attn_metric_value={attn_metric_value:.6f}, "
                        f"attn_metric_threshold={attn_metric_threshold:.6f}, "
                        f"post_insert_attn_avg={avg_post_insert_attn:.6f}, "
                        f"post_insert_attn_avg_first10={avg_post_insert_attn_first10:.6f}, "
                        f"samples={len(post_insert_selected_attn_scores)}, "
                        f"mode={selection_mode}"
                    )

                path_orig_segment = tokenizer.decode(
                    tokens_orig[parallel_start_len:], skip_special_tokens=True
                )
                path_safety_segment = tokenizer.decode(
                    tokens_safety[parallel_start_len:], skip_special_tokens=True
                )

                def decode_tail_with_special(tokens, start_idx, tail_tokens=20):
                    seg = tokens[start_idx:]
                    seg_list = seg.tolist() if isinstance(seg, torch.Tensor) else list(seg)
                    if not seg_list:
                        return ""
                    tail_ids = seg_list[-tail_tokens:]
                    return tokenizer.decode(tail_ids, skip_special_tokens=False)

                orig_last_id = int(tokens_orig[-1].item()) if len(tokens_orig) > 0 else -1
                safety_last_id = int(tokens_safety[-1].item()) if len(tokens_safety) > 0 else -1
                orig_last_text = (
                    tokenizer.decode([orig_last_id], skip_special_tokens=False)
                    if orig_last_id >= 0
                    else ""
                )
                safety_last_text = (
                    tokenizer.decode([safety_last_id], skip_special_tokens=False)
                    if safety_last_id >= 0
                    else ""
                )
                path_orig_tail_raw = decode_tail_with_special(
                    tokens_orig, parallel_start_len, tail_tokens=20
                )
                path_safety_tail_raw = decode_tail_with_special(
                    tokens_safety, parallel_start_len, tail_tokens=20
                )
                attn_focus_path_segment = ""
                if safety_attn_best_token_ids:
                    attn_focus_path_segment = tokenizer.decode(
                        safety_attn_best_token_ids, skip_special_tokens=True
                    ).strip()

                log_print(f"\n{'#' * 80}")
                log_print(f"📋 [redacted] {split_count + 1} [redacted] — [redacted][redacted][redacted][redacted]")
                log_print(f"   [redacted]: {trigger_reason}")
                log_print(
                    f"   [redacted] {len(log_probs_orig)} tokens, [redacted] {len(log_probs_safety)} tokens"
                )
                log_print("-" * 80)
                log_print("[redacted] main [redacted]")
                log_print(path_orig_segment)
                log_print("-" * 80)
                log_print("[redacted] safety [redacted]")
                log_print(path_safety_segment)
                log_print("-" * 80)
                log_print(
                    f"   [debug] orig_ended_early={orig_ended_early}, safety_ended_early={safety_ended_early}"
                )
                log_print(
                    f"   [debug] [redacted]token: id={orig_last_id}, text={repr(orig_last_text)} | "
                    f"[redacted]token: id={safety_last_id}, text={repr(safety_last_text)}"
                )
                log_print("[redacted] main [redacted][redacted][redacted]special[redacted][redacted]")
                log_print(path_orig_tail_raw)
                log_print("[redacted] safety [redacted][redacted][redacted]special[redacted][redacted]")
                log_print(path_safety_tail_raw)
                if attn_focus_path_segment:
                    log_print("[redacted] attention[redacted][redacted][redacted]top-k[redacted]token[redacted][redacted]")
                    log_print(attn_focus_path_segment)
                log_print("-" * 80)
                matched_inserted_prefix_len = _match_inserted_prefix_len(
                    tokens_safety,
                    parallel_start_len,
                    token_sets["SORRY_ASSIST_IDS"],
                )
                attention_source_positions = list(
                    range(
                        parallel_start_len,
                        parallel_start_len + matched_inserted_prefix_len,
                    )
                )
                attention_source_tokens = _debug_token_details(
                    tokenizer, tokens_safety, attention_source_positions
                )
                _debug_block(
                    "Branch",
                    "Branch Segments",
                    [
                        ("split_index", split_count + 1),
                        ("original_range", f"[{parallel_start_len}, {len(tokens_orig)})"),
                        ("safety_range", f"[{parallel_start_len}, {len(tokens_safety)})"),
                        ("original_tokens", len(log_probs_orig)),
                        ("safety_tokens", len(log_probs_safety)),
                        ("original_eos", bool(orig_ended_early)),
                        ("safety_eos", bool(safety_ended_early)),
                    ],
                    body=(
                        "Original branch:\n"
                        f"{_debug_snippet(path_orig_segment)}\n\n"
                        "Safety branch:\n"
                        f"{_debug_snippet(path_safety_segment)}"
                    ),
                )
                _debug_block(
                    "Selection",
                    "Attention-Based Selection",
                    [
                        ("split_index", split_count + 1),
                        ("attention_source_positions", attention_source_positions),
                        (
                            "attention_source_tokens",
                            json.dumps(attention_source_tokens, ensure_ascii=False),
                        ),
                        ("original_attention_score", "N/A (no inserted safety span)"),
                        ("safety_attention_score", f"{avg_safety_selected_attn:.6f}"),
                        ("post_insert_attention_score", f"{avg_post_insert_attn:.6f}"),
                        ("metric", attn_selection_metric),
                        ("metric_raw", f"{attn_metric_value_raw:.6f}"),
                        ("attn_length_scale", f"{attn_length_scale:.6f}"),
                        ("metric_scaled", f"{attn_metric_value:.6f}"),
                        ("threshold", f"{attn_metric_threshold:.6f}"),
                        (
                            "comparison",
                            f"{attn_metric_value:.6f} >= {attn_metric_threshold:.6f} => "
                            f"{bool(attn_metric_value >= attn_metric_threshold)}",
                        ),
                        ("metric_count", attn_metric_count),
                        ("selection_mode", selection_mode),
                        ("selected_path", "safety" if choose_safety else "main"),
                    ],
                )

                split_count += 1
                if choose_safety:
                    tokens_main = tokens_safety
                    selected_path = "safety"
                    if safety_ended_early and not orig_ended_early:
                        reason = "[redacted](EOS)"
                    elif selection_mode in {"attn_budget_gate", "attn_only"}:
                        reason = (
                            "attention[redacted] "
                            f"(metric={attn_selection_metric}, "
                            f"value={attn_metric_value:.6f} >= "
                            f"threshold={attn_metric_threshold:.6f}, "
                            f"mode={selection_mode})"
                        )
                    else:
                        reason = (
                            f"[redacted] (mean_lp_safety={mean_lp_safety:.4f} > "
                            f"mean_lp_orig={mean_lp_orig:.4f})"
                        )
                    log_print(f"📌 [redacted] {split_count} [redacted] → [redacted] safety [redacted] [{reason}]")
                else:
                    tokens_main = tokens_orig
                    selected_path = "main"
                    if orig_ended_early and not safety_ended_early:
                        reason = "[redacted](EOS)"
                    elif selection_mode in {"attn_budget_gate", "attn_only"}:
                        reason = (
                            "attention[redacted] "
                            f"(metric={attn_selection_metric}, "
                            f"value={attn_metric_value:.6f} < "
                            f"threshold={attn_metric_threshold:.6f}, "
                            f"mode={selection_mode})"
                        )
                    elif selection_mode == "attn_only_insufficient_signal":
                        min_tokens = int(getattr(config, "POST_INSERT_ATTN_MIN_TOKENS", 1))
                        reason = (
                            "attention[redacted] "
                            f"(count={attn_metric_count} < min_tokens={min_tokens}, "
                            f"mode={selection_mode})"
                        )
                    elif selection_mode == "anti_degen_gate":
                        reason = (
                            "anti-degen[redacted] "
                            f"(punct_ratio={punct_ratio:.3f} >= "
                            f"max={float(getattr(config, 'ANTI_DEGEN_MAX_PUNCT_RATIO', 0.6)):.3f})"
                        )
                    else:
                        reason = (
                            f"[redacted] (mean_lp_orig={mean_lp_orig:.4f} >= "
                            f"mean_lp_safety={mean_lp_safety:.4f})"
                        )
                    log_print(f"📌 [redacted] {split_count} [redacted] → [redacted] main [redacted] [{reason}]")
                selected_path_history.append(selected_path)
                insertion_attention_trace["splits"].append(
                    {
                        "split_index": split_count,
                        "selected_path": selected_path,
                        "selection_mode": selection_mode,
                        "attn_selection_metric": attn_selection_metric,
                        "attn_selection_value": attn_metric_value,
                        "attn_selection_value_raw": attn_metric_value_raw,
                        "attn_selection_threshold": attn_metric_threshold,
                        "attn_selection_compare": bool(
                            attn_metric_value >= attn_metric_threshold
                        ),
                        "attn_selection_count": attn_metric_count,
                        "prompt_token_len": int(prompt_token_len),
                        "attn_length_scale": float(attn_length_scale),
                        "attn_length_scale_p0": float(
                            getattr(config, "ATTN_PROMPTLEN_SCALE_P0", 0.0)
                        ),
                        "attn_length_scale_beta": float(
                            getattr(config, "ATTN_PROMPTLEN_SCALE_BETA", 0.0)
                        ),
                        "post_insert_selected_attn_avg": avg_post_insert_attn,
                        "post_insert_selected_attn_avg_first10": avg_post_insert_attn_first10,
                        "post_insert_selected_attn_count": len(post_insert_selected_attn_scores),
                        "safety_selected_attn_avg": avg_safety_selected_attn,
                        "safety_selected_attn_avg_first10": avg_safety_selected_attn_first10,
                        "safety_selected_attn_count": len(safety_selected_attn_scores),
                        "attention_source_positions": attention_source_positions,
                        "attention_source_tokens": attention_source_tokens,
                        "attn_focus_token_ids": safety_attn_best_token_ids,
                        "attn_focus_path_text": attn_focus_path_segment,
                        "attn_focus_records": safety_attn_records,
                        "attn_diag_stats": attn_diag_stats,
                        "selected_token_attn_per_intervention_token": safety_selected_per_intervention_token,
                        "anti_degen": {
                            "enabled": bool(getattr(config, "ANTI_DEGEN_ENABLED", False)),
                            "punct_like_count": int(safety_punct_like_count),
                            "safety_token_count": int(safety_token_count),
                            "punct_ratio": float(punct_ratio),
                            "degenerate_safety": bool(degenerate_safety),
                        },
                    }
                )
                log_print(f"{'#' * 80}\n")

                stop_on_selected_eos = (choose_safety and safety_ended_early) or (
                    (not choose_safety) and orig_ended_early
                )
                if stop_on_selected_eos:
                    log_print("⏹️ [redacted] EOS[redacted][redacted]")
                    break
                continue

            next_id = _sample_next_token_from_logits(logits[0])
            tokens_main = torch.cat([tokens_main, torch.tensor([next_id], device=device)])
            if next_id == tokenizer.eos_token_id:
                break
            if answer_only_mode:
                generated_text = tokenizer.decode(
                    tokens_main[initial_token_count:], skip_special_tokens=True
                )
                if _extract_direct_option_letter(generated_text):
                    break
                if len(tokens_main) - initial_token_count >= 4:
                    break

    result_text = tokenizer.decode(tokens_main[initial_token_count:], skip_special_tokens=True)
    if answer_only_mode:
        extracted = _extract_direct_option_letter(result_text)
        result_text = extracted if extracted else result_text.strip()
    return (
        result_text,
        selected_path,
        split_count,
        selected_path_history,
        insertion_attention_trace,
    )


def run_original_response(prompt, runtime):
    if _is_sglang_backend(runtime):
        return _run_sglang_original_response(prompt=prompt, runtime=runtime)

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]

    system_prompt = config.build_system_prompt()
    if system_prompt == "":
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    initial_token_count = len(input_ids[0])
    tokens = input_ids[0]

    with torch.no_grad():
        while True:
            if len(tokens) - initial_token_count >= config.MAX_TOKENS:
                break
            out = model(tokens.unsqueeze(0))
            logits = out.logits[:, -1, :].clone()
            next_id = _sample_next_token_from_logits(logits[0], force_greedy=False)
            tokens = torch.cat([tokens, torch.tensor([next_id], device=device)])
            if next_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(tokens[initial_token_count:], skip_special_tokens=True)
