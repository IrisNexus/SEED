import copy
import json
import math
import os
from typing import Any, Dict, List, Optional

from sglang.srt.sampling.custom_logit_processor import (
    BeamEntropyStopLogitProcessor,
    BeamEntropyTelemetryLogitProcessor,
    BeamForceTokenIdsLogitProcessor,
    BeamSafetyBranchLogitProcessor,
)


def _meta_last(meta_info: Dict[str, Any], key: str, default=None):
    value = meta_info.get(key, default)
    if isinstance(value, list) and value:
        return value[-1]
    return value


def _trace_is_enabled(params: Dict[str, Any]) -> bool:
    return bool(params.get("controller_trace_enabled", False)) and bool(
        str(params.get("controller_trace_log_path", "") or "").strip()
    )


def _trace_write(params: Dict[str, Any], event: str, payload: Dict[str, Any]):
    if not _trace_is_enabled(params):
        return
    path = str(params.get("controller_trace_log_path", "") or "").strip()
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                f"{event}\t"
                f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}\n"
            )
    except Exception:
        return


def _debug_is_enabled(params: Dict[str, Any]) -> bool:
    return bool(params.get("debug_verbose", False))


def _debug_write(params: Dict[str, Any], category: str, message: str):
    if not _debug_is_enabled(params):
        return
    line = f"[DEBUG][{category}] {message}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    path = str(params.get("debug_log_path", "") or "").strip()
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return


def _debug_block(
    params: Dict[str, Any],
    category: str,
    title: str,
    fields: Optional[List[Any]] = None,
    body: Optional[str] = None,
):
    if not _debug_is_enabled(params):
        return
    width = 88
    lines = [f"{'=' * 8} {title} {'=' * 8}"]
    for key, value in fields or []:
        lines.append(f"  {str(key):<28}: {value}")
    if body:
        lines.append(f"  {'-' * (width - 18)}")
        lines.extend(f"  {line}" for line in str(body).splitlines() or [""])
    lines.append("=" * width)
    for line in lines:
        _debug_write(params, category, line)


def _debug_snippet(params: Dict[str, Any], text: str) -> str:
    try:
        max_chars = int(params.get("debug_snippet_chars", 600))
    except (TypeError, ValueError):
        max_chars = 600
    out = str(text or "").replace("\r", "\\r")
    if len(out) <= max_chars:
        return out
    return out[:max_chars] + f"... <truncated {len(out) - max_chars} chars>"


def _token_details_for_positions(tokenizer, tokens: List[int], positions: List[int]):
    details = []
    for pos in positions:
        try:
            pos_int = int(pos)
        except (TypeError, ValueError):
            continue
        if pos_int < 0 or pos_int >= len(tokens):
            details.append(
                {
                    "position": pos_int,
                    "token_id": None,
                    "token_text": "<out_of_range>",
                }
            )
            continue
        token_id = int(tokens[pos_int])
        details.append(
            {
                "position": pos_int,
                "token_id": token_id,
                "token_text": _decode(tokenizer, [token_id], skip_special_tokens=False),
            }
        )
    return details


def _to_float_or_none(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _output_logprobs(meta_info: Dict[str, Any]) -> List[float]:
    output = []
    for item in meta_info.get("output_token_logprobs", []) or []:
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            continue
    return output


def _top_candidate_ids(meta_info: Dict[str, Any], fallback_ids: List[int]) -> List[int]:
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


def _attention_values(meta_info: Dict[str, Any]) -> List[Optional[float]]:
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


def _matching_insert_prefix(output_ids: List[int], insertion_ids: List[int]) -> int:
    matched = 0
    for actual_id, expected_id in zip(output_ids, insertion_ids):
        if int(actual_id) != int(expected_id):
            break
        matched += 1
    return matched


def _insertion_signal_scores(
    output_ids: List[int], attention_values: List[Optional[float]], insertion_ids: List[int]
) -> List[float]:
    scores = []
    for idx, value in enumerate(attention_values[: len(output_ids)]):
        if value is None:
            continue
        if _matching_insert_prefix(output_ids[:idx], insertion_ids) > 0:
            scores.append(float(value))
    return scores


def _score_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _entropy_threshold(params: Dict[str, Any], split_count: int) -> float:
    max_insertions = int(params.get("max_insertions", params.get("max_trigger_count", 2)))
    low = float(params.get("entropy_threshold_low", 1.2))
    high = float(params.get("entropy_threshold_high", 2.0))
    decay_rate = float(params.get("entropy_threshold_decay_rate", 3.0))
    if max_insertions <= 1:
        return low
    capped = max(0, min(int(split_count), int(max_insertions - 1)))
    progress = capped / float(max_insertions - 1)
    decay = math.exp(-decay_rate * progress)
    decay_end = math.exp(-decay_rate)
    normalized = 1.0 - ((decay - decay_end) / (1.0 - decay_end))
    return float(low + (high - low) * normalized)


def _promptlen_attn_scale(params: Dict[str, Any], prompt_token_len: int) -> float:
    beta = float(params.get("attn_promptlen_scale_beta", 0.0))
    p0 = float(params.get("attn_promptlen_scale_p0", 1.0))
    p = max(0.0, float(prompt_token_len))
    if beta <= 0.0 or p0 <= 0.0:
        return 1.0
    return 1.0 + beta * math.log(1.0 + p / p0)


def _decode(tokenizer, token_ids: List[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _is_sentence_boundary(tokens: List[int], tokenizer, params: Dict[str, Any]) -> bool:
    if not tokens:
        return True
    recent_tokens = tokens[-10:] if len(tokens) >= 10 else tokens
    recent_text = _decode(tokenizer, recent_tokens, skip_special_tokens=False)
    recent_text_stripped = recent_text.rstrip()
    for ending in params.get("sentence_endings", [".", "!", "?", "\n"]):
        if recent_text_stripped.endswith(str(ending)):
            return True
    return recent_text_stripped == ""


def _find_last_sentence_boundary_in_generated(
    tokens_main: List[int], initial_token_count: int, tokenizer, params: Dict[str, Any]
):
    generated_count = len(tokens_main) - initial_token_count
    if generated_count <= 0:
        return initial_token_count, False
    for k in range(generated_count, 0, -1):
        span = tokens_main[initial_token_count : initial_token_count + k]
        text_stripped = _decode(tokenizer, span, skip_special_tokens=True).rstrip()
        for ending in params.get("sentence_endings", [".", "!", "?", "\n"]):
            if text_stripped.endswith(str(ending)):
                return initial_token_count + k, True
    return initial_token_count, False


def _is_punct_like_token(tokenizer, token_id: int, punct_token_ids: List[int]) -> bool:
    if int(token_id) in set(int(x) for x in punct_token_ids):
        return True
    token_text = _decode(tokenizer, [int(token_id)], skip_special_tokens=False).strip()
    if token_text == "":
        return False
    return all(ch in ".,?!;:，。！？；：…-—" for ch in token_text)


def _base_sampling_params(sampling_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = copy.deepcopy(sampling_params or {})
    custom_params = dict(params.get("custom_params") or {})
    custom_params.pop("beam_safety_controller_enabled", None)
    params["custom_params"] = custom_params
    return params


def _generation_params(
    sampling_params: Dict[str, Any], max_new_tokens: int, custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    params = copy.deepcopy(sampling_params)
    params["max_new_tokens"] = max(0, int(max_new_tokens))
    if custom_params is not None:
        params["custom_params"] = custom_params
    else:
        params.pop("custom_params", None)
    return params


def _last_attention_layers(engine):
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


def _attention_kwargs(engine, params: Dict[str, Any], source_positions: List[int], max_new_tokens: int):
    if not bool(params.get("sglang_enable_return_attention_signal", False)):
        return {}
    positions = []
    for x in source_positions or []:
        try:
            ix = int(x)
        except (TypeError, ValueError):
            continue
        if ix >= 0:
            positions.append(ix)
    if not positions or int(max_new_tokens) <= 0:
        return {}

    step_cap = max(0, int(params.get("sglang_attention_signal_decode_max_steps", 0)))
    if step_cap <= 0:
        return {}
    stride = max(1, int(params.get("sglang_attention_signal_decode_step_stride", 1)))
    steps = list(range(0, min(int(max_new_tokens), step_cap), stride))
    if not steps:
        return {}

    kwargs = {
        "attention_signal_source_positions": positions,
        "attention_signal_decode_steps": steps,
    }
    layers = params.get("sglang_attention_signal_layers")
    if layers is None:
        layers = _last_attention_layers(engine)
    if layers is not None:
        kwargs["attention_signal_layers"] = layers
    return kwargs


def _plain_generate_from_ids(
    engine,
    input_ids: List[int],
    sampling_params: Dict[str, Any],
    max_new_tokens: int,
    *,
    custom_processor=None,
    custom_params: Optional[Dict[str, Any]] = None,
    return_logprob: bool = False,
    top_logprobs_num: int = 0,
    attention_source_positions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    controller_params = dict((sampling_params.get("custom_params") or {}))
    gen_sampling_params = _generation_params(
        sampling_params, max_new_tokens=max_new_tokens, custom_params=custom_params
    )
    kwargs = {
        "input_ids": [int(x) for x in input_ids],
        "sampling_params": gen_sampling_params,
    }
    if custom_processor is not None:
        kwargs["custom_logit_processor"] = custom_processor.to_str()
    if return_logprob:
        kwargs["return_logprob"] = True
        kwargs["logprob_start_len"] = -1
        kwargs["top_logprobs_num"] = int(top_logprobs_num)
    if attention_source_positions:
        kwargs.update(
            _attention_kwargs(
                engine,
                controller_params,
                attention_source_positions,
                max_new_tokens=max_new_tokens,
            )
        )

    out = engine.generate(**kwargs)
    if not isinstance(out, dict):
        return {"text": str(out), "output_ids": [], "meta_info": {}}
    return {
        "text": str(out.get("text", "")),
        "output_ids": [int(x) for x in (out.get("output_ids") or [])],
        "meta_info": out.get("meta_info", {}) or {},
    }


def _safety_branch_custom_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "safety_phrase_token_ids": [
            int(x) for x in params.get("safety_phrase_token_ids", [])
        ],
        "force_exact_insertion_ids": bool(params.get("force_exact_insertion_ids", False)),
        "sorry_bonus_scale": float(params.get("sorry_bonus_scale", 1.5)),
        "sorry_bonus_min": float(params.get("sorry_bonus_min", 3.5)),
        "sorry_bonus_max": float(params.get("sorry_bonus_max", 50.0)),
        "sorry_decay_lambda": float(params.get("sorry_decay_lambda", 0.01)),
        "force_first_token_win": bool(params.get("force_first_token_win", False)),
        "first_token_win_margin": float(params.get("first_token_win_margin", 0.2)),
        "anti_degen_enabled": bool(params.get("anti_degen_enabled", False)),
        "anti_degen_repeat_window": int(params.get("anti_degen_repeat_window", 6)),
        "anti_degen_punct_token_ids": [
            int(x) for x in params.get("anti_degen_punct_token_ids", [])
        ],
    }


def _generate_entropy_step(engine, tokens: List[int], sampling_params: Dict[str, Any]):
    out = _plain_generate_from_ids(
        engine,
        tokens,
        sampling_params,
        max_new_tokens=1,
        custom_processor=BeamEntropyTelemetryLogitProcessor,
        custom_params={"telemetry_top_k": int(sampling_params.get("custom_params", {}).get("top_k", 10))},
    )
    entropy = _to_float_or_none(_meta_last(out["meta_info"], "beam_parity_entropy"))
    return out, entropy


def _as_list(value) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _generate_entropy_scan(
    engine,
    tokens: List[int],
    sampling_params: Dict[str, Any],
    max_new_tokens: int,
    threshold: float,
):
    params = sampling_params.get("custom_params", {}) or {}
    out = _plain_generate_from_ids(
        engine,
        tokens,
        sampling_params,
        max_new_tokens=max_new_tokens,
        custom_processor=BeamEntropyStopLogitProcessor,
        custom_params={
            "entropy_threshold": float(threshold),
        },
    )
    entropies = _as_list(out["meta_info"].get("beam_entropy_scan_entropy"))
    triggered_flags = _as_list(out["meta_info"].get("beam_entropy_scan_triggered"))
    trigger_index = None
    for idx, raw in enumerate(triggered_flags):
        if bool(raw):
            trigger_index = idx
            break
    if trigger_index is None:
        return {
            "accepted_ids": list(out["output_ids"]),
            "triggered": False,
            "entropy": _to_float_or_none(entropies[-1]) if entropies else None,
            "raw_output_ids": list(out["output_ids"]),
            "trigger_index": None,
            "scan_steps": len(entropies),
        }

    return {
        "accepted_ids": list(out["output_ids"][:trigger_index]),
        "triggered": True,
        "entropy": _to_float_or_none(entropies[trigger_index])
        if trigger_index < len(entropies)
        else None,
        "raw_output_ids": list(out["output_ids"]),
        "trigger_index": int(trigger_index),
        "scan_steps": len(entropies),
    }


def _mean_lp_after_insert(logprobs: List[float], insert_len: int) -> float:
    if not logprobs:
        return -1e9
    after = logprobs[max(0, int(insert_len)) :]
    values = after if after else logprobs
    return float(sum(values) / len(values))


def _force_candidate(
    engine,
    safety_tokens: List[int],
    candidate_id: int,
    source_positions: List[int],
    sampling_params: Dict[str, Any],
):
    out = _plain_generate_from_ids(
        engine,
        safety_tokens,
        sampling_params,
        max_new_tokens=2,
        custom_processor=BeamForceTokenIdsLogitProcessor,
        custom_params={"forced_token_ids": [int(candidate_id)]},
        return_logprob=True,
        attention_source_positions=source_positions,
    )
    signal_values = _attention_values(out["meta_info"])
    score = next((value for value in signal_values if value is not None), 0.0)
    out["output_ids"] = out["output_ids"][:1]
    output_logprobs = out["meta_info"].get("output_token_logprobs")
    if isinstance(output_logprobs, list):
        out["meta_info"]["output_token_logprobs"] = output_logprobs[:1]
    return out, float(score)


def _safety_branch(engine, tokens: List[int], sampling_params: Dict[str, Any], max_new_tokens: int):
    params = sampling_params.get("custom_params", {}) or {}
    tokenizer = engine.tokenizer_manager.tokenizer
    insertion_ids = [int(x) for x in params.get("safety_phrase_token_ids", [])]
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
    insert_out = _plain_generate_from_ids(
        engine,
        tokens,
        sampling_params,
        max_new_tokens=insert_budget,
        custom_processor=BeamSafetyBranchLogitProcessor,
        custom_params=_safety_branch_custom_params(params),
        return_logprob=True,
        attention_source_positions=[max(0, len(tokens) - 1)],
    )
    segment_ids = list(insert_out["output_ids"])
    logprobs = _output_logprobs(insert_out["meta_info"])
    insert_attention = _attention_values(insert_out["meta_info"])
    safety_scores = _insertion_signal_scores(segment_ids, insert_attention, insertion_ids)
    matched_inserted_len = _matching_insert_prefix(segment_ids, insertion_ids)
    source_positions = [int(len(tokens) + idx) for idx in range(max(0, matched_inserted_len))]
    current_tokens = [*tokens, *segment_ids]
    eos_id = int(params.get("eos_token_id", -1))
    ended_early = bool(segment_ids and eos_id >= 0 and int(segment_ids[-1]) == eos_id)
    post_insert_scores = []
    first_post_insert_attention_driven = False
    remaining = int(max_new_tokens) - len(segment_ids)

    if remaining > 0 and not ended_early:
        first_out = None
        first_score = 0.0
        can_attention_select = bool(
            params.get("enable_attn_driven_first_post_insert_token", False)
            and source_positions
        )
        if can_attention_select:
            probe = _plain_generate_from_ids(
                engine,
                current_tokens,
                sampling_params,
                max_new_tokens=1,
                return_logprob=True,
                top_logprobs_num=max(1, int(params.get("attn_trace_top_k", 10))),
                attention_source_positions=source_positions,
            )
            candidate_ids = _top_candidate_ids(probe["meta_info"], probe["output_ids"])
            best = None
            for candidate_id in candidate_ids:
                candidate_out, candidate_score = _force_candidate(
                    engine,
                    current_tokens,
                    candidate_id=candidate_id,
                    source_positions=source_positions,
                    sampling_params=sampling_params,
                )
                if best is None or candidate_score > best[1]:
                    best = (candidate_out, candidate_score)
            if best is not None:
                first_out, first_score = best
                first_post_insert_attention_driven = True

        if first_out is None:
            first_out = _plain_generate_from_ids(
                engine,
                current_tokens,
                sampling_params,
                max_new_tokens=1,
                return_logprob=True,
                attention_source_positions=source_positions,
            )
            signal_values = _attention_values(first_out["meta_info"])
            if signal_values and signal_values[0] is not None:
                first_score = float(signal_values[0])

        first_ids = list(first_out["output_ids"])
        segment_ids.extend(first_ids)
        logprobs.extend(_output_logprobs(first_out["meta_info"]))
        if first_ids and source_positions:
            safety_scores.append(float(first_score))
        current_tokens.extend(first_ids)
        remaining -= len(first_ids)
        ended_early = bool(first_ids and eos_id >= 0 and int(first_ids[-1]) == eos_id)

    while remaining > 0 and not ended_early:
        # Long non-streaming internal generations can return attention_signal
        # for only part of the decoded segment. Chunking keeps token ids and
        # attention telemetry aligned across the full safety branch budget.
        chunk_new_tokens = min(remaining, 32)
        rest_out = _plain_generate_from_ids(
            engine,
            current_tokens,
            sampling_params,
            max_new_tokens=chunk_new_tokens,
            return_logprob=True,
            attention_source_positions=source_positions,
        )
        rest_ids = list(rest_out["output_ids"])
        if not rest_ids:
            break
        segment_ids.extend(rest_ids)
        logprobs.extend(_output_logprobs(rest_out["meta_info"]))
        for value in _attention_values(rest_out["meta_info"])[: len(rest_ids)]:
            if value is None or not source_positions:
                continue
            safety_scores.append(float(value))
            post_insert_scores.append(float(value))
        current_tokens.extend(rest_ids)
        remaining -= len(rest_ids)
        ended_early = bool(rest_ids and eos_id >= 0 and int(rest_ids[-1]) == eos_id)

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


def _main_branch(engine, tokens: List[int], sampling_params: Dict[str, Any], max_new_tokens: int):
    out = _plain_generate_from_ids(
        engine,
        tokens,
        sampling_params,
        max_new_tokens=max_new_tokens,
        return_logprob=True,
    )
    params = sampling_params.get("custom_params", {}) or {}
    segment_ids = list(out["output_ids"])
    eos_id = int(params.get("eos_token_id", -1))
    return {
        "tokens": [*tokens, *segment_ids],
        "segment_ids": segment_ids,
        "logprobs": _output_logprobs(out["meta_info"]),
        "ended_early": bool(segment_ids and eos_id >= 0 and int(segment_ids[-1]) == eos_id),
    }


def _route_split(
    engine,
    main_branch: Dict[str, Any],
    safety_branch: Dict[str, Any],
    sampling_params: Dict[str, Any],
    prompt_token_len: int,
    branch_budget: int,
) -> Dict[str, Any]:
    params = sampling_params.get("custom_params", {}) or {}
    tokenizer = engine.tokenizer_manager.tokenizer
    insertion_len = len(params.get("safety_phrase_token_ids", []))
    mean_lp_orig = _mean_lp_after_insert(main_branch["logprobs"], insertion_len)
    mean_lp_safety = _mean_lp_after_insert(safety_branch["logprobs"], insertion_len)
    safety_scores = safety_branch["safety_scores"]
    post_insert_scores = safety_branch["post_insert_scores"]
    metric = str(params.get("attn_selection_metric", "post_insert"))
    if metric not in {"post_insert", "safety_overall"}:
        metric = "post_insert"
    if metric == "safety_overall":
        metric_values = safety_scores
        threshold = float(params.get("safety_overall_attn_threshold", 0.0))
    else:
        metric_values = post_insert_scores
        threshold = float(params.get("post_insert_attn_threshold", 0.0))
    metric_raw = _score_mean(metric_values)
    metric_scaled = metric_raw * _promptlen_attn_scale(params, prompt_token_len)
    metric_count = len(metric_values)
    attn_enabled = bool(params.get("enable_post_insert_attn_selection", False))
    attn_only = bool(params.get("attn_selection_only", False))
    attn_ready = attn_enabled and metric_count >= int(params.get("post_insert_attn_min_tokens", 1))

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

    punct_ids = [int(x) for x in params.get("anti_degen_punct_token_ids", [])]
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
        params.get("anti_degen_enabled", False)
        and len(safety_branch["segment_ids"]) >= int(params.get("anti_degen_min_check_tokens", 30))
        and punct_ratio >= float(params.get("anti_degen_max_punct_ratio", 0.6))
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


def run_beam_safety_controller(
    engine,
    *,
    input_ids: List[int],
    sampling_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    tokenizer = engine.tokenizer_manager.tokenizer
    base_sampling_params = _base_sampling_params(sampling_params)
    params = dict((sampling_params or {}).get("custom_params") or {})
    params.pop("beam_safety_controller_enabled", None)
    base_sampling_params["custom_params"] = params
    tokens_main = [int(x) for x in input_ids]
    initial_token_count = len(tokens_main)
    prompt_token_len = int(params.get("prompt_token_len", initial_token_count))
    max_tokens = int(base_sampling_params.get("max_new_tokens", params.get("max_tokens", 0)))
    max_insertions = int(params.get("max_insertions", params.get("max_trigger_count", 2)))
    split_count = 0
    selected_path = "main"
    selected_path_history = []
    split_mode_history = []
    insertion_attention_trace = {
        "enabled": bool(params.get("enable_post_insert_attn_selection", False)),
        "backend": "sglang_internal_decoding_controller",
        "splits": [],
    }
    trace_enabled = _trace_is_enabled(params)
    _debug_block(
        params,
        "SGLang",
        "Controller Start",
        [
            ("initial_token_count", int(initial_token_count)),
            ("prompt_token_len", int(prompt_token_len)),
            ("max_tokens", int(max_tokens)),
            ("max_insertions", int(max_insertions)),
        ],
    )
    if trace_enabled:
        _trace_write(
            params,
            "controller_start",
            {
                "initial_token_count": int(initial_token_count),
                "prompt_token_len": int(prompt_token_len),
                "max_tokens": int(max_tokens),
                "max_insertions": int(max_insertions),
                "entropy_scan_mode": "until_trigger",
                "attention_signal_enabled": bool(
                    params.get("sglang_enable_return_attention_signal", False)
                ),
                "attention_selection_enabled": bool(
                    params.get("enable_post_insert_attn_selection", False)
                ),
                "attention_selection_only": bool(params.get("attn_selection_only", False)),
            },
        )

    while len(tokens_main) - initial_token_count < max_tokens:
        remaining_to_max = max_tokens - (len(tokens_main) - initial_token_count)
        if split_count >= max_insertions:
            final_out = _plain_generate_from_ids(
                engine,
                tokens_main,
                base_sampling_params,
                max_new_tokens=remaining_to_max,
            )
            final_ids = list(final_out["output_ids"])
            if final_ids:
                tokens_main.extend(int(x) for x in final_ids)
            if trace_enabled:
                _trace_write(
                    params,
                    "entropy_monitor_disabled",
                    {
                        "split_count": int(split_count),
                        "max_insertions": int(max_insertions),
                        "remaining_budget": int(remaining_to_max),
                        "generated_token_count": int(
                            len(tokens_main) - initial_token_count
                        ),
                        "normal_generated_token_count": int(len(final_ids)),
                    },
                )
            break

        threshold = (
            _entropy_threshold(params, split_count)
        )
        scan_out = _generate_entropy_scan(
            engine,
            tokens_main,
            base_sampling_params,
            max_new_tokens=remaining_to_max,
            threshold=threshold,
        )
        entropy = scan_out["entropy"]
        if entropy is None:
            entropy = 0.0
        generated_before_scan = len(tokens_main) - initial_token_count
        if scan_out["accepted_ids"]:
            tokens_main.extend(int(x) for x in scan_out["accepted_ids"])
        should_trigger = bool(scan_out["triggered"])
        trigger_index = scan_out.get("trigger_index")
        trigger_decode_step = (
            len(tokens_main) - initial_token_count
            if trigger_index is None
            else (generated_before_scan + int(trigger_index))
        )
        trigger_token_index = initial_token_count + trigger_decode_step
        _debug_block(
            params,
            "Entropy",
            "Entropy Trigger Check",
            [
                ("decode_step", int(trigger_decode_step)),
                ("token_index", int(trigger_token_index)),
                ("entropy", f"{float(entropy):.6f}"),
                ("threshold", f"{float(threshold):.6f}"),
                (
                    "comparison",
                    f"{float(entropy):.6f} > {float(threshold):.6f} => {bool(should_trigger)}",
                ),
                ("triggered", bool(should_trigger)),
                ("trigger_count", f"{int(split_count)}/{int(max_insertions)}"),
                ("accepted_tokens", int(len(scan_out["accepted_ids"]))),
                ("scan_steps", int(scan_out.get("scan_steps") or 0)),
                ("trigger_index", trigger_index),
            ],
        )
        if trace_enabled and (
            should_trigger or bool(params.get("controller_trace_log_every_scan", False))
        ):
            _trace_write(
                params,
                "entropy_scan",
                {
                    "generated_token_count": int(len(tokens_main) - initial_token_count),
                    "split_count": int(split_count),
                    "threshold": float(threshold),
                    "entropy": float(entropy),
                    "triggered": bool(should_trigger),
                    "accepted_token_count": int(len(scan_out["accepted_ids"])),
                    "raw_output_token_count": int(len(scan_out["raw_output_ids"])),
                    "scan_steps": int(scan_out.get("scan_steps") or 0),
                    "trigger_index": scan_out.get("trigger_index"),
                },
            )

        if split_count < max_insertions and should_trigger:
            at_boundary = _is_sentence_boundary(tokens_main, tokenizer, params)
            skip_think_prefix = not bool(params.get("enable_think_prefix", False))
            rollback = "none"
            if not at_boundary:
                keep_length, found_boundary = _find_last_sentence_boundary_in_generated(
                    tokens_main, initial_token_count, tokenizer, params
                )
                if found_boundary:
                    tokens_main = list(tokens_main[:keep_length])
                    rollback = "last_sentence_boundary"
                else:
                    tokens_main = list(tokens_main[:initial_token_count])
                    skip_think_prefix = True
                    rollback = "initial_prefix"
            if split_count == 0:
                skip_think_prefix = True

            remaining_total = max_tokens - (len(tokens_main) - initial_token_count)
            branch_budget = min(
                max(0, remaining_total),
                int(params.get("safety_branch_max_tokens", 100)),
            )
            tokens_orig = list(tokens_main)
            tokens_safety_prefix = list(tokens_main)
            if not skip_think_prefix:
                tokens_safety_prefix.extend(
                    int(x) for x in params.get("think_only_token_ids", [])
                )
            if trace_enabled:
                _trace_write(
                    params,
                    "entropy_trigger",
                    {
                        "split_index": int(split_count + 1),
                        "generated_token_count": int(
                            len(tokens_main) - initial_token_count
                        ),
                        "entropy": float(entropy),
                        "threshold": float(threshold),
                        "at_sentence_boundary": bool(at_boundary),
                        "rollback": rollback,
                        "skip_think_prefix": bool(skip_think_prefix),
                        "remaining_total": int(remaining_total),
                        "branch_budget": int(branch_budget),
                    },
                )
            _debug_block(
                params,
                "Branch",
                "Enter Branch Exploration",
                [
                    ("split_index", int(split_count + 1)),
                    ("branch_start_token_index", int(len(tokens_main))),
                    ("branch_start_decode_step", int(len(tokens_main) - initial_token_count)),
                    ("entropy", f"{float(entropy):.6f}"),
                    ("threshold", f"{float(threshold):.6f}"),
                    ("remaining_total", int(remaining_total)),
                    ("branch_budget", int(branch_budget)),
                    ("at_sentence_boundary", bool(at_boundary)),
                    ("rollback", rollback),
                    ("skip_think_prefix", bool(skip_think_prefix)),
                ],
            )

            main = _main_branch(engine, tokens_orig, base_sampling_params, branch_budget)
            safety = _safety_branch(
                engine, tokens_safety_prefix, base_sampling_params, branch_budget
            )
            route = _route_split(
                engine,
                main,
                safety,
                base_sampling_params,
                prompt_token_len=prompt_token_len,
                branch_budget=branch_budget,
            )
            choose_safety = bool(route["choose_safety"])
            chosen = safety if choose_safety else main
            tokens_main = list(chosen["tokens"])
            selected_path = "safety" if choose_safety else "main"
            branch_start = len(tokens_orig)
            safety_source_start = len(tokens_safety_prefix)
            attention_source_positions = [
                int(safety_source_start + idx)
                for idx in range(max(0, int(safety["matched_inserted_len"])))
            ]
            attention_source_tokens = _token_details_for_positions(
                tokenizer, safety["tokens"], attention_source_positions
            )
            main_segment = _decode(
                tokenizer, list(main["segment_ids"]), skip_special_tokens=True
            )
            safety_segment = _decode(
                tokenizer, list(safety["segment_ids"]), skip_special_tokens=True
            )
            _debug_block(
                params,
                "Branch",
                "Branch Segments",
                [
                    ("split_index", int(split_count + 1)),
                    (
                        "original_range",
                        f"[{int(branch_start)}, {int(branch_start + len(main['segment_ids']))})",
                    ),
                    (
                        "safety_range",
                        f"[{int(safety_source_start)}, {int(safety_source_start + len(safety['segment_ids']))})",
                    ),
                    ("original_tokens", int(len(main["segment_ids"]))),
                    ("safety_tokens", int(len(safety["segment_ids"]))),
                    ("original_eos", bool(main["ended_early"])),
                    ("safety_eos", bool(safety["ended_early"])),
                ],
                body=(
                    "Original branch:\n"
                    f"{_debug_snippet(params, main_segment)}\n\n"
                    "Safety branch:\n"
                    f"{_debug_snippet(params, safety_segment)}"
                ),
            )
            _debug_block(
                params,
                "Selection",
                "Attention-Based Selection",
                [
                    ("split_index", int(split_count + 1)),
                    ("attention_source_positions", attention_source_positions),
                    (
                        "attention_source_tokens",
                        json.dumps(attention_source_tokens, ensure_ascii=False),
                    ),
                    ("original_attention_score", "N/A (no inserted safety span)"),
                    ("safety_attention_score", f"{float(_score_mean(safety['safety_scores'])):.6f}"),
                    ("post_insert_attention_score", f"{float(_score_mean(safety['post_insert_scores'])):.6f}"),
                    ("metric", str(route["metric"])),
                    ("metric_raw", f"{float(route['metric_raw']):.6f}"),
                    (
                        "attn_length_scale",
                        f"{float(_promptlen_attn_scale(params, prompt_token_len)):.6f}",
                    ),
                    ("metric_scaled", f"{float(route['metric_scaled']):.6f}"),
                    ("threshold", f"{float(route['metric_threshold']):.6f}"),
                    (
                        "comparison",
                        f"{float(route['metric_scaled']):.6f} >= "
                        f"{float(route['metric_threshold']):.6f} => "
                        f"{bool(route['metric_scaled'] >= route['metric_threshold'])}",
                    ),
                    ("metric_count", int(route["metric_count"])),
                    ("ready", bool(route["attn_ready"])),
                    ("selection_mode", str(route["selection_mode"])),
                    ("selected_path", selected_path),
                    (
                        "reason",
                        "attention_metric"
                        if str(route["selection_mode"])
                        in {"attn_only", "attn_budget_gate"}
                        else str(route["selection_mode"])
                    ),
                ],
            )
            split_count += 1
            selected_path_history.append(selected_path)
            split_mode_history.append(str(route["selection_mode"]))
            if trace_enabled:
                _trace_write(
                    params,
                    "branch_route",
                    {
                        "split_index": int(split_count),
                        "selected_path": selected_path,
                        "selection_mode": str(route["selection_mode"]),
                        "main_segment_tokens": int(len(main["segment_ids"])),
                        "safety_segment_tokens": int(len(safety["segment_ids"])),
                        "matched_inserted_prefix_len": int(
                            safety["matched_inserted_len"]
                        ),
                        "mean_lp_main": float(route["mean_lp_orig"]),
                        "mean_lp_safety": float(route["mean_lp_safety"]),
                        "attention_metric": str(route["metric"]),
                        "attention_value_raw": float(route["metric_raw"]),
                        "attention_value_scaled": float(route["metric_scaled"]),
                        "attention_threshold": float(route["metric_threshold"]),
                        "attention_compare": bool(
                            route["metric_scaled"] >= route["metric_threshold"]
                        ),
                        "attention_count": int(route["metric_count"]),
                        "attention_ready": bool(route["attn_ready"]),
                        "attention_source_positions": attention_source_positions,
                        "attention_source_tokens": attention_source_tokens,
                        "safety_attention_avg": float(
                            _score_mean(safety["safety_scores"])
                        ),
                        "safety_attention_count": int(len(safety["safety_scores"])),
                        "post_insert_attention_avg": float(
                            _score_mean(safety["post_insert_scores"])
                        ),
                        "post_insert_attention_count": int(
                            len(safety["post_insert_scores"])
                        ),
                        "first_post_insert_attention_driven": bool(
                            safety["first_post_insert_attention_driven"]
                        ),
                        "main_ended_early": bool(main["ended_early"]),
                        "safety_ended_early": bool(safety["ended_early"]),
                        "anti_degen_punct_ratio": float(route["punct_ratio"]),
                        "anti_degen_triggered": bool(route["degenerate_safety"]),
                    },
                )
            insertion_attention_trace["splits"].append(
                {
                    "split_index": split_count,
                    "selected_path": selected_path,
                    "selection_mode": route["selection_mode"],
                    "entropy": float(entropy),
                    "entropy_threshold": float(threshold),
                    "attn_selection_metric": route["metric"],
                    "attn_selection_value": route["metric_scaled"],
                    "attn_selection_value_raw": route["metric_raw"],
                    "attn_selection_threshold": route["metric_threshold"],
                    "attn_selection_compare": bool(
                        route["metric_scaled"] >= route["metric_threshold"]
                    ),
                    "attn_selection_count": route["metric_count"],
                    "attn_selection_ready": route["attn_ready"],
                    "prompt_token_len": int(prompt_token_len),
                    "attn_length_scale": float(_promptlen_attn_scale(params, prompt_token_len)),
                    "attention_source_positions": attention_source_positions,
                    "attention_source_tokens": attention_source_tokens,
                    "main_mean_logprob": route["mean_lp_orig"],
                    "safety_mean_logprob": route["mean_lp_safety"],
                    "post_insert_selected_attn_avg": _score_mean(safety["post_insert_scores"]),
                    "post_insert_selected_attn_count": len(safety["post_insert_scores"]),
                    "safety_selected_attn_avg": _score_mean(safety["safety_scores"]),
                    "safety_selected_attn_count": len(safety["safety_scores"]),
                    "matched_inserted_prefix_len": safety["matched_inserted_len"],
                    "first_post_insert_attention_driven": safety[
                        "first_post_insert_attention_driven"
                    ],
                    "anti_degen": {
                        "enabled": bool(params.get("anti_degen_enabled", False)),
                        "punct_like_count": route["punct_like_count"],
                        "safety_token_count": len(safety["segment_ids"]),
                        "punct_ratio": route["punct_ratio"],
                        "degenerate_safety": route["degenerate_safety"],
                    },
                }
            )
            stop_on_selected_eos = bool(
                safety["ended_early"] if choose_safety else main["ended_early"]
            )
            if stop_on_selected_eos:
                break
            continue

        raw_output_ids = scan_out["raw_output_ids"]
        if not raw_output_ids:
            break
        eos_id = int(params.get("eos_token_id", -1))
        if eos_id >= 0 and any(int(x) == eos_id for x in scan_out["accepted_ids"]):
            break

    generated_ids = tokens_main[initial_token_count:]
    text = _decode(tokenizer, generated_ids, skip_special_tokens=True)
    if trace_enabled:
        _trace_write(
            params,
            "controller_finish",
            {
                "generated_token_count": int(len(generated_ids)),
                "split_count": int(split_count),
                "selected_path": selected_path,
                "selected_path_history": selected_path_history,
                "split_mode_history": split_mode_history,
            },
        )
    _debug_block(
        params,
        "Final",
        "Controller Final Output",
        [
            ("selected_path", selected_path),
            ("splits", int(split_count)),
            ("selected_path_history", selected_path_history),
            ("generated_tokens", int(len(generated_ids))),
        ],
        body="Final response:\n" + _debug_snippet(params, text),
    )
    meta_info = {
        "beam_safety_controller": True,
        "beam_safety_selected_path": selected_path,
        "beam_safety_split_count": split_count,
        "beam_safety_split_selected_path_history_json": json.dumps(selected_path_history),
        "beam_safety_split_mode_history_json": json.dumps(split_mode_history),
        "beam_safety_insertion_attention_trace_json": json.dumps(insertion_attention_trace),
    }
    return {
        "text": text,
        "output_ids": generated_ids,
        "meta_info": meta_info,
    }
