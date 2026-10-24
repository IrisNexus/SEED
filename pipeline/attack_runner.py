import json
import random


def run_attack_dataset(dataset_path, runtime, *, config, generate, generate_original,
                       log_print, verbose=False, max_samples=None,
                       include_original_response=False):
    with open(dataset_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Attack dataset must be a JSON list.")
    if max_samples is not None and int(max_samples) < 0:
        raise ValueError("max_samples must be non-negative.")
    if max_samples:
        sample_size = min(int(max_samples), len(data))
        if getattr(config, "MAX_SAMPLES_RANDOM", True):
            data = random.Random(getattr(config, "MAX_SAMPLES_RANDOM_SEED", None)).sample(data, sample_size)
        else:
            data = data[:sample_size]

    results = []
    attention_traces = []
    for idx, item in enumerate(data, 1):
        row = {"sample_index": idx}
        trace = None
        try:
            if not isinstance(item, dict) or "choices" in item:
                raise ValueError("Expected an attack sample with instruction or question.")
            instruction = item.get("instruction", item.get("question"))
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Attack sample has no non-empty instruction or question.")
            row["instruction"] = instruction
            for key in ("id", "question", "category", "attack", "source"):
                if key in item:
                    row[key] = item[key]
            log_print(f"[{idx}/{len(data)}] Generating attack response")
            if include_original_response:
                row["original_response"] = generate_original(instruction, runtime)
            beam_result = generate(instruction, runtime)
            response, selected_path, splits = beam_result[:3]
            history = beam_result[3] if len(beam_result) > 3 else ([selected_path] if splits else [])
            trace = beam_result[4] if len(beam_result) > 4 else {"enabled": False, "splits": []}
            row.update(response=response, modified_response=response, selected_path=selected_path,
                       selected_path_history=history, splits=splits)
            if verbose:
                log_print(f"  Modified Response: {response[:200]}")
            log_print(f"  path={selected_path}, splits={splits}")
        except Exception as exc:
            row.update(error=str(exc), selected_path="error")
            log_print(f"  ERROR: {exc}")
        if getattr(config, "SAVE_ATTENTION_TRACE_TO_SEPARATE_FILE", True):
            attention_traces.append(trace)
        else:
            row["insertion_attention_trace"] = trace
        results.append(row)

    successful = [row for row in results if "error" not in row]
    safety_paths = sum(row["selected_path"] == "safety" for row in successful)
    output = {
        "config": {"generation_model": config.MODEL_NAME, "generation_system_prompt": config.build_system_prompt(),
                   "dataset_path": str(dataset_path), "max_samples_limit": max_samples,
                   "include_original_response": bool(include_original_response)},
        "total": len(data), "successful": len(successful), "errors": len(results) - len(successful),
        "safety_path_count": safety_paths,
        "safety_path_rate": safety_paths / len(successful) if successful else None,
        "split_count": sum(row["splits"] for row in successful),
        "results": results,
    }
    if getattr(config, "SAVE_ATTENTION_TRACE_TO_SEPARATE_FILE", True):
        output["attention_traces"] = attention_traces
    return output
