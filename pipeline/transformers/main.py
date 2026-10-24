import json
import os
import sys
from datetime import datetime

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    __package__ = "pipeline.transformers"

from . import config
from .dataset_runner import run_attack_dataset
from .logger import close_log_file, log_print, set_log_file
from .modeling import load_model_and_tokenizer


def _is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _dump_config_snapshot(run_output_dir):
    """Persist current config.py parameters for reproducibility."""
    snapshot = {}
    for key, value in vars(config).items():
        if key.startswith("_"):
            continue
        if callable(value):
            continue
        if _is_json_serializable(value):
            snapshot[key] = value
        else:
            snapshot[key] = str(value)

    snapshot_path = os.path.join(run_output_dir, "config_snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return snapshot_path


def _parse_args(argv):
    datasets = list(config.DEFAULT_DATASETS)
    verbose = config.DEFAULT_VERBOSE
    max_samples = config.DEFAULT_MAX_SAMPLES
    include_original_response = config.DEFAULT_INCLUDE_ORIGINAL_RESPONSE

    if len(argv) > 1:
        datasets = [argv[1]]
    if len(argv) > 2:
        verbose = argv[2].lower() == "true"
    if len(argv) > 3:
        max_samples = int(argv[3])
    if len(argv) > 4:
        include_original_response = argv[4].lower() == "true"

    return datasets, verbose, max_samples, include_original_response


def _dataset_output_label(dataset_name):
    stem = os.path.splitext(dataset_name)[0]
    return stem.replace("\\", "_").replace("/", "_")


def run_cli(base_dir, argv=None):
    if argv is None:
        argv = sys.argv

    datasets, verbose, max_samples, include_original_response = _parse_args(argv)

    output_root_dir = config.results_dir(base_dir)
    os.makedirs(output_root_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_root_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    log_file_path = os.path.join(run_output_dir, "beam_safety_test_log.txt")
    config_snapshot_path = _dump_config_snapshot(run_output_dir)
    log_file = open(log_file_path, "w", encoding="utf-8")
    set_log_file(log_file)

    tokenizer, model = load_model_and_tokenizer()
    token_sets = config.build_token_sets(tokenizer)

    runtime = {
        "tokenizer": tokenizer,
        "model": model,
        "device": config.DEVICE,
        "token_sets": token_sets,
    }

    try:
        log_print("=" * 80)
        log_print(f"[redacted]: {run_output_dir}")
        log_print(f"[redacted]: {config_snapshot_path}")
        log_print(f"[redacted]: {log_file_path}")
        log_print("=" * 80)

        all_results = {}

        for dataset_name in datasets:
            dataset_path = config.dataset_path(base_dir, dataset_name)
            log_print(f"\n{'=' * 80}")
            log_print(f"[redacted]: {dataset_name}")
            log_print(f"[redacted]: {dataset_path}")

            if not os.path.exists(dataset_path):
                log_print(f"[redacted]: [redacted]: {dataset_path}[redacted][redacted]")
                continue

            stats = run_attack_dataset(
                dataset_path,
                runtime=runtime,
                verbose=verbose,
                max_samples=max_samples,
                include_original_response=include_original_response,
            )

            dataset_basename = _dataset_output_label(dataset_name)
            output_file = os.path.join(
                run_output_dir, f"beam_safety_test_results_{dataset_basename}.json"
            )
            attention_traces = stats.pop("attention_traces", None)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            log_print(f"\n[redacted]: {output_file}")
            if attention_traces is not None:
                attn_file = os.path.join(
                    run_output_dir,
                    f"beam_safety_test_results_{dataset_basename}_attention.json",
                )
                with open(attn_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "dataset": dataset_name,
                            "total": len(attention_traces),
                            "attention_traces": attention_traces,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                log_print(f"Attention [redacted]: {attn_file}")

            all_results[dataset_name] = stats

        if len(datasets) > 1 and len(all_results) > 1:
            summary_file = os.path.join(
                run_output_dir, "beam_safety_test_results_all.json"
            )
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            log_print(f"\n[redacted]: {summary_file}")
    finally:
        log_print(f"\n[redacted]: {log_file_path}")
        close_log_file()


def main():
    pipeline_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(pipeline_dir, "..", ".."))
    run_cli(base_dir, sys.argv)


if __name__ == "__main__":
    main()
