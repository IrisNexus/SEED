from ..attack_runner import run_attack_dataset as _run_attack_dataset
from . import config
from .beam_core import run_beam_search, run_original_response
from .logger import log_print


def run_attack_dataset(dataset_path, runtime, verbose=False, max_samples=None,
                       include_original_response=False):
    return _run_attack_dataset(
        dataset_path, runtime, config=config, generate=run_beam_search,
        generate_original=run_original_response, log_print=log_print,
        verbose=verbose, max_samples=max_samples,
        include_original_response=include_original_response,
    )
