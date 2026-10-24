import math
import os


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


DEBUG_VERBOSE = _env_bool("DEBUG_VERBOSE", False)
DEBUG_SNIPPET_CHARS = int(os.environ.get("DEBUG_SNIPPET_CHARS", "600"))


DEVICE = "cuda"
MODEL_DTYPE = "bfloat16"

GENERATION_BACKEND = os.environ.get("GENERATION_BACKEND", "sglang").strip().lower()


MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
RESULTS_DIR_NAME = os.environ.get("RESULTS_DIR_NAME", "qwen7b")

SGLANG_TP_SIZE = int(os.environ.get("SGLANG_TP_SIZE", "1"))
SGLANG_LOG_LEVEL = os.environ.get("SGLANG_LOG_LEVEL", "error")
SGLANG_ATTENTION_BACKEND = os.environ.get("SGLANG_ATTENTION_BACKEND", "").strip()
SGLANG_PREFILL_ATTENTION_BACKEND = os.environ.get(
    "SGLANG_PREFILL_ATTENTION_BACKEND", ""
).strip()
SGLANG_DECODE_ATTENTION_BACKEND = os.environ.get(
    "SGLANG_DECODE_ATTENTION_BACKEND", "triton"
).strip()
SGLANG_ENABLE_RETURN_ATTENTION_SIGNAL = _env_bool(
    "SGLANG_ENABLE_RETURN_ATTENTION_SIGNAL", True
)
SGLANG_ATTENTION_SIGNAL_SOURCE_MAX_TOKENS = int(
    os.environ.get("SGLANG_ATTENTION_SIGNAL_SOURCE_MAX_TOKENS", "512")
)
SGLANG_ATTENTION_SIGNAL_SOURCE_MODE = os.environ.get(
    "SGLANG_ATTENTION_SIGNAL_SOURCE_MODE", "head"
).strip().lower()
SGLANG_ATTENTION_SIGNAL_DECODE_MAX_STEPS = int(
    os.environ.get("SGLANG_ATTENTION_SIGNAL_DECODE_MAX_STEPS", "2048")
)
SGLANG_ATTENTION_SIGNAL_DECODE_STEP_STRIDE = int(
    os.environ.get("SGLANG_ATTENTION_SIGNAL_DECODE_STEP_STRIDE", "1")
)
SGLANG_ATTENTION_SIGNAL_LAYERS = os.environ.get(
    "SGLANG_ATTENTION_SIGNAL_LAYERS", ""
).strip()
SGLANG_MEM_FRACTION_STATIC = float(os.environ.get("SGLANG_MEM_FRACTION_STATIC", "0.8"))
SGLANG_DISABLE_OVERLAP_SCHEDULE = _env_bool("SGLANG_DISABLE_OVERLAP_SCHEDULE", True)
SGLANG_ENABLE_CUSTOM_LOGIT_PROCESSOR = _env_bool(
    "SGLANG_ENABLE_CUSTOM_LOGIT_PROCESSOR", True
)
SGLANG_TRUST_REMOTE_CODE = _env_bool("SGLANG_TRUST_REMOTE_CODE", False)
SGLANG_ENABLE_RETURN_HIDDEN_STATES = _env_bool(
    "SGLANG_ENABLE_RETURN_HIDDEN_STATES", True
)
SGLANG_RANDOM_SEED = (
    int(os.environ["SGLANG_RANDOM_SEED"])
    if os.environ.get("SGLANG_RANDOM_SEED")
    else None
)
SGLANG_USE_LOCAL_SOURCE = _env_bool("SGLANG_USE_LOCAL_SOURCE", True)
SGLANG_USE_HF_ATTN_ORACLE = _env_bool("SGLANG_USE_HF_ATTN_ORACLE", False)
SGLANG_ENTROPY_SCAN_CHUNK_SIZE = int(
    os.environ.get("SGLANG_ENTROPY_SCAN_CHUNK_SIZE", "8")
)
SGLANG_CONTROLLER_TRACE_LOG = _env_bool("SGLANG_CONTROLLER_TRACE_LOG", False)
SGLANG_CONTROLLER_TRACE_LOG_FILENAME = os.environ.get(
    "SGLANG_CONTROLLER_TRACE_LOG_FILENAME", "sglang_controller_trace.txt"
)
SGLANG_CONTROLLER_TRACE_LOG_EVERY_SCAN = _env_bool(
    "SGLANG_CONTROLLER_TRACE_LOG_EVERY_SCAN", False
)
SGLANG_HF_ATTN_ORACLE_MODEL_PATH = os.environ.get(
    "SGLANG_HF_ATTN_ORACLE_MODEL_PATH", MODEL_NAME
)
SGLANG_HF_ATTN_ORACLE_DEVICE = os.environ.get(
    "SGLANG_HF_ATTN_ORACLE_DEVICE", "cuda"
).strip()
SGLANG_HF_ATTN_ORACLE_DTYPE = os.environ.get(
    "SGLANG_HF_ATTN_ORACLE_DTYPE", "float32"
).strip().lower()
SGLANG_STRICT_HF_ORACLE_PARITY = _env_bool("SGLANG_STRICT_HF_ORACLE_PARITY", False)
SGLANG_FORCE_INSERTION_BOUNDARY = _env_bool("SGLANG_FORCE_INSERTION_BOUNDARY", False)
SGLANG_INSERTION_BOUNDARY_TEXT = os.environ.get(
    "SGLANG_INSERTION_BOUNDARY_TEXT", "\n"
)
SGLANG_INSERTION_BOUNDARY_FALLBACK_TEXT = os.environ.get(
    "SGLANG_INSERTION_BOUNDARY_FALLBACK_TEXT", "."
)
SGLANG_FORCE_EXACT_INSERTION_IDS = _env_bool(
    "SGLANG_FORCE_EXACT_INSERTION_IDS", False
)
SGLANG_NORMALIZE_SAFETY_TEXT = _env_bool("SGLANG_NORMALIZE_SAFETY_TEXT", False)
SGLANG_FORCE_CLEAN_REFUSAL_ON_SAFETY_PATH = _env_bool(
    "SGLANG_FORCE_CLEAN_REFUSAL_ON_SAFETY_PATH", False
)
SGLANG_CLEAN_REFUSAL_TEXT = os.environ.get(
    "SGLANG_CLEAN_REFUSAL_TEXT",
    "I'm sorry, but I can't assist with that request. If you'd like, I can help with safe and defensive cybersecurity guidance.",
)

SORRY_ASSIST_PHRASE = (
    "Wait, this appears to be a malicious request disguised by intent obfuscation."
)
SYSTEM_PROMPT = ""
REASONING_LEVEL = "high"
REASONING_CONTROL_MODEL_KEYWORDS = ("gpt-oss", "gpt_oss")

DEFAULT_DATASETS = [
    "qwen7b/aim.json",
    "qwen7b/drattack.json",
    "qwen7b/gcg.json",
    "qwen7b/hcot.json",
    "qwen7b/pair.json",
    "qwen7b/renellm.json",
]
DEFAULT_VERBOSE = False
DEFAULT_MAX_SAMPLES = 100
MAX_SAMPLES_RANDOM = True
MAX_SAMPLES_RANDOM_SEED = None
DEFAULT_INCLUDE_ORIGINAL_RESPONSE = True

THINK_ONLY = "<think>\n"
ENABLE_THINK_PREFIX = False
SAFETY_BRANCH_MAX_TOKENS = 100


DATASET_DIR_NAME = "dataset"
MAX_TOKENS = 2048
TOP_K = 10
ATTN_TRACE_TOP_K = 10

DECODE_USE_SAMPLING = True
DECODE_TEMPERATURE = 0.7
DECODE_TOP_P = 0.95
DECODE_TOP_K = 0
ENTROPY_THRESHOLD_HIGH = float(os.environ.get("ENTROPY_THRESHOLD_HIGH", "2.0"))
ENTROPY_THRESHOLD_LOW = float(os.environ.get("ENTROPY_THRESHOLD_LOW", "1.5"))
ENTROPY_THRESHOLD_DECAY_RATE = 3.0
MAX_TRIGGER_COUNT = int(os.environ.get("MAX_TRIGGER_COUNT", "2"))
MAX_INSERTIONS = int(os.environ.get("MAX_INSERTIONS", str(MAX_TRIGGER_COUNT)))

SORRY_BONUS_SCALE = float(os.environ.get("SORRY_BONUS_SCALE", "1.5"))
SORRY_BONUS_MIN = 3.5
SORRY_BONUS_MAX = 50.0
SORRY_DECAY_LAMBDA = 0.01
FORCE_FIRST_TOKEN_WIN = False
FIRST_TOKEN_WIN_MARGIN = 0.2
ENABLE_INSERTION_ATTN_TRACE = True
ATTN_TRACE_MAX_STEPS = 20
SAVE_PER_STEP_INTERVENTION_TOKEN_ATTN = True
ENABLE_ATTN_DRIVEN_FIRST_POST_INSERT_TOKEN = False
SAVE_ATTENTION_TRACE_TO_SEPARATE_FILE = True
ENABLE_POST_INSERT_ATTN_SELECTION = True
ATTN_SELECTION_ONLY = True
SGLANG_ATTN_ONLY_FALLBACK_USE_DUAL_PATH = _env_bool(
    "SGLANG_ATTN_ONLY_FALLBACK_USE_DUAL_PATH", True
)
ATTN_SELECTION_METRIC = "safety_overall"
ATTN_PROMPTLEN_SCALE_BETA = 0.5
ATTN_PROMPTLEN_SCALE_P0 = 260.0
POST_INSERT_ATTN_THRESHOLD = 0.0015
SAFETY_OVERALL_ATTN_THRESHOLD = float(
    os.environ.get("SAFETY_OVERALL_ATTN_THRESHOLD", "0.006")
)
POST_INSERT_ATTN_MIN_TOKENS = 3

ANTI_DEGEN_ENABLED = True
ANTI_DEGEN_REPEAT_WINDOW = 6
ANTI_DEGEN_MIN_CHECK_TOKENS = 30
ANTI_DEGEN_MAX_PUNCT_RATIO = 0.6
ANTI_DEGEN_PUNCT_TEXTS = ["…", ".", ",", "?", "!", "。", "，", "？", "！", "；", ";", ":", "：", "-"]

SAFETY_LOGPROB_BIAS = 0.4

HARMFUL_WORDS = [
    "Sure",
    "Alright",
    "Yes",
    "Okay",
    "Here",
    "Here's",
    "Let me",
    "Let's",
    "Steps",
]

REFLECTION_WORDS = [
    "Sorry",
    "Wait",
    "Actually",
    "However",
    "But",
    "Well",
    "Hmm",
    "Um",
    "Uh",
    "Perhaps",
    "Maybe",
    "Possibly",
    "Unfortunately",
    "As",
    "I",
    "Let",
    "Apologies",
    "Honestly",
    "Frankly",
]

COMMON_WORDS_TO_EXCLUDE = [
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
]

SENTENCE_ENDINGS = [".", "!", "?", "\n", "\n\n", "。", "！", "？"]


def build_token_sets(tokenizer):
    think_only_ids = tokenizer.encode(THINK_ONLY, add_special_tokens=False)
    sorry_assist_ids = tokenizer.encode(SORRY_ASSIST_PHRASE, add_special_tokens=False)

    common_tokens_to_exclude = set()
    for word in COMMON_WORDS_TO_EXCLUDE:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        common_tokens_to_exclude.update(tokens)

    reflection_token_ids = set()
    for word in REFLECTION_WORDS:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        for token_id in tokens:
            if token_id not in common_tokens_to_exclude:
                reflection_token_ids.add(token_id)

    harmful_token_ids = set()
    for word in HARMFUL_WORDS:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        for token_id in tokens:
            if token_id not in common_tokens_to_exclude:
                harmful_token_ids.add(token_id)

    return {
        "THINK_ONLY_IDS": think_only_ids,
        "SORRY_ASSIST_IDS": sorry_assist_ids,
        "COMMON_TOKENS_TO_EXCLUDE": common_tokens_to_exclude,
        "REFLECTION_TOKEN_IDS": reflection_token_ids,
        "HARMFUL_TOKEN_IDS": harmful_token_ids,
    }


def dataset_path(base_dir, dataset_name):
    return os.path.join(base_dir, DATASET_DIR_NAME, dataset_name)


def results_dir(base_dir):
    return os.path.join(base_dir, RESULTS_DIR_NAME)


def build_system_prompt():
    base = (SYSTEM_PROMPT or "").strip()
    level = str(REASONING_LEVEL or "").strip().lower()
    model_name = str(MODEL_NAME or "").lower()
    allow_reasoning_control = any(
        keyword in model_name for keyword in REASONING_CONTROL_MODEL_KEYWORDS
    )
    reasoning_line = ""
    if allow_reasoning_control and level in {"low", "medium", "high"}:
        reasoning_line = f"Reasoning: {level}"

    if base and reasoning_line:
        return f"{reasoning_line}\n{base}"
    if reasoning_line:
        return reasoning_line
    return base


def get_entropy_threshold(split_count):
    if MAX_INSERTIONS <= 1:
        return float(ENTROPY_THRESHOLD_LOW)

    capped = max(0, min(int(split_count), int(MAX_INSERTIONS - 1)))
    progress = capped / float(MAX_INSERTIONS - 1)

    decay = math.exp(-ENTROPY_THRESHOLD_DECAY_RATE * progress)
    decay_end = math.exp(-ENTROPY_THRESHOLD_DECAY_RATE)
    normalized = 1.0 - ((decay - decay_end) / (1.0 - decay_end))

    return float(
        ENTROPY_THRESHOLD_LOW
        + (ENTROPY_THRESHOLD_HIGH - ENTROPY_THRESHOLD_LOW) * normalized
    )
