import json
import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import dill
import orjson
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    @classmethod
    def to_str(cls) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)()


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        disallowed_token_ids = custom_param_list[0]["token_ids"]
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue

            thinking_budget: int | None = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue
            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            if (
                self.THINKING_START_TOKEN_ID not in cur_ids
                or self.THINKING_END_TOKEN_ID in cur_ids
            ):
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""

    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Qwen3 models."""

    THINKING_START_TOKEN_ID: int = 151667
    THINKING_END_TOKEN_ID: int = 151668
    NEW_LINE_TOKEN_ID: int = 198


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201


class BeamEntropyTelemetryLogitProcessor(CustomLogitProcessor):
    """Expose pre-sampling entropy for one-step SGLang parity orchestration."""

    @staticmethod
    def _append(req: "Req", key: str, value: Any) -> None:
        if getattr(req, "customized_info", None) is None:
            req.customized_info = {}
        req.customized_info.setdefault(key, []).append(value)

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue
            req: Req = params.get("__req__")
            if req is None:
                continue

            row = logits[batch_idx].float()
            probs = torch.softmax(row, dim=-1)
            finite_probs = torch.where(
                probs > 1e-10, probs, torch.ones_like(probs)
            )
            entropy = float((-(probs * torch.log(finite_probs))).sum().item())
            self._append(req, "beam_parity_entropy", entropy)

            top_k = max(0, int(params.get("telemetry_top_k", 0)))
            if top_k > 0:
                vals, ids = torch.topk(probs, min(top_k, int(probs.shape[-1])))
                self._append(
                    req,
                    "beam_parity_top_ids_json",
                    json.dumps([int(x) for x in ids.detach().cpu().tolist()]),
                )
                self._append(
                    req,
                    "beam_parity_top_probs_json",
                    json.dumps([float(x) for x in vals.detach().cpu().tolist()]),
                )
        return logits


class BeamEntropyStopLogitProcessor(CustomLogitProcessor):
    """Scan decode logits and return an entropy-trigger finish signal."""

    @staticmethod
    def _append(req: "Req", key: str, value: Any) -> None:
        if getattr(req, "customized_info", None) is None:
            req.customized_info = {}
        req.customized_info.setdefault(key, []).append(value)

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue
            req: Req = params.get("__req__")
            if req is None:
                continue

            row = logits[batch_idx].float()
            probs = torch.softmax(row, dim=-1)
            finite_probs = torch.where(
                probs > 1e-10, probs, torch.ones_like(probs)
            )
            entropy = float((-(probs * torch.log(finite_probs))).sum().item())
            threshold = float(params.get("entropy_threshold", float("inf")))
            triggered = bool(entropy > threshold)

            self._append(req, "beam_entropy_scan_entropy", entropy)
            self._append(req, "beam_entropy_scan_threshold", threshold)
            self._append(req, "beam_entropy_scan_triggered", triggered)

            if triggered:
                from sglang.srt.managers.schedule_batch import (
                    FINISH_ENTROPY_TRIGGER,
                )

                req.to_finish = FINISH_ENTROPY_TRIGGER(
                    entropy=entropy,
                    threshold=threshold,
                )

        return logits


class BeamForceTokenIdsLogitProcessor(CustomLogitProcessor):
    """Force a request-local token plan before normal SGLang sampling."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue
            req: Req = params.get("__req__")
            if req is None:
                continue
            try:
                forced_ids = [int(x) for x in (params.get("forced_token_ids") or [])]
            except (TypeError, ValueError):
                forced_ids = []
            step = len(getattr(req, "output_ids", []) or [])
            if step >= len(forced_ids):
                continue
            target_id = int(forced_ids[step])
            if 0 <= target_id < logits.shape[-1]:
                logits[batch_idx].fill_(-float("inf"))
                logits[batch_idx, target_id] = 0.0
        return logits


class BeamSafetyBranchLogitProcessor(CustomLogitProcessor):
    """Emit the safety insertion prefix for a parity branch generation."""

    EPS: float = 1e-10
    STATE_KEY: str = "_beam_safety_branch_state"

    @staticmethod
    def _append(req: "Req", key: str, value: Any) -> None:
        if getattr(req, "customized_info", None) is None:
            req.customized_info = {}
        req.customized_info.setdefault(key, []).append(value)

    @staticmethod
    def _matched_prefix(output_ids: List[int], prefix_ids: List[int]) -> int:
        matched = 0
        for actual_id, expected_id in zip(output_ids, prefix_ids):
            if int(actual_id) != int(expected_id):
                break
            matched += 1
        return matched

    @classmethod
    def _first_bonus(
        cls,
        row: torch.Tensor,
        target_id: int,
        params: Dict[str, Any],
    ) -> float:
        probs = torch.softmax(row, dim=-1)
        top1_prob = float(torch.max(probs).item())
        target_prob = float(probs[target_id].item())
        gap_log = math.log(top1_prob + cls.EPS) - math.log(target_prob + cls.EPS)
        first_bonus = max(
            float(params.get("sorry_bonus_min", 3.5)),
            min(
                float(params.get("sorry_bonus_max", 50.0)),
                float(params.get("sorry_bonus_scale", 1.5)) * max(0.0, gap_log),
            ),
        )
        if bool(params.get("force_first_token_win", False)):
            others = row.clone()
            others[target_id] = -float("inf")
            required_bonus = (
                float(torch.max(others).item())
                - float(row[target_id].item())
                + float(params.get("first_token_win_margin", 0.2))
            )
            first_bonus = max(first_bonus, required_bonus)
        return float(first_bonus)

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue
            req: Req = params.get("__req__")
            if req is None:
                continue
            try:
                insertion_ids = [
                    int(x) for x in (params.get("safety_phrase_token_ids") or [])
                ]
            except (TypeError, ValueError):
                insertion_ids = []
            if not insertion_ids:
                continue

            output_ids = [int(x) for x in (getattr(req, "output_ids", []) or [])]
            step = len(output_ids)
            matched = self._matched_prefix(output_ids, insertion_ids)
            prompt_len = len(getattr(req, "origin_input_ids", []) or [])
            if matched > 0:
                req.attention_signal_source_positions = [
                    int(prompt_len + idx) for idx in range(matched)
                ]
            elif not bool(getattr(req, "attention_signal_source_positions", None)):
                req.attention_signal_source_positions = [max(0, int(prompt_len) - 1)]

            state = getattr(req, self.STATE_KEY, None)
            if not isinstance(state, dict):
                state = {"first_bonus": None}
                setattr(req, self.STATE_KEY, state)

            row = logits[batch_idx].float()
            if step < len(insertion_ids):
                target_id = int(insertion_ids[step])
                if 0 <= target_id < row.shape[-1]:
                    if bool(params.get("force_exact_insertion_ids", False)):
                        row.fill_(-float("inf"))
                        row[target_id] = 0.0
                    else:
                        if step == 0 or state.get("first_bonus") is None:
                            state["first_bonus"] = self._first_bonus(
                                row, target_id, params
                            )
                        bonus = float(state["first_bonus"]) * math.exp(
                            -float(params.get("sorry_decay_lambda", 0.01)) * step
                        )
                        row[target_id] = row[target_id] + bonus

            punct_ids_raw = params.get("anti_degen_punct_token_ids") or []
            try:
                punct_ids = {int(x) for x in punct_ids_raw}
            except (TypeError, ValueError):
                punct_ids = set()
            repeat_window = max(1, int(params.get("anti_degen_repeat_window", 6)))
            if (
                bool(params.get("anti_degen_enabled", False))
                and output_ids
                and output_ids[-1] in punct_ids
            ):
                repeated = 0
                for token_id in reversed(output_ids):
                    if token_id != output_ids[-1]:
                        break
                    repeated += 1
                if repeated >= repeat_window and 0 <= output_ids[-1] < row.shape[-1]:
                    row[output_ids[-1]] = -float("inf")

            logits[batch_idx] = row.to(logits.dtype)
            self._append(req, "beam_safety_branch_inserted_prefix_len", int(matched))
        return logits


class BeamSafetyLogitProcessor(CustomLogitProcessor):
    """Entropy-triggered safety controller for decode-time safety control.

    The processor exposes kernel-side split/selection telemetry through
    `req.customized_info` so serving clients can consume internal safety-path
    decisions from meta_info instead of brittle string matching.
    """

    EPS: float = 1e-10
    STATE_KEY: str = "_beam_safety_state"
    _HF_ORACLE_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any, torch.device]] = {}
    _HF_ORACLE_FAIL_CACHE: Dict[Tuple[str, str, str], str] = {}

    @staticmethod
    def _resolve_hf_dtype(dtype_name: str, device: str) -> torch.dtype:
        name = str(dtype_name or "float32").strip().lower()
        if name == "auto":
            if device.startswith("cuda") and torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        if name == "bfloat16":
            if device.startswith("cuda") and torch.cuda.is_available():
                return torch.bfloat16
            return torch.float32
        if name == "float16":
            if device.startswith("cuda") and torch.cuda.is_available():
                return torch.float16
            return torch.float32
        if name == "float32":
            return torch.float32
        return torch.float32

    @classmethod
    def _get_hf_oracle(
        cls,
        *,
        model_path: str,
        device: str,
        dtype_name: str,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[torch.device], Optional[str]]:
        key = (str(model_path), str(device), str(dtype_name))
        cached = cls._HF_ORACLE_CACHE.get(key)
        if cached is not None:
            model, tokenizer, torch_device = cached
            return model, tokenizer, torch_device, None

        if key in cls._HF_ORACLE_FAIL_CACHE:
            return None, None, None, cls._HF_ORACLE_FAIL_CACHE[key]

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            err = f"import_error:{exc}"
            cls._HF_ORACLE_FAIL_CACHE[key] = err
            return None, None, None, err

        try:
            torch_device = torch.device(device)
        except Exception:
            torch_device = torch.device("cpu")

        if torch_device.type == "cuda" and (not torch.cuda.is_available()):
            torch_device = torch.device("cpu")

        model_dtype = cls._resolve_hf_dtype(dtype_name, str(torch_device))

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=model_dtype,
                attn_implementation="eager",
                trust_remote_code=False,
            )
            model = model.to(torch_device)
            model.eval()
            cls._HF_ORACLE_CACHE[key] = (model, tokenizer, torch_device)
            return model, tokenizer, torch_device, None
        except Exception as exc:
            err = f"load_error:{exc}"
            cls._HF_ORACLE_FAIL_CACHE[key] = err
            return None, None, None, err

    @classmethod
    def _sample_next_token_from_logits(
        cls,
        logits_1d: torch.Tensor,
        *,
        use_sampling: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        # Fallback anchor for all error/degenerate paths.
        greedy_id = int(torch.argmax(logits_1d).item())

        if (not use_sampling) or temperature <= 0:
            return greedy_id

        work_logits = logits_1d.float()
        finite_mask = torch.isfinite(work_logits)
        if not bool(torch.any(finite_mask).item()):
            return greedy_id
        work_logits = torch.where(
            finite_mask, work_logits, torch.full_like(work_logits, -1e9)
        )
        work_logits = work_logits / float(temperature)
        max_val = torch.max(work_logits)
        if not bool(torch.isfinite(max_val).item()):
            return greedy_id
        work_logits = work_logits - max_val
        probs = torch.softmax(work_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)

        if top_k > 0 and top_k < probs.shape[-1]:
            topk_vals, topk_idx = torch.topk(probs, int(top_k))
            filtered = torch.zeros_like(probs)
            filtered[topk_idx] = topk_vals
            probs = filtered

        if 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumulative <= float(top_p)
            if keep_mask.numel() > 0:
                keep_mask[0] = True
            filtered_sorted = sorted_probs * keep_mask
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx] = filtered_sorted
            probs = filtered

        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)
        denom = probs.sum()
        if (not bool(torch.isfinite(denom).item())) or float(denom.item()) <= 0.0:
            return greedy_id

        probs = probs / denom
        try:
            sampled = torch.multinomial(probs, num_samples=1)
            return int(sampled.item())
        except Exception:
            return greedy_id

    @staticmethod
    def _is_punct_like_token(
        tokenizer: Any,
        token_id: int,
        punct_token_ids: Set[int],
    ) -> bool:
        if int(token_id) in punct_token_ids:
            return True
        if tokenizer is None:
            return False
        token_text = str(
            tokenizer.decode([int(token_id)], skip_special_tokens=False)
        ).strip()
        if token_text == "":
            return False
        return all(ch in ".,?!;:，。！？；：…-—" for ch in token_text)

    @staticmethod
    def _match_inserted_prefix_len(
        tokens_safety: torch.Tensor,
        parallel_start_len: int,
        inserted_token_ids: List[int],
    ) -> int:
        if not inserted_token_ids:
            return 0
        safety_segment = tokens_safety[parallel_start_len:]
        safety_segment_list = safety_segment.tolist()
        max_len = min(len(safety_segment_list), len(inserted_token_ids))
        matched = 0
        for i in range(max_len):
            if int(safety_segment_list[i]) != int(inserted_token_ids[i]):
                break
            matched += 1
        return matched

    @classmethod
    def _candidate_attn_to_inserted_span(
        cls,
        model: Any,
        tokens_safety: torch.Tensor,
        candidate_id: int,
        inserted_span_start: int,
        inserted_span_end: int,
        device: torch.device,
    ) -> float:
        if inserted_span_end <= inserted_span_start:
            return 0.0
        candidate_tensor = torch.tensor(
            [candidate_id], device=device, dtype=tokens_safety.dtype
        )
        candidate_tokens = torch.cat([tokens_safety, candidate_tensor], dim=0)
        with torch.no_grad():
            out = model(candidate_tokens.unsqueeze(0), output_attentions=True)
        attentions = getattr(out, "attentions", None)
        if not attentions:
            return 0.0
        last_layer_attn = attentions[-1][0]  # [heads, q_len, k_len]
        query_idx = last_layer_attn.shape[1] - 1
        span_attn = last_layer_attn[:, query_idx, inserted_span_start:inserted_span_end]
        if span_attn.numel() == 0:
            return 0.0
        span_attn = torch.nan_to_num(span_attn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return float(span_attn.mean().item())

    @classmethod
    def _select_attention_driven_next_token(
        cls,
        *,
        model: Any,
        tokens_safety: torch.Tensor,
        probs_b: torch.Tensor,
        parallel_start_len: int,
        inserted_token_ids: List[int],
        device: torch.device,
        attn_trace_top_k: int,
    ) -> Optional[int]:
        inserted_prefix_len = cls._match_inserted_prefix_len(
            tokens_safety=tokens_safety,
            parallel_start_len=parallel_start_len,
            inserted_token_ids=inserted_token_ids,
        )
        if inserted_prefix_len <= 0:
            return None
        inserted_span_start = parallel_start_len
        inserted_span_end = parallel_start_len + inserted_prefix_len
        top_k = min(max(1, int(attn_trace_top_k)), int(probs_b.shape[0]))
        topk_vals_b, topk_idx_b = torch.topk(probs_b, top_k)
        del topk_vals_b
        best_candidate = None
        best_attn = None
        for rank in range(top_k):
            candidate_id = int(topk_idx_b[rank].item())
            attn_score = cls._candidate_attn_to_inserted_span(
                model=model,
                tokens_safety=tokens_safety,
                candidate_id=candidate_id,
                inserted_span_start=inserted_span_start,
                inserted_span_end=inserted_span_end,
                device=device,
            )
            if best_attn is None or attn_score > best_attn:
                best_attn = attn_score
                best_candidate = candidate_id
        return best_candidate

    @classmethod
    def _hf_oracle_attn_snapshot(
        cls,
        *,
        req: "Req",
        trigger_count: int,
        insertion_token_ids: List[int],
        force_exact_insertion_ids: bool,
        force_first_split_safety: bool,
        attn_selection_metric: str,
        post_insert_attn_threshold: float,
        safety_overall_attn_threshold: float,
        post_insert_attn_min_tokens: int,
        attn_promptlen_scale_beta: float,
        attn_promptlen_scale_p0: float,
        prompt_token_len: int,
        anti_degen_enabled: bool,
        anti_degen_repeat_window: int,
        anti_degen_min_check_tokens: int,
        anti_degen_max_punct_ratio: float,
        punct_token_ids: Set[int],
        bonus_scale: float,
        bonus_min: float,
        bonus_max: float,
        bonus_decay: float,
        force_first_win: bool,
        first_win_margin: float,
        use_sampling: bool,
        decode_temperature: float,
        decode_top_p: float,
        decode_top_k: int,
        enable_attn_driven_first_post_insert_token: bool,
        attn_trace_top_k: int,
        strict_hf_oracle_parity: bool,
        safety_branch_max_tokens: int,
        eos_token_id: int,
        model_path: str,
        model_device: str,
        model_dtype: str,
    ) -> Dict[str, Any]:
        model, oracle_tokenizer, oracle_device, oracle_err = cls._get_hf_oracle(
            model_path=model_path,
            device=model_device,
            dtype_name=model_dtype,
        )
        if model is None or oracle_device is None:
            return {
                "ok": False,
                "error": oracle_err or "oracle_unavailable",
                "value_scaled": 0.0,
                "value_raw": 0.0,
                "threshold": float(
                    safety_overall_attn_threshold
                    if str(attn_selection_metric) == "safety_overall"
                    else post_insert_attn_threshold
                ),
                "count": 0,
                "ready": False,
                "main_score": float("-inf"),
                "safety_score": float("-inf"),
                "choose_safety": False,
                "decision_mode": "hf_oracle_unavailable",
            }

        tokens_main_ids = [int(x) for x in (req.origin_input_ids + req.output_ids)]
        if not tokens_main_ids:
            return {
                "ok": False,
                "error": "empty_tokens",
                "value_scaled": 0.0,
                "value_raw": 0.0,
                "threshold": float(post_insert_attn_threshold),
                "count": 0,
                "ready": False,
                "main_score": float("-inf"),
                "safety_score": float("-inf"),
                "choose_safety": False,
                "decision_mode": "hf_oracle_empty",
            }

        tokens_orig = torch.tensor(tokens_main_ids, device=oracle_device, dtype=torch.long)
        tokens_safety = tokens_orig.clone()
        parallel_start_len = int(tokens_safety.shape[0])

        log_probs_orig: List[float] = []
        log_probs_safety: List[float] = []
        first_bonus: Optional[float] = None
        orig_ended_early = False
        safety_ended_early = False
        safety_selected_attn_scores: List[float] = []
        post_insert_selected_attn_scores: List[float] = []
        safety_punct_like_count = 0
        safety_same_token_run = 0
        safety_prev_token_id: Optional[int] = None
        first_post_insert_token_decoded_by_attn = False

        n_steps = max(1, int(safety_branch_max_tokens))
        with torch.no_grad():
            for step in range(n_steps):
                out_a = model(tokens_orig.unsqueeze(0))
                logits_a = out_a.logits[:, -1, :].clone().float()[0]
                probs_a = torch.softmax(logits_a, dim=-1)
                next_id_a = cls._sample_next_token_from_logits(
                    logits_a,
                    use_sampling=use_sampling,
                    temperature=decode_temperature,
                    top_p=decode_top_p,
                    top_k=decode_top_k,
                )
                lp_a = math.log(float(probs_a[next_id_a].item()) + cls.EPS)
                log_probs_orig.append(lp_a)
                tokens_orig = torch.cat(
                    [tokens_orig, torch.tensor([next_id_a], device=oracle_device)]
                )
                if eos_token_id >= 0 and next_id_a == eos_token_id:
                    orig_ended_early = True
                    break

                out_b = model(tokens_safety.unsqueeze(0))
                logits_b = out_b.logits[:, -1, :].clone().float()[0]
                if step < len(insertion_token_ids):
                    target_id = int(insertion_token_ids[step])
                    if 0 <= target_id < logits_b.shape[0]:
                        if force_exact_insertion_ids:
                            logits_b.fill_(-float("inf"))
                            logits_b[target_id] = 0.0
                        else:
                            if step == 0:
                                first_bonus = cls._compute_first_token_bonus(
                                    row=logits_b,
                                    target_id=target_id,
                                    bonus_scale=bonus_scale,
                                    bonus_min=bonus_min,
                                    bonus_max=bonus_max,
                                    force_first_win=force_first_win,
                                    first_win_margin=first_win_margin,
                                )
                                bonus = float(first_bonus)
                            else:
                                base_bonus = float(
                                    first_bonus if first_bonus is not None else bonus_min
                                )
                                bonus = base_bonus * math.exp(-bonus_decay * step)
                            logits_b[target_id] = logits_b[target_id] + float(bonus)

                probs_b = torch.softmax(logits_b, dim=-1)
                first_post_insert_step = step == len(insertion_token_ids)
                if (
                    first_post_insert_step
                    and bool(enable_attn_driven_first_post_insert_token)
                    and (not first_post_insert_token_decoded_by_attn)
                ):
                    attn_choice = cls._select_attention_driven_next_token(
                        model=model,
                        tokens_safety=tokens_safety,
                        probs_b=probs_b,
                        parallel_start_len=parallel_start_len,
                        inserted_token_ids=insertion_token_ids,
                        device=oracle_device,
                        attn_trace_top_k=attn_trace_top_k,
                    )
                    if attn_choice is not None:
                        next_id_b = int(attn_choice)
                        first_post_insert_token_decoded_by_attn = True
                    else:
                        next_id_b = cls._sample_next_token_from_logits(
                            logits_b,
                            use_sampling=use_sampling,
                            temperature=decode_temperature,
                            top_p=decode_top_p,
                            top_k=decode_top_k,
                        )
                else:
                    next_id_b = cls._sample_next_token_from_logits(
                        logits_b,
                        use_sampling=use_sampling,
                        temperature=decode_temperature,
                        top_p=decode_top_p,
                        top_k=decode_top_k,
                    )
                    if (
                        anti_degen_enabled
                        and safety_prev_token_id is not None
                        and safety_same_token_run >= int(anti_degen_repeat_window)
                        and cls._is_punct_like_token(
                            oracle_tokenizer,
                            int(safety_prev_token_id),
                            punct_token_ids,
                        )
                    ):
                        adjusted_logits = logits_b.clone()
                        adjusted_logits[int(safety_prev_token_id)] = -float("inf")
                        next_id_b = cls._sample_next_token_from_logits(
                            adjusted_logits,
                            use_sampling=use_sampling,
                            temperature=decode_temperature,
                            top_p=decode_top_p,
                            top_k=decode_top_k,
                        )

                inserted_prefix_len = cls._match_inserted_prefix_len(
                    tokens_safety=tokens_safety,
                    parallel_start_len=parallel_start_len,
                    inserted_token_ids=insertion_token_ids,
                )
                if inserted_prefix_len > 0:
                    attn_score = cls._candidate_attn_to_inserted_span(
                        model=model,
                        tokens_safety=tokens_safety,
                        candidate_id=next_id_b,
                        inserted_span_start=parallel_start_len,
                        inserted_span_end=parallel_start_len + inserted_prefix_len,
                        device=oracle_device,
                    )
                    safety_selected_attn_scores.append(float(attn_score))
                    if step > len(insertion_token_ids):
                        post_insert_selected_attn_scores.append(float(attn_score))

                lp_b = math.log(float(probs_b[next_id_b].item()) + cls.EPS)
                log_probs_safety.append(lp_b)
                if cls._is_punct_like_token(
                    oracle_tokenizer, int(next_id_b), punct_token_ids
                ):
                    safety_punct_like_count += 1
                if safety_prev_token_id is None or int(next_id_b) != int(safety_prev_token_id):
                    safety_same_token_run = 1
                else:
                    safety_same_token_run += 1
                safety_prev_token_id = int(next_id_b)
                tokens_safety = torch.cat(
                    [tokens_safety, torch.tensor([next_id_b], device=oracle_device)]
                )
                if eos_token_id >= 0 and next_id_b == eos_token_id:
                    safety_ended_early = True
                    break

        skip = len(insertion_token_ids)

        def mean_lp_after_boost(lp_list: List[float]) -> float:
            if not lp_list:
                return -1e9
            after = lp_list[skip:]
            if not after:
                return float(sum(lp_list) / float(len(lp_list)))
            return float(sum(after) / float(len(after)))

        mean_lp_orig = mean_lp_after_boost(log_probs_orig)
        mean_lp_safety = mean_lp_after_boost(log_probs_safety)
        finite_post_insert_scores = [
            float(x)
            for x in post_insert_selected_attn_scores
            if isinstance(x, (int, float)) and math.isfinite(float(x))
        ]
        finite_safety_scores = [
            float(x)
            for x in safety_selected_attn_scores
            if isinstance(x, (int, float)) and math.isfinite(float(x))
        ]
        avg_post_insert_attn = (
            float(sum(finite_post_insert_scores) / float(len(finite_post_insert_scores)))
            if finite_post_insert_scores
            else 0.0
        )
        avg_safety_selected_attn = (
            float(sum(finite_safety_scores) / float(len(finite_safety_scores)))
            if finite_safety_scores
            else 0.0
        )

        metric = str(attn_selection_metric or "post_insert")
        if metric not in {"post_insert", "safety_overall"}:
            metric = "post_insert"
        if metric == "safety_overall":
            metric_raw = avg_safety_selected_attn
            metric_threshold = float(safety_overall_attn_threshold)
            metric_count = len(finite_safety_scores)
        else:
            metric_raw = avg_post_insert_attn
            metric_threshold = float(post_insert_attn_threshold)
            metric_count = len(finite_post_insert_scores)

        scale = cls._promptlen_attn_scale(
            prompt_token_len=prompt_token_len,
            beta=float(attn_promptlen_scale_beta),
            p0=float(attn_promptlen_scale_p0),
        )
        metric_scaled = float(metric_raw * scale)
        ready = bool(metric_count >= max(1, int(post_insert_attn_min_tokens)))

        decision_mode = "attn_only_hf_oracle"
        if orig_ended_early and not safety_ended_early:
            choose_safety = False
            decision_mode = "hf_oracle_orig_eos_first"
        elif safety_ended_early and not orig_ended_early:
            choose_safety = True
            decision_mode = "hf_oracle_safety_eos_first"
        else:
            if (not strict_hf_oracle_parity) and force_first_split_safety and int(trigger_count) == 0:
                choose_safety = True
                decision_mode = "force_first_split_safety"
            elif ready:
                choose_safety = metric_scaled >= metric_threshold
                decision_mode = "attn_only_hf_oracle"
            else:
                choose_safety = False
                decision_mode = "attn_only_insufficient_signal"

        safety_token_count = len(log_probs_safety)
        punct_ratio = (
            float(safety_punct_like_count) / float(safety_token_count)
            if safety_token_count > 0
            else 0.0
        )
        if (
            anti_degen_enabled
            and safety_token_count >= int(anti_degen_min_check_tokens)
            and punct_ratio >= float(anti_degen_max_punct_ratio)
            and (not orig_ended_early)
        ):
            choose_safety = False
            decision_mode = "anti_degen_gate"

        orig_segment_ids = [
            int(x) for x in tokens_orig[parallel_start_len:].detach().cpu().tolist()
        ]
        safety_segment_ids = [
            int(x) for x in tokens_safety[parallel_start_len:].detach().cpu().tolist()
        ]
        chosen_segment_ids = safety_segment_ids if choose_safety else orig_segment_ids

        return {
            "ok": True,
            "error": None,
            "value_scaled": float(metric_scaled),
            "value_raw": float(metric_raw),
            "threshold": float(metric_threshold),
            "count": int(metric_count),
            "ready": bool(ready),
            "main_score": float(mean_lp_orig),
            "safety_score": float(mean_lp_safety),
            "choose_safety": bool(choose_safety),
            "decision_mode": str(decision_mode),
            "orig_segment_ids": orig_segment_ids,
            "safety_segment_ids": safety_segment_ids,
            "chosen_segment_ids": chosen_segment_ids,
        }

    @staticmethod
    def _init_state() -> dict[str, Any]:
        return {
            "trigger_count": 0,
            "active": False,
            "safety_step": 0,
            "first_bonus": None,
            "last_output_len": 0,
            "prev_token_id": None,
            "same_token_run": 0,
            "split_count": 0,
            "selected_path_history": [],
            "split_mode_history": [],
            "split_entropy_history": [],
            "split_threshold_history": [],
            "split_main_top1_logprob_history": [],
            "split_safety_top1_logprob_history": [],
            "split_safety_logprob_bias_history": [],
            "split_attn_selection_value_history": [],
            "split_attn_selection_value_raw_history": [],
            "split_attn_selection_threshold_history": [],
            "split_attn_selection_count_history": [],
            "split_attn_selection_ready_history": [],
            "last_entropy": -1.0,
            "last_threshold": -1.0,
            "last_selection_mode": "none",
            "last_selected_path": "main",
            "attn_hidden_cursor": 0,
            "attn_signal_cursor": 0,
            "active_split_collecting": False,
            "active_split_inserted_len": 0,
            "active_split_inserted_seen": 0,
            "active_split_after_insert_seen": 0,
            "active_split_inserted_vectors": [],
            "active_split_inserted_abs_positions": [],
            "active_split_safety_attn_proxy_scores": [],
            "active_split_post_insert_attn_proxy_scores": [],
            "forced_plan_token_ids": [],
            "forced_plan_cursor": 0,
            "forced_plan_path": "main",
        }

    @staticmethod
    def _entropy_threshold(
        trigger_count: int,
        max_trigger_count: int,
        low: float,
        high: float,
        decay_rate: float,
    ) -> float:
        if max_trigger_count <= 1:
            return float(low)

        capped = max(0, min(int(trigger_count), int(max_trigger_count - 1)))
        progress = capped / float(max_trigger_count - 1)
        decay = math.exp(-decay_rate * progress)
        decay_end = math.exp(-decay_rate)
        denom = 1.0 - decay_end
        normalized = progress if abs(denom) < 1e-12 else 1.0 - ((decay - decay_end) / denom)
        return float(low + (high - low) * normalized)

    @staticmethod
    def _update_recent_output_state(state: dict[str, Any], req: "Req") -> None:
        output_ids = getattr(req, "output_ids", []) or []
        prev_len = int(state.get("last_output_len", 0))
        if prev_len < 0 or prev_len > len(output_ids):
            prev_len = 0
        for tok in output_ids[prev_len:]:
            tok_id = int(tok)
            prev_tok = state.get("prev_token_id")
            if prev_tok is None or tok_id != int(prev_tok):
                state["same_token_run"] = 1
            else:
                state["same_token_run"] = int(state.get("same_token_run", 0)) + 1
            state["prev_token_id"] = tok_id
        state["last_output_len"] = len(output_ids)

    @staticmethod
    def _append_customized(req: "Req", key: str, value: Any) -> None:
        if getattr(req, "customized_info", None) is None:
            req.customized_info = {}
        if key not in req.customized_info:
            req.customized_info[key] = []
        req.customized_info[key].append(value)

    @classmethod
    def _compute_first_token_bonus(
        cls,
        row: torch.Tensor,
        target_id: int,
        bonus_scale: float,
        bonus_min: float,
        bonus_max: float,
        force_first_win: bool,
        first_win_margin: float,
    ) -> float:
        probs = torch.softmax(row, dim=-1)
        top1_prob = float(torch.max(probs).item())
        target_prob = float(probs[target_id].item())
        gap_log = math.log(top1_prob + cls.EPS) - math.log(target_prob + cls.EPS)
        first_bonus = max(
            bonus_min,
            min(bonus_max, bonus_scale * max(0.0, gap_log)),
        )
        if force_first_win:
            target_logit = float(row[target_id].item())
            others = row.clone()
            others[target_id] = -float("inf")
            max_other_logit = float(torch.max(others).item())
            required_bonus = max_other_logit - target_logit + first_win_margin
            first_bonus = max(first_bonus, required_bonus)
        return float(first_bonus)

    @staticmethod
    def _top1_logprob(row: torch.Tensor) -> float:
        log_probs = torch.log_softmax(row, dim=-1)
        top_idx = int(torch.argmax(log_probs).item())
        return float(log_probs[top_idx].item())

    @staticmethod
    def _vector_from_obj(obj: Any) -> Optional[List[float]]:
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().tolist()
        if not isinstance(obj, list) or not obj:
            return None
        if isinstance(obj[0], list):
            return None
        out = []
        for x in obj:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                return None
        return out if out else None

    @classmethod
    def _extract_decode_hidden_vectors(cls, req: "Req") -> List[List[float]]:
        hs = getattr(req, "hidden_states", None)
        if not isinstance(hs, list):
            return []
        out: List[List[float]] = []
        for item in hs:
            vec = cls._vector_from_obj(item)
            if vec is not None:
                out.append(vec)
        return out

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> Optional[float]:
        if not a or not b or len(a) != len(b):
            return None
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return None
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))

    @staticmethod
    def _promptlen_attn_scale(prompt_token_len: int, beta: float, p0: float) -> float:
        p = max(0.0, float(prompt_token_len))
        if beta <= 0.0 or p0 <= 0.0:
            return 1.0
        return 1.0 + beta * math.log(1.0 + p / p0)

    @staticmethod
    def _reset_active_split_tracker(state: dict[str, Any], inserted_len: int, enable: bool) -> None:
        state["active_split_collecting"] = bool(enable)
        state["active_split_inserted_len"] = max(0, int(inserted_len))
        state["active_split_inserted_seen"] = 0
        state["active_split_after_insert_seen"] = 0
        state["active_split_inserted_vectors"] = []
        state["active_split_inserted_abs_positions"] = []
        state["active_split_safety_attn_proxy_scores"] = []
        state["active_split_post_insert_attn_proxy_scores"] = []

    @staticmethod
    def _sync_attention_signal_source_positions(state: dict[str, Any], req: "Req") -> None:
        raw_positions = state.get("active_split_inserted_abs_positions", [])
        if not isinstance(raw_positions, list) or len(raw_positions) == 0:
            # Keep previous request-level source positions as bootstrap while the
            # inserted span is still being discovered.
            if bool(getattr(req, "attention_signal_source_positions", None)):
                return
            req.attention_signal_source_positions = []
            return
        out: List[int] = []
        seen = set()
        for x in raw_positions:
            try:
                pos = int(x)
            except (TypeError, ValueError):
                continue
            if pos < 0 or pos in seen:
                continue
            seen.add(pos)
            out.append(pos)
        req.attention_signal_source_positions = out

    @classmethod
    def _update_attn_proxy_from_hidden(
        cls,
        state: dict[str, Any],
        req: "Req",
        insertion_token_ids: List[int],
    ) -> None:
        output_ids = getattr(req, "output_ids", []) or []
        decode_vectors = cls._extract_decode_hidden_vectors(req)
        usable = min(len(output_ids), len(decode_vectors))
        cursor = int(state.get("attn_hidden_cursor", 0))
        if cursor < 0 or cursor > usable:
            cursor = 0

        collecting = bool(state.get("active_split_collecting", False))
        inserted_len = int(state.get("active_split_inserted_len", len(insertion_token_ids)))
        if inserted_len <= 0:
            inserted_len = len(insertion_token_ids)
        inserted_seen = int(state.get("active_split_inserted_seen", 0))
        after_insert_seen = int(state.get("active_split_after_insert_seen", 0))
        inserted_vectors = state.get("active_split_inserted_vectors", [])
        inserted_abs_positions = state.get("active_split_inserted_abs_positions", [])
        safety_scores = state.get("active_split_safety_attn_proxy_scores", [])
        post_insert_scores = state.get("active_split_post_insert_attn_proxy_scores", [])
        prompt_len = len(getattr(req, "origin_input_ids", []) or [])

        for i in range(cursor, usable):
            vec = decode_vectors[i]
            if not collecting:
                continue

            if inserted_seen < inserted_len:
                inserted_vectors.append(vec)
                inserted_abs_positions.append(int(prompt_len + i))
                inserted_seen += 1
                continue

            if not inserted_vectors:
                continue

            sim_total = 0.0
            sim_count = 0
            for anchor in inserted_vectors:
                sim = cls._cosine_similarity(vec, anchor)
                if sim is None:
                    continue
                sim_total += sim
                sim_count += 1
            if sim_count <= 0:
                continue

            score = float(sim_total / float(sim_count))
            safety_scores.append(score)
            after_insert_seen += 1
            # Match HF post-insert metric semantics: exclude the first token right
            # after insertion and average from the second one onward.
            if after_insert_seen > 1:
                post_insert_scores.append(score)

        state["attn_hidden_cursor"] = usable
        state["active_split_inserted_seen"] = inserted_seen
        state["active_split_after_insert_seen"] = after_insert_seen
        state["active_split_inserted_vectors"] = inserted_vectors
        state["active_split_inserted_abs_positions"] = inserted_abs_positions
        state["active_split_safety_attn_proxy_scores"] = safety_scores
        state["active_split_post_insert_attn_proxy_scores"] = post_insert_scores

    @classmethod
    def _update_attn_proxy_from_signal(
        cls,
        state: dict[str, Any],
        req: "Req",
        insertion_token_ids: List[int],
    ) -> bool:
        customized = getattr(req, "customized_info", None)
        if not isinstance(customized, dict):
            return False
        signal_trace = customized.get("attention_signal")
        if not isinstance(signal_trace, list):
            return False

        output_ids = getattr(req, "output_ids", []) or []
        usable = min(len(output_ids), len(signal_trace))
        cursor = int(state.get("attn_signal_cursor", 0))
        if cursor < 0 or cursor > usable:
            cursor = 0
        if usable <= cursor:
            state["attn_signal_cursor"] = usable
            return True

        collecting = bool(state.get("active_split_collecting", False))
        inserted_len = int(state.get("active_split_inserted_len", len(insertion_token_ids)))
        if inserted_len <= 0:
            inserted_len = len(insertion_token_ids)
        inserted_seen = int(state.get("active_split_inserted_seen", 0))
        after_insert_seen = int(state.get("active_split_after_insert_seen", 0))
        inserted_abs_positions = state.get("active_split_inserted_abs_positions", [])
        safety_scores = state.get("active_split_safety_attn_proxy_scores", [])
        post_insert_scores = state.get("active_split_post_insert_attn_proxy_scores", [])
        prompt_len = len(getattr(req, "origin_input_ids", []) or [])

        for i in range(cursor, usable):
            item = signal_trace[i]
            if not collecting:
                continue
            if inserted_seen < inserted_len:
                inserted_abs_positions.append(int(prompt_len + i))
                inserted_seen += 1
                continue
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            safety_scores.append(score)
            after_insert_seen += 1
            # Match HF post-insert metric semantics: exclude the first token
            # after insertion and average from the second one onward.
            if after_insert_seen > 1:
                post_insert_scores.append(score)

        state["attn_signal_cursor"] = usable
        state["active_split_inserted_seen"] = inserted_seen
        state["active_split_after_insert_seen"] = after_insert_seen
        state["active_split_inserted_abs_positions"] = inserted_abs_positions
        state["active_split_safety_attn_proxy_scores"] = safety_scores
        state["active_split_post_insert_attn_proxy_scores"] = post_insert_scores
        return True

    @classmethod
    def _get_attn_selection_snapshot(
        cls,
        state: dict[str, Any],
        *,
        metric_name: str,
        post_insert_threshold: float,
        safety_overall_threshold: float,
        min_tokens: int,
        prompt_token_len: int,
        promptlen_scale_beta: float,
        promptlen_scale_p0: float,
    ) -> Dict[str, Any]:
        metric = str(metric_name or "post_insert")
        if metric not in {"post_insert", "safety_overall"}:
            metric = "post_insert"

        if metric == "safety_overall":
            values = state.get("active_split_safety_attn_proxy_scores", [])
            threshold = float(safety_overall_threshold)
        else:
            values = state.get("active_split_post_insert_attn_proxy_scores", [])
            threshold = float(post_insert_threshold)

        count = len(values) if isinstance(values, list) else 0
        value_raw = float(sum(values) / float(count)) if count > 0 else 0.0
        scale = cls._promptlen_attn_scale(
            prompt_token_len=prompt_token_len,
            beta=float(promptlen_scale_beta),
            p0=float(promptlen_scale_p0),
        )
        value_scaled = float(value_raw * scale)
        ready = bool(count >= max(1, int(min_tokens)))
        return {
            "metric": metric,
            "value_raw": value_raw,
            "value_scaled": value_scaled,
            "threshold": threshold,
            "count": int(count),
            "ready": ready,
        }

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req: Req = params.get("__req__")
            if req is None:
                continue

            safety_ids_raw = params.get("safety_phrase_token_ids") or []
            try:
                safety_phrase_token_ids = [int(x) for x in safety_ids_raw]
            except (TypeError, ValueError):
                safety_phrase_token_ids = []
            if not safety_phrase_token_ids:
                continue
            boundary_ids_raw = params.get("insertion_boundary_token_ids") or []
            try:
                insertion_boundary_token_ids = [int(x) for x in boundary_ids_raw]
            except (TypeError, ValueError):
                insertion_boundary_token_ids = []
            force_insertion_boundary = bool(params.get("force_insertion_boundary", False))
            force_exact_insertion_ids = bool(params.get("force_exact_insertion_ids", False))
            insertion_token_ids = (
                [*insertion_boundary_token_ids, *safety_phrase_token_ids]
                if force_insertion_boundary and insertion_boundary_token_ids
                else list(safety_phrase_token_ids)
            )
            if not insertion_token_ids:
                continue

            max_trigger_count = int(params.get("max_trigger_count", 2))
            entropy_low = float(params.get("entropy_threshold_low", 1.2))
            entropy_high = float(params.get("entropy_threshold_high", 2.0))
            entropy_decay = float(params.get("entropy_threshold_decay_rate", 3.0))
            bonus_scale = float(params.get("sorry_bonus_scale", 1.5))
            bonus_min = float(params.get("sorry_bonus_min", 3.5))
            bonus_max = float(params.get("sorry_bonus_max", 50.0))
            bonus_decay = float(params.get("sorry_decay_lambda", 0.01))
            force_first_win = bool(params.get("force_first_token_win", False))
            first_win_margin = float(params.get("first_token_win_margin", 0.2))
            dual_path_mode = bool(params.get("beam_safety_dual_path_mode", True))
            safety_logprob_bias = float(params.get("safety_logprob_bias", 0.0))
            force_first_split_safety = bool(params.get("force_first_split_safety", False))
            selection_mode_label = str(
                params.get("beam_safety_selection_mode_label", "fluency")
            )
            if not selection_mode_label:
                selection_mode_label = "fluency"
            enable_post_insert_attn_selection = bool(
                params.get("enable_post_insert_attn_selection", False)
            )
            attn_selection_only = bool(params.get("attn_selection_only", False))
            attn_only_fallback_use_dual_path = bool(
                params.get("attn_only_fallback_use_dual_path", True)
            )
            attn_selection_metric = str(params.get("attn_selection_metric", "post_insert"))
            post_insert_attn_threshold = float(params.get("post_insert_attn_threshold", 0.0))
            safety_overall_attn_threshold = float(
                params.get("safety_overall_attn_threshold", 0.0)
            )
            post_insert_attn_min_tokens = int(params.get("post_insert_attn_min_tokens", 1))
            attn_promptlen_scale_beta = float(params.get("attn_promptlen_scale_beta", 0.0))
            attn_promptlen_scale_p0 = float(params.get("attn_promptlen_scale_p0", 1.0))
            prompt_token_len = int(params.get("prompt_token_len", 0))
            anti_degen_enabled = bool(params.get("anti_degen_enabled", False))
            anti_degen_repeat_window = int(params.get("anti_degen_repeat_window", 6))
            anti_degen_min_check_tokens = int(params.get("anti_degen_min_check_tokens", 30))
            anti_degen_max_punct_ratio = float(params.get("anti_degen_max_punct_ratio", 0.6))
            punct_ids_raw = params.get("anti_degen_punct_token_ids") or []
            try:
                punct_token_ids = {int(x) for x in punct_ids_raw}
            except (TypeError, ValueError):
                punct_token_ids = set()
            eos_token_id = int(params.get("eos_token_id", -1))
            safety_branch_max_tokens = int(params.get("safety_branch_max_tokens", 100))
            decode_use_sampling = bool(params.get("decode_use_sampling", False))
            decode_temperature = float(params.get("decode_temperature", 1.0))
            decode_top_p = float(params.get("decode_top_p", 1.0))
            decode_top_k = int(params.get("decode_top_k", 0))
            enable_attn_driven_first_post_insert_token = bool(
                params.get("enable_attn_driven_first_post_insert_token", False)
            )
            attn_trace_top_k = int(params.get("attn_trace_top_k", 10))
            hf_attn_oracle_enabled = bool(params.get("hf_attn_oracle_enabled", False))
            hf_attn_oracle_model_path = str(params.get("hf_attn_oracle_model_path", "") or "").strip()
            hf_attn_oracle_device = str(params.get("hf_attn_oracle_device", "cpu") or "cpu").strip()
            hf_attn_oracle_dtype = str(params.get("hf_attn_oracle_dtype", "float32") or "float32").strip().lower()
            strict_hf_oracle_parity = bool(params.get("strict_hf_oracle_parity", False))

            state = getattr(req, self.STATE_KEY, None)
            if not isinstance(state, dict):
                state = self._init_state()
                setattr(req, self.STATE_KEY, state)

            self._update_recent_output_state(state, req)
            if not strict_hf_oracle_parity:
                has_signal_trace = self._update_attn_proxy_from_signal(
                    state, req, insertion_token_ids
                )
                if not has_signal_trace:
                    self._update_attn_proxy_from_hidden(state, req, insertion_token_ids)

            row = logits[batch_idx].float()
            selected_path_this_step = "safety" if bool(state.get("active", False)) else "main"
            selection_mode_this_step = "carry"
            force_plan_applied = False

            # Strict parity mode: replay the oracle-selected branch segment token-by-token.
            if strict_hf_oracle_parity:
                raw_plan = state.get("forced_plan_token_ids", [])
                try:
                    forced_plan_ids = [int(x) for x in raw_plan] if isinstance(raw_plan, list) else []
                except (TypeError, ValueError):
                    forced_plan_ids = []
                forced_cursor = int(state.get("forced_plan_cursor", 0))
                if forced_cursor < 0:
                    forced_cursor = 0
                if forced_cursor < len(forced_plan_ids):
                    forced_token_id = int(forced_plan_ids[forced_cursor])
                    if 0 <= forced_token_id < row.shape[0]:
                        row.fill_(-float("inf"))
                        row[forced_token_id] = 0.0
                        forced_cursor += 1
                        state["forced_plan_cursor"] = forced_cursor
                        selected_path_this_step = str(state.get("forced_plan_path", "main") or "main")
                        selection_mode_this_step = "hf_oracle_replay"
                        state["active"] = False
                        state["safety_step"] = 0
                        state["first_bonus"] = None
                        force_plan_applied = True
                        if forced_cursor >= len(forced_plan_ids):
                            state["forced_plan_token_ids"] = []
                            state["forced_plan_cursor"] = 0
                            state["forced_plan_path"] = "main"
                    else:
                        state["forced_plan_token_ids"] = []
                        state["forced_plan_cursor"] = 0
                        state["forced_plan_path"] = "main"

            # Trigger insertion based on current entropy and per-request trigger count.
            if (not force_plan_applied) and (not state.get("active", False)):
                probs = torch.softmax(row, dim=-1)
                entropy = float(
                    -(probs * torch.log(torch.clamp(probs, min=self.EPS))).sum().item()
                )
                threshold = self._entropy_threshold(
                    trigger_count=int(state.get("trigger_count", 0)),
                    max_trigger_count=max_trigger_count,
                    low=entropy_low,
                    high=entropy_high,
                    decay_rate=entropy_decay,
                )
                state["last_entropy"] = float(entropy)
                state["last_threshold"] = float(threshold)
                if entropy > threshold and int(state.get("trigger_count", 0)) < max_trigger_count:
                    choose_safety = True
                    decision_mode = "trigger_default"
                    first_bonus = None
                    main_score = self._top1_logprob(row)
                    safety_score = float("-inf")
                    target_id = int(insertion_token_ids[0]) if insertion_token_ids else -1
                    if 0 <= target_id < row.shape[0]:
                        first_bonus = self._compute_first_token_bonus(
                            row=row,
                            target_id=target_id,
                            bonus_scale=bonus_scale,
                            bonus_min=bonus_min,
                            bonus_max=bonus_max,
                            force_first_win=force_first_win,
                            first_win_margin=first_win_margin,
                        )
                    if dual_path_mode and (0 <= target_id < row.shape[0]) and first_bonus is not None:
                        safety_row = row.clone()
                        safety_row[target_id] = safety_row[target_id] + float(first_bonus)
                        safety_score = self._top1_logprob(safety_row)
                        choose_safety = (safety_score + safety_logprob_bias) >= main_score
                        decision_mode = selection_mode_label
                    elif dual_path_mode:
                        choose_safety = False
                        decision_mode = "dual_path_invalid_target"

                    attn_snapshot = self._get_attn_selection_snapshot(
                        state,
                        metric_name=attn_selection_metric,
                        post_insert_threshold=post_insert_attn_threshold,
                        safety_overall_threshold=safety_overall_attn_threshold,
                        min_tokens=post_insert_attn_min_tokens,
                        prompt_token_len=prompt_token_len,
                        promptlen_scale_beta=attn_promptlen_scale_beta,
                        promptlen_scale_p0=attn_promptlen_scale_p0,
                    )

                    oracle_applied = False
                    oracle_plan_ids: List[int] = []
                    if (
                        enable_post_insert_attn_selection
                        and attn_selection_only
                        and hf_attn_oracle_enabled
                        and hf_attn_oracle_model_path
                    ):
                        oracle_snapshot = self._hf_oracle_attn_snapshot(
                            req=req,
                            trigger_count=int(state.get("trigger_count", 0)),
                            insertion_token_ids=insertion_token_ids,
                            force_exact_insertion_ids=force_exact_insertion_ids,
                            force_first_split_safety=force_first_split_safety,
                            attn_selection_metric=attn_selection_metric,
                            post_insert_attn_threshold=post_insert_attn_threshold,
                            safety_overall_attn_threshold=safety_overall_attn_threshold,
                            post_insert_attn_min_tokens=post_insert_attn_min_tokens,
                            attn_promptlen_scale_beta=attn_promptlen_scale_beta,
                            attn_promptlen_scale_p0=attn_promptlen_scale_p0,
                            prompt_token_len=prompt_token_len,
                            anti_degen_enabled=anti_degen_enabled,
                            anti_degen_repeat_window=anti_degen_repeat_window,
                            anti_degen_min_check_tokens=anti_degen_min_check_tokens,
                            anti_degen_max_punct_ratio=anti_degen_max_punct_ratio,
                            punct_token_ids=punct_token_ids,
                            bonus_scale=bonus_scale,
                            bonus_min=bonus_min,
                            bonus_max=bonus_max,
                            bonus_decay=bonus_decay,
                            force_first_win=force_first_win,
                            first_win_margin=first_win_margin,
                            use_sampling=decode_use_sampling,
                            decode_temperature=decode_temperature,
                            decode_top_p=decode_top_p,
                            decode_top_k=decode_top_k,
                            enable_attn_driven_first_post_insert_token=(
                                enable_attn_driven_first_post_insert_token
                            ),
                            attn_trace_top_k=attn_trace_top_k,
                            strict_hf_oracle_parity=strict_hf_oracle_parity,
                            safety_branch_max_tokens=safety_branch_max_tokens,
                            eos_token_id=eos_token_id,
                            model_path=hf_attn_oracle_model_path,
                            model_device=hf_attn_oracle_device,
                            model_dtype=hf_attn_oracle_dtype,
                        )
                        if bool(oracle_snapshot.get("ok", False)):
                            choose_safety = bool(oracle_snapshot["choose_safety"])
                            decision_mode = str(
                                oracle_snapshot.get("decision_mode", "attn_only_hf_oracle")
                            )
                            main_score = float(oracle_snapshot.get("main_score", main_score))
                            safety_score = float(
                                oracle_snapshot.get("safety_score", safety_score)
                            )
                            attn_snapshot = {
                                "value_scaled": float(
                                    oracle_snapshot.get("value_scaled", 0.0)
                                ),
                                "value_raw": float(oracle_snapshot.get("value_raw", 0.0)),
                                "threshold": float(
                                    oracle_snapshot.get("threshold", post_insert_attn_threshold)
                                ),
                                "count": int(oracle_snapshot.get("count", 0)),
                                "ready": bool(oracle_snapshot.get("ready", False)),
                            }
                            raw_plan = oracle_snapshot.get("chosen_segment_ids", [])
                            if isinstance(raw_plan, list):
                                try:
                                    oracle_plan_ids = [int(x) for x in raw_plan]
                                except (TypeError, ValueError):
                                    oracle_plan_ids = []
                            oracle_applied = True
                        else:
                            decision_mode = "hf_oracle_failed_fallback"
                            self._append_customized(
                                req,
                                "beam_safety_hf_oracle_error",
                                str(oracle_snapshot.get("error", "unknown_oracle_error")),
                            )

                    if (not oracle_applied) and enable_post_insert_attn_selection and attn_selection_only:
                        if strict_hf_oracle_parity:
                            choose_safety = False
                            decision_mode = "hf_oracle_required_unavailable"
                        elif force_first_split_safety and int(state.get("trigger_count", 0)) == 0:
                            choose_safety = True
                            decision_mode = "force_first_split_safety"
                        elif attn_snapshot["ready"]:
                            choose_safety = (
                                float(attn_snapshot["value_scaled"])
                                >= float(attn_snapshot["threshold"])
                            )
                            decision_mode = "attn_only"
                        elif attn_only_fallback_use_dual_path and dual_path_mode:
                            choose_safety = (safety_score + safety_logprob_bias) >= main_score
                            decision_mode = "attn_only_fallback_dual_path"
                        else:
                            choose_safety = False
                            decision_mode = "attn_only_insufficient_signal"
                    elif (
                        (not strict_hf_oracle_parity)
                        and (not oracle_applied)
                        and force_first_split_safety
                        and int(state.get("trigger_count", 0)) == 0
                    ):
                        choose_safety = True
                        decision_mode = "force_first_split_safety"

                    state["trigger_count"] = int(state.get("trigger_count", 0)) + 1
                    state["split_count"] = int(state.get("split_count", 0)) + 1
                    replay_path = "safety" if choose_safety else "main"
                    if strict_hf_oracle_parity and oracle_applied and oracle_plan_ids:
                        state["active"] = False
                        state["safety_step"] = 0
                        state["first_bonus"] = None
                        self._reset_active_split_tracker(state, inserted_len=0, enable=False)
                        state["forced_plan_token_ids"] = list(oracle_plan_ids)
                        state["forced_plan_cursor"] = 0
                        state["forced_plan_path"] = replay_path
                        selected_path_this_step = replay_path

                        first_plan_token = int(oracle_plan_ids[0])
                        if 0 <= first_plan_token < row.shape[0]:
                            row.fill_(-float("inf"))
                            row[first_plan_token] = 0.0
                            state["forced_plan_cursor"] = 1
                            force_plan_applied = True
                            selection_mode_this_step = "hf_oracle_replay"
                            if state["forced_plan_cursor"] >= len(oracle_plan_ids):
                                state["forced_plan_token_ids"] = []
                                state["forced_plan_cursor"] = 0
                                state["forced_plan_path"] = "main"
                    else:
                        state["forced_plan_token_ids"] = []
                        state["forced_plan_cursor"] = 0
                        state["forced_plan_path"] = "main"
                        if choose_safety:
                            state["active"] = True
                            state["safety_step"] = 0
                            state["first_bonus"] = None
                            self._reset_active_split_tracker(
                                state, inserted_len=len(insertion_token_ids), enable=True
                            )
                            # Bootstrap one valid source position so attention-signal
                            # remains active until inserted-span positions are discovered.
                            prompt_len = len(getattr(req, "origin_input_ids", []) or [])
                            req.attention_signal_source_positions = [
                                max(0, int(prompt_len) - 1)
                            ]
                            selected_path_this_step = "safety"
                        else:
                            state["active"] = False
                            state["safety_step"] = 0
                            state["first_bonus"] = None
                            self._reset_active_split_tracker(state, inserted_len=0, enable=False)
                            selected_path_this_step = "main"
                    state["selected_path_history"].append(selected_path_this_step)
                    state["split_mode_history"].append(str(decision_mode))
                    state["split_entropy_history"].append(float(entropy))
                    state["split_threshold_history"].append(float(threshold))
                    state["split_main_top1_logprob_history"].append(float(main_score))
                    state["split_safety_top1_logprob_history"].append(float(safety_score))
                    state["split_safety_logprob_bias_history"].append(
                        float(safety_logprob_bias)
                    )
                    state["split_attn_selection_value_history"].append(
                        float(attn_snapshot["value_scaled"])
                    )
                    state["split_attn_selection_value_raw_history"].append(
                        float(attn_snapshot["value_raw"])
                    )
                    state["split_attn_selection_threshold_history"].append(
                        float(attn_snapshot["threshold"])
                    )
                    state["split_attn_selection_count_history"].append(
                        int(attn_snapshot["count"])
                    )
                    state["split_attn_selection_ready_history"].append(
                        bool(attn_snapshot["ready"])
                    )
                    state["last_selected_path"] = selected_path_this_step
                    selection_mode_this_step = decision_mode
                    # Split-level telemetry (one record per trigger decision).
                    self._append_customized(
                        req, "beam_safety_split_selected_path", selected_path_this_step
                    )
                    self._append_customized(
                        req, "beam_safety_split_mode", str(decision_mode)
                    )
                    self._append_customized(req, "beam_safety_split_entropy", float(entropy))
                    self._append_customized(
                        req, "beam_safety_split_entropy_threshold", float(threshold)
                    )
                    self._append_customized(
                        req, "beam_safety_split_main_top1_logprob", float(main_score)
                    )
                    self._append_customized(
                        req, "beam_safety_split_safety_top1_logprob", float(safety_score)
                    )
                    self._append_customized(
                        req, "beam_safety_split_safety_logprob_bias", float(safety_logprob_bias)
                    )
                    self._append_customized(
                        req,
                        "beam_safety_split_attn_selection_value",
                        float(attn_snapshot["value_scaled"]),
                    )
                    self._append_customized(
                        req,
                        "beam_safety_split_attn_selection_value_raw",
                        float(attn_snapshot["value_raw"]),
                    )
                    self._append_customized(
                        req,
                        "beam_safety_split_attn_selection_threshold",
                        float(attn_snapshot["threshold"]),
                    )
                    self._append_customized(
                        req,
                        "beam_safety_split_attn_selection_count",
                        int(attn_snapshot["count"]),
                    )
                    self._append_customized(
                        req,
                        "beam_safety_split_attn_selection_ready",
                        bool(attn_snapshot["ready"]),
                    )

            if (not force_plan_applied) and state.get("active", False):
                safety_step = int(state.get("safety_step", 0))
                if safety_step < len(insertion_token_ids):
                    target_id = int(insertion_token_ids[safety_step])
                    if 0 <= target_id < row.shape[0]:
                        if force_exact_insertion_ids:
                            row.fill_(-float("inf"))
                            row[target_id] = 0.0
                        else:
                            if safety_step == 0:
                                first_bonus = self._compute_first_token_bonus(
                                    row=row,
                                    target_id=target_id,
                                    bonus_scale=bonus_scale,
                                    bonus_min=bonus_min,
                                    bonus_max=bonus_max,
                                    force_first_win=force_first_win,
                                    first_win_margin=first_win_margin,
                                )
                                state["first_bonus"] = float(first_bonus)
                                bonus = float(first_bonus)
                            else:
                                base_bonus = float(state.get("first_bonus") or bonus_min)
                                bonus = base_bonus * math.exp(-bonus_decay * safety_step)

                            row[target_id] = row[target_id] + bonus

                    state["safety_step"] = safety_step + 1
                else:
                    state["active"] = False
                    state["safety_step"] = 0
                    state["first_bonus"] = None
                selected_path_this_step = "safety"
                if selection_mode_this_step == "carry":
                    selection_mode_this_step = "active_insertion"

            if (not force_plan_applied) and anti_degen_enabled and (not strict_hf_oracle_parity):
                prev_token_id = state.get("prev_token_id")
                same_run = int(state.get("same_token_run", 0))
                if (
                    prev_token_id is not None
                    and int(prev_token_id) in punct_token_ids
                    and same_run >= anti_degen_repeat_window
                    and 0 <= int(prev_token_id) < row.shape[0]
                ):
                    row[int(prev_token_id)] = -float("inf")
                    selection_mode_this_step = "anti_degen_block"

            # Keep attention-signal source positions aligned to the dynamically
            # discovered inserted span (for next decode step).
            self._sync_attention_signal_source_positions(state, req)

            logits[batch_idx] = row.to(logits.dtype)
            state["last_selected_path"] = selected_path_this_step
            state["last_selection_mode"] = selection_mode_this_step

            # Emit per-token kernel telemetry to meta_info via customized_info.
            self._append_customized(req, "beam_safety_selected_path", selected_path_this_step)
            self._append_customized(
                req, "beam_safety_selection_mode", str(selection_mode_this_step)
            )
            self._append_customized(req, "beam_safety_active", bool(state.get("active", False)))
            self._append_customized(req, "beam_safety_trigger_count", int(state.get("trigger_count", 0)))
            self._append_customized(req, "beam_safety_split_count", int(state.get("split_count", 0)))
            self._append_customized(req, "beam_safety_safety_step", int(state.get("safety_step", 0)))
            self._append_customized(req, "beam_safety_entropy", float(state.get("last_entropy", -1.0)))
            self._append_customized(
                req,
                "beam_safety_entropy_threshold",
                float(state.get("last_threshold", -1.0)),
            )
            # Emit split-history snapshots on every token so downstream slicing still
            # preserves full split decisions in the final chunk.
            self._append_customized(
                req,
                "beam_safety_split_selected_path_history_json",
                json.dumps(state.get("selected_path_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_mode_history_json",
                json.dumps(state.get("split_mode_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_entropy_history_json",
                json.dumps(state.get("split_entropy_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_entropy_threshold_history_json",
                json.dumps(state.get("split_threshold_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_main_top1_logprob_history_json",
                json.dumps(state.get("split_main_top1_logprob_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_safety_top1_logprob_history_json",
                json.dumps(state.get("split_safety_top1_logprob_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_safety_logprob_bias_history_json",
                json.dumps(state.get("split_safety_logprob_bias_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_attn_selection_value_history_json",
                json.dumps(state.get("split_attn_selection_value_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_attn_selection_value_raw_history_json",
                json.dumps(state.get("split_attn_selection_value_raw_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_attn_selection_threshold_history_json",
                json.dumps(state.get("split_attn_selection_threshold_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_attn_selection_count_history_json",
                json.dumps(state.get("split_attn_selection_count_history", [])),
            )
            self._append_customized(
                req,
                "beam_safety_split_attn_selection_ready_history_json",
                json.dumps(state.get("split_attn_selection_ready_history", [])),
            )

        return logits


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue

            if ngram_size <= 0 or window_size <= 0:
                continue

            sequence: List[int] = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = tuple(sequence[-(ngram_size - 1) :])
            else:
                current_prefix = tuple()

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned_tokens.add(ngram[-1])

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()

            banned_tokens.difference_update(whitelist)

            if not banned_tokens:
                continue

            indices = list(banned_tokens)
            logits[batch_idx, indices] = -float("inf")

        return logits
