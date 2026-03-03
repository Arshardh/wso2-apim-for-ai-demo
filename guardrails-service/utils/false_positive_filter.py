"""
RECAP Framework — Phase 3b: LLM Context-Based False Positive Filter
====================================================================
This module implements the false positive filtering step described in the
RECAP architecture. For "risky" entity detections (e.g. a 2-digit number
flagged as DATE_TIME), it passes the match and its surrounding sentence
context to a local SLM and asks a YES/NO question:

    "Is the highlighted text actually a <ENTITY_TYPE> in this sentence?"

The model answers YES (keep the detection) or NO (discard as false positive).

Design Decisions
----------------
- Model-agnostic: Any HuggingFace instruction-following model can be plugged
  in via the `model_name` constructor parameter.
- Default model: microsoft/Phi-3.5-mini-instruct — fast, accurate, small
  enough for GPU deployment alongside the main guardrail models.
- Context window: ±150 characters around the matched span, to give the model
  enough sentence context without blowing up the prompt.
- Lazy loading: The model is NOT loaded until the first call to
  `is_false_positive()`. This avoids loading it at all if no risky entities
  are detected in a given request.
- Temperature: Set to 0.0 (greedy decoding) for fully deterministic output.
"""

from __future__ import annotations

import re
import torch
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

# System prompt: instruct the model to act as a strict binary classifier
_SYSTEM_PROMPT = (
    "You are a precise PII detection assistant. "
    "Your only job is to answer YES or NO. "
    "Do not explain. Do not add punctuation. Output only the single word YES or NO."
)

# User prompt template: filled at runtime with the entity type, matched text,
# and surrounding sentence context.
_USER_PROMPT_TEMPLATE = (
    "Sentence context: \"{context}\"\n"
    "Highlighted text: \"{match}\"\n\n"
    "Is the highlighted text actually a {entity_type} in this sentence context?\n"
    "Answer YES if it is a real {entity_type}. Answer NO if it is not.\n"
    "Answer:"
)

# How many characters to include on each side of the matched span as context
_CONTEXT_WINDOW = 150


class FalsePositiveFilter:
    """
    Wraps a local instruction-following SLM to validate risky PII detections.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: microsoft/Phi-3.5-mini-instruct.
        Can be any instruction-following model (Phi, Mistral, LLaMA, Qwen, etc.)
    quant : bool
        If True, load the model in 8-bit quantization using bitsandbytes.
        Reduces VRAM at a small accuracy cost.
    device : str or int
        "cpu", "mps", or a CUDA device index (e.g. 0). Default: -1 (auto).
    cache_dir : str
        Directory to cache downloaded model weights.
    hf_token : bool or str
        If True, uses the HF_TOKEN env variable. Pass a string to use directly.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        quant: bool = False,
        device: Optional[Union[str, int]] = -1,
        cache_dir: str = "./models",
        hf_token: Union[bool, str] = True,
    ):
        self.model_name = model_name
        self._quantize = quant
        self._cache_dir = cache_dir
        self._hf_token = hf_token
        self._device_arg = device

        # Resolved at load time
        self._pipe = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy model loader
    # ------------------------------------------------------------------

    def _load(self):
        """Load the tokenizer, model, and pipeline. Called once on first use."""
        if self._loaded:
            return

        print(f"[FalsePositiveFilter] Loading model: {self.model_name}")

        compute_dtype = torch.float16
        attn_implementation = "sdpa"

        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                try:
                    import flash_attn  # noqa: F401
                    attn_implementation = "flash_attention_2"
                    print("[FalsePositiveFilter] ✓ Flash Attention 2")
                except ImportError:
                    print("[FalsePositiveFilter] ⚠ Flash Attention 2 not available, using SDPA")

        model_kwargs: dict = {
            "cache_dir": self._cache_dir,
            "token": self._hf_token,
            "torch_dtype": compute_dtype,
            "attn_implementation": attn_implementation,
        }

        if self._quantize:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            elif str(self._device_arg).lower() == "mps":
                model_kwargs["device_map"] = "mps"
            else:
                model_kwargs["device_map"] = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self._cache_dir,
            token=self._hf_token,
        )

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=5,        # We only need YES or NO — keep it tight
            do_sample=False,         # Greedy decoding — fully deterministic
            temperature=None,        # Must be None when do_sample=False for some models
            return_full_text=False,  # Return only generated tokens, not the prompt
        )

        self._loaded = True
        print(f"[FalsePositiveFilter] ✓ Model loaded: {self.model_name}")

    # ------------------------------------------------------------------
    # Context extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context(text: str, start: int, end: int, window: int = _CONTEXT_WINDOW) -> str:
        """
        Extract a substring of `text` centered around the span [start, end],
        padded by `window` characters on each side.

        Returns clean context string with leading/trailing whitespace stripped.
        """
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end].strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_false_positive(
        self,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> bool:
        """
        Ask the SLM whether the detected span is a genuine PII entity or a
        false positive.

        Parameters
        ----------
        text        : Full original text being analyzed
        start       : Start character index of the detected span
        end         : End character index of the detected span
        entity_type : Presidio entity type (e.g. "DATE_TIME", "PHONE_NUMBER")

        Returns
        -------
        True  → The detection IS a false positive (should be discarded)
        False → The detection is genuine PII (should be kept)
        """
        # Lazy-load the model on first call
        self._load()

        matched_text = text[start:end]
        context = self._extract_context(text, start, end)

        # Build a human-readable entity label for the prompt
        entity_label = entity_type.replace("_", " ").title()

        prompt = _USER_PROMPT_TEMPLATE.format(
            context=context,
            match=matched_text,
            entity_type=entity_label,
        )

        # Build messages in chat format for instruction-following models
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            result = self._pipe(messages)
            raw_output: str = result[0]["generated_text"].strip().upper()
        except Exception as e:
            # On any model error, default to KEEPING the detection (safe side)
            print(f"[FalsePositiveFilter] ⚠ Inference error: {e}. Defaulting to keep detection.")
            return False

        # Parse the model's answer
        # We look for YES/NO anywhere in the output in case the model adds
        # extra tokens despite our tight max_new_tokens=5 setting
        if re.search(r"\bNO\b", raw_output):
            return True   # False positive — discard this detection
        elif re.search(r"\bYES\b", raw_output):
            return False  # Genuine PII — keep this detection
        else:
            # Ambiguous output — default to keeping the detection (safe side)
            print(
                f"[FalsePositiveFilter] ⚠ Ambiguous model output: '{raw_output}' "
                f"for '{matched_text}' ({entity_type}). Defaulting to keep."
            )
            return False

    def batch_filter(
        self,
        text: str,
        detections: list,
        risky_types: set,
    ) -> list:
        """
        Filter a list of RecognizerResult-like objects, running the SLM context
        check only on those whose entity_type is in `risky_types`.

        Parameters
        ----------
        text        : Full original text
        detections  : List of objects with .start, .end, .entity_type attributes
        risky_types : Set of entity type strings that need context validation

        Returns
        -------
        Filtered list with false positives removed.
        """
        if not detections:
            return detections

        # Check whether any risky entities are present at all
        # If not, skip loading the model entirely
        has_risky = any(d.entity_type in risky_types for d in detections)
        if not has_risky:
            return detections

        kept = []
        for detection in detections:
            if detection.entity_type in risky_types:
                fp = self.is_false_positive(
                    text=text,
                    start=detection.start,
                    end=detection.end,
                    entity_type=detection.entity_type,
                )
                if fp:
                    print(
                        f"[FalsePositiveFilter] Discarded false positive: "
                        f"'{text[detection.start:detection.end]}' ({detection.entity_type})"
                    )
                    continue  # Drop this detection
            kept.append(detection)

        return kept
"""
RECAP Framework — Phase 3b: LLM Context-Based False Positive Filter
====================================================================
This module implements the false positive filtering step described in the
RECAP architecture. For "risky" entity detections (e.g. a 2-digit number
flagged as DATE_TIME), it passes the match and its surrounding sentence
context to a local SLM and asks a YES/NO question:

    "Is the highlighted text actually a <ENTITY_TYPE> in this sentence?"

The model answers YES (keep the detection) or NO (discard as false positive).

Design Decisions
----------------
- Model-agnostic: Any HuggingFace instruction-following model can be plugged
  in via the `model_name` constructor parameter.
- Default model: microsoft/Phi-3.5-mini-instruct — fast, accurate, small
  enough for GPU deployment alongside the main guardrail models.
- Context window: ±150 characters around the matched span, to give the model
  enough sentence context without blowing up the prompt.
- Lazy loading: The model is NOT loaded until the first call to
  `is_false_positive()`. This avoids loading it at all if no risky entities
  are detected in a given request.
- Temperature: Set to 0.0 (greedy decoding) for fully deterministic output.
"""

from __future__ import annotations

import re
import torch
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

# System prompt: instruct the model to act as a strict binary classifier
_SYSTEM_PROMPT = (
    "You are a precise PII detection assistant. "
    "Your only job is to answer YES or NO. "
    "Do not explain. Do not add punctuation. Output only the single word YES or NO."
)

# User prompt template: filled at runtime with the entity type, matched text,
# and surrounding sentence context.
_USER_PROMPT_TEMPLATE = (
    "Sentence context: \"{context}\"\n"
    "Highlighted text: \"{match}\"\n\n"
    "Is the highlighted text actually a {entity_type} in this sentence context?\n"
    "Answer YES if it is a real {entity_type}. Answer NO if it is not.\n"
    "Answer:"
)

# How many characters to include on each side of the matched span as context
_CONTEXT_WINDOW = 150


class FalsePositiveFilter:
    """
    Wraps a local instruction-following SLM to validate risky PII detections.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: microsoft/Phi-3.5-mini-instruct.
        Can be any instruction-following model (Phi, Mistral, LLaMA, Qwen, etc.)
    quant : bool
        If True, load the model in 8-bit quantization using bitsandbytes.
        Reduces VRAM at a small accuracy cost.
    device : str or int
        "cpu", "mps", or a CUDA device index (e.g. 0). Default: -1 (auto).
    cache_dir : str
        Directory to cache downloaded model weights.
    hf_token : bool or str
        If True, uses the HF_TOKEN env variable. Pass a string to use directly.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        quant: bool = False,
        device: Optional[Union[str, int]] = -1,
        cache_dir: str = "./models",
        hf_token: Union[bool, str] = True,
    ):
        self.model_name = model_name
        self._quantize = quant
        self._cache_dir = cache_dir
        self._hf_token = hf_token
        self._device_arg = device

        # Resolved at load time
        self._pipe = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy model loader
    # ------------------------------------------------------------------

    def _load(self):
        """Load the tokenizer, model, and pipeline. Called once on first use."""
        if self._loaded:
            return

        print(f"[FalsePositiveFilter] Loading model: {self.model_name}")

        compute_dtype = torch.float16
        attn_implementation = "sdpa"

        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                try:
                    import flash_attn  # noqa: F401
                    attn_implementation = "flash_attention_2"
                    print("[FalsePositiveFilter] ✓ Flash Attention 2")
                except ImportError:
                    print("[FalsePositiveFilter] ⚠ Flash Attention 2 not available, using SDPA")

        model_kwargs: dict = {
            "cache_dir": self._cache_dir,
            "token": self._hf_token,
            "torch_dtype": compute_dtype,
            "attn_implementation": attn_implementation,
        }

        if self._quantize:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            elif str(self._device_arg).lower() == "mps":
                model_kwargs["device_map"] = "mps"
            else:
                model_kwargs["device_map"] = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self._cache_dir,
            token=self._hf_token,
        )

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=5,        # We only need YES or NO — keep it tight
            do_sample=False,         # Greedy decoding — fully deterministic
            temperature=None,        # Must be None when do_sample=False for some models
            return_full_text=False,  # Return only generated tokens, not the prompt
        )

        self._loaded = True
        print(f"[FalsePositiveFilter] ✓ Model loaded: {self.model_name}")

    # ------------------------------------------------------------------
    # Context extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context(text: str, start: int, end: int, window: int = _CONTEXT_WINDOW) -> str:
        """
        Extract a substring of `text` centered around the span [start, end],
        padded by `window` characters on each side.

        Returns clean context string with leading/trailing whitespace stripped.
        """
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end].strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_false_positive(
        self,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> bool:
        """
        Ask the SLM whether the detected span is a genuine PII entity or a
        false positive.

        Parameters
        ----------
        text        : Full original text being analyzed
        start       : Start character index of the detected span
        end         : End character index of the detected span
        entity_type : Presidio entity type (e.g. "DATE_TIME", "PHONE_NUMBER")

        Returns
        -------
        True  → The detection IS a false positive (should be discarded)
        False → The detection is genuine PII (should be kept)
        """
        # Lazy-load the model on first call
        self._load()

        matched_text = text[start:end]
        context = self._extract_context(text, start, end)

        # Build a human-readable entity label for the prompt
        entity_label = entity_type.replace("_", " ").title()

        prompt = _USER_PROMPT_TEMPLATE.format(
            context=context,
            match=matched_text,
            entity_type=entity_label,
        )

        # Build messages in chat format for instruction-following models
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            result = self._pipe(messages)
            raw_output: str = result[0]["generated_text"].strip().upper()
        except Exception as e:
            # On any model error, default to KEEPING the detection (safe side)
            print(f"[FalsePositiveFilter] ⚠ Inference error: {e}. Defaulting to keep detection.")
            return False

        # Parse the model's answer
        # We look for YES/NO anywhere in the output in case the model adds
        # extra tokens despite our tight max_new_tokens=5 setting
        if re.search(r"\bNO\b", raw_output):
            return True   # False positive — discard this detection
        elif re.search(r"\bYES\b", raw_output):
            return False  # Genuine PII — keep this detection
        else:
            # Ambiguous output — default to keeping the detection (safe side)
            print(
                f"[FalsePositiveFilter] ⚠ Ambiguous model output: '{raw_output}' "
                f"for '{matched_text}' ({entity_type}). Defaulting to keep."
            )
            return False

    def batch_filter(
        self,
        text: str,
        detections: list,
        risky_types: set,
    ) -> list:
        """
        Filter a list of RecognizerResult-like objects, running the SLM context
        check only on those whose entity_type is in `risky_types`.

        Parameters
        ----------
        text        : Full original text
        detections  : List of objects with .start, .end, .entity_type attributes
        risky_types : Set of entity type strings that need context validation

        Returns
        -------
        Filtered list with false positives removed.
        """
        if not detections:
            return detections

        # Check whether any risky entities are present at all
        # If not, skip loading the model entirely
        has_risky = any(d.entity_type in risky_types for d in detections)
        if not has_risky:
            return detections

        kept = []
        for detection in detections:
            if detection.entity_type in risky_types:
                fp = self.is_false_positive(
                    text=text,
                    start=detection.start,
                    end=detection.end,
                    entity_type=detection.entity_type,
                )
                if fp:
                    print(
                        f"[FalsePositiveFilter] Discarded false positive: "
                        f"'{text[detection.start:detection.end]}' ({detection.entity_type})"
                    )
                    continue  # Drop this detection
            kept.append(detection)

        return kept