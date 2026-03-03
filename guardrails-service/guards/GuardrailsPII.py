import json
import os
from typing import Any, Callable, Dict, Optional, List, Tuple, Sequence, cast

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.validator_base import ErrorSpan

# ---------------------------------------------------------------------------
# NOTE: AnonymizerEngine and the old Presidio imports are no longer used
# directly in this file. Detection and tokenization are now handled entirely
# by the RECAP engine pipeline. They are kept commented below for reference.
#
# from presidio_anonymizer import AnonymizerEngine
# from presidio_analyzer import RecognizerRegistry, EntityRecognizer
# from presidio_anonymizer import RecognizerResult as AnonymizerRecognizerResult
# ---------------------------------------------------------------------------

from utils.recap_engine import RecapEngine, RecapResult
from utils.false_positive_filter import FalsePositiveFilter
from utils.constants import PRESIDIO_TO_GLINER, DEFAULT_THRESHOLDS

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Threshold helper — identical to the original, kept for backwards compat
# ---------------------------------------------------------------------------

def get_entity_threshold(entity: str) -> float:
    if entity in DEFAULT_THRESHOLDS:
        return DEFAULT_THRESHOLDS[entity]
    if entity in PRESIDIO_TO_GLINER:
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Pydantic Models
#
# InferenceInput / InferenceOutputResult are identical to the original.
# InferenceOutput gains two new fields (synapse_memory_map, token) that are
# optional with defaults so any existing callers won't break.
# PIIValidationResponse gains safe_payload + synapse_memory_map (optional).
# ---------------------------------------------------------------------------

class InferenceInput(BaseModel):
    text: str
    entities: List[str]


class InferenceOutputResult(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float
    token: str = ""                    # NEW: assigned RECAP token e.g. "[NAME_1]"


class InferenceOutput(BaseModel):
    results: List[InferenceOutputResult]
    anonymized_text: str               # Now contains RECAP tokenized text
    synapse_memory_map: Dict[str, str] = {}  # NEW: token → original value map


class PIIDetectionRequest(BaseModel):
    text: str = Field(..., description="Text to validate for PII")
    piiEntities: List[str] = Field(..., description="List of PII entity types to detect and filter")


class PIIEntity(BaseModel):
    piiEntity: str
    piiValue: str


class PIIValidationResponse(BaseModel):
    verdict: bool
    assessment: List[PIIEntity]
    safe_payload: str = ""                   # NEW: tokenized text safe to send to LLM
    synapse_memory_map: Dict[str, str] = {}  # NEW: for restoration after LLM responds


# ---------------------------------------------------------------------------
# Sanitize / Restore models — for the dedicated /sanitize and /restore endpoints
# ---------------------------------------------------------------------------

class SanitizeRequest(BaseModel):
    payload: str = Field(..., description="Raw text to sanitize before sending to LLM")
    piiEntities: Optional[List[str]] = Field(
        None,
        description="Optional list of PII entity types. Defaults to all supported entities."
    )


class SanitizeResponse(BaseModel):
    status: str
    safe_payload: str
    synapse_memory_map: Dict[str, str]
    detections: List[InferenceOutputResult] = []


class RestoreRequest(BaseModel):
    llm_response: str = Field(..., description="LLM response containing RECAP tokens")
    synapse_memory_map: Dict[str, str] = Field(
        ..., description="Memory map returned by /sanitize"
    )


class RestoreResponse(BaseModel):
    status: str
    final_restored_payload: str


# ---------------------------------------------------------------------------
# GuardrailsPII — RECAP + ECI Enhanced
# ---------------------------------------------------------------------------

@register_validator(name="guardrails/guardrails_pii", data_type="string")
class GuardrailsPII(Validator):
    """
    Drop-in replacement for the original GuardrailsPII validator.

    Upgrades the original Presidio + GLiNER anonymization pipeline with the
    full RECAP Framework:

        Phase 1a — Presidio regex (structured PII: emails, SSNs, credit cards)
        Phase 1b — GLiNER NER (contextual PII: names, addresses, dates)
        Phase 1c — ECI keyword/regex (internal project names, HR terms, servers)
        Phase 1d — ECI SLM contextual (catches unlabeled internal context)
        Phase 2/3a — Span-swallowing overlap resolution
        Phase 3b — LLM false positive filter for risky detections
        Phase 4 — Reversible tokenization → [NAME_1], [PROJECT_1], etc.
                  + synapse_memory_map for downstream restoration

    Backwards Compatibility
    -----------------------
    - Constructor signature is fully backwards compatible. All original
      parameters (entities, model_name, get_entity_threshold, on_fail,
      use_local) work exactly as before.
    - Three new optional parameters added: fp_filter_model, fp_filter_quant,
      eci_config. All have safe defaults so existing code needs no changes.
    - _validate() still returns PassResult / FailResult with error_spans.
    - fix_value in FailResult now contains RECAP tokenized text instead of
      Presidio anonymized text — it is reversible via /restore.
    - anonymize() return signature extended to 3-tuple:
        (safe_payload, synapse_memory_map, error_spans)
      If you only unpack 2 values, wrap the call or use the new signature.

    New Parameters
    --------------
    fp_filter_model : str or None
        HuggingFace model for Phase 3b false positive filtering + Phase 1d
        ECI contextual detection. Both phases share the same loaded pipeline
        to avoid loading model weights twice into VRAM.
        Set to None to disable both Phase 3b and Phase 1d.
        Default: "microsoft/Phi-3.5-mini-instruct"

    fp_filter_quant : bool
        Load the fp_filter model in 8-bit quantization (bitsandbytes).
        Reduces VRAM at a small accuracy cost. Default: False.

    eci_config : ECIConfig or None
        Enterprise Confidential Information configuration loaded from
        config/eci_config.json via load_eci_config(). If None, Phases 1c
        and 1d are skipped and only standard PII is detected.
        Default: None
    """

    PII_ENTITIES_MAP = {
        "pii": [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "DOMAIN_NAME",
            "IP_ADDRESS",
            "DATE_TIME",
            "LOCATION",
            "PERSON",
            "URL",
        ],
        "spi": [
            "CREDIT_CARD",
            "CRYPTO",
            "IBAN_CODE",
            "NRP",
            "MEDICAL_LICENSE",
            "US_BANK_NUMBER",
            "US_DRIVER_LICENSE",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
        ],
    }

    # Full default entity list — identical to the original
    DEFAULT_ENTITIES = [
        "CREDIT_CARD",
        "CRYPTO",
        "DATE_TIME",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "NRP",
        "LOCATION",
        "PERSON",
        "PHONE_NUMBER",
        "MEDICAL_LICENSE",
        "URL",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "US_ITIN",
        "US_PASSPORT",
        "US_SSN",
        "UK_NHS",
        "ES_NIF",
        "ES_NIE",
        "IT_FISCAL_CODE",
        "IT_DRIVER_LICENSE",
        "IT_VAT_CODE",
        "IT_PASSPORT",
        "IT_IDENTITY_CARD",
        "PL_PESEL",
        "SG_NRIC_FIN",
        "SG_UEN",
        "AU_ABN",
        "AU_ACN",
        "AU_TFN",
        "AU_MEDICARE",
        "IN_PAN",
        "IN_AADHAAR",
        "IN_VEHICLE_REGISTRATION",
        "IN_VOTER",
        "IN_PASSPORT",
        "FI_PERSONAL_IDENTITY_CODE",
    ]

    def __init__(
        self,
        entities: str | List[str] | None = None,
        model_name: str = "urchade/gliner_small-v2.1",
        get_entity_threshold: Callable = get_entity_threshold,
        on_fail: Optional[Callable] = None,
        use_local: bool = True,
        # ---- NEW parameters (all optional with safe defaults) ----
        fp_filter_model: Optional[str] = "microsoft/Phi-3.5-mini-instruct",
        fp_filter_quant: bool = False,
        eci_config=None,   # Optional[ECIConfig]
        **kwargs,
    ):
        """
        Parameters match the original exactly. Three new optional parameters
        added at the end. Existing instantiation code requires zero changes.
        """
        # Resolve entities before calling super() so the list is ready
        if entities is None:
            resolved_entities = self.DEFAULT_ENTITIES
        elif isinstance(entities, str):
            assert entities in self.PII_ENTITIES_MAP, f"Invalid entity type: {entities}"
            resolved_entities = self.PII_ENTITIES_MAP[entities]
        else:
            resolved_entities = entities

        super().__init__(
            on_fail=on_fail,
            model_name=model_name,
            entities=entities,
            get_entity_threshold=get_entity_threshold,
            use_local=use_local,
            **kwargs,
        )

        self.entities = resolved_entities
        self.model_name = model_name
        self.get_entity_threshold = get_entity_threshold
        self.eci_config = eci_config

        if eci_config is not None:
            print(
                f"[GuardrailsPII] ECI detection enabled for '{eci_config.company_name}' "
                f"({len(eci_config.enabled_categories)} categories: "
                f"{', '.join(eci_config.enabled_categories.keys())})"
            )

        if self.use_local:
            # ----------------------------------------------------------------
            # Phase 3b: False Positive Filter
            # This instance is also shared with Phase 1d (ECI SLM recognizer)
            # so the SLM model weights are only loaded into VRAM once.
            # ----------------------------------------------------------------
            fp_filter = None
            if fp_filter_model is not None:
                fp_filter = FalsePositiveFilter(
                    model_name=fp_filter_model,
                    quant=fp_filter_quant,
                )
                print(
                    f"[GuardrailsPII] Phase 3b FP filter enabled: {fp_filter_model} "
                    f"(quant={fp_filter_quant})"
                )
            else:
                print("[GuardrailsPII] Phase 3b FP filter disabled.")

            # ----------------------------------------------------------------
            # RECAP Engine — orchestrates all phases (1a–1d, 2/3a, 3b, 4)
            # ----------------------------------------------------------------
            self.recap_engine = RecapEngine(
                entities=self.entities,
                model_name=model_name,
                get_entity_threshold=get_entity_threshold,
                fp_filter=fp_filter,
                eci_config=eci_config,
            )

    # ------------------------------------------------------------------
    # Core inference — local
    # ------------------------------------------------------------------

    def _inference_local(self, model_input: InferenceInput) -> InferenceOutput:
        """
        Run the full RECAP pipeline and return structured output.

        Replaces the original Presidio anonymize() call. Key differences:
          - anonymized_text now contains reversible RECAP tokens ([NAME_1])
            instead of permanent Presidio labels (<PERSON>)
          - synapse_memory_map carries the token→value dictionary
          - each result carries the assigned token string
        """
        recap_result: RecapResult = self.recap_engine.sanitize(
            text=model_input.text,
            entities=model_input.entities,
        )

        results = [
            InferenceOutputResult(
                entity_type=d.entity_type,
                start=d.start,
                end=d.end,
                score=d.score,
                token=d.token,
            )
            for d in recap_result.detections
        ]

        return InferenceOutput(
            anonymized_text=recap_result.safe_payload,
            synapse_memory_map=recap_result.synapse_memory_map,
            results=results,
        )

    # ------------------------------------------------------------------
    # Core inference — remote (unchanged from original)
    # ------------------------------------------------------------------

    def _inference_remote(self, model_input: InferenceInput) -> InferenceOutput:
        request_body = {
            "text": model_input.text,
            "entities": model_input.entities,
        }
        response = self._hub_inference_request(
            json.dumps(request_body), self.validation_endpoint  # type: ignore
        )
        return InferenceOutput.model_validate(response)

    # ------------------------------------------------------------------
    # Public: anonymize()
    #
    # CHANGED: now returns a 3-tuple (safe_payload, synapse_memory_map, error_spans)
    # The original returned a 2-tuple (anonymized_text, error_spans).
    #
    # If you have existing code that unpacks 2 values:
    #   anonymized_text, error_spans = guard.anonymize(text, entities)
    # Update it to:
    #   safe_payload, synapse_memory_map, error_spans = guard.anonymize(text, entities)
    # ------------------------------------------------------------------

    def anonymize(
        self,
        text: str,
        entities: List[str],
    ) -> Tuple[str, Dict[str, str], List[ErrorSpan]]:
        """
        Sanitize text and return the tokenized payload, memory map, and spans.

        Returns
        -------
        safe_payload : str
            Text with all PII/ECI replaced by reversible RECAP tokens.
        synapse_memory_map : dict
            { "[NAME_1]": "Alex Morgan", "[PROJECT_1]": "Phoenix", ... }
            Pass this to /restore after the LLM responds.
        error_spans : list[ErrorSpan]
            One ErrorSpan per detected entity (start, end, reason=entity_type).
        """
        input_request = InferenceInput(text=text, entities=entities)
        output = self._inference(input_request)

        output = cast(InferenceOutput, output)

        error_spans = [
            ErrorSpan(start=r.start, end=r.end, reason=r.entity_type)
            for r in output.results
        ]

        return output.anonymized_text, output.synapse_memory_map, error_spans

    # ------------------------------------------------------------------
    # guardrails-ai interface: _validate()
    #
    # Identical contract to the original — returns PassResult or FailResult.
    # Changes:
    #   - fix_value is now RECAP tokenized text (reversible) instead of
    #     Presidio anonymized text (permanent)
    #   - synapse_memory_map and safe_payload are written back into metadata
    #     so the FastAPI endpoint can retrieve them after guard.validate()
    # ------------------------------------------------------------------

    def _validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:
        entities = metadata.get("entities", self.entities)
        if entities is None:
            raise ValueError(
                "`entities` must be set in order to use the `GuardrailsPII` validator."
            )

        safe_payload, synapse_memory_map, error_spans = self.anonymize(
            text=value,
            entities=entities,
        )

        # Write RECAP outputs back into metadata so server.py can retrieve them
        # after calling await guard.validate(text, metadata=metadata)
        metadata["synapse_memory_map"] = synapse_memory_map
        metadata["safe_payload"] = safe_payload

        if len(error_spans) == 0:
            return PassResult()
        else:
            return FailResult(
                error_message=f"The following text contains PII:\n{value}",
                fix_value=safe_payload,   # Reversible RECAP token text
                error_spans=error_spans,
            )