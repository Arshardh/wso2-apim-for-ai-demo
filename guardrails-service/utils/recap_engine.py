"""
RECAP Framework — Core Engine
==============================
Implements the full RECAP pipeline with ECI (Enterprise Confidential Information) support.

Pipeline Overview
-----------------
Phase 1a — Structured PII detection via Presidio regex recognizers
Phase 1b — Contextual PII detection via GLiNER NER model
Phase 1c — ECI keyword/regex detection (customer-configured internal terms)
Phase 1d — ECI SLM contextual detection (catches unlabeled internal context)
Phase 2/3a — Unified span overlap resolution (span-swallowing across all sources)
Phase 3b — LLM false positive filter for risky detections
Phase 4 — Reversible tokenization -> safe_payload + synapse_memory_map
Phase 5 — /restore: string-replace to detokenize LLM response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple

from presidio_analyzer import (
    RecognizerRegistry,
    RecognizerResult,
)

from utils.analyzer_engine import AnalyzerEngine
from utils.gliner_recognizer import GLiNERRecognizer
from utils.false_positive_filter import FalsePositiveFilter
from utils.constants import (
    PRESIDIO_TO_GLINER,
    DEFAULT_THRESHOLDS,
    ENTITY_TO_TOKEN_NAME,
    RISKY_ENTITY_TYPES,
)
from utils.eci_constants import ALL_ECI_ENTITY_TYPES, ECI_RISKY_ENTITY_TYPES


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RecapResult:
    """Output of a full RECAP sanitization pass."""
    safe_payload: str
    synapse_memory_map: Dict[str, str]
    detections: List["DetectedEntity"]


@dataclass
class DetectedEntity:
    """A single confirmed detection after all phases and filtering."""
    entity_type: str
    start: int
    end: int
    score: float
    value: str
    token: str = field(default="")
    source: str = field(default="")


# ---------------------------------------------------------------------------
# Phase 2/3a: Span Overlap Resolution
# ---------------------------------------------------------------------------

def resolve_overlaps(detections: List[RecognizerResult]) -> List[RecognizerResult]:
    """
    Resolve overlapping detections via span-swallowing.
    Sort by length descending; accept only non-overlapping spans.
    Larger spans win — works uniformly across PII and ECI detections.
    """
    if not detections:
        return []

    sorted_detections = sorted(
        detections,
        key=lambda r: (r.end - r.start),
        reverse=True
    )

    accepted: List[RecognizerResult] = []
    for candidate in sorted_detections:
        overlaps = any(
            candidate.start < kept.end and candidate.end > kept.start
            for kept in accepted
        )
        if not overlaps:
            accepted.append(candidate)

    accepted.sort(key=lambda r: r.start)
    return accepted


# ---------------------------------------------------------------------------
# Phase 4: Tokenizer
# ---------------------------------------------------------------------------

def build_token(entity_type: str, count: int) -> str:
    """
    Generate a gateway-compatible token string.
    build_token("PERSON", 1)       -> "[NAME_1]"
    build_token("ECI_PROJECT", 1)  -> "[PROJECT_1]"
    build_token("ECI_HR_DATA", 2)  -> "[HR_DATA_2]"
    """
    label = ENTITY_TO_TOKEN_NAME.get(entity_type, entity_type)
    return f"[{label}_{count}]"


def tokenize(
    text: str,
    detections: List[RecognizerResult],
) -> Tuple[str, Dict[str, str], List[DetectedEntity]]:
    """
    Phase 4: Replace all confirmed spans with numbered tokens.
    Processes right-to-left to preserve character index integrity.
    Returns (safe_payload, synapse_memory_map, detected_entities).
    """
    type_counters: Dict[str, int] = {}

    entities: List[DetectedEntity] = []
    for r in detections:
        source = ""
        if hasattr(r, "recognition_metadata") and r.recognition_metadata:
            rname = r.recognition_metadata.get(RecognizerResult.RECOGNIZER_NAME_KEY, "")
            if "GLiNER" in rname:
                source = "gliner"
            elif "ECIKeyword" in rname:
                source = "eci_keyword"
            elif "ECISLM" in rname:
                source = "eci_slm"
            else:
                source = "presidio"

        entities.append(DetectedEntity(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=r.score,
            value=text[r.start:r.end],
            source=source,
        ))

    synapse_memory_map: Dict[str, str] = {}
    safe_payload = text

    for entity in reversed(entities):
        type_counters[entity.entity_type] = type_counters.get(entity.entity_type, 0) + 1
        token = build_token(entity.entity_type, type_counters[entity.entity_type])
        entity.token = token
        synapse_memory_map[token] = entity.value
        safe_payload = safe_payload[:entity.start] + token + safe_payload[entity.end:]

    entities.reverse()
    return safe_payload, synapse_memory_map, entities


# ---------------------------------------------------------------------------
# Phase 5: Restoration
# ---------------------------------------------------------------------------

def restore(llm_response: str, synapse_memory_map: Dict[str, str]) -> str:
    """
    Phase 5: Detokenize an LLM response using the synapse_memory_map.
    Simple string-replace — order doesn't matter (tokens are non-overlapping).
    """
    restored = llm_response
    for token, original_value in synapse_memory_map.items():
        restored = restored.replace(token, original_value)
    return restored


# ---------------------------------------------------------------------------
# RECAP Engine
# ---------------------------------------------------------------------------

class RecapEngine:
    """
    Full RECAP pipeline orchestrator combining PII + ECI detection.

    Parameters
    ----------
    entities         : Standard Presidio PII entity types to detect.
    model_name       : GLiNER model for Phase 1b contextual NER.
    get_entity_threshold : Score threshold callable per entity type.
    fp_filter        : FalsePositiveFilter for Phase 3b. Its SLM pipeline
                       is also shared with Phase 1d to avoid loading twice.
    risky_entity_types : Entity types needing Phase 3b validation.
    eci_config       : ECIConfig instance. If None, Phases 1c/1d are skipped.
    """

    def __init__(
        self,
        entities: List[str],
        model_name: str = "urchade/gliner_small-v2.1",
        get_entity_threshold: Callable[[str], float] = None,
        fp_filter: Optional[FalsePositiveFilter] = None,
        risky_entity_types: Optional[set] = None,
        eci_config=None,
    ):
        self.entities = entities
        self.model_name = model_name
        self.get_entity_threshold = get_entity_threshold or self._default_threshold
        self.fp_filter = fp_filter
        self.eci_config = eci_config

        # Merge standard + ECI risky types for Phase 3b
        base_risky = risky_entity_types or RISKY_ENTITY_TYPES
        self.risky_entity_types = base_risky | ECI_RISKY_ENTITY_TYPES

        # ---- Phase 1b: GLiNER ------------------------------------------------
        self.gliner_recognizer = GLiNERRecognizer(
            supported_entities=self.entities,
            model_name=model_name,
        )

        # ---- Presidio registry (Phases 1a + 1b) ------------------------------
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=["en", "es", "it", "pl"])
        registry.add_recognizer(self.gliner_recognizer)

        # ---- Phase 1c: ECI Keyword/Regex Recognizer --------------------------
        self.eci_keyword_recognizer = None
        if eci_config is not None and eci_config.enabled_categories:
            from utils.eci_keyword_recognizer import ECIKeywordRecognizer
            self.eci_keyword_recognizer = ECIKeywordRecognizer(eci_config=eci_config)
            registry.add_recognizer(self.eci_keyword_recognizer)
            print(
                f"[RecapEngine] Phase 1c: ECI keyword recognizer loaded "
                f"({len(eci_config.enabled_categories)} categories)"
            )

        self.analyzer = AnalyzerEngine(
            registry=registry,
            supported_languages=["en"],
        )

        # ---- Phase 1d: ECI SLM Recognizer ------------------------------------
        # Shares fp_filter's SLM pipeline — same model weights, no double VRAM.
        self.eci_slm_recognizer = None
        if (
            eci_config is not None
            and eci_config.slm_enabled
            and eci_config.enabled_categories
            and fp_filter is not None
        ):
            fp_filter._load()
            if fp_filter._pipe is not None:
                from utils.eci_slm_recognizer import ECISLMRecognizer
                self.eci_slm_recognizer = ECISLMRecognizer(
                    eci_config=eci_config,
                    slm_pipe=fp_filter._pipe,
                )
                print("[RecapEngine] Phase 1d: ECI SLM recognizer loaded (shared pipeline)")
            else:
                print("[RecapEngine] Phase 1d: SLM pipeline not available. Skipping.")

    @staticmethod
    def _default_threshold(entity_type: str) -> float:
        return DEFAULT_THRESHOLDS.get(entity_type, 0.5)

    def _detect(self, text: str, entities: List[str]) -> List[RecognizerResult]:
        """
        Run all detection phases (1a-1d) and return unified filtered results.

        Routing rules:
          ECI entities       -> only from ECI recognizers (1c or 1d)
          Contextual PII     -> only from GLiNER (1b)
          Structured PII     -> only from Presidio regex (1a)
        """
        eci_entity_types = (
            self.eci_config.all_entity_types if self.eci_config else []
        )
        all_entities = list(set(entities) | set(eci_entity_types))

        # Phases 1a + 1b + 1c through unified Presidio analyzer
        raw_results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=all_entities,
            deduplicate=False,
        )

        # Phase 1d: SLM runs independently, results merged
        slm_results = []
        if self.eci_slm_recognizer is not None:
            slm_results = self.eci_slm_recognizer.analyze(
                text=text,
                entities=eci_entity_types,
            )

        all_results = raw_results + slm_results

        # Recognizer name lookups for routing
        eci_kw_name  = self.eci_keyword_recognizer.name if self.eci_keyword_recognizer else ""
        eci_slm_name = self.eci_slm_recognizer.name if self.eci_slm_recognizer else ""
        gliner_name  = self.gliner_recognizer.name

        filtered = []
        for r in all_results:
            rname = ""
            if hasattr(r, "recognition_metadata") and r.recognition_metadata:
                rname = r.recognition_metadata.get(RecognizerResult.RECOGNIZER_NAME_KEY, "")

            is_eci        = r.entity_type in ALL_ECI_ENTITY_TYPES
            is_contextual = r.entity_type in PRESIDIO_TO_GLINER

            if is_eci:
                if rname in (eci_kw_name, eci_slm_name):
                    filtered.append(r)
            elif is_contextual:
                if rname == gliner_name:
                    filtered.append(r)
            else:
                if r.entity_type in entities:
                    filtered.append(r)

        # Apply per-entity confidence thresholds
        return [
            r for r in filtered
            if r.score >= self.get_entity_threshold(r.entity_type)
        ]

    def sanitize(self, text: str, entities: Optional[List[str]] = None) -> RecapResult:
        """
        Run the full RECAP pipeline (Phases 1a-1d, 2/3a, 3b, 4).

        ECI entity types from eci_config are included automatically —
        the caller does not need to list them in `entities`.
        """
        if entities is None:
            entities = self.entities

        detections = self._detect(text, entities)
        detections = resolve_overlaps(detections)

        if self.fp_filter is not None:
            detections = self.fp_filter.batch_filter(
                text=text,
                detections=detections,
                risky_types=self.risky_entity_types,
            )

        safe_payload, synapse_memory_map, detected_entities = tokenize(text, detections)

        return RecapResult(
            safe_payload=safe_payload,
            synapse_memory_map=synapse_memory_map,
            detections=detected_entities,
        )

    @staticmethod
    def restore(llm_response: str, synapse_memory_map: Dict[str, str]) -> str:
        """Phase 5: Restore original values into an LLM response."""
        return restore(llm_response, synapse_memory_map)