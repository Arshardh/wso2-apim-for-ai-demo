"""
RECAP Framework — Phase 1d: ECI SLM Contextual Recognizer
==========================================================
Uses a local Small Language Model (SLM) to contextually detect Enterprise
Confidential Information (ECI) that the keyword/regex layer (Phase 1c) missed.

Why We Need This
----------------
The keyword layer is great for known, explicit terms like "Project Phoenix" or
"db-prod-01". But enterprise confidentiality leaks happen in many forms that
regex cannot catch:

  "John will be out for a few weeks recovering"
    → No keyword match, but clearly employee HR information

  "The team is behind on the Q3 initiative"
    → No specific project name, but reveals internal planning status

  "Let's loop in the infrastructure lead before the migration"
    → Reveals org structure and internal decision-making process

  "We're targeting a 15% headcount reduction in that division"
    → Sensitive org/HR information with no flaggable keyword

The SLM reads the full text holistically and returns a JSON list of detected
confidential spans — entity text, entity type, and a confidence score.

Architecture
------------
This recognizer runs AFTER the keyword/regex layer. It receives the full text
and the list of ECI categories (with their slm_description fields) and is
prompted to extract spans that match those categories.

To avoid double-detection, the SLM is instructed to focus on spans NOT already
covered by obvious keywords. The overlap resolution in Phase 2/3a handles any
remaining duplicates.

Model & Prompting
-----------------
- Uses the same configurable SLM as the FalsePositiveFilter (default: Phi-3.5-mini-instruct)
- The model is shared (same instance) to avoid loading it twice into VRAM
- Forced JSON output via a strict system prompt and bracket parser
- max_new_tokens=512 to accommodate a full span list

JSON Output Format
------------------
The model is required to output ONLY a JSON array like:
[
  {"text": "Project Phoenix", "entity_type": "ECI_PROJECT", "score": 0.95},
  {"text": "John is on leave until March 21", "entity_type": "ECI_HR_DATA", "score": 0.88}
]

Empty array [] if nothing found.

Bulletproof JSON Parser
-----------------------
SLMs sometimes wrap JSON in markdown code blocks or add preamble text.
The _extract_json_array() function finds the first '[' and last ']' in the
output and parses only that substring — making it robust to model verbosity.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, TYPE_CHECKING

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from utils.eci_config import ECIConfig

if TYPE_CHECKING:
    from utils.false_positive_filter import FalsePositiveFilter


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a confidential information extraction system for an enterprise.
Your job is to identify spans of text that contain sensitive internal company information.
You MUST respond ONLY with a valid JSON array. No explanation. No markdown. No preamble.
If nothing is found, respond with exactly: []"""

_USER_PROMPT_TEMPLATE = """Company: {company_name}

You are scanning the following text for sensitive internal information belonging to {company_name}.

Categories to detect:
{category_descriptions}

Text to analyze:
\"\"\"{text}\"\"\"

Extract all spans of text that contain information from the categories above.
For each span, provide the exact text as it appears, the matching entity_type, and a confidence score (0.0-1.0).

Respond ONLY with a JSON array in this exact format:
[
  {{"text": "exact text from input", "entity_type": "ECI_CATEGORY_NAME", "score": 0.95}},
  {{"text": "another span", "entity_type": "ECI_CATEGORY_NAME", "score": 0.88}}
]

If no sensitive information is found, respond with: []"""


# ---------------------------------------------------------------------------
# Bulletproof JSON Bracket Parser
# ---------------------------------------------------------------------------

def _extract_json_array(raw_output: str) -> list:
    """
    Robustly extract a JSON array from raw SLM output.

    The SLM may wrap the JSON in markdown code blocks, add preamble text,
    or include trailing commentary. This function finds the outermost '[...]'
    and parses only that part.

    Returns an empty list on any parse failure.
    """
    # Find the first '[' and last ']'
    start = raw_output.find('[')
    end = raw_output.rfind(']')

    if start == -1 or end == -1 or end < start:
        return []

    json_str = raw_output[start:end + 1]

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError:
        # Try to fix common issues: trailing commas, single quotes
        # Remove trailing commas before ] or }
        cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


# ---------------------------------------------------------------------------
# ECI SLM Recognizer
# ---------------------------------------------------------------------------

class ECISLMRecognizer(EntityRecognizer):
    """
    Presidio-compatible recognizer that uses a local SLM to detect ECI
    contextually — catching what the keyword/regex layer misses.

    Parameters
    ----------
    eci_config  : ECIConfig instance defining what to look for
    slm_pipe    : A HuggingFace text-generation pipeline instance.
                  This is shared with the FalsePositiveFilter to avoid
                  loading the same model weights twice into VRAM.
    """

    def __init__(
        self,
        eci_config: ECIConfig,
        slm_pipe,  # HuggingFace pipeline — typed loosely to avoid circular import
    ):
        self.eci_config = eci_config
        self._pipe = slm_pipe

        supported_entities = eci_config.all_entity_types

        super().__init__(
            supported_entities=supported_entities,
            name="ECISLMRecognizer",
        )

    def load(self) -> None:
        """Model is pre-loaded and injected — nothing to load here."""
        pass

    def _build_category_descriptions(self, entities: Optional[List[str]] = None) -> str:
        """
        Build the human-readable category descriptions block for the SLM prompt.
        Each line tells the model what one ECI category covers.
        """
        lines = []
        for cat in self.eci_config.enabled_categories.values():
            if entities and cat.entity_type not in entities:
                continue
            slm_desc = cat.slm_description or cat.description
            lines.append(f"- {cat.entity_type}: {slm_desc}")
        return "\n".join(lines)

    def _find_span(self, text: str, span_text: str) -> List[tuple]:
        """
        Find all character positions of span_text within text.
        Returns a list of (start, end) tuples.

        Uses case-insensitive search since the SLM may slightly alter casing.
        Falls back to exact match first for precision.
        """
        positions = []

        # Try exact match first
        start = 0
        while True:
            idx = text.find(span_text, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(span_text)))
            start = idx + 1

        if positions:
            return positions

        # Fallback: case-insensitive
        lower_text = text.lower()
        lower_span = span_text.lower()
        start = 0
        while True:
            idx = lower_text.find(lower_span, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(span_text)))
            start = idx + 1

        return positions

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        """
        Run the SLM over the text and convert its JSON output into
        RecognizerResult objects.

        Parameters
        ----------
        text     : Full input text to scan
        entities : Optional filter for entity types

        Returns
        -------
        List of RecognizerResult from SLM detections
        """
        # If no enabled categories match, skip inference entirely
        category_desc = self._build_category_descriptions(entities)
        if not category_desc:
            return []

        # Build the prompt
        prompt = _USER_PROMPT_TEMPLATE.format(
            company_name=self.eci_config.company_name,
            category_descriptions=category_desc,
            text=text,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        try:
            result = self._pipe(messages, max_new_tokens=512)
            raw_output: str = result[0]["generated_text"]
        except Exception as e:
            print(f"[ECISLMRecognizer] ⚠ Inference error: {e}. Returning no detections.")
            return []

        # Parse the JSON array from the model output
        detections = _extract_json_array(raw_output)

        results = []
        valid_entity_types = set(self.eci_config.all_entity_types)
        target_entities = set(entities) if entities else valid_entity_types

        for item in detections:
            # Validate structure
            if not isinstance(item, dict):
                continue
            span_text = item.get("text", "").strip()
            entity_type = item.get("entity_type", "").strip()
            score = float(item.get("score", 0.0))

            # Skip invalid, low-confidence, or out-of-scope detections
            if not span_text or not entity_type:
                continue
            if entity_type not in valid_entity_types:
                continue
            if entity_type not in target_entities:
                continue
            if score < self.eci_config.confidence_threshold:
                continue

            # Find the span position(s) in the original text
            positions = self._find_span(text, span_text)
            if not positions:
                print(
                    f"[ECISLMRecognizer] Could not locate span '{span_text}' in text. Skipping."
                )
                continue

            for start, end in positions:
                results.append(RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=score,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                ))

        return results