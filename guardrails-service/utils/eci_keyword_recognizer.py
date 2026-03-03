"""
RECAP Framework — Phase 1c: ECI Keyword & Regex Recognizer
===========================================================
A Presidio-compatible EntityRecognizer that detects Enterprise Confidential
Information (ECI) using the customer-provided keyword and regex rules from
eci_config.json.

This runs as Phase 1c — after Presidio regex (1a) and GLiNER NER (1b), but
before the SLM contextual layer (1d). It feeds into the same unified span
pool that goes through overlap resolution (Phase 2/3a).

Detection Logic
---------------
For each enabled ECI category in the config:

  Plain keywords:
    Scanned using word-boundary regex so "Phoenix" doesn't match "APhoenix".
    Case-insensitive. Multi-word phrases also supported ("Project Phoenix").

  Regex patterns:
    Customer-defined regex patterns compiled once at recognizer init time.
    Applied directly against the full text with re.finditer().

Each match produces a RecognizerResult with:
  - entity_type: the category's entity_type (e.g. "ECI_PROJECT")
  - start/end: character span in the original text
  - score: 0.85 for keyword matches, 0.90 for regex matches
    (slightly higher for regex since patterns are more precise)

Score Rationale
---------------
We assign fixed high scores (0.85-0.90) rather than model confidence scores
because keyword/regex detection is deterministic — if the word "Phoenix" is
in the text and the customer said to flag "Phoenix", it IS a match. The score
is used downstream by the threshold filter, so we set it high enough to always
pass the default 0.5 threshold but leave headroom below 1.0 for the system to
distinguish keyword matches from perfect structured matches.
"""

from __future__ import annotations

import re
from typing import List, Optional

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from utils.eci_config import ECIConfig, ECICategory


# Fixed confidence scores for each detection method
_KEYWORD_SCORE = 0.85
_REGEX_SCORE   = 0.90


class ECIKeywordRecognizer(EntityRecognizer):
    """
    Presidio EntityRecognizer for keyword/regex-based ECI detection.

    Reads detection rules from an ECIConfig object and scans text for matches.
    Plugged into the Presidio RecognizerRegistry alongside the standard
    recognizers and GLiNERRecognizer.

    Parameters
    ----------
    eci_config : ECIConfig instance loaded from eci_config.json
    """

    def __init__(self, eci_config: ECIConfig):
        self.eci_config = eci_config

        # Collect all entity types from enabled categories
        supported_entities = eci_config.all_entity_types

        # Pre-compile all patterns at init time so we don't recompile per request
        # Structure: { entity_type: [(compiled_pattern, score), ...] }
        self._compiled: dict = {}
        for cat in eci_config.enabled_categories.values():
            patterns = []

            # Compile keyword rules as word-boundary regex
            for rule in cat.keywords:
                # Escape the keyword for safe regex use, then wrap in word boundaries
                # Use \b for single-word terms, but for multi-word phrases we need
                # to match the full phrase as a unit
                escaped = re.escape(rule.value)
                # Multi-word phrases: word boundaries on first and last word
                pattern = re.compile(
                    r'\b' + escaped + r'\b',
                    re.IGNORECASE
                )
                patterns.append((pattern, _KEYWORD_SCORE, rule.description))

            # Compile regex rules directly
            for rule in cat.patterns:
                if rule.is_regex:
                    try:
                        pattern = re.compile(rule.value, re.IGNORECASE)
                        patterns.append((pattern, _REGEX_SCORE, rule.description))
                    except re.error as e:
                        print(
                            f"[ECIKeywordRecognizer] Invalid regex in category "
                            f"'{cat.entity_type}': {rule.value!r} — {e}"
                        )

            self._compiled[cat.entity_type] = patterns

        super().__init__(
            supported_entities=supported_entities,
            name="ECIKeywordRecognizer",
        )

    def load(self) -> None:
        """No model loading required — patterns are compiled in __init__."""
        pass

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        """
        Scan the text for all keyword/regex matches across all enabled ECI categories.

        Parameters
        ----------
        text     : The full input text to scan
        entities : Optional filter — only scan for these entity types.
                   If None, scan all enabled categories.

        Returns
        -------
        List of RecognizerResult, one per match found.
        """
        results = []
        target_entities = set(entities) if entities else set(self._compiled.keys())

        for entity_type, patterns in self._compiled.items():
            if entity_type not in target_entities:
                continue

            for compiled_pattern, score, description in patterns:
                for match in compiled_pattern.finditer(text):
                    start, end = match.start(), match.end()

                    # Skip empty matches (shouldn't happen but defensive)
                    if start == end:
                        continue

                    results.append(RecognizerResult(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        score=score,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                            "match_description": description,
                        },
                    ))

        return results