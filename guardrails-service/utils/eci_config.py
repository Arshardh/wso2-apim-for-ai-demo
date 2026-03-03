"""
Enterprise Confidential Information (ECI) Configuration
=========================================================
This module defines the schema and loader for the customer-specific
Enterprise Confidential Information (ECI) configuration.

Unlike standard PII (names, emails, SSNs) which are universally defined,
ECI is company-specific. "Project Phoenix", "db-prod-01", and
"Sarah reports to the CTO" mean nothing to GDPR regulators, but they are
highly sensitive to the company that owns them.

The customer populates config/eci_config.json with their specific:
  - HR keywords and patterns (leave types, role titles, salary markers)
  - Project names, codenames, and roadmap terms
  - Org structure terms (team names, reporting language, department names)
  - System names, server patterns, credential markers

Config File Location
---------------------
Default: config/eci_config.json (relative to the server working directory)
Override: Set the ECI_CONFIG_PATH environment variable

Config Schema
-------------
See ECIConfig and its nested models below for the full schema.
See config/eci_config.json for a ready-to-edit example.

How Detection Works
-------------------
Each ECI category has two detection layers:

1. keywords: Exact/case-insensitive word or phrase matches.
   Fast, deterministic, zero false positives for known terms.
   Best for: known project names, server names, specific HR phrases.

2. patterns: Regex patterns for structured internal formats.
   Best for: server naming conventions (db-prod-\\d+), ticket IDs,
   credential patterns (API_KEY=...), salary ranges.

The SLM contextual layer (Phase 1d) then catches anything the
keyword/regex layer missed by reading the text holistically.

Sensitivity Levels
------------------
Each category has a sensitivity_level (1-3):
  1 = Confidential   — mask and tokenize, include in audit log
  2 = Internal Only  — mask and tokenize, flag for review
  3 = Restricted     — mask and tokenize, alert security team

All levels result in the same tokenization behavior — the level is
metadata for downstream audit/alerting systems.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern


# ---------------------------------------------------------------------------
# Schema: Individual detection rule
# ---------------------------------------------------------------------------

@dataclass
class ECIDetectionRule:
    """
    A single keyword or regex pattern for detecting internal information.

    Attributes
    ----------
    value       : The keyword string or regex pattern string
    is_regex    : True if value is a regex pattern, False for plain keyword
    description : Human-readable description of what this rule detects
    """
    value: str
    is_regex: bool = False
    description: str = ""

    def compile(self) -> Optional[Pattern]:
        """Compile and return the regex pattern, or None if plain keyword."""
        if self.is_regex:
            return re.compile(self.value, re.IGNORECASE)
        return None


# ---------------------------------------------------------------------------
# Schema: ECI Category
# ---------------------------------------------------------------------------

@dataclass
class ECICategory:
    """
    One category of enterprise confidential information.

    Attributes
    ----------
    entity_type     : Presidio-style entity type string used in tokens.
                      e.g. "ECI_PROJECT", "ECI_HR_DATA"
    token_label     : Short label used in replacement tokens.
                      e.g. "PROJECT" → [PROJECT_1]
    description     : Human-readable description of this category
    sensitivity_level : 1 (Confidential), 2 (Internal Only), 3 (Restricted)
    keywords        : List of plain-text keyword/phrase rules
    patterns        : List of regex-based detection rules
    slm_description : Natural language description passed to the SLM so it
                      knows what to look for in this category.
                      e.g. "project names, codenames, internal initiative titles"
    enabled         : If False, this category is skipped entirely
    """
    entity_type: str
    token_label: str
    description: str
    sensitivity_level: int = 1
    keywords: List[ECIDetectionRule] = field(default_factory=list)
    patterns: List[ECIDetectionRule] = field(default_factory=list)
    slm_description: str = ""
    enabled: bool = True

    def all_rules(self) -> List[ECIDetectionRule]:
        """Return all keyword and pattern rules combined."""
        return self.keywords + self.patterns


# ---------------------------------------------------------------------------
# Schema: Root ECI Config
# ---------------------------------------------------------------------------

@dataclass
class ECIConfig:
    """
    Root configuration object for Enterprise Confidential Information detection.

    Attributes
    ----------
    company_name    : Used in SLM prompts for context
    categories      : Dict of category_key → ECICategory
    slm_enabled     : If True, Phase 1d SLM contextual detection runs after
                      the keyword/regex layer
    slm_model       : HuggingFace model for SLM contextual detection.
                      Shares the FalsePositiveFilter model by default.
    confidence_threshold : Minimum score for SLM detections (0.0–1.0)
    """
    company_name: str = "the organization"
    categories: Dict[str, ECICategory] = field(default_factory=dict)
    slm_enabled: bool = True
    slm_model: str = "microsoft/Phi-3.5-mini-instruct"
    confidence_threshold: float = 0.7

    @property
    def enabled_categories(self) -> Dict[str, ECICategory]:
        """Return only enabled categories."""
        return {k: v for k, v in self.categories.items() if v.enabled}

    @property
    def all_entity_types(self) -> List[str]:
        """Return entity type strings for all enabled categories."""
        return [cat.entity_type for cat in self.enabled_categories.values()]


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _parse_rules(rules_data: List[dict]) -> List[ECIDetectionRule]:
    """Parse a list of rule dicts into ECIDetectionRule objects."""
    rules = []
    for r in rules_data:
        rules.append(ECIDetectionRule(
            value=r["value"],
            is_regex=r.get("is_regex", False),
            description=r.get("description", ""),
        ))
    return rules


def load_eci_config(config_path: Optional[str] = None) -> ECIConfig:
    """
    Load the ECI configuration from a JSON file.

    Parameters
    ----------
    config_path : Path to the JSON config file.
                  Defaults to the ECI_CONFIG_PATH env var, then
                  config/eci_config.json in the working directory.

    Returns
    -------
    ECIConfig instance. Returns an empty config (no categories, SLM disabled)
    if the file doesn't exist, so the system degrades gracefully.
    """
    if config_path is None:
        config_path = os.getenv(
            "ECI_CONFIG_PATH",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "eci_config.json")
        )

    if not os.path.exists(config_path):
        print(f"[ECIConfig] No config file found at {config_path}. ECI detection disabled.")
        return ECIConfig(slm_enabled=False)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    categories = {}
    for key, cat_data in raw.get("categories", {}).items():
        categories[key] = ECICategory(
            entity_type=cat_data["entity_type"],
            token_label=cat_data["token_label"],
            description=cat_data.get("description", ""),
            sensitivity_level=cat_data.get("sensitivity_level", 1),
            keywords=_parse_rules(cat_data.get("keywords", [])),
            patterns=_parse_rules(cat_data.get("patterns", [])),
            slm_description=cat_data.get("slm_description", ""),
            enabled=cat_data.get("enabled", True),
        )

    config = ECIConfig(
        company_name=raw.get("company_name", "the organization"),
        categories=categories,
        slm_enabled=raw.get("slm_enabled", True),
        slm_model=raw.get("slm_model", "microsoft/Phi-3.5-mini-instruct"),
        confidence_threshold=raw.get("confidence_threshold", 0.7),
    )

    enabled_count = len(config.enabled_categories)
    print(
        f"[ECIConfig] Loaded config for '{config.company_name}': "
        f"{enabled_count} ECI categories enabled."
    )
    return config