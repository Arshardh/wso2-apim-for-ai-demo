"""
RECAP Framework — ECI Entity Type Constants
============================================
Defines the ECI entity type strings and their token label mappings.
These are merged into the main ENTITY_TO_TOKEN_NAME dict in constants.py
and are used by the tokenizer (Phase 4) to generate gateway-compatible tokens.

Entity Type Naming Convention
------------------------------
All ECI entity types are prefixed with "ECI_" to clearly distinguish them
from standard Presidio PII entity types in logs, audit trails, and API responses.

Token Label Convention
----------------------
Token labels are SHORT_CAPS strings that appear in the synapse_memory_map:
  "ECI_HR_DATA"      → [HR_DATA_1], [HR_DATA_2], ...
  "ECI_PROJECT"      → [PROJECT_1], [PROJECT_2], ...
  "ECI_ORG_INFO"     → [ORG_INFO_1], ...
  "ECI_SYSTEM_INFO"  → [SYSTEM_INFO_1], ...

These four categories map to the four customer requirements:
  - Employee HR data (leave, sick days, performance, salary)
  - Internal project names, codenames, roadmaps
  - Org structure (who reports to whom, team names)
  - Internal system names, server names, credentials
"""

# ---------------------------------------------------------------------------
# ECI Entity Type strings — used as entity_type in RecognizerResult
# and in the eci_config.json "entity_type" field
# ---------------------------------------------------------------------------

ECI_HR_DATA     = "ECI_HR_DATA"       # Employee leave, salary, performance, sick days
ECI_PROJECT     = "ECI_PROJECT"       # Project names, codenames, roadmap items, initiatives
ECI_ORG_INFO    = "ECI_ORG_INFO"      # Team names, reporting structure, org hierarchy
ECI_SYSTEM_INFO = "ECI_SYSTEM_INFO"   # Server names, credentials, API keys, internal systems

# Convenience set of all ECI entity types
ALL_ECI_ENTITY_TYPES = {
    ECI_HR_DATA,
    ECI_PROJECT,
    ECI_ORG_INFO,
    ECI_SYSTEM_INFO,
}

# ---------------------------------------------------------------------------
# ECI Token Label Mapping
# Merged into ENTITY_TO_TOKEN_NAME in constants.py
# ---------------------------------------------------------------------------

ECI_ENTITY_TO_TOKEN_NAME = {
    ECI_HR_DATA:     "HR_DATA",
    ECI_PROJECT:     "PROJECT",
    ECI_ORG_INFO:    "ORG_INFO",
    ECI_SYSTEM_INFO: "SYSTEM_INFO",
}

# ---------------------------------------------------------------------------
# ECI Risky Entity Types
# ECI detections from the SLM layer carry model confidence scores and should
# pass through Phase 3b false positive filtering if their score is borderline.
# Keyword/regex matches (score 0.85-0.90) are deterministic and don't need it.
# We mark ECI_HR_DATA as risky because contextual HR detection (e.g. "out for
# a few weeks") can sometimes be ambiguous.
# ---------------------------------------------------------------------------

ECI_RISKY_ENTITY_TYPES = {
    ECI_HR_DATA,    # "John is recovering" might not always mean leave
}

# ---------------------------------------------------------------------------
# Default ECI Thresholds
# Applied by get_entity_threshold() in GuardrailsPII
# ---------------------------------------------------------------------------

ECI_DEFAULT_THRESHOLDS = {
    ECI_HR_DATA:     0.70,   # Slightly more lenient — HR leaks are high risk
    ECI_PROJECT:     0.75,
    ECI_ORG_INFO:    0.75,
    ECI_SYSTEM_INFO: 0.80,   # Higher threshold — system names need confidence
}