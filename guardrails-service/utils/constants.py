PRESIDIO_TO_GLINER = {
    "LOCATION": ["location", "place", "address"],
    "DATE_TIME": ["date", "time", "date of birth"],
    "PERSON": ["person", "name"],
    "PHONE_NUMBER": [
        "phone number",
    ],
}

GLINER_TO_PRESIDIO = {}
for presidio, entities in PRESIDIO_TO_GLINER.items():
    for entity in entities:
        GLINER_TO_PRESIDIO[entity] = presidio

DEFAULT_THRESHOLDS = {
    "LOCATION": 0.5,
    "DATE_TIME": 0.5,
    "PERSON": 0.5,
    "PHONE_NUMBER": 0.5,
    "EMAIL_ADDRESS": 1.0,
}

# ---------------------------------------------------------------------------
# RECAP Framework: Entity → Token Name Mapping
#
# Maps Presidio entity type codes to human-readable, gateway-compatible token
# names used in the synapse_memory_map.
#
# Example: "PERSON" → token becomes [NAME_1], [NAME_2], ...
#          "EMAIL_ADDRESS" → token becomes [EMAIL_1], [EMAIL_2], ...
# ---------------------------------------------------------------------------
ENTITY_TO_TOKEN_NAME = {
    # Core contextual entities (handled by GLiNER)
    "PERSON":               "NAME",
    "LOCATION":             "ADDRESS",
    "DATE_TIME":            "DATE",
    "PHONE_NUMBER":         "PHONE",

    # Structured financial / identity entities (handled by Presidio regex)
    "EMAIL_ADDRESS":        "EMAIL",
    "CREDIT_CARD":          "CREDIT_CARD",
    "IBAN_CODE":            "IBAN",
    "CRYPTO":               "CRYPTO",
    "IP_ADDRESS":           "IP",
    "URL":                  "URL",
    "DOMAIN_NAME":          "DOMAIN",

    # US-specific identifiers
    "US_SSN":               "SSN",
    "US_PASSPORT":          "PASSPORT",
    "US_DRIVER_LICENSE":    "DRIVER_LICENSE",
    "US_BANK_NUMBER":       "BANK_ACCOUNT",
    "US_ITIN":              "ITIN",

    # Medical / professional
    "MEDICAL_LICENSE":      "MEDICAL_LICENSE",
    "NRP":                  "NRP",

    # UK
    "UK_NHS":               "NHS",

    # Spain
    "ES_NIF":               "NIF",
    "ES_NIE":               "NIE",

    # Italy
    "IT_FISCAL_CODE":       "FISCAL_CODE",
    "IT_DRIVER_LICENSE":    "DRIVER_LICENSE",
    "IT_VAT_CODE":          "VAT",
    "IT_PASSPORT":          "PASSPORT",
    "IT_IDENTITY_CARD":     "ID_CARD",

    # Poland
    "PL_PESEL":             "PESEL",

    # Singapore
    "SG_NRIC_FIN":          "NRIC",
    "SG_UEN":               "UEN",

    # Australia
    "AU_ABN":               "ABN",
    "AU_ACN":               "ACN",
    "AU_TFN":               "TFN",
    "AU_MEDICARE":          "MEDICARE",

    # India
    "IN_PAN":               "PAN",
    "IN_AADHAAR":           "AADHAAR",
    "IN_VEHICLE_REGISTRATION": "VEHICLE_REG",
    "IN_VOTER":             "VOTER_ID",
    "IN_PASSPORT":          "PASSPORT",

    # Finland
    "FI_PERSONAL_IDENTITY_CODE": "PERSONAL_ID",
}

# ---------------------------------------------------------------------------
# RECAP Framework: Risky Entity Types for Phase 3b False Positive Filtering
#
# These entity types are prone to false positives from regex detection and
# should be passed through the LLM context checker before being committed
# to the synapse_memory_map.
#
# Rationale per type:
#   DATE_TIME   — "12" could be a time, a quantity, a page number, etc.
#   PHONE_NUMBER — short numeric strings can look like phone numbers
#   US_SSN      — partial matches on unrelated number sequences
#   NRP         — national registration patterns can match generic numbers
#   AGE         — not a Presidio type but included if added in future
# ---------------------------------------------------------------------------
RISKY_ENTITY_TYPES = {
    "DATE_TIME",
    "PHONE_NUMBER",
    "US_SSN",
    "NRP",
}

# ---------------------------------------------------------------------------
# RECAP Framework: Structured (Regex-Primary) vs Contextual (SLM-Primary)
#
# Used in Phase 1 to route each entity type to the appropriate detection
# engine. Structured entities are reliably caught by Presidio's regex
# recognizers. Contextual entities need GLiNER / SLM understanding.
# ---------------------------------------------------------------------------
STRUCTURED_ENTITIES = {
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "CRYPTO",
    "IP_ADDRESS",
    "URL",
    "DOMAIN_NAME",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
    "US_ITIN",
    "MEDICAL_LICENSE",
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
    "NRP",
}

CONTEXTUAL_ENTITIES = {
    "PERSON",
    "LOCATION",
    "DATE_TIME",
    "PHONE_NUMBER",
}

# ---------------------------------------------------------------------------
# Merge ECI entity token names into the master map
# Imported here to keep ENTITY_TO_TOKEN_NAME as the single source of truth
# for the tokenizer (Phase 4), regardless of whether the entity is standard
# PII or enterprise confidential.
# ---------------------------------------------------------------------------
from utils.eci_constants import ECI_ENTITY_TO_TOKEN_NAME, ECI_DEFAULT_THRESHOLDS  # noqa: E402

ENTITY_TO_TOKEN_NAME.update(ECI_ENTITY_TO_TOKEN_NAME)
DEFAULT_THRESHOLDS.update(ECI_DEFAULT_THRESHOLDS)