import re
from datetime import datetime

# =============================================================================
# RISK MANAGEMENT CONFIGURATION
# =============================================================================

PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
# DATE_PATTERN = re.compile(r"\d{2}[/\-\.]\d{2}[/\-\.]\d{4}")
DATE_PATTERN = re.compile(r"\d{2}[/\-\.]\d{2}[/\-\.]\d{4}")

# Add a fuzzy pattern for OCR-merged dates like "20104/1964" or "2041964"
DATE_PATTERN_FUZZY = re.compile(r"\d{1,2}[/\-\.]?\d{1,2}[/\-\.]\d{4}")

# Position 4 of PAN must be one of these valid taxpayer entity codes
VALID_FOURTH_CHARS = set("PCHFATBLJG")

# Known fraudulent testing hashes
KNOWN_FAKE_PANS = {
    "ABCDE1234F",
    "AAAAA0000A",
    "AAAAA1111A",
    "PANNO0000P",
    "XXXXX0000X",
    "TESTPAN123T",
}

# Core structural words used to bypass name-matching
TEMPLATE_BOILERPLATE = {
    "INCOME",
    "TAX",
    "DEPARTMENT",
    "GOVT",
    "INDIA",
    "GOVERNMENT",
    "ACCOUNT",
    "NUMBER",
    "CARD",
    "MINISTRY",
    "PERMANENT",
    "NAME",
    "FATHER",
    "FATHERS",
    "DATE",
    "BIRTH",
    "SIGNATURE",
}

# =============================================================================
# CORE VALIDATION ENGINES
# =============================================================================


def normalize_pan(word: str) -> str:
    """
    Safely corrects common OCR alphanumeric confusions on PAN strings.
    Strictly applies fixes ONLY to the expected character-type positions.
    """
    w = word.strip().upper()
    if len(w) != 10:
        return w

    chars = list(w)

    # Positions 0-4 MUST be letters
    for i in range(5):
        if chars[i] == "0":
            chars[i] = "O"
        if chars[i] == "1":
            chars[i] = "I"
        if chars[i] == "8":
            chars[i] = "B"
        if chars[i] == "5":
            chars[i] = "S"

    # Position 9 MUST be a letter
    if chars[9] == "0":
        chars[9] = "O"
    if chars[9] == "1":
        chars[9] = "I"

    # Positions 5-8 MUST be digits
    for i in range(5, 9):
        if chars[i] == "O":
            chars[i] = "0"
        if chars[i] == "I":
            chars[i] = "1"
        if chars[i] == "B":
            chars[i] = "8"
        if chars[i] == "S":
            chars[i] = "5"
        if chars[i] == "G":
            chars[i] = "6"

    return "".join(chars)


def validate_pan_structure(pan: str) -> list:
    """Deterministic structural verification of the PAN String."""
    issues = []

    if len(pan) != 10:
        issues.append("invalid_pan_length")
        return issues

    if not pan[:3].isalpha():
        issues.append("first_3_chars_not_alpha")

    if pan[3] not in VALID_FOURTH_CHARS:
        issues.append(f"invalid_taxpayer_entity_code:{pan[3]}")

    if not pan[4].isalpha():
        issues.append("5th_char_surname_initial_not_alpha")

    if not pan[5:9].isdigit():
        issues.append("sequential_digits_invalid")

    if not pan[9].isalpha():
        issues.append("last_char_checksum_not_alpha")

    return issues


def validate_date(date_str: str) -> list:
    """
    Strict calendar logic. Evaluates real-world existence of the date.
    Traps impossible parameters (e.g. 42/13/2029).
    """
    issues = []
    try:
        # Standardize separators
        normalized = date_str.replace("-", "/").replace(".", "/")
        parts = normalized.split("/")

        if len(parts) != 3:
            issues.append("malformed_date_string")
            return issues

        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        current_year = datetime.now().year  # Max year ceiling

        # 1. Native Python Calendar Validation (traps 31st Feb, 13th month, 42nd day, etc)
        datetime(year, month, day)

        # 2. PAN Business Logic Constraints
        if year > current_year:
            issues.append(f"future_birth_year:{year}")
        if year < 1900:
            issues.append(f"impossible_historical_birth_year:{year}")

    except ValueError:
        # datetime() throws ValueError on dates like 42/13/1990
        issues.append("impossible_calendar_date_value")

    return issues


def verify_pan_surname_crosscheck(pan: str, words: list) -> bool:
    """
    Verifies that the 5th character of the PAN (Surname Initial)
    exists as the starting letter of at least one valid personal name on the card.
    """
    if len(pan) != 10 or not pan[4].isalpha():
        return False

    surname_initial = pan[4].upper()

    for word in words:
        w_clean = word.strip().upper()
        # Clean out numbers, punctuation, and structural boilerplate
        if not w_clean.isalpha() or len(w_clean) < 3:
            continue
        if w_clean in TEMPLATE_BOILERPLATE:
            continue

        # If we find a valid identity string starting with the 5th character, it passes.
        if w_clean.startswith(surname_initial):
            return True

    return False


# =============================================================================
# DATA EXTRACTION & PIPELINE ORCHESTRATION
# =============================================================================


def extract_fields(ocr_entry: dict) -> dict:
    """Isolates core variables from raw OCR arrays."""
    words = ocr_entry.get("words", [])
    scores = ocr_entry.get("scores", [])

    # 1. PAN Extraction (Strict vs Fuzzy)
    strict_pans = [
        w.strip().upper() for w in words if PAN_PATTERN.match(w.strip().upper())
    ]
    fuzzy_pans = []

    for w in words:
        normalized = normalize_pan(w)
        if PAN_PATTERN.match(normalized) and normalized not in strict_pans:
            fuzzy_pans.append(normalized)

    pan_numbers = strict_pans if strict_pans else fuzzy_pans
    pan_needed_normalization = len(strict_pans) == 0 and len(fuzzy_pans) > 0

    # 2. Date Extraction
    dates = [w.strip() for w in words if DATE_PATTERN.match(w.strip())]

    # 3. Quality Tracking
    avg_confidence = sum(scores) / len(scores) if scores else 0.0
    full_text = " ".join(words).upper()
    return {
        "pan_numbers": pan_numbers,
        "pan_needed_normalization": pan_needed_normalization,
        "dates": dates,
        "avg_confidence": round(avg_confidence, 3),
        "full_text": full_text,
        "words": words,
        "image_quality": "poor" if avg_confidence < 0.75 else "acceptable",
    }


def validate(fields: dict) -> dict:
    """Applies definitive risk management constraints."""
    violations = []
    score = 0

    full_text = fields.get("full_text", "")
    hard_fail_reasons = []

    # Rule 1: Mandatory Card Origin Keywords
    # Done fuzzily to prevent OCR line-breaks from creating false flags
    has_issuer = ("INCOME" in full_text and "TAX" in full_text) or (
        "GOVT" in full_text or "GOVERNMENT" in full_text
    )
    has_card_type = "PERMANENT" in full_text and "ACCOUNT" in full_text

    if not has_issuer or not has_card_type:
        violations.append("missing_critical_issuer_keywords")
        score += 35

    # Rule 2: PAN Existence and OCR Quality
    if not fields["pan_numbers"]:
        violations.append("missing_pan_number_block")
        hard_fail_reasons.append("missing_pan_number_block")
    else:
        if fields.get("pan_needed_normalization"):
            violations.append("pan_recovered_via_normalization")
            score += 15  # Minor penalty for bad OCR clarity

        # Rule 3: Multiple PANs detected
        if len(set(fields["pan_numbers"])) > 1:
            violations.append("multiple_conflicting_pans_detected")
            score += 50

        # Rule 4: Structural Validation of the primary PAN
        primary_pan = fields["pan_numbers"][0]
        if primary_pan in KNOWN_FAKE_PANS:
            violations.append("known_testing_pan_hash")
            score += 100
        else:
            pan_issues = validate_pan_structure(primary_pan)
            if pan_issues:
                violations.extend(pan_issues)
                score += 40
            else:
                # Rule 5: 5th Character Surname Cross-Check
                if not verify_pan_surname_crosscheck(primary_pan, fields["words"]):
                    violations.append("pan_5th_character_surname_mismatch")
                    score += 35

    # Rule 6: Date of Birth Logic
    if not fields["dates"]:
        violations.append("missing_date_of_birth")
        hard_fail_reasons.append("missing_date_of_birth")
    else:
        date_issues = validate_date(fields["dates"][0])
        if date_issues:
            violations.extend(date_issues)
            score += 50

    if hard_fail_reasons:
        return {
            "violations": violations,
            "violation_count": len(violations),
            "risk_score": 100,
            "verdict": "SUSPICIOUS",
            "image_quality": fields["image_quality"],
            "hard_fail": True,
            "hard_fail_reasons": hard_fail_reasons,
        }

    return {
        "violations": violations,
        "violation_count": len(violations),
        "risk_score": min(score, 100),
        "verdict": "SUSPICIOUS" if score >= 30 else "CLEAN",
        "image_quality": fields["image_quality"],
        "hard_fail": False,
        "hard_fail_reasons": [],
    }


def assess(ocr_entry: dict) -> dict:
    """Main execution point called by the orchestrator."""
    fields = extract_fields(ocr_entry)
    result = validate(fields)
    return {
        "doc_id": ocr_entry.get("doc_id"),
        "fields": fields,
        "validation": result,
    }
