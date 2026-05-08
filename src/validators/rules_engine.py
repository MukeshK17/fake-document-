from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Verhoeff multiplication and permutation tables
_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0],
]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8],
]
_VERHOEFF_INV = [0,4,3,2,1,9,8,7,6,5]


def _verhoeff_check(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


class RulesEngine:

    def __init__(self, config: dict[str, Any]) -> None:
        v = config.get("validators", {})
        self._pan_regex    = re.compile(v.get("pan_card",  {}).get("pan_regex",  r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$"))
        self._name_min_len = int(v.get("pan_card",  {}).get("name_min_length", 3))
        self._aadhaar_digits = int(v.get("aadhar_card", {}).get("digit_count", 12))
        self._bal_tolerance  = float(v.get("bank_statement", {}).get("balance_tolerance_inr", 1.0))

        # Dispatch table: doc_type → validator method
        self._dispatch: dict[str, Any] = {
            "PAN_CARD":        self._validate_pan,
            "AADHAR_CARD":     self._validate_aadhaar,
            "BANK_STATEMENT":  self._validate_bank,
            "SALARY_SLIP":     self._validate_salary,
            "ITR_FORM":        self._validate_itr,
        }
        logger.info("RulesEngine ready.")

    def validate(
        self,
        document_type: str,
        words: list[str],
        boxes: list[list[int]],
    ) -> dict[str, Any]:
        fn = self._dispatch.get(document_type)
        if fn is None:
            logger.debug("No rules for document_type='%s'; score=0.0", document_type)
            return {"score": 0.0, "detail": {"no_rules_for_type": False}}

        text = " ".join(words)
        detail: dict[str, bool] = fn(text, words, boxes)
        passed = sum(detail.values())
        score  = round(passed / len(detail), 4) if detail else 0.0
        logger.debug("Rules [%s] → %d/%d passed | score=%.2f", document_type, passed, len(detail), score)
        return {"score": score, "detail": detail}


    # Per-type validators

    def _validate_pan(self, text: str, words: list[str], _boxes: list) -> dict[str, bool]:
        pan_match  = bool(re.search(self._pan_regex, text))
        has_name   = any(len(w) >= self._name_min_len and w.isalpha() and w.isupper() for w in words)
        has_dob    = bool(re.search(r"\d{2}/\d{2}/\d{4}", text))
        has_father = bool(re.search(r"FATHER|S/O|C/O", text, re.IGNORECASE))
        return {
            "pan_format_valid": pan_match,
            "name_present":     has_name,
            "dob_present":      has_dob,
            "father_name_hint": has_father,
        }

    def _validate_aadhaar(self, text: str, _words: list, _boxes: list) -> dict[str, bool]:
        digits = re.findall(r"\b\d{4}\b", text)          # Aadhaar shown as XXXX XXXX XXXX
        joined = "".join(digits)
        has_12 = len(joined) == self._aadhaar_digits
        checksum_ok = _verhoeff_check(joined) if has_12 else False
        has_uid_label = bool(re.search(r"unique\s+identification|aadhaar|आधार", text, re.IGNORECASE))
        has_dob  = bool(re.search(r"DOB|Year of Birth|\d{2}/\d{2}/\d{4}", text, re.IGNORECASE))
        return {
            "twelve_digit_present": has_12,
            "verhoeff_checksum":    checksum_ok,
            "uid_label_present":    has_uid_label,
            "dob_present":          has_dob,
        }

    def _validate_bank(self, text: str, _words: list, _boxes: list) -> dict[str, bool]:
        has_account = bool(re.search(r"account\s*(no|number|#)?\.?\s*:?\s*\d{9,18}", text, re.IGNORECASE))
        has_ifsc    = bool(re.search(r"[A-Z]{4}0[A-Z0-9]{6}", text))
        has_balance = bool(re.search(r"balance|closing|opening", text, re.IGNORECASE))
        has_dates   = bool(re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text))
        return {
            "account_number_present": has_account,
            "ifsc_code_present":      has_ifsc,
            "balance_field_present":  has_balance,
            "dates_present":          has_dates,
        }

    def _validate_salary(self, text: str, _words: list, _boxes: list) -> dict[str, bool]:
        has_employer = bool(re.search(r"employer|company|organisation|pvt|ltd|llp", text, re.IGNORECASE))
        has_employee = bool(re.search(r"employee|name\s*:", text, re.IGNORECASE))
        has_net_pay  = bool(re.search(r"net\s*pay|net\s*salary|take\s*home", text, re.IGNORECASE))
        has_month    = bool(re.search(r"january|february|march|april|may|june|july|august|september|october|november|december|salary\s*slip", text, re.IGNORECASE))
        return {
            "employer_present":  has_employer,
            "employee_present":  has_employee,
            "net_pay_present":   has_net_pay,
            "month_year_present": has_month,
        }

    def _validate_itr(self, text: str, _words: list, _boxes: list) -> dict[str, bool]:
        has_pan       = bool(re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text))
        has_ay        = bool(re.search(r"assessment\s+year|A\.?Y\.?\s*20\d{2}", text, re.IGNORECASE))
        has_income    = bool(re.search(r"total\s+income|gross\s+total", text, re.IGNORECASE))
        has_signature = bool(re.search(r"e-?verification|digital\s+signature|acknowledgement", text, re.IGNORECASE))
        return {
            "pan_present":         has_pan,
            "assessment_year":     has_ay,
            "income_field_present": has_income,
            "verification_present": has_signature,
        }
