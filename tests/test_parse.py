import sys

sys.path.insert(0, ".")
from src.validators.field_validator import assess

SAMPLE = {
    "doc_id": "PAN_148",
    "words": [
        "BTRCY",
        "farsT",
        "HRRR",
        "INCOME TAX DEPARTMENT",
        "GOVT.OF INDIA",
        "Permanent Account Number Card",
        "BJUPC0498Q",
        "/Name",
        "CHIRAG CHATTAR",
        "faT Ta/Father's Name",
        "MUKESHCHATTAR",
        "20012017",
        "  a/Date of Birth",
        "Onirag",
        "16/10/1997",
        "ar&r/Signature",
    ],
    "scores": [
        0.601874589920044,
        0.5255069136619568,
        0.8316422700881958,
        0.9497025012969971,
        0.9671987295150767,
        0.9629560112953186,
        0.9699205160140991,
        0.9817250370979309,
        0.9596745371818542,
        0.82770836353302,
        0.9936404824256897,
        0.9978076219558716,
        0.7837382555007935,
        0.5264487862586975,
        0.9970760345458984,
        0.8066743016242981,
    ],
}


def test_assess_extracts_document_id_and_validation() -> None:
    result = assess(SAMPLE)

    assert result["doc_id"] == "PAN_148"
    assert "validation" in result
    assert isinstance(result["validation"]["risk_score"], (int, float))
