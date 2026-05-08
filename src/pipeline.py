from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.ingestion import DocumentIngester
from src.data.preprocessing import DocumentPreprocessor
from src.extractors.paddleOcr_extractor import PaddleOCRExtractor
from src.models.layoutlm_classifier import LayoutLMv3Classifier
from src.validators.rules_engine import RulesEngine

logger = logging.getLogger(__name__)


class DocumentVerificationPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg            = config
        self._thresholds     = config["risk_thresholds"]
        self._weights        = config["ensemble_weights"]
        self._output_cfg     = config.get("output", {})

        self._ingester       = DocumentIngester(config)
        self._preprocessor   = DocumentPreprocessor(config)
        self._ocr            = PaddleOCRExtractor(config)
        self._classifier     = LayoutLMv3Classifier(config)
        self._rules          = RulesEngine(config)

        self._audit_path     = Path(self._output_cfg.get("audit_log_path", "logs/audit.jsonl"))
        self._results_path   = Path(self._output_cfg.get("results_dir", "results"))

        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._results_path.mkdir(parents=True, exist_ok=True)

        logger.info("DocumentVerificationPipeline initialised.")

    def load(self) -> None:
        logger.info("Loading OCR engine…")
        self._ocr.load()
        logger.info("Loading LayoutLMv3 classifier…")
        self._classifier.load()
        logger.info("Pipeline ready.")

    def run(self, file_path: str | Path, doc_id: str | None = None) -> dict[str, Any]:
        doc_id    = doc_id or str(uuid.uuid4())[:8]
        file_path = str(file_path)
        started   = time.perf_counter()

        result: dict[str, Any] = {
            "doc_id":          doc_id,
            "file_path":       file_path,
            "document_type":   "UNKNOWN",
            "type_confidence": 0.0,
            "layout_score":    0.0,
            "rules_score":     0.0,
            "risk_score":      0.0,
            "verdict":         "REJECTED",
            "is_fake":         True,
            "rules_detail":    {},
            "ocr_word_count":  0,
            "pages_processed": 0,
            "error":           None,
        }

        try:
            result = self._run_stages(result, file_path, doc_id)
        except Exception as exc:
            logger.error("Pipeline error on %s: %s", doc_id, exc, exc_info=True)
            result["error"] = str(exc)

        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["timestamp"]  = datetime.now(timezone.utc).isoformat()

        self._write_audit(result)
        self._write_csv_row(result)

        logger.info(
            "[%s] %-16s | type=%-16s conf=%.2f | risk=%.2f | %s",
            doc_id, Path(file_path).name,
            result["document_type"], result["type_confidence"],
            result["risk_score"], result["verdict"],
        )
        return result

    def run_batch(self, file_paths: list[str | Path]) -> list[dict[str, Any]]:
        """Run :meth:`run` on a list of files and return all results."""
        return [self.run(fp) for fp in file_paths]


    # Pipeline stages


    def _run_stages(
        self, result: dict[str, Any], file_path: str, doc_id: str
    ) -> dict[str, Any]:

        # Stage 1: Ingest
        pages = self._ingester.load(file_path)
        result["pages_processed"] = len(pages)

        # Stage 2: Preprocess (no augmentation at inference)
        image: Image.Image = self._preprocessor.process(pages[0], augment=False)

        # Stage 3: OCR
        ocr = self._ocr.extract(image)
        result["ocr_word_count"] = len(ocr["words"])

        # Stage 4: Classify
        clf = self._classifier.inference(
            image=image,
            words=ocr["words"],
            boxes=ocr["boxes"],
        )
        result["document_type"]   = clf["label"]
        result["type_confidence"] = clf["confidence"]
        result["layout_score"]    = clf["confidence"]

        # Stage 5: Rules validation
        rules_result = self._rules.validate(
            document_type=clf["label"],
            words=ocr["words"],
            boxes=ocr["boxes"],
        )
        result["rules_score"]  = rules_result["score"]
        result["rules_detail"] = rules_result["detail"]

        # Stage 6: Ensemble + verdict
        result["risk_score"] = self._ensemble(
            layout_score=result["layout_score"],
            cnn_score=1.0,              # neutral until cnn_tampering.py is built
            rules_score=result["rules_score"],
        )
        result["verdict"] = self._verdict(result["risk_score"])
        result["is_fake"] = result["verdict"] == "REJECTED"

        return result

    def _ensemble(
        self, layout_score: float, cnn_score: float, rules_score: float
    ) -> float:
        w = self._weights
        raw = (
            layout_score * w["layoutlmv3_classification"]
            + cnn_score  * w["cnn_tampering_detection"]
            + rules_score * w["regex_rules_engine"]
        )
        return round(raw, 4)

    def _verdict(self, risk_score: float) -> str:
        t = self._thresholds
        if risk_score >= t["auto_approve"]:
            return "APPROVED"
        if risk_score >= t["manual_review"]:
            return "MANUAL_REVIEW"
        return "REJECTED"

    def _write_audit(self, result: dict[str, Any]) -> None:
        with self._audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")

    def _write_csv_row(self, result: dict[str, Any]) -> None:
        csv_path = self._results_path / "results.csv"
        write_header = not csv_path.exists()
        fields = [
            "timestamp", "doc_id", "file_path", "document_type",
            "type_confidence", "risk_score", "verdict", "is_fake",
            "rules_score", "ocr_word_count", "pages_processed",
            "elapsed_ms", "error",
        ]
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            if write_header:
                fh.write(",".join(fields) + "\n")
            fh.write(",".join(str(result.get(f, "")) for f in fields) + "\n")