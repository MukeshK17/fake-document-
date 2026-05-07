from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image
from torch.utils.data import Dataset

from src.data.ingestion import DocumentIngester
from src.data.preprocessing import DocumentPreprocessor

logger = logging.getLogger(__name__)

BoundingBox = list[int]


class DocumentSample(NamedTuple):
    image:  Image.Image
    words:  list[str]
    boxes:  list[BoundingBox]
    label:  int
    doc_id: str


class DocumentDataset(Dataset):
    """Loads and preprocesses labelled document pages from a manifest file.

    Parameters
    ----------
    manifest_path : str | Path   CSV or JSONL manifest for this split.
    config        : dict         Top-level pipeline config.
    split         : str          ``"train"`` | ``"val"`` | ``"test"`` (controls augmentation).
    """

    def __init__(self, manifest_path: str | Path, config: dict[str, Any], split: str = "train") -> None:
        self._manifest_path = Path(manifest_path)
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self._manifest_path}")

        self._split       = split
        self._ingester    = DocumentIngester(config)
        self._preprocessor = DocumentPreprocessor(config)

        raw = config["layoutlmv3"]["id2label"]
        self.id2label: dict[int, str]  = {int(k): v for k, v in raw.items()}
        self.label2id: dict[str, int]  = {v: k for k, v in self.id2label.items()}

        self._samples   = self._load_manifest()
        self._ocr_cache = self._load_ocr_cache()

        logger.info("DocumentDataset [%s] | %d samples | labels=%s",
                    split, len(self._samples), sorted(self.label2id))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> DocumentSample:
        row       = self._samples[idx]
        doc_id    = row["doc_id"]
        label_str = row["label"]

        if label_str not in self.label2id:
            raise ValueError(f"Unknown label '{label_str}' at row {idx}. "
                             f"Known: {sorted(self.label2id)}")

        pages = self._ingester.load(row["file_path"])
        if not pages:
            raise RuntimeError(f"Zero pages loaded from: {row['file_path']}")

        image = self._preprocessor.process(pages[0], augment=self._split == "train")
        ocr   = self._ocr_cache.get(doc_id, {})

        return DocumentSample(
            image=image,
            words=ocr.get("words", []),
            boxes=ocr.get("boxes", []),
            label=self.label2id[label_str],
            doc_id=doc_id,
        )


    # Private helpers

    def _load_manifest(self) -> list[dict[str, str]]:
        suffix = self._manifest_path.suffix.lower()
        if suffix == ".csv":
            return self._parse_csv()
        if suffix == ".jsonl":
            return self._parse_jsonl()
        raise ValueError(f"Unsupported manifest format '{suffix}'. Expected .csv or .jsonl")

    def _parse_csv(self) -> list[dict[str, str]]:
        with self._manifest_path.open(newline="", encoding="utf-8") as fh:
            return [{"doc_id":    r["doc_id"].strip(),
                     "file_path": r["file_path"].strip(),
                     "label":     r["label"].strip()}
                    for r in csv.DictReader(fh)]

    def _parse_jsonl(self) -> list[dict[str, str]]:
        rows = []
        with self._manifest_path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {i} of {self._manifest_path}") from exc
                rows.append({"doc_id": str(obj["doc_id"]),
                             "file_path": str(obj["file_path"]),
                             "label": str(obj["label"])})
        return rows

    def _load_ocr_cache(self) -> dict[str, dict[str, Any]]:
        cache_path = self._manifest_path.with_name(self._manifest_path.stem + "_ocr_cache.jsonl")
        if not cache_path.exists():
            logger.debug("No OCR cache at %s; words/boxes will be empty.", cache_path)
            return {}
        cache = {}
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line := line.strip():
                    obj = json.loads(line)
                    cache[obj["doc_id"]] = {"words": obj.get("words", []),
                                            "boxes": obj.get("boxes", [])}
        logger.info("OCR cache loaded | %d entries | %s", len(cache), cache_path.name)
        return cache
