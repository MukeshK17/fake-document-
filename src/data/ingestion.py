"""
Raw document ingestion — single entry-point for all file I/O.

Always returns ``list[PIL.Image.Image]`` in RGB mode (one item per page),
regardless of whether the source is a JPEG, PNG, TIFF, BMP, WEBP, PDF,
or S3 URI. Nothing downstream ever calls ``open()`` or ``boto3`` directly.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"})


class DocumentIngester:

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("ingestion", {})
        self._pdf_dpi:   int       = int(cfg.get("pdf_dpi",   200))
        self._max_pages: int | None = cfg.get("max_pages", None)
        logger.info("DocumentIngester ready | pdf_dpi=%d | max_pages=%s",
                    self._pdf_dpi, self._max_pages)

    def load(self, source: str | Path) -> list[Image.Image]:
        """Return one RGB PIL image per page from a local path or S3 URI."""
        src = str(source)
        if src.startswith("s3://"):
            return self._from_s3(src)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        if path.suffix.lower() == ".pdf":
            return self._from_pdf(path)
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            logger.debug("Loading image: %s", path)
            return [Image.open(path).convert("RGB")]
        raise ValueError(f"Unsupported type '{path.suffix}'. "
                         f"Supported: PDF + {sorted(_IMAGE_SUFFIXES)}")

    def _from_pdf(self, path: Path) -> list[Image.Image]:
        try:
            from pdf2image import convert_from_path  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install pdf2image: pip install pdf2image") from exc
        logger.debug("Rasterising PDF at %d DPI: %s", self._pdf_dpi, path)
        pages = convert_from_path(str(path), dpi=self._pdf_dpi, last_page=self._max_pages)
        pages = [p.convert("RGB") for p in pages]
        logger.info("Loaded %d page(s) from PDF: %s", len(pages), path.name)
        return pages

    def _from_s3(self, uri: str) -> list[Image.Image]:
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install boto3: pip install boto3") from exc
        without_scheme = uri[5:]  # strip "s3://"
        if "/" not in without_scheme:
            raise ValueError(f"Cannot parse S3 URI: {uri!r}")
        bucket, key = without_scheme.split("/", 1)
        logger.debug("Downloading s3://%s/%s", bucket, key)
        with tempfile.NamedTemporaryFile(suffix=Path(key).suffix, delete=False) as tmp:
            boto3.client("s3").download_fileobj(bucket, key, tmp)
            tmp_path = Path(tmp.name)
        result = self.load(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return result
