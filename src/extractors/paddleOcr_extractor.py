from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BoundingBox = list[int]                      # [x_min, y_min, x_max, y_max] in [0, 1000]
_Polygon    = list[list[float]]              # 4 corner points from PaddleOCR


class PaddleOCRExtractor:


    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("paddleocr", {})
        rt  = config.get("runtime",   {})

        self._lang          = str(cfg.get("lang",          "en"))
        self._use_angle_cls = bool(cfg.get("use_angle_cls", True))
        self._det_db_thresh = float(cfg.get("det_db_thresh", 0.3))
        self._rec_batch_num = int(cfg.get("rec_batch_num",  6))

        device = str(rt.get("device", "auto"))
        self._use_gpu = bool(cfg.get("use_gpu", device not in ("cpu", "auto")))

        raw_cache = cfg.get("cache_dir")
        self._cache_dir: Path | None = Path(raw_cache) if raw_cache else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._ocr: Any = None
        logger.info("PaddleOCRExtractor | lang=%s | gpu=%s | cache=%s",
                    self._lang, self._use_gpu, self._cache_dir)

    @property
    def is_loaded(self) -> bool:
        return self._ocr is not None

    def load(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pip install paddleocr paddlepaddle") from exc
        logger.info("Loading PaddleOCR (lang=%s, gpu=%s)…", self._lang, self._use_gpu)
        self._ocr = PaddleOCR(use_angle_cls=self._use_angle_cls, lang=self._lang,
                              use_gpu=self._use_gpu, det_db_thresh=self._det_db_thresh,
                              rec_batch_num=self._rec_batch_num, show_log=False)
        logger.info("PaddleOCR loaded.")

    def extract(self, image: Image.Image) -> dict[str, list]:

        if not self.is_loaded:
            raise RuntimeError("Call load() before extract().")
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL.Image.Image, got {type(image)}")

        image = image.convert("RGB")

        if self._cache_dir:
            cached = self._read_cache(image)
            if cached:
                logger.debug("OCR cache hit.")
                return cached

        img_w, img_h = image.size
        words, boxes, scores = [], [], []

        raw = self._ocr.ocr(np.array(image), cls=self._use_angle_cls)
        if raw and raw[0] is not None:
            for polygon, (text, score) in ((l[0], l[1]) for l in raw[0]):
                if not text.strip() or score < 0.5:
                    continue
                words.append(text)
                boxes.append(self._norm_box(polygon, img_w, img_h))
                scores.append(float(score))

        result = {"words": words, "boxes": boxes, "scores": scores}
        logger.debug("Extracted %d tokens.", len(words))

        if self._cache_dir:
            self._write_cache(image, result)
        return result

    def extract_batch(self, images: list[Image.Image]) -> list[dict[str, list]]:
        return [self.extract(img) for img in images]


    # Private helpers

    @staticmethod
    def _norm_box(polygon: _Polygon, img_w: int, img_h: int) -> BoundingBox:
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        x_min = max(0,    int(round(min(xs) / img_w * 1000)))
        y_min = max(0,    int(round(min(ys) / img_h * 1000)))
        x_max = min(1000, int(round(max(xs) / img_w * 1000)))
        y_max = min(1000, int(round(max(ys) / img_h * 1000)))
        # Clamp degenerate boxes
        x_max = max(x_max, x_min + 1)
        y_max = max(y_max, y_min + 1)
        return [x_min, y_min, min(x_max, 1000), min(y_max, 1000)]

    @staticmethod
    def _img_hash(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def _cache_path(self, image: Image.Image) -> Path:
        return self._cache_dir / f"{self._img_hash(image)}.json"  # type: ignore[operator]

    def _read_cache(self, image: Image.Image) -> dict[str, list] | None:
        p = self._cache_path(image)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _write_cache(self, image: Image.Image, result: dict[str, list]) -> None:
        self._cache_path(image).write_text(json.dumps(result), encoding="utf-8")
