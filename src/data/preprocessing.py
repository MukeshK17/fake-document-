from __future__ import annotations

import logging
import random
from io import BytesIO
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_SIZE: tuple[int, int] = (1024, 1024)


# Module-level helper


def _fires(stage_cfg: dict[str, Any]) -> bool:
    """Return True if a stage is enabled and its probability roll succeeds."""
    return stage_cfg.get("enabled", True) and random.random() < float(
        stage_cfg.get("p", 0.5)
    )


# DocumentPreprocessor


class DocumentPreprocessor:
    def __init__(self, config: dict[str, Any]) -> None:
        pre_cfg: dict[str, Any] = config.get("preprocessing", {})
        raw_size = pre_cfg.get("target_size", list(_DEFAULT_TARGET_SIZE))
        self._target_size: tuple[int, int] = (int(raw_size[0]), int(raw_size[1]))

        _all_clean: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
            ("correct_orientation", self._apply_orientation_correction),
            ("deskew", self._apply_deskew),
            ("denoise", self._apply_denoise),
            ("normalise_contrast", self._apply_clahe),
        ]
        self._clean_stages = [fn for key, fn in _all_clean if pre_cfg.get(key, True)]

        aug_cfg: dict[str, Any] = pre_cfg.get("augmentation", {})
        self._aug_enabled: bool = bool(aug_cfg.get("enabled", True))
        self._aug_cfg = aug_cfg
        self._aug_dispatch: list[
            tuple[str, Callable[[Image.Image, dict], Image.Image]]
        ] = [
            ("random_rotation", self._aug_random_rotation),
            ("random_perspective", self._aug_random_perspective),
            ("random_brightness", self._aug_random_brightness),
            ("random_contrast", self._aug_random_contrast),
            ("random_gaussian_blur", self._aug_random_gaussian_blur),
            ("random_jpeg_compression", self._aug_random_jpeg_compression),
            ("random_noise", self._aug_random_noise),
            ("random_shadow", self._aug_random_shadow),
            ("random_ink_bleed", self._aug_random_ink_bleed),
            ("random_erode_text", self._aug_random_erode_text),
            ("cutout", self._aug_cutout),
        ]

        logger.info(
            "DocumentPreprocessor ready | clean_stages=%d | augmentation=%s | target_size=%s",
            len(self._clean_stages),
            self._aug_enabled,
            self._target_size,
        )

    # Public API

    def process(self, image: Image.Image, augment: bool = False) -> Image.Image:
        arr = self._pil_to_bgr(image)

        # Deterministic cleaning — iterate pre-filtered stage list (no if/else)
        for stage_fn in self._clean_stages:
            arr = stage_fn(arr)
        arr = self._resize_with_padding(arr, self._target_size)

        # Stochastic augmentation — training only
        if augment and self._aug_enabled:
            arr = self._pil_to_bgr(self._apply_augmentations(self._bgr_to_pil(arr)))

        return self._bgr_to_pil(arr)

    # Augmentation orchestrator

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Iterate the dispatch table; fire each stage if its probability rolls."""
        for key, fn in self._aug_dispatch:
            stage_cfg = self._aug_cfg.get(key, {})
            if _fires(stage_cfg):
                image = fn(image, stage_cfg)
        return image

    # Augmentation implementations

    @staticmethod
    def _aug_random_rotation(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Rotate ±max_angle(degree) — simulates crooked scanner placement."""

        max_a = float(cfg.get("max_angle", 5.0))
        return image.rotate(
            random.uniform(-max_a, max_a), expand=False, fillcolor=(255, 255, 255)
        )

    @staticmethod
    def _aug_random_perspective(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Mild projective warp — simulates off-axis phone capture."""

        scale = float(cfg.get("distortion_scale", 0.05))
        w, h = image.size
        arr = np.array(image, dtype=np.uint8)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.array(
            [
                [
                    random.uniform(-w * scale, w * scale),
                    random.uniform(-h * scale, h * scale),
                ]
                for _ in range(4)
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src, dst)
        return Image.fromarray(
            cv2.warpPerspective(
                arr,
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        )

    @staticmethod
    def _aug_random_brightness(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Random brightness — simulates scanner lamp drift."""

        lo, hi = cfg.get("factor_range", [0.7, 1.3])
        return ImageEnhance.Brightness(image).enhance(
            random.uniform(float(lo), float(hi))
        )

    @staticmethod
    def _aug_random_contrast(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Random contrast — simulates faded photocopy or high-contrast reprint."""

        lo, hi = cfg.get("factor_range", [0.7, 1.3])
        return ImageEnhance.Contrast(image).enhance(
            random.uniform(float(lo), float(hi))
        )

    @staticmethod
    def _aug_random_gaussian_blur(
        image: Image.Image, cfg: dict[str, Any]
    ) -> Image.Image:
        """Random Gaussian blur — simulates low-DPI / out-of-focus capture."""

        lo, hi = cfg.get("radius_range", [0.5, 2.0])
        return image.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(float(lo), float(hi)))
        )

    @staticmethod
    def _aug_random_jpeg_compression(
        image: Image.Image, cfg: dict[str, Any]
    ) -> Image.Image:
        """Random JPEG round-trip — introduces block artefacts from repeated saves."""

        lo, hi = cfg.get("quality_range", [40, 85])
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=random.randint(int(lo), int(hi)))
        buf.seek(0)
        return Image.open(buf).copy()

    @staticmethod
    def _aug_random_noise(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Gaussian pixel noise — simulates CCD sensor / phone camera grain."""

        lo, hi = cfg.get("std_range", [5.0, 25.0])
        arr = np.array(image, dtype=np.float32)
        return Image.fromarray(
            np.clip(
                arr
                + np.random.normal(
                    0.0, random.uniform(float(lo), float(hi)), arr.shape
                ),
                0,
                255,
            ).astype(np.uint8)
        )

    @staticmethod
    def _aug_random_shadow(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Vertical shadow band — simulates book-spine or hand shadow during scanning."""

        lo, hi = cfg.get("alpha_range", [0.3, 0.7])
        alpha = random.uniform(float(lo), float(hi))
        w, _ = image.size
        shadow_w = int(w * float(cfg.get("width_fraction", 0.35)))
        x0 = random.randint(0, max(0, w - shadow_w))
        arr = np.array(image, dtype=np.float32)
        arr[:, x0 : x0 + shadow_w] *= 1.0 - alpha
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @staticmethod
    def _aug_random_ink_bleed(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Morphological dilation on dark regions — simulates inkjet ink smear."""

        k = int(cfg.get("kernel_size", 2))
        kernel = np.ones((k, k), np.uint8)
        arr = np.array(image)
        return Image.fromarray(
            cv2.bitwise_not(cv2.dilate(cv2.bitwise_not(arr), kernel))
        )

    @staticmethod
    def _aug_random_erode_text(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """Morphological erosion on dark regions — simulates faded / thin ink."""

        k = int(cfg.get("kernel_size", 2))
        kernel = np.ones((k, k), np.uint8)
        arr = np.array(image)
        return Image.fromarray(cv2.bitwise_not(cv2.erode(cv2.bitwise_not(arr), kernel)))

    @staticmethod
    def _aug_cutout(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        """White-rectangle occlusion — forces classification from partial evidence."""

        arr = np.array(image)
        img_h, img_w = arr.shape[:2]
        for _ in range(int(cfg.get("num_holes", 1))):
            hole_h = random.randint(
                1, max(1, int(img_h * float(cfg.get("max_h_ratio", 0.2))))
            )
            hole_w = random.randint(
                1, max(1, int(img_w * float(cfg.get("max_w_ratio", 0.2))))
            )
            y1, x1 = (
                random.randint(0, img_h - hole_h),
                random.randint(0, img_w - hole_w),
            )
            arr[y1 : y1 + hole_h, x1 : x1 + hole_w] = 255
        return Image.fromarray(arr)

    # Deterministic cleaning stages

    @staticmethod
    def _apply_orientation_correction(bgr: np.ndarray) -> np.ndarray:
        """Correct 90/180/270° page rotation using Tesseract OSD."""
        try:
            import pytesseract  # type: ignore[import]
        except ImportError:
            logger.debug("pytesseract not installed; skipping orientation correction.")
            return bgr

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
            angle, conf = (
                int(osd.get("rotate", 0)),
                float(osd.get("orientation_conf", 0.0)),
            )
        except Exception as exc:
            logger.debug("Tesseract OSD failed (%s); skipping.", exc)
            return bgr

        rotation_map = {
            90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE,
        }
        code = rotation_map.get(angle) if conf >= 2.0 else None
        if code is not None:
            logger.debug("Correcting orientation by %d°", angle)
            return cv2.rotate(bgr, code)
        return bgr

    @staticmethod
    def _apply_deskew(bgr: np.ndarray) -> np.ndarray:
        """Correct small rotational skew (±0.5°–10°) via Hough lines."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(
            cv2.Canny(gray, 50, 150, apertureSize=3),
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )
        if lines is None:
            return bgr
        angles = [
            np.degrees(np.arctan2(y2 - y1, x2 - x1))
            for x1, y1, x2, y2 in lines[:, 0]
            if x2 != x1
        ]
        if not angles:
            return bgr
        median_angle = float(np.median(angles))
        if not (0.5 <= abs(median_angle) <= 10.0):
            return bgr
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
        logger.debug("Deskewed by %.2f°", median_angle)
        return cv2.warpAffine(
            bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    @staticmethod
    def _apply_denoise(bgr: np.ndarray) -> np.ndarray:
        """Remove scanner noise via non-local means denoising."""
        return cv2.fastNlMeansDenoisingColored(
            bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )

    @staticmethod
    def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
        """CLAHE contrast normalisation in LAB colour space (L channel only)."""
        l_ch, a_ch, b_ch = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
        lab_norm = cv2.merge(
            [
                cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch),
                a_ch,
                b_ch,
            ]
        )
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _resize_with_padding(
        bgr: np.ndarray, target_size: tuple[int, int]
    ) -> np.ndarray:
        """Uniform-scale + white-pad to exactly ``target_size``."""
        tw, th = target_size
        sh, sw = bgr.shape[:2]
        scale = min(tw / sw, th / sh)
        nw, nh = int(sw * scale), int(sh * scale)
        canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
        canvas[
            (th - nh) // 2 : (th - nh) // 2 + nh, (tw - nw) // 2 : (tw - nw) // 2 + nw
        ] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return canvas

    # Conversion helpers

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
