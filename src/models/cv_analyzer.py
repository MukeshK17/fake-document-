"""CV and ELA analysis utilities."""

import os
import tempfile

import cv2
import numpy as np

SUSPICIOUS_RESOLUTIONS = {
    (1280, 720),
    (1920, 1080),
    (1366, 768),
    (1024, 768),
    (1586, 992),
}


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return img


def check_resolution(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    total_pixels = h * w
    aspect_ratio = w / h if h > 0 else 0

    issues = []

    if total_pixels < 50000:
        issues.append("very_low_resolution")

    if (w, h) in SUSPICIOUS_RESOLUTIONS:
        issues.append(f"suspicious_screen_resolution:{w}x{h}")

    if not (0.5 <= aspect_ratio <= 0.85) and not (1.3 <= aspect_ratio <= 2.2):
        issues.append(f"unusual_aspect_ratio:{round(aspect_ratio, 2)}")

    return {
        "width": w,
        "height": h,
        "total_pixels": total_pixels,
        "aspect_ratio": round(aspect_ratio, 3),
        "issues": issues,
    }


def check_noise_uniformity(img: np.ndarray) -> dict:
    """Detect uneven noise by comparing quadrant intensity spread."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mh, mw = h // 2, w // 2

    quadrants = {
        "top_left": gray[:mh, :mw],
        "top_right": gray[:mh, mw:],
        "bottom_left": gray[mh:, :mw],
        "bottom_right": gray[mh:, mw:],
    }

    stds = {k: float(np.std(v)) for k, v in quadrants.items()}
    max_std = max(stds.values())
    min_std = min(stds.values())
    variance_ratio = (max_std - min_std) / (max_std + 1e-6)

    issues = []
    if variance_ratio > 0.65:
        issues.append(f"uneven_noise_distribution:{round(variance_ratio, 2)}")

    return {
        "quadrant_stds": stds,
        "variance_ratio": round(variance_ratio, 3),
        "issues": issues,
    }


def check_ela(img: np.ndarray, quality: int = 90) -> dict:
    """
    Error Level Analysis — detects regions saved at different
    compression levels, which happens when content is pasted in.
    """
    issues = []

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imread(tmp_path)

        if recompressed is None or recompressed.shape != img.shape:
            return {"issues": ["ela_failed"]}

        diff = cv2.absdiff(img, recompressed).astype(np.float32)
        ela_mean = float(np.mean(diff))
        ela_max = float(np.max(diff))
        ela_std = float(np.std(diff))

        if ela_std > 15:
            issues.append(f"high_ela_variance:{round(ela_std, 1)}")
        if ela_max > 80:
            issues.append(f"high_ela_peak:{round(ela_max, 1)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "ela_mean": round(ela_mean, 3),
        "ela_max": round(ela_max, 3),
        "ela_std": round(ela_std, 3),
        "issues": issues,
    }


def analyze_image(image_path: str) -> dict:
    """Full CV analysis pipeline."""
    try:
        img = load_image(image_path)
    except Exception as e:
        return {
            "cv_risk_score": 0,
            "error": str(e),
            "all_issues": ["image_load_failed"],
        }

    resolution = check_resolution(img)
    noise = check_noise_uniformity(img)
    ela = check_ela(img)

    all_issues = resolution["issues"] + noise["issues"] + ela["issues"]

    cv_score = 0
    if "very_low_resolution" in str(all_issues):
        cv_score += 20
    if "suspicious_screen_resolution" in str(all_issues):
        cv_score += 50
    if "unusual_aspect_ratio" in str(all_issues):
        cv_score += 15
    if "uneven_noise_distribution" in str(all_issues):
        cv_score += 25
    if "high_ela_variance" in str(all_issues):
        cv_score += 30
    if "high_ela_peak" in str(all_issues):
        cv_score += 20

    return {
        "cv_risk_score": min(cv_score, 100),
        "all_issues": all_issues,
    }
