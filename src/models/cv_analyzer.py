
import cv2
import numpy as np

SUSPICIOUS_RESOLUTIONS = {
    (1280, 720),   # standard screenshot
    (1920, 1080),  # full HD screenshot
    (1366, 768),   # laptop screenshot
    (1024, 768),   # common screen
    (1586, 992),   # downloads
}

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(str(image_path))
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

    # Check for suspicious screen resolutions
    if (w, h) in SUSPICIOUS_RESOLUTIONS:
        issues.append(f"suspicious_screen_resolution:{w}x{h}")

    # Allow portrait scans and landscape cards
    if not (0.5 <= aspect_ratio <= 0.85) and not (1.3 <= aspect_ratio <= 2.2):
        issues.append(f"unusual_aspect_ratio:{round(aspect_ratio,2)}")

    return {
        "width": w,
        "height": h,
        "total_pixels": total_pixels,
        "aspect_ratio": round(aspect_ratio, 3),
        "issues": issues
    }


def check_noise_uniformity(img: np.ndarray) -> dict:
    """
    Detect uneven noise — a sign of spliced/edited regions.
    Divide image into quadrants, compare noise levels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mh, mw = h // 2, w // 2

    quadrants = {
        "top_left":     gray[:mh, :mw],
        "top_right":    gray[:mh, mw:],
        "bottom_left":  gray[mh:, :mw],
        "bottom_right": gray[mh:, mw:],
    }

    stds = {k: float(np.std(v)) for k, v in quadrants.items()}
    max_std = max(stds.values())
    min_std = min(stds.values())
    variance_ratio = (max_std - min_std) / (max_std + 1e-6)

    issues = []
    if variance_ratio > 0.65:
        issues.append(f"uneven_noise_distribution:{round(variance_ratio,2)}")

    return {
        "quadrant_stds": stds,
        "variance_ratio": round(variance_ratio, 3),
        "issues": issues
    }


def check_ela(img: np.ndarray, quality: int = 90) -> dict:
    """
    Error Level Analysis — detects regions saved at different
    compression levels, which happens when content is pasted in.
    """
    import os
    import tempfile
    issues = []

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        recompressed = cv2.imread(tmp_path)

        if recompressed is None or recompressed.shape != img.shape:
            return {"issues": ["ela_failed"]}

        # ELA difference
        diff = cv2.absdiff(img, recompressed).astype(np.float32)
        ela_mean = float(np.mean(diff))
        ela_max = float(np.max(diff))
        ela_std = float(np.std(diff))

        # High ELA std = uneven compression = possible paste
        if ela_std > 15:
            issues.append(f"high_ela_variance:{round(ela_std,1)}")
        if ela_max > 80:
            issues.append(f"high_ela_peak:{round(ela_max,1)}")

    finally:
        os.unlink(tmp_path)

    return {
        "ela_mean": round(ela_mean, 3),
        "ela_max": round(ela_max, 3),
        "ela_std": round(ela_std, 3),
        "issues": issues
    }


def analyze_image(image_path: str) -> dict:
    """
    Full CV analysis pipeline.
    Returns dict of all signals and combined cv_risk_score.
    """
    try:
        img = load_image(image_path)
    except Exception as e:
        return {
            "cv_risk_score": 0,
            "error": str(e),
            "issues": ["image_load_failed"]
        }

    resolution = check_resolution(img)
    noise = check_noise_uniformity(img)
    ela = check_ela(img)

    all_issues = (
        resolution["issues"] +
        noise["issues"] +
        ela["issues"]
    )

    # CV risk score
    cv_score = 0
    if "very_low_resolution" in str(all_issues):
        cv_score += 20
    if "suspicious_screen_resolution" in str(all_issues):
        cv_score += 50  # very strong signal
    if "unusual_aspect_ratio" in str(all_issues):
        cv_score += 15
    if "uneven_noise_distribution" in str(all_issues):
        cv_score += 25
    if "high_ela_variance" in str(all_issues):
        cv_score += 30
    if "high_ela_peak" in str(all_issues):
        cv_score += 20

    return {
        "resolution": resolution,
        "noise": noise,
        "ela": ela,
        "all_issues": all_issues,
        "cv_risk_score": min(cv_score, 100)
    }

def check_regional_sharpness(img: np.ndarray, boxes: list) -> dict:
    """
    Compare sharpness of text regions against document average.
    Pasted/typed text often has different sharpness than scanned text.
    Uses Laplacian variance as sharpness measure.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Document-level sharpness baseline
    doc_sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if not boxes:
        return {"doc_sharpness": round(doc_sharpness, 2), "issues": []}

    region_sharpness = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        region = gray[y1:y2, x1:x2]
        if region.size < 50:
            continue
        sharpness = float(cv2.Laplacian(region, cv2.CV_64F).var())
        region_sharpness.append(sharpness)

    if not region_sharpness:
        return {"doc_sharpness": round(doc_sharpness, 2), "issues": []}

    avg_region_sharpness = sum(region_sharpness) / len(region_sharpness)
    max_region_sharpness = max(region_sharpness)

    issues = []
    # If one region is dramatically sharper than document average
    if doc_sharpness > 0 and max_region_sharpness / doc_sharpness > 3.0:
        issues.append(f"sharpness_outlier_region:{round(max_region_sharpness/doc_sharpness, 1)}x")

    return {
        "doc_sharpness": round(doc_sharpness, 2),
        "avg_region_sharpness": round(avg_region_sharpness, 2),
        "max_region_sharpness": round(max_region_sharpness, 2),
        "issues": issues
    }
