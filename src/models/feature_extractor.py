
import cv2
import numpy as np


def extract_features(ocr_entry: dict, image_path: str) -> dict:
    """
    Extract numerical features from a document for anomaly detection.
    All features normalized to comparable scales.
    """
    features = {}
    words = ocr_entry.get("words", [])
    scores = ocr_entry.get("scores", [])
    boxes = ocr_entry.get("boxes", [])

    # OCR-based features
    features["avg_ocr_confidence"] = (
        sum(scores) / len(scores) if scores else 0.0
    )
    features["min_ocr_confidence"] = min(scores) if scores else 0.0
    features["low_conf_ratio"] = (
        sum(1 for s in scores if s < 0.7) / len(scores)
        if scores else 0.0
    )
    features["word_count"] = len(words)

    # Confidence variance — high variance = inconsistent quality
    if len(scores) > 1:
        mean = features["avg_ocr_confidence"]
        features["conf_variance"] = sum(
            (s - mean) ** 2 for s in scores
        ) / len(scores)
    else:
        features["conf_variance"] = 0.0

    # Image-based features
    img = cv2.imread(str(image_path))
    if img is None:
        # Return zero features if image not loadable
        return {k: 0.0 for k in [
            "avg_ocr_confidence", "min_ocr_confidence",
            "low_conf_ratio", "word_count", "conf_variance",
            "width", "height", "aspect_ratio", "total_pixels",
            "bg_sharpness", "avg_region_sharpness",
            "max_sharpness_ratio", "ela_mean", "ela_std",
            "noise_variance_ratio", "mean_brightness",
            "brightness_std", "edge_density",
            "saturation_mean", "saturation_std"
        ]}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Resolution features
    features["width"] = float(w)
    features["height"] = float(h)
    features["aspect_ratio"] = round(w / h if h > 0 else 0, 3)
    features["total_pixels"] = float(w * h)

    # Sharpness features
    bg_region = gray[h//4: 3*h//4, w//4: 3*w//4]
    bg_sharp = float(cv2.Laplacian(bg_region, cv2.CV_64F).var())
    features["bg_sharpness"] = bg_sharp

    region_sharps = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        region = gray[y1:y2, x1:x2]
        region_sharps.append(
            float(cv2.Laplacian(region, cv2.CV_64F).var())
        )

    if region_sharps:
        avg_sharp = sum(region_sharps) / len(region_sharps)
        max_sharp = max(region_sharps)
        features["avg_region_sharpness"] = avg_sharp
        features["max_sharpness_ratio"] = min(
            max_sharp / (bg_sharp + 1e-6), 50.0
        )
    else:
        features["avg_region_sharpness"] = 0.0
        features["max_sharpness_ratio"] = 0.0

    # ELA features
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        recomp = cv2.imread(tmp_path)
        if recomp is not None and recomp.shape == img.shape:
            diff = cv2.absdiff(img, recomp).astype(np.float32)
            features["ela_mean"] = float(np.mean(diff))
            features["ela_std"] = float(np.std(diff))
        else:
            features["ela_mean"] = 0.0
            features["ela_std"] = 0.0
    finally:
        os.unlink(tmp_path)

    # Noise uniformity
    mh, mw = h // 2, w // 2
    quads = [
        gray[:mh, :mw], gray[:mh, mw:],
        gray[mh:, :mw], gray[mh:, mw:]
    ]
    stds = [float(np.std(q)) for q in quads]
    max_std, min_std = max(stds), min(stds)
    features["noise_variance_ratio"] = (
        (max_std - min_std) / (max_std + 1e-6)
    )

    # Brightness features
    features["mean_brightness"] = float(np.mean(gray))
    features["brightness_std"] = float(np.std(gray))

    # Edge density — printed/scanned fakes have different edge profiles
    edges = cv2.Canny(gray, 50, 150)
    features["edge_density"] = float(np.sum(edges > 0)) / (h * w)

    # Color saturation — B&W scans vs color photos vs digital fakes
    saturation = hsv[:, :, 1].astype(float)
    features["saturation_mean"] = float(np.mean(saturation))
    features["saturation_std"] = float(np.std(saturation))

    return features
