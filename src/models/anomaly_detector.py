import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    # Resolution — strong signal for screenshot fakes
    "width",
    "height",
    "aspect_ratio",
    "total_pixels",
    # Texture — captures scan vs digital vs photo
    "noise_variance_ratio",
    "saturation_mean",
    "saturation_std",
    "edge_density",
    # Sharpness — captures tampered regions
    "bg_sharpness",
    "max_sharpness_ratio",
    # ELA — captures compression artifacts
    "ela_std",
    # OCR quality
    "avg_ocr_confidence",
    "low_conf_ratio",
    "conf_variance",
]

MODEL_PATH = Path("data/models/isolation_forest.pkl")
SCALER_PATH = Path("data/models/scaler.pkl")


def load_features(splits_dir: Path, raw_dirs: dict) -> tuple:
    """
    Load features for all documents.
    Returns X (features), labels, doc_ids.
    """
    import sys

    sys.path.insert(0, ".")
    from src.models.feature_extractor import extract_features

    X, labels, doc_ids = [], [], []

    for split in ["train", "val", "test"]:
        cache_path = splits_dir / f"{split}_ocr_cache.jsonl"
        csv_path = splits_dir / f"{split}.csv"

        doc_labels = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                doc_labels[row["doc_id"]] = row["label"]

        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry["doc_id"]
                label = doc_labels.get(doc_id, "UNKNOWN")

                if label not in ("PAN_CARD", "FAKE_PAN"):
                    continue

                img_dir = raw_dirs.get(label)
                if img_dir is None:
                    continue

                matches = list(img_dir.glob(f"{doc_id}.*"))
                if not matches:
                    continue

                feats = extract_features(entry, str(matches[0]))
                feat_vector = [feats.get(k, 0.0) for k in FEATURE_NAMES]

                X.append(feat_vector)
                labels.append(label)
                doc_ids.append(doc_id)

                if len(X) % 100 == 0:
                    print(f"  Extracted features: {len(X)}")

    return np.array(X), labels, doc_ids


def train(splits_dir: Path, raw_dirs: dict):
    """Train IsolationForest on real documents only."""
    print("Extracting features from all documents...")
    X, labels, doc_ids = load_features(splits_dir, raw_dirs)

    # Train only on real documents
    real_mask = np.array([label == "PAN_CARD" for label in labels])
    X_real = X[real_mask]
    print(f"\nTraining on {len(X_real)} real documents")

    # Scale features
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)

    # Train IsolationForest
    # contamination=0.05 means we expect ~5% of real docs
    # to look unusual (conservative)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,  # expect ~8% anomalies
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_real_scaled)

    # Save model and scaler
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {MODEL_PATH}")

    # Evaluate immediately on all docs
    X_scaled = scaler.transform(X)
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    # Convert: IsolationForest returns -1 (anomaly) or 1 (normal)
    # We convert to: 1 = suspicious, 0 = clean
    flagged = predictions == -1

    real_labels = np.array(labels)
    tp = sum(flagged[real_labels == "FAKE_PAN"])
    fn = sum(~flagged[real_labels == "FAKE_PAN"])
    fp = sum(flagged[real_labels == "PAN_CARD"])
    tn = sum(~flagged[real_labels == "PAN_CARD"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n{'=' * 45}")
    print("ISOLATION FOREST RESULTS")
    print(f"{'=' * 45}")
    print(f"Real docs:   {sum(real_labels == 'PAN_CARD')}")
    print(f"Fake docs:   {sum(real_labels == 'FAKE_PAN')}")
    print(f"{'=' * 45}")
    print(f"TP: {tp}  FN: {fn}  FP: {fp}  TN: {tn}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1:         {f1:.3f}")
    print(f"FPR:        {fpr:.3f}")
    print(f"{'=' * 45}")

    return model, scaler, scores, labels, doc_ids


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from pathlib import Path

    SPLITS_DIR = Path("data/splits")
    RAW_DIRS = {
        "PAN_CARD": Path("data/raw/pan_card"),
        "FAKE_PAN": Path("data/raw/fake_pan_card"),
    }
    train(SPLITS_DIR, RAW_DIRS)
