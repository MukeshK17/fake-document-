import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ANN_DIR = Path("data/raw/fake_pan_card_annotations")
IMG_DIR = Path("data/raw/fake_pan_card")

# Fields we care about for training
TARGET_FIELDS = {
    "name",
    "fathers name",
    "date of birth",
    "pan number",
    "photo",
    "signature",
    "income tax department",
    "govt of india",
    "bharat sarkar",
    "ayakar vibhag",
    "pan label",
    "national",
    "tampered",
}

field_positions = defaultdict(list)

for jf in sorted(ANN_DIR.glob("*.json")):
    stem = jf.stem
    img_path = None
    for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue
    H, W = img.shape[:2]

    data = json.load(open(jf))
    items = data if isinstance(data, list) else [data]

    for item in items:
        raw = item.get("labels", {}).get("labelName", "")
        label = raw.strip().strip("'\"").strip().lower()
        if label not in TARGET_FIELDS:
            continue

        r = item["rectMask"]
        x1 = float(r["xMin"]) / W
        y1 = float(r["yMin"]) / H
        bw = float(r["width"]) / W
        bh = float(r["height"]) / H
        field_positions[label].append([x1, y1, bw, bh])

# Print template with variance
print(
    f"{'Field':<30} {'x1':>6} {'y1':>6} {'w':>6} {'h':>6} {'std_x':>7} {'std_y':>7} {'n':>5}"
)
print("─" * 75)
template = {}
for label in sorted(field_positions.keys()):
    arr = np.array(field_positions[label])
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    template[label] = avg.tolist()
    print(
        f"{label:<30} {avg[0]:>6.3f} {avg[1]:>6.3f} {avg[2]:>6.3f} {avg[3]:>6.3f} "
        f"{std[0]:>7.3f} {std[1]:>7.3f} {len(arr):>5}"
    )

with open("data/field_position_template.json", "w") as f:
    json.dump(template, f, indent=2)
print("\nSaved template")
