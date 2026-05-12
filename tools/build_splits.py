from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"})

# folder name  →  exact label string that matches pipeline_prod.yaml id2label
FOLDER_TO_LABEL: dict[str, str] = {
    "pan_card":       "PAN_CARD",
    "aadhar_card":    "AADHAR_CARD",
    "bank_statement": "BANK_STATEMENT",
    "salary_slip":    "SALARY_SLIP",
    "itr_form":       "ITR_FORM",
}

# ANSI
G, R, Y, B, BOLD, RST = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"


def scan_raw(raw_dir: Path) -> list[dict[str, str]]:
    """Return a list of {doc_id, file_path, label} dicts for every image found."""
    records: list[dict[str, str]] = []
    for folder, label in FOLDER_TO_LABEL.items():
        folder_path = raw_dir / folder
        if not folder_path.exists():
            print(f"  {Y}skip{RST}  {folder}/  (folder not found)")
            continue
        images = sorted(
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
        if not images:
            print(f"  {Y}warn{RST}  {folder}/  exists but contains no supported images")
            continue
        for img in images:
            # doc_id  = filename without extension  e.g.  PAN_001
            # file_path = relative from repo root   e.g.  data/raw/pan_card/PAN_001.jpg
            records.append({
                "doc_id":    img.stem,
                "file_path": str(img.relative_to(ROOT)),
                "label":     label,
            })
        print(f"  {G}found{RST}  {folder}/  →  {len(images)} images  ({label})")
    return records


def split_records(
    records: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    Stratified split — each label keeps the same ratio in every split.
    This prevents a small class from landing entirely in one split.
    """
    rng = random.Random(seed)

    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    train, val, test = [], [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n      = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # test gets whatever is left
        train += items[:n_train]
        val   += items[n_train:n_train + n_val]
        test  += items[n_train + n_val:]

    return train, val, test


def write_csv(path: Path, records: list[dict[str, str]], dry_run: bool) -> None:
    if dry_run:
        print(f"  {B}[dry-run]{RST}  would write  {path}  ({len(records)} rows)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["doc_id", "file_path", "label"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  {G}wrote{RST}   {path}  ({len(records)} rows)")


def write_label_map(splits_dir: Path, dry_run: bool) -> None:
    label_map = {v: k for k, v in {
        "0": "PAN_CARD", "1": "AADHAR_CARD", "2": "BANK_STATEMENT",
        "3": "SALARY_SLIP", "4": "ITR_FORM", "5": "UNKNOWN",
    }.items()}
    path = splits_dir / "label_map.json"
    if dry_run:
        print(f"  {B}[dry-run]{RST}  would write  {path}")
        return
    path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"  {G}wrote{RST}   {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test CSV manifests from data/raw/"
    )
    parser.add_argument("--raw-dir",   default=str(ROOT / "data" / "raw"),
                        help="Path to raw images root  (default: data/raw/)")
    parser.add_argument("--splits-dir", default=str(ROOT / "data" / "splits"),
                        help="Output directory for CSVs  (default: data/splits/)")
    parser.add_argument("--train", type=float, default=0.70, metavar="RATIO")
    parser.add_argument("--val",   type=float, default=0.15, metavar="RATIO")
    parser.add_argument("--test",  type=float, default=0.15, metavar="RATIO")
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without touching disk")
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print(f"{R}Error: --train + --val + --test must equal 1.0{RST}")
        sys.exit(1)

    raw_dir    = Path(args.raw_dir)
    splits_dir = Path(args.splits_dir)

    print(f"\n{BOLD}{B}build_splits.py{RST}\n")
    print(f"  raw dir    : {raw_dir}")
    print(f"  splits dir : {splits_dir}")
    print(f"  ratio      : train={args.train}  val={args.val}  test={args.test}")
    print(f"  seed       : {args.seed}\n")

    # scan
    records = scan_raw(raw_dir)
    if not records:
        print(f"\n{R}No images found. Check that data/raw/<label_folder>/ exists.{RST}\n")
        sys.exit(1)

    print(f"\n  {BOLD}total images found: {len(records)}{RST}\n")

    # split
    train, val, test = split_records(records, args.train, args.val, args.seed)

    # write
    write_csv(splits_dir / "train.csv", train, args.dry_run)
    write_csv(splits_dir / "val.csv",   val,   args.dry_run)
    write_csv(splits_dir / "test.csv",  test,  args.dry_run)
    write_label_map(splits_dir, args.dry_run)

    print(f"\n{BOLD}  Summary{RST}")
    print(f"  train : {len(train):>4} images")
    print(f"  val   : {len(val):>4} images")
    print(f"  test  : {len(test):>4} images")
    print(f"  total : {len(records):>4} images\n")

    if not args.dry_run:
        print(f"  {G}{BOLD}Done. Run tools/build_ocr_cache.py next.{RST}\n")


if __name__ == "__main__":
    main()
