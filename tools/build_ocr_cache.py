"""
Usage:

    # build cache for all three splits
    python tools/build_ocr_cache.py

    # build cache for one specific split only
    python tools/build_ocr_cache.py --split train

    # use a different config file
    python tools/build_ocr_cache.py --config configs/pipeline_dev.yaml

    # dry run — check manifests exist and images are readable, write nothing
    python tools/build_ocr_cache.py --dry-run

    # overwrite existing cache entries (default: skip already-cached doc_ids)
    python tools/build_ocr_cache.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)

G, R, Y, B, BOLD, RST = (
    "\033[92m",
    "\033[91m",
    "\033[93m",
    "\033[96m",
    "\033[1m",
    "\033[0m",
)


def load_existing_cache(cache_path: Path) -> dict[str, dict]:
    """Load an existing JSONL cache into a dict keyed by doc_id."""
    if not cache_path.exists():
        return {}
    cache = {}
    with cache_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line := line.strip():
                obj = json.loads(line)
                cache[obj["doc_id"]] = obj
    return cache


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Return rows from a manifest CSV as a list of dicts."""
    if not manifest_path.exists():
        print(f"  {R}missing{RST}  {manifest_path}")
        return []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_cache_for_split(
    split: str,
    splits_dir: Path,
    cfg: dict,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Process one split.  Returns (processed, skipped, failed) counts.
    """
    from src.data.ingestion import DocumentIngester
    from src.data.preprocessing import DocumentPreprocessor
    from src.extractors.paddleOcr_extractor import PaddleOCRExtractor

    manifest_path = splits_dir / f"{split}.csv"
    cache_path = splits_dir / f"{split}_ocr_cache.jsonl"

    rows = read_manifest(manifest_path)
    if not rows:
        return 0, 0, 0

    existing = {} if overwrite else load_existing_cache(cache_path)
    to_process = [r for r in rows if r["doc_id"] not in existing]
    skipped = len(rows) - len(to_process)

    print(f"\n  {BOLD}Split: {split}{RST}")
    print(f"  manifest  : {manifest_path}  ({len(rows)} rows)")
    print(f"  cache     : {cache_path}")
    print(f"  to process: {len(to_process)}  |  already cached: {skipped}")

    if dry_run:
        print(
            f"  {B}[dry-run]{RST}  would process {len(to_process)} images — writing nothing"
        )
        return 0, skipped, 0

    if not to_process:
        print(f"  {G}all images already cached — nothing to do{RST}")
        return 0, skipped, 0

    # Initialise pipeline modules
    # Use preprocessing_overrides from smoke_test section for speed if available,
    # otherwise use production config as-is
    ingester = DocumentIngester(cfg)
    preprocessor = DocumentPreprocessor(cfg)
    extractor = PaddleOCRExtractor(cfg)

    print("  loading PaddleOCR…", end=" ", flush=True)
    extractor.load()
    print(f"{G}ready{RST}\n")

    processed = failed = 0
    t_start = time.perf_counter()

    with cache_path.open("a", encoding="utf-8") as out_fh:
        # write any existing entries back first if overwrite=True
        if overwrite and existing:
            for entry in existing.values():
                out_fh.write(json.dumps(entry) + "\n")

        for i, row in enumerate(to_process, 1):
            doc_id = row["doc_id"]
            file_path = ROOT / row["file_path"]
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / i) * (len(to_process) - i) if i > 1 else 0

            print(
                f"  [{i:>4}/{len(to_process)}]  {doc_id:<20}"
                f"  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                end="  ",
                flush=True,
            )

            try:
                pages = ingester.load(file_path)
                # No augmentation for OCR cache — want clean deterministic tokens
                image = preprocessor.process(pages[0], augment=False)
                result = extractor.extract(image)

                entry = {
                    "doc_id": doc_id,
                    "words": result["words"],
                    "boxes": result["boxes"],
                    "scores": result["scores"],
                }
                out_fh.write(json.dumps(entry) + "\n")
                out_fh.flush()  # flush every line — safe to interrupt mid-run

                print(f"{G}✓{RST}  {len(result['words'])} tokens")
                processed += 1

            except Exception as exc:
                print(f"{R}✗  {exc}{RST}")
                failed += 1

    return processed, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PaddleOCR cache for train/val/test manifests"
    )
    parser.add_argument("--config", default="configs/pipeline_prod.yaml")
    parser.add_argument("--splits-dir", default=str(ROOT / "data" / "splits"))
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to process  (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process and overwrite already-cached entries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests and images without writing anything",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"{R}Config not found: {config_path}{RST}")
        sys.exit(1)

    cfg = yaml.safe_load(config_path.read_text())

    # Disable slow preprocessing stages for cache building speed
    # (same images, but we don't need perfect quality for cache — OCR is fast enough)
    cfg["preprocessing"]["correct_orientation"] = False
    cfg["preprocessing"]["denoise"] = False

    splits_dir = Path(args.splits_dir)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    print(f"\n{BOLD}{B}build_ocr_cache.py{RST}\n")
    print(f"  config     : {config_path}")
    print(f"  splits dir : {splits_dir}")
    print(f"  splits     : {splits}")
    print(f"  overwrite  : {args.overwrite}")

    try:
        import paddleocr  # noqa: F401
    except ImportError:
        print(f"\n{R}PaddleOCR not installed.{RST}")
        print("Install with:  pip install paddleocr paddlepaddle\n")
        sys.exit(1)

    total_processed = total_skipped = total_failed = 0

    for split in splits:
        p, s, f = build_cache_for_split(
            split, splits_dir, cfg, args.overwrite, args.dry_run
        )
        total_processed += p
        total_skipped += s
        total_failed += f

    print(f"\n{BOLD}  Final summary{RST}")
    print(f"  processed : {total_processed}")
    print(f"  skipped   : {total_skipped}  (already cached)")
    print(f"  failed    : {total_failed}")

    if total_failed > 0:
        print(f"\n  {Y}Some images failed. Check the errors above.{RST}")
        print("  Re-run with --overwrite to retry failed entries.\n")
        sys.exit(1)

    if not args.dry_run:
        print(f"\n  {G}{BOLD}Cache built. Run tools/smoke_test.py to validate.{RST}\n")


if __name__ == "__main__":
    main()
