from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch

# allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from PIL import Image, ImageDraw

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger("smoke_test")

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pipeline_prod.yaml"

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
SKIP = f"{YELLOW}⊘ SKIP{RESET}"


# Synthetic document factory

def make_synthetic_doc(doc_type: str = "AADHAR_CARD") -> tuple[Path, list[str], list[list[int]]]:
    """
    Create a minimal synthetic document image with text drawn on it.
    Returns (temp_png_path, words, boxes) — the OCR output is pre-built
    so we don't need a real OCR engine for most stages.
    """
    templates: dict[str, dict] = {
        "AADHAR_CARD": {
            "lines": [
                "Unique Identification Authority of India",
                "Name: RAHUL SHARMA",
                "DOB: 12/08/1990",
                "Male",
                "1234 5678 9012",
                "आधार",
            ],
            "words": ["Unique", "Identification", "Authority", "of", "India",
                      "Name:", "RAHUL", "SHARMA", "DOB:", "12/08/1990",
                      "Male", "1234", "5678", "9012"],
            "boxes": [
                [10,10,180,25],[185,10,310,25],[315,10,420,25],[425,10,445,25],[450,10,520,25],
                [10,40,60,55],[65,40,140,55],[145,40,240,55],
                [10,70,55,85],[60,70,180,85],
                [10,100,70,115],
                [10,130,80,145],[85,130,165,145],[170,130,250,145],
            ],
        },
        "PAN_CARD": {
            "lines": [
                "Income Tax Department",
                "GOVT OF INDIA",
                "Permanent Account Number Card",
                "ABCDE1234F",
                "Name: JOHN DOE",
                "Father: JAMES DOE",
                "DOB: 01/01/1985",
            ],
            "words": ["Income", "Tax", "Department", "GOVT", "OF", "INDIA",
                      "ABCDE1234F", "Name:", "JOHN", "DOE",
                      "Father:", "JAMES", "DOE", "DOB:", "01/01/1985"],
            "boxes": [
                [10,10,90,25],[95,10,125,25],[130,10,220,25],
                [10,40,65,55],[70,40,100,55],[105,40,155,55],
                [10,70,145,85],
                [10,100,55,115],[60,100,115,115],[120,100,170,115],
                [10,130,65,145],[70,130,130,145],[135,130,180,145],
                [10,160,55,175],[60,160,180,175],
            ],
        },
    }

    tmpl = templates.get(doc_type, templates["AADHAR_CARD"])

    img = Image.new("RGB", (400, 250), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 10
    for line in tmpl["lines"]:
        draw.text((10, y), line, fill=(20, 20, 20))
        y += 30
    draw.rectangle([5, 5, 395, 245], outline=(150, 150, 150), width=2)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return Path(tmp.name), tmpl["words"], tmpl["boxes"]


# Individual stage tests
def test_stage_1_ingest(cfg: dict, img_path: Path, verbose: bool) -> bool:
    from src.data.ingestion import DocumentIngester
    ingester = DocumentIngester(cfg)
    pages = ingester.load(img_path)
    ok = len(pages) == 1 and pages[0].mode == "RGB"
    if verbose:
        print(f"      size={pages[0].size}  mode={pages[0].mode}")
    return ok


def test_stage_2_preprocess(cfg: dict, img_path: Path, verbose: bool) -> bool:
    from src.data.ingestion import DocumentIngester
    from src.data.preprocessing import DocumentPreprocessor
    ingester = DocumentIngester(cfg)
    preprocessor = DocumentPreprocessor(cfg)
    pages = ingester.load(img_path)
    clean = preprocessor.process(pages[0], augment=False)
    target = tuple(cfg["preprocessing"]["target_size"])
    ok = clean.size == target and clean.mode == "RGB"
    if verbose:
        print(f"      cleaned size={clean.size}  expected={target}")
    return ok


def test_stage_2b_augment(cfg: dict, img_path: Path, verbose: bool) -> bool:
    from src.data.ingestion import DocumentIngester
    from src.data.preprocessing import DocumentPreprocessor
    ingester = DocumentIngester(cfg)
    preprocessor = DocumentPreprocessor(cfg)
    pages = ingester.load(img_path)
    aug = preprocessor.process(pages[0], augment=True)
    ok = aug.mode == "RGB" and aug.size == tuple(cfg["preprocessing"]["target_size"])
    if verbose:
        print(f"      augmented size={aug.size}")
    return ok


def test_stage_3_ocr(cfg: dict, img_path: Path, words: list, boxes: list, verbose: bool) -> tuple[bool, bool]:
    """Returns (ran, passed). ran=False means paddleocr not installed → skip."""
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False, False

    from src.data.ingestion import DocumentIngester
    from src.data.preprocessing import DocumentPreprocessor
    from src.extractors.paddleOcr_extractor import PaddleOCRExtractor

    ingester = DocumentIngester(cfg)
    preprocessor = DocumentPreprocessor(cfg)
    extractor = PaddleOCRExtractor(cfg)
    extractor.load()

    pages = ingester.load(img_path)
    clean = preprocessor.process(pages[0])
    result = extractor.extract(clean)
    ok = isinstance(result["words"], list) and isinstance(result["boxes"], list)
    if verbose:
        print(f"      tokens={len(result['words'])}  boxes={len(result['boxes'])}")
    return True, ok


def test_stage_4_classifier(cfg: dict, img_path: Path, words: list, boxes: list, verbose: bool) -> tuple[bool, bool]:
    """Returns (ran, passed). ran=False means transformers not installed → skip."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False, False

    from src.data.ingestion import DocumentIngester
    from src.data.preprocessing import DocumentPreprocessor
    from src.models.layoutlm_classifier import LayoutLMv3Classifier

    ingester = DocumentIngester(cfg)
    preprocessor = DocumentPreprocessor(cfg)

    # Mock the heavy model load so smoke test runs without downloading weights
    with patch("transformers.LayoutLMv3ForSequenceClassification.from_pretrained") as mock_model, \
         patch("transformers.LayoutLMv3Processor.from_pretrained") as mock_proc:

        # Build a fake forward pass that returns plausible logits
        num_labels = len(cfg["layoutlmv3"]["id2label"])
        import torch
        fake_logits = torch.zeros(1, num_labels)
        fake_logits[0][1] = 3.5   # AADHAR_CARD is index 1 → highest score

        fake_output = MagicMock()
        fake_output.logits = fake_logits

        mock_model.return_value.eval.return_value = None
        mock_model.return_value.to.return_value = mock_model.return_value
        mock_model.return_value.return_value = fake_output
        mock_model.return_value.__call__ = lambda self, **kw: fake_output

        # Fake processor returns minimal tensor dict
        fake_encoding = {
            "input_ids":      torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
            "bbox":           torch.zeros(1, 512, 4, dtype=torch.long),
            "pixel_values":   torch.zeros(1, 3, 224, 224),
        }
        mock_proc.return_value.return_value = fake_encoding
        mock_proc.return_value.__call__ = lambda *a, **kw: fake_encoding

        clf = LayoutLMv3Classifier(cfg)
        clf.load()

        pages = ingester.load(img_path)
        clean = preprocessor.process(pages[0])
        result = clf.inference(image=clean, words=words, boxes=boxes)

    ok = (
        result["label"] in cfg["layoutlmv3"]["id2label"].values()
        and 0.0 <= result["confidence"] <= 1.0
    )
    if verbose:
        print(f"      label={result['label']}  confidence={result['confidence']:.4f}")
    return True, ok


def test_stage_5_rules(cfg: dict, doc_type: str, words: list, boxes: list, verbose: bool) -> bool:
    from src.validators.rules_engine import RulesEngine
    engine = RulesEngine(cfg)
    result = engine.validate(document_type=doc_type, words=words, boxes=boxes)
    ok = 0.0 <= result["score"] <= 1.0 and isinstance(result["detail"], dict)
    if verbose:
        for rule, passed in result["detail"].items():
            mark = "✓" if passed else "✗"
            print(f"      {mark}  {rule}")
        print(f"      score={result['score']:.2f}")
    return ok


def test_stage_6_ensemble(cfg: dict, verbose: bool) -> bool:
    from src.pipeline import DocumentVerificationPipeline

    # We only test the private helpers, no load() needed
    pipe = DocumentVerificationPipeline.__new__(DocumentVerificationPipeline)
    pipe._cfg        = cfg
    pipe._thresholds = cfg["risk_thresholds"]
    pipe._weights    = cfg["ensemble_weights"]

    score   = pipe._ensemble(layout_score=0.95, cnn_score=1.0, rules_score=0.75)
    verdict = pipe._verdict(score)
    ok = isinstance(score, float) and verdict in ("APPROVED", "MANUAL_REVIEW", "REJECTED")
    if verbose:
        print(f"      risk_score={score:.4f}  verdict={verdict}")
    return ok


def test_stage_7_output(cfg: dict, img_path: Path, words: list, boxes: list, verbose: bool) -> bool:
    """Full pipeline run with mocked models — verifies CSV + audit log are written."""
    import torch

    from src.pipeline import DocumentVerificationPipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_copy = json.loads(json.dumps(cfg))
        cfg_copy["output"]["results_dir"]    = tmp_dir + "/results"
        cfg_copy["output"]["audit_log_path"] = tmp_dir + "/logs/audit.jsonl"

        # Mock OCR
        mock_ocr = MagicMock()
        mock_ocr.is_loaded = True
        mock_ocr.extract.return_value = {"words": words, "boxes": boxes, "scores": [0.95]*len(words)}

        # Mock classifier
        num_labels = len(cfg["layoutlmv3"]["id2label"])
        fake_logits = torch.zeros(1, num_labels)
        fake_logits[0][1] = 3.5
        fake_output = MagicMock()
        fake_output.logits = fake_logits

        mock_clf_model = MagicMock()
        mock_clf_model.return_value = fake_output

        with patch("src.pipeline.PaddleOCRExtractor") as MockOCR, \
             patch("src.pipeline.LayoutLMv3Classifier") as MockClf, \
             patch("transformers.LayoutLMv3ForSequenceClassification.from_pretrained", return_value=mock_clf_model), \
             patch("transformers.LayoutLMv3Processor.from_pretrained"):

            MockOCR.return_value = mock_ocr
            fake_encoding = {
                "input_ids":      torch.zeros(1, 512, dtype=torch.long),
                "attention_mask": torch.ones(1, 512, dtype=torch.long),
                "bbox":           torch.zeros(1, 512, 4, dtype=torch.long),
                "pixel_values":   torch.zeros(1, 3, 224, 224),
            }
            MockClf.return_value.is_loaded = True
            MockClf.return_value.inference.return_value = {
                "label": "AADHAR_CARD", "confidence": 0.94,
                "scores": {}, "label_id": 1,
            }

            pipe = DocumentVerificationPipeline(cfg_copy)
            pipe._ocr = mock_ocr
            pipe.load()
            result = pipe.run(str(img_path), doc_id="smoke-001")

        audit_path  = Path(tmp_dir) / "logs"  / "audit.jsonl"
        results_csv = Path(tmp_dir) / "results" / "results.csv"

        audit_ok  = audit_path.exists()  and audit_path.stat().st_size > 0
        csv_ok    = results_csv.exists() and results_csv.stat().st_size > 0
        result_ok = result["document_type"] == "AADHAR_CARD"

        if verbose:
            print(f"      audit_log  written={audit_ok}  size={audit_path.stat().st_size if audit_ok else 0}B")
            print(f"      results_csv written={csv_ok}")
            print(f"      verdict={result['verdict']}  is_fake={result['is_fake']}")

        return audit_ok and csv_ok and result_ok


def test_stage_8_dataloader(cfg: dict, img_path: Path, words: list, boxes: list, verbose: bool) -> bool:
    import torch

    from src.data.dataloader import document_collate_fn
    from src.data.dataset import DocumentDataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write a tiny manifest
        manifest = Path(tmp_dir) / "train.csv"
        manifest.write_text("doc_id,file_path,label\n"
                            f"doc001,{img_path},AADHAR_CARD\n"
                            f"doc002,{img_path},PAN_CARD\n")

        # Write a tiny OCR cache
        cache = Path(tmp_dir) / "train_ocr_cache.jsonl"
        cache.write_text(
            json.dumps({"doc_id": "doc001", "words": words, "boxes": boxes}) + "\n" +
            json.dumps({"doc_id": "doc002", "words": words, "boxes": boxes}) + "\n"
        )

        ds = DocumentDataset(manifest, cfg, split="val")
        sample = ds[0]

        batch = document_collate_fn([ds[0], ds[1]], max_seq_length=32)

        ok = (
            len(ds) == 2
            and batch["labels"].shape == torch.Size([2])
            and len(batch["words"]) == 2
            and len(batch["words"][0]) == 32    # padded to max_seq_length
        )
        if verbose:
            print(f"      dataset_len={len(ds)}  batch_labels={batch['labels'].tolist()}")
            print(f"      words_padded_len={len(batch['words'][0])}  expected=32")
    return ok


# Runner

def run_stage(name: str, fn, *args) -> tuple[str, bool, bool]:
    """Run one stage function, catch exceptions, return (name, ran, passed)."""
    try:
        result = fn(*args)
        if isinstance(result, tuple):
            ran, passed = result
        else:
            ran, passed = True, result
        return name, ran, passed
    except Exception:
        logger.debug("Stage '%s' raised:\n%s", name, traceback.format_exc())
        return name, True, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake Doc Detector — smoke test")
    parser.add_argument("--stages", default="all",
                        help="Comma-separated stage numbers to run, e.g. 1,2,5 (default: all)")
    parser.add_argument("--doc-type", default="AADHAR_CARD",
                        choices=["AADHAR_CARD","PAN_CARD"],
                        help="Synthetic document type to generate")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    active = set(args.stages.split(",")) if args.stages != "all" else None

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Fake Document Detector — Smoke Test{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")

    if not CONFIG_PATH.exists():
        print(f"{RED}Config not found: {CONFIG_PATH}{RESET}")
        sys.exit(1)

    cfg: dict = yaml.safe_load(CONFIG_PATH.read_text())

    # Override preprocessing target_size to something small for speed
    cfg["preprocessing"]["target_size"] = [224, 224]
    cfg["preprocessing"]["correct_orientation"] = False
    cfg["preprocessing"]["denoise"] = False

    img_path, words, boxes = make_synthetic_doc(args.doc_type)
    print(f"  Synthetic doc  : {args.doc_type}")
    print(f"  Temp image     : {img_path}")
    print(f"  OCR tokens     : {len(words)}\n")

    stages = [
        ("1",  "Ingest",                 test_stage_1_ingest,       (cfg, img_path, args.verbose)),
        ("2",  "Preprocess (clean)",     test_stage_2_preprocess,   (cfg, img_path, args.verbose)),
        ("2b", "Preprocess (augment)",   test_stage_2b_augment,     (cfg, img_path, args.verbose)),
        ("3",  "OCR extractor",          test_stage_3_ocr,          (cfg, img_path, words, boxes, args.verbose)),
        ("4",  "LayoutLMv3 classifier",  test_stage_4_classifier,   (cfg, img_path, words, boxes, args.verbose)),
        ("5",  "Rules engine",           test_stage_5_rules,        (cfg, args.doc_type, words, boxes, args.verbose)),
        ("6",  "Ensemble + verdict",     test_stage_6_ensemble,     (cfg, args.verbose)),
        ("7",  "CSV + audit output",     test_stage_7_output,       (cfg, img_path, words, boxes, args.verbose)),
        ("8",  "DataLoader batch",       test_stage_8_dataloader,   (cfg, img_path, words, boxes, args.verbose)),
    ]

    results = []
    for num, label, fn, fn_args in stages:
        if active and num not in active:
            continue
        print(f"  {BOLD}Stage {num:<3}{RESET}  {label:<28}", end="  ", flush=True)
        name, ran, passed = run_stage(label, fn, *fn_args)
        if not ran:
            print(SKIP + "  (dependency not installed)")
            results.append((label, "skip"))
        elif passed:
            print(PASS)
            results.append((label, "pass"))
        else:
            print(FAIL)
            results.append((label, "fail"))
            if args.verbose:
                print()

    # Summary
    total  = len(results)
    passed = sum(1 for _, s in results if s == "pass")
    skipped= sum(1 for _, s in results if s == "skip")
    failed = sum(1 for _, s in results if s == "fail")

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"  {BOLD}Results{RESET}   {GREEN}{passed} passed{RESET}  "
          f"{YELLOW}{skipped} skipped{RESET}  {RED}{failed} failed{RESET}  "
          f"(of {total} stages)\n")

    if failed == 0:
        print(f"  {GREEN}{BOLD}All stages passed — data pipeline is healthy ✓{RESET}\n")
    else:
        print(f"  {RED}{BOLD}Fix the failing stages before moving to model training.{RESET}\n")
        sys.exit(1)

    # Clean up temp file
    try:
        img_path.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
