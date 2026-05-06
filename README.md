# Fake Document Detection Pipeline 

An asynchronous machine learning pipeline built for Risk Management to identify fake government documents and financial records such as PAN cards, Aadhaar cards, and bank statements.

## Overview
Forgery methods are becoming more advanced, so relying only on visual inspection is not enough anymore. This project uses a multimodal approach that combines image analysis, metadata checking, and document text-layout understanding to detect fake documents more reliably.

The system analyzes documents and generates a **Risk Confidence Score** to help identify suspicious cases that may need manual review.

## Core Architecture
This project follows a modular and configuration-based structurex . The pipeline includes the following components:

* **Document Routing & KIE:** `LayoutLMv3` (Microsoft) for spatial layout understanding, document classification, and Key Information Extraction.
* **Text Extraction:** `PaddleOCR` for robust spatial text bounding and extraction.
* **Visual Forensics:** Convolutional Neural Networks (CNNs) for pixel-level tampering detection (Copy-Move, JPEG Ghosting).
* **Semantic Rules Engine:** A deterministic validator using Regular Expressions and mathematical logic (e.g., Verhoeff checksums) to catch logical inconsistencies.

##  Repository Blueprint
```text
fake-document-/
├── configs/               # Control room for YAML configurations (model paths, risk thresholds)
├── data/                  # Local datasets (ignored by Git, managed via DVC later)
├── demo/                  # Quick-start scripts and basic FastAPI interfaces
├── docs/                  # Documentation and API specs
├── src/                   # Core application code
│   ├── models/            # Wrappers for LayoutLMv3, CNNs, PaddleOCR
│   ├── validators/        # Deterministic Regex & Mathematical rules engines
│   └── utils/             # Helper functions (logging, image resizing)
├── tests/                 # Pytest suite for unit and integration testing
├── tools/                 # Execution scripts (train.py, evaluate.py)
├── pyproject.toml         # Ruff formatting and linting rules
├── requirements.txt       # Locked dependencies
└── README.md
