"""
Handles two non-trivial concerns:
- Variable-length OCR sequences: pads/truncates words+boxes to max_seq_length
  so every batch tensor has a fixed shape (PyTorch default collate cannot do this).
- Class imbalance: WeightedRandomSampler up-samples minority classes during training.
  Val/test loaders are always deterministic (no sampler, no shuffle).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import DocumentDataset, DocumentSample

logger = logging.getLogger(__name__)


def document_collate_fn(
    batch: list[DocumentSample],
    max_seq_length: int = 512,
) -> dict[str, Any]:
    """Pad/truncate words+boxes to fixed length and stack labels into a tensor.

    Returns dict with keys: ``images``, ``words``, ``boxes``, ``labels``, ``doc_ids``.
    """
    padded_words, padded_boxes = [], []
    for s in batch:
        w = s.words[:max_seq_length]
        b = s.boxes[:max_seq_length]
        pad = max_seq_length - len(w)
        padded_words.append(w + [""] * pad)
        padded_boxes.append(b + [[0, 0, 0, 0]] * pad)

    return {
        "images": [s.image for s in batch],
        "words": padded_words,
        "boxes": padded_boxes,
        "labels": torch.tensor([s.label for s in batch], dtype=torch.long),
        "doc_ids": [s.doc_id for s in batch],
    }


def _build_weighted_sampler(dataset: DocumentDataset) -> WeightedRandomSampler:
    """Inverse-frequency sampler so every class appears equally during training."""
    labels = [dataset.label2id[row["label"]] for row in dataset._samples]
    counts = Counter(labels)
    total = len(labels)
    weights = [total / counts[lbl] for lbl in labels]
    logger.debug(
        "WeightedSampler | %s",
        {dataset.id2label[k]: v for k, v in sorted(counts.items())},
    )
    return WeightedRandomSampler(weights, num_samples=total, replacement=True)


def _eval_loader(
    manifest: str | Path,
    config: dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    collate: Any,
) -> DataLoader:
    return DataLoader(
        DocumentDataset(manifest, config, split=split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_dataloaders(
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) from manifest paths + config."""
    dl_cfg = config.get("dataloader", {})
    rt_cfg = config.get("runtime", {})
    seq_len = int(
        config.get("layoutlmv3", {}).get("processor", {}).get("max_seq_length", 512)
    )

    num_workers = int(rt_cfg.get("num_workers", 4))
    train_batch_size = int(dl_cfg.get("train_batch_size", 8))
    eval_batch_size = int(dl_cfg.get("eval_batch_size", 16))
    use_sampler = bool(dl_cfg.get("use_weighted_sampler", True))
    pin_memory = bool(dl_cfg.get("pin_memory", True))

    collate = lambda batch: document_collate_fn(batch, max_seq_length=seq_len)  # noqa: E731

    train_ds = DocumentDataset(train_manifest, config, split="train")
    sampler = _build_weighted_sampler(train_ds) if use_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = _eval_loader(
        val_manifest, config, "val", eval_batch_size, num_workers, pin_memory, collate
    )
    test_loader = _eval_loader(
        test_manifest, config, "test", eval_batch_size, num_workers, pin_memory, collate
    )

    logger.info(
        "DataLoaders | train=%d (%d batches) | val=%d | test=%d",
        len(train_ds),
        len(train_loader),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    return train_loader, val_loader, test_loader
