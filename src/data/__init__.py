"""
src/data
========
Data layer for the fake-document-detector pipeline.

Public surface
--------------
ingestion      → DocumentIngester
preprocessing  → DocumentPreprocessor
dataset        → DocumentDataset
dataloader     → build_dataloaders
"""

from src.data.dataloader import build_dataloaders
from src.data.dataset import DocumentDataset
from src.data.ingestion import DocumentIngester
from src.data.preprocessing import DocumentPreprocessor

__all__ = [
    "DocumentIngester",
    "DocumentPreprocessor",
    "DocumentDataset",
    "build_dataloaders",
]
