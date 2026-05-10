"""
widget_vlm.py
=================

This module provides simple utilities to maintain a dataset of vision–language
models (VLMs) and to extract model‐related information from research papers.

The dataset is stored in a CSV file (`vlm_dataset.csv`) with the following
columns:

    model_name, year, vision_encoder, language_model,
    model_size_parameters, fusion_strategy,
    novel_technique, citation

The functions in this module allow you to load the dataset into memory,
add new entries, save changes, and (optionally) parse PDF files to
auto‑extract candidate fields for new entries.  Because extracting
structured information from research papers automatically is still an
open research problem, the parsing functions here use simple
heuristics to highlight lines that mention relevant components (e.g.,
"vision encoder", "language model", "ViT" etc.).  You should manually
validate and complete the extracted information before inserting into
the dataset.

Example usage:

    from widget_vlm import VLMDataSet

    # load existing dataset
    ds = VLMDataSet("vlm_dataset.csv")

    # parse a new paper and inspect the extracted snippets
    candidates = ds.parse_paper("some_model_paper.pdf")
    for line in candidates:
        print(line)

    # after manually determining the correct fields, add to dataset
    ds.add_entry({
        "model_name": "SomeModel",
        "year": "2024",
        "vision_encoder": "ViT‑Huge",
        "language_model": "LLama2‑13B",
        "model_size_parameters": "13B",
        "fusion_strategy": "Cross‑attention",
        "novel_technique": "Dual gating",
        "citation": "Paper citation here"
    })
    ds.save()

When working in this environment you can run the module directly to
preview candidate lines extracted from a PDF:

    python widget_vlm.py path/to/paper.pdf

This will print lines containing keywords such as "vision encoder",
"language model", "ViT", etc., and is intended to assist you in
manually filling out the dataset.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore


@dataclass
class VLMDataSet:
    """A simple dataset manager for vision‑language models."""

    csv_path: str
    entries: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Load dataset from CSV if it exists
        if os.path.exists(self.csv_path):
            self.load()

    def load(self) -> None:
        """Load entries from the CSV file into memory."""
        self.entries.clear()
        with open(self.csv_path, "r", newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append({k: v.strip() for k, v in row.items()})

    def save(self) -> None:
        """Save the current entries back to the CSV file."""
        if not self.entries:
            raise ValueError("No entries to save.")
        fieldnames = list(self.entries[0].keys())
        with open(self.csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(entry)

    def add_entry(self, data: Dict[str, str]) -> None:
        """Append a new entry to the dataset.

        Args:
            data: A dictionary containing all required fields.
        """
        required_fields = {
            "model_name",
            "year",
            "vision_encoder",
            "language_model",
            "model_size_parameters",
            "fusion_strategy",
            "novel_technique",
            "citation",
        }
        missing = required_fields - data.keys()
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(sorted(missing))}")
        # Append the entry
        self.entries.append({field: data.get(field, "").strip() for field in required_fields})

    # ---------------------- Paper Parsing Utilities ---------------------- #
    # Regular expressions for capturing lines mentioning components
    KEYWORDS = [
        re.compile(r"vision encoder", re.IGNORECASE),
        re.compile(r"language model", re.IGNORECASE),
        re.compile(r"LLM", re.IGNORECASE),
        re.compile(r"ViT", re.IGNORECASE),
        re.compile(r"ResNet", re.IGNORECASE),
        re.compile(r"parameters", re.IGNORECASE),
        re.compile(r"B\b|M\b", re.IGNORECASE),  # matches 'B' or 'M' for billions/millions
        re.compile(r"cross[- ]?attention", re.IGNORECASE),
        re.compile(r"adapter", re.IGNORECASE),
        re.compile(r"query", re.IGNORECASE),
        re.compile(r"fusion", re.IGNORECASE),
    ]

    def parse_paper(self, pdf_path: str) -> List[str]:
        """Extract candidate lines from a PDF that mention VLM components.

        This function opens a PDF file and extracts all text.  It then
        returns a list of lines containing any of the keywords defined
        above.  These lines may include information about the vision
        encoder, language model, parameter counts, fusion strategy, etc.

        Args:
            pdf_path: The path to the PDF file to parse.

        Returns:
            A list of strings, each representing a candidate line from
            the paper containing relevant keywords.
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required to parse PDFs. Please install it via pip.")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            # Extract text; suppress footnotes/annotation extraction
            page_text = page.get_text("text")
            text += page_text + "\n"
        # Split into lines and filter for keyword matches
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidates: List[str] = []
        for line in lines:
            # Check if any keyword matches
            if any(pattern.search(line) for pattern in self.KEYWORDS):
                # Filter out extremely long lines (>300 characters) which are
                # unlikely to be meaningful on their own.
                if len(line) <= 300:
                    candidates.append(line)
        return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse a PDF paper and print candidate lines containing VLM keywords.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--dataset", help="Optional path to existing dataset CSV", default=None)
    args = parser.parse_args()

    if args.dataset:
        ds = VLMDataSet(args.dataset)
    else:
        ds = VLMDataSet("vlm_dataset.csv")
    try:
        candidates = ds.parse_paper(args.pdf_path)
    except Exception as e:
        print(f"Error parsing paper: {e}")
        raise SystemExit(1)
    print(f"Found {len(candidates)} candidate lines with VLM-related keywords:\n")
    for line in candidates:
        print(f"- {line}")