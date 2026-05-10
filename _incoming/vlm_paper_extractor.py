"""
vlm_paper_extractor.py
======================

This module implements a minimalistic information extractor for vision-language
model (VLM) papers.  Given the raw text of a paper (for example the
body of an arXiv preprint or the parsed output of a PDF), the extractor
attempts to identify a handful of high‑level attributes commonly used in
VLM taxonomies:

* **model** – a candidate name for the system described in the paper.  A
  simple heuristic picks the first capitalised token from the title or
  prominently formatted text.
* **release_year** – the first four‑digit year observed in the text.  This
  roughly corresponds to the year of publication or first release.
* **architecture** – rough classification of the backbone architecture
  (e.g. "Transformer", "Diffusion", "CNN").  The implementation scans
  for indicative keywords and returns all hits.
* **tasks** – a set of downstream tasks addressed by the paper, such as
  visual question answering (VQA), captioning, retrieval, OCR/document
  understanding, etc.  Tasks are detected via regex patterns defined in
  the `TASK_PATTERNS` table.
* **techniques** – a set of key training or modelling techniques (e.g.
  contrastive learning, diffusion, masked autoencoders, LoRA, etc.).  The
  extractor uses a dictionary of keywords to populate this field.
* **fusion_techniques** – mechanisms used to combine modalities, such as
  early fusion, late fusion or cross‑modal fusion.  Keywords are matched
  case‑insensitively.
* **attention_types** – mentions of attention mechanisms like
  self‑attention or cross‑attention.
* **novelty** – a short snippet capturing the claimed novelty.  The
  extractor looks for sentences containing words like "propose" or
  "introduce" and returns the first such sentence if found.

The heuristics used here are deliberately simple and rule‑based; they
should be viewed as a starting point rather than a complete taxonomy.
They do not depend on external APIs or large language models and run
entirely offline.  If more sophisticated extraction is needed (e.g.
involving fine‑tuned language models), this script can serve as a
baseline or be extended accordingly.

Usage:

    python vlm_paper_extractor.py /path/to/paper.txt

This will read the file as plain text and emit a JSON object with
extracted fields.  If you have a PDF file, convert it to text first
using the provided PDF reading service (see README for details).
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

# Patterns for detecting VLM tasks.  Each entry maps a canonical task
# label to a list of regex patterns.  All matches are case‑insensitive.
TASK_PATTERNS: Dict[str, List[str]] = {
    "visual_question_answering": [
        r"\bVQA\b",
        r"visual question answering",
        r"OK[- ]VQA",
        r"TextVQA",
        r"GQA",
        r"VizWiz"
    ],
    "captioning": [
        r"\bcaption(?:ing)?\b",
        r"COCO Caption",
        r"Flickr30k"
    ],
    "retrieval": [
        r"\bretrieval\b",
        r"retrieving",
        r"retriever"
    ],
    "classification": [
        r"\bclassification\b",
        r"classify"
    ],
    "ocr": [
        r"\bOCR\b",
        r"document understanding",
        r"DocVQA",
        r"TextCaps",
        r"COCO-Text",
        r"SVT",
        r"IC13",
        r"IIIT5K"
    ],
    "chart_table": [
        r"ChartQA",
        r"table question answering",
        r"chart reasoning"
    ],
    "science_question_answering": [
        r"ScienceQA"
    ],
    "massive_multitask": [
        r"MMMU"
    ]
}

# Patterns for techniques.  Each keyword maps to a list of regex patterns
# used to detect that technique in a paper.  These lists are not
# exhaustive but cover many common VLM building blocks.
TECHNIQUE_PATTERNS: Dict[str, List[str]] = {
    "contrastive": [r"contrastive", r"InfoNCE", r"CLIP", r"SigLIP"],
    "masked_modeling": [r"masked language modeling", r"MAE", r"MIM"],
    "generative": [r"diffusion", r"autoregressive", r"VQ[- ]?VAE", r"VQ[- ]?GAN"],
    "pretrained_backbone": [r"BLIP[- ]?2", r"Q[- ]?Former", r"LoRA", r"adapter"],
    "optimization": [r"instruction[- ]tuning", r"RLHF", r"SNR", r"scheduler"],
    "data_curation": [r"LAION", r"DataComp", r"filtering", r"synthetic captions"],
    "evaluation": [r"Winoground", r"ARO", r"TIFA", r"hallucination", r"red team"],
}

# Separate dictionaries for fusion and attention.  Each key represents
# a specific mechanism with its own patterns.
FUSION_PATTERNS: Dict[str, List[str]] = {
    "late_fusion": [r"late fusion"],
    "early_fusion": [r"early fusion"],
    "cross_modal_fusion": [r"cross[- ]modal fusion", r"cross[- ]modal integration"],
    "fusion_module": [r"fusion module"]
}

ATTENTION_PATTERNS: Dict[str, List[str]] = {
    "cross_attention": [r"cross[- ]attention"],
    "self_attention": [r"self[- ]attention"],
    "multi_head_attention": [r"multi[- ]head attention"]
}

# Separate dictionary for architectures.  Keys are human‑readable
# architecture labels and the associated patterns to detect them.
ARCHITECTURE_PATTERNS: Dict[str, List[str]] = {
    "Transformer": [r"Transformer"],
    "Diffusion": [r"diffusion"],
    "CNN": [r"Convolutional", r"CNN"],
    "GPT": [r"GPT"],
    "LSTM": [r"LSTM"]
}

def extract_model_name(text: str) -> Optional[str]:
    """
    Heuristically identify a candidate model name from the paper text.

    This implementation looks at the beginning of the document for
    capitalised words or identifiers that may denote the model name.  If
    multiple uppercase tokens appear consecutively (e.g. "LLaVA 2"), the
    function returns the first one.  If no such token exists, returns
    ``None``.
    """
    # Extract the first line (title) and tokenise
    first_line = text.split("\n", 1)[0]
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-\.]{2,}\b", first_line)
    if tokens:
        return tokens[0]
    return None

def extract_release_year(text: str) -> Optional[str]:
    """
    Find the earliest year (four digits) mentioned in the text.

    Returns the first match as a string or ``None`` if none found.
    """
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None

def extract_items(text: str, pattern_dict: Dict[str, List[str]]) -> List[str]:
    """
    Return a sorted list of keys from ``pattern_dict`` whose any pattern
    matches the ``text``.  Matching is case‑insensitive.  Keys are
    deduplicated and sorted alphabetically.
    """
    hits = []
    for key, patterns in pattern_dict.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                hits.append(key)
                break
    return sorted(set(hits))

def extract_novelty_sentences(text: str) -> Optional[str]:
    """
    Extract a short description of the paper's novelty.

    The heuristic searches for sentences containing verbs such as
    "propose", "introduce", "present" or "develop" within the first
    3000 characters.  The first such sentence is returned, stripped of
    leading/trailing whitespace.  If none is found, returns ``None``.
    """
    first_chunk = text[:3000]
    # Split into sentences using simple period delimiting
    sentences = re.split(r"(?<=[.!?])\s+", first_chunk)
    for sent in sentences:
        if re.search(r"\b(propose|introduce|present|develop|analyze|analyse)\b", sent, re.IGNORECASE):
            return sent.strip()
    return None

def extract_vlm_metadata(text: str) -> Dict[str, Optional[str]]:
    """
    Extract high‑level VLM metadata from the supplied text.

    Args:
        text: The raw text of a paper, including title and abstract.

    Returns:
        A dictionary containing the extracted fields.
    """
    result: Dict[str, Optional[str]] = {}
    # Model name
    model_name = extract_model_name(text)
    result["model"] = model_name
    # Release year
    result["release_year"] = extract_release_year(text)
    # Architecture(s)
    result["architecture"] = extract_items(text, ARCHITECTURE_PATTERNS)
    # Tasks
    tasks = extract_items(text, TASK_PATTERNS)
    result["tasks"] = tasks
    # Techniques (excluding architecture and fusion/attention)
    technique_keys = {k: v for k, v in TECHNIQUE_PATTERNS.items() if k not in {"fusion", "attention"}}
    techniques = extract_items(text, technique_keys)
    result["techniques"] = techniques
    # Fusion techniques (specific labels)
    result["fusion_techniques"] = extract_items(text, FUSION_PATTERNS)
    # Attention types (specific labels)
    result["attention_types"] = extract_items(text, ATTENTION_PATTERNS)
    # Novelty sentence
    result["novelty"] = extract_novelty_sentences(text)
    return result

def read_text_file(path: str) -> str:
    """
    Read a file as UTF‑8 text.  Returns the contents as a single
    string.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VLM taxonomy fields from a paper.")
    parser.add_argument("input", help="Path to a plain text file containing the paper's content.")
    parser.add_argument("-o", "--output", help="Optional output JSON file path.")
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")
    text = read_text_file(args.input)
    metadata = extract_vlm_metadata(text)
    json_output = json.dumps(metadata, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            out_f.write(json_output)
        print(f"Metadata written to {args.output}")
    else:
        print(json_output)

if __name__ == "__main__":
    main()
