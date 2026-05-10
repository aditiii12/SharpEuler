#!/usr/bin/env python3
"""
order_eval_vlmeval.py

Evaluate BOTH modality orderings for a single (model, dataset TSV) pair using a
VLMEvalKit-configured model and write:
  - per_example.jsonl
  - summary.json

Designed for MCQ-style TSVs like MMStar / AI2D / RealWorldQA exports.

Usage
-----
python order_eval_vlmeval.py \
  --model "InternVL2-8B" \
  --tsv ~/LMUData/MMStar.tsv \
  --img_root ~/LMUData \
  --out_dir outputs/order_eval_matrix/internvl2_8b/mmstar \
  --max_samples 0 \
  --debug

Notes
-----
- This script tries several wrapper call styles because VLMEvalKit wrappers vary.
- It is intentionally verbose for the first few examples.
- It expects a model registry in vlmeval.config called supported_VLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

# ---------------------------
# Helpers
# ---------------------------

LETTER_SET = set(list("ABCD"))


def normalize_letter(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"\b([ABCD])\b", s.upper())
    if m:
        return m.group(1)
    # common integer labels
    if s in {"0", "1", "2", "3"}:
        return "ABCD"[int(s)]
    return None


def parse_letter_from_output(text: str) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()

    # strongest patterns first
    patterns = [
        r"answer\s*[:：]\s*([ABCD])\b",
        r"option\s*([ABCD])\b",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ]
    up = s.upper()
    for p in patterns:
        m = re.search(p, up)
        if m:
            return m.group(1)

    # fall back to first capital A-D if nothing else
    m = re.search(r"([ABCD])", up)
    if m:
        return m.group(1)
    return None


def bool_mean(xs: List[bool]) -> float:
    return (sum(1 for x in xs if x) / len(xs)) if xs else float("nan")


def safe_relpath(path: str, root: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(root) / path)


def detect_image_col(df: pd.DataFrame) -> str:
    candidates = [
        "image", "image_path", "img", "img_path", "image_file", "filename", "file_name"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    # fallback: anything containing "image"
    for c in df.columns:
        if "image" in c.lower():
            return c
    raise ValueError(f"Could not detect image column from columns={list(df.columns)}")


def detect_question_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "prompt", "question", "query", "instruction", "text"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def detect_answer_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "answer", "correct", "label", "gt", "target"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def detect_option_cols(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    exact = {c.lower(): c for c in df.columns}
    for letter in "ABCD":
        for cand in [
            letter,
            f"option_{letter.lower()}",
            f"option_{letter}",
            f"choice_{letter.lower()}",
            f"choice_{letter}",
        ]:
            if cand.lower() in exact:
                out[letter] = exact[cand.lower()]
                break
    return out


def build_prompt(row: pd.Series, question_col: Optional[str], option_cols: Dict[str, str]) -> str:
    if question_col is not None:
        q = str(row[question_col]).strip()
    else:
        q = ""

    if option_cols:
        lines = [q, ""]
        for L in "ABCD":
            if L in option_cols:
                lines.append(f"{L}. {str(row[option_cols[L]]).strip()}")
        lines.append("")
        lines.append("Answer with a single letter: A, B, C, or D.")
        return "\n".join(lines).strip()

    # If prompt is already fully formatted
    if q:
        if "Answer with" not in q:
            q = q + "\n\nAnswer with a single letter: A, B, C, or D."
        return q.strip()

    raise ValueError("Could not build prompt: no question/prompt column and no option columns found.")


def list_candidate_methods(model) -> List[str]:
    cands = [
        "generate", "chat", "generate_inner", "infer", "predict", "__call__"
    ]
    return [m for m in cands if hasattr(model, m)]


def make_messages(img_path: str, prompt: str, order: str):
    if order == "image_first":
        return [
            {"type": "image", "value": img_path},
            {"type": "text", "value": prompt},
        ]
    elif order == "question_first":
        return [
            {"type": "text", "value": prompt},
            {"type": "image", "value": img_path},
        ]
    else:
        raise ValueError(f"Unknown order={order}")


def try_call_model(model, messages, prompt: str, img_path: str, debug: bool = False) -> str:
    """
    Tries several common VLMEval wrapper calling conventions.
    """
    attempts = []

    # Common multimodal message styles
    attempts.append(("generate(message=...)", lambda: model.generate(message=messages)))
    attempts.append(("generate(messages=...)", lambda: model.generate(messages=messages)))
    attempts.append(("chat(message=...)", lambda: model.chat(message=messages)))
    attempts.append(("chat(messages=...)", lambda: model.chat(messages=messages)))
    attempts.append(("generate_inner(message=...)", lambda: model.generate_inner(message=messages)))
    attempts.append(("infer(message=...)", lambda: model.infer(message=messages)))
    attempts.append(("__call__(message=...)", lambda: model(message=messages)))

    # Some wrappers accept raw text + image path
    attempts.append(("generate(prompt=..., image=...)", lambda: model.generate(prompt=prompt, image=img_path)))
    attempts.append(("chat(prompt=..., image=...)", lambda: model.chat(prompt=prompt, image=img_path)))

    # Some wrappers accept PIL.Image
    pil_img = None
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception:
        pil_img = None

    if pil_img is not None:
        attempts.append(("generate(prompt=..., image=PIL)", lambda: model.generate(prompt=prompt, image=pil_img)))
        attempts.append(("chat(prompt=..., image=PIL)", lambda: model.chat(prompt=prompt, image=pil_img)))

    last_exc = None
    for name, fn in attempts:
        try:
            out = fn()
            if debug:
                print(f"[DEBUG] model call succeeded with {name}", flush=True)
            return str(out)
        except Exception as e:
            last_exc = e
            if debug:
                print(f"[DEBUG] model call failed with {name}: {type(e).__name__}: {e}", flush=True)

    raise RuntimeError(
        f"All wrapper call styles failed. Last error: {type(last_exc).__name__}: {last_exc}"
    )


def maybe_load_registry():
    try:
        from vlmeval.config import supported_VLM
        return supported_VLM
    except Exception as e:
        raise RuntimeError(
            "Failed to import supported_VLM from vlmeval.config. "
            "Run this script from inside your VLMEvalKit repo / env."
        ) from e


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Exact model name from vlmeval/config.py")
    ap.add_argument("--tsv", required=True, help="Path to dataset TSV")
    ap.add_argument("--img_root", required=True, help="Image root dir")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[DEBUG] ==================================================", flush=True)
    print("[DEBUG] order_eval_vlmeval.py", flush=True)
    print(f"[DEBUG] model={args.model}", flush=True)
    print(f"[DEBUG] tsv={args.tsv}", flush=True)
    print(f"[DEBUG] img_root={args.img_root}", flush=True)
    print(f"[DEBUG] out_dir={out_dir}", flush=True)
    print(f"[DEBUG] max_samples={args.max_samples}", flush=True)
    print("[DEBUG] ==================================================", flush=True)

    supported_VLM = maybe_load_registry()
    if args.model not in supported_VLM:
        close = [k for k in supported_VLM.keys() if args.model.lower() in k.lower() or k.lower() in args.model.lower()]
        raise KeyError(
            f"Model {args.model!r} not found in supported_VLM. Close matches: {close[:10]}"
        )

    print(f"[DEBUG] registry hit: {args.model}", flush=True)
    ctor = supported_VLM[args.model]
    print(f"[DEBUG] constructor type: {type(ctor)}", flush=True)

    print("[DEBUG] instantiating model ...", flush=True)
    model = ctor()
    print(f"[DEBUG] model instance type: {type(model)}", flush=True)
    print(f"[DEBUG] candidate methods: {list_candidate_methods(model)}", flush=True)

    print("[DEBUG] loading TSV ...", flush=True)
    df = pd.read_csv(args.tsv, sep="\t")
    print(f"[DEBUG] loaded rows={len(df)} cols={list(df.columns)}", flush=True)

    image_col = detect_image_col(df)
    question_col = detect_question_col(df)
    answer_col = detect_answer_col(df)
    option_cols = detect_option_cols(df)

    print("[DEBUG] detected columns:", flush=True)
    print(f"  image_col   = {image_col}", flush=True)
    print(f"  question_col= {question_col}", flush=True)
    print(f"  answer_col  = {answer_col}", flush=True)
    print(f"  option_cols = {option_cols}", flush=True)

    if len(df) == 0:
        raise ValueError("Empty TSV")

    if args.max_samples and args.max_samples > 0:
        df = df.iloc[:args.max_samples].copy()
        print(f"[DEBUG] truncated to first {len(df)} rows", flush=True)

    # first-row prompt sanity
    row0 = df.iloc[0]
    prompt0 = build_prompt(row0, question_col, option_cols)
    img0 = safe_relpath(str(row0[image_col]), args.img_root)

    print("[DEBUG] first-example sanity:", flush=True)
    print(f"  raw image path = {row0[image_col]}", flush=True)
    print(f"  resolved image = {img0}", flush=True)
    print(f"  image exists?  = {Path(img0).exists()}", flush=True)
    print("  prompt preview =", flush=True)
    print("  " + repr(prompt0[:500]), flush=True)

    per_example_path = out_dir / "per_example.jsonl"
    img_ok = []
    qf_ok = []
    disagreements = []
    img_unparsed = 0
    qf_unparsed = 0

    with open(per_example_path, "w") as fout:
        for idx, row in df.iterrows():
            img_path = safe_relpath(str(row[image_col]), args.img_root)
            prompt = build_prompt(row, question_col, option_cols)
            gold = normalize_letter(row[answer_col]) if answer_col is not None else None

            if not Path(img_path).exists():
                print(f"[WARN] missing image at row={idx}: {img_path}", flush=True)
                continue

            img_msg = make_messages(img_path, prompt, "image_first")
            qf_msg = make_messages(img_path, prompt, "question_first")

            try:
                img_out = try_call_model(model, img_msg, prompt, img_path, debug=(args.debug and idx < 2))
            except Exception as e:
                img_out = f"[ERROR] {type(e).__name__}: {e}"
                if args.debug:
                    print(f"[DEBUG] image_first failed at row={idx}", flush=True)
                    traceback.print_exc()

            try:
                qf_out = try_call_model(model, qf_msg, prompt, img_path, debug=(args.debug and idx < 2))
            except Exception as e:
                qf_out = f"[ERROR] {type(e).__name__}: {e}"
                if args.debug:
                    print(f"[DEBUG] question_first failed at row={idx}", flush=True)
                    traceback.print_exc()

            img_pred = parse_letter_from_output(img_out)
            qf_pred = parse_letter_from_output(qf_out)

            if img_pred is None:
                img_unparsed += 1
            if qf_pred is None:
                qf_unparsed += 1

            record = {
                "index": int(idx),
                "image_path": img_path,
                "gold": gold,
                "img_output": img_out,
                "qf_output": qf_out,
                "img_pred": img_pred,
                "qf_pred": qf_pred,
            }

            if gold is not None:
                record["img_correct"] = (img_pred == gold)
                record["qf_correct"] = (qf_pred == gold)
                img_ok.append(record["img_correct"])
                qf_ok.append(record["qf_correct"])

            if img_pred is not None and qf_pred is not None:
                record["disagree"] = (img_pred != qf_pred)
                disagreements.append(record["disagree"])

            fout.write(json.dumps(record) + "\n")

            if args.debug and idx < 5:
                print(f"[DEBUG] row={idx} gold={gold} | img_pred={img_pred} | qf_pred={qf_pred}", flush=True)
                print(f"[DEBUG]   img_out={repr(str(img_out)[:200])}", flush=True)
                print(f"[DEBUG]   qf_out ={repr(str(qf_out)[:200])}", flush=True)

    img_acc = bool_mean(img_ok) if img_ok else float("nan")
    qf_acc = bool_mean(qf_ok) if qf_ok else float("nan")
    gap = (img_acc - qf_acc) if (img_ok and qf_ok) else float("nan")
    disagr = bool_mean(disagreements) if disagreements else float("nan")

    summary = {
        "model": args.model,
        "dataset": Path(args.tsv).stem,
        "img_first_acc": None if pd.isna(img_acc) else round(float(img_acc), 6),
        "qf_acc": None if pd.isna(qf_acc) else round(float(qf_acc), 6),
        "gap_img_minus_qf": None if pd.isna(gap) else round(float(gap), 6),
        "disagreement_rate": None if pd.isna(disagr) else round(float(disagr), 6),
        "n_eval": int(len(img_ok)) if img_ok else int(len(df)),
        "img_unparsed": int(img_unparsed),
        "qf_unparsed": int(qf_unparsed),
        "notes": "",
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[DEBUG] ================= SUMMARY =================", flush=True)
    for k, v in summary.items():
        print(f"[DEBUG] {k}: {v}", flush=True)
    print("[DEBUG] per-example written to:", per_example_path, flush=True)
    print("[DEBUG] summary written to   :", summary_path, flush=True)
    print("[DEBUG] ==========================================\n", flush=True)


if __name__ == "__main__":
    main()
