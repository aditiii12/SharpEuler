#!/usr/bin/env python3
"""
order_eval_vlmeval_fixed.py

Evaluate BOTH modality orderings for a single (model, dataset TSV) pair using a
VLMEvalKit-configured model and write:
  - per_example.jsonl
  - summary.json

Key fixes vs earlier draft:
- supports inline/base64 images in TSVs like MMStar
- materializes every image to a real temp PNG path, so wrappers that expect paths work
- saves one sanity image for visual inspection
- prints one example prompt + message ordering for BOTH image-first and question-first

Usage
-----
python order_eval_vlmeval_fixed.py \
  --model "InternVL2-8B" \
  --tsv ~/LMUData/MMStar.tsv \
  --img_root ~/LMUData \
  --out_dir outputs/order_eval_matrix/internvl2_8b/mmstar \
  --max_samples 0 \
  --debug
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

LETTERS = "ABCD"


# ---------------------------
# Generic helpers
# ---------------------------

def normalize_letter(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"\b([ABCD])\b", s.upper())
    if m:
        return m.group(1)
    if s in {"0", "1", "2", "3"}:
        return LETTERS[int(s)]
    return None


def parse_letter_from_output(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    up = s.upper()

    patterns = [
        r"ANSWER\s*[:：]\s*([ABCD])\b",
        r"OPTION\s*([ABCD])\b",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ]
    for p in patterns:
        m = re.search(p, up)
        if m:
            return m.group(1)

    m = re.search(r"([ABCD])", up)
    return m.group(1) if m else None


def bool_mean(xs: List[bool]) -> float:
    return (sum(1 for x in xs if x) / len(xs)) / len(xs) if xs else float("nan")


def truncate_text(s: str, n: int = 220) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "...<truncated>"


# ---------------------------
# TSV column detection
# ---------------------------

def detect_image_col(df: pd.DataFrame) -> str:
    candidates = ["image", "image_path", "img", "img_path", "image_file", "filename", "file_name"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    for c in df.columns:
        if "image" in c.lower():
            return c
    raise ValueError(f"Could not detect image column from columns={list(df.columns)}")


def detect_question_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["prompt", "question", "query", "instruction", "text"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def detect_answer_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["answer", "correct", "label", "gt", "target"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def detect_option_cols(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    exact = {c.lower(): c for c in df.columns}
    for letter in LETTERS:
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
    q = str(row[question_col]).strip() if question_col is not None else ""
    if option_cols:
        lines = [q, ""]
        for L in LETTERS:
            if L in option_cols:
                lines.append(f"{L}. {str(row[option_cols[L]]).strip()}")
        lines.append("")
        lines.append("Answer with a single letter: A, B, C, or D.")
        return "\n".join(lines).strip()

    if q:
        if "Answer with" not in q:
            q += "\n\nAnswer with a single letter: A, B, C, or D."
        return q.strip()

    raise ValueError("Could not build prompt: no question/prompt column and no option columns found.")


# ---------------------------
# Image loading / materialization
# ---------------------------

def clean_image_field(x: Any) -> str:
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    return s.strip()


def is_inline_image_payload(s: str) -> bool:
    if s.startswith("data:image/"):
        return True
    if s.startswith("/9j/"):  # common JPEG base64 prefix
        return True
    if len(s) > 500:
        return True
    return False


def try_path_resolution(s: str, img_root: str) -> Optional[Path]:
    p = Path(s)
    if p.exists():
        return p
    p2 = Path(img_root) / s
    if p2.exists():
        return p2
    return None


def load_image_from_field(image_field: Any, img_root: str) -> Tuple[Image.Image, str]:
    """
    Returns:
      image_pil, storage_type
    storage_type in {"path", "inline_base64", "data_url"}
    """
    s = clean_image_field(image_field)

    resolved = try_path_resolution(s, img_root)
    if resolved is not None:
        return Image.open(resolved).convert("RGB"), "path"

    if s.startswith("data:image/"):
        payload = s.split(",", 1)[1]
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB"), "data_url"

    if is_inline_image_payload(s):
        raw = base64.b64decode(s)
        return Image.open(io.BytesIO(raw)).convert("RGB"), "inline_base64"

    raise FileNotFoundError(f"Could not resolve image as path or inline payload. Prefix={s[:80]!r}")


def materialize_temp_image(img: Image.Image, tmp_dir: Path, row_idx: int) -> str:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"row_{row_idx:06d}.png"
    img.save(out_path)
    return str(out_path)


# ---------------------------
# Model registry / calling
# ---------------------------

def maybe_load_registry():
    try:
        from vlmeval.config import supported_VLM
        return supported_VLM
    except Exception as e:
        raise RuntimeError(
            "Failed to import supported_VLM from vlmeval.config. "
            "Run this script from inside your VLMEvalKit repo / env."
        ) from e


def list_candidate_methods(model) -> List[str]:
    cands = ["generate", "chat", "generate_inner", "infer", "predict", "__call__"]
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


def try_call_model(model, messages, prompt: str, img_path: str, pil_img: Image.Image, debug: bool = False) -> str:
    """
    Tries several common VLMEval wrapper calling conventions.
    """
    attempts = []

    # Message-style APIs
    attempts.append(("generate(message=...)", lambda: model.generate(message=messages)))
    attempts.append(("generate(messages=...)", lambda: model.generate(messages=messages)))
    attempts.append(("chat(message=...)", lambda: model.chat(message=messages)))
    attempts.append(("chat(messages=...)", lambda: model.chat(messages=messages)))
    attempts.append(("generate_inner(message=...)", lambda: model.generate_inner(message=messages)))
    attempts.append(("infer(message=...)", lambda: model.infer(message=messages)))
    attempts.append(("__call__(message=...)", lambda: model(message=messages)))

    # prompt + path
    attempts.append(("generate(prompt=..., image=path)", lambda: model.generate(prompt=prompt, image=img_path)))
    attempts.append(("chat(prompt=..., image=path)", lambda: model.chat(prompt=prompt, image=img_path)))

    # prompt + PIL
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
    tmp_img_dir = out_dir / "_tmp_images"

    print("[DEBUG] ==================================================", flush=True)
    print("[DEBUG] order_eval_vlmeval_fixed.py", flush=True)
    print(f"[DEBUG] model={args.model}", flush=True)
    print(f"[DEBUG] tsv={args.tsv}", flush=True)
    print(f"[DEBUG] img_root={args.img_root}", flush=True)
    print(f"[DEBUG] out_dir={out_dir}", flush=True)
    print(f"[DEBUG] max_samples={args.max_samples}", flush=True)
    print("[DEBUG] ==================================================", flush=True)

    supported_VLM = maybe_load_registry()
    if args.model not in supported_VLM:
        close = [k for k in supported_VLM.keys() if args.model.lower() in k.lower() or k.lower() in args.model.lower()]
        raise KeyError(f"Model {args.model!r} not found in supported_VLM. Close matches: {close[:10]}")

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
    print(f"  image_col    = {image_col}", flush=True)
    print(f"  question_col = {question_col}", flush=True)
    print(f"  answer_col   = {answer_col}", flush=True)
    print(f"  option_cols  = {option_cols}", flush=True)

    if len(df) == 0:
        raise ValueError("Empty TSV")

    if args.max_samples and args.max_samples > 0:
        df = df.iloc[:args.max_samples].copy()
        print(f"[DEBUG] truncated to first {len(df)} rows", flush=True)

    # -------- first example sanity --------
    row0 = df.iloc[0]
    prompt0 = build_prompt(row0, question_col, option_cols)
    img0_pil, img0_type = load_image_from_field(row0[image_col], args.img_root)
    img0_tmp = materialize_temp_image(img0_pil, tmp_img_dir, 0)
    sanity_img_path = out_dir / "sanity_first_example.png"
    img0_pil.save(sanity_img_path)

    img0_msg = make_messages(img0_tmp, prompt0, "image_first")
    qf0_msg = make_messages(img0_tmp, prompt0, "question_first")

    print("[DEBUG] first-example sanity:", flush=True)
    print(f"  raw image field prefix = {truncate_text(str(row0[image_col]), 120)}", flush=True)
    print(f"  image storage type     = {img0_type}", flush=True)
    print(f"  materialized path      = {img0_tmp}", flush=True)
    print(f"  temp image exists?     = {Path(img0_tmp).exists()}", flush=True)
    print(f"  saved sanity image     = {sanity_img_path}", flush=True)
    print(f"  sanity image size      = {img0_pil.size}", flush=True)
    print("  prompt preview         =", flush=True)
    print("   " + truncate_text(prompt0, 500), flush=True)
    print("  image-first message    =", flush=True)
    print("   " + repr(img0_msg), flush=True)
    print("  question-first message =", flush=True)
    print("   " + repr(qf0_msg), flush=True)

    # -------- evaluation loop --------
    per_example_path = out_dir / "per_example.jsonl"
    img_ok: List[bool] = []
    qf_ok: List[bool] = []
    disagreements: List[bool] = []
    img_unparsed = 0
    qf_unparsed = 0
    n_skipped = 0

    with open(per_example_path, "w") as fout:
        for idx, row in df.iterrows():
            try:
                pil_img, storage_type = load_image_from_field(row[image_col], args.img_root)
                img_path = materialize_temp_image(pil_img, tmp_img_dir, int(idx))
            except Exception as e:
                n_skipped += 1
                print(f"[WARN] failed to load image at row={idx}: {type(e).__name__}: {e}", flush=True)
                continue

            prompt = build_prompt(row, question_col, option_cols)
            gold = normalize_letter(row[answer_col]) if answer_col is not None else None

            img_msg = make_messages(img_path, prompt, "image_first")
            qf_msg = make_messages(img_path, prompt, "question_first")

            try:
                img_out = try_call_model(model, img_msg, prompt, img_path, pil_img, debug=(args.debug and idx < 2))
            except Exception as e:
                img_out = f"[ERROR] {type(e).__name__}: {e}"
                if args.debug:
                    print(f"[DEBUG] image_first failed at row={idx}", flush=True)
                    traceback.print_exc()

            try:
                qf_out = try_call_model(model, qf_msg, prompt, img_path, pil_img, debug=(args.debug and idx < 2))
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
                "image_storage_type": storage_type,
                "materialized_image_path": img_path,
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
                print(f"[DEBUG]   img_out={repr(str(img_out)[:220])}", flush=True)
                print(f"[DEBUG]   qf_out ={repr(str(qf_out)[:220])}", flush=True)

    img_acc = bool_mean(img_ok) if img_ok else float("nan")
    qf_acc = bool_mean(qf_ok) if qf_ok else float("nan")
    gap = (img_acc - qf_acc) if img_ok and qf_ok else float("nan")
    disagr = bool_mean(disagreements) if disagreements else float("nan")

    summary = {
        "model": args.model,
        "dataset": Path(args.tsv).stem,
        "img_first_acc": None if pd.isna(img_acc) else round(float(img_acc), 6),
        "qf_acc": None if pd.isna(qf_acc) else round(float(qf_acc), 6),
        "gap_img_minus_qf": None if pd.isna(gap) else round(float(gap), 6),
        "disagreement_rate": None if pd.isna(disagr) else round(float(disagr), 6),
        "n_eval": int(len(img_ok)) if img_ok else 0,
        "n_skipped": int(n_skipped),
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
    print(f"[DEBUG] per-example written to: {per_example_path}", flush=True)
    print(f"[DEBUG] summary written to   : {summary_path}", flush=True)
    print(f"[DEBUG] sanity image written to: {sanity_img_path}", flush=True)
    print("[DEBUG] ==========================================\n", flush=True)


if __name__ == "__main__":
    main()
