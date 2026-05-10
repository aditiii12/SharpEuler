#!/usr/bin/env python3
"""
eval_cmmmu_val.py

Evaluate CMMMU_VAL predictions saved in an .xlsx (typically from VLMEvalKit-style runs).

Features
- Robust parsing of predicted option letter (A/B/C/D/E/...) from messy strings
- Handles common formats: "A", "A.", "Answer: A", "(A)", "Option A", "A) ...", "A. ..."
- Computes overall accuracy and per-category breakdown if columns exist
- Writes a JSON + CSV summary next to the input file (or in --out-dir)
- Optionally prints sample wrong rows

Typical usage:
    python eval_cmmmu_val.py /path/to/WeThink-Qwen2.5VL-7B_CMMMU_VAL.xlsx

If your sheet name is unknown, the script auto-selects the first sheet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -------- parsing helpers --------
_LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)  # allow up to H just in case
_LETTER_PUNCT_RE = re.compile(r"\b([A-H])\s*[\.\):]\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"(?:final\s*answer|answer|option|choice)\s*[:\-]?\s*([A-H])\b", re.IGNORECASE)


def normalize_choice(x) -> Optional[str]:
    """
    Normalize a prediction/answer cell to a single uppercase option letter (A-H).
    Returns None if parsing fails or x is NaN/empty.
    """
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None

    # If cell is exactly a letter
    if len(s) == 1 and s.upper() in "ABCDEFGH":
        return s.upper()

    # Prefer explicit "Answer: X" / "Final answer X" patterns
    m = _ANSWER_RE.search(s)
    if m:
        return m.group(1).upper()

    # Then look for "A." / "A)" / "A:"
    m = _LETTER_PUNCT_RE.search(s)
    if m:
        return m.group(1).upper()

    # Then look for a standalone letter token, but try to avoid picking letters from words
    m = _LETTER_RE.search(s)
    if m:
        return m.group(1).upper()

    return None


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fuzzy: contains
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def load_xlsx(path: Path, sheet: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    xl = pd.ExcelFile(path)
    sheet_name = sheet or xl.sheet_names[0]
    df = xl.parse(sheet_name)
    return df, sheet_name


# -------- evaluation --------
def evaluate(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    id_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
) -> dict:
    out = {}

    gt = df[gt_col].apply(normalize_choice)
    pred = df[pred_col].apply(normalize_choice)

    valid = gt.notna()
    out["n_total_rows"] = int(len(df))
    out["n_with_gt"] = int(valid.sum())

    correct = (gt == pred) & valid
    out["n_correct"] = int(correct.sum())
    out["accuracy"] = float(out["n_correct"] / out["n_with_gt"]) if out["n_with_gt"] > 0 else float("nan")

    # coverage: how often we produced a parseable prediction
    pred_parsed = pred.notna() & valid
    out["prediction_parse_rate"] = float(pred_parsed.mean()) if out["n_with_gt"] > 0 else float("nan")

    # confusion (for A-D only) if possible
    letters = sorted({c for c in gt.dropna().unique().tolist() + pred.dropna().unique().tolist() if c is not None})
    if letters:
        cm = pd.crosstab(gt[valid], pred[valid], dropna=False).reindex(index=letters, columns=letters, fill_value=0)
        out["confusion_matrix"] = cm.to_dict()

    # group breakdowns
    group_cols = group_cols or []
    breakdowns = {}
    for g in group_cols:
        if g in df.columns:
            tmp = df.loc[valid, [g]].copy()
            tmp["gt"] = gt[valid].values
            tmp["pred"] = pred[valid].values
            tmp["correct"] = (tmp["gt"] == tmp["pred"]).astype(int)
            agg = tmp.groupby(g, dropna=False)["correct"].agg(["count", "mean"]).reset_index()
            agg.rename(columns={"count": "n", "mean": "acc"}, inplace=True)
            breakdowns[g] = agg.to_dict(orient="records")
    if breakdowns:
        out["breakdowns"] = breakdowns

    # wrong examples
    wrong_idx = df.index[valid & (~correct)].tolist()
    out["n_wrong"] = int(len(wrong_idx))
    out["_internal_wrong_idx"] = wrong_idx  # removed before saving if requested

    # attach a minimal row-level table for easy exporting
    row_table = df.copy()
    row_table["_gt_norm"] = gt
    row_table["_pred_norm"] = pred
    row_table["_correct"] = correct.astype(int)
    out["_row_table"] = row_table

    # include id column if present
    if id_col and id_col in df.columns:
        out["id_col"] = id_col

    return out


def write_outputs(summary: dict, input_path: Path, out_dir: Optional[Path], prefix: Optional[str], keep_internal: bool):
    out_dir = out_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = prefix or input_path.stem
    json_path = out_dir / f"{base}.eval.json"
    csv_path = out_dir / f"{base}.eval_rows.csv"

    # row table
    row_table = summary.pop("_row_table")
    wrong_idx = summary.pop("_internal_wrong_idx")

    # write row-level CSV
    row_table.to_csv(csv_path, index=False)

    # remove any internal keys unless asked to keep
    if not keep_internal:
        # nothing extra to remove right now
        pass
    else:
        summary["wrong_idx"] = wrong_idx

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return json_path, csv_path


def main():
    p = argparse.ArgumentParser(description="Evaluate CMMMU_VAL .xlsx outputs.")
    p.add_argument("xlsx", type=str, help="Path to .xlsx file containing predictions.")
    p.add_argument("--sheet", type=str, default=None, help="Sheet name (default: first sheet).")
    p.add_argument("--gt-col", type=str, default=None, help="Ground-truth column name (auto-detect if omitted).")
    p.add_argument("--pred-col", type=str, default=None, help="Prediction column name (auto-detect if omitted).")
    p.add_argument("--id-col", type=str, default=None, help="ID/index column name (optional).")
    p.add_argument("--group-cols", type=str, default="category,l2_category,bench",
                   help="Comma-separated columns to report breakdowns for (if present).")
    p.add_argument("--out-dir", type=str, default=None, help="Directory to write outputs (default: alongside xlsx).")
    p.add_argument("--prefix", type=str, default=None, help="Prefix for output files (default: xlsx stem).")
    p.add_argument("--show-wrong", type=int, default=10, help="Print up to N wrong examples.")
    p.add_argument("--keep-internal", action="store_true", help="Include internal indices in JSON output.")
    args = p.parse_args()

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Not found: {xlsx_path}")

    df, sheet_name = load_xlsx(xlsx_path, sheet=args.sheet)

    # auto detect columns
    gt_col = args.gt_col or find_col(df, ["answer", "gt", "gold", "label", "correct_answer"])
    pred_col = args.pred_col or find_col(df, ["prediction", "pred", "model_output", "output"])

    if gt_col is None:
        raise ValueError(f"Could not auto-detect gt column in columns={list(df.columns)}. Use --gt-col.")
    if pred_col is None:
        raise ValueError(f"Could not auto-detect prediction column in columns={list(df.columns)}. Use --pred-col.")

    group_cols = [c.strip() for c in (args.group_cols or "").split(",") if c.strip()]

    summary = evaluate(df, gt_col=gt_col, pred_col=pred_col, id_col=args.id_col, group_cols=group_cols)

    # print summary
    acc = summary["accuracy"]
    print(f"File: {xlsx_path}")
    print(f"Sheet: {sheet_name}")
    print(f"GT column: {gt_col} | Pred column: {pred_col}")
    print(f"Rows with GT: {summary['n_with_gt']} / {summary['n_total_rows']}")
    print(f"Prediction parse rate: {summary['prediction_parse_rate']:.4f}")
    print(f"Accuracy: {acc:.4f}" if not np.isnan(acc) else "Accuracy: NaN")

    # print breakdowns (compact)
    if "breakdowns" in summary:
        for g, rows in summary["breakdowns"].items():
            # show top 10 biggest groups by n
            dfb = pd.DataFrame(rows).sort_values("n", ascending=False)
            print(f"\nBreakdown: {g} (top 10 by n)")
            print(dfb.head(10).to_string(index=False))

    # show wrong examples
    n_show = max(0, int(args.show_wrong))
    if n_show > 0:
        row_table = summary["_row_table"]
        wrong = row_table[row_table["_correct"] == 0].copy()
        if len(wrong) > 0:
            print(f"\nSample wrong rows (showing up to {n_show}):")
            cols_to_show = []
            for c in ["index", "id", "qid", "question", "answer", "prediction", "_gt_norm", "_pred_norm"]:
                if c in wrong.columns:
                    cols_to_show.append(c)
            if not cols_to_show:
                cols_to_show = wrong.columns[:8].tolist()
            print(wrong[cols_to_show].head(n_show).to_string(index=False))
        else:
            print("\nNo wrong rows 🎉")

    # write outputs
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    json_path, csv_path = write_outputs(summary, xlsx_path, out_dir, args.prefix, args.keep_internal)
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
