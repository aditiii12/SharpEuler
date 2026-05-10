#!/usr/bin/env python3
"""
run_ordering_matrix.py

Automate:
1) run baseline ordering evaluation commands for a model/dataset matrix
2) parse the resulting shard JSON files
3) write / update a CSV with:
   model_short, model_id, family, dataset, img_first_acc, qf_acc,
   gap_img_minus_qf, disagreement_rate, n_eval, notes

This is intentionally runner-agnostic:
- each experiment provides a shell command to execute
- each experiment provides a result_dir where shard JSONs are written

Expected shard JSON shape (same as your ablation-style runners):
{
  "step0": [...],               # image-first correctness booleans
  "qf_step0": [...],            # question-first correctness booleans
  "meta": [{"img_pred": "...", "qf_pred": "..."}, ...]
}
or
{
  "disagreements": [...]
}

Usage:
    python run_ordering_matrix.py --config ordering_matrix_config.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

CSV_COLUMNS = [
    "model_short",
    "model_id",
    "family",
    "dataset",
    "img_first_acc",
    "qf_acc",
    "gap_img_minus_qf",
    "disagreement_rate",
    "n_eval",
    "notes",
]

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def mean_bool(xs: List[Any]) -> float:
    if not xs:
        return float("nan")
    vals = [1.0 if bool(x) else 0.0 for x in xs]
    return sum(vals) / len(vals)

def collect_shards(result_dir: str, json_glob: str = "gpu*.json") -> List[str]:
    paths = sorted(glob.glob(os.path.join(result_dir, json_glob)))
    paths = [p for p in paths if not p.endswith("_trace.jsonl")]
    return paths

def parse_metrics_from_result_dir(result_dir: str, json_glob: str = "gpu*.json") -> Dict[str, Any]:
    shard_paths = collect_shards(result_dir, json_glob=json_glob)
    if not shard_paths:
        raise FileNotFoundError(f"No shard JSONs found in {result_dir} matching {json_glob}")

    all_step0 = []
    all_qf_step0 = []
    all_disagreements = []
    n_meta = 0

    for path in shard_paths:
        d = load_json(path)

        step0 = d.get("step0", [])
        qf_step0 = d.get("qf_step0", [])
        all_step0.extend(step0)
        all_qf_step0.extend(qf_step0)

        if "disagreements" in d and d["disagreements"]:
            all_disagreements.extend([bool(x) for x in d["disagreements"]])
        else:
            for m in d.get("meta", []):
                if "img_pred" in m and "qf_pred" in m:
                    all_disagreements.append(m["img_pred"] != m["qf_pred"])
            n_meta += len(d.get("meta", []))

    if not all_step0 or not all_qf_step0:
        raise ValueError(
            f"Parsed shard JSONs in {result_dir}, but did not find non-empty step0/qf_step0 arrays."
        )

    img_acc = mean_bool(all_step0)
    qf_acc = mean_bool(all_qf_step0)
    gap = img_acc - qf_acc
    disagr = mean_bool(all_disagreements) if all_disagreements else float("nan")
    n_eval = len(all_step0)

    return {
        "img_first_acc": round(img_acc, 6),
        "qf_acc": round(qf_acc, 6),
        "gap_img_minus_qf": round(gap, 6),
        "disagreement_rate": round(disagr, 6) if disagr == disagr else "",
        "n_eval": n_eval,
    }

def run_command(cmd: str, workdir: str | None = None) -> None:
    print(f"\n[RUN] {cmd}", flush=True)
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=workdir,
        executable="/bin/bash",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}: {cmd}")

def upsert_csv(csv_path: str, row: Dict[str, Any]) -> None:
    csv_file = Path(csv_path)
    existing = []

    if csv_file.exists():
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

    key = (row["model_short"], row["dataset"])
    replaced = False
    for i, r in enumerate(existing):
        if (r["model_short"], r["dataset"]) == key:
            existing[i] = {k: row.get(k, "") for k in CSV_COLUMNS}
            replaced = True
            break

    if not replaced:
        existing.append({k: row.get(k, "") for k in CSV_COLUMNS})

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)

def validate_experiment(exp: Dict[str, Any]) -> None:
    required = ["model_short", "model_id", "family", "dataset", "result_dir", "run_command"]
    missing = [k for k in required if k not in exp]
    if missing:
        raise ValueError(f"Experiment missing required keys: {missing}\n{exp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to ordering_matrix_config.json")
    ap.add_argument("--only_model", default=None, help="Optional: run only one model_short")
    ap.add_argument("--only_dataset", default=None, help="Optional: run only one dataset")
    ap.add_argument("--skip_run", action="store_true", help="Only parse existing result dirs and update CSV")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    csv_path = cfg.get("csv_path", "modality_order_eval_results.csv")
    workdir = cfg.get("workdir", None)
    experiments = cfg.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in config.")

    for exp in experiments:
        validate_experiment(exp)

        if args.only_model and exp["model_short"] != args.only_model:
            continue
        if args.only_dataset and exp["dataset"] != args.only_dataset:
            continue

        cmd = exp["run_command"]
        if "REPLACE_ME" in cmd:
            print(f"[SKIP] Placeholder command for {exp['model_short']} / {exp['dataset']}")
            continue

        if not args.skip_run:
            run_command(cmd, workdir=workdir)

        metrics = parse_metrics_from_result_dir(
            exp["result_dir"],
            json_glob=exp.get("json_glob", "gpu*.json"),
        )

        row = {
            "model_short": exp["model_short"],
            "model_id": exp["model_id"],
            "family": exp["family"],
            "dataset": exp["dataset"],
            **metrics,
            "notes": exp.get("notes", ""),
        }
        upsert_csv(csv_path, row)
        print(f"[DONE] {exp['model_short']} / {exp['dataset']} -> {csv_path}", flush=True)

    print(f"\nFinished. CSV written to: {csv_path}")

if __name__ == "__main__":
    main()
