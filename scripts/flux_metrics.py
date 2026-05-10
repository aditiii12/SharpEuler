"""flux_metrics.py — aggregate both blocks, compute final FID over all 80, emit LaTeX."""

import os, json, glob
import numpy as np
import torch_fidelity

BASE_SAVE_DIR = "/pscratch/sd/a/aditi_12/projects/flux_gamma_sweep"
NFE_BUDGETS = [8, 12]
METHOD_ORDER = ["pipeline", "shifted", "gamma033", "gamma05", "gamma1"]
LABELS = {
    "pipeline": "Pipeline",
    "shifted":  "Shifted $\\alpha{=}3$",
    "gamma033": "SharpEuler $\\gamma{=}0.33$",
    "gamma05":  "SharpEuler $\\gamma{=}0.5$",
    "gamma1":   "SharpEuler $\\gamma{=}1.0$",
}

agg = {}
for B in NFE_BUDGETS:
    B_master = os.path.join(BASE_SAVE_DIR, f"B{B}")
    B_fid    = os.path.join(B_master, "_fid")

    rows = []
    for jf in glob.glob(os.path.join(B_master, "results_block_*.json")):
        rows.extend(json.load(open(jf)))
    print(f"\nB={B}: {len(rows)} prompts aggregated")

    metrics = {}
    for m in METHOD_ORDER:
        rmses = [r[f"rmse_{m}"] for r in rows]
        clips = [r[f"clip_{m}"] for r in rows]
        metrics[m] = {
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "clip_mean": float(np.mean(clips)), "clip_std": float(np.std(clips)),
        }

    print(f"  computing FID...")
    ref_fid_dir = os.path.join(B_fid, "reference")
    for m in METHOD_ORDER:
        result = torch_fidelity.calculate_metrics(
            input1=os.path.join(B_fid, m), input2=ref_fid_dir,
            cuda=True, fid=True, verbose=False)
        metrics[m]["fid"] = float(result["frechet_inception_distance"])

    agg[B] = metrics


# print summary
print(f"\n{'='*70}\nFINAL")
for B in NFE_BUDGETS:
    print(f"\nB={B}")
    print(f"{'method':<12} {'RMSE':>12} {'CLIP':>14} {'FID':>10}")
    for m in METHOD_ORDER:
        d = agg[B][m]
        print(f"{m:<12} {d['rmse_mean']:>6.2f}±{d['rmse_std']:>4.2f} "
              f"{d['clip_mean']:>6.4f}±{d['clip_std']:>4.4f} {d['fid']:>10.2f}")


# emit LaTeX
print(f"\n{'='*70}\nLaTeX rows (methods × budget × {{RMSE, CLIP, FID}}):")
print()
for m in METHOD_ORDER:
    cells = [LABELS[m]]
    for B in NFE_BUDGETS:
        d = agg[B][m]
        cells.extend([f"{d['rmse_mean']:.2f}", f"{d['clip_mean']:.4f}", f"{d['fid']:.2f}"])
    print(" & ".join(cells) + r" \\")


with open(os.path.join(BASE_SAVE_DIR, "summary.json"), "w") as f:
    json.dump(agg, f, indent=2)
print(f"\nSaved {os.path.join(BASE_SAVE_DIR, 'summary.json')}")