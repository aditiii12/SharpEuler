#!/usr/bin/env python3
"""
Experiment 50B: TTT on RealWorldQA — push image_first using disagreement-weighted
strong-branch margin maximization.

Idea:
  - image_first is the stronger baseline
  - question_first is only a probe for instability
  - optimize ONLY the image_first branch
  - use JS(strong, weak) to decide where to push harder
  - maximize the top1-top2 margin on image_first over A/B/C/D logits

Per question:
  1. Run image_first and question_first
  2. Compute JS disagreement over A/B/C/D probs
  3. Take N_STEPS updates on layers 0-6 ONLY (top 1/4 of 28 layers)
     with loss = - JS * margin(image_first)
     optionally gated by strong confidence
  4. Evaluate both orderings after each step
  5. Reset weights before next question

This is the simplest margin-based version faithful to "use weak ordering to push the
stronger image_first baseline".
"""

import argparse, gc, json, os, random, io, base64, subprocess, datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID       = "yangjie-cv/WeThink-Qwen2.5VL-7B"
N_STEPS        = 10
LETTERS        = ["A", "B", "C", "D"]
TOTAL_LAYERS   = 28
N_LAYERS_TTT   = TOTAL_LAYERS // 4   # top 1/4 layers = first 7 layers: 0-6
CONF_THRESH    = 0.45
JS_MIN         = 0.00
EPS            = 1e-8

LAYER_FN = lambda n: any(f"language_model.layers.{l}." in n for l in range(N_LAYERS_TTT))

SWEEP_CONFIGS = [
    ("img_margin_js_lr1e-4", 1e-4, 0.9),
]
ALL_CONFIGS = ["baseline", "img_margin_js_lr1e-4"]


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_metadata(result_dir, config_name, args):
    cfg = next((c for c in SWEEP_CONFIGS if c[0] == config_name), None)
    meta = {
        "experiment": "exp50_strong_margin_weighted",
        "config": config_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "git_hash": get_git_hash(),
        "model_id": MODEL_ID,
        "n_questions": args.n_questions,
        "n_gpus": args.n_gpus,
        "seed": args.seed,
        "n_steps": N_STEPS,
        "layer_range": f"0-{N_LAYERS_TTT-1}",
        "dataset": "RealWorldQA",
        "tsv": args.tsv,
        "strong_ordering": "image_first",
        "weak_ordering": "question_first",
        "objective": "loss = - JS(strong,weak) * (top1_logit - top2_logit) on strong",
        "confidence_threshold": args.conf_thresh,
        "js_min": args.js_min,
        "lr": cfg[1] if cfg else None,
        "momentum": cfg[2] if cfg else None,
        "fp32_optimizer": True,
    }
    path = os.path.join(result_dir, f"run_metadata_{config_name}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata -> {path}")


def write_questions(result_dir, questions):
    path = os.path.join(result_dir, "questions_used.json")
    if os.path.exists(path):
        return
    payload = [{"idx": i, "question": q["question"], "correct": q["correct"]} for i, q in enumerate(questions)]
    with open(path, "w") as f:
        json.dump({"n": len(payload), "questions": payload}, f, indent=2)
    print(f"Questions -> {path}")


def build_prompt(row):
    return (
        f"{str(row['question']).strip()}\n"
        f"A. {str(row['A']).strip()}\nB. {str(row['B']).strip()}\n"
        f"C. {str(row['C']).strip()}\nD. {str(row['D']).strip()}\n"
        f"Answer with only the letter A, B, C, or D."
    )


def normalize_answer(row):
    ans = str(row["answer"]).strip()
    if ans.upper() in LETTERS:
        return ans.upper()
    for k in LETTERS:
        if ans.lower() == str(row[k]).strip().lower():
            return k
    return None


def load_questions(args):
    df = pd.read_csv(args.tsv, sep="\t")
    rng = random.Random(args.seed)
    idx = list(df.index)
    rng.shuffle(idx)
    questions = []
    for _, row in df.loc[idx].reset_index(drop=True).iterrows():
        if args.n_questions and len(questions) >= args.n_questions:
            break
        correct = normalize_answer(row)
        if correct is None:
            continue
        questions.append({
            "question": str(row["question"]),
            "prompt": build_prompt(row),
            "image_path": str(row["image"]),
            "correct": correct,
        })
    return questions


def load_image(q, img_root):
    p = q["image_path"]
    if p.startswith("data:") or len(p) > 500:
        return Image.open(io.BytesIO(base64.b64decode(p.split(",")[-1]))).convert("RGB")
    return Image.open(os.path.join(img_root, p)).convert("RGB")


def get_inputs(proc, image, prompt, ordering, device):
    content = ([{"type": "image"}, {"type": "text", "text": prompt}] if ordering == "image_first"
               else [{"type": "text", "text": prompt}, {"type": "image"}])
    chat = proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )
    inp = proc(text=chat, images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in inp.items()}


def get_letter_ids(proc):
    ids = {}
    for L in LETTERS:
        for s in [L, " " + L]:
            tids = proc.tokenizer.encode(s, add_special_tokens=False)
            if len(tids) == 1:
                ids[L] = int(tids[0])
                break
        if L not in ids:
            ids[L] = int(proc.tokenizer.convert_tokens_to_ids(L))
    return ids


@torch.no_grad()
def predict_from_logits(logits, letter_ids):
    return max(LETTERS, key=lambda L: float(logits[letter_ids[L]]))


@torch.no_grad()
def predict(model, inputs, letter_ids):
    logits = model(**inputs).logits[0, -1].float()
    return predict_from_logits(logits, letter_ids)


@torch.no_grad()
def get_letter_probs_and_stats(model, inputs, letter_ids):
    logits = model(**inputs).logits[0, -1].float()
    letter_logits = torch.stack([logits[letter_ids[L]] for L in LETTERS])
    probs = F.softmax(letter_logits, dim=0)
    pred_idx = int(torch.argmax(probs))
    pred = LETTERS[pred_idx]
    conf = float(probs[pred_idx])
    return probs, pred, conf, letter_logits, logits


def js_divergence_from_probs(p, q):
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + EPS) - torch.log(m + EPS)))
    kl_qm = torch.sum(q * (torch.log(q + EPS) - torch.log(m + EPS)))
    return 0.5 * (kl_pm + kl_qm)


def save_snapshot(model):
    return {n: p.cpu().clone() for n, p in model.named_parameters() if LAYER_FN(n)}


def restore_snapshot(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n].to(p.device))
    gc.collect(); torch.cuda.empty_cache()


def baseline_worker(gpu_id, questions, img_root, result_dir):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    device = f"cuda:{gpu_id}"
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    ).to(device).eval()
    letter_ids = get_letter_ids(proc)
    results = {"img": [], "qf": []}

    from tqdm import tqdm
    for qi, q in enumerate(tqdm(questions, desc=f"[GPU {gpu_id}] baseline", position=gpu_id, leave=True)):
        image = load_image(q, img_root)
        with torch.no_grad():
            inp_img = get_inputs(proc, image, q["prompt"], "image_first", device)
            inp_qf = get_inputs(proc, image, q["prompt"], "question_first", device)
            img_pred = predict(model, inp_img, letter_ids)
            qf_pred = predict(model, inp_qf, letter_ids)
            results["img"].append(img_pred == q["correct"])
            results["qf"].append(qf_pred == q["correct"])
            del inp_img, inp_qf; torch.cuda.empty_cache()
        print(
            f"[GPU {gpu_id}] Q{qi+1}/{len(questions)} img={img_pred}({'V' if img_pred==q['correct'] else 'X'}) "
            f"qf={qf_pred}({'V' if qf_pred==q['correct'] else 'X'}) correct={q['correct']} "
            f"running img={np.mean(results['img']):.3f} qf={np.mean(results['qf']):.3f}",
            flush=True,
        )

    out = os.path.join(result_dir, f"gpu{gpu_id}_baseline_results.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"[GPU {gpu_id}] Saved -> {out}", flush=True)


def ttt_worker(gpu_id, questions, img_root, result_dir, active_configs, conf_thresh, js_min):
    print(f"[GPU {gpu_id}] worker started", flush=True)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    device = f"cuda:{gpu_id}"
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    ).to(device).eval()
    print(f"[GPU {gpu_id}] model ready", flush=True)
    letter_ids = get_letter_ids(proc)
    cfg_label = active_configs[0][0]
    results = {cfg[0]: {"qf": [], "img": [], "js": [], "conf": [], "did_update": [], "margin": []} for cfg in active_configs}

    from tqdm import tqdm
    for qi, q in enumerate(tqdm(questions, desc=f"[GPU {gpu_id}] {cfg_label}", position=gpu_id, leave=True)):
        image = load_image(q, img_root)
        correct = q["correct"]
        snap = save_snapshot(model)

        inp_img = get_inputs(proc, image, q["prompt"], "image_first", device)
        inp_qf = get_inputs(proc, image, q["prompt"], "question_first", device)
        inp_img_cpu = {k: v.cpu() for k, v in inp_img.items()}
        inp_qf_cpu = {k: v.cpu() for k, v in inp_qf.items()}

        for cfg_lbl, lr, momentum in active_configs:
            params = [p for n, p in model.named_parameters() if LAYER_FN(n) and p.requires_grad]
            params_fp32 = [p.detach().float().requires_grad_(True) for p in params]
            opt = torch.optim.SGD(params_fp32, lr=lr, momentum=momentum)

            qf_steps, img_steps, js_steps, conf_steps, upd_steps, margin_steps = [], [], [], [], [], []

            try:
                for step in range(N_STEPS + 1):
                    model.eval()
                    with torch.no_grad():
                        img_gpu = {k: v.to(device) for k, v in inp_img_cpu.items()}
                        qf_gpu = {k: v.to(device) for k, v in inp_qf_cpu.items()}

                        p_s, pred_s, conf_s, letter_logits_s, full_logits_s = get_letter_probs_and_stats(model, img_gpu, letter_ids)
                        p_w, pred_w, conf_w, _, _ = get_letter_probs_and_stats(model, qf_gpu, letter_ids)
                        js = float(js_divergence_from_probs(p_s, p_w))
                        top2_vals, _ = torch.topk(letter_logits_s, 2)
                        margin = float(top2_vals[0] - top2_vals[1])

                        img_steps.append(pred_s == correct)
                        qf_steps.append(pred_w == correct)
                        js_steps.append(js)
                        conf_steps.append(conf_s)
                        margin_steps.append(margin)

                        gate = (conf_s >= conf_thresh) and (js >= js_min)
                        upd_steps.append(bool(gate and step < N_STEPS))

                        if qi < 5:
                            print(
                                f"[GPU {gpu_id}] Q{qi+1} step{step:02d} strong={pred_s} weak={pred_w} "
                                f"correct={correct} strong_conf={conf_s:.4f} weak_conf={conf_w:.4f} js={js:.6f} "
                                f"margin={margin:.4f} gate={gate} img_ok={img_steps[-1]} qf_ok={qf_steps[-1]}",
                                flush=True,
                            )

                        del qf_gpu
                        if step == N_STEPS:
                            del img_gpu
                            torch.cuda.empty_cache()
                            break

                    if not gate:
                        del img_gpu
                        torch.cuda.empty_cache()
                        continue

                    model.train()
                    opt.zero_grad()
                    logits = model(**img_gpu).logits[0, -1].float()
                    letter_logits = torch.stack([logits[letter_ids[L]] for L in LETTERS])
                    vals, _ = torch.topk(letter_logits, 2)
                    margin_t = vals[0] - vals[1]
                    loss = - js * margin_t
                    loss.backward()

                    for p, p32 in zip(params, params_fp32):
                        if p.grad is not None:
                            p32.grad = p.grad.float().clone()
                        p.grad = None
                    opt.step()

                    with torch.no_grad():
                        for p, p32 in zip(params, params_fp32):
                            p.copy_(p32.bfloat16())

                    if qi < 5:
                        print(
                            f"[GPU {gpu_id}] Q{qi+1} UPDATE step{step:02d} loss={float(loss):.6f} "
                            f"pred_strong={pred_s} conf={conf_s:.4f} js={js:.6f} margin_before={margin:.4f}",
                            flush=True,
                        )
                    del img_gpu, logits, letter_logits, vals, margin_t, loss
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"[GPU {gpu_id}] Q{qi+1} OOM", flush=True)
                qf_steps = [False] * (N_STEPS + 1)
                img_steps = [False] * (N_STEPS + 1)
                js_steps = [0.0] * (N_STEPS + 1)
                conf_steps = [0.0] * (N_STEPS + 1)
                upd_steps = [False] * (N_STEPS + 1)
                margin_steps = [0.0] * (N_STEPS + 1)
            except Exception as e:
                print(f"[GPU {gpu_id}] Q{qi+1} ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
                qf_steps = [False] * (N_STEPS + 1)
                img_steps = [False] * (N_STEPS + 1)
                js_steps = [0.0] * (N_STEPS + 1)
                conf_steps = [0.0] * (N_STEPS + 1)
                upd_steps = [False] * (N_STEPS + 1)
                margin_steps = [0.0] * (N_STEPS + 1)

            opt.zero_grad(set_to_none=True)
            del opt, params, params_fp32
            results[cfg_lbl]["qf"].append(qf_steps)
            results[cfg_lbl]["img"].append(img_steps)
            results[cfg_lbl]["js"].append(js_steps)
            results[cfg_lbl]["conf"].append(conf_steps)
            results[cfg_lbl]["did_update"].append(upd_steps)
            results[cfg_lbl]["margin"].append(margin_steps)

            print(
                f"[GPU {gpu_id}] Q{qi+1}/{len(questions)} {cfg_lbl} correct={correct} "
                f"img_s0={img_steps[0]} img_peak={max(img_steps)} img_last={img_steps[-1]} "
                f"avg_js={np.mean(js_steps):.6f} avg_conf={np.mean(conf_steps):.4f} "
                f"margin0={margin_steps[0]:.4f} margin_last={margin_steps[-1]:.4f} "
                f"n_updates={sum(upd_steps[:-1])} running img_last={np.mean([x[-1] for x in results[cfg_lbl]['img']]):.3f}",
                flush=True,
            )

            restore_snapshot(model, snap)

        del inp_img, inp_qf, inp_img_cpu, inp_qf_cpu, snap
        gc.collect(); torch.cuda.empty_cache()

    out = os.path.join(result_dir, f"gpu{gpu_id}_{cfg_label}_results.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"[GPU {gpu_id}] Saved -> {out}", flush=True)


def run_mp(worker_fn, n_gpus, questions, img_root, result_dir, extra_args=()):
    splits = [[] for _ in range(n_gpus)]
    for i, q in enumerate(questions):
        splits[i % n_gpus].append(q)
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
    procs = []
    for gpu_id in range(n_gpus):
        if not splits[gpu_id]:
            continue
        args_ = (gpu_id, splits[gpu_id], img_root, result_dir) + tuple(extra_args)
        p = mp.Process(target=worker_fn, args=args_)
        p.start(); procs.append(p)
    for p in procs:
        p.join()


def pad_or_trim(arr, target):
    if len(arr) >= target:
        return arr[:target]
    return arr + [arr[-1]] * (target - len(arr))


def make_figure(summary, out_dir, n, baseline_img, baseline_qf):
    steps = np.arange(0, N_STEPS + 1)
    fig, axes = plt.subplots(1, 4, figsize=(28, 5))
    cfg_lbl = "img_margin_js_lr1e-4"

    vals_img = np.array(summary[cfg_lbl]["img"])
    vals_qf = np.array(summary[cfg_lbl]["qf"])
    vals_js = np.array(summary[cfg_lbl]["js"])
    vals_margin = np.array(summary[cfg_lbl]["margin"])

    axes[0].plot(steps, vals_img, color="#C44E52", linewidth=2.5, marker="o", label=cfg_lbl)
    axes[0].axhline(baseline_img, color="#333333", linestyle=":", linewidth=1.5, label=f"baseline image_first ({baseline_img:.4f})")
    axes[0].axhline(baseline_qf, color="#AAAAAA", linestyle=":", linewidth=1.2, label=f"baseline question_first ({baseline_qf:.4f})")
    peak_step = int(np.argmax(vals_img))
    axes[0].annotate(f"{vals_img[peak_step]:.4f} @ {peak_step}", (peak_step, vals_img[peak_step]),
                     (peak_step + 0.4, vals_img[peak_step] + 0.004), fontsize=9)
    axes[0].set_title("image_first accuracy")
    axes[0].set_xlabel("TTT step")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(steps, vals_qf, color="#4C72B0", linewidth=2.5, marker="o", label=cfg_lbl)
    axes[1].axhline(baseline_qf, color="#333333", linestyle=":", linewidth=1.5, label=f"baseline question_first ({baseline_qf:.4f})")
    axes[1].axhline(baseline_img, color="#AAAAAA", linestyle=":", linewidth=1.2, label=f"baseline image_first ({baseline_img:.4f})")
    axes[1].set_title("question_first accuracy")
    axes[1].set_xlabel("TTT step")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(steps, vals_js, color="#55A868", linewidth=2.5, marker="o")
    axes[2].set_title("mean JS(strong, weak)")
    axes[2].set_xlabel("TTT step")
    axes[2].set_ylabel("JS")
    axes[2].grid(True, alpha=0.2)

    axes[3].plot(steps, vals_margin, color="#8172B2", linewidth=2.5, marker="o")
    axes[3].set_title("mean strong top1-top2 margin")
    axes[3].set_xlabel("TTT step")
    axes[3].set_ylabel("margin")
    axes[3].grid(True, alpha=0.2)

    fig.suptitle(
        f"WeThink-Qwen2.5VL-7B | RealWorldQA | Layers 0-6 | n={n}\n"
        f"Strong-targeted TTT: disagreement-weighted margin on image_first",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "results.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def consolidate(args):
    result_dir = args.result_dir
    bfiles = [f for f in os.listdir(result_dir) if f.endswith("_results.json") and "baseline" in f]
    if not bfiles:
        print("ERROR: No baseline found. Run --config baseline first.")
        return

    img_c, qf_c = [], []
    for fname in bfiles:
        with open(os.path.join(result_dir, fname)) as f:
            d = json.load(f)
        img_c.extend(d["img"]); qf_c.extend(d["qf"])
    baseline_img = float(np.mean(img_c))
    baseline_qf = float(np.mean(qf_c))
    n_base = len(img_c)
    print(f"\nBaseline (n={n_base}):")
    print(f"  image_first:    {baseline_img:.4f}")
    print(f"  question_first: {baseline_qf:.4f}")
    print(f"  ordering gap:   {baseline_img - baseline_qf:.4f}")

    tfiles = [f for f in os.listdir(result_dir) if f.endswith("_results.json") and "baseline" not in f]
    merged = {cfg[0]: {"qf": [], "img": [], "js": [], "conf": [], "did_update": [], "margin": []} for cfg in SWEEP_CONFIGS}
    for fname in tfiles:
        with open(os.path.join(result_dir, fname)) as f:
            data = json.load(f)
        for cfg_lbl, d in data.items():
            if cfg_lbl in merged:
                for k in merged[cfg_lbl].keys():
                    merged[cfg_lbl][k].extend(d[k])

    target = N_STEPS + 1
    summary = {}
    print(f"\n{'Config':<22} {'metric':<10} {'peak':>7} {'@step':>6} {'delta':>10}")
    print(f"  {'-'*58}")
    for cfg_lbl, _, _ in SWEEP_CONFIGS:
        d = merged[cfg_lbl]
        if not d["img"]:
            continue
        qf_m = np.array([pad_or_trim(x, target) for x in d["qf"]], dtype=float).mean(axis=0)
        img_m = np.array([pad_or_trim(x, target) for x in d["img"]], dtype=float).mean(axis=0)
        js_m = np.array([pad_or_trim(x, target) for x in d["js"]], dtype=float).mean(axis=0)
        conf_m = np.array([pad_or_trim(x, target) for x in d["conf"]], dtype=float).mean(axis=0)
        upd_m = np.array([pad_or_trim(x, target) for x in d["did_update"]], dtype=float).mean(axis=0)
        margin_m = np.array([pad_or_trim(x, target) for x in d["margin"]], dtype=float).mean(axis=0)
        summary[cfg_lbl] = {
            "qf": qf_m.tolist(), "img": img_m.tolist(), "js": js_m.tolist(),
            "conf": conf_m.tolist(), "did_update": upd_m.tolist(), "margin": margin_m.tolist(),
        }
        for metric, arr, base in [("img", img_m, baseline_img), ("qf", qf_m, baseline_qf)]:
            peak = float(np.max(arr)); step = int(np.argmax(arr))
            marker = " <-- TARGET" if metric == "img" and peak > baseline_img else ""
            print(f"  {cfg_lbl:<22} {metric:<10} {peak:>7.4f} {step:>6} {peak-base:>+10.4f}{marker}")
        print(f"  {cfg_lbl:<22} {'margin@end':<10} {margin_m[-1]:>7.4f}")
        print(f"  {cfg_lbl:<22} {'upd/step':<10} {np.mean(upd_m[:-1]):>7.4f}")

    counts = [len(merged[c[0]]["img"]) for c in SWEEP_CONFIGS if merged[c[0]]["img"]]
    n = max(counts) if counts else 0
    summary["baseline_img"] = baseline_img
    summary["baseline_qf"] = baseline_qf
    summary["n"] = n
    os.makedirs(args.out_dir, exist_ok=True)
    if summary:
        make_figure(summary, args.out_dir, n, baseline_img, baseline_qf)
    with open(args.out_json, "w") as f:
        json.dump({"curves": summary}, f, indent=2)
    print(f"\nSaved: {args.out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--img_root", default=None)
    ap.add_argument("--n_questions", type=int, default=None)
    ap.add_argument("--n_gpus", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None)
    ap.add_argument("--result_dir", default="outputs/exp50_strong_margin_weighted")
    ap.add_argument("--out_json", default="outputs/exp50_strong_margin_weighted_summary.json")
    ap.add_argument("--out_dir", default="outputs/exp50_strong_margin_weighted_figures")
    ap.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    ap.add_argument("--js_min", type=float, default=JS_MIN)
    ap.add_argument("--consolidate", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    if args.consolidate:
        consolidate(args)
        return

    if not args.config:
        print(f"--config required. Options: {ALL_CONFIGS}")
        return
    if not args.tsv:
        print("Need --tsv")
        return
    if not args.img_root:
        print("Need --img_root")
        return
    if args.config not in ALL_CONFIGS:
        print(f"Unknown config. Options: {ALL_CONFIGS}")
        return

    questions = load_questions(args)
    print(f"Questions: {len(questions)}, config: {args.config}, conf_thresh={args.conf_thresh}, js_min={args.js_min}")
    n_gpus = min(args.n_gpus, len(questions))

    if args.config == "baseline":
        write_questions(args.result_dir, questions)
        write_metadata(args.result_dir, "baseline", args)
        run_mp(baseline_worker, n_gpus, questions, args.img_root, args.result_dir)
    else:
        bfiles = [f for f in os.listdir(args.result_dir) if f.endswith("_results.json") and "baseline" in f]
        if not bfiles:
            print("ERROR: Run --config baseline first.")
            return
        active = [c for c in SWEEP_CONFIGS if c[0] == args.config]
        write_metadata(args.result_dir, args.config, args)
        run_mp(ttt_worker, n_gpus, questions, args.img_root, args.result_dir, (active, args.conf_thresh, args.js_min))
    print("Done.")


if __name__ == "__main__":
    main()
