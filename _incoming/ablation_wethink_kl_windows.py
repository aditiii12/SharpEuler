#!/usr/bin/env python3
"""
KL ordering ablation — WeThink-Qwen2.5VL-7B.

Adds:
- fixed-teacher vs current-teacher KL
- early / mid / late layer-window ablation
- 5-step runs with accuracy recorded after every step
- peak step / peak accuracy / final accuracy reporting
- consolidated plot across all configs
- GPU id selection (defaults to 1,2,3,4,5,6,7)

Example:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ablation_wethink_kl_windows.py \
    --tsv ~/LMUData/RealWorldQA.tsv --img_root ~/LMUData \
    --n_questions 200 --config fixed_early_lr1e4 --result_dir outputs/wethink_realworldqa

  python ablation_wethink_kl_windows.py --consolidate --result_dir outputs/wethink_realworldqa
"""

import argparse
import base64
import gc
import glob
import io
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "yangjie-cv/WeThink-Qwen2.5VL-7B"
LETTERS = ["A", "B", "C", "D"]
SEED = 42
DEFAULT_STEPS = 5
DEFAULT_GPU_IDS = [1, 2, 3, 4, 5, 6, 7]

# Keep this small and sharp.
# name -> dict(steps, lr, mom, teacher_mode, layer_window)
CONFIGS = {
    "baseline": {
        "steps": 0,
        "lr": None,
        "mom": None,
        "teacher_mode": "none",
        "layer_window": "none",
    },
    "current_early_lr1e4": {
        "steps": DEFAULT_STEPS,
        "lr": 1e-4,
        "mom": 0.9,
        "teacher_mode": "current",
        "layer_window": "early",
    },
    "fixed_early_lr1e4": {
        "steps": DEFAULT_STEPS,
        "lr": 1e-4,
        "mom": 0.9,
        "teacher_mode": "fixed",
        "layer_window": "early",
    },
    "fixed_mid_lr1e4": {
        "steps": DEFAULT_STEPS,
        "lr": 1e-4,
        "mom": 0.9,
        "teacher_mode": "fixed",
        "layer_window": "mid",
    },
    "fixed_late_lr1e4": {
        "steps": DEFAULT_STEPS,
        "lr": 1e-4,
        "mom": 0.9,
        "teacher_mode": "fixed",
        "layer_window": "late",
    },
}
ALL_CONFIGS = list(CONFIGS.keys())


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
    for key in LETTERS:
        if ans.lower() == str(row[key]).strip().lower():
            return key
    return None


def load_questions(tsv, n_q):
    df = pd.read_csv(tsv, sep="\t")
    rng = random.Random(SEED)
    idx = list(df.index)
    rng.shuffle(idx)
    questions = []
    for _, row in df.loc[idx].reset_index(drop=True).iterrows():
        if n_q and len(questions) >= n_q:
            break
        correct = normalize_answer(row)
        if correct is None:
            continue
        questions.append(
            {
                "prompt": build_prompt(row),
                "image_path": str(row["image"]),
                "correct": correct,
                "category": str(row.get("category", "overall")),
            }
        )
    return questions


def load_image(img_path, img_root):
    if img_path.startswith("["):
        img_path = img_path.strip("[]'\"")
    if img_path.startswith("data:") or len(img_path) > 500:
        payload = img_path.split(",")[-1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    return Image.open(os.path.join(img_root, img_path)).convert("RGB")


def get_inputs(proc, image, prompt, ordering, device):
    content = (
        [{"type": "image"}, {"type": "text", "text": prompt}]
        if ordering == "image_first"
        else [{"type": "text", "text": prompt}, {"type": "image"}]
    )
    chat = proc.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inp = proc(text=chat, images=image, return_tensors="pt")
    return {key: value.to(device) for key, value in inp.items()}


def get_letter_ids(proc):
    ids = {}
    for letter in LETTERS:
        for text in [letter, " " + letter]:
            tids = proc.tokenizer.encode(text, add_special_tokens=False)
            if len(tids) == 1:
                ids[letter] = int(tids[0])
                break
        if letter not in ids:
            ids[letter] = int(proc.tokenizer.convert_tokens_to_ids(letter))
    return ids


@torch.no_grad()
def predict_with_conf(model, inputs, letter_ids):
    logits = model(**inputs).logits[0, -1].float()
    probs = torch.softmax(torch.stack([logits[letter_ids[letter]] for letter in LETTERS]), dim=0)
    return LETTERS[probs.argmax().item()], probs.max().item()


@torch.no_grad()
def get_last_token_logits(model, inputs):
    return model(**inputs).logits[0, -1].float()


def infer_num_layers(model):
    layer_idxs = set()
    pattern = re.compile(r"language_model\.layers\.(\d+)\.")
    for name, _ in model.named_parameters():
        match = pattern.search(name)
        if match:
            layer_idxs.add(int(match.group(1)))
    if not layer_idxs:
        raise RuntimeError("Could not infer language_model.layers.* indices from model parameters.")
    return max(layer_idxs) + 1


def select_layer_indices(num_layers, window, width=7):
    if window == "none":
        return []
    if num_layers <= width:
        return list(range(num_layers))
    if window == "early":
        return list(range(width))
    if window == "late":
        return list(range(num_layers - width, num_layers))
    if window == "mid":
        start = max(0, (num_layers - width) // 2)
        return list(range(start, start + width))
    raise ValueError(f"Unknown layer window: {window}")


def make_layer_fn(selected_layers):
    selected = set(selected_layers)

    def layer_fn(name):
        for idx in selected:
            if f"language_model.layers.{idx}." in name:
                return True
        return False

    return layer_fn


def save_snapshot(model, layer_fn):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if layer_fn(name)}


def restore_snapshot(model, snap):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snap:
                p.copy_(snap[name].to(p.device))
    gc.collect()
    torch.cuda.empty_cache()


def worker(gpu_id, questions, img_root, config_name, result_dir):
    cfg = CONFIGS[config_name]
    n_steps = cfg["steps"]
    lr = cfg["lr"]
    mom = cfg["mom"]
    teacher_mode = cfg["teacher_mode"]
    layer_window = cfg["layer_window"]

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    device = f"cuda:{gpu_id}"

    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    ).to(device).eval()

    num_layers = infer_num_layers(model)
    selected_layers = select_layer_indices(num_layers, layer_window, width=7)
    layer_fn = make_layer_fn(selected_layers)
    letter_ids = get_letter_ids(proc)

    step0_correct = []
    img_steps = []

    from tqdm import tqdm

    for qi, q in enumerate(
        tqdm(questions, desc=f"[GPU {gpu_id}] {config_name}", position=min(gpu_id, 15), leave=True)
    ):
        image = load_image(q["image_path"], img_root)
        correct = q["correct"]

        inp_img = get_inputs(proc, image, q["prompt"], "image_first", device)
        inp_qf = get_inputs(proc, image, q["prompt"], "question_first", device)

        with torch.no_grad():
            img_pred, _ = predict_with_conf(model, inp_img, letter_ids)
        step0_correct.append(img_pred == correct)
        q_steps = [img_pred == correct]

        if teacher_mode == "none" or n_steps == 0:
            img_steps.append(q_steps)
        else:
            snap = save_snapshot(model, layer_fn)
            params = [p for name, p in model.named_parameters() if layer_fn(name) and p.requires_grad]
            params_fp32 = [p.detach().float().requires_grad_(True) for p in params]
            opt = torch.optim.SGD(params_fp32, lr=lr, momentum=mom)

            fixed_p_img = None
            if teacher_mode == "fixed":
                with torch.no_grad():
                    fixed_logits_img = get_last_token_logits(model, inp_img)
                    fixed_p_img = F.softmax(fixed_logits_img, dim=-1).unsqueeze(0)
                del fixed_logits_img

            try:
                for step in range(n_steps):
                    model.train()
                    opt.zero_grad()

                    if teacher_mode == "current":
                        with torch.no_grad():
                            logits_img = get_last_token_logits(model, inp_img)
                            p_img = F.softmax(logits_img, dim=-1).unsqueeze(0)
                    elif teacher_mode == "fixed":
                        p_img = fixed_p_img
                    else:
                        raise ValueError(f"Unexpected teacher mode: {teacher_mode}")

                    logits_qf = get_last_token_logits(model, inp_qf)
                    log_p_qf = F.log_softmax(logits_qf, dim=-1).unsqueeze(0)
                    loss = F.kl_div(log_p_qf, p_img, reduction="batchmean")
                    loss.backward()

                    for p, p32 in zip(params, params_fp32):
                        if p.grad is not None:
                            p32.grad = p.grad.float().clone()
                        p.grad = None
                    opt.step()

                    with torch.no_grad():
                        for p, p32 in zip(params, params_fp32):
                            p.copy_(p32.to(dtype=p.dtype))
                    model.eval()
                    del loss, logits_qf, log_p_qf
                    if teacher_mode == "current":
                        del logits_img, p_img

                    with torch.no_grad():
                        step_pred, _ = predict_with_conf(model, inp_img, letter_ids)
                    q_steps.append(step_pred == correct)

                    if qi < 3:
                        print(
                            f"[GPU {gpu_id}] {config_name} Q{qi + 1} step{step + 1} "
                            f"img={step_pred}({'V' if step_pred == correct else 'X'}) "
                            f"teacher={teacher_mode} layers={selected_layers[0] if selected_layers else '-'}"
                            f"-{selected_layers[-1] if selected_layers else '-'}",
                            flush=True,
                        )
            except torch.cuda.OutOfMemoryError:
                print(f"[GPU {gpu_id}] Q{qi + 1} OOM", flush=True)
                model.eval()
                gc.collect()
                torch.cuda.empty_cache()

            img_steps.append(q_steps)
            opt.zero_grad(set_to_none=True)
            del opt, params, params_fp32, fixed_p_img
            restore_snapshot(model, snap)
            del snap

        del inp_img, inp_qf
        gc.collect()
        torch.cuda.empty_cache()

        after_curve = [np.mean([steps[min(i, len(steps) - 1)] for steps in img_steps]) for i in range(len(img_steps[-1]))]
        peak_acc = max(after_curve)
        peak_step = int(np.argmax(after_curve))
        print(
            f"[GPU {gpu_id}] {config_name} Q{qi + 1}/{len(questions)} "
            f"step0={'V' if step0_correct[-1] else 'X'} correct={correct} "
            f"after={'V' if q_steps[-1] else 'X'} "
            f"step0_acc={np.mean(step0_correct):.3f} "
            f"peak_acc={peak_acc:.3f} peak_step={peak_step} "
            f"final_acc={after_curve[-1]:.3f} delta={peak_acc - np.mean(step0_correct):+.3f}",
            flush=True,
        )

    out = os.path.join(result_dir, f"gpu{gpu_id}_{config_name}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "config": config_name,
                "gpu_id": gpu_id,
                "teacher_mode": teacher_mode,
                "layer_window": layer_window,
                "selected_layers": selected_layers,
                "num_layers": num_layers,
                "step0": step0_correct,
                "img_steps": img_steps,
            },
            f,
        )
    print(f"[GPU {gpu_id}] Saved -> {out}", flush=True)


def run_mp(config_name, gpu_ids, questions, img_root, result_dir):
    gpu_ids = list(gpu_ids)
    splits = [[] for _ in range(len(gpu_ids))]
    for i, q in enumerate(questions):
        splits[i % len(gpu_ids)].append(q)

    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    procs = []
    for slot, gpu_id in enumerate(gpu_ids):
        if not splits[slot]:
            continue
        p = mp.Process(target=worker, args=(gpu_id, splits[slot], img_root, config_name, result_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def consolidate(result_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = glob.glob(os.path.join(result_dir, "gpu*_*.json"))
    grouped = defaultdict(lambda: {"step0": [], "img_steps": []})
    meta = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        cfg = data["config"]
        grouped[cfg]["step0"].extend(data["step0"])
        grouped[cfg]["img_steps"].extend(data["img_steps"])
        meta[cfg] = {
            "teacher_mode": data.get("teacher_mode", "?"),
            "layer_window": data.get("layer_window", "?"),
            "selected_layers": data.get("selected_layers", []),
            "num_layers": data.get("num_layers", None),
        }

    if not grouped:
        print(f"No result json files found in {result_dir}")
        return

    max_steps = max(max((len(step_list) for step_list in d["img_steps"]), default=1) for d in grouped.values())

    def pad(arr, target):
        return arr + [arr[-1]] * (target - len(arr)) if len(arr) < target else arr[:target]

    print(f"\n{'=' * 110}")
    print(f"WeThink-Qwen2.5VL-7B  |  KL windows ablation  |  {result_dir}")
    print(
        f"{'Config':<22} {'n':>6} {'step0':>7} {'peak':>7} {'@step':>6} {'final':>7} {'delta':>8} {'teacher':>9} {'layers':>16}"
    )
    print(f"{'-' * 110}")

    rows = []
    curves = {}
    baseline_val = None

    for cfg_name in ALL_CONFIGS:
        if cfg_name not in grouped:
            continue
        data = grouped[cfg_name]
        n = len(data["step0"])
        step0 = float(np.mean(data["step0"])) if n > 0 else float("nan")
        padded = np.array([pad(s, max_steps) for s in data["img_steps"]], dtype=float)
        curve = padded.mean(axis=0).tolist()
        peak = max(curve)
        peak_step = int(np.argmax(curve))
        final = curve[-1]
        delta = peak - step0
        info = meta.get(cfg_name, {})
        sel = info.get("selected_layers", [])
        layer_str = "-" if not sel else f"{sel[0]}-{sel[-1]}"
        teacher_str = info.get("teacher_mode", "?")
        rows.append((cfg_name, n, step0, peak, peak_step, final, delta, teacher_str, layer_str))
        curves[cfg_name] = curve
        if cfg_name == "baseline":
            baseline_val = step0

    best_delta = max(r[6] for r in rows if r[0] != "baseline") if any(r[0] != "baseline" for r in rows) else 0.0
    for row in rows:
        cfg_name, n, step0, peak, peak_step, final, delta, teacher_str, layer_str = row
        marker = "  <- BEST" if cfg_name != "baseline" and abs(delta - best_delta) < 1e-12 else ""
        print(
            f"{cfg_name:<22} {n:>6} {step0:>7.3f} {peak:>7.3f} {peak_step:>6} {final:>7.3f} {delta:>+8.3f} {teacher_str:>9} {layer_str:>16}{marker}"
        )
    print(f"{'=' * 110}")

    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    x_steps = np.arange(max_steps)

    for (cfg_name, curve), color in zip(curves.items(), colors):
        ax.plot(
            x_steps,
            curve,
            label=cfg_name,
            color=color,
            linewidth=2.5 if cfg_name == "baseline" else 1.8,
            linestyle="--" if cfg_name == "baseline" else "-",
            marker="o",
            markersize=3,
        )

    if baseline_val is not None:
        ax.axhline(
            baseline_val,
            color="black",
            linewidth=1,
            linestyle=":",
            label=f"baseline ({baseline_val:.3f})",
            zorder=5,
        )

    ax.set_xlabel("TTT step", fontsize=11)
    ax.set_ylabel("image_first accuracy", fontsize=11)
    ax.set_title(f"WeThink-Qwen2.5VL-7B  |  KL windows ablation  |  {result_dir}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    plt.tight_layout()

    out_path = os.path.join(result_dir, "kl_windows_img_steps.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


def parse_gpu_ids(text):
    if not text:
        return DEFAULT_GPU_IDS
    ids = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        ids.append(int(piece))
    if not ids:
        raise ValueError("No valid GPU ids parsed.")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--img_root", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n_questions", type=int, default=None)
    ap.add_argument("--gpu_ids", type=str, default=",".join(map(str, DEFAULT_GPU_IDS)))
    ap.add_argument("--result_dir", default="outputs/wethink_kl_windows")
    ap.add_argument("--consolidate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    if args.consolidate:
        consolidate(args.result_dir)
        return

    if not args.config:
        print("--config required. Options:\n  " + "\n  ".join(ALL_CONFIGS))
        return
    if args.config not in CONFIGS:
        print("Unknown config. Options:\n  " + "\n  ".join(ALL_CONFIGS))
        return
    if not args.tsv or not args.img_root:
        print("Need --tsv and --img_root")
        return

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    existing = glob.glob(os.path.join(args.result_dir, f"gpu*_{args.config}.json"))
    if existing:
        print(f"\nWARNING: {len(existing)} result file(s) already exist for '{args.config}'")
        ans = input("Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        for f in existing:
            os.remove(f)

    questions = load_questions(args.tsv, args.n_questions)
    print(
        f"Questions: {len(questions)}, config: {args.config}, seed={SEED}, gpu_ids={gpu_ids}, steps={CONFIGS[args.config]['steps']}"
    )

    run_mp(args.config, gpu_ids, questions, args.img_root, args.result_dir)
    print("Done.")


if __name__ == "__main__":
    main()
