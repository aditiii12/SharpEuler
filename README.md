# SharpEuler: Sharpness-Aware Flow Matching

Official implementation of Sharpness-Aware Flow Matching - https://arxiv.org/pdf/2605.11547

## Installation

```bash
git clone [email protected]:aditiii12/SharpEuler.git
cd SharpEuler
pip install -e .
```

Optional extras:

```bash
pip install -e .[flux]   # FLUX evaluation: diffusers, transformers, torch-fidelity, open-clip
pip install -e .[toys]   # toy experiments: POT (for Wasserstein-2)
pip install -e .[dev]    # linting + testing
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. FLUX experiments need a CUDA GPU (≥ 24 GB VRAM at 1024×1024).

## Quick start

### 1. Toy 2D experiment

Two toy checkpoints are included in `checkpoints/`:

```python
import torch
from sharpeuler.toys import FMNet, FMNet2
from sharpeuler.core import (
    collect_dense_trajectories,
    compute_sharpness_profile,
    build_adaptive_schedule_from_signal,
    euler_fm_sample,
)

model = FMNet().to("cuda").eval()
model.load_state_dict(torch.load("checkpoints/fm_branchy_flower_3.pth")["ema"])

# 1. calibrate
sched_dense, _, vel_hist = collect_dense_trajectories(model, n_paths=1024, N_dense=1000)
profile = compute_sharpness_profile(sched_dense, vel_hist)

# 2. build schedule (γ = 0.5, LTE-prescribed)
phi = profile["accel_norm"] ** 0.5
schedule = build_adaptive_schedule_from_signal(profile["t_mid"], phi, B=16, sigma=1.0)

# 3. sample
samples = euler_fm_sample(model, n=10_000, schedule=schedule)
```

### 2. FLUX evaluation

Calibrate the sharpness profile (one-time, ~30 minutes on a single H100):

```bash
python scripts/flux_calibrate_profile.py
```

Run the eval block over budgets `B ∈ {8, 12, 16, 20}` against the 50-step pipeline reference:

```bash
python scripts/flux_eval_block.py     # generates per-prompt panels and per-method images
python scripts/flux_metrics.py        # aggregates RMSE / CLIP / FID
```

Paths in these scripts are currently hardcoded for our setup; edit the constants at the top to point to your FLUX checkpoint and output directory.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{sharpeuler2026,
  title  = {Sharpness-Aware Flow Matching},
  author = {Gupta, Aditi and Lim, Soon Hoe and Yu, Annan and Erichson, N. Benjamin},
  year   = {2026},
  note   = {arXiv preprint (link forthcoming)}
}
```

## License

[MIT](LICENSE).

## Acknowledgments

This work was conducted at ICSI Berkeley with support from Lawrence Berkeley National Laboratory and NERSC computing resources.
