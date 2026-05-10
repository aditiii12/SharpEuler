import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


# =========================================================
# Device
# =========================================================
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================================================
# Data
# =========================================================
def make_gaussian_grid(n_modes=9, n_samples=3000, std=0.05, seed=0):
    """
    n_modes must be a perfect square.
    n_samples is PER mode.
    Returns:
        X: (n_modes * n_samples, 2)
        centers: 1D array of grid coordinates
    """
    rng = np.random.default_rng(seed)
    grid_size = int(np.sqrt(n_modes))
    assert grid_size * grid_size == n_modes, "n_modes must be a perfect square"

    centers = np.linspace(-1.0, 1.0, grid_size)
    data = []
    for cx in centers:
        for cy in centers:
            samples = rng.standard_normal((n_samples, 2)) * std + np.array([cx, cy])
            data.append(samples)
    return np.vstack(data).astype(np.float32), centers


def make_loader(X, batch_size=512, shuffle=True, num_workers=0, device="cuda"):
    X = torch.as_tensor(X, dtype=torch.float32)
    ds = TensorDataset(X)
    pin = (device is not None) and ("cuda" in str(device))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
    )


# =========================================================
# Time embedding
# =========================================================
class TimeMLP(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

    def forward(self, t):
        # t: (B,)
        return self.net(t[:, None])  # (B, time_dim)


# =========================================================
# Original residual MLP blocks
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.to_scale = nn.Linear(time_dim, hidden_dim)
        self.to_shift = nn.Linear(time_dim, hidden_dim)

        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x, t_emb):
        h = self.fc(self.norm(x))
        h = F.silu(h)

        scale = self.to_scale(t_emb)
        shift = self.to_shift(t_emb)
        h = h * (1.0 + scale) + shift
        return x + h


class FMNet(nn.Module):
    """
    Your original time-conditioned residual MLP.
    """
    def __init__(self, data_dim=2, hidden_dim=128, time_dim=32, n_blocks=2):
        super().__init__()
        self.time_mlp = TimeMLP(time_dim)
        self.inp = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, time_dim) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, data_dim)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = F.silu(self.norm(h))
        return self.out(h)

    def forward_with_feat(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = F.silu(self.norm(h))
        y = self.out(h)
        return y, h


# =========================================================
# U-Net-shaped MLP blocks
# =========================================================
class FiLMResBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_scale1 = nn.Linear(time_dim, dim)
        self.to_shift1 = nn.Linear(time_dim, dim)
        self.to_scale2 = nn.Linear(time_dim, dim)
        self.to_shift2 = nn.Linear(time_dim, dim)

        nn.init.zeros_(self.to_scale1.weight)
        nn.init.zeros_(self.to_scale1.bias)
        nn.init.zeros_(self.to_shift1.weight)
        nn.init.zeros_(self.to_shift1.bias)
        nn.init.zeros_(self.to_scale2.weight)
        nn.init.zeros_(self.to_scale2.bias)
        nn.init.zeros_(self.to_shift2.weight)
        nn.init.zeros_(self.to_shift2.bias)

    def _film(self, h, scale, shift):
        return h * (1.0 + scale) + shift

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.fc1(h)
        h = F.silu(h)
        h = self._film(h, self.to_scale1(t_emb), self.to_shift1(t_emb))

        h = self.norm2(h)
        h = self.fc2(h)
        h = F.silu(h)
        h = self._film(h, self.to_scale2(t_emb), self.to_shift2(t_emb))

        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.res = FiLMResBlock(out_dim, time_dim)

    def forward(self, x, t_emb):
        h = self.proj(x)
        h = F.silu(h)
        h = self.res(h, t_emb)
        return h


class UpBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, time_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim + skip_dim, out_dim)
        self.res = FiLMResBlock(out_dim, time_dim)

    def forward(self, x, skip, t_emb):
        h = torch.cat([x, skip], dim=-1)
        h = self.proj(h)
        h = F.silu(h)
        h = self.res(h, t_emb)
        return h


class FMUNet(nn.Module):
    """
    U-Net-shaped MLP for low-dimensional toy data.
    """
    def __init__(self, data_dim=2, base_dim=128, time_dim=64):
        super().__init__()
        self.time_mlp = TimeMLP(time_dim)

        self.inp = nn.Linear(data_dim, base_dim)

        self.down1 = DownBlock(base_dim, base_dim, time_dim)
        self.down2 = DownBlock(base_dim, base_dim // 2, time_dim)
        self.down3 = DownBlock(base_dim // 2, base_dim // 4, time_dim)

        self.mid1 = FiLMResBlock(base_dim // 4, time_dim)
        self.mid2 = FiLMResBlock(base_dim // 4, time_dim)

        self.up3 = UpBlock(base_dim // 4, base_dim // 4, base_dim // 2, time_dim)
        self.up2 = UpBlock(base_dim // 2, base_dim // 2, base_dim, time_dim)
        self.up1 = UpBlock(base_dim, base_dim, base_dim, time_dim)

        self.norm = nn.LayerNorm(base_dim)
        self.out = nn.Linear(base_dim, data_dim)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h0 = F.silu(self.inp(x))
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)

        m = self.mid1(h3, t_emb)
        m = self.mid2(m, t_emb)

        u3 = self.up3(m, h3, t_emb)
        u2 = self.up2(u3, h2, t_emb)
        u1 = self.up1(u2, h1, t_emb)

        h = F.silu(self.norm(u1))
        return self.out(h)

    def forward_with_feat(self, x, t):
        t_emb = self.time_mlp(t)

        h0 = F.silu(self.inp(x))
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)

        m = self.mid1(h3, t_emb)
        m = self.mid2(m, t_emb)

        u3 = self.up3(m, h3, t_emb)
        u2 = self.up2(u3, h2, t_emb)
        u1 = self.up1(u2, h1, t_emb)

        h = F.silu(self.norm(u1))
        y = self.out(h)
        return y, h


# =========================================================
# EMA / scheduler
# =========================================================
def make_ema(model):
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def ema_update(model, ema_model, decay=0.999):
    for p, q in zip(model.parameters(), ema_model.parameters()):
        q.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def make_warmup_cosine(optimizer, total_steps, warmup_steps=1000, eta_min=1e-6):
    base_lr = optimizer.defaults["lr"]
    min_factor = eta_min / max(base_lr, 1e-12)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        r = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * r))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================
# Flow matching training pairs
# =========================================================
@torch.no_grad()
def sample_fm_training_pairs(x0):
    """
    Straight-line flow matching:
        x_t = (1-t) x0 + t z
        target_v = z - x0

    Returns:
        xt:       (B, D)
        t:        (B,)
        target_v: (B, D)
        z:        (B, D)
    """
    B = x0.shape[0]
    z = torch.randn_like(x0)
    t = torch.rand(B, device=x0.device)
    xt = (1.0 - t[:, None]) * x0 + t[:, None] * z
    target_v = z - x0
    return xt, t, target_v, z


# =========================================================
# Training
# =========================================================
def train_fm(
    model,
    ema_model,
    loader,
    epochs=200,
    lr=1e-3,
    clip_grad=1.0,
    ema_decay=0.999,
    eta_min=1e-5,
    warmup_steps=500,
    device="cuda",
    save_name="fm_final.pth",
):
    model.to(device)
    ema_model.to(device)
    model.train()
    ema_model.eval()

    opt = optim.Adam(model.parameters(), lr=lr)
    total_steps = max(1, epochs * len(loader))
    scheduler = make_warmup_cosine(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=eta_min,
    )

    epoch_losses = []
    epoch_bar = tqdm(range(epochs), desc="FM epochs")

    for epoch in epoch_bar:
        running_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
        for (x0_cpu,) in batch_bar:
            x0 = x0_cpu.to(device, non_blocking=True)

            xt, t, target_v, _ = sample_fm_training_pairs(x0)
            pred_v = model(xt, t)
            loss = ((pred_v - target_v) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()
            scheduler.step()
            ema_update(model, ema_model, decay=ema_decay)

            loss_val = loss.item()
            running_loss += loss_val
            n_batches += 1

            batch_bar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_loss = running_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        epoch_bar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    torch.save(
        {
            "model_state": model.state_dict(),
            "ema_model_state": ema_model.state_dict(),
        },
        save_name,
    )
    print(f"saved FM checkpoint to {save_name}")
    return epoch_losses


# =========================================================
# Sampling
# =========================================================
def make_uniform_schedule(B, device=None):
    # B steps => B+1 time points, descending from 1 to 0
    return torch.linspace(1.0, 0.0, B + 1, device=device)


@torch.no_grad()
def euler_fm_sample(model, n, schedule, device="cuda", x1=None, gen=None, return_traj=False):
    """
    Solve dx/dt = v(x,t) from t=1 to t=0 using explicit Euler on a descending schedule.
    """
    model.eval()
    schedule = schedule.to(device)

    assert schedule.ndim == 1, "schedule must be 1D"
    assert torch.all(schedule[:-1] >= schedule[1:]), "schedule must be nonincreasing"

    dtype = next(model.parameters()).dtype

    if x1 is None:
        x = torch.randn((n, 2), device=device, dtype=dtype, generator=gen)
    else:
        x = x1.to(device=device, dtype=dtype).clone()

    traj = [x.clone()] if return_traj else None
    vel_hist = []

    for k in range(len(schedule) - 1):
        t_k = schedule[k]
        t_next = schedule[k + 1]
        h = t_k - t_next  # positive

        t_batch = torch.full((x.shape[0],), float(t_k), device=device, dtype=dtype)
        v = model(x, t_batch)
        x = x - h * v

        vel_hist.append(v.clone())
        if return_traj:
            traj.append(x.clone())

    if return_traj:
        return x, traj, vel_hist
    return x


# =========================================================
# Dense trajectories + profiles
# =========================================================
@torch.no_grad()
def collect_dense_trajectories(model, n_paths=512, N_dense=1000, device="cuda", gen=None):
    schedule_dense = make_uniform_schedule(N_dense, device=device)
    x0, traj, vel_hist = euler_fm_sample(
        model=model,
        n=n_paths,
        schedule=schedule_dense,
        device=device,
        gen=gen,
        return_traj=True,
    )
    return schedule_dense, traj, vel_hist


@torch.no_grad()
def compute_sharpness_profile_from_velocities(schedule_dense, vel_hist, eps=1e-12):
    """
    Returns:
        dict with:
            t_mid
            accel_norm
            kappa_approx
            kappa_geom
    """
    accel_norm = []
    kappa_geom = []
    kappa_approx = []
    t_mid = []

    for i in range(len(vel_hist) - 1):
        v1 = vel_hist[i]
        v2 = vel_hist[i + 1]
        dt = float(schedule_dense[i] - schedule_dense[i + 1])  # positive

        # use sign-consistent derivative wrt descending time grid
        a_hat = (v1 - v2) / max(dt, eps)

        v1_sq = (v1 * v1).sum(dim=1, keepdim=True).clamp_min(eps)
        a_dot_v = (a_hat * v1).sum(dim=1, keepdim=True)

        a_par = (a_dot_v / v1_sq) * v1
        a_orth = a_hat - a_par

        accel = torch.norm(a_hat, dim=1)
        kap_app = torch.norm(a_hat, dim=1) / torch.sqrt(v1_sq.squeeze(1))
        kap_geo = torch.norm(a_orth, dim=1) / v1_sq.squeeze(1)

        accel_norm.append(accel.mean())
        kappa_approx.append(kap_app.mean())
        kappa_geom.append(kap_geo.mean())
        t_mid.append(0.5 * (schedule_dense[i] + schedule_dense[i + 1]))

    return {
        "t_mid": torch.stack(t_mid),
        "accel_norm": torch.stack(accel_norm),
        "kappa_approx": torch.stack(kappa_approx),
        "kappa_geom": torch.stack(kappa_geom),
    }


# =========================================================
# Adaptive schedule utilities
# =========================================================
def gaussian_smooth_1d_torch(x, sigma=5.0):
    if sigma <= 0:
        return x.clone()

    radius = max(1, int(3 * sigma))
    grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()

    x_pad = F.pad(x[None, None, :], (radius, radius), mode="reflect")
    y = F.conv1d(x_pad, kernel[None, None, :])[0, 0]
    return y


@torch.no_grad()
def build_importance_density(profile_vals, sigma=5.0, eps=1e-12):
    phi = profile_vals.clamp_min(0.0)
    phi = gaussian_smooth_1d_torch(phi, sigma=sigma)

    s = phi.sum()
    if s <= eps:
        return torch.full_like(phi, 1.0 / len(phi))
    return phi / s


@torch.no_grad()
def build_adaptive_schedule_robust(
    t_support_desc,
    profile_vals,
    B,
    sigma=12.0,
    temper="log1p",
    power=0.5,
    mix_uniform=0.25,
    trim_frac=0.02,
    eps=1e-12,
    min_step=1e-6,
    verbose=True,
):
    """
    t_support_desc: descending support, e.g. profile["t_mid"]
    profile_vals: same length as support
    returns:
        schedule_adaptive: (B+1,) descending
        phi_processed
        p_desc
    """
    assert t_support_desc.ndim == 1
    assert profile_vals.ndim == 1
    assert t_support_desc.numel() == profile_vals.numel()
    assert bool(torch.all(t_support_desc[:-1] >= t_support_desc[1:]))

    assert B * min_step < 1.0, "min_step too large for requested budget"

    phi = profile_vals.clone().float().clamp_min(0.0)
    M = phi.numel()

    if verbose:
        print("\n[build_adaptive_schedule_robust] INPUT")
        print("  t_support_desc shape:", tuple(t_support_desc.shape))
        print("  profile_vals shape:  ", tuple(profile_vals.shape))
        print("  first/last t:", float(t_support_desc[0]), float(t_support_desc[-1]))
        print("  profile min/max:", float(phi.min()), float(phi.max()))

    trim = int(trim_frac * M)
    if trim > 0:
        phi[:trim] = phi[:trim].median()
        phi[-trim:] = phi[-trim:].median()

    if temper == "log1p":
        phi = torch.log1p(phi)
    elif temper == "sqrt":
        phi = torch.sqrt(phi)
    elif temper == "pow":
        phi = phi.pow(power)
    elif temper is None:
        pass
    else:
        raise ValueError(f"Unknown temper={temper}")

    phi = gaussian_smooth_1d_torch(phi, sigma=sigma)

    if phi.sum() <= eps:
        phi = torch.ones_like(phi)

    phi = (1.0 - mix_uniform) * phi + mix_uniform * torch.ones_like(phi)
    p_desc = phi / phi.sum().clamp_min(eps)

    if verbose:
        print("\n  Probability checks")
        print("  p_desc sum:", float(p_desc.sum()))
        print("  p_desc min/max:", float(p_desc.min()), float(p_desc.max()))

    t_support_asc = torch.flip(t_support_desc, dims=[0])
    p_asc = torch.flip(p_desc, dims=[0])
    cdf_asc = torch.cumsum(p_asc, dim=0)

    t_aug = torch.cat([
        torch.zeros(1, device=t_support_desc.device, dtype=t_support_desc.dtype),
        t_support_asc,
        torch.ones(1, device=t_support_desc.device, dtype=t_support_desc.dtype),
    ], dim=0)

    cdf_aug = torch.cat([
        torch.zeros(1, device=cdf_asc.device, dtype=cdf_asc.dtype),
        cdf_asc,
        torch.ones(1, device=cdf_asc.device, dtype=cdf_asc.dtype),
    ], dim=0)

    cdf_aug = cdf_aug.clamp(0.0, 1.0)
    cdf_aug, _ = torch.cummax(cdf_aug, dim=0)

    u = torch.linspace(0.0, 1.0, B + 1, device=t_support_desc.device, dtype=t_support_desc.dtype)
    t_quant_asc = torch.empty_like(u)

    for i, ui in enumerate(u):
        idx = torch.searchsorted(cdf_aug, ui, right=False).item()
        if idx <= 0:
            t_quant_asc[i] = t_aug[0]
        elif idx >= len(cdf_aug):
            t_quant_asc[i] = t_aug[-1]
        else:
            c0, c1 = cdf_aug[idx - 1], cdf_aug[idx]
            t0, t1 = t_aug[idx - 1], t_aug[idx]
            w = (ui - c0) / (c1 - c0 + eps)
            t_quant_asc[i] = t0 + w * (t1 - t0)

    schedule_adaptive = torch.flip(t_quant_asc, dims=[0]).contiguous()
    schedule_adaptive[0] = 1.0
    schedule_adaptive[-1] = 0.0

    for i in range(len(schedule_adaptive) - 2, 0, -1):
        if schedule_adaptive[i] >= schedule_adaptive[i - 1] - min_step:
            schedule_adaptive[i] = schedule_adaptive[i - 1] - min_step
        schedule_adaptive[i] = schedule_adaptive[i].clamp(min=schedule_adaptive[i + 1] + min_step)

    schedule_adaptive[0] = 1.0
    schedule_adaptive[-1] = 0.0

    step_sizes = schedule_adaptive[:-1] - schedule_adaptive[1:]

    if verbose:
        print("\n[build_adaptive_schedule_robust] OUTPUT")
        print("  first 10:", schedule_adaptive[:10])
        print("  last 10: ", schedule_adaptive[-10:])
        print("  descending:", bool(torch.all(schedule_adaptive[:-1] >= schedule_adaptive[1:])))
        print("  step min/max:", float(step_sizes.min()), float(step_sizes.max()))
        print("  zero steps:", int((step_sizes <= 0).sum().item()))

    assert torch.all(schedule_adaptive[:-1] > schedule_adaptive[1:]), "schedule has duplicate/non-descending steps"

    return schedule_adaptive, phi, p_desc


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_schedule_rmse(model, schedule_test, schedule_ref, n_eval=2048, device="cuda", seed=0):
    """
    Compare low-budget schedule against dense reference on the SAME x1.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    dtype = next(model.parameters()).dtype
    x1 = torch.randn((n_eval, 2), device=device, generator=g, dtype=dtype)

    x_ref = euler_fm_sample(model, n_eval, schedule_ref, device=device, x1=x1)
    x_test = euler_fm_sample(model, n_eval, schedule_test, device=device, x1=x1)

    rmse = torch.sqrt(((x_ref - x_test) ** 2).sum(dim=1).mean())
    return float(rmse), x_ref, x_test


# =========================================================
# Experiment driver
# =========================================================
@dataclass
class FMConfig:
    model_type: str = "mlp"   # "mlp" or "unet"
    n_modes: int = 9
    n_samples_per_mode: int = 3000
    std: float = 0.05
    seed: int = 0

    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    clip_grad: float = 1.0
    ema_decay: float = 0.999
    eta_min: float = 1e-5
    warmup_steps: int = 200

    hidden_dim: int = 128
    time_dim: int = 32
    n_blocks: int = 2
    base_dim: int = 128

    N_dense: int = 1000
    n_paths_profile: int = 1024
    save_name: str = "fm_gaussian_grid.pth"


def make_model(cfg: FMConfig):
    if cfg.model_type == "mlp":
        return FMNet(
            data_dim=2,
            hidden_dim=cfg.hidden_dim,
            time_dim=cfg.time_dim,
            n_blocks=cfg.n_blocks,
        )
    elif cfg.model_type == "unet":
        return FMUNet(
            data_dim=2,
            base_dim=cfg.base_dim,
            time_dim=max(cfg.time_dim, 64),
        )
    else:
        raise ValueError(f"Unknown model_type={cfg.model_type}")


def run_default_experiment(cfg: FMConfig | None = None):
    if cfg is None:
        cfg = FMConfig()

    device = get_device()
    print("device:", device)

    X, centers = make_gaussian_grid(
        n_modes=cfg.n_modes,
        n_samples=cfg.n_samples_per_mode,
        std=cfg.std,
        seed=cfg.seed,
    )
    loader = make_loader(X, batch_size=cfg.batch_size, shuffle=True, num_workers=0, device=device)

    model = make_model(cfg).to(device)
    ema_model = make_ema(model)

    losses = train_fm(
        model,
        ema_model,
        loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        clip_grad=cfg.clip_grad,
        ema_decay=cfg.ema_decay,
        eta_min=cfg.eta_min,
        warmup_steps=cfg.warmup_steps,
        device=device,
        save_name=cfg.save_name,
    )

    g_off = torch.Generator(device=device).manual_seed(123)
    schedule_dense, traj_dense, vel_hist_dense = collect_dense_trajectories(
        ema_model,
        n_paths=cfg.n_paths_profile,
        N_dense=cfg.N_dense,
        device=device,
        gen=g_off,
    )
    profile = compute_sharpness_profile_from_velocities(schedule_dense, vel_hist_dense)

    return {
        "X": X,
        "centers": centers,
        "model": model,
        "ema_model": ema_model,
        "losses": losses,
        "schedule_dense": schedule_dense,
        "traj_dense": traj_dense,
        "vel_hist_dense": vel_hist_dense,
        "profile": profile,
        "device": device,
        "cfg": cfg,
    }


if __name__ == "__main__":
    cfg = FMConfig(
        model_type="mlp",   # change to "unet" to use FMUNet
        n_modes=9,
        n_samples_per_mode=3000,
        std=0.05,
        epochs=50,
        save_name="fm_gaussian_grid_demo.pth",
    )
    out = run_default_experiment(cfg)

    profile = out["profile"]
    device = out["device"]
    ema_model = out["ema_model"]
    schedule_dense = out["schedule_dense"]

    # Example adaptive schedule
    B = 10
    profile_for_sched = torch.sqrt(profile["accel_norm"] + 1e-12)

    schedule_adaptive, _, _ = build_adaptive_schedule_robust(
        t_support_desc=profile["t_mid"],
        profile_vals=profile_for_sched,
        B=B,
        sigma=12.0,
        temper=None,
        mix_uniform=0.35,
        trim_frac=0.03,
        verbose=True,
    )

    schedule_uniform = make_uniform_schedule(B, device=device)

    rmse_u, _, _ = evaluate_schedule_rmse(
        ema_model, schedule_uniform, schedule_dense, n_eval=2048, device=device, seed=42
    )
    rmse_a, _, _ = evaluate_schedule_rmse(
        ema_model, schedule_adaptive, schedule_dense, n_eval=2048, device=device, seed=42
    )

    print("\nDemo comparison")
    print(f"Uniform RMSE : {rmse_u:.4f}")
    print(f"Adaptive RMSE: {rmse_a:.4f}")
    print(f"Improvement  : {(rmse_u - rmse_a) / rmse_u * 100:.2f}%")