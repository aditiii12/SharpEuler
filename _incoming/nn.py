import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Time embedding
# -----------------------------
class TimeMLP(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim), nn.SiLU()
        )

    def forward(self, t):   # t: (B,)
        return self.net(t[:, None])  # (B, time_dim)


# -----------------------------
# FiLM residual MLP block
# -----------------------------
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

        nn.init.zeros_(self.to_scale1.weight); nn.init.zeros_(self.to_scale1.bias)
        nn.init.zeros_(self.to_shift1.weight); nn.init.zeros_(self.to_shift1.bias)
        nn.init.zeros_(self.to_scale2.weight); nn.init.zeros_(self.to_scale2.bias)
        nn.init.zeros_(self.to_shift2.weight); nn.init.zeros_(self.to_shift2.bias)

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


# -----------------------------
# Down block
# -----------------------------
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


# -----------------------------
# Up block with skip connection
# -----------------------------
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


# -----------------------------
# U-Net-shaped MLP for FM
# -----------------------------
class FMUNet(nn.Module):
    """
    U-Net-like architecture in feature space for low-dim toy data.
    Input/output signature matches your FMNet:
        x: (B, data_dim)
        t: (B,)
        returns v(x,t): (B, data_dim)
    """
    def __init__(self, data_dim=2, base_dim=128, time_dim=64):
        super().__init__()
        self.time_mlp = TimeMLP(time_dim)

        # input stem
        self.inp = nn.Linear(data_dim, base_dim)

        # encoder
        self.down1 = DownBlock(base_dim, base_dim, time_dim)          # 128
        self.down2 = DownBlock(base_dim, base_dim // 2, time_dim)     # 64
        self.down3 = DownBlock(base_dim // 2, base_dim // 4, time_dim)  # 32

        # bottleneck
        self.mid1 = FiLMResBlock(base_dim // 4, time_dim)
        self.mid2 = FiLMResBlock(base_dim // 4, time_dim)

        # decoder
        self.up3 = UpBlock(base_dim // 4, base_dim // 4, base_dim // 2, time_dim)  # 32+32 -> 64
        self.up2 = UpBlock(base_dim // 2, base_dim // 2, base_dim, time_dim)        # 64+64 -> 128
        self.up1 = UpBlock(base_dim, base_dim, base_dim, time_dim)                   # 128+128 -> 128

        self.norm = nn.LayerNorm(base_dim)
        self.out = nn.Linear(base_dim, data_dim)

        # start near zero vector field
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h0 = F.silu(self.inp(x))     # (B, 128)

        # encoder
        h1 = self.down1(h0, t_emb)   # (B, 128)
        h2 = self.down2(h1, t_emb)   # (B, 64)
        h3 = self.down3(h2, t_emb)   # (B, 32)

        # bottleneck
        m = self.mid1(h3, t_emb)
        m = self.mid2(m, t_emb)

        # decoder
        u3 = self.up3(m, h3, t_emb)  # (B, 64)
        u2 = self.up2(u3, h2, t_emb) # (B, 128)
        u1 = self.up1(u2, h1, t_emb) # (B, 128)

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