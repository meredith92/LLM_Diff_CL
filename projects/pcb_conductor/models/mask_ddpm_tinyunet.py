# projects/pcb_conductor/models/mask_ddpm_tinyunet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# DDPM beta schedule
# -------------------------
def make_beta_schedule(num_timesteps: int,
                       beta_start: float = 1e-4,
                       beta_end: float = 2e-2,
                       schedule: str = "linear") -> torch.Tensor:
    """
    Returns betas: [T]
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule == "cosine":
        # Nichol & Dhariwal cosine schedule
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        s = 0.008
        alphas_bar = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-6, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# -------------------------
# Time embedding
# -------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] int64 timesteps
        returns: [B, dim]
        """
        half = self.dim // 2
        device = t.device
        emb_scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -emb_scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# -------------------------
# UNet blocks
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time = nn.Linear(tdim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, in_ch, H, W]
        t_emb: [B, tdim]
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb)).view(t_emb.size(0), -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyUNet(nn.Module):
    """
    Tiny UNet for 1-channel mask DDPM.
    Input:  [B,1,H,W] in [-1,1] (or noisy sample x_t)
    Output: [B,1,H,W] predicted noise epsilon
    """
    def __init__(self, base: int = 64, tdim: int = 256):
        super().__init__()
        self.tdim = tdim

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        self.rb1 = ResBlock(base, base, tdim)
        self.down1 = Down(base)

        self.rb2 = ResBlock(base, base * 2, tdim)
        self.down2 = Down(base * 2)

        self.mid1 = ResBlock(base * 2, base * 2, tdim)
        self.mid2 = ResBlock(base * 2, base * 2, tdim)

        self.up2 = Up(base * 2)
        self.rb3 = ResBlock(base * 4, base, tdim)

        self.up1 = Up(base)
        self.rb4 = ResBlock(base * 2, base, tdim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W]
        t: [B] int64
        """
        t_emb = self.time_emb(t)

        x0 = self.in_conv(x)          # [B,base,H,W]
        x1 = self.rb1(x0, t_emb)
        d1 = self.down1(x1)           # [B,base,H/2,W/2]

        x2 = self.rb2(d1, t_emb)
        d2 = self.down2(x2)           # [B,2base,H/4,W/4]

        m = self.mid1(d2, t_emb)
        m = self.mid2(m, t_emb)

        u2 = self.up2(m)              # [B,2base,H/2,W/2]
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.rb3(u2, t_emb)      # [B,base,H/2,W/2]

        u1 = self.up1(u2)             # [B,base,H,W]
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.rb4(u1, t_emb)      # [B,base,H,W]

        out = self.out_conv(F.silu(self.out_norm(u1)))  # [B,1,H,W]
        return out
