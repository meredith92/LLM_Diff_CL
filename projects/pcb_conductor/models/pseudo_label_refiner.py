# projects/xxx/models/pseudo_label_refiner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_prior import DiffusionPrior, DiffusionPriorConfig


@dataclass
class PseudoLabelRefinerConfig:
    # diffusion prior
    diffusion: DiffusionPriorConfig = DiffusionPriorConfig()

    # uncertainty sampling
    K: int = 2                  # number of diffusion samples (2 is enough to start)
    steps: int = 20             # diffusion denoise steps per sample
    downsample: int = 4         # do diffusion on lower-res to save time
    u_thr: float = 0.15         # uncertainty threshold for hard gating

    # weighting mode
    mode: str = "hard"          # "hard" or "soft"
    alpha: float = 5.0          # for soft weighting: exp(-alpha * u)

    # safety
    eps: float = 1e-6


class PseudoLabelRefiner(nn.Module):
    """
    Turn teacher prob p_t into:
      - uncertainty map u in [0,1]
      - diffusion weight map w_diff (hard or soft)
      - optionally fuse with existing selective weight map w_sel

    Typical usage:
      u, w_diff, w = refiner(p_t, w_sel=w_sel)
    """
    def __init__(self, cfg: PseudoLabelRefinerConfig):
        super().__init__()
        self.cfg = cfg
        self.prior = DiffusionPrior(cfg.diffusion)

    @torch.no_grad()
    def forward(
        self,
        p_t: torch.Tensor,
        w_sel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        p_t:   teacher probability, [B,1,H,W] in [0,1]
        w_sel: your existing selective_weight_map, [B,1,H,W] (0/1 or 0~1)

        returns:
          u:      uncertainty map [B,1,H,W] in [0,1]
          w_diff: diffusion-based weight [B,1,H,W]
          w:      fused weight [B,1,H,W]
        """
        assert p_t.dim() == 4, f"p_t should be [B,1,H,W], got {p_t.shape}"
        p_t = p_t.clamp(0, 1)

        # 1) Downsample for diffusion (speed)
        if self.cfg.downsample > 1:
            p_in = F.interpolate(
                p_t, scale_factor=1.0 / self.cfg.downsample,
                mode="bilinear", align_corners=False
            )
        else:
            p_in = p_t

        # 2) K samples -> variance (uncertainty)
        samples = []
        for _ in range(self.cfg.K):
            yk = self.prior.sample(condition=p_in, steps=self.cfg.steps)  # [B,1,h,w] in [0,1]
            samples.append(yk)

        S = torch.stack(samples, dim=0)               # [K,B,1,h,w]
        u = S.var(dim=0, unbiased=False)              # [B,1,h,w]

        # 3) Normalize uncertainty to [0,1] per-image for stable thresholding
        B = u.shape[0]
        u_flat = u.view(B, -1)
        u_max = u_flat.max(dim=1)[0].view(B, 1, 1, 1).clamp_min(self.cfg.eps)
        u = (u / u_max).clamp(0, 1)

        # 4) Upsample back
        if self.cfg.downsample > 1:
            u = F.interpolate(u, size=p_t.shape[-2:], mode="bilinear", align_corners=False)

        # 5) Convert to diffusion weights
        if self.cfg.mode == "hard":
            w_diff = (u < self.cfg.u_thr).float()
        elif self.cfg.mode == "soft":
            w_diff = torch.exp(-self.cfg.alpha * u).clamp(0, 1)
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}, choose from ['hard','soft']")

        # 6) Fuse with existing selective weights
        if w_sel is None:
            w = w_diff
        else:
            # ensure same dtype/device
            w = w_sel.to(w_diff.dtype) * w_diff

        return u, w_diff, w
