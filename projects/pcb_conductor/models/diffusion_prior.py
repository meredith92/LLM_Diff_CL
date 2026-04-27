# projects/pcb_conductor/models/diffusion_prior.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_ddpm_tinyunet import TinyUNet, make_beta_schedule  # 你复制过去的文件
from .mask_ddpm_backbones import UNetSmall,AttnUNet, ResUNet,CLIPTextEncoder# 或 AttnUNet / ResUNet


PROMPT_PINS = (
    "thin metallic silver leads or wires, "
    "narrow and reflective, separated from each other, "
    "binary mask of only the silver parts, "
    "exclude gold or yellow pads and large rectangles"
)

class DiffusionPriorConfig:
    def __init__(self, ckpt=None, num_train_timesteps=1000, beta_start=1e-4, beta_end=2e-2, beta_schedule="linear"):
        self.ckpt = ckpt
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule


class DiffusionPrior(nn.Module):
    def __init__(self, cfg: DiffusionPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.num_train_timesteps

        # self.unet = TinyUNet(base=64, tdim=256)
        self.unet = AttnUNet(base=96, tdim=256,cond_dim=512)

        self.clip_text_encoder = CLIPTextEncoder(
            prompt_set=PROMPT_PINS,
            pool="eos",
            cache_on_gpu=True,
        )
        # load ckpt (supports raw state_dict OR dict with "model")
        if cfg.ckpt:
            if not os.path.isfile(cfg.ckpt):
                raise FileNotFoundError(f"Diffusion ckpt not found: {cfg.ckpt}")
            state = torch.load(cfg.ckpt, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.unet.load_state_dict(state, strict=True)

        self.unet.eval()

        betas = make_beta_schedule(self.T, cfg.beta_start, cfg.beta_end, schedule=cfg.beta_schedule)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, steps: int = 20) -> torch.Tensor:
        """
        condition: [B,1,H,W] in [0,1]
        return:    [B,1,H,W] in [0,1]
        """
        device = condition.device
        self.unet.to(device)
        # text_encoder = CLIPTextEncoder(
        #     model_name="openai/clip-vit-base-patch32",
        #     device=str(device),
        #     max_length=77,
        #     normalize=True,
        # ).to(device)

        B, C, H, W = condition.shape
        x = torch.randn((B, C, H, W), device=device)

        # fast sampling timesteps
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()

        # weak tether to condition for stability
        cond = condition.clamp(0, 1) * 2 - 1
        gamma = 0.10
        # llm
        guidance_scale = 2.0  # 1.5~3.0 常用

        for t in ts:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            #eps = self.unet(x, t_batch)
            #llm
            prompt_id = torch.zeros(B, dtype=torch.long, device=device)  # 全是 0
            cond_emb = self.clip_text_encoder(prompt_ids=prompt_id)  # [B, D] 查表
            # cond_emb = text_encoder([PROMPT_PINS] * B)  # 或者用缓存 expand 的方式

            eps_uncond = self.unet(x, t_batch, None)
            eps_cond = self.unet(x, t_batch, cond_emb)

            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            ab_t = self.alphas_bar[t]
            x0 = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t + 1e-8)

            if t > 0:
                ab_prev = self.alphas_bar[t - 1]
                # simple mean update
                x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
                # add noise (keeps diversity for uncertainty)
                x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)
            else:
                x = x0

            x = (1 - gamma) * x + gamma * cond

        return torch.sigmoid(x)





