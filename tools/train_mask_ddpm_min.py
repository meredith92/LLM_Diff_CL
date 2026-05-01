import os
import math
import glob
import argparse
from dataclasses import dataclass
from typing import List
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from projects.pcb_conductor.models.mask_ddpm_backbones import UNetSmall,make_beta_schedule,AttnUNet,ResUNet,CLIPTextEncoder
# import torch
# import torch.nn.functional as F
# from transformers import CLIPTokenizer, CLIPTextModel

PROMPT_PINS = (
    "a row of separate vertical rectangular pins, "
    "uniform width, evenly spaced, flat bottoms, "
    "not connected to each other, clean binary mask"
)


# register_all_modules(init_default_scope=True)
# -------------------------
# Utilities: DDPM schedule
# -------------------------
# def make_beta_schedule(num_timesteps: int, beta_start=1e-4, beta_end=2e-2, schedule="linear"):
#     if schedule == "linear":
#         return torch.linspace(beta_start, beta_end, num_timesteps)
#     elif schedule == "cosine":
#         # cosine schedule (Nichol & Dhariwal style)
#         # produces alphas_bar then convert to betas
#         steps = num_timesteps + 1
#         x = torch.linspace(0, num_timesteps, steps)
#         s = 0.008
#         alphas_bar = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
#         alphas_bar = alphas_bar / alphas_bar[0]
#         betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
#         return betas.clamp(1e-6, 0.999)
#     else:
#         raise ValueError(f"Unknown schedule: {schedule}")


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """Extract a[t] for each batch element, reshape to [B,1,1,1] broadcastable."""
    B = t.shape[0]
    out = a.gather(0, t).view(B, 1, 1, 1)
    return out.expand(x_shape)


# -------------------------
# Dataset
# -------------------------
class MaskFolderDataset(Dataset):
    """
    Read masks from a folder (recursively). Each mask is converted to 1xHxW float in [0,1].
    Assumes masks are single-channel or RGB; we convert to L.
    """
    def __init__(self, root: str, image_size: int = 128):
        self.paths: List[str] = []
        exts = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        for e in exts:
            self.paths += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No mask images found under: {root}")

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # [0,1], shape [1,H,W] for L
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("L")
        x = self.tf(img)  # [1,H,W] in [0,1]
        # binarize softly (works for 0/255 or already soft)
        x = (x > 0.5).float()
        # map to [-1, 1] as typical DDPM input range
        x = x * 2.0 - 1.0
        return x




# -------------------------
# Training
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_root", type=str, required=True, help="Folder containing mask images.")
    ap.add_argument("--out_dir", type=str, default="work_dirs/mask_ddpm_min")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num_timesteps", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MaskFolderDataset(args.mask_root, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    text_encoder = CLIPTextEncoder(
        model_name="openai/clip-vit-base-patch32",
        device=str(device),
        max_length=77,
        normalize=True,
    ).to(device)

    # _cached_prompt = None
    _cached_emb1 = None  # [1, 512] for ViT-B/32

    @torch.no_grad()
    def get_cond_emb(batch_size: int, device: torch.device, drop_prob: float = 0.1):
        """
        Returns:
            cond_emb: [B, 512] or None  (None 用于 CFG 的 unconditional)
        """
        nonlocal _cached_emb1

        # CFG dropout (train only)
        if drop_prob > 0 and torch.rand(1, device=device).item() < drop_prob:
            return None

        if _cached_emb1 is None:
            _cached_emb1 = text_encoder(PROMPT_PINS).to(device)  # [1,512]

        return _cached_emb1.expand(batch_size, -1).contiguous()

    #model = AttnUNet(base=96, tdim=256).to(device)
    #llm
    model = AttnUNet(
        base=96,
        tdim=256,
        cond_dim=512  # 👈 新增
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    betas = make_beta_schedule(args.num_timesteps, args.beta_start, args.beta_end, schedule=args.schedule).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    step = 0
    model.train()
    for ep in range(1, args.epochs + 1):
        for x0 in dl:
            x0 = x0.to(device, non_blocking=True)  # [B,1,H,W] in [-1,1]
            B = x0.size(0)

            t = torch.randint(0, args.num_timesteps, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)

            a_bar = extract(alphas_bar, t, x0.shape)
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                #no llm
                #pred_noise = model(x_t, t)
                # llm
                B = x_t.size(0)
                cond_emb = get_cond_emb(B, x_t.device, drop_prob=0.1)  # 10% unconditional for CFG training

                pred_noise = model(x_t, t, cond_emb)
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            step += 1
            if step % 100 == 0:
                print(f"[ep {ep:03d}] step {step:06d}  loss={loss.item():.6f}")

        if ep % args.save_every == 0 or ep == args.epochs:
            ckpt = {
                "model": model.state_dict(),
                "epoch": ep,
                "args": vars(args),
            }
            path = os.path.join(args.out_dir, f"unet_ep{ep:03d}.pth")
            torch.save(ckpt, path)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "unet_llm.pth"))
            print(f"[OK] saved: {path}")

    print("[DONE] training finished.")


if __name__ == "__main__":
    main()
