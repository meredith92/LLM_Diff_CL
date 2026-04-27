import math
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Optional

# -------- beta schedule (optional reuse) --------
def make_beta_schedule(num_timesteps: int,
                       beta_start: float = 1e-4,
                       beta_end: float = 2e-2,
                       schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule == "cosine":
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        s = 0.008
        alphas_bar = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-6, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        max_length: int = 77,
        normalize: bool = True,
        use_cls_token: bool = False,   # 兼容你原来的参数
        prompt_set: Optional[List[str]] = None,
        pool: str = "eos",             # "eos" | "cls" | "mean" | "pooler"
        cache_on_gpu: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_length = max_length
        self.normalize = normalize
        self.use_cls_token = use_cls_token
        self.pool = pool

        self.prompt_set = prompt_set
        self.prompt2id = {p: i for i, p in enumerate(prompt_set)} if prompt_set else None

        # 只有在需要构建 cache 时才加载 tokenizer/model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.register_buffer("text_cache", None, persistent=False)  # [N, D]
        if prompt_set is not None and len(prompt_set) > 0:
            self._build_cache(prompt_set, cache_on_gpu=cache_on_gpu)

    @torch.no_grad()
    def _encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.text_model(**inputs)
        last = out.last_hidden_state  # [B, T, D]

        # 兼容你原来的 use_cls_token
        if self.use_cls_token or self.pool == "cls":
            emb = last[:, 0, :]
        elif self.pool == "pooler" and hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        elif self.pool == "mean":
            # 只对有效 token 平均更合理
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            # 默认：EOS/最后有效 token 向量（CLIP 常见用法）
            eos_idx = inputs["attention_mask"].sum(dim=1) - 1  # [B]
            emb = last[torch.arange(last.size(0), device=last.device), eos_idx]

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb  # [B, D]

    @torch.no_grad()
    def _build_cache(self, prompt_set: list[str], batch_size: int = 128, cache_on_gpu: bool = True):
        feats = []
        for i in range(0, len(prompt_set), batch_size):
            feats.append(self._encode_prompts(prompt_set[i:i+batch_size]).detach().cpu())
        cache = torch.cat(feats, dim=0)  # [N, D]
        if cache_on_gpu:
            cache = cache.to(self.device, non_blocking=True)
        self.text_cache = cache

    @torch.no_grad()
    def forward(self, prompts=None, prompt_ids=None):
        """
        用法：
          1) forward(prompt_ids=LongTensor[B])  # 最快（推荐你训练时用这个）
          2) forward(prompts=list[str])         # 会把 str 映射成 id 再查表（仍然快）
          3) prompt_set=None 时，退化为“即时编码”（和你原来一样）
        """
        # 有 cache：走查表
        if self.text_cache is not None:
            if prompt_ids is None:
                if prompts is None:
                    raise ValueError("Either prompts or prompt_ids must be provided.")
                if isinstance(prompts, str):
                    prompts = [prompts]
                if self.prompt2id is None:
                    raise ValueError("prompt_set is None, cannot map prompts to ids.")
                prompt_ids = torch.tensor([self.prompt2id[p] for p in prompts], dtype=torch.long, device=self.device)
            else:
                if not torch.is_tensor(prompt_ids):
                    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
                else:
                    prompt_ids = prompt_ids.to(self.device)

            return self.text_cache.index_select(0, prompt_ids)  # [B, D]

        # 没 cache：退化为即时编码（你的原逻辑）
        if prompts is None:
            raise ValueError("prompt_set is None; please pass prompts for on-the-fly encoding.")
        if isinstance(prompts, str):
            prompts = [prompts]
        return self._encode_prompts(prompts)
# class CLIPTextEncoder(torch.nn.Module):
#     def __init__(
#         self,
#         model_name: str = "openai/clip-vit-base-patch32",
#         device: str = "cuda",
#         max_length: int = 77,
#         normalize: bool = True,
#         use_cls_token: bool = False,
#     ):
#         super().__init__()
#         self.device = torch.device(device)
#         self.max_length = max_length
#         self.normalize = normalize
#         self.use_cls_token = use_cls_token
#
#         self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
#         self.text_model = CLIPTextModel.from_pretrained(model_name).to(self.device)
#         self.text_model.eval()
#         for p in self.text_model.parameters():
#             p.requires_grad = False
#
#     @torch.no_grad()
#     def forward(self, prompts):
#         if isinstance(prompts, str):
#             prompts = [prompts]
#
#         inputs = self.tokenizer(
#             prompts,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         ).to(self.device)
#
#         out = self.text_model(**inputs)
#         if self.use_cls_token:
#             emb = out.last_hidden_state[:, 0, :]  # [B, D]
#         else:
#             # 通常可用；如你的 transformers 版本没有 pooler_output，就改为 mean pooling
#             if hasattr(out, "pooler_output") and out.pooler_output is not None:
#                 emb = out.pooler_output
#             else:
#                 emb = out.last_hidden_state.mean(dim=1)
#
#         if self.normalize:
#             emb = F.normalize(emb, p=2, dim=-1)
#
#         return emb  # [B, D]

# -------- time embedding --------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        emb_scale = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -emb_scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _gn(ch: int) -> nn.GroupNorm:
    # small ch might not be divisible by 8; choose a safe group count
    groups = 8
    while ch % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(tdim, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        #llm
        self.cond_proj = nn.Linear(tdim, out_ch)

    # def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
    #     h = self.conv1(F.silu(self.norm1(x)))
    #     h = h + self.time(F.silu(t_emb)).view(t_emb.size(0), -1, 1, 1)
    #     h = self.conv2(self.dropout(F.silu(self.norm2(h))))
    #     return h + self.skip(x)

    #llm
    def forward(self, x, t_emb, cond_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        t_out = self.time(t_emb)

        if cond_emb is not None:
            c_out = self.cond_proj(cond_emb)
            t_out = t_out  + c_out  # 👈 核心：语言调制

        h = h + t_out[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

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


# -------- attention block (2D) --------
class SelfAttention2d(nn.Module):
    """
    Lightweight MHSA over H*W tokens. Use on low-res stages (e.g., 32x32 or 16x16).
    """
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)


        # [B, heads, head_dim, HW]
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        # attention: [B, heads, HW, HW]
        q = q.permute(0, 1, 3, 2)  # [B,heads,HW,head_dim]
        k = k.permute(0, 1, 2, 3)  # [B,heads,head_dim,HW]
        attn = torch.matmul(q, k) / math.sqrt(head_dim)
        attn = attn.softmax(dim=-1)

        v = v.permute(0, 1, 3, 2)  # [B,heads,HW,head_dim]
        out = torch.matmul(attn, v)  # [B,heads,HW,head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        out = self.proj(out)
        return x + out

class UNetSmall(nn.Module):
    """
    A stronger drop-in replacement for TinyUNet.
    """

    def __init__(self, base: int = 96, tdim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        # down 1
        self.rb1a = ResBlock(base, base, tdim, dropout)
        self.rb1b = ResBlock(base, base, tdim, dropout)
        self.down1 = Down(base)

        # down 2
        self.rb2a = ResBlock(base, base * 2, tdim, dropout)
        self.rb2b = ResBlock(base * 2, base * 2, tdim, dropout)
        self.down2 = Down(base * 2)

        # down 3 (lowest)
        self.rb3a = ResBlock(base * 2, base * 3, tdim, dropout)
        self.rb3b = ResBlock(base * 3, base * 3, tdim, dropout)

        # up 2
        self.up2 = Up(base * 3)
        self.rb_up2a = ResBlock(base * 3 + base * 2, base * 2, tdim, dropout)
        self.rb_up2b = ResBlock(base * 2, base * 2, tdim, dropout)

        # up 1
        self.up1 = Up(base * 2)
        self.rb_up1a = ResBlock(base * 2 + base, base, tdim, dropout)
        self.rb_up1b = ResBlock(base, base, tdim, dropout)

        self.out_norm = _gn(base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        x0 = self.in_conv(x)

        x1 = self.rb1b(self.rb1a(x0, t_emb), t_emb)
        d1 = self.down1(x1)

        x2 = self.rb2b(self.rb2a(d1, t_emb), t_emb)
        d2 = self.down2(x2)

        x3 = self.rb3b(self.rb3a(d2, t_emb), t_emb)

        u2 = self.up2(x3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.rb_up2b(self.rb_up2a(u2, t_emb), t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.rb_up1b(self.rb_up1a(u1, t_emb), t_emb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out

class AttnUNet(nn.Module):
    """
    UNetSmall + attention at the bottleneck (and optionally one up stage).
    """

    def __init__(self, base: int = 96, tdim: int = 256, dropout: float = 0.0, heads: int = 4,cond_dim=512):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        self.rb1 = ResBlock(base, base, tdim, dropout)
        self.down1 = Down(base)

        self.rb2 = ResBlock(base, base * 2, tdim, dropout)
        self.down2 = Down(base * 2)

        self.rb3 = ResBlock(base * 2, base * 3, tdim, dropout)

        # attention at bottleneck
        self.attn = SelfAttention2d(base * 3, num_heads=heads)

        self.up2 = Up(base * 3)
        self.rb_up2 = ResBlock(base * 3 + base * 2, base * 2, tdim, dropout)

        # optional attention after up2 (still low-ish res)
        self.attn_up = SelfAttention2d(base * 2, num_heads=heads)

        self.up1 = Up(base * 2)
        self.rb_up1 = ResBlock(base * 2 + base, base, tdim, dropout)

        self.out_norm = _gn(base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

        #llm
        self.cond_proj = nn.Linear(cond_dim, tdim)

    # def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #llm
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_emb: torch.Tensor = None):

        #llm
        t_emb = self.time_emb(t)
        if cond_emb is not None:
            cond_emb = self.cond_proj(cond_emb)

        t_emb = self.time_emb(t)

        x0 = self.in_conv(x)

        # x1 = self.rb1(x0, t_emb)
        #llm
        x1 = self.rb1(x0, t_emb, cond_emb)
        d1 = self.down1(x1)

        #x2 = self.rb2(d1, t_emb)
        #llm
        x2 = self.rb2(d1, t_emb, cond_emb)
        d2 = self.down2(x2)

        #x3 = self.rb3(d2, t_emb)
        #llm
        x3 = self.rb3(d2, t_emb,cond_emb)
        x3 = self.attn(x3)

        u2 = self.up2(x3)
        u2 = torch.cat([u2, x2], dim=1)
        #u2 = self.rb_up2(u2, t_emb)
        #llm
        u2 = self.rb_up2(u2, t_emb,cond_emb)
        u2 = self.attn_up(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        #u1 = self.rb_up1(u1, t_emb)
        # llm
        u1 = self.rb_up1(u1, t_emb,cond_emb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out

class ResUNet(nn.Module):
    """
    Deeper residual UNet. Stronger prior, slower.
    """
    def __init__(self, base: int = 96, tdim: int = 256, dropout: float = 0.0, blocks_per_stage: int = 3):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        def make_stack(in_ch, out_ch):
            layers = [ResBlock(in_ch, out_ch, tdim, dropout)]
            for _ in range(blocks_per_stage - 1):
                layers.append(ResBlock(out_ch, out_ch, tdim, dropout))
            return nn.ModuleList(layers)

        self.s1 = make_stack(base, base)
        self.down1 = Down(base)

        self.s2 = make_stack(base, base * 2)
        self.down2 = Down(base * 2)

        self.s3 = make_stack(base * 2, base * 3)

        self.up2 = Up(base * 3)
        self.u2 = make_stack(base * 3 + base * 2, base * 2)

        self.up1 = Up(base * 2)
        self.u1 = make_stack(base * 2 + base, base)

        self.out_norm = _gn(base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def _run_stack(self, stack: nn.ModuleList, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for blk in stack:
            x = blk(x, t_emb)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        x0 = self.in_conv(x)

        x1 = self._run_stack(self.s1, x0, t_emb)
        d1 = self.down1(x1)

        x2 = self._run_stack(self.s2, d1, t_emb)
        d2 = self.down2(x2)

        x3 = self._run_stack(self.s3, d2, t_emb)

        u2 = self.up2(x3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self._run_stack(self.u2, u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self._run_stack(self.u1, u1, t_emb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out



