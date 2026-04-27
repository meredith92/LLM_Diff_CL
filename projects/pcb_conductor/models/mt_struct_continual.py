import torch
import torch.nn.functional as F
from copy import deepcopy
import requests
import numpy as np
import cv2

from mmseg.registry import MODELS
from mmseg.models import EncoderDecoder

from projects.pcb_conductor.models.diffusion_prior import DiffusionPrior, DiffusionPriorConfig


def _sigmoid(x):
    return torch.sigmoid(x)

def _maxpool(x, k):
    pad = k // 2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)

def _minpool(x, k):
    return -_maxpool(-x, k)

@torch.no_grad()
def boundary_band(prob, k=5):
    d = _maxpool(prob, k)
    e = _minpool(prob, k)
    return (d - e).clamp(0, 1)

@torch.no_grad()
def selective_weight_map(p_t, conf_thr=0.8, band_k=5, drop_boundary=True):
    conf = (p_t > conf_thr) | (p_t < (1.0 - conf_thr))
    w = conf.float()
    if drop_boundary:
        band = boundary_band(p_t, band_k)
        w = w * (band < 1e-6).float()
    return w

def soft_skeleton_proxy(prob, ks=(3, 5, 7)):
    outs = []
    for k in ks:
        e = _minpool(prob, k)
        outs.append((prob - e).clamp(min=0.0))
    return torch.stack(outs, dim=0).mean(dim=0)

def skel_consistency(p_s, p_t, w):
    sk_s = soft_skeleton_proxy(p_s)
    sk_t = soft_skeleton_proxy(p_t)
    diff = (sk_s - sk_t).abs() * w
    eps = torch.finfo(diff.dtype).eps
    return diff.sum() / (w.sum() + eps)

import json, re

def _safe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def _build_llm_prompt(info: dict, score_rule: float, target_n: int = 24):
    return f"""You are a strict judge for a binary segmentation mask of "separate vertical pins".
You will be given ONLY numeric metrics. Do not ask for images.

IMPORTANT:
- Output MUST be ONLY one JSON object on a single line.
- Do NOT include any extra text before/after JSON.
- Use double quotes for JSON keys/strings.

Return ONLY valid JSON:
{{"score":0-1,"error_types":[...],"suggestions":[...],"usable_for_kd":true/false}}

error_types must be chosen from:
["merged","noise","broken","wrong_orientation","uneven_width","bad_bottom","spacing_irregular","too_few","too_many","low_confidence"]

Metrics:
expected_pins={target_n}
n_cc={info.get("n_cc", -1)}
bottom_fill={info.get("bottom_fill", -1)}
wide_ratio={info.get("wide_ratio", -1)}
thin_ratio={info.get("thin_ratio", -1)}
good_aspect_ratio={info.get("good_aspect_ratio", -1)}
gap_outlier_ratio={info.get("gap_outlier_ratio", -1)}
conf={info.get("conf", -1)}
score_rule={score_rule}
"""


def _ollama_call(prompt: str,
                 model: str = "qwen2.5:7b-instruct",
                 timeout_ms: int = 800,
                 temperature: float = 0.0,
                 num_predict: int = 200):
    """
    Returns: model raw text (should be JSON per our prompt)
    """
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        }
    }
    r = requests.post(url, json=payload, timeout=timeout_ms / 1000.0)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def _llm_call_stub(prompt: str) -> str:
    return _ollama_call(prompt)


# -------------------------
# Pseudo-label Judge (24 pins, no equal spacing assumption)
# -------------------------
def pseudo_label_judge_24pins_no_equal_spacing(
    p: np.ndarray,                 # HxW prob in [0,1]
    thr: float = 0.5,
    min_area: int = 20,
    bottom_band_ratio: float = 0.12,
    target_n: int = 24,
    n_tol: int = 6,                # allow break/merge
):
    """
    Returns:
      s_img: float in [0,1]
      m_rel: HxW uint8 reliable pixels mask {0,1}
      info: dict metrics
    """
    H, W = p.shape
    y = (p >= thr).astype(np.uint8)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(y, connectivity=8)

    comps = []
    for i in range(1, num):
        x, y0, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        comps.append(dict(i=i, x=x, y=y0, w=w, h=h, area=area, cx=cx, cy=cy))

    n_cc = len(comps)
    if n_cc == 0:
        m_rel = (np.abs(p - 0.5) > 0.25).astype(np.uint8)
        return 0.0, m_rel, dict(n_cc=0, reason="no_components")

    ws  = np.array([c["w"] for c in comps], dtype=np.float32)
    hs  = np.array([c["h"] for c in comps], dtype=np.float32)
    cxs = np.array([c["cx"] for c in comps], dtype=np.float32)

    # verticality
    aspect = hs / (ws + 1e-6)
    good_aspect_ratio = float(np.mean(aspect > 2.0))

    # width: allow thin pins, mainly penalize overly wide (merged)
    med_w = float(np.median(ws) + 1e-6)
    wide_thr = 2.1 * med_w
    wide_ratio = float(np.mean(ws > wide_thr))
    thin_thr = 0.35 * med_w
    thin_ratio = float(np.mean(ws < thin_thr))
    s_width = float(np.clip(1.0 - (0.90 * wide_ratio + 0.10 * thin_ratio), 0.0, 1.0))

    # bottom connection check
    b = max(int(H * bottom_band_ratio), 1)
    bottom = y[H - b : H, :]
    bottom_fill = float(np.mean(bottom))
    s_bottom = float(np.clip((0.22 - bottom_fill) / (0.22 - 0.12 + 1e-6), 0.0, 1.0))

    # spacing sanity (no equal spacing): only penalize extreme gaps/overlaps
    cxs_sorted = np.sort(cxs)
    if len(cxs_sorted) >= 3:
        dx = np.diff(cxs_sorted)
        med_dx = float(np.median(dx) + 1e-6)
        gap_outlier = (dx < 0.3 * med_dx) | (dx > 3.0 * med_dx)
        gap_outlier_ratio = float(np.mean(gap_outlier))
    else:
        gap_outlier_ratio = 1.0
    s_gap = float(np.clip(1.0 - gap_outlier_ratio, 0.0, 1.0))

    # count prior: soft peak at 24
    count_err = abs(n_cc - target_n)
    s_count = float(np.clip(1.0 - (count_err / (n_tol + 1e-6)), 0.0, 1.0))

    # aspect score
    s_aspect = float(np.clip((good_aspect_ratio - 0.5) / (0.9 - 0.5 + 1e-6), 0.0, 1.0))

    # confidence proxy
    conf = float(np.mean(np.abs(p - 0.5) * 2.0))  # 0..1

    # final score
    s_img = (
        0.32 * s_count +
        0.22 * s_bottom +
        0.22 * s_width +
        0 * s_aspect +
        0.05 * s_gap +
        0.05 * conf
    )
    s_img = float(np.clip(s_img, 0.0, 1.0))

    # reliable pixels: confident pixels + remove tiny/noisy positives already filtered by min_area
    m_conf = (np.abs(p - 0.5) > 0.20).astype(np.uint8)

    keep = np.zeros((H, W), dtype=np.uint8)
    kept_ids = {c["i"] for c in comps}
    for i in kept_ids:
        keep[labels == i] = 1

    y_bin = (p >= thr).astype(np.uint8)
    m_rel = (m_conf & ((1 - y_bin) | keep)).astype(np.uint8)

    info = dict(
        n_cc=n_cc,
        s_count=s_count,
        bottom_fill=bottom_fill,
        s_bottom=s_bottom,
        med_w=med_w,
        wide_ratio=wide_ratio,
        thin_ratio=thin_ratio,
        s_width=s_width,
        good_aspect_ratio=good_aspect_ratio,
        s_aspect=s_aspect,
        gap_outlier_ratio=gap_outlier_ratio,
        s_gap=s_gap,
        conf=conf,
        s_img=s_img,
    )
    return s_img, m_rel, info


@MODELS.register_module()
class MTStructContinual(EncoderDecoder):
    def __init__(
        self,
        ema=0.99,
        conf_thr=0.8,
        band_k=5,
        lam_u=1.0,
        lam_skel=0.1,
        lam_lwf=1.0,
        use_lwf=True,
        use_selective=True,
        drop_boundary=True,
        # ---- diffusion gate ----
        use_diffusion=True,
        u_thr=0.15,
        diff_K=2,
        diff_steps=10,
        diff_down=4,
        diff_ckpt=None,
        # ---- pseudo judge ----
        use_judge=True,
        judge_thr=0.35,   # if you want hard filter, use >0; else keep as soft weight
        judge_min_area=20,
        judge_bottom_band=0.12,
        judge_target_n=24,
        judge_n_tol=6,
        # ---- LLM judge ----
        use_llm_judge=False,
        llm_alpha=0.3,  # LLM 融合权重
        llm_keep_thr=0.35,  # LLM hard filter (可设 None 或 <=0 关闭)
        llm_uncertain_band=(0.25, 0.70),  # 只在 rule 分数不确定时才问 LLM（省开销）
        llm_timeout_ms=800,  # 你自己实现时用
        llm_every_n=50,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ema = ema
        self.conf_thr = conf_thr
        self.band_k = band_k
        self.lam_u = lam_u
        self.lam_skel = lam_skel
        self.lam_lwf = lam_lwf
        self.use_lwf = use_lwf
        self.use_selective = use_selective
        self.drop_boundary = drop_boundary

        # diffusion params
        self.use_diffusion = use_diffusion
        self.u_thr = u_thr
        self.diff_K = diff_K
        self.diff_steps = diff_steps
        self.diff_down = diff_down

        if self.use_diffusion:
            cfg = DiffusionPriorConfig(ckpt=diff_ckpt)
            self.diffusion = DiffusionPrior(cfg)
            self.diffusion.requires_grad_(False)
        else:
            self.diffusion = None

        # judge params
        self.use_judge = use_judge
        self.judge_thr = judge_thr
        self.judge_min_area = judge_min_area
        self.judge_bottom_band = judge_bottom_band
        self.judge_target_n = judge_target_n
        self.judge_n_tol = judge_n_tol

        # ---- LLM judge ----
        self.use_llm_judge = use_llm_judge
        self.llm_alpha = llm_alpha
        self.llm_keep_thr = llm_keep_thr
        self.llm_uncertain_band = llm_uncertain_band
        self.llm_timeout_ms = llm_timeout_ms
        self.llm_every_n = llm_every_n
        self._iter = 0

        self._llm_cache = {}  # 简单缓存：key->(score, keep)

        self.teacher = None
        self.old_model = None

    def llm_judge(self, info: dict, score_rule: float):
        # cache key：把连续值粗量化，避免浮点抖动导致 cache miss
        key = (
            int(info.get("n_cc", -1)),
            round(float(info.get("bottom_fill", 0.0)), 3),
            round(float(info.get("wide_ratio", 0.0)), 3),
            round(float(info.get("thin_ratio", 0.0)), 3),
            round(float(info.get("good_aspect_ratio", 0.0)), 3),
            round(float(info.get("gap_outlier_ratio", 0.0)), 3),
            round(float(info.get("conf", 0.0)), 3),
            round(float(score_rule), 3),
        )
        if key in self._llm_cache:
            return self._llm_cache[key]

        prompt = _build_llm_prompt(info, score_rule, target_n=self.judge_target_n)

        try:
            out = _llm_call_stub(prompt)  # TODO: 替换
            j = _safe_json(out) or {}
            s = float(j.get("score", score_rule))
            s = max(0.0, min(1.0, s))
            usable = bool(j.get("usable_for_kd", s >= (self.judge_thr or 0.0)))
        except Exception:
            # fallback: 失败就用 rule
            s = float(score_rule)
            usable = bool(s >= (self.judge_thr or 0.0))

        self._llm_cache[key] = (s, usable)
        return s, usable

    def init_teacher(self):
        if self.teacher is None:
            self.teacher = deepcopy(self)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def diffusion_uncertainty(self, p_t: torch.Tensor) -> torch.Tensor:
        assert self.diffusion is not None
        p_t = p_t.clamp(0, 1)

        if self.diff_down > 1:
            p_in = F.interpolate(p_t, scale_factor=1.0 / self.diff_down,
                                 mode="bilinear", align_corners=False)
        else:
            p_in = p_t

        samples = []
        for _ in range(self.diff_K):
            yk = self.diffusion.sample(condition=p_in, steps=self.diff_steps)  # [B,1,h,w]
            samples.append(yk)
        S = torch.stack(samples, dim=0)          # [K,B,1,h,w]
        u = S.var(dim=0, unbiased=False)         # [B,1,h,w]

        B = u.shape[0]
        u_max = u.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1).clamp_min(1e-6)
        u = (u / u_max).clamp(0, 1)

        if self.diff_down > 1:
            u = F.interpolate(u, size=p_t.shape[-2:], mode="bilinear", align_corners=False)

        return u

    @torch.no_grad()
    def ema_update(self):
        self.init_teacher()
        for (n_s, p_s), (n_t, p_t) in zip(self.named_parameters(), self.teacher.named_parameters()):
            if n_s.startswith("diffusion."):
                continue
            p_t.data.mul_(self.ema).add_(p_s.data, alpha=(1.0 - self.ema))

    @torch.no_grad()
    def set_old_model(self):
        self.old_model = deepcopy(self)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)

    def _forward_logits(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs, dim=0)

        if inputs.dtype == torch.uint8:
            inputs = inputs.float() / 255.0
        elif inputs.dtype != torch.float32:
            inputs = inputs.float()

        if inputs.dtype == torch.float32 and inputs.max() > 1.5:
            inputs = inputs / 255.0

        x = self.extract_feat(inputs)
        logits = self.decode_head(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits

    @staticmethod
    def _stack_field(data_samples, name, device, fallback=None):
        xs = []
        for i, ds in enumerate(data_samples):
            x = getattr(ds, name, None)
            if x is None:
                assert fallback is not None, f"Missing {name} and no fallback provided."
                x = fallback[i]
            xs.append(x)
        return torch.stack(xs, dim=0).to(device)

    @staticmethod
    def _get_is_labeled(data_samples, device):
        vals = [ds.metainfo.get("is_labeled", 1) for ds in data_samples]
        return torch.tensor(vals, device=device, dtype=torch.long)

    def loss(self, inputs, data_samples, train_cfg=None, **kwargs):
        self.init_teacher()

        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs, dim=0)
        if not isinstance(data_samples, (list, tuple)):
            data_samples = [data_samples]

        device = inputs.device

        inputs_s = inputs
        inputs_w = self._stack_field(data_samples, "inputs_w", device=device, fallback=inputs_s)
        is_labeled = self._get_is_labeled(data_samples, device=device).long()

        if inputs_s.dtype == torch.float32 and inputs_s.max() > 1.5:
            inputs_s = inputs_s / 255.0
        if inputs_w is not None and inputs_w.dtype == torch.float32 and inputs_w.max() > 1.5:
            inputs_w = inputs_w / 255.0

        losses = {}

        # -------------------------
        # labeled branch
        # -------------------------
        idx_l = torch.where(is_labeled > 0)[0]
        if idx_l.numel() > 0:
            inp_l = inputs_s[idx_l]
            ds_l = [data_samples[i] for i in idx_l.tolist()]

            if inp_l.dtype == torch.uint8:
                inp_l = inp_l.float() / 255.0
            elif inp_l.dtype != torch.float32:
                inp_l = inp_l.float()
            if inp_l.dtype == torch.float32 and inp_l.max() > 1.5:
                inp_l = inp_l / 255.0

            loss_sup = super().loss(inp_l, ds_l)
            for k, v in loss_sup.items():
                losses[f"loss_sup.{k}"] = v

            if self.use_lwf and (self.old_model is not None):
                logits_l = self._forward_logits(inp_l)
                with torch.no_grad():
                    logits_old = self.old_model._forward_logits(inp_l)
                losses["loss_lwf"] = self.lam_lwf * F.mse_loss(torch.sigmoid(logits_l), torch.sigmoid(logits_old))

        self._iter += 1
        llm_active = (self.use_llm_judge and (self.llm_every_n is None or self._iter % self.llm_every_n == 0))
        # -------------------------
        # unlabeled branch
        # -------------------------
        idx_u = torch.where(is_labeled <= 0)[0]
        if idx_u.numel() > 0:
            assert inputs_w is not None, "Need inputs_w stored in data_samples for unlabeled batch."

            inp_w = inputs_w[idx_u].float()  # teacher view (weak)
            inp_s = inputs_s[idx_u].float()  # student view (strong)

            with torch.no_grad():
                logits_t = self.teacher._forward_logits(inp_w)
                p_t = _sigmoid(logits_t)  # [B,1,H,W]

                # (A) selective weight map
                if self.use_selective:
                    w_sel = selective_weight_map(
                        p_t,
                        conf_thr=self.conf_thr,
                        band_k=self.band_k,
                        drop_boundary=self.drop_boundary
                    )
                else:
                    w_sel = torch.ones_like(p_t)

                # (B) diffusion uncertainty gate
                if self.use_diffusion:
                    u = self.diffusion_uncertainty(p_t)
                    w_diff = (u < self.u_thr).to(p_t.dtype)
                else:
                    w_diff = torch.ones_like(p_t)

                # (C) pseudo-label judge (structure)
                # if self.use_judge:
                #     s_list = []
                #     m_list = []
                #     p_cpu = p_t.detach().cpu()  # [B,1,H,W]
                #
                #     B_u = p_cpu.shape[0]
                #     for b in range(B_u):
                #         p_np = p_cpu[b, 0].numpy()  # HxW

                    #     s_img, m_rel, info = pseudo_label_judge_24pins_no_equal_spacing(
                    #         p_np,
                    #         thr=0.5,
                    #         min_area=self.judge_min_area,
                    #         bottom_band_ratio=self.judge_bottom_band,
                    #         target_n=self.judge_target_n,
                    #         n_tol=self.judge_n_tol,
                    #     )
                    #     s_list.append(s_img)
                    #     m_list.append(m_rel)
                    #
                    # s_img = torch.tensor(s_list, device=device, dtype=p_t.dtype)  # [B]
                    # w_judge = torch.stack(
                    #     [torch.from_numpy(m).to(device) for m in m_list],
                    #     dim=0
                    # ).unsqueeze(1).to(p_t.dtype)  # [B,1,H,W]
                    #
                    # # optional hard filter at image level
                    # if self.judge_thr is not None and self.judge_thr > 0:
                    #     keep = (s_img >= float(self.judge_thr)).to(p_t.dtype).view(-1, 1, 1, 1)
                    # else:
                    #     keep = torch.ones((p_t.shape[0], 1, 1, 1), device=device, dtype=p_t.dtype)
                    #
                    # # combine: pixel mask * image score * keep
                    # w = w_sel.to(p_t.dtype) * w_diff * w_judge
                    # w = w * (s_img.view(-1, 1, 1, 1) * keep)
                # rule score
                # s_rule = torch.tensor(s_list, device=device, dtype=p_t.dtype)  # [B]

                # ---- LLM judge (image-level) ----
                # (C) pseudo-label judge (structure) + optional LLM judge
                if self.use_judge:
                    s_list = []
                    m_list = []
                    info_list = []

                    p_cpu = p_t.detach().cpu()  # [B,1,H,W]
                    B_u = p_cpu.shape[0]

                    for b in range(B_u):
                        p_np = p_cpu[b, 0].numpy()  # HxW

                        s_img, m_rel, info = pseudo_label_judge_24pins_no_equal_spacing(
                            p_np,
                            thr=0.5,
                            min_area=self.judge_min_area,
                            bottom_band_ratio=self.judge_bottom_band,
                            target_n=self.judge_target_n,
                            n_tol=self.judge_n_tol,
                        )
                        s_list.append(float(s_img))
                        m_list.append(m_rel)
                        info_list.append(info)

                    # pixel reliable mask from rule
                    w_judge = torch.stack(
                        [torch.from_numpy(m).to(device) for m in m_list],
                        dim=0
                    ).unsqueeze(1).to(p_t.dtype)  # [B,1,H,W]

                    # rule score tensor
                    s_rule = torch.tensor(s_list, device=device, dtype=p_t.dtype)  # [B]

                    # ---- LLM judge (image-level) ----
                    if llm_active:
                        lo, hi = self.llm_uncertain_band
                        s_llm_list = []
                        keep_llm_list = []

                        for b in range(B_u):
                            sr = float(s_list[b])
                            if (sr >= lo) and (sr <= hi):
                                s_llm, usable = self.llm_judge(info_list[b], sr)
                            else:
                                s_llm, usable = sr, (sr >= float(self.judge_thr or 0.0))

                            s_llm_list.append(float(s_llm))
                            keep_llm_list.append(1.0 if usable else 0.0)

                        s_llm = torch.tensor(s_llm_list, device=device, dtype=p_t.dtype)  # [B]
                        keep_llm = torch.tensor(keep_llm_list, device=device, dtype=p_t.dtype).view(-1, 1, 1, 1)

                        alpha = float(self.llm_alpha)
                        s_final = (1 - alpha) * s_rule + alpha * s_llm
                    else:
                        s_final = s_rule
                        keep_llm = torch.ones((B_u, 1, 1, 1), device=device, dtype=p_t.dtype)

                    # rule hard filter
                    if self.judge_thr is not None and self.judge_thr > 0:
                        keep_rule = (s_rule >= float(self.judge_thr)).to(p_t.dtype).view(-1, 1, 1, 1)
                    else:
                        keep_rule = torch.ones((B_u, 1, 1, 1), device=device, dtype=p_t.dtype)

                    # optional LLM hard filter on fused score
                    if (self.use_llm_judge and self.llm_keep_thr is not None and self.llm_keep_thr > 0):
                        keep_score = (s_final >= float(self.llm_keep_thr)).to(p_t.dtype).view(-1, 1, 1, 1)
                    else:
                        keep_score = torch.ones((B_u, 1, 1, 1), device=device, dtype=p_t.dtype)

                    # combine all weights
                    w = w_sel.to(p_t.dtype) * w_diff * w_judge
                    w = w * (s_final.view(-1, 1, 1, 1) * keep_rule * keep_llm * keep_score)

                else:
                    w = w_sel.to(p_t.dtype) * w_diff

                # else:
                #     w = w_sel.to(p_t.dtype) * w_diff

            # student forward
            logits_s = self._forward_logits(inp_s)
            p_s = _sigmoid(logits_s)

            cons = (p_s - p_t).pow(2) * w
            losses["loss_unsup"] = self.lam_u * (cons.sum() / (w.sum() + 1e-6))
            losses["loss_skel"] = self.lam_skel * skel_consistency(p_s, p_t, w)

        return losses
