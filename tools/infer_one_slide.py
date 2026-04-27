import os
import numpy as np
import mmcv
import torch
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from mmseg.apis import init_model
from mmseg.models.data_preprocessor import SegDataPreProcessor  # ✅ 关键

CONFIG = r'projects/pcb_conductor/configs/segformer_mt_vb.py'
CKPT   = r'work_dirs/segformer_mt_vb/llmrule500-4.pth'
IMG    = r'data/B/images/train/2025-09-16_09-17-15_945.bmp'  # <-- 改成你的图片路径
OUTDIR = r'work_dirs\segformer_mt_vb'
os.makedirs(OUTDIR, exist_ok=True)

cfg = Config.fromfile(CONFIG)

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(99999, 1024), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackSegInputs')
]

model = init_model(cfg, CKPT, device='cuda:0')  # 没GPU改 'cpu'
model.eval()

device = next(model.parameters()).device
# 3) ✅ 强制替换模型的 data_preprocessor（保证 uint8->float + normalize）
model.data_preprocessor = SegDataPreProcessor(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
).to(device)


# 3) 强制 slide（兼容 wrapper / 非 wrapper 两种结构）
if 'test_cfg' in cfg.model:
    cfg.model['test_cfg'] = dict(mode='slide', crop_size=(256, 1024), stride=(128, 768))
elif 'model' in cfg.model and isinstance(cfg.model['model'], dict):
    cfg.model['model']['test_cfg'] = dict(mode='slide', crop_size=(256, 1024), stride=(128, 768))

# 5) pipeline -> collate（default_collate 强制 stack）
pipeline = Compose(cfg.test_pipeline)
data = pipeline(dict(img_path=IMG))
batch = default_collate([data])

# 终极兜底：确保 float
if batch['inputs'].dtype == torch.uint8:
    batch['inputs'] = batch['inputs'].float()

print('inputs dtype:', batch['inputs'].dtype, 'shape:', tuple(batch['inputs'].shape))

# ✅ move to cuda
device = next(model.parameters()).device
batch['inputs'] = batch['inputs'].to(device)
batch['data_samples'] = [ds.to(device) for ds in batch['data_samples']]

with torch.no_grad():
    outputs = model.test_step(batch)
out = outputs[0]

pred = out.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
mmcv.imwrite(pred * 255, os.path.join(OUTDIR, 'smallunet4500.png'))
print('saved:', os.path.join(OUTDIR, 'smallunet4500.png'))
