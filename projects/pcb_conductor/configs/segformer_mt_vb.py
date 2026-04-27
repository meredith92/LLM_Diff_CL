
default_scope = 'mmseg'
custom_imports = dict(
    imports=['projects.pcb_conductor'],
    allow_failed_imports=False
)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size_divisor=32,   # ✅ 强制 pad/stack
    pad_val=0,
    seg_pad_val=255
)
# ---------- pipelines ----------
# B: strip 2500x260 -> resize height to 256, crop (256,1024)
b_crop = (256, 1024)

b_weak = [
    dict(type='Resize', scale=(99999, 256), keep_ratio=True),
    dict(type='RandomCrop', crop_size=b_crop),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
]

b_strong = [
    dict(type='Resize', scale=(99999, 256), keep_ratio=True),
    dict(type='RandomCrop', crop_size=b_crop),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
]

b_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TwoViewAug', weak=b_weak, strong=b_strong),

]

# A: labeled -> crop same shape for unified collate
a_crop = (256, 1024)
a_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(99999, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=a_crop),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackSegInputs'),
    dict(type='EnsureLabeledUnlabeledKeys'),
]

test_pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(99999, 1024), keep_ratio=True),
            dict(type='Pad', size_divisor=32),
            dict(type='PackSegInputs')
        ]

# Note: For B we already Pack inside TwoViewAug, so don't PackSegInputs again.

# ---------- datasets ----------
data_root = 'data'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='DualStreamDataset',
        labeled_ratio=0.5,
        labeled_dataset=dict(
            type='PCBAConductorDataset',
            data_root=f'{data_root}/A',
            data_prefix=dict(img_path='images/train', seg_map_path='labels/train'),
            pipeline=a_pipeline
        ),
        unlabeled_dataset=dict(
            type='PCBBUnlabeledDataset',
            data_root=f'{data_root}/B',
            data_prefix=dict(img_path='images/train'),
            pipeline=b_pipeline
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='PCBAConductorDataset',
        data_root=f'{data_root}/A',
        data_prefix=dict(img_path='images/val', seg_map_path='labels/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]
    )
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
val_cfg = dict()   # 可不写；但写一个空 dict 最稳

# ---------- model (SegFormer-B0) ----------
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='MTStructContinual',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,  # binary sigmoid
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(256, 1024), stride=(128, 768)),
    # test_cfg = dict(mode='whole'),

    ema=0.99,
    conf_thr=0.8,
    band_k=5,
    lam_u=1.0,
    lam_skel=0.3,
    lam_lwf=1.0,
    use_lwf=False,
    use_selective=True,
    drop_boundary=True,
    use_diffusion=True,
    u_thr=0.12,
    diff_K=8,
    diff_steps=20,
    diff_down=2,
    diff_ckpt='work_dirs/mask_ddpm_min/unet_llm.pth',
)

# test_dataloader = val_dataloader
# test_evaluator = val_evaluator
# test_cfg = val_cfg  # 或者 test_cfg = dict()
# ---------- hooks ----------
custom_hooks = [
    dict(type='EMAUpdateHook')
]

# ---------- optim ----------
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=6e-5, weight_decay=0.01))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=3000, val_interval=500)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,      # ⭐ 核心
        interval=500,       # 每 1000 iter 保存
        max_keep_ckpts=10,   # 防止磁盘爆
        save_last=True,
        save_best='mDice',   # 或 'mIoU'
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)
)