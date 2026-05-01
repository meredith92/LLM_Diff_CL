# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import logging
import os
import os.path as osp
import sys

# Real-time stdout
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmseg.utils import register_all_modules


register_all_modules(init_default_scope=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")

    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest checkpoint in the work_dir automatically"
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training"
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: key=value"
    )

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher"
    )

    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)

    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # -------------------------
    # Load config
    # -------------------------
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # -------------------------
    # Work dir
    # -------------------------
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs",
            osp.splitext(osp.basename(args.config))[0]
        )

    # -------------------------
    # AMP
    # -------------------------
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type

        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when optimizer wrapper type is "
                f"`OptimWrapper`, but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # -------------------------
    # Resume
    # -------------------------
    cfg.resume = args.resume

    # -------------------------
    # Logger hook fallback
    # -------------------------
    if "default_hooks" not in cfg:
        cfg.default_hooks = {}

    if "logger" not in cfg.default_hooks:
        cfg.default_hooks["logger"] = dict(
            type="LoggerHook",
            interval=50
        )
    else:
        if cfg.default_hooks["logger"].get("interval", 50) > 100:
            print(
                f"[WARNING] Logger interval too large "
                f"({cfg.default_hooks['logger']['interval']}), set to 50"
            )
            cfg.default_hooks["logger"]["interval"] = 50

    # -------------------------
    # Debug config info
    # -------------------------
    # print("=" * 80)
    # print("[DBG] FINAL TRAIN CONFIG")
    # print(f"[DBG] config: {args.config}")
    # print(f"[DBG] work_dir: {cfg.work_dir}")
    # print(f"[DBG] resume: {cfg.resume}")
    # print(f"[DBG] load_from: {cfg.get('load_from', None)}")
    # print(f"[DBG] max_iters: {cfg.train_cfg.get('max_iters', None)}")
    # print(f"[DBG] logger interval: {cfg.default_hooks.logger.interval}")

    try:
        dataset = cfg.train_dataloader.dataset
        # print(f"[DBG] train dataset type: {dataset.type}")

        # if hasattr(dataset, "labeled_dataset"):
        #     print(
        #         "[DBG] labeled data_prefix:",
        #         dataset.labeled_dataset.data_prefix
        #     )

        # if hasattr(dataset, "unlabeled_dataset"):
        #     print(
        #         "[DBG] unlabeled data_prefix:",
        #         dataset.unlabeled_dataset.data_prefix
        #     )

    except Exception as e:
        print(f"[DBG] Could not print dataset info: {e}")

    print("=" * 80)
    sys.stdout.flush()
    sys.stderr.flush()

    # -------------------------
    # Build runner
    # -------------------------
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # -------------------------
    # Set old_model for LwF continual learning
    # -------------------------
    ckpt = cfg.get("load_from", None)
    use_lwf = cfg.model.get("use_lwf", False)

    if use_lwf and ckpt is not None:
        if hasattr(runner.model, "set_old_model"):
            print(f">>> Setting old_model from: {ckpt}")
            runner.model.set_old_model(ckpt)
        else:
            print(
                "[WARNING] cfg.model.use_lwf=True, "
                "but runner.model has no set_old_model() method."
            )

    # -------------------------
    # Debug weight status
    # -------------------------
    try:
        model_dict = runner.model.state_dict()

        total_params = 0
        zero_params = 0

        for name, param in model_dict.items():
            total_params += param.numel()
            if param.abs().sum() == 0:
                zero_params += param.numel()

        print(
            f"[DEBUG] Model weight status: "
            f"{zero_params}/{total_params} parameters are zero"
        )

        if zero_params == total_params:
            print(
                "[WARNING] ALL model parameters are zero! "
                "Model weights may not be loaded properly."
            )

        sample_keys = [k for k in model_dict.keys() if "weight" in k][:3]

        for k in sample_keys:
            p = model_dict[k]
            print(
                f"[DEBUG] {k}: "
                f"min={p.min():.6f}, "
                f"max={p.max():.6f}, "
                f"mean={p.mean():.6f}"
            )

    except Exception as e:
        print(f"[WARNING] Could not check model weights: {e}")

    sys.stdout.flush()
    sys.stderr.flush()

    # -------------------------
    # Start training
    # -------------------------
    print("\n" + "=" * 80)
    print(">>> STARTING TRAINING")
    print(f"    Work directory: {cfg.work_dir}")
    print(f"    Max iterations: {cfg.train_cfg.get('max_iters', None)}")
    print(f"    Logger interval: {cfg.default_hooks.logger.interval}")
    print("=" * 80 + "\n")

    sys.stdout.flush()
    sys.stderr.flush()

    runner.train()


if __name__ == "__main__":
    main()