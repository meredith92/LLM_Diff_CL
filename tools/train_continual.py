#!/usr/bin/env python3
"""
End-to-end Continual Learning Experiment Pipeline for PCB Conductor Segmentation

Pipeline:
1. Pretrain diffusion model on Domain A labels
2. Train segmentation model on Domain A
3. Evaluate on both Domain A and B
4. Continual finetune on Domain B
5. Evaluate on both domains again
6. Generate continual learning metrics and visualizations

Usage:
    python tools/train_continual.py \
        --stage pretrain_diffusion \
        --mask_root data/A/labels/train \
        --out_dir work_dirs/continual_experiment
"""
import glob
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from projects.pcb_conductor.tools.continual_eval import ContinualSegmentationEvaluator


class ContinualExperimentOrchestrator:
    """
    Orchestrate the full continual learning experiment pipeline.
    """
    
    def __init__(
        self,
        base_work_dir: str = 'work_dirs/continual_experiment',
        domain_a_config: str = 'projects/pcb_conductor/configs/segformer_mt_vb_a.py',
        domain_b_config: str = 'projects/pcb_conductor/configs/segformer_mt_vb_b.py',
    ):
        self.base_work_dir = Path(base_work_dir)
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        
        self.domain_a_config = domain_a_config
        self.domain_b_config = domain_b_config
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.base_work_dir / f'experiment_log_{self.timestamp}.txt'
        
        self.evaluator = ContinualSegmentationEvaluator(
            domain_order=['domain_A', 'domain_B'],
            metric_key='mIoU'
        )
        
    def log(self, msg: str, level: str = 'INFO'):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def run_command(self, cmd, desc=None):
        import subprocess
        import sys
        import os

        if desc:
            self.log("=" * 80)
            self.log(desc)
            self.log("=" * 80)

        # 强制 python 实时输出
        cmd = cmd.replace("python tools/train.py", "python -u tools/train.py")

        self.log(f"Running command:\n{cmd}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=None,     # 关键：不要 PIPE，不要 capture
            stderr=None,     # 直接继承当前终端
            env=env,
        )

        process.wait()

        if process.returncode == 0:
            self.log(f"Command finished successfully: {desc}")
        else:
            self.log(
                f"ERROR: Command failed with return code {process.returncode}",
                level="ERROR"
            )

        return process.returncode
    
    def stage_pretrain_diffusion(
        self,
        mask_root: str,
        image_size: int = 128,
        batch_size: int = 16,
        epochs: int = 50,
        lr: float = 2e-4,
    ) -> str:
        """
        Stage 1: Pretrain diffusion model on Domain A segmentation masks.
        
        Returns:
            path to saved diffusion checkpoint
        """
        self.log("="*80)
        self.log("STAGE 1: Pretraining Diffusion Model on Domain A")
        self.log("="*80)
        
        out_dir = str(self.base_work_dir / 'stage1_diffusion_pretraining')
        
        cmd = (
            f"python tools/train_mask_ddpm_min.py "
            f"--mask_root {mask_root} "
            f"--out_dir {out_dir} "
            f"--image_size {image_size} "
            f"--batch_size {batch_size} "
            f"--epochs {epochs} "
            f"--lr {lr} "
            f"--amp"
        )
        
        ret = self.run_command(cmd, "Pretraining diffusion model")
        
        if ret == 0:
            ckpt_path = os.path.join(out_dir, 'unet_llm.pth')
            print(ckpt_path)
            self.log(f"Diffusion pretraining completed. Checkpoint: {ckpt_path}")
            return ckpt_path
        else:
            self.log("Diffusion pretraining failed!", level='ERROR')
            return None
    
    def stage_train_domain_a(self, diffusion_ckpt: Optional[str] = None) -> str:
        """
        Stage 2: Train segmentation model on Domain A.
        
        Args:
            diffusion_ckpt: path to pretrained diffusion checkpoint
        
        Returns:
            path to saved model checkpoint
        """
        self.log("="*80)
        self.log("STAGE 2: Training Segmentation Model on Domain A")
        self.log("="*80)
        
        work_dir = str(self.base_work_dir / 'stage2_train_domain_a')
        
        # Update config to use diffusion checkpoint if provided
        cfg_opt_str = ''
        if diffusion_ckpt:
            diffusion_ckpt = str(diffusion_ckpt).replace("\\", "/")
            cfg_opt_str = f"--cfg-options model.diff_ckpt={diffusion_ckpt}"
        
        # cfg_opt_str = ' '.join(cfg_options) if cfg_options else ''
        
        cmd = (
            f"python tools/train.py "
            f"--config {self.domain_a_config} "
            f"--work-dir {work_dir} "
            f"{cfg_opt_str} "
            f"--amp"
        )
        
        ret = self.run_command(cmd, "Training segmentation on Domain A")
        
        if ret == 0:
            self.log(f"Domain A training completed. Work dir: {work_dir}")
            return work_dir
        else:
            self.log("Domain A training failed!", level='ERROR')
            return None
    
    def stage_evaluate_both_domains(self, model_work_dir: str, stage_name: str = 'after_A'):
        """
        Stage 3: Evaluate the model on both Domain A and B.
        
        Args:
            model_work_dir: work directory containing the trained model checkpoint
            stage_name: identifier for this evaluation stage
        
        Returns:
            dict of metrics: {'domain_A': float, 'domain_B': float}
        """
        self.log("="*80)
        self.log(f"EVALUATING: {stage_name}")
        self.log("="*80)
        
        metrics = {}
        
        # Check if model_work_dir is valid
        if model_work_dir is None:
            self.log("Cannot evaluate: model_work_dir is None (likely training failed)", level="WARNING")
            return {'domain_A': 0.0, 'domain_B': 0.0}
        
        if not os.path.exists(model_work_dir):
            self.log(f"Cannot evaluate: model_work_dir does not exist: {model_work_dir}", level="WARNING")
            return {'domain_A': 0.0, 'domain_B': 0.0}
        
        # Find the best checkpoint
        best_ckpts = glob.glob(os.path.join(model_work_dir, "best_*.pth"))
        iter_ckpts = glob.glob(os.path.join(model_work_dir, "iter_*.pth"))
        
        if best_ckpts:
            ckpt = max(best_ckpts, key=os.path.getmtime)
        elif iter_ckpts:
            ckpt = max(iter_ckpts, key=os.path.getmtime)
        else:
            self.log(f"No checkpoint found in {model_work_dir}", level="WARNING")
            return {'domain_A': 0.0, 'domain_B': 0.0}
        
        self.log(f"Using checkpoint: {ckpt}")
        
        # Evaluate on Domain A
        eval_dir_a = f"{model_work_dir}/eval_domain_a"
        cmd_a = (
            f"python tools/test.py "
            f"{self.domain_a_config} "
            f"{ckpt} "
            f"--work-dir {eval_dir_a} "
            f"--out {eval_dir_a}"
        )
        ret_a = self.run_command(cmd_a, "Evaluating on Domain A")
        
        # Parse Domain A metrics from work_dir
        domain_a_miou = self._parse_metrics(eval_dir_a)
        if domain_a_miou is None:
            domain_a_miou = 0.0
            self.log("Could not parse Domain A metrics, using 0.0", level="WARNING")
        
        metrics['domain_A'] = domain_a_miou
        
        # Evaluate on Domain B
        eval_dir_b = f"{model_work_dir}/eval_domain_b"
        cmd_b = (
            f"python tools/test.py "
            f"{self.domain_b_config} "
            f"{ckpt} "
            f"--work-dir {eval_dir_b} "
            f"--out {eval_dir_b}"
        )
        ret_b = self.run_command(cmd_b, "Evaluating on Domain B")
        
        # Parse Domain B metrics from work_dir
        domain_b_miou = self._parse_metrics(eval_dir_b)
        if domain_b_miou is None:
            domain_b_miou = 0.0
            self.log("Could not parse Domain B metrics, using 0.0", level="WARNING")
        
        metrics['domain_B'] = domain_b_miou
        
        self.log(f"Domain A mIoU: {metrics['domain_A']:.4f}")
        self.log(f"Domain B mIoU: {metrics['domain_B']:.4f}")
        
        return metrics
    
    def _parse_metrics(self, eval_work_dir: str) -> Optional[float]:
        """
        Parse mIoU from evaluation results (from metrics.json or latest.json).
        
        Args:
            eval_work_dir: directory containing evaluation results
        
        Returns:
            mIoU value or None if not found
        """
        if not os.path.exists(eval_work_dir):
            self.log(f"[WARNING] eval_work_dir does not exist: {eval_work_dir}")
            return None
        
        # Collect all possible locations for metrics file
        possible_files = [
            os.path.join(eval_work_dir, "metrics.json"),
            os.path.join(eval_work_dir, "latest_metrics.json"),
            os.path.join(eval_work_dir, "latest.json"),
        ]
        
        # Check directory structure
        actual_files = os.listdir(eval_work_dir)
        
        # Look for timestamp subdirectories (mmengine creates YYYYMMDD_HHMMSS directories)
        timestamp_dirs = []
        timestamp_json_files = []
        
        for item in actual_files:
            item_path = os.path.join(eval_work_dir, item)
            
            # Case 1: Direct timestamp JSON files (e.g., 20260501_094154.json)
            if item.endswith('.json') and len(item) == 19:  # YYYYMMDD_HHMMSS.json
                timestamp_json_files.append(item_path)
            
            # Case 2: Timestamp subdirectories (e.g., 20260501_094154/)
            if os.path.isdir(item_path) and len(item) == 15:  # YYYYMMDD_HHMMSS
                timestamp_dirs.append(item_path)
        
        # Add timestamp JSON files to search list
        possible_files.extend(timestamp_json_files)
        
        # Add files within timestamp subdirectories
        for ts_dir in sorted(timestamp_dirs)[-1:]:  # Use the latest timestamp dir
            for subfile in [f for f in os.listdir(ts_dir) if f.endswith('.json')]:
                possible_files.append(os.path.join(ts_dir, subfile))
        
        self.log(f"[DEBUG] Searching for metrics in: {possible_files}")
        
        for metrics_file in possible_files:
            if os.path.exists(metrics_file):
                self.log(f"[DEBUG] Found metrics file: {metrics_file}")
                try:
                    with open(metrics_file, 'r') as f:
                        results = json.load(f)
                        self.log(f"[DEBUG] Loaded metrics: {results}")
                        # Look for mIoU in results with various possible keys
                        for key in ['mmseg/mIoU', 'mIoU', 'iou', 'IoU']:
                            if key in results:
                                miou_val = float(results[key])
                                self.log(f"Parsed mIoU={miou_val} from {metrics_file} (key: {key})")
                                return miou_val
                except Exception as e:
                    self.log(f"Error parsing {metrics_file}: {e}", level="WARNING")
        
        # If no metrics file found, log a warning
        self.log(f"[WARNING] No metrics file found in {eval_work_dir}", level="WARNING")
        return None
    
    def stage_continual_finetune_domain_b(
        self,
        init_from_work_dir: str,
        finetune_iters: int = 500,
    ) -> str:
        """
        Stage 4: Continual finetune on Domain B (starting from Domain A model).
        
        Args:
            init_from_work_dir: work directory from Domain A training
            finetune_iters: number of iterations for finetuning
        
        Returns:
            path to finetuned model work directory
        """
        if init_from_work_dir is None:
            self.log("Error: Domain A work directory is None. Cannot proceed with Domain B finetuning.", level="ERROR")
            return None
        
        self.log("="*80)
        self.log("STAGE 4: Continual Finetune on Domain B")
        self.log("="*80)
        
        work_dir = str(self.base_work_dir / 'stage4_continual_finetune_domain_b')
        
        # Copy config and modify for Domain B data
        # cfg_options = [
        #     f"train_cfg.max_iters={finetune_iters}",
        # ]
        best_ckpts = glob.glob(os.path.join(init_from_work_dir, "best_*.pth"))
        iter_ckpts = glob.glob(os.path.join(init_from_work_dir, "iter_*.pth"))

        if best_ckpts:
            domain_a_ckpt = max(best_ckpts, key=os.path.getmtime)
        elif iter_ckpts:
            domain_a_ckpt = max(iter_ckpts, key=os.path.getmtime)
        else:
            self.log("No Domain A checkpoint found!", level="ERROR")
            return None
        cfg_opt_str = ''
        cfg_opt_str = f"--cfg-options load_from={domain_a_ckpt} train_cfg.max_iters={finetune_iters} "
        
        # cfg_opt_str = ' '.join(cfg_options)
        
        cmd = (
            f"python tools/train.py "
            f"--config {self.domain_b_config} "
            f"--work-dir {work_dir} "
            f"{cfg_opt_str} "
            f"--amp"
        )
        
        ret = self.run_command(cmd, "Continual finetuning on Domain B")
        
        if ret == 0:
            self.log(f"Domain B finetuning completed. Work dir: {work_dir}")
            return work_dir
        else:
            self.log("Domain B finetuning failed!", level='ERROR')
            return None
    
    def run_full_pipeline(
        self,
        mask_root: str,
        skip_diffusion: bool = False,
        skip_domain_a: bool = False,
    ):
        """
        Run the complete continual learning experiment pipeline.
        """
        self.log(f"Starting continual learning experiment at {self.timestamp}")
        
        
        # Stage 1: Pretrain diffusion
        ckpt_path = os.path.join(self.base_work_dir,"stage1_diffusion_pretraining","unet_llm.pth")
        if not skip_diffusion:
            if os.path.exists(ckpt_path):
                self.log(f"Found existing diffusion checkpoint, skipping training: {ckpt_path}")
                diffusion_ckpt = ckpt_path
            else:
                diffusion_ckpt = self.stage_pretrain_diffusion(mask_root)
        
        # Stage 2: Train on Domain A
        domain_a_work_dir = os.path.join(
            self.base_work_dir,
            "stage2_train_domain_a"
        )

        domain_a_ckpts = []
        for pattern in ["best_*.pth", "latest.pth", "iter_*.pth"]:
            domain_a_ckpts.extend(
                glob.glob(os.path.join(domain_a_work_dir, pattern))
            )

        if domain_a_ckpts:
            latest_domain_a_ckpt = max(domain_a_ckpts, key=os.path.getmtime)
            self.log(
                f"Found existing Domain A checkpoint, skipping Domain A training: "
                f"{latest_domain_a_ckpt}"
            )
        else:
            if not skip_domain_a:
                domain_a_work_dir = self.stage_train_domain_a(diffusion_ckpt)
                if domain_a_work_dir is None:
                    self.log("Domain A training failed! Stopping pipeline.", level='ERROR')
                    return
            else:
                self.log(
                    "skip_domain_a=True but no existing Domain A checkpoint found!",
                    level="ERROR"
                )
                return
        
        # Stage 3a: Evaluate after Domain A training
        metrics_after_a = self.stage_evaluate_both_domains(
            domain_a_work_dir,
            stage_name='After Domain A Training'
        )
        self.evaluator.add_eval_result(stage_idx=0, domain_metrics=metrics_after_a)
        
        # Stage 4: Continual finetune on Domain B
        domain_b_work_dir = self.stage_continual_finetune_domain_b(domain_a_work_dir)
        
        if not domain_b_work_dir:
            self.log("Domain B finetuning failed!", level='ERROR')
            return
        
        # Stage 3b: Evaluate after Domain B training
        metrics_after_b = self.stage_evaluate_both_domains(
            domain_b_work_dir,
            stage_name='After Domain B Training'
        )
        self.evaluator.add_eval_result(stage_idx=1, domain_metrics=metrics_after_b)
        
        # Report final results
        self.log("="*80)
        self.log("FINAL CONTINUAL LEARNING RESULTS")
        self.log("="*80)
        self.log(str(self.evaluator))
        
        # Save results
        results_file = str(self.base_work_dir / 'continual_results.json')
        self.evaluator.save(results_file)
        self.log(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning Experiment Pipeline for PCB Conductor Segmentation'
    )
    
    parser.add_argument(
        '--base-work-dir',
        type=str,
        default='work_dirs/continual_experiment',
        help='Base directory for all experiment outputs'
    )
    
    parser.add_argument(
        '--mask-root',
        type=str,
        required=True,
        help='Root directory containing segmentation masks for diffusion pretraining'
    )
    
    parser.add_argument(
        '--domain-a-config',
        type=str,
        default='projects/pcb_conductor/configs/segformer_mt_vb_a.py',
        help='Config file for Domain A training'
    )
    
    parser.add_argument(
        '--domain-b-config',
        type=str,
        default='projects/pcb_conductor/configs/segformer_mt_vb_b.py',
        help='Config file for Domain B training'
    )
    
    parser.add_argument(
        '--skip-diffusion',
        action='store_true',
        help='Skip diffusion pretraining (use existing checkpoint)'
    )
    
    parser.add_argument(
        '--skip-domain-a',
        action='store_true',
        help='Skip Domain A training (use existing checkpoint)'
    )
    
    args = parser.parse_args()
    
    orchestrator = ContinualExperimentOrchestrator(
        base_work_dir=args.base_work_dir,
        domain_a_config=args.domain_a_config,
        domain_b_config=args.domain_b_config,
    )
    
    orchestrator.run_full_pipeline(
        mask_root=args.mask_root,
        skip_diffusion=args.skip_diffusion,
        skip_domain_a=args.skip_domain_a,
    )


if __name__ == '__main__':
    main()
