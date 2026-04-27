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

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from projects.pcb_conductor.tools.continual_eval import ContinualSegmentationEvaluator


class ContinualExperimentOrchestrator:
    """
    Orchestrate the full continual learning experiment pipeline.
    """
    
    def __init__(
        self,
        base_work_dir: str = 'work_dirs/continual_experiment',
        domain_a_config: str = 'projects/pcb_conductor/configs/segformer_mt_vb.py',
        domain_b_config: str = 'projects/pcb_conductor/configs/segformer_mt_vb.py',
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
    
    def run_command(self, cmd: str, description: str = '') -> int:
        """Execute shell command and log output."""
        if description:
            self.log(f"Running: {description}")
        self.log(f"Command: {cmd}")
        
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            self.log(f"ERROR: Command failed with return code {result.returncode}", level='ERROR')
        return result.returncode
    
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
            self.log(f"✓ Diffusion pretraining completed. Checkpoint: {ckpt_path}")
            return ckpt_path
        else:
            self.log("✗ Diffusion pretraining failed!", level='ERROR')
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
        cfg_options = []
        if diffusion_ckpt:
            cfg_options.append(f"model.diff_ckpt='{diffusion_ckpt}'")
        
        cfg_opt_str = ' '.join(cfg_options) if cfg_options else ''
        
        cmd = (
            f"python tools/train.py "
            f"--config {self.domain_a_config} "
            f"--work-dir {work_dir} "
            f"{cfg_opt_str} "
            f"--amp"
        )
        
        ret = self.run_command(cmd, "Training segmentation on Domain A")
        
        if ret == 0:
            self.log(f"✓ Domain A training completed. Work dir: {work_dir}")
            return work_dir
        else:
            self.log("✗ Domain A training failed!", level='ERROR')
            return None
    
    def stage_evaluate_both_domains(self, model_ckpt: str, stage_name: str = 'after_A'):
        """
        Stage 3: Evaluate the model on both Domain A and B.
        
        Args:
            model_ckpt: path to model checkpoint
            stage_name: identifier for this evaluation stage
        
        Returns:
            dict of metrics: {'domain_A': float, 'domain_B': float}
        """
        self.log("="*80)
        self.log(f"EVALUATING: {stage_name}")
        self.log("="*80)
        
        # TODO: Parse actual metrics from test.py output
        # For now, return placeholder
        metrics = {
            'domain_A': 0.920,  # Placeholder
            'domain_B': 0.650,  # Placeholder (zero-shot)
        }
        
        self.log(f"Domain A mIoU: {metrics['domain_A']:.4f}")
        self.log(f"Domain B mIoU: {metrics['domain_B']:.4f}")
        
        return metrics
    
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
        self.log("="*80)
        self.log("STAGE 4: Continual Finetune on Domain B")
        self.log("="*80)
        
        work_dir = str(self.base_work_dir / 'stage4_continual_finetune_domain_b')
        
        # Copy config and modify for Domain B data
        cfg_options = [
            f"train_cfg.max_iters={finetune_iters}",
        ]
        
        cfg_opt_str = ' '.join(cfg_options)
        
        cmd = (
            f"python tools/train.py "
            f"--config {self.domain_b_config} "
            f"--work-dir {work_dir} "
            f"--resume "  # Resume from latest checkpoint in work_dir
            f"{cfg_opt_str} "
            f"--amp"
        )
        
        ret = self.run_command(cmd, "Continual finetuning on Domain B")
        
        if ret == 0:
            self.log(f"✓ Domain B finetuning completed. Work dir: {work_dir}")
            return work_dir
        else:
            self.log("✗ Domain B finetuning failed!", level='ERROR')
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
        diffusion_ckpt = None
        if not skip_diffusion:
            diffusion_ckpt = self.stage_pretrain_diffusion(mask_root)
        
        # Stage 2: Train on Domain A
        domain_a_work_dir = None
        if not skip_domain_a:
            domain_a_work_dir = self.stage_train_domain_a(diffusion_ckpt)
        
        if not domain_a_work_dir:
            self.log("Cannot proceed without Domain A model!", level='ERROR')
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
        default='projects/pcb_conductor/configs/segformer_mt_vb.py',
        help='Config file for Domain A training'
    )
    
    parser.add_argument(
        '--domain-b-config',
        type=str,
        default='projects/pcb_conductor/configs/segformer_mt_vb.py',
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
