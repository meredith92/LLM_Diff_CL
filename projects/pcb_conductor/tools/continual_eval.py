"""
Continual Learning Evaluation Framework for PCB Conductor Segmentation

Protocol: Domain A (source) → Domain B (target)
Metrics: Target mIoU, Forgetting, Plasticity, Backward Transfer
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from mmengine.config import Config
from mmengine.runner import Runner


class ContinualSegmentationEvaluator:
    """
    Evaluate continual learning performance on sequential domain adaptation.
    
    Key metrics:
    - Target Performance: mIoU on new domain after adaptation
    - Forgetting: loss in old domain after adapting to new domain
    - Plasticity: improvement on new domain
    - Backward Transfer: negative impact of new domain on old domain
    """
    
    def __init__(self, domain_order: List[str], metric_key: str = 'mIoU'):
        """
        Args:
            domain_order: list of domain names in training order, e.g., ['domain_A', 'domain_B']
            metric_key: metric to track, e.g., 'mIoU', 'mDice'
        """
        self.domain_order = domain_order
        self.num_domains = len(domain_order)
        self.metric_key = metric_key
        
        # [time_step, domain_idx] = metric value
        self.iou_matrix = np.zeros((self.num_domains, self.num_domains))
        self.iou_matrix[:] = np.nan  # NaN for unseen domain-time combinations
        
        self.history = []  # List of (stage, domain_results)
    
    def add_eval_result(self, stage_idx: int, domain_metrics: Dict[str, float]):
        """
        Add evaluation results after training on stage_idx.
        
        Args:
            stage_idx: which domain just finished training (0=domain_A, 1=domain_B, etc.)
            domain_metrics: dict mapping domain_name -> metric value
                e.g., {'domain_A': 0.92, 'domain_B': 0.85}
        """
        for domain_name, metric_val in domain_metrics.items():
            domain_idx = self.domain_order.index(domain_name)
            self.iou_matrix[stage_idx, domain_idx] = metric_val
        
        self.history.append({
            'stage': stage_idx,
            'domain_order_trained': self.domain_order[:stage_idx+1],
            'metrics': domain_metrics,
        })
    
    @property
    def target_performance(self) -> float:
        """
        Performance on the final domain.
        Target mIoU = mIoU(last_domain) after all training.
        """
        last_stage = self.num_domains - 1
        if not np.isnan(self.iou_matrix[last_stage, last_stage]):
            return float(self.iou_matrix[last_stage, last_stage])
        return np.nan
    
    @property
    def average_forgetting(self) -> float:
        """
        Average forgetting over all old domains.
        
        For domain j < current_stage:
            F_j = max_{t < current_stage} mIoU(D_j) - mIoU(D_j, final_model)
        
        Average Forgetting = mean(F_j)
        """
        current_stage = self.num_domains - 1
        forgetting_list = []
        
        for domain_idx in range(current_stage):
            # Max IoU before current domain was trained
            max_iou_before = np.nanmax(self.iou_matrix[:current_stage, domain_idx])
            # Final IoU after all training
            final_iou = self.iou_matrix[current_stage, domain_idx]
            
            if not (np.isnan(max_iou_before) or np.isnan(final_iou)):
                forgetting = max_iou_before - final_iou
                forgetting_list.append(forgetting)
        
        if len(forgetting_list) > 0:
            return float(np.mean(forgetting_list))
        return np.nan
    
    @property
    def backward_transfer(self) -> float:
        """
        Backward Transfer: performance change on source domain.
        BWT = mIoU(source, after_all) - mIoU(source, after_source)
        """
        if self.num_domains < 2:
            return np.nan
        
        source_iou_initial = self.iou_matrix[0, 0]
        source_iou_final = self.iou_matrix[-1, 0]
        
        if not (np.isnan(source_iou_initial) or np.isnan(source_iou_final)):
            return float(source_iou_final - source_iou_initial)
        return np.nan
    
    @property
    def forward_transfer(self) -> float:
        """
        Forward Transfer: how much learning source helps target.
        FWT = mIoU(target, after_source) - baseline(target)
        
        Note: baseline(target) not available without zero-shot evaluation.
        Simplified: use improvement on target domain.
        """
        if self.num_domains < 2:
            return np.nan
        
        target_idx = self.num_domains - 1
        target_iou_after_source = self.iou_matrix[0, target_idx]  # Evaluate target after source training
        
        # Note: This is NaN if we don't evaluate target after each stage.
        # For true forward transfer, need zero-shot baseline.
        if not np.isnan(target_iou_after_source):
            return float(target_iou_after_source)
        return np.nan
    
    @property
    def plasticity(self) -> float:
        """
        Plasticity: ability to improve on new domains.
        Simplified: improvement on current domain from initial to final.
        """
        # Improvement on last domain (most recent)
        last_idx = self.num_domains - 1
        final_iou = self.iou_matrix[last_idx, last_idx]
        
        if not np.isnan(final_iou):
            return float(final_iou)
        return np.nan
    
    def get_summary(self) -> Dict:
        """Return a comprehensive summary of continual learning metrics."""
        return {
            'target_performance': self.target_performance,
            'average_forgetting': self.average_forgetting,
            'backward_transfer': self.backward_transfer,
            'plasticity': self.plasticity,
            'iou_matrix': self.iou_matrix.tolist(),
            'history': self.history,
        }
    
    def to_dict(self) -> Dict:
        """Export as dictionary for saving."""
        return {
            'domain_order': self.domain_order,
            'summary': self.get_summary(),
            'iou_matrix': self.iou_matrix.tolist(),
        }
    
    def save(self, filepath: str):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[OK] Saved continual eval results to {filepath}")
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"ContinualSegmentationEvaluator(\n"
            f"  domains={self.domain_order},\n"
            f"  target_mIoU={summary['target_performance']:.4f},\n"
            f"  avg_forgetting={summary['average_forgetting']:.4f},\n"
            f"  backward_transfer={summary['backward_transfer']:.4f},\n"
            f"  plasticity={summary['plasticity']:.4f}\n"
            f")"
        )


def run_single_domain_training(config_path: str, work_dir: str, device: str = 'cuda'):
    """
    Train model on a single domain using MMSegmentation.
    
    Returns:
        metrics_dict: dict of computed metrics on val set
    """
    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    # Evaluation is done inside runner.train() and saved to work_dir
    # Parse the metrics from the last checkpoint's eval results
    eval_results = runner.evaluate()  # This varies by MMSeg version
    
    return eval_results


def example_continual_experiment():
    """
    Example: Domain A → Domain B continual learning experiment
    """
    
    # 1. Setup
    evaluator = ContinualSegmentationEvaluator(
        domain_order=['domain_A', 'domain_B'],
        metric_key='mIoU'
    )
    
    # 2. Stage 1: Train on Domain A
    print("\n" + "="*60)
    print("Stage 1: Training on Domain A")
    print("="*60)
    config_a = 'projects/pcb_conductor/configs/segformer_mt_vb.py'
    work_dir_a = 'work_dirs/continual_stage1_domain_a'
    
    # For now, simulate results (in real experiment, call run_single_domain_training)
    # metrics_a = run_single_domain_training(config_a, work_dir_a)
    metrics_a = {'domain_A': 0.920, 'domain_B': 0.650}  # Simulated: B is unseen
    evaluator.add_eval_result(stage_idx=0, domain_metrics=metrics_a)
    
    print(f"Domain A mIoU: {metrics_a['domain_A']:.4f}")
    print(f"Domain B mIoU (zero-shot from A): {metrics_a['domain_B']:.4f}")
    
    # 3. Stage 2: Continual finetune on Domain B
    print("\n" + "="*60)
    print("Stage 2: Continual Finetune on Domain B")
    print("="*60)
    config_b = 'projects/pcb_conductor/configs/segformer_mt_vb.py'  # Same config, different data
    work_dir_b = 'work_dirs/continual_stage2_domain_b'
    
    # metrics_b = run_single_domain_training(config_b, work_dir_b, init_from=work_dir_a)
    metrics_b = {'domain_A': 0.895, 'domain_B': 0.910}  # Simulated: improved B, slight drop in A
    evaluator.add_eval_result(stage_idx=1, domain_metrics=metrics_b)
    
    print(f"Domain A mIoU (after B training): {metrics_b['domain_A']:.4f}  [Forgetting: {metrics_a['domain_A'] - metrics_b['domain_A']:.4f}]")
    print(f"Domain B mIoU (after B training): {metrics_b['domain_B']:.4f}")
    
    # 4. Report results
    print("\n" + "="*60)
    print("CONTINUAL LEARNING RESULTS")
    print("="*60)
    print(evaluator)
    
    # 5. Save
    evaluator.save('work_dirs/continual_results.json')
    
    return evaluator


if __name__ == '__main__':
    # Example usage
    evaluator = example_continual_experiment()
