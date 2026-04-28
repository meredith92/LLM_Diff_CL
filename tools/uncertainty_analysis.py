#!/usr/bin/env python3
"""
Diffusion Uncertainty Analysis for PCB Conductor Segmentation

Analyzes uncertainty estimates from diffusion model predictions:
1. Timestep-wise uncertainty: confidence at different denoising steps
2. Spatial uncertainty: pixel-level confidence maps
3. Statistical analysis: distribution of uncertainty across domains
4. Uncertainty calibration: how well uncertainty correlates with error

This implements Section 4.3 of the paper.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import seaborn as sns


class DiffusionUncertaintyAnalyzer:
    """
    Analyze uncertainty in diffusion-based segmentation predictions.
    
    Key metrics:
    - Timestep uncertainty: variance across denoising timesteps
    - Spatial uncertainty: per-pixel prediction confidence
    - Prediction-error correlation: uncertainty vs actual error
    - Domain adaptation uncertainty: shift in uncertainty distribution
    """
    
    def __init__(self, num_timesteps: int = 1000):
        """
        Args:
            num_timesteps: number of diffusion timesteps
        """
        self.num_timesteps = num_timesteps
        self.timestep_uncertainties = []  # (T, H, W) for each sample
        self.spatial_uncertainties = []   # (H, W) final uncertainty map
        self.predictions = []              # model predictions
        self.ground_truth = []             # ground truth masks
        self.errors = []                   # prediction errors
        
    def add_sample(self,
                   timestep_predictions: torch.Tensor,
                   final_prediction: torch.Tensor,
                   ground_truth: torch.Tensor):
        """
        Add a sample with multi-timestep predictions.
        
        Args:
            timestep_predictions: (T, 1, H, W) predictions at each timestep
            final_prediction: (1, H, W) final denoised prediction
            ground_truth: (H, W) ground truth binary mask
        """
        # Ensure tensors are on CPU and detached
        if isinstance(timestep_predictions, torch.Tensor):
            timestep_predictions = timestep_predictions.detach().cpu()
        if isinstance(final_prediction, torch.Tensor):
            final_prediction = final_prediction.detach().cpu()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu()
        
        # Compute timestep-wise uncertainty as variance across timesteps
        ts_pred = timestep_predictions.numpy() if isinstance(timestep_predictions, torch.Tensor) else timestep_predictions
        ts_uncertainty = np.var(ts_pred, axis=0)  # (1, H, W)
        self.timestep_uncertainties.append(ts_uncertainty.squeeze())
        
        # Compute spatial uncertainty as softmax entropy of final prediction
        final_pred = final_prediction.numpy() if isinstance(final_prediction, torch.Tensor) else final_prediction
        final_pred = final_pred.squeeze()
        
        # Convert to probability if in [-1, 1] range
        if final_pred.min() < 0:
            prob = (final_pred + 1) / 2.0
        else:
            prob = final_pred
        
        # Clamp to valid probability range
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        
        # Entropy-based uncertainty
        entropy = -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))
        self.spatial_uncertainties.append(entropy)
        
        # Store prediction and ground truth
        binary_pred = (final_pred > 0.5).astype(np.float32)
        self.predictions.append(binary_pred)
        
        gt = ground_truth.numpy() if isinstance(ground_truth, torch.Tensor) else ground_truth
        gt = (gt > 0.5).astype(np.float32)
        self.ground_truth.append(gt)
        
        # Compute error
        error = np.abs(binary_pred - gt)
        self.errors.append(error)
    
    def compute_timestep_uncertainty_profile(self) -> np.ndarray:
        """
        Compute average uncertainty across timesteps.
        
        Returns:
            (T,) array of mean uncertainty at each timestep
        """
        if len(self.timestep_uncertainties) == 0:
            return np.array([])
        
        # Average spatial uncertainty for each timestep
        ts_unc = np.array(self.timestep_uncertainties)  # (N, H, W)
        return np.mean(ts_unc, axis=(1, 2))  # (N,) -> average per timestep
    
    def compute_spatial_uncertainty_map(self) -> np.ndarray:
        """
        Compute average spatial uncertainty map across all samples.
        
        Returns:
            (H, W) average uncertainty map
        """
        if len(self.spatial_uncertainties) == 0:
            return np.array([])
        
        spatial_unc = np.array(self.spatial_uncertainties)  # (N, H, W)
        return np.mean(spatial_unc, axis=0)  # (H, W)
    
    def compute_uncertainty_error_correlation(self) -> float:
        """
        Measure correlation between prediction uncertainty and actual error.
        
        Returns:
            correlation coefficient (0 to 1, higher is better)
        """
        if len(self.spatial_uncertainties) == 0:
            return 0.0
        
        # Flatten uncertainty and error maps
        uncertainties = np.concatenate([u.flatten() for u in self.spatial_uncertainties])
        errors = np.concatenate([e.flatten() for e in self.errors])
        
        # Compute correlation
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            correlation = 0.0
        
        return float(correlation)
    
    def compute_mean_uncertainty(self) -> float:
        """
        Compute average uncertainty across all predictions.
        
        Returns:
            mean uncertainty value
        """
        if len(self.spatial_uncertainties) == 0:
            return 0.0
        
        spatial_unc = np.array(self.spatial_uncertainties)
        return float(np.mean(spatial_unc))
    
    def compute_uncertainty_std(self) -> float:
        """
        Compute standard deviation of uncertainty.
        
        Returns:
            std of uncertainty
        """
        if len(self.spatial_uncertainties) == 0:
            return 0.0
        
        spatial_unc = np.array(self.spatial_uncertainties)
        return float(np.std(spatial_unc))
    
    def get_high_uncertainty_regions(self, threshold: float = 0.75) -> np.ndarray:
        """
        Get mask of high-uncertainty regions.
        
        Args:
            threshold: percentile threshold (0-100)
        
        Returns:
            binary mask of high-uncertainty pixels
        """
        spatial_map = self.compute_spatial_uncertainty_map()
        if spatial_map.size == 0:
            return np.array([])
        
        threshold_val = np.percentile(spatial_map, threshold)
        return (spatial_map > threshold_val).astype(np.float32)
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive uncertainty analysis summary.
        
        Returns:
            dict with all computed metrics
        """
        return {
            'num_samples': len(self.spatial_uncertainties),
            'mean_uncertainty': self.compute_mean_uncertainty(),
            'std_uncertainty': self.compute_uncertainty_std(),
            'uncertainty_error_correlation': self.compute_uncertainty_error_correlation(),
            'timestep_uncertainty_profile': self.compute_timestep_uncertainty_profile().tolist(),
            'spatial_uncertainty_map_shape': (
                self.compute_spatial_uncertainty_map().shape 
                if len(self.spatial_uncertainties) > 0 else None
            ),
        }
    
    def save_uncertainty_map(self, output_path: str):
        """
        Save uncertainty map as visualization.
        
        Args:
            output_path: where to save the image
        """
        spatial_map = self.compute_spatial_uncertainty_map()
        if spatial_map.size == 0:
            print(f"[WARNING] No samples to visualize")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Normalize to [0, 255]
        spatial_map_norm = ((spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)) * 255
        spatial_map_norm = spatial_map_norm.astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(spatial_map_norm, cv2.COLORMAP_JET)
        
        # Save
        cv2.imwrite(output_path, colored)
        print(f"[OK] Saved uncertainty map to {output_path}")
    
    def plot_uncertainty_distribution(self, output_path: str):
        """
        Plot uncertainty distribution histogram.
        
        Args:
            output_path: where to save the figure
        """
        if len(self.spatial_uncertainties) == 0:
            print(f"[WARNING] No samples to plot")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        spatial_unc = np.concatenate([u.flatten() for u in self.spatial_uncertainties])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
        
        # Histogram
        ax = axes[0]
        ax.hist(spatial_unc, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Uncertainty (Entropy)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # KDE plot
        ax = axes[1]
        ax.hist(spatial_unc, bins=50, alpha=0.5, density=True, color='steelblue', label='Data')
        
        # Add KDE
        from scipy import stats
        kde = stats.gaussian_kde(spatial_unc)
        x_range = np.linspace(spatial_unc.min(), spatial_unc.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax.set_xlabel('Uncertainty (Entropy)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Uncertainty Distribution (KDE)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved uncertainty distribution to {output_path}")
    
    def plot_timestep_uncertainty(self, output_path: str):
        """
        Plot uncertainty evolution across denoising timesteps.
        
        Args:
            output_path: where to save the figure
        """
        ts_profile = self.compute_timestep_uncertainty_profile()
        if len(ts_profile) == 0:
            print(f"[WARNING] No timestep data to plot")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        timesteps = np.arange(len(ts_profile))
        ax.plot(timesteps, ts_profile, 'o-', linewidth=2, markersize=6, color='steelblue')
        ax.fill_between(timesteps, ts_profile, alpha=0.3, color='steelblue')
        
        ax.set_xlabel('Denoising Timestep', fontsize=12)
        ax.set_ylabel('Mean Uncertainty', fontsize=12)
        ax.set_title('Uncertainty Evolution During Diffusion Denoising', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved timestep uncertainty to {output_path}")


def analyze_uncertainty_for_dataset(dataset_dir: str,
                                   output_dir: str = 'work_dirs/uncertainty_analysis',
                                   num_samples: Optional[int] = None) -> DiffusionUncertaintyAnalyzer:
    """
    Analyze uncertainty for all samples in a dataset.
    
    Args:
        dataset_dir: directory containing prediction and GT files
        output_dir: where to save results
        num_samples: limit number of samples (None = use all)
    
    Returns:
        DiffusionUncertaintyAnalyzer with results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DiffusionUncertaintyAnalyzer()
    
    # This is a placeholder - in real usage, you would:
    # 1. Load timestep predictions from diffusion model inference
    # 2. Load final predictions and ground truth
    # 3. Call analyzer.add_sample() for each
    
    print(f"[INFO] Uncertainty analysis would process dataset from: {dataset_dir}")
    print(f"[INFO] Results will be saved to: {output_dir}")
    
    return analyzer


def main():
    parser = argparse.ArgumentParser(
        description='Diffusion Uncertainty Analysis for Segmentation'
    )
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Directory with predictions and ground truth')
    parser.add_argument('--output-dir', type=str, default='work_dirs/uncertainty_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Limit number of samples (None = all)')
    parser.add_argument('--save-maps', action='store_true',
                       help='Save uncertainty maps')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Diffusion Uncertainty Analysis")
    print("="*80)
    
    # Analyze uncertainty
    analyzer = analyze_uncertainty_for_dataset(
        args.dataset_dir,
        args.output_dir,
        args.num_samples
    )
    
    # Save results
    summary = analyzer.get_summary()
    results_file = os.path.join(args.output_dir, 'uncertainty_summary.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved uncertainty summary to {results_file}")
    
    # Generate visualizations
    if args.save_maps:
        analyzer.save_uncertainty_map(
            os.path.join(args.output_dir, 'uncertainty_map.png')
        )
    
    if args.plot:
        analyzer.plot_uncertainty_distribution(
            os.path.join(args.output_dir, 'uncertainty_distribution.png')
        )
        analyzer.plot_timestep_uncertainty(
            os.path.join(args.output_dir, 'timestep_uncertainty.png')
        )
    
    # Print summary
    print("\n" + "="*80)
    print("UNCERTAINTY ANALYSIS SUMMARY")
    print("="*80)
    for key, val in summary.items():
        if isinstance(val, (int, float)):
            print(f"{key:.<40} {val:.4f}")
        elif isinstance(val, list):
            print(f"{key:.<40} [list with {len(val)} items]")
        else:
            print(f"{key:.<40} {val}")


if __name__ == '__main__':
    main()
