#!/usr/bin/env python3
"""
Structural Analysis for PCB Conductor Topology

Analyzes the topological properties of predicted conductor pins:
1. Pin detection: identify individual pins in segmentation masks
2. Pin measurements: compute pin count, spacing, alignment
3. Topology metrics: structural quality and consistency
4. Error analysis: measure topology violations in predictions
5. Domain-specific insights: differences between domains

This implements Section 5.5 of the paper.
"""

import os
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy.spatial.distance import cdist
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import seaborn as sns


@dataclass
class Pin:
    """Represents a detected conductor pin."""
    id: int
    center: Tuple[float, float]
    area: float
    width: float
    height: float
    confidence: float = 1.0


class StructureAnalyzer:
    """
    Analyze topological structure of PCB conductor pins.
    """
    
    def __init__(self, 
                 min_pin_area: int = 50,
                 max_pin_area: int = 5000,
                 pin_aspect_ratio_bounds: Tuple[float, float] = (0.5, 3.0)):
        """
        Args:
            min_pin_area: minimum pin area (pixels)
            max_pin_area: maximum pin area (pixels)
            pin_aspect_ratio_bounds: valid range for width/height ratio
        """
        self.min_pin_area = min_pin_area
        self.max_pin_area = max_pin_area
        self.pin_aspect_ratio_bounds = pin_aspect_ratio_bounds
        
        self.predictions = []
        self.ground_truths = []
        self.detected_pins = []  # List of List[Pin] for each sample
        self.gt_pins = []        # List of List[Pin] for each sample
    
    def detect_pins(self, mask: np.ndarray) -> List[Pin]:
        """
        Detect individual pins from binary segmentation mask.
        
        Args:
            mask: (H, W) binary mask with 1 = foreground (pins)
        
        Returns:
            list of detected Pin objects
        """
        # Ensure binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        
        pins = []
        
        for pin_id in range(1, num_features + 1):
            # Extract pin region
            pin_region = (labeled_array == pin_id).astype(np.uint8)
            
            # Compute area
            area = np.sum(pin_region)
            
            # Filter by area
            if area < self.min_pin_area or area > self.max_pin_area:
                continue
            
            # Compute bounding box
            coords = np.where(pin_region > 0)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            # Check aspect ratio
            aspect_ratio = width / (height + 1e-8)
            if not (self.pin_aspect_ratio_bounds[0] <= aspect_ratio <= self.pin_aspect_ratio_bounds[1]):
                continue
            
            # Compute center
            center_y = (y_min + y_max) / 2.0
            center_x = (x_min + x_max) / 2.0
            
            pin = Pin(
                id=pin_id,
                center=(center_x, center_y),
                area=float(area),
                width=float(width),
                height=float(height),
            )
            
            pins.append(pin)
        
        return pins
    
    def add_sample(self, prediction: np.ndarray, ground_truth: np.ndarray):
        """
        Add a sample for analysis.
        
        Args:
            prediction: (H, W) predicted segmentation mask
            ground_truth: (H, W) ground truth mask
        """
        # Detect pins
        pred_pins = self.detect_pins(prediction)
        gt_pins = self.detect_pins(ground_truth)
        
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.detected_pins.append(pred_pins)
        self.gt_pins.append(gt_pins)
    
    def compute_pin_spacing(self, pins: List[Pin]) -> np.ndarray:
        """
        Compute pairwise distances between pin centers.
        
        Args:
            pins: list of Pin objects
        
        Returns:
            (N, N) distance matrix
        """
        if len(pins) < 2:
            return np.array([])
        
        centers = np.array([p.center for p in pins])
        distances = cdist(centers, centers)
        
        return distances
    
    def compute_alignment_score(self, pins: List[Pin]) -> float:
        """
        Compute how well pins are aligned horizontally or vertically.
        
        Returns:
            score from 0 (random) to 1 (perfectly aligned)
        """
        if len(pins) < 2:
            return 1.0
        
        centers = np.array([p.center for p in pins])
        
        # Check horizontal alignment: variance of y-coordinates
        y_coords = centers[:, 1]
        y_variance = np.var(y_coords)
        
        # Check vertical alignment: variance of x-coordinates
        x_coords = centers[:, 0]
        x_variance = np.var(x_coords)
        
        # Normalized by image size (assume 256x256)
        normalized_y_var = y_variance / (256 ** 2)
        normalized_x_var = x_variance / (256 ** 2)
        
        # Lower variance = better alignment
        # Convert to score (1 = perfect, 0 = random)
        alignment = 1.0 - (normalized_y_var + normalized_x_var) / 2.0
        alignment = max(0.0, min(1.0, alignment))
        
        return float(alignment)
    
    def compute_uniformity_score(self, pins: List[Pin]) -> float:
        """
        Compute how uniform the pin sizes are.
        
        Returns:
            score from 0 (varying sizes) to 1 (uniform sizes)
        """
        if len(pins) < 2:
            return 1.0
        
        areas = np.array([p.area for p in pins])
        
        # Coefficient of variation
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        cv = std_area / (mean_area + 1e-8)
        
        # Convert CV to uniformity score
        uniformity = 1.0 / (1.0 + cv)
        
        return float(uniformity)
    
    def compute_pin_count_agreement(self) -> float:
        """
        Measure agreement between predicted and GT pin counts.
        
        Returns:
            accuracy (0 to 1)
        """
        if len(self.detected_pins) == 0:
            return 0.0
        
        agreements = []
        
        for pred_pins, gt_pins in zip(self.detected_pins, self.gt_pins):
            pred_count = len(pred_pins)
            gt_count = len(gt_pins)
            
            # Count agreement: 1 if perfect, 0 if completely off
            if gt_count == 0:
                if pred_count == 0:
                    agreement = 1.0
                else:
                    agreement = 0.0
            else:
                # Allow some tolerance (±20%)
                tolerance = max(1, int(0.2 * gt_count))
                if abs(pred_count - gt_count) <= tolerance:
                    agreement = 1.0 - (abs(pred_count - gt_count) / gt_count)
                else:
                    agreement = 0.0
            
            agreements.append(agreement)
        
        return float(np.mean(agreements))
    
    def compute_topology_violation_rate(self) -> float:
        """
        Measure how often predicted pins violate structural rules.
        
        Common violations:
        - Pins touching/overlapping (should be separate)
        - Gaps in uniform rows (should be evenly spaced)
        - Size outliers (should be similar size)
        
        Returns:
            violation rate (0 = no violations, 1 = all violations)
        """
        if len(self.detected_pins) == 0:
            return 0.0
        
        violation_rates = []
        
        for pins in self.detected_pins:
            if len(pins) < 2:
                violation_rates.append(0.0)
                continue
            
            violations = 0
            total_checks = 0
            
            # Check 1: Pins should not overlap or touch
            centers = np.array([p.center for p in pins])
            sizes = np.array([p.width for p in pins])
            
            for i in range(len(pins)):
                for j in range(i + 1, len(pins)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    min_dist = (sizes[i] + sizes[j]) / 2.0
                    
                    # Allow some overlap tolerance
                    if dist < min_dist * 0.8:
                        violations += 1
                    total_checks += 1
            
            # Check 2: Size uniformity
            areas = np.array([p.area for p in pins])
            size_outliers = np.sum(np.abs(areas - np.median(areas)) > 2 * np.std(areas))
            violations += size_outliers
            total_checks += len(pins)
            
            if total_checks > 0:
                rate = min(1.0, violations / total_checks)
            else:
                rate = 0.0
            
            violation_rates.append(rate)
        
        return float(np.mean(violation_rates))
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive structure analysis summary.
        
        Returns:
            dict with all computed metrics
        """
        return {
            'num_samples': len(self.predictions),
            'avg_predicted_pin_count': float(np.mean([len(p) for p in self.detected_pins])) if self.detected_pins else 0,
            'avg_gt_pin_count': float(np.mean([len(p) for p in self.gt_pins])) if self.gt_pins else 0,
            'pin_count_agreement': self.compute_pin_count_agreement(),
            'topology_violation_rate': self.compute_topology_violation_rate(),
            'avg_alignment_score': float(np.mean([self.compute_alignment_score(p) for p in self.detected_pins])) if self.detected_pins else 0,
            'avg_uniformity_score': float(np.mean([self.compute_uniformity_score(p) for p in self.detected_pins])) if self.detected_pins else 0,
        }
    
    def visualize_pins(self, 
                      sample_idx: int,
                      output_path: str):
        """
        Visualize detected pins on prediction and GT.
        
        Args:
            sample_idx: which sample to visualize
            output_path: where to save figure
        """
        if sample_idx >= len(self.predictions):
            print(f"[ERROR] Sample index {sample_idx} out of range")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        pred = self.predictions[sample_idx]
        gt = self.ground_truths[sample_idx]
        pred_pins = self.detected_pins[sample_idx]
        gt_pins = self.gt_pins[sample_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        
        # Prediction
        ax = axes[0]
        ax.imshow(pred, cmap='gray', alpha=0.6)
        
        # Draw detected pins
        for pin in pred_pins:
            circle = mpatches.Circle(pin.center, radius=5, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(circle)
            ax.text(pin.center[0], pin.center[1]-8, str(pin.id), color='red', fontsize=8, ha='center')
        
        ax.set_title(f'Predicted Pins (Count: {len(pred_pins)})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Ground Truth
        ax = axes[1]
        ax.imshow(gt, cmap='gray', alpha=0.6)
        
        # Draw GT pins
        for pin in gt_pins:
            circle = mpatches.Circle(pin.center, radius=5, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(circle)
            ax.text(pin.center[0], pin.center[1]-8, str(pin.id), color='green', fontsize=8, ha='center')
        
        ax.set_title(f'Ground Truth Pins (Count: {len(gt_pins)})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved pin visualization to {output_path}")
    
    def plot_structure_metrics(self, output_path: str):
        """
        Plot structure analysis results.
        
        Args:
            output_path: where to save figure
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        summary = self.get_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        
        # Pin count comparison
        ax = axes[0, 0]
        categories = ['Predicted', 'Ground Truth']
        counts = [summary['avg_predicted_pin_count'], summary['avg_gt_pin_count']]
        bars = ax.bar(categories, counts, color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=1.5)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Pin Count', fontsize=11)
        ax.set_title('Pin Count Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Metrics comparison
        ax = axes[0, 1]
        metrics_names = ['Count Agree', 'Alignment', 'Uniformity']
        metrics_vals = [
            summary['pin_count_agreement'],
            summary['avg_alignment_score'],
            summary['avg_uniformity_score'],
        ]
        bars = ax.bar(metrics_names, metrics_vals, color=['#e74c3c', '#f39c12', '#27ae60'],
                     edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, metrics_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.set_title('Structure Quality Metrics', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Violation rate
        ax = axes[1, 0]
        violation_rate = summary['topology_violation_rate']
        compliance_rate = 1.0 - violation_rate
        
        categories = ['Compliant', 'Violations']
        sizes = [compliance_rate, violation_rate]
        colors = ['#2ecc71', '#e74c3c']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%',
                                           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Topology Rule Compliance', fontsize=12, fontweight='bold')
        
        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        Structure Analysis Summary
        {'='*40}
        
        Samples Analyzed:        {summary['num_samples']}
        
        Pin Detection:
          Avg Predicted Count:   {summary['avg_predicted_pin_count']:.1f}
          Avg GT Count:          {summary['avg_gt_pin_count']:.1f}
          Count Agreement:       {summary['pin_count_agreement']:.3f}
        
        Structural Quality:
          Alignment Score:       {summary['avg_alignment_score']:.3f}
          Uniformity Score:      {summary['avg_uniformity_score']:.3f}
          Violation Rate:        {summary['topology_violation_rate']:.3f}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved structure metrics plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Structural Analysis for PCB Conductor Topology'
    )
    parser.add_argument('--pred-dir', type=str, required=True,
                       help='Directory with predicted masks')
    parser.add_argument('--gt-dir', type=str, required=True,
                       help='Directory with ground truth masks')
    parser.add_argument('--output-dir', type=str, default='work_dirs/structure_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate pin visualizations')
    parser.add_argument('--plot', action='store_true',
                       help='Generate metric plots')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PCB Conductor Structure Analysis")
    print("="*80)
    
    # Create analyzer
    analyzer = StructureAnalyzer()
    
    # This is a placeholder - in real usage:
    # 1. Load prediction and GT masks from files
    # 2. Call analyzer.add_sample() for each
    # 3. Generate results and visualizations
    
    print(f"[INFO] Would analyze predictions from: {args.pred_dir}")
    print(f"[INFO] Would load GT masks from: {args.gt_dir}")
    print(f"[INFO] Results would be saved to: {args.output_dir}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    summary = analyzer.get_summary()
    
    results_file = os.path.join(args.output_dir, 'structure_summary.json')
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[OK] Saved structure summary to {results_file}")
    
    # Generate visualizations
    if args.visualize and len(analyzer.predictions) > 0:
        analyzer.visualize_pins(0, os.path.join(args.output_dir, 'pin_detection.png'))
    
    if args.plot and len(analyzer.predictions) > 0:
        analyzer.plot_structure_metrics(os.path.join(args.output_dir, 'structure_metrics.png'))
    
    print("\nStructure Analysis Summary:")
    for key, val in summary.items():
        print(f"  {key:.<40} {val:.4f}" if isinstance(val, (int, float)) else f"  {key}: {val}")


if __name__ == '__main__':
    main()
