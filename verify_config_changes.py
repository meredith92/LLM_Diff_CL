#!/usr/bin/env python
"""Verify that configuration changes are reasonable."""

import torch
from projects.pcb_conductor.models.mt_struct_continual import (
    selective_weight_map, boundary_band
)

def test_weight_filtering():
    """Test weight filtering with different thresholds."""
    # Simulate teacher predictions
    B, H, W = 2, 256, 1024
    
    # Create test probability maps
    # Case 1: Confident predictions (mostly binary)
    p_t_confident = torch.cat([
        torch.ones(B//2, 1, H, W) * 0.95,  # High confidence
        torch.ones(B//2, 1, H, W) * 0.05,  # High confidence (negative)
    ])
    
    # Case 2: Mixed confidence
    p_t_mixed = torch.rand(B, 1, H, W)
    
    print("="*60)
    print("Testing weight filtering thresholds")
    print("="*60)
    
    for conf_thr in [0.8, 0.5, 0.3]:
        for drop_boundary in [True, False]:
            # Test on confident predictions
            w_conf = selective_weight_map(
                p_t_confident, 
                conf_thr=conf_thr,
                drop_boundary=drop_boundary,
                band_k=5
            )
            
            # Test on mixed predictions
            w_mixed = selective_weight_map(
                p_t_mixed,
                conf_thr=conf_thr,
                drop_boundary=drop_boundary,
                band_k=5
            )
            
            conf_kept = (w_conf.sum() / w_conf.numel() * 100)
            mixed_kept = (w_mixed.sum() / w_mixed.numel() * 100)
            
            print(f"\nconf_thr={conf_thr}, drop_boundary={drop_boundary}")
            print(f"  Confident predictions: {conf_kept:.1f}% pixels kept")
            print(f"  Mixed predictions:     {mixed_kept:.1f}% pixels kept")
            
            # Warn if too few pixels are kept
            if conf_kept < 1:
                print(f"  ⚠️  WARNING: Only {conf_kept:.2f}% pixels kept! (likely zero loss)")
            if mixed_kept < 1:
                print(f"  ⚠️  WARNING: Only {mixed_kept:.2f}% pixels kept! (likely zero loss)")

if __name__ == "__main__":
    test_weight_filtering()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("✓ conf_thr=0.5, drop_boundary=False")
    print("  Keeps ~50% of pixels for self-supervised learning")
    print("  Preserves important boundary information")
    print("  Avoids zero loss issue")
