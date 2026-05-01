#!/usr/bin/env python
"""Quick test to verify metrics parsing works correctly."""

import os
import json
from pathlib import Path

# Simulate the _parse_metrics function
def parse_metrics(eval_work_dir: str):
    """Test version of metrics parser."""
    if not os.path.exists(eval_work_dir):
        print(f"[WARNING] eval_work_dir does not exist: {eval_work_dir}")
        return None
    
    possible_files = [
        os.path.join(eval_work_dir, "metrics.json"),
        os.path.join(eval_work_dir, "latest_metrics.json"),
        os.path.join(eval_work_dir, "latest.json"),
    ]
    
    actual_files = os.listdir(eval_work_dir)
    
    timestamp_dirs = []
    timestamp_json_files = []
    
    for item in actual_files:
        item_path = os.path.join(eval_work_dir, item)
        
        # Case 1: Direct timestamp JSON files (e.g., 20260501_094154.json)
        if item.endswith('.json') and len(item) == 19:  # YYYYMMDD_HHMMSS.json
            timestamp_json_files.append(item_path)
            print(f"Found timestamp JSON: {item}")
        
        # Case 2: Timestamp subdirectories
        if os.path.isdir(item_path) and len(item) == 15:  # YYYYMMDD_HHMMSS
            timestamp_dirs.append(item_path)
            print(f"Found timestamp dir: {item}")
    
    possible_files.extend(timestamp_json_files)
    
    for ts_dir in sorted(timestamp_dirs)[-1:]:
        for subfile in [f for f in os.listdir(ts_dir) if f.endswith('.json')]:
            possible_files.append(os.path.join(ts_dir, subfile))
            print(f"Adding file from dir: {subfile}")
    
    print(f"Total possible files: {len(possible_files)}")
    
    for metrics_file in possible_files:
        if os.path.exists(metrics_file):
            print(f"✓ Found metrics file: {metrics_file}")
            try:
                with open(metrics_file, 'r') as f:
                    results = json.load(f)
                    print(f"  Metrics: {results}")
                    for key in ['mmseg/mIoU', 'mIoU', 'iou', 'IoU']:
                        if key in results:
                            miou_val = float(results[key])
                            print(f"  ✓ Parsed mIoU={miou_val} (key: {key})")
                            return miou_val
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"[WARNING] No metrics file found")
    return None


if __name__ == "__main__":
    # Test on actual directories
    domain_a_dir = r"work_dirs\continual_experiment\stage2_train_domain_a\eval_domain_a"
    domain_b_dir = r"work_dirs\continual_experiment\stage2_train_domain_a\eval_domain_b"
    
    print("=" * 60)
    print("Testing Domain A metrics parsing")
    print("=" * 60)
    miou_a = parse_metrics(domain_a_dir)
    print(f"Result: {miou_a}\n")
    
    print("=" * 60)
    print("Testing Domain B metrics parsing")
    print("=" * 60)
    miou_b = parse_metrics(domain_b_dir)
    print(f"Result: {miou_b}\n")
    
    if miou_a is not None and miou_b is not None:
        print(f"✓ SUCCESS: Domain A mIoU={miou_a:.2f}, Domain B mIoU={miou_b:.2f}")
    else:
        print(f"✗ FAILED: Could not parse metrics")
