#!/usr/bin/env python
"""Monitor training progress by watching work_dir for changes."""

import os
import sys
import time
import glob
from pathlib import Path

def monitor_training(work_dir: str, check_interval: int = 5):
    """Monitor training by checking for checkpoint/log updates."""
    
    work_dir = Path(work_dir)
    if not work_dir.exists():
        print(f"[ERROR] Work directory does not exist: {work_dir}")
        return
    
    print(f"🔍 Monitoring training in: {work_dir}")
    print(f"   Check interval: {check_interval}s")
    print("   (Press Ctrl+C to stop)\n")
    
    last_checkpoint = None
    last_log_size = 0
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Check for checkpoints
            checkpoints = sorted(glob.glob(str(work_dir / "*.pth")))
            if checkpoints and checkpoints[-1] != last_checkpoint:
                last_checkpoint = checkpoints[-1]
                mtime = os.path.getmtime(last_checkpoint)
                size_mb = os.path.getsize(last_checkpoint) / (1024*1024)
                print(f"✅ [{iteration}] Checkpoint updated: {Path(last_checkpoint).name} ({size_mb:.1f}MB)")
            
            # Check for log file growth
            log_files = glob.glob(str(work_dir / "*.log"))
            if log_files:
                log_file = log_files[-1]
                current_size = os.path.getsize(log_file)
                if current_size > last_log_size:
                    size_kb = (current_size - last_log_size) / 1024
                    print(f"📝 [{iteration}] Log updated (+{size_kb:.1f}KB)")
                    last_log_size = current_size
                    
                    # Show last few lines of log
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()[-3:]  # Last 3 lines
                        for line in lines:
                            if line.strip():
                                print(f"    {line.rstrip()}")
            else:
                print(f"ℹ️  [{iteration}] No log file found yet...")
            
            # Check directory size
            total_size = sum(f.stat().st_size for f in work_dir.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024*1024)
            print(f"📊 Work dir size: {total_size_mb:.1f}MB\n")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--work-dir', required=True, help='Training work directory')
    parser.add_argument('--interval', type=int, default=5, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    monitor_training(args.work_dir, args.interval)
